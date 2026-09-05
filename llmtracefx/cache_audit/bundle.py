"""Write, sanitize, and verify deterministic cache-audit evidence bundles."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from llmtracefx.evidence.core import PRIVACY_PATTERNS, canonical_json
from llmtracefx.optimizer._artifact_io import (
    MAX_EVIDENCE_ARTIFACT_BYTES,
    read_bounded_regular_bytes,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from llmtracefx.optimizer.collectors._shared import atomic_write_text

from .report import build_claim_matrix, build_summary
from .report_html import render_html, render_reuse_alignment_svg
from .schema import (
    AuditManifest,
    CacheEventRecord,
    CacheStateSnapshot,
    CostEvidence,
    EvidenceFact,
    Limitation,
    MemoryEvidence,
    OutputEvidence,
    PublicationMode,
    RequestEvidence,
    ReuseEvidence,
    unavailable,
)
from .verdicts import classify_request
from .workloads import adversarial_requests, workload_digest

BUNDLE_DATA_FILES = (
    "audit-manifest.json",
    "request-evidence.jsonl",
    "cache-events.jsonl",
    "claim-matrix.json",
    "summary.json",
    "reuse-alignment.svg",
    "report.html",
    "evidence_bundle.py",
)
BUNDLE_FILES = (*BUNDLE_DATA_FILES, "SHA256SUMS")
_CHECKSUM = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._-]*)$")


class CacheAuditBundleError(ValueError):
    """Raised when a cache-audit evidence bundle fails closed."""


def package_source_digest() -> str:
    """Digest the complete Python package that can affect bundle verification."""

    package_dir = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_dir.rglob("*.py")):
        relative = path.relative_to(package_dir).as_posix().encode("utf-8")
        content = _normalized_source(
            relative.decode("utf-8"),
            path.read_bytes(),
        )
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return "sha256:" + digest.hexdigest()


def _normalized_source(relative: str, content: bytes) -> bytes:
    if relative != "evidence/registry.py":
        return content
    text = content.decode("utf-8")
    text = re.sub(
        r'^_CACHE_AUDIT_SOURCE_COMMIT = "[^"]*"$',
        '_CACHE_AUDIT_SOURCE_COMMIT = "<bound-at-generation>"',
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^_CACHE_AUDIT_PACKAGE_DIGEST = "[^"]*"$',
        '_CACHE_AUDIT_PACKAGE_DIGEST = "<bound-at-generation>"',
        text,
        flags=re.MULTILINE,
    )
    return text.encode("utf-8")


def _portable_verifier(manifest: AuditManifest) -> str:
    if manifest.generator_package_digest is None:
        raise CacheAuditBundleError(
            "portable verification requires a generator package digest"
        )
    return f'''"""Offline verifier bound to one audited source implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

EXPECTED_COMMIT = {manifest.generator_commit!r}
EXPECTED_PACKAGE_DIGEST = {manifest.generator_package_digest!r}
BUNDLE_DATA_FILES = {BUNDLE_DATA_FILES!r}
BUNDLE_FILES = BUNDLE_DATA_FILES + ("SHA256SUMS",)
MAX_FILE_BYTES = {MAX_EVIDENCE_ARTIFACT_BYTES}


def fail(message: str) -> None:
    raise SystemExit(message)


def safe_bytes(path: Path) -> bytes:
    if path.is_symlink():
        fail(f"unsafe symlink: {{path}}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        fail(f"cannot open {{path}}: {{exc}}")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
            fail(f"unsafe or oversized file: {{path}}")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65536, MAX_FILE_BYTES - total + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_FILE_BYTES:
                fail(f"oversized file: {{path}}")
        after = os.fstat(descriptor)
        signature = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
        )
        if signature(before) != signature(after):
            fail(f"file changed while reading: {{path}}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def verify_bundle_envelope(bundle: Path) -> None:
    if bundle.is_symlink() or not bundle.is_dir():
        fail("bundle must be a regular directory")
    names = {{item.name for item in bundle.iterdir()}}
    if names != set(BUNDLE_FILES):
        fail("bundle file allowlist mismatch")
    checksum_lines = safe_bytes(bundle / "SHA256SUMS").decode("ascii").splitlines()
    checksums = {{}}
    for line in checksum_lines:
        parts = line.split("  ")
        if len(parts) != 2 or len(parts[0]) != 64 or any(
            character not in "0123456789abcdef" for character in parts[0]
        ):
            fail("invalid checksum manifest")
        digest, name = parts
        if name in checksums:
            fail("duplicate checksum entry")
        checksums[name] = digest
    if set(checksums) != set(BUNDLE_DATA_FILES):
        fail("checksum file allowlist mismatch")
    for name, expected in checksums.items():
        if hashlib.sha256(safe_bytes(bundle / name)).hexdigest() != expected:
            fail(f"checksum mismatch: {{name}}")
    try:
        manifest = json.loads(safe_bytes(bundle / "audit-manifest.json"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"invalid audit manifest: {{exc}}")
    if manifest.get("schema_version") != "2":
        fail("unsupported audit schema")
    if manifest.get("generator_commit") != EXPECTED_COMMIT:
        fail("bundle generator commit does not match embedded trust anchor")
    if manifest.get("generator_package_digest") != EXPECTED_PACKAGE_DIGEST:
        fail("bundle package digest does not match embedded trust anchor")


def normalized_source(relative: str, content: bytes) -> bytes:
    if relative != "evidence/registry.py":
        return content
    import re

    text = content.decode("utf-8")
    text = re.sub(
        r'^_CACHE_AUDIT_SOURCE_COMMIT = "[^"]*"$',
        '_CACHE_AUDIT_SOURCE_COMMIT = "<bound-at-generation>"',
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^_CACHE_AUDIT_PACKAGE_DIGEST = "[^"]*"$',
        '_CACHE_AUDIT_PACKAGE_DIGEST = "<bound-at-generation>"',
        text,
        flags=re.MULTILINE,
    )
    return text.encode("utf-8")


def snapshot_package(root: Path) -> tuple[Path, str]:
    package = root / "llmtracefx"
    if package.is_symlink() or not package.is_dir():
        fail("candidate has no regular llmtracefx package")
    temporary = Path(tempfile.mkdtemp(prefix="llmtracefx-verifier-"))
    snapshot = temporary / "llmtracefx"
    snapshot.mkdir(mode=0o700)
    digest = hashlib.sha256()
    paths = sorted(package.rglob("*.py"))
    if not paths:
        fail("candidate llmtracefx package has no Python sources")
    for source in paths:
        relative_path = source.relative_to(package)
        for parent in (source, *source.parents):
            if parent == package.parent:
                break
            if parent.is_symlink():
                fail(f"candidate package contains symlink: {{parent}}")
        content = safe_bytes(source)
        digest_content = normalized_source(relative_path.as_posix(), content)
        relative = relative_path.as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(digest_content).to_bytes(8, "big"))
        digest.update(digest_content)
        destination = snapshot / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            os.write(descriptor, content)
        finally:
            os.close(descriptor)
    return temporary, "sha256:" + digest.hexdigest()


def commit_matches(root: Path) -> bool:
    if EXPECTED_COMMIT is None or not (root / ".git").exists():
        return True
    import subprocess

    listing = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-tree",
            "-r",
            "--name-only",
            EXPECTED_COMMIT,
            "--",
            "llmtracefx",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if listing.returncode != 0:
        return False
    digest = hashlib.sha256()
    paths = sorted(
        line.decode("utf-8").removeprefix("llmtracefx/")
        for line in listing.stdout.splitlines()
        if line.endswith(b".py")
    )
    for relative_text in paths:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "show",
                EXPECTED_COMMIT + ":llmtracefx/" + relative_text,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            return False
        content = normalized_source(relative_text, result.stdout)
        relative = relative_text.encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return "sha256:" + digest.hexdigest() == EXPECTED_PACKAGE_DIGEST


def resolve_package(bundle: Path, requested: Path | None) -> Path:
    candidates = [requested.resolve()] if requested is not None else []
    if requested is None:
        candidates.extend((bundle.resolve(), *bundle.resolve().parents))
    for root in candidates:
        temporary, digest = snapshot_package(root)
        if digest == EXPECTED_PACKAGE_DIGEST and commit_matches(root):
            return temporary
        import shutil

        shutil.rmtree(temporary)
    fail(
        "matching llmtracefx source not found; pass --package-root for the "
        "repository or installed package recorded by this bundle"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--public-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--package-root", type=Path)
    args = parser.parse_args()
    bundle = args.public_dir.resolve()
    verify_bundle_envelope(bundle)
    snapshot_root = resolve_package(bundle, args.package_root)
    sys.path.insert(0, str(snapshot_root))
    from llmtracefx.cache_audit.bundle import verify_bundle

    print(json.dumps({{"verified": True, **verify_bundle(bundle)}}, sort_keys=True))


if __name__ == "__main__":
    main()
'''


def _verify_public_synthetic_provenance(
    manifest: AuditManifest, records: Sequence[RequestEvidence]
) -> None:
    approved = adversarial_requests()
    if (
        manifest.backend != "synthetic_reference"
        or manifest.model_id != "synthetic-tiny-model"
        or manifest.tokenizer_id != "integer-tokenizer-v1"
        or tuple(record.spec for record in records) != approved
        or manifest.adapter_version != "2"
        or manifest.workload_digest != workload_digest(approved)
        or manifest.generator_commit is None
        or manifest.generator_package_digest != package_source_digest()
    ):
        raise CacheAuditBundleError(
            "public_synthetic publication requires the approved built-in "
            "synthetic reference workload and identities"
        )


def _verify_manifest_backend_contract(manifest: AuditManifest) -> None:
    if manifest.publication_mode is PublicationMode.PUBLIC_REDACTED:
        return
    if manifest.backend == "synthetic_reference":
        if (
            manifest.backend_version != "1"
            or manifest.cache_config.cache_type != "token_trie"
            or manifest.cache_config.hash_algorithm is not None
            or manifest.cache_config.hash_block_size is not None
            or manifest.cache_config.physical_block_sizes
            or manifest.cache_config.fine_grained_hits
            or manifest.model_artifact_digest is not None
            or manifest.runtime_identity.get("synthetic_engine")
            != "independent-state-machine-v2"
        ):
            raise CacheAuditBundleError(
                "synthetic-reference manifest contradicts adapter-owned identity"
            )
        return
    if manifest.backend == "mlx_lm_local":
        from .adapters.mlx import REQUIRED_MLX_LM_VERSION, REQUIRED_MLX_VERSION

        if (
            manifest.backend_version != REQUIRED_MLX_LM_VERSION
            or manifest.runtime_identity.get("mlx") != REQUIRED_MLX_VERSION
            or manifest.runtime_identity.get("mlx_lm") != REQUIRED_MLX_LM_VERSION
            or manifest.cache_config.cache_type != "mlx_lru_prompt_cache"
            or manifest.cache_config.hash_algorithm is not None
            or manifest.cache_config.hash_block_size is not None
            or manifest.cache_config.physical_block_sizes
            or manifest.cache_config.fine_grained_hits
            or manifest.model_artifact_digest is None
        ):
            raise CacheAuditBundleError(
                "MLX manifest contradicts pinned adapter-owned identity"
            )
        return
    raise CacheAuditBundleError(f"unsupported bundle backend: {manifest.backend}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(
        read_bounded_regular_bytes(path, MAX_EVIDENCE_ARTIFACT_BYTES)
    ).hexdigest()


def _jsonl(records: Sequence[RequestEvidence], *, include_tokens: bool) -> str:
    return "".join(
        json.dumps(
            record.to_dict(include_tokens=include_tokens),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
        for record in records
    )


def _events_jsonl(records: Sequence[RequestEvidence]) -> str:
    rows = []
    for record in records:
        for event in record.events:
            rows.append(
                {
                    "request_id": record.spec.request_id,
                    "event": event.to_dict(),
                }
            )
    return "".join(
        json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
        for row in rows
    )


def write_bundle(
    output_dir: Path,
    manifest: AuditManifest,
    records: Sequence[RequestEvidence],
) -> None:
    """Write one complete bundle atomically at file granularity."""

    if output_dir.is_symlink():
        raise CacheAuditBundleError("output directory must not be a symlink")
    if output_dir.exists():
        if not output_dir.is_dir():
            raise CacheAuditBundleError("output path must be a directory")
        if any(output_dir.iterdir()):
            raise CacheAuditBundleError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if tuple(record.spec.request_id for record in records) != manifest.request_order:
        raise CacheAuditBundleError("record order does not match manifest")
    if workload_digest([record.spec for record in records]) != manifest.workload_digest:
        raise CacheAuditBundleError(
            "recorded request specifications do not match manifest"
        )
    if manifest.generator_package_digest != package_source_digest():
        raise CacheAuditBundleError(
            "manifest generating-package digest does not match this package"
        )
    _verify_manifest_backend_contract(manifest)
    if manifest.publication_mode is PublicationMode.PUBLIC_SYNTHETIC:
        _verify_public_synthetic_provenance(manifest, records)
        _verify_reference_oracle(manifest, records)
    include_tokens = manifest.publication_mode is not PublicationMode.PUBLIC_REDACTED
    atomic_write_text(
        output_dir / "audit-manifest.json", canonical_json(manifest.to_dict())
    )
    atomic_write_text(
        output_dir / "request-evidence.jsonl",
        _jsonl(records, include_tokens=include_tokens),
    )
    atomic_write_text(output_dir / "cache-events.jsonl", _events_jsonl(records))
    atomic_write_text(
        output_dir / "claim-matrix.json",
        canonical_json(build_claim_matrix(records)),
    )
    atomic_write_text(
        output_dir / "summary.json",
        canonical_json(build_summary(records, manifest)),
    )
    atomic_write_text(
        output_dir / "reuse-alignment.svg",
        render_reuse_alignment_svg(records),
    )
    atomic_write_text(output_dir / "report.html", render_html(manifest, records))
    atomic_write_text(
        output_dir / "evidence_bundle.py",
        _portable_verifier(manifest),
    )
    checksums = "".join(
        f"{_sha256(output_dir / name)}  {name}\n" for name in BUNDLE_DATA_FILES
    )
    atomic_write_text(output_dir / "SHA256SUMS", checksums)


def _load_json(path: Path) -> Any:
    text = read_bounded_regular_text(
        path,
        max_bytes=MAX_EVIDENCE_ARTIFACT_BYTES,
    )
    try:
        return json.loads(text, parse_constant=reject_non_finite_json_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise CacheAuditBundleError(f"invalid JSON in {path.name}: {exc}") from exc


def _load_records(path: Path) -> list[RequestEvidence]:
    text = read_bounded_regular_text(
        path,
        max_bytes=MAX_EVIDENCE_ARTIFACT_BYTES,
    )
    records: list[RequestEvidence] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise CacheAuditBundleError(
                f"blank line in request-evidence.jsonl at {line_number}"
            )
        try:
            value = json.loads(line, parse_constant=reject_non_finite_json_constant)
            records.append(RequestEvidence.from_dict(value))
        except (json.JSONDecodeError, ValueError) as exc:
            raise CacheAuditBundleError(
                f"invalid request evidence at line {line_number}: {exc}"
            ) from exc
    return records


def _verify_checksums(bundle_dir: Path) -> None:
    text = read_bounded_regular_text(
        bundle_dir / "SHA256SUMS",
        max_bytes=MAX_EVIDENCE_ARTIFACT_BYTES,
    )
    found: dict[str, str] = {}
    for line in text.splitlines():
        match = _CHECKSUM.fullmatch(line)
        if match is None:
            raise CacheAuditBundleError(f"invalid checksum line: {line!r}")
        digest, name = match.groups()
        if name in found:
            raise CacheAuditBundleError(f"duplicate checksum entry: {name}")
        found[name] = digest
    if set(found) != set(BUNDLE_DATA_FILES):
        raise CacheAuditBundleError("checksum file list is incomplete or unexpected")
    for name, expected in found.items():
        if _sha256(bundle_dir / name) != expected:
            raise CacheAuditBundleError(f"checksum mismatch: {name}")


def _verify_public_privacy(bundle_dir: Path) -> None:
    for name in BUNDLE_FILES:
        text = read_bounded_regular_text(
            bundle_dir / name,
            max_bytes=MAX_EVIDENCE_ARTIFACT_BYTES,
        )
        for pattern, label in PRIVACY_PATTERNS:
            if pattern.search(text):
                raise CacheAuditBundleError(f"{name} contains {label}")
        if '"cache_salt"' in text or '"raw_hash"' in text:
            raise CacheAuditBundleError(f"{name} contains private cache identity")


def _facts(record: RequestEvidence) -> tuple[EvidenceFact[Any], ...]:
    snapshots = tuple(
        fact
        for snapshot in (record.cache_before, record.cache_after)
        if snapshot is not None
        for fact in (
            snapshot.entry_count,
            snapshot.logical_bytes,
            snapshot.valid_token_offsets,
            snapshot.cache_classes,
        )
    )
    return (
        *(getattr(record.reuse, name) for name in record.reuse.__dataclass_fields__),
        *(getattr(record.memory, name) for name in record.memory.__dataclass_fields__),
        record.output.token_identity,
        record.output.correctness,
        record.cost.billed,
        record.cost.estimated,
        *snapshots,
    )


def _verify_public_redacted_shape(
    manifest: AuditManifest, records: Sequence[RequestEvidence]
) -> None:
    if (
        manifest.run_id != "cache-audit-public-redacted"
        or manifest.backend
        not in {"synthetic_reference", "mlx_lm_local", "vllm", "redacted-backend"}
        or manifest.backend_version != "redacted"
        or manifest.adapter_version != "redacted"
        or manifest.model_id != "redacted-model"
        or manifest.tokenizer_id != "redacted-tokenizer"
        or manifest.model_artifact_digest is not None
        or manifest.runtime_identity != {"redaction": "public"}
        or manifest.cache_config.namespace_id != "redacted-namespace"
        or manifest.cache_config.cache_type != "redacted-cache"
        or manifest.cache_config.hash_algorithm is not None
        or manifest.cache_config.cache_salt_relationship is not None
        or manifest.generator_commit is not None
        or manifest.generator_package_digest is None
    ):
        raise CacheAuditBundleError("public-redacted manifest is not de-identified")
    expected_ids = tuple(f"request-{index:04d}" for index in range(len(records)))
    if manifest.request_order != expected_ids:
        raise CacheAuditBundleError("public-redacted request identifiers are invalid")
    for limitation in manifest.limitations:
        if (
            re.fullmatch(r"redacted_limitation_[0-9]{4}", limitation.code) is None
            or limitation.message
            != "Limitation details are available only in the private bundle."
        ):
            raise CacheAuditBundleError(
                "public-redacted manifest limitation is invalid"
            )
    request_id_set = set(expected_ids)
    for index, record in enumerate(records):
        if record.spec.request_id != expected_ids[index]:
            raise CacheAuditBundleError("public-redacted request identifier is invalid")
        if re.fullmatch(r"namespace-[0-9]{4}", record.spec.namespace_id) is None:
            raise CacheAuditBundleError("public-redacted namespace is invalid")
        if re.fullmatch(r"replicate-[0-9]{4}", record.spec.replicate_id) is None:
            raise CacheAuditBundleError(
                "public-redacted replicate identifier is invalid"
            )
        if (
            record.spec.pair_id is not None
            and re.fullmatch(r"pair-[0-9]{4}", record.spec.pair_id) is None
        ):
            raise CacheAuditBundleError("public-redacted pair identifier is invalid")
        if not set(record.spec.expected_predecessors) <= request_id_set:
            raise CacheAuditBundleError("public-redacted predecessor is invalid")
        for fact in _facts(record):
            if fact.source != f"redacted.{fact.basis.value}":
                raise CacheAuditBundleError(
                    "public-redacted evidence source is invalid"
                )
            if not set(fact.limitations) <= {
                "details_redacted",
                "token_identity_redacted",
                "output_tokens_redacted",
                "cache_classes_redacted",
            }:
                raise CacheAuditBundleError(
                    "public-redacted evidence limitation is invalid"
                )
        for limitation in record.limitations:
            if limitation.code == "public_tokens_redacted":
                continue
            if re.fullmatch(r"redacted_limitation_[0-9]{4}", limitation.code) is None:
                raise CacheAuditBundleError(
                    "public-redacted limitation identifier is invalid"
                )
            if (
                limitation.message
                != "Limitation details are available only in the private bundle."
            ):
                raise CacheAuditBundleError(
                    "public-redacted limitation message is invalid"
                )
        for event in record.events:
            if event.event_type not in {
                "stored",
                "removed",
                "cleared",
                "preempted",
                "BlockStored",
                "BlockRemoved",
                "AllBlocksCleared",
                "redacted",
            } or event.medium not in {
                None,
                "CPU",
                "GPU",
                "cpu",
                "gpu",
                "host",
                "device",
            }:
                raise CacheAuditBundleError("public-redacted cache event is invalid")
            if not set(event.limitations) <= {"details_redacted"}:
                raise CacheAuditBundleError(
                    "public-redacted cache event limitation is invalid"
                )


def _verify_reference_oracle(
    manifest: AuditManifest, records: Sequence[RequestEvidence]
) -> None:
    if manifest.backend != "synthetic_reference":
        return
    if any(record.spec.input_token_ids is None for record in records):
        return
    from .adapters.reference import ReferenceCacheAdapter

    expected_records = ReferenceCacheAdapter(
        max_entries=manifest.cache_config.max_entries or 32,
        max_bytes=manifest.cache_config.max_bytes or (1 << 30),
    ).run([record.spec for record in records])
    for expected, observed in zip(expected_records, records, strict=True):
        fields = tuple(expected.reuse.__dataclass_fields__)
        for field in fields:
            if (
                getattr(expected.reuse, field).value
                != getattr(observed.reuse, field).value
            ):
                raise CacheAuditBundleError(
                    f"independent oracle mismatch for "
                    f"{observed.spec.request_id}.{field}"
                )
        if expected.output.output_token_ids != observed.output.output_token_ids:
            raise CacheAuditBundleError(
                f"output control mismatch for {observed.spec.request_id}"
            )
        if expected.output.baseline_token_ids != observed.output.baseline_token_ids:
            raise CacheAuditBundleError(
                f"baseline control mismatch for {observed.spec.request_id}"
            )


def verify_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Verify checksums, schema, request order, verdicts, reports, and privacy."""

    if not bundle_dir.is_dir() or bundle_dir.is_symlink():
        raise CacheAuditBundleError("bundle must be a regular directory")
    actual = {item.name for item in bundle_dir.iterdir()}
    if actual != set(BUNDLE_FILES):
        raise CacheAuditBundleError(
            f"bundle files differ: missing={sorted(set(BUNDLE_FILES) - actual)}, "
            f"extra={sorted(actual - set(BUNDLE_FILES))}"
        )
    for name in BUNDLE_FILES:
        path = bundle_dir / name
        if path.is_symlink() or not path.is_file():
            raise CacheAuditBundleError(f"{name} must be a regular non-symlink file")
    _verify_checksums(bundle_dir)
    manifest = AuditManifest.from_dict(_load_json(bundle_dir / "audit-manifest.json"))
    if manifest.generator_package_digest != package_source_digest():
        raise CacheAuditBundleError(
            "bundle belongs to a different llmtracefx cache-audit package"
        )
    _verify_manifest_backend_contract(manifest)
    records = _load_records(bundle_dir / "request-evidence.jsonl")
    if tuple(record.spec.request_id for record in records) != manifest.request_order:
        raise CacheAuditBundleError("request evidence order does not match manifest")
    if workload_digest([record.spec for record in records]) != manifest.workload_digest:
        raise CacheAuditBundleError(
            "request specifications do not match workload digest"
        )
    if manifest.publication_mode is PublicationMode.PUBLIC_SYNTHETIC:
        _verify_public_synthetic_provenance(manifest, records)
    if manifest.publication_mode is PublicationMode.PUBLIC_REDACTED:
        for record in records:
            if (
                record.spec.input_token_ids is not None
                or record.output.output_token_ids is not None
                or record.output.baseline_token_ids is not None
            ):
                raise CacheAuditBundleError(
                    "public-redacted bundle contains exact token IDs"
                )
        _verify_public_redacted_shape(manifest, records)
    if read_bounded_regular_text(
        bundle_dir / "cache-events.jsonl",
        max_bytes=MAX_EVIDENCE_ARTIFACT_BYTES,
    ) != _events_jsonl(records):
        raise CacheAuditBundleError("cache event stream is not derived from requests")
    _verify_reference_oracle(manifest, records)
    for record in records:
        output_tokens = record.output.output_token_ids
        baseline_tokens = record.output.baseline_token_ids
        if (output_tokens is None) != (baseline_tokens is None):
            raise CacheAuditBundleError(
                f"incomplete output comparison for request {record.spec.request_id}"
            )
        if output_tokens is None:
            if record.output.token_identity.value is not None:
                raise CacheAuditBundleError(
                    f"token identity lacks exact arrays for request "
                    f"{record.spec.request_id}"
                )
        else:
            assert baseline_tokens is not None
            if (
                len(output_tokens) > record.spec.output_tokens
                or len(baseline_tokens) > record.spec.output_tokens
            ):
                raise CacheAuditBundleError(
                    f"output exceeds requested length for {record.spec.request_id}"
                )
            if record.output.token_identity.value != (output_tokens == baseline_tokens):
                raise CacheAuditBundleError(
                    f"output token identity mismatch for {record.spec.request_id}"
                )
        classified = classify_request(replace(record, verdict=None, verdict_reasons=()))
        if classified.verdict != record.verdict:
            raise CacheAuditBundleError(
                f"verdict mismatch for request {record.spec.request_id}"
            )
        if classified.verdict_reasons != record.verdict_reasons:
            raise CacheAuditBundleError(
                f"verdict reasons mismatch for request {record.spec.request_id}"
            )
        if classified.eligibility != record.eligibility:
            raise CacheAuditBundleError(
                f"claim eligibility mismatch for request {record.spec.request_id}"
            )
    if _load_json(bundle_dir / "claim-matrix.json") != build_claim_matrix(records):
        raise CacheAuditBundleError("claim matrix is not derived from request evidence")
    if _load_json(bundle_dir / "summary.json") != build_summary(records, manifest):
        raise CacheAuditBundleError("summary is not derived from request evidence")
    if read_bounded_regular_text(
        bundle_dir / "reuse-alignment.svg",
        max_bytes=MAX_EVIDENCE_ARTIFACT_BYTES,
    ) != render_reuse_alignment_svg(records):
        raise CacheAuditBundleError("reuse SVG is not deterministic")
    if read_bounded_regular_text(
        bundle_dir / "report.html",
        max_bytes=MAX_EVIDENCE_ARTIFACT_BYTES,
    ) != render_html(manifest, records):
        raise CacheAuditBundleError("HTML report is not deterministic")
    if read_bounded_regular_text(
        bundle_dir / "evidence_bundle.py",
        max_bytes=MAX_EVIDENCE_ARTIFACT_BYTES,
    ) != _portable_verifier(manifest):
        raise CacheAuditBundleError("portable verifier wrapper is not deterministic")
    if manifest.publication_mode is not PublicationMode.PRIVATE:
        _verify_public_privacy(bundle_dir)
    return {
        "run_id": manifest.run_id,
        "backend": manifest.backend,
        "request_count": len(records),
        "verdict_counts": dict(
            sorted(
                Counter(
                    "unclassified" if record.verdict is None else record.verdict.value
                    for record in records
                ).items()
            )
        ),
        "token_identity_reproducible": (
            manifest.publication_mode is not PublicationMode.PUBLIC_REDACTED
        ),
    }


def read_bundle(bundle_dir: Path) -> tuple[AuditManifest, list[RequestEvidence]]:
    """Load a checksummed bundle after full verification."""

    verify_bundle(bundle_dir)
    return (
        AuditManifest.from_dict(_load_json(bundle_dir / "audit-manifest.json")),
        _load_records(bundle_dir / "request-evidence.jsonl"),
    )


def _redacted_fact(fact: EvidenceFact[Any]) -> EvidenceFact[Any]:
    return replace(
        fact,
        source=f"redacted.{fact.basis.value}",
        limitations=("details_redacted",) if fact.limitations else (),
    )


def _redacted_snapshot(
    snapshot: CacheStateSnapshot | None,
) -> CacheStateSnapshot | None:
    if snapshot is None:
        return None
    return replace(
        snapshot,
        entry_count=_redacted_fact(snapshot.entry_count),
        logical_bytes=_redacted_fact(snapshot.logical_bytes),
        valid_token_offsets=_redacted_fact(snapshot.valid_token_offsets),
        cache_classes=unavailable("redacted.unavailable", "cache_classes_redacted"),
    )


def _redacted_limitations(
    limitations: Sequence[Limitation],
) -> tuple[Limitation, ...]:
    return tuple(
        Limitation(
            code=f"redacted_limitation_{index:04d}",
            message="Limitation details are available only in the private bundle.",
            blocks_verdict=limitation.blocks_verdict,
        )
        for index, limitation in enumerate(limitations)
    )


def sanitized_records(
    records: Sequence[RequestEvidence],
) -> list[RequestEvidence]:
    """Return structurally de-identified public records and reclassify them."""

    limitation = Limitation(
        code="public_tokens_redacted",
        message=(
            "Exact input/output token IDs remain private; public token-identity "
            "claims are not independently reproducible."
        ),
        blocks_verdict=False,
    )
    request_ids = {
        record.spec.request_id: f"request-{index:04d}"
        for index, record in enumerate(records)
    }
    pair_ids = {
        pair_id: f"pair-{index:04d}"
        for index, pair_id in enumerate(
            dict.fromkeys(
                record.spec.pair_id
                for record in records
                if record.spec.pair_id is not None
            )
        )
    }
    namespace_ids = {
        namespace_id: f"namespace-{index:04d}"
        for index, namespace_id in enumerate(
            dict.fromkeys(record.spec.namespace_id for record in records)
        )
    }
    sanitized: list[RequestEvidence] = []
    for index, record in enumerate(records):
        spec = replace(
            record.spec,
            request_id=request_ids[record.spec.request_id],
            input_token_ids=None,
            pair_id=(
                None if record.spec.pair_id is None else pair_ids[record.spec.pair_id]
            ),
            expected_predecessors=tuple(
                request_ids[predecessor]
                for predecessor in record.spec.expected_predecessors
            ),
            namespace_id=namespace_ids[record.spec.namespace_id],
            replicate_id=f"replicate-{index:04d}",
        )
        redacted = replace(
            record,
            spec=spec,
            reuse=ReuseEvidence(
                semantic_prefix_tokens=unavailable(
                    "redacted.unavailable", "token_identity_redacted"
                ),
                policy_reusable_tokens=unavailable(
                    "redacted.unavailable", "token_identity_redacted"
                ),
                reusable_blocks=unavailable(
                    "redacted.unavailable", "token_identity_redacted"
                ),
                partial_block_tokens=unavailable(
                    "redacted.unavailable", "token_identity_redacted"
                ),
                engine_cached_tokens=_redacted_fact(record.reuse.engine_cached_tokens),
                engine_cached_blocks=_redacted_fact(record.reuse.engine_cached_blocks),
                engine_created_tokens=_redacted_fact(
                    record.reuse.engine_created_tokens
                ),
                observed_prompt_tokens=_redacted_fact(
                    record.reuse.observed_prompt_tokens
                ),
                policy_required_prompt_tokens=unavailable(
                    "redacted.unavailable", "token_identity_redacted"
                ),
                unexpected_recomputed_tokens=unavailable(
                    "redacted.unavailable", "token_identity_redacted"
                ),
                eviction_observed=unavailable(
                    "redacted.unavailable", "token_identity_redacted"
                ),
                preemption_observed=unavailable(
                    "redacted.unavailable", "token_identity_redacted"
                ),
                prior_residency_observed=unavailable(
                    "redacted.unavailable", "token_identity_redacted"
                ),
                residency_absence_observed=unavailable(
                    "redacted.unavailable", "token_identity_redacted"
                ),
            ),
            memory=MemoryEvidence(
                **{
                    name: _redacted_fact(getattr(record.memory, name))
                    for name in record.memory.__dataclass_fields__
                }
            ),
            output=OutputEvidence(
                output_token_ids=None,
                baseline_token_ids=None,
                token_identity=unavailable(
                    "redacted.unavailable", "output_tokens_redacted"
                ),
                correctness=unavailable(
                    "redacted.unavailable", "output_tokens_redacted"
                ),
                finish_reason=(
                    record.output.finish_reason
                    if record.output.finish_reason in {"stop", "length", "eos"}
                    else None
                ),
            ),
            cost=CostEvidence(
                billed=_redacted_fact(record.cost.billed),
                estimated=_redacted_fact(record.cost.estimated),
                currency=None,
            ),
            limitations=_redacted_limitations(record.limitations) + (limitation,),
            cache_before=_redacted_snapshot(record.cache_before),
            cache_after=_redacted_snapshot(record.cache_after),
            events=tuple(
                CacheEventRecord(
                    sequence=event.sequence,
                    event_type=(
                        event.event_type
                        if event.event_type
                        in {
                            "stored",
                            "removed",
                            "cleared",
                            "preempted",
                            "BlockStored",
                            "BlockRemoved",
                            "AllBlocksCleared",
                        }
                        else "redacted"
                    ),
                    basis=event.basis,
                    token_count=event.token_count,
                    block_count=event.block_count,
                    medium=(
                        event.medium
                        if event.medium
                        in {"CPU", "GPU", "cpu", "gpu", "host", "device"}
                        else None
                    ),
                    group_index=event.group_index,
                    limitations=("details_redacted",) if event.limitations else (),
                )
                for event in record.events
            ),
            verdict=None,
            verdict_reasons=(),
        )
        sanitized.append(classify_request(redacted))
    return sanitized


def sanitized_manifest(
    manifest: AuditManifest, records: Sequence[RequestEvidence]
) -> AuditManifest:
    """Return a non-correlating manifest bound to the sanitized request specs."""

    return replace(
        manifest,
        run_id="cache-audit-public-redacted",
        backend=(
            manifest.backend
            if manifest.backend in {"synthetic_reference", "mlx_lm_local", "vllm"}
            else "redacted-backend"
        ),
        backend_version="redacted",
        adapter_version="redacted",
        model_id="redacted-model",
        tokenizer_id="redacted-tokenizer",
        model_artifact_digest=None,
        runtime_identity={"redaction": "public"},
        cache_config=replace(
            manifest.cache_config,
            namespace_id="redacted-namespace",
            cache_type="redacted-cache",
            hash_algorithm=None,
            cache_salt_relationship=None,
        ),
        publication_mode=PublicationMode.PUBLIC_REDACTED,
        request_order=tuple(record.spec.request_id for record in records),
        workload_digest=workload_digest([record.spec for record in records]),
        generator_commit=None,
        limitations=_redacted_limitations(manifest.limitations),
    )


def sanitize_bundle_records(
    manifest: AuditManifest, records: Sequence[RequestEvidence]
) -> tuple[AuditManifest, list[RequestEvidence]]:
    """Sanitize records and return the correspondingly rebound manifest."""

    redacted_records = sanitized_records(records)
    return sanitized_manifest(manifest, redacted_records), redacted_records
