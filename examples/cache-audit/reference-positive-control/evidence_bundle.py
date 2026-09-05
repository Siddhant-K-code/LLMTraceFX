"""Offline verifier bound to one audited source implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from datetime import datetime
from pathlib import Path

EXPECTED_COMMIT = "a5d63a90a0f31a5c1d6ada571cf98baf42fe308d"
EXPECTED_COMMIT_AT = "2026-09-05T22:37:54+05:30"
EXPECTED_PACKAGE_DIGEST = (
    "sha256:a0d9ea03f94b12383011d5a03295426cbac3ffc42f40c58b47cf7a36d812f5e3"
)
EXPECTED_GENERATED_AT = "2026-09-05T17:07:59Z"
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
BUNDLE_FILES = BUNDLE_DATA_FILES + ("SHA256SUMS",)
MAX_FILE_BYTES = 67108864
GIT_ENV = {**os.environ, "GIT_NO_LAZY_FETCH": "1"}


def fail(message: str) -> None:
    raise SystemExit(message)


def safe_bytes(path: Path) -> bytes:
    if path.is_symlink():
        fail(f"unsafe symlink: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        fail(f"cannot open {path}: {exc}")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
            fail(f"unsafe or oversized file: {path}")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65536, MAX_FILE_BYTES - total + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_FILE_BYTES:
                fail(f"oversized file: {path}")
        after = os.fstat(descriptor)

        def signature(value):
            return (
                value.st_dev,
                value.st_ino,
                value.st_mode,
                value.st_size,
                value.st_mtime_ns,
            )

        if signature(before) != signature(after):
            fail(f"file changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def verify_bundle_envelope(bundle: Path) -> None:
    if bundle.is_symlink() or not bundle.is_dir():
        fail("bundle must be a regular directory")
    names = {item.name for item in bundle.iterdir()}
    if names != set(BUNDLE_FILES):
        fail("bundle file allowlist mismatch")
    checksum_lines = safe_bytes(bundle / "SHA256SUMS").decode("ascii").splitlines()
    checksums = {}
    for line in checksum_lines:
        parts = line.split("  ")
        if (
            len(parts) != 2
            or len(parts[0]) != 64
            or any(character not in "0123456789abcdef" for character in parts[0])
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
            fail(f"checksum mismatch: {name}")
    try:
        manifest = json.loads(safe_bytes(bundle / "audit-manifest.json"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"invalid audit manifest: {exc}")
    if manifest.get("schema_version") != "2":
        fail("unsupported audit schema")
    if manifest.get("generator_commit") != EXPECTED_COMMIT:
        fail("bundle generator commit does not match embedded trust anchor")
    if manifest.get("generator_commit_at") != EXPECTED_COMMIT_AT:
        fail("bundle generator commit timestamp does not match embedded trust anchor")
    if manifest.get("generator_package_digest") != EXPECTED_PACKAGE_DIGEST:
        fail("bundle package digest does not match embedded trust anchor")
    if manifest.get("generated_at") != EXPECTED_GENERATED_AT:
        fail("bundle generation timestamp does not match embedded trust anchor")
    try:
        captured_at = datetime.fromisoformat(
            manifest["created_at"].replace("Z", "+00:00")
        )
        generated_at = datetime.fromisoformat(
            manifest["generated_at"].replace("Z", "+00:00")
        )
        commit_at = (
            None
            if EXPECTED_COMMIT_AT is None
            else datetime.fromisoformat(EXPECTED_COMMIT_AT.replace("Z", "+00:00"))
        )
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"invalid bundle chronology: {exc}")
    if captured_at.tzinfo is None or generated_at.tzinfo is None:
        fail("bundle chronology must include timezone offsets")
    if generated_at < captured_at:
        fail("bundle generation predates evidence capture")
    if (EXPECTED_COMMIT is None) != (commit_at is None):
        fail("bundle generator commit timestamp binding is incomplete")
    if commit_at is not None and generated_at < commit_at:
        fail("bundle generation predates generator commit")


def normalized_source(relative: str, content: bytes) -> bytes:
    if relative != "evidence/registry.py":
        return content
    import re

    text = content.decode("utf-8")
    text = re.sub(
        r'^_CACHE_AUDIT_SOURCE_COMMIT = (?:"[^"]*"|\(\n\s*"[^"]*"\n\))$',
        '_CACHE_AUDIT_SOURCE_COMMIT = "<bound-at-generation>"',
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^_CACHE_AUDIT_PACKAGE_DIGEST = (?:"[^"]*"|\(\n\s*"[^"]*"\n\))$',
        '_CACHE_AUDIT_PACKAGE_DIGEST = "<bound-at-generation>"',
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^_CACHE_AUDIT_CAPTURED_AT = (?:"[^"]*"|\(\n\s*"[^"]*"\n\))$',
        '_CACHE_AUDIT_CAPTURED_AT = "<bound-at-generation>"',
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^_CACHE_AUDIT_IMPLEMENTATION_BOUND_AT = (?:"[^"]*"|\(\n\s*"[^"]*"\n\))$',
        '_CACHE_AUDIT_IMPLEMENTATION_BOUND_AT = "<bound-at-generation>"',
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
                fail(f"candidate package contains symlink: {parent}")
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


def repository_is_incomplete(root: Path) -> bool:
    import subprocess

    shallow = subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(root),
            "rev-parse",
            "--is-shallow-repository",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
        env=GIT_ENV,
    )
    if shallow.returncode != 0 or shallow.stdout.strip() not in {"true", "false"}:
        fail("candidate repository has invalid shallow-checkout metadata")
    if shallow.stdout.strip() == "true":
        return True
    partial = subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(root),
            "config",
            "--type=bool",
            "--get-regexp",
            r"^remote\..*\.promisor$",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
        env=GIT_ENV,
    )
    if partial.returncode not in {0, 1}:
        fail("candidate repository has invalid partial-clone metadata")
    if partial.returncode == 1:
        return False
    values = [line.rsplit(maxsplit=1)[-1] for line in partial.stdout.splitlines()]
    if not values or any(value not in {"true", "false"} for value in values):
        fail("candidate repository has invalid partial-clone metadata")
    return any(value == "true" for value in values)


def package_objects_missing(root: Path) -> bool | None:
    if EXPECTED_COMMIT is None:
        return False
    import subprocess

    result = subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(root),
            "rev-list",
            "--objects",
            "--missing=print",
            EXPECTED_COMMIT,
            "--",
            "llmtracefx",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
        env=GIT_ENV,
    )
    if result.returncode != 0:
        return None
    return any(line.startswith("?") for line in result.stdout.splitlines())


def commit_matches(root: Path) -> str | None:
    if EXPECTED_COMMIT is None or not (root / ".git").exists():
        return "unavailable"
    import subprocess

    object_type = subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(root),
            "cat-file",
            "-t",
            EXPECTED_COMMIT,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
        env=GIT_ENV,
    )
    if object_type.returncode != 0:
        return "unavailable" if repository_is_incomplete(root) else None
    if object_type.stdout.strip() != "commit":
        return None
    listing = subprocess.run(
        [
            "git",
            "--no-replace-objects",
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
        env=GIT_ENV,
    )
    if listing.returncode != 0:
        return None
    timestamp = subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(root),
            "show",
            "-s",
            "--format=%cI",
            EXPECTED_COMMIT,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
        text=True,
        env=GIT_ENV,
    )
    if timestamp.returncode != 0:
        return None
    try:
        commit_time = datetime.fromisoformat(timestamp.stdout.strip())
        generated_time = datetime.fromisoformat(
            EXPECTED_GENERATED_AT.replace("Z", "+00:00")
        )
    except ValueError:
        return None
    if EXPECTED_COMMIT_AT is None:
        return None
    try:
        expected_commit_time = datetime.fromisoformat(
            EXPECTED_COMMIT_AT.replace("Z", "+00:00")
        )
    except ValueError:
        return None
    if commit_time != expected_commit_time or generated_time < commit_time:
        return None
    if repository_is_incomplete(root):
        missing = package_objects_missing(root)
        if missing is None:
            return None
        if missing:
            return "unavailable"
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
                "--no-replace-objects",
                "-C",
                str(root),
                "show",
                EXPECTED_COMMIT + ":llmtracefx/" + relative_text,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            env=GIT_ENV,
        )
        if result.returncode != 0:
            return None
        content = normalized_source(relative_text, result.stdout)
        relative = relative_text.encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    if "sha256:" + digest.hexdigest() != EXPECTED_PACKAGE_DIGEST:
        return None
    return "verified"


def resolve_package(bundle: Path, requested: Path | None) -> tuple[Path, str]:
    candidates = [requested.resolve()] if requested is not None else []
    if requested is None:
        candidates.extend((bundle.resolve(), *bundle.resolve().parents))
    for root in candidates:
        package = root / "llmtracefx"
        if package.is_symlink() or not package.is_dir():
            continue
        temporary, digest = snapshot_package(root)
        corroboration = commit_matches(root)
        if digest == EXPECTED_PACKAGE_DIGEST and corroboration is not None:
            return temporary, corroboration
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
    snapshot_root, corroboration = resolve_package(bundle, args.package_root)
    sys.path.insert(0, str(snapshot_root))
    from llmtracefx.cache_audit.bundle import verify_bundle

    result = verify_bundle(bundle)
    result["repository_chronology_corroboration"] = corroboration
    print(json.dumps({"verified": True, **result}, sort_keys=True))


if __name__ == "__main__":
    main()
