"""Deterministic indexing, verification, and rendering for public evidence."""

from __future__ import annotations

import hashlib
import html
import json
import math
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from llmtracefx.brand import LOCKUP_SVG, TOKENS_CSS
from llmtracefx.optimizer._artifact_io import (
    MAX_EVIDENCE_ARTIFACT_BYTES,
    MAX_METADATA_ARTIFACT_BYTES,
    read_bounded_regular_bytes,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from llmtracefx.optimizer.collectors._shared import atomic_write_text

from .registry import (
    ADAPTERS,
    CATALOG_SCHEMA_VERSION,
    CLAIM_DIMENSIONS,
    LEGACY_PINNED_SHA256,
    SOURCES,
)

MAX_CATALOG_BYTES = 2 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_ITEMS = 250_000
MAX_STRING_LENGTH = 32 * 1024
CATALOG_FILES = (
    "README.md",
    "catalog.json",
    "catalog.schema.json",
    "claim-matrix.json",
    "graph.dot",
    "graph.json",
    "graph.svg",
    "index.html",
    "registry.json",
)
CATALOG_PUBLIC_FILES = (*CATALOG_FILES, "SHA256SUMS")
OUTCOMES = {"completed", "oom", "refused", "comparison"}
STATUSES = {"verified"}
CLAIM_STATES = {"supported", "unsupported", "not_applicable"}
SUPPORTED_BUNDLE_SCHEMA_VERSIONS = {"1", "2"}
RELATIONS = {
    "derived_from",
    "compares",
    "same_model_as",
    "uses_workload_contract",
    "supersedes",
}
KINDS = {
    "metal_attribution",
    "model_lab",
    "fit_frontier",
    "oom_autopsy",
    "positive_control",
    "hosted_comparison",
    "provider_preflight",
    "compile_break_even",
    "cache_truth_audit",
}
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
SAFE_PATH_SEGMENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
CHECKSUM_LINE = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._-]*)$")
PRIVACY_PATTERNS = (
    (re.compile(r"/Users/|/home/|[A-Za-z]:\\Users\\"), "private home path"),
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "email address"),
    (
        re.compile(
            r"\b(?:gh[pousr]_|github_pat_|sk-|hf[_-]|wk-|ws-)" r"[A-Za-z0-9_-]{8,}\b"
        ),
        "secret-shaped token",
    ),
    (
        re.compile(
            r"\b[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}\b",
            re.IGNORECASE,
        ),
        "UUID",
    ),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
)
PRIVATE_JSON_KEYS = {
    "absolute_path",
    "account_id",
    "account_identifier",
    "api_key",
    "cache_path",
    "cookie",
    "email",
    "home",
    "host_name",
    "hostname",
    "local_path",
    "model_path",
    "pid",
    "process_id",
    "raw_prompt",
    "raw_response",
    "reasoning_text",
    "user_id",
    "user_name",
    "username",
}
SCRIPT_ADAPTERS = {
    "metal_public_v1": (
        "examples/metal_evidence/evidence_demo.py",
        ("verify", "--public-dir", "{bundle}"),
    ),
    "oom_autopsy_v1": (
        "examples/optimizer/m5-pro-qwen3.8-27b-oom-autopsy/evidence_bundle.py",
        ("verify", "--public-dir", "{bundle}"),
    ),
    "openrouter_glm_v1": (
        "examples/optimizer/openrouter-glm-2k/evidence_bundle.py",
        ("verify",),
    ),
    "modal_preflight_v1": (
        "examples/optimizer/modal-glm53flash-preflight/evidence_bundle.py",
        ("verify",),
    ),
    "cloudrift_preflight_v1": (
        "examples/optimizer/cloudrift-glm53flash-preflight/evidence_bundle.py",
        ("verify",),
    ),
    "cloudrift_compile_v1": (
        "examples/optimizer/qwen3-8b-vllm-compile-break-even/evidence_bundle.py",
        ("verify",),
    ),
    "vllm_crossover_protocol_v1": (
        "examples/optimizer/qwen3-8b-vllm-crossover-protocol/evidence_bundle.py",
        ("verify",),
    ),
    "vllm_crossover_results_v1": (
        "llmtracefx/evidence/vllm_crossover_results_verifier.py",
        ("verify", "--bundle", "{bundle}"),
    ),
    "modal_l4_crossover_protocol_v1": (
        "llmtracefx/evidence/modal_l4_crossover_verifier.py",
        ("verify-protocol", "--bundle", "{bundle}"),
    ),
    "modal_l4_crossover_results_v1": (
        "llmtracefx/evidence/modal_l4_crossover_verifier.py",
        ("verify-results", "--bundle", "{bundle}"),
    ),
}


class CatalogError(ValueError):
    """Raised when catalog data or a referenced evidence bundle is unsafe."""


def canonical_json(value: Any) -> str:
    """Return the repository's canonical, human-readable JSON form."""
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def _json_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], context: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise CatalogError(
            f"{context} keys differ; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _require_string(
    value: Any, context: str, *, allow_none: bool = False
) -> str | None:
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value or len(value) > MAX_STRING_LENGTH:
        raise CatalogError(f"{context} must be a non-empty bounded string")
    return value


def _require_string_list(value: Any, context: str) -> list[str]:
    if not isinstance(value, list) or len(value) > MAX_JSON_ITEMS:
        raise CatalogError(f"{context} must be a bounded list")
    result: list[str] = []
    for index, item in enumerate(value):
        text = _require_string(item, f"{context}[{index}]")
        assert text is not None
        result.append(text)
    return result


def _walk_json(value: Any, context: str = "$", depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise CatalogError(f"{context} exceeds maximum JSON depth")
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise CatalogError(f"{context} contains a non-finite number")
        return
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            raise CatalogError(f"{context} contains an oversized string")
        return
    if isinstance(value, list):
        if len(value) > MAX_JSON_ITEMS:
            raise CatalogError(f"{context} contains too many list items")
        for index, item in enumerate(value):
            _walk_json(item, f"{context}[{index}]", depth + 1)
        return
    if isinstance(value, dict):
        if len(value) > MAX_JSON_ITEMS:
            raise CatalogError(f"{context} contains too many object members")
        for key, item in value.items():
            if not isinstance(key, str) or len(key) > 256:
                raise CatalogError(f"{context} contains an invalid object key")
            _walk_json(item, f"{context}.{key}", depth + 1)
        return
    raise CatalogError(
        f"{context} contains unsupported JSON type {type(value).__name__}"
    )


def _load_json(path: Path, max_bytes: int = MAX_METADATA_ARTIFACT_BYTES) -> Any:
    try:
        text = read_bounded_regular_text(path, max_bytes)
        value = json.loads(text, parse_constant=reject_non_finite_json_constant)
    except RecursionError as exc:
        raise CatalogError(f"{path.name} exceeds maximum JSON depth") from exc
    except (OSError, UnicodeError, ValueError) as exc:
        raise CatalogError(
            f"could not load bounded strict JSON {path.name}: {exc}"
        ) from exc
    _walk_json(value)
    return value


def _validate_relative_path(value: str, context: str) -> tuple[str, ...]:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 512
        or "\\" in value
        or "\x00" in value
        or value.startswith("/")
        or re.match(r"^[A-Za-z]:", value) is not None
    ):
        raise CatalogError(f"{context} is not a safe relative path")
    raw_parts = value.split("/")
    if any(
        part in ("", ".", "..") or SAFE_PATH_SEGMENT.fullmatch(part) is None
        for part in raw_parts
    ):
        raise CatalogError(f"{context} is not a contained relative path")
    return tuple(raw_parts)


def _resolve_contained(
    repo_root: Path, relative: str, *, directory: bool = False
) -> Path:
    parts = _validate_relative_path(relative, "public_path")
    root = repo_root.resolve(strict=True)
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise CatalogError(f"{relative} contains a symlink")
    try:
        resolved = current.resolve(strict=True)
    except OSError as exc:
        raise CatalogError(f"{relative} does not exist") from exc
    if resolved != root and root not in resolved.parents:
        raise CatalogError(f"{relative} escapes the repository root")
    if directory and not resolved.is_dir():
        raise CatalogError(f"{relative} must be a directory")
    return resolved


def _scan_privacy(name: str, text: str) -> None:
    for pattern, description in PRIVACY_PATTERNS:
        if pattern.search(text):
            raise CatalogError(f"{name} contains {description}")


def _scan_json_privacy(value: Any, context: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in PRIVATE_JSON_KEYS:
                raise CatalogError(f"{context}.{key} is a private evidence field")
            _scan_json_privacy(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_json_privacy(item, f"{context}[{index}]")


def _artifact_set_hash(
    repo_root: Path, source: Mapping[str, Any], *, scan_privacy: bool = True
) -> str:
    bundle = _resolve_contained(repo_root, source["public_path"], directory=True)
    expected = tuple(source["artifact_files"])
    if not expected or len(set(expected)) != len(expected):
        raise CatalogError(f"{source['evidence_id']} has an invalid artifact allowlist")
    actual: list[str] = []
    for child in bundle.iterdir():
        if child.is_symlink() or not child.is_file():
            raise CatalogError(
                f"{source['evidence_id']} contains a non-regular artifact"
            )
        actual.append(child.name)
    if sorted(actual) != sorted(expected):
        raise CatalogError(
            f"{source['evidence_id']} artifact allowlist drifted; "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"unexpected={sorted(set(actual) - set(expected))}"
        )
    inventory: list[dict[str, Any]] = []
    for name in sorted(expected):
        _validate_relative_path(name, "artifact filename")
        if len(Path(name).parts) != 1:
            raise CatalogError("artifact allowlists may contain only filenames")
        path = bundle / name
        try:
            data = read_bounded_regular_bytes(path, MAX_EVIDENCE_ARTIFACT_BYTES)
        except (OSError, ValueError) as exc:
            raise CatalogError(
                f"{source['evidence_id']} has unsafe artifact {name}"
            ) from exc
        if scan_privacy:
            try:
                _scan_privacy(name, data.decode("utf-8"))
            except UnicodeDecodeError:
                pass
        if name.endswith(".json"):
            _scan_json_privacy(
                _load_json(path, MAX_EVIDENCE_ARTIFACT_BYTES),
                f"{source['evidence_id']}.{name}",
            )
        inventory.append({"name": name, "bytes": len(data), "sha256": _sha256(data)})
    return _json_hash(inventory)


def _parse_sha256sums(bundle: Path) -> dict[str, str]:
    text = read_bounded_regular_text(bundle / "SHA256SUMS", MAX_METADATA_ARTIFACT_BYTES)
    parsed: dict[str, str] = {}
    for line in text.splitlines():
        match = CHECKSUM_LINE.fullmatch(line)
        if match is None:
            raise CatalogError("SHA256SUMS contains a malformed line")
        digest, name = match.groups()
        if name in parsed:
            raise CatalogError(f"SHA256SUMS repeats {name}")
        parsed[name] = digest
    return parsed


def _verify_sha256_allowlist(repo_root: Path, source: Mapping[str, Any]) -> None:
    bundle = _resolve_contained(repo_root, source["public_path"], directory=True)
    expected_names = set(source["artifact_files"]) - {"SHA256SUMS", "README.md"}
    recorded = _parse_sha256sums(bundle)
    if set(recorded) != expected_names:
        raise CatalogError(
            f"{source['evidence_id']} SHA256SUMS allowlist does not match"
        )
    for name, expected in recorded.items():
        actual = _sha256(
            read_bounded_regular_bytes(bundle / name, MAX_EVIDENCE_ARTIFACT_BYTES)
        )
        if actual != expected:
            raise CatalogError(f"{source['evidence_id']} checksum mismatch for {name}")


def _verify_legacy_pins(repo_root: Path, source: Mapping[str, Any]) -> None:
    for name in source["artifact_files"]:
        relative = f"{source['public_path']}/{name}"
        expected = LEGACY_PINNED_SHA256.get(relative)
        if expected is None:
            raise CatalogError(f"{source['evidence_id']} lacks a closed historical pin")
        path = _resolve_contained(repo_root, relative)
        actual = _sha256(read_bounded_regular_bytes(path, MAX_EVIDENCE_ARTIFACT_BYTES))
        if actual != expected:
            raise CatalogError(f"{source['evidence_id']} historical artifact changed")


def _run_script_verifier(repo_root: Path, source: Mapping[str, Any]) -> None:
    adapter = source["adapter"]
    script_relative, argument_template = SCRIPT_ADAPTERS[adapter]
    bundle = _resolve_contained(repo_root, source["public_path"], directory=True)
    script = _resolve_contained(repo_root, script_relative)
    arguments = tuple(
        str(bundle) if value == "{bundle}" else value for value in argument_template
    )
    command = (
        sys.executable,
        "-I",
        str(Path(__file__).with_name("_offline_runner.py").resolve()),
        str(repo_root),
        str(script),
        *arguments,
    )
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
    }
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
            shell=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise CatalogError(f"{source['evidence_id']} verifier could not run") from exc
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).strip().splitlines()
        reason = detail[-1][:240] if detail else "no diagnostic"
        raise CatalogError(
            f"{source['evidence_id']} existing verifier failed: {reason}"
        )


def _verify_source_bindings(repo_root: Path, source: Mapping[str, Any]) -> None:
    bundle = _resolve_contained(repo_root, source["public_path"], directory=True)
    evidence_id = source["evidence_id"]
    schema: Any
    captured: Any
    source_commit: Any = None
    model_id: Any = None
    model_revision: Any = None
    if evidence_id == "metal-attribution-m5-pro-20260831":
        manifest = _load_json(bundle / "experiment-manifest.json")
        schema = manifest["schema_version"]
        captured = manifest["captured_at"]
        if manifest["environment"]["hardware"] != source["hardware"]["system"]:
            raise CatalogError(f"{evidence_id} hardware binding drifted")
    elif evidence_id == "qwen38-27b-m5-pro-lab-oom-20260831":
        manifest = _load_json(bundle / "evidence-summary.json")
        schema = manifest["schema_version"]
        captured = manifest["generated_at"]
        model_id = manifest["model"]["repository_id"]
        model_revision = manifest["model"]["revision"]
    elif evidence_id == "qwen38-27b-m5-pro-fit-frontier-20260901":
        manifest = _load_json(bundle / "fit-frontier-summary.json")
        schema = manifest["schema_version"]
        captured = manifest["generated_at"]
        model_id = manifest["model"]["repository_id"]
        model_revision = manifest["model"]["revision"]
    elif evidence_id == "qwen38-27b-m5-pro-clean-boot-autopsy-20260901":
        manifest = _load_json(bundle / "evidence-manifest.json")
        schema = manifest["schema_version"]
        captured = manifest["run"]["completed_at_utc"]
        source_commit = manifest["run"]["code_checkout_commit"]
        model_id = manifest["model"]["repository_id"]
        model_revision = manifest["model"]["revision"]
    elif evidence_id == "qwen3-8b-m5-pro-control-20260902":
        manifest = _load_json(bundle / "evidence-manifest.json")
        schema = manifest["schema_version"]
        captured = manifest["run"]["completed_at_utc"]
        source_commit = manifest["run"]["code_checkout_commit"]
        model_id = manifest["model"]["official_id"]
        model_revision = manifest["model"]["official_revision"]
    elif evidence_id == "openrouter-glm-2k-comparison-20260902":
        manifest = _load_json(bundle / "experiment-manifest.json")
        comparison = _load_json(bundle / "comparison.json")
        generation = _load_json(bundle / "generation-metadata.json")
        schema = manifest["schema_version"]
        captured = comparison["generated_at"]
        source_commit = manifest["run"]["base_checkout_commit"]
        requested_models = sorted(manifest["systems"]["requested_model_ids"])
        resolved_builds = manifest["systems"]["generation_metadata_observations"][
            "resolved_model_builds"
        ]
        observed_builds = {
            observation["requested_model_id"]: observation["resolved_model_build"]
            for observation in generation["observations"]
        }
        if (
            len(observed_builds) != len(generation["observations"])
            or set(resolved_builds) != set(requested_models)
            or observed_builds != resolved_builds
        ):
            raise CatalogError(f"{evidence_id} provider build binding drifted")
        model_id = " and ".join(requested_models)
        model_revision = " and ".join(sorted(resolved_builds.values()))
    elif evidence_id == "modal-glm53flash-preflight-20260902":
        manifest = _load_json(bundle / "experiment-manifest.json")
        schema = manifest["schema_version"]
        captured = manifest["as_of"]
        source_commit = manifest["repository_head"]
        model_id = manifest["model"]["repo_id"]
        model_revision = manifest["model"]["revision"]
    elif evidence_id == "cloudrift-glm53flash-preflight-20260902":
        manifest = _load_json(bundle / "experiment-manifest.json")
        inventory = _load_json(bundle / "model-inventory-reference.json")
        schema = manifest["schema_version"]
        captured = manifest["as_of"]
        source_commit = manifest["repository_base"]
        model_id = inventory["repo_id"]
        model_revision = inventory["revision"]
    elif evidence_id == "qwen3-8b-cloudrift-vllm-compile-20260903":
        contract = _load_json(bundle / "experiment-contract.json")
        inventory = _load_json(bundle / "model-inventory.json")
        lifecycle = [
            json.loads(line)
            for line in read_bounded_regular_text(
                bundle / "lifecycle-records.jsonl", MAX_EVIDENCE_ARTIFACT_BYTES
            ).splitlines()
        ]
        schema = contract["schema_version"]
        captured = lifecycle[-1]["ended_at"]
        source_commit = contract["collection_source_commit"]
        model_id = inventory["model_id"]
        model_revision = inventory["revision"]
    elif evidence_id == "qwen3-8b-vllm-crossover-protocol-20260904":
        contract = _load_json(bundle / "evidence-contract.json")
        plan = _load_json(bundle / "experiment-plan.json")
        schema = contract["schema_version"]
        captured = contract["captured_at"]
        model_id = plan["model"]["id"]
        model_revision = plan["model"]["revision"]
    elif evidence_id == "cache-audit-reference-positive-control-20260905":
        manifest = _load_json(bundle / "audit-manifest.json")
        schema = manifest["schema_version"]
        captured = manifest["created_at"]
        source_commit = manifest["generator_commit"]
        model_id = manifest["model_id"]
        binding = source["cache_binding"]
        observed_binding = {
            "publication_mode": manifest["publication_mode"],
            "backend": manifest["backend"],
            "workload_digest": manifest["workload_digest"],
            "adapter_version": manifest["adapter_version"],
            "generator_package_digest": manifest["generator_package_digest"],
            "privacy_status": "verified_public_synthetic",
        }
        if observed_binding != binding:
            raise CatalogError(f"{evidence_id} cache provenance binding drifted")
    else:  # pragma: no cover - SOURCES is a closed registry
        raise CatalogError(f"{evidence_id} has no source binding contract")
    if str(schema) != source["bundle_schema_version"]:
        raise CatalogError(f"{evidence_id} schema binding drifted")
    if captured != source["captured_at"]:
        raise CatalogError(f"{evidence_id} capture timestamp binding drifted")
    if source_commit != source["source_commit"]:
        raise CatalogError(f"{evidence_id} source commit binding drifted")
    if model_id != source["model"]["id"]:
        raise CatalogError(f"{evidence_id} model identity binding drifted")
    if model_revision != source["model"]["revision"]:
        raise CatalogError(f"{evidence_id} model revision binding drifted")


def verify_source(repo_root: Path, source: Mapping[str, Any]) -> None:
    """Verify one source through its closed adapter and exact artifact allowlist."""
    adapter = source["adapter"]
    if adapter not in ADAPTERS:
        raise CatalogError(f"unknown built-in adapter {adapter!r}")
    _artifact_set_hash(repo_root, source)
    _verify_source_bindings(repo_root, source)
    if adapter == "legacy_pinned_v1":
        _verify_legacy_pins(repo_root, source)
    elif adapter == "sha256_allowlist_v1":
        _verify_sha256_allowlist(repo_root, source)
    elif adapter == "cache_audit_v1":
        from llmtracefx.cache_audit.bundle import verify_bundle

        verify_bundle(
            _resolve_contained(repo_root, source["public_path"], directory=True)
        )
    elif adapter in SCRIPT_ADAPTERS:
        _run_script_verifier(repo_root, source)
    else:  # pragma: no cover - registry and branch are intentionally closed together
        raise CatalogError(f"adapter {adapter!r} has no implementation")


def _discover_candidates(repo_root: Path) -> list[str]:
    examples = _resolve_contained(repo_root, "examples", directory=True)
    markers = {
        "SHA256SUMS",
        "evidence-manifest.json",
        "evidence-summary.json",
        "experiment-manifest.json",
        "fit-frontier-summary.json",
    }
    registered = {
        _resolve_contained(repo_root, source["public_path"], directory=True)
        for source in SOURCES
    }
    generated_catalog = (repo_root / "examples" / "evidence-catalog").resolve()
    candidates: set[Path] = set()
    for path in examples.rglob("*"):
        if path.is_symlink():
            raise CatalogError("examples contains a symlink; candidate scan refused")
        if path.is_file() and path.name in markers:
            parent = path.parent.resolve()
            if parent != generated_catalog and generated_catalog not in parent.parents:
                candidates.add(parent)
    unregistered: list[str] = []
    root = repo_root.resolve()
    for candidate in sorted(candidates):
        if not any(
            candidate == known or known in candidate.parents for known in registered
        ):
            unregistered.append(candidate.relative_to(root).as_posix())
    return unregistered


def _public_registry_document() -> dict[str, Any]:
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "adapters": ADAPTERS,
        "sources": [dict(source) for source in SOURCES],
    }


def build_catalog(repo_root: str | Path) -> dict[str, Any]:
    """Build a deterministic catalog from the closed source registry."""
    root = Path(repo_root).resolve(strict=True)
    entries: list[dict[str, Any]] = []
    for source in SOURCES:
        adapter = ADAPTERS[source["adapter"]]
        entry = {
            key: value
            for key, value in source.items()
            if key not in {"adapter", "artifact_files", "cache_binding"}
        }
        entry["verifier"] = dict(adapter)
        entry["artifact_set_hash"] = _artifact_set_hash(root, source)
        entry["measurements"] = list(entry["measurements"])
        entry["supported_claims"] = list(entry["supported_claims"])
        entry["unsupported_claims"] = list(entry["unsupported_claims"])
        entry["dependencies"] = list(entry["dependencies"])
        entry["limitations"] = list(entry["limitations"])
        entries.append(entry)
    entries.sort(key=lambda item: item["evidence_id"])
    edges = sorted(
        (
            {
                "source": entry["evidence_id"],
                "target": dependency["evidence_id"],
                "relation": dependency["relation"],
            }
            for entry in entries
            for dependency in entry["dependencies"]
        ),
        key=lambda edge: (edge["source"], edge["target"], edge["relation"]),
    )
    body = {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "generator": {
            "name": "llmtracefx-evidence",
            "version": CATALOG_SCHEMA_VERSION,
            "determinism": "source-only; no clock, cwd, host, network, or model state",
        },
        "registry_hash": _json_hash(_public_registry_document()),
        "claim_dimensions": list(CLAIM_DIMENSIONS),
        "entries": entries,
        "edges": edges,
        "unregistered_candidates": _discover_candidates(root),
    }
    catalog = dict(body)
    catalog["catalog_hash"] = _json_hash(body)
    validate_catalog_document(catalog)
    return catalog


def _validate_timestamp(value: Any, context: str) -> None:
    text = _require_string(value, context)
    assert text is not None
    try:
        datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CatalogError(f"{context} is not an ISO-8601 date or timestamp") from exc


def _validate_entry(entry: Any, context: str) -> None:
    if not isinstance(entry, dict):
        raise CatalogError(f"{context} must be an object")
    expected = {
        "evidence_id",
        "kind",
        "status",
        "outcome",
        "public_path",
        "bundle_schema_version",
        "verifier",
        "artifact_set_hash",
        "captured_at",
        "source_commit",
        "model",
        "runtime",
        "hardware",
        "workload",
        "measurements",
        "claims",
        "supported_claims",
        "unsupported_claims",
        "budget",
        "dependencies",
        "limitations",
    }
    _require_exact_keys(entry, expected, context)
    evidence_id = _require_string(entry["evidence_id"], f"{context}.evidence_id")
    if evidence_id is None or SAFE_ID.fullmatch(evidence_id) is None:
        raise CatalogError(f"{context}.evidence_id is invalid")
    if not isinstance(entry["kind"], str) or entry["kind"] not in KINDS:
        raise CatalogError(f"{context}.kind is unknown")
    if not isinstance(entry["status"], str) or entry["status"] not in STATUSES:
        raise CatalogError(f"{context}.status is unknown")
    if not isinstance(entry["outcome"], str) or entry["outcome"] not in OUTCOMES:
        raise CatalogError(f"{context}.outcome is unknown")
    _validate_relative_path(entry["public_path"], f"{context}.public_path")
    if entry["bundle_schema_version"] not in SUPPORTED_BUNDLE_SCHEMA_VERSIONS:
        raise CatalogError(f"{context}.bundle_schema_version is unsupported")
    if not isinstance(entry["verifier"], dict):
        raise CatalogError(f"{context}.verifier must be an object")
    _require_exact_keys(entry["verifier"], {"name", "version"}, f"{context}.verifier")
    _require_string(entry["verifier"]["name"], f"{context}.verifier.name")
    _require_string(entry["verifier"]["version"], f"{context}.verifier.version")
    if entry["verifier"] not in ADAPTERS.values():
        raise CatalogError(f"{context}.verifier is not in the closed registry")
    if not isinstance(entry["artifact_set_hash"], str) or not SHA256.fullmatch(
        entry["artifact_set_hash"]
    ):
        raise CatalogError(f"{context}.artifact_set_hash is invalid")
    _validate_timestamp(entry["captured_at"], f"{context}.captured_at")
    source_commit = entry["source_commit"]
    if source_commit is not None and (
        not isinstance(source_commit, str) or COMMIT.fullmatch(source_commit) is None
    ):
        raise CatalogError(f"{context}.source_commit is invalid")
    for field, keys in (
        ("model", {"id", "revision", "quantization"}),
        ("runtime", {"name", "version", "provider"}),
        ("hardware", {"system", "architecture"}),
        ("workload", {"identity", "context", "request"}),
    ):
        value = entry[field]
        if not isinstance(value, dict):
            raise CatalogError(f"{context}.{field} must be an object")
        _require_exact_keys(value, keys, f"{context}.{field}")
        for key in keys:
            _require_string(value[key], f"{context}.{field}.{key}", allow_none=True)
    measurements = entry["measurements"]
    if not isinstance(measurements, list) or not measurements or len(measurements) > 64:
        raise CatalogError(f"{context}.measurements must be a non-empty bounded list")
    for index, measurement in enumerate(measurements):
        if not isinstance(measurement, dict):
            raise CatalogError(f"{context}.measurements[{index}] must be an object")
        _require_exact_keys(
            measurement, {"scope", "provenance"}, f"{context}.measurements[{index}]"
        )
        _require_string(measurement["scope"], f"{context}.measurements[{index}].scope")
        _require_string(
            measurement["provenance"],
            f"{context}.measurements[{index}].provenance",
        )
    claims = entry["claims"]
    if not isinstance(claims, dict):
        raise CatalogError(f"{context}.claims must be an object")
    _require_exact_keys(claims, set(CLAIM_DIMENSIONS), f"{context}.claims")
    for dimension, claim in claims.items():
        if not isinstance(claim, dict):
            raise CatalogError(f"{context}.claims.{dimension} must be an object")
        _require_exact_keys(
            claim, {"state", "provenance"}, f"{context}.claims.{dimension}"
        )
        if not isinstance(claim["state"], str) or claim["state"] not in CLAIM_STATES:
            raise CatalogError(f"{context}.claims.{dimension}.state is unknown")
        _require_string(claim["provenance"], f"{context}.claims.{dimension}.provenance")
    if not _require_string_list(
        entry["supported_claims"], f"{context}.supported_claims"
    ):
        raise CatalogError(f"{context}.supported_claims must not be empty")
    if not _require_string_list(
        entry["unsupported_claims"], f"{context}.unsupported_claims"
    ):
        raise CatalogError(f"{context}.unsupported_claims must not be empty")
    if not _require_string_list(entry["limitations"], f"{context}.limitations"):
        raise CatalogError(f"{context}.limitations must not be empty")
    budget = entry["budget"]
    if not isinstance(budget, dict):
        raise CatalogError(f"{context}.budget must be an object")
    _require_exact_keys(
        budget,
        {
            "scope",
            "authorized_usd",
            "planned_usd",
            "reported_usd",
            "inferred_usd",
            "limitation",
        },
        f"{context}.budget",
    )
    _require_string(budget["scope"], f"{context}.budget.scope")
    _require_string(budget["limitation"], f"{context}.budget.limitation")
    for field in ("authorized_usd", "planned_usd", "reported_usd", "inferred_usd"):
        value = budget[field]
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise CatalogError(
                f"{context}.budget.{field} must be finite and nonnegative"
            )
    dependencies = entry["dependencies"]
    if not isinstance(dependencies, list) or len(dependencies) > 128:
        raise CatalogError(f"{context}.dependencies must be a bounded list")
    seen_dependencies: set[tuple[str, str]] = set()
    for index, dependency in enumerate(dependencies):
        if not isinstance(dependency, dict):
            raise CatalogError(f"{context}.dependencies[{index}] must be an object")
        _require_exact_keys(
            dependency,
            {"evidence_id", "relation"},
            f"{context}.dependencies[{index}]",
        )
        target = _require_string(
            dependency["evidence_id"],
            f"{context}.dependencies[{index}].evidence_id",
        )
        relation = dependency["relation"]
        if not isinstance(relation, str) or relation not in RELATIONS:
            raise CatalogError(f"{context}.dependencies[{index}].relation is unknown")
        dependency_key = (str(target), relation)
        if dependency_key in seen_dependencies:
            raise CatalogError(f"{context} repeats a dependency")
        seen_dependencies.add(dependency_key)


def _validate_graph(entries: list[dict[str, Any]], edges: Any) -> None:
    ids = [entry["evidence_id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise CatalogError("catalog contains duplicate evidence IDs")
    known = set(ids)
    expected_edges = sorted(
        (
            {
                "source": entry["evidence_id"],
                "target": dependency["evidence_id"],
                "relation": dependency["relation"],
            }
            for entry in entries
            for dependency in entry["dependencies"]
        ),
        key=lambda edge: (edge["source"], edge["target"], edge["relation"]),
    )
    if edges != expected_edges:
        raise CatalogError("top-level edges do not exactly match entry dependencies")
    adjacency: dict[str, list[str]] = {evidence_id: [] for evidence_id in ids}
    for edge in expected_edges:
        if edge["source"] not in known or edge["target"] not in known:
            raise CatalogError("catalog contains a dangling dependency edge")
        if edge["source"] == edge["target"]:
            raise CatalogError("catalog contains a self dependency")
        adjacency[edge["source"]].append(edge["target"])
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise CatalogError("catalog dependency graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for target in adjacency[node]:
            visit(target)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(adjacency):
        visit(node)
    hash_to_path: dict[str, str] = {}
    path_to_hash: dict[str, str] = {}
    for entry in entries:
        digest = entry["artifact_set_hash"]
        path = entry["public_path"]
        if digest in hash_to_path and hash_to_path[digest] != path:
            raise CatalogError("conflicting paths share one artifact content identity")
        if path in path_to_hash and path_to_hash[path] != digest:
            raise CatalogError("one public path has conflicting content identities")
        hash_to_path[digest] = path
        path_to_hash[path] = digest


def validate_catalog_document(catalog: Any) -> None:
    """Strictly validate an untrusted catalog document."""
    _walk_json(catalog)
    if not isinstance(catalog, dict):
        raise CatalogError("catalog must be an object")
    _require_exact_keys(
        catalog,
        {
            "schema_version",
            "generator",
            "registry_hash",
            "claim_dimensions",
            "entries",
            "edges",
            "unregistered_candidates",
            "catalog_hash",
        },
        "catalog",
    )
    if catalog["schema_version"] != CATALOG_SCHEMA_VERSION:
        raise CatalogError("unsupported catalog schema version")
    generator = catalog["generator"]
    if not isinstance(generator, dict):
        raise CatalogError("catalog.generator must be an object")
    _require_exact_keys(
        generator, {"name", "version", "determinism"}, "catalog.generator"
    )
    if generator["name"] != "llmtracefx-evidence":
        raise CatalogError("unknown catalog generator")
    _require_string(generator["version"], "catalog.generator.version")
    _require_string(generator["determinism"], "catalog.generator.determinism")
    if not isinstance(catalog["registry_hash"], str) or not SHA256.fullmatch(
        catalog["registry_hash"]
    ):
        raise CatalogError("catalog.registry_hash is invalid")
    if catalog["claim_dimensions"] != list(CLAIM_DIMENSIONS):
        raise CatalogError("catalog claim dimensions drifted")
    entries = catalog["entries"]
    if not isinstance(entries, list) or not entries or len(entries) > 256:
        raise CatalogError("catalog.entries must be a non-empty bounded list")
    for index, entry in enumerate(entries):
        _validate_entry(entry, f"catalog.entries[{index}]")
    if entries != sorted(entries, key=lambda entry: entry["evidence_id"]):
        raise CatalogError("catalog entries are not deterministically ordered")
    unregistered = _require_string_list(
        catalog["unregistered_candidates"], "catalog.unregistered_candidates"
    )
    if unregistered != sorted(set(unregistered)):
        raise CatalogError("unregistered candidates are not unique and sorted")
    for index, path in enumerate(unregistered):
        _validate_relative_path(path, f"catalog.unregistered_candidates[{index}]")
    _validate_graph(entries, catalog["edges"])
    recorded_hash = catalog["catalog_hash"]
    if not isinstance(recorded_hash, str) or not SHA256.fullmatch(recorded_hash):
        raise CatalogError("catalog.catalog_hash is invalid")
    body = dict(catalog)
    del body["catalog_hash"]
    if recorded_hash != _json_hash(body):
        raise CatalogError("catalog hash does not verify")
    _scan_privacy("catalog.json", canonical_json(catalog))


def _claim_matrix(catalog: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "dimensions": list(CLAIM_DIMENSIONS),
        "rows": [
            {
                "evidence_id": entry["evidence_id"],
                "outcome": entry["outcome"],
                "claims": entry["claims"],
            }
            for entry in catalog["entries"]
        ],
    }


def _graph_document(catalog: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CATALOG_SCHEMA_VERSION,
        "catalog_hash": catalog["catalog_hash"],
        "nodes": [
            {
                "evidence_id": entry["evidence_id"],
                "kind": entry["kind"],
                "status": entry["status"],
                "outcome": entry["outcome"],
                "model": entry["model"]["id"],
                "system": entry["hardware"]["system"],
                "supported_claim_scope": [
                    dimension
                    for dimension in CLAIM_DIMENSIONS
                    if entry["claims"][dimension]["state"] == "supported"
                ],
            }
            for entry in catalog["entries"]
        ],
        "edges": catalog["edges"],
    }


def _dot(graph: Mapping[str, Any]) -> str:
    def quote(value: str) -> str:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    colors = {
        "completed": "#17513a",
        "oom": "#8c1d28",
        "refused": "#4a5157",
        "comparison": "#c23d16",
    }
    lines = [
        "digraph evidence {",
        '  graph [bgcolor="#fbfaf7", rankdir=LR];',
        '  node [shape=box, style="rounded,filled", fillcolor="#f4f1ea", '
        'fontname="Helvetica", color="#16181a"];',
        '  edge [fontname="Helvetica", color="#5b6167"];',
    ]
    for node in graph["nodes"]:
        label = (
            f"{node['evidence_id']}\\n{node['outcome']} / {node['status']}\\n"
            f"{node['model'] or 'no model'}\\n"
            f"claims: {', '.join(node['supported_claim_scope']) or 'none'}"
        )
        lines.append(
            f"  {quote(node['evidence_id'])} "
            f"[label={quote(label)}, fontcolor={quote(colors[node['outcome']])}];"
        )
    for edge in graph["edges"]:
        lines.append(
            f"  {quote(edge['source'])} -> {quote(edge['target'])} "
            f"[label={quote(edge['relation'])}];"
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


def _svg(graph: Mapping[str, Any]) -> str:
    width = 1220
    node_width = 520
    node_height = 126
    x_positions = (55, 645)
    y_step = 174
    positions: dict[str, tuple[int, int]] = {}
    for index, node in enumerate(graph["nodes"]):
        positions[node["evidence_id"]] = (
            x_positions[index % 2],
            72 + (index // 2) * y_step,
        )
    height = 72 + ((len(graph["nodes"]) + 1) // 2) * y_step + 120
    outcome_color = {
        "completed": "#17513a",
        "oom": "#8c1d28",
        "refused": "#4a5157",
        "comparison": "#c23d16",
    }
    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">LLMTraceFX evidence lineage graph</title>',
        f'<desc id="desc">{len(graph["nodes"])} verified public evidence bundles '
        f"and {len(graph['edges'])} typed metadata-backed lineage edges.</desc>",
        '<rect width="100%" height="100%" fill="#f4f1ea"/>',
        '<text x="55" y="42" fill="#16181a" font-family="system-ui,sans-serif" '
        'font-size="26" font-weight="700">Evidence lineage</text>',
    ]
    for edge in graph["edges"]:
        sx, sy = positions[edge["source"]]
        tx, ty = positions[edge["target"]]
        x1 = sx + (0 if tx < sx else node_width)
        x2 = tx + (node_width if tx < sx else 0)
        y1 = sy + node_height // 2
        y2 = ty + node_height // 2
        parts.extend(
            (
                f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                'stroke="#5b6167" stroke-width="2"/>',
                f'<text x="{(x1 + x2) / 2:.1f}" y="{(y1 + y2) / 2 - 6:.1f}" '
                'text-anchor="middle" fill="#5b6167" '
                'font-family="ui-monospace,monospace" font-size="11">'
                f"{html.escape(edge['relation'])}</text>",
            )
        )
    for node in graph["nodes"]:
        x, y = positions[node["evidence_id"]]
        color = outcome_color[node["outcome"]]
        claims = ", ".join(node["supported_claim_scope"]) or "none"
        model = node["model"] or "no model"
        parts.extend(
            (
                f'<rect x="{x}" y="{y}" width="{node_width}" height="{node_height}" '
                f'rx="8" fill="#fbfaf7" stroke="{color}" stroke-width="3"/>',
                f'<text x="{x + 18}" y="{y + 28}" fill="#16181a" '
                'font-family="ui-monospace,monospace" font-size="14" '
                f'font-weight="700">{html.escape(node["evidence_id"])}</text>',
                f'<text x="{x + 18}" y="{y + 53}" fill="{color}" '
                'font-family="system-ui,sans-serif" font-size="14">'
                f"{html.escape(node['outcome'])} / {html.escape(node['status'])}</text>",
                f'<text x="{x + 18}" y="{y + 78}" fill="#5b6167" '
                'font-family="system-ui,sans-serif" font-size="13">'
                f"{html.escape(model[:68])}</text>",
                f'<text x="{x + 18}" y="{y + 103}" fill="#5b6167" '
                'font-family="system-ui,sans-serif" font-size="12">'
                f"supported: {html.escape(claims)}</text>",
            )
        )
    parts.append("</svg>\n")
    return "".join(parts)


def _html(catalog: Mapping[str, Any], graph_svg: str) -> str:
    matrix = _claim_matrix(catalog)
    rows: list[str] = []
    for row in matrix["rows"]:
        cells = "".join(
            "<td><strong>"
            + html.escape(row["claims"][dimension]["state"])
            + "</strong><small>"
            + html.escape(row["claims"][dimension]["provenance"])
            + "</small></td>"
            for dimension in CLAIM_DIMENSIONS
        )
        rows.append(
            '<tr><th scope="row">'
            + html.escape(row["evidence_id"])
            + "<small>"
            + html.escape(row["outcome"])
            + "</small></th>"
            + cells
            + "</tr>"
        )
    headers = "".join(
        f'<th scope="col">{html.escape(dimension.replace("_", " "))}</th>'
        for dimension in CLAIM_DIMENSIONS
    )
    tokens = TOKENS_CSS
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLMTraceFX evidence catalog</title>
<style>
{tokens}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--field); color: var(--ink); font-family: var(--sans); }}
main {{ max-width: 1480px; margin: 0 auto; padding: 32px; }}
.lockup {{ width: 260px; color: var(--ink); }}
.lede {{ max-width: 78ch; color: var(--muted); }}
.panel {{ margin-top: 28px; padding: 24px; background: var(--sheet); border: 1px solid var(--rule); }}
.graph svg {{ width: 100%; height: auto; }}
.table-wrap {{ overflow-x: auto; }}
table {{ border-collapse: collapse; min-width: 1320px; width: 100%; }}
th, td {{ border-bottom: 1px solid var(--rule); padding: 10px; text-align: left; vertical-align: top; }}
thead th {{ color: var(--signal); }}
tbody th {{ width: 260px; font-family: var(--mono); }}
small {{ display: block; margin-top: 5px; color: var(--muted); font-weight: 400; }}
strong {{ color: var(--ink); }}
code {{ font-family: var(--mono); }}
</style>
</head>
<body><main>
{LOCKUP_SVG}
<h1>Offline evidence catalog</h1>
<p class="lede">A deterministic index of {len(catalog["entries"])} verified public bundles. Claim cells
state supported, unsupported, or not applicable; absence is never converted to yes.</p>
<p><code>{html.escape(catalog["catalog_hash"])}</code></p>
<section class="panel graph" aria-labelledby="graph-title">
<h2 id="graph-title">Lineage graph</h2>
{graph_svg}
</section>
<section class="panel" aria-labelledby="matrix-title">
<h2 id="matrix-title">Claim matrix</h2>
<div class="table-wrap"><table>
<thead><tr><th scope="col">Evidence</th>{headers}</tr></thead>
<tbody>{"".join(rows)}</tbody>
</table></div>
</section>
</main></body></html>
"""


def _schema_document() -> dict[str, Any]:
    string_or_null = {
        "type": ["string", "null"],
        "minLength": 1,
        "maxLength": MAX_STRING_LENGTH,
    }
    claim = {
        "type": "object",
        "additionalProperties": False,
        "required": ["state", "provenance"],
        "properties": {
            "state": {"enum": sorted(CLAIM_STATES)},
            "provenance": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_STRING_LENGTH,
            },
        },
    }
    claims = {
        "type": "object",
        "additionalProperties": False,
        "required": list(CLAIM_DIMENSIONS),
        "properties": dict.fromkeys(CLAIM_DIMENSIONS, claim),
    }
    path_pattern = (
        r"^(?![A-Za-z]:)[A-Za-z0-9][A-Za-z0-9._-]*" r"(?:/[A-Za-z0-9][A-Za-z0-9._-]*)*$"
    )
    entry_required = [
        "evidence_id",
        "kind",
        "status",
        "outcome",
        "public_path",
        "bundle_schema_version",
        "verifier",
        "artifact_set_hash",
        "captured_at",
        "source_commit",
        "model",
        "runtime",
        "hardware",
        "workload",
        "measurements",
        "claims",
        "supported_claims",
        "unsupported_claims",
        "budget",
        "dependencies",
        "limitations",
    ]
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://github.com/Siddhant-K-code/LLMTraceFX/evidence-catalog-v1",
        "title": "LLMTraceFX offline evidence catalog",
        "type": "object",
        "additionalProperties": False,
        "$defs": {
            "nonEmptyString": {
                "type": "string",
                "minLength": 1,
                "maxLength": MAX_STRING_LENGTH,
            },
            "stringList": {
                "type": "array",
                "items": {"$ref": "#/$defs/nonEmptyString"},
                "minItems": 1,
                "maxItems": MAX_JSON_ITEMS,
            },
            "verifier": {
                "oneOf": [
                    {
                        "const": adapter,
                    }
                    for adapter in sorted(
                        ADAPTERS.values(),
                        key=lambda item: (item["name"], item["version"]),
                    )
                ],
            },
            "model": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "revision", "quantization"],
                "properties": {
                    "id": string_or_null,
                    "revision": string_or_null,
                    "quantization": string_or_null,
                },
            },
            "runtime": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "version", "provider"],
                "properties": {
                    "name": string_or_null,
                    "version": string_or_null,
                    "provider": string_or_null,
                },
            },
            "hardware": {
                "type": "object",
                "additionalProperties": False,
                "required": ["system", "architecture"],
                "properties": {
                    "system": string_or_null,
                    "architecture": string_or_null,
                },
            },
            "workload": {
                "type": "object",
                "additionalProperties": False,
                "required": ["identity", "context", "request"],
                "properties": {
                    "identity": string_or_null,
                    "context": string_or_null,
                    "request": string_or_null,
                },
            },
            "measurement": {
                "type": "object",
                "additionalProperties": False,
                "required": ["scope", "provenance"],
                "properties": {
                    "scope": {"$ref": "#/$defs/nonEmptyString"},
                    "provenance": {"$ref": "#/$defs/nonEmptyString"},
                },
            },
            "dependency": {
                "type": "object",
                "additionalProperties": False,
                "required": ["evidence_id", "relation"],
                "properties": {
                    "evidence_id": {
                        "type": "string",
                        "pattern": SAFE_ID.pattern,
                    },
                    "relation": {"enum": sorted(RELATIONS)},
                },
            },
            "budget": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "scope",
                    "authorized_usd",
                    "planned_usd",
                    "reported_usd",
                    "inferred_usd",
                    "limitation",
                ],
                "properties": {
                    "scope": {"$ref": "#/$defs/nonEmptyString"},
                    "authorized_usd": {
                        "type": ["number", "null"],
                        "minimum": 0,
                    },
                    "planned_usd": {"type": ["number", "null"], "minimum": 0},
                    "reported_usd": {"type": ["number", "null"], "minimum": 0},
                    "inferred_usd": {"type": ["number", "null"], "minimum": 0},
                    "limitation": {"$ref": "#/$defs/nonEmptyString"},
                },
            },
            "entry": {
                "type": "object",
                "additionalProperties": False,
                "required": entry_required,
                "properties": {
                    "evidence_id": {
                        "type": "string",
                        "pattern": SAFE_ID.pattern,
                    },
                    "kind": {"enum": sorted(KINDS)},
                    "status": {"enum": sorted(STATUSES)},
                    "outcome": {"enum": sorted(OUTCOMES)},
                    "public_path": {
                        "type": "string",
                        "maxLength": 512,
                        "pattern": path_pattern,
                    },
                    "bundle_schema_version": {
                        "enum": sorted(SUPPORTED_BUNDLE_SCHEMA_VERSIONS)
                    },
                    "verifier": {"$ref": "#/$defs/verifier"},
                    "artifact_set_hash": {
                        "type": "string",
                        "pattern": "^sha256:[0-9a-f]{64}$",
                    },
                    "captured_at": {
                        "type": "string",
                        "minLength": 10,
                        "maxLength": MAX_STRING_LENGTH,
                        "anyOf": [{"format": "date"}, {"format": "date-time"}],
                    },
                    "source_commit": {
                        "type": ["string", "null"],
                        "pattern": "^[0-9a-f]{40}$",
                    },
                    "model": {"$ref": "#/$defs/model"},
                    "runtime": {"$ref": "#/$defs/runtime"},
                    "hardware": {"$ref": "#/$defs/hardware"},
                    "workload": {"$ref": "#/$defs/workload"},
                    "measurements": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/measurement"},
                        "minItems": 1,
                        "maxItems": 64,
                    },
                    "claims": claims,
                    "supported_claims": {
                        "allOf": [{"$ref": "#/$defs/stringList"}],
                        "minItems": 1,
                    },
                    "unsupported_claims": {
                        "allOf": [{"$ref": "#/$defs/stringList"}],
                        "minItems": 1,
                    },
                    "budget": {"$ref": "#/$defs/budget"},
                    "dependencies": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/dependency"},
                        "maxItems": 128,
                    },
                    "limitations": {"$ref": "#/$defs/stringList"},
                },
            },
            "edge": {
                "type": "object",
                "additionalProperties": False,
                "required": ["source", "target", "relation"],
                "properties": {
                    "source": {"type": "string", "pattern": SAFE_ID.pattern},
                    "target": {"type": "string", "pattern": SAFE_ID.pattern},
                    "relation": {"enum": sorted(RELATIONS)},
                },
            },
        },
        "required": [
            "schema_version",
            "generator",
            "registry_hash",
            "claim_dimensions",
            "entries",
            "edges",
            "unregistered_candidates",
            "catalog_hash",
        ],
        "properties": {
            "schema_version": {"const": CATALOG_SCHEMA_VERSION},
            "generator": {
                "type": "object",
                "additionalProperties": False,
                "required": ["name", "version", "determinism"],
                "properties": {
                    "name": {"const": "llmtracefx-evidence"},
                    "version": {"$ref": "#/$defs/nonEmptyString"},
                    "determinism": {"$ref": "#/$defs/nonEmptyString"},
                },
            },
            "registry_hash": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
            "claim_dimensions": {
                "type": "array",
                "prefixItems": [{"const": value} for value in CLAIM_DIMENSIONS],
                "items": False,
                "minItems": len(CLAIM_DIMENSIONS),
                "maxItems": len(CLAIM_DIMENSIONS),
            },
            "entries": {
                "type": "array",
                "items": {"$ref": "#/$defs/entry"},
                "minItems": 1,
                "maxItems": 256,
            },
            "edges": {
                "type": "array",
                "items": {"$ref": "#/$defs/edge"},
                "maxItems": 4096,
            },
            "unregistered_candidates": {
                "type": "array",
                "items": {
                    "type": "string",
                    "maxLength": 512,
                    "pattern": path_pattern,
                },
                "maxItems": 4096,
            },
            "catalog_hash": {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"},
        },
    }


def _readme() -> str:
    return """# LLMTraceFX offline evidence catalog

This directory is generated from the closed registry in
`llmtracefx/evidence/registry.py`. Catalog metadata is not trusted: verification
revalidates the strict schema, catalog hash, exact bundle allowlists, every bundle's
closed verifier adapter, privacy gates, content identities, and lineage graph.

Reproduce from a source checkout without network access, credentials, model loads,
cloud authentication, or paid execution:

```bash
uv run llmtracefx-evidence index
uv run llmtracefx-evidence verify
uv run llmtracefx-evidence graph
make evidence-catalog
```

From an installed wheel and unrelated working directory, pass the committed catalog
explicitly; the repository root is inferred from that path:

```bash
llmtracefx-evidence verify --catalog /path/to/repo/examples/evidence-catalog/catalog.json
```

`catalog.json` is canonical machine-readable metadata. `graph.json` and `graph.dot`
carry the same typed lineage. `graph.svg` and `index.html` are self-contained static
views. `claim-matrix.json` preserves supported, unsupported, and not-applicable
claim states with provenance. `SHA256SUMS` covers every other generated file and intentionally excludes itself
to avoid a circular checksum.

Unknown example directories are never inferred into the catalog. Candidate evidence
directories not present in the closed registry appear in `unregistered_candidates`
and make verification fail until explicitly reviewed and registered.
"""


def render_catalog_artifacts(catalog: Mapping[str, Any]) -> dict[str, str]:
    """Render all deterministic catalog artifacts without writing them."""
    graph = _graph_document(catalog)
    graph_svg = _svg(graph)
    artifacts = {
        "README.md": _readme(),
        "catalog.json": canonical_json(catalog),
        "catalog.schema.json": canonical_json(_schema_document()),
        "claim-matrix.json": canonical_json(_claim_matrix(catalog)),
        "graph.dot": _dot(graph),
        "graph.json": canonical_json(graph),
        "graph.svg": graph_svg,
        "index.html": _html(catalog, graph_svg),
        "registry.json": canonical_json(_public_registry_document()),
    }
    for name, text in artifacts.items():
        _scan_privacy(name, text)
        if "<script" in text.lower() or "@import" in text.lower():
            raise CatalogError(f"{name} contains active or external content")
    return artifacts


def _checksums(artifacts: Mapping[str, str]) -> str:
    return "".join(
        f"{_sha256(artifacts[name].encode('utf-8'))}  {name}\n"
        for name in CATALOG_FILES
    )


def generate_catalog_artifacts(
    repo_root: str | Path, output_dir: str | Path | None = None
) -> dict[str, Any]:
    """Generate the canonical catalog, graph, matrix, and static views."""
    root = Path(repo_root).resolve(strict=True)
    destination = (
        Path(output_dir)
        if output_dir is not None
        else root / "examples" / "evidence-catalog"
    )
    if destination.exists() and destination.is_symlink():
        raise CatalogError("catalog output directory must not be a symlink")
    destination.mkdir(parents=True, exist_ok=True)
    actual = {path.name for path in destination.iterdir()}
    unexpected = actual - set(CATALOG_PUBLIC_FILES)
    if unexpected:
        raise CatalogError(
            f"catalog output directory has unexpected files: {sorted(unexpected)}"
        )
    catalog = build_catalog(root)
    artifacts = render_catalog_artifacts(catalog)
    artifacts["SHA256SUMS"] = _checksums(artifacts)
    for name in CATALOG_PUBLIC_FILES:
        atomic_write_text(destination / name, artifacts[name])
    return catalog


def _infer_repo_root(catalog_path: Path) -> Path:
    for candidate in (catalog_path.parent, *catalog_path.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "examples").is_dir()
            and (candidate / "llmtracefx" / "evidence" / "registry.py").is_file()
        ):
            return candidate.resolve()
    raise CatalogError("repository root could not be inferred; pass --repo-root")


def _verify_generated_files(catalog_path: Path, catalog: Mapping[str, Any]) -> None:
    directory = catalog_path.parent
    if not (directory / "SHA256SUMS").exists():
        raise CatalogError("generated catalog artifact is missing: SHA256SUMS")
    actual = {path.name for path in directory.iterdir()}
    if actual != set(CATALOG_PUBLIC_FILES):
        raise CatalogError("generated catalog artifact allowlist drifted")
    expected_artifacts = render_catalog_artifacts(catalog)
    expected_artifacts["SHA256SUMS"] = _checksums(expected_artifacts)
    for name in CATALOG_PUBLIC_FILES:
        recorded = read_bounded_regular_text(
            directory / name,
            (
                MAX_CATALOG_BYTES
                if name.endswith((".json", ".html", ".svg"))
                else MAX_METADATA_ARTIFACT_BYTES
            ),
        )
        if recorded != expected_artifacts[name]:
            raise CatalogError(f"generated catalog artifact changed: {name}")


def verify_catalog(
    catalog_path: str | Path,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify a committed catalog and every referenced source, failing closed."""
    unresolved_path = Path(catalog_path)
    if unresolved_path.is_symlink():
        raise CatalogError("catalog must not be a symlink")
    path = unresolved_path.resolve(strict=True)
    catalog = _load_json(path, MAX_CATALOG_BYTES)
    validate_catalog_document(catalog)
    root = (
        Path(repo_root).resolve(strict=True)
        if repo_root is not None
        else _infer_repo_root(path)
    )
    expected = build_catalog(root)
    if catalog != expected:
        raise CatalogError(
            "catalog does not match the closed registry and current bundles"
        )
    if catalog["unregistered_candidates"]:
        raise CatalogError("unregistered candidate evidence directories were found")
    verified: list[str] = []
    for source in SOURCES:
        verify_source(root, source)
        verified.append(source["evidence_id"])
    _verify_generated_files(path, catalog)
    return {
        "verified": True,
        "catalog_hash": catalog["catalog_hash"],
        "entries": len(catalog["entries"]),
        "edges": len(catalog["edges"]),
        "verified_evidence_ids": sorted(verified),
        "unregistered_candidates": [],
    }


__all__ = [
    "CATALOG_FILES",
    "CATALOG_PUBLIC_FILES",
    "CatalogError",
    "build_catalog",
    "canonical_json",
    "generate_catalog_artifacts",
    "render_catalog_artifacts",
    "validate_catalog_document",
    "verify_catalog",
    "verify_source",
]
