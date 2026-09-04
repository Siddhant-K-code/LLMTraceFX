"""Build and verify the offline vLLM crossover protocol evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..._artifact_io import (
    MAX_EVIDENCE_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_bytes,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from ...collectors._shared import atomic_write_text
from .cloudrift_crossover import offline_plan_document
from .vllm_compile import (
    BOOTSTRAP_RESAMPLES,
    CONTROLLED_REQUESTS_PER_CELL,
    PAIRS_PER_LANE,
    PROTOCOL_ID,
    SIGN_FLIP_ENUMERATIONS,
    VLLMCompileContractError,
    build_default_plan,
)

EVIDENCE_SCHEMA_VERSION = "1"
CAPTURED_AT = "2026-09-04T14:41:34.327+05:30"
IMPLEMENTATION_BASE_HEAD = "266b42bc7659b3ab147f199f293bfc033324f763"
BUNDLE_FILES = (
    "README.md",
    "SHA256SUMS",
    "budget-plan.json",
    "claim-matrix.json",
    "evidence-contract.json",
    "evidence_bundle.py",
    "experiment-plan.json",
    "methodology.svg",
    "offline-preflight.json",
    "protocol-sources.json",
    "report.html",
)
HASHED_FILES = tuple(sorted(set(BUNDLE_FILES) - {"SHA256SUMS"}))
SOURCE_FILES = (
    "llmtracefx/optimizer/lab/qwen3_8b/vllm_compile.py",
    "llmtracefx/optimizer/lab/qwen3_8b/cloudrift_crossover.py",
    "llmtracefx/optimizer/lab/qwen3_8b/cloudrift_crossover_runner.py",
    "llmtracefx/optimizer/lab/qwen3_8b/cloudrift_crossover_evidence.py",
    "llmtracefx/optimizer/lab/qwen3_8b/cloudrift_crossover_results.py",
)
_CHECKSUM = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._-]*)$")
_PRIVATE_PATTERNS = (
    (re.compile(r"/Users/|/home/|[A-Za-z]:\\Users\\"), "private home path"),
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "email address"),
    (
        re.compile(r"\b(?!127\.0\.0\.1\b)(?:\d{1,3}\.){3}\d{1,3}\b"),
        "IP address",
    ),
    (re.compile(r"\bGPU-[0-9a-f-]{16,}\b", re.I), "GPU UUID"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (
        re.compile(
            r"\b(?:gh[pousr]_|github_pat_|sk-|hf[_-]|AKIA)" r"[A-Za-z0-9_-]{8,}\b"
        ),
        "credential-shaped value",
    ),
)

README = """# Controlled Qwen3-8B vLLM crossover protocol

This is a verified offline protocol bundle, not benchmark evidence. It defines
two separate fresh-lifecycle lanes, eight eager/compiled pairs per lane,
counterbalanced ABBA/BAAB order, exact fixed-token-count and natural-output
workloads, whole-pair inference, a strict list-rate budget, and fail-closed
claim gates.

No CloudRift or Modal authentication occurred. No instance was created, model
downloaded, GPU used, or paid operation performed. All performance, crossover,
quality, runtime-component, and provider-spend claims remain unsupported until
a separately authorized complete execution bundle passes verification.

The controlled lane fixes 144 prompt requests and exactly 96 generated token
steps per request. This is fixed token count, not output control. Only observed
token-array identity can support an output-identical qualification. The
natural lane uses separate fresh lifecycles and gates end-to-end serving claims;
unequal outputs are never used for a causal speedup claim.

The independent analysis unit is a whole adjacent eager/compiled lifecycle
pair. Requests are repeated measures and are never bootstrapped independently.
The protocol reports first and sustained integer-request crossings, preserves
no-crossing outcomes as right-censored through request 144, and performs no
headline extrapolation.

Run `uv run --offline --no-sync python -I evidence_bundle.py verify` from this
directory in a clean checkout.
"""

WRAPPER = '''"""Verify the committed offline vLLM crossover protocol bundle."""

import importlib
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

verify_offline_bundle = importlib.import_module(
    "llmtracefx.optimizer.lab.qwen3_8b.cloudrift_crossover_evidence"
).verify_offline_bundle

if __name__ == "__main__":
    if sys.argv[1:] != ["verify"]:
        raise SystemExit("usage: evidence_bundle.py verify")
    verify_offline_bundle(Path(__file__).resolve().parent)
    print("Offline vLLM crossover protocol verified")
'''


class CrossoverEvidenceError(ValueError):
    """Raised when crossover evidence is unsafe, incomplete, or inconsistent."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_uri(value: bytes) -> str:
    return "sha256:" + _sha256(value)


def _canonical_json(value: Any) -> str:
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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        text = read_bounded_regular_text(path, MAX_EVIDENCE_ARTIFACT_BYTES)
        value = json.loads(text, parse_constant=reject_non_finite_json_constant)
    except (OSError, ArtifactReadError, ValueError, RecursionError) as exc:
        raise CrossoverEvidenceError(f"{path.name} is not safe JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CrossoverEvidenceError(f"{path.name} must contain an object")
    if text != _canonical_json(value):
        raise CrossoverEvidenceError(f"{path.name} is not canonical JSON")
    return value


def _source_document(repo_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative in SOURCE_FILES:
        path = repo_root / relative
        try:
            data = read_bounded_regular_bytes(path, MAX_EVIDENCE_ARTIFACT_BYTES)
        except (OSError, ArtifactReadError) as exc:
            raise CrossoverEvidenceError(
                f"protocol source is unavailable: {relative}"
            ) from exc
        files.append(
            {
                "path": relative,
                "bytes": len(data),
                "sha256": _sha256_uri(data),
            }
        )
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "implementation_base_head": IMPLEMENTATION_BASE_HEAD,
        "files": files,
        "source_set_sha256": _sha256_uri(
            json.dumps(
                files,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("utf-8")
        ),
    }


def _claim_matrix() -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "execution_state": "not_run",
        "claims": [
            {
                "claim_id": "offline-protocol-defined",
                "state": "supported",
                "provenance": "repository",
                "evidence": "experiment-plan.json",
            },
            {
                "claim_id": "zero-spend-offline-generation",
                "state": "supported",
                "provenance": "offline_process",
                "evidence": "offline-preflight.json",
            },
            {
                "claim_id": "fixed-token-count-crossover",
                "state": "unsupported",
                "provenance": "not_observed",
                "evidence": None,
            },
            {
                "claim_id": "output-identical-generation-crossover",
                "state": "unsupported",
                "provenance": "not_observed",
                "evidence": None,
            },
            {
                "claim_id": "numerically-reproducible-generation-crossover",
                "state": "unsupported",
                "provenance": "not_observed",
                "evidence": None,
            },
            {
                "claim_id": "forward-pass-identical",
                "state": "not_applicable",
                "provenance": "unsupported_interface",
                "evidence": None,
            },
            {
                "claim_id": "natural-output-quality-preserved",
                "state": "unsupported",
                "provenance": "not_observed",
                "evidence": None,
            },
            {
                "claim_id": "natural-end-to-end-causal-speedup",
                "state": "unsupported",
                "provenance": "not_observed",
                "evidence": None,
            },
            {
                "claim_id": "compile-cuda-graph-component-timing",
                "state": "unsupported",
                "provenance": "not_observed",
                "evidence": None,
            },
            {
                "claim_id": "budget-reservations-within-hard-cap",
                "state": "unsupported",
                "provenance": "not_observed",
                "evidence": None,
            },
            {
                "claim_id": "active-operation-list-rate-equivalent-within-hard-cap",
                "state": "unsupported",
                "provenance": "not_observed",
                "evidence": None,
            },
            {
                "claim_id": "provider-billed-cost-within-hard-cap",
                "state": "unsupported",
                "provenance": "not_observed",
                "evidence": None,
            },
            {
                "claim_id": "provider-teardown",
                "state": "unsupported",
                "provenance": "not_observed",
                "evidence": None,
            },
        ],
    }


def _contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "captured_at": CAPTURED_AT,
        "execution_state": "not_run",
        "evidence_kind": "offline_protocol_preflight",
        "historical_evidence_dependency": ("qwen3-8b-cloudrift-vllm-compile-20260903"),
        "independent_sample_unit": "adjacent eager-compiled lifecycle pair",
        "pairs_per_lane": PAIRS_PER_LANE,
        "controlled_requests_per_cell": CONTROLLED_REQUESTS_PER_CELL,
        "bootstrap": {
            "unit": "whole_pair",
            "resamples": BOOTSTRAP_RESAMPLES,
            "simultaneous_curve_band": True,
            "request_level_resampling": False,
            "controlled_small_sample_limitation": (
                "The eight-pair simultaneous band may under-cover; controlled "
                "claim support also requires the terminal sign-symmetry test."
            ),
            "natural_small_sample_limitation": (
                "Natural timing and nondegenerate quality percentile bootstraps "
                "use eight pairs, have no sign-symmetry backstop, and may under-cover."
            ),
        },
        "sign_flip_enumerations": SIGN_FLIP_ENUMERATIONS,
        "sign_flip_semantics": (
            "Exhaustive sign-symmetry permutation test, not randomized assignment "
            "inference; validity assumes sign-symmetric lifecycle-pair effects."
        ),
        "claim_requirements": plan["claim_requirements"],
        "quality_preservation": plan["quality_preservation"],
        "crossing": {
            "first": "first integer request with cumulative compiled <= eager",
            "sustained": (
                "first integer request with cumulative compiled <= eager through 144"
            ),
            "right_censor_at_request": CONTROLLED_REQUESTS_PER_CELL,
            "headline_extrapolation": False,
        },
        "forced_token_lane": {
            "state": "not_applicable",
            "reason": (
                "vLLM 0.28.0 has no stable dedicated forced-token replay API; "
                "prompt_logprobs changes cache behavior"
            ),
        },
        "component_observability": {
            "compile_time": "optional_version_pinned_internal_or_null",
            "cuda_graph_capture_time": "null_no_stable_hook",
            "debug_log_parsing": False,
        },
        "provider_access": "external_and_not_authorized",
        "methodology_sources": [
            {
                "scope": "vLLM documented metrics",
                "url": (
                    "https://github.com/vllm-project/vllm/blob/v0.28.0/"
                    "docs/design/metrics.md"
                ),
            },
            {
                "scope": "vLLM fixed decode-count latency benchmark",
                "url": (
                    "https://github.com/vllm-project/vllm/blob/v0.28.0/"
                    "vllm/benchmarks/latency.py"
                ),
            },
            {
                "scope": "vLLM batch-invariance limitations",
                "url": (
                    "https://github.com/vllm-project/vllm/blob/v0.28.0/"
                    "docs/features/batch_invariance.md"
                ),
            },
            {
                "scope": "PyTorch 2.13 reproducibility",
                "url": "https://docs.pytorch.org/docs/2.13/notes/randomness.html",
            },
            {
                "scope": "CUDA cuBLAS reproducibility",
                "url": (
                    "https://docs.nvidia.com/cuda/cublas/"
                    "index.html#results-reproducibility"
                ),
            },
            {
                "scope": "Qwen3-8B pinned model guidance",
                "url": (
                    "https://huggingface.co/Qwen/Qwen3-8B/blob/"
                    "b968826d9c46dd6066d109eabc6255188de91218/README.md"
                ),
            },
            {
                "scope": "paired bootstrap methodology",
                "url": "https://aclanthology.org/W04-3250/",
            },
            {
                "scope": "MLPerf inference reproducibility conventions",
                "url": (
                    "https://github.com/mlcommons/inference_policies/blob/"
                    "master/inference_rules.adoc"
                ),
            },
        ],
        "execution_authorization": {
            "state": "absent",
            "required_bindings": [
                "canonical authorization content hash",
                "OpenSSH detached signature from the authorized coordinator",
                "exact resolved workspace path hash",
                "exact plan hash",
                "exact clean source head",
                "pinned derived image digest",
                "public experiment nonce",
                "billing start and list rate",
                "USD 3 hard cap",
                "scheduled shutdown at billing start plus 19,680 seconds",
                "zero automatic retries",
                "approved local unix:///var/run/docker.sock endpoint",
                "externally managed provider access",
            ],
        },
        "host_receipts": {
            "initial_page_cache_reset": "inside measured preflight allowance",
            "between_cell_page_cache_resets": 31,
            "hardware_observations": (
                "preflight plus before and after every measured cell"
            ),
            "hardware_identity": "salted commitment shared with every cell receipt",
            "thermal_resource_failure": "terminal_no_retries",
            "provider_teardown": "external_and_never_inferred",
        },
    }


def _budget_document(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "execution_state": "not_run",
        "spend_authority": False,
        "actual_spend_usd": "0",
        "budget": plan["budget"],
        "later_authorization_required": True,
        "provider_reported_spend_usd": None,
        "provider_reported_spend_null_reason": "no provider operation occurred",
    }


def _render_svg(plan: Mapping[str, Any]) -> str:
    schedule = plan["schedule"]
    cells = []
    for index, cell in enumerate(schedule):
        x = 24 + index * 24
        color = "#175cd3" if cell["mode"] == "eager" else "#b54708"
        cells.append(
            f'<rect x="{x}" y="70" width="18" height="54" rx="2" fill="{color}"/>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="840" height="180" viewBox="0 0 840 180" role="img" aria-labelledby="title desc">
<title id="title">Counterbalanced vLLM crossover schedule</title>
<desc id="desc">Thirty-two fresh cells form eight controlled and eight natural eager-compiled pairs.</desc>
<rect width="840" height="180" fill="white"/>
<text x="24" y="34" font-family="system-ui" font-size="19" fill="#17202a">32 fresh cells; adjacent paired lifecycles</text>
{"".join(cells)}
<rect x="24" y="145" width="14" height="14" fill="#175cd3"/><text x="44" y="157" font-family="system-ui" font-size="13">eager</text>
<rect x="104" y="145" width="14" height="14" fill="#b54708"/><text x="124" y="157" font-family="system-ui" font-size="13">compiled</text>
</svg>
"""


def _render_report(
    plan: Mapping[str, Any],
    preflight: Mapping[str, Any],
    claims: Mapping[str, Any],
) -> str:
    summary = plan["budget"]["summary"]
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['claim_id'])}</td>"
        f"<td>{html.escape(item['state'])}</td>"
        f"<td>{html.escape(item['provenance'])}</td>"
        "</tr>"
        for item in claims["claims"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Controlled vLLM crossover protocol</title>
  <style>
    body {{ color:#17202a; font:16px/1.5 system-ui,sans-serif; margin:2rem auto; max-width:980px; padding:0 1rem; }}
    .notice {{ background:#fffaeb; border-left:.4rem solid #b54708; padding:1rem; }}
    table {{ border-collapse:collapse; width:100%; }}
    th,td {{ border-bottom:1px solid #d0d5dd; padding:.55rem; text-align:left; }}
  </style>
</head>
<body>
  <h1>Controlled Qwen3-8B vLLM crossover protocol</h1>
  <p class="notice"><strong>Offline protocol only.</strong> No provider access,
  GPU use, model download, or spend occurred. Crossover and quality claims are
  unsupported.</p>
  <p>Two separate lanes each contain {PAIRS_PER_LANE} adjacent fresh lifecycle
  pairs. The controlled lane measures {CONTROLLED_REQUESTS_PER_CELL} sequential
  requests per cell at a fixed 96-token decode count. The natural lane uses
  separate fresh lifecycles for bounded output and correctness.</p>
  <p>The active later-run envelope is {summary["active_planned_seconds"]} seconds
  (${summary["active_planned_usd"]}) with an untouched
  {summary["untouched_margin_seconds"]}-second termination margin. This is not
  spending authority.</p>
  <p>Offline blockers: {len(preflight["blockers"])}. Whole lifecycle pairs, not
  requests, are the inferential unit.</p>
  <table><thead><tr><th>Claim</th><th>State</th><th>Provenance</th></tr></thead>
  <tbody>{rows}</tbody></table>
</body>
</html>
"""


def _expected_documents(repo_root: Path) -> dict[str, bytes]:
    plan = build_default_plan().to_dict()
    preflight = offline_plan_document()
    claims = _claim_matrix()
    documents = {
        "README.md": README.encode("utf-8"),
        "budget-plan.json": _canonical_json(_budget_document(plan)).encode("utf-8"),
        "claim-matrix.json": _canonical_json(claims).encode("utf-8"),
        "evidence-contract.json": _canonical_json(_contract(plan)).encode("utf-8"),
        "evidence_bundle.py": WRAPPER.encode("utf-8"),
        "experiment-plan.json": _canonical_json(plan).encode("utf-8"),
        "methodology.svg": _render_svg(plan).encode("utf-8"),
        "offline-preflight.json": _canonical_json(preflight).encode("utf-8"),
        "protocol-sources.json": _canonical_json(_source_document(repo_root)).encode(
            "utf-8"
        ),
        "report.html": _render_report(plan, preflight, claims).encode("utf-8"),
    }
    checksums = "\n".join(
        f"{_sha256(documents[name])}  {name}" for name in HASHED_FILES
    )
    documents["SHA256SUMS"] = (checksums + "\n").encode("utf-8")
    return documents


def build_offline_bundle(output_dir: Path, *, repo_root: Path) -> None:
    """Write the deterministic refusal bundle and verify it immediately."""

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink():
        raise CrossoverEvidenceError("bundle directory must not be a symlink")
    expected = _expected_documents(repo_root.resolve())
    unexpected = {path.name for path in output_dir.iterdir()} - set(BUNDLE_FILES)
    if unexpected:
        raise CrossoverEvidenceError(
            f"bundle directory contains unexpected files: {sorted(unexpected)}"
        )
    for name, data in expected.items():
        atomic_write_text(output_dir / name, data.decode("utf-8"))
    verify_offline_bundle(output_dir, repo_root=repo_root)


def _scan_privacy(name: str, text: str) -> None:
    for pattern, description in _PRIVATE_PATTERNS:
        if pattern.search(text):
            raise CrossoverEvidenceError(f"{name} contains {description}")


def verify_offline_bundle(
    bundle_dir: Path,
    *,
    repo_root: Path | None = None,
) -> None:
    """Verify exact files, source bindings, semantics, privacy, and checksums."""

    root = bundle_dir.resolve()
    repository = (
        Path(__file__).resolve().parents[4]
        if repo_root is None
        else repo_root.resolve()
    )
    if bundle_dir.is_symlink() or not root.is_dir():
        raise CrossoverEvidenceError("bundle must be a non-symlink directory")
    actual = {path.name for path in root.iterdir()}
    if actual != set(BUNDLE_FILES):
        raise CrossoverEvidenceError(
            f"bundle file set differs: {sorted(actual ^ set(BUNDLE_FILES))}"
        )
    expected = _expected_documents(repository)
    for name in BUNDLE_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise CrossoverEvidenceError(f"{name} must be a regular file")
        data = read_bounded_regular_bytes(path, MAX_EVIDENCE_ARTIFACT_BYTES)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CrossoverEvidenceError(f"{name} is not UTF-8") from exc
        _scan_privacy(name, text)
        if data != expected[name]:
            raise CrossoverEvidenceError(f"{name} differs from the protocol contract")

    checksum_text = (root / "SHA256SUMS").read_text(encoding="utf-8")
    names: set[str] = set()
    for line in checksum_text.splitlines():
        match = _CHECKSUM.fullmatch(line)
        if match is None:
            raise CrossoverEvidenceError("SHA256SUMS contains a malformed line")
        digest, name = match.groups()
        if name in names or name not in HASHED_FILES:
            raise CrossoverEvidenceError("SHA256SUMS allowlist differs")
        names.add(name)
        if digest != _sha256((root / name).read_bytes()):
            raise CrossoverEvidenceError(f"checksum mismatch for {name}")
    if names != set(HASHED_FILES):
        raise CrossoverEvidenceError("SHA256SUMS is incomplete")

    plan = _read_json(root / "experiment-plan.json")
    preflight = _read_json(root / "offline-preflight.json")
    claims = _read_json(root / "claim-matrix.json")
    if VLLMCompilePlanProxy.verify(plan) is not True:
        raise CrossoverEvidenceError("experiment plan does not verify")
    if (
        preflight["execution_authorized"] is not False
        or preflight["offline_only"] is not True
        or preflight["spend_usd"] != "0"
        or preflight["gpu_used"] is not False
        or claims["execution_state"] != "not_run"
    ):
        raise CrossoverEvidenceError("offline refusal semantics drifted")
    encoded = json.dumps(
        [plan, preflight, claims],
        allow_nan=False,
        ensure_ascii=True,
    )
    if "NaN" in encoded or "Infinity" in encoded:
        raise CrossoverEvidenceError("non-finite value found")


class VLLMCompilePlanProxy:
    """Small indirection keeping verifier error messages evidence-specific."""

    @staticmethod
    def verify(value: Mapping[str, Any]) -> bool:
        from .vllm_compile import VLLMCompilePlan

        try:
            VLLMCompilePlan.from_dict(dict(value))
        except VLLMCompileContractError as exc:
            raise CrossoverEvidenceError("experiment plan is invalid") from exc
        return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="action", required=True)
    build = subparsers.add_parser("build", allow_abbrev=False)
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument("--repo-root", required=True, type=Path)
    verify = subparsers.add_parser("verify", allow_abbrev=False)
    verify.add_argument("--bundle-dir", required=True, type=Path)
    verify.add_argument("--repo-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "build":
            build_offline_bundle(args.output_dir, repo_root=args.repo_root)
        else:
            verify_offline_bundle(args.bundle_dir, repo_root=args.repo_root)
        return 0
    except (OSError, ValueError, CrossoverEvidenceError) as exc:
        print(f"vLLM crossover evidence failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
