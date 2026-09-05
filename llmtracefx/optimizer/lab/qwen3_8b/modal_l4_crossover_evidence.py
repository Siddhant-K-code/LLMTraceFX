"""Build and verify the offline Modal L4 crossover protocol evidence.

Two deterministic, fail-closed contracts live here.

``build_offline_bundle``/``verify_offline_bundle`` publish the
preregistration and refusal bundle for the Modal L4 delta: the plan, the
budget chain, the claim matrix with every provider-conditioned claim
still unsupported, and the result contract a future run must satisfy.

``verify_result_bundle`` adjudicates a completed run. It validates the
Modal-specific envelope first (attempt receipts, rate receipt, memory
gate, application ledger, teardown, limitations) and only then delegates
the statistics to the existing crossover results verifier. The
statistical core is reused, not reimplemented.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal
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
from .cloudrift_crossover_results import CrossoverResultsError
from .modal_l4_crossover import (
    BLOCKED_CLAIM_IDS,
    COMPUTE_PLANNED_SECONDS,
    COMPUTE_PLANNED_USD,
    CPU_FUNCTION_USD_PER_SECOND,
    CREDENTIAL_EXPOSURE_CONFIRMERS,
    GPU_FUNCTION_USD_PER_SECOND,
    HARD_CAP_USD,
    OFFICIAL_RATE_URL,
    PROTOCOL_ID,
    STORAGE_PLANNED_USD,
    TOTAL_PLANNED_USD,
    UNCONTROLLED_CACHE_LIMITATIONS,
    UNTOUCHED_MARGIN_USD,
    ModalL4ContractError,
    ModalL4Plan,
    assert_provider_sdk_absent,
    build_default_plan,
    evaluate_attempt_receipts,
    evaluate_memory_gate,
    evaluate_teardown_receipt,
    verify_ledger_document,
    verify_official_rate_receipt,
    verify_profile_authentication,
)
from .modal_l4_crossover_results import (
    REUSED_STATISTICAL_PRIMITIVES,
    ModalL4ResultsError,
    analyze_modal_run,
)
from .modal_l4_rates import RateRefreshError, verify_rate_refresh
from .vllm_compile import PROTOCOL_ID as BASE_PROTOCOL_ID
from .vllm_compile import canonical_decimal

EVIDENCE_SCHEMA_VERSION = "1"
CAPTURED_AT = "2026-09-04T19:52:50.511+05:30"
IMPLEMENTATION_BASE_HEAD = "6b5448790551f61c57cbe23aa9001e728cc73e43"
BUNDLE_FILES = (
    "README.md",
    "SHA256SUMS",
    "budget-plan.json",
    "claim-matrix.json",
    "evidence-contract.json",
    "experiment-plan.json",
    "methodology.svg",
    "offline-preflight.json",
    "protocol-sources.json",
    "report.html",
    "result-contract.json",
)
HASHED_FILES = tuple(sorted(set(BUNDLE_FILES) - {"SHA256SUMS"}))
SOURCE_FILES = (
    "llmtracefx/optimizer/lab/qwen3_8b/vllm_compile.py",
    "llmtracefx/optimizer/lab/qwen3_8b/cloudrift_crossover_runner.py",
    "llmtracefx/optimizer/lab/qwen3_8b/cloudrift_crossover_results.py",
    "llmtracefx/optimizer/lab/qwen3_8b/modal_l4_crossover.py",
    "llmtracefx/optimizer/lab/qwen3_8b/modal_l4_cell_runner.py",
    "llmtracefx/optimizer/lab/qwen3_8b/modal_l4_app.py",
    "llmtracefx/optimizer/lab/qwen3_8b/modal_l4_rates.py",
    "llmtracefx/optimizer/lab/qwen3_8b/modal_l4_execute.py",
    "llmtracefx/optimizer/lab/qwen3_8b/modal_l4_crossover_results.py",
    "llmtracefx/optimizer/lab/qwen3_8b/modal_l4_crossover_evidence.py",
)
CELLS_DIRECTORY = "cells"
ORCHESTRATION_FILE = "orchestration-receipt.json"
RESULT_ENVELOPE_FILES = (
    "application-ledger.json",
    "credential-exposure.json",
    "memory-gate.json",
    "modal-attempt-receipts.json",
    "modal-limitations.json",
    "modal-rate-receipt.json",
    "modal-rate-refresh.json",
    "modal-teardown.json",
    "profile-authentication.json",
    "source-checkout.json",
)
# Deterministic artifacts regenerated and byte-compared by the result verifier.
# analysis.json is the full re-derived analysis (crossover distribution, the
# right-censored bootstrap interval, and the lean pair records); claim-matrix
# and the two renders are projected from it.
RESULT_ARTIFACT_FILES = (
    "analysis.json",
    "claim-matrix.json",
    "crossover.svg",
    "report.html",
)
RESULT_CHECKSUM_FILE = "SHA256SUMS"
# Every top-level file the SHA256SUMS manifest covers: the sealed orchestration
# receipt, each standalone envelope, and each regenerated artifact. The 32 cell
# receipts live under ``cells/`` and are validated by their own inner seals
# through ``analyze_modal_run``; the checksum-line grammar admits no path
# separator, so they are covered cryptographically rather than by this manifest.
RESULT_HASHED_FILES = tuple(
    sorted({ORCHESTRATION_FILE, *RESULT_ENVELOPE_FILES, *RESULT_ARTIFACT_FILES})
)
RESULT_TOP_LEVEL_FILES = tuple(sorted({*RESULT_HASHED_FILES, RESULT_CHECKSUM_FILE}))
# The statistics are no longer delegated to the CloudRift bundle verifier,
# which is bound to receipts a Modal run cannot produce. Instead the
# provider-neutral statistical primitives are reused directly over the sealed
# inner cell receipts. The reference is kept honest here.
DELEGATED_STATISTICAL_VERIFIER = (
    "llmtracefx.optimizer.lab.qwen3_8b.modal_l4_crossover_results.analyze_modal_run"
)
REUSED_STATISTICAL_PRIMITIVE_NAMES = tuple(REUSED_STATISTICAL_PRIMITIVES)
PROVENANCE_DOMAINS = (
    "modal.com",
    "github.com",
    "docs.pytorch.org",
    "docs.nvidia.com",
    "huggingface.co",
    "aclanthology.org",
)
_CHECKSUM = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._-]*)$")
_PRIVATE_PATTERNS = (
    (re.compile(r"/Users/|/home/|[A-Za-z]:\\Users\\"), "private home path"),
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "email address"),
    (re.compile(r"\b(?!127\.0\.0\.1\b)(?:\d{1,3}\.){3}\d{1,3}\b"), "IP address"),
    (re.compile(r"\bGPU-[0-9a-f-]{16,}\b", re.I), "GPU UUID"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (
        re.compile(r"\b(?:gh[pousr]_|github_pat_|sk-|hf[_-]|AKIA)[A-Za-z0-9_-]{8,}\b"),
        "credential-shaped value",
    ),
    (re.compile(r'"(?:modal_token[a-z_]*|token_secret)"'), "credential field"),
)

README = """# Modal L4 delta for the Qwen3-8B vLLM crossover protocol

This is a verified offline protocol bundle, not benchmark evidence. It
preregisters one future Modal L4 execution of the sealed crossover
experiment. The scientific core is unchanged: the same pinned model
revision, the same runtime pins, two lanes, eight adjacent eager/compiled
pairs per lane, the same sealed 32-cell counterbalanced schedule, 144
fixed-token-count controlled requests and 12 natural requests per cell,
whole-pair statistics, no replacement cells, no adaptive stopping, and no
extrapolation.

Only the provider envelope differs. Work runs through Modal Functions and
RPC, never a public web endpoint, on one L4 with four physical CPU cores
and 32 GiB of memory, one live cell at a time, single-use containers,
zero retries, and an explicit timeout on every stage.

No Modal authentication occurred. The Modal SDK is not imported on any
planning or verification path. No container was created, model
downloaded, GPU used, or paid operation performed.

Two things this protocol deliberately does not claim. Modal exposes no
host page-cache control and no dedicated-host reservation, so the
CloudRift cache-control requirements are removed here rather than
asserted unverifiably: results are descriptive, provider-conditioned
paired comparisons, and pure causal compilation and natural causal
speedup claims are unsupported by construction. The application ledger
records reservations at published list rates and is explicitly not
provider proof; provider-reported spend stays null until an external
sanitized receipt exists.

A GPU memory admission gate precedes every measured cell. Immutable
runner arguments stay BF16, tensor parallel 1, one sequence, 0.94 memory
utilization, no prefix or speculative decoding, and a context length of
exactly the longest frozen prompt array plus 96. If either canary fails,
the run publishes a refusal. Nothing is tuned to make it pass.
"""


class ModalL4EvidenceError(ValueError):
    """Raised when Modal L4 evidence is unsafe, incomplete, or inconsistent."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_uri(value: bytes) -> str:
    return "sha256:" + _sha256(value)


def _canonical_json(value: Any) -> str:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False)
        + "\n"
    )


def _compact_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _read_json(path: Path, *, require_canonical: bool = True) -> dict[str, Any]:
    try:
        text = read_bounded_regular_text(path, MAX_EVIDENCE_ARTIFACT_BYTES)
        value = json.loads(text, parse_constant=reject_non_finite_json_constant)
    except (OSError, ArtifactReadError, ValueError, RecursionError) as exc:
        raise ModalL4EvidenceError(f"{path.name} is not safe JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ModalL4EvidenceError(f"{path.name} must contain an object")
    if require_canonical and text != _canonical_json(value):
        raise ModalL4EvidenceError(f"{path.name} is not canonical JSON")
    return value


def _require_provenance_domain(url: str, *, field: str) -> str:
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ModalL4EvidenceError(f"{field} must be an https URL")
    host = url.split("/")[2].split("@")[-1].split(":")[0].lower()
    if not any(
        host == domain or host.endswith(f".{domain}") for domain in PROVENANCE_DOMAINS
    ):
        raise ModalL4EvidenceError(f"{field} is not an allowed provenance domain")
    return url


def _source_document(repo_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for relative in SOURCE_FILES:
        path = repo_root / relative
        try:
            data = read_bounded_regular_bytes(path, MAX_EVIDENCE_ARTIFACT_BYTES)
        except (OSError, ArtifactReadError) as exc:
            raise ModalL4EvidenceError(
                f"protocol source is unavailable: {relative}"
            ) from exc
        files.append(
            {"path": relative, "bytes": len(data), "sha256": _sha256_uri(data)}
        )
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "implementation_base_head": IMPLEMENTATION_BASE_HEAD,
        "reused_statistical_core": DELEGATED_STATISTICAL_VERIFIER,
        "files": files,
        "source_set_sha256": _sha256_uri(_compact_json(files).encode("utf-8")),
    }


def _claim_matrix() -> dict[str, Any]:
    blocked = dict.fromkeys(BLOCKED_CLAIM_IDS, "unsupported_by_construction_on_modal")
    claims = [
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
            "claim_id": "no-provider-authentication",
            "state": "supported",
            "provenance": "offline_process",
            "evidence": "offline-preflight.json",
        },
        {
            "claim_id": "exposed-profile-credential-never-used-by-experiment",
            "state": "supported",
            "provenance": "offline_process_never_authenticated",
            "evidence": "offline-preflight.json",
        },
        {
            "claim_id": "exposed-profile-credential-revocation-confirmed",
            "state": "unsupported",
            "provenance": "not_observed",
            "evidence": None,
        },
        {
            "claim_id": "fresh-local-profile-created-without-sharing",
            "state": "unsupported",
            "provenance": "not_observed",
            "evidence": None,
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
            "claim_id": "numerically-reproducible-generation",
            "state": "unsupported",
            "provenance": "not_observed",
            "evidence": None,
        },
        {
            "claim_id": "natural-output-quality-preserved",
            "state": "unsupported",
            "provenance": "not_observed",
            "evidence": None,
        },
        {
            "claim_id": "cache-state-controlled-comparison",
            "state": "not_applicable",
            "provenance": "unobservable_provider_placement_and_page_cache",
            "evidence": None,
        },
        {
            "claim_id": "memory-gate-passed",
            "state": "unsupported",
            "provenance": "not_observed",
            "evidence": None,
        },
        {
            "claim_id": "application-ledger-within-hard-cap",
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
    ]
    claims.extend(
        {
            "claim_id": claim_id,
            "state": "not_applicable",
            "provenance": reason,
            "evidence": None,
        }
        for claim_id, reason in sorted(blocked.items())
    )
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "execution_state": "not_run",
        "claims": sorted(claims, key=lambda item: item["claim_id"]),
    }


def _budget_document(plan: Mapping[str, Any]) -> dict[str, Any]:
    budget = plan["budget"]
    chain = {
        "rates": plan["pricing"],
        "stages": budget["stages"],
        "compute_planned_seconds": COMPUTE_PLANNED_SECONDS,
        "compute_planned_usd": canonical_decimal(COMPUTE_PLANNED_USD),
        "storage_planned_usd": canonical_decimal(STORAGE_PLANNED_USD),
        "total_planned_usd": canonical_decimal(TOTAL_PLANNED_USD),
        "untouched_margin_usd": canonical_decimal(UNTOUCHED_MARGIN_USD),
        "hard_cap_usd": canonical_decimal(HARD_CAP_USD),
    }
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "execution_state": "not_run",
        "spend_authority": False,
        "actual_spend_usd": "0",
        "plan_sha256": plan["plan_sha256"],
        "budget": budget,
        "budget_chain": chain,
        "budget_chain_sha256": _sha256_uri(_compact_json(chain).encode("utf-8")),
        "contingency_is_never_spent_on_science": True,
        "application_ledger_required": True,
        "application_ledger_is_provider_proof": False,
        "later_authorization_required": True,
        "provider_reported_spend_usd": None,
        "provider_reported_spend_null_reason": "no provider operation occurred",
    }


def _result_contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "execution_state": "not_run",
        "orchestration_file": ORCHESTRATION_FILE,
        "cells_directory": CELLS_DIRECTORY,
        "provider_native_results_verifier": DELEGATED_STATISTICAL_VERIFIER,
        "reused_statistical_primitives": sorted(REUSED_STATISTICAL_PRIMITIVE_NAMES),
        "modal_envelope_files": sorted(RESULT_ENVELOPE_FILES),
        "result_artifact_files": sorted(RESULT_ARTIFACT_FILES),
        "checksum_manifest": RESULT_CHECKSUM_FILE,
        "verification_order": [
            "modal envelope files exist and are canonical",
            "the credential-exposure gate is cleared by a coordinator attestation",
            "attempt receipts show one terminal attempt per planned lifecycle",
            "official rate receipt is at or below the committed rates",
            "both memory-gate canaries passed without tuning",
            "application ledger verifies within the hard cap",
            "teardown receipt is complete with no live named volumes",
            "the orchestration receipt is a published, complete result",
            "all 32 sealed inner cell receipts validate with reused primitives",
            "L4, runtime-pin, driver and nonce-bound commitment continuity holds",
            "the fresh official-rate capture recomputes and binds to the receipt",
            "the source-checkout receipt binds to the authorized clean checkout",
            "the boolean-only local-profile verdict binds to the orchestration",
            "analysis, claim matrix, report, and figure regenerate byte-for-byte",
            "the SHA256SUMS manifest recomputes for every top-level file",
            "claims blocked by construction are not marked supported",
        ],
        "fail_closed": True,
        "blocked_claim_ids": sorted(BLOCKED_CLAIM_IDS),
        "invalidating_observations": plan["invalidating_observations"],
        "provider_spend_nullable": True,
    }


def _contract(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "base_protocol_id": BASE_PROTOCOL_ID,
        "captured_at": CAPTURED_AT,
        "execution_state": "not_run",
        "evidence_kind": "offline_protocol_preregistration",
        "provider": plan["provider"],
        "provider_access": "not_authenticated_and_not_accessed",
        "preserved_from_base_protocol": plan["preserved_from_base_protocol"],
        "changed_from_base_protocol": plan["changed_from_base_protocol"],
        "statistics": plan["preserved_core"]["statistics"],
        "memory_gate": plan["memory_gate"],
        "cache_claims": plan["cache_claims"],
        "measurement_delta": plan["measurement_delta"],
        "authentication_policy": plan["authentication_policy"],
        "credential_exposure_gate": plan["credential_exposure_gate"],
        "teardown_contract": plan["teardown_contract"],
        "accepted_residual_risk": plan["accepted_residual_risk"],
        "execution_surface": {
            "provider_sdk": plan["provider_sdk"],
            "functions": plan["functions"],
            "call_sequence": plan["call_sequence"],
            "statistical_publication": plan["statistical_publication"],
            "provider_module": (
                "llmtracefx.optimizer.lab.qwen3_8b.modal_l4_app is the only module "
                "that imports the provider SDK; no planning, verification, or "
                "evidence path imports it"
            ),
            "sdk_import_point": "inside the run action, after every gate has passed",
            "authorization": {
                "schema": "llmtracefx.modal_l4_crossover.authorization",
                "binding": [
                    "exact plan hash",
                    "exact clean source head",
                    "public experiment nonce and run-scoped resource names",
                    "pinned base image reference",
                    "hash of the authorized structured rate receipt",
                    "hash of the resolved workspace path",
                    "explicit acceptance of the Modal crash-reschedule residual",
                ],
                "authentication": "openssh_detached_signature",
            },
        },
        "methodology_sources": [
            {"scope": "Modal published pricing", "url": OFFICIAL_RATE_URL},
            {
                "scope": "Modal GPU and container lifecycle reference",
                "url": "https://modal.com/docs/guide/gpu",
            },
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
        ],
        "execution_authorization": {
            "state": "absent",
            "required_bindings": [
                "exact plan hash",
                "exact clean source head",
                "public experiment nonce and run-scoped resource names",
                "re-fetched and hashed official Modal rate receipt",
                "standard local Modal profile with no credential or routing override",
                "USD 6 hard cap with an untouched contingency margin",
                "zero retries and one live cell",
                "explicit timeout for every stage",
                "terminal teardown obligation on every exit path",
            ],
        },
    }


def _preflight_document() -> dict[str, Any]:
    from .modal_l4_crossover import offline_plan_document

    return offline_plan_document()


def _render_svg(plan: Mapping[str, Any]) -> str:
    cells = []
    for index, cell in enumerate(plan["preserved_core"]["schedule"]):
        x = 24 + index * 24
        color = "#175cd3" if cell["mode"] == "eager" else "#b54708"
        cells.append(
            f'<rect x="{x}" y="70" width="18" height="54" rx="2" fill="{color}"/>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="840" height="180" viewBox="0 0 840 180" role="img" aria-labelledby="title desc">
<title id="title">Modal L4 counterbalanced crossover schedule</title>
<desc id="desc">Thirty-two single-use Modal containers form eight controlled and eight natural eager-compiled pairs on one L4.</desc>
<rect width="840" height="180" fill="white"/>
<text x="24" y="34" font-family="system-ui" font-size="19" fill="#17202a">32 single-use L4 cells; one live cell at a time</text>
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
    budget = plan["budget"]
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(item['claim_id'])}</td>"
        f"<td>{html.escape(item['state'])}</td>"
        f"<td>{html.escape(item['provenance'])}</td>"
        "</tr>"
        for item in claims["claims"]
    )
    limitations = "".join(
        f"<li>{html.escape(item)}</li>" for item in UNCONTROLLED_CACHE_LIMITATIONS
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Modal L4 vLLM crossover protocol delta</title>
  <style>
    body {{ color:#17202a; font:16px/1.5 system-ui,sans-serif; margin:2rem auto; max-width:980px; padding:0 1rem; }}
    .notice {{ background:#fffaeb; border-left:.4rem solid #b54708; padding:1rem; }}
    table {{ border-collapse:collapse; width:100%; }}
    th,td {{ border-bottom:1px solid #d0d5dd; padding:.55rem; text-align:left; }}
  </style>
</head>
<body>
  <h1>Modal L4 delta for the Qwen3-8B vLLM crossover protocol</h1>
  <p class="notice"><strong>Offline protocol only.</strong> No Modal
  authentication, SDK import, container, GPU use, model download, or spend
  occurred. Crossover, quality, and spend claims are unsupported.</p>
  <p>The sealed scientific core is preserved from
  {html.escape(str(plan["preserved_core"]["base_protocol_id"]))}: two lanes,
  eight adjacent pairs per lane, and a 32-cell counterbalanced schedule of
  144 fixed-token-count controlled requests and 12 natural requests per cell.</p>
  <p>The priced envelope is {budget["compute_planned_seconds"]} container
  seconds (${budget["compute_planned_usd"]}) plus ${budget["storage"]["planned_usd"]}
  of volume reservation, totalling ${budget["total_planned_usd"]} against a
  ${budget["hard_cap_usd"]} hard cap with an untouched
  ${budget["untouched_margin_usd"]} contingency that is never spent on science.
  This is an application-side ledger at published list rates, not provider proof.</p>
  <p>Offline blockers: {len(preflight["blockers"])}. Whole lifecycle pairs, not
  requests, remain the inferential unit.</p>
  <h2>Uncontrolled on this provider</h2>
  <ul>{limitations}</ul>
  <table><thead><tr><th>Claim</th><th>State</th><th>Provenance</th></tr></thead>
  <tbody>{rows}</tbody></table>
</body>
</html>
"""


def _expected_documents(repo_root: Path) -> dict[str, bytes]:
    plan = build_default_plan().to_dict()
    preflight = _preflight_document()
    claims = _claim_matrix()
    documents = {
        "README.md": README.encode("utf-8"),
        "budget-plan.json": _canonical_json(_budget_document(plan)).encode("utf-8"),
        "claim-matrix.json": _canonical_json(claims).encode("utf-8"),
        "evidence-contract.json": _canonical_json(_contract(plan)).encode("utf-8"),
        "experiment-plan.json": _canonical_json(plan).encode("utf-8"),
        "methodology.svg": _render_svg(plan).encode("utf-8"),
        "offline-preflight.json": _canonical_json(preflight).encode("utf-8"),
        "protocol-sources.json": _canonical_json(_source_document(repo_root)).encode(
            "utf-8"
        ),
        "report.html": _render_report(plan, preflight, claims).encode("utf-8"),
        "result-contract.json": _canonical_json(_result_contract(plan)).encode("utf-8"),
    }
    checksums = "\n".join(
        f"{_sha256(documents[name])}  {name}" for name in HASHED_FILES
    )
    documents["SHA256SUMS"] = (checksums + "\n").encode("utf-8")
    return documents


def _scan_privacy(name: str, text: str) -> None:
    for pattern, description in _PRIVATE_PATTERNS:
        if pattern.search(text):
            raise ModalL4EvidenceError(f"{name} contains {description}")


def build_offline_bundle(output_dir: Path, *, repo_root: Path) -> None:
    """Write the deterministic preregistration bundle and verify it now."""

    assert_provider_sdk_absent()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink():
        raise ModalL4EvidenceError("bundle directory must not be a symlink")
    expected = _expected_documents(repo_root.resolve())
    unexpected = {path.name for path in output_dir.iterdir()} - set(BUNDLE_FILES)
    if unexpected:
        raise ModalL4EvidenceError(
            f"bundle directory contains unexpected files: {sorted(unexpected)}"
        )
    for name, data in expected.items():
        atomic_write_text(output_dir / name, data.decode("utf-8"))
    verify_offline_bundle(output_dir, repo_root=repo_root)


def verify_offline_bundle(
    bundle_dir: Path,
    *,
    repo_root: Path | None = None,
) -> None:
    """Verify files, source bindings, semantics, privacy, and checksums."""

    assert_provider_sdk_absent()
    root = bundle_dir.resolve()
    repository = (
        Path(__file__).resolve().parents[4]
        if repo_root is None
        else repo_root.resolve()
    )
    if bundle_dir.is_symlink() or not root.is_dir():
        raise ModalL4EvidenceError("bundle must be a non-symlink directory")
    actual = {path.name for path in root.iterdir()}
    if actual != set(BUNDLE_FILES):
        raise ModalL4EvidenceError(
            f"bundle file set differs: {sorted(actual ^ set(BUNDLE_FILES))}"
        )
    expected = _expected_documents(repository)
    for name in BUNDLE_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ModalL4EvidenceError(f"{name} must be a regular file")
        data = read_bounded_regular_bytes(path, MAX_EVIDENCE_ARTIFACT_BYTES)
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ModalL4EvidenceError(f"{name} is not UTF-8") from exc
        _scan_privacy(name, text)
        if data != expected[name]:
            raise ModalL4EvidenceError(f"{name} differs from the protocol contract")

    checksum_text = (root / "SHA256SUMS").read_text(encoding="utf-8")
    names: set[str] = set()
    for line in checksum_text.splitlines():
        match = _CHECKSUM.fullmatch(line)
        if match is None:
            raise ModalL4EvidenceError("SHA256SUMS contains a malformed line")
        digest, name = match.groups()
        if name in names or name not in HASHED_FILES:
            raise ModalL4EvidenceError("SHA256SUMS allowlist differs")
        names.add(name)
        if digest != _sha256((root / name).read_bytes()):
            raise ModalL4EvidenceError(f"checksum mismatch for {name}")
    if names != set(HASHED_FILES):
        raise ModalL4EvidenceError("SHA256SUMS is incomplete")

    plan = _read_json(root / "experiment-plan.json")
    preflight = _read_json(root / "offline-preflight.json")
    claims = _read_json(root / "claim-matrix.json")
    budget = _read_json(root / "budget-plan.json")
    contract = _read_json(root / "evidence-contract.json")
    try:
        ModalL4Plan.from_dict(plan)
    except ModalL4ContractError as exc:
        raise ModalL4EvidenceError("experiment plan is invalid") from exc
    for entry in contract["methodology_sources"]:
        _require_provenance_domain(entry["url"], field="methodology source")
    if (
        preflight["execution_authorized"] is not False
        or preflight["offline_only"] is not True
        or preflight["provider_authentication_used"] is not False
        or preflight["provider_sdk_imported"] is not False
        or preflight["spend_usd"] != "0"
        or preflight["gpu_used"] is not False
        or claims["execution_state"] != "not_run"
        or preflight["credential_exposure_gate"]["cleared"] is not False
        or preflight["credential_exposure_gate"]["action"]
        != "refuse_provider_execution"
        or preflight["exposed_profile_credential_never_used_by_experiment"] is not True
        or budget["spend_authority"] is not False
        or budget["application_ledger_is_provider_proof"] is not False
        or budget["provider_reported_spend_usd"] is not None
    ):
        raise ModalL4EvidenceError("offline refusal semantics drifted")
    blocked = {
        item["claim_id"]: item["state"]
        for item in claims["claims"]
        if item["claim_id"] in BLOCKED_CLAIM_IDS
    }
    if set(blocked) != set(BLOCKED_CLAIM_IDS) or any(
        state != "not_applicable" for state in blocked.values()
    ):
        raise ModalL4EvidenceError("claims blocked by construction are not closed")
    chain = budget["budget_chain"]
    if budget["budget_chain_sha256"] != _sha256_uri(
        _compact_json(chain).encode("utf-8")
    ):
        raise ModalL4EvidenceError("budget chain hash does not verify")
    if (
        chain["hard_cap_usd"] != canonical_decimal(HARD_CAP_USD)
        or Decimal(chain["compute_planned_usd"]) + Decimal(chain["storage_planned_usd"])
        != Decimal(chain["total_planned_usd"])
        or Decimal(chain["total_planned_usd"]) + Decimal(chain["untouched_margin_usd"])
        != HARD_CAP_USD
        or chain["rates"]["gpu_function_usd_per_second"]
        != canonical_decimal(GPU_FUNCTION_USD_PER_SECOND)
        or chain["rates"]["cpu_function_usd_per_second"]
        != canonical_decimal(CPU_FUNCTION_USD_PER_SECOND)
    ):
        raise ModalL4EvidenceError("budget chain does not reconcile")
    encoded = json.dumps(
        [plan, preflight, claims, budget, contract], allow_nan=False, ensure_ascii=True
    )
    if "NaN" in encoded or "Infinity" in encoded:
        raise ModalL4EvidenceError("non-finite value found")


def _require_cleared_exposure(document: Mapping[str, Any]) -> None:
    """Refuse a published result until the exposure gate verdict is cleared."""

    forbidden = sorted(
        key
        for key in document
        if key
        not in {
            "gate",
            "cleared",
            "exposed_profile_credential_never_used_by_experiment",
            "exposed_profile_credential_revocation_confirmed",
            "fresh_local_profile_created_without_sharing",
            "fresh_profile_shared_anywhere",
            "confirmed_by",
            "confirmed_at",
            "reason",
            "records_credential_values",
            "action",
        }
    )
    if forbidden:
        raise ModalL4EvidenceError(
            "credential-exposure verdict carries fields outside its closed "
            "allowlist: " + ", ".join(forbidden)
        )
    if (
        document.get("cleared") is not True
        or document.get("exposed_profile_credential_never_used_by_experiment")
        is not True
        or document.get("exposed_profile_credential_revocation_confirmed") is not True
        or document.get("fresh_local_profile_created_without_sharing") is not True
        or document.get("fresh_profile_shared_anywhere") is not False
        or document.get("records_credential_values") is not False
        or document.get("confirmed_by") not in CREDENTIAL_EXPOSURE_CONFIRMERS
    ):
        raise ModalL4EvidenceError(
            "credential-exposure gate is not cleared; results are refused until "
            "revocation and fresh-profile creation are confirmed"
        )


def _require_all_passed(gate: Mapping[str, Any]) -> None:
    observations = gate.get("canaries")
    if not isinstance(observations, list) or len(observations) != 2:
        raise ModalL4EvidenceError("memory gate must contain both canaries")
    modes = set()
    for observation in observations:
        verdict = evaluate_memory_gate(observation)
        modes.add(verdict["mode"])
        if not verdict["passed"]:
            raise ModalL4EvidenceError(
                f"memory gate failed for the {verdict['mode']} canary: "
                + ", ".join(verdict["failures"])
            )
    if modes != {"eager", "compiled"}:
        raise ModalL4EvidenceError("memory gate is missing a canary mode")
    if gate.get("tuning_applied") is not False:
        raise ModalL4EvidenceError("memory gate reports tuning; results are refused")


def _verify_result_ledger(
    document: Mapping[str, Any], *, source_head: str, experiment_nonce: str
) -> None:
    """Validate the application ledger comprehensively, reusing the trusted core.

    Rather than inspecting ``reserved_usd`` alone, this reuses
    :func:`verify_ledger_document` -- the same seal, event hash-chain,
    per-lifecycle state machine, and reconciled-total validation the live
    ledger and the orchestration analyzer use -- bound to this plan, source
    head, and nonce, and then confirms the exact planned envelope within the
    hard cap.
    """

    if document.get("is_provider_proof") is not False:
        raise ModalL4EvidenceError("application ledger must not claim provider proof")
    try:
        summary = verify_ledger_document(
            document,
            plan=build_default_plan(),
            source_head=source_head,
            experiment_nonce=experiment_nonce,
        )
    except ModalL4ContractError as exc:
        raise ModalL4EvidenceError(
            f"application ledger does not verify: {exc}"
        ) from exc
    if summary["reserved_usd"] != canonical_decimal(TOTAL_PLANNED_USD):
        raise ModalL4EvidenceError(
            "application ledger total differs from the planned envelope"
        )
    if Decimal(summary["reserved_usd"]) > HARD_CAP_USD:
        raise ModalL4EvidenceError("application ledger exceeds the hard cap")


def _expected_memory_gate_envelope(orchestration: Mapping[str, Any]) -> dict[str, Any]:
    gate = orchestration["memory_gate"]
    return {
        "tuning_applied": gate["tuning_applied"],
        "canaries": [entry["receipt"]["observation"] for entry in gate["canaries"]],
    }


def _expected_credential_exposure_projection(
    envelope: Mapping[str, Any],
) -> dict[str, Any]:
    return {key: value for key, value in envelope.items() if key != "reason"}


def _require_envelope_binds_to_orchestration(
    name: str,
    envelope: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if dict(envelope) != dict(expected):
        raise ModalL4EvidenceError(
            f"{name} is not bound to the orchestration receipt; a mix-and-match "
            "envelope was refused"
        )


def _read_bundle_cells(root: Path) -> dict[str, Any]:
    cells_dir = root / CELLS_DIRECTORY
    if cells_dir.is_symlink() or not cells_dir.is_dir():
        raise ModalL4EvidenceError("result bundle is missing the cells directory")
    cells: dict[str, Any] = {}
    for entry in sorted(cells_dir.iterdir()):
        if entry.is_symlink() or not entry.is_file() or entry.suffix != ".json":
            raise ModalL4EvidenceError(
                "the cells directory may contain only non-symlink JSON files"
            )
        _scan_privacy(
            f"{CELLS_DIRECTORY}/{entry.name}",
            read_bounded_regular_text(entry, MAX_EVIDENCE_ARTIFACT_BYTES),
        )
        cells[entry.stem] = _read_json(entry, require_canonical=False)
    return cells


def _require_no_foreign_bundle_entries(root: Path) -> None:
    """Refuse any bundle entry outside the closed result allowlist."""

    permitted = (
        set(RESULT_ENVELOPE_FILES)
        | set(RESULT_ARTIFACT_FILES)
        | {ORCHESTRATION_FILE, RESULT_CHECKSUM_FILE, CELLS_DIRECTORY}
    )
    foreign = sorted(
        entry.name
        for entry in root.iterdir()
        if entry.name not in permitted or entry.is_symlink()
    )
    if foreign:
        raise ModalL4EvidenceError(
            "result bundle contains files, directories, or symlinks outside its "
            "closed allowlist: " + ", ".join(foreign)
        )


def verify_result_bundle(bundle_dir: Path) -> dict[str, Any]:
    """Adjudicate a completed Modal L4 run; every failure is terminal.

    The orchestration receipt is the single sealed source of truth. It is
    validated comprehensively by ``analyze_modal_run`` (its own seal, the exact
    call sequence, both memory canaries, the application ledger, the teardown,
    the sealed cell inventory), and then every standalone envelope file is bound
    by exact equality to the corresponding orchestration content -- ledger,
    teardown, memory gate, attempts, rate receipt, exposure, and limitations --
    so a mix-and-match of a valid orchestration with a tampered or swapped
    envelope (or the reverse) is impossible. Extra files, directories, and
    symlinks are refused.
    """

    assert_provider_sdk_absent()
    root = bundle_dir.resolve()
    if bundle_dir.is_symlink() or not root.is_dir():
        raise ModalL4EvidenceError("result bundle must be a non-symlink directory")
    _require_no_foreign_bundle_entries(root)
    for name in RESULT_ENVELOPE_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ModalL4EvidenceError(f"result bundle is missing {name}")
        _scan_privacy(
            name, read_bounded_regular_text(path, MAX_EVIDENCE_ARTIFACT_BYTES)
        )
    orchestration_path = root / ORCHESTRATION_FILE
    if orchestration_path.is_symlink() or not orchestration_path.is_file():
        raise ModalL4EvidenceError(f"result bundle is missing {ORCHESTRATION_FILE}")
    _scan_privacy(
        ORCHESTRATION_FILE,
        read_bounded_regular_text(orchestration_path, MAX_EVIDENCE_ARTIFACT_BYTES),
    )
    for name in (*RESULT_ARTIFACT_FILES, RESULT_CHECKSUM_FILE):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ModalL4EvidenceError(f"result bundle is missing {name}")
        _scan_privacy(
            name, read_bounded_regular_text(path, MAX_EVIDENCE_ARTIFACT_BYTES)
        )
    orchestration = _read_json(orchestration_path, require_canonical=False)
    cells = _read_bundle_cells(root)

    # The single trusted validation: consume the sealed orchestration receipt
    # and the thirty-two sealed inner cell receipts and re-derive every fact.
    try:
        analysis = analyze_modal_run(orchestration=orchestration, cells=cells)
    except (ModalL4ResultsError, CrossoverResultsError) as exc:
        raise ModalL4EvidenceError(
            f"provider result validation or analysis failed: {exc}"
        ) from exc
    source_head = analysis["source_head"]
    experiment_nonce = analysis["experiment_nonce"]

    # Cross-bind every standalone envelope file to the orchestration content by
    # exact equality, so a swapped or tampered envelope is refused.
    exposure = _read_json(root / "credential-exposure.json", require_canonical=False)
    _require_cleared_exposure(exposure)
    _require_envelope_binds_to_orchestration(
        "credential-exposure.json",
        _expected_credential_exposure_projection(exposure),
        orchestration["credential_exposure"],
    )
    attempts = _read_json(root / "modal-attempt-receipts.json", require_canonical=False)
    verdict = evaluate_attempt_receipts(attempts.get("receipts"))
    if not verdict["valid"]:
        raise ModalL4EvidenceError(
            "run is invalidated by provider attempt receipts: "
            + ", ".join(
                sorted({finding["observation"] for finding in verdict["findings"]})
            )
        )
    _require_envelope_binds_to_orchestration(
        "modal-attempt-receipts.json",
        attempts,
        {"receipts": orchestration["attempt_receipts"]},
    )
    rate_receipt = _read_json(root / "modal-rate-receipt.json", require_canonical=False)
    verify_official_rate_receipt(rate_receipt)
    _require_envelope_binds_to_orchestration(
        "modal-rate-receipt.json", rate_receipt, orchestration["rate_receipt"]
    )
    memory_gate = _read_json(root / "memory-gate.json", require_canonical=False)
    _require_all_passed(memory_gate)
    _require_envelope_binds_to_orchestration(
        "memory-gate.json",
        memory_gate,
        _expected_memory_gate_envelope(orchestration),
    )
    ledger = _read_json(root / "application-ledger.json", require_canonical=False)
    _verify_result_ledger(
        ledger, source_head=source_head, experiment_nonce=experiment_nonce
    )
    _require_envelope_binds_to_orchestration(
        "application-ledger.json", ledger, orchestration["ledger"]
    )
    teardown_envelope = _read_json(
        root / "modal-teardown.json", require_canonical=False
    )
    teardown = evaluate_teardown_receipt(teardown_envelope)
    if not teardown["complete"]:
        raise ModalL4EvidenceError(
            "teardown receipt is incomplete: " + ", ".join(teardown["failures"])
        )
    _require_envelope_binds_to_orchestration(
        "modal-teardown.json",
        teardown_envelope,
        {
            key: value
            for key, value in orchestration["teardown"].items()
            if key != "adjudication"
        },
    )
    limitations = _read_json(root / "modal-limitations.json", require_canonical=False)
    if sorted(limitations.get("uncontrolled", ())) != sorted(
        UNCONTROLLED_CACHE_LIMITATIONS
    ):
        raise ModalL4EvidenceError(
            "uncontrolled provider limitations are not published"
        )
    _require_envelope_binds_to_orchestration(
        "modal-limitations.json",
        limitations,
        {"uncontrolled": orchestration["uncontrolled_limitations"]},
    )

    # The freshly captured official-rate provenance is bound to the orchestration
    # and independently recomputed -- entirely offline, no network -- from the
    # standalone rate receipt and its hashed capture, so a published result
    # proves list-rate provenance without a fetch and a swapped capture is
    # refused.
    rate_refresh = _read_json(root / "modal-rate-refresh.json", require_canonical=False)
    _require_envelope_binds_to_orchestration(
        "modal-rate-refresh.json", rate_refresh, orchestration["rate_refresh"]
    )
    capture = rate_refresh.get("capture")
    verification = rate_refresh.get("verification")
    if not isinstance(capture, Mapping) or not isinstance(verification, Mapping):
        raise ModalL4EvidenceError(
            "rate refresh must carry a capture and a verification object"
        )
    try:
        recomputed_verification = verify_rate_refresh(rate_receipt, capture=capture)
    except RateRefreshError as exc:
        raise ModalL4EvidenceError(
            f"rate refresh does not bind to the standalone rate receipt: {exc}"
        ) from exc
    if dict(verification) != recomputed_verification:
        raise ModalL4EvidenceError(
            "rate refresh verification does not recompute from the standalone "
            "rate receipt and its captured documents"
        )

    # The source-checkout receipt and the boolean-only local-profile verdict are
    # each bound to the orchestration content that analyze_modal_run already
    # validated, so a swapped or tampered standalone receipt is refused.
    source_checkout = _read_json(root / "source-checkout.json", require_canonical=False)
    _require_envelope_binds_to_orchestration(
        "source-checkout.json", source_checkout, orchestration["source_checkout"]
    )
    profile = _read_json(root / "profile-authentication.json", require_canonical=False)
    try:
        verify_profile_authentication(profile)
    except ModalL4ContractError as exc:
        raise ModalL4EvidenceError(
            f"profile authentication verdict is invalid: {exc}"
        ) from exc
    _require_envelope_binds_to_orchestration(
        "profile-authentication.json",
        profile,
        orchestration["profile_authentication"],
    )

    supported_blocked = sorted(
        item["claim_id"]
        for item in analysis["claim_matrix"].get("claims", ())
        if isinstance(item, Mapping)
        and item.get("claim_id") in BLOCKED_CLAIM_IDS
        and item.get("state") == "supported"
    )
    if supported_blocked:
        raise ModalL4EvidenceError(
            "claims unsupported by construction are marked supported: "
            + ", ".join(supported_blocked)
        )

    # Regenerate every deterministic artifact from the re-derived analysis and
    # require byte identity, so a hand-edited analysis document, claim matrix,
    # report, or crossover figure cannot ride along with a valid orchestration.
    for name, expected in _result_artifact_documents(analysis).items():
        actual = read_bounded_regular_bytes(root / name, MAX_EVIDENCE_ARTIFACT_BYTES)
        if actual != expected:
            raise ModalL4EvidenceError(
                f"{name} does not match the regenerated artifact"
            )

    # The checksum manifest is verified last: every top-level file it names
    # recomputes to its recorded digest, the allowlist is exact, and nothing is
    # missing, so any residual byte drift is terminal.
    _verify_result_checksums(root)
    return {
        "verified": True,
        "protocol_id": PROTOCOL_ID,
        "delegated_statistical_verifier": DELEGATED_STATISTICAL_VERIFIER,
        "reused_statistical_primitives": list(REUSED_STATISTICAL_PRIMITIVE_NAMES),
        "pair_count": analysis["pair_count"],
        "claim_matrix": analysis["claim_matrix"],
        "provider_reported_spend_usd": teardown["provider_reported_spend_usd"],
        "provider_reported_spend_null_reason": teardown[
            "provider_reported_spend_null_reason"
        ],
    }


def _render_result_svg(analysis: Mapping[str, Any]) -> str:
    """Render the controlled mean-difference curve as a deterministic figure.

    The curve is compiled-minus-eager cumulative seconds across the 144
    controlled requests. It starts positive (compile and warmup overhead) and
    ends negative (the sustained speedup), so it crosses zero once: the sealed
    provider-conditioned crossover. A zero baseline and, when observed, the
    sustained-crossing request are marked. Every coordinate is formatted to two
    decimals so two builds of the same run are byte-identical.
    """

    controlled = analysis["crossover_inference"]["controlled"]
    curve = [float(value) for value in controlled["mean_difference_curve"]]
    count = len(curve)
    left, right, top, bottom = 60.0, 800.0, 24.0, 200.0
    low = min(curve)
    high = max(curve)
    span = (high - low) or 1.0
    step = right - left
    denominator = float(count - 1) if count > 1 else 1.0

    def _x(index: int) -> float:
        return left + step * (index / denominator)

    def _y(value: float) -> float:
        return bottom - (bottom - top) * ((value - low) / span)

    points = " ".join(f"{_x(i):.2f},{_y(v):.2f}" for i, v in enumerate(curve))
    baseline = ""
    if low <= 0.0 <= high:
        zero_y = _y(0.0)
        baseline = (
            f'<line x1="{left:.2f}" y1="{zero_y:.2f}" x2="{right:.2f}" '
            f'y2="{zero_y:.2f}" stroke="#98a2b3" stroke-dasharray="4 4"/>'
            f'<text x="{left:.2f}" y="{zero_y - 6:.2f}" font-family="system-ui" '
            'font-size="12" fill="#475467">compiled = eager</text>'
        )
    crossing = ""
    sustained = controlled["aggregate_sustained_crossing_request_count"]
    if isinstance(sustained, int) and not isinstance(sustained, bool):
        if 1 <= sustained <= count:
            crossing_x = _x(sustained - 1)
            crossing = (
                f'<line x1="{crossing_x:.2f}" y1="{top:.2f}" x2="{crossing_x:.2f}" '
                f'y2="{bottom:.2f}" stroke="#b54708"/>'
                f'<text x="{crossing_x + 4:.2f}" y="{top + 12:.2f}" '
                'font-family="system-ui" font-size="12" fill="#b54708">'
                f"sustained crossing @ request {sustained}</text>"
            )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="840" height="240" viewBox="0 0 840 240" role="img" aria-labelledby="title desc">
<title id="title">Modal L4 controlled compiled-minus-eager cumulative time</title>
<desc id="desc">Mean cumulative compiled-minus-eager seconds across 144 controlled requests, with the zero baseline and the sustained provider-conditioned crossover request marked.</desc>
<rect width="840" height="240" fill="white"/>
<text x="24" y="20" font-family="system-ui" font-size="16" fill="#17202a">Controlled crossover: compiled minus eager cumulative seconds</text>
{baseline}
<polyline fill="none" stroke="#175cd3" stroke-width="2" points="{points}"/>
{crossing}
</svg>
"""


def _render_result_report(analysis: Mapping[str, Any]) -> str:
    """Render a deterministic, sanitized HTML summary of a completed run.

    The report projects the re-derived claim matrix, the controlled crossover
    inference (first and sustained crossings, the right-censored bootstrap
    interval, and the terminal sign-flip p-value), and the standing provider
    limitations. It states plainly that provider-reported spend, pure causal
    compilation, host page-cache control, and natural causal speedup remain
    unsupported on this provider.
    """

    controlled = analysis["crossover_inference"]["controlled"]
    interval = controlled["bootstrap_sustained_crossing_interval"]

    def _endpoint_text(endpoint: Mapping[str, Any]) -> str:
        count = endpoint.get("request_count")
        return f"request {count}" if count is not None else "open (right-censored)"

    interval_lower = _endpoint_text(interval["lower"])
    interval_median = _endpoint_text(interval["median"])
    interval_upper = _endpoint_text(interval["upper"])
    claim_rows_parts: list[str] = []
    for claim in analysis["claim_matrix"]["claims"]:
        blockers = ", ".join(claim.get("blockers", [])) or "none"
        claim_rows_parts.append(
            "<tr>"
            f"<td>{html.escape(str(claim['claim_id']))}</td>"
            f"<td>{html.escape(str(claim['state']))}</td>"
            f"<td>{html.escape(blockers)}</td>"
            "</tr>"
        )
    claim_rows = "".join(claim_rows_parts)
    limitation_items = "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in analysis["uncontrolled_limitations"]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Modal L4 vLLM crossover result</title>
  <style>
    body {{ color:#17202a; font:16px/1.5 system-ui,sans-serif; margin:2rem auto; max-width:980px; padding:0 1rem; }}
    .notice {{ background:#fffaeb; border-left:.4rem solid #b54708; padding:1rem; }}
    table {{ border-collapse:collapse; width:100%; }}
    th,td {{ border-bottom:1px solid #d0d5dd; padding:.55rem; text-align:left; }}
    figure {{ margin:1.5rem 0; }}
  </style>
</head>
<body>
  <h1>Modal L4 crossover result for the Qwen3-8B vLLM protocol</h1>
  <p class="notice"><strong>Provider-conditioned, descriptive result.</strong>
  Provider-reported spend stays null until an external sanitized receipt exists.
  Pure causal compilation, host page-cache control, and natural causal speedup
  are unsupported by construction on this provider.</p>
  <p>Source head {html.escape(str(analysis["source_head"]))}; experiment nonce
  {html.escape(str(analysis["experiment_nonce"]))}. {analysis["cell_count"]}
  sealed single-use cells form {analysis["pair_count"]} adjacent eager/compiled
  lifecycle pairs, the whole-pair inferential unit.</p>
  <h2>Controlled crossover inference</h2>
  <p>Aggregate first crossing at request
  {html.escape(str(controlled["aggregate_first_crossing_request_count"]))};
  aggregate sustained crossing at request
  {html.escape(str(controlled["aggregate_sustained_crossing_request_count"]))}.
  The {controlled["resample_count"]}-resample whole-pair bootstrap places the
  sustained crossing between {html.escape(interval_lower)} and
  {html.escape(interval_upper)} (median {html.escape(interval_median)}, interval
  state {html.escape(str(interval["state"]))}; right-censored at request
  {html.escape(str(interval["censor_at_request_count"]))}). Terminal-effect
  sign-flip p-value
  {html.escape(str(controlled["terminal_effect_sign_flip_p_value"]))};
  request-level resampling {html.escape(str(controlled["request_level_resampling"]))}.</p>
  <figure><img src="crossover.svg" alt="Controlled compiled-minus-eager cumulative time with the crossover marked" width="840" height="240"></figure>
  <h2>Uncontrolled on this provider</h2>
  <ul>{limitation_items}</ul>
  <h2>Claim matrix</h2>
  <table><thead><tr><th>Claim</th><th>State</th><th>Blockers</th></tr></thead>
  <tbody>{claim_rows}</tbody></table>
</body>
</html>
"""


def _result_artifact_documents(analysis: Mapping[str, Any]) -> dict[str, bytes]:
    """Return the four regenerated result artifacts as canonical bytes."""

    return {
        "analysis.json": _canonical_json(analysis).encode("utf-8"),
        "claim-matrix.json": _canonical_json(analysis["claim_matrix"]).encode("utf-8"),
        "crossover.svg": _render_result_svg(analysis).encode("utf-8"),
        "report.html": _render_result_report(analysis).encode("utf-8"),
    }


def _result_bundle_documents(
    orchestration: Mapping[str, Any],
    analysis: Mapping[str, Any],
    *,
    exposure_reason: str,
) -> dict[str, bytes]:
    """Project every top-level result-bundle file, plus the checksum manifest.

    Each standalone envelope is a redundant projection of the sealed
    orchestration content, so the verifier can cross-bind it back. The four
    artifacts are regenerated from the re-derived analysis. Everything is
    canonical JSON (or deterministic text) so two builds of the same run are
    byte-identical, and the SHA256SUMS manifest binds the whole set.
    """

    envelopes: dict[str, Any] = {
        "application-ledger.json": orchestration["ledger"],
        "credential-exposure.json": {
            **orchestration["credential_exposure"],
            "reason": exposure_reason,
        },
        "memory-gate.json": _expected_memory_gate_envelope(orchestration),
        "modal-attempt-receipts.json": {"receipts": orchestration["attempt_receipts"]},
        "modal-limitations.json": {
            "uncontrolled": orchestration["uncontrolled_limitations"]
        },
        "modal-rate-receipt.json": orchestration["rate_receipt"],
        "modal-rate-refresh.json": orchestration["rate_refresh"],
        "modal-teardown.json": {
            key: value
            for key, value in orchestration["teardown"].items()
            if key != "adjudication"
        },
        "profile-authentication.json": orchestration["profile_authentication"],
        "source-checkout.json": orchestration["source_checkout"],
    }
    documents: dict[str, bytes] = {
        ORCHESTRATION_FILE: _canonical_json(orchestration).encode("utf-8")
    }
    for name, value in envelopes.items():
        documents[name] = _canonical_json(value).encode("utf-8")
    documents.update(_result_artifact_documents(analysis))
    missing = set(RESULT_HASHED_FILES) - set(documents)
    if missing:
        raise ModalL4EvidenceError(
            "internal error: result bundle is missing " + ", ".join(sorted(missing))
        )
    checksums = "\n".join(
        f"{_sha256(documents[name])}  {name}" for name in RESULT_HASHED_FILES
    )
    documents[RESULT_CHECKSUM_FILE] = (checksums + "\n").encode("utf-8")
    return documents


def _verify_result_checksums(root: Path) -> None:
    """Recompute every manifest digest; the allowlist must be exact and whole."""

    checksum_text = read_bounded_regular_text(
        root / RESULT_CHECKSUM_FILE, MAX_EVIDENCE_ARTIFACT_BYTES
    )
    names: set[str] = set()
    for line in checksum_text.splitlines():
        match = _CHECKSUM.fullmatch(line)
        if match is None:
            raise ModalL4EvidenceError("result SHA256SUMS contains a malformed line")
        digest, name = match.groups()
        if name in names or name not in RESULT_HASHED_FILES:
            raise ModalL4EvidenceError("result SHA256SUMS allowlist differs")
        names.add(name)
        actual = read_bounded_regular_bytes(root / name, MAX_EVIDENCE_ARTIFACT_BYTES)
        if digest != _sha256(actual):
            raise ModalL4EvidenceError(f"result checksum mismatch for {name}")
    if names != set(RESULT_HASHED_FILES):
        raise ModalL4EvidenceError("result SHA256SUMS is incomplete")


def _require_published_result(orchestration: Mapping[str, Any]) -> None:
    """Refuse to bundle anything but a published, complete result receipt."""

    if (
        orchestration.get("kind") != "llmtracefx.modal_l4_crossover.result"
        or orchestration.get("published") is not True
        or orchestration.get("status") != "complete"
    ):
        raise ModalL4EvidenceError(
            "execution workspace does not hold a published, complete result; a "
            "refusal, incomplete teardown, or invalidated run cannot be bundled"
        )


def _read_workspace_cells(workspace: Path) -> dict[str, Any]:
    cells_dir = workspace / CELLS_DIRECTORY
    if cells_dir.is_symlink() or not cells_dir.is_dir():
        raise ModalL4EvidenceError("execution workspace is missing its cells directory")
    cells: dict[str, Any] = {}
    for entry in sorted(cells_dir.iterdir()):
        if entry.is_symlink() or not entry.is_file() or entry.suffix != ".json":
            raise ModalL4EvidenceError(
                "the workspace cells directory may contain only non-symlink JSON files"
            )
        cells[entry.stem] = _read_json(entry, require_canonical=False)
    return cells


def _read_exposure_reason(workspace: Path) -> str:
    exposure_path = workspace / "credential-exposure.json"
    if exposure_path.is_symlink() or not exposure_path.is_file():
        raise ModalL4EvidenceError(
            "execution workspace is missing its credential-exposure verdict"
        )
    exposure = _read_json(exposure_path, require_canonical=False)
    reason = exposure.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ModalL4EvidenceError(
            "execution workspace credential-exposure verdict is missing its "
            "confirmation reason"
        )
    return reason


def build_result_bundle(execution_workspace: Path, output_dir: Path) -> dict[str, Any]:
    """Build the deterministic, closed result bundle from an execution workspace.

    The workspace must hold a *published, complete* run: its sealed
    orchestration receipt and exactly 32 sealed cell receipts. The run is
    re-derived and validated by ``analyze_modal_run`` before anything is
    written, so a refusal, an incomplete teardown, or an invalidated attempt
    can never be published. Every standalone envelope is a projection of the
    sealed orchestration content, the four analysis artifacts are regenerated
    from the re-derived analysis, and a SHA256SUMS manifest binds the set. The
    bundle is verified in place before returning; the provider SDK is never
    imported.
    """

    assert_provider_sdk_absent()
    workspace = execution_workspace.resolve()
    if execution_workspace.is_symlink() or not workspace.is_dir():
        raise ModalL4EvidenceError(
            "execution workspace must be a non-symlink directory"
        )
    orchestration_path = workspace / ORCHESTRATION_FILE
    if orchestration_path.is_symlink() or not orchestration_path.is_file():
        raise ModalL4EvidenceError(
            "execution workspace has no published orchestration receipt; a "
            "refusal writes none"
        )
    _scan_privacy(
        ORCHESTRATION_FILE,
        read_bounded_regular_text(orchestration_path, MAX_EVIDENCE_ARTIFACT_BYTES),
    )
    orchestration = _read_json(orchestration_path, require_canonical=False)
    _require_published_result(orchestration)
    cells = _read_workspace_cells(workspace)
    exposure_reason = _read_exposure_reason(workspace)

    # The single trusted validation. Any seal, binding, ledger, teardown, or
    # attempt failure is terminal here, before a single artifact is written.
    try:
        analysis = analyze_modal_run(orchestration=orchestration, cells=cells)
    except (ModalL4ResultsError, CrossoverResultsError) as exc:
        raise ModalL4EvidenceError(
            f"execution workspace does not analyze as a valid published run: {exc}"
        ) from exc

    documents = _result_bundle_documents(
        orchestration, analysis, exposure_reason=exposure_reason
    )
    cell_texts = {cell_id: _canonical_json(cells[cell_id]) for cell_id in sorted(cells)}

    # Sanitize every artifact before it touches the disk.
    for name, data in documents.items():
        _scan_privacy(name, data.decode("utf-8"))
    for cell_id, text in cell_texts.items():
        _scan_privacy(f"{CELLS_DIRECTORY}/{cell_id}.json", text)

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink():
        raise ModalL4EvidenceError("result bundle directory must not be a symlink")
    expected_entries = set(documents) | {CELLS_DIRECTORY}
    unexpected = sorted(
        {entry.name for entry in output_dir.iterdir()} - expected_entries
    )
    if unexpected:
        raise ModalL4EvidenceError(
            "result bundle directory contains unexpected entries: "
            + ", ".join(unexpected)
        )
    for name, data in documents.items():
        atomic_write_text(output_dir / name, data.decode("utf-8"))
    cells_dir = output_dir / CELLS_DIRECTORY
    cells_dir.mkdir(exist_ok=True)
    for cell_id, text in cell_texts.items():
        atomic_write_text(cells_dir / f"{cell_id}.json", text)

    # Verify the freshly written bundle end to end before returning.
    verify_result_bundle(output_dir)
    return {
        "built": True,
        "protocol_id": PROTOCOL_ID,
        "pair_count": analysis["pair_count"],
        "cell_count": analysis["cell_count"],
        "files": list(RESULT_TOP_LEVEL_FILES),
        "cell_files": len(cell_texts),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmtracefx-modal-l4-crossover-evidence", allow_abbrev=False
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    build = subparsers.add_parser("build", allow_abbrev=False)
    build.add_argument("--output-dir", required=True, type=Path)
    build.add_argument("--repo-root", required=True, type=Path)
    verify = subparsers.add_parser("verify", allow_abbrev=False)
    verify.add_argument("--bundle-dir", required=True, type=Path)
    verify.add_argument("--repo-root", type=Path)
    build_results = subparsers.add_parser("build-results", allow_abbrev=False)
    build_results.add_argument("--execution-workspace", required=True, type=Path)
    build_results.add_argument("--output-dir", required=True, type=Path)
    results = subparsers.add_parser("verify-results", allow_abbrev=False)
    results.add_argument("--bundle-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action == "build":
            build_offline_bundle(args.output_dir, repo_root=args.repo_root)
        elif args.action == "verify":
            verify_offline_bundle(args.bundle_dir, repo_root=args.repo_root)
        elif args.action == "build-results":
            build_result_bundle(args.execution_workspace, args.output_dir)
        else:
            verify_result_bundle(args.bundle_dir)
        return 0
    except (
        OSError,
        ValueError,
        ModalL4ContractError,
        ModalL4EvidenceError,
        CrossoverResultsError,
    ) as exc:
        print(f"Modal L4 crossover evidence failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
