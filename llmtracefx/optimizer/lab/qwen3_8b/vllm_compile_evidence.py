"""Pure offline evidence builder and verifier for the Qwen3 vLLM experiment."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from ...collectors._shared import atomic_write_text
from ...workloads.catalog import workload_by_id
from ...workloads.evaluators import evaluate_workload
from ...workloads.schema import WorkloadCategory
from .vllm_compile import (
    APPROVED_PLAN_SHA256,
    CELLS,
    EXPECTED_MODEL_BYTES,
    EXPECTED_MODEL_FILE_COUNT,
    HARD_CAP_USD,
    MODEL_ID,
    MODEL_REVISION,
    OFFICIAL_VLLM_IMAGE_DIGEST,
    REQUESTS_PER_CELL,
    HardwareIdentity,
    VLLMCompilePlan,
    canonical_decimal,
    canonical_json,
    validate_hardware_identity,
    workload_descriptors,
)

BUNDLE_FILES = (
    "README.md",
    "experiment-contract.json",
    "pricing-snapshot.json",
    "model-inventory.json",
    "runtime-image.json",
    "workload-contract.json",
    "lifecycle-records.jsonl",
    "request-records.jsonl",
    "correctness-report.json",
    "break-even.json",
    "cost-ledger.json",
    "teardown-report.json",
    "claim-matrix.json",
    "report.html",
    "break-even.svg",
    "evidence_bundle.py",
    "SHA256SUMS",
)
JSON_FILES = frozenset(name for name in BUNDLE_FILES if name.endswith(".json"))
JSONL_FILES = frozenset(name for name in BUNDLE_FILES if name.endswith(".jsonl"))
PROVENANCE = frozenset(
    {
        "client_observed",
        "vllm",
        "cuda",
        "modal_provider",
        "model_reported",
        "derived",
    }
)
_RAW_REQUEST_FIELD_PROVENANCE = {
    "started_at": "client_observed",
    "ended_at": "client_observed",
    "wall_clock_seconds": "client_observed",
    "input_token_count": "derived",
    "input_token_ids_sha256": "derived",
    "output_token_count": "derived",
    "output_tokens_per_second": "derived",
    "output_rate_basis": "derived",
    "output_token_ids": "model_reported",
    "decoded_output": "model_reported",
    "finish_reason": "model_reported",
    "ttft_seconds": "vllm",
    "correctness": "derived",
}
_PUBLIC_REQUEST_FIELD_PROVENANCE = {
    "cell_id": "derived",
    "ordinal": "derived",
    "request_id": "derived",
    "workload_id": "derived",
    "context_tier": "derived",
    "repetition": "derived",
    "input_token_count": "derived",
    "input_token_ids_sha256": "derived",
    "output_token_count": "derived",
    "output_token_ids": "model_reported",
    "decoded_output": "model_reported",
    "finish_reason": "model_reported",
    "started_at": "client_observed",
    "ended_at": "client_observed",
    "latency_seconds": "client_observed",
    "ttft_seconds": "vllm",
    "output_tokens_per_second": "derived",
    "output_rate_basis": "derived",
    "terminal": "model_reported",
    "correctness": "derived",
    "evaluator": "derived",
}
RUNTIME_PINS = {
    "python_version": "3.12",
    "vllm_version": "0.28.0",
    "torch_version": "2.13.0+cu130",
    "cuda_version": "13.0",
}
IMAGE_REFERENCE = "vllm/vllm-openai:v0.28.0@" + OFFICIAL_VLLM_IMAGE_DIGEST
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_STRING_BYTES = 131_072
MAX_ITEMS = 250_000
MAX_DEPTH = 20
MAX_OUTPUT_BYTES = 65_536
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_GIT_HEAD = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,31}[a-z0-9])?$")
_EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-" r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.I,
)
_PROVIDER_ID = re.compile(r"\b(?:ap|vo|ta)-[A-Za-z0-9_-]{20,}\b")
_CREDENTIAL = re.compile(
    r"(?<![A-Za-z0-9])(?:hf_[A-Za-z0-9_-]{8,}|"
    r"gh[pousr]_[A-Za-z0-9_-]{8,}|github_pat_[A-Za-z0-9_-]{8,}|"
    r"sk-[A-Za-z0-9_-]{16,}|AKIA[0-9A-Z]{16}|"
    r"modal[_-](?:token|secret)[A-Za-z0-9_-]{8,})",
    re.I,
)
_PRIVATE_KEYS = frozenset(
    {
        "account",
        "account_id",
        "user",
        "user_id",
        "workspace",
        "workspace_id",
        "profile",
        "profile_name",
        "password",
        "api_key",
        "secret",
        "modal_token_id",
        "modal_token_secret",
        "hf_token",
    }
)
_FINISH_REASONS = frozenset({"stop", "length"})
_FUNCTIONS = ("l40s_eager", "l40s_compiled", "h100_eager", "h100_compiled")

README = """# Qwen3 8B vLLM compilation evidence

This bundle reports four vLLM 0.28.0 cells under one fixed workload contract.
Correctness is evaluated offline with deterministic structured JSON and prose
evaluators. Model output is never executed.

Break-even results compare eager and compiled execution on the same accelerator.
Each pair is comparable only when eager and compiled produce identical output
token counts for every request and compiled correctness is not worse. Exact
token-ID divergence is reported separately. Each cold cell is observed once,
so initialization deltas have no variance estimate.
The compiled mode combines vLLM compilation and CUDA graph capture. Component
timings remain null when vLLM does not expose them, and warm compile caches are
outside scope.

Observed crossings are limited to the 12 measured requests. A modeled crossing
is published only when no observed crossing exists and positive savings from an
exact repetition of that 12-request cycle would repay the measured initialization
penalty. Modeled crossings are not observations.

List-rate estimates are inferred from observed client lifecycle duration and the
pinned pricing snapshot. Provider account billing remains separate and may be
unavailable or delayed.

MLX results are outside this comparison because their runtime, quantization, and
hardware scope are incompatible. They are not included in rankings.
"""

VERIFY_SCRIPT = '''"""Verify this evidence bundle with the installed package."""
from __future__ import annotations

import argparse
from pathlib import Path

from llmtracefx.optimizer.lab.qwen3_8b.vllm_compile_evidence import verify_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", nargs="?", type=Path, default=Path(__file__).parent)
    verify_bundle(parser.parse_args().bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


class VLLMCompileEvidenceError(ValueError):
    """Raised when evidence cannot be safely made public or verified."""


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def _strict_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise VLLMCompileEvidenceError(f"{label} schema is not exact")
    return value


def _decimal(value: Any, label: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise VLLMCompileEvidenceError(f"{label} must be decimal")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise VLLMCompileEvidenceError(f"{label} must be decimal") from exc
    if not result.is_finite() or result < 0 or (positive and result <= 0):
        raise VLLMCompileEvidenceError(f"{label} is outside its finite bound")
    return result


def _timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or len(value) > 64:
        raise VLLMCompileEvidenceError(f"{label} must be an ISO timestamp")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise VLLMCompileEvidenceError(f"{label} must be an ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise VLLMCompileEvidenceError(f"{label} must include timezone")
    return parsed.astimezone(timezone.utc)


def _duration(start: Any, end: Any, label: str) -> Decimal:
    delta = _timestamp(end, f"{label} end") - _timestamp(start, f"{label} start")
    seconds = Decimal(delta.days * 86_400 + delta.seconds) + Decimal(
        delta.microseconds
    ) / Decimal(1_000_000)
    if seconds < 0:
        raise VLLMCompileEvidenceError(f"{label} duration is negative")
    return seconds


def _walk_safe(value: Any, *, depth: int = 0, counter: list[int] | None = None) -> None:
    if depth > MAX_DEPTH:
        raise VLLMCompileEvidenceError("evidence nesting exceeds bound")
    if counter is None:
        counter = [0]
    counter[0] += 1
    if counter[0] > MAX_ITEMS:
        raise VLLMCompileEvidenceError("evidence item count exceeds bound")
    if value is None or isinstance(value, bool):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise VLLMCompileEvidenceError("non-finite evidence is forbidden")
        return
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise VLLMCompileEvidenceError("non-finite evidence is forbidden")
        return
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_STRING_BYTES:
            raise VLLMCompileEvidenceError("evidence string exceeds bound")
        if (
            _SHA256.fullmatch(value)
            or _HEX64.fullmatch(value)
            or value == IMAGE_REFERENCE
        ):
            return
        credential_scan = re.sub(r"(?:sha256:)?[0-9a-f]{64}", "", value)
        if (
            "/Users/" in value
            or "/home/" in value
            or _EMAIL.search(value)
            or _UUID.search(value)
            or _PROVIDER_ID.search(value)
            or _CREDENTIAL.search(credential_scan)
        ):
            raise VLLMCompileEvidenceError("private or credential-shaped content")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise VLLMCompileEvidenceError("evidence keys must be strings")
            if key.lower() in _PRIVATE_KEYS:
                raise VLLMCompileEvidenceError("private or credential-shaped key")
            _walk_safe(key, depth=depth + 1, counter=counter)
            _walk_safe(item, depth=depth + 1, counter=counter)
        return
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            _walk_safe(item, depth=depth + 1, counter=counter)
        return
    raise VLLMCompileEvidenceError("unsupported evidence value")


def _verify_seal(value: Mapping[str, Any], field: str) -> None:
    material = dict(value)
    expected = material.pop(field, None)
    if not isinstance(expected, str) or expected != _sha256_json(material):
        raise VLLMCompileEvidenceError(f"{field} does not verify")


def _conversion_inventory() -> list[dict[str, Any]]:
    path = Path(__file__).parent / "data" / "qwen3-8b-conversion-manifest-v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data["source"]["files"])


def _harness_hashes() -> tuple[str, str]:
    workload = {
        "schema_version": "1",
        "descriptors": [item.to_dict() for item in workload_descriptors()],
        "sampling": {
            "max_tokens": 96,
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 20260831,
        },
        "tokenizer": {
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "messages": "single_user_message",
        },
    }
    output = {
        "schema_version": "1",
        "request_terminal_required": True,
        "finish_reason_required": True,
        "input_count_source": "persisted_prompt_token_ids",
        "output_count_source": "request_output_token_ids",
        "decoded_output_max_utf8_bytes": MAX_OUTPUT_BYTES,
        "remote_correctness_evaluation": False,
        "provenance_domains": sorted(PROVENANCE),
    }
    return _sha256_json(workload), _sha256_json(output)


def _validate_contract(value: Any) -> tuple[dict[str, Any], VLLMCompilePlan]:
    contract = _strict_keys(
        value,
        {
            "schema_version",
            "experiment_id",
            "git_head",
            "approved_plan_sha256",
            "reproduction_command_argv",
            "plan",
        },
        "execution contract",
    )
    if (
        contract["schema_version"] != "1"
        or not isinstance(contract["experiment_id"], str)
        or not _SAFE_ID.fullmatch(contract["experiment_id"])
        or not isinstance(contract["git_head"], str)
        or not _GIT_HEAD.fullmatch(contract["git_head"])
        or contract["approved_plan_sha256"] != APPROVED_PLAN_SHA256
    ):
        raise VLLMCompileEvidenceError("execution contract identity is invalid")
    expected_argv = [
        "uv",
        "run",
        "python",
        "-m",
        "llmtracefx.optimizer.lab.qwen3_8b.modal_orchestrator",
        "--approval",
        "<approved-plan-path>",
        "--approval-sha256",
        APPROVED_PLAN_SHA256,
        "--git-head",
        contract["git_head"],
        "--workspace",
        "<repository-root>",
        "--output-dir",
        "<private-output-directory>",
        "--ledger",
        "<private-ledger-path>",
        "--experiment-id",
        contract["experiment_id"],
    ]
    if contract["reproduction_command_argv"] != expected_argv:
        raise VLLMCompileEvidenceError("reproduction command is invalid")
    try:
        plan = VLLMCompilePlan.from_dict(contract["plan"])
    except (ValueError, TypeError) as exc:
        raise VLLMCompileEvidenceError("execution plan is invalid") from exc
    if plan.runtime_pins.to_dict() != RUNTIME_PINS:
        raise VLLMCompileEvidenceError("runtime pins differ from approved runtime")
    if plan.image_digest != OFFICIAL_VLLM_IMAGE_DIGEST:
        raise VLLMCompileEvidenceError("image digest differs from approved image")
    return dict(contract), plan


def _validate_pricing(value: Any, plan: VLLMCompilePlan) -> dict[str, Any]:
    pricing = _strict_keys(
        value,
        {
            "schema_version",
            "retrieved_date",
            "pricing_page",
            "volumes_page",
            "rates_response_sha256",
        },
        "pricing snapshot",
    )
    if pricing["schema_version"] != "1":
        raise VLLMCompileEvidenceError("pricing schema version is invalid")
    for name in ("pricing_page", "volumes_page"):
        page = _strict_keys(pricing[name], {"status", "sha256"}, name)
        if page["status"] != 200 or not _SHA256.fullmatch(str(page["sha256"])):
            raise VLLMCompileEvidenceError(f"{name} policy evidence is invalid")
    try:
        parsed_date = datetime.strptime(pricing["retrieved_date"], "%Y-%m-%d").date()
    except (TypeError, ValueError) as exc:
        raise VLLMCompileEvidenceError("pricing retrieval date is invalid") from exc
    if (
        parsed_date.isoformat() != plan.validation_as_of_date
        or pricing["rates_response_sha256"] != plan.prices.source_sha256
    ):
        raise VLLMCompileEvidenceError("pricing snapshot does not bind the plan")
    return {
        **pricing,
        "rates": plan.prices.to_dict(),
        "pricing_sha256": plan.prices.content_sha256,
    }


def _validate_staging(
    value: Any, plan: VLLMCompilePlan
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not isinstance(value, dict) or value.get("schema_version") != "1":
        raise VLLMCompileEvidenceError("staging receipt schema is invalid")
    expected_keys = {
        "schema_version",
        "plan_sha256",
        "workload_sha256",
        "output_contract_sha256",
        "runtime_sha256",
        "image_sha256",
        "image_digest",
        "model_id",
        "model_revision",
        "model_file_count",
        "model_bytes",
        "inventory",
        "prompts",
        "prompt_ids_sha256",
        "staged_at",
        "receipt_sha256",
    }
    _strict_keys(value, expected_keys, "staging receipt")
    _timestamp(value["staged_at"], "staging receipt")
    _verify_seal(value, "receipt_sha256")
    workload_hash, output_hash = _harness_hashes()
    expected = {
        "plan_sha256": plan.content_sha256,
        "workload_sha256": workload_hash,
        "output_contract_sha256": output_hash,
        "runtime_sha256": _sha256_json(RUNTIME_PINS),
        "image_sha256": _sha256_json({"reference": IMAGE_REFERENCE}),
        "image_digest": OFFICIAL_VLLM_IMAGE_DIGEST,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_file_count": EXPECTED_MODEL_FILE_COUNT,
        "model_bytes": EXPECTED_MODEL_BYTES,
    }
    if any(value.get(key) != expected_item for key, expected_item in expected.items()):
        raise VLLMCompileEvidenceError("staging receipt contract binding mismatch")
    inventory = value.get("inventory")
    if inventory != _conversion_inventory():
        raise VLLMCompileEvidenceError("model inventory differs from exact manifest")
    prompts = value.get("prompts")
    if not isinstance(prompts, list) or len(prompts) != 6:
        raise VLLMCompileEvidenceError("staging must contain six prompts")
    descriptor_hashes = {
        f"{item.context_tier}/{item.workload_id}": item.prompt_sha256
        for item in workload_descriptors()
    }
    prompt_map: dict[str, dict[str, Any]] = {}
    arrays: dict[str, list[int]] = {}
    for prompt in prompts:
        if not isinstance(prompt, dict):
            raise VLLMCompileEvidenceError("staged prompt is invalid")
        required = {
            "key",
            "prompt_sha256",
            "decoded_prompt_sha256",
            "prompt_token_ids",
            "prompt_token_ids_sha256",
            "input_token_count",
        }
        _strict_keys(prompt, required, "staged prompt")
        key, token_ids = prompt["key"], prompt["prompt_token_ids"]
        if (
            not isinstance(key, str)
            or key in prompt_map
            or descriptor_hashes.get(key) != prompt["prompt_sha256"]
            or not _SHA256.fullmatch(str(prompt["decoded_prompt_sha256"]))
            or not isinstance(token_ids, list)
            or not token_ids
            or len(token_ids) > 65_536
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in token_ids
            )
            or prompt["input_token_count"] != len(token_ids)
            or prompt["prompt_token_ids_sha256"] != _sha256_json(token_ids)
        ):
            raise VLLMCompileEvidenceError("staged prompt token evidence is invalid")
        prompt_map[key] = dict(prompt)
        arrays[key] = token_ids
    if set(prompt_map) != set(descriptor_hashes):
        raise VLLMCompileEvidenceError("staged prompt identities are incomplete")
    token_payload = {
        "schema_version": "1",
        "workload_sha256": workload_hash,
        "prompts": arrays,
    }
    if value.get("prompt_ids_sha256") != _sha256_json(token_payload):
        raise VLLMCompileEvidenceError("staged prompt collection hash is invalid")
    return dict(value), prompt_map


def _validate_lifecycle(value: Any, cell_index: int) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VLLMCompileEvidenceError("lifecycle record is invalid")
    if "artifact_sha256" not in value:
        raise VLLMCompileEvidenceError("lifecycle record seal is missing")
    _verify_seal(value, "artifact_sha256")
    required = {"function", "started_at", "events", "ended_at"}
    if set(value) - {"artifact_sha256"} != required:
        raise VLLMCompileEvidenceError("lifecycle record schema is not exact")
    expected_function = (
        "l40s_eager",
        "l40s_compiled",
        "h100_eager",
        "h100_compiled",
    )[cell_index]
    events = value["events"]
    duration = _duration(value["started_at"], value["ended_at"], "cell lifecycle")
    if (
        value["function"] != expected_function
        or duration <= 0
        or not isinstance(events, list)
        or not events
    ):
        raise VLLMCompileEvidenceError("lifecycle identity or duration is invalid")
    public_events: list[dict[str, str]] = []
    for observed in events:
        if not isinstance(observed, dict) or set(observed) != {"received_at", "event"}:
            raise VLLMCompileEvidenceError("lifecycle event wrapper is invalid")
        _timestamp(observed["received_at"], "event received_at")
        event = observed["event"]
        if not isinstance(event, dict) or event.get("provenance") not in PROVENANCE:
            raise VLLMCompileEvidenceError("lifecycle provenance is invalid")
        event_name = event.get("event")
        if not isinstance(event_name, str) or not event_name:
            raise VLLMCompileEvidenceError("lifecycle event name is invalid")
        public_events.append(
            {
                "received_at": observed["received_at"],
                "event": event_name,
                "provenance": event["provenance"],
            }
        )
    if (
        events[0]["event"].get("event") != "container_started"
        or events[-1]["event"].get("event") != "cell_terminal"
        or sum(item["event"].get("event") == "cell_terminal" for item in events) != 1
    ):
        raise VLLMCompileEvidenceError("lifecycle terminal boundaries are invalid")
    first_event_at = public_events[0]["received_at"]
    return {
        "cell_id": CELLS[cell_index].cell_id,
        "function": expected_function,
        "started_at": value["started_at"],
        "ended_at": value["ended_at"],
        "duration_seconds": canonical_decimal(duration),
        "first_event_received_at": first_event_at,
        "invocation_to_first_event_seconds": canonical_decimal(
            _duration(
                value["started_at"],
                first_event_at,
                "invocation to first lifecycle event",
            )
        ),
        "events": public_events,
    }


def _optional_positive(value: Any, label: str) -> str | None:
    if value is None:
        return None
    parsed = _decimal(value, label, positive=True)
    return canonical_decimal(parsed)


def _validate_cell(
    value: Any,
    cell_index: int,
    plan: VLLMCompilePlan,
    receipt: Mapping[str, Any],
    prompts: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(value, dict) or value.get("schema_version") != "1":
        raise VLLMCompileEvidenceError("cell terminal schema is invalid")
    _strict_keys(
        value,
        {
            "schema_version",
            "cell",
            "plan_sha256",
            "staging_receipt_sha256",
            "workload_sha256",
            "output_contract_sha256",
            "runtime_sha256",
            "image_sha256",
            "hardware",
            "runtime",
            "initialization_started_at",
            "initialization_ready_at",
            "compilation_seconds",
            "cuda_graph_seconds",
            "peak_gpu_memory_mib",
            "requests",
            "correctness_evaluated_remotely",
            "terminal",
            "cell_sha256",
        },
        "cell terminal",
    )
    _verify_seal(value, "cell_sha256")
    cell = CELLS[cell_index]
    workload_hash, output_hash = _harness_hashes()
    expected_bindings = {
        "cell": cell.to_dict(),
        "plan_sha256": plan.content_sha256,
        "staging_receipt_sha256": receipt["receipt_sha256"],
        "workload_sha256": workload_hash,
        "output_contract_sha256": output_hash,
        "runtime_sha256": _sha256_json(RUNTIME_PINS),
        "image_sha256": _sha256_json({"reference": IMAGE_REFERENCE}),
        "runtime": RUNTIME_PINS,
        "terminal": True,
        "correctness_evaluated_remotely": False,
    }
    if any(value.get(key) != expected for key, expected in expected_bindings.items()):
        raise VLLMCompileEvidenceError("cell terminal binding is invalid")
    hardware = _strict_keys(
        value.get("hardware"),
        {
            "gpu_name",
            "gpu_count",
            "driver_version",
            "memory_total_mib",
            "memory_used_mib",
        },
        "hardware identity",
    )
    if hardware.get("gpu_count") != 1 or "H200" in str(hardware.get("gpu_name")):
        raise VLLMCompileEvidenceError("hardware substitution is forbidden")
    try:
        validate_hardware_identity(
            cell,
            HardwareIdentity(hardware["gpu_name"], hardware["gpu_count"]),
        )
    except (KeyError, ValueError) as exc:
        raise VLLMCompileEvidenceError("hardware substitution is forbidden") from exc
    total_memory = _optional_positive(
        hardware.get("memory_total_mib"), "total GPU memory"
    )
    _optional_positive(hardware.get("memory_used_mib"), "used GPU memory")
    driver = hardware.get("driver_version")
    if driver is not None and (not isinstance(driver, str) or not driver):
        raise VLLMCompileEvidenceError("CUDA driver must be string or null")
    init_seconds = _duration(
        value.get("initialization_started_at"),
        value.get("initialization_ready_at"),
        "initialization",
    )
    requests = value.get("requests")
    descriptors = workload_descriptors()
    if not isinstance(requests, list) or len(requests) != REQUESTS_PER_CELL:
        raise VLLMCompileEvidenceError("cell must contain exactly 12 requests")
    public_requests: list[dict[str, Any]] = []
    correctness: list[dict[str, Any]] = []
    previous_request_end: datetime | None = None
    initialization_ready = _timestamp(
        value["initialization_ready_at"], "initialization ready"
    )
    for request, descriptor in zip(requests, descriptors, strict=True):
        if not isinstance(request, dict):
            raise VLLMCompileEvidenceError("terminal request is invalid")
        _strict_keys(
            request,
            {
                "ordinal",
                "request_id",
                "workload_id",
                "workload_version",
                "context_tier",
                "repetition",
                "prompt_sha256",
                "warmup",
                "terminal",
                "started_at",
                "ended_at",
                "wall_clock_seconds",
                "input_token_count",
                "input_token_ids_sha256",
                "output_token_count",
                "output_tokens_per_second",
                "output_rate_basis",
                "output_token_ids",
                "decoded_output",
                "finish_reason",
                "ttft_seconds",
                "evaluator_input",
                "correctness",
                "provenance",
                "field_provenance",
            },
            "terminal request",
        )
        key = f"{descriptor.context_tier}/{descriptor.workload_id}"
        prompt = prompts[key]
        output_ids = request.get("output_token_ids")
        elapsed = _decimal(
            request.get("wall_clock_seconds"), "request wall clock", positive=True
        )
        if (
            any(
                request.get(field) != descriptor.to_dict()[field]
                for field in descriptor.to_dict()
            )
            or request.get("terminal") is not True
            or request.get("finish_reason") not in _FINISH_REASONS
            or request.get("correctness") is not None
            or request.get("input_token_count") != prompt["input_token_count"]
            or request.get("input_token_ids_sha256")
            != prompt["prompt_token_ids_sha256"]
            or not isinstance(output_ids, list)
            or not output_ids
            or len(output_ids) > 96
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in output_ids
            )
            or request.get("output_token_count") != len(output_ids)
            or not isinstance(request.get("decoded_output"), str)
            or len(request["decoded_output"].encode("utf-8")) > MAX_OUTPUT_BYTES
            or request.get("provenance") not in PROVENANCE
        ):
            raise VLLMCompileEvidenceError("terminal request contract is invalid")
        field_provenance = request.get("field_provenance")
        if field_provenance != _RAW_REQUEST_FIELD_PROVENANCE:
            raise VLLMCompileEvidenceError("request field provenance is invalid")
        expected_rate = Decimal(len(output_ids)) / elapsed
        observed_rate = _decimal(
            request.get("output_tokens_per_second"), "output rate", positive=True
        )
        if not math.isclose(
            float(observed_rate), float(expected_rate), rel_tol=1e-12, abs_tol=0.0
        ):
            raise VLLMCompileEvidenceError("output rate does not recompute")
        if (
            request.get("output_rate_basis")
            != "output_tokens_per_complete_response_second"
        ):
            raise VLLMCompileEvidenceError("output rate basis is invalid")
        ttft = _optional_positive(request.get("ttft_seconds"), "TTFT")
        if ttft is not None and _decimal(ttft, "TTFT") > elapsed:
            raise VLLMCompileEvidenceError("TTFT exceeds complete request latency")
        request_started = _timestamp(request.get("started_at"), "request start")
        request_ended = _timestamp(request.get("ended_at"), "request end")
        timestamp_elapsed = Decimal((request_ended - request_started).total_seconds())
        if (
            request_started < initialization_ready
            or request_ended < request_started
            or previous_request_end is not None
            and request_started < previous_request_end
            or abs(timestamp_elapsed - elapsed)
            > max(Decimal("0.05"), elapsed * Decimal("0.01"))
        ):
            raise VLLMCompileEvidenceError(
                "request timing is not sequential or does not reconcile"
            )
        previous_request_end = request_ended
        evaluator_input = _strict_keys(
            request.get("evaluator_input"),
            {
                "workload_id",
                "context_tier",
                "decoded_output",
                "output_token_ids",
            },
            "evaluator input",
        )
        if evaluator_input != {
            "workload_id": descriptor.workload_id,
            "context_tier": descriptor.context_tier,
            "decoded_output": request["decoded_output"],
            "output_token_ids": output_ids,
        }:
            raise VLLMCompileEvidenceError("evaluator input differs from response")
        workload = workload_by_id(descriptor.workload_id)
        if workload.category not in {
            WorkloadCategory.STRUCTURED_JSON,
            WorkloadCategory.PROSE_REASONING,
        }:
            raise VLLMCompileEvidenceError("executable workload is forbidden")
        outcome = evaluate_workload(workload, request["decoded_output"])
        evaluator = {
            "success": outcome.success,
            "quality_score": outcome.quality_score,
            "quality_metric": outcome.quality_metric,
            "notes": outcome.notes,
            "evaluator": "evaluate_workload",
            "executed_model_output": False,
        }
        public = {
            "cell_id": cell.cell_id,
            "ordinal": descriptor.ordinal,
            "request_id": descriptor.request_id,
            "workload_id": descriptor.workload_id,
            "context_tier": descriptor.context_tier,
            "repetition": descriptor.repetition,
            "input_token_count": request["input_token_count"],
            "input_token_ids_sha256": request["input_token_ids_sha256"],
            "output_token_count": len(output_ids),
            "output_token_ids": output_ids,
            "decoded_output": request["decoded_output"],
            "finish_reason": request["finish_reason"],
            "started_at": request["started_at"],
            "ended_at": request["ended_at"],
            "latency_seconds": canonical_decimal(elapsed),
            "ttft_seconds": ttft,
            "output_tokens_per_second": canonical_decimal(expected_rate),
            "output_rate_basis": request["output_rate_basis"],
            "terminal": True,
            "provenance": request["provenance"],
            "correctness": outcome.success,
            "evaluator": evaluator,
            "field_provenance": _PUBLIC_REQUEST_FIELD_PROVENANCE,
        }
        public_requests.append(public)
        correctness.append(
            {
                "cell_id": cell.cell_id,
                "request_id": descriptor.request_id,
                **evaluator,
            }
        )
    peak_memory = _optional_positive(
        value.get("peak_gpu_memory_mib"), "peak GPU memory"
    )
    if (
        peak_memory is not None
        and total_memory is not None
        and _decimal(peak_memory, "peak GPU memory")
        > _decimal(total_memory, "total GPU memory")
    ):
        raise VLLMCompileEvidenceError("peak GPU memory exceeds total memory")
    compilation = _optional_positive(
        value.get("compilation_seconds"), "compilation timing"
    )
    if not cell.compile_enabled and compilation is not None:
        raise VLLMCompileEvidenceError("eager cell cannot report compilation timing")
    summary = {
        "cell": cell.to_dict(),
        "hardware": {
            "gpu_name": hardware["gpu_name"],
            "gpu_count": 1,
            "driver_version": driver,
            "memory_total_mib": total_memory,
        },
        "runtime": RUNTIME_PINS,
        "image_reference": IMAGE_REFERENCE,
        "initialization_started_at": value["initialization_started_at"],
        "initialization_ready_at": value["initialization_ready_at"],
        "initialization_seconds": canonical_decimal(init_seconds),
        "compilation_seconds": compilation,
        "cuda_graph_seconds": _optional_positive(
            value.get("cuda_graph_seconds"), "CUDA graph timing"
        ),
        "peak_gpu_memory_mib": peak_memory,
        "terminal_outcome": "complete",
        "field_provenance": {
            "hardware": "cuda",
            "runtime": "vllm",
            "image_reference": "derived",
            "initialization_started_at": "client_observed",
            "initialization_ready_at": "vllm",
            "initialization_seconds": "derived",
            "compilation_seconds": "vllm",
            "cuda_graph_seconds": "vllm",
            "peak_gpu_memory_mib": "cuda",
            "terminal_outcome": "derived",
        },
    }
    return summary, public_requests, correctness


def _validate_ledger(value: Any, plan: VLLMCompilePlan) -> dict[str, Any]:
    ledger = _strict_keys(
        value,
        {
            "schema_version",
            "plan_sha256",
            "revision",
            "reserved_usd",
            "remaining_usd",
            "lines",
        },
        "ledger projection",
    )
    lines = ledger["lines"]
    if (
        ledger["schema_version"] != "1"
        or ledger["plan_sha256"] != plan.content_sha256
        or ledger["revision"] != len(plan.lines)
        or not isinstance(lines, list)
        or len(lines) != len(plan.lines)
    ):
        raise VLLMCompileEvidenceError("ledger projection binding is invalid")
    for observed, expected in zip(lines, plan.lines, strict=True):
        if observed != {
            "line_id": expected.line_id,
            "reserved_usd": canonical_decimal(expected.amount_usd),
        }:
            raise VLLMCompileEvidenceError("ledger line differs from plan")
    if (
        ledger["reserved_usd"] != canonical_decimal(plan.first_pass_usd)
        or _decimal(ledger["remaining_usd"], "remaining budget")
        != HARD_CAP_USD - plan.first_pass_usd
    ):
        raise VLLMCompileEvidenceError("ledger totals do not recompute")
    return dict(ledger)


def _validate_billing(value: Any, label: str) -> dict[str, Any]:
    billing = _strict_keys(
        value, {"facts", "unavailable_reason", "unsupported_fields"}, label
    )
    unsupported = billing["unsupported_fields"]
    if unsupported != ["credits_usd", "budget_usd", "spend_limit_usd"]:
        raise VLLMCompileEvidenceError(f"{label} unsupported fields are invalid")
    if billing["facts"] is None:
        if (
            not isinstance(billing["unavailable_reason"], str)
            or not billing["unavailable_reason"]
        ):
            raise VLLMCompileEvidenceError(f"{label} missing reason is required")
        return dict(billing)
    if billing["unavailable_reason"] is not None or not isinstance(
        billing["facts"], dict
    ):
        raise VLLMCompileEvidenceError(f"{label} optional semantics are invalid")
    allowed = {
        "metered_cost",
        "billed_cost",
        "credits_usd",
        "budget_usd",
        "spend_limit_usd",
    }
    if set(billing["facts"]) != allowed:
        raise VLLMCompileEvidenceError(f"{label} facts schema is invalid")
    facts: dict[str, str | None] = {}
    for key, raw in billing["facts"].items():
        facts[key] = (
            None if raw is None else canonical_decimal(_decimal(raw, f"{label}.{key}"))
        )
    return {
        "facts": facts,
        "unavailable_reason": None,
        "unsupported_fields": unsupported,
    }


def _billing_delta(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> dict[str, Any]:
    if before["facts"] is None or after["facts"] is None:
        return {"delta_usd": None, "unavailable_reason": "not_comparable"}
    first = before["facts"].get("metered_cost")
    last = after["facts"].get("metered_cost")
    if first is None or last is None:
        return {"delta_usd": None, "unavailable_reason": "not_comparable"}
    delta = _decimal(last, "billing after") - _decimal(first, "billing before")
    if delta < 0:
        return {"delta_usd": None, "unavailable_reason": "not_comparable"}
    return {"delta_usd": canonical_decimal(delta), "unavailable_reason": None}


def _validate_teardown(value: Any, billing_after: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise VLLMCompileEvidenceError("teardown report is invalid")
    if "artifact_sha256" not in value:
        raise VLLMCompileEvidenceError("teardown report seal is missing")
    _verify_seal(value, "artifact_sha256")
    required = {
        "schema_version",
        "experiment_id",
        "complete",
        "steps",
        "billing_after",
        "billing_after_unavailable_reason",
        "billing_unsupported_fields",
        "secrets_created",
        "credentials_to_revoke",
        "post_delete_storage_billing_days_accounted",
        "provider_inventory_after",
        "inventory_status",
    }
    if set(value) - {"artifact_sha256"} != required:
        raise VLLMCompileEvidenceError("teardown report schema is not exact")
    if (
        value["schema_version"] != "1"
        or value["complete"] is not True
        or value["inventory_status"] != "complete"
        or value["billing_after"] != billing_after["facts"]
        or value["billing_after_unavailable_reason"]
        != billing_after["unavailable_reason"]
        or value["billing_unsupported_fields"] != billing_after["unsupported_fields"]
        or value["secrets_created"] != 0
        or value["credentials_to_revoke"] != []
        or value["post_delete_storage_billing_days_accounted"] != 4
    ):
        raise VLLMCompileEvidenceError("teardown is incomplete")
    steps = value["steps"]
    if (
        not isinstance(steps, list)
        or not steps
        or any(
            not isinstance(step, dict)
            or step.get("status") not in {"complete", "already_stopped", "not_required"}
            for step in steps
        )
    ):
        raise VLLMCompileEvidenceError("teardown steps are incomplete")
    expected_operations = ("stop_app", "delete_volume")
    if len(steps) != 2:
        raise VLLMCompileEvidenceError("teardown step count is invalid")
    for step, operation in zip(steps, expected_operations, strict=True):
        if set(step) != {"operation", "status"} or step["operation"] != operation:
            raise VLLMCompileEvidenceError("teardown step schema is invalid")
    inventories = value["provider_inventory_after"]
    if not isinstance(inventories, dict) or set(inventories) != {
        "apps",
        "volumes",
        "containers",
        "secrets",
    }:
        raise VLLMCompileEvidenceError("teardown inventory is invalid")
    live = {
        "running",
        "deployed",
        "ephemeral",
        "ephemeral (detached)",
        "initializing...",
        "stopping...",
    }
    for inventory in inventories.values():
        if not isinstance(inventory, dict) or set(inventory) != {
            "count",
            "status_counts",
        }:
            raise VLLMCompileEvidenceError("teardown inventory facts are invalid")
        counts = inventory["status_counts"]
        if (
            isinstance(inventory["count"], bool)
            or not isinstance(inventory["count"], int)
            or not isinstance(counts, dict)
            or any(
                isinstance(count, bool) or not isinstance(count, int) or count < 0
                for count in counts.values()
            )
            or sum(counts.values()) != inventory["count"]
            or any(counts.get(status, 0) for status in live)
        ):
            raise VLLMCompileEvidenceError("live or ambiguous teardown inventory")
    return {
        "schema_version": "1",
        "complete": True,
        "steps": steps,
        "inventory_status": "complete",
        "provider_inventory_after": inventories,
        "billing_after": billing_after,
        "secrets_created": 0,
        "credentials_to_revoke": [],
        "post_delete_storage_billing_days_accounted": 4,
    }


def _cell_rate(plan: VLLMCompilePlan, cell_index: int) -> Decimal:
    rates = dict(plan.prices.rates)
    gpu = (
        rates["l40s_gpu_second_usd"]
        if CELLS[cell_index].accelerator == "L40S"
        else rates["h100_gpu_second_usd"]
    )
    return (
        gpu
        + Decimal(4) * rates["cpu_core_second_usd"]
        + Decimal(32) * rates["memory_gib_second_usd"]
    )


def _cost_report(
    plan: VLLMCompilePlan,
    ledger: Mapping[str, Any],
    lifecycles: Sequence[Mapping[str, Any]],
    billing_before: Mapping[str, Any],
    billing_after: Mapping[str, Any],
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    total = Decimal()
    for index, lifecycle in enumerate(lifecycles):
        duration = _decimal(lifecycle["duration_seconds"], "lifecycle duration")
        rate = _cell_rate(plan, index)
        cost = duration * rate
        total += cost
        cells.append(
            {
                "cell_id": CELLS[index].cell_id,
                "observed_client_lifecycle_seconds": canonical_decimal(duration),
                "list_rate_usd_per_second": canonical_decimal(rate),
                "inferred_list_rate_cost_usd": canonical_decimal(cost),
                "basis": "client_lifecycle_seconds_times_pinned_list_rate",
            }
        )
    return {
        "schema_version": "1",
        "hard_cap_usd": canonical_decimal(HARD_CAP_USD),
        "reserved_first_pass_usd": ledger["reserved_usd"],
        "remaining_lifecycle_usd": ledger["remaining_usd"],
        "reservation_revision": ledger["revision"],
        "reservations": ledger["lines"],
        "inferred_cells": cells,
        "inferred_cell_lifecycle_total_usd": canonical_decimal(total),
        "inferred_scope": (
            "four observed cell lifecycles only; excludes image preparation, "
            "staging, and retained storage"
        ),
        "provider_billing_before": billing_before,
        "provider_billing_after": billing_after,
        "provider_account_billing": _billing_delta(billing_before, billing_after),
    }


def _break_even_pair(
    eager_summary: Mapping[str, Any],
    compiled_summary: Mapping[str, Any],
    eager_requests: Sequence[Mapping[str, Any]],
    compiled_requests: Sequence[Mapping[str, Any]],
    *,
    eager_rate: Decimal,
    compiled_rate: Decimal,
) -> dict[str, Any]:
    eager_init = _decimal(eager_summary["initialization_seconds"], "eager init")
    compiled_init = _decimal(
        compiled_summary["initialization_seconds"], "compiled init"
    )
    output_count_parity = all(
        eager["output_token_count"] == compiled["output_token_count"]
        for eager, compiled in zip(eager_requests, compiled_requests, strict=True)
    )
    identical_output_token_ids = all(
        eager["output_token_ids"] == compiled["output_token_ids"]
        for eager, compiled in zip(eager_requests, compiled_requests, strict=True)
    )
    first_divergent_request_ordinal = next(
        (
            index
            for index, (eager, compiled) in enumerate(
                zip(eager_requests, compiled_requests, strict=True), start=1
            )
            if eager["output_token_ids"] != compiled["output_token_ids"]
        ),
        None,
    )
    correctness_not_worse = all(
        not eager["correctness"] or compiled["correctness"]
        for eager, compiled in zip(eager_requests, compiled_requests, strict=True)
    )
    comparable = output_count_parity and correctness_not_worse
    eager_cumulative = eager_init
    compiled_cumulative = compiled_init
    observed: int | None = None
    list_observed: int | None = None
    prefix_savings: list[Decimal] = []
    cumulative_saving = Decimal()
    for index, (eager, compiled) in enumerate(
        zip(eager_requests, compiled_requests, strict=True), start=1
    ):
        eager_latency = _decimal(eager["latency_seconds"], "eager latency")
        compiled_latency = _decimal(compiled["latency_seconds"], "compiled latency")
        eager_cumulative += eager_latency
        compiled_cumulative += compiled_latency
        cumulative_saving += eager_latency - compiled_latency
        prefix_savings.append(cumulative_saving)
        if comparable and observed is None and compiled_cumulative <= eager_cumulative:
            observed = index
        if (
            comparable
            and list_observed is None
            and compiled_cumulative * compiled_rate <= eager_cumulative * eager_rate
        ):
            list_observed = index
    cycle_saving = prefix_savings[-1]
    overhead = compiled_init - eager_init
    extrapolated: int | None = None
    if comparable and observed is None and cycle_saving > 0:
        candidates: list[int] = []
        for prefix, saving in enumerate(prefix_savings, start=1):
            required = overhead - saving
            cycles = max(
                0,
                int(
                    (required / cycle_saving).to_integral_value(rounding=ROUND_CEILING)
                ),
            )
            candidates.append(cycles * REQUESTS_PER_CELL + prefix)
        extrapolated = min(candidates)
    return {
        "comparable": comparable,
        "paired_output_count_parity": output_count_parity,
        "identical_output_token_ids": identical_output_token_ids,
        "first_divergent_request_ordinal": first_divergent_request_ordinal,
        "compiled_correctness_not_worse": correctness_not_worse,
        "observed_requests": observed,
        "observed_lower_bound_requests": (
            REQUESTS_PER_CELL if comparable and observed is None else None
        ),
        "extrapolated_requests": (
            extrapolated if comparable and cycle_saving > 0 else None
        ),
        "initialization_delta_seconds": canonical_decimal(overhead),
        "initialization_penalty_seconds": canonical_decimal(max(overhead, Decimal())),
        "no_measured_cold_start_penalty": overhead <= 0,
        "full_cycle_request_saving_seconds": canonical_decimal(cycle_saving),
        "basis": "initialization_plus_complete_request_wall_clock",
        "list_rate_cumulative_crossing_requests": list_observed,
        "list_rate_basis": "cumulative_seconds_times_cell_list_rate",
    }


def _break_even(
    summaries: Sequence[Mapping[str, Any]],
    requests_by_cell: Sequence[Sequence[Mapping[str, Any]]],
    plan: VLLMCompilePlan,
) -> dict[str, Any]:
    pairs = [
        {
            "accelerator": "L40S",
            **_break_even_pair(
                summaries[0],
                summaries[1],
                requests_by_cell[0],
                requests_by_cell[1],
                eager_rate=_cell_rate(plan, 0),
                compiled_rate=_cell_rate(plan, 1),
            ),
        },
        {
            "accelerator": "H100",
            **_break_even_pair(
                summaries[2],
                summaries[3],
                requests_by_cell[2],
                requests_by_cell[3],
                eager_rate=_cell_rate(plan, 2),
                compiled_rate=_cell_rate(plan, 3),
            ),
        },
    ]
    return {
        "schema_version": "1",
        "pairs": pairs,
        "list_rate_cumulative_crossings": [
            {
                "accelerator": pair["accelerator"],
                "observed_requests": pair["list_rate_cumulative_crossing_requests"],
                "basis": pair["list_rate_basis"],
            }
            for pair in pairs
        ],
        "ranking_scope": [cell.cell_id for cell in CELLS],
        "incompatible_scope_limitation": (
            "MLX is excluded from rankings because runtime, quantization, and "
            "hardware are incompatible."
        ),
    }


def _json_text(value: Any) -> str:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def _jsonl_text(values: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
        for value in values
    )


def _report_html(break_even: Mapping[str, Any]) -> str:
    def crossing(value: int | None, *, comparable: bool) -> str:
        if not comparable:
            return "not comparable"
        return str(value) if value is not None else "not observed"

    rows = "".join(
        "<tr><td>"
        + html.escape(pair["accelerator"])
        + "</td><td>"
        + html.escape(
            crossing(pair["observed_requests"], comparable=pair["comparable"])
        )
        + "</td><td>"
        + html.escape(
            crossing(pair["extrapolated_requests"], comparable=pair["comparable"])
        )
        + "</td><td>"
        + html.escape(pair["initialization_delta_seconds"])
        + "</td><td>"
        + html.escape(pair["full_cycle_request_saving_seconds"])
        + "</td><td>"
        + html.escape(str(pair["no_measured_cold_start_penalty"]).lower())
        + "</td></tr>"
        for pair in break_even["pairs"]
    )
    return (
        '<!doctype html>\n<html lang="en"><meta charset="utf-8">'
        "<title>Qwen3 compilation evidence</title><body>"
        "<h1>Qwen3 8B vLLM compilation evidence</h1>"
        "<table><thead><tr><th>Accelerator</th><th>Observed crossing</th>"
        f"<th>Modeled crossing by repeated {REQUESTS_PER_CELL}-request cycles</th>"
        "<th>Compiled initialization delta (s)</th>"
        f"<th>{REQUESTS_PER_CELL}-request saving (s)</th>"
        "<th>No measured cold-start penalty</th></tr></thead><tbody>"
        + rows
        + "</tbody></table><p>MLX is outside the compatible ranking scope.</p>"
        "</body></html>\n"
    )


def _svg(break_even: Mapping[str, Any]) -> str:
    def label(pair: Mapping[str, Any]) -> str:
        if not pair["comparable"]:
            return "not comparable"
        if pair["observed_requests"] is not None:
            return f"observed {pair['observed_requests']}"
        if pair["extrapolated_requests"] is not None:
            return f"modeled {pair['extrapolated_requests']}"
        return "no repayment"

    labels = [label(pair) for pair in break_even["pairs"]]
    values = [
        (
            pair["observed_requests"] or pair["extrapolated_requests"] or 0
            if label(pair).startswith(("observed", "modeled"))
            else 0
        )
        for pair in break_even["pairs"]
    ]
    widths = [min(360, int(value) * 10) for value in values]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="520" height="150" '
        'viewBox="0 0 520 150" role="img" '
        'aria-label="Observed or modeled break-even requests">\n'
        '<rect width="520" height="150" fill="white"/>\n'
        '<text x="10" y="22" font-family="sans-serif" font-size="16">'
        "Break-even request count</text>\n"
        f'<text x="10" y="62" font-family="sans-serif">L40S</text>'
        f'<rect x="100" y="45" width="{widths[0]}" height="22" fill="#0969da"/>\n'
        f'<text x="470" y="62" text-anchor="end" font-family="sans-serif">'
        f"{html.escape(labels[0])}</text>\n"
        f'<text x="10" y="112" font-family="sans-serif">H100</text>'
        f'<rect x="100" y="95" width="{widths[1]}" height="22" fill="#1a7f37"/>\n'
        f'<text x="470" y="112" text-anchor="end" font-family="sans-serif">'
        f"{html.escape(labels[1])}</text>\n"
        "</svg>\n"
    )


def _claim_matrix() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "claims": [
            {
                "cell_id": cell.cell_id,
                "relation": "uses_workload_contract",
                "target": "qwen3-8b-vllm-compile-workload-v1",
            }
            for cell in CELLS
        ],
        "ranking_scope": [cell.cell_id for cell in CELLS],
        "limitations": [
            (
                "MLX is an incompatible scope because runtime, quantization, "
                "and hardware differ. It is not ranked."
            )
        ],
    }


def _write_bundle(directory: Path, files: Mapping[str, str]) -> None:
    for name, content in files.items():
        atomic_write_text(directory / name, content)
    checksum_lines = [
        f"{hashlib.sha256(files[name].encode('utf-8')).hexdigest()}  {name}\n"
        for name in sorted(files)
    ]
    atomic_write_text(directory / "SHA256SUMS", "".join(checksum_lines))


def build_bundle(
    output_dir: str | Path,
    *,
    execution_contract: Any,
    pricing_snapshot: Any,
    staging_receipt: Any,
    cell_records: Sequence[Any],
    lifecycle_records: Sequence[Any],
    ledger_snapshot: Any,
    billing_before: Any,
    billing_after: Any,
    teardown_report: Any,
) -> dict[str, str]:
    """Validate raw evidence, evaluate correctness, and render a public bundle."""

    destination = Path(output_dir)
    if (
        not destination.is_dir()
        or destination.is_symlink()
        or any(destination.iterdir())
    ):
        raise VLLMCompileEvidenceError("bundle directory must be empty and regular")
    raw_inputs = [
        execution_contract,
        pricing_snapshot,
        staging_receipt,
        list(cell_records),
        list(lifecycle_records),
        ledger_snapshot,
        billing_before,
        billing_after,
        teardown_report,
    ]
    for value in raw_inputs:
        _walk_safe(value)
    contract, plan = _validate_contract(execution_contract)
    pricing = _validate_pricing(pricing_snapshot, plan)
    receipt, prompts = _validate_staging(staging_receipt, plan)
    if len(cell_records) != 4 or len(lifecycle_records) != 4:
        raise VLLMCompileEvidenceError("exactly four cells and lifecycles are required")
    lifecycles = [
        _validate_lifecycle(value, index)
        for index, value in enumerate(lifecycle_records)
    ]
    summaries: list[dict[str, Any]] = []
    requests_by_cell: list[list[dict[str, Any]]] = []
    correctness: list[dict[str, Any]] = []
    for index, value in enumerate(cell_records):
        summary, requests, evaluations = _validate_cell(
            value, index, plan, receipt, prompts
        )
        summaries.append(summary)
        requests_by_cell.append(requests)
        correctness.extend(evaluations)
    ledger = _validate_ledger(ledger_snapshot, plan)
    before = _validate_billing(billing_before, "billing before")
    after = _validate_billing(billing_after, "billing after")
    teardown = _validate_teardown(teardown_report, after)
    cost = _cost_report(plan, ledger, lifecycles, before, after)
    break_even = _break_even(summaries, requests_by_cell, plan)
    requests = [item for group in requests_by_cell for item in group]
    workload_public = {
        "schema_version": "1",
        "workload_contract_sha256": _harness_hashes()[0],
        "prompts": [prompts[key] for key in sorted(prompts)],
        "request_bindings": [
            {
                **descriptor.to_dict(),
                "prompt_key": f"{descriptor.context_tier}/{descriptor.workload_id}",
                "prompt_token_ids_sha256": prompts[
                    f"{descriptor.context_tier}/{descriptor.workload_id}"
                ]["prompt_token_ids_sha256"],
            }
            for descriptor in workload_descriptors()
        ],
    }
    runtime_public = {
        "schema_version": "1",
        "runtime_pins": RUNTIME_PINS,
        "image_reference": IMAGE_REFERENCE,
        "image_digest": OFFICIAL_VLLM_IMAGE_DIGEST,
        "cells": summaries,
    }
    model_public = {
        "schema_version": "1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "file_count": EXPECTED_MODEL_FILE_COUNT,
        "total_bytes": EXPECTED_MODEL_BYTES,
        "files": receipt["inventory"],
    }
    correctness_public = {
        "schema_version": "1",
        "evaluator": "evaluate_workload",
        "model_output_executed": False,
        "records": correctness,
        "passed": sum(item["success"] for item in correctness),
        "failed": sum(not item["success"] for item in correctness),
        "cells": [
            {
                "cell_id": cell.cell_id,
                "passed": sum(
                    item["success"]
                    for item in correctness
                    if item["cell_id"] == cell.cell_id
                ),
                "failed": sum(
                    not item["success"]
                    for item in correctness
                    if item["cell_id"] == cell.cell_id
                ),
            }
            for cell in CELLS
        ],
    }
    files = {
        "README.md": README,
        "experiment-contract.json": _json_text(contract),
        "pricing-snapshot.json": _json_text(pricing),
        "model-inventory.json": _json_text(model_public),
        "runtime-image.json": _json_text(runtime_public),
        "workload-contract.json": _json_text(workload_public),
        "lifecycle-records.jsonl": _jsonl_text(lifecycles),
        "request-records.jsonl": _jsonl_text(requests),
        "correctness-report.json": _json_text(correctness_public),
        "break-even.json": _json_text(break_even),
        "cost-ledger.json": _json_text(cost),
        "teardown-report.json": _json_text(teardown),
        "claim-matrix.json": _json_text(_claim_matrix()),
        "report.html": _report_html(break_even),
        "break-even.svg": _svg(break_even),
        "evidence_bundle.py": VERIFY_SCRIPT,
    }
    _write_bundle(destination, files)
    verify_bundle(destination)
    return {
        name: _sha256_bytes((destination / name).read_bytes()) for name in BUNDLE_FILES
    }


def _read_file(path: Path) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise VLLMCompileEvidenceError(f"{path.name} must be regular and non-symlink")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise VLLMCompileEvidenceError(f"{path.name} exceeds size bound")
    return path.read_bytes()


def _load_private_sealed(
    path: Path, *, seal_field: str = "artifact_sha256", keep_seal: bool = False
) -> dict[str, Any]:
    raw = _read_file(path)
    try:
        value = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise VLLMCompileEvidenceError(f"{path.name} is invalid JSON") from exc
    if not isinstance(value, dict) or raw.decode("utf-8") != canonical_json(value):
        raise VLLMCompileEvidenceError(f"{path.name} is not canonical")
    _verify_seal(value, seal_field)
    _walk_safe(value)
    if keep_seal:
        return value
    result = dict(value)
    result.pop(seal_field)
    return result


def build_from_execution_directory(
    execution_dir: str | Path, output_dir: str | Path
) -> dict[str, str]:
    """Build the public bundle from one complete orchestrator output directory."""

    root = Path(execution_dir)
    if not root.is_dir() or root.is_symlink():
        raise VLLMCompileEvidenceError("execution directory must be regular")
    contract = _load_private_sealed(root / "evidence-contract-input.json")
    pricing = _load_private_sealed(root / "pricing-snapshot-input.json")
    billing_before = _load_private_sealed(root / "billing-before-input.json")
    ledger = _load_private_sealed(root / "ledger-projection-input.json")
    receipt = _load_private_sealed(
        root / "staging-receipt.json",
        seal_field="receipt_sha256",
        keep_seal=True,
    )
    cells = [
        _load_private_sealed(
            root / f"{name}-terminal.json",
            seal_field="cell_sha256",
            keep_seal=True,
        )
        for name in _FUNCTIONS
    ]
    lifecycles = [
        _load_private_sealed(root / f"{name}-lifecycle.json", keep_seal=True)
        for name in _FUNCTIONS
    ]
    teardown = _load_private_sealed(root / "teardown-report.json", keep_seal=True)
    billing_after = {
        "facts": teardown["billing_after"],
        "unavailable_reason": teardown["billing_after_unavailable_reason"],
        "unsupported_fields": teardown["billing_unsupported_fields"],
    }
    return build_bundle(
        output_dir,
        execution_contract=contract,
        pricing_snapshot=pricing,
        staging_receipt=receipt,
        cell_records=cells,
        lifecycle_records=lifecycles,
        ledger_snapshot=ledger,
        billing_before=billing_before,
        billing_after=billing_after,
        teardown_report=teardown,
    )


def _load_json_file(path: Path) -> Any:
    raw = _read_file(path)
    try:
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise VLLMCompileEvidenceError(f"{path.name} is invalid JSON") from exc
    if raw.decode("ascii") != _json_text(value):
        raise VLLMCompileEvidenceError(f"{path.name} is not canonical")
    _walk_safe(value)
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    raw = _read_file(path)
    try:
        text = raw.decode("ascii")
        values = [
            json.loads(
                line,
                parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
            )
            for line in text.splitlines()
        ]
    except (UnicodeDecodeError, ValueError) as exc:
        raise VLLMCompileEvidenceError(f"{path.name} is invalid JSONL") from exc
    if text != _jsonl_text(values):
        raise VLLMCompileEvidenceError(f"{path.name} is not canonical JSONL")
    for value in values:
        if not isinstance(value, dict):
            raise VLLMCompileEvidenceError(f"{path.name} rows must be objects")
        _walk_safe(value)
    return values


def verify_bundle(directory: str | Path) -> dict[str, Any]:
    """Strictly verify a public bundle and recompute every substantive result."""

    root = Path(directory)
    if not root.is_dir() or root.is_symlink():
        raise VLLMCompileEvidenceError("bundle must be a regular directory")
    names = {path.name for path in root.iterdir()}
    if names != set(BUNDLE_FILES):
        raise VLLMCompileEvidenceError("bundle top-level file allowlist differs")
    sums_raw = _read_file(root / "SHA256SUMS")
    try:
        sums_text = sums_raw.decode("ascii")
    except UnicodeDecodeError as exc:
        raise VLLMCompileEvidenceError("SHA256SUMS must be ASCII") from exc
    expected_names = sorted(set(BUNDLE_FILES) - {"SHA256SUMS"})
    expected_lines: list[str] = []
    for name in expected_names:
        digest = hashlib.sha256(_read_file(root / name)).hexdigest()
        expected_lines.append(f"{digest}  {name}\n")
    if sums_text != "".join(expected_lines):
        raise VLLMCompileEvidenceError("checksum coverage or digest is invalid")

    loaded = {name: _load_json_file(root / name) for name in JSON_FILES}
    lifecycles = _load_jsonl(root / "lifecycle-records.jsonl")
    requests = _load_jsonl(root / "request-records.jsonl")
    if _read_file(root / "README.md").decode("ascii") != README:
        raise VLLMCompileEvidenceError("README prose differs from allowlisted text")
    if _read_file(root / "evidence_bundle.py").decode("ascii") != VERIFY_SCRIPT:
        raise VLLMCompileEvidenceError("verification CLI differs from allowlisted text")

    contract, plan = _validate_contract(loaded["experiment-contract.json"])
    expected_pricing = _validate_pricing(
        {
            key: loaded["pricing-snapshot.json"][key]
            for key in (
                "schema_version",
                "retrieved_date",
                "pricing_page",
                "volumes_page",
                "rates_response_sha256",
            )
        },
        plan,
    )
    if loaded["pricing-snapshot.json"] != expected_pricing:
        raise VLLMCompileEvidenceError("public pricing snapshot differs from plan")
    model = loaded["model-inventory.json"]
    if model != {
        "schema_version": "1",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "file_count": EXPECTED_MODEL_FILE_COUNT,
        "total_bytes": EXPECTED_MODEL_BYTES,
        "files": _conversion_inventory(),
    }:
        raise VLLMCompileEvidenceError("public model inventory is invalid")
    workload = loaded["workload-contract.json"]
    _strict_keys(
        workload,
        {
            "schema_version",
            "workload_contract_sha256",
            "prompts",
            "request_bindings",
        },
        "public workload contract",
    )
    if (
        not isinstance(workload, dict)
        or workload.get("schema_version") != "1"
        or workload.get("workload_contract_sha256") != _harness_hashes()[0]
        or not isinstance(workload.get("prompts"), list)
        or len(workload["prompts"]) != 6
    ):
        raise VLLMCompileEvidenceError("public workload contract is invalid")
    prompt_map = {item["key"]: item for item in workload["prompts"]}
    if len(prompt_map) != 6:
        raise VLLMCompileEvidenceError("public prompt identities are duplicated")
    bindings = workload.get("request_bindings")
    descriptors = workload_descriptors()
    if not isinstance(bindings, list) or len(bindings) != 12:
        raise VLLMCompileEvidenceError("public request bindings are invalid")
    for binding, descriptor in zip(bindings, descriptors, strict=True):
        key = f"{descriptor.context_tier}/{descriptor.workload_id}"
        _strict_keys(
            prompt_map[key],
            {
                "key",
                "prompt_sha256",
                "decoded_prompt_sha256",
                "prompt_token_ids",
                "prompt_token_ids_sha256",
                "input_token_count",
            },
            "public prompt",
        )
        expected = {
            **descriptor.to_dict(),
            "prompt_key": key,
            "prompt_token_ids_sha256": prompt_map[key]["prompt_token_ids_sha256"],
        }
        if binding != expected:
            raise VLLMCompileEvidenceError("public request binding order differs")
        tokens = prompt_map[key]["prompt_token_ids"]
        if (
            prompt_map[key]["input_token_count"] != len(tokens)
            or prompt_map[key]["prompt_token_ids_sha256"] != _sha256_json(tokens)
            or prompt_map[key]["prompt_sha256"] != descriptor.prompt_sha256
            or not _SHA256.fullmatch(prompt_map[key]["decoded_prompt_sha256"])
        ):
            raise VLLMCompileEvidenceError("public prompt tokens do not verify")
    runtime = loaded["runtime-image.json"]
    _strict_keys(
        runtime,
        {
            "schema_version",
            "runtime_pins",
            "image_reference",
            "image_digest",
            "cells",
        },
        "public runtime image",
    )
    if (
        runtime.get("runtime_pins") != RUNTIME_PINS
        or runtime.get("image_reference") != IMAGE_REFERENCE
        or runtime.get("image_digest") != OFFICIAL_VLLM_IMAGE_DIGEST
        or not isinstance(runtime.get("cells"), list)
        or len(runtime["cells"]) != 4
    ):
        raise VLLMCompileEvidenceError("public runtime image evidence is invalid")
    for index, summary in enumerate(runtime["cells"]):
        _strict_keys(
            summary,
            {
                "cell",
                "hardware",
                "runtime",
                "image_reference",
                "initialization_started_at",
                "initialization_ready_at",
                "initialization_seconds",
                "compilation_seconds",
                "cuda_graph_seconds",
                "peak_gpu_memory_mib",
                "terminal_outcome",
                "field_provenance",
            },
            "public runtime cell",
        )
        public_hardware = _strict_keys(
            summary.get("hardware"),
            {
                "gpu_name",
                "gpu_count",
                "driver_version",
                "memory_total_mib",
            },
            "public hardware",
        )
        _optional_positive(
            public_hardware["memory_total_mib"], "public total GPU memory"
        )
        if public_hardware["driver_version"] is not None and (
            not isinstance(public_hardware["driver_version"], str)
            or not public_hardware["driver_version"]
        ):
            raise VLLMCompileEvidenceError("public CUDA driver is invalid")
        if (
            summary.get("cell") != CELLS[index].to_dict()
            or summary.get("runtime") != RUNTIME_PINS
            or summary.get("image_reference") != IMAGE_REFERENCE
            or public_hardware.get("gpu_count") != 1
            or summary.get("terminal_outcome") != "complete"
            or summary.get("field_provenance")
            != {
                "hardware": "cuda",
                "runtime": "vllm",
                "image_reference": "derived",
                "initialization_started_at": "client_observed",
                "initialization_ready_at": "vllm",
                "initialization_seconds": "derived",
                "compilation_seconds": "vllm",
                "cuda_graph_seconds": "vllm",
                "peak_gpu_memory_mib": "cuda",
                "terminal_outcome": "derived",
            }
            or _duration(
                summary["initialization_started_at"],
                summary["initialization_ready_at"],
                "public initialization",
            )
            != _decimal(summary["initialization_seconds"], "public initialization")
        ):
            raise VLLMCompileEvidenceError("public cell runtime identity is invalid")
        try:
            validate_hardware_identity(
                CELLS[index],
                HardwareIdentity(
                    public_hardware["gpu_name"], public_hardware["gpu_count"]
                ),
            )
        except (KeyError, ValueError) as exc:
            raise VLLMCompileEvidenceError(
                "public cell hardware identity is invalid"
            ) from exc
        for optional in (
            "compilation_seconds",
            "cuda_graph_seconds",
            "peak_gpu_memory_mib",
        ):
            _optional_positive(summary[optional], optional)
        if (
            not CELLS[index].compile_enabled
            and summary["compilation_seconds"] is not None
        ):
            raise VLLMCompileEvidenceError("public eager cell has compilation timing")
    if len(lifecycles) != 4 or len(requests) != 48:
        raise VLLMCompileEvidenceError("public cell or request count is invalid")
    for index, lifecycle in enumerate(lifecycles):
        _strict_keys(
            lifecycle,
            {
                "cell_id",
                "function",
                "started_at",
                "ended_at",
                "duration_seconds",
                "first_event_received_at",
                "invocation_to_first_event_seconds",
                "events",
            },
            "public lifecycle",
        )
        if (
            lifecycle["cell_id"] != CELLS[index].cell_id
            or lifecycle["function"]
            != (
                "l40s_eager",
                "l40s_compiled",
                "h100_eager",
                "h100_compiled",
            )[index]
            or not isinstance(lifecycle["events"], list)
            or not lifecycle["events"]
            or _duration(
                lifecycle["started_at"],
                lifecycle["ended_at"],
                "public lifecycle",
            )
            != _decimal(lifecycle["duration_seconds"], "public lifecycle")
            or _duration(
                lifecycle["started_at"],
                lifecycle["first_event_received_at"],
                "public invocation to first event",
            )
            != _decimal(
                lifecycle["invocation_to_first_event_seconds"],
                "public invocation to first event",
            )
            or lifecycle["first_event_received_at"]
            != lifecycle["events"][0]["received_at"]
        ):
            raise VLLMCompileEvidenceError("public lifecycle does not verify")
        for event in lifecycle["events"]:
            _strict_keys(event, {"received_at", "event", "provenance"}, "public event")
            _timestamp(event["received_at"], "public event")
            if (
                not isinstance(event["event"], str)
                or not event["event"]
                or event["provenance"] not in PROVENANCE
            ):
                raise VLLMCompileEvidenceError("public event is invalid")
        if (
            lifecycle["events"][0]["event"] != "container_started"
            or lifecycle["events"][-1]["event"] != "cell_terminal"
        ):
            raise VLLMCompileEvidenceError("public event boundaries differ")
    requests_by_cell: list[list[dict[str, Any]]] = []
    correctness_rows: list[dict[str, Any]] = []
    for cell_index, cell in enumerate(CELLS):
        group = requests[cell_index * 12 : (cell_index + 1) * 12]
        requests_by_cell.append(group)
        for request, descriptor in zip(group, descriptors, strict=True):
            _strict_keys(
                request,
                {
                    "cell_id",
                    "ordinal",
                    "request_id",
                    "workload_id",
                    "context_tier",
                    "repetition",
                    "input_token_count",
                    "input_token_ids_sha256",
                    "output_token_count",
                    "output_token_ids",
                    "decoded_output",
                    "finish_reason",
                    "started_at",
                    "ended_at",
                    "latency_seconds",
                    "ttft_seconds",
                    "output_tokens_per_second",
                    "output_rate_basis",
                    "terminal",
                    "provenance",
                    "correctness",
                    "evaluator",
                    "field_provenance",
                },
                "public request",
            )
            key = f"{descriptor.context_tier}/{descriptor.workload_id}"
            output_ids = request.get("output_token_ids")
            latency = _decimal(
                request.get("latency_seconds"), "public latency", positive=True
            )
            if (
                request.get("cell_id") != cell.cell_id
                or request.get("request_id") != descriptor.request_id
                or request.get("input_token_count")
                != prompt_map[key]["input_token_count"]
                or request.get("input_token_ids_sha256")
                != prompt_map[key]["prompt_token_ids_sha256"]
                or not isinstance(output_ids, list)
                or not output_ids
                or len(output_ids) > 96
                or any(
                    isinstance(token, bool) or not isinstance(token, int) or token < 0
                    for token in output_ids
                )
                or request.get("output_token_count") != len(output_ids)
                or request.get("finish_reason") not in _FINISH_REASONS
                or request.get("terminal") is not True
                or request.get("provenance") not in PROVENANCE
                or request.get("field_provenance") != _PUBLIC_REQUEST_FIELD_PROVENANCE
                or not isinstance(request.get("decoded_output"), str)
                or len(request["decoded_output"].encode("utf-8")) > MAX_OUTPUT_BYTES
                or request.get("output_rate_basis")
                != "output_tokens_per_complete_response_second"
                or _duration(
                    request.get("started_at"),
                    request.get("ended_at"),
                    "public request",
                )
                - latency
                > Decimal("1")
                or latency
                - _duration(
                    request.get("started_at"),
                    request.get("ended_at"),
                    "public request",
                )
                > Decimal("1")
                or _decimal(
                    request.get("output_tokens_per_second"),
                    "public output rate",
                    positive=True,
                )
                != Decimal(len(output_ids)) / latency
            ):
                raise VLLMCompileEvidenceError("public request record is invalid")
            public_ttft = _optional_positive(request.get("ttft_seconds"), "public TTFT")
            if (
                public_ttft is not None
                and _decimal(public_ttft, "public TTFT") > latency
            ):
                raise VLLMCompileEvidenceError("public TTFT exceeds complete latency")
            outcome = evaluate_workload(
                workload_by_id(descriptor.workload_id), request["decoded_output"]
            )
            evaluator = _strict_keys(
                request.get("evaluator"),
                {
                    "success",
                    "quality_score",
                    "quality_metric",
                    "notes",
                    "evaluator",
                    "executed_model_output",
                },
                "public evaluator",
            )
            if (
                request.get("correctness") is not outcome.success
                or evaluator.get("success") is not outcome.success
                or evaluator.get("quality_score") != outcome.quality_score
                or evaluator.get("quality_metric") != outcome.quality_metric
                or evaluator.get("notes") != outcome.notes
                or evaluator.get("evaluator") != "evaluate_workload"
                or evaluator.get("executed_model_output") is not False
            ):
                raise VLLMCompileEvidenceError("correctness recomputation differs")
            correctness_rows.append(
                {
                    "cell_id": cell.cell_id,
                    "request_id": descriptor.request_id,
                    **evaluator,
                }
            )
    correctness = loaded["correctness-report.json"]
    _strict_keys(
        correctness,
        {
            "schema_version",
            "evaluator",
            "model_output_executed",
            "records",
            "passed",
            "failed",
            "cells",
        },
        "public correctness report",
    )
    if (
        correctness.get("records") != correctness_rows
        or correctness.get("passed")
        != sum(item["success"] for item in correctness_rows)
        or correctness.get("failed")
        != sum(not item["success"] for item in correctness_rows)
        or correctness.get("model_output_executed") is not False
        or correctness.get("evaluator") != "evaluate_workload"
        or correctness.get("cells")
        != [
            {
                "cell_id": cell.cell_id,
                "passed": sum(
                    item["success"]
                    for item in correctness_rows
                    if item["cell_id"] == cell.cell_id
                ),
                "failed": sum(
                    not item["success"]
                    for item in correctness_rows
                    if item["cell_id"] == cell.cell_id
                ),
            }
            for cell in CELLS
        ]
    ):
        raise VLLMCompileEvidenceError("correctness report differs")
    expected_break_even = _break_even(runtime["cells"], requests_by_cell, plan)
    if loaded["break-even.json"] != expected_break_even:
        raise VLLMCompileEvidenceError("break-even recomputation differs")
    ledger = loaded["cost-ledger.json"]
    _strict_keys(
        ledger,
        {
            "schema_version",
            "hard_cap_usd",
            "reserved_first_pass_usd",
            "remaining_lifecycle_usd",
            "reservation_revision",
            "reservations",
            "inferred_cells",
            "inferred_cell_lifecycle_total_usd",
            "inferred_scope",
            "provider_billing_before",
            "provider_billing_after",
            "provider_account_billing",
        },
        "public cost ledger",
    )
    before = _validate_billing(ledger["provider_billing_before"], "billing before")
    after = _validate_billing(ledger["provider_billing_after"], "billing after")
    expected_cells = _cost_report(
        plan,
        {
            "reserved_usd": canonical_decimal(plan.first_pass_usd),
            "remaining_usd": canonical_decimal(HARD_CAP_USD - plan.first_pass_usd),
            "revision": len(plan.lines),
            "lines": [
                {
                    "line_id": line.line_id,
                    "reserved_usd": canonical_decimal(line.amount_usd),
                }
                for line in plan.lines
            ],
        },
        lifecycles,
        before,
        after,
    )
    for field in (
        "schema_version",
        "hard_cap_usd",
        "reserved_first_pass_usd",
        "remaining_lifecycle_usd",
        "reservation_revision",
        "reservations",
        "inferred_cells",
        "inferred_cell_lifecycle_total_usd",
        "inferred_scope",
        "provider_billing_before",
        "provider_billing_after",
        "provider_account_billing",
    ):
        if ledger.get(field) != expected_cells[field]:
            raise VLLMCompileEvidenceError("inferred cost recomputation differs")
    teardown = loaded["teardown-report.json"]
    _strict_keys(
        teardown,
        {
            "schema_version",
            "complete",
            "steps",
            "inventory_status",
            "provider_inventory_after",
            "billing_after",
            "secrets_created",
            "credentials_to_revoke",
            "post_delete_storage_billing_days_accounted",
        },
        "public teardown",
    )
    if (
        teardown.get("complete") is not True
        or teardown.get("inventory_status") != "complete"
    ):
        raise VLLMCompileEvidenceError("public teardown is incomplete")
    synthetic_raw_teardown = {
        "schema_version": "1",
        "experiment_id": contract["experiment_id"],
        "complete": teardown["complete"],
        "steps": teardown["steps"],
        "billing_after": teardown["billing_after"]["facts"],
        "billing_after_unavailable_reason": teardown["billing_after"][
            "unavailable_reason"
        ],
        "billing_unsupported_fields": teardown["billing_after"]["unsupported_fields"],
        "secrets_created": teardown["secrets_created"],
        "credentials_to_revoke": teardown["credentials_to_revoke"],
        "post_delete_storage_billing_days_accounted": teardown[
            "post_delete_storage_billing_days_accounted"
        ],
        "provider_inventory_after": teardown["provider_inventory_after"],
        "inventory_status": teardown["inventory_status"],
    }
    synthetic_raw_teardown["artifact_sha256"] = _sha256_json(synthetic_raw_teardown)
    _validate_teardown(synthetic_raw_teardown, after)
    claim = loaded["claim-matrix.json"]
    if (
        claim != _claim_matrix()
        or any(
            item.get("relation") != "uses_workload_contract" for item in claim["claims"]
        )
        or any("mlx" in item.lower() for item in claim["ranking_scope"])
    ):
        raise VLLMCompileEvidenceError("claim matrix is not conservative")
    if _read_file(root / "report.html").decode("ascii") != _report_html(
        expected_break_even
    ):
        raise VLLMCompileEvidenceError("HTML report differs")
    if _read_file(root / "break-even.svg").decode("ascii") != _svg(expected_break_even):
        raise VLLMCompileEvidenceError("SVG report differs")
    for value in loaded.values():
        _walk_safe(value)
    _walk_safe(_read_file(root / "README.md").decode("ascii"))
    _walk_safe(_read_file(root / "report.html").decode("ascii"))
    _walk_safe(_read_file(root / "break-even.svg").decode("ascii"))
    _walk_safe(_read_file(root / "evidence_bundle.py").decode("ascii"))
    return {
        "schema_version": "1",
        "plan_sha256": plan.content_sha256,
        "files_verified": len(BUNDLE_FILES),
        "cells_verified": 4,
        "requests_verified": 48,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qwen3-vllm-evidence-verify")
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args(argv)
    try:
        verify_bundle(args.bundle)
    except VLLMCompileEvidenceError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
