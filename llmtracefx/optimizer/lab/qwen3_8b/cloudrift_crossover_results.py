"""Build and verify completed vLLM crossover result evidence.

The builder consumes only local, sealed execution artifacts.  It deliberately
contains no provider, network, Docker, or GPU operations.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import random
import re
import sys
from collections.abc import Mapping, Sequence
from decimal import Decimal, InvalidOperation
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
from ...workloads.catalog import workload_by_id
from ...workloads.evaluators import evaluate_workload
from . import vllm_compile as core
from .cloudrift_crossover import (
    AUTHORIZATION_SIGNATURE_NAMESPACE,
    AUTHORIZATION_SIGNER_IDENTITY,
    MAX_BASELINE_GPU_MEMORY_MIB,
    MAX_IDLE_GPU_TEMPERATURE_C,
    MAX_IDLE_GPU_UTILIZATION_PERCENT,
    CrossoverOrchestratorError,
    ExecutionAuthorization,
)

RESULT_SCHEMA_VERSION = "1"
BOOTSTRAP_RESAMPLES = core.BOOTSTRAP_RESAMPLES
CONTROLLED_SIGN_SYMMETRY_ALPHA = core.CONTROLLED_SIGN_SYMMETRY_ALPHA
SUPPORTED_CROSSING_BASIS = (
    "simultaneous_upper_band_sustained_crossing_observed_and_"
    "terminal_effect_sign_symmetry_p_value_at_most_0.05"
)
MAX_OUTPUT_BYTES = 8_388_608
MAX_BUNDLE_FILE_BYTES = 33_554_432
BUNDLE_FILES = (
    "SHA256SUMS",
    "analysis.json",
    "budget-teardown.json",
    "claim-matrix.json",
    "correctness.json",
    "crossover.svg",
    "evidence_bundle.py",
    "lifecycle-pairs.json",
    "protocol.json",
    "provenance-null-matrix.json",
    "report.html",
    "request-records.jsonl",
)
HASHED_FILES = tuple(name for name in BUNDLE_FILES if name != "SHA256SUMS")
JSON_FILES = (
    "analysis.json",
    "budget-teardown.json",
    "claim-matrix.json",
    "correctness.json",
    "lifecycle-pairs.json",
    "protocol.json",
    "provenance-null-matrix.json",
)
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_SAFE_PROCESS_NAME = re.compile(r"^[A-Za-z0-9._:+-]{1,128}$")
_CHECKSUM = re.compile(r"^([0-9a-f]{64})  ([A-Za-z0-9][A-Za-z0-9._-]*)$")
_PRIVATE_PATTERNS = (
    (re.compile(r"/Users/|/home/|[A-Za-z]:\\Users\\"), "private path"),
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "email address"),
    (
        re.compile(r"\b(?!127\.0\.0\.1\b)(?:\d{1,3}\.){3}\d{1,3}\b"),
        "IP address",
    ),
    (re.compile(r"\bGPU-[0-9a-f-]{16,}\b", re.I), "GPU UUID"),
    (
        re.compile(r"\bgpu_(?:uuid|identity)(?:_sha256|_commitment)?\b", re.I),
        "GPU identity derivative",
    ),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private key"),
    (
        re.compile(r"\b(?:gh[pousr]_|github_pat_|sk-|hf[_-]|AKIA)[A-Za-z0-9_-]{8,}\b"),
        "credential-shaped value",
    ),
    (
        re.compile(r'"(?:host(?:name)?|user(?:name)?|port|experiment_nonce)"\s*:'),
        "private connection field",
    ),
)


class CrossoverResultsError(ValueError):
    """Raised when completed evidence is unsafe, incomplete, or inconsistent."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_uri(data: bytes) -> str:
    return "sha256:" + _digest(data)


def _sha256_json(value: Any) -> str:
    return _sha256_uri(core.canonical_json(value).encode("utf-8"))


def _json_text(value: Any) -> str:
    try:
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
    except (TypeError, ValueError, OverflowError) as exc:
        raise CrossoverResultsError(
            f"non-finite or invalid result value: {exc}"
        ) from exc


def _jsonl_text(values: Sequence[Mapping[str, Any]]) -> str:
    try:
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
    except (TypeError, ValueError, OverflowError) as exc:
        raise CrossoverResultsError(f"invalid JSONL result value: {exc}") from exc


def _safe_json(path: Path, *, require_canonical: bool = False) -> dict[str, Any]:
    try:
        text = read_bounded_regular_text(path, MAX_EVIDENCE_ARTIFACT_BYTES)
        value = json.loads(text, parse_constant=reject_non_finite_json_constant)
    except (OSError, ArtifactReadError, ValueError, RecursionError) as exc:
        raise CrossoverResultsError(f"{path.name} is not safe JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise CrossoverResultsError(f"{path.name} must contain an object")
    if require_canonical and text != _json_text(value):
        raise CrossoverResultsError(f"{path.name} is not canonical JSON")
    return value


def _verify_seal(value: Mapping[str, Any], field: str) -> None:
    body = dict(value)
    observed = body.pop(field, None)
    if not isinstance(observed, str) or observed != _sha256_json(body):
        raise CrossoverResultsError(f"{field} does not verify")


def _positive_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise CrossoverResultsError(f"{field} must be finite and positive")
    return float(value)


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CrossoverResultsError(f"{field} must be a positive integer")
    return value


def _typed_seconds(
    value: Any,
    field: str,
    *,
    expected_clock_domain: str,
    expected_provenance: str | None = None,
) -> float:
    if not isinstance(value, dict) or set(value) != {
        "value",
        "unit",
        "clock_domain",
        "provenance",
        "observability_state",
        "null_reason",
    }:
        raise CrossoverResultsError(f"{field} is not a typed measurement")
    if (
        value["unit"] != "seconds"
        or value["observability_state"] != "observed"
        or value["null_reason"] is not None
        or value["clock_domain"] != expected_clock_domain
        or not isinstance(value["provenance"], str)
        or not value["provenance"]
    ):
        raise CrossoverResultsError(f"{field} must be an observed seconds measurement")
    if expected_provenance is not None and value["provenance"] != expected_provenance:
        raise CrossoverResultsError(f"{field} provenance differs")
    return _positive_number(value["value"], f"{field}.value")


def _validate_typed_measurement(
    value: Any,
    field: str,
    *,
    unit: str,
    clock_domain: str,
    provenance: str,
    null_reasons: Mapping[str, set[str]] | None = None,
) -> bool:
    expected_keys = {
        "value",
        "unit",
        "clock_domain",
        "provenance",
        "observability_state",
        "null_reason",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("unit") != unit
        or value.get("clock_domain") != clock_domain
        or value.get("provenance") != provenance
    ):
        raise CrossoverResultsError(f"{field} typed measurement schema differs")
    state = value["observability_state"]
    if state == "observed":
        if value["null_reason"] is not None:
            raise CrossoverResultsError(f"{field} observed value has a null reason")
        _positive_number(value["value"], f"{field}.value")
        return True
    if null_reasons is None or state not in null_reasons:
        raise CrossoverResultsError(f"{field} observability state is invalid")
    if value["value"] is not None or value["null_reason"] not in null_reasons[state]:
        raise CrossoverResultsError(f"{field} null provenance is incomplete")
    return False


def _request_indices(index: int) -> tuple[int, int]:
    return ((index - 1) // 12 + 1, (index - 1) % 12 + 1)


def _validate_process_tree(value: Any, cell_id: str) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value)
        != {
            "nodes",
            "clock_domain",
            "provenance",
            "observability_state",
            "null_reason",
        }
        or value.get("clock_domain") != "runner_process_snapshot"
        or value.get("provenance") != "linux_procfs_stat"
    ):
        raise CrossoverResultsError(f"{cell_id} process tree schema differs")
    nodes = value["nodes"]
    if value["observability_state"] == "unobservable":
        if nodes is not None or value["null_reason"] != "linux_procfs_unavailable":
            raise CrossoverResultsError(f"{cell_id} process tree null state differs")
        return dict(value)
    if (
        value["observability_state"] != "observed"
        or value["null_reason"] is not None
        or not isinstance(nodes, list)
        or not nodes
        or len(nodes) > 4096
    ):
        raise CrossoverResultsError(f"{cell_id} process tree state differs")
    for index, node in enumerate(nodes):
        if (
            not isinstance(node, dict)
            or set(node) != {"node_index", "parent_node_index", "process_name"}
            or node["node_index"] != index
            or not isinstance(node["process_name"], str)
            or not _SAFE_PROCESS_NAME.fullmatch(node["process_name"])
        ):
            raise CrossoverResultsError(f"{cell_id} process tree node differs")
        parent = node["parent_node_index"]
        if (index == 0 and parent is not None) or (
            index > 0
            and (
                isinstance(parent, bool)
                or not isinstance(parent, int)
                or not 0 <= parent < index
            )
        ):
            raise CrossoverResultsError(f"{cell_id} process tree topology differs")
    return {**value, "nodes": [dict(node) for node in nodes]}


def _validate_request(
    request: Any,
    *,
    cell: core.ScheduleCell,
    descriptor: core.WorkloadDescriptor,
    index: int,
) -> dict[str, Any]:
    if not isinstance(request, dict):
        raise CrossoverResultsError(f"{cell.cell_id} request {index} is not an object")
    descriptor_data = descriptor.to_dict()
    expected_keys = set(descriptor_data) | {
        "cycle_index",
        "base_ordinal",
        "request_sequence_index",
        "cell_id",
        "pair_id",
        "lane",
        "mode",
        "input_token_count",
        "input_token_ids_sha256",
        "output_token_count",
        "output_token_ids",
        "output_token_ids_sha256",
        "finish_reason",
        "timing",
        "metrics",
        "terminal",
    }
    if cell.lane == "natural":
        expected_keys.add("decoded_output")
    if set(request) != expected_keys:
        raise CrossoverResultsError(f"{cell.cell_id} request {index} keys differ")
    for key, expected in descriptor_data.items():
        if request.get(key) != expected:
            raise CrossoverResultsError(
                f"{cell.cell_id} request {index} descriptor differs at {key}"
            )
    cycle, base = _request_indices(index)
    fixed = {
        "cycle_index": cycle,
        "base_ordinal": base,
        "request_sequence_index": index,
        "cell_id": cell.cell_id,
        "pair_id": cell.pair_id,
        "lane": cell.lane,
        "mode": cell.mode,
        "terminal": True,
    }
    if any(request.get(key) != expected for key, expected in fixed.items()):
        raise CrossoverResultsError(f"{cell.cell_id} request {index} binding differs")
    _positive_int(request.get("input_token_count"), "input_token_count")
    if not isinstance(
        request.get("input_token_ids_sha256"), str
    ) or not _SHA256.fullmatch(request["input_token_ids_sha256"]):
        raise CrossoverResultsError("input token identity is invalid")
    output_ids = request.get("output_token_ids")
    if not isinstance(output_ids, list) or any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in output_ids
    ):
        raise CrossoverResultsError("output token IDs are invalid")
    if request.get("output_token_count") != len(output_ids):
        raise CrossoverResultsError("output token count differs from token IDs")
    if request.get("output_token_ids_sha256") != core.token_ids_sha256(output_ids):
        raise CrossoverResultsError("output token identity seal differs")
    if cell.lane == "controlled":
        if len(output_ids) != 96 or request.get("finish_reason") != "length":
            raise CrossoverResultsError("controlled request terminal shape differs")
        if "decoded_output" in request:
            raise CrossoverResultsError(
                "controlled request unexpectedly contains output text"
            )
    else:
        decoded = request.get("decoded_output")
        if (
            not isinstance(decoded, str)
            or len(decoded.encode("utf-8")) > MAX_OUTPUT_BYTES
            or request.get("finish_reason") not in {"stop", "length"}
        ):
            raise CrossoverResultsError("natural request terminal shape differs")
    timing = request.get("timing")
    if not isinstance(timing, dict) or set(timing) != {
        "latency_seconds",
        "cumulative_from_initialization_seconds",
        "latency_perf_counter_ns",
        "cumulative_from_initialization_perf_counter_ns",
        "output_token_rate_tokens_per_second",
    }:
        raise CrossoverResultsError("request timing must be an object")
    latency = _typed_seconds(
        timing.get("latency_seconds"),
        "request latency",
        expected_clock_domain="same_process_perf_counter",
        expected_provenance="measured_perf_counter_ns",
    )
    cumulative = _typed_seconds(
        timing.get("cumulative_from_initialization_seconds"),
        "request cumulative timing",
        expected_clock_domain="same_process_perf_counter",
        expected_provenance="measured_perf_counter_ns",
    )
    latency_ns = _positive_int(timing.get("latency_perf_counter_ns"), "latency ns")
    cumulative_ns = _positive_int(
        timing.get("cumulative_from_initialization_perf_counter_ns"),
        "cumulative ns",
    )
    if not math.isclose(latency, latency_ns / 1_000_000_000, abs_tol=1e-12):
        raise CrossoverResultsError("latency typed and integer timings differ")
    if not math.isclose(cumulative, cumulative_ns / 1_000_000_000, abs_tol=1e-12):
        raise CrossoverResultsError("cumulative typed and integer timings differ")
    if cumulative_ns < latency_ns:
        raise CrossoverResultsError("cumulative timing precedes request latency")
    rate = timing["output_token_rate_tokens_per_second"]
    _validate_typed_measurement(
        rate,
        "output token rate",
        unit="tokens_per_second",
        clock_domain="same_process_perf_counter",
        provenance="derived_exact_token_count_over_perf_counter_latency",
    )
    observed_rate = _positive_number(rate.get("value"), "output token rate")
    if not math.isclose(observed_rate, len(output_ids) / latency, rel_tol=1e-12):
        raise CrossoverResultsError("output token rate differs from exact token count")
    metrics = request.get("metrics")
    metric_null_reasons = {
        "queue_seconds": "request_state_stats_has_no_queue_duration_field",
        "prefill_seconds": "request_state_stats_has_no_prefill_duration_field",
        "inference_seconds": "request_state_stats_has_no_inference_duration_field",
        "decode_seconds": "request_state_stats_has_no_decode_duration_field",
        "mean_time_per_output_token_seconds": (
            "request_state_stats_has_no_mean_output_token_duration_field"
        ),
        "e2e_seconds": "request_state_stats_has_no_e2e_duration_field",
    }
    if not isinstance(metrics, dict) or set(metrics) != {
        "ttft_seconds",
        *metric_null_reasons,
    }:
        raise CrossoverResultsError("request metric provenance is missing")
    metric_provenance = "version_pinned_vllm_0_28_request_state_stats"
    _validate_typed_measurement(
        metrics["ttft_seconds"],
        "metrics.ttft_seconds",
        unit="seconds",
        clock_domain="request_output_metrics",
        provenance=metric_provenance,
        null_reasons={
            "unobservable": {"request_state_stats_first_token_latency_unavailable"}
        },
    )
    for name, null_reason in metric_null_reasons.items():
        if _validate_typed_measurement(
            metrics[name],
            f"metrics.{name}",
            unit="seconds",
            clock_domain="request_output_metrics",
            provenance=metric_provenance,
            null_reasons={"unobservable": {null_reason}},
        ):
            raise CrossoverResultsError(f"metrics.{name} must remain null")
    public = {
        **descriptor_data,
        "cycle_index": cycle,
        "base_ordinal": base,
        "request_sequence_index": index,
        "cell_id": cell.cell_id,
        "pair_id": cell.pair_id,
        "lane": cell.lane,
        "mode": cell.mode,
        "input_token_count": request["input_token_count"],
        "input_token_ids_sha256": request["input_token_ids_sha256"],
        "output_token_count": len(output_ids),
        "output_token_ids": output_ids,
        "output_token_ids_sha256": request["output_token_ids_sha256"],
        "finish_reason": request["finish_reason"],
        "latency_seconds": latency,
        "cumulative_from_initialization_seconds": cumulative,
        "timing": {
            "latency_seconds": dict(timing["latency_seconds"]),
            "cumulative_from_initialization_seconds": dict(
                timing["cumulative_from_initialization_seconds"]
            ),
            "output_token_rate_tokens_per_second": dict(rate),
        },
        "metrics": {name: dict(value) for name, value in metrics.items()},
        "terminal": True,
    }
    if cell.lane == "natural":
        public["decoded_output"] = request["decoded_output"]
    return public


def _validate_cell(
    raw: dict[str, Any],
    cell: core.ScheduleCell,
    *,
    plan: core.VLLMCompilePlan,
    host_lifecycle_duration_ns: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], bool]:
    _verify_seal(raw, "cell_sha256")
    expected_keys = {
        "schema_version",
        "protocol_id",
        "cell",
        "plan_sha256",
        "analysis_seed",
        "model",
        "budget",
        "runtime",
        "deterministic_environment",
        "hardware_commitment",
        "process_tree",
        "measurements",
        "request_count_expected",
        "request_count_observed",
        "prompt_ids_sha256",
        "staging_receipt_sha256",
        "requests",
        "terminal",
        "cell_sha256",
    }
    if set(raw) != expected_keys:
        raise CrossoverResultsError(f"{cell.cell_id} terminal receipt keys differ")
    fixed = {
        "schema_version": "2",
        "protocol_id": core.PROTOCOL_ID,
        "cell": cell.to_dict(),
        "plan_sha256": plan.content_sha256,
        "analysis_seed": core.ANALYSIS_SEED,
        "request_count_expected": cell.requests_per_cell,
        "request_count_observed": cell.requests_per_cell,
        "terminal": True,
    }
    if any(raw.get(key) != value for key, value in fixed.items()):
        raise CrossoverResultsError(f"{cell.cell_id} terminal receipt binding differs")
    expected_model = {
        "id": core.MODEL_ID,
        "revision": core.MODEL_REVISION,
        "expected_file_count": core.EXPECTED_MODEL_FILE_COUNT,
        "expected_bytes": core.EXPECTED_MODEL_BYTES,
        "state_receipt": "staging-receipt.json",
        "prompt_receipt": "prompt-token-ids.json",
    }
    if raw["model"] != expected_model or raw["budget"] != {"hard_cap_usd": "3.00"}:
        raise CrossoverResultsError(f"{cell.cell_id} model or budget binding differs")
    runtime = raw["runtime"]
    plan_document = plan.to_dict()
    expected_resolved = plan_document["execution_modes"][cell.mode]
    if (
        not isinstance(runtime, dict)
        or set(runtime)
        != {
            "pins",
            "expected_pins",
            "runtime_image",
            "resolved_execution_config",
            "optional_version_pinned_fields",
        }
        or runtime.get("pins") != core.RUNTIME_PINS
        or runtime.get("expected_pins") != core.RUNTIME_PINS
        or runtime.get("runtime_image")
        != {
            "base_reference": core.BASE_IMAGE_REFERENCE,
            "derived_image_id": core.DERIVED_IMAGE_ID,
        }
        or runtime.get("resolved_execution_config") != expected_resolved
    ):
        raise CrossoverResultsError(f"{cell.cell_id} runtime commitment differs")
    environment = raw["deterministic_environment"]
    expected_variables = plan_document["reproducibility"]["environment"]
    expected_cache_roles = {
        "vllm": ("VLLM_CACHE_ROOT", "vllm"),
        "torchinductor": ("TORCHINDUCTOR_CACHE_DIR", "torchinductor"),
        "triton": ("TRITON_CACHE_DIR", "triton"),
        "cuda": ("CUDA_CACHE_PATH", "cuda"),
        "home": ("HOME", "home"),
        "huggingface": ("HF_HOME", "huggingface"),
        "xdg": ("XDG_CACHE_HOME", "xdg"),
    }
    if (
        not isinstance(environment, dict)
        or set(environment) != {"variables", "cache_root_role", "cache_roles"}
        or environment.get("variables") != expected_variables
        or plan_document["lifecycle_controls"]["cell_unique_cache_directories"]
        != list(expected_cache_roles)
    ):
        raise CrossoverResultsError(f"{cell.cell_id} deterministic environment differs")
    cache_root_role = environment["cache_root_role"]
    cache_roles = environment["cache_roles"]
    if (
        not isinstance(cache_root_role, dict)
        or set(cache_root_role) != {"relative_identity", "path_sha256"}
        or cache_root_role.get("relative_identity") != cell.cell_id
        or not isinstance(cache_root_role.get("path_sha256"), str)
        or not _SHA256.fullmatch(cache_root_role["path_sha256"])
        or not isinstance(cache_roles, dict)
        or set(cache_roles) != set(expected_cache_roles)
    ):
        raise CrossoverResultsError(f"{cell.cell_id} cache attestation differs")
    for role, (env_var, relative_path) in expected_cache_roles.items():
        attestation = cache_roles[role]
        if (
            not isinstance(attestation, dict)
            or set(attestation) != {"env_var", "relative_path", "path_sha256"}
            or attestation.get("env_var") != env_var
            or attestation.get("relative_path") != relative_path
            or not isinstance(attestation.get("path_sha256"), str)
            or not _SHA256.fullmatch(attestation["path_sha256"])
        ):
            raise CrossoverResultsError(f"{cell.cell_id} cache role differs")
    hardware = raw["hardware_commitment"]
    if (
        not isinstance(hardware, dict)
        or set(hardware)
        != {
            "gpu_name",
            "gpu_count",
            "driver_version",
            "memory_total_mib",
            "memory_used_mib",
            "public_experiment_nonce",
            "gpu_identity_commitment",
        }
        or hardware.get("gpu_name") != core.EXPECTED_GPU_NAME
        or hardware.get("gpu_count") != 1
        or hardware.get("driver_version") != core.EXPECTED_DRIVER
        or hardware.get("memory_total_mib") != core.EXPECTED_MEMORY_MIB
        or isinstance(hardware.get("memory_used_mib"), bool)
        or not isinstance(hardware.get("memory_used_mib"), int)
        or not 0 <= hardware["memory_used_mib"] <= core.EXPECTED_MEMORY_MIB
        or not isinstance(hardware.get("gpu_identity_commitment"), str)
        or not _SHA256.fullmatch(hardware["gpu_identity_commitment"])
    ):
        raise CrossoverResultsError(f"{cell.cell_id} hardware commitment differs")
    process_tree = _validate_process_tree(raw["process_tree"], cell.cell_id)
    measurements = raw["measurements"]
    if not isinstance(measurements, dict) or set(measurements) != {
        "initialization_seconds",
        "initialization_perf_counter_ns",
        "peak_gpu_memory_mib",
        "gpu_memory_series",
    }:
        raise CrossoverResultsError(f"{cell.cell_id} measurements are missing")
    initialization = _typed_seconds(
        measurements.get("initialization_seconds"),
        "initialization timing",
        expected_clock_domain="same_process_perf_counter",
        expected_provenance="measured_perf_counter_ns",
    )
    initialization_ns = _positive_int(
        measurements.get("initialization_perf_counter_ns"),
        "initialization ns",
    )
    if not math.isclose(
        initialization, initialization_ns / 1_000_000_000, abs_tol=1e-12
    ):
        raise CrossoverResultsError("initialization typed and integer timings differ")
    peak_observed = _validate_typed_measurement(
        measurements.get("peak_gpu_memory_mib"),
        "peak GPU memory",
        unit="MiB",
        clock_domain="sampled_nvidia_smi",
        provenance="sampled_nvidia_smi",
        null_reasons={"unobservable": {"nvidia_smi_peak_memory_unavailable"}},
    )
    memory_series = measurements["gpu_memory_series"]
    series_keys = {
        "value",
        "unit",
        "clock_domain",
        "provenance",
        "observability_state",
        "null_reason",
        "target_interval_ms",
        "sampling_error_count",
        "sampling_error_types",
    }
    if (
        not isinstance(memory_series, dict)
        or set(memory_series) != series_keys
        or memory_series["unit"] != "MiB"
        or memory_series["clock_domain"] != "same_process_perf_counter_offset_ns"
        or memory_series["provenance"] != "sampled_nvidia_smi"
        or memory_series["target_interval_ms"] != 200
        or isinstance(memory_series["sampling_error_count"], bool)
        or not isinstance(memory_series["sampling_error_count"], int)
        or memory_series["sampling_error_count"] < 0
        or not isinstance(memory_series["sampling_error_types"], list)
        or memory_series["sampling_error_count"]
        != len(memory_series["sampling_error_types"])
        or any(
            not isinstance(error, str) or not error or len(error) > 100
            for error in memory_series["sampling_error_types"]
        )
    ):
        raise CrossoverResultsError("GPU memory series schema differs")
    if memory_series["observability_state"] == "observed":
        samples = memory_series["value"]
        if (
            memory_series["null_reason"] is not None
            or not isinstance(samples, list)
            or not samples
            or len(samples) > 1_000_000
            or any(
                not isinstance(sample, dict)
                or set(sample) != {"offset_ns", "memory_used_mib"}
                or isinstance(sample["offset_ns"], bool)
                or not isinstance(sample["offset_ns"], int)
                or sample["offset_ns"] < 0
                or isinstance(sample["memory_used_mib"], bool)
                or not isinstance(sample["memory_used_mib"], int)
                or not 0 <= sample["memory_used_mib"] <= core.EXPECTED_MEMORY_MIB
                for sample in samples
            )
            or any(
                current["offset_ns"] <= previous["offset_ns"]
                for previous, current in zip(samples, samples[1:], strict=False)
            )
            or not peak_observed
            or measurements["peak_gpu_memory_mib"]["value"]
            != max(sample["memory_used_mib"] for sample in samples)
        ):
            raise CrossoverResultsError("GPU memory series values differ")
    elif (
        memory_series["observability_state"] != "unobservable"
        or memory_series["value"] is not None
        or memory_series["null_reason"] != "nvidia_smi_memory_series_unavailable"
        or peak_observed
    ):
        raise CrossoverResultsError("GPU memory series null provenance differs")
    descriptors = core.lane_request_descriptors(cell.lane)
    requests = raw["requests"]
    if not isinstance(requests, list) or len(requests) != len(descriptors):
        raise CrossoverResultsError(f"{cell.cell_id} request count differs")
    public_requests = [
        _validate_request(request, cell=cell, descriptor=descriptor, index=index)
        for index, (descriptor, request) in enumerate(
            zip(descriptors, requests, strict=True), start=1
        )
    ]
    curve = [
        request["cumulative_from_initialization_seconds"] for request in public_requests
    ]
    if curve[0] < initialization or any(
        current <= previous for previous, current in zip(curve, curve[1:], strict=False)
    ):
        raise CrossoverResultsError(f"{cell.cell_id} cumulative curve is not monotonic")
    optional = runtime.get("optional_version_pinned_fields")
    if not isinstance(optional, dict) or set(optional) != {
        "compiled_mode_expected",
        "compilation_config_fields",
        "encoder_compilation_config",
        "compilation_time_seconds",
        "encoder_compilation_time_seconds",
        "cuda_graph_capture_duration_seconds",
        "cuda_graph_dispatch_counter",
    }:
        raise CrossoverResultsError("optional runtime provenance is missing")
    if optional["compiled_mode_expected"] is not (cell.mode == "compiled"):
        raise CrossoverResultsError("optional runtime mode differs")
    config_fields = optional["compilation_config_fields"]
    if (
        not isinstance(config_fields, dict)
        or set(config_fields)
        != {
            "value",
            "unit",
            "clock_domain",
            "provenance",
            "observability_state",
            "null_reason",
        }
        or config_fields["unit"] != "json"
        or config_fields["clock_domain"] != "resolved_runtime_config"
        or config_fields["provenance"] != "version_pinned_vllm_0_28_internal"
        or config_fields["observability_state"] not in {"observed", "unobservable"}
        or (
            config_fields["observability_state"] == "observed"
            and (
                not isinstance(config_fields["value"], dict)
                or set(config_fields["value"])
                != {
                    "backend",
                    "compile_sizes",
                    "inductor_compile_config",
                    "pass_config",
                    "splitting_ops",
                }
                or config_fields["null_reason"] is not None
            )
        )
        or (
            config_fields["observability_state"] == "unobservable"
            and (
                config_fields["value"] is not None
                or config_fields["null_reason"]
                != "optional_compilation_fields_not_exposed"
            )
        )
    ):
        raise CrossoverResultsError("compilation config attestation differs")
    encoder_config = optional["encoder_compilation_config"]
    if (
        not isinstance(encoder_config, dict)
        or set(encoder_config)
        != {
            "field_name",
            "value",
            "unit",
            "clock_domain",
            "provenance",
            "observability_state",
            "null_reason",
        }
        or encoder_config["field_name"] != "encoder_compilation_config"
        or encoder_config["unit"] != "json"
        or encoder_config["clock_domain"] != "resolved_runtime_config"
        or encoder_config["provenance"] != "version_pinned_vllm_0_28_internal"
        or encoder_config["observability_state"] not in {"observed", "unobservable"}
        or (
            encoder_config["observability_state"] == "observed"
            and (
                encoder_config["value"] is None
                or encoder_config["null_reason"] is not None
            )
        )
        or (
            encoder_config["observability_state"] == "unobservable"
            and (
                encoder_config["value"] is not None
                or encoder_config["null_reason"]
                != "encoder_compilation_config_not_exposed"
            )
        )
    ):
        raise CrossoverResultsError("encoder config attestation differs")
    null_state = "unobservable" if cell.mode == "compiled" else "not_applicable"
    null_reason = None if cell.mode == "compiled" else "not_applicable_eager_mode"
    component_observed_values = []
    for name, compiled_null_reason in (
        ("compilation_time_seconds", "compilation_time_not_exposed_by_vllm_0_28"),
        (
            "encoder_compilation_time_seconds",
            "encoder_compilation_time_not_exposed_by_vllm_0_28",
        ),
        (
            "cuda_graph_capture_duration_seconds",
            "cuda_graph_capture_duration_not_exposed_by_vllm",
        ),
    ):
        reasons = (
            {"unobservable": {compiled_null_reason}}
            if null_reason is None
            else {null_state: {null_reason}}
        )
        component_observed_values.append(
            _validate_typed_measurement(
                optional[name],
                f"runtime.{name}",
                unit="seconds",
                clock_domain="vllm_internal_runtime",
                provenance="version_pinned_vllm_0_28_internal",
                null_reasons=reasons,
            )
        )
    component_observed = all(component_observed_values)
    dispatch_counter = optional["cuda_graph_dispatch_counter"]
    dispatch_state = "unobservable" if cell.mode == "compiled" else "not_applicable"
    dispatch_reason = (
        "offline_llm_has_no_stable_cuda_graph_dispatch_metric_snapshot_hook"
        if cell.mode == "compiled"
        else "not_applicable_eager_mode"
    )
    if _validate_typed_measurement(
        dispatch_counter,
        "runtime.cuda_graph_dispatch_counter",
        unit="requests",
        clock_domain="vllm_metrics_registry",
        provenance="documented_vllm_0_28_metric",
        null_reasons={dispatch_state: {dispatch_reason}},
    ):
        raise CrossoverResultsError("CUDA graph dispatch counter must remain null")
    public_cell = {
        "cell_id": cell.cell_id,
        "period_index": cell.period_index,
        "mode": cell.mode,
        "process_tree": process_tree,
        "initialization_seconds": initialization,
        "cumulative_seconds": curve,
        "measurements": {
            "host_lifecycle_seconds": {
                "value": host_lifecycle_duration_ns / 1_000_000_000,
                "unit": "seconds",
                "clock_domain": "host_perf_counter",
                "provenance": "operation_receipt_duration_ns",
                "observability_state": "observed",
                "null_reason": None,
            },
            "initialization_seconds": dict(measurements["initialization_seconds"]),
            "peak_gpu_memory_mib": dict(measurements["peak_gpu_memory_mib"]),
            "gpu_memory_series": dict(memory_series),
        },
        "compile_component_measurements": {
            name: dict(optional[name])
            for name in (
                "compilation_time_seconds",
                "encoder_compilation_time_seconds",
                "cuda_graph_capture_duration_seconds",
                "cuda_graph_dispatch_counter",
            )
        },
        "terminal": True,
    }
    return public_cell, public_requests, component_observed


def _endpoint(
    value: int | None, *, limit: int, sustained: bool = False
) -> dict[str, Any]:
    if value is not None:
        return {"state": "observed", "request_count": value, "lower_bound": None}
    return {
        "state": "right_censored" if sustained else "open",
        "request_count": None,
        "lower_bound": limit,
    }


def _interval_endpoint(
    value: int | None, *, is_open: bool, limit: int
) -> dict[str, Any]:
    if is_open:
        if value is not None and value <= limit:
            raise CrossoverResultsError("open bootstrap endpoint is not censored")
        return {
            "state": "open",
            "request_count": None,
            "lower_bound": limit,
            "censor_sentinel_request_count": value or limit + 1,
        }
    if value is None or value > limit:
        raise CrossoverResultsError("closed bootstrap endpoint has no value")
    return {
        "state": "observed",
        "request_count": value,
        "lower_bound": None,
        "censor_sentinel_request_count": None,
    }


def _identity_summary(
    cells: Sequence[core.ScheduleCell],
    requests_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    lane: str,
) -> dict[str, Any]:
    lane_cells = [cell for cell in cells if cell.lane == lane]
    count = lane_cells[0].requests_per_cell
    mismatches: list[dict[str, Any]] = []
    within_mode_identical = True
    cross_mode_identical = True
    for index in range(count):
        by_mode: dict[str, list[tuple[int, ...]]] = {"eager": [], "compiled": []}
        for cell in lane_cells:
            ids = tuple(requests_by_cell[cell.cell_id][index]["output_token_ids"])
            by_mode[cell.mode].append(ids)
        if len(set(by_mode["eager"])) != 1:
            within_mode_identical = False
            mismatches.append(
                {"request_sequence_index": index + 1, "scope": "eager_lifecycles"}
            )
        if len(set(by_mode["compiled"])) != 1:
            within_mode_identical = False
            mismatches.append(
                {"request_sequence_index": index + 1, "scope": "compiled_lifecycles"}
            )
        pair_matches = []
        for pair_index in range(1, core.PAIRS_PER_LANE + 1):
            pair = [cell for cell in lane_cells if cell.pair_index == pair_index]
            pair_by_mode = {cell.mode: cell for cell in pair}
            pair_matches.append(
                requests_by_cell[pair_by_mode["eager"].cell_id][index][
                    "output_token_ids"
                ]
                == requests_by_cell[pair_by_mode["compiled"].cell_id][index][
                    "output_token_ids"
                ]
            )
        if not all(pair_matches):
            cross_mode_identical = False
            mismatches.append({"request_sequence_index": index + 1, "scope": "modes"})
    return {
        "cross_mode_pair_outputs_identical": cross_mode_identical,
        "within_mode_lifecycles_identical": within_mode_identical,
        "all_corresponding_outputs_identical": (
            cross_mode_identical and within_mode_identical
        ),
        "mismatches": mismatches,
    }


_PAIR_EFFECT_UNITS = {
    "initialization": "seconds",
    "host_lifecycle": "seconds",
    "request_phase": "seconds",
    "cumulative_init_to_terminal": "seconds",
    "mean_ttft": "seconds",
    "mean_prefill": "seconds",
    "mean_decode": "seconds",
    "mean_output_rate": "tokens_per_second",
    "peak_gpu_memory": "MiB",
}
_PAIR_EFFECT_PROVENANCE = {
    "initialization": "compiled_minus_eager_cell_initialization_perf_counter",
    "host_lifecycle": "compiled_minus_eager_host_operation_receipt_duration_ns",
    "request_phase": "compiled_minus_eager_sum_request_terminal_latency",
    "cumulative_init_to_terminal": (
        "compiled_minus_eager_terminal_cumulative_from_initialization"
    ),
    "mean_ttft": "compiled_minus_eager_mean_request_ttft",
    "mean_prefill": "compiled_minus_eager_mean_request_prefill",
    "mean_decode": "compiled_minus_eager_mean_request_decode",
    "mean_output_rate": "compiled_minus_eager_mean_exact_output_token_rate",
    "peak_gpu_memory": "compiled_minus_eager_sampled_nvidia_smi_peak",
}


def _pair_effect_value(
    metric: str,
    value: float | None,
    *,
    null_reason: str | None = None,
) -> dict[str, Any]:
    if value is not None and not math.isfinite(value):
        raise CrossoverResultsError(f"{metric} pair effect is nonfinite")
    if (value is None) is (null_reason is None):
        raise CrossoverResultsError(f"{metric} pair effect null provenance differs")
    return {
        "value": value,
        "unit": _PAIR_EFFECT_UNITS[metric],
        "provenance": _PAIR_EFFECT_PROVENANCE[metric],
        "null_reason": null_reason,
    }


def _observed_request_mean(
    requests: Sequence[Mapping[str, Any]],
    *,
    container: str,
    metric: str,
) -> float | None:
    measurements = [request[container][metric] for request in requests]
    if any(item["observability_state"] != "observed" for item in measurements):
        return None
    return sum(float(item["value"]) for item in measurements) / len(measurements)


def _compute_pair_effects(
    eager: Mapping[str, Any],
    compiled: Mapping[str, Any],
    eager_requests: Sequence[Mapping[str, Any]],
    compiled_requests: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    scalar_values = {
        "initialization": (
            float(compiled["measurements"]["initialization_seconds"]["value"])
            - float(eager["measurements"]["initialization_seconds"]["value"])
        ),
        "host_lifecycle": (
            float(compiled["measurements"]["host_lifecycle_seconds"]["value"])
            - float(eager["measurements"]["host_lifecycle_seconds"]["value"])
        ),
        "request_phase": (
            sum(float(request["latency_seconds"]) for request in compiled_requests)
            - sum(float(request["latency_seconds"]) for request in eager_requests)
        ),
        "cumulative_init_to_terminal": (
            float(compiled["cumulative_seconds"][-1])
            - float(eager["cumulative_seconds"][-1])
        ),
    }
    effects = {
        metric: _pair_effect_value(metric, value)
        for metric, value in scalar_values.items()
    }
    for metric, request_metric in (
        ("mean_ttft", "ttft_seconds"),
        ("mean_prefill", "prefill_seconds"),
        ("mean_decode", "decode_seconds"),
    ):
        eager_mean = _observed_request_mean(
            eager_requests, container="metrics", metric=request_metric
        )
        compiled_mean = _observed_request_mean(
            compiled_requests, container="metrics", metric=request_metric
        )
        effects[metric] = _pair_effect_value(
            metric,
            (
                None
                if eager_mean is None or compiled_mean is None
                else compiled_mean - eager_mean
            ),
            null_reason=(
                "not_all_requests_observed_in_both_cells"
                if eager_mean is None or compiled_mean is None
                else None
            ),
        )
    eager_rate = _observed_request_mean(
        eager_requests,
        container="timing",
        metric="output_token_rate_tokens_per_second",
    )
    compiled_rate = _observed_request_mean(
        compiled_requests,
        container="timing",
        metric="output_token_rate_tokens_per_second",
    )
    effects["mean_output_rate"] = _pair_effect_value(
        "mean_output_rate",
        (
            None
            if eager_rate is None or compiled_rate is None
            else compiled_rate - eager_rate
        ),
        null_reason=(
            "not_all_requests_observed_in_both_cells"
            if eager_rate is None or compiled_rate is None
            else None
        ),
    )
    eager_peak = eager["measurements"]["peak_gpu_memory_mib"]
    compiled_peak = compiled["measurements"]["peak_gpu_memory_mib"]
    peak_observed = (
        eager_peak["observability_state"] == "observed"
        and compiled_peak["observability_state"] == "observed"
    )
    effects["peak_gpu_memory"] = _pair_effect_value(
        "peak_gpu_memory",
        (
            float(compiled_peak["value"]) - float(eager_peak["value"])
            if peak_observed
            else None
        ),
        null_reason=(
            None
            if peak_observed
            else "peak_gpu_memory_unobserved_in_eager_or_compiled_cell"
        ),
    )
    return effects


def _validate_public_pair_curves(
    pair_records: Sequence[Mapping[str, Any]],
    requests_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    schedule: Sequence[Mapping[str, Any]],
) -> None:
    expected_pairs: list[dict[str, Any]] = []
    for lane in core.LANES:
        for pair_index in range(1, core.PAIRS_PER_LANE + 1):
            pair_cells = [
                cell
                for cell in schedule
                if cell["lane"] == lane and cell["pair_index"] == pair_index
            ]
            if (
                len(pair_cells) != 2
                or abs(schedule.index(pair_cells[0]) - schedule.index(pair_cells[1]))
                != 1
                or {cell["mode"] for cell in pair_cells} != {"eager", "compiled"}
                or len({cell["pair_id"] for cell in pair_cells}) != 1
                or len({cell["order"] for cell in pair_cells}) != 1
            ):
                raise CrossoverResultsError(
                    "canonical lifecycle pair schedule is invalid"
                )
            by_mode = {cell["mode"]: cell for cell in pair_cells}
            expected_pairs.append(
                {
                    "lane": lane,
                    "pair_index": pair_index,
                    "pair_id": pair_cells[0]["pair_id"],
                    "order": pair_cells[0]["order"],
                    "cell_ids_in_execution_order": [
                        cell["cell_id"] for cell in pair_cells
                    ],
                    "eager": by_mode["eager"],
                    "compiled": by_mode["compiled"],
                }
            )
    if len(pair_records) != len(expected_pairs):
        raise CrossoverResultsError("public lifecycle pair inventory differs")
    expected_cell_ids = {cell["cell_id"] for cell in schedule}
    if set(requests_by_cell) != expected_cell_ids:
        raise CrossoverResultsError("public request cell inventory differs")
    seen_cells: set[str] = set()
    for pair, expected in zip(pair_records, expected_pairs, strict=True):
        for field in (
            "lane",
            "pair_index",
            "pair_id",
            "order",
            "cell_ids_in_execution_order",
        ):
            if pair.get(field) != expected[field]:
                raise CrossoverResultsError(
                    "public lifecycle pair canonical binding differs"
                )
        for mode in ("eager", "compiled"):
            cell = pair[mode]
            expected_cell = expected[mode]
            cell_id = cell["cell_id"]
            if (
                cell_id != expected_cell["cell_id"]
                or cell.get("mode") != mode
                or cell.get("period_index") != expected_cell["period_index"]
                or cell.get("terminal") is not True
            ):
                raise CrossoverResultsError(
                    "public lifecycle cell canonical binding differs"
                )
            if cell_id in seen_cells:
                raise CrossoverResultsError(
                    "public lifecycle cell appears more than once"
                )
            seen_cells.add(cell_id)
            requests = sorted(
                requests_by_cell[cell_id],
                key=lambda request: request["request_sequence_index"],
            )
            if len(requests) != expected_cell["requests_per_cell"] or any(
                request.get("cell_id") != cell_id
                or request.get("lane") != expected["lane"]
                or request.get("mode") != mode
                or request.get("pair_id") != expected["pair_id"]
                for request in requests
            ):
                raise CrossoverResultsError(
                    "public request canonical role binding differs"
                )
            if [request["request_sequence_index"] for request in requests] != list(
                range(1, len(requests) + 1)
            ):
                raise CrossoverResultsError("public request sequence is not contiguous")
            curve = [
                request["cumulative_from_initialization_seconds"]
                for request in requests
            ]
            if (
                cell["cumulative_seconds"] != curve
                or not curve
                or curve[0] < cell["initialization_seconds"]
                or any(
                    current <= previous
                    for previous, current in zip(curve, curve[1:], strict=False)
                )
            ):
                raise CrossoverResultsError(
                    "public cell cumulative curve does not match requests"
                )
        expected_difference = [
            compiled - eager
            for eager, compiled in zip(
                pair["eager"]["cumulative_seconds"],
                pair["compiled"]["cumulative_seconds"],
                strict=True,
            )
        ]
        if pair["compiled_minus_eager_seconds"] != expected_difference:
            raise CrossoverResultsError(
                "public lifecycle pair difference curve does not recompute"
            )
    if len(seen_cells) != len(pair_records) * 2:
        raise CrossoverResultsError("public lifecycle cell cardinality differs")


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _pair_effect_distributions(
    pair_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    lanes: dict[str, Any] = {}
    for lane in core.LANES:
        lane_pairs = [pair for pair in pair_records if pair["lane"] == lane]
        metrics: dict[str, Any] = {}
        for metric in _PAIR_EFFECT_UNITS:
            effect_records = [
                {
                    "pair_id": pair["pair_id"],
                    "value": pair["pair_effects"][metric]["value"],
                    "null_reason": pair["pair_effects"][metric]["null_reason"],
                }
                for pair in lane_pairs
            ]
            observed = [
                float(effect["value"])
                for effect in effect_records
                if effect["value"] is not None
            ]
            mean: float | None
            median: float | None
            if observed:
                ordered = sorted(observed)
                midpoint = len(ordered) // 2
                lower = ordered[:midpoint]
                upper = ordered[(len(ordered) + 1) // 2 :]
                q1 = _median(lower) if lower else ordered[0]
                q3 = _median(upper) if upper else ordered[-1]
                mean = sum(observed) / len(observed)
                median = _median(observed)
                range_value: dict[str, float | None] = {
                    "minimum": ordered[0],
                    "maximum": ordered[-1],
                }
                iqr: dict[str, float | None] = {
                    "first_quartile": q1,
                    "third_quartile": q3,
                    "width": q3 - q1,
                }
                summary_null_reason = None
            else:
                mean = median = None
                range_value = {"minimum": None, "maximum": None}
                iqr = {
                    "first_quartile": None,
                    "third_quartile": None,
                    "width": None,
                }
                summary_null_reason = "no_observed_pair_effects"
            metrics[metric] = {
                "unit": _PAIR_EFFECT_UNITS[metric],
                "provenance": _PAIR_EFFECT_PROVENANCE[metric],
                "pair_count": len(effect_records),
                "observed_effect_count": len(observed),
                "effects": effect_records,
                "mean": mean,
                "median": median,
                "iqr": iqr,
                "range": range_value,
                "summary_null_reason": summary_null_reason,
            }
        lanes[lane] = metrics
    return {
        "unit_of_analysis": "lifecycle_pair",
        "request_level_resampling": False,
        "summary_method": (
            "descriptive_exclusive_halves_moore_mccabe_over_observed_pair_effects"
        ),
        "lanes": lanes,
    }


def _component_observability(
    pair_records: Sequence[Mapping[str, Any]],
) -> bool:
    component_names = (
        "compilation_time_seconds",
        "encoder_compilation_time_seconds",
        "cuda_graph_capture_duration_seconds",
    )
    compiled_cells = [pair["compiled"] for pair in pair_records]
    return bool(compiled_cells) and all(
        cell["compile_component_measurements"][name]["observability_state"]
        == "observed"
        for cell in compiled_cells
        for name in component_names
    )


def _claim_matrix_document(
    *,
    analysis: Mapping[str, Any],
    controlled_identity: Mapping[str, Any],
    natural_identity: Mapping[str, Any],
    natural_all_correct: bool,
    quality_preservation: Mapping[str, Any],
    component_observability: bool,
) -> dict[str, Any]:
    gate = core.ClaimGate(
        terminal=True,
        completeness=True,
        fixed_count=True,
        controlled_supported_crossing=_controlled_supported_crossing(
            analysis["controlled"]
        ),
        controlled_output_identity=controlled_identity[
            "cross_mode_pair_outputs_identical"
        ],
        controlled_numeric_reproducibility=controlled_identity[
            "within_mode_lifecycles_identical"
        ],
        natural_output_identity=natural_identity["cross_mode_pair_outputs_identical"],
        natural_numeric_reproducibility=natural_identity[
            "within_mode_lifecycles_identical"
        ],
        natural_supported_speedup=analysis["controlled"]["natural_timing"][
            "speedup_supported"
        ],
        natural_absolute_correctness=(
            natural_all_correct and quality_preservation["noninferiority_supported"]
        ),
        component_observability=component_observability,
    )
    claims = [decision.to_dict() for decision in gate.matrix()]
    claims.extend(
        [
            {
                "claim_id": "budget-reservations-within-hard-cap",
                "state": "supported",
                "blockers": [],
            },
            {
                "claim_id": "active-operation-list-rate-equivalent-within-hard-cap",
                "state": "supported",
                "blockers": [],
            },
            {
                "claim_id": "provider-billed-cost-within-hard-cap",
                "state": "unsupported",
                "blockers": ["external_provider_end_receipt_absent"],
            },
            {
                "claim_id": "provider-teardown",
                "state": "unsupported",
                "blockers": ["external_provider_fact_not_observed"],
            },
        ]
    )
    return {"schema_version": RESULT_SCHEMA_VERSION, "claims": claims}


def _controlled_supported_crossing(controlled: Mapping[str, Any]) -> bool:
    return (
        controlled["simultaneous_band_sustained_crossing_request_count"] is not None
        and controlled["terminal_effect_sign_flip_p_value"]
        <= CONTROLLED_SIGN_SYMMETRY_ALPHA
    )


def _analysis_document(
    pairs: Sequence[core.PairCurve],
    *,
    natural_identity: bool,
    natural_terminal_effects: Sequence[float],
    pair_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result = core.analyze_pair_curves(
        pairs,
        resample_count=BOOTSTRAP_RESAMPLES,
        analysis_seed=core.ANALYSIS_SEED,
    ).to_dict()
    limit = core.CONTROLLED_REQUESTS_PER_CELL
    result["aggregate_first_crossing"] = _endpoint(
        result["aggregate_first_crossing_request_count"], limit=limit
    )
    result["aggregate_sustained_crossing"] = _endpoint(
        result["aggregate_sustained_crossing_request_count"],
        limit=limit,
        sustained=True,
    )
    result["simultaneous_band_first_crossing"] = _endpoint(
        result["simultaneous_band_first_crossing_request_count"], limit=limit
    )
    result["simultaneous_band_sustained_crossing"] = _endpoint(
        result["simultaneous_band_sustained_crossing_request_count"],
        limit=limit,
        sustained=True,
    )
    result["supported_first_crossing"] = dict(
        result["simultaneous_band_first_crossing"]
    )
    supported_gate = _controlled_supported_crossing(result)
    result["supported_crossing_gate_satisfied"] = supported_gate
    result["supported_sustained_crossing"] = (
        dict(result["simultaneous_band_sustained_crossing"])
        if supported_gate
        else {
            "state": "unsupported",
            "request_count": None,
            "lower_bound": None,
            "reason": "combined_band_crossing_and_sign_symmetry_gate_not_satisfied",
        }
    )
    result["supported_crossing_basis"] = SUPPORTED_CROSSING_BASIS
    lower_open = result["bootstrap_sustained_crossing_lower_is_open"]
    upper_open = result["bootstrap_sustained_crossing_upper_is_open"]
    median = result["bootstrap_sustained_crossing_median_request_count"]
    median_open = median is None or median > limit
    result["bootstrap_sustained_crossing_interval"] = {
        "state": (
            "open"
            if lower_open and upper_open
            else ("partially_open" if lower_open or upper_open else "observed")
        ),
        "lower": _interval_endpoint(
            result["bootstrap_sustained_crossing_lower_request_count"],
            is_open=lower_open,
            limit=limit,
        ),
        "median": _interval_endpoint(
            median,
            is_open=median_open,
            limit=limit,
        ),
        "upper": _interval_endpoint(
            result["bootstrap_sustained_crossing_upper_request_count"],
            is_open=upper_open,
            limit=limit,
        ),
        "censor_at_request_count": limit,
        "censor_sentinel_request_count": limit + 1,
    }
    result["bootstrap_unit"] = "whole_lifecycle_pair"
    result["request_level_resampling"] = False
    natural_rng = random.Random(core.ANALYSIS_SEED)
    natural_bootstrap = [
        sum(
            natural_terminal_effects[
                natural_rng.randrange(len(natural_terminal_effects))
            ]
            for _ in natural_terminal_effects
        )
        / len(natural_terminal_effects)
        for _ in range(BOOTSTRAP_RESAMPLES)
    ]
    natural_mean = sum(natural_terminal_effects) / len(natural_terminal_effects)
    natural_lower = _ordered_quantile(natural_bootstrap, 0.025)
    natural_upper = _ordered_quantile(natural_bootstrap, 0.975)
    natural_speedup_supported = (
        natural_identity and natural_mean < 0 and natural_upper <= 0
    )
    result["natural_timing"] = {
        "causal_claim_eligible": natural_identity,
        "pair_terminal_compiled_minus_eager_seconds": list(natural_terminal_effects),
        "mean_terminal_compiled_minus_eager_seconds": natural_mean,
        "analysis_seed": core.ANALYSIS_SEED,
        "resample_count": BOOTSTRAP_RESAMPLES,
        "bootstrap_unit": "whole_lifecycle_pair",
        "request_level_resampling": False,
        "lower_confidence_endpoint_seconds": natural_lower,
        "upper_confidence_endpoint_seconds": natural_upper,
        "speedup_supported": natural_speedup_supported,
        "causal_claim_blocker": (
            None
            if natural_identity
            else "natural_outputs_differ_across_modes_or_lifecycles"
        ),
    }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "controlled": result,
        "pair_effect_distributions": _pair_effect_distributions(pair_records),
    }


def _ordered_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise CrossoverResultsError("quality bootstrap quantile has no values")
    index = max(0, min(len(values) - 1, math.ceil(quantile * len(values)) - 1))
    return sorted(values)[index]


def _quality_preservation(
    plan: core.VLLMCompilePlan,
    evaluations: Sequence[Mapping[str, Any]],
    *,
    resample_count: int | None = None,
) -> dict[str, Any]:
    if resample_count is None:
        resample_count = BOOTSTRAP_RESAMPLES
    if (
        isinstance(resample_count, bool)
        or not isinstance(resample_count, int)
        or resample_count <= 0
    ):
        raise CrossoverResultsError("quality bootstrap resample count is invalid")
    by_cell: dict[str, list[Mapping[str, Any]]] = {}
    for evaluation in evaluations:
        by_cell.setdefault(str(evaluation["cell_id"]), []).append(evaluation)
    pair_effects: list[dict[str, Any]] = []
    effects: list[float] = []
    for pair_index in range(1, core.PAIRS_PER_LANE + 1):
        pair_cells = [
            cell
            for cell in plan.schedule
            if cell.lane == "natural" and cell.pair_index == pair_index
        ]
        if len(pair_cells) != 2:
            raise CrossoverResultsError("natural quality lifecycle pair is incomplete")
        by_mode = {cell.mode: cell for cell in pair_cells}
        eager = by_cell.get(by_mode["eager"].cell_id, [])
        compiled = by_cell.get(by_mode["compiled"].cell_id, [])
        if len(eager) != core.NATURAL_REQUESTS_PER_CELL or len(compiled) != (
            core.NATURAL_REQUESTS_PER_CELL
        ):
            raise CrossoverResultsError("natural quality request count differs")
        eager_rate = sum(bool(item["success"]) for item in eager) / len(eager)
        compiled_rate = sum(bool(item["success"]) for item in compiled) / len(compiled)
        effect = compiled_rate - eager_rate
        effects.append(effect)
        pair_effects.append(
            {
                "pair_id": pair_cells[0].pair_id,
                "pair_index": pair_index,
                "eager_request_success_rate": eager_rate,
                "compiled_request_success_rate": compiled_rate,
                "compiled_minus_eager_request_success_rate": effect,
            }
        )
    margin = float(core.QUALITY_NONINFERIORITY_MARGIN)
    support_threshold = 0.0 if margin == 0 else -margin
    rng = random.Random(core.ANALYSIS_SEED)
    bootstrap = []
    for _ in range(resample_count):
        sampled = [effects[rng.randrange(len(effects))] for _ in effects]
        bootstrap.append(sum(sampled) / len(sampled))
    if len(set(effects)) == 1:
        confidence_method = "not_applicable_deterministic_pair_effects"
        confidence_level = None
        lower = upper = None
        supported = effects[0] >= support_threshold
        inference_state = (
            "deterministic_complete_agreement"
            if supported
            else "deterministic_noninferiority_failed"
        )
    else:
        lower = _ordered_quantile(bootstrap, 0.025)
        upper = _ordered_quantile(bootstrap, 0.975)
        confidence_method = "deterministic_whole_pair_percentile_bootstrap"
        confidence_level = "0.95"
        supported = lower >= support_threshold
        inference_state = "whole_pair_bootstrap"
    return {
        "lane": "natural",
        "evaluator": "evaluate_workload",
        "independent_unit": "adjacent_eager_compiled_lifecycle_pair",
        "effect": "compiled_minus_eager_request_success_rate",
        "pair_effects": pair_effects,
        "mean_pair_effect": sum(effects) / len(effects),
        "noninferiority_margin": core.canonical_decimal(
            core.QUALITY_NONINFERIORITY_MARGIN
        ),
        "confidence_method": confidence_method,
        "confidence_level": confidence_level,
        "analysis_seed": core.ANALYSIS_SEED,
        "resample_count": resample_count,
        "bootstrap_unit": "whole_lifecycle_pair",
        "request_level_resampling": False,
        "lower_confidence_endpoint": lower,
        "upper_confidence_endpoint": upper,
        "inference_state": inference_state,
        "support_threshold": support_threshold,
        "noninferiority_supported": supported,
    }


def _natural_evaluation(request: Mapping[str, Any]) -> dict[str, Any]:
    outcome = evaluate_workload(
        workload_by_id(str(request["workload_id"])),
        str(request["decoded_output"]),
    )
    return {
        "cell_id": request["cell_id"],
        "request_sequence_index": request["request_sequence_index"],
        "success": outcome.success,
        "quality_score": outcome.quality_score,
        "quality_metric": outcome.quality_metric,
        "notes": outcome.notes,
        "evaluator": "evaluate_workload",
        "workload_version": request["workload_version"],
        "output_token_ids_sha256": request["output_token_ids_sha256"],
    }


def _scan_privacy(name: str, text: str) -> None:
    for pattern, description in _PRIVATE_PATTERNS:
        if pattern.search(text):
            raise CrossoverResultsError(f"{name} contains {description}")


def _scan_raw_privacy(name: str, text: str) -> None:
    patterns = (
        (re.compile(r"/Users/|/home/|[A-Za-z]:\\Users\\"), "private path"),
        (re.compile(r"\bGPU-[0-9a-f-]{8,}\b", re.I), "raw GPU UUID"),
        (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "email address"),
        (
            re.compile(r"\b(?!127\.0\.0\.1\b)(?:\d{1,3}\.){3}\d{1,3}\b"),
            "IP address",
        ),
        (
            re.compile(
                r"\b(?:gh[pousr]_|github_pat_|sk-|hf[_-]|AKIA)" r"[A-Za-z0-9_-]{8,}\b"
            ),
            "credential-shaped value",
        ),
    )
    for pattern, description in patterns:
        if pattern.search(text):
            raise CrossoverResultsError(f"{name} contains {description}")


def _validate_operation_receipts(
    receipts: Any,
    *,
    ledger: Mapping[str, Any],
) -> dict[str, int]:
    entries = ledger["entries"]
    if not isinstance(receipts, list) or len(receipts) != len(entries):
        raise CrossoverResultsError("operation receipt cardinality differs")
    previous_ended_ns = -1
    durations: dict[str, int] = {}
    for receipt, entry in zip(receipts, entries, strict=True):
        if not isinstance(receipt, dict) or set(receipt) != {
            "command_id",
            "lifecycle_id",
            "line_id",
            "clock_domain",
            "started_ns",
            "ended_ns",
            "duration_ns",
            "status",
        }:
            raise CrossoverResultsError("operation receipt shape differs")
        if (
            receipt["command_id"] != entry["command_id"]
            or receipt["lifecycle_id"] != entry["lifecycle_id"]
            or receipt["line_id"] != entry["line_id"]
            or receipt["clock_domain"] != "host_perf_counter"
            or receipt["status"] != "completed"
        ):
            raise CrossoverResultsError("operation receipt ledger binding differs")
        started_ns = _positive_int(receipt["started_ns"], "operation started_ns")
        ended_ns = _positive_int(receipt["ended_ns"], "operation ended_ns")
        duration_ns = _positive_int(
            receipt["duration_ns"], "operation receipt duration_ns"
        )
        if (
            ended_ns - started_ns != duration_ns
            or started_ns < previous_ended_ns
            or ended_ns <= previous_ended_ns
            or math.ceil(duration_ns / 1_000_000_000) != entry["actual_seconds"]
        ):
            raise CrossoverResultsError(
                "operation duration/timing sequence or ledger duration differs"
            )
        previous_ended_ns = ended_ns
        durations[receipt["lifecycle_id"]] = duration_ns
    return durations


def _validate_hardware_observations(
    observations: Any,
    *,
    plan: core.VLLMCompilePlan,
) -> dict[str, Any]:
    expected_ids = ["preflight-after-reset"]
    for cell in plan.schedule:
        expected_ids.extend(
            (
                f"{cell.cell_id}-before-container",
                f"{cell.cell_id}-after-container",
            )
        )
    if not isinstance(observations, list) or len(observations) != 65:
        raise CrossoverResultsError("hardware observation cardinality differs")
    expected_keys = {
        "observation_id",
        "clock_domain",
        "host_perf_counter_ns",
        "gpu_identity_commitment",
        "gpu_name",
        "driver_version",
        "memory_total_mib",
        "memory_used_mib",
        "temperature_c",
        "utilization_percent",
        "power_limit_watts",
        "sm_clock_mhz",
        "compute_capability",
    }
    counters: list[int] = []
    commitments: set[str] = set()
    power_limits: set[float] = set()
    for observation, expected_id in zip(observations, expected_ids, strict=True):
        if not isinstance(observation, dict) or set(observation) != expected_keys:
            raise CrossoverResultsError("hardware observation shape differs")
        if (
            observation["observation_id"] != expected_id
            or observation["clock_domain"] != "host_perf_counter"
            or observation["gpu_name"] != core.EXPECTED_GPU_NAME
            or observation["driver_version"] != core.EXPECTED_DRIVER
            or observation["memory_total_mib"] != core.EXPECTED_MEMORY_MIB
            or observation["compute_capability"] != "8.9"
        ):
            raise CrossoverResultsError("hardware observation identity differs")
        counter = _positive_int(
            observation["host_perf_counter_ns"], "hardware perf counter"
        )
        memory_used = observation["memory_used_mib"]
        temperature = observation["temperature_c"]
        utilization = observation["utilization_percent"]
        sm_clock = observation["sm_clock_mhz"]
        if (
            isinstance(memory_used, bool)
            or not isinstance(memory_used, int)
            or not 0 <= memory_used <= MAX_BASELINE_GPU_MEMORY_MIB
            or isinstance(temperature, bool)
            or not isinstance(temperature, int)
            or not 0 <= temperature <= MAX_IDLE_GPU_TEMPERATURE_C
            or isinstance(utilization, bool)
            or not isinstance(utilization, int)
            or not 0 <= utilization <= MAX_IDLE_GPU_UTILIZATION_PERCENT
            or isinstance(sm_clock, bool)
            or not isinstance(sm_clock, int)
            or sm_clock <= 0
        ):
            raise CrossoverResultsError("hardware baseline guard differs")
        power = _positive_number(
            observation["power_limit_watts"], "hardware power limit"
        )
        commitment = observation["gpu_identity_commitment"]
        if not isinstance(commitment, str) or not _SHA256.fullmatch(commitment):
            raise CrossoverResultsError("hardware identity commitment is invalid")
        counters.append(counter)
        commitments.add(commitment)
        power_limits.add(power)
    if any(
        current <= previous
        for previous, current in zip(counters, counters[1:], strict=False)
    ):
        raise CrossoverResultsError("hardware observation clock is not monotonic")
    if len(commitments) != 1 or len(power_limits) != 1:
        raise CrossoverResultsError("hardware identity or power limit changed")
    return {
        "observation_count": len(observations),
        "clock_domain": "host_perf_counter",
        "gpu_name": core.EXPECTED_GPU_NAME,
        "driver_version": core.EXPECTED_DRIVER,
        "memory_total_mib": core.EXPECTED_MEMORY_MIB,
        "compute_capability": "8.9",
        "maximum_baseline_memory_used_mib": max(
            observation["memory_used_mib"] for observation in observations
        ),
        "maximum_idle_temperature_c": max(
            observation["temperature_c"] for observation in observations
        ),
        "maximum_idle_utilization_percent": max(
            observation["utilization_percent"] for observation in observations
        ),
        "power_limit_watts": next(iter(power_limits)),
        "minimum_sm_clock_mhz": min(
            observation["sm_clock_mhz"] for observation in observations
        ),
        "maximum_sm_clock_mhz": max(
            observation["sm_clock_mhz"] for observation in observations
        ),
    }


def _validate_workspace(workspace: Path) -> dict[str, Any]:
    root = workspace.resolve()
    if workspace.is_symlink() or not root.is_dir():
        raise CrossoverResultsError("workspace must be a non-symlink directory")
    plan = core.build_default_plan()
    authorization_raw = _safe_json(root / "authorization.json")
    _scan_raw_privacy("authorization.json", core.canonical_json(authorization_raw))
    try:
        authorization = ExecutionAuthorization.from_dict(authorization_raw)
    except (CrossoverOrchestratorError, TypeError, ValueError) as exc:
        raise CrossoverResultsError(f"authorization is invalid: {exc}") from exc
    if authorization.plan_sha256 != plan.content_sha256:
        raise CrossoverResultsError("authorization plan hash differs")
    ledger_path = root / "budget-ledger.json"
    ledger_text = read_bounded_regular_text(ledger_path, core.MAX_LEDGER_ARTIFACT_BYTES)
    _scan_raw_privacy("budget-ledger.json", ledger_text)
    ledger_reader = core.LifecycleBudgetLedger(
        ledger_path,
        plan=plan,
        git_head=authorization.source_head,
        workspace_path=root,
    )
    try:
        ledger = ledger_reader._read()
    except (OSError, core.VLLMCompileContractError) as exc:
        raise CrossoverResultsError(f"budget ledger is invalid: {exc}") from exc
    if (
        any(entry.get("status") != "completed" for entry in ledger["entries"])
        or any(event.get("event_type") == "abort" for event in ledger["events"])
        or len(ledger["entries"]) != len(plan.budget_lifecycles)
    ):
        raise CrossoverResultsError("every planned lifecycle must complete once")
    orchestration = _safe_json(root / "orchestration-receipt.json")
    _scan_raw_privacy("orchestration-receipt.json", core.canonical_json(orchestration))
    _verify_seal(orchestration, "orchestration_sha256")
    expected_orchestration_keys = {
        "schema_version",
        "protocol_id",
        "plan_sha256",
        "source_head",
        "runtime_image_id",
        "authorization_sha256",
        "authorization_authentication",
        "scheduled_shutdown_at",
        "repository_path_sha256",
        "workspace_path_sha256",
        "completed_cell_ids",
        "operation_receipts",
        "hardware_observations",
        "ledger_abort_failures",
        "status",
        "failure",
        "teardown_status",
        "host_shutdown_observed_at",
        "host_shutdown_observed_null_reason",
        "external_provider_console_confirmation",
        "external_provider_console_confirmation_null_reason",
        "independently_verified_provider_termination",
        "independently_verified_provider_termination_null_reason",
        "provider_teardown",
        "provider_teardown_null_reason",
        "ledger_sha256",
        "orchestration_sha256",
    }
    if set(orchestration) != expected_orchestration_keys:
        raise CrossoverResultsError("orchestration receipt keys differ")
    authentication = orchestration["authorization_authentication"]
    if (
        not isinstance(authentication, dict)
        or set(authentication)
        != {
            "mechanism",
            "namespace",
            "signer_identity",
            "signature_sha256",
            "authorized_signers_sha256",
            "verified",
        }
        or authentication["mechanism"] != "openssh_detached_signature"
        or authentication["namespace"] != AUTHORIZATION_SIGNATURE_NAMESPACE
        or authentication["signer_identity"] != AUTHORIZATION_SIGNER_IDENTITY
        or not isinstance(authentication["signature_sha256"], str)
        or not _SHA256.fullmatch(authentication["signature_sha256"])
        or not isinstance(authentication["authorized_signers_sha256"], str)
        or not _SHA256.fullmatch(authentication["authorized_signers_sha256"])
        or authentication["verified"] is not True
    ):
        raise CrossoverResultsError("authorization authentication evidence differs")
    if (
        orchestration["schema_version"] != "1"
        or orchestration["protocol_id"] != core.PROTOCOL_ID
        or orchestration["plan_sha256"] != plan.content_sha256
        or orchestration["source_head"] != authorization.source_head
        or orchestration["runtime_image_id"] != authorization.runtime_image_id
        or orchestration["authorization_sha256"] != authorization.authorization_sha256
        or orchestration["scheduled_shutdown_at"] != authorization.scheduled_shutdown_at
        or orchestration["workspace_path_sha256"] != authorization.workspace_sha256
        or orchestration["completed_cell_ids"]
        != [cell.cell_id for cell in plan.schedule]
        or orchestration["ledger_abort_failures"] != []
        or orchestration["status"] != "complete"
        or orchestration["failure"] is not None
        or orchestration["teardown_status"] != "local_cleanup_complete"
        or orchestration["host_shutdown_observed_at"] is not None
        or orchestration["host_shutdown_observed_null_reason"]
        != "the local process cannot observe its own later host shutdown"
        or orchestration["external_provider_console_confirmation"] is not None
        or orchestration["external_provider_console_confirmation_null_reason"]
        != "external operator confirmation was not supplied to the local runner"
        or orchestration["independently_verified_provider_termination"] is not None
        or orchestration["independently_verified_provider_termination_null_reason"]
        != "no provider API receipt is available"
        or orchestration["provider_teardown"] is not None
        or orchestration["provider_teardown_null_reason"]
        != "provider teardown remains externally user-confirmed"
        or not isinstance(orchestration["repository_path_sha256"], str)
        or not _SHA256.fullmatch(orchestration["repository_path_sha256"])
        or not isinstance(orchestration["workspace_path_sha256"], str)
        or not _SHA256.fullmatch(orchestration["workspace_path_sha256"])
        or orchestration["ledger_sha256"] != _sha256_uri(ledger_text.encode("utf-8"))
    ):
        raise CrossoverResultsError("orchestration completion binding differs")
    operation_durations = _validate_operation_receipts(
        orchestration["operation_receipts"], ledger=ledger
    )
    cell_lifecycles = {
        lifecycle.cell_id: lifecycle.lifecycle_id
        for lifecycle in plan.budget_lifecycles
        if lifecycle.cell_id is not None
    }
    hardware_summary = _validate_hardware_observations(
        orchestration["hardware_observations"], plan=plan
    )
    raw_dir = root / "raw"
    if raw_dir.is_symlink() or not raw_dir.is_dir():
        raise CrossoverResultsError("raw receipt directory is unsafe")
    expected_names = {
        name
        for cell in plan.schedule
        for name in (
            f"{cell.cell_id}.json",
            f".{cell.cell_id}-progress.json",
        )
    }
    raw_paths = list(raw_dir.iterdir())
    if {path.name for path in raw_paths} != expected_names or len(raw_paths) != 64:
        raise CrossoverResultsError("exact raw receipt inventory differs")
    for path in raw_paths:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > MAX_OUTPUT_BYTES
        ):
            raise CrossoverResultsError("raw directory contains an unsafe artifact")
        _scan_raw_privacy(
            path.name,
            read_bounded_regular_text(path, MAX_OUTPUT_BYTES),
        )
    public_cells: dict[str, dict[str, Any]] = {}
    requests_by_cell: dict[str, list[dict[str, Any]]] = {}
    raw_cells: dict[str, dict[str, Any]] = {}
    for cell in plan.schedule:
        raw = _safe_json(raw_dir / f"{cell.cell_id}.json")
        progress = _safe_json(raw_dir / f".{cell.cell_id}-progress.json")
        _verify_seal(progress, "progress_sha256")
        if set(progress) != {
            "schema_version",
            "protocol_id",
            "cell_id",
            "lane",
            "mode",
            "request_count_expected",
            "request_count_completed",
            "last_request_sequence_index",
            "requests",
            "terminal",
            "progress_sha256",
        }:
            raise CrossoverResultsError(f"{cell.cell_id} progress receipt keys differ")
        expected_progress = {
            "schema_version": "2",
            "protocol_id": core.PROTOCOL_ID,
            "cell_id": cell.cell_id,
            "lane": cell.lane,
            "mode": cell.mode,
            "request_count_expected": cell.requests_per_cell,
            "request_count_completed": cell.requests_per_cell,
            "last_request_sequence_index": cell.requests_per_cell,
            "terminal": False,
        }
        if any(
            progress.get(key) != value for key, value in expected_progress.items()
        ) or progress.get("requests") != raw.get("requests"):
            raise CrossoverResultsError(
                f"{cell.cell_id} progress receipt differs from terminal cell"
            )
        public, requests, _component_observed = _validate_cell(
            raw,
            cell,
            plan=plan,
            host_lifecycle_duration_ns=operation_durations[
                cell_lifecycles[cell.cell_id]
            ],
        )
        public_cells[cell.cell_id] = public
        requests_by_cell[cell.cell_id] = requests
        raw_cells[cell.cell_id] = raw
    prompt_seals = {cell["prompt_ids_sha256"] for cell in raw_cells.values()}
    staging_seals = {cell["staging_receipt_sha256"] for cell in raw_cells.values()}
    image_ids = {
        cell["runtime"]["runtime_image"]["derived_image_id"]
        for cell in raw_cells.values()
    }
    if (
        len(prompt_seals) != 1
        or len(staging_seals) != 1
        or any(
            not isinstance(value, str) or not _SHA256.fullmatch(value)
            for value in (*prompt_seals, *staging_seals)
        )
    ):
        raise CrossoverResultsError("prompt/staging receipt continuity differs")
    if image_ids != {authorization.runtime_image_id}:
        raise CrossoverResultsError("runtime image commitment continuity differs")
    observation_commitments = {
        observation["gpu_identity_commitment"]
        for observation in orchestration["hardware_observations"]
    }
    cell_commitments = {
        cell["hardware_commitment"]["gpu_identity_commitment"]
        for cell in raw_cells.values()
    }
    if (
        len(observation_commitments) != 1
        or len(cell_commitments) != 1
        or observation_commitments != cell_commitments
    ):
        raise CrossoverResultsError(
            "host and cell hardware identity commitments differ"
        )
    if any(
        cell["hardware_commitment"].get("public_experiment_nonce")
        != authorization.experiment_nonce
        for cell in raw_cells.values()
    ):
        raise CrossoverResultsError("hardware commitment authorization differs")
    input_identities: dict[tuple[int, str, str], set[tuple[int, str]]] = {}
    for cell in plan.schedule:
        for request in requests_by_cell[cell.cell_id]:
            key = (
                request["base_ordinal"],
                request["context_tier"],
                request["workload_id"],
            )
            input_identities.setdefault(key, set()).add(
                (request["input_token_count"], request["input_token_ids_sha256"])
            )
    if any(len(values) != 1 for values in input_identities.values()):
        raise CrossoverResultsError("prompt token identities differ across cells")
    controlled_identity = _identity_summary(
        plan.schedule, requests_by_cell, "controlled"
    )
    natural_identity = _identity_summary(plan.schedule, requests_by_cell, "natural")
    pair_records: list[dict[str, Any]] = []
    pair_curves: list[core.PairCurve] = []
    natural_terminal_effects: list[float] = []
    for lane in core.LANES:
        for pair_index in range(1, core.PAIRS_PER_LANE + 1):
            pair_cells = [
                cell
                for cell in plan.schedule
                if cell.lane == lane and cell.pair_index == pair_index
            ]
            if (
                len(pair_cells) != 2
                or abs(
                    plan.schedule.index(pair_cells[0])
                    - plan.schedule.index(pair_cells[1])
                )
                != 1
            ):
                raise CrossoverResultsError("lifecycle pair is not adjacent")
            by_mode = {cell.mode: cell for cell in pair_cells}
            eager = public_cells[by_mode["eager"].cell_id]
            compiled = public_cells[by_mode["compiled"].cell_id]
            difference = [
                compiled_value - eager_value
                for eager_value, compiled_value in zip(
                    eager["cumulative_seconds"],
                    compiled["cumulative_seconds"],
                    strict=True,
                )
            ]
            record = {
                "pair_id": pair_cells[0].pair_id,
                "pair_index": pair_index,
                "lane": lane,
                "order": pair_cells[0].order,
                "cell_ids_in_execution_order": [cell.cell_id for cell in pair_cells],
                "eager": eager,
                "compiled": compiled,
                "compiled_minus_eager_seconds": difference,
                "pair_effects": _compute_pair_effects(
                    eager,
                    compiled,
                    requests_by_cell[by_mode["eager"].cell_id],
                    requests_by_cell[by_mode["compiled"].cell_id],
                ),
            }
            pair_records.append(record)
            if lane == "controlled":
                pair_curves.append(
                    core.PairCurve(
                        pair_id=pair_cells[0].pair_id,
                        order=pair_cells[0].order,
                        eager_cumulative=tuple(eager["cumulative_seconds"]),
                        compiled_cumulative=tuple(compiled["cumulative_seconds"]),
                    )
                )
            else:
                natural_terminal_effects.append(difference[-1])
    natural_evaluations: list[dict[str, Any]] = []
    public_requests: list[dict[str, Any]] = []
    for cell in plan.schedule:
        for request in requests_by_cell[cell.cell_id]:
            if cell.lane == "natural":
                evaluation = _natural_evaluation(request)
                natural_evaluations.append(evaluation)
                request = {**request, "correctness": evaluation}
            public_requests.append(request)
    all_natural_correct = all(item["success"] for item in natural_evaluations)
    quality_preservation = _quality_preservation(plan, natural_evaluations)
    analysis = _analysis_document(
        pair_curves,
        natural_identity=natural_identity["all_corresponding_outputs_identical"],
        natural_terminal_effects=natural_terminal_effects,
        pair_records=pair_records,
    )
    claims = _claim_matrix_document(
        analysis=analysis,
        controlled_identity=controlled_identity,
        natural_identity=natural_identity,
        natural_all_correct=all_natural_correct,
        quality_preservation=quality_preservation,
        component_observability=_component_observability(pair_records),
    )
    active_operation_seconds = sum(
        entry["actual_seconds"] for entry in ledger["entries"]
    )
    active_operation_equivalent = (
        Decimal(active_operation_seconds)
        * core.ANTICIPATED_RATE_USD_PER_HOUR
        / Decimal(3600)
    )
    absent_provider_end = "external_provider_end_receipt_absent"
    return {
        "plan": plan,
        "authorization": authorization,
        "ledger": ledger,
        "orchestration": orchestration,
        "pair_records": pair_records,
        "public_requests": public_requests,
        "analysis": analysis,
        "correctness": {
            "schema_version": RESULT_SCHEMA_VERSION,
            "evaluator": "evaluate_workload",
            "natural_all_correct": all_natural_correct,
            "natural_output_identity": natural_identity,
            "controlled_output_identity": controlled_identity,
            "quality_preservation": quality_preservation,
            "evaluations": natural_evaluations,
        },
        "claims": claims,
        "budget": {
            "schema_version": RESULT_SCHEMA_VERSION,
            "hard_cap_usd": "3",
            "reserved_usd": ledger["reserved_usd"],
            "reservations_within_hard_cap": Decimal(ledger["reserved_usd"])
            <= core.HARD_CAP_USD,
            "active_operation_seconds": active_operation_seconds,
            "active_operation_list_rate_usd_per_hour": core.canonical_decimal(
                core.ANTICIPATED_RATE_USD_PER_HOUR
            ),
            "active_operation_list_rate_equivalent_usd": core.canonical_decimal(
                active_operation_equivalent
            ),
            "active_operation_equivalent_within_hard_cap": (
                active_operation_equivalent <= core.HARD_CAP_USD
            ),
            "provider_billed_seconds": None,
            "provider_billed_seconds_null_reason": absent_provider_end,
            "provider_reported_spend_usd": None,
            "provider_reported_spend_null_reason": absent_provider_end,
            "provider_list_rate_cost_usd": None,
            "provider_list_rate_cost_null_reason": absent_provider_end,
            "actual_cost_usd": None,
            "actual_cost_null_reason": absent_provider_end,
            "automatic_retries": 0,
            "all_lifecycles_completed": True,
            "local_cleanup": "complete",
            "host_shutdown_observed_at": orchestration["host_shutdown_observed_at"],
            "host_shutdown_observed_null_reason": orchestration[
                "host_shutdown_observed_null_reason"
            ],
            "external_provider_console_confirmation": orchestration[
                "external_provider_console_confirmation"
            ],
            "external_provider_console_confirmation_null_reason": orchestration[
                "external_provider_console_confirmation_null_reason"
            ],
            "independently_verified_provider_termination": orchestration[
                "independently_verified_provider_termination"
            ],
            "independently_verified_provider_termination_null_reason": orchestration[
                "independently_verified_provider_termination_null_reason"
            ],
            "provider_teardown": None,
            "provider_teardown_null_reason": orchestration[
                "provider_teardown_null_reason"
            ],
            "provider_teardown_provenance": "external_fact_not_inferred",
        },
        "hardware_summary": hardware_summary,
    }


def _protocol_document(data: Mapping[str, Any]) -> dict[str, Any]:
    plan = data["plan"]
    authorization = data["authorization"]
    orchestration = data["orchestration"]
    plan_document = plan.to_dict()
    quality_preservation = dict(plan_document["quality_preservation"])
    quality_preservation["executed_resamples"] = BOOTSTRAP_RESAMPLES
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": core.PROTOCOL_ID,
        "plan_sha256": plan.content_sha256,
        "schedule_sha256": plan_document["schedule_sha256"],
        "schedule": [cell.to_dict() for cell in plan.schedule],
        "execution_modes": plan_document["execution_modes"],
        "lifecycle_controls": plan_document["lifecycle_controls"],
        "measurement_contract": plan_document["measurement_contract"],
        "sample_stopping_rule": plan_document["sample_stopping_rule"],
        "reproducibility": plan_document["reproducibility"],
        "source_head": authorization.source_head,
        "runtime_image_id": authorization.runtime_image_id,
        "authorization_authentication": dict(
            orchestration["authorization_authentication"]
        ),
        "bindings_verified": {
            "authorization": True,
            "orchestration": True,
            "ledger": True,
            "cells": True,
            "progress_receipts": True,
            "operation_receipts": True,
            "hardware_observations": True,
            "source_runtime_hardware_continuity": True,
        },
        "hardware_observations": data["hardware_summary"],
        "analysis": {
            "unit": "whole_lifecycle_pair",
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "request_level_resampling": False,
            "controlled_curve_request_count": core.CONTROLLED_REQUESTS_PER_CELL,
        },
        "quality_preservation": quality_preservation,
    }


def _provenance_document(data: Mapping[str, Any]) -> dict[str, Any]:
    natural = data["analysis"]["controlled"]["natural_timing"]
    request = data["public_requests"][0]
    compiled_cell = next(
        pair["compiled"]
        for pair in data["pair_records"]
        if pair["lane"] == "controlled"
    )

    def measurement_row(
        field: str,
        measurement: Mapping[str, Any],
    ) -> dict[str, Any]:
        observed = measurement["observability_state"] == "observed"
        return {
            "field": field,
            "provenance": measurement["provenance"],
            "value_state": "observed" if observed else "null",
            "null_reason": None if observed else measurement["null_reason"],
        }

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "fields": [
            {
                "field": "provider_lifecycle",
                "provenance": "external_provider_fact_not_inferred",
                "value_state": "null",
                "null_reason": "provider_lifecycle_not_observed_by_local_orchestrator",
            },
            measurement_row(
                "host_lifecycle",
                compiled_cell["measurements"]["host_lifecycle_seconds"],
            ),
            {
                "field": "process_lifecycle",
                "provenance": "not_exposed",
                "value_state": "null",
                "null_reason": "process_boundary_not_separately_retained",
            },
            measurement_row("process_tree", compiled_cell["process_tree"]),
            measurement_row(
                "model_initialization",
                compiled_cell["measurements"]["initialization_seconds"],
            ),
            measurement_row(
                "compile_time",
                compiled_cell["compile_component_measurements"][
                    "compilation_time_seconds"
                ],
            ),
            measurement_row(
                "cuda_graph_capture",
                compiled_cell["compile_component_measurements"][
                    "cuda_graph_capture_duration_seconds"
                ],
            ),
            measurement_row(
                "cuda_graph_dispatch_counter",
                compiled_cell["compile_component_measurements"][
                    "cuda_graph_dispatch_counter"
                ],
            ),
            measurement_row("prefill", request["metrics"]["prefill_seconds"]),
            measurement_row("decode", request["metrics"]["decode_seconds"]),
            measurement_row(
                "per_output_token",
                request["metrics"]["mean_time_per_output_token_seconds"],
            ),
            measurement_row("ttft", request["metrics"]["ttft_seconds"]),
            measurement_row("terminal_latency", request["timing"]["latency_seconds"]),
            measurement_row(
                "output_rate",
                request["timing"]["output_token_rate_tokens_per_second"],
            ),
            measurement_row(
                "sampled_gpu_memory_peak",
                compiled_cell["measurements"]["peak_gpu_memory_mib"],
            ),
            measurement_row(
                "sampled_gpu_memory_series",
                compiled_cell["measurements"]["gpu_memory_series"],
            ),
            {
                "field": "correctness",
                "provenance": "evaluate_workload",
                "value_state": "observed",
                "null_reason": None,
            },
            {
                "field": "natural_causal_timing",
                "provenance": "derived",
                "value_state": (
                    "observed" if natural["causal_claim_eligible"] else "null"
                ),
                "null_reason": natural["causal_claim_blocker"],
            },
            {
                "field": "budget_reservations",
                "provenance": "sealed_preregistered_lifecycle_ledger",
                "value_state": "observed",
                "null_reason": None,
            },
            {
                "field": "active_operation_list_rate_equivalent",
                "provenance": "derived_from_completed_ledger_seconds",
                "value_state": "observed",
                "null_reason": None,
            },
            {
                "field": "provider_billed_seconds",
                "provenance": "external_provider_fact_not_inferred",
                "value_state": "null",
                "null_reason": data["budget"]["provider_billed_seconds_null_reason"],
            },
            {
                "field": "provider_reported_spend",
                "provenance": "external_provider_fact_not_inferred",
                "value_state": "null",
                "null_reason": data["budget"]["provider_reported_spend_null_reason"],
            },
            {
                "field": "provider_list_rate_cost",
                "provenance": "external_provider_fact_not_inferred",
                "value_state": "null",
                "null_reason": data["budget"]["provider_list_rate_cost_null_reason"],
            },
            {
                "field": "actual_cost",
                "provenance": "external_provider_fact_not_inferred",
                "value_state": "null",
                "null_reason": data["budget"]["actual_cost_null_reason"],
            },
            {
                "field": "host_shutdown_observed_at",
                "provenance": "external_host_lifecycle_fact_not_inferred",
                "value_state": "null",
                "null_reason": data["budget"]["host_shutdown_observed_null_reason"],
            },
            {
                "field": "external_provider_console_confirmation",
                "provenance": "external_operator_fact_not_inferred",
                "value_state": "null",
                "null_reason": data["budget"][
                    "external_provider_console_confirmation_null_reason"
                ],
            },
            {
                "field": "independently_verified_provider_termination",
                "provenance": "external_provider_receipt_not_inferred",
                "value_state": "null",
                "null_reason": data["budget"][
                    "independently_verified_provider_termination_null_reason"
                ],
            },
            {
                "field": "provider_teardown",
                "provenance": "external_provider_fact_not_inferred",
                "value_state": "null",
                "null_reason": data["budget"]["provider_teardown_null_reason"],
            },
        ],
    }


def _render_svg(analysis: Mapping[str, Any]) -> str:
    controlled = analysis["controlled"]
    curve = controlled["mean_difference_curve"]
    low = controlled["simultaneous_band_lower"]
    high = controlled["simultaneous_band_upper"]
    values = [*curve, *low, *high, 0.0]
    minimum, maximum = min(values), max(values)
    span = maximum - minimum or 1.0

    def point(index: int, value: float) -> str:
        x = 36 + index * (728 / (len(curve) - 1))
        y = 220 - (value - minimum) * 180 / span
        return f"{x:.2f},{y:.2f}"

    mean_points = " ".join(point(i, value) for i, value in enumerate(curve))
    upper_points = " ".join(point(i, value) for i, value in enumerate(high))
    lower_points = " ".join(
        point(i, value) for i, value in reversed(list(enumerate(low)))
    )
    zero_y = 220 - (0.0 - minimum) * 180 / span
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="800" height="270" viewBox="0 0 800 270" role="img" aria-labelledby="title desc">
<title id="title">Controlled cumulative crossover curve</title>
<desc id="desc">Mean compiled minus eager cumulative time with a whole-pair bootstrap simultaneous band.</desc>
<rect width="800" height="270" fill="white"/>
<line x1="36" y1="{zero_y:.2f}" x2="764" y2="{zero_y:.2f}" stroke="#667085"/>
<polygon points="{upper_points} {lower_points}" fill="#d1e9ff"/>
<polyline points="{mean_points}" fill="none" stroke="#175cd3" stroke-width="2"/>
<text x="36" y="252" font-family="system-ui" font-size="13">request 1</text>
<text x="680" y="252" font-family="system-ui" font-size="13">request {len(curve)}</text>
</svg>
"""


def _render_report(data: Mapping[str, Any]) -> str:
    controlled = data["analysis"]["controlled"]
    natural_timing = controlled["natural_timing"]
    quality = data["correctness"]["quality_preservation"]
    claims = data["claims"]["claims"]
    rows = "".join(
        "<tr>"
        f"<td>{html.escape(claim['claim_id'])}</td>"
        f"<td>{html.escape(claim['state'])}</td>"
        f"<td>{html.escape(', '.join(claim['blockers']) or 'none')}</td>"
        "</tr>"
        for claim in claims
    )
    sustained = controlled["supported_sustained_crossing"]
    crossing = (
        str(sustained["request_count"])
        if sustained["state"] == "observed"
        else (
            f"unsupported ({sustained['reason']})"
            if sustained["state"] == "unsupported"
            else f"{sustained['state']} at {sustained['lower_bound']}"
        )
    )
    if quality["inference_state"] == "whole_pair_bootstrap":
        quality_summary = (
            f"paired effect: {quality['mean_pair_effect']:.6f}; "
            "95% whole-pair percentile interval "
            f"[{quality['lower_confidence_endpoint']:.6f}, "
            f"{quality['upper_confidence_endpoint']:.6f}]"
        )
    else:
        quality_summary = (
            f"deterministic identical pair effect: "
            f"{quality['mean_pair_effect']:.6f}; confidence interval not applicable"
        )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Completed vLLM crossover result</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#182230}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #d0d5dd;padding:.5rem;text-align:left}}</style></head>
<body><h1>Completed Qwen3-8B vLLM crossover</h1>
<p>Eight adjacent eager/compiled lifecycle pairs per lane were completed (16 total). The controlled analysis resampled {BOOTSTRAP_RESAMPLES:,} whole pairs and never requests. With only 8 pairs per lane, simultaneous and percentile intervals may under-cover.</p>
<p>Supported sustained crossing: {html.escape(crossing)}. Basis: {html.escape(controlled["supported_crossing_basis"])}. Terminal sign-symmetry p-value: {controlled["terminal_effect_sign_flip_p_value"]:.6f}.</p>
<p>Natural timing mean compiled-minus-eager terminal effect: {natural_timing["mean_terminal_compiled_minus_eager_seconds"]:.6f} seconds; 95% whole-pair percentile interval [{natural_timing["lower_confidence_endpoint_seconds"]:.6f}, {natural_timing["upper_confidence_endpoint_seconds"]:.6f}]. Causal eligibility: {str(natural_timing["causal_claim_eligible"]).lower()}; causal speedup supported: {str(natural_timing["speedup_supported"]).lower()}.</p>
<p>Natural quality {quality_summary}. Noninferiority supported: {str(quality["noninferiority_supported"]).lower()}.</p>
<img src="crossover.svg" alt="Controlled cumulative crossover curve">
<h2>Claim matrix</h2><table><thead><tr><th>Claim</th><th>State</th><th>Blockers</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>
"""


VERIFIER = r'''"""Self-contained verifier for a completed crossover evidence bundle."""
import hashlib, html, json, math, random, re, sys
from decimal import Decimal
from pathlib import Path

BOOTSTRAP_RESAMPLES = __BOOTSTRAP_RESAMPLES__
CONTROLLED_SIGN_SYMMETRY_ALPHA = __CONTROLLED_SIGN_SYMMETRY_ALPHA__
CLAIM_REQUIREMENTS = __CLAIM_REQUIREMENTS__
CANONICAL_SCHEDULE = __CANONICAL_SCHEDULE__
SAMPLE_STOPPING_RULE = __SAMPLE_STOPPING_RULE__
SUPPORTED_CROSSING_BASIS = "simultaneous_upper_band_sustained_crossing_observed_and_terminal_effect_sign_symmetry_p_value_at_most_0.05"
FILES = {"SHA256SUMS","analysis.json","budget-teardown.json","claim-matrix.json","correctness.json","crossover.svg","evidence_bundle.py","lifecycle-pairs.json","protocol.json","provenance-null-matrix.json","report.html","request-records.jsonl"}
HASHED = sorted(FILES-{"SHA256SUMS"})
JSON_FILES = {"analysis.json","budget-teardown.json","claim-matrix.json","correctness.json","lifecycle-pairs.json","protocol.json","provenance-null-matrix.json"}
PRIVATE = [re.compile("/"+"Users/|/"+r"home/|[A-Za-z]:\\"+"Users"+r"\\"),re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),re.compile(r"\b(?!127\.0\.0\.1\b)(?:\d{1,3}\.){3}\d{1,3}\b"),re.compile(r"\bGPU-[0-9a-f-]{16,}\b",re.I),re.compile(r"\bgpu_(?:uuid|identity)(?:_sha256|_commitment)?\b",re.I),re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),re.compile(r"\b(?:gh[pousr]_|github_pat_|sk-|hf[_-]|AKIA)[A-Za-z0-9_-]{8,}\b"),re.compile(r'"(?:host(?:name)?|user(?:name)?|port|experiment_nonce)"\s*:')]
def canonical(v):
    return json.dumps(v,indent=2,sort_keys=True,ensure_ascii=True,allow_nan=False)+"\n"
def load(path):
    text=path.read_text(encoding="utf-8")
    value=json.loads(text,parse_constant=lambda x:(_ for _ in ()).throw(ValueError("nonfinite")))
    if not isinstance(value,dict) or text!=canonical(value): raise ValueError(path.name+" is not canonical JSON")
    return value
def evaluate(item):
    text=item["decoded_output"]
    workload=item["workload_id"]
    if workload=="structured-json-profile-extraction":
        stripped=text.strip()
        try:
            try: payload=json.loads(stripped)
            except json.JSONDecodeError:
                start=stripped.find("{"); end=stripped.rfind("}")
                if start==-1 or end==-1 or end<=start: raise json.JSONDecodeError("no JSON object found",stripped,0)
                payload=json.loads(stripped[start:end+1])
        except json.JSONDecodeError as exc:
            success=False; score=0.0; metric="structured_json_exact_field_match"; notes="response did not contain a parseable JSON object: "+str(exc)
        else:
            if not isinstance(payload,dict):
                success=False; score=0.0; metric="structured_json_exact_field_match"; notes="parsed JSON root must be an object, got "+type(payload).__name__
            else:
                fields=(("name","str",str),("age","int",int),("is_active","bool",bool))
                problems=[]; matched=0
                for name,type_name,type_check in fields:
                    if name not in payload:
                        problems.append("missing field '"+name+"'"); continue
                    value=payload[name]
                    if type_name in ("int","float") and isinstance(value,bool):
                        problems.append("field '"+name+"' is bool, expected "+type_name); continue
                    if not isinstance(value,type_check):
                        problems.append("field '"+name+"' is "+type(value).__name__+", expected "+type_name); continue
                    matched+=1
                score=matched/len(fields); success=matched==len(fields); metric="structured_json_exact_field_match"; notes=None if success else "; ".join(problems)[:500]
    elif workload=="prose-reasoning-two-train-gap":
        success=re.search(r"^\s*3(?:\.0+)?(?=\s|$)",text,re.IGNORECASE) is not None
        score=1.0 if success else 0.0; metric="exact_answer_pattern_match"; notes=None if success else "expected answer pattern not found in response"
    else:
        raise ValueError("unexpected natural evaluator")
    return {"cell_id":item["cell_id"],"request_sequence_index":item["request_sequence_index"],"success":success,"quality_score":score,"quality_metric":metric,"notes":notes,"evaluator":"evaluate_workload","workload_version":item["workload_version"],"output_token_ids_sha256":item["output_token_ids_sha256"]}
def first_crossing(curve):
    return next((index for index,value in enumerate(curve,1) if value<=0),None)
def sustained_crossing(curve):
    suffix=True; earliest=None
    for reverse_index,value in enumerate(reversed(curve),1):
        if value>0: suffix=False
        elif suffix: earliest=len(curve)-reverse_index+1
        else: suffix=False
    return earliest
def mean_curve(curves):
    totals=[0.0]*len(curves[0])
    for curve in curves:
        for index,value in enumerate(curve): totals[index]+=value
    return [value/float(len(curves)) for value in totals]
def quantile(values,q):
    return sorted(values)[max(0,min(len(values)-1,math.ceil(q*len(values))-1))]
def median(values):
    ordered=sorted(values); middle=len(ordered)//2
    return ordered[middle] if len(ordered)%2 else (ordered[middle-1]+ordered[middle])/2
def endpoint(value,limit,sustained=False):
    return {"state":"observed","request_count":value,"lower_bound":None} if value is not None else {"state":"right_censored" if sustained else "open","request_count":None,"lower_bound":limit}
def interval_endpoint(value,is_open,limit):
    if is_open: return {"state":"open","request_count":None,"lower_bound":limit,"censor_sentinel_request_count":value or limit+1}
    return {"state":"observed","request_count":value,"lower_bound":None,"censor_sentinel_request_count":None}
def effect_distributions(pair_records):
    units={"initialization":"seconds","host_lifecycle":"seconds","request_phase":"seconds","cumulative_init_to_terminal":"seconds","mean_ttft":"seconds","mean_prefill":"seconds","mean_decode":"seconds","mean_output_rate":"tokens_per_second","peak_gpu_memory":"MiB"}
    provenance={"initialization":"compiled_minus_eager_cell_initialization_perf_counter","host_lifecycle":"compiled_minus_eager_host_operation_receipt_duration_ns","request_phase":"compiled_minus_eager_sum_request_terminal_latency","cumulative_init_to_terminal":"compiled_minus_eager_terminal_cumulative_from_initialization","mean_ttft":"compiled_minus_eager_mean_request_ttft","mean_prefill":"compiled_minus_eager_mean_request_prefill","mean_decode":"compiled_minus_eager_mean_request_decode","mean_output_rate":"compiled_minus_eager_mean_exact_output_token_rate","peak_gpu_memory":"compiled_minus_eager_sampled_nvidia_smi_peak"}
    lanes={}
    for lane in ("natural","controlled"):
        lane_pairs=[pair for pair in pair_records if pair["lane"]==lane]; metrics={}
        for metric in units:
            records=[{"pair_id":pair["pair_id"],"value":pair["pair_effects"][metric]["value"],"null_reason":pair["pair_effects"][metric]["null_reason"]} for pair in lane_pairs]
            observed=[float(record["value"]) for record in records if record["value"] is not None]
            if observed:
                ordered=sorted(observed); midpoint=len(ordered)//2; lower=ordered[:midpoint]; upper=ordered[(len(ordered)+1)//2:]; q1=median(lower) if lower else ordered[0]; q3=median(upper) if upper else ordered[-1]
                mean=sum(observed)/len(observed); med=median(observed); range_value={"minimum":ordered[0],"maximum":ordered[-1]}; iqr={"first_quartile":q1,"third_quartile":q3,"width":q3-q1}; reason=None
            else:
                mean=med=None; range_value={"minimum":None,"maximum":None}; iqr={"first_quartile":None,"third_quartile":None,"width":None}; reason="no_observed_pair_effects"
            metrics[metric]={"unit":units[metric],"provenance":provenance[metric],"pair_count":len(records),"observed_effect_count":len(observed),"effects":records,"mean":mean,"median":med,"iqr":iqr,"range":range_value,"summary_null_reason":reason}
        lanes[lane]=metrics
    return {"unit_of_analysis":"lifecycle_pair","request_level_resampling":False,"summary_method":"descriptive_exclusive_halves_moore_mccabe_over_observed_pair_effects","lanes":lanes}
def identity(schedule,requests_by_cell,lane):
    cells=[cell for cell in schedule if cell["lane"]==lane]; count=cells[0]["requests_per_cell"]; mismatches=[]; within=True; cross=True
    for index in range(count):
        by_mode={"eager":[],"compiled":[]}
        for cell in cells: by_mode[cell["mode"]].append(tuple(requests_by_cell[cell["cell_id"]][index]["output_token_ids"]))
        if len(set(by_mode["eager"]))!=1: within=False; mismatches.append({"request_sequence_index":index+1,"scope":"eager_lifecycles"})
        if len(set(by_mode["compiled"]))!=1: within=False; mismatches.append({"request_sequence_index":index+1,"scope":"compiled_lifecycles"})
        matches=[]
        for pair_index in range(1,9):
            pair=[cell for cell in cells if cell["pair_index"]==pair_index]; modes={cell["mode"]:cell for cell in pair}
            matches.append(requests_by_cell[modes["eager"]["cell_id"]][index]["output_token_ids"]==requests_by_cell[modes["compiled"]["cell_id"]][index]["output_token_ids"])
        if not all(matches): cross=False; mismatches.append({"request_sequence_index":index+1,"scope":"modes"})
    return {"cross_mode_pair_outputs_identical":cross,"within_mode_lifecycles_identical":within,"all_corresponding_outputs_identical":cross and within,"mismatches":mismatches}
def full_analysis(pair_records,resamples,natural_identical):
    controlled=[pair for pair in pair_records if pair["lane"]=="controlled"]; curves=[]
    for pair in controlled:
        difference=[compiled-eager for eager,compiled in zip(pair["eager"]["cumulative_seconds"],pair["compiled"]["cumulative_seconds"])]
        if pair["compiled_minus_eager_seconds"]!=difference: raise ValueError("pair difference curve differs")
        curves.append(difference)
    observed=mean_curve(curves); rng=random.Random(20260905); deviations=[]; sustained_values=[]; censored=0
    for _ in range(resamples):
        sampled=[curves[rng.randrange(8)] for _ in range(8)]; sampled_mean=mean_curve(sampled); deviations.append(max(abs(sampled_mean[index]-observed[index]) for index in range(144))); crossing=sustained_crossing(sampled_mean)
        if crossing is None: censored+=1; sustained_values.append(145)
        else: sustained_values.append(crossing)
    radius=quantile(deviations,.95); lower=[value-radius for value in observed]; upper=[value+radius for value in observed]; crossing_median=quantile(sustained_values,.5); crossing_lower=quantile(sustained_values,.025); crossing_upper=quantile(sustained_values,.975)
    core_effects=[]
    for pair,curve in zip(controlled,curves):
        first=first_crossing(curve); sustained=sustained_crossing(curve)
        core_effects.append({"pair_id":pair["pair_id"],"order":pair["order"],"first_crossing_request_count":first,"sustained_crossing_request_count":sustained,"right_censored":sustained is None,"terminal_effect_seconds":curve[-1],"mean_effect_seconds":sum(curve)/len(curve),"minimum_effect_seconds":min(curve),"maximum_effect_seconds":max(curve)})
    terminals=[effect["terminal_effect_seconds"] for effect in core_effects]; observed_terminal=abs(sum(terminals)/len(terminals)); extreme=0
    for mask in range(1<<len(terminals)):
        total=sum((-1.0 if (mask>>index)&1 else 1.0)*effect for index,effect in enumerate(terminals))
        if abs(total/len(terminals))>=observed_terminal-1e-12: extreme+=1
    aggregate_first=first_crossing(observed); aggregate_sustained=sustained_crossing(observed); band_first=first_crossing(upper); band_sustained=sustained_crossing(upper)
    result={"protocol_id":"qwen3-8b-vllm-crossover-v2","analysis_seed":20260905,"resample_count":resamples,"pair_effects":core_effects,"mean_difference_curve":observed,"simultaneous_band_lower":lower,"simultaneous_band_upper":upper,"aggregate_first_crossing_request_count":aggregate_first,"aggregate_sustained_crossing_request_count":aggregate_sustained,"simultaneous_band_first_crossing_request_count":band_first,"simultaneous_band_sustained_crossing_request_count":band_sustained,"bootstrap_uncensored_resamples":resamples-censored,"bootstrap_censored_resamples":censored,"bootstrap_sustained_crossing_median_request_count":crossing_median if crossing_median<=144 else None,"bootstrap_sustained_crossing_lower_request_count":crossing_lower if crossing_lower<=144 else None,"bootstrap_sustained_crossing_upper_request_count":crossing_upper if crossing_upper<=144 else None,"bootstrap_sustained_crossing_lower_is_open":crossing_lower>144,"bootstrap_sustained_crossing_upper_is_open":crossing_upper>144,"terminal_effect_sign_flip_p_value":extreme/float(1<<len(terminals))}
    result["aggregate_first_crossing"]=endpoint(aggregate_first,144); result["aggregate_sustained_crossing"]=endpoint(aggregate_sustained,144,True); result["simultaneous_band_first_crossing"]=endpoint(band_first,144); result["simultaneous_band_sustained_crossing"]=endpoint(band_sustained,144,True); result["supported_first_crossing"]=dict(result["simultaneous_band_first_crossing"]); supported_gate=band_sustained is not None and result["terminal_effect_sign_flip_p_value"]<=CONTROLLED_SIGN_SYMMETRY_ALPHA; result["supported_crossing_gate_satisfied"]=supported_gate; result["supported_sustained_crossing"]=dict(result["simultaneous_band_sustained_crossing"]) if supported_gate else {"state":"unsupported","request_count":None,"lower_bound":None,"reason":"combined_band_crossing_and_sign_symmetry_gate_not_satisfied"}; result["supported_crossing_basis"]=SUPPORTED_CROSSING_BASIS
    median_open=crossing_median>144; lower_open=crossing_lower>144; upper_open=crossing_upper>144
    result["bootstrap_sustained_crossing_interval"]={"state":"open" if lower_open and upper_open else ("partially_open" if lower_open or upper_open else "observed"),"lower":interval_endpoint(None if lower_open else crossing_lower,lower_open,144),"median":interval_endpoint(None if median_open else crossing_median,median_open,144),"upper":interval_endpoint(None if upper_open else crossing_upper,upper_open,144),"censor_at_request_count":144,"censor_sentinel_request_count":145}
    result["bootstrap_unit"]="whole_lifecycle_pair"; result["request_level_resampling"]=False
    natural_effects=[pair["compiled_minus_eager_seconds"][-1] for pair in pair_records if pair["lane"]=="natural"]
    natural_rng=random.Random(20260905); natural_bootstrap=[sum(natural_effects[natural_rng.randrange(len(natural_effects))] for _ in natural_effects)/len(natural_effects) for _ in range(BOOTSTRAP_RESAMPLES)]; natural_mean=sum(natural_effects)/len(natural_effects); natural_lower=quantile(natural_bootstrap,.025); natural_upper=quantile(natural_bootstrap,.975); natural_supported=natural_identical and natural_mean<0 and natural_upper<=0
    result["natural_timing"]={"causal_claim_eligible":natural_identical,"pair_terminal_compiled_minus_eager_seconds":natural_effects,"mean_terminal_compiled_minus_eager_seconds":natural_mean,"analysis_seed":20260905,"resample_count":BOOTSTRAP_RESAMPLES,"bootstrap_unit":"whole_lifecycle_pair","request_level_resampling":False,"lower_confidence_endpoint_seconds":natural_lower,"upper_confidence_endpoint_seconds":natural_upper,"speedup_supported":natural_supported,"causal_claim_blocker":None if natural_identical else "natural_outputs_differ_across_modes_or_lifecycles"}
    return {"schema_version":"1","controlled":result,"pair_effect_distributions":effect_distributions(pair_records)}
def claim_matrix(analysis,controlled_identity,natural_identity,natural_absolute_correctness,component_observable):
    controlled=analysis["controlled"]
    flags={"terminal":True,"completeness":True,"fixed_count":True,"controlled_supported_crossing":controlled["supported_crossing_gate_satisfied"],"controlled_output_identity":controlled_identity["cross_mode_pair_outputs_identical"],"controlled_numeric_reproducibility":controlled_identity["within_mode_lifecycles_identical"],"natural_output_identity":natural_identity["cross_mode_pair_outputs_identical"],"natural_numeric_reproducibility":natural_identity["within_mode_lifecycles_identical"],"natural_supported_speedup":analysis["controlled"]["natural_timing"]["speedup_supported"],"natural_absolute_correctness":natural_absolute_correctness,"component_observability":component_observable}
    requirements=CLAIM_REQUIREMENTS.items()
    claims=[]
    for claim_id,required in requirements:
        blockers=[name for name in required if not flags[name]]; claims.append({"claim_id":claim_id,"state":"supported" if not blockers else "unsupported","blockers":blockers})
    claims.append({"claim_id":"forward-pass-identical","state":"not_applicable","blockers":["forward_pass_identity_is_out_of_scope"]})
    claims.extend(({"claim_id":"budget-reservations-within-hard-cap","state":"supported","blockers":[]},{"claim_id":"active-operation-list-rate-equivalent-within-hard-cap","state":"supported","blockers":[]},{"claim_id":"provider-billed-cost-within-hard-cap","state":"unsupported","blockers":["external_provider_end_receipt_absent"]},{"claim_id":"provider-teardown","state":"unsupported","blockers":["external_provider_fact_not_observed"]}))
    return {"schema_version":"1","claims":claims}
def render_svg(analysis):
    controlled=analysis["controlled"]; curve=controlled["mean_difference_curve"]; low=controlled["simultaneous_band_lower"]; high=controlled["simultaneous_band_upper"]; values=[*curve,*low,*high,0.0]; minimum,maximum=min(values),max(values); span=maximum-minimum or 1.0
    def point(index,value):
        return f"{36+index*(728/(len(curve)-1)):.2f},{220-(value-minimum)*180/span:.2f}"
    mean_points=" ".join(point(i,value) for i,value in enumerate(curve)); upper_points=" ".join(point(i,value) for i,value in enumerate(high)); lower_points=" ".join(point(i,value) for i,value in reversed(list(enumerate(low)))); zero_y=220-(0.0-minimum)*180/span
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="800" height="270" viewBox="0 0 800 270" role="img" aria-labelledby="title desc">
<title id="title">Controlled cumulative crossover curve</title>
<desc id="desc">Mean compiled minus eager cumulative time with a whole-pair bootstrap simultaneous band.</desc>
<rect width="800" height="270" fill="white"/>
<line x1="36" y1="{zero_y:.2f}" x2="764" y2="{zero_y:.2f}" stroke="#667085"/>
<polygon points="{upper_points} {lower_points}" fill="#d1e9ff"/>
<polyline points="{mean_points}" fill="none" stroke="#175cd3" stroke-width="2"/>
<text x="36" y="252" font-family="system-ui" font-size="13">request 1</text>
<text x="680" y="252" font-family="system-ui" font-size="13">request {len(curve)}</text>
</svg>
"""
def render_report(analysis,correctness,claim_document):
    controlled=analysis["controlled"]; natural_timing=controlled["natural_timing"]; quality=correctness["quality_preservation"]; claims=claim_document["claims"]
    rows="".join("<tr>"+f"<td>{html.escape(claim['claim_id'])}</td>"+f"<td>{html.escape(claim['state'])}</td>"+f"<td>{html.escape(', '.join(claim['blockers']) or 'none')}</td>"+"</tr>" for claim in claims)
    sustained=controlled["supported_sustained_crossing"]
    crossing=str(sustained["request_count"]) if sustained["state"]=="observed" else (f"unsupported ({sustained['reason']})" if sustained["state"]=="unsupported" else f"{sustained['state']} at {sustained['lower_bound']}")
    if quality["inference_state"]=="whole_pair_bootstrap":
        quality_summary=f"paired effect: {quality['mean_pair_effect']:.6f}; 95% whole-pair percentile interval [{quality['lower_confidence_endpoint']:.6f}, {quality['upper_confidence_endpoint']:.6f}]"
    else:
        quality_summary=f"deterministic identical pair effect: {quality['mean_pair_effect']:.6f}; confidence interval not applicable"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Completed vLLM crossover result</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#182230}}table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #d0d5dd;padding:.5rem;text-align:left}}</style></head>
<body><h1>Completed Qwen3-8B vLLM crossover</h1>
<p>Eight adjacent eager/compiled lifecycle pairs per lane were completed (16 total). The controlled analysis resampled {BOOTSTRAP_RESAMPLES:,} whole pairs and never requests. With only 8 pairs per lane, simultaneous and percentile intervals may under-cover.</p>
<p>Supported sustained crossing: {html.escape(crossing)}. Basis: {html.escape(controlled["supported_crossing_basis"])}. Terminal sign-symmetry p-value: {controlled["terminal_effect_sign_flip_p_value"]:.6f}.</p>
<p>Natural timing mean compiled-minus-eager terminal effect: {natural_timing["mean_terminal_compiled_minus_eager_seconds"]:.6f} seconds; 95% whole-pair percentile interval [{natural_timing["lower_confidence_endpoint_seconds"]:.6f}, {natural_timing["upper_confidence_endpoint_seconds"]:.6f}]. Causal eligibility: {str(natural_timing["causal_claim_eligible"]).lower()}; causal speedup supported: {str(natural_timing["speedup_supported"]).lower()}.</p>
<p>Natural quality {quality_summary}. Noninferiority supported: {str(quality["noninferiority_supported"]).lower()}.</p>
<img src="crossover.svg" alt="Controlled cumulative crossover curve">
<h2>Claim matrix</h2><table><thead><tr><th>Claim</th><th>State</th><th>Blockers</th></tr></thead><tbody>{rows}</tbody></table>
</body></html>
"""
def provenance_document(analysis,requests,pair_records,budget):
    natural=analysis["controlled"]["natural_timing"]; request=requests[0]; compiled=next(pair["compiled"] for pair in pair_records if pair["lane"]=="controlled")
    def measurement(field,value):
        observed=value["observability_state"]=="observed"
        return {"field":field,"provenance":value["provenance"],"value_state":"observed" if observed else "null","null_reason":None if observed else value["null_reason"]}
    fields=[
        {"field":"provider_lifecycle","provenance":"external_provider_fact_not_inferred","value_state":"null","null_reason":"provider_lifecycle_not_observed_by_local_orchestrator"},
        measurement("host_lifecycle",compiled["measurements"]["host_lifecycle_seconds"]),
        {"field":"process_lifecycle","provenance":"not_exposed","value_state":"null","null_reason":"process_boundary_not_separately_retained"},
        measurement("process_tree",compiled["process_tree"]),
        measurement("model_initialization",compiled["measurements"]["initialization_seconds"]),
        measurement("compile_time",compiled["compile_component_measurements"]["compilation_time_seconds"]),
        measurement("cuda_graph_capture",compiled["compile_component_measurements"]["cuda_graph_capture_duration_seconds"]),
        measurement("cuda_graph_dispatch_counter",compiled["compile_component_measurements"]["cuda_graph_dispatch_counter"]),
        measurement("prefill",request["metrics"]["prefill_seconds"]),
        measurement("decode",request["metrics"]["decode_seconds"]),
        measurement("per_output_token",request["metrics"]["mean_time_per_output_token_seconds"]),
        measurement("ttft",request["metrics"]["ttft_seconds"]),
        measurement("terminal_latency",request["timing"]["latency_seconds"]),
        measurement("output_rate",request["timing"]["output_token_rate_tokens_per_second"]),
        measurement("sampled_gpu_memory_peak",compiled["measurements"]["peak_gpu_memory_mib"]),
        measurement("sampled_gpu_memory_series",compiled["measurements"]["gpu_memory_series"]),
        {"field":"correctness","provenance":"evaluate_workload","value_state":"observed","null_reason":None},
        {"field":"natural_causal_timing","provenance":"derived","value_state":"observed" if natural["causal_claim_eligible"] else "null","null_reason":natural["causal_claim_blocker"]},
        {"field":"budget_reservations","provenance":"sealed_preregistered_lifecycle_ledger","value_state":"observed","null_reason":None},
        {"field":"active_operation_list_rate_equivalent","provenance":"derived_from_completed_ledger_seconds","value_state":"observed","null_reason":None},
        {"field":"provider_billed_seconds","provenance":"external_provider_fact_not_inferred","value_state":"null","null_reason":budget["provider_billed_seconds_null_reason"]},
        {"field":"provider_reported_spend","provenance":"external_provider_fact_not_inferred","value_state":"null","null_reason":budget["provider_reported_spend_null_reason"]},
        {"field":"provider_list_rate_cost","provenance":"external_provider_fact_not_inferred","value_state":"null","null_reason":budget["provider_list_rate_cost_null_reason"]},
        {"field":"actual_cost","provenance":"external_provider_fact_not_inferred","value_state":"null","null_reason":budget["actual_cost_null_reason"]},
        {"field":"host_shutdown_observed_at","provenance":"external_host_lifecycle_fact_not_inferred","value_state":"null","null_reason":budget["host_shutdown_observed_null_reason"]},
        {"field":"external_provider_console_confirmation","provenance":"external_operator_fact_not_inferred","value_state":"null","null_reason":budget["external_provider_console_confirmation_null_reason"]},
        {"field":"independently_verified_provider_termination","provenance":"external_provider_receipt_not_inferred","value_state":"null","null_reason":budget["independently_verified_provider_termination_null_reason"]},
        {"field":"provider_teardown","provenance":"external_provider_fact_not_inferred","value_state":"null","null_reason":budget["provider_teardown_null_reason"]},
    ]
    return {"schema_version":"1","fields":fields}
def verify(root):
    root=root.resolve()
    if not root.is_dir() or any(p.is_symlink() or not p.is_file() or p.stat().st_size>33554432 for p in root.iterdir()): raise ValueError("unsafe bundle")
    if {p.name for p in root.iterdir()}!=FILES: raise ValueError("bundle inventory differs")
    for p in root.iterdir():
        text=p.read_text(encoding="utf-8")
        if any(pattern.search(text) for pattern in PRIVATE): raise ValueError(p.name+" contains private data")
    expected="\n".join(hashlib.sha256((root/n).read_bytes()).hexdigest()+"  "+n for n in HASHED)+"\n"
    if (root/"SHA256SUMS").read_text(encoding="utf-8")!=expected: raise ValueError("checksums differ")
    docs={n:load(root/n) for n in JSON_FILES}
    protocol=docs["protocol.json"]; pairs=docs["lifecycle-pairs.json"]; analysis=docs["analysis.json"]
    quality=docs["correctness.json"]["quality_preservation"]
    hardware=protocol["hardware_observations"]
    authentication=protocol["authorization_authentication"]; sha=re.compile(r"^sha256:[0-9a-f]{64}$")
    if protocol["schedule"]!=CANONICAL_SCHEDULE or protocol.get("sample_stopping_rule")!=SAMPLE_STOPPING_RULE or len(pairs["pairs"])!=16 or protocol["bindings_verified"]["progress_receipts"] is not True or hardware["observation_count"]!=65 or hardware["compute_capability"]!="8.9" or hardware["maximum_baseline_memory_used_mib"]>2048 or hardware["maximum_idle_temperature_c"]>80 or hardware["maximum_idle_utilization_percent"]>5 or set(authentication)!={"mechanism","namespace","signer_identity","signature_sha256","authorized_signers_sha256","verified"} or authentication["mechanism"]!="openssh_detached_signature" or authentication["namespace"]!="llmtracefx-vllm-crossover-authorization-v1" or authentication["signer_identity"]!="vllm-crossover-coordinator" or not sha.fullmatch(authentication["signature_sha256"]) or not sha.fullmatch(authentication["authorized_signers_sha256"]) or authentication["verified"] is not True: raise ValueError("protocol cardinality, authentication, or hardware evidence differs")
    if protocol["analysis"]["bootstrap_resamples"]!=BOOTSTRAP_RESAMPLES or protocol["quality_preservation"]["executed_resamples"]!=BOOTSTRAP_RESAMPLES or quality["resample_count"]!=BOOTSTRAP_RESAMPLES: raise ValueError("bootstrap execution count differs")
    requests=[]
    for line in (root/"request-records.jsonl").read_text(encoding="utf-8").splitlines():
        item=json.loads(line,parse_constant=lambda x:(_ for _ in ()).throw(ValueError("nonfinite")))
        if json.dumps(item,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False)!=line: raise ValueError("noncanonical JSONL")
        requests.append(item)
    if len(requests)!=2496 or any(r.get("terminal") is not True for r in requests): raise ValueError("request cardinality differs")
    requests_by_cell={}
    for item in requests: requests_by_cell.setdefault(item["cell_id"],[]).append(item)
    for cell_requests in requests_by_cell.values(): cell_requests.sort(key=lambda request:request["request_sequence_index"])
    effect_units={"initialization":"seconds","host_lifecycle":"seconds","request_phase":"seconds","cumulative_init_to_terminal":"seconds","mean_ttft":"seconds","mean_prefill":"seconds","mean_decode":"seconds","mean_output_rate":"tokens_per_second","peak_gpu_memory":"MiB"}
    expected_pairs=[]
    for lane in ("controlled","natural"):
        for pair_index in range(1,9):
            cells=[cell for cell in CANONICAL_SCHEDULE if cell["lane"]==lane and cell["pair_index"]==pair_index]
            if len(cells)!=2 or abs(CANONICAL_SCHEDULE.index(cells[0])-CANONICAL_SCHEDULE.index(cells[1]))!=1 or {cell["mode"] for cell in cells}!={"eager","compiled"} or len({cell["pair_id"] for cell in cells})!=1 or len({cell["order"] for cell in cells})!=1: raise ValueError("canonical pair schedule invalid")
            by_mode={cell["mode"]:cell for cell in cells}
            expected_pairs.append({"lane":lane,"pair_index":pair_index,"pair_id":cells[0]["pair_id"],"order":cells[0]["order"],"cell_ids_in_execution_order":[cell["cell_id"] for cell in cells],"eager":by_mode["eager"],"compiled":by_mode["compiled"]})
    if len(pairs["pairs"])!=len(expected_pairs) or set(requests_by_cell)!={cell["cell_id"] for cell in CANONICAL_SCHEDULE}: raise ValueError("canonical pair or request inventory differs")
    for pair,expected_pair in zip(pairs["pairs"],expected_pairs):
        if any(pair.get(field)!=expected_pair[field] for field in ("lane","pair_index","pair_id","order","cell_ids_in_execution_order")): raise ValueError("pair canonical binding differs")
        for mode in ("eager","compiled"):
            cell=pair[mode]; expected_cell=expected_pair[mode]
            if cell["cell_id"]!=expected_cell["cell_id"] or cell.get("mode")!=mode or cell.get("period_index")!=expected_cell["period_index"] or cell.get("terminal") is not True: raise ValueError("cell canonical binding differs")
            cell_requests=requests_by_cell[cell["cell_id"]]
            if len(cell_requests)!=expected_cell["requests_per_cell"] or any(request.get("cell_id")!=cell["cell_id"] or request.get("lane")!=expected_pair["lane"] or request.get("mode")!=mode or request.get("pair_id")!=expected_pair["pair_id"] for request in cell_requests): raise ValueError("request canonical role binding differs")
        eager=pair["eager"]; compiled=pair["compiled"]; eager_requests=requests_by_cell[eager["cell_id"]]; compiled_requests=requests_by_cell[compiled["cell_id"]]; effects=pair["pair_effects"]
        for cell,cell_requests in ((eager,eager_requests),(compiled,compiled_requests)):
            if [request["request_sequence_index"] for request in cell_requests]!=list(range(1,len(cell_requests)+1)): raise ValueError("request sequence differs")
            request_curve=[request["cumulative_from_initialization_seconds"] for request in cell_requests]
            if cell["cumulative_seconds"]!=request_curve or not request_curve or request_curve[0]<cell["initialization_seconds"] or any(current<=previous for previous,current in zip(request_curve,request_curve[1:])): raise ValueError("cell curve differs from requests")
        expected_difference=[compiled_value-eager_value for eager_value,compiled_value in zip(eager["cumulative_seconds"],compiled["cumulative_seconds"])]
        if pair["compiled_minus_eager_seconds"]!=expected_difference: raise ValueError("pair difference curve differs")
        if set(effects)!=set(effect_units) or any(effects[m]["unit"]!=u for m,u in effect_units.items()): raise ValueError("pair effect units differ")
        direct={"initialization":compiled["measurements"]["initialization_seconds"]["value"]-eager["measurements"]["initialization_seconds"]["value"],"host_lifecycle":compiled["measurements"]["host_lifecycle_seconds"]["value"]-eager["measurements"]["host_lifecycle_seconds"]["value"],"request_phase":sum(r["latency_seconds"] for r in compiled_requests)-sum(r["latency_seconds"] for r in eager_requests),"cumulative_init_to_terminal":compiled["cumulative_seconds"][-1]-eager["cumulative_seconds"][-1]}
        for metric,value in direct.items():
            if effects[metric]["null_reason"] is not None or not math.isclose(effects[metric]["value"],value,abs_tol=1e-12): raise ValueError("direct pair effect differs")
        for metric,request_metric in (("mean_ttft","ttft_seconds"),("mean_prefill","prefill_seconds"),("mean_decode","decode_seconds")):
            eager_values=[r["metrics"][request_metric] for r in eager_requests]; compiled_values=[r["metrics"][request_metric] for r in compiled_requests]; observed=all(v["observability_state"]=="observed" for v in eager_values+compiled_values)
            expected=(sum(v["value"] for v in compiled_values)/len(compiled_values)-sum(v["value"] for v in eager_values)/len(eager_values)) if observed else None
            if (expected is None and (effects[metric]["value"] is not None or effects[metric]["null_reason"] is None)) or (expected is not None and (effects[metric]["null_reason"] is not None or not math.isclose(effects[metric]["value"],expected,abs_tol=1e-12))): raise ValueError("request metric pair effect differs")
        eager_rates=[r["timing"]["output_token_rate_tokens_per_second"] for r in eager_requests]; compiled_rates=[r["timing"]["output_token_rate_tokens_per_second"] for r in compiled_requests]; rates_observed=all(v["observability_state"]=="observed" for v in eager_rates+compiled_rates)
        expected_rate=(sum(v["value"] for v in compiled_rates)/len(compiled_rates)-sum(v["value"] for v in eager_rates)/len(eager_rates)) if rates_observed else None
        if (expected_rate is None and (effects["mean_output_rate"]["value"] is not None or effects["mean_output_rate"]["null_reason"] is None)) or (expected_rate is not None and (effects["mean_output_rate"]["null_reason"] is not None or not math.isclose(effects["mean_output_rate"]["value"],expected_rate,abs_tol=1e-12))): raise ValueError("output rate pair effect differs")
        eager_peak=eager["measurements"]["peak_gpu_memory_mib"]; compiled_peak=compiled["measurements"]["peak_gpu_memory_mib"]; peak_observed=eager_peak["observability_state"]=="observed" and compiled_peak["observability_state"]=="observed"; expected_peak=compiled_peak["value"]-eager_peak["value"] if peak_observed else None
        if (expected_peak is None and (effects["peak_gpu_memory"]["value"] is not None or effects["peak_gpu_memory"]["null_reason"] is None)) or (expected_peak is not None and (effects["peak_gpu_memory"]["null_reason"] is not None or not math.isclose(effects["peak_gpu_memory"]["value"],expected_peak,abs_tol=1e-12))): raise ValueError("peak memory pair effect differs")
    distributions=analysis["pair_effect_distributions"]
    if distributions["request_level_resampling"] is not False: raise ValueError("pair effects must not request-bootstrap")
    for lane in ("controlled","natural"):
        for metric,unit in effect_units.items():
            summary=distributions["lanes"][lane][metric]
            if summary["pair_count"]!=8 or len(summary["effects"])!=8 or summary["unit"]!=unit: raise ValueError("pair effect distribution differs")
    controlled_identity=identity(protocol["schedule"],requests_by_cell,"controlled"); natural_identity=identity(protocol["schedule"],requests_by_cell,"natural")
    expected_analysis=full_analysis(pairs["pairs"],BOOTSTRAP_RESAMPLES,natural_identity["all_corresponding_outputs_identical"])
    if analysis!=expected_analysis: raise ValueError("complete controlled analysis differs")
    evaluations=[]
    for item in requests:
        if item["lane"]=="natural":
            expected_evaluation=evaluate(item)
            if item.get("correctness")!=expected_evaluation: raise ValueError("request correctness does not recompute")
            evaluations.append(expected_evaluation)
        elif "correctness" in item:
            raise ValueError("controlled request contains correctness")
    correctness=docs["correctness.json"]
    if correctness["evaluations"]!=evaluations or correctness["natural_all_correct"]!=all(e["success"] for e in evaluations) or correctness["controlled_output_identity"]!=controlled_identity or correctness["natural_output_identity"]!=natural_identity: raise ValueError("correctness report differs")
    controlled=analysis["controlled"]
    if controlled["resample_count"]!=BOOTSTRAP_RESAMPLES or controlled["bootstrap_unit"]!="whole_lifecycle_pair" or controlled["request_level_resampling"] is not False: raise ValueError("analysis protocol differs")
    quality_effects=quality["pair_effects"]
    if len(quality_effects)!=8 or quality["analysis_seed"]!=20260905 or quality["resample_count"]!=BOOTSTRAP_RESAMPLES or quality["bootstrap_unit"]!="whole_lifecycle_pair" or quality["request_level_resampling"] is not False or quality["noninferiority_margin"]!="0" or quality["support_threshold"]!=0.0 or not math.isclose(sum(p["compiled_minus_eager_request_success_rate"] for p in quality_effects)/8,quality["mean_pair_effect"],abs_tol=1e-12): raise ValueError("quality preservation analysis differs")
    by_cell={}
    for evaluation in evaluations: by_cell.setdefault(evaluation["cell_id"],[]).append(evaluation)
    recomputed_effects=[]; expected_quality_effects=[]
    for pair_index in range(1,9):
        cells=[c for c in protocol["schedule"] if c["lane"]=="natural" and c["pair_index"]==pair_index]
        modes={c["mode"]:c for c in cells}
        eager=by_cell[modes["eager"]["cell_id"]]; compiled=by_cell[modes["compiled"]["cell_id"]]
        eager_rate=sum(e["success"] for e in eager)/12; compiled_rate=sum(e["success"] for e in compiled)/12
        effect=compiled_rate-eager_rate
        recomputed_effects.append(effect)
        expected_quality_effects.append({"pair_id":cells[0]["pair_id"],"pair_index":pair_index,"eager_request_success_rate":eager_rate,"compiled_request_success_rate":compiled_rate,"compiled_minus_eager_request_success_rate":effect})
        observed=quality_effects[pair_index-1]
        if not math.isclose(eager_rate,observed["eager_request_success_rate"],abs_tol=1e-12) or not math.isclose(compiled_rate,observed["compiled_request_success_rate"],abs_tol=1e-12) or not math.isclose(effect,observed["compiled_minus_eager_request_success_rate"],abs_tol=1e-12): raise ValueError("quality pair effect differs")
    rng=random.Random(20260905); boot=[sum(recomputed_effects[rng.randrange(8)] for _ in range(8))/8 for _ in range(BOOTSTRAP_RESAMPLES)]
    if len(set(recomputed_effects))==1:
        expected_supported=recomputed_effects[0]>=0.0; expected_state="deterministic_complete_agreement" if expected_supported else "deterministic_noninferiority_failed"
        if quality["confidence_method"]!="not_applicable_deterministic_pair_effects" or quality["confidence_level"] is not None or quality["lower_confidence_endpoint"] is not None or quality["upper_confidence_endpoint"] is not None or quality["inference_state"]!=expected_state or quality["noninferiority_supported"] is not expected_supported: raise ValueError("deterministic quality inference differs")
        expected_method="not_applicable_deterministic_pair_effects"; expected_level=None; lower=upper=None
    else:
        lower=quantile(boot,.025); upper=quantile(boot,.975)
        if quality["confidence_method"]!="deterministic_whole_pair_percentile_bootstrap" or quality["confidence_level"]!="0.95" or quality["inference_state"]!="whole_pair_bootstrap" or not math.isclose(lower,quality["lower_confidence_endpoint"],abs_tol=1e-12) or not math.isclose(upper,quality["upper_confidence_endpoint"],abs_tol=1e-12) or quality["noninferiority_supported"] is not (lower>=0.0): raise ValueError("quality bootstrap differs")
        expected_supported=lower>=0.0; expected_state="whole_pair_bootstrap"; expected_method="deterministic_whole_pair_percentile_bootstrap"; expected_level="0.95"
    expected_quality={"lane":"natural","evaluator":"evaluate_workload","independent_unit":"adjacent_eager_compiled_lifecycle_pair","effect":"compiled_minus_eager_request_success_rate","pair_effects":expected_quality_effects,"mean_pair_effect":sum(recomputed_effects)/len(recomputed_effects),"noninferiority_margin":"0","confidence_method":expected_method,"confidence_level":expected_level,"analysis_seed":20260905,"resample_count":BOOTSTRAP_RESAMPLES,"bootstrap_unit":"whole_lifecycle_pair","request_level_resampling":False,"lower_confidence_endpoint":lower,"upper_confidence_endpoint":upper,"inference_state":expected_state,"support_threshold":0.0,"noninferiority_supported":expected_supported}
    if quality!=expected_quality: raise ValueError("complete quality inference differs")
    component_observable=all(pair["compiled"]["compile_component_measurements"][name]["observability_state"]=="observed" for pair in pairs["pairs"] for name in ("compilation_time_seconds","encoder_compilation_time_seconds","cuda_graph_capture_duration_seconds"))
    expected_claims=claim_matrix(expected_analysis,controlled_identity,natural_identity,all(e["success"] for e in evaluations) and quality["noninferiority_supported"],component_observable)
    if docs["claim-matrix.json"]!=expected_claims: raise ValueError("complete claim matrix differs")
    budget=docs["budget-teardown.json"]; absent="external_provider_end_receipt_absent"
    budget_keys={"schema_version","hard_cap_usd","reserved_usd","reservations_within_hard_cap","active_operation_seconds","active_operation_list_rate_usd_per_hour","active_operation_list_rate_equivalent_usd","active_operation_equivalent_within_hard_cap","provider_billed_seconds","provider_billed_seconds_null_reason","provider_reported_spend_usd","provider_reported_spend_null_reason","provider_list_rate_cost_usd","provider_list_rate_cost_null_reason","actual_cost_usd","actual_cost_null_reason","automatic_retries","all_lifecycles_completed","local_cleanup","host_shutdown_observed_at","host_shutdown_observed_null_reason","external_provider_console_confirmation","external_provider_console_confirmation_null_reason","independently_verified_provider_termination","independently_verified_provider_termination_null_reason","provider_teardown","provider_teardown_null_reason","provider_teardown_provenance"}
    expected_equivalent=Decimal(budget["active_operation_seconds"])*Decimal("0.39")/Decimal(3600)
    nulls=(("provider_billed_seconds","provider_billed_seconds_null_reason"),("provider_reported_spend_usd","provider_reported_spend_null_reason"),("provider_list_rate_cost_usd","provider_list_rate_cost_null_reason"),("actual_cost_usd","actual_cost_null_reason"))
    teardown_nulls=(("host_shutdown_observed_at","host_shutdown_observed_null_reason","the local process cannot observe its own later host shutdown"),("external_provider_console_confirmation","external_provider_console_confirmation_null_reason","external operator confirmation was not supplied to the local runner"),("independently_verified_provider_termination","independently_verified_provider_termination_null_reason","no provider API receipt is available"))
    if set(budget)!=budget_keys or type(budget["active_operation_seconds"]) is not int or budget["active_operation_seconds"]<=0 or "within_hard_cap" in budget or budget["provider_teardown"] is not None or budget["active_operation_list_rate_usd_per_hour"]!="0.39" or Decimal(budget["active_operation_list_rate_equivalent_usd"])!=expected_equivalent or budget["active_operation_equivalent_within_hard_cap"]!=(expected_equivalent<=Decimal(budget["hard_cap_usd"])) or budget["reservations_within_hard_cap"]!=(Decimal(budget["reserved_usd"])<=Decimal(budget["hard_cap_usd"])) or any(budget[v] is not None or budget[r]!=absent for v,r in nulls) or any(budget[v] is not None or budget[r]!=reason for v,r,reason in teardown_nulls): raise ValueError("budget semantics differ")
    if (root/"crossover.svg").read_bytes()!=render_svg(expected_analysis).encode(): raise ValueError("crossover.svg does not recompute")
    if (root/"report.html").read_bytes()!=render_report(expected_analysis,correctness,expected_claims).encode(): raise ValueError("report.html does not recompute")
    if (root/"provenance-null-matrix.json").read_bytes()!=canonical(provenance_document(expected_analysis,requests,pairs["pairs"],budget)).encode(): raise ValueError("provenance-null-matrix.json does not recompute")
if __name__=="__main__":
    if sys.argv[1:]!=["verify"]: raise SystemExit("usage: evidence_bundle.py verify")
    verify(Path(__file__).resolve().parent)
    print("Completed vLLM crossover evidence verified")
'''


def _verifier_source(plan: core.VLLMCompilePlan) -> str:
    plan_document = plan.to_dict()
    return (
        VERIFIER.replace("__BOOTSTRAP_RESAMPLES__", str(BOOTSTRAP_RESAMPLES))
        .replace(
            "__CONTROLLED_SIGN_SYMMETRY_ALPHA__",
            repr(CONTROLLED_SIGN_SYMMETRY_ALPHA),
        )
        .replace(
            "__CLAIM_REQUIREMENTS__",
            repr(
                {
                    claim_id: tuple(requirements)
                    for claim_id, requirements in core.CLAIM_REQUIREMENTS.items()
                }
            ),
        )
        .replace(
            "__CANONICAL_SCHEDULE__",
            repr([cell.to_dict() for cell in plan.schedule]),
        )
        .replace(
            "__SAMPLE_STOPPING_RULE__",
            repr(plan_document["sample_stopping_rule"]),
        )
    )


def _documents(data: Mapping[str, Any]) -> dict[str, bytes]:
    protocol = _protocol_document(data)
    pairs = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "pairs": data["pair_records"],
    }
    provenance = _provenance_document(data)
    documents = {
        "analysis.json": _json_text(data["analysis"]).encode(),
        "budget-teardown.json": _json_text(data["budget"]).encode(),
        "claim-matrix.json": _json_text(data["claims"]).encode(),
        "correctness.json": _json_text(data["correctness"]).encode(),
        "crossover.svg": _render_svg(data["analysis"]).encode(),
        "evidence_bundle.py": _verifier_source(data["plan"]).encode(),
        "lifecycle-pairs.json": _json_text(pairs).encode(),
        "protocol.json": _json_text(protocol).encode(),
        "provenance-null-matrix.json": _json_text(provenance).encode(),
        "report.html": _render_report(data).encode(),
        "request-records.jsonl": _jsonl_text(data["public_requests"]).encode(),
    }
    for name, payload in documents.items():
        _scan_privacy(name, payload.decode("utf-8"))
    documents["SHA256SUMS"] = (
        "\n".join(f"{_digest(documents[name])}  {name}" for name in HASHED_FILES) + "\n"
    ).encode()
    return documents


def build_bundle(workspace: Path, output_dir: Path) -> None:
    """Validate a complete raw workspace, publish a deterministic public bundle."""

    data = _validate_workspace(workspace)
    documents = _documents(data)
    if output_dir.exists():
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise CrossoverResultsError("output must be a non-symlink directory")
        actual = {path.name for path in output_dir.iterdir()}
        if actual - set(BUNDLE_FILES):
            raise CrossoverResultsError("output directory contains extra files")
        for path in output_dir.iterdir():
            if path.is_symlink() or not path.is_file():
                raise CrossoverResultsError("output directory contains an unsafe entry")
    else:
        output_dir.mkdir(parents=True)
    for name, payload in documents.items():
        path = output_dir / name
        atomic_write_text(path, payload.decode("utf-8"))
    verify_bundle(output_dir)


def verify_bundle(bundle_dir: Path) -> None:
    """Verify inventory, bounded files, privacy, canonical data, and semantics."""

    root = bundle_dir.resolve()
    if bundle_dir.is_symlink() or not root.is_dir():
        raise CrossoverResultsError("bundle must be a non-symlink directory")
    actual = {path.name for path in root.iterdir()}
    if actual != set(BUNDLE_FILES):
        raise CrossoverResultsError(f"bundle file set differs: {sorted(actual)}")
    for name in BUNDLE_FILES:
        path = root / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > MAX_BUNDLE_FILE_BYTES
        ):
            raise CrossoverResultsError(f"{name} is not a bounded regular file")
        try:
            text = read_bounded_regular_bytes(path, MAX_BUNDLE_FILE_BYTES).decode(
                "utf-8"
            )
        except (OSError, ArtifactReadError, UnicodeDecodeError) as exc:
            raise CrossoverResultsError(f"{name} is unsafe: {exc}") from exc
        _scan_privacy(name, text)
    expected_checksums = (
        "\n".join(
            f"{_digest((root / name).read_bytes())}  {name}" for name in HASHED_FILES
        )
        + "\n"
    )
    checksum_text = (root / "SHA256SUMS").read_text(encoding="utf-8")
    if checksum_text != expected_checksums:
        raise CrossoverResultsError("SHA256SUMS does not verify")
    for line in checksum_text.splitlines():
        match = _CHECKSUM.fullmatch(line)
        if match is None:
            raise CrossoverResultsError("SHA256SUMS contains a malformed line")
    documents = {
        name: _safe_json(root / name, require_canonical=True) for name in JSON_FILES
    }
    protocol = documents["protocol.json"]
    pairs = documents["lifecycle-pairs.json"]
    analysis_document = documents["analysis.json"]
    analysis = analysis_document["controlled"]
    quality = documents["correctness.json"].get("quality_preservation")
    hardware = protocol.get("hardware_observations")
    bindings = protocol.get("bindings_verified")
    authentication = protocol.get("authorization_authentication")
    if (
        protocol.get("protocol_id") != core.PROTOCOL_ID
        or protocol.get("plan_sha256") != core.build_default_plan().content_sha256
        or protocol.get("schedule")
        != [cell.to_dict() for cell in core.build_default_plan().schedule]
        or protocol.get("sample_stopping_rule")
        != core.build_default_plan().to_dict()["sample_stopping_rule"]
        or len(pairs.get("pairs", [])) != 16
        or protocol["analysis"]["bootstrap_resamples"] != BOOTSTRAP_RESAMPLES
        or analysis.get("resample_count") != BOOTSTRAP_RESAMPLES
        or analysis.get("bootstrap_unit") != "whole_lifecycle_pair"
        or analysis.get("request_level_resampling") is not False
        or not isinstance(hardware, dict)
        or hardware.get("observation_count") != 65
        or hardware.get("clock_domain") != "host_perf_counter"
        or hardware.get("gpu_name") != core.EXPECTED_GPU_NAME
        or hardware.get("driver_version") != core.EXPECTED_DRIVER
        or hardware.get("memory_total_mib") != core.EXPECTED_MEMORY_MIB
        or hardware.get("compute_capability") != "8.9"
        or hardware.get("maximum_baseline_memory_used_mib", math.inf)
        > MAX_BASELINE_GPU_MEMORY_MIB
        or hardware.get("maximum_idle_temperature_c", math.inf)
        > MAX_IDLE_GPU_TEMPERATURE_C
        or hardware.get("maximum_idle_utilization_percent", math.inf)
        > MAX_IDLE_GPU_UTILIZATION_PERCENT
        or not isinstance(bindings, dict)
        or bindings.get("operation_receipts") is not True
        or bindings.get("progress_receipts") is not True
        or bindings.get("hardware_observations") is not True
        or not isinstance(quality, dict)
        or protocol.get("quality_preservation", {}).get("executed_resamples")
        != BOOTSTRAP_RESAMPLES
        or quality.get("resample_count") != BOOTSTRAP_RESAMPLES
        or quality.get("bootstrap_unit") != "whole_lifecycle_pair"
        or quality.get("request_level_resampling") is not False
        or not isinstance(authentication, dict)
        or set(authentication)
        != {
            "mechanism",
            "namespace",
            "signer_identity",
            "signature_sha256",
            "authorized_signers_sha256",
            "verified",
        }
        or authentication.get("mechanism") != "openssh_detached_signature"
        or authentication.get("namespace") != AUTHORIZATION_SIGNATURE_NAMESPACE
        or authentication.get("signer_identity") != AUTHORIZATION_SIGNER_IDENTITY
        or not isinstance(authentication.get("signature_sha256"), str)
        or not _SHA256.fullmatch(authentication["signature_sha256"])
        or not isinstance(authentication.get("authorized_signers_sha256"), str)
        or not _SHA256.fullmatch(authentication["authorized_signers_sha256"])
        or authentication.get("verified") is not True
    ):
        raise CrossoverResultsError("bundle protocol semantics differ")
    quality_effects = quality.get("pair_effects")
    mean_pair_effect = quality.get("mean_pair_effect")
    support_threshold = quality.get("support_threshold")
    if (
        not isinstance(quality_effects, list)
        or len(quality_effects) != core.PAIRS_PER_LANE
        or isinstance(mean_pair_effect, bool)
        or not isinstance(mean_pair_effect, (int, float))
        or not math.isfinite(mean_pair_effect)
        or isinstance(support_threshold, bool)
        or not isinstance(support_threshold, (int, float))
        or not math.isfinite(support_threshold)
        or not math.isclose(
            sum(
                effect["compiled_minus_eager_request_success_rate"]
                for effect in quality_effects
            )
            / core.PAIRS_PER_LANE,
            mean_pair_effect,
            abs_tol=1e-12,
        )
    ):
        raise CrossoverResultsError("natural quality preservation analysis differs")
    requests: list[dict[str, Any]] = []
    text = (root / "request-records.jsonl").read_text(encoding="utf-8")
    for line in text.splitlines():
        try:
            item = json.loads(line, parse_constant=reject_non_finite_json_constant)
        except (ValueError, RecursionError) as exc:
            raise CrossoverResultsError("request JSONL is invalid") from exc
        if (
            not isinstance(item, dict)
            or core.canonical_json(item) != line
            or item.get("terminal") is not True
        ):
            raise CrossoverResultsError("request JSONL is not canonical terminal data")
        requests.append(item)
    if len(requests) != 2496:
        raise CrossoverResultsError("public request cardinality differs")
    requests_by_cell: dict[str, list[dict[str, Any]]] = {}
    for request in requests:
        requests_by_cell.setdefault(request["cell_id"], []).append(request)
    for cell_requests in requests_by_cell.values():
        cell_requests.sort(key=lambda request: request["request_sequence_index"])
    pair_records = pairs["pairs"]
    try:
        _validate_public_pair_curves(
            pair_records, requests_by_cell, protocol["schedule"]
        )
    except CrossoverResultsError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise CrossoverResultsError("public lifecycle pair curves are invalid") from exc
    try:
        for pair in pair_records:
            eager = pair["eager"]
            compiled = pair["compiled"]
            recomputed_effects = _compute_pair_effects(
                eager,
                compiled,
                requests_by_cell[eager["cell_id"]],
                requests_by_cell[compiled["cell_id"]],
            )
            if pair.get("pair_effects") != recomputed_effects:
                raise CrossoverResultsError("lifecycle pair effects do not recompute")
        expected_distributions = _pair_effect_distributions(pair_records)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise CrossoverResultsError(
            "lifecycle pair effect evidence is invalid"
        ) from exc
    if analysis_document.get("pair_effect_distributions") != expected_distributions:
        raise CrossoverResultsError("pair effect distributions do not recompute")
    plan = core.build_default_plan()
    try:
        controlled_identity = _identity_summary(
            plan.schedule, requests_by_cell, "controlled"
        )
        natural_identity = _identity_summary(plan.schedule, requests_by_cell, "natural")
        controlled_curves = [
            core.PairCurve(
                pair_id=pair["pair_id"],
                order=pair["order"],
                eager_cumulative=tuple(pair["eager"]["cumulative_seconds"]),
                compiled_cumulative=tuple(pair["compiled"]["cumulative_seconds"]),
            )
            for pair in pair_records
            if pair["lane"] == "controlled"
        ]
        natural_terminal_effects = [
            pair["compiled_minus_eager_seconds"][-1]
            for pair in pair_records
            if pair["lane"] == "natural"
        ]
        expected_analysis = _analysis_document(
            controlled_curves,
            natural_identity=natural_identity["all_corresponding_outputs_identical"],
            natural_terminal_effects=natural_terminal_effects,
            pair_records=pair_records,
        )
    except (KeyError, TypeError, ValueError, core.VLLMCompileContractError) as exc:
        raise CrossoverResultsError("published analysis inputs are invalid") from exc
    if analysis_document != expected_analysis:
        raise CrossoverResultsError("complete controlled analysis does not recompute")
    correctness = documents["correctness.json"]
    if (
        correctness.get("controlled_output_identity") != controlled_identity
        or correctness.get("natural_output_identity") != natural_identity
    ):
        raise CrossoverResultsError("published output identity decisions differ")
    recomputed_evaluations: list[dict[str, Any]] = []
    for request in requests:
        if request.get("lane") != "natural":
            if "correctness" in request:
                raise CrossoverResultsError(
                    "controlled request contains correctness evaluation"
                )
            continue
        try:
            expected_evaluation = _natural_evaluation(request)
        except (KeyError, TypeError, ValueError) as exc:
            raise CrossoverResultsError(
                "published natural request cannot be evaluated"
            ) from exc
        if request.get("correctness") != expected_evaluation:
            raise CrossoverResultsError(
                "published request correctness does not recompute"
            )
        recomputed_evaluations.append(expected_evaluation)
    if correctness.get("evaluations") != recomputed_evaluations or correctness.get(
        "natural_all_correct"
    ) is not all(item["success"] for item in recomputed_evaluations):
        raise CrossoverResultsError("correctness report evaluations differ")
    try:
        recomputed_quality = _quality_preservation(
            core.build_default_plan(),
            recomputed_evaluations,
            resample_count=BOOTSTRAP_RESAMPLES,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise CrossoverResultsError(
            "natural quality evaluation records are invalid"
        ) from exc
    if recomputed_quality != quality:
        raise CrossoverResultsError("natural quality preservation does not recompute")
    expected_claims = _claim_matrix_document(
        analysis=expected_analysis,
        controlled_identity=controlled_identity,
        natural_identity=natural_identity,
        natural_all_correct=all(item["success"] for item in recomputed_evaluations),
        quality_preservation=recomputed_quality,
        component_observability=_component_observability(pair_records),
    )
    if documents["claim-matrix.json"] != expected_claims:
        raise CrossoverResultsError("complete claim matrix does not recompute")
    curves = [
        pair["compiled_minus_eager_seconds"]
        for pair in pairs["pairs"]
        if pair.get("lane") == "controlled"
    ]
    if len(curves) != 8 or any(len(curve) != 144 for curve in curves):
        raise CrossoverResultsError("controlled pair curve cardinality differs")
    recomputed = [
        sum(curve[index] for curve in curves) / len(curves) for index in range(144)
    ]
    observed_mean = analysis.get("mean_difference_curve")
    if (
        not isinstance(observed_mean, list)
        or len(observed_mean) != 144
        or any(
            not math.isclose(expected, observed, abs_tol=1e-12)
            for expected, observed in zip(recomputed, observed_mean, strict=True)
        )
    ):
        raise CrossoverResultsError("controlled mean curve differs")
    budget = documents["budget-teardown.json"]
    expected_budget_keys = {
        "schema_version",
        "hard_cap_usd",
        "reserved_usd",
        "reservations_within_hard_cap",
        "active_operation_seconds",
        "active_operation_list_rate_usd_per_hour",
        "active_operation_list_rate_equivalent_usd",
        "active_operation_equivalent_within_hard_cap",
        "provider_billed_seconds",
        "provider_billed_seconds_null_reason",
        "provider_reported_spend_usd",
        "provider_reported_spend_null_reason",
        "provider_list_rate_cost_usd",
        "provider_list_rate_cost_null_reason",
        "actual_cost_usd",
        "actual_cost_null_reason",
        "automatic_retries",
        "all_lifecycles_completed",
        "local_cleanup",
        "host_shutdown_observed_at",
        "host_shutdown_observed_null_reason",
        "external_provider_console_confirmation",
        "external_provider_console_confirmation_null_reason",
        "independently_verified_provider_termination",
        "independently_verified_provider_termination_null_reason",
        "provider_teardown",
        "provider_teardown_null_reason",
        "provider_teardown_provenance",
    }
    absent_provider_end = "external_provider_end_receipt_absent"
    try:
        active_seconds = budget["active_operation_seconds"]
        expected_equivalent = (
            Decimal(active_seconds) * core.ANTICIPATED_RATE_USD_PER_HOUR / Decimal(3600)
        )
        arithmetic_valid = (
            not isinstance(active_seconds, bool)
            and isinstance(active_seconds, int)
            and active_seconds > 0
            and budget["active_operation_list_rate_usd_per_hour"]
            == core.canonical_decimal(core.ANTICIPATED_RATE_USD_PER_HOUR)
            and budget["active_operation_list_rate_equivalent_usd"]
            == core.canonical_decimal(expected_equivalent)
            and budget["active_operation_equivalent_within_hard_cap"]
            is (expected_equivalent <= core.HARD_CAP_USD)
            and budget["reservations_within_hard_cap"]
            is (Decimal(budget["reserved_usd"]) <= core.HARD_CAP_USD)
        )
    except (KeyError, InvalidOperation, TypeError, ValueError):
        arithmetic_valid = False
    null_fields = (
        ("provider_billed_seconds", "provider_billed_seconds_null_reason"),
        ("provider_reported_spend_usd", "provider_reported_spend_null_reason"),
        ("provider_list_rate_cost_usd", "provider_list_rate_cost_null_reason"),
        ("actual_cost_usd", "actual_cost_null_reason"),
    )
    teardown_domain_nulls = (
        (
            "host_shutdown_observed_at",
            "host_shutdown_observed_null_reason",
            "the local process cannot observe its own later host shutdown",
        ),
        (
            "external_provider_console_confirmation",
            "external_provider_console_confirmation_null_reason",
            "external operator confirmation was not supplied to the local runner",
        ),
        (
            "independently_verified_provider_termination",
            "independently_verified_provider_termination_null_reason",
            "no provider API receipt is available",
        ),
    )
    if (
        set(budget) != expected_budget_keys
        or not arithmetic_valid
        or any(
            budget[value_field] is not None
            or budget[reason_field] != absent_provider_end
            for value_field, reason_field in null_fields
        )
        or any(
            budget[value_field] is not None or budget[reason_field] != reason
            for value_field, reason_field, reason in teardown_domain_nulls
        )
        or "within_hard_cap" in budget
        or budget.get("provider_teardown") is not None
    ):
        raise CrossoverResultsError("budget or provider teardown semantics differ")
    claims_by_id = {
        claim["claim_id"]: claim for claim in documents["claim-matrix.json"]["claims"]
    }
    if (
        claims_by_id.get("budget-reservations-within-hard-cap", {}).get("state")
        != "supported"
        or claims_by_id.get(
            "active-operation-list-rate-equivalent-within-hard-cap", {}
        ).get("state")
        != "supported"
        or claims_by_id.get("provider-billed-cost-within-hard-cap", {}).get("state")
        != "unsupported"
    ):
        raise CrossoverResultsError("budget claim semantics differ")
    artifact_data = {
        "analysis": expected_analysis,
        "correctness": correctness,
        "claims": expected_claims,
        "public_requests": requests,
        "pair_records": pair_records,
        "budget": budget,
    }
    expected_artifacts = {
        "report.html": _render_report(artifact_data),
        "crossover.svg": _render_svg(expected_analysis),
        "provenance-null-matrix.json": _json_text(_provenance_document(artifact_data)),
        "evidence_bundle.py": _verifier_source(core.build_default_plan()),
    }
    for name, expected_text in expected_artifacts.items():
        if (root / name).read_bytes() != expected_text.encode("utf-8"):
            raise CrossoverResultsError(f"{name} does not recompute")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cloudrift-crossover-results", allow_abbrev=False
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    build = subparsers.add_parser("build", allow_abbrev=False)
    build.add_argument("--workspace", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    verify = subparsers.add_parser("verify", allow_abbrev=False)
    verify.add_argument("--bundle", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "build":
        build_bundle(args.workspace, args.output)
        print("Completed vLLM crossover evidence built and verified")
    else:
        verify_bundle(args.bundle)
        print("Completed vLLM crossover evidence verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
