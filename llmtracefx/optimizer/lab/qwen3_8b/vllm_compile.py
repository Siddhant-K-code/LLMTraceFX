"""Pure, offline contract for the Qwen3-8B vLLM crossover protocol.

This module preserves the historical 12-request workload contract used by the
existing CloudRift evidence while adding the controlled/natural crossover core
for protocol v2. It intentionally imports no vLLM runtime code and performs no
network access.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import random
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Literal

from ..._artifact_io import (
    MAX_METADATA_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from ...collectors._shared import atomic_write_text

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
VLLM_SOURCE_COMMIT = "2cf0a6915ce544dc493a0990f2ea38d81601128a"
EXPECTED_MODEL_FILE_COUNT = 15
EXPECTED_MODEL_BYTES = 16_397_461_266
MODEL_DIRECTORY = f"model-{MODEL_REVISION}"
REQUESTS_PER_CELL = 12
WORKLOAD_IDS = (
    "structured-json-profile-extraction",
    "prose-reasoning-two-train-gap",
)
CONTEXT_TIERS = ("2k", "8k", "16k")

PROTOCOL_ID = "qwen3-8b-vllm-crossover-v2"
PLAN_SCHEMA_VERSION = "2"
LEDGER_SCHEMA_VERSION = "2"
SCHEDULE_SEED = 20260904
ANALYSIS_SEED = 20260905
SAMPLING_SEED = 20260831
PAIRS_PER_LANE = 8
LANES = ("controlled", "natural")
MODES = ("eager", "compiled")
PAIR_MODE_BLOCKS = ("ABBA", "BAAB")
CONTROLLED_CYCLES_PER_CELL = 12
CONTROLLED_REQUESTS_PER_CELL = REQUESTS_PER_CELL * CONTROLLED_CYCLES_PER_CELL
NATURAL_REQUESTS_PER_CELL = REQUESTS_PER_CELL
ANTICIPATED_RATE_USD_PER_HOUR = Decimal("0.39")
HARD_CAP_USD = Decimal("3.00")
PREFLIGHT_ALLOWANCE_SECONDS = 2700
CONTROLLED_CELL_ALLOWANCE_SECONDS = 480
NATURAL_CELL_ALLOWANCE_SECONDS = 240
RESET_ALLOWANCE_SECONDS = 60
EXPORT_ALLOWANCE_SECONDS = 900
TEARDOWN_ALLOWANCE_SECONDS = 2700
ACTIVE_PLANNED_SECONDS = 19_680
UNTOUCHED_MARGIN_SECONDS = 8_012
ABSOLUTE_CEILING_SECONDS = 27_692
BOOTSTRAP_RESAMPLES = 20_000
SIGN_FLIP_ENUMERATIONS = 256
CONTROLLED_SIGN_SYMMETRY_ALPHA = 0.05
MAX_LEDGER_ARTIFACT_BYTES = 1_048_576
QUALITY_NONINFERIORITY_MARGIN = Decimal("0")
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 4090"
EXPECTED_MEMORY_MIB = 24_564
EXPECTED_DRIVER = "580.159.03"
BASE_IMAGE_REFERENCE = (
    "vllm/vllm-openai:v0.28.0@"
    "sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635"
)
DERIVED_IMAGE_ID = (
    "sha256:fd34de17a99d2890ed1031fd32fff4c74837bbc92df7dcb955caf610266cffb3"
)
RUNTIME_PINS = {
    "python_version": "3.12",
    "vllm_version": "0.28.0",
    "torch_version": "2.13.0+cu130",
    "cuda_version": "13.0",
    "transformers_version": "5.15.1",
    "typing_extensions_version": "4.15.0",
}
DETERMINISTIC_ENVIRONMENT = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "HF_HUB_OFFLINE": "1",
    "PYTHONHASHSEED": str(SAMPLING_SEED),
    "PYTHONDONTWRITEBYTECODE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "VLLM_BATCH_INVARIANT": "0",
    "VLLM_DISABLE_COMPILE_CACHE": "1",
    "VLLM_NO_USAGE_STATS": "1",
}
EXECUTION_MODES = {
    "eager": {
        "enforce_eager": True,
        "compilation_mode": "NONE",
        "cuda_graph_mode": "NONE",
    },
    "compiled": {
        "enforce_eager": False,
        "compilation_mode": "VLLM_COMPILE",
        "cuda_graph_mode": "FULL_AND_PIECEWISE",
    },
}
RESOURCE_SETTINGS = {
    "gpu_count": 1,
    "cpu_cores": 4,
    "memory_gib": 32,
    "concurrency": 1,
    "retries": 0,
    "max_live_cells": 1,
    "network": "none",
}
LIFECYCLE_CONTROLS = {
    "fresh_container_per_cell": True,
    "adjacent_pair_cells": True,
    "hidden_generation_warmups": 0,
    "initial_host_page_cache_resets": 1,
    "between_cell_host_page_cache_resets": 31,
    "cell_unique_cache_directories": [
        "vllm",
        "torchinductor",
        "triton",
        "cuda",
        "home",
        "huggingface",
        "xdg",
    ],
    "prefix_caching": False,
    "speculative_decoding": False,
    "tensor_parallel_size": 1,
    "max_num_seqs": 1,
}
MEASUREMENT_CONTRACT = {
    "provider_lifecycle": {
        "provenance": "external_receipt_only",
        "missing": "null",
    },
    "host_lifecycle": {
        "provenance": "host_perf_counter",
        "cross_clock_subtraction": False,
    },
    "process_tree": {
        "provenance": "linux_procfs_stat_without_process_ids",
        "missing": "null",
    },
    "process_model_initialization": {
        "provenance": "same_process_perf_counter",
        "includes_required_compile_and_cuda_graph_setup": True,
    },
    "compile_time": {
        "provenance": "version_pinned_vllm_0_28_internal",
        "missing": "null",
    },
    "cuda_graph_capture_time": {
        "provenance": "no_stable_hook",
        "value": None,
    },
    "cuda_graph_dispatch_counter": {
        "provenance": "documented_metric_without_stable_offline_snapshot_hook",
        "value": None,
    },
    "request_terminal_latency": {
        "provenance": "same_process_perf_counter",
    },
    "cumulative_init_to_terminal": {
        "provenance": "initialization_plus_sum_of_request_perf_counter_durations",
        "excludes_inter_request_progress_receipt_io": True,
    },
    "ttft": {
        "provenance": "version_pinned_vllm_0_28_request_state_stats",
        "missing": "null",
    },
    "queue_prefill_inference_decode_e2e": {
        "provenance": "version_pinned_vllm_0_28_request_state_stats",
        "missing": "null",
    },
    "per_token_decode_series": {
        "provenance": "no_stable_hook",
        "value": None,
    },
    "output_rate": {
        "provenance": "exact_token_count_over_same_clock_terminal_latency",
    },
    "gpu_memory": {
        "provenance": "sampled_nvidia_smi",
        "target_interval_ms": 200,
        "scope": "whole_device",
    },
    "correctness": {
        "lane": "natural",
        "provenance": "pinned_deterministic_evaluator",
    },
    "list_rate_cost": {
        "provenance": "decimal_seconds_times_committed_list_rate",
    },
}
PROVENANCE_POLICY = {
    "offline_only": True,
    "imports_vllm": False,
    "path_bound": True,
    "source_head_bound": True,
    "canonical_json_only": True,
}
NULL_POLICY = {
    "allow_null_unobservable_components": True,
    "null_requires_explicit_reason": True,
    "missing_observation_is_not_zero": True,
}
_SAMPLE_STOPPING_RULE = {
    "crossover_analysis_lane": "controlled",
    "pair_count": PAIRS_PER_LANE,
    "curve_length": CONTROLLED_REQUESTS_PER_CELL,
    "first_crossing_rule": (
        "The first request ordinal where compiled_minus_eager cumulative time is "
        "less than or equal to zero."
    ),
    "sustained_crossing_rule": (
        "The first request ordinal where compiled_minus_eager cumulative time is "
        "less than or equal to zero for that request and every later request; "
        "otherwise the pair is right censored at 144 requests."
    ),
    "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
    "bootstrap_unit": "whole_pair",
    "sign_flip_enumerations": SIGN_FLIP_ENUMERATIONS,
    "sign_flip_semantics": (
        "Exhaustive sign-symmetry permutation test; not randomized assignment "
        "inference and assumes pair effects are sign-symmetric."
    ),
    "controlled_support_rule": (
        "The simultaneous upper-band sustained crossing is observed and the "
        "terminal-effect sign-symmetry permutation p-value is at most 0.05."
    ),
    "natural_timing_support_rule": (
        "Correct, identical, reproducible natural outputs; negative mean terminal "
        "compiled-minus-eager effect; and a whole-pair 95% percentile-bootstrap "
        "upper endpoint less than or equal to zero."
    ),
    "small_sample_limitation": (
        "The nonparametric simultaneous band uses eight lifecycle pairs and may "
        "under-cover; supported claims also require the sign-symmetry test."
    ),
}

CLAIM_REQUIREMENTS = {
    "fixed-token-count-crossover": (
        "terminal",
        "completeness",
        "fixed_count",
        "controlled_supported_crossing",
    ),
    "output-identical-generation-crossover": (
        "terminal",
        "completeness",
        "fixed_count",
        "controlled_supported_crossing",
        "controlled_output_identity",
        "controlled_numeric_reproducibility",
    ),
    "numerically-reproducible-generation-crossover": (
        "terminal",
        "completeness",
        "fixed_count",
        "controlled_supported_crossing",
        "controlled_output_identity",
        "controlled_numeric_reproducibility",
    ),
    "natural-output-quality-preserved": (
        "terminal",
        "completeness",
        "natural_correctness",
    ),
    "natural-end-to-end-causal-speedup": (
        "terminal",
        "completeness",
        "natural_output_identity",
        "natural_numeric_reproducibility",
        "natural_correctness",
        "natural_supported_speedup",
    ),
    "compile-cuda-graph-component-timing": (
        "terminal",
        "completeness",
        "component_observability",
    ),
}

ClaimState = Literal["supported", "unsupported", "not_applicable"]

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_HEAD = re.compile(r"^[0-9a-f]{40}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_PROMPT_MANIFEST = (
    Path(__file__).parent / "data" / "qwen3-8b-control-manifest-template-v1.json"
)


class VLLMCompileContractError(ValueError):
    """Raised when the immutable experiment contract is violated."""


@dataclass(frozen=True)
class WorkloadDescriptor:
    """One request in the fixed two-workload, three-tier experiment."""

    ordinal: int
    workload_id: str
    workload_version: str
    context_tier: str
    repetition: int
    prompt_sha256: str
    warmup: bool = False

    @property
    def request_id(self) -> str:
        return f"{self.context_tier}-{self.workload_id}-rep-{self.repetition:02d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "request_id": self.request_id,
            "workload_id": self.workload_id,
            "workload_version": self.workload_version,
            "context_tier": self.context_tier,
            "repetition": self.repetition,
            "prompt_sha256": self.prompt_sha256,
            "warmup": self.warmup,
        }


def canonical_json(value: Any) -> str:
    """Return finite, stable JSON used for experiment hashes."""

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VLLMCompileContractError(f"value is not canonical JSON: {exc}") from exc


def canonical_decimal(value: Decimal) -> str:
    """Return a finite Decimal in canonical plain-string form."""

    if not isinstance(value, Decimal) or not value.is_finite():
        raise VLLMCompileContractError("decimal value must be finite")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in ("", "-0") else rendered


def _decimal(value: Any, *, field: str, positive: bool = False) -> Decimal:
    if not isinstance(value, str) or not value:
        raise VLLMCompileContractError(f"{field} must be a canonical decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise VLLMCompileContractError(
            f"{field} must be a canonical decimal string"
        ) from exc
    if not result.is_finite() or (result <= 0 if positive else result < 0):
        relation = "> 0" if positive else ">= 0"
        raise VLLMCompileContractError(f"{field} must be finite and {relation}")
    if canonical_decimal(result) != value:
        raise VLLMCompileContractError(
            f"{field} must use canonical decimal spelling {canonical_decimal(result)!r}"
        )
    return result


def _sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _timestamp(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise VLLMCompileContractError(
            f"{field} must be a non-empty ISO-8601 timestamp"
        )
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise VLLMCompileContractError(
            f"{field} must be a non-empty ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise VLLMCompileContractError(f"{field} must include a timezone")
    return value


def _prompt_hashes() -> dict[str, dict[str, str]]:
    path = _PROMPT_MANIFEST
    if path.is_symlink() or not path.is_file():
        raise VLLMCompileContractError("packaged Qwen manifest is unavailable")
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise VLLMCompileContractError("packaged Qwen manifest exceeds its bound")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(
                VLLMCompileContractError(f"non-finite manifest value: {item}")
            ),
        )
        workloads = value["workloads"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise VLLMCompileContractError(
            f"packaged Qwen manifest is invalid: {exc}"
        ) from exc
    if not isinstance(workloads, list) or len(workloads) != len(WORKLOAD_IDS):
        raise VLLMCompileContractError("packaged workloads differ from the contract")
    result: dict[str, dict[str, str]] = {}
    for workload_id, item in zip(WORKLOAD_IDS, workloads, strict=True):
        if not isinstance(item, dict) or item.get("workload_id") != workload_id:
            raise VLLMCompileContractError("packaged workload order drifted")
        hashes = item.get("prompt_hashes")
        if not isinstance(hashes, dict) or set(hashes) != set(CONTEXT_TIERS):
            raise VLLMCompileContractError("packaged prompt hashes are incomplete")
        if any(
            not isinstance(candidate, str) or _SHA256.fullmatch(candidate) is None
            for candidate in hashes.values()
        ):
            raise VLLMCompileContractError("packaged prompt hash is invalid")
        result[workload_id] = dict(hashes)
    return result


def workload_descriptors() -> tuple[WorkloadDescriptor, ...]:
    """Return the exact 12-request order with no warmup requests."""

    hashes = _prompt_hashes()
    descriptors: list[WorkloadDescriptor] = []
    for tier in CONTEXT_TIERS:
        for workload_id in WORKLOAD_IDS:
            for repetition in (1, 2):
                descriptors.append(
                    WorkloadDescriptor(
                        ordinal=len(descriptors) + 1,
                        workload_id=workload_id,
                        workload_version="1",
                        context_tier=tier,
                        repetition=repetition,
                        prompt_sha256=hashes[workload_id][tier],
                    )
                )
    if len(descriptors) != REQUESTS_PER_CELL:
        raise VLLMCompileContractError("request count drifted")
    return tuple(descriptors)


def token_ids_sha256(token_ids: list[int]) -> str:
    """Return the canonical identity of one exact token array."""

    if (
        not token_ids
        or any(
            isinstance(item, bool) or not isinstance(item, int) for item in token_ids
        )
        or any(item < 0 for item in token_ids)
    ):
        raise VLLMCompileContractError("token IDs must be non-negative integers")
    return "sha256:" + hashlib.sha256(canonical_json(token_ids).encode()).hexdigest()


@dataclass(frozen=True)
class SamplingContract:
    temperature: int
    top_p: int
    seed: int
    n: int
    best_of: int
    max_tokens: int
    min_tokens: int
    ignore_eos: bool
    stop: tuple[str, ...]
    detokenize: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
            "n": self.n,
            "best_of": self.best_of,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "ignore_eos": self.ignore_eos,
            "stop": list(self.stop),
            "detokenize": self.detokenize,
        }

    @classmethod
    def from_dict(
        cls,
        data: Any,
        *,
        expected: SamplingContract | None = None,
    ) -> SamplingContract:
        if not isinstance(data, dict):
            raise VLLMCompileContractError("sampling contract must be an object")
        if expected is None:
            raise VLLMCompileContractError(
                "sampling contract requires an expected baseline"
            )
        if data != expected.to_dict():
            raise VLLMCompileContractError(
                "sampling contract does not exactly match the frozen canonical contract"
            )
        return expected


CONTROLLED_SAMPLING = SamplingContract(
    temperature=0,
    top_p=1,
    seed=SAMPLING_SEED,
    n=1,
    best_of=1,
    max_tokens=96,
    min_tokens=96,
    ignore_eos=True,
    stop=(),
    detokenize=False,
)
NATURAL_SAMPLING = SamplingContract(
    temperature=0,
    top_p=1,
    seed=SAMPLING_SEED,
    n=1,
    best_of=1,
    max_tokens=96,
    min_tokens=0,
    ignore_eos=False,
    stop=(),
    detokenize=True,
)


@dataclass(frozen=True)
class ModeContract:
    lane: str
    descriptor_cycles_per_cell: int
    requests_per_cell: int
    sampling: SamplingContract
    analysis_eligible: bool

    def to_dict(self) -> dict[str, Any]:
        descriptors = lane_request_descriptors(self.lane)
        return {
            "lane": self.lane,
            "descriptor_cycles_per_cell": self.descriptor_cycles_per_cell,
            "base_requests_per_cycle": REQUESTS_PER_CELL,
            "requests_per_cell": self.requests_per_cell,
            "analysis_eligible": self.analysis_eligible,
            "sampling": self.sampling.to_dict(),
            "base_descriptor_sha256": _sha256_json(
                [descriptor.to_dict() for descriptor in workload_descriptors()]
            ),
            "cell_descriptor_sha256": _sha256_json(
                [descriptor.to_dict() for descriptor in descriptors]
            ),
        }


def lane_request_descriptors(lane: str) -> tuple[WorkloadDescriptor, ...]:
    """Return the exact immutable descriptor sequence for one lane."""

    base = workload_descriptors()
    if lane == "controlled":
        descriptors = base * CONTROLLED_CYCLES_PER_CELL
    elif lane == "natural":
        descriptors = base
    else:
        raise VLLMCompileContractError(f"unknown lane {lane!r}")
    return descriptors


@dataclass(frozen=True)
class ScheduleCell:
    cell_id: str
    lane: str
    pair_index: int
    pair_id: str
    period_index: int
    mode: str
    order: str
    requests_per_cell: int
    descriptor_cycles_per_cell: int

    def __post_init__(self) -> None:
        if self.lane not in LANES:
            raise VLLMCompileContractError(f"unknown lane {self.lane!r}")
        if self.mode not in MODES:
            raise VLLMCompileContractError(f"unknown mode {self.mode!r}")
        if self.order not in ("eager-compiled", "compiled-eager"):
            raise VLLMCompileContractError("schedule order is invalid")
        if (
            isinstance(self.pair_index, bool)
            or not isinstance(self.pair_index, int)
            or not 1 <= self.pair_index <= PAIRS_PER_LANE
        ):
            raise VLLMCompileContractError("schedule pair_index is invalid")
        if self.period_index not in (1, 2):
            raise VLLMCompileContractError("schedule period_index is invalid")
        expected_pair_id = f"{PROTOCOL_ID}-{self.lane}-pair-{self.pair_index:02d}"
        if self.pair_id != expected_pair_id:
            raise VLLMCompileContractError("schedule pair_id does not match its fields")
        expected_mode = self.order.split("-")[self.period_index - 1]
        if self.mode != expected_mode:
            raise VLLMCompileContractError("schedule mode does not match its order")
        expected_cell_id = (
            f"{expected_pair_id}-period-{self.period_index:02d}-{self.mode}"
        )
        if self.cell_id != expected_cell_id:
            raise VLLMCompileContractError("schedule cell_id does not match its fields")
        expected_requests = (
            CONTROLLED_REQUESTS_PER_CELL
            if self.lane == "controlled"
            else NATURAL_REQUESTS_PER_CELL
        )
        expected_cycles = CONTROLLED_CYCLES_PER_CELL if self.lane == "controlled" else 1
        if self.requests_per_cell != expected_requests:
            raise VLLMCompileContractError(
                "schedule requests_per_cell does not match its lane"
            )
        if self.descriptor_cycles_per_cell != expected_cycles:
            raise VLLMCompileContractError(
                "schedule descriptor_cycles_per_cell does not match its lane"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "lane": self.lane,
            "pair_index": self.pair_index,
            "pair_id": self.pair_id,
            "period_index": self.period_index,
            "mode": self.mode,
            "order": self.order,
            "requests_per_cell": self.requests_per_cell,
            "descriptor_cycles_per_cell": self.descriptor_cycles_per_cell,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ScheduleCell:
        if not isinstance(data, dict):
            raise VLLMCompileContractError("schedule cell must be an object")
        try:
            cell = cls(
                cell_id=data["cell_id"],
                lane=data["lane"],
                pair_index=data["pair_index"],
                pair_id=data["pair_id"],
                period_index=data["period_index"],
                mode=data["mode"],
                order=data["order"],
                requests_per_cell=data["requests_per_cell"],
                descriptor_cycles_per_cell=data["descriptor_cycles_per_cell"],
            )
        except KeyError as exc:
            raise VLLMCompileContractError(
                f"schedule cell is missing a required field: {exc}"
            ) from exc
        if data != cell.to_dict():
            raise VLLMCompileContractError(
                "schedule cell does not exactly match the canonical contract"
            )
        return cell


def _first_mode_blocks(lane: str) -> tuple[str, str]:
    digest = hashlib.sha256(
        f"{PROTOCOL_ID}:{SCHEDULE_SEED}:{lane}:blocks".encode()
    ).digest()
    return (
        PAIR_MODE_BLOCKS
        if digest[0] % 2 == 0
        else (PAIR_MODE_BLOCKS[1], PAIR_MODE_BLOCKS[0])
    )


def lane_first_mode_symbols(lane: str) -> tuple[str, ...]:
    if lane not in LANES:
        raise VLLMCompileContractError(f"unknown lane {lane!r}")
    return tuple("".join(_first_mode_blocks(lane)))


def lane_pair_orders(lane: str) -> tuple[str, ...]:
    orders: list[str] = []
    for symbol in lane_first_mode_symbols(lane):
        first = "eager" if symbol == "A" else "compiled"
        second = "compiled" if first == "eager" else "eager"
        orders.append(f"{first}-{second}")
    return tuple(orders)


def _pair_unit_lane_order() -> tuple[str, ...]:
    rng = random.Random(SCHEDULE_SEED)
    remaining = dict.fromkeys(LANES, PAIRS_PER_LANE)
    order: list[str] = []
    while len(order) < PAIRS_PER_LANE * len(LANES):
        choices: list[tuple[tuple[int, float], str]] = []
        for lane in LANES:
            if remaining[lane] <= 0:
                continue
            if len(order) >= 2 and order[-1] == order[-2] == lane:
                continue
            choices.append(((remaining[lane], rng.random()), lane))
        if not choices:
            raise VLLMCompileContractError(
                "deterministic schedule interleave is impossible"
            )
        _, lane = max(choices)
        order.append(lane)
        remaining[lane] -= 1
    return tuple(order)


def _build_schedule() -> tuple[ScheduleCell, ...]:
    order_by_lane = {lane: lane_pair_orders(lane) for lane in LANES}
    pair_counts = dict.fromkeys(LANES, 0)
    cells: list[ScheduleCell] = []
    for lane in _pair_unit_lane_order():
        pair_counts[lane] += 1
        pair_index = pair_counts[lane]
        order = order_by_lane[lane][pair_index - 1]
        pair_id = f"{PROTOCOL_ID}-{lane}-pair-{pair_index:02d}"
        requests_per_cell = (
            CONTROLLED_REQUESTS_PER_CELL
            if lane == "controlled"
            else NATURAL_REQUESTS_PER_CELL
        )
        descriptor_cycles = CONTROLLED_CYCLES_PER_CELL if lane == "controlled" else 1
        for period_index, mode in enumerate(order.split("-"), start=1):
            cells.append(
                ScheduleCell(
                    cell_id=(f"{pair_id}-period-{period_index:02d}-{mode}"),
                    lane=lane,
                    pair_index=pair_index,
                    pair_id=pair_id,
                    period_index=period_index,
                    mode=mode,
                    order=order,
                    requests_per_cell=requests_per_cell,
                    descriptor_cycles_per_cell=descriptor_cycles,
                )
            )
    return tuple(cells)


CROSSOVER_SCHEDULE = _build_schedule()


def crossover_schedule() -> tuple[ScheduleCell, ...]:
    return CROSSOVER_SCHEDULE


@dataclass(frozen=True)
class BudgetLine:
    line_id: str
    kind: str
    occurrences: int
    seconds_per_occurrence: int
    total_seconds: int
    amount_usd: Decimal
    reservable: bool
    lifecycle_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_id": self.line_id,
            "kind": self.kind,
            "occurrences": self.occurrences,
            "seconds_per_occurrence": self.seconds_per_occurrence,
            "total_seconds": self.total_seconds,
            "amount_usd": canonical_decimal(self.amount_usd),
            "rate_usd_per_hour": canonical_decimal(ANTICIPATED_RATE_USD_PER_HOUR),
            "reservable": self.reservable,
            "lifecycle_ids": list(self.lifecycle_ids),
        }


@dataclass(frozen=True)
class BudgetLifecycle:
    lifecycle_id: str
    line_id: str
    kind: str
    ordinal: int
    planned_seconds: int
    ceiling_usd: Decimal
    cell_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lifecycle_id": self.lifecycle_id,
            "line_id": self.line_id,
            "kind": self.kind,
            "ordinal": self.ordinal,
            "planned_seconds": self.planned_seconds,
            "ceiling_usd": canonical_decimal(self.ceiling_usd),
            "rate_usd_per_hour": canonical_decimal(ANTICIPATED_RATE_USD_PER_HOUR),
            "cell_id": self.cell_id,
        }


@dataclass(frozen=True)
class BudgetSummary:
    active_planned_seconds: int
    active_planned_usd: Decimal
    untouched_margin_seconds: int
    untouched_margin_usd: Decimal
    absolute_ceiling_seconds: int
    absolute_ceiling_usd: Decimal

    def to_dict(self) -> dict[str, Any]:
        return {
            "anticipated_rate_usd_per_hour": canonical_decimal(
                ANTICIPATED_RATE_USD_PER_HOUR
            ),
            "hard_cap_usd": canonical_decimal(HARD_CAP_USD),
            "active_planned_seconds": self.active_planned_seconds,
            "active_planned_usd": canonical_decimal(self.active_planned_usd),
            "untouched_margin_seconds": self.untouched_margin_seconds,
            "untouched_margin_usd": canonical_decimal(self.untouched_margin_usd),
            "absolute_ceiling_seconds": self.absolute_ceiling_seconds,
            "absolute_ceiling_usd": canonical_decimal(self.absolute_ceiling_usd),
        }


def _cost_for_seconds(seconds: int) -> Decimal:
    if isinstance(seconds, bool) or not isinstance(seconds, int) or seconds < 0:
        raise VLLMCompileContractError("seconds must be a non-negative integer")
    with localcontext() as context:
        context.prec = 28
        return Decimal(seconds) * ANTICIPATED_RATE_USD_PER_HOUR / Decimal(3600)


def _build_budget() -> (
    tuple[tuple[BudgetLine, ...], tuple[BudgetLifecycle, ...], BudgetSummary]
):
    lifecycles: list[BudgetLifecycle] = []

    def add_lifecycle(
        lifecycle_id: str,
        *,
        line_id: str,
        kind: str,
        planned_seconds: int,
        cell_id: str | None = None,
    ) -> None:
        lifecycles.append(
            BudgetLifecycle(
                lifecycle_id=lifecycle_id,
                line_id=line_id,
                kind=kind,
                ordinal=len(lifecycles) + 1,
                planned_seconds=planned_seconds,
                ceiling_usd=_cost_for_seconds(planned_seconds),
                cell_id=cell_id,
            )
        )

    add_lifecycle(
        "preflight-01",
        line_id="preflight",
        kind="preflight",
        planned_seconds=PREFLIGHT_ALLOWANCE_SECONDS,
    )
    for index, cell in enumerate(CROSSOVER_SCHEDULE, start=1):
        add_lifecycle(
            f"{cell.lane}-cell-{cell.pair_index:02d}-{cell.period_index:02d}",
            line_id=f"{cell.lane}-cell",
            kind="cell",
            planned_seconds=(
                CONTROLLED_CELL_ALLOWANCE_SECONDS
                if cell.lane == "controlled"
                else NATURAL_CELL_ALLOWANCE_SECONDS
            ),
            cell_id=cell.cell_id,
        )
        if index != len(CROSSOVER_SCHEDULE):
            add_lifecycle(
                f"reset-{index:02d}",
                line_id="reset",
                kind="reset",
                planned_seconds=RESET_ALLOWANCE_SECONDS,
            )
    add_lifecycle(
        "export-01",
        line_id="export",
        kind="export",
        planned_seconds=EXPORT_ALLOWANCE_SECONDS,
    )
    add_lifecycle(
        "teardown-01",
        line_id="teardown",
        kind="teardown",
        planned_seconds=TEARDOWN_ALLOWANCE_SECONDS,
    )

    lifecycle_ids_by_line: dict[str, list[str]] = {}
    for lifecycle in lifecycles:
        lifecycle_ids_by_line.setdefault(lifecycle.line_id, []).append(
            lifecycle.lifecycle_id
        )

    lines = (
        BudgetLine(
            line_id="preflight",
            kind="preflight",
            occurrences=1,
            seconds_per_occurrence=PREFLIGHT_ALLOWANCE_SECONDS,
            total_seconds=PREFLIGHT_ALLOWANCE_SECONDS,
            amount_usd=_cost_for_seconds(PREFLIGHT_ALLOWANCE_SECONDS),
            reservable=True,
            lifecycle_ids=tuple(lifecycle_ids_by_line["preflight"]),
        ),
        BudgetLine(
            line_id="controlled-cell",
            kind="cell",
            occurrences=16,
            seconds_per_occurrence=CONTROLLED_CELL_ALLOWANCE_SECONDS,
            total_seconds=16 * CONTROLLED_CELL_ALLOWANCE_SECONDS,
            amount_usd=_cost_for_seconds(16 * CONTROLLED_CELL_ALLOWANCE_SECONDS),
            reservable=True,
            lifecycle_ids=tuple(lifecycle_ids_by_line["controlled-cell"]),
        ),
        BudgetLine(
            line_id="natural-cell",
            kind="cell",
            occurrences=16,
            seconds_per_occurrence=NATURAL_CELL_ALLOWANCE_SECONDS,
            total_seconds=16 * NATURAL_CELL_ALLOWANCE_SECONDS,
            amount_usd=_cost_for_seconds(16 * NATURAL_CELL_ALLOWANCE_SECONDS),
            reservable=True,
            lifecycle_ids=tuple(lifecycle_ids_by_line["natural-cell"]),
        ),
        BudgetLine(
            line_id="reset",
            kind="reset",
            occurrences=31,
            seconds_per_occurrence=RESET_ALLOWANCE_SECONDS,
            total_seconds=31 * RESET_ALLOWANCE_SECONDS,
            amount_usd=_cost_for_seconds(31 * RESET_ALLOWANCE_SECONDS),
            reservable=True,
            lifecycle_ids=tuple(lifecycle_ids_by_line["reset"]),
        ),
        BudgetLine(
            line_id="export",
            kind="export",
            occurrences=1,
            seconds_per_occurrence=EXPORT_ALLOWANCE_SECONDS,
            total_seconds=EXPORT_ALLOWANCE_SECONDS,
            amount_usd=_cost_for_seconds(EXPORT_ALLOWANCE_SECONDS),
            reservable=True,
            lifecycle_ids=tuple(lifecycle_ids_by_line["export"]),
        ),
        BudgetLine(
            line_id="teardown",
            kind="teardown",
            occurrences=1,
            seconds_per_occurrence=TEARDOWN_ALLOWANCE_SECONDS,
            total_seconds=TEARDOWN_ALLOWANCE_SECONDS,
            amount_usd=_cost_for_seconds(TEARDOWN_ALLOWANCE_SECONDS),
            reservable=True,
            lifecycle_ids=tuple(lifecycle_ids_by_line["teardown"]),
        ),
        BudgetLine(
            line_id="untouched-margin",
            kind="margin",
            occurrences=1,
            seconds_per_occurrence=UNTOUCHED_MARGIN_SECONDS,
            total_seconds=UNTOUCHED_MARGIN_SECONDS,
            amount_usd=_cost_for_seconds(UNTOUCHED_MARGIN_SECONDS),
            reservable=False,
            lifecycle_ids=(),
        ),
    )
    active_planned_usd = sum(
        (line.amount_usd for line in lines if line.reservable),
        Decimal(),
    )
    summary = BudgetSummary(
        active_planned_seconds=ACTIVE_PLANNED_SECONDS,
        active_planned_usd=active_planned_usd,
        untouched_margin_seconds=UNTOUCHED_MARGIN_SECONDS,
        untouched_margin_usd=_cost_for_seconds(UNTOUCHED_MARGIN_SECONDS),
        absolute_ceiling_seconds=ABSOLUTE_CEILING_SECONDS,
        absolute_ceiling_usd=_cost_for_seconds(ABSOLUTE_CEILING_SECONDS),
    )
    return lines, tuple(lifecycles), summary


BUDGET_LINES, BUDGET_LIFECYCLES, BUDGET_SUMMARY = _build_budget()


@dataclass(frozen=True)
class VLLMCompilePlan:
    """Strict immutable plan for the offline crossover protocol."""

    mode_contracts: tuple[ModeContract, ...]
    schedule: tuple[ScheduleCell, ...]
    budget_lines: tuple[BudgetLine, ...]
    budget_lifecycles: tuple[BudgetLifecycle, ...]
    budget_summary: BudgetSummary

    @classmethod
    def create(cls) -> VLLMCompilePlan:
        mode_contracts = (
            ModeContract(
                lane="controlled",
                descriptor_cycles_per_cell=CONTROLLED_CYCLES_PER_CELL,
                requests_per_cell=CONTROLLED_REQUESTS_PER_CELL,
                sampling=CONTROLLED_SAMPLING,
                analysis_eligible=True,
            ),
            ModeContract(
                lane="natural",
                descriptor_cycles_per_cell=1,
                requests_per_cell=NATURAL_REQUESTS_PER_CELL,
                sampling=NATURAL_SAMPLING,
                analysis_eligible=False,
            ),
        )
        return cls(
            mode_contracts=mode_contracts,
            schedule=CROSSOVER_SCHEDULE,
            budget_lines=BUDGET_LINES,
            budget_lifecycles=BUDGET_LIFECYCLES,
            budget_summary=BUDGET_SUMMARY,
        )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "schedule_seed": SCHEDULE_SEED,
            "analysis_seed": ANALYSIS_SEED,
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "expected_file_count": EXPECTED_MODEL_FILE_COUNT,
                "expected_bytes": EXPECTED_MODEL_BYTES,
                "directory": MODEL_DIRECTORY,
            },
            "runtime": {
                "vllm_source_commit": VLLM_SOURCE_COMMIT,
                "base_image_reference": BASE_IMAGE_REFERENCE,
                "derived_image_id": DERIVED_IMAGE_ID,
                "runtime_pins": dict(RUNTIME_PINS),
                "expected_gpu_name": EXPECTED_GPU_NAME,
                "expected_memory_mib": EXPECTED_MEMORY_MIB,
                "expected_driver": EXPECTED_DRIVER,
            },
            "resource_settings": dict(RESOURCE_SETTINGS),
            "execution_modes": dict(EXECUTION_MODES),
            "vllm_0_28_sampling_adapter": {
                "omitted_runtime_arguments": {
                    "best_of": (
                        "vLLM 0.28.0 removed the best_of SamplingParams argument; "
                        "n=1 preserves one effective candidate"
                    )
                }
            },
            "lifecycle_controls": dict(LIFECYCLE_CONTROLS),
            "measurement_contract": dict(MEASUREMENT_CONTRACT),
            "claim_requirements": {
                claim_id: list(requirements)
                for claim_id, requirements in CLAIM_REQUIREMENTS.items()
            },
            "reproducibility": {
                "schedule_seed": SCHEDULE_SEED,
                "analysis_seed": ANALYSIS_SEED,
                "sampling_seed": SAMPLING_SEED,
                "deterministic_schedule": True,
                "deterministic_bootstrap": True,
                "whole_pair_resampling_only": True,
                "local_files_only": True,
                "environment": dict(DETERMINISTIC_ENVIRONMENT),
                "torch_deterministic_algorithms": True,
                "cudnn_deterministic": True,
                "cudnn_benchmark": False,
                "tf32": False,
                "float32_matmul_precision": "highest",
            },
            "modes": {mode.lane: mode.to_dict() for mode in self.mode_contracts},
            "schedule": [cell.to_dict() for cell in self.schedule],
            "schedule_sha256": _sha256_json([cell.to_dict() for cell in self.schedule]),
            "sample_stopping_rule": dict(_SAMPLE_STOPPING_RULE),
            "quality_preservation": {
                "lane": "natural",
                "evaluator": "evaluate_workload",
                "independent_unit": "adjacent eager-compiled lifecycle pair",
                "effect": "compiled_minus_eager_request_success_rate",
                "noninferiority_margin": canonical_decimal(
                    QUALITY_NONINFERIORITY_MARGIN
                ),
                "inference_method": (
                    "deterministic whole-pair percentile bootstrap unless every "
                    "pair effect is identical; identical effects are reported as "
                    "a deterministic observed-workload fact without CI endpoints"
                ),
                "confidence_level": "0.95",
                "resamples": BOOTSTRAP_RESAMPLES,
                "support_rule": (
                    "lower confidence endpoint >= negative margin; when all pair "
                    "effects are identical, the shared deterministic effect >= "
                    "negative margin"
                ),
            },
            "budget": {
                "lines": [line.to_dict() for line in self.budget_lines],
                "lifecycles": [
                    lifecycle.to_dict() for lifecycle in self.budget_lifecycles
                ],
                "summary": self.budget_summary.to_dict(),
            },
            "provenance_policy": dict(PROVENANCE_POLICY),
            "null_policy": dict(NULL_POLICY),
        }

    @property
    def content_sha256(self) -> str:
        return _sha256_json(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        data = self._content_dict()
        data["plan_sha256"] = self.content_sha256
        return data

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Any) -> VLLMCompilePlan:
        if not isinstance(data, dict):
            raise VLLMCompileContractError("plan must be an object")
        expected = cls.create().to_dict()
        missing = sorted(set(expected) - set(data))
        extra = sorted(set(data) - set(expected))
        if missing or extra:
            raise VLLMCompileContractError(
                f"plan keys must match exactly; missing={missing!r} extra={extra!r}"
            )
        if data != expected:
            raise VLLMCompileContractError(
                "plan does not exactly match the frozen canonical crossover contract"
            )
        return cls.create()

    @classmethod
    def from_json(cls, payload: str) -> VLLMCompilePlan:
        try:
            data = json.loads(payload, parse_constant=reject_non_finite_json_constant)
        except (ValueError, RecursionError) as exc:
            raise VLLMCompileContractError(f"invalid plan JSON: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def read_json(cls, path: str | Path) -> VLLMCompilePlan:
        try:
            payload = read_bounded_regular_text(path, MAX_METADATA_ARTIFACT_BYTES)
        except (OSError, ArtifactReadError) as exc:
            raise VLLMCompileContractError(f"failed to read plan: {exc}") from exc
        return cls.from_json(payload)


def build_default_plan() -> VLLMCompilePlan:
    return VLLMCompilePlan.create()


build_plan = VLLMCompilePlan.create


def _event_hash(event: dict[str, Any]) -> str:
    material = dict(event)
    material.pop("event_sha256", None)
    return _sha256_json(material)


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    material = dict(payload)
    material.pop("ledger_sha256", None)
    material["ledger_sha256"] = _sha256_json(material)
    return material


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class LifecycleBudgetLedger:
    """Path-bound append-only reservation ledger for crossover lifecycles."""

    def __init__(
        self,
        path: str | Path,
        *,
        plan: VLLMCompilePlan,
        git_head: str,
        workspace_path: str | Path,
    ) -> None:
        if not _GIT_HEAD.fullmatch(git_head):
            raise VLLMCompileContractError("git head must be an exact 40-hex commit")
        workspace = Path(workspace_path)
        if not workspace.exists() or not workspace.is_dir() or workspace.is_symlink():
            raise VLLMCompileContractError(
                "workspace path must be an existing non-symlink directory"
            )
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")
        self.plan = plan
        self.git_head = git_head
        self.workspace_path = workspace.resolve()
        self._high_water_revision = 0
        self._high_water_reserved = Decimal()
        self._lifecycle_by_id = {
            lifecycle.lifecycle_id: lifecycle
            for lifecycle in self.plan.budget_lifecycles
        }

    @classmethod
    def initialize(
        cls,
        path: str | Path,
        *,
        plan: VLLMCompilePlan,
        git_head: str,
        workspace_path: str | Path,
    ) -> LifecycleBudgetLedger:
        ledger = cls(path, plan=plan, git_head=git_head, workspace_path=workspace_path)
        with _locked(ledger.lock_path):
            if ledger.path.exists():
                raise VLLMCompileContractError(
                    "lifecycle ledger already exists and cannot be reset"
                )
            ledger._write(ledger._initial_payload())
        return ledger

    def _initial_payload(self) -> dict[str, Any]:
        entries = [
            {
                **lifecycle.to_dict(),
                "status": "planned",
                "command_id": None,
                "reserved_at": None,
                "completed_at": None,
                "aborted_at": None,
                "actual_seconds": None,
                "actual_cost_usd": None,
                "abort_reason": None,
                "argv_sha256": None,
            }
            for lifecycle in self.plan.budget_lifecycles
        ]
        return {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "plan_sha256": self.plan.content_sha256,
            "source_head": self.git_head,
            "workspace_path_sha256": _sha256_text(str(self.workspace_path)),
            "ledger_path_sha256": _sha256_text(str(self.path)),
            "hard_cap_usd": canonical_decimal(HARD_CAP_USD),
            "revision": 0,
            "reserved_usd": "0",
            "remaining_usd": canonical_decimal(HARD_CAP_USD),
            "entries": entries,
            "events": [],
        }

    def _write(self, payload: dict[str, Any]) -> None:
        atomic_write_text(
            self.path,
            json.dumps(_seal(payload), indent=2, sort_keys=True, allow_nan=False)
            + "\n",
        )
        revision = payload.get("revision")
        reserved = payload.get("reserved_usd")
        if isinstance(revision, int) and isinstance(reserved, str):
            self._high_water_revision = max(self._high_water_revision, revision)
            self._high_water_reserved = max(
                self._high_water_reserved,
                _decimal(reserved, field="ledger reserved_usd"),
            )

    def _read(self) -> dict[str, Any]:
        try:
            payload = json.loads(
                read_bounded_regular_text(self.path, MAX_LEDGER_ARTIFACT_BYTES),
                parse_constant=reject_non_finite_json_constant,
            )
        except (OSError, ArtifactReadError, ValueError, RecursionError) as exc:
            raise VLLMCompileContractError(
                f"failed to read lifecycle ledger: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise VLLMCompileContractError("lifecycle ledger must be an object")
        expected_seal = payload.get("ledger_sha256")
        if (
            not isinstance(expected_seal, str)
            or _seal(payload)["ledger_sha256"] != expected_seal
        ):
            raise VLLMCompileContractError(
                "lifecycle ledger integrity seal does not verify"
            )
        expected = self._initial_payload()
        for field in (
            "schema_version",
            "protocol_id",
            "plan_sha256",
            "source_head",
            "workspace_path_sha256",
            "ledger_path_sha256",
            "hard_cap_usd",
        ):
            if payload.get(field) != expected[field]:
                raise VLLMCompileContractError(
                    f"lifecycle ledger {field} binding does not match"
                )
        revision = payload.get("revision")
        entries = payload.get("entries")
        events = payload.get("events")
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 0
            or not isinstance(entries, list)
            or not isinstance(events, list)
            or revision != len(events)
        ):
            raise VLLMCompileContractError(
                "lifecycle ledger revision/event count is invalid"
            )
        replayed = self._replay_events(events)
        expected_entries = [
            replayed["entries"][lifecycle.lifecycle_id]
            for lifecycle in self.plan.budget_lifecycles
        ]
        if entries != expected_entries:
            raise VLLMCompileContractError(
                "lifecycle ledger entries do not match the append-only event log"
            )
        total = replayed["reserved_usd"]
        if total > HARD_CAP_USD:
            raise VLLMCompileContractError("lifecycle reservations exceed hard cap")
        if payload.get("reserved_usd") != canonical_decimal(total):
            raise VLLMCompileContractError("lifecycle reserved total does not verify")
        if payload.get("remaining_usd") != canonical_decimal(HARD_CAP_USD - total):
            raise VLLMCompileContractError("lifecycle remaining total does not verify")
        if revision < self._high_water_revision or total < self._high_water_reserved:
            raise VLLMCompileContractError("lifecycle ledger rollback detected")
        self._high_water_revision = revision
        self._high_water_reserved = total
        return payload

    def _replay_events(self, events: Sequence[Any]) -> dict[str, Any]:
        entries = {
            lifecycle.lifecycle_id: {
                **lifecycle.to_dict(),
                "status": "planned",
                "command_id": None,
                "reserved_at": None,
                "completed_at": None,
                "aborted_at": None,
                "actual_seconds": None,
                "actual_cost_usd": None,
                "abort_reason": None,
                "argv_sha256": None,
            }
            for lifecycle in self.plan.budget_lifecycles
        }
        total = Decimal()
        previous: str | None = None
        commands: dict[str, str] = {}
        for index, raw in enumerate(events, start=1):
            if not isinstance(raw, dict):
                raise VLLMCompileContractError("lifecycle event must be an object")
            if raw.get("index") != index:
                raise VLLMCompileContractError("lifecycle event index is invalid")
            if raw.get("previous_event_sha256") != previous:
                raise VLLMCompileContractError("lifecycle event chain is broken")
            if raw.get("event_sha256") != _event_hash(raw):
                raise VLLMCompileContractError("lifecycle event hash does not verify")
            event_type = raw.get("event_type")
            command_id = raw.get("command_id")
            if not isinstance(command_id, str) or not _SAFE_ID.fullmatch(command_id):
                raise VLLMCompileContractError(
                    "lifecycle event command identity is invalid"
                )
            if event_type == "reserve":
                lifecycle_id = raw.get("lifecycle_id")
                line_id = raw.get("line_id")
                if lifecycle_id not in entries:
                    raise VLLMCompileContractError(
                        "lifecycle event lifecycle_id is not in the plan"
                    )
                entry = entries[lifecycle_id]
                if entry["status"] != "planned":
                    raise VLLMCompileContractError("lifecycle reservation was replayed")
                if command_id in commands:
                    raise VLLMCompileContractError(
                        "lifecycle event command identity is duplicated"
                    )
                if line_id != entry["line_id"]:
                    raise VLLMCompileContractError(
                        "lifecycle event line_id does not match the planned lifecycle"
                    )
                reserved = _decimal(
                    raw.get("reserved_usd"),
                    field="lifecycle event reserved_usd",
                    positive=True,
                )
                if reserved != _decimal(
                    entry["ceiling_usd"],
                    field="planned lifecycle ceiling_usd",
                    positive=True,
                ):
                    raise VLLMCompileContractError(
                        "lifecycle event reservation differs from its planned ceiling"
                    )
                _timestamp(raw.get("reserved_at"), field="reserved_at")
                argv_sha256 = raw.get("argv_sha256")
                if (
                    not isinstance(argv_sha256, str)
                    or _SHA256.fullmatch(argv_sha256) is None
                ):
                    raise VLLMCompileContractError(
                        "lifecycle reservation argv hash is invalid"
                    )
                entry["status"] = "reserved"
                entry["command_id"] = command_id
                entry["reserved_at"] = raw["reserved_at"]
                entry["argv_sha256"] = argv_sha256
                commands[command_id] = lifecycle_id
                total += reserved
            elif event_type == "complete":
                if command_id not in commands:
                    raise VLLMCompileContractError("lifecycle completion is unknown")
                entry = entries[commands[command_id]]
                if entry["status"] != "reserved":
                    raise VLLMCompileContractError("lifecycle completion was replayed")
                _timestamp(raw.get("completed_at"), field="completed_at")
                actual_seconds = raw.get("actual_seconds")
                if (
                    isinstance(actual_seconds, bool)
                    or not isinstance(actual_seconds, int)
                    or actual_seconds < 0
                    or actual_seconds > entry["planned_seconds"]
                ):
                    raise VLLMCompileContractError(
                        "lifecycle completion actual_seconds is invalid"
                    )
                actual_cost = _decimal(
                    raw.get("actual_cost_usd"),
                    field="lifecycle event actual_cost_usd",
                )
                if actual_cost != _cost_for_seconds(actual_seconds):
                    raise VLLMCompileContractError(
                        "lifecycle completion actual_cost_usd does not match "
                        "actual_seconds"
                    )
                if actual_cost > _decimal(
                    entry["ceiling_usd"],
                    field="planned lifecycle ceiling_usd",
                    positive=True,
                ):
                    raise VLLMCompileContractError(
                        "lifecycle completion actual_cost_usd exceeds the "
                        "reserved ceiling"
                    )
                entry["status"] = "completed"
                entry["completed_at"] = raw["completed_at"]
                entry["actual_seconds"] = actual_seconds
                entry["actual_cost_usd"] = raw["actual_cost_usd"]
            elif event_type == "abort":
                if command_id not in commands:
                    raise VLLMCompileContractError("lifecycle abort is unknown")
                entry = entries[commands[command_id]]
                if entry["status"] != "reserved":
                    raise VLLMCompileContractError("lifecycle abort was replayed")
                _timestamp(raw.get("aborted_at"), field="aborted_at")
                reason = raw.get("abort_reason")
                if not isinstance(reason, str) or not reason:
                    raise VLLMCompileContractError(
                        "lifecycle abort reason must be non-empty"
                    )
                entry["status"] = "aborted"
                entry["aborted_at"] = raw["aborted_at"]
                entry["abort_reason"] = reason
            else:
                raise VLLMCompileContractError("lifecycle event type is invalid")
            previous = raw["event_sha256"]
        return {"entries": entries, "reserved_usd": total}

    def snapshot(self) -> dict[str, Any]:
        with _locked(self.lock_path):
            return self._read()

    def reserve(
        self,
        command_id: str,
        *,
        line_id: str,
        lifecycle_id: str,
        ceiling_usd: Decimal,
        argv: Sequence[str],
        reserved_at: str,
    ) -> dict[str, Any]:
        if not isinstance(command_id, str) or not _SAFE_ID.fullmatch(command_id):
            raise VLLMCompileContractError("command_id is not a safe stable identity")
        if lifecycle_id not in self._lifecycle_by_id:
            raise VLLMCompileContractError(
                "reservation lifecycle_id is not in the plan"
            )
        lifecycle = self._lifecycle_by_id[lifecycle_id]
        if line_id != lifecycle.line_id:
            raise VLLMCompileContractError(
                "reservation line_id does not match the planned lifecycle"
            )
        if ceiling_usd != lifecycle.ceiling_usd:
            raise VLLMCompileContractError(
                "command ceiling must exactly match its approved lifecycle ceiling"
            )
        if (
            isinstance(argv, (str, bytes))
            or not argv
            or any(not isinstance(item, str) or not item for item in argv)
        ):
            raise VLLMCompileContractError("argv must be non-empty immutable strings")
        _timestamp(reserved_at, field="reserved_at")
        with _locked(self.lock_path):
            payload = self._read()
            entries = {
                entry["lifecycle_id"]: entry
                for entry in payload["entries"]
                if isinstance(entry, dict)
            }
            entry = entries[lifecycle_id]
            if entry["status"] != "planned":
                raise VLLMCompileContractError(
                    "planned lifecycle is already reserved or completed"
                )
            if any(event["command_id"] == command_id for event in payload["events"]):
                raise VLLMCompileContractError(
                    f"command {command_id!r} is already reserved"
                )
            already = _decimal(payload["reserved_usd"], field="ledger reserved_usd")
            if already + ceiling_usd > HARD_CAP_USD:
                raise VLLMCompileContractError(
                    "pre-command reservation refused: hard cap would be exceeded"
                )
            previous = (
                payload["events"][-1]["event_sha256"] if payload["events"] else None
            )
            event = {
                "index": len(payload["events"]) + 1,
                "event_type": "reserve",
                "command_id": command_id,
                "line_id": line_id,
                "lifecycle_id": lifecycle_id,
                "reserved_usd": canonical_decimal(ceiling_usd),
                "reserved_at": reserved_at,
                "argv_sha256": _sha256_json(list(argv)),
                "previous_event_sha256": previous,
            }
            event["event_sha256"] = _event_hash(event)
            entry["status"] = "reserved"
            entry["command_id"] = command_id
            entry["reserved_at"] = reserved_at
            entry["argv_sha256"] = event["argv_sha256"]
            payload["events"].append(event)
            payload["revision"] += 1
            payload["reserved_usd"] = canonical_decimal(already + ceiling_usd)
            payload["remaining_usd"] = canonical_decimal(
                HARD_CAP_USD - already - ceiling_usd
            )
            self._write(payload)
            return dict(event)

    def complete(
        self,
        command_id: str,
        *,
        completed_at: str,
        actual_seconds: int,
    ) -> dict[str, Any]:
        if not isinstance(command_id, str) or not _SAFE_ID.fullmatch(command_id):
            raise VLLMCompileContractError("command_id is not a safe stable identity")
        _timestamp(completed_at, field="completed_at")
        if (
            isinstance(actual_seconds, bool)
            or not isinstance(actual_seconds, int)
            or actual_seconds < 0
        ):
            raise VLLMCompileContractError(
                "actual_seconds must be a non-negative integer"
            )
        with _locked(self.lock_path):
            payload = self._read()
            entries = [
                entry
                for entry in payload["entries"]
                if entry["command_id"] == command_id
            ]
            if len(entries) != 1:
                raise VLLMCompileContractError("completion command_id is unknown")
            entry = entries[0]
            if entry["status"] != "reserved":
                raise VLLMCompileContractError(
                    "completion command_id is not reservable"
                )
            if actual_seconds > entry["planned_seconds"]:
                raise VLLMCompileContractError(
                    "actual_seconds exceeds the planned lifecycle ceiling"
                )
            previous = (
                payload["events"][-1]["event_sha256"] if payload["events"] else None
            )
            event = {
                "index": len(payload["events"]) + 1,
                "event_type": "complete",
                "command_id": command_id,
                "completed_at": completed_at,
                "actual_seconds": actual_seconds,
                "actual_cost_usd": canonical_decimal(_cost_for_seconds(actual_seconds)),
                "previous_event_sha256": previous,
            }
            event["event_sha256"] = _event_hash(event)
            entry["status"] = "completed"
            entry["completed_at"] = completed_at
            entry["actual_seconds"] = actual_seconds
            entry["actual_cost_usd"] = event["actual_cost_usd"]
            payload["events"].append(event)
            payload["revision"] += 1
            self._write(payload)
            return dict(event)

    def abort(self, command_id: str, *, aborted_at: str, reason: str) -> dict[str, Any]:
        if not isinstance(command_id, str) or not _SAFE_ID.fullmatch(command_id):
            raise VLLMCompileContractError("command_id is not a safe stable identity")
        _timestamp(aborted_at, field="aborted_at")
        if not isinstance(reason, str) or not reason:
            raise VLLMCompileContractError("abort reason must be a non-empty string")
        with _locked(self.lock_path):
            payload = self._read()
            entries = [
                entry
                for entry in payload["entries"]
                if entry["command_id"] == command_id
            ]
            if len(entries) != 1:
                raise VLLMCompileContractError("abort command_id is unknown")
            entry = entries[0]
            if entry["status"] != "reserved":
                raise VLLMCompileContractError("abort command_id is not reservable")
            previous = (
                payload["events"][-1]["event_sha256"] if payload["events"] else None
            )
            event = {
                "index": len(payload["events"]) + 1,
                "event_type": "abort",
                "command_id": command_id,
                "aborted_at": aborted_at,
                "abort_reason": reason,
                "previous_event_sha256": previous,
            }
            event["event_sha256"] = _event_hash(event)
            entry["status"] = "aborted"
            entry["aborted_at"] = aborted_at
            entry["abort_reason"] = reason
            payload["events"].append(event)
            payload["revision"] += 1
            self._write(payload)
            return dict(event)


SealedLifecycleLedger = LifecycleBudgetLedger


@dataclass(frozen=True)
class PairCurve:
    pair_id: str
    order: str
    eager_cumulative: tuple[float, ...]
    compiled_cumulative: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise VLLMCompileContractError("pair_id must be a non-empty string")
        if self.order not in ("eager-compiled", "compiled-eager"):
            raise VLLMCompileContractError(
                "pair order must be eager-compiled or compiled-eager"
            )
        eager = self._normalize_curve(self.eager_cumulative, field="eager_cumulative")
        compiled = self._normalize_curve(
            self.compiled_cumulative, field="compiled_cumulative"
        )
        object.__setattr__(self, "eager_cumulative", eager)
        object.__setattr__(self, "compiled_cumulative", compiled)

    @staticmethod
    def _normalize_curve(values: Sequence[float], *, field: str) -> tuple[float, ...]:
        if isinstance(values, (str, bytes)):
            raise VLLMCompileContractError(f"{field} must be a numeric sequence")
        result = tuple(float(value) for value in values)
        if len(result) != CONTROLLED_REQUESTS_PER_CELL:
            raise VLLMCompileContractError(
                f"{field} must contain exactly "
                f"{CONTROLLED_REQUESTS_PER_CELL} cumulative values"
            )
        previous = -math.inf
        for value in result:
            if not math.isfinite(value):
                raise VLLMCompileContractError(f"{field} must be finite")
            if value < previous:
                raise VLLMCompileContractError(
                    f"{field} must be cumulative and non-decreasing"
                )
            previous = value
        return result

    @property
    def difference_curve(self) -> tuple[float, ...]:
        return tuple(
            compiled - eager
            for eager, compiled in zip(
                self.eager_cumulative, self.compiled_cumulative, strict=True
            )
        )

    @property
    def terminal_effect_seconds(self) -> float:
        return self.difference_curve[-1]


@dataclass(frozen=True)
class PairEffect:
    pair_id: str
    order: str
    first_crossing_request_count: int | None
    sustained_crossing_request_count: int | None
    right_censored: bool
    terminal_effect_seconds: float
    mean_effect_seconds: float
    minimum_effect_seconds: float
    maximum_effect_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "order": self.order,
            "first_crossing_request_count": self.first_crossing_request_count,
            "sustained_crossing_request_count": self.sustained_crossing_request_count,
            "right_censored": self.right_censored,
            "terminal_effect_seconds": self.terminal_effect_seconds,
            "mean_effect_seconds": self.mean_effect_seconds,
            "minimum_effect_seconds": self.minimum_effect_seconds,
            "maximum_effect_seconds": self.maximum_effect_seconds,
        }


@dataclass(frozen=True)
class PairedAnalysis:
    pair_effects: tuple[PairEffect, ...]
    mean_difference_curve: tuple[float, ...]
    simultaneous_band_lower: tuple[float, ...]
    simultaneous_band_upper: tuple[float, ...]
    aggregate_first_crossing_request_count: int | None
    aggregate_sustained_crossing_request_count: int | None
    simultaneous_band_first_crossing_request_count: int | None
    simultaneous_band_sustained_crossing_request_count: int | None
    bootstrap_uncensored_resamples: int
    bootstrap_censored_resamples: int
    bootstrap_sustained_crossing_median_request_count: int | None
    bootstrap_sustained_crossing_lower_request_count: int | None
    bootstrap_sustained_crossing_upper_request_count: int | None
    bootstrap_sustained_crossing_lower_is_open: bool
    bootstrap_sustained_crossing_upper_is_open: bool
    terminal_effect_sign_flip_p_value: float
    analysis_seed: int
    resample_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "protocol_id": PROTOCOL_ID,
            "analysis_seed": self.analysis_seed,
            "resample_count": self.resample_count,
            "pair_effects": [effect.to_dict() for effect in self.pair_effects],
            "mean_difference_curve": list(self.mean_difference_curve),
            "simultaneous_band_lower": list(self.simultaneous_band_lower),
            "simultaneous_band_upper": list(self.simultaneous_band_upper),
            "aggregate_first_crossing_request_count": (
                self.aggregate_first_crossing_request_count
            ),
            "aggregate_sustained_crossing_request_count": (
                self.aggregate_sustained_crossing_request_count
            ),
            "simultaneous_band_first_crossing_request_count": (
                self.simultaneous_band_first_crossing_request_count
            ),
            "simultaneous_band_sustained_crossing_request_count": (
                self.simultaneous_band_sustained_crossing_request_count
            ),
            "bootstrap_uncensored_resamples": self.bootstrap_uncensored_resamples,
            "bootstrap_censored_resamples": self.bootstrap_censored_resamples,
            "bootstrap_sustained_crossing_median_request_count": (
                self.bootstrap_sustained_crossing_median_request_count
            ),
            "bootstrap_sustained_crossing_lower_request_count": (
                self.bootstrap_sustained_crossing_lower_request_count
            ),
            "bootstrap_sustained_crossing_upper_request_count": (
                self.bootstrap_sustained_crossing_upper_request_count
            ),
            "bootstrap_sustained_crossing_lower_is_open": (
                self.bootstrap_sustained_crossing_lower_is_open
            ),
            "bootstrap_sustained_crossing_upper_is_open": (
                self.bootstrap_sustained_crossing_upper_is_open
            ),
            "terminal_effect_sign_flip_p_value": self.terminal_effect_sign_flip_p_value,
        }


def _first_crossing(curve: Sequence[float]) -> int | None:
    for index, value in enumerate(curve, start=1):
        if value <= 0:
            return index
    return None


def _sustained_crossing(curve: Sequence[float]) -> int | None:
    suffix_ok = True
    earliest: int | None = None
    for reverse_index, value in enumerate(reversed(curve), start=1):
        if value > 0:
            suffix_ok = False
        elif suffix_ok:
            earliest = len(curve) - reverse_index + 1
        else:
            suffix_ok = False
    return earliest


def _mean_curve(curves: Sequence[Sequence[float]]) -> tuple[float, ...]:
    curve_length = len(curves[0])
    totals = [0.0] * curve_length
    for curve in curves:
        for index, value in enumerate(curve):
            totals[index] += value
    denominator = float(len(curves))
    return tuple(total / denominator for total in totals)


def _ordered_quantile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise VLLMCompileContractError("quantile requires at least one value")
    index = max(0, min(len(values) - 1, math.ceil(quantile * len(values)) - 1))
    return sorted(values)[index]


def _ordered_quantile_int(values: Sequence[int], quantile: float) -> int:
    if not values:
        raise VLLMCompileContractError("quantile requires at least one value")
    index = max(0, min(len(values) - 1, math.ceil(quantile * len(values)) - 1))
    return sorted(values)[index]


def _pair_effect(pair: PairCurve) -> PairEffect:
    difference = pair.difference_curve
    first = _first_crossing(difference)
    sustained = _sustained_crossing(difference)
    return PairEffect(
        pair_id=pair.pair_id,
        order=pair.order,
        first_crossing_request_count=first,
        sustained_crossing_request_count=sustained,
        right_censored=sustained is None,
        terminal_effect_seconds=difference[-1],
        mean_effect_seconds=sum(difference) / len(difference),
        minimum_effect_seconds=min(difference),
        maximum_effect_seconds=max(difference),
    )


def _sign_flip_p_value(terminal_effects: Sequence[float]) -> float:
    observed = abs(sum(terminal_effects) / len(terminal_effects))
    extreme = 0
    for mask in range(1 << len(terminal_effects)):
        total = 0.0
        for index, effect in enumerate(terminal_effects):
            sign = -1.0 if (mask >> index) & 1 else 1.0
            total += sign * effect
        if abs(total / len(terminal_effects)) >= observed - 1e-12:
            extreme += 1
    return extreme / float(1 << len(terminal_effects))


def analyze_pair_curves(
    pairs: Sequence[PairCurve],
    *,
    resample_count: int = BOOTSTRAP_RESAMPLES,
    analysis_seed: int = ANALYSIS_SEED,
) -> PairedAnalysis:
    if len(pairs) != PAIRS_PER_LANE:
        raise VLLMCompileContractError(f"expected exactly {PAIRS_PER_LANE} pair curves")
    if (
        isinstance(resample_count, bool)
        or not isinstance(resample_count, int)
        or resample_count <= 0
    ):
        raise VLLMCompileContractError("resample_count must be a positive integer")
    identifiers = [pair.pair_id for pair in pairs]
    if len(set(identifiers)) != len(identifiers):
        raise VLLMCompileContractError("pair_id values must be unique")
    pair_effects = tuple(_pair_effect(pair) for pair in pairs)
    difference_curves = [pair.difference_curve for pair in pairs]
    observed_mean = _mean_curve(difference_curves)
    bootstrap_max_deviation: list[float] = []
    bootstrap_sustained: list[int] = []
    bootstrap_censored = 0
    rng = random.Random(analysis_seed)
    pair_count = len(difference_curves)
    for _ in range(resample_count):
        sampled = [
            difference_curves[rng.randrange(pair_count)] for _ in range(pair_count)
        ]
        sampled_mean = _mean_curve(sampled)
        bootstrap_max_deviation.append(
            max(
                abs(sampled_mean[index] - observed_mean[index])
                for index in range(CONTROLLED_REQUESTS_PER_CELL)
            )
        )
        sustained = _sustained_crossing(sampled_mean)
        if sustained is None:
            bootstrap_censored += 1
            bootstrap_sustained.append(CONTROLLED_REQUESTS_PER_CELL + 1)
        else:
            bootstrap_sustained.append(sustained)
    band_radius = _ordered_quantile(bootstrap_max_deviation, 0.95)
    band_lower = tuple(value - band_radius for value in observed_mean)
    band_upper = tuple(value + band_radius for value in observed_mean)
    crossing_median = _ordered_quantile_int(bootstrap_sustained, 0.5)
    crossing_lower = _ordered_quantile_int(bootstrap_sustained, 0.025)
    crossing_upper = _ordered_quantile_int(bootstrap_sustained, 0.975)
    terminal_effects = [effect.terminal_effect_seconds for effect in pair_effects]
    return PairedAnalysis(
        pair_effects=pair_effects,
        mean_difference_curve=observed_mean,
        simultaneous_band_lower=band_lower,
        simultaneous_band_upper=band_upper,
        aggregate_first_crossing_request_count=_first_crossing(observed_mean),
        aggregate_sustained_crossing_request_count=_sustained_crossing(observed_mean),
        simultaneous_band_first_crossing_request_count=_first_crossing(band_upper),
        simultaneous_band_sustained_crossing_request_count=_sustained_crossing(
            band_upper
        ),
        bootstrap_uncensored_resamples=resample_count - bootstrap_censored,
        bootstrap_censored_resamples=bootstrap_censored,
        bootstrap_sustained_crossing_median_request_count=(
            crossing_median if crossing_median <= CONTROLLED_REQUESTS_PER_CELL else None
        ),
        bootstrap_sustained_crossing_lower_request_count=(
            crossing_lower if crossing_lower <= CONTROLLED_REQUESTS_PER_CELL else None
        ),
        bootstrap_sustained_crossing_upper_request_count=(
            crossing_upper if crossing_upper <= CONTROLLED_REQUESTS_PER_CELL else None
        ),
        bootstrap_sustained_crossing_lower_is_open=(
            crossing_lower > CONTROLLED_REQUESTS_PER_CELL
        ),
        bootstrap_sustained_crossing_upper_is_open=(
            crossing_upper > CONTROLLED_REQUESTS_PER_CELL
        ),
        terminal_effect_sign_flip_p_value=_sign_flip_p_value(terminal_effects),
        analysis_seed=analysis_seed,
        resample_count=resample_count,
    )


@dataclass(frozen=True)
class ClaimDecision:
    claim_id: str
    state: ClaimState
    blockers: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "state": self.state,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class ClaimGate:
    terminal: bool
    completeness: bool
    fixed_count: bool
    controlled_supported_crossing: bool
    controlled_output_identity: bool
    controlled_numeric_reproducibility: bool
    natural_output_identity: bool
    natural_numeric_reproducibility: bool
    natural_correctness: bool
    natural_supported_speedup: bool
    component_observability: bool

    def _flag_map(self) -> dict[str, bool]:
        return {
            "terminal": self.terminal,
            "completeness": self.completeness,
            "fixed_count": self.fixed_count,
            "controlled_supported_crossing": self.controlled_supported_crossing,
            "controlled_output_identity": self.controlled_output_identity,
            "controlled_numeric_reproducibility": (
                self.controlled_numeric_reproducibility
            ),
            "natural_output_identity": self.natural_output_identity,
            "natural_numeric_reproducibility": self.natural_numeric_reproducibility,
            "natural_correctness": self.natural_correctness,
            "natural_supported_speedup": self.natural_supported_speedup,
            "component_observability": self.component_observability,
        }

    def evaluate(self, claim_id: str) -> ClaimDecision:
        if claim_id == "forward-pass-identical":
            return ClaimDecision(
                claim_id=claim_id,
                state="not_applicable",
                blockers=("forward_pass_identity_is_out_of_scope",),
            )
        if claim_id not in CLAIM_REQUIREMENTS:
            raise VLLMCompileContractError(f"unknown claim_id {claim_id!r}")
        flags = self._flag_map()
        blockers = tuple(
            name for name in CLAIM_REQUIREMENTS[claim_id] if not flags[name]
        )
        state: ClaimState = "supported" if not blockers else "unsupported"
        return ClaimDecision(claim_id=claim_id, state=state, blockers=blockers)

    def matrix(self) -> tuple[ClaimDecision, ...]:
        return tuple(
            self.evaluate(claim_id)
            for claim_id in (
                "fixed-token-count-crossover",
                "output-identical-generation-crossover",
                "numerically-reproducible-generation-crossover",
                "natural-output-quality-preserved",
                "natural-end-to-end-causal-speedup",
                "compile-cuda-graph-component-timing",
                "forward-pass-identical",
            )
        )


__all__ = [
    "ABSOLUTE_CEILING_SECONDS",
    "ACTIVE_PLANNED_SECONDS",
    "ANALYSIS_SEED",
    "ANTICIPATED_RATE_USD_PER_HOUR",
    "BASE_IMAGE_REFERENCE",
    "BOOTSTRAP_RESAMPLES",
    "BudgetLifecycle",
    "BudgetLine",
    "BudgetSummary",
    "BUDGET_LIFECYCLES",
    "BUDGET_LINES",
    "BUDGET_SUMMARY",
    "CONTROLLED_CELL_ALLOWANCE_SECONDS",
    "CONTROLLED_CYCLES_PER_CELL",
    "CONTROLLED_REQUESTS_PER_CELL",
    "CONTROLLED_SIGN_SYMMETRY_ALPHA",
    "CONTROLLED_SAMPLING",
    "CONTEXT_TIERS",
    "CROSSOVER_SCHEDULE",
    "ClaimDecision",
    "ClaimGate",
    "CLAIM_REQUIREMENTS",
    "DERIVED_IMAGE_ID",
    "EXPORT_ALLOWANCE_SECONDS",
    "EXPECTED_DRIVER",
    "EXPECTED_GPU_NAME",
    "EXPECTED_MEMORY_MIB",
    "EXPECTED_MODEL_BYTES",
    "EXPECTED_MODEL_FILE_COUNT",
    "HARD_CAP_USD",
    "LANES",
    "LifecycleBudgetLedger",
    "MODEL_DIRECTORY",
    "MODEL_ID",
    "MODEL_REVISION",
    "MODES",
    "ModeContract",
    "NATURAL_CELL_ALLOWANCE_SECONDS",
    "NATURAL_REQUESTS_PER_CELL",
    "NATURAL_SAMPLING",
    "NULL_POLICY",
    "PAIR_MODE_BLOCKS",
    "PAIRS_PER_LANE",
    "PLAN_SCHEMA_VERSION",
    "PREFLIGHT_ALLOWANCE_SECONDS",
    "PROTOCOL_ID",
    "QUALITY_NONINFERIORITY_MARGIN",
    "PROVENANCE_POLICY",
    "PairedAnalysis",
    "PairCurve",
    "PairEffect",
    "REQUESTS_PER_CELL",
    "RESET_ALLOWANCE_SECONDS",
    "RESOURCE_SETTINGS",
    "RUNTIME_PINS",
    "SAMPLING_SEED",
    "SCHEDULE_SEED",
    "SIGN_FLIP_ENUMERATIONS",
    "SamplingContract",
    "ScheduleCell",
    "SealedLifecycleLedger",
    "TEARDOWN_ALLOWANCE_SECONDS",
    "UNTOUCHED_MARGIN_SECONDS",
    "VLLMCompileContractError",
    "VLLMCompilePlan",
    "VLLM_SOURCE_COMMIT",
    "WorkloadDescriptor",
    "LEDGER_SCHEMA_VERSION",
    "analyze_pair_curves",
    "build_default_plan",
    "build_plan",
    "canonical_decimal",
    "canonical_json",
    "crossover_schedule",
    "lane_first_mode_symbols",
    "lane_pair_orders",
    "lane_request_descriptors",
    "token_ids_sha256",
    "workload_descriptors",
]
