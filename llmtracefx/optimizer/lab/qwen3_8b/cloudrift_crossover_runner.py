"""Measured CloudRift runner for one approved Qwen3-8B crossover cell."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import random
import re
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import cloudrift_runner as base_runner
from .vllm_compile import (
    ANALYSIS_SEED,
    BASE_IMAGE_REFERENCE,
    DERIVED_IMAGE_ID,
    DETERMINISTIC_ENVIRONMENT,
    EXPECTED_MODEL_BYTES,
    EXPECTED_MODEL_FILE_COUNT,
    HARD_CAP_USD,
    MODEL_ID,
    MODEL_REVISION,
    NATURAL_SAMPLING,
    PROTOCOL_ID,
    RUNTIME_PINS,
    SAMPLING_SEED,
    ModeContract,
    ScheduleCell,
    VLLMCompileContractError,
    VLLMCompilePlan,
    build_default_plan,
    canonical_json,
    lane_request_descriptors,
    token_ids_sha256,
)

_DETERMINISTIC_ENV = dict(DETERMINISTIC_ENVIRONMENT)
_REQUEST_STATE_STATS_PROVENANCE = "version_pinned_vllm_0_28_request_state_stats"
_VLLM_INTERNAL_PROVENANCE = "version_pinned_vllm_0_28_internal"
_NULL_REQUEST_METRICS = (
    ("queue_seconds", "request_state_stats_has_no_queue_duration_field"),
    ("prefill_seconds", "request_state_stats_has_no_prefill_duration_field"),
    ("inference_seconds", "request_state_stats_has_no_inference_duration_field"),
    ("decode_seconds", "request_state_stats_has_no_decode_duration_field"),
    (
        "mean_time_per_output_token_seconds",
        "request_state_stats_has_no_mean_output_token_duration_field",
    ),
    ("e2e_seconds", "request_state_stats_has_no_e2e_duration_field"),
)
_OPTIONAL_COMPILATION_FIELDS = (
    "backend",
    "compile_sizes",
    "inductor_compile_config",
    "pass_config",
    "splitting_ops",
)
_MEMORY_SAMPLE_INTERVAL_SECONDS = 0.2
_MEMORY_SAMPLE_TIMEOUT_SECONDS = 5
_SAFE_PROCESS_NAME = re.compile(r"^[A-Za-z0-9._:+-]{1,128}$")


def _load_plan_cell(cell_id: str) -> tuple[VLLMCompilePlan, ScheduleCell]:
    plan = build_default_plan()
    matches = [cell for cell in plan.schedule if cell.cell_id == cell_id]
    if len(matches) != 1:
        raise VLLMCompileContractError(f"unknown crossover cell_id {cell_id!r}")
    return plan, matches[0]


def _mode_contract(plan: VLLMCompilePlan, lane: str) -> ModeContract:
    for mode_contract in plan.mode_contracts:
        if mode_contract.lane == lane:
            return mode_contract
    raise VLLMCompileContractError(f"missing mode contract for lane {lane!r}")


def _require_directory(path: Path, *, field: str) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.exists() and (resolved.is_symlink() or not resolved.is_dir()):
        raise VLLMCompileContractError(f"{field} must be a non-symlink directory path")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _cache_layout(cell: ScheduleCell, cache_root: Path) -> dict[str, str]:
    cache_root = _require_directory(cache_root, field="cache_root")
    if any(cache_root.iterdir()):
        raise VLLMCompileContractError("cache_root must be empty before cell launch")
    root = cache_root / cell.cell_id
    root.mkdir()
    paths = {
        "cache_root": str(root),
        "vllm_cache_root": str(root / "vllm"),
        "torchinductor_cache_dir": str(root / "torchinductor"),
        "triton_cache_dir": str(root / "triton"),
        "cuda_cache_path": str(root / "cuda"),
        "home": str(root / "home"),
        "hf_home": str(root / "huggingface"),
        "xdg_cache_home": str(root / "xdg"),
    }
    for value in paths.values():
        Path(value).mkdir(parents=True, exist_ok=True)
    return paths


def prepare_deterministic_environment(
    cell: ScheduleCell,
    cache_root: Path,
) -> dict[str, str]:
    cache_paths = _cache_layout(cell, cache_root)
    env_updates = {
        **_DETERMINISTIC_ENV,
        "VLLM_CACHE_ROOT": cache_paths["vllm_cache_root"],
        "TORCHINDUCTOR_CACHE_DIR": cache_paths["torchinductor_cache_dir"],
        "TRITON_CACHE_DIR": cache_paths["triton_cache_dir"],
        "CUDA_CACHE_PATH": cache_paths["cuda_cache_path"],
        "HOME": cache_paths["home"],
        "HF_HOME": cache_paths["hf_home"],
        "XDG_CACHE_HOME": cache_paths["xdg_cache_home"],
    }
    for key, value in env_updates.items():
        os.environ[key] = value
        if os.environ.get(key) != value:
            raise VLLMCompileContractError(
                f"failed to apply deterministic environment {key}={value!r}"
            )
    return {**env_updates, **cache_paths}


def _public_environment_attestation(
    cell: ScheduleCell,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    cache_root = Path(environment["cache_root"])
    cache_roles: dict[str, dict[str, str]] = {}
    for role, variable in (
        ("vllm", "VLLM_CACHE_ROOT"),
        ("torchinductor", "TORCHINDUCTOR_CACHE_DIR"),
        ("triton", "TRITON_CACHE_DIR"),
        ("cuda", "CUDA_CACHE_PATH"),
        ("home", "HOME"),
        ("huggingface", "HF_HOME"),
        ("xdg", "XDG_CACHE_HOME"),
    ):
        cache_path = Path(environment[variable])
        cache_roles[role] = {
            "env_var": variable,
            "relative_path": cache_path.relative_to(cache_root).as_posix(),
            "path_sha256": _sha256_text(str(cache_path)),
        }
    return {
        "variables": dict(_DETERMINISTIC_ENV),
        "cache_root_role": {
            "relative_identity": cell.cell_id,
            "path_sha256": _sha256_text(str(cache_root)),
        },
        "cache_roles": cache_roles,
    }


def _import_module(name: str) -> Any:
    return importlib.import_module(name)


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_safe_value(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if base_runner.math.isfinite(value) else None
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            return None
        return {
            key: _json_safe_value(item, depth=depth + 1) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item, depth=depth + 1) for item in value]
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    if hasattr(value, "__dict__"):
        return _json_safe_value(vars(value), depth=depth + 1)
    return None


def _import_runtime_stack() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    numpy_module = _import_module("numpy")
    torch_module = _import_module("torch")
    vllm_module = _import_module("vllm")
    config_module = _import_module("vllm.config")
    compilation_module = _import_module("vllm.config.compilation")
    inputs_module = _import_module("vllm.inputs")
    return (
        numpy_module,
        torch_module,
        vllm_module,
        config_module,
        compilation_module,
        inputs_module,
        vllm_module.SamplingParams,
    )


def _set_python_and_numpy_seeds(numpy_module: Any) -> None:
    random.seed(SAMPLING_SEED)
    numpy_module.random.seed(SAMPLING_SEED)


def _set_torch_determinism(torch_module: Any) -> None:
    torch_module.manual_seed(SAMPLING_SEED)
    if not hasattr(torch_module, "cuda"):
        raise VLLMCompileContractError("torch.cuda is unavailable")
    torch_module.cuda.manual_seed_all(SAMPLING_SEED)
    torch_module.use_deterministic_algorithms(True, warn_only=False)
    cudnn = getattr(torch_module.backends, "cudnn", None)
    cuda_backend = getattr(torch_module.backends, "cuda", None)
    if cudnn is None or cuda_backend is None or not hasattr(cuda_backend, "matmul"):
        raise VLLMCompileContractError("torch deterministic backend controls missing")
    cudnn.deterministic = True
    cudnn.benchmark = False
    cudnn.allow_tf32 = False
    cuda_backend.matmul.allow_tf32 = False
    if hasattr(torch_module, "set_float32_matmul_precision"):
        torch_module.set_float32_matmul_precision("highest")
    if hasattr(torch_module, "are_deterministic_algorithms_enabled"):
        if not torch_module.are_deterministic_algorithms_enabled():
            raise VLLMCompileContractError(
                "torch deterministic algorithms are not enabled"
            )
    if cudnn.deterministic is not True or cudnn.benchmark is not False:
        raise VLLMCompileContractError("torch cudnn deterministic controls drifted")
    if cudnn.allow_tf32 is not False or cuda_backend.matmul.allow_tf32 is not False:
        raise VLLMCompileContractError("torch TF32 controls drifted")


def _prompt_key(descriptor: Any) -> str:
    return f"{descriptor.context_tier}/{descriptor.workload_id}"


def _typed_measurement(
    *,
    value: float | int | None,
    unit: str,
    clock_domain: str,
    provenance: str,
    null_reason: str,
) -> dict[str, Any]:
    observed_value: float | None = None
    if not isinstance(value, bool) and isinstance(value, (int, float)):
        candidate = float(value)
        if candidate > 0 and base_runner.math.isfinite(candidate):
            observed_value = candidate
    return {
        "value": observed_value,
        "unit": unit,
        "clock_domain": clock_domain,
        "provenance": provenance,
        "observability_state": (
            "observed" if observed_value is not None else "unobservable"
        ),
        "null_reason": None if observed_value is not None else null_reason,
    }


class _MemorySeriesSampler:
    """Collect a same-process timestamped whole-device memory series."""

    def __init__(self) -> None:
        self._origin_ns = time.perf_counter_ns()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._lock = threading.Lock()
        self._samples: list[dict[str, int]] = []
        self._errors: list[str] = []

    def _observe(self) -> None:
        completed = subprocess.run(
            (
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ),
            check=True,
            capture_output=True,
            text=True,
            shell=False,
            timeout=_MEMORY_SAMPLE_TIMEOUT_SECONDS,
        )
        memory_used_mib = int(completed.stdout.strip())
        if memory_used_mib < 0:
            raise ValueError("GPU memory observation is negative")
        sample = {
            "offset_ns": time.perf_counter_ns() - self._origin_ns,
            "memory_used_mib": memory_used_mib,
        }
        with self._lock:
            self._samples.append(sample)

    def _observe_or_record_error(self) -> None:
        try:
            self._observe()
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            with self._lock:
                self._errors.append(type(exc).__name__)

    def _sample(self) -> None:
        while not self._stop.wait(_MEMORY_SAMPLE_INTERVAL_SECONDS):
            self._observe_or_record_error()

    def start(self) -> None:
        self._observe()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=_MEMORY_SAMPLE_TIMEOUT_SECONDS + 1)
        if self._thread.is_alive():
            with self._lock:
                self._errors.append("sampling_thread_timeout")
        else:
            self._observe_or_record_error()

    @property
    def peak_mib(self) -> int | None:
        with self._lock:
            if not self._samples:
                return None
            return max(sample["memory_used_mib"] for sample in self._samples)

    def receipt(self) -> dict[str, Any]:
        with self._lock:
            samples = [dict(sample) for sample in self._samples]
            errors = list(self._errors)
        return {
            "value": samples if samples else None,
            "unit": "MiB",
            "clock_domain": "same_process_perf_counter_offset_ns",
            "provenance": "sampled_nvidia_smi",
            "observability_state": "observed" if samples else "unobservable",
            "null_reason": None if samples else "nvidia_smi_memory_series_unavailable",
            "target_interval_ms": int(_MEMORY_SAMPLE_INTERVAL_SECONDS * 1000),
            "sampling_error_count": len(errors),
            "sampling_error_types": errors,
        }


def _process_tree_receipt() -> dict[str, Any]:
    proc = Path("/proc")
    if not proc.is_dir():
        return {
            "nodes": None,
            "clock_domain": "runner_process_snapshot",
            "provenance": "linux_procfs_stat",
            "observability_state": "unobservable",
            "null_reason": "linux_procfs_unavailable",
        }
    processes: dict[int, tuple[int, str]] = {}
    try:
        paths = sorted(
            (path for path in proc.iterdir() if path.name.isdecimal()),
            key=lambda path: int(path.name),
        )
        if len(paths) > 4096:
            raise VLLMCompileContractError("process snapshot exceeds PID limit")
        for path in paths:
            try:
                stat = (path / "stat").read_text(encoding="utf-8")
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
            close = stat.rfind(")")
            open_ = stat.find("(")
            if open_ < 0 or close <= open_:
                raise VLLMCompileContractError("procfs process stat is malformed")
            fields = stat[close + 1 :].split()
            if len(fields) < 2:
                raise VLLMCompileContractError("procfs process stat is incomplete")
            pid = int(path.name)
            ppid = int(fields[1])
            name = stat[open_ + 1 : close]
            safe_name = (
                name
                if _SAFE_PROCESS_NAME.fullmatch(name)
                else "sha256:" + hashlib.sha256(name.encode()).hexdigest()
            )
            processes[pid] = (ppid, safe_name)
    except (OSError, UnicodeError, ValueError) as exc:
        raise VLLMCompileContractError("could not record process tree") from exc

    root = os.getpid()
    if root not in processes:
        raise VLLMCompileContractError("runner process is absent from process snapshot")
    selected = {root}
    while True:
        added = {
            pid
            for pid, (ppid, _) in processes.items()
            if ppid in selected and pid not in selected
        }
        if not added:
            break
        selected.update(added)
    children: dict[int, list[int]] = {pid: [] for pid in selected}
    for pid in selected:
        if pid != root:
            children[processes[pid][0]].append(pid)
    nodes: list[dict[str, Any]] = []
    indices = {root: 0}
    queue = [root]
    while queue:
        pid = queue.pop(0)
        parent_pid = processes[pid][0]
        nodes.append(
            {
                "node_index": indices[pid],
                "parent_node_index": None if pid == root else indices[parent_pid],
                "process_name": processes[pid][1],
            }
        )
        ordered_children = sorted(
            children[pid],
            key=lambda child: (processes[child][1], child),
        )
        for child in ordered_children:
            indices[child] = len(indices)
            queue.append(child)
    nodes.sort(key=lambda node: node["node_index"])
    return {
        "nodes": nodes,
        "clock_domain": "runner_process_snapshot",
        "provenance": "linux_procfs_stat",
        "observability_state": "observed",
        "null_reason": None,
    }


def _null_measurement(
    *,
    unit: str,
    clock_domain: str,
    provenance: str,
    null_reason: str,
    observability_state: str = "unobservable",
) -> dict[str, Any]:
    return {
        "value": None,
        "unit": unit,
        "clock_domain": clock_domain,
        "provenance": provenance,
        "observability_state": observability_state,
        "null_reason": null_reason,
    }


def _request_state_stats_ttft(metrics: Any) -> float | None:
    candidate = getattr(metrics, "first_token_latency", None)
    if isinstance(candidate, (int, float)) and candidate > 0:
        return float(candidate)
    return None


def _request_metric_adapter(metrics: Any) -> dict[str, Any]:
    values: dict[str, Any] = {}
    ttft = _request_state_stats_ttft(metrics)
    values["ttft_seconds"] = _typed_measurement(
        value=ttft,
        unit="seconds",
        clock_domain="request_output_metrics",
        provenance=_REQUEST_STATE_STATS_PROVENANCE,
        null_reason="request_state_stats_first_token_latency_unavailable",
    )
    for name, null_reason in _NULL_REQUEST_METRICS:
        values[name] = _null_measurement(
            unit="seconds",
            clock_domain="request_output_metrics",
            provenance=_REQUEST_STATE_STATS_PROVENANCE,
            null_reason=null_reason,
        )
    return values


def _optional_compilation_time(
    *,
    value: Any,
    compiled: bool,
    null_reason: str,
) -> dict[str, Any]:
    if not compiled:
        return _null_measurement(
            unit="seconds",
            clock_domain="vllm_internal_runtime",
            provenance=_VLLM_INTERNAL_PROVENANCE,
            null_reason="not_applicable_eager_mode",
            observability_state="not_applicable",
        )
    return _typed_measurement(
        value=float(value) if isinstance(value, (int, float)) else None,
        unit="seconds",
        clock_domain="vllm_internal_runtime",
        provenance=_VLLM_INTERNAL_PROVENANCE,
        null_reason=null_reason,
    )


def _optional_config_snapshot(
    llm: Any,
    *,
    compiled: bool,
) -> dict[str, Any]:
    config = llm.llm_engine.vllm_config
    compilation = config.compilation_config
    optional_fields = {
        field: _json_safe_value(getattr(compilation, field, None))
        for field in _OPTIONAL_COMPILATION_FIELDS
    }
    observed_optional = any(value is not None for value in optional_fields.values())
    encoder_field_name = "encoder_compilation_config"
    raw_encoder_value = getattr(config, encoder_field_name, None)
    encoder_value = _json_safe_value(raw_encoder_value)
    return {
        "compiled_mode_expected": compiled,
        "compilation_config_fields": {
            "value": optional_fields if observed_optional else None,
            "unit": "json",
            "clock_domain": "resolved_runtime_config",
            "provenance": _VLLM_INTERNAL_PROVENANCE,
            "observability_state": (
                "observed" if observed_optional else "unobservable"
            ),
            "null_reason": (
                None if observed_optional else "optional_compilation_fields_not_exposed"
            ),
        },
        "encoder_compilation_config": {
            "field_name": encoder_field_name,
            "value": encoder_value,
            "unit": "json",
            "clock_domain": "resolved_runtime_config",
            "provenance": _VLLM_INTERNAL_PROVENANCE,
            "observability_state": (
                "observed" if encoder_value is not None else "unobservable"
            ),
            "null_reason": (
                None
                if encoder_value is not None
                else "encoder_compilation_config_not_exposed"
            ),
        },
        "compilation_time_seconds": _optional_compilation_time(
            value=getattr(compilation, "compilation_time", None),
            compiled=compiled,
            null_reason="compilation_time_not_exposed_by_vllm_0_28",
        ),
        "encoder_compilation_time_seconds": _optional_compilation_time(
            value=getattr(raw_encoder_value, "compilation_time", None),
            compiled=compiled,
            null_reason="encoder_compilation_time_not_exposed_by_vllm_0_28",
        ),
        "cuda_graph_capture_duration_seconds": (
            _null_measurement(
                unit="seconds",
                clock_domain="vllm_internal_runtime",
                provenance=_VLLM_INTERNAL_PROVENANCE,
                null_reason="cuda_graph_capture_duration_not_exposed_by_vllm",
            )
            if compiled
            else _null_measurement(
                unit="seconds",
                clock_domain="vllm_internal_runtime",
                provenance=_VLLM_INTERNAL_PROVENANCE,
                null_reason="not_applicable_eager_mode",
                observability_state="not_applicable",
            )
        ),
        "cuda_graph_dispatch_counter": (
            _null_measurement(
                unit="requests",
                clock_domain="vllm_metrics_registry",
                provenance="documented_vllm_0_28_metric",
                null_reason=(
                    "offline_llm_has_no_stable_cuda_graph_dispatch_metric_snapshot_hook"
                ),
            )
            if compiled
            else _null_measurement(
                unit="requests",
                clock_domain="vllm_metrics_registry",
                provenance="documented_vllm_0_28_metric",
                null_reason="not_applicable_eager_mode",
                observability_state="not_applicable",
            )
        ),
    }


def _gpu_commitment(
    hardware: Mapping[str, Any],
    experiment_nonce: str,
) -> dict[str, Any]:
    if not isinstance(experiment_nonce, str) or not experiment_nonce:
        raise VLLMCompileContractError("experiment nonce must be a non-empty string")
    private_uuid_hash = hardware.get("gpu_uuid_sha256")
    if not isinstance(private_uuid_hash, str):
        raise VLLMCompileContractError("hardware helper did not expose gpu_uuid_sha256")
    return {
        "gpu_name": hardware["gpu_name"],
        "gpu_count": hardware["gpu_count"],
        "driver_version": hardware["driver_version"],
        "memory_total_mib": hardware["memory_total_mib"],
        "memory_used_mib": hardware["memory_used_mib"],
        "public_experiment_nonce": experiment_nonce,
        "gpu_identity_commitment": base_runner._sha256_json(
            {
                "public_experiment_nonce": experiment_nonce,
                "private_gpu_uuid_sha256": private_uuid_hash,
            }
        ),
    }


def _request_indices(index: int) -> dict[str, int]:
    cycle_index = ((index - 1) // 12) + 1
    base_ordinal = ((index - 1) % 12) + 1
    return {
        "cycle_index": cycle_index,
        "base_ordinal": base_ordinal,
        "request_sequence_index": index,
    }


def _verify_terminal_shape(
    cell: ScheduleCell,
    request_record: Mapping[str, Any],
) -> None:
    output_ids = request_record["output_token_ids"]
    finish_reason = request_record["finish_reason"]
    if cell.lane == "controlled":
        if len(output_ids) != 96:
            raise VLLMCompileContractError(
                "controlled lane must return exactly 96 output token IDs"
            )
        if finish_reason != "length":
            raise VLLMCompileContractError(
                "controlled lane finish_reason must be exactly 'length'"
            )
    else:
        if finish_reason not in {"stop", "length"}:
            raise VLLMCompileContractError(
                "natural lane finish_reason must be stop or length"
            )
        decoded = request_record.get("decoded_output")
        if not isinstance(decoded, str):
            raise VLLMCompileContractError(
                "natural lane must retain decoded text output"
            )
        if len(decoded.encode("utf-8")) > base_runner.MAX_OUTPUT_BYTES:
            raise VLLMCompileContractError("decoded output exceeds bound")


def _write_progress(
    progress_path: Path,
    *,
    cell: ScheduleCell,
    request_count_expected: int,
    requests: Sequence[dict[str, Any]],
) -> None:
    payload = base_runner._seal(
        {
            "schema_version": "2",
            "protocol_id": PROTOCOL_ID,
            "cell_id": cell.cell_id,
            "lane": cell.lane,
            "mode": cell.mode,
            "request_count_expected": request_count_expected,
            "request_count_completed": len(requests),
            "last_request_sequence_index": len(requests),
            "requests": list(requests),
            "terminal": False,
        },
        "progress_sha256",
    )
    base_runner._atomic_json(progress_path, payload)


def _sampling_params_kwargs(contract: ModeContract) -> dict[str, Any]:
    parameters = contract.sampling.to_dict()
    if parameters.pop("best_of") != 1:
        raise VLLMCompileContractError("effective best_of contract must remain one")
    return parameters


def _llm_kwargs(
    *,
    cell: ScheduleCell,
    mode_contract: ModeContract,
    maximum_model_len: int,
    model_path: Path,
    config_module: Any,
    compilation_module: Any,
) -> dict[str, Any]:
    compiled = cell.mode == "compiled"
    return {
        "model": str(model_path),
        "trust_remote_code": False,
        "dtype": "bfloat16",
        "max_model_len": maximum_model_len,
        "tensor_parallel_size": 1,
        "max_num_seqs": 1,
        "gpu_memory_utilization": 0.94,
        "enable_prefix_caching": False,
        "disable_custom_all_reduce": True,
        "disable_log_stats": False,
        "seed": mode_contract.sampling.seed,
        "enforce_eager": not compiled,
        "speculative_config": None,
        "compilation_config": config_module.CompilationConfig(
            mode=(
                compilation_module.CompilationMode.VLLM_COMPILE
                if compiled
                else compilation_module.CompilationMode.NONE
            ),
            cudagraph_mode=(
                compilation_module.CUDAGraphMode.FULL_AND_PIECEWISE
                if compiled
                else compilation_module.CUDAGraphMode.NONE
            ),
        ),
    }


def _request_record(
    *,
    cell: ScheduleCell,
    descriptor: Any,
    request_index: int,
    prompt_ids: Sequence[int],
    completion: Any,
    response: Any,
    started_ns: int,
    ended_ns: int,
    initialization_started_ns: int,
) -> dict[str, Any]:
    output_ids = list(completion.token_ids)
    latency_seconds = (ended_ns - started_ns) / 1_000_000_000
    record = {
        **descriptor.to_dict(),
        **_request_indices(request_index),
        "cell_id": cell.cell_id,
        "pair_id": cell.pair_id,
        "lane": cell.lane,
        "mode": cell.mode,
        "input_token_count": len(prompt_ids),
        "input_token_ids_sha256": base_runner._sha256_json(list(prompt_ids)),
        "output_token_count": len(output_ids),
        "output_token_ids": output_ids,
        "output_token_ids_sha256": token_ids_sha256(output_ids),
        "finish_reason": completion.finish_reason,
        "timing": {
            "latency_seconds": _typed_measurement(
                value=latency_seconds,
                unit="seconds",
                clock_domain="same_process_perf_counter",
                provenance="measured_perf_counter_ns",
                null_reason="same_process_perf_counter_unavailable",
            ),
            "cumulative_from_initialization_seconds": _typed_measurement(
                value=(ended_ns - initialization_started_ns) / 1_000_000_000,
                unit="seconds",
                clock_domain="same_process_perf_counter",
                provenance="measured_perf_counter_ns",
                null_reason="same_process_perf_counter_unavailable",
            ),
            "latency_perf_counter_ns": ended_ns - started_ns,
            "cumulative_from_initialization_perf_counter_ns": (
                ended_ns - initialization_started_ns
            ),
            "output_token_rate_tokens_per_second": _typed_measurement(
                value=(
                    len(output_ids) / latency_seconds if latency_seconds > 0 else None
                ),
                unit="tokens_per_second",
                clock_domain="same_process_perf_counter",
                provenance="derived_exact_token_count_over_perf_counter_latency",
                null_reason="perf_counter_latency_nonpositive",
            ),
        },
        "metrics": _request_metric_adapter(getattr(response, "metrics", object())),
        "terminal": True,
    }
    if cell.lane == "natural":
        record["decoded_output"] = completion.text
    return record


def run_cell(
    cell_id: str,
    *,
    model_path: Path,
    state_path: Path,
    cache_root: Path,
    output: Path,
    experiment_nonce: str,
) -> None:
    plan, cell = _load_plan_cell(cell_id)
    mode_contract = _mode_contract(plan, cell.lane)
    if mode_contract.sampling != NATURAL_SAMPLING and cell.lane == "natural":
        raise VLLMCompileContractError("natural lane sampling contract drifted")
    private_environment = prepare_deterministic_environment(cell, cache_root)
    runtime = base_runner._verify_runtime()
    hardware = base_runner._hardware()
    staging = base_runner._read_json(state_path / base_runner.STAGING_FILE)
    prompts = base_runner._read_json(state_path / base_runner.PROMPT_FILE)
    base_runner._verify_seal(staging, "receipt_sha256")
    base_runner._verify_seal(prompts, "prompt_ids_sha256")
    base_runner._verify_staging_binding(staging, prompts, model_path)
    prompt_ids_by_key = prompts["prompts"]
    if not isinstance(prompt_ids_by_key, dict) or not prompt_ids_by_key:
        raise VLLMCompileContractError("prompt receipt is incomplete")
    descriptors = lane_request_descriptors(cell.lane)
    maximum_prompt_len = 0
    for descriptor in descriptors:
        ids = prompt_ids_by_key.get(_prompt_key(descriptor))
        if (
            not isinstance(ids, list)
            or not ids
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in ids
            )
        ):
            raise VLLMCompileContractError("prompt receipt contains invalid token IDs")
        maximum_prompt_len = max(maximum_prompt_len, len(ids))
    maximum_model_len = maximum_prompt_len + mode_contract.sampling.max_tokens

    (
        numpy_module,
        torch_module,
        vllm_module,
        config_module,
        compilation_module,
        inputs_module,
        sampling_params_type,
    ) = _import_runtime_stack()
    _set_python_and_numpy_seeds(numpy_module)
    _set_torch_determinism(torch_module)

    initialization_started_ns = time.perf_counter_ns()
    sampler = _MemorySeriesSampler()
    sampler.start()
    llm = None
    try:
        llm = vllm_module.LLM(
            **_llm_kwargs(
                cell=cell,
                mode_contract=mode_contract,
                maximum_model_len=maximum_model_len,
                model_path=model_path,
                config_module=config_module,
                compilation_module=compilation_module,
            )
        )
        resolved = base_runner._resolved(llm, cell.mode == "compiled")
        initialization_ready_ns = time.perf_counter_ns()
        requests: list[dict[str, Any]] = []
        sampling = sampling_params_type(**_sampling_params_kwargs(mode_contract))
        tokens_prompt_type = inputs_module.TokensPrompt
        progress_path = output.with_name(f".{output.stem}-progress.json")
        for request_index, descriptor in enumerate(descriptors, start=1):
            prompt_ids = list(prompt_ids_by_key[_prompt_key(descriptor)])
            started_ns = time.perf_counter_ns()
            generated = llm.generate(
                [tokens_prompt_type(prompt_token_ids=prompt_ids)],
                sampling,
                use_tqdm=False,
            )
            ended_ns = time.perf_counter_ns()
            if len(generated) != 1 or not generated[0].finished:
                raise VLLMCompileContractError("request did not complete")
            response = generated[0]
            if len(response.outputs) != 1:
                raise VLLMCompileContractError("request returned multiple completions")
            completion = response.outputs[0]
            record = _request_record(
                cell=cell,
                descriptor=descriptor,
                request_index=request_index,
                prompt_ids=prompt_ids,
                completion=completion,
                response=response,
                started_ns=started_ns,
                ended_ns=ended_ns,
                initialization_started_ns=initialization_started_ns,
            )
            _verify_terminal_shape(cell, record)
            requests.append(record)
            _write_progress(
                progress_path,
                cell=cell,
                request_count_expected=mode_contract.requests_per_cell,
                requests=requests,
            )
        if len(requests) != mode_contract.requests_per_cell:
            raise VLLMCompileContractError(
                f"cell did not complete {mode_contract.requests_per_cell} requests"
            )
    finally:
        sampler.stop()

    if llm is None:
        raise VLLMCompileContractError("LLM failed to initialize")

    terminal_payload = base_runner._seal(
        {
            "schema_version": "2",
            "protocol_id": PROTOCOL_ID,
            "cell": cell.to_dict(),
            "plan_sha256": plan.content_sha256,
            "analysis_seed": ANALYSIS_SEED,
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "expected_file_count": EXPECTED_MODEL_FILE_COUNT,
                "expected_bytes": EXPECTED_MODEL_BYTES,
                "state_receipt": base_runner.STAGING_FILE,
                "prompt_receipt": base_runner.PROMPT_FILE,
            },
            "budget": {
                "hard_cap_usd": str(HARD_CAP_USD),
            },
            "runtime": {
                "pins": runtime,
                "expected_pins": RUNTIME_PINS,
                "runtime_image": {
                    "base_reference": BASE_IMAGE_REFERENCE,
                    "derived_image_id": DERIVED_IMAGE_ID,
                },
                "resolved_execution_config": resolved,
                "optional_version_pinned_fields": _optional_config_snapshot(
                    llm,
                    compiled=cell.mode == "compiled",
                ),
            },
            "deterministic_environment": _public_environment_attestation(
                cell,
                private_environment,
            ),
            "hardware_commitment": _gpu_commitment(hardware, experiment_nonce),
            "process_tree": _process_tree_receipt(),
            "measurements": {
                "initialization_seconds": _typed_measurement(
                    value=(initialization_ready_ns - initialization_started_ns)
                    / 1_000_000_000,
                    unit="seconds",
                    clock_domain="same_process_perf_counter",
                    provenance="measured_perf_counter_ns",
                    null_reason="initialization_timing_unavailable",
                ),
                "initialization_perf_counter_ns": (
                    initialization_ready_ns - initialization_started_ns
                ),
                "peak_gpu_memory_mib": _typed_measurement(
                    value=sampler.peak_mib,
                    unit="MiB",
                    clock_domain="sampled_nvidia_smi",
                    provenance="sampled_nvidia_smi",
                    null_reason="nvidia_smi_peak_memory_unavailable",
                ),
                "gpu_memory_series": sampler.receipt(),
            },
            "request_count_expected": mode_contract.requests_per_cell,
            "request_count_observed": len(requests),
            "prompt_ids_sha256": prompts["prompt_ids_sha256"],
            "staging_receipt_sha256": staging["receipt_sha256"],
            "requests": requests,
            "terminal": True,
        },
        "cell_sha256",
    )
    if "gpu_uuid_sha256" in canonical_json(terminal_payload):
        raise VLLMCompileContractError(
            "terminal payload leaked a raw GPU UUID derivative"
        )
    base_runner._atomic_json(output, terminal_payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_cell_parser = subparsers.add_parser("run-cell")
    run_cell_parser.add_argument("--cell-id", required=True)
    run_cell_parser.add_argument("--model-path", required=True, type=Path)
    run_cell_parser.add_argument("--state-path", required=True, type=Path)
    run_cell_parser.add_argument("--cache-root", required=True, type=Path)
    run_cell_parser.add_argument("--output", required=True, type=Path)
    run_cell_parser.add_argument("--experiment-nonce", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "run-cell":
        raise SystemExit("unsupported command")
    run_cell(
        args.cell_id,
        model_path=args.model_path,
        state_path=args.state_path,
        cache_root=args.cache_root,
        output=args.output,
        experiment_nonce=args.experiment_nonce,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
