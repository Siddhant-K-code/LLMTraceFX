"""Container-side measured runner for the Modal L4 crossover delta.

Everything measurable here is the CloudRift crossover runner's code,
imported rather than copied: the deterministic environment, the memory
series sampler, the frozen ``_llm_kwargs``, the per-request record, the
terminal-shape check, and the process-tree receipt. Only two things are
different, and both are honest differences rather than reimplementations.

The first is the hardware gate. ``cloudrift_runner._hardware`` admits one
exact RTX 4090 with one exact driver, which an L4 can never satisfy, so
this module has its own L4 gate that pins the accelerator name and count
and records the provider-managed driver instead of pinning it.

The second is that a Modal Function returns a value instead of leaving a
file on a host. Ordinary and out-of-memory failures are converted into
terminal refusal receipts so the orchestrator can tear down and publish a
refusal, rather than losing the reason inside a provider stack trace.

This module imports no provider SDK. It is imported inside the container
by the app module, and imported directly by the offline tests.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import cloudrift_crossover_runner as cell_runner
from . import cloudrift_runner as base_runner
from .modal_l4_crossover import (
    CONTAINER_CACHE_ROOT,
    DECODE_STEPS,
    EXPECTED_GPU_NAME,
    GPU_COUNT,
    MIN_TOTAL_VRAM_MIB,
    MODEL_MOUNT_PATH,
    PROTOCOL_ID,
    STATE_MOUNT_PATH,
    VRAM_HEADROOM_MIB,
    ModalL4ContractError,
    build_default_plan,
    max_model_len,
    runtime_image_identity,
)
from .vllm_compile import (
    EXPECTED_MODEL_BYTES,
    EXPECTED_MODEL_FILE_COUNT,
    MODEL_DIRECTORY,
    MODEL_ID,
    MODEL_REVISION,
    RUNTIME_PINS,
    VLLMCompileContractError,
    canonical_json,
    crossover_schedule,
    lane_request_descriptors,
)

RECEIPT_SCHEMA_VERSION = "1"
STAGE_DOWNLOAD_FILE = "modal-download-receipt.json"
GPU_QUERY_FIELDS = ("name", "driver_version", "memory.total", "memory.used", "uuid")
HARDWARE_COMMAND_TIMEOUT_SECONDS = 20
# vLLM 0.28.0 reports a length-capped completion with this exact finish reason
# when the controlled sampler forces the full decode budget. Pinned shape.
FINISH_REASON_LENGTH = "length"


def _now() -> str:
    return base_runner._now()


def container_identity() -> str | None:
    """Return a salted digest of the provider task identity, or null.

    Two measured cells that report the same identity were not two fresh
    single-use containers, which is the one second-attempt signal a
    returned receipt can actually carry. The raw identity never leaves
    this function.
    """

    raw = (os.environ.get("MODAL_TASK_ID") or "").strip()
    if not raw:
        return None
    return "sha256:" + hashlib.sha256(f"{PROTOCOL_ID}:{raw}".encode()).hexdigest()


def _refusal(
    kind: str,
    *,
    reason: str,
    detail: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a sealed terminal refusal receipt for a failed provider call."""

    payload: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "kind": kind,
        "status": "refused",
        "terminal": True,
        "observed_at": _now(),
        "reason": reason,
        "detail": detail,
        "container_identity_sha256": container_identity(),
        **(dict(extra) if extra else {}),
    }
    return base_runner._seal(payload, "receipt_sha256")


_FAILURE_KIND = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
# Allowlisted, stable failure categories. Anything unrecognized collapses to
# "unexpected_error"; no free-form text is ever derived from the exception.
_FAILURE_CATEGORIES = (
    "out_of_memory",
    "contract_violation",
    "timeout",
    "subprocess_error",
    "io_error",
    "value_error",
    "runtime_error",
    "unexpected_error",
)


def _failure_category(exc: BaseException) -> str:
    """Return one allowlisted, stable category for a failure."""

    if _is_out_of_memory(exc):
        return "out_of_memory"
    if isinstance(exc, (ModalL4ContractError, VLLMCompileContractError)):
        return "contract_violation"
    if isinstance(exc, subprocess.SubprocessError):
        return "subprocess_error"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, OSError):
        return "io_error"
    if isinstance(exc, ValueError):
        return "value_error"
    if isinstance(exc, RuntimeError):
        return "runtime_error"
    return "unexpected_error"


def _failure_detail(exc: BaseException) -> str:
    """Return only the stable exception class name for a failure.

    The exception message is deliberately discarded: it can carry a private
    path, a URL, a provider identifier, or a credential substring. Only the
    class name (a stable Python identifier), validated against a strict
    identifier charset, is persisted; the category is recorded separately.
    """

    kind = type(exc).__name__
    return kind if _FAILURE_KIND.fullmatch(kind) else "SanitizedError"


def _is_out_of_memory(exc: BaseException) -> bool:
    text = f"{type(exc).__name__} {exc}".lower()
    return "out of memory" in text or "outofmemory" in text or "oom" in text


def model_path() -> Path:
    return Path(MODEL_MOUNT_PATH) / MODEL_DIRECTORY


def state_path() -> Path:
    return Path(STATE_MOUNT_PATH)


def l4_hardware() -> dict[str, Any]:
    """Return the observed accelerator identity, pinned to one L4.

    The driver version is recorded, never pinned: Modal manages it and a
    pin the operator cannot control would turn an ordinary provider
    upgrade into a fake scientific failure. The device UUID never leaves
    this function in raw form.
    """

    completed = subprocess.run(
        (
            "nvidia-smi",
            f"--query-gpu={','.join(GPU_QUERY_FIELDS)}",
            "--format=csv,noheader,nounits",
        ),
        check=True,
        capture_output=True,
        text=True,
        shell=False,
        timeout=HARDWARE_COMMAND_TIMEOUT_SECONDS,
    )
    lines = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    if len(lines) != GPU_COUNT:
        raise ModalL4ContractError(
            f"expected exactly {GPU_COUNT} accelerator, observed {len(lines)}"
        )
    fields = [item.strip() for item in lines[0].split(",")]
    if len(fields) != len(GPU_QUERY_FIELDS):
        raise ModalL4ContractError("GPU identity is incomplete")
    name, driver, total, used, gpu_uuid = fields
    if name != EXPECTED_GPU_NAME:
        raise ModalL4ContractError(
            f"accelerator is {name!r}, not the approved {EXPECTED_GPU_NAME!r}"
        )
    if int(total) < MIN_TOTAL_VRAM_MIB:
        raise ModalL4ContractError("accelerator memory is below the approved device")
    if not gpu_uuid.startswith("GPU-"):
        raise ModalL4ContractError("GPU identity is incomplete")
    return {
        "gpu_name": name,
        "gpu_count": GPU_COUNT,
        "driver_version": driver,
        "driver_pinned": False,
        "memory_total_mib": int(total),
        "memory_used_mib": int(used),
        "gpu_uuid_sha256": "sha256:" + hashlib.sha256(gpu_uuid.encode()).hexdigest(),
    }


def _prompt_receipts() -> tuple[dict[str, Any], dict[str, Any]]:
    staging = base_runner._read_json(state_path() / base_runner.STAGING_FILE)
    prompts = base_runner._read_json(state_path() / base_runner.PROMPT_FILE)
    base_runner._verify_seal(staging, "receipt_sha256")
    base_runner._verify_seal(prompts, "prompt_ids_sha256")
    base_runner._verify_staging_binding(staging, prompts, model_path())
    return staging, prompts


def longest_controlled_prompt(prompts: Mapping[str, Any]) -> tuple[str, list[int]]:
    """Return the key and token array of the longest sealed prompt."""

    ids_by_key = prompts.get("prompts")
    if not isinstance(ids_by_key, dict) or not ids_by_key:
        raise ModalL4ContractError("prompt receipt is incomplete")
    keys = {
        cell_runner._prompt_key(descriptor)
        for descriptor in lane_request_descriptors("controlled")
    }
    candidates: list[tuple[int, str]] = []
    for key in sorted(keys):
        ids = ids_by_key.get(key)
        if (
            not isinstance(ids, list)
            or not ids
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in ids
            )
        ):
            raise ModalL4ContractError("prompt receipt contains invalid token IDs")
        candidates.append((len(ids), key))
    length, key = max(candidates)
    del length
    return key, list(ids_by_key[key])


def stage_model(*, volume_committer: Any = None) -> dict[str, Any]:
    """Download the pinned revision onto the run-scoped volume. CPU only."""

    started_at = _now()
    try:
        from huggingface_hub import snapshot_download

        target = model_path()
        target.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=str(target),
            token=False,
        )
        files = [
            path
            for path in sorted(target.rglob("*"))
            if path.is_file() and ".cache/huggingface/" not in path.as_posix()
        ]
        total_bytes = sum(path.stat().st_size for path in files)
        if (
            len(files) != EXPECTED_MODEL_FILE_COUNT
            or total_bytes != EXPECTED_MODEL_BYTES
        ):
            return _refusal(
                "modal_stage",
                reason="staged model inventory differs from the pinned revision",
                detail=(
                    f"observed {len(files)} files and {total_bytes} bytes; "
                    f"expected {EXPECTED_MODEL_FILE_COUNT} and {EXPECTED_MODEL_BYTES}"
                ),
            )
        state_path().mkdir(parents=True, exist_ok=True)
        receipt = base_runner._seal(
            {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "kind": "modal_stage",
                "status": "completed",
                "provider": "modal",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "model_directory": MODEL_DIRECTORY,
                "model_file_count": len(files),
                "model_bytes": total_bytes,
                "started_at": started_at,
                "ended_at": _now(),
                "terminal": True,
            },
            "receipt_sha256",
        )
        base_runner._atomic_json(state_path() / STAGE_DOWNLOAD_FILE, receipt)
        if volume_committer is not None:
            volume_committer()
        return receipt
    except Exception as exc:  # noqa: BLE001 - a refusal receipt beats a stack trace
        return _refusal(
            "modal_stage",
            reason="staging failed",
            detail=_failure_detail(exc),
            extra={"failure_category": _failure_category(exc)},
        )


def verify_and_seal(*, volume_committer: Any = None) -> dict[str, Any]:
    """Verify the staged bytes and seal the frozen prompt token arrays."""

    started_at = _now()
    try:
        target = model_path()
        inventory = base_runner._inventory(target)
        prompts, ids = base_runner._tokenize(target)
        prompt_payload = base_runner._seal(
            {"schema_version": "1", "prompts": ids}, "prompt_ids_sha256"
        )
        base_runner._atomic_json(state_path() / base_runner.PROMPT_FILE, prompt_payload)
        longest_tokens = max(len(value) for value in ids.values())
        staging = base_runner._seal(
            {
                "schema_version": "1",
                "provider": "modal",
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "model_file_count": len(inventory),
                "model_bytes": sum(item["size_bytes"] for item in inventory),
                "inventory": inventory,
                "prompts": prompts,
                "prompt_ids_sha256": prompt_payload["prompt_ids_sha256"],
                "runtime": base_runner._verify_runtime(),
                "runtime_image": runtime_image_identity(),
                "started_at": started_at,
                "ended_at": _now(),
                "terminal": True,
            },
            "receipt_sha256",
        )
        base_runner._atomic_json(state_path() / base_runner.STAGING_FILE, staging)
        if volume_committer is not None:
            volume_committer()
        return base_runner._seal(
            {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "kind": "modal_verify",
                "status": "completed",
                "model_file_count": len(inventory),
                "model_bytes": sum(item["size_bytes"] for item in inventory),
                "prompt_ids_sha256": prompt_payload["prompt_ids_sha256"],
                "staging_receipt_sha256": staging["receipt_sha256"],
                "longest_prompt_tokens": longest_tokens,
                "max_model_len": max_model_len(longest_tokens),
                "started_at": started_at,
                "ended_at": _now(),
                "terminal": True,
            },
            "receipt_sha256",
        )
    except Exception as exc:  # noqa: BLE001 - a refusal receipt beats a stack trace
        return _refusal(
            "modal_verify",
            reason="staging verification failed",
            detail=_failure_detail(exc),
            extra={"failure_category": _failure_category(exc)},
        )


def _canary_cell(mode: str) -> Any:
    for cell in crossover_schedule():
        if cell.lane == "controlled" and cell.mode == mode:
            return cell
    raise ModalL4ContractError(f"no controlled cell exists for mode {mode!r}")


def run_canary(
    mode: str,
    *,
    experiment_nonce: str,
    cache_root: str | None = None,
) -> dict[str, Any]:
    """Run one isolated memory-gate canary and report an observation.

    The runner arguments come from the unchanged crossover ``_llm_kwargs``
    so the canary measures the configuration the cells will use, not a
    smaller one that would pass more easily. The canary carries the same
    nonce-bound hardware commitment the measured cells do, so the result
    path can prove driver/runtime/hardware continuity from both canaries
    through every cell without any receipt exposing a raw GPU UUID.
    """

    if mode not in ("eager", "compiled"):
        return _refusal("modal_canary", reason=f"unknown canary mode {mode!r}")
    started_at = _now()
    sampler = None
    try:
        _, prompts = _prompt_receipts()
        key, prompt_ids = longest_controlled_prompt(prompts)
        cell = _canary_cell(mode)
        plan = build_default_plan()
        del plan
        root = Path(cache_root or CONTAINER_CACHE_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        cell_runner.prepare_deterministic_environment(cell, root)
        hardware = l4_hardware()
        hardware_commitment = cell_runner._gpu_commitment(hardware, experiment_nonce)
        runtime = base_runner._verify_runtime()
        mode_contract = cell_runner._mode_contract(
            cell_runner._load_plan_cell(cell.cell_id)[0], "controlled"
        )
        maximum_model_len = max_model_len(len(prompt_ids))
        (
            numpy_module,
            torch_module,
            vllm_module,
            config_module,
            compilation_module,
            inputs_module,
            sampling_params_type,
        ) = cell_runner._import_runtime_stack()
        cell_runner._set_python_and_numpy_seeds(numpy_module)
        cell_runner._set_torch_determinism(torch_module)
        sampler = cell_runner._MemorySeriesSampler()
        sampler.start()
        started_ns = time.perf_counter_ns()
        llm = vllm_module.LLM(
            **cell_runner._llm_kwargs(
                cell=cell,
                mode_contract=mode_contract,
                maximum_model_len=maximum_model_len,
                model_path=model_path(),
                config_module=config_module,
                compilation_module=compilation_module,
            )
        )
        resolved = base_runner._resolved(llm, mode == "compiled")
        sampling = sampling_params_type(
            **cell_runner._sampling_params_kwargs(mode_contract)
        )
        generated = llm.generate(
            [inputs_module.TokensPrompt(prompt_token_ids=list(prompt_ids))],
            sampling,
            use_tqdm=False,
        )
        ended_ns = time.perf_counter_ns()
        sampler.stop()
        if len(generated) != 1 or not generated[0].finished:
            return _refusal(
                "modal_canary",
                reason="canary did not return exactly one finished request",
                extra={"mode": mode},
            )
        if len(generated[0].outputs) != 1:
            return _refusal(
                "modal_canary",
                reason="canary returned more than one completion",
                extra={"mode": mode},
            )
        completion = generated[0].outputs[0]
        output_ids = list(completion.token_ids)
        generated_tokens = len(output_ids)
        finish_reason = completion.finish_reason
        if generated_tokens != DECODE_STEPS or finish_reason != FINISH_REASON_LENGTH:
            return _refusal(
                "modal_canary",
                reason=(
                    "canary did not reach the pinned terminal shape of "
                    f"{DECODE_STEPS} tokens finishing on length"
                ),
                extra={
                    "mode": mode,
                    "generated_tokens": generated_tokens,
                    "finish_reason": finish_reason,
                },
            )
        if len(prompt_ids) + DECODE_STEPS != maximum_model_len:
            return _refusal(
                "modal_canary",
                reason="canary context length is not the full longest request",
                extra={"mode": mode},
            )
        kv_tokens, kv_blocks = _kv_capacity(llm)
        observation = {
            "mode": mode,
            "gpu_name": hardware["gpu_name"],
            "gpu_count": hardware["gpu_count"],
            "runtime_pins": dict(runtime),
            "total_vram_mib": hardware["memory_total_mib"],
            "peak_vram_mib": sampler.peak_mib,
            "kv_cache_blocks": kv_blocks,
            "kv_cache_tokens": kv_tokens,
            "max_model_len": maximum_model_len,
            "out_of_memory": False,
            "generated_tokens": generated_tokens,
            "finish_reason": finish_reason,
            "terminal": generated_tokens == DECODE_STEPS
            and finish_reason == FINISH_REASON_LENGTH,
            "used_longest_controlled_prompt": True,
            "runner_kwargs": {
                "dtype": "bfloat16",
                "tensor_parallel_size": 1,
                "max_num_seqs": 1,
                "gpu_memory_utilization": "0.94",
                "enable_prefix_caching": False,
                "speculative_config": None,
                "enforce_eager": mode == "eager",
                "max_model_len": maximum_model_len,
            },
        }
        return base_runner._seal(
            {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "kind": "modal_canary",
                "status": "completed",
                "mode": mode,
                "container_identity_sha256": container_identity(),
                "prompt_key": key,
                "prompt_token_count": len(prompt_ids),
                "headroom_mib_required": VRAM_HEADROOM_MIB,
                "resolved_execution_config": resolved,
                "expected_runtime_pins": dict(RUNTIME_PINS),
                "hardware_commitment": hardware_commitment,
                "runtime_image": runtime_image_identity(),
                "observation": observation,
                "memory_series": sampler.receipt(),
                "elapsed_seconds": (ended_ns - started_ns) / 1_000_000_000,
                "started_at": started_at,
                "ended_at": _now(),
                "terminal": True,
            },
            "receipt_sha256",
        )
    except Exception as exc:  # noqa: BLE001 - a refusal receipt beats a stack trace
        peak = sampler.peak_mib if sampler is not None else None
        if sampler is not None:
            sampler.stop()
        return _refusal(
            "modal_canary",
            reason=(
                "canary exhausted device memory"
                if _is_out_of_memory(exc)
                else "canary failed"
            ),
            detail=_failure_detail(exc),
            extra={
                "mode": mode,
                "out_of_memory": _is_out_of_memory(exc),
                "peak_vram_mib": peak,
                "failure_category": _failure_category(exc),
            },
        )


def _kv_capacity(llm: Any) -> tuple[int, int]:
    """Return observed KV capacity in tokens and blocks, or fail closed.

    The memory admission gate is only meaningful if the KV cache the engine
    actually allocated can be read back. vLLM 0.28.0 exposes it on the
    engine's ``cache_config`` (``num_gpu_blocks`` and ``block_size``); these
    are version-pinned internal fields, labelled as such. When the pinned
    access path is absent the canary refuses rather than reporting a null
    capacity that would silently weaken the gate.
    """

    for attribute in ("llm_engine", "engine"):
        engine = getattr(llm, attribute, None)
        cache_config = getattr(getattr(engine, "cache_config", None), "__dict__", None)
        if not isinstance(cache_config, dict):
            continue
        blocks = cache_config.get("num_gpu_blocks")
        block_size = cache_config.get("block_size")
        if isinstance(blocks, bool) or isinstance(block_size, bool):
            continue
        if isinstance(blocks, int) and isinstance(block_size, int) and blocks > 0:
            return blocks * block_size, blocks
    raise ModalL4ContractError(
        "vLLM 0.28.0 KV cache capacity is not readable on the pinned "
        "engine.cache_config path; refusing to admit without it"
    )


@contextmanager
def _l4_hardware_gate() -> Iterator[None]:
    """Install the L4 accelerator observer for one delegated ``run_cell``.

    ``cloudrift_crossover_runner.run_cell`` calls ``base_runner._hardware``
    on the shared ``cloudrift_runner`` module, whose gate admits exactly one
    RTX 4090. That module is part of the committed CloudRift evidence and is
    never edited; instead this context manager swaps in the L4 observer for
    the scope of the call and restores the original afterwards, even on
    error, so CloudRift's default behaviour is unchanged everywhere else.
    ``base_runner`` here and inside the crossover runner are the same module
    object, so the swap is what the delegated call sees.
    """

    original = base_runner._hardware
    base_runner._hardware = l4_hardware
    try:
        yield
    finally:
        base_runner._hardware = original


def run_measured_cell(
    cell_id: str,
    *,
    experiment_nonce: str,
    cache_root: str | None = None,
    output_root: str | None = None,
) -> dict[str, Any]:
    """Execute one sealed cell and return its sealed terminal receipt.

    The measurement itself is ``cloudrift_crossover_runner.run_cell``, run
    unchanged and never copied. The one honest difference is the hardware
    gate: the CloudRift runner admits exactly one RTX 4090, which an L4 can
    never satisfy, so for the duration of the delegated call this module
    installs its own L4 observer in place of the shared
    ``cloudrift_runner._hardware`` gate and restores it afterwards. The
    CloudRift source is not modified and its default behaviour is untouched;
    the L4 observer preserves the same receipt shape, including the
    nonce-bound hardware commitment, so continuity holds without any raw GPU
    UUID leaving the container.
    """

    started_at = _now()
    try:
        cell = next(
            (item for item in crossover_schedule() if item.cell_id == cell_id), None
        )
        if cell is None:
            return _refusal(
                "modal_cell",
                reason=f"cell {cell_id!r} is not in the sealed schedule",
            )
        hardware = l4_hardware()
        root = Path(cache_root or CONTAINER_CACHE_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        outputs = Path(output_root or "/run-output")
        outputs.mkdir(parents=True, exist_ok=True)
        output = outputs / f"{cell_id}.json"
        with _l4_hardware_gate():
            cell_runner.run_cell(
                cell_id,
                model_path=model_path(),
                state_path=state_path(),
                cache_root=root,
                output=output,
                experiment_nonce=experiment_nonce,
            )
        payload = json.loads(output.read_text(encoding="utf-8"))
        if "gpu_uuid_sha256" in canonical_json(payload):
            return _refusal(
                "modal_cell",
                reason="cell receipt leaked a raw GPU UUID derivative",
                extra={"cell_id": cell_id},
            )
        return base_runner._seal(
            {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "kind": "modal_cell",
                "status": "completed",
                "cell_id": cell_id,
                "container_identity_sha256": container_identity(),
                "provider_hardware": {
                    key: value
                    for key, value in hardware.items()
                    if key != "gpu_uuid_sha256"
                },
                "runtime_image": runtime_image_identity(),
                "cell_receipt": payload,
                "started_at": started_at,
                "ended_at": _now(),
                "terminal": True,
            },
            "receipt_sha256",
        )
    except (
        VLLMCompileContractError,
        ModalL4ContractError,
        OSError,
        ValueError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as exc:
        return _refusal(
            "modal_cell",
            reason=(
                "cell exhausted device memory"
                if _is_out_of_memory(exc)
                else "cell failed"
            ),
            detail=_failure_detail(exc),
            extra={
                "cell_id": cell_id,
                "out_of_memory": _is_out_of_memory(exc),
                "failure_category": _failure_category(exc),
            },
        )
    except BaseException as exc:  # noqa: BLE001 - refuse, then let teardown run
        return _refusal(
            "modal_cell",
            reason="cell failed with an unexpected error",
            detail=_failure_detail(exc),
            extra={
                "cell_id": cell_id,
                "failure_category": _failure_category(exc),
            },
        )


def analysis_inventory(cell_ids: Sequence[str]) -> dict[str, Any]:
    """Return a sanitized inventory of the run's container-side state."""

    started_at = _now()
    try:
        staging, prompts = _prompt_receipts()
        return base_runner._seal(
            {
                "schema_version": RECEIPT_SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "kind": "modal_analysis",
                "status": "completed",
                "expected_cell_ids": sorted(cell_ids),
                "sealed_cell_ids": sorted(
                    cell.cell_id for cell in crossover_schedule()
                ),
                "staging_receipt_sha256": staging["receipt_sha256"],
                "prompt_ids_sha256": prompts["prompt_ids_sha256"],
                "model_file_count": staging["model_file_count"],
                "model_bytes": staging["model_bytes"],
                "statistical_publication": (
                    "delegated to the existing results core; never computed here"
                ),
                "started_at": started_at,
                "ended_at": _now(),
                "terminal": True,
            },
            "receipt_sha256",
        )
    except Exception as exc:  # noqa: BLE001 - a refusal receipt beats a stack trace
        return _refusal(
            "modal_analysis",
            reason="analysis inventory failed",
            detail=_failure_detail(exc),
            extra={"failure_category": _failure_category(exc)},
        )


def container_environment_is_clean() -> bool:
    """Return whether the container inherited no credential-shaped variable."""

    return not any(
        name.startswith("MODAL_TOKEN") or name.endswith("_API_KEY")
        for name, value in os.environ.items()
        if value
    )
