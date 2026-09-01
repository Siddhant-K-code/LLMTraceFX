"""Safety-gated orchestration for the pinned M5 Pro Qwen3.8 lab."""

from __future__ import annotations

import hashlib
import html
import json
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from statistics import mean
from typing import Any

from ..collectors._shared import atomic_write_text, config_hash
from ..collectors.mlx import MLXVLMRuntime, mlx_collection_contract_hash
from ..compare.compare import compare
from ..compare.policy import CompareConstraints, CompareObjective, ComparePolicy
from ..compare.report_html import render_compare_report_html
from ..schema import ExperimentRecord, MetricProvenance, utc_now_iso
from ..tune.policy import TuneConstraints, TuneObjective, TunePolicy
from ..tune.report_html import render_tune_report_html
from ..tune.tuner import tune
from ..workloads.aggregate import correct_cases_per_minute
from ..workloads.catalog import workload_by_id
from ..workloads.materialize import MaterializedPrompt, materialize_prompt
from ..workloads.matrix import (
    DECODE_MODE_AUTOREGRESSIVE,
    MatrixEntry,
)
from ..workloads.schema import CONTEXT_TIER_TARGET_TOKENS, ContextTier
from ..workloads.verify import (
    RowResult,
    RowStatus,
    RowVerification,
    RunBinding,
    execute_row,
)
from .manifest import LabManifest, LabManifestError

LAB_REPORT_SCHEMA_VERSION = "1"
MODEL_VERIFICATION_SCHEMA_VERSION = "1"
LAB_STATE_SCHEMA_VERSION = "1"

_PRIVATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"/Users/[^/\s]+"),
    re.compile(r"/home/[^/\s]+"),
    re.compile(r"[A-Za-z]:\\Users\\[^\\\s]+"),
    re.compile(r"\b(?:hf[_-]|sk-)[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?<![:/\w])(?:/|\.{1,2}/|[A-Za-z]:\\)[^\s'\"<>]+"),
)


class LabError(RuntimeError):
    """Raised when a lab action cannot proceed safely."""


@dataclass(frozen=True)
class HostSnapshot:
    collected_at: str
    os_name: str
    os_release: str
    architecture: str
    python_implementation: str
    python_version: str
    cpu_count: int | None
    chip: str | None
    total_memory_bytes: int | None
    memory_free_percent: float | None
    swap_used_bytes: int | None
    disk_free_bytes: int
    package_versions: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SafetyDecision:
    safe: bool
    blockers: tuple[str, ...]
    snapshot: HostSnapshot

    def to_dict(self) -> dict[str, Any]:
        return {
            "safe": self.safe,
            "blockers": list(self.blockers),
            "snapshot": self.snapshot.to_dict(),
        }


def _run_text(argv: list[str]) -> str | None:
    try:
        return subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            shell=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _parse_byte_quantity(value: str) -> int | None:
    match = re.search(r"([0-9]+(?:\.[0-9]+)?)([KMG])", value)
    if match is None:
        return None
    factors = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(match.group(1)) * factors[match.group(2)])


def collect_host_snapshot(path: Path) -> HostSnapshot:
    disk_probe = path
    while not disk_probe.exists() and disk_probe != disk_probe.parent:
        disk_probe = disk_probe.parent
    memory_raw = _run_text(["sysctl", "-n", "hw.memsize"])
    chip = _run_text(["sysctl", "-n", "machdep.cpu.brand_string"])
    pressure = _run_text(["memory_pressure"])
    swap = _run_text(["sysctl", "-n", "vm.swapusage"])
    free_match = (
        re.search(r"System-wide memory free percentage:\s*([0-9.]+)%", pressure)
        if pressure
        else None
    )
    swap_match = re.search(r"\bused\s*=\s*([0-9.]+[KMG])", swap) if swap else None
    versions: dict[str, str] = {}
    for package in (
        "llmtracefx",
        "mlx",
        "mlx-lm",
        "mlx-vlm",
        "transformers",
        "huggingface-hub",
    ):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            continue
    return HostSnapshot(
        collected_at=utc_now_iso(),
        os_name=platform.system(),
        os_release=platform.release(),
        architecture=platform.machine(),
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        cpu_count=(
            None
            if (cpu_count := _run_text(["sysctl", "-n", "hw.ncpu"])) is None
            else int(cpu_count)
        ),
        chip=chip,
        total_memory_bytes=(
            int(memory_raw) if memory_raw and memory_raw.isdigit() else None
        ),
        memory_free_percent=(
            float(free_match.group(1)) if free_match is not None else None
        ),
        swap_used_bytes=(
            _parse_byte_quantity(swap_match.group(1))
            if swap_match is not None
            else None
        ),
        disk_free_bytes=shutil.disk_usage(disk_probe).free,
        package_versions=dict(sorted(versions.items())),
    )


def assess_safety(
    manifest: LabManifest,
    path: Path,
    *,
    include_download: bool,
) -> SafetyDecision:
    snapshot = collect_host_snapshot(path)
    blockers: list[str] = []
    required_free = manifest.safety.minimum_free_disk_after_download_bytes
    if include_download:
        required_free += manifest.model.expected_download_bytes
    if snapshot.disk_free_bytes < required_free:
        blockers.append(
            f"disk free {snapshot.disk_free_bytes} bytes is below required "
            f"{required_free} bytes"
        )
    if snapshot.os_name != "Darwin" or snapshot.architecture != "arm64":
        blockers.append("the selected MLX lab requires Apple Silicon macOS")
    if snapshot.chip != manifest.safety.required_chip:
        blockers.append(
            f"chip is {snapshot.chip or 'unavailable'}, expected "
            f"{manifest.safety.required_chip}"
        )
    if snapshot.total_memory_bytes is None:
        blockers.append("total physical memory could not be measured")
    elif snapshot.total_memory_bytes != manifest.safety.required_total_memory_bytes:
        blockers.append(
            f"physical memory is {snapshot.total_memory_bytes} bytes, expected "
            f"{manifest.safety.required_total_memory_bytes} bytes"
        )
    if snapshot.memory_free_percent is None:
        blockers.append("current memory headroom could not be measured")
    elif snapshot.memory_free_percent < manifest.safety.minimum_memory_free_percent:
        blockers.append(
            f"memory free {snapshot.memory_free_percent:g}% is below "
            f"{manifest.safety.minimum_memory_free_percent:g}%"
        )
    if snapshot.swap_used_bytes is None:
        blockers.append("current swap usage could not be measured")
    elif snapshot.swap_used_bytes > manifest.safety.maximum_swap_used_bytes:
        blockers.append(
            f"swap used {snapshot.swap_used_bytes} bytes exceeds "
            f"{manifest.safety.maximum_swap_used_bytes} bytes"
        )
    expected_versions = {
        "mlx": manifest.runtime.mlx_version,
        "mlx-lm": manifest.runtime.mlx_lm_version,
        manifest.runtime.name: manifest.runtime.version,
        "transformers": manifest.runtime.transformers_version,
    }
    for package, expected in expected_versions.items():
        observed = snapshot.package_versions.get(package)
        if observed != expected:
            blockers.append(
                f"{package} version is {observed or 'not installed'}, expected {expected}"
            )
    return SafetyDecision(
        safe=not blockers, blockers=tuple(blockers), snapshot=snapshot
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(manifest: LabManifest, model_path: Path) -> dict[str, Any]:
    failures: list[str] = []
    files: list[dict[str, Any]] = []
    expected_paths = {pin.path for pin in manifest.model.files}
    if model_path.is_dir():
        unexpected = sorted(
            path.name
            for path in model_path.iterdir()
            if path.name != ".cache" and path.name not in expected_paths
        )
        failures.extend(f"unpinned model-root entry: {name}" for name in unexpected)
    for pin in manifest.model.files:
        path = model_path / pin.path
        if not path.is_file() or path.is_symlink():
            failures.append(f"missing regular model file: {pin.path}")
            continue
        size = path.stat().st_size
        if size != pin.size_bytes:
            failures.append(f"{pin.path} size {size} does not match {pin.size_bytes}")
            continue
        digest = _sha256_file(path)
        if digest != pin.sha256:
            failures.append(f"{pin.path} sha256 {digest} does not match {pin.sha256}")
            continue
        files.append({"path": pin.path, "size_bytes": size, "sha256": digest})
    result = {
        "schema_version": MODEL_VERIFICATION_SCHEMA_VERSION,
        "repository_id": manifest.model.repository_id,
        "revision": manifest.model.revision,
        "verified": not failures,
        "failures": failures,
        "files": files,
    }
    if failures:
        raise LabError("model verification failed: " + "; ".join(failures))
    return result


def acquire_model(
    manifest: LabManifest, *, model_path: Path, workspace: Path
) -> dict[str, Any]:
    workspace_decision = assess_safety(manifest, workspace, include_download=False)
    model_decision = assess_safety(manifest, model_path.parent, include_download=True)
    blockers = tuple(
        dict.fromkeys(workspace_decision.blockers + model_decision.blockers)
    )
    if blockers:
        raise LabError("download blocked by safety preflight: " + "; ".join(blockers))
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise LabError("huggingface-hub is required; install the `mlx` extra") from exc
    downloaded = Path(
        snapshot_download(
            repo_id=manifest.model.repository_id,
            revision=manifest.model.revision,
            local_dir=model_path,
            allow_patterns=[pin.path for pin in manifest.model.files],
        )
    )
    result = verify_model(manifest, downloaded)
    workspace.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        workspace / "model-verification.json",
        json.dumps(result, indent=2, sort_keys=False) + "\n",
    )
    return result


def model_files_present(manifest: LabManifest, model_path: Path) -> bool:
    return all(
        (model_path / pin.path).is_file()
        and not (model_path / pin.path).is_symlink()
        and (model_path / pin.path).stat().st_size == pin.size_bytes
        for pin in manifest.model.files
    )


def verify_catalog(manifest: LabManifest) -> None:
    expected_tiers = {tier.value: tier for tier in ContextTier}
    for tier in manifest.context_tiers:
        enum_tier = expected_tiers.get(tier.name)
        if enum_tier is None:
            raise LabManifestError(
                f"context tier {tier.name!r} is not supported by the workload catalog"
            )
        if CONTEXT_TIER_TARGET_TOKENS[enum_tier] != tier.target_tokens:
            raise LabManifestError(
                f"context tier {tier.name!r} target changed from pinned "
                f"{tier.target_tokens} to {CONTEXT_TIER_TARGET_TOKENS[enum_tier]}"
            )
    for pin in manifest.workloads:
        try:
            workload = workload_by_id(pin.workload_id)
        except KeyError as exc:
            raise LabManifestError(
                f"pinned workload {pin.workload_id!r} is not in the catalog"
            ) from exc
        if workload.version != pin.version:
            raise LabManifestError(
                f"workload {pin.workload_id!r} version changed from "
                f"{pin.version} to {workload.version}"
            )
        for tier_name, expected_hash in pin.prompt_hashes.items():
            materialized = materialize_prompt(workload, expected_tiers[tier_name])
            if materialized.prompt_hash != expected_hash:
                raise LabManifestError(
                    f"workload {pin.workload_id!r} {tier_name} prompt hash "
                    f"changed from {expected_hash} to {materialized.prompt_hash}"
                )


def _entry(
    manifest: LabManifest,
    *,
    workload_id: str,
    tier: ContextTier,
    repetition_index: int,
    prompt: MaterializedPrompt,
    prompt_path: Path,
    output_dir: Path,
    warmup: bool,
) -> MatrixEntry:
    label = "warmup" if warmup else "rep"
    run_id = f"{workload_id}-{tier.value}-{label}-{repetition_index + 1:02d}"
    return MatrixEntry(
        run_id=run_id,
        workload_id=workload_id,
        workload_version=workload_by_id(workload_id).version,
        category=workload_by_id(workload_id).category.value,
        context_tier=tier.value,
        decode_mode=DECODE_MODE_AUTOREGRESSIVE,
        configured_depth=None,
        prompt=prompt,
        prompt_path=str(prompt_path.resolve()),
        runner_results_dir=str((output_dir / "runner" / run_id).resolve()),
        collector_output_dir=str(
            (output_dir / "runs" / run_id / "collection").resolve()
        ),
        runnable=True,
        unsupported_reason=None,
        command_argv=("llmtracefx-m5-lab", "run"),
        max_tokens=manifest.generation.max_output_tokens,
    )


def _binding(
    manifest: LabManifest,
    model_path: Path,
    *,
    repetition_index: int,
    hardware_fingerprint: str,
    measured_repetitions: int | None = None,
) -> RunBinding:
    return RunBinding(
        target_model_path=model_path,
        seed=manifest.generation.seed,
        temperature=manifest.generation.temperature,
        top_p=manifest.generation.top_p,
        enable_thinking=manifest.generation.enable_thinking,
        prefill_step_size=manifest.runtime.prefill_step_size,
        model_revision=manifest.model.revision,
        tokenizer_revision=manifest.model.revision,
        quantization=manifest.model.quantization,
        model_family=manifest.model.model_family,
        timeout_seconds=manifest.cooperative_timeout_seconds,
        warmup_repetitions=manifest.repetitions.warmup_per_tier,
        measured_repetitions=(
            measured_repetitions
            if measured_repetitions is not None
            else manifest.repetitions.measured_per_workload
        ),
        repetition_index=repetition_index,
        hardware_fingerprint=hardware_fingerprint,
    )


def _write_prompt(path: Path, prompt: MaterializedPrompt) -> None:
    atomic_write_text(path, prompt.text)


def _peak_bytes(result: RowResult) -> float | None:
    if result.final_record is None or result.final_record.memory.peak is None:
        return None
    return result.final_record.memory.peak.value


def _tier_is_safe(
    manifest: LabManifest,
    results: tuple[RowResult, ...],
    postflight: SafetyDecision,
) -> tuple[bool, tuple[str, ...]]:
    blockers = list(postflight.blockers)
    for result in results:
        if (
            manifest.safety.stop_on_any_failed_row
            and result.verification.status
            not in (RowStatus.COMPLETED, RowStatus.SKIPPED)
        ):
            blockers.append(
                f"{result.entry.run_id} ended as " f"{result.verification.status.value}"
            )
        peak = _peak_bytes(result)
        if peak is None:
            blockers.append(f"{result.entry.run_id} has no observed MLX peak memory")
        elif peak > manifest.safety.maximum_peak_memory_bytes:
            blockers.append(
                f"{result.entry.run_id} peak memory {peak:.0f} exceeds "
                f"{manifest.safety.maximum_peak_memory_bytes} bytes"
            )
    return not blockers, tuple(blockers)


def run_lab(
    manifest: LabManifest,
    *,
    workspace: Path,
    model_path: Path,
    max_tier: str,
    resume: bool,
) -> dict[str, Any]:
    verify_catalog(manifest)
    verify_model(manifest, model_path)
    preflight = assess_safety(manifest, workspace, include_download=False)
    if not preflight.safe:
        raise LabError(
            "run blocked by safety preflight: " + "; ".join(preflight.blockers)
        )

    selected_max = manifest.tier(max_tier)
    tiers = tuple(
        tier for tier in manifest.context_tiers if tier.order <= selected_max.order
    )
    hardware_fingerprint = config_hash(
        {
            "os_name": preflight.snapshot.os_name,
            "os_release": preflight.snapshot.os_release,
            "architecture": preflight.snapshot.architecture,
            "python_implementation": preflight.snapshot.python_implementation,
            "python_version": preflight.snapshot.python_version,
            "cpu_count": preflight.snapshot.cpu_count,
            "chip": preflight.snapshot.chip,
            "total_memory_bytes": preflight.snapshot.total_memory_bytes,
            "package_versions": preflight.snapshot.package_versions,
        }
    )

    state: dict[str, Any] = {
        "schema_version": LAB_STATE_SCHEMA_VERSION,
        "lab_id": manifest.lab_id,
        "started_at": utc_now_iso(),
        "status": "running",
        "requested_max_tier": max_tier,
        "preflight": preflight.to_dict(),
        "tiers": [],
        "stopped_before_tier": None,
        "stopped_during": None,
        "unattempted_tiers": [],
        "stop_reasons": [],
    }
    state["execution_id"] = config_hash(
        {
            "lab_id": manifest.lab_id,
            "started_at": state["started_at"],
            "hardware_fingerprint": hardware_fingerprint,
        }
    )
    workspace.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        workspace / "state.json",
        json.dumps(state, indent=2, sort_keys=False) + "\n",
    )

    runtime = MLXVLMRuntime(
        temperature=manifest.generation.temperature,
        top_p=manifest.generation.top_p,
        enable_thinking=manifest.generation.enable_thinking,
        prefill_step_size=manifest.runtime.prefill_step_size,
    )

    def runtime_factory() -> MLXVLMRuntime:
        return runtime

    results_root = workspace / "results"

    for tier_pin in tiers:
        tier = ContextTier(tier_pin.name)
        tier_preflight = assess_safety(manifest, workspace, include_download=False)
        if not tier_preflight.safe:
            state["stopped_before_tier"] = tier.value
            state["unattempted_tiers"] = [
                candidate.name
                for candidate in manifest.context_tiers
                if tier_pin.order <= candidate.order <= selected_max.order
            ]
            state["stop_reasons"] = list(tier_preflight.blockers)
            break

        warmup_results: list[RowResult] = []
        if manifest.repetitions.warmup_per_tier:
            warmup_pin = manifest.workloads[0]
            warmup_workload = workload_by_id(warmup_pin.workload_id)
            prompt = materialize_prompt(warmup_workload, tier)
            prompt_path = (
                workspace
                / "prompts"
                / (f"{warmup_workload.workload_id}-{tier.value}.txt")
            )
            _write_prompt(prompt_path, prompt)
            for index in range(manifest.repetitions.warmup_per_tier):
                warmup_entry = _entry(
                    manifest,
                    workload_id=warmup_workload.workload_id,
                    tier=tier,
                    repetition_index=index,
                    prompt=prompt,
                    prompt_path=prompt_path,
                    output_dir=workspace / "warmups" / tier.value,
                    warmup=True,
                )
                warmup_binding = _binding(
                    manifest,
                    model_path,
                    repetition_index=index,
                    hardware_fingerprint=hardware_fingerprint,
                    measured_repetitions=max(manifest.repetitions.warmup_per_tier, 1),
                )
                result = execute_row(
                    warmup_entry,
                    manifest_dir=workspace,
                    output_dir=workspace / "warmups" / tier.value,
                    model_id=manifest.model.repository_id,
                    binding=warmup_binding,
                    resume=False,
                    runtime_factory=runtime_factory,
                )
                warmup_results.append(result)
                if (
                    manifest.safety.stop_on_any_failed_row
                    and result.verification.status
                    not in (RowStatus.COMPLETED, RowStatus.SKIPPED)
                ):
                    break
        if any(
            result.verification.status not in (RowStatus.COMPLETED, RowStatus.SKIPPED)
            for result in warmup_results
        ):
            reasons = [
                result.verification.reason
                or f"{result.entry.run_id} ended as {result.verification.status.value}"
                for result in warmup_results
                if result.verification.status
                not in (RowStatus.COMPLETED, RowStatus.SKIPPED)
            ]
            state["tiers"].append(
                {
                    "tier": tier.value,
                    "status": "failed_warmup",
                    "run_ids": [result.entry.run_id for result in warmup_results],
                    "maximum_observed_peak_memory_bytes": max(
                        (
                            peak
                            for result in warmup_results
                            if (peak := _peak_bytes(result)) is not None
                        ),
                        default=None,
                    ),
                    "postflight": assess_safety(
                        manifest, workspace, include_download=False
                    ).to_dict(),
                    "blockers": reasons,
                }
            )
            state["stopped_during"] = {
                "tier": tier.value,
                "phase": "warmup",
            }
            state["stopped_before_tier"] = None
            state["unattempted_tiers"] = [
                later.name
                for later in manifest.context_tiers
                if later.order > tier_pin.order and later.order <= selected_max.order
            ]
            state["stop_reasons"] = reasons
            break
        warmup_postflight = assess_safety(manifest, workspace, include_download=False)
        warmup_safe, warmup_blockers = _tier_is_safe(
            manifest, tuple(warmup_results), warmup_postflight
        )
        if not warmup_safe:
            state["tiers"].append(
                {
                    "tier": tier.value,
                    "status": "failed_warmup_safety",
                    "run_ids": [result.entry.run_id for result in warmup_results],
                    "maximum_observed_peak_memory_bytes": max(
                        (
                            peak
                            for result in warmup_results
                            if (peak := _peak_bytes(result)) is not None
                        ),
                        default=None,
                    ),
                    "postflight": warmup_postflight.to_dict(),
                    "blockers": list(warmup_blockers),
                }
            )
            state["stopped_during"] = {
                "tier": tier.value,
                "phase": "warmup_safety",
            }
            state["stopped_before_tier"] = None
            state["unattempted_tiers"] = [
                later.name
                for later in manifest.context_tiers
                if later.order > tier_pin.order and later.order <= selected_max.order
            ]
            state["stop_reasons"] = list(warmup_blockers)
            break

        tier_results: list[RowResult] = []
        for workload_pin in manifest.workloads:
            workload = workload_by_id(workload_pin.workload_id)
            prompt = materialize_prompt(workload, tier)
            prompt_path = (
                workspace / "prompts" / (f"{workload.workload_id}-{tier.value}.txt")
            )
            _write_prompt(prompt_path, prompt)
            for index in range(manifest.repetitions.measured_per_workload):
                entry = _entry(
                    manifest,
                    workload_id=workload.workload_id,
                    tier=tier,
                    repetition_index=index,
                    prompt=prompt,
                    prompt_path=prompt_path,
                    output_dir=results_root,
                    warmup=False,
                )
                result = execute_row(
                    entry,
                    manifest_dir=workspace,
                    output_dir=results_root,
                    model_id=manifest.model.repository_id,
                    binding=_binding(
                        manifest,
                        model_path,
                        repetition_index=index,
                        hardware_fingerprint=hardware_fingerprint,
                    ),
                    resume=resume,
                    runtime_factory=runtime_factory,
                )
                tier_results.append(result)
                if (
                    manifest.safety.stop_on_any_failed_row
                    and result.verification.status
                    not in (RowStatus.COMPLETED, RowStatus.SKIPPED)
                ):
                    break
            if (
                tier_results
                and manifest.safety.stop_on_any_failed_row
                and tier_results[-1].verification.status
                not in (RowStatus.COMPLETED, RowStatus.SKIPPED)
            ):
                break

        postflight = assess_safety(manifest, workspace, include_download=False)
        safe, blockers = _tier_is_safe(manifest, tuple(tier_results), postflight)
        state["tiers"].append(
            {
                "tier": tier.value,
                "status": "passed" if safe else "failed",
                "run_ids": [result.entry.run_id for result in tier_results],
                "maximum_observed_peak_memory_bytes": max(
                    (
                        peak
                        for result in tier_results
                        if (peak := _peak_bytes(result)) is not None
                    ),
                    default=None,
                ),
                "postflight": postflight.to_dict(),
                "blockers": list(blockers),
            }
        )
        atomic_write_text(
            workspace / "state.json",
            json.dumps(state, indent=2, sort_keys=False) + "\n",
        )
        if not safe:
            remaining = [
                later.name
                for later in manifest.context_tiers
                if later.order > tier_pin.order and later.order <= selected_max.order
            ]
            state["stopped_before_tier"] = remaining[0] if remaining else None
            state["stopped_during"] = {
                "tier": tier.value,
                "phase": "measured_or_postflight",
            }
            state["unattempted_tiers"] = remaining
            state["stop_reasons"] = list(blockers)
            break

    state["ended_at"] = utc_now_iso()
    state["status"] = "stopped" if state["stop_reasons"] else "completed"
    atomic_write_text(
        workspace / "state.json",
        json.dumps(state, indent=2, sort_keys=False) + "\n",
    )
    return state


def _measurement(record: ExperimentRecord, name: str) -> float | None:
    value = getattr(record.timing, name)
    return None if value is None else value.value


def _tier_summary(
    tier: str, rows: list[tuple[RowVerification, ExperimentRecord]]
) -> dict[str, Any]:
    evaluated = [
        (verification, record)
        for verification, record in rows
        if verification.status in (RowStatus.COMPLETED, RowStatus.SKIPPED)
    ]
    passed = [record for _, record in evaluated if record.outcome.success]

    def average(values: list[float]) -> float | None:
        return mean(values) if values else None

    total_values = [
        value
        for _, record in evaluated
        if (value := _measurement(record, "total")) is not None
    ]
    pass_total = sum(
        value
        for record in passed
        if (value := _measurement(record, "total")) is not None
    )
    peaks = [
        record.memory.peak.value
        for _, record in evaluated
        if record.memory.peak is not None
    ]
    quality = [
        record.outcome.quality_score
        for _, record in evaluated
        if record.outcome.quality_score is not None
    ]
    decode_ms = [
        value
        for _, record in evaluated
        if (value := _measurement(record, "decode")) is not None and value > 0
    ]
    token_rates = []
    for _, record in evaluated:
        tokens = record.tokens.generated_tokens
        duration = _measurement(record, "decode")
        if tokens is not None and tokens > 1 and duration is not None and duration > 0:
            # ``decode`` starts when the first token is observed, so only
            # subsequent tokens belong in its derived throughput numerator.
            token_rates.append((tokens - 1) / (duration / 1000))
    return {
        "tier": tier,
        "runs": len(rows),
        "evaluated_runs": len(evaluated),
        "passing_runs": len(passed),
        "pass_rate": len(passed) / len(evaluated) if evaluated else None,
        "mean_quality_score": average([float(value) for value in quality]),
        "mean_total_ms": average(total_values),
        "mean_prefill_ms": average(
            [
                value
                for _, record in evaluated
                if (value := _measurement(record, "prefill")) is not None
            ]
        ),
        "mean_decode_ms": average(decode_ms),
        "mean_decode_tokens_per_second": average(token_rates),
        "max_peak_memory_bytes": max(peaks) if peaks else None,
        "correct_cases_per_minute": correct_cases_per_minute(len(passed), pass_total),
        "timing_provenance": MetricProvenance.MEASURED_WALL_CLOCK.value,
        "memory_provenance": (
            MetricProvenance.MEASURED_NATIVE.value if peaks else None
        ),
        "token_rate_provenance": (
            MetricProvenance.DERIVED.value if token_rates else None
        ),
    }


def _argv_value(record: ExperimentRecord, flag: str) -> str | None:
    argv = record.command.argv
    try:
        index = argv.index(flag)
    except ValueError:
        return None
    return argv[index + 1] if index + 1 < len(argv) else None


def _record_matches_manifest(
    manifest: LabManifest,
    verification: RowVerification,
    record: ExperimentRecord,
    *,
    warmup: bool,
    allowed_tiers: set[str],
) -> bool:
    workload_pins = {workload.workload_id: workload for workload in manifest.workloads}
    workload = workload_pins.get(verification.workload_id)
    if workload is None or verification.context_tier not in allowed_tiers:
        return False
    expected_hash = workload.prompt_hashes.get(verification.context_tier)
    if (
        verification.workload_version != workload.version
        or verification.verified_prompt_hash != expected_hash
        or record.command.workload_hash != expected_hash
    ):
        return False
    label = "warmup" if warmup else "rep"
    expected_run_id = (
        f"{workload.workload_id}-{verification.context_tier}-{label}-"
        f"{record.repetition.repetition_index + 1:02d}"
    )
    if verification.run_id != expected_run_id or record.run_id != expected_run_id:
        return False
    expected_measured = (
        max(manifest.repetitions.warmup_per_tier, 1)
        if warmup
        else manifest.repetitions.measured_per_workload
    )
    expected_warmups = manifest.repetitions.warmup_per_tier
    if (
        record.repetition.measured_repetitions != expected_measured
        or record.repetition.warmup_repetitions != expected_warmups
        or record.repetition.seed != manifest.generation.seed
    ):
        return False
    expected_memory_gb = manifest.safety.required_total_memory_bytes / 1024**3
    expected_config_hash = mlx_collection_contract_hash(
        model_id=manifest.model.repository_id,
        model_revision=manifest.model.revision,
        tokenizer_revision=manifest.model.revision,
        quantization=manifest.model.quantization,
        model_family=manifest.model.model_family,
        max_tokens=manifest.generation.max_output_tokens,
        seed=manifest.generation.seed,
        temperature=manifest.generation.temperature,
        top_p=manifest.generation.top_p,
        enable_thinking=manifest.generation.enable_thinking,
        prefill_step_size=manifest.runtime.prefill_step_size,
        draft_enabled=False,
        num_draft_tokens=2,
        timeout_seconds=manifest.cooperative_timeout_seconds,
    )
    return (
        record.model.model_id == manifest.model.repository_id
        and record.model.model_revision == manifest.model.revision
        and record.model.tokenizer_revision == manifest.model.revision
        and record.model.quantization == manifest.model.quantization
        and record.model.model_family == manifest.model.model_family
        and record.runtime.name == manifest.runtime.name
        and record.runtime.version == manifest.runtime.version
        and record.platform.accelerator == manifest.safety.required_chip
        and record.platform.total_memory_gb == expected_memory_gb
        and _argv_value(record, "--max-tokens")
        == str(manifest.generation.max_output_tokens)
        and _argv_value(record, "--temperature") == str(manifest.generation.temperature)
        and _argv_value(record, "--top-p") == str(manifest.generation.top_p)
        and record.command.config_hash == expected_config_hash
    )


def _environment_errors(
    manifest: LabManifest, environment_path: Path
) -> tuple[str, ...]:
    try:
        payload = json.loads(environment_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return (f"environment artifact is unreadable: {exc}",)
    if not isinstance(payload, dict):
        return ("environment artifact must be a JSON object",)
    versions = payload.get("package_versions")
    if not isinstance(versions, dict):
        return ("environment package_versions is missing",)
    expected = {
        "mlx": manifest.runtime.mlx_version,
        "mlx-lm": manifest.runtime.mlx_lm_version,
        "transformers": manifest.runtime.transformers_version,
    }
    # The runtime's own package (``manifest.runtime.name``, e.g. "mlx-vlm"
    # for the VLM-based 27B lab or "mlx-lm" for a plain mlx-lm control) is
    # keyed dynamically so this check generalizes to any pinned runtime
    # package instead of assuming "mlx-vlm" is always present.
    expected[manifest.runtime.name] = manifest.runtime.version
    errors = [
        f"{package} recorded as {versions.get(package)!r}, expected {version!r}"
        for package, version in expected.items()
        if versions.get(package) != version
    ]
    return tuple(errors)


def _current_state_scope(
    manifest: LabManifest, state: dict[str, Any]
) -> tuple[set[str], set[str], set[str]]:
    measured_run_ids: set[str] = set()
    warmup_run_ids: set[str] = set()
    attempted_tiers: set[str] = set()
    for tier_state in state.get("tiers", []):
        if not isinstance(tier_state, dict):
            continue
        tier_name = tier_state.get("tier")
        if isinstance(tier_name, str):
            attempted_tiers.add(tier_name)
        if tier_state.get("status") in {
            "failed_warmup",
            "failed_warmup_safety",
        }:
            warmup_run_ids.update(
                run_id
                for run_id in tier_state.get("run_ids", [])
                if isinstance(run_id, str)
            )
        else:
            measured_run_ids.update(
                run_id
                for run_id in tier_state.get("run_ids", [])
                if isinstance(run_id, str)
            )
            if isinstance(tier_name, str):
                warmup_run_ids.update(
                    f"{manifest.workloads[0].workload_id}-{tier_name}-"
                    f"warmup-{index + 1:02d}"
                    for index in range(manifest.repetitions.warmup_per_tier)
                )
    stopped_during = state.get("stopped_during")
    if isinstance(stopped_during, dict):
        tier_name = stopped_during.get("tier")
        if isinstance(tier_name, str):
            attempted_tiers.add(tier_name)
    legacy_stopped = state.get("stopped_before_tier")
    if (
        "stopped_during" not in state
        and not attempted_tiers
        and isinstance(legacy_stopped, str)
    ):
        attempted_tiers.add(legacy_stopped)
    valid_tiers = {tier.name for tier in manifest.context_tiers}
    attempted_tiers.intersection_update(valid_tiers)
    if not warmup_run_ids and "stopped_during" not in state:
        warmup_run_ids.update(
            f"{manifest.workloads[0].workload_id}-{tier}-warmup-01"
            for tier in attempted_tiers
        )
    return attempted_tiers, measured_run_ids, warmup_run_ids


def build_shareable_report(manifest: LabManifest, *, workspace: Path) -> dict[str, Any]:
    state_path = workspace / "state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {}
    )
    if state.get("status") == "running":
        raise LabError(
            "current lab execution is incomplete; report generation is refused"
        )
    allowed_tiers, measured_run_ids, warmup_run_ids = _current_state_scope(
        manifest, state
    )
    environment: dict[str, Any] | None = None
    rows_by_tier: dict[str, list[tuple[RowVerification, ExperimentRecord]]] = {}
    runs_dir = workspace / "results" / "runs"
    if runs_dir.is_dir():
        for verification_path in sorted(runs_dir.glob("*/verification.json")):
            verification = RowVerification.read_json(verification_path)
            if (
                verification.final_record_path is None
                or verification.run_id not in measured_run_ids
            ):
                continue
            record = ExperimentRecord.read_json(
                verification_path.parent / "final_record.json"
            )
            if not _record_matches_manifest(
                manifest,
                verification,
                record,
                warmup=False,
                allowed_tiers=allowed_tiers,
            ):
                continue
            if _environment_errors(
                manifest,
                verification_path.parent / "collection" / "environment.json",
            ):
                continue
            environment_path = (
                verification_path.parent / "collection" / "environment.json"
            )
            if environment is None and environment_path.is_file():
                raw_environment = json.loads(
                    environment_path.read_text(encoding="utf-8")
                )
                environment = {
                    "os_name": raw_environment.get("os_name"),
                    "os_release": raw_environment.get("os_release"),
                    "architecture": raw_environment.get("architecture"),
                    "python_implementation": raw_environment.get(
                        "python_implementation"
                    ),
                    "python_version": raw_environment.get("python_version"),
                    "cpu_count": raw_environment.get("cpu_count"),
                    "total_memory_gb": raw_environment.get("total_memory_gb"),
                    "accelerator": record.platform.accelerator,
                    "package_versions": raw_environment.get("package_versions", {}),
                }
            rows_by_tier.setdefault(verification.context_tier, []).append(
                (verification, record)
            )
    tier_order = {tier.name: tier.order for tier in manifest.context_tiers}
    tiers = [
        _tier_summary(name, rows)
        for name, rows in sorted(
            rows_by_tier.items(), key=lambda item: tier_order[item[0]]
        )
    ]
    failed_attempts: list[dict[str, Any]] = []
    warmups_dir = workspace / "warmups"
    if warmups_dir.is_dir():
        for verification_path in sorted(warmups_dir.glob("*/runs/*/verification.json")):
            verification = RowVerification.read_json(verification_path)
            if verification.run_id not in warmup_run_ids:
                continue
            record_path = verification_path.parent / "final_record.json"
            failed_record = (
                ExperimentRecord.read_json(record_path)
                if verification.final_record_path is not None and record_path.is_file()
                else None
            )
            if failed_record is not None and not _record_matches_manifest(
                manifest,
                verification,
                failed_record,
                warmup=True,
                allowed_tiers=allowed_tiers,
            ):
                continue
            if failed_record is not None and _environment_errors(
                manifest,
                verification_path.parent / "collection" / "environment.json",
            ):
                continue
            environment_path = (
                verification_path.parent / "collection" / "environment.json"
            )
            if (
                environment is None
                and verification.collection_dir is not None
                and environment_path.is_file()
            ):
                raw_environment = json.loads(
                    environment_path.read_text(encoding="utf-8")
                )
                environment = {
                    "os_name": raw_environment.get("os_name"),
                    "os_release": raw_environment.get("os_release"),
                    "architecture": raw_environment.get("architecture"),
                    "python_implementation": raw_environment.get(
                        "python_implementation"
                    ),
                    "python_version": raw_environment.get("python_version"),
                    "cpu_count": raw_environment.get("cpu_count"),
                    "total_memory_gb": raw_environment.get("total_memory_gb"),
                    "accelerator": (
                        failed_record.platform.accelerator
                        if failed_record is not None
                        else None
                    ),
                    "package_versions": raw_environment.get("package_versions", {}),
                }
            if verification.status in (RowStatus.COMPLETED, RowStatus.SKIPPED):
                continue
            failed_attempts.append(
                {
                    "run_id": verification.run_id,
                    "tier": verification.context_tier,
                    "phase": "warmup",
                    "status": verification.status.value,
                    "reason": verification.reason,
                    "input_tokens": (
                        failed_record.tokens.input_tokens
                        if failed_record is not None
                        else None
                    ),
                    "total_ms": (
                        _measurement(failed_record, "total")
                        if failed_record is not None
                        else None
                    ),
                    "prefill_ms": (
                        _measurement(failed_record, "prefill")
                        if failed_record is not None
                        else None
                    ),
                    "decode_ms": (
                        _measurement(failed_record, "decode")
                        if failed_record is not None
                        else None
                    ),
                    "peak_memory_bytes": (
                        failed_record.memory.peak.value
                        if failed_record is not None
                        and failed_record.memory.peak is not None
                        else None
                    ),
                }
            )
    for tier_rows in rows_by_tier.values():
        for verification, record in tier_rows:
            if verification.status in (RowStatus.COMPLETED, RowStatus.SKIPPED):
                continue
            failed_attempts.append(
                {
                    "run_id": verification.run_id,
                    "tier": verification.context_tier,
                    "phase": "measured",
                    "status": verification.status.value,
                    "reason": verification.reason,
                    "input_tokens": record.tokens.input_tokens,
                    "total_ms": _measurement(record, "total"),
                    "prefill_ms": _measurement(record, "prefill"),
                    "decode_ms": _measurement(record, "decode"),
                    "peak_memory_bytes": (
                        record.memory.peak.value
                        if record.memory.peak is not None
                        else None
                    ),
                }
            )
    stop_reasons = state.get("stop_reasons", [])
    if failed_attempts:
        stop_reasons = [
            attempt["reason"]
            for attempt in failed_attempts
            if attempt["reason"] is not None
        ]
    stopped_during = state.get("stopped_during")
    if stopped_during is None and failed_attempts:
        stopped_during = {
            "tier": failed_attempts[0]["tier"],
            "phase": failed_attempts[0]["phase"],
        }
    unattempted_tiers = state.get("unattempted_tiers")
    if not isinstance(unattempted_tiers, list) and stopped_during is not None:
        stopped_tier = manifest.tier(stopped_during["tier"])
        unattempted_tiers = [
            tier.name
            for tier in manifest.context_tiers
            if tier.order > stopped_tier.order
        ]
    stopped_before_tier = (
        state.get("stopped_before_tier") if stopped_during is None else None
    )
    requested_max_tier = state.get("requested_max_tier")
    executed_tiers = set(allowed_tiers)
    not_executed_tiers = [
        tier.name for tier in manifest.context_tiers if tier.name not in executed_tiers
    ]
    preflight_snapshot = state.get("preflight", {}).get("snapshot", {})
    tier_states = state.get("tiers", [])
    last_postflight = (
        tier_states[-1].get("postflight", {}).get("snapshot", {})
        if tier_states and isinstance(tier_states[-1], dict)
        else {}
    )
    report = {
        "schema_version": LAB_REPORT_SCHEMA_VERSION,
        "lab_id": manifest.lab_id,
        "generated_at": state.get("ended_at"),
        "model": {
            "official_id": manifest.model.official_id,
            "official_revision": manifest.model.official_revision,
            "repository_id": manifest.model.repository_id,
            "revision": manifest.model.revision,
            "license": manifest.model.license,
            "quantization": manifest.model.quantization,
            "converter": manifest.model.converter,
            "converter_revision": manifest.model.converter_revision,
            "download_bytes": manifest.model.expected_download_bytes,
            "sources": list(manifest.model.sources),
        },
        "environment": environment,
        "runtime_contract": asdict(manifest.runtime),
        "generation": asdict(manifest.generation),
        "repetitions": asdict(manifest.repetitions),
        "cooperative_timeout_seconds": manifest.cooperative_timeout_seconds,
        "safety": asdict(manifest.safety),
        "safety_observations": {
            "provenance": "macOS memory_pressure and sysctl",
            "preflight_memory_free_percent": preflight_snapshot.get(
                "memory_free_percent"
            ),
            "preflight_swap_used_bytes": preflight_snapshot.get("swap_used_bytes"),
            "postflight_memory_free_percent": last_postflight.get(
                "memory_free_percent"
            ),
            "postflight_swap_used_bytes": last_postflight.get("swap_used_bytes"),
        },
        "tiers": tiers,
        "failed_attempts": failed_attempts,
        "stopped_before_tier": stopped_before_tier,
        "stopped_during": stopped_during,
        "requested_max_tier": requested_max_tier,
        "unattempted_tiers": unattempted_tiers or [],
        "not_executed_tiers": not_executed_tiers,
        "stop_reasons": stop_reasons,
        "limitations": [
            "Results apply only to the pinned model, runtime, workloads, context tiers, and this 24 GB M5 Pro configuration.",
            "Total, prefill, and decode durations are host wall-clock boundaries; they are not kernel timings.",
            "Peak memory is MLX allocator peak memory when available, not whole-system resident memory.",
            "Decode token rate is derived from measured generated tokens and decode wall time.",
            "The configured timeout is cooperative and cannot preempt a blocked native model-load or prefill call.",
            "No GPU utilization, bandwidth, power, energy, or kernel timing is measured or inferred.",
            "The 4-bit conversion card names the upstream model but does not cryptographically bind its source revision.",
        ],
    }
    assert_shareable(report)
    return report


def _stale_result_run_ids(manifest: LabManifest, *, workspace: Path) -> tuple[str, ...]:
    state_path = workspace / "state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {}
    )
    allowed_tiers, measured_run_ids, _ = _current_state_scope(manifest, state)
    stale: list[str] = []
    runs_dir = workspace / "results" / "runs"
    if not runs_dir.is_dir():
        return ()
    for verification_path in sorted(runs_dir.glob("*/verification.json")):
        try:
            verification = RowVerification.read_json(verification_path)
            record = ExperimentRecord.read_json(
                verification_path.parent / "final_record.json"
            )
        except (OSError, ValueError):
            if verification_path.parent.name in measured_run_ids:
                stale.append(verification_path.parent.name)
            continue
        if verification.run_id not in measured_run_ids:
            continue
        if not _record_matches_manifest(
            manifest,
            verification,
            record,
            warmup=False,
            allowed_tiers=allowed_tiers,
        ):
            stale.append(verification.run_id)
            continue
        environment_errors = _environment_errors(
            manifest,
            verification_path.parent / "collection" / "environment.json",
        )
        if environment_errors:
            stale.append(verification.run_id)
    return tuple(stale)


def _materialize_current_results(manifest: LabManifest, *, workspace: Path) -> Path:
    state_path = workspace / "state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {}
    )
    _, measured_run_ids, _ = _current_state_scope(manifest, state)
    state_digest = config_hash(state).removeprefix("sha256:")[:16]
    filtered = workspace / "reports" / f"current-results-{state_digest}"
    for run_id in sorted(measured_run_ids):
        source = workspace / "results" / "runs" / run_id
        verification = RowVerification.read_json(source / "verification.json")
        record = ExperimentRecord.read_json(source / "final_record.json")
        destination = filtered / "runs" / run_id
        final_record_path = destination / "final_record.json"
        record.write_json(final_record_path)
        filtered_verification = replace(
            verification,
            final_record_path=str(final_record_path.resolve()),
            collection_dir=None,
        )
        atomic_write_text(
            destination / "verification.json",
            filtered_verification.to_json(),
        )
    return filtered


def assert_shareable(value: Any, *, context: str = "report") -> None:
    if isinstance(value, dict):
        forbidden_keys = {
            "serial",
            "serial_number",
            "hardware_uuid",
            "hostname",
            "username",
            "home",
            "model_path",
            "workspace",
        }
        overlap = forbidden_keys.intersection(str(key).lower() for key in value)
        if overlap:
            raise LabError(f"{context} contains private key(s): {sorted(overlap)}")
        for key, item in value.items():
            assert_shareable(item, context=f"{context}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_shareable(item, context=f"{context}[{index}]")
    elif isinstance(value, str):
        for pattern in _PRIVATE_PATTERNS:
            if pattern.search(value):
                raise LabError(f"{context} contains private path or credential data")


def _fmt(value: float | int | None, *, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _bar_chart(
    tiers: list[dict[str, Any]],
    *,
    metric: str,
    title: str,
    unit: str,
    scale: float = 1.0,
) -> str:
    values = [
        None if tier[metric] is None else float(tier[metric]) / scale for tier in tiers
    ]
    maximum = max((value for value in values if value is not None), default=1.0)
    rows: list[str] = []
    for index, (tier, value) in enumerate(zip(tiers, values, strict=True)):
        y = 36 + index * 54
        width = 0 if value is None or maximum <= 0 else 420 * value / maximum
        label = "n/a" if value is None else f"{value:.2f} {unit}"
        rows.append(
            f'<text x="0" y="{y + 18}" class="axis">'
            f"{html.escape(str(tier['tier']))}</text>"
            f'<rect x="52" y="{y}" width="{width:.2f}" height="28" rx="3" '
            'class="bar"/>'
            f'<text x="{58 + width:.2f}" y="{y + 19}" class="value">{label}</text>'
        )
    height = 64 + len(tiers) * 54
    return (
        f'<section class="chart"><h2>{title}</h2>'
        f'<svg viewBox="0 0 620 {height}" role="img" aria-label="{title}">'
        + "".join(rows)
        + "</svg></section>"
    )


def render_lab_report_html(report: dict[str, Any]) -> str:
    assert_shareable(report)
    tiers = list(report["tiers"])
    model = report["model"]
    environment = report["environment"] or {}
    safety_observations = report["safety_observations"]

    def esc(value: Any) -> str:
        return html.escape(str(value), quote=True)

    rows = "".join(
        "<tr>"
        f"<td>{esc(tier['tier'])}</td>"
        f"<td>{tier['evaluated_runs']}</td>"
        f"<td>{_fmt(tier['pass_rate'])}</td>"
        f"<td>{_fmt(tier['mean_quality_score'])}</td>"
        f"<td>{_fmt(tier['mean_total_ms'])}</td>"
        f"<td>{_fmt(tier['mean_prefill_ms'])}</td>"
        f"<td>{_fmt(tier['mean_decode_ms'])}</td>"
        f"<td>{_fmt(tier['mean_decode_tokens_per_second'])}</td>"
        f"<td>{_fmt(tier['correct_cases_per_minute'])}</td>"
        "</tr>"
        for tier in tiers
    )
    limitations = "".join(f"<li>{esc(item)}</li>" for item in report["limitations"])
    failed_rows = "".join(
        "<tr>"
        f"<td>{esc(attempt['tier'])}</td>"
        f"<td>{esc(attempt['phase'])}</td>"
        f"<td>{esc(attempt['status'])}</td>"
        f"<td>{esc(attempt['input_tokens'] if attempt['input_tokens'] is not None else 'n/a')}</td>"
        f"<td>{esc(attempt['reason'] or 'unrecorded')}</td>"
        "</tr>"
        for attempt in report["failed_attempts"]
    )
    stopped_during = report["stopped_during"]
    stopped_label = (
        f"{stopped_during['tier']} / {stopped_during['phase']}"
        if isinstance(stopped_during, dict)
        else (
            f"before {report['stopped_before_tier']}"
            if report["stopped_before_tier"]
            else "none"
        )
    )
    charts = "".join(
        (
            _bar_chart(
                tiers,
                metric="mean_total_ms",
                title="Mean total latency by safe context tier",
                unit="ms",
            ),
            _bar_chart(
                tiers,
                metric="pass_rate",
                title="Deterministic evaluator pass rate",
                unit="",
            ),
            _bar_chart(
                tiers,
                metric="correct_cases_per_minute",
                title="Correct cases per minute",
                unit="cases/min",
            ),
            _bar_chart(
                tiers,
                metric="max_peak_memory_bytes",
                title="Observed MLX peak memory",
                unit="GiB",
                scale=1024**3,
            ),
        )
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="robots" content="noindex, nofollow">
<title>M5 Pro Qwen3.8 local inference lab</title>
<style>
:root{{--ink:#17202a;--muted:#5d6670;--paper:#f7f3ea;--panel:#fffdf8;--accent:#f05d23;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--paper);color:var(--ink);
font:15px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}} main{{max-width:1120px;
margin:auto;padding:40px 24px}} h1{{font-size:clamp(28px,5vw,54px);line-height:1.02}}
.stamp{{color:var(--muted)}} .card,.chart{{background:var(--panel);border:1px solid #d9d1c3;
padding:20px;margin:20px 0;overflow:auto}} table{{width:100%;border-collapse:collapse;
min-width:900px}} th,td{{text-align:right;padding:10px;border-bottom:1px solid #ddd5c8}}
th:first-child,td:first-child{{text-align:left}} svg{{width:100%;min-width:560px}}
.bar{{fill:var(--accent)}} .axis,.value{{font:14px ui-monospace,SFMono-Regular,Menlo,monospace;
fill:var(--ink)}} code{{overflow-wrap:anywhere}} .warning{{border-left:5px solid var(--accent)}}
</style>
</head>
<body><main>
<p class="stamp">LLMTraceFX / local evidence / no external references</p>
<h1>M5 Pro × Qwen3.8-27B</h1>
<p>Measured evidence for <code>{esc(model['repository_id'])}@{esc(model['revision'])}</code>,
{esc(model['quantization'])}, under the pinned workloads and constraints below. This
is not a universal fastest-model claim.</p>
<section class="card"><h2>Run contract</h2>
<p>Official model: <code>{esc(model['official_id'])}@{esc(model['official_revision'])}</code><br>
License: {esc(model['license'])}<br>Converter: {esc(model['converter'])} at
<code>{esc(model['converter_revision'])}</code><br>Generated:
{esc(report['generated_at'] or 'not run')}</p></section>
<section class="card"><h2>Environment</h2><p>
Accelerator: {esc(environment.get('accelerator') or 'unrecorded')}<br>
Memory: {esc(environment.get('total_memory_gb') or 'unrecorded')} GiB<br>
OS: {esc(environment.get('os_name') or 'unrecorded')}
{esc(environment.get('os_release') or '')} / {esc(environment.get('architecture') or 'unrecorded')}<br>
Python: {esc(environment.get('python_version') or 'unrecorded')}<br>
Runtime: mlx-vlm {esc(environment.get('package_versions', {}).get('mlx-vlm', 'unrecorded'))},
MLX {esc(environment.get('package_versions', {}).get('mlx', 'unrecorded'))}
</p></section>
<section class="card"><h2>Safety observations</h2><p>
Preflight free memory: {esc(safety_observations['preflight_memory_free_percent'])}%,
swap used: {esc(safety_observations['preflight_swap_used_bytes'])} bytes.<br>
Postflight free memory: {esc(safety_observations['postflight_memory_free_percent'])}%,
swap used: {esc(safety_observations['postflight_swap_used_bytes'])} bytes.<br>
Provenance: {esc(safety_observations['provenance'])}.
</p></section>
<section class="card"><h2>Measured outcomes</h2><table><thead><tr>
<th>Tier</th><th>Runs</th><th>Pass rate</th><th>Mean quality</th>
<th>Total ms</th><th>Prefill ms</th><th>Decode ms</th><th>Decode tok/s</th>
<th>Correct/min</th></tr></thead><tbody>{rows}</tbody></table></section>
<section class="card warning"><h2>Stopped attempts</h2><table><thead><tr>
<th>Tier</th><th>Phase</th><th>Status</th><th>Input tokens</th><th>Reason</th>
</tr></thead><tbody>{failed_rows}</tbody></table></section>
{charts}
<section class="card warning"><h2>Limits and provenance</h2><ul>{limitations}</ul>
<p>Stop point: {esc(stopped_label)}. Requested maximum:
{esc(report['requested_max_tier'] or 'unrecorded')}. Not executed:
{esc(', '.join(report['not_executed_tiers']) or 'none')}.</p></section>
</main></body></html>
"""


def write_reports(
    manifest: LabManifest,
    *,
    workspace: Path,
    shareable_dir: Path | None = None,
) -> dict[str, Any]:
    evidence_verification = verify_evidence(manifest, workspace=workspace)
    if not evidence_verification["verified"]:
        raise LabError(
            "report generation refused unverified current-run evidence: "
            + "; ".join(evidence_verification["failures"])
        )
    stale = _stale_result_run_ids(manifest, workspace=workspace)
    if stale:
        raise LabError(
            "report generation refused stale or mismatched run evidence: "
            + ", ".join(stale)
        )
    report = build_shareable_report(manifest, workspace=workspace)
    reports_dir = workspace / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        reports_dir / "lab-report.json",
        json.dumps(report, indent=2, sort_keys=False) + "\n",
    )
    atomic_write_text(reports_dir / "lab-report.html", render_lab_report_html(report))

    results_dir = _materialize_current_results(manifest, workspace=workspace)
    tune_policy = TunePolicy(
        name="M5 Pro constraints-first local evidence",
        description=(
            "Rank only repetitions that pass deterministic evaluation, remain "
            "under the configured MLX peak-memory ceiling, and share identity."
        ),
        objective=TuneObjective.MAX_CORRECT_CASES_PER_MINUTE,
        constraints=TuneConstraints(
            min_pass_rate=1.0,
            max_peak_memory_bytes=float(manifest.safety.maximum_peak_memory_bytes),
            allowed_provenances=frozenset(
                {
                    MetricProvenance.MEASURED_WALL_CLOCK,
                    MetricProvenance.MEASURED_NATIVE,
                }
            ),
            min_measured_repetitions=manifest.repetitions.measured_per_workload,
        ),
    )
    tune_report = tune(results_dirs=(results_dir,), policy=tune_policy)
    atomic_write_text(reports_dir / "tune-report.json", tune_report.to_json() + "\n")
    atomic_write_text(
        reports_dir / "tune-report.html",
        render_tune_report_html(tune_report, redact_paths=True),
    )

    compare_policy = ComparePolicy(
        name="Pinned local-system evidence",
        description=(
            "Apply the same deterministic pass-rate and repetition bar through "
            "the merged comparison layer; no hosted systems are included."
        ),
        objective=CompareObjective.MAX_CORRECT_CASES_PER_MINUTE,
        constraints=CompareConstraints(
            min_pass_rate=1.0,
            allowed_provenances=frozenset({MetricProvenance.MEASURED_WALL_CLOCK}),
            min_measured_repetitions=manifest.repetitions.measured_per_workload,
        ),
    )
    compare_report = compare(results_dirs=(results_dir,), policy=compare_policy)
    atomic_write_text(
        reports_dir / "compare-report.json",
        compare_report.to_json() + "\n",
    )
    atomic_write_text(
        reports_dir / "compare-report.html",
        render_compare_report_html(compare_report, redact_paths=True),
    )

    if shareable_dir is not None:
        shareable_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(
            shareable_dir / "evidence-summary.json",
            json.dumps(report, indent=2, sort_keys=False) + "\n",
        )
        atomic_write_text(shareable_dir / "report.html", render_lab_report_html(report))
    return report


def verify_evidence(manifest: LabManifest, *, workspace: Path) -> dict[str, Any]:
    verify_catalog(manifest)
    state_path = workspace / "state.json"
    state = (
        json.loads(state_path.read_text(encoding="utf-8"))
        if state_path.exists()
        else {}
    )
    if state.get("status") == "running":
        return {
            "schema_version": "1",
            "verified": False,
            "records_checked": 0,
            "failures": [
                "current lab execution is incomplete; verification is refused"
            ],
        }
    allowed_tiers, measured_run_ids, warmup_run_ids = _current_state_scope(
        manifest, state
    )
    checked = 0
    found_run_ids: set[str] = set()
    failures: list[str] = []
    roots = (
        (
            workspace / "results" / "runs",
            "*/verification.json",
            measured_run_ids,
            False,
        ),
        (
            workspace / "warmups",
            "*/runs/*/verification.json",
            warmup_run_ids,
            True,
        ),
    )
    for root, pattern, current_run_ids, warmup in roots:
        verification_paths = sorted(root.glob(pattern)) if root.is_dir() else []
        for verification_path in verification_paths:
            try:
                verification = RowVerification.read_json(verification_path)
                if verification.run_id not in current_run_ids:
                    continue
                found_run_ids.add(verification.run_id)
                if verification.final_record_path is None:
                    failures.append(f"{verification_path.parent.name}: no final record")
                    continue
                record_path = verification_path.parent / "final_record.json"
                record = ExperimentRecord.read_json(record_path)
                if not _record_matches_manifest(
                    manifest,
                    verification,
                    record,
                    warmup=warmup,
                    allowed_tiers=allowed_tiers,
                ):
                    failures.append(
                        f"{verification.run_id}: current manifest binding mismatch"
                    )
                    continue
                environment_errors = _environment_errors(
                    manifest,
                    verification_path.parent / "collection" / "environment.json",
                )
                if environment_errors:
                    failures.extend(
                        f"{verification.run_id}: {error}"
                        for error in environment_errors
                    )
                    continue
                checked += 1
            except (OSError, ValueError) as exc:
                failures.append(f"{verification_path.parent.name}: {exc}")
    for missing_run_id in sorted((measured_run_ids | warmup_run_ids) - found_run_ids):
        failures.append(f"{missing_run_id}: expected current-run artifact is missing")
    report_path = workspace / "reports" / "lab-report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            assert_shareable(report)
        except (OSError, ValueError, LabError) as exc:
            failures.append(f"shareable report: {exc}")
    return {
        "schema_version": "1",
        "verified": not failures and checked > 0,
        "records_checked": checked,
        "failures": failures,
    }
