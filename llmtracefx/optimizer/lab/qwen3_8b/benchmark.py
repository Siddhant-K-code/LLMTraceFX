"""Subprocess-isolated benchmark runner for the Qwen3-8B M5 Pro control.

This differs from the packaged 27B lab's ``lab.core.run_lab`` in exactly
the ways this control requires and the 27B lab does not:

* every warmup and every measured repetition runs in its own fresh,
  no-shell subprocess and process group (``run_lab`` executes an entire
  tiered sweep in one long-lived process);
* a parent-enforced wall-clock timeout with TERM->KILL escalation wraps
  each of those subprocesses, independent of ``collect_mlx``'s own
  cooperative in-process timeout;
* the checkpoint's own mlx-lm chat template (``enable_thinking=false``)
  tokenizes every prompt, so each row's *actual* tokenizer count is
  tracked next to the workload catalog's *requested* target-tier token
  count instead of assuming they coincide;
* a run mode gate (``exploratory`` by default; ``publication`` only with
  an explicit clean-boot assertion) matches ``lab.frontier``'s pattern.

It otherwise deliberately reuses the load-bearing pieces of the 27B lab
unmodified: ``LabManifest`` parsing/validation, ``verify_model``/
``verify_catalog``/``assess_safety``, the workload catalog and
deterministic prompt materialization, the safe evaluators and MLX
collector (both reached through ``workloads.verify.execute_row``, which
this module's child subprocess calls directly), and the tune/compare/
report layers in ``lab.core`` once results are written in the exact
directory shape ``lab.core`` already expects.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...collectors._shared import atomic_write_text, config_hash
from ...schema import ExperimentRecord, utc_now_iso
from ...workloads.catalog import workload_by_id
from ...workloads.materialize import MaterializedPrompt, materialize_prompt
from ...workloads.matrix import DECODE_MODE_AUTOREGRESSIVE, MatrixEntry
from ...workloads.schema import ContextTier
from ...workloads.verify import RowResult, RowStatus, RowVerification, execute_row
from ..core import (
    LabError,
    _binding,
    _peak_bytes,
    _record_matches_manifest,
    _tier_is_safe,
    assess_safety,
    verify_catalog,
    verify_model,
)
from ..frontier import _clean_process_group
from ..manifest import LabManifest
from .runtime import Qwen3ChatMLXLMRuntime

BENCHMARK_STATE_SCHEMA_VERSION = "1"
RUN_MODES = ("exploratory", "publication")
DEFAULT_ROW_TIMEOUT_SECONDS = 300.0
DEFAULT_CLEANUP_GRACE_SECONDS = 15.0


def _row_run_id(
    workload_id: str, tier: ContextTier, *, warmup: bool, index: int
) -> str:
    label = "warmup" if warmup else "rep"
    return f"{workload_id}-{tier.value}-{label}-{index + 1:02d}"


def _control_entry(
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
    run_id = _row_run_id(workload_id, tier, warmup=warmup, index=repetition_index)
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
        command_argv=("llmtracefx-m5-control", "run"),
        max_tokens=manifest.generation.max_output_tokens,
    )


def _write_prompt(path: Path, prompt: MaterializedPrompt) -> None:
    atomic_write_text(path, prompt.text)


def _row_is_ok(result: RowResult) -> bool:
    """Whether a row's evidence is trustworthy *and* task-successful.

    A row whose collector status is ``COMPLETED``/``SKIPPED`` but whose
    evaluator reported ``outcome_success=False`` executed cleanly but
    failed the task -- an evaluator failure, not merely an infrastructure
    one. It must gate exactly like any other failed row (stop immediately,
    warmup or measured, never let the tier or the next tier pass), so it
    is never treated as "ok" here even though its collector status looks
    like success.
    """
    return (
        result.verification.status in (RowStatus.COMPLETED, RowStatus.SKIPPED)
        and result.verification.outcome_success is True
    )


def _evaluator_failure_blockers(results: tuple[RowResult, ...]) -> tuple[str, ...]:
    return tuple(
        f"{result.entry.run_id} completed but the evaluator reported "
        "outcome_success=False (task quality failure)"
        for result in results
        if result.verification.status in (RowStatus.COMPLETED, RowStatus.SKIPPED)
        and result.verification.outcome_success is False
    )


def _row_result_from_disk(entry: MatrixEntry, *, output_dir: Path) -> RowResult | None:
    run_dir = output_dir / "runs" / entry.run_id
    verification_path = run_dir / "verification.json"
    if not verification_path.is_file():
        return None
    verification = RowVerification.read_json(verification_path)
    record = (
        ExperimentRecord.read_json(run_dir / "final_record.json")
        if verification.final_record_path is not None
        and (run_dir / "final_record.json").is_file()
        else None
    )
    return RowResult(entry=entry, verification=verification, final_record=record)


def _synthetic_failure_result(
    entry: MatrixEntry, *, reason: str, started_at: str
) -> RowResult:
    verification = RowVerification(
        schema_version="2",
        run_id=entry.run_id,
        workload_id=entry.workload_id,
        workload_version=entry.workload_version,
        category=entry.category,
        context_tier=entry.context_tier,
        decode_mode=entry.decode_mode,
        status=RowStatus.FAILED,
        reason=reason,
        recorded_prompt_hash=entry.prompt.prompt_hash,
        verified_prompt_hash=None,
        run_binding_hash=None,
        resumed=False,
        outcome_success=False,
        quality_score=None,
        total_ms=None,
        started_at=started_at,
        ended_at=utc_now_iso(),
        final_record_path=None,
        collection_dir=None,
    )
    return RowResult(entry=entry, verification=verification, final_record=None)


class ChildLaunchResult:
    """Outcome of one row's isolated child subprocess."""

    __slots__ = ("exit_code", "timed_out", "descendants_cleaned")

    def __init__(
        self, *, exit_code: int | None, timed_out: bool, descendants_cleaned: bool
    ):
        self.exit_code = exit_code
        self.timed_out = timed_out
        self.descendants_cleaned = descendants_cleaned


def launch_row_subprocess(
    *,
    manifest_path: Path,
    workload_id: str,
    tier: ContextTier,
    repetition_index: int,
    warmup: bool,
    model_path: Path,
    output_dir: Path,
    resume: bool,
    timeout_seconds: float,
    cleanup_grace_seconds: float,
) -> ChildLaunchResult:
    argv = [
        sys.executable,
        "-m",
        "llmtracefx.optimizer.lab.qwen3_8b.benchmark",
        "_child",
        "--manifest",
        str(manifest_path),
        "--workload-id",
        workload_id,
        "--tier",
        tier.value,
        "--repetition-index",
        str(repetition_index),
        "--model-path",
        str(model_path),
        "--output-dir",
        str(output_dir),
        "--warmup" if warmup else "--no-warmup",
        "--resume" if resume else "--no-resume",
    ]
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        shell=False,
    )
    process_group = process.pid
    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
    descendants_cleaned = _clean_process_group(
        process, process_group, cleanup_grace_seconds
    )
    if process.poll() is None:
        try:
            process.wait(timeout=cleanup_grace_seconds)
        except subprocess.TimeoutExpired:
            descendants_cleaned = False
    return ChildLaunchResult(
        exit_code=process.returncode,
        timed_out=timed_out,
        descendants_cleaned=descendants_cleaned,
    )


def _run_row_isolated(
    manifest: LabManifest,
    *,
    manifest_path: Path,
    workload_id: str,
    tier: ContextTier,
    repetition_index: int,
    warmup: bool,
    model_path: Path,
    output_dir: Path,
    resume: bool,
    timeout_seconds: float,
    cleanup_grace_seconds: float,
    launcher: Callable[..., ChildLaunchResult],
) -> RowResult:
    started_at = utc_now_iso()
    entry_tier_value = tier.value
    run_id = _row_run_id(workload_id, tier, warmup=warmup, index=repetition_index)
    child = launcher(
        manifest_path=manifest_path,
        workload_id=workload_id,
        tier=tier,
        repetition_index=repetition_index,
        warmup=warmup,
        model_path=model_path,
        output_dir=output_dir,
        resume=resume,
        timeout_seconds=timeout_seconds,
        cleanup_grace_seconds=cleanup_grace_seconds,
    )
    workload = workload_by_id(workload_id)
    prompt = materialize_prompt(workload, tier)
    fallback_entry = MatrixEntry(
        run_id=run_id,
        workload_id=workload_id,
        workload_version=workload.version,
        category=workload.category.value,
        context_tier=entry_tier_value,
        decode_mode=DECODE_MODE_AUTOREGRESSIVE,
        configured_depth=None,
        prompt=prompt,
        prompt_path="",
        runner_results_dir="",
        collector_output_dir="",
        runnable=True,
        unsupported_reason=None,
        command_argv=("llmtracefx-m5-control", "run"),
        max_tokens=manifest.generation.max_output_tokens,
    )
    if child.timed_out:
        return _synthetic_failure_result(
            fallback_entry,
            reason=f"row {run_id} exceeded its {timeout_seconds:g}s parent-enforced timeout",
            started_at=started_at,
        )
    if not child.descendants_cleaned:
        return _synthetic_failure_result(
            fallback_entry,
            reason=f"row {run_id} process-group cleanup failed after TERM/KILL escalation",
            started_at=started_at,
        )
    if child.exit_code not in (0, None):
        # A non-zero exit means the child process itself errored (an
        # uncaught exception, an import failure, etc.), not that
        # ``execute_row`` cleanly recorded a row failure. Never trust
        # whatever verification.json happens to exist at that path in
        # this case -- it could be stale evidence left over from an
        # earlier, unrelated attempt; treat this purely as a failure
        # instead of ever reading the child's own claimed artifact.
        return _synthetic_failure_result(
            fallback_entry,
            reason=(
                f"row {run_id} child exited with a non-zero status "
                f"({child.exit_code}); its own written evidence, if any, "
                "is never trusted after a crash"
            ),
            started_at=started_at,
        )
    result = _row_result_from_disk(fallback_entry, output_dir=output_dir)
    if result is None:
        return _synthetic_failure_result(
            fallback_entry,
            reason=(
                f"row {run_id} child exited (code {child.exit_code}) without "
                "writing supported evidence"
            ),
            started_at=started_at,
        )
    verification = result.verification
    expected_identity = (
        run_id,
        workload_id,
        workload.version,
        tier.value,
        DECODE_MODE_AUTOREGRESSIVE,
    )
    observed_identity = (
        verification.run_id,
        verification.workload_id,
        verification.workload_version,
        verification.context_tier,
        verification.decode_mode,
    )
    if observed_identity != expected_identity:
        return _synthetic_failure_result(
            fallback_entry,
            reason=f"row {run_id} child artifact identity is stale or mismatched",
            started_at=started_at,
        )
    if verification.status in (RowStatus.COMPLETED, RowStatus.SKIPPED) and (
        result.final_record is None
        or not _record_matches_manifest(
            manifest,
            verification,
            result.final_record,
            warmup=warmup,
            allowed_tiers={tier.value},
        )
    ):
        return _synthetic_failure_result(
            fallback_entry,
            reason=f"row {run_id} child artifact binding is stale or mismatched",
            started_at=started_at,
        )
    if (
        verification.status in (RowStatus.COMPLETED, RowStatus.SKIPPED)
        and verification.outcome_success is None
    ):
        return _synthetic_failure_result(
            fallback_entry,
            reason=f"row {run_id} child artifact lacks evaluator evidence",
            started_at=started_at,
        )
    return result


def run_child_row(
    manifest: LabManifest,
    *,
    workload_id: str,
    tier: ContextTier,
    repetition_index: int,
    warmup: bool,
    model_path: Path,
    output_dir: Path,
    manifest_dir: Path,
    resume: bool,
    hardware_fingerprint: str,
) -> RowResult:
    """Executed inside the isolated child subprocess: build the exact same
    ``MatrixEntry``/``RunBinding`` the parent would have, then reuse
    ``execute_row`` verbatim (safe evaluators, MLX collector, resume-by-
    hash, atomic artifact writes)."""
    workload = workload_by_id(workload_id)
    prompt = materialize_prompt(workload, tier)
    prompt_path = manifest_dir / "prompts" / f"{workload_id}-{tier.value}.txt"
    _write_prompt(prompt_path, prompt)
    entry = _control_entry(
        manifest,
        workload_id=workload_id,
        tier=tier,
        repetition_index=repetition_index,
        prompt=prompt,
        prompt_path=prompt_path,
        output_dir=output_dir,
        warmup=warmup,
    )
    binding = _binding(
        manifest,
        model_path,
        repetition_index=repetition_index,
        hardware_fingerprint=hardware_fingerprint,
        measured_repetitions=(
            max(manifest.repetitions.warmup_per_tier, 1)
            if warmup
            else manifest.repetitions.measured_per_workload
        ),
    )

    def runtime_factory() -> Qwen3ChatMLXLMRuntime:
        return Qwen3ChatMLXLMRuntime(
            temperature=manifest.generation.temperature,
            top_p=manifest.generation.top_p,
            enable_thinking=manifest.generation.enable_thinking,
        )

    return execute_row(
        entry,
        manifest_dir=manifest_dir,
        output_dir=output_dir,
        model_id=manifest.model.repository_id,
        binding=binding,
        resume=resume,
        runtime_factory=runtime_factory,
    )


def run_benchmark(
    manifest: LabManifest,
    *,
    manifest_path: Path,
    workspace: Path,
    model_path: Path,
    max_tier: str,
    run_mode: str = "exploratory",
    clean_boot_confirmed: bool = False,
    resume: bool = True,
    row_timeout_seconds: float = DEFAULT_ROW_TIMEOUT_SECONDS,
    cleanup_grace_seconds: float = DEFAULT_CLEANUP_GRACE_SECONDS,
    launcher: Callable[..., ChildLaunchResult] = launch_row_subprocess,
) -> dict[str, Any]:
    if run_mode not in RUN_MODES:
        raise LabError(f"unsupported run mode {run_mode!r}")
    if run_mode == "publication" and not clean_boot_confirmed:
        raise LabError(
            "publication mode requires the operator assertion --confirm-clean-boot"
        )
    if run_mode == "exploratory" and clean_boot_confirmed:
        raise LabError("--confirm-clean-boot is only valid in publication mode")

    verify_catalog(manifest)
    verify_model(manifest, model_path)
    preflight = assess_safety(manifest, workspace, include_download=False)
    if not preflight.safe:
        raise LabError(
            "benchmark run blocked by safety preflight: "
            + "; ".join(preflight.blockers)
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
        "schema_version": BENCHMARK_STATE_SCHEMA_VERSION,
        "lab_id": manifest.lab_id,
        "run_mode": run_mode,
        "clean_boot_confirmed": clean_boot_confirmed,
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
            "run_mode": run_mode,
        }
    )
    workspace.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        workspace / "state.json", json.dumps(state, indent=2, sort_keys=False) + "\n"
    )

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
            warmup_workload_id = manifest.workloads[0].workload_id
            for index in range(manifest.repetitions.warmup_per_tier):
                result = _run_row_isolated(
                    manifest,
                    manifest_path=manifest_path,
                    workload_id=warmup_workload_id,
                    tier=tier,
                    repetition_index=index,
                    warmup=True,
                    model_path=model_path,
                    output_dir=workspace / "warmups" / tier.value,
                    resume=False,
                    timeout_seconds=row_timeout_seconds,
                    cleanup_grace_seconds=cleanup_grace_seconds,
                    launcher=launcher,
                )
                warmup_results.append(result)
                if manifest.safety.stop_on_any_failed_row and not _row_is_ok(result):
                    break

        if any(not _row_is_ok(result) for result in warmup_results):
            reasons = [
                result.verification.reason
                or (
                    f"{result.entry.run_id} completed but the evaluator "
                    "reported outcome_success=False (task quality failure)"
                    if result.verification.status
                    in (RowStatus.COMPLETED, RowStatus.SKIPPED)
                    else f"{result.entry.run_id} ended as {result.verification.status.value}"
                )
                for result in warmup_results
                if not _row_is_ok(result)
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
            state["stopped_during"] = {"tier": tier.value, "phase": "warmup"}
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
        warmup_evaluator_blockers = _evaluator_failure_blockers(tuple(warmup_results))
        if warmup_evaluator_blockers:
            warmup_safe = False
            warmup_blockers = tuple(warmup_blockers) + warmup_evaluator_blockers
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
            state["stopped_during"] = {"tier": tier.value, "phase": "warmup_safety"}
            state["unattempted_tiers"] = [
                later.name
                for later in manifest.context_tiers
                if later.order > tier_pin.order and later.order <= selected_max.order
            ]
            state["stop_reasons"] = list(warmup_blockers)
            break

        tier_results: list[RowResult] = []
        stopped = False
        for workload_pin in manifest.workloads:
            for index in range(manifest.repetitions.measured_per_workload):
                result = _run_row_isolated(
                    manifest,
                    manifest_path=manifest_path,
                    workload_id=workload_pin.workload_id,
                    tier=tier,
                    repetition_index=index,
                    warmup=False,
                    model_path=model_path,
                    output_dir=results_root,
                    resume=resume,
                    timeout_seconds=row_timeout_seconds,
                    cleanup_grace_seconds=cleanup_grace_seconds,
                    launcher=launcher,
                )
                tier_results.append(result)
                if manifest.safety.stop_on_any_failed_row and not _row_is_ok(result):
                    stopped = True
                    break
            if stopped:
                break

        postflight = assess_safety(manifest, workspace, include_download=False)
        safe, blockers = _tier_is_safe(manifest, tuple(tier_results), postflight)
        evaluator_blockers = _evaluator_failure_blockers(tuple(tier_results))
        if evaluator_blockers:
            safe = False
            blockers = tuple(blockers) + evaluator_blockers
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
        workspace / "state.json", json.dumps(state, indent=2, sort_keys=False) + "\n"
    )
    return state


def _read_manifest(path: Path) -> LabManifest:
    return LabManifest.read_json(path)


def _child_main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="qwen3-8b-benchmark-child")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--repetition-index", required=True, type=int)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--warmup", dest="warmup", action="store_true")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument("--resume", dest="resume", action="store_true")
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    manifest = _read_manifest(manifest_path)
    output_dir = Path(args.output_dir)
    manifest_dir = manifest_path.parent
    preflight = assess_safety(manifest, manifest_dir, include_download=False)
    if not preflight.safe:
        raise LabError(
            "isolated row blocked by safety preflight: " + "; ".join(preflight.blockers)
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
    run_child_row(
        manifest,
        workload_id=args.workload_id,
        tier=ContextTier(args.tier),
        repetition_index=args.repetition_index,
        warmup=args.warmup,
        model_path=Path(args.model_path),
        output_dir=output_dir,
        manifest_dir=manifest_dir,
        resume=args.resume,
        hardware_fingerprint=hardware_fingerprint,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    raw_argv = sys.argv[1:] if argv is None else argv
    if raw_argv[:1] == ["_child"]:
        return _child_main(raw_argv[1:])
    raise SystemExit("this module only supports the internal `_child` action")


if __name__ == "__main__":
    raise SystemExit(main())
