"""Tiered execution lifecycle shared by the M5 lab and the Qwen3-8B control.

``lab.core.run_lab`` and ``qwen3_8b.benchmark.run_benchmark`` walk the
manifest's context tiers in order. For each tier they run a safety
preflight, the warmup rows, a warmup safety gate, the measured rows, and a
postflight gate, then record the tier in ``state.json``. The first failed
gate stops the run. This module owns that loop, the ``state.json`` layout,
and the status values. Callers supply how a row is executed and how
strictly a row counts as ok.

Run status moves ``running`` -> ``completed`` or ``running`` -> ``stopped``.
A tier is recorded once with one of the ``TierStatus`` values and never
changes afterwards.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from ..collectors._shared import atomic_write_text, config_hash
from ..schema import utc_now_iso
from ..workloads.schema import ContextTier
from ..workloads.verify import RowResult, RowStatus
from .manifest import ContextTierPin, LabManifest
from .run_owner import current_run_owner


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"


class TierStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    FAILED_WARMUP = "failed_warmup"
    FAILED_WARMUP_SAFETY = "failed_warmup_safety"


class StopPhase(StrEnum):
    WARMUP = "warmup"
    WARMUP_SAFETY = "warmup_safety"
    MEASURED_OR_POSTFLIGHT = "measured_or_postflight"


# Plain strings, for membership checks against values read from state.json.
WARMUP_FAILURE_TIER_STATUSES = frozenset(
    {TierStatus.FAILED_WARMUP.value, TierStatus.FAILED_WARMUP_SAFETY.value}
)

_RUN_TRANSITIONS: dict[RunStatus, frozenset[RunStatus]] = {
    RunStatus.RUNNING: frozenset({RunStatus.COMPLETED, RunStatus.STOPPED}),
    RunStatus.COMPLETED: frozenset(),
    RunStatus.STOPPED: frozenset(),
}


class InvalidRunTransition(ValueError):
    """Raised when a run status change is not allowed."""


def transition_run_status(state: dict[str, Any], new: RunStatus) -> None:
    current = RunStatus(state["status"])
    if new not in _RUN_TRANSITIONS[current]:
        raise InvalidRunTransition(
            f"run status cannot move from {current.value!r} to {new.value!r}"
        )
    state["status"] = new.value


class SafetyDecision(Protocol):
    """The parts of ``lab.core.SafetyDecision`` this module reads."""

    @property
    def safe(self) -> bool:
        """Whether the host may run the next step."""

    @property
    def blockers(self) -> tuple[str, ...]:
        """Reasons the host is not safe."""

    def to_dict(self) -> dict[str, Any]:
        """JSON form stored in ``state.json``."""


class RowRunner(Protocol):
    def __call__(
        self,
        *,
        workload_id: str,
        tier: ContextTier,
        repetition_index: int,
        warmup: bool,
        output_dir: Path,
        resume: bool,
    ) -> RowResult:
        """Execute one warmup or measured row and return its result."""


def row_status_ok(result: RowResult) -> bool:
    return result.verification.status in (RowStatus.COMPLETED, RowStatus.SKIPPED)


def row_ended_as(result: RowResult) -> str:
    return f"{result.entry.run_id} ended as {result.verification.status.value}"


def no_extra_blockers(results: tuple[RowResult, ...]) -> tuple[str, ...]:
    return ()


def peak_bytes(result: RowResult) -> float | None:
    if result.final_record is None or result.final_record.memory.peak is None:
        return None
    return result.final_record.memory.peak.value


def tier_is_safe(
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
        peak = peak_bytes(result)
        if peak is None:
            blockers.append(f"{result.entry.run_id} has no observed MLX peak memory")
        elif peak > manifest.safety.maximum_peak_memory_bytes:
            blockers.append(
                f"{result.entry.run_id} peak memory {peak:.0f} exceeds "
                f"{manifest.safety.maximum_peak_memory_bytes} bytes"
            )
    return not blockers, tuple(blockers)


def write_state(workspace: Path, state: dict[str, Any]) -> None:
    atomic_write_text(
        workspace / "state.json",
        json.dumps(state, indent=2, sort_keys=False) + "\n",
    )


def start_run(
    *,
    workspace: Path,
    schema_version: str,
    lab_id: str,
    requested_max_tier: str,
    preflight: SafetyDecision,
    hardware_fingerprint: str,
    header: dict[str, Any] | None = None,
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the ``running`` state, persist it, and return it.

    ``header`` fields are placed right after ``lab_id``. ``identity`` fields
    are added to the inputs of ``execution_id``.
    """
    state: dict[str, Any] = {
        "schema_version": schema_version,
        "lab_id": lab_id,
        **(header or {}),
        "started_at": utc_now_iso(),
        "status": RunStatus.RUNNING.value,
        "owner": current_run_owner(),
        "requested_max_tier": requested_max_tier,
        "preflight": preflight.to_dict(),
        "tiers": [],
        "stopped_before_tier": None,
        "stopped_during": None,
        "unattempted_tiers": [],
        "stop_reasons": [],
    }
    state["execution_id"] = config_hash(
        {
            "lab_id": lab_id,
            "started_at": state["started_at"],
            "hardware_fingerprint": hardware_fingerprint,
            **(identity or {}),
        }
    )
    workspace.mkdir(parents=True, exist_ok=True)
    write_state(workspace, state)
    return state


def _tier_record(
    tier: ContextTier,
    status: TierStatus,
    results: list[RowResult],
    postflight: SafetyDecision,
    blockers: list[str],
) -> dict[str, Any]:
    return {
        "tier": tier.value,
        "status": status.value,
        "run_ids": [result.entry.run_id for result in results],
        "maximum_observed_peak_memory_bytes": max(
            (peak for result in results if (peak := peak_bytes(result)) is not None),
            default=None,
        ),
        "postflight": postflight.to_dict(),
        "blockers": blockers,
    }


def run_tiers(
    manifest: LabManifest,
    *,
    state: dict[str, Any],
    workspace: Path,
    max_tier: str,
    resume: bool,
    run_row: RowRunner,
    assess: Callable[[], SafetyDecision],
    row_ok: Callable[[RowResult], bool] = row_status_ok,
    describe_failure: Callable[[RowResult], str] = row_ended_as,
    extra_blockers: Callable[
        [tuple[RowResult, ...]], tuple[str, ...]
    ] = no_extra_blockers,
) -> dict[str, Any]:
    """Run every tier up to ``max_tier`` and finish ``state``.

    ``assess`` returns a fresh host safety decision. ``row_ok`` decides
    whether a row may let the run continue. ``describe_failure`` explains a
    failed warmup row that has no recorded reason. ``extra_blockers`` adds
    caller-specific blockers to the warmup and postflight gates.
    """
    selected_max = manifest.tier(max_tier)
    tiers = tuple(
        tier for tier in manifest.context_tiers if tier.order <= selected_max.order
    )
    stop_on_failure = manifest.safety.stop_on_any_failed_row
    results_root = workspace / "results"

    def later_tiers(tier_pin: ContextTierPin) -> list[str]:
        return [
            later.name
            for later in manifest.context_tiers
            if later.order > tier_pin.order and later.order <= selected_max.order
        ]

    for tier_pin in tiers:
        tier = ContextTier(tier_pin.name)
        tier_preflight = assess()
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
                result = run_row(
                    workload_id=warmup_workload_id,
                    tier=tier,
                    repetition_index=index,
                    warmup=True,
                    output_dir=workspace / "warmups" / tier.value,
                    resume=False,
                )
                warmup_results.append(result)
                if stop_on_failure and not row_ok(result):
                    break

        if any(not row_ok(result) for result in warmup_results):
            reasons = [
                result.verification.reason or describe_failure(result)
                for result in warmup_results
                if not row_ok(result)
            ]
            state["tiers"].append(
                _tier_record(
                    tier, TierStatus.FAILED_WARMUP, warmup_results, assess(), reasons
                )
            )
            state["stopped_during"] = {
                "tier": tier.value,
                "phase": StopPhase.WARMUP.value,
            }
            state["unattempted_tiers"] = later_tiers(tier_pin)
            state["stop_reasons"] = reasons
            break

        warmup_postflight = assess()
        warmup_safe, warmup_blockers = tier_is_safe(
            manifest, tuple(warmup_results), warmup_postflight
        )
        warmup_extra = extra_blockers(tuple(warmup_results))
        if warmup_extra:
            warmup_safe = False
            warmup_blockers = tuple(warmup_blockers) + warmup_extra
        if not warmup_safe:
            state["tiers"].append(
                _tier_record(
                    tier,
                    TierStatus.FAILED_WARMUP_SAFETY,
                    warmup_results,
                    warmup_postflight,
                    list(warmup_blockers),
                )
            )
            state["stopped_during"] = {
                "tier": tier.value,
                "phase": StopPhase.WARMUP_SAFETY.value,
            }
            state["unattempted_tiers"] = later_tiers(tier_pin)
            state["stop_reasons"] = list(warmup_blockers)
            break

        tier_results: list[RowResult] = []
        stopped = False
        for workload_pin in manifest.workloads:
            for index in range(manifest.repetitions.measured_per_workload):
                result = run_row(
                    workload_id=workload_pin.workload_id,
                    tier=tier,
                    repetition_index=index,
                    warmup=False,
                    output_dir=results_root,
                    resume=resume,
                )
                tier_results.append(result)
                if stop_on_failure and not row_ok(result):
                    stopped = True
                    break
            if stopped:
                break

        postflight = assess()
        safe, blockers = tier_is_safe(manifest, tuple(tier_results), postflight)
        measured_extra = extra_blockers(tuple(tier_results))
        if measured_extra:
            safe = False
            blockers = tuple(blockers) + measured_extra
        state["tiers"].append(
            _tier_record(
                tier,
                TierStatus.PASSED if safe else TierStatus.FAILED,
                tier_results,
                postflight,
                list(blockers),
            )
        )
        write_state(workspace, state)
        if not safe:
            remaining = later_tiers(tier_pin)
            state["stopped_before_tier"] = remaining[0] if remaining else None
            state["stopped_during"] = {
                "tier": tier.value,
                "phase": StopPhase.MEASURED_OR_POSTFLIGHT.value,
            }
            state["unattempted_tiers"] = remaining
            state["stop_reasons"] = list(blockers)
            break

    state["ended_at"] = utc_now_iso()
    transition_run_status(
        state, RunStatus.STOPPED if state["stop_reasons"] else RunStatus.COMPLETED
    )
    write_state(workspace, state)
    return state
