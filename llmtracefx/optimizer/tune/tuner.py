"""Constraint evaluation and ranking engine for the ``tune`` command.

Given already loaded, identity-checked evidence (``loader.load_evidence``),
this module:

1. Buckets runs into comparable groups (``identity.GroupKey``) and, within
   each group, into distinct candidate configurations
   (``identity.CandidateKey``).
2. Evaluates every candidate against the policy's constraints, collecting
   *every* violated constraint rather than stopping at the first one.
3. Ranks the candidates that satisfy every constraint by exactly one
   configured objective, with deterministic tie-breaking, and flags a
   group as inconclusive if no candidate survives or the top two
   candidates are indistinguishable from measurement noise.
4. Optionally compares a speculative-decoding winner against the best
   available autoregressive baseline candidate in the same group, reusing
   ``doctor.speculative.diagnose_speculative_regression`` verbatim rather
   than re-implementing similar logic.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..doctor.speculative import diagnose_speculative_regression
from ..schema import Measurement, MetricProvenance, utc_now_iso
from ..workloads.aggregate import correct_cases_per_minute, pass_rate
from .identity import CandidateKey, GroupKey, candidate_key_for, group_key_for
from .loader import ExcludedRun, RunEvidence, load_evidence
from .policy import TuneConstraints, TuneObjective, TunePolicy
from .report import (
    TUNE_REPORT_SCHEMA_VERSION,
    BaselineComparison,
    CandidateReport,
    GroupOutcome,
    GroupReport,
    RejectedCandidateReport,
    TuneReport,
)


def _provenance_allowed(
    measurement: Measurement, allowed: frozenset[MetricProvenance] | None
) -> bool:
    if allowed is None:
        return True
    return measurement.provenance in allowed


@dataclass(frozen=True)
class _CandidateEvaluation:
    """Internal, pre-ranking evaluation of one candidate within a group."""

    candidate_key: CandidateKey
    runs: tuple[RunEvidence, ...]
    eligible_runs: tuple[RunEvidence, ...]
    timed_runs: tuple[RunEvidence, ...]
    accepted: bool
    reasons: tuple[str, ...]
    objective_value: float | None
    mean_total_latency_ms: float | None
    stdev_total_latency_ms: float | None
    coefficient_of_variation: float | None
    correct_cases_per_minute: float | None
    pass_rate: float | None
    mean_quality_score: float | None
    quality_metric: str | None
    mean_peak_memory_bytes: float | None
    max_peak_memory_bytes: float | None


def _evaluate_candidate(
    candidate_key: CandidateKey,
    runs: Sequence[RunEvidence],
    *,
    constraints: TuneConstraints,
    objective: TuneObjective,
) -> _CandidateEvaluation:
    reasons: list[str] = []

    eligible = [
        run for run in runs if run.verification.status in constraints.required_statuses
    ]
    for run in runs:
        if run.verification.status not in constraints.required_statuses:
            reasons.append(
                f"run {run.run_id}: status '{run.verification.status.value}' is "
                "not one of the required statuses "
                f"{sorted(s.value for s in constraints.required_statuses)}"
            )

    # Numeric ceilings are checked across *every* run of this candidate,
    # not only status-eligible ones: a run that is already disqualified by
    # status can still independently violate a latency/memory ceiling, and
    # a rejected candidate's report should show every violation, not stop
    # at the first one.
    for run in runs:
        total = run.final_record.timing.total
        if (
            total is not None
            and _provenance_allowed(total, constraints.allowed_provenances)
            and constraints.max_total_latency_ms is not None
            and total.value > constraints.max_total_latency_ms
        ):
            reasons.append(
                f"run {run.run_id}: total latency {total.value:.2f} ms exceeds "
                f"the maximum {constraints.max_total_latency_ms} ms"
            )
        peak = run.final_record.memory.peak
        if (
            peak is not None
            and _provenance_allowed(peak, constraints.allowed_provenances)
            and constraints.max_peak_memory_bytes is not None
            and peak.value > constraints.max_peak_memory_bytes
        ):
            reasons.append(
                f"run {run.run_id}: peak memory {peak.value:.0f} bytes exceeds "
                f"the maximum {constraints.max_peak_memory_bytes:.0f} bytes"
            )

    if not eligible:
        return _CandidateEvaluation(
            candidate_key=candidate_key,
            runs=tuple(runs),
            eligible_runs=(),
            timed_runs=(),
            accepted=False,
            reasons=tuple(reasons),
            objective_value=None,
            mean_total_latency_ms=None,
            stdev_total_latency_ms=None,
            coefficient_of_variation=None,
            correct_cases_per_minute=None,
            pass_rate=None,
            mean_quality_score=None,
            quality_metric=None,
            mean_peak_memory_bytes=None,
            max_peak_memory_bytes=None,
        )

    # --- Task success / pass rate -------------------------------------
    successes = [run for run in eligible if run.final_record.outcome.success]
    pass_rate_value = pass_rate(len(successes), len(eligible))
    if constraints.min_pass_rate is not None and (
        pass_rate_value is None or pass_rate_value < constraints.min_pass_rate
    ):
        reasons.append(
            f"pass rate {pass_rate_value} is below the required minimum "
            f"{constraints.min_pass_rate}"
        )

    # --- Quality score / metric ----------------------------------------
    quality_metrics_seen: set[str] = set()
    quality_scores: list[float] = []
    for run in eligible:
        metric = run.final_record.outcome.quality_metric
        score = run.final_record.outcome.quality_score
        if metric is not None:
            quality_metrics_seen.add(metric)
        if (
            constraints.required_quality_metric is not None
            and metric is not None
            and metric != constraints.required_quality_metric
        ):
            reasons.append(
                f"run {run.run_id}: quality_metric {metric!r} does not match "
                f"the required quality_metric "
                f"{constraints.required_quality_metric!r}"
            )
        if score is not None:
            quality_scores.append(score)
        elif constraints.min_quality_score is not None:
            reasons.append(
                f"run {run.run_id}: missing outcome.quality_score, required "
                f"by min_quality_score={constraints.min_quality_score}"
            )
    if len(quality_metrics_seen) > 1:
        reasons.append(
            "runs report inconsistent outcome.quality_metric values "
            f"{sorted(quality_metrics_seen)}; they cannot be treated as one "
            "candidate's quality evidence"
        )
    mean_quality = statistics.mean(quality_scores) if quality_scores else None
    if (
        constraints.min_quality_score is not None
        and mean_quality is not None
        and mean_quality < constraints.min_quality_score
    ):
        reasons.append(
            f"mean quality_score {mean_quality:.4f} is below the required "
            f"minimum {constraints.min_quality_score}"
        )
    quality_metric_label = (
        next(iter(quality_metrics_seen)) if len(quality_metrics_seen) == 1 else None
    )

    # --- Timing / latency ------------------------------------------------
    timed: list[tuple[RunEvidence, float]] = []
    for run in eligible:
        total = run.final_record.timing.total
        if total is None:
            reasons.append(f"run {run.run_id}: missing timing.total measurement")
            continue
        if not _provenance_allowed(total, constraints.allowed_provenances):
            allowed = constraints.allowed_provenances or frozenset()
            reasons.append(
                f"run {run.run_id}: timing.total provenance "
                f"'{total.provenance.value}' is not in the allowed provenance "
                f"set {sorted(p.value for p in allowed)}"
            )
            continue
        timed.append((run, total.value))

    mean_total: float | None
    stdev_total: float | None
    cv: float | None
    if not timed:
        reasons.append(
            "no eligible run has a usable timing.total measurement; latency "
            "and throughput cannot be computed for this candidate"
        )
        mean_total = None
        stdev_total = None
        cv = None
    else:
        totals = [total_value for _run, total_value in timed]
        mean_total = statistics.mean(totals)
        stdev_total = statistics.pstdev(totals) if len(totals) > 1 else 0.0
        cv = stdev_total / mean_total if mean_total > 0 else None
        if (
            len(totals) > 1
            and constraints.max_coefficient_of_variation is not None
            and cv is not None
            and cv > constraints.max_coefficient_of_variation
        ):
            reasons.append(
                f"coefficient of variation {cv:.4f} exceeds the maximum "
                f"{constraints.max_coefficient_of_variation}"
            )

    if len(timed) < constraints.min_measured_repetitions:
        reasons.append(
            f"only {len(timed)} run(s) with usable timing evidence; policy "
            f"requires at least {constraints.min_measured_repetitions}"
        )

    # --- Memory ------------------------------------------------------
    peaks: list[float] = []
    for run in eligible:
        peak = run.final_record.memory.peak
        if peak is None:
            if constraints.max_peak_memory_bytes is not None:
                reasons.append(
                    f"run {run.run_id}: missing memory.peak measurement, "
                    "required by max_peak_memory_bytes="
                    f"{constraints.max_peak_memory_bytes}"
                )
            continue
        if not _provenance_allowed(peak, constraints.allowed_provenances):
            if constraints.max_peak_memory_bytes is not None:
                allowed = constraints.allowed_provenances or frozenset()
                reasons.append(
                    f"run {run.run_id}: memory.peak provenance "
                    f"'{peak.provenance.value}' is not in the allowed "
                    f"provenance set "
                    f"{sorted(p.value for p in allowed)}"
                )
            continue
        peaks.append(peak.value)
    mean_peak = statistics.mean(peaks) if peaks else None
    max_peak = max(peaks) if peaks else None

    # --- Objective ------------------------------------------------------
    passing_timed = [
        total_value for run, total_value in timed if run.final_record.outcome.success
    ]
    pass_total_ms = sum(passing_timed)
    ccpm = correct_cases_per_minute(len(passing_timed), pass_total_ms)

    objective_value: float | None
    if objective == TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS:
        objective_value = mean_total
        if objective_value is None:
            reasons.append(
                "objective 'min_mean_total_latency_ms' cannot be computed: no "
                "usable timing.total evidence"
            )
    elif objective == TuneObjective.MAX_CORRECT_CASES_PER_MINUTE:
        objective_value = ccpm
        if objective_value is None:
            reasons.append(
                "objective 'max_correct_cases_per_minute' cannot be computed: "
                "no passing run has usable timing.total evidence"
            )
    else:  # pragma: no cover - TuneObjective is exhaustively defined above
        raise AssertionError(f"unhandled tune objective: {objective!r}")

    accepted = not reasons

    return _CandidateEvaluation(
        candidate_key=candidate_key,
        runs=tuple(runs),
        eligible_runs=tuple(eligible),
        timed_runs=tuple(run for run, _total_value in timed),
        accepted=accepted,
        reasons=tuple(reasons),
        objective_value=objective_value,
        mean_total_latency_ms=mean_total,
        stdev_total_latency_ms=stdev_total,
        coefficient_of_variation=cv,
        correct_cases_per_minute=ccpm,
        pass_rate=pass_rate_value,
        mean_quality_score=mean_quality,
        quality_metric=quality_metric_label,
        mean_peak_memory_bytes=mean_peak,
        max_peak_memory_bytes=max_peak,
    )


def _objective_noise(
    evaluation: _CandidateEvaluation, objective: TuneObjective
) -> float:
    """A rough absolute noise band for tie detection, in the objective's units."""
    if objective == TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS:
        return evaluation.stdev_total_latency_ms or 0.0
    if evaluation.coefficient_of_variation and evaluation.objective_value:
        return abs(evaluation.objective_value * evaluation.coefficient_of_variation)
    return 0.0


def _tie_reason(
    best: _CandidateEvaluation,
    second: _CandidateEvaluation | None,
    objective: TuneObjective,
) -> str | None:
    if second is None:
        return None
    # Both are accepted, so objective_value is never None here.
    best_value = best.objective_value
    second_value = second.objective_value
    if best_value is None or second_value is None:  # pragma: no cover - defensive
        raise AssertionError("accepted candidates must have an objective_value")
    if best_value == second_value:
        return (
            f"top two candidates are exactly tied on {objective.value} "
            f"({best_value:.6g})"
        )
    noise = _objective_noise(best, objective) + _objective_noise(second, objective)
    delta = abs(best_value - second_value)
    if noise > 0 and delta < noise:
        return (
            f"top two candidates differ by {delta:.6g} on {objective.value}, "
            f"smaller than their combined measurement noise ({noise:.6g}); "
            "collect more repetitions to distinguish them"
        )
    return None


def _to_candidate_report(
    evaluation: _CandidateEvaluation, *, rank: int, objective: TuneObjective
) -> CandidateReport:
    objective_value = evaluation.objective_value
    if objective_value is None:  # pragma: no cover - defensive; accepted implies set
        raise AssertionError("accepted candidate must have an objective_value")
    return CandidateReport(
        candidate_key=evaluation.candidate_key,
        rank=rank,
        run_ids=tuple(run.run_id for run in evaluation.eligible_runs),
        verification_paths=tuple(
            str(run.verification_path) for run in evaluation.eligible_runs
        ),
        final_record_paths=tuple(
            str(run.final_record_path) for run in evaluation.eligible_runs
        ),
        evidence_count=len(evaluation.timed_runs),
        objective_name=objective.value,
        objective_value=objective_value,
        mean_total_latency_ms=evaluation.mean_total_latency_ms,
        stdev_total_latency_ms=evaluation.stdev_total_latency_ms,
        coefficient_of_variation=evaluation.coefficient_of_variation,
        correct_cases_per_minute=evaluation.correct_cases_per_minute,
        pass_rate=evaluation.pass_rate,
        mean_quality_score=evaluation.mean_quality_score,
        quality_metric=evaluation.quality_metric,
        mean_peak_memory_bytes=evaluation.mean_peak_memory_bytes,
        max_peak_memory_bytes=evaluation.max_peak_memory_bytes,
    )


def _to_rejected_report(evaluation: _CandidateEvaluation) -> RejectedCandidateReport:
    return RejectedCandidateReport(
        candidate_key=evaluation.candidate_key,
        run_ids=tuple(run.run_id for run in evaluation.runs),
        verification_paths=tuple(str(run.verification_path) for run in evaluation.runs),
        final_record_paths=tuple(str(run.final_record_path) for run in evaluation.runs),
        reasons=evaluation.reasons,
    )


def _maybe_baseline_comparison(
    winner: _CandidateEvaluation,
    accepted: Sequence[_CandidateEvaluation],
) -> BaselineComparison | None:
    """Compare a speculative-decoding winner to the best autoregressive baseline.

    Reuses ``doctor.speculative.diagnose_speculative_regression`` verbatim.
    Returns ``None`` (not attempted, never fabricated) when the winner is
    not speculative-enabled or no accepted autoregressive-baseline
    candidate exists in the same group.
    """
    if not winner.candidate_key.speculative_enabled:
        return None
    baseline_candidates = [
        evaluation
        for evaluation in accepted
        if not evaluation.candidate_key.speculative_enabled
    ]
    if not baseline_candidates:
        return None
    baseline = max(
        baseline_candidates,
        key=lambda evaluation: (
            len(evaluation.timed_runs),
            evaluation.candidate_key.sort_key(),
        ),
    )
    doctor_report = diagnose_speculative_regression(
        [run.final_record for run in baseline.timed_runs],
        [run.final_record for run in winner.timed_runs],
    )
    return BaselineComparison(
        baseline_candidate_key=baseline.candidate_key,
        speculative_candidate_key=winner.candidate_key,
        report=doctor_report,
    )


def _build_group_report(
    group_key: GroupKey,
    candidates_by_key: dict[CandidateKey, list[RunEvidence]],
    policy: TunePolicy,
) -> GroupReport:
    evaluations = [
        _evaluate_candidate(
            candidate_key,
            runs,
            constraints=policy.constraints,
            objective=policy.objective,
        )
        for candidate_key, runs in candidates_by_key.items()
    ]
    evaluations.sort(key=lambda evaluation: evaluation.candidate_key.sort_key())

    accepted_evals = [evaluation for evaluation in evaluations if evaluation.accepted]
    rejected_evals = [
        evaluation for evaluation in evaluations if not evaluation.accepted
    ]

    minimize = policy.objective == TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS

    def _sort_metric(evaluation: _CandidateEvaluation) -> tuple[float, tuple[Any, ...]]:
        value = evaluation.objective_value
        if value is None:  # pragma: no cover - defensive; accepted implies set
            raise AssertionError("accepted candidate must have an objective_value")
        return (value if minimize else -value, evaluation.candidate_key.sort_key())

    accepted_evals.sort(key=_sort_metric)

    rejected_reports = tuple(
        _to_rejected_report(evaluation) for evaluation in rejected_evals
    )

    if not accepted_evals:
        return GroupReport(
            group_key=group_key,
            outcome=GroupOutcome.INCONCLUSIVE,
            recommended=None,
            accepted=(),
            rejected=rejected_reports,
            inconclusive_reason=(
                "no candidate satisfied every constraint in this policy"
            ),
        )

    accepted_reports = tuple(
        _to_candidate_report(evaluation, rank=index + 1, objective=policy.objective)
        for index, evaluation in enumerate(accepted_evals)
    )

    best = accepted_evals[0]
    second = accepted_evals[1] if len(accepted_evals) > 1 else None
    tie_reason = _tie_reason(best, second, policy.objective)

    if tie_reason is not None:
        return GroupReport(
            group_key=group_key,
            outcome=GroupOutcome.INCONCLUSIVE,
            recommended=None,
            accepted=accepted_reports,
            rejected=rejected_reports,
            inconclusive_reason=tie_reason,
        )

    return GroupReport(
        group_key=group_key,
        outcome=GroupOutcome.RECOMMENDED,
        recommended=accepted_reports[0],
        accepted=accepted_reports,
        rejected=rejected_reports,
        inconclusive_reason=None,
        baseline_comparison=_maybe_baseline_comparison(best, accepted_evals),
    )


def _build_groups(
    runs: Sequence[RunEvidence],
) -> tuple[dict[GroupKey, dict[CandidateKey, list[RunEvidence]]], list[ExcludedRun]]:
    groups: dict[GroupKey, dict[CandidateKey, list[RunEvidence]]] = {}
    excluded: list[ExcludedRun] = []
    for run in runs:
        try:
            group_key = group_key_for(run.verification, run.final_record)
            candidate_key = candidate_key_for(run.verification, run.final_record)
        except ValueError as exc:
            excluded.append(
                ExcludedRun(
                    run_id=run.run_id,
                    source_results_dir=run.source_results_dir,
                    reason=f"could not derive a comparable-group identity: {exc}",
                )
            )
            continue
        groups.setdefault(group_key, {}).setdefault(candidate_key, []).append(run)
    return groups, excluded


def tune(*, results_dirs: Sequence[Path], policy: TunePolicy) -> TuneReport:
    """Load, group, constrain, and rank evidence into a full ``TuneReport``."""
    loaded = load_evidence(tuple(results_dirs))
    groups, extra_excluded = _build_groups(loaded.usable)

    group_reports = tuple(
        _build_group_report(group_key, groups[group_key], policy)
        for group_key in sorted(groups, key=lambda key: key.sort_key())
    )

    all_excluded = sorted(
        (*loaded.excluded, *extra_excluded),
        key=lambda run: (run.source_results_dir, run.run_id),
    )

    return TuneReport(
        schema_version=TUNE_REPORT_SCHEMA_VERSION,
        generated_at=utc_now_iso(),
        results_dirs=tuple(str(directory) for directory in results_dirs),
        policy=policy,
        groups=group_reports,
        excluded_runs=tuple(all_excluded),
    )
