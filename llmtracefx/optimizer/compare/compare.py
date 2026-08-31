"""Constraint evaluation, single-objective ranking and frontier for ``compare``.

Given already loaded, identity-checked cross-system evidence
(``evidence.load_comparison_evidence``), this module:

1. Splits runs into comparable units (``identity.ComparableUnitKey``) and,
   within each unit, into systems (``identity.SystemKey``). Two runs that do
   not share a unit are never placed side by side, and two runs that do not
   share a system are never averaged together.
2. Evaluates every system against the policy's constraints, collecting
   *every* violation rather than stopping at the first.
3. Ranks the systems that cleared every constraint by exactly one configured
   objective, with deterministic tie-breaking, and declares the unit
   inconclusive when nothing survives or the leaders cannot be told apart
   from measurement noise.
4. Places every ranked system on a Pareto-style evidence frontier, so the
   report can show the trade-off instead of claiming a universal winner.

Formulas are reused, not restated. ``pass_rate`` and
``correct_cases_per_minute`` are imported from ``workloads.aggregate`` (the
same functions the tuner uses), mean/population-stdev/coefficient-of-
variation follow ``tune.tuner`` exactly, the median/p95 pair uses the same
definitions the API collector already uses for its latency distribution
(``statistics.median`` and nearest rank), and the tie-versus-noise test
mirrors ``tune.tuner``'s: a gap smaller than the two leaders' combined noise
band is not a difference.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schema import Measurement, MetricProvenance, utc_now_iso
from ..workloads.aggregate import correct_cases_per_minute, pass_rate
from ..workloads.verify import RowStatus
from .cost import (
    CostBreakdown,
    TokenUsage,
    correct_cases_per_currency_unit,
    cost_per_case,
    estimate_run_cost,
)
from .evidence import (
    CompareEvidenceError,
    SystemRun,
    load_comparison_evidence,
)
from .identity import ComparableUnitKey, SystemKey
from .policy import CompareConstraints, CompareObjective, ComparePolicy
from .pricing import PricingEntry, PricingError, PricingManifest
from .report import (
    COMPARE_REPORT_SCHEMA_VERSION,
    CompareReport,
    CostSummary,
    FrontierEntry,
    ParetoAxis,
    PricingProvenance,
    RejectedSystemReport,
    StratumOutcome,
    StratumReport,
    SystemReport,
    TtftBasis,
    UsageTotals,
)

#: The frontier axes, in reading order. Only the axes every ranked system in
#: a unit has evidence for are used; the rest are dropped from that unit's
#: frontier and named in ``missing_evidence`` rather than filled in.
FRONTIER_AXES: tuple[ParetoAxis, ...] = (
    ParetoAxis.MAX_PASS_RATE,
    ParetoAxis.MAX_CORRECT_CASES_PER_MINUTE,
    ParetoAxis.MIN_MEAN_TOTAL_LATENCY_MS,
    ParetoAxis.MIN_COST_PER_CORRECT_CASE,
)


def _percentiles(values: Sequence[float]) -> tuple[float | None, float | None]:
    """Median and nearest-rank p95, matching the API collector's definitions."""
    if not values:
        return None, None
    ordered = sorted(values)
    rank = max(1, -(-95 * len(ordered) // 100))
    # The median of an even-length list is ``(a + b) / 2`` in float, which
    # overflows to ``inf`` for two large-but-finite timings. Filtering the
    # inputs for finiteness therefore does not make the median finite, and an
    # infinite p50 is not a measurement: it prints as "p50 inf ms" and makes
    # ``to_json`` raise, since the report is written with ``allow_nan=False``.
    # Every other statistic here is already routed through this filter.
    return (
        _finite_or_none(float(statistics.median(ordered))),
        _finite_or_none(ordered[rank - 1]),
    )


def _finite_or_none(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return value


def _provenance_allowed(
    measurement: Measurement, allowed: frozenset[MetricProvenance] | None
) -> bool:
    return allowed is None or measurement.provenance in allowed


@dataclass(frozen=True)
class _SystemEvaluation:
    """Internal, pre-ranking evaluation of one system within a unit."""

    system_key: SystemKey
    runs: tuple[SystemRun, ...]
    eligible_runs: tuple[SystemRun, ...]
    timed_runs: tuple[SystemRun, ...]
    accepted: bool
    reasons: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    objective_value: float | None
    pass_rate: float | None
    mean_quality_score: float | None
    quality_metric: str | None
    mean_total_latency_ms: float | None
    p50_total_latency_ms: float | None
    p95_total_latency_ms: float | None
    stdev_total_latency_ms: float | None
    coefficient_of_variation: float | None
    throughput_coefficient_of_variation: float | None
    correct_cases_per_minute: float | None
    mean_ttft_ms: float | None
    ttft_basis: TtftBasis | None
    usage: UsageTotals | None
    cost: CostSummary | None
    cost_per_correct_case: float | None
    cost_dispersion: float | None
    mean_run_cost: float | None
    correct_cases: int
    mean_peak_memory_bytes: float | None
    max_peak_memory_bytes: float | None


def _collect_ttft(
    runs: Sequence[SystemRun],
) -> tuple[float | None, TtftBasis | None, list[str]]:
    """Mean time-to-first-token, with the measurement it actually is.

    A local prefill and a hosted API's client-observed first-token offset are
    different quantities. If one system somehow produced both, neither is
    reported: averaging them would publish a number that is not a
    measurement of anything.
    """
    missing: list[str] = []
    samples: dict[TtftBasis, list[float]] = defaultdict(list)
    for run in runs:
        if run.api_evidence is not None:
            value = run.api_evidence.client_ttft_ms
            if value is not None:
                samples[TtftBasis.CLIENT_OBSERVED_STREAM].append(value)
                continue
        else:
            value = run.local_prefill_ms
            if value is not None:
                samples[TtftBasis.LOCAL_PREFILL].append(value)
                continue
        missing.append(f"run {run.run_id}: no time-to-first-token was recorded")

    if not samples:
        return None, None, missing
    if len(samples) > 1:
        missing.append(
            "runs mix a local prefill measurement with a client-observed "
            "stream offset; these are different quantities and are not "
            "averaged, so no time-to-first-token is reported"
        )
        return None, None, missing
    basis, values = next(iter(samples.items()))
    return _finite_or_none(statistics.mean(values)), basis, missing


def _collect_usage(runs: Sequence[SystemRun]) -> UsageTotals | None:
    """Sum provider-reported usage across runs, or ``None`` if none reported.

    A total is only produced when *every* run in the set reports that
    component. Summing across the subset that happened to report would emit a
    number smaller than the real total while still labelling it
    ``input_tokens``, and any cost or ratio derived from it would understate
    the truth by however many runs were silently skipped. Partial knowledge
    is reported as no total, with the shortfall visible in
    ``runs_reporting_usage`` versus ``runs_total``.
    """
    reporting = [
        run
        for run in runs
        if run.api_evidence is not None and run.api_evidence.usage_reported
    ]
    if not reporting:
        return None

    def total(field: str) -> int | None:
        values = [
            (
                None
                if run.api_evidence is None or not run.api_evidence.usage_reported
                else getattr(run.api_evidence.usage, field)
            )
            for run in runs
        ]
        # A single unreported component makes the total unknowable, not
        # smaller: it stays null rather than summing the rest.
        if any(value is None for value in values):
            return None
        return sum(int(value) for value in values if value is not None)

    return UsageTotals(
        runs_reporting_usage=len(reporting),
        runs_total=len(runs),
        input_tokens=total("prompt_tokens"),
        output_tokens=total("completion_tokens"),
        cached_input_tokens=total("cached_prompt_tokens"),
        reasoning_tokens=total("reasoning_tokens"),
    )


def _collect_cost(
    runs: Sequence[SystemRun],
    *,
    entry: PricingEntry,
    correct_cases: int,
) -> tuple[CostSummary, float | None, tuple[float, ...]]:
    """Total, per-case and per-correct-case cost for one system's runs.

    Also returns the per-run amounts. Those are the only dispersion this
    module has in monetary units, and a cost objective must not borrow the
    latency coefficient of variation to decide whether two systems are
    distinguishable on price.
    """
    reasons: list[str] = []
    breakdowns: list[CostBreakdown] = []
    for run in runs:
        if run.api_evidence is None or not run.api_evidence.usage_reported:
            reasons.append(
                f"run {run.run_id}: no provider-reported usage, so its cost "
                "cannot be estimated"
            )
            continue
        breakdown = estimate_run_cost(run.api_evidence.usage, entry)
        breakdowns.append(breakdown)
        reasons.extend(f"run {run.run_id}: {reason}" for reason in breakdown.reasons)

    complete = len(breakdowns) == len(runs) and all(
        breakdown.available for breakdown in breakdowns
    )
    total: float | None = None
    if complete and breakdowns:
        total = sum(breakdown.amount or 0.0 for breakdown in breakdowns)
        total = _finite_or_none(total)
        if total is None:  # pragma: no cover - defensive
            reasons.append("summed cost is not a finite number")
    elif breakdowns:
        reasons.append(
            "at least one run could not be priced, so no total is reported; a "
            "partial total would understate the real spend"
        )

    per_case = cost_per_case(total, len(runs))
    per_correct = cost_per_case(total, correct_cases)
    if total is not None and correct_cases == 0:
        reasons.append(
            "no run passed, so cost per correct case is undefined rather than "
            "zero or infinite"
        )
    summary = CostSummary(
        currency=entry.currency,
        pricing_entry_id=entry.entry_id,
        pricing_entry_sha256=entry.content_sha256,
        rates_are_illustrative=entry.rates_are_illustrative,
        total_amount=total,
        cost_per_case=per_case,
        cost_per_correct_case=per_correct,
        correct_cases_per_currency_unit=correct_cases_per_currency_unit(
            correct_cases, total
        ),
        reasons=tuple(dict.fromkeys(reasons)),
    )
    per_run_amounts = tuple(
        breakdown.amount for breakdown in breakdowns if breakdown.amount is not None
    )
    return summary, per_correct, per_run_amounts


def _evaluate_system(
    system_key: SystemKey,
    runs: Sequence[SystemRun],
    *,
    constraints: CompareConstraints,
    objective: CompareObjective,
    pricing: PricingManifest | None,
) -> _SystemEvaluation:
    reasons: list[str] = []
    missing: list[str] = []

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

    if not eligible:
        return _SystemEvaluation(
            system_key=system_key,
            runs=tuple(runs),
            eligible_runs=(),
            timed_runs=(),
            accepted=False,
            reasons=tuple(reasons),
            missing_evidence=tuple(missing),
            objective_value=None,
            pass_rate=None,
            mean_quality_score=None,
            quality_metric=None,
            mean_total_latency_ms=None,
            p50_total_latency_ms=None,
            p95_total_latency_ms=None,
            stdev_total_latency_ms=None,
            coefficient_of_variation=None,
            throughput_coefficient_of_variation=None,
            correct_cases_per_minute=None,
            mean_ttft_ms=None,
            ttft_basis=None,
            usage=None,
            cost=None,
            cost_per_correct_case=None,
            cost_dispersion=None,
            mean_run_cost=None,
            correct_cases=0,
            mean_peak_memory_bytes=None,
            max_peak_memory_bytes=None,
        )

    # --- Correctness -----------------------------------------------------
    successes = [run for run in eligible if run.record.outcome.success]
    pass_rate_value = pass_rate(len(successes), len(eligible))
    if constraints.min_pass_rate is not None and (
        pass_rate_value is None or pass_rate_value < constraints.min_pass_rate
    ):
        reasons.append(
            f"pass rate {pass_rate_value} is below the required minimum "
            f"{constraints.min_pass_rate}"
        )

    # --- Quality ---------------------------------------------------------
    # The evaluator/quality metric is part of the comparable unit key, so
    # every run here already agrees on it; this only reads the scores.
    quality_metric = eligible[0].record.outcome.quality_metric
    if (
        constraints.required_quality_metric is not None
        and quality_metric != constraints.required_quality_metric
    ):
        reasons.append(
            f"quality_metric {quality_metric!r} does not match the required "
            f"quality_metric {constraints.required_quality_metric!r}"
        )
    quality_scores: list[float] = []
    for run in eligible:
        score = run.record.outcome.quality_score
        if score is None:
            if constraints.min_quality_score is not None:
                reasons.append(
                    f"run {run.run_id}: missing outcome.quality_score, required "
                    f"by min_quality_score={constraints.min_quality_score}"
                )
            else:
                missing.append(f"run {run.run_id}: no quality score was recorded")
            continue
        if not math.isfinite(score):
            reasons.append(
                f"run {run.run_id}: outcome.quality_score is non-finite and unusable"
            )
            continue
        quality_scores.append(score)
    mean_quality = (
        _finite_or_none(statistics.mean(quality_scores)) if quality_scores else None
    )
    if (
        constraints.min_quality_score is not None
        and mean_quality is not None
        and mean_quality < constraints.min_quality_score
    ):
        reasons.append(
            f"mean quality_score {mean_quality:.4f} is below the required "
            f"minimum {constraints.min_quality_score}"
        )

    # --- Latency ---------------------------------------------------------
    timed: list[tuple[SystemRun, float]] = []
    for run in eligible:
        total = run.record.timing.total
        if total is None:
            reasons.append(f"run {run.run_id}: missing timing.total measurement")
            continue
        value = run.total_ms
        if value is None:
            reasons.append(f"run {run.run_id}: timing.total is non-finite and unusable")
            continue
        if not _provenance_allowed(total, constraints.allowed_provenances):
            allowed = constraints.allowed_provenances or frozenset()
            reasons.append(
                f"run {run.run_id}: timing.total provenance "
                f"'{total.provenance.value}' is not in the allowed provenance "
                f"set {sorted(p.value for p in allowed)}"
            )
            continue
        timed.append((run, value))

    mean_total: float | None = None
    stdev_total: float | None = None
    cv: float | None = None
    p50: float | None = None
    p95: float | None = None
    if not timed:
        reasons.append(
            "no eligible run has a usable timing.total measurement; latency and "
            "throughput cannot be computed for this system"
        )
    else:
        totals = [value for _run, value in timed]
        mean_total = _finite_or_none(statistics.mean(totals))
        stdev_total = (
            _finite_or_none(statistics.pstdev(totals)) if len(totals) > 1 else 0.0
        )
        p50, p95 = _percentiles(totals)
        cv = (
            _finite_or_none(stdev_total / mean_total)
            if mean_total is not None and stdev_total is not None and mean_total > 0
            else None
        )
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
        if (
            constraints.max_mean_total_latency_ms is not None
            and mean_total is not None
            and mean_total > constraints.max_mean_total_latency_ms
        ):
            reasons.append(
                f"mean total latency {mean_total:.2f} ms exceeds the maximum "
                f"{constraints.max_mean_total_latency_ms} ms"
            )

    if len(timed) < constraints.min_measured_repetitions:
        reasons.append(
            f"only {len(timed)} run(s) with usable timing evidence; policy "
            f"requires at least {constraints.min_measured_repetitions}"
        )

    passing_timed = [value for run, value in timed if run.record.outcome.success]
    ccpm = _finite_or_none(
        correct_cases_per_minute(len(passing_timed), sum(passing_timed))
    )
    # Throughput counts only the runs that both passed and were timed, so its
    # noise band has to come from those same samples. Reusing the latency
    # coefficient of variation above would mix in the failed runs' timings,
    # which contribute nothing to correct cases per minute: a system whose
    # failures are erratic but whose passes are steady would be called noisy
    # on a figure none of that erratic evidence feeds, and a system whose
    # failures happen to be uniform would look steadier than its throughput
    # really is.
    throughput_cv: float | None = None
    if len(passing_timed) > 1:
        passing_mean = _finite_or_none(statistics.mean(passing_timed))
        passing_stdev = _finite_or_none(statistics.pstdev(passing_timed))
        if passing_mean is not None and passing_stdev is not None and passing_mean > 0:
            throughput_cv = _finite_or_none(passing_stdev / passing_mean)

    # --- Time to first token ---------------------------------------------
    mean_ttft, ttft_basis, ttft_missing = _collect_ttft(eligible)
    missing.extend(ttft_missing)

    # --- Local-only memory ------------------------------------------------
    mean_peak: float | None = None
    max_peak: float | None = None
    if system_key.is_local:
        peaks = [
            value
            for value in (run.peak_memory_bytes for run in eligible)
            if value is not None
        ]
        if peaks:
            mean_peak = _finite_or_none(statistics.mean(peaks))
            max_peak = max(peaks)
        else:
            missing.append("no run recorded a local peak memory measurement")
    else:
        missing.append(
            "peak memory is a local-only measurement and is not available for a "
            "system executed by a hosted provider"
        )

    # --- Provider usage and money -----------------------------------------
    usage = _collect_usage(eligible)
    if usage is not None and not usage.complete:
        missing.append(
            f"only {usage.runs_reporting_usage} of {usage.runs_total} run(s) "
            "carry provider-reported token usage"
        )

    cost: CostSummary | None = None
    per_correct_cost: float | None = None
    per_run_costs: tuple[float, ...] = ()
    if pricing is not None:
        try:
            entry = pricing.resolve(
                provider=system_key.provider,
                model_id=system_key.model_id,
                model_revision=system_key.model_revision,
            )
        except PricingError as exc:
            reasons.append(f"pricing lookup failed: {exc}")
            entry = None
        if entry is None:
            missing.append(
                "the pricing manifest has no entry for provider "
                f"{system_key.provider!r} model {system_key.model_id!r}"
                + (
                    f" revision {system_key.model_revision!r}"
                    if system_key.model_revision
                    else ""
                )
                + "; no cost is estimated for this system"
            )
        else:
            cost, per_correct_cost, per_run_costs = _collect_cost(
                eligible, entry=entry, correct_cases=len(successes)
            )

    # The cost ceiling is evaluated here, outside every pricing branch, on
    # purpose. Nested inside "an entry resolved" it would exempt every system
    # the manifest did not price -- which includes every local system, since a
    # provider-keyed manifest can never match one -- and would publish
    # ``accepted`` meaning "cleared every constraint" for a system whose
    # ceiling was never checked. An unevaluable ceiling is a rejection, not a
    # pass.
    if constraints.max_cost_per_correct_case is not None:
        if per_correct_cost is None:
            reasons.append(
                "constraints.max_cost_per_correct_case is configured but no "
                "cost per correct case could be estimated for this system"
            )
        elif per_correct_cost > constraints.max_cost_per_correct_case:
            currency = f" {cost.currency}" if cost is not None else ""
            reasons.append(
                f"estimated cost per correct case {per_correct_cost:.6g}"
                f"{currency} exceeds the maximum "
                f"{constraints.max_cost_per_correct_case}"
            )

    # --- Objective ---------------------------------------------------------
    objective_value: float | None
    if objective == CompareObjective.MIN_MEAN_TOTAL_LATENCY_MS:
        objective_value = mean_total
        if objective_value is None:
            reasons.append(
                "objective 'min_mean_total_latency_ms' cannot be computed: no "
                "usable timing.total evidence"
            )
    elif objective == CompareObjective.MAX_CORRECT_CASES_PER_MINUTE:
        objective_value = ccpm
        if objective_value is None:
            reasons.append(
                "objective 'max_correct_cases_per_minute' cannot be computed: "
                "no passing run has usable timing.total evidence"
            )
    elif objective == CompareObjective.MIN_COST_PER_CORRECT_CASE:
        objective_value = per_correct_cost
        if objective_value is None:
            reasons.append(
                "objective 'min_cost_per_correct_case' cannot be computed: no "
                "estimated cost per correct case is available for this system"
            )
    elif objective == CompareObjective.MAX_CORRECT_CASES_PER_CURRENCY_UNIT:
        objective_value = None if cost is None else cost.correct_cases_per_currency_unit
        if objective_value is None:
            reasons.append(
                "objective 'max_correct_cases_per_currency_unit' cannot be "
                "computed: no estimated spend is available for this system"
            )
    else:  # pragma: no cover - CompareObjective is exhaustively handled above
        raise AssertionError(f"unhandled compare objective: {objective!r}")

    return _SystemEvaluation(
        system_key=system_key,
        runs=tuple(runs),
        eligible_runs=tuple(eligible),
        timed_runs=tuple(run for run, _value in timed),
        accepted=not reasons,
        reasons=tuple(reasons),
        missing_evidence=tuple(dict.fromkeys(missing)),
        objective_value=objective_value,
        pass_rate=pass_rate_value,
        mean_quality_score=mean_quality,
        quality_metric=quality_metric,
        mean_total_latency_ms=mean_total,
        p50_total_latency_ms=p50,
        p95_total_latency_ms=p95,
        stdev_total_latency_ms=stdev_total,
        coefficient_of_variation=cv,
        throughput_coefficient_of_variation=throughput_cv,
        correct_cases_per_minute=ccpm,
        mean_ttft_ms=mean_ttft,
        ttft_basis=ttft_basis,
        usage=usage,
        cost=cost,
        cost_per_correct_case=per_correct_cost,
        cost_dispersion=(
            _finite_or_none(statistics.pstdev(per_run_costs))
            if len(per_run_costs) > 1
            else None
        ),
        mean_run_cost=(
            _finite_or_none(statistics.mean(per_run_costs)) if per_run_costs else None
        ),
        correct_cases=len(successes),
        mean_peak_memory_bytes=mean_peak,
        max_peak_memory_bytes=max_peak,
    )


def _objective_noise(
    evaluation: _SystemEvaluation, objective: CompareObjective
) -> float:
    """A rough absolute noise band in the objective's own units.

    Each objective uses the dispersion of the quantity it actually ranks on.
    Latency uses the measured population standard deviation of the timings,
    following ``tune.tuner._objective_noise``. The two money objectives use
    the dispersion of the per-run costs, because the timing coefficient of
    variation says nothing about whether two systems differ on price: a
    system can be metronomically steady in latency while its token usage, and
    therefore its cost, swings run to run, and borrowing the timing figure
    would then declare a real price difference to be noise, or vice versa.

    Returning ``0.0`` means "no dispersion evidence", which leaves only an
    exact tie detectable. That is the honest default: with one priced run
    there is nothing to estimate a band from, and inventing one from an
    unrelated axis would be worse than admitting the gap.
    """
    if objective == CompareObjective.MIN_MEAN_TOTAL_LATENCY_MS:
        return evaluation.stdev_total_latency_ms or 0.0

    if objective == CompareObjective.MIN_COST_PER_CORRECT_CASE:
        # The per-run amounts are per *attempt*. The objective divides the
        # total by the number of *correct* cases, so with mixed pass/fail the
        # two are on different scales and the raw attempt dispersion
        # understates the band by exactly the pass rate. Scaling by
        # attempts/correct puts it back in the objective's units: with four
        # attempts and two passes, a swing of x per attempt moves cost per
        # correct case by 2x.
        if evaluation.cost_dispersion is None or not evaluation.correct_cases:
            return 0.0
        attempts = len(evaluation.eligible_runs)
        if attempts <= 0:  # pragma: no cover - eligible_runs is non-empty here
            return 0.0
        return abs(evaluation.cost_dispersion * attempts / evaluation.correct_cases)

    if objective == CompareObjective.MAX_CORRECT_CASES_PER_CURRENCY_UNIT:
        # This objective is the reciprocal of a cost, so the band is scaled
        # by the *relative* cost dispersion rather than its absolute value.
        # A relative spread is already dimensionless, so the attempts-versus-
        # correct-cases scaling above does not apply: it cancels in the
        # ratio.
        if (
            evaluation.cost_dispersion
            and evaluation.mean_run_cost
            and evaluation.objective_value
        ):
            relative = evaluation.cost_dispersion / evaluation.mean_run_cost
            return abs(evaluation.objective_value * relative)
        return 0.0

    # Throughput. Its dispersion comes from the passing timed runs only,
    # which are exactly the samples correct cases per minute is computed
    # from.
    if evaluation.throughput_coefficient_of_variation and evaluation.objective_value:
        return abs(
            evaluation.objective_value * evaluation.throughput_coefficient_of_variation
        )
    return 0.0


def _tie_reason(
    best: _SystemEvaluation,
    second: _SystemEvaluation | None,
    objective: CompareObjective,
) -> str | None:
    """Whether the top two systems are indistinguishable on this evidence."""
    if second is None:
        return None
    best_value = best.objective_value
    second_value = second.objective_value
    if best_value is None or second_value is None:  # pragma: no cover - defensive
        raise AssertionError("accepted systems must have an objective_value")
    if best_value == second_value:
        return (
            f"top two systems are exactly tied on {objective.value} "
            f"({best_value:.6g})"
        )
    noise = _objective_noise(best, objective) + _objective_noise(second, objective)
    delta = abs(best_value - second_value)
    if noise > 0 and delta < noise:
        return (
            f"top two systems differ by {delta:.6g} on {objective.value}, "
            f"smaller than their combined measurement noise ({noise:.6g}); "
            "collect more repetitions to distinguish them"
        )
    return None


def _axis_value(evaluation: _SystemEvaluation, axis: ParetoAxis) -> float | None:
    if axis == ParetoAxis.MAX_PASS_RATE:
        return evaluation.pass_rate
    if axis == ParetoAxis.MAX_CORRECT_CASES_PER_MINUTE:
        return evaluation.correct_cases_per_minute
    if axis == ParetoAxis.MIN_MEAN_TOTAL_LATENCY_MS:
        return evaluation.mean_total_latency_ms
    return evaluation.cost_per_correct_case


def _build_frontier(
    evaluations: Sequence[_SystemEvaluation],
) -> tuple[tuple[ParetoAxis, ...], tuple[FrontierEntry, ...], list[str]]:
    """Non-dominated systems over every axis all of them have evidence for.

    Dominance is computed on point estimates, so a system that wins only
    inside the noise band still shows as dominating. That is why the frontier
    is presented as evidence rather than as a verdict, and why the ranking
    (which does apply the noise test) is reported separately.
    """
    notes: list[str] = []
    if not evaluations:
        return (), (), notes

    usable_axes = tuple(
        axis
        for axis in FRONTIER_AXES
        if all(_axis_value(item, axis) is not None for item in evaluations)
    )
    dropped = [axis.value for axis in FRONTIER_AXES if axis not in usable_axes]
    if dropped:
        notes.append(
            "frontier axes "
            + ", ".join(sorted(dropped))
            + " were dropped because at least one ranked system has no evidence "
            "for them"
        )
    if not usable_axes:
        notes.append(
            "no frontier could be computed: the ranked systems share no axis "
            "on which all of them carry evidence"
        )
        return (), (), notes

    def dominates(left: _SystemEvaluation, right: _SystemEvaluation) -> bool:
        strictly_better = False
        for axis in usable_axes:
            left_value = _axis_value(left, axis)
            right_value = _axis_value(right, axis)
            assert left_value is not None and right_value is not None
            if axis.prefers_lower:
                if left_value > right_value:
                    return False
                if left_value < right_value:
                    strictly_better = True
            else:
                if left_value < right_value:
                    return False
                if left_value > right_value:
                    strictly_better = True
        return strictly_better

    entries: list[FrontierEntry] = []
    for candidate in evaluations:
        dominators = tuple(
            other.system_key.label()
            for other in evaluations
            if other is not candidate and dominates(other, candidate)
        )
        entries.append(
            FrontierEntry(
                system_key=candidate.system_key,
                dominated=bool(dominators),
                dominated_by=dominators,
            )
        )
    return usable_axes, tuple(entries), notes


def _to_system_report(
    evaluation: _SystemEvaluation, *, rank: int, objective: CompareObjective
) -> SystemReport:
    objective_value = evaluation.objective_value
    if objective_value is None:  # pragma: no cover - accepted implies set
        raise AssertionError("accepted system must have an objective_value")
    return SystemReport(
        system_key=evaluation.system_key,
        rank=rank,
        run_ids=tuple(run.run_id for run in evaluation.eligible_runs),
        verification_paths=tuple(
            str(run.verification_path) for run in evaluation.eligible_runs
        ),
        record_paths=tuple(str(run.record_path) for run in evaluation.eligible_runs),
        evidence_count=len(evaluation.timed_runs),
        objective_name=objective.value,
        objective_value=objective_value,
        pass_rate=evaluation.pass_rate,
        mean_quality_score=evaluation.mean_quality_score,
        quality_metric=evaluation.quality_metric,
        mean_total_latency_ms=evaluation.mean_total_latency_ms,
        p50_total_latency_ms=evaluation.p50_total_latency_ms,
        p95_total_latency_ms=evaluation.p95_total_latency_ms,
        stdev_total_latency_ms=evaluation.stdev_total_latency_ms,
        coefficient_of_variation=evaluation.coefficient_of_variation,
        correct_cases_per_minute=evaluation.correct_cases_per_minute,
        mean_ttft_ms=evaluation.mean_ttft_ms,
        ttft_basis=evaluation.ttft_basis,
        usage=evaluation.usage,
        cost=evaluation.cost,
        mean_peak_memory_bytes=evaluation.mean_peak_memory_bytes,
        max_peak_memory_bytes=evaluation.max_peak_memory_bytes,
        missing_evidence=evaluation.missing_evidence,
    )


def _to_rejected_report(evaluation: _SystemEvaluation) -> RejectedSystemReport:
    return RejectedSystemReport(
        system_key=evaluation.system_key,
        run_ids=tuple(run.run_id for run in evaluation.runs),
        verification_paths=tuple(str(run.verification_path) for run in evaluation.runs),
        record_paths=tuple(str(run.record_path) for run in evaluation.runs),
        reasons=evaluation.reasons,
    )


def _build_stratum(
    unit_key: ComparableUnitKey,
    runs: Sequence[SystemRun],
    *,
    policy: ComparePolicy,
    pricing: PricingManifest | None,
) -> StratumReport:
    by_system: dict[SystemKey, list[SystemRun]] = defaultdict(list)
    for run in runs:
        by_system[run.system_key].append(run)

    evaluations = [
        _evaluate_system(
            system_key,
            sorted(system_runs, key=lambda run: run.run_id),
            constraints=policy.constraints,
            objective=policy.objective,
            pricing=pricing,
        )
        for system_key, system_runs in sorted(
            by_system.items(), key=lambda item: item[0].sort_key()
        )
    ]

    accepted = [item for item in evaluations if item.accepted]
    rejected = tuple(
        _to_rejected_report(item) for item in evaluations if not item.accepted
    )

    # Deterministic ordering: objective first, then the system's own sort key,
    # so two systems with an identical objective value never swap places
    # between runs. The objective value is read explicitly rather than through
    # an ``or`` fallback, which would silently substitute a value for a
    # genuine 0.0 objective and would hide a missing one behind a plausible
    # number instead of failing.
    def rank_sort_key(evaluation: _SystemEvaluation) -> tuple[Any, ...]:
        value = evaluation.objective_value
        if value is None:  # pragma: no cover - defensive; accepted implies set
            raise AssertionError("accepted system must have an objective_value")
        return (
            value if policy.objective.prefers_lower else -value,
            evaluation.system_key.sort_key(),
        )

    accepted.sort(key=rank_sort_key)

    missing: list[str] = []
    if len(by_system) < 2:
        missing.append(
            f"only {len(by_system)} system produced evidence for this comparable "
            "unit; a comparison needs at least two"
        )

    frontier_axes, frontier, frontier_notes = _build_frontier(accepted)
    missing.extend(frontier_notes)

    ranked = tuple(
        _to_system_report(item, rank=index, objective=policy.objective)
        for index, item in enumerate(accepted, start=1)
    )

    if not accepted:
        return StratumReport(
            unit_key=unit_key,
            outcome=StratumOutcome.INCONCLUSIVE,
            objective_name=policy.objective.value,
            ranked=(),
            rejected=rejected,
            recommended=None,
            inconclusive_reason=(
                "no system cleared every constraint on the available evidence"
            ),
            frontier_axes=frontier_axes,
            frontier=frontier,
            missing_evidence=tuple(dict.fromkeys(missing)),
        )

    tie = _tie_reason(
        accepted[0], accepted[1] if len(accepted) > 1 else None, policy.objective
    )
    if tie is not None:
        return StratumReport(
            unit_key=unit_key,
            outcome=StratumOutcome.INCONCLUSIVE,
            objective_name=policy.objective.value,
            ranked=ranked,
            rejected=rejected,
            recommended=None,
            inconclusive_reason=tie,
            frontier_axes=frontier_axes,
            frontier=frontier,
            missing_evidence=tuple(dict.fromkeys(missing)),
        )

    if len(by_system) < 2:
        return StratumReport(
            unit_key=unit_key,
            outcome=StratumOutcome.INCONCLUSIVE,
            objective_name=policy.objective.value,
            ranked=ranked,
            rejected=rejected,
            recommended=None,
            inconclusive_reason=(
                "only one system produced evidence for this comparable unit, so "
                "nothing was compared against anything"
            ),
            frontier_axes=frontier_axes,
            frontier=frontier,
            missing_evidence=tuple(dict.fromkeys(missing)),
        )

    return StratumReport(
        unit_key=unit_key,
        outcome=StratumOutcome.RECOMMENDED,
        objective_name=policy.objective.value,
        ranked=ranked,
        rejected=rejected,
        recommended=ranked[0],
        inconclusive_reason=None,
        frontier_axes=frontier_axes,
        frontier=frontier,
        missing_evidence=tuple(dict.fromkeys(missing)),
    )


def compare(
    *,
    results_dirs: tuple[Path, ...],
    policy: ComparePolicy,
    pricing: PricingManifest | None = None,
    pricing_manifest_path: str | None = None,
    tune_report_paths: tuple[str, ...] = (),
) -> CompareReport:
    """Compare already-collected evidence across systems, offline.

    Reads nothing but artifacts on disk: no model is loaded, no API is
    called, and no benchmark is executed.
    """
    if policy.objective.requires_cost and pricing is None:
        raise CompareEvidenceError(
            f"objective {policy.objective.value!r} ranks on money, so a pricing "
            "manifest is required; refusing to rank on a cost that was never "
            "supplied"
        )
    if policy.constraints.max_cost_per_correct_case is not None and pricing is None:
        raise CompareEvidenceError(
            "constraints.max_cost_per_correct_case is configured but no pricing "
            "manifest was supplied; refusing to run a comparison whose cost "
            "ceiling could never be evaluated for any system"
        )
    if pricing is not None and pricing_manifest_path is None:
        raise CompareEvidenceError(
            "a pricing manifest was supplied without its path; every monetary "
            "value must be attributable to the file it came from"
        )

    loaded = load_comparison_evidence(results_dirs)

    by_unit: dict[ComparableUnitKey, list[SystemRun]] = defaultdict(list)
    for run in loaded.runs:
        by_unit[run.unit_key].append(run)

    strata = tuple(
        _build_stratum(unit_key, unit_runs, policy=policy, pricing=pricing)
        for unit_key, unit_runs in sorted(
            by_unit.items(), key=lambda item: item[0].sort_key()
        )
    )

    provenance: PricingProvenance | None = None
    if pricing is not None and pricing_manifest_path is not None:
        used = sorted(
            {
                system.cost.pricing_entry_id
                for stratum in strata
                for system in stratum.ranked
                if system.cost is not None
            }
        )
        provenance = PricingProvenance(
            manifest_path=pricing_manifest_path,
            manifest_sha256=pricing.content_sha256,
            currency=pricing.currency,
            rates_are_illustrative=pricing.rates_are_illustrative,
            entry_ids_used=tuple(used),
        )

    return CompareReport(
        schema_version=COMPARE_REPORT_SCHEMA_VERSION,
        generated_at=utc_now_iso(),
        results_dirs=tuple(str(path) for path in results_dirs),
        tune_report_paths=tune_report_paths,
        policy=policy,
        strata=strata,
        pricing=provenance,
        excluded_runs=loaded.excluded,
    )


__all__ = [
    "FRONTIER_AXES",
    "RowStatus",
    "TokenUsage",
    "compare",
]
