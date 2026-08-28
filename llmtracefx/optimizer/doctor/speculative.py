"""'Doctor' analysis rule: is speculative decoding / MTP a net regression?

Compares a set of speculative-decoding (MTP) runs against a set of
comparable autoregressive baseline runs and reports whether speculative
decoding measurably helped, hurt, or produced no reliable signal.
Deliberately conservative: returns
:attr:`DoctorVerdict.INCONCLUSIVE` rather than guessing whenever the runs
are not comparable, too few, or the observed delta is smaller than the
run-to-run noise.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from ..schema import ExperimentRecord


class DoctorVerdict(str, Enum):
    """Outcome of a doctor diagnosis."""

    REGRESSION = "regression"
    """Speculative decoding measurably increased total latency."""

    IMPROVEMENT = "improvement"
    """Speculative decoding measurably decreased total latency."""

    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"
    """A delta was measurable but too small to matter."""

    INCONCLUSIVE = "inconclusive"
    """Evidence was insufficient, incomparable, or too noisy to judge."""


ComparabilityKey = tuple[
    str,
    str | None,
    str | None,
    str,
    str | None,
    str | None,
    str,
    str,
    str | None,
    int | None,
    int | None,
]


def comparability_key(record: ExperimentRecord) -> ComparabilityKey:
    """Key used to decide whether two runs are comparable.

    Two runs are only comparable for a latency comparison if they share
    the same model/quantization, runtime/backend, hardware/OS, and
    workload shape (input/generated token counts).
    """
    return (
        record.model.model_id,
        record.model.model_revision,
        record.model.quantization,
        record.runtime.name,
        record.runtime.version,
        record.runtime.backend,
        record.platform.os_name,
        record.platform.architecture,
        record.platform.accelerator,
        record.tokens.input_tokens,
        record.tokens.generated_tokens,
    )


@dataclass(frozen=True)
class SpeculativeRegressionReport:
    """Result of comparing speculative-decoding runs to a baseline."""

    verdict: DoctorVerdict
    reason: str
    baseline_run_ids: tuple[str, ...] = ()
    speculative_run_ids: tuple[str, ...] = ()
    baseline_mean_total_ms: float | None = None
    speculative_mean_total_ms: float | None = None
    delta_ms: float | None = None
    delta_pct: float | None = None


def _eligible_totals(
    records: Sequence[ExperimentRecord], *, speculative_enabled: bool
) -> list[ExperimentRecord]:
    return [
        record
        for record in records
        if record.outcome.success
        and record.speculative.enabled is speculative_enabled
        and record.timing.total is not None
    ]


def diagnose_speculative_regression(
    baseline_records: Sequence[ExperimentRecord],
    speculative_records: Sequence[ExperimentRecord],
    *,
    min_repetitions: int = 2,
    relative_threshold: float = 0.03,
) -> SpeculativeRegressionReport:
    """Diagnose whether speculative decoding is a net regression.

    ``baseline_records`` and ``speculative_records`` may contain a mix of
    runs; only successful runs with a recorded total time and the
    expected ``speculative.enabled`` flag are used. ``relative_threshold``
    is the minimum relative delta (default 3%) required to call a
    direction rather than "no significant difference".
    """
    baseline = _eligible_totals(baseline_records, speculative_enabled=False)
    speculative = _eligible_totals(speculative_records, speculative_enabled=True)

    if not baseline or not speculative:
        return SpeculativeRegressionReport(
            verdict=DoctorVerdict.INCONCLUSIVE,
            reason=(
                "no successful runs with a recorded total time were found for "
                f"{'the baseline' if not baseline else 'the speculative-decoding'} group"
            ),
        )

    baseline_keys = {comparability_key(record) for record in baseline}
    speculative_keys = {comparability_key(record) for record in speculative}
    if len(baseline_keys) > 1 or len(speculative_keys) > 1:
        return SpeculativeRegressionReport(
            verdict=DoctorVerdict.INCONCLUSIVE,
            reason="runs within a group are not comparable (mixed model/runtime/hardware/workload)",
            baseline_run_ids=tuple(r.run_id for r in baseline),
            speculative_run_ids=tuple(r.run_id for r in speculative),
        )
    if baseline_keys != speculative_keys:
        return SpeculativeRegressionReport(
            verdict=DoctorVerdict.INCONCLUSIVE,
            reason=(
                "baseline and speculative-decoding runs are not comparable "
                "(different model/runtime/hardware/workload)"
            ),
            baseline_run_ids=tuple(r.run_id for r in baseline),
            speculative_run_ids=tuple(r.run_id for r in speculative),
        )

    if len(baseline) < min_repetitions or len(speculative) < min_repetitions:
        return SpeculativeRegressionReport(
            verdict=DoctorVerdict.INCONCLUSIVE,
            reason=(
                f"need at least {min_repetitions} comparable repetitions per group "
                f"(got {len(baseline)} baseline, {len(speculative)} speculative)"
            ),
            baseline_run_ids=tuple(r.run_id for r in baseline),
            speculative_run_ids=tuple(r.run_id for r in speculative),
        )

    baseline_totals = [
        r.timing.total.value for r in baseline if r.timing.total is not None
    ]
    speculative_totals = [
        r.timing.total.value for r in speculative if r.timing.total is not None
    ]

    baseline_mean = statistics.mean(baseline_totals)
    speculative_mean = statistics.mean(speculative_totals)

    # The generic Measurement schema only requires timing values to be
    # >= 0 (zero and, since it does not check finiteness, NaN/inf all
    # pass ExperimentRecord.validate()). None of those are usable
    # evidence here: a zero-or-negative baseline makes the relative delta
    # below undefined/infinite, and a non-finite mean would silently
    # propagate into a nonsensical verdict rather than a crash.
    if not math.isfinite(baseline_mean) or baseline_mean <= 0:
        return SpeculativeRegressionReport(
            verdict=DoctorVerdict.INCONCLUSIVE,
            reason=(
                "baseline mean total time is not a positive, finite measurement "
                f"({baseline_mean!r} ms); cannot compute a reliable relative delta"
            ),
            baseline_run_ids=tuple(r.run_id for r in baseline),
            speculative_run_ids=tuple(r.run_id for r in speculative),
        )
    if not math.isfinite(speculative_mean) or speculative_mean < 0:
        return SpeculativeRegressionReport(
            verdict=DoctorVerdict.INCONCLUSIVE,
            reason=(
                "speculative-decoding mean total time is not a finite, non-negative "
                f"measurement ({speculative_mean!r} ms); cannot compute a reliable delta"
            ),
            baseline_run_ids=tuple(r.run_id for r in baseline),
            speculative_run_ids=tuple(r.run_id for r in speculative),
        )

    baseline_noise = statistics.pstdev(baseline_totals)
    speculative_noise = statistics.pstdev(speculative_totals)

    delta_ms = speculative_mean - baseline_mean
    # baseline_mean is guaranteed positive and finite by the guard above,
    # so this division is always well-defined -- delta_pct is never None
    # by the time it reaches the percentage-formatted reasons below.
    delta_pct = delta_ms / baseline_mean

    run_ids = tuple(r.run_id for r in baseline), tuple(r.run_id for r in speculative)

    combined_noise = baseline_noise + speculative_noise
    if combined_noise > 0 and abs(delta_ms) < combined_noise:
        return SpeculativeRegressionReport(
            verdict=DoctorVerdict.INCONCLUSIVE,
            reason=(
                f"measured delta ({delta_ms:+.2f} ms) is smaller than the combined "
                f"repetition noise ({combined_noise:.2f} ms); collect more repetitions"
            ),
            baseline_run_ids=run_ids[0],
            speculative_run_ids=run_ids[1],
            baseline_mean_total_ms=baseline_mean,
            speculative_mean_total_ms=speculative_mean,
            delta_ms=delta_ms,
            delta_pct=delta_pct,
        )

    if abs(delta_pct) < relative_threshold:
        verdict = DoctorVerdict.NO_SIGNIFICANT_DIFFERENCE
        reason = f"delta ({delta_pct:+.1%}) is below the {relative_threshold:.0%} significance threshold"
    elif delta_ms > 0:
        verdict = DoctorVerdict.REGRESSION
        reason = (
            f"speculative decoding increased mean total time by {delta_ms:.2f} ms "
            f"({delta_pct:+.1%}) versus the autoregressive baseline"
        )
    else:
        verdict = DoctorVerdict.IMPROVEMENT
        reason = (
            f"speculative decoding decreased mean total time by {-delta_ms:.2f} ms "
            f"({delta_pct:+.1%}) versus the autoregressive baseline"
        )

    return SpeculativeRegressionReport(
        verdict=verdict,
        reason=reason,
        baseline_run_ids=run_ids[0],
        speculative_run_ids=run_ids[1],
        baseline_mean_total_ms=baseline_mean,
        speculative_mean_total_ms=speculative_mean,
        delta_ms=delta_ms,
        delta_pct=delta_pct,
    )
