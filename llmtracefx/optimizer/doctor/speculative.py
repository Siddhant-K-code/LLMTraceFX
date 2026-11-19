"""'Doctor' analysis rule: is speculative decoding / MTP a net regression?

Compares a set of speculative-decoding (MTP) runs against a set of
comparable autoregressive baseline runs and reports whether speculative
decoding measurably helped, hurt, or produced no reliable signal.
Deliberately conservative: returns
:attr:`DoctorVerdict.INCONCLUSIVE` rather than guessing whenever the runs
are not comparable, too few, or the observed delta is smaller than the
run-to-run noise.

The latency comparison is the client's view. When server metrics evidence
for the speculative runs carries counter-derived acceptance (see
``llmtracefx.optimizer.server_metrics.speculative``), the server's view
can overrule it: a run the server never verified a draft for, a run with no
accepted draft, or a speed-up larger than the acceptance length allows.
``SpeculativeRegressionReport.basis`` says which view decided.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from ..schema import ExperimentRecord


class DoctorVerdict(StrEnum):
    """Outcome of a doctor diagnosis."""

    REGRESSION = "regression"
    """Speculative decoding measurably increased total latency, or the
    server accepted no draft token in any speculative run."""

    IMPROVEMENT = "improvement"
    """Speculative decoding measurably decreased total latency."""

    NO_SIGNIFICANT_DIFFERENCE = "no_significant_difference"
    """A delta was measurable but too small to matter."""

    INCONCLUSIVE = "inconclusive"
    """Evidence was insufficient, incomparable, or too noisy to judge, or
    the server's acceptance evidence contradicts the latency comparison."""


CLIENT_LATENCY = "client_latency"
SERVER_ACCEPTANCE = "server_acceptance"


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
    basis: str = CLIENT_LATENCY
    """``client_latency`` when the verdict comes from the timing comparison,
    ``server_acceptance`` when server acceptance counters decided it."""
    acceptance: Mapping[str, Any] | None = None
    """Server acceptance pooled over the speculative runs, when evidence
    was supplied."""


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
    server_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> SpeculativeRegressionReport:
    """Diagnose whether speculative decoding is a net regression.

    ``baseline_records`` and ``speculative_records`` may contain a mix of
    runs; only successful runs with a recorded total time and the
    expected ``speculative.enabled`` flag are used. ``relative_threshold``
    is the minimum relative delta (default 3%) required to call a
    direction rather than "no significant difference".

    ``server_evidence`` maps a speculative run's ``run_id`` to the server
    metrics evidence recorded for its window. Without it, or when none of
    it carries measured acceptance, the latency verdict stands with basis
    ``client_latency``.
    """
    report = _latency_report(
        baseline_records,
        speculative_records,
        min_repetitions=min_repetitions,
        relative_threshold=relative_threshold,
    )
    if server_evidence is None or report.speculative_mean_total_ms is None:
        return report
    return _apply_acceptance(report, server_evidence)


def _latency_report(
    baseline_records: Sequence[ExperimentRecord],
    speculative_records: Sequence[ExperimentRecord],
    *,
    min_repetitions: int,
    relative_threshold: float,
) -> SpeculativeRegressionReport:
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


_COUNTER_ROLES = ("verify_steps", "draft_tokens", "accepted_tokens", "emitted_tokens")


def _pool_acceptance(
    run_ids: Sequence[str], server_evidence: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    run_status: dict[str, str] = {}
    pooled: dict[str, float] = {}
    roles: set[str] | None = None
    for run_id in run_ids:
        evidence = server_evidence.get(run_id)
        block = None if evidence is None else evidence.get("speculative")
        if not isinstance(block, Mapping):
            run_status[run_id] = "no_evidence"
            continue
        status = str(block.get("status"))
        run_status[run_id] = status
        if status != "measured":
            continue
        counters = block.get("counters")
        if not isinstance(counters, Mapping):
            run_status[run_id] = "no_evidence"
            continue
        values: dict[str, float] = {}
        for role in _COUNTER_ROLES:
            entry = counters.get(role)
            value = entry.get("value") if isinstance(entry, Mapping) else None
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                and value >= 0
            ):
                values[role] = float(value)
        roles = set(values) if roles is None else roles & set(values)
        for role, value in values.items():
            pooled[role] = pooled.get(role, 0.0) + value
    measured = [r for r, status in run_status.items() if status == "measured"]
    usable = sorted(roles or ())
    totals = {role: pooled[role] for role in usable}
    steps = totals.get("verify_steps", 0.0)
    length: float | None = None
    length_basis: str | None = None
    if steps > 0 and "accepted_tokens" in totals:
        length = 1.0 + totals["accepted_tokens"] / steps
        length_basis = "1 + accepted draft tokens / verification steps"
    elif steps > 0 and "emitted_tokens" in totals:
        length = totals["emitted_tokens"] / steps
        length_basis = "tokens emitted / verification steps"
    rate: float | None = None
    if totals.get("draft_tokens", 0.0) > 0 and "accepted_tokens" in totals:
        rate = totals["accepted_tokens"] / totals["draft_tokens"]
    return {
        "provenance": "derived",
        "statistic": "ratio_of_pooled_deltas",
        "runs": len(run_ids),
        "measured_runs": len(measured),
        "run_status": run_status,
        "pooled_counters": {
            role: int(v) if v.is_integer() else v for role, v in totals.items()
        },
        "acceptance_length": length,
        "acceptance_length_basis": length_basis,
        "acceptance_rate": rate,
    }


def _apply_acceptance(
    report: SpeculativeRegressionReport,
    server_evidence: Mapping[str, Mapping[str, Any]],
) -> SpeculativeRegressionReport:
    acceptance = _pool_acceptance(report.speculative_run_ids, server_evidence)
    run_status = acceptance["run_status"]
    no_drafts = sorted(r for r, status in run_status.items() if status == "no_drafts")
    if no_drafts:
        return replace(
            report,
            verdict=DoctorVerdict.INCONCLUSIVE,
            basis=SERVER_ACCEPTANCE,
            acceptance=acceptance,
            reason=(
                f"run(s) {', '.join(no_drafts)} are labelled speculative but the "
                "server ran no verification step in their window, so the "
                "comparison does not measure speculative decoding"
            ),
        )
    measured = acceptance["measured_runs"]
    runs = acceptance["runs"]
    length = acceptance["acceptance_length"]
    if measured < runs or length is None:
        unusable = sorted(
            f"{r} ({status})"
            for r, status in run_status.items()
            if status != "measured"
        )
        detail = (
            f"no usable acceptance for {', '.join(unusable)}"
            if unusable
            else "the pooled counters give no acceptance length"
        )
        return replace(
            report,
            acceptance=acceptance,
            reason=(
                f"{report.reason}; server acceptance covers {measured} of {runs} "
                f"speculative run(s) ({detail}), so the latency verdict stands"
            ),
        )

    baseline_mean = report.baseline_mean_total_ms
    speculative_mean = report.speculative_mean_total_ms
    if (
        report.verdict is DoctorVerdict.IMPROVEMENT
        and baseline_mean is not None
        and speculative_mean is not None
        and speculative_mean > 0
        and baseline_mean / speculative_mean > length
    ):
        speedup = baseline_mean / speculative_mean
        return replace(
            report,
            verdict=DoctorVerdict.INCONCLUSIVE,
            basis=SERVER_ACCEPTANCE,
            acceptance=acceptance,
            reason=(
                f"the speculative runs were {speedup:.2f}x faster, but the server "
                f"emitted {length:.2f} tokens per verification step; a "
                "verification step costs at least one ordinary decode step, so "
                f"speculation alone cannot explain more than {length:.2f}x and "
                "something else differs between the groups"
            ),
        )
    accepted = acceptance["pooled_counters"].get("accepted_tokens")
    if accepted == 0:
        drafted = acceptance["pooled_counters"].get("draft_tokens")
        none = (
            "no draft token" if drafted is None else f"none of {drafted} draft tokens"
        )
        return replace(
            report,
            verdict=DoctorVerdict.REGRESSION,
            basis=SERVER_ACCEPTANCE,
            acceptance=acceptance,
            reason=(
                f"the server accepted {none} across "
                f"{runs} speculative run(s); every verification step emitted "
                "only the target model's token, so drafting was pure overhead "
                f"(latency alone said: {report.reason})"
            ),
        )
    return replace(
        report,
        acceptance=acceptance,
        reason=(
            f"{report.reason}; the server's acceptance length of {length:.2f} "
            "tokens per verification step is consistent with it"
        ),
    )
