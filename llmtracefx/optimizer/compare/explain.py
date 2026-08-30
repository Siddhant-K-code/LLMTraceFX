"""Human-readable rendering of an already-computed ``CompareReport``.

Mirrors ``tune.explain``: a pure formatter over a validated report. It never
recomputes a value, never re-ranks anything, and never fills a gap. A metric
that is unavailable prints as ``n/a`` with the reason available in the
report's ``missing_evidence``, so a reader can see what was not known rather
than being handed a plausible-looking zero.
"""

from __future__ import annotations

from .policy import CompareConstraints
from .report import CompareReport, StratumOutcome, StratumReport, SystemReport


def _number(value: float | None, *, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _money(value: float | None, currency: str) -> str:
    return "n/a" if value is None else f"{value:.6g} {currency}"


def _system_line(system: SystemReport) -> str:
    parts = [
        f"  #{system.rank} {system.system_key.label()}",
        f"      objective {system.objective_name} = "
        + f"{_number(system.objective_value)}",
        f"      pass rate {_percent(system.pass_rate)}"
        + f"  quality {system.quality_metric or 'n/a'} = "
        + f"{_number(system.mean_quality_score, digits=3)}",
        f"      latency mean {_number(system.mean_total_latency_ms, digits=2)} ms"
        + f"  p50 {_number(system.p50_total_latency_ms, digits=2)} ms"
        + f"  p95 {_number(system.p95_total_latency_ms, digits=2)} ms",
    ]
    if system.mean_ttft_ms is not None and system.ttft_basis is not None:
        parts.append(
            f"      ttft {_number(system.mean_ttft_ms, digits=2)} ms "
            f"({system.ttft_basis.value})"
        )
    parts.append(
        f"      correct cases/min {_number(system.correct_cases_per_minute, digits=3)}"
    )
    if system.usage is not None:
        usage = system.usage
        parts.append(
            "      provider-reported usage: "
            f"input={usage.input_tokens} output={usage.output_tokens} "
            f"cached={usage.cached_input_tokens} "
            f"reasoning={usage.reasoning_tokens} "
            f"({usage.runs_reporting_usage}/{usage.runs_total} runs)"
        )
    if system.cost is not None:
        cost = system.cost
        label = "illustrative" if cost.rates_are_illustrative else "estimated"
        parts.append(
            f"      cost ({label}): total "
            f"{_money(cost.total_amount, cost.currency)}"
            f"  per correct case "
            f"{_money(cost.cost_per_correct_case, cost.currency)}"
            f"  correct per {cost.currency} "
            f"{_number(cost.correct_cases_per_currency_unit, digits=3)}"
        )
    if system.system_key.is_local and system.mean_peak_memory_bytes is not None:
        parts.append(
            "      local peak memory "
            f"{system.mean_peak_memory_bytes / (1024 * 1024):.1f} MB mean"
        )
    return "\n".join(parts)


def _stratum_text(
    stratum: StratumReport, *, constraints: CompareConstraints, verbose: bool
) -> str:
    lines = [
        f"Comparable unit: {stratum.unit_key.label()}",
        f"  outcome: {stratum.outcome.value} (objective {stratum.objective_name})",
    ]
    if stratum.outcome == StratumOutcome.RECOMMENDED and stratum.recommended:
        lines.append(
            f"  recommended for this unit only: "
            f"{stratum.recommended.system_key.label()}"
        )
    elif stratum.inconclusive_reason:
        lines.append(f"  inconclusive: {stratum.inconclusive_reason}")

    # A verdict without the bar it cleared is not readable evidence, and the
    # README promises each recommendation is stated with its constraints.
    lines.append("  Constraints in force:")
    lines.extend(f"      {item}" for item in constraints.active_summary())

    if stratum.ranked:
        lines.append("  Ranked systems:")
        shown = stratum.ranked if verbose else stratum.ranked[:3]
        lines.extend(_system_line(system) for system in shown)
        if not verbose and len(stratum.ranked) > len(shown):
            lines.append(
                f"      ... {len(stratum.ranked) - len(shown)} more "
                "(rerun with --explain)"
            )

    if stratum.frontier:
        on_frontier = [
            entry.system_key.label()
            for entry in stratum.frontier
            if not entry.dominated
        ]
        axes = ", ".join(axis.value for axis in stratum.frontier_axes)
        lines.append(f"  Frontier ({axes}):")
        for label in on_frontier:
            lines.append(f"      on frontier: {label}")
        if verbose:
            for entry in stratum.frontier:
                if entry.dominated:
                    lines.append(
                        f"      dominated:   {entry.system_key.label()} "
                        f"(by {', '.join(entry.dominated_by)})"
                    )

    if stratum.rejected:
        lines.append(f"  Rejected systems ({len(stratum.rejected)}):")
        for system in stratum.rejected:
            lines.append(f"      {system.system_key.label()}")
            reasons = system.reasons if verbose else system.reasons[:1]
            lines.extend(f"        - {reason}" for reason in reasons)
            if not verbose and len(system.reasons) > 1:
                lines.append(
                    f"        - ... {len(system.reasons) - 1} more "
                    "(rerun with --explain)"
                )

    if stratum.missing_evidence:
        lines.append("  Missing evidence:")
        lines.extend(f"      - {note}" for note in stratum.missing_evidence)
    return "\n".join(lines)


def format_compare_report_text(report: CompareReport, *, verbose: bool = False) -> str:
    """Render ``report`` as plain text for a terminal."""
    lines = [
        f"Cross-system comparison ({report.policy.name or 'unnamed policy'})",
        f"  generated at: {report.generated_at}",
        f"  objective:    {report.policy.objective.value}",
        f"  comparable units: {len(report.strata)}",
    ]
    if report.pricing is not None:
        pricing = report.pricing
        lines.append(
            f"  pricing:      {pricing.manifest_path} "
            f"({pricing.currency}, sha256 {pricing.manifest_sha256[:12]})"
        )
        if pricing.rates_are_illustrative:
            lines.append(
                "                rates are declared illustrative examples; "
                "every monetary figure below is a demonstration, not a price"
            )
    else:
        lines.append("  pricing:      none supplied, so no monetary values")

    if not report.strata:
        lines.append("")
        lines.append(
            "No comparable units were found. Nothing was compared, so nothing "
            "is recommended."
        )
        return "\n".join(lines)

    for stratum in report.strata:
        lines.append("")
        lines.append(
            _stratum_text(
                stratum,
                constraints=report.policy.constraints,
                verbose=verbose,
            )
        )

    if report.excluded_runs:
        lines.append("")
        lines.append(f"Excluded runs ({len(report.excluded_runs)}):")
        for run in report.excluded_runs:
            lines.append(f"  {run.run_id}: {run.reason}")

    lines.append("")
    lines.append(
        "Every recommendation above applies only to the workload, context "
        "tier, evaluator, decode settings, objective and constraints named "
        "with it. There is no universal winner on this page."
    )
    return "\n".join(lines)


__all__ = ["format_compare_report_text"]
