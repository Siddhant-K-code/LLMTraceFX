"""Human-readable terminal rendering of a ``TuneReport``.

Two levels of detail are supported: a concise default (one line per
rejected candidate, showing why it lost) and a verbose ``--explain`` mode
(every violated constraint for every rejected candidate, plus the full
accepted-candidate ranking). Both always explain the winner: which
candidate was recommended, its measured objective value, and -- when more
than one candidate was accepted -- what distinguishes it from the runner
up.
"""

from __future__ import annotations

from .report import CandidateReport, GroupOutcome, GroupReport, TuneReport


def _format_candidate_summary(candidate: CandidateReport) -> str:
    parts = [
        f"objective({candidate.objective_name})={candidate.objective_value:.4f}",
        f"evidence={candidate.evidence_count}",
    ]
    if candidate.mean_total_latency_ms is not None:
        parts.append(f"mean_total_ms={candidate.mean_total_latency_ms:.2f}")
    if candidate.correct_cases_per_minute is not None:
        parts.append(f"correct_cases_per_min={candidate.correct_cases_per_minute:.3f}")
    if candidate.pass_rate is not None:
        parts.append(f"pass_rate={candidate.pass_rate:.2f}")
    if candidate.mean_peak_memory_bytes is not None:
        parts.append(
            f"peak_mem_mb={candidate.mean_peak_memory_bytes / (1024 * 1024):.1f}"
        )
    return ", ".join(parts)


def _format_group(group: GroupReport, *, verbose: bool) -> list[str]:
    lines = [f"Group: {group.group_key.label()}"]

    if group.outcome == GroupOutcome.RECOMMENDED and group.recommended is not None:
        winner = group.recommended
        lines.append(f"  RECOMMENDED: {winner.candidate_key.label()}")
        lines.append(f"    {_format_candidate_summary(winner)}")
        lines.append(f"    run(s): {', '.join(winner.run_ids)}")
        if len(group.accepted) > 1:
            runner_up = group.accepted[1]
            lines.append(
                "    beat runner-up "
                f"{runner_up.candidate_key.label()} "
                f"({_format_candidate_summary(runner_up)})"
            )
        if group.baseline_comparison is not None:
            comparison = group.baseline_comparison
            lines.append(
                "    vs. autoregressive baseline "
                f"({comparison.baseline_candidate_key.label()}): "
                f"{comparison.report.verdict.value} -- {comparison.report.reason}"
            )
    else:
        lines.append(f"  INCONCLUSIVE: {group.inconclusive_reason}")

    if group.accepted and (verbose or group.outcome == GroupOutcome.INCONCLUSIVE):
        lines.append("  Accepted candidates:")
        for candidate in group.accepted:
            lines.append(
                f"    #{candidate.rank} {candidate.candidate_key.label()}: "
                f"{_format_candidate_summary(candidate)}"
            )

    if group.rejected:
        lines.append(f"  Rejected candidates ({len(group.rejected)}):")
        for rejected in group.rejected:
            lines.append(
                f"    - {rejected.candidate_key.label()} "
                f"(run(s): {', '.join(rejected.run_ids)})"
            )
            reasons = rejected.reasons if verbose else rejected.reasons[:1]
            for reason in reasons:
                lines.append(f"        reason: {reason}")
            omitted = len(rejected.reasons) - len(reasons)
            if omitted > 0:
                lines.append(
                    f"        (+{omitted} more reason(s); rerun with --explain "
                    "to see all)"
                )
    return lines


def format_report_text(report: TuneReport, *, verbose: bool = False) -> str:
    """Render ``report`` as a human-readable terminal summary."""
    lines: list[str] = [
        f"Tune report (schema {report.schema_version}), generated "
        f"{report.generated_at}",
        f"Objective: {report.policy.objective.value}"
        + (f" ({report.policy.name})" if report.policy.name else ""),
        f"Results directories: {', '.join(report.results_dirs)}",
    ]

    if report.excluded_runs:
        lines.append(
            f"{len(report.excluded_runs)} run(s) excluded (unusable evidence):"
        )
        for run in report.excluded_runs:
            lines.append(f"  - {run.run_id} [{run.source_results_dir}]: {run.reason}")

    if not report.groups:
        lines.append("")
        lines.append(
            "No comparable groups were found in the provided results " "directories."
        )
        return "\n".join(lines)

    for group in report.groups:
        lines.append("")
        lines.extend(_format_group(group, verbose=verbose))

    return "\n".join(lines)
