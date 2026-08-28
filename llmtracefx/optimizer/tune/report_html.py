"""Self-contained static HTML rendering of a ``TuneReport``.

This is a product surface over already-computed tuning evidence, not a new
scoring system: every value rendered here comes straight from a validated
``TuneReport`` (see ``report.py``) and nothing is recomputed or guessed. The
output is a single portable HTML file with inline CSS and no JavaScript, so
it opens directly in a browser (no server, no CDN, no network access) and is
safe to attach to an issue, a chat message, or a shared drive.

Two properties are deliberate:

* **Determinism.** Rendering the same ``TuneReport`` twice produces
  byte-identical HTML (the only "clock" in the document is the report's own
  ``generated_at``, never a new timestamp taken at render time).
* **Escaping.** Every string that ultimately originates from the report JSON
  (policy name, reasons, model/candidate identity fields, artifact paths,
  run ids) is passed through ``html.escape`` before being written into the
  document, so a maliciously crafted report cannot inject markup or script
  content into the rendered page.

Local artifact paths are privacy-sensitive (they usually live under a
user's home directory) and are redacted to a stable, non-identifying label
by default -- see ``_redact_path``. Pass ``redact_paths=False`` to include
the full path as plain text instead.
"""

from __future__ import annotations

import html
import math
from collections.abc import Iterable
from pathlib import PurePosixPath

from ..doctor.speculative import DoctorVerdict
from .policy import TuneConstraints, TunePolicy
from .report import (
    BaselineComparison,
    CandidateReport,
    GroupOutcome,
    GroupReport,
    RejectedCandidateReport,
    TuneReport,
)


def _esc(value: object) -> str:
    """HTML-escape any value by way of its string representation."""
    return html.escape(str(value), quote=True)


def _redact_path(raw: str) -> str:
    """Replace a local artifact path with a stable, non-identifying label.

    Every artifact this project ever links to lives under
    ``<results_dir>/runs/<run_id>/...``; when that shape is recognizable the
    label keeps the ``runs/<run_id>/<file>`` suffix (identifying which run
    without leaking the absolute, possibly-home-directory-containing
    prefix). Otherwise falls back to the final path component.
    """
    posix_raw = raw.replace("\\", "/")
    parts = PurePosixPath(posix_raw).parts
    if "runs" in parts:
        return "/".join(parts[parts.index("runs") :])
    name = PurePosixPath(posix_raw).name
    return name or raw


def _path_label(raw: str, *, redact_paths: bool) -> str:
    return raw if not redact_paths else _redact_path(raw)


def _fmt_number(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if not math.isfinite(value):
        # Defense in depth: TuneReport.from_dict already rejects non-finite
        # values before a report ever reaches this renderer.
        return "n/a"
    return f"{value:.{digits}f}"


def _fmt_bytes_mb(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value / (1024 * 1024):.1f} MB"


def _fmt_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.1f}%"


def _list_items(values: Iterable[str]) -> str:
    items = [f"<li><code>{_esc(value)}</code></li>" for value in values]
    return (
        '<ul class="evidence-list">' + "".join(items) + "</ul>"
        if items
        else '<p class="muted">none</p>'
    )


def _style() -> str:
    return """
    :root {
      color-scheme: light;
      --ink: #1b1f24;
      --muted: #5b6472;
      --border: #d7dce2;
      --bg: #ffffff;
      --bg-alt: #f6f8fa;
      --accent-good: #1a7f37;
      --accent-good-bg: #ddf4e4;
      --accent-bad: #b91c1c;
      --accent-bad-bg: #fde2e1;
      --accent-warn: #9a6700;
      --accent-warn-bg: #fff3cd;
    }
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
      color: var(--ink);
      background: var(--bg-alt);
      margin: 0;
      padding: 2rem 1rem 4rem;
      line-height: 1.5;
    }
    main {
      max-width: 960px;
      margin: 0 auto;
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 2rem;
    }
    h1 { font-size: 1.6rem; margin-top: 0; }
    h2 { font-size: 1.25rem; border-bottom: 1px solid var(--border); padding-bottom: 0.35rem; margin-top: 2.5rem; }
    h3 { font-size: 1.05rem; margin-bottom: 0.4rem; }
    h4 { font-size: 0.95rem; margin-bottom: 0.3rem; }
    p { margin: 0.4rem 0; }
    .muted { color: var(--muted); }
    code { background: var(--bg-alt); padding: 0.05rem 0.3rem; border-radius: 4px; font-size: 0.85em; }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
      gap: 0.75rem;
      margin: 1rem 0;
    }
    .summary-card {
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 0.6rem 0.8rem;
      background: var(--bg-alt);
      text-align: center;
    }
    .summary-card .value { font-size: 1.4rem; font-weight: 600; display: block; }
    .summary-card .label { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.03em; }
    table { border-collapse: collapse; width: 100%; margin: 0.75rem 0; font-size: 0.9rem; }
    th, td { border: 1px solid var(--border); padding: 0.4rem 0.55rem; text-align: left; vertical-align: top; }
    th { background: var(--bg-alt); font-weight: 600; }
    tr.recommended-row { background: var(--accent-good-bg); }
    dl.identity { display: grid; grid-template-columns: max-content 1fr; gap: 0.15rem 1rem; margin: 0.5rem 0 1rem; }
    dl.identity dt { color: var(--muted); font-size: 0.85rem; }
    dl.identity dd { margin: 0; font-size: 0.9rem; }
    .badge { display: inline-block; padding: 0.15rem 0.55rem; border-radius: 999px; font-size: 0.78rem; font-weight: 600; }
    .badge-recommended { background: var(--accent-good-bg); color: var(--accent-good); }
    .badge-inconclusive { background: var(--accent-warn-bg); color: var(--accent-warn); }
    .panel { border: 1px solid var(--border); border-radius: 6px; padding: 0.75rem 1rem; margin: 0.75rem 0; }
    .panel-good { border-color: var(--accent-good); background: var(--accent-good-bg); }
    .panel-warn { border-color: var(--accent-warn); background: var(--accent-warn-bg); }
    .panel-bad { border-color: var(--accent-bad); background: var(--accent-bad-bg); }
    ul.violations { margin: 0.3rem 0 0.6rem 1.1rem; padding: 0; }
    ul.violations li { color: var(--accent-bad); }
    ul.evidence-list { margin: 0.2rem 0; padding-left: 1.1rem; font-size: 0.85rem; }
    details { margin: 0.4rem 0; }
    details summary { cursor: pointer; color: var(--muted); font-size: 0.85rem; }
    footer { margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid var(--border); font-size: 0.8rem; color: var(--muted); }
    @media print {
      body { background: var(--bg); padding: 0; }
      main { border: none; padding: 0; }
      details { display: block; }
      details summary { display: none; }
    }
    """.strip()


def _render_constraints(constraints: TuneConstraints) -> str:
    rows: list[tuple[str, str]] = [
        (
            "Required row statuses",
            ", ".join(sorted(status.value for status in constraints.required_statuses)),
        ),
        ("Minimum measured repetitions", str(constraints.min_measured_repetitions)),
    ]
    if constraints.min_pass_rate is not None:
        rows.append(("Minimum pass rate", _fmt_percent(constraints.min_pass_rate)))
    if constraints.min_quality_score is not None:
        rows.append(
            (
                "Minimum quality score",
                f"{_fmt_number(constraints.min_quality_score, digits=3)} "
                f"({_esc(constraints.required_quality_metric)})",
            )
        )
    if constraints.max_peak_memory_bytes is not None:
        rows.append(
            ("Maximum peak memory", _fmt_bytes_mb(constraints.max_peak_memory_bytes))
        )
    if constraints.max_total_latency_ms is not None:
        rows.append(
            (
                "Maximum total latency",
                f"{_fmt_number(constraints.max_total_latency_ms, digits=2)} ms",
            )
        )
    if constraints.max_coefficient_of_variation is not None:
        rows.append(
            (
                "Maximum coefficient of variation",
                _fmt_number(constraints.max_coefficient_of_variation, digits=3),
            )
        )
    if constraints.allowed_provenances is not None:
        rows.append(
            (
                "Allowed metric provenances",
                ", ".join(sorted(p.value for p in constraints.allowed_provenances)),
            )
        )
    items = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in rows)
    return f'<dl class="identity">{items}</dl>'


def _render_policy(policy: TunePolicy) -> str:
    name = policy.name or "(unnamed policy)"
    description = (
        f'<p class="muted">{_esc(policy.description)}</p>' if policy.description else ""
    )
    return (
        f"<h2>Policy</h2>"
        f"<p><strong>{_esc(name)}</strong></p>"
        f"{description}"
        f"<p>Objective: <code>{_esc(policy.objective.value)}</code></p>"
        f"<h3>Constraints</h3>"
        f"{_render_constraints(policy.constraints)}"
    )


def _summary_card(value: int, label: str) -> str:
    return (
        '<div class="summary-card">'
        f'<span class="value">{value}</span>'
        f'<span class="label">{_esc(label)}</span>'
        "</div>"
    )


def _render_summary(report: TuneReport) -> str:
    recommended_groups = sum(
        1 for group in report.groups if group.outcome == GroupOutcome.RECOMMENDED
    )
    inconclusive_groups = sum(
        1 for group in report.groups if group.outcome == GroupOutcome.INCONCLUSIVE
    )
    accepted_total = sum(len(group.accepted) for group in report.groups)
    rejected_total = sum(len(group.rejected) for group in report.groups)
    cards = "".join(
        [
            _summary_card(len(report.groups), "Groups"),
            _summary_card(recommended_groups, "Recommendations"),
            _summary_card(inconclusive_groups, "Inconclusive"),
            _summary_card(accepted_total, "Accepted candidates"),
            _summary_card(rejected_total, "Rejected candidates"),
            _summary_card(len(report.excluded_runs), "Excluded runs"),
        ]
    )
    return f'<h2>Summary</h2><div class="summary-grid">{cards}</div>'


def _render_group_identity(group: GroupReport) -> str:
    key = group.group_key
    rows = [
        ("Workload", f"{key.workload_id} (v{key.workload_version})"),
        ("Context tier", key.context_tier),
        (
            "Model",
            key.model_id + (f" / {key.model_family}" if key.model_family else ""),
        ),
        ("Accelerator", key.accelerator or "unknown"),
        (
            "Runtime / backend",
            f"{key.runtime_name} / {key.runtime_backend or 'unknown'}",
        ),
        ("Prompt hash", key.workload_prompt_hash),
    ]
    items = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in rows)
    return f'<dl class="identity">{items}</dl>'


def _candidate_evidence(candidate: CandidateReport, *, redact_paths: bool) -> str:
    run_ids = _list_items(candidate.run_ids)
    verification = _list_items(
        _path_label(p, redact_paths=redact_paths) for p in candidate.verification_paths
    )
    final_records = _list_items(
        _path_label(p, redact_paths=redact_paths) for p in candidate.final_record_paths
    )
    return (
        "<details><summary>Source run IDs and artifact paths</summary>"
        f"<h4>Run IDs</h4>{run_ids}"
        f"<h4>Verification artifacts</h4>{verification}"
        f"<h4>Final record artifacts</h4>{final_records}"
        "</details>"
    )


def _render_recommendation_rationale(group: GroupReport) -> str:
    winner = group.recommended
    assert winner is not None
    parts = [
        f"Won on objective <code>{_esc(winner.objective_name)}</code> with value "
        f"<code>{_fmt_number(winner.objective_value)}</code>, backed by "
        f"{winner.evidence_count} accepted run(s)."
    ]
    if len(group.accepted) > 1:
        runner_up = group.accepted[1]
        parts.append(
            "Ranked ahead of runner-up "
            f"<code>{_esc(runner_up.candidate_key.label())}</code> "
            f"(objective value <code>{_fmt_number(runner_up.objective_value)}</code>)."
        )
    else:
        parts.append("It was the only candidate accepted in this group.")
    return " ".join(parts)


def _render_accepted_table(
    accepted: tuple[CandidateReport, ...],
    *,
    recommended_rank: int | None,
    redact_paths: bool,
) -> str:
    if not accepted:
        return '<p class="muted">No accepted candidates.</p>'
    header = (
        "<tr><th>Rank</th><th>Candidate</th><th>Objective</th>"
        "<th>Mean latency (ms)</th><th>Pass rate</th>"
        "<th>Quality metric / score</th><th>Peak memory (mean / max)</th>"
        "<th>Evidence</th><th>CV</th></tr>"
    )
    rows: list[str] = []
    for candidate in accepted:
        row_class = (
            ' class="recommended-row"' if candidate.rank == recommended_rank else ""
        )
        quality = (
            f"{_esc(candidate.quality_metric)} = "
            f"{_fmt_number(candidate.mean_quality_score, digits=3)}"
            if candidate.quality_metric is not None
            else "n/a"
        )
        memory = (
            f"{_fmt_bytes_mb(candidate.mean_peak_memory_bytes)} / "
            f"{_fmt_bytes_mb(candidate.max_peak_memory_bytes)}"
        )
        rows.append(
            f"<tr{row_class}>"
            f"<td>#{candidate.rank}</td>"
            f"<td>{_esc(candidate.candidate_key.label())}</td>"
            f"<td>{_fmt_number(candidate.objective_value)}</td>"
            f"<td>{_fmt_number(candidate.mean_total_latency_ms, digits=2)}</td>"
            f"<td>{_fmt_percent(candidate.pass_rate)}</td>"
            f"<td>{quality}</td>"
            f"<td>{memory}</td>"
            f"<td>{candidate.evidence_count}</td>"
            f"<td>{_fmt_number(candidate.coefficient_of_variation, digits=4)}</td>"
            "</tr>"
        )
        rows.append(
            f'<tr{row_class}><td colspan="9">'
            + _candidate_evidence(candidate, redact_paths=redact_paths)
            + "</td></tr>"
        )
    return f"<table>{header}{''.join(rows)}</table>"


def _render_rejected(
    rejected: tuple[RejectedCandidateReport, ...], *, redact_paths: bool
) -> str:
    if not rejected:
        return ""
    blocks = []
    for candidate in rejected:
        reasons = "".join(f"<li>{_esc(reason)}</li>" for reason in candidate.reasons)
        run_ids = _list_items(candidate.run_ids)
        verification = _list_items(
            _path_label(p, redact_paths=redact_paths)
            for p in candidate.verification_paths
        )
        final_records = _list_items(
            _path_label(p, redact_paths=redact_paths)
            for p in candidate.final_record_paths
        )
        blocks.append(
            '<div class="panel panel-bad">'
            f"<strong>{_esc(candidate.candidate_key.label())}</strong>"
            f'<ul class="violations">{reasons}</ul>'
            "<details><summary>Source run IDs and artifact paths</summary>"
            f"<h4>Run IDs</h4>{run_ids}"
            f"<h4>Verification artifacts</h4>{verification}"
            f"<h4>Final record artifacts</h4>{final_records}"
            "</details>"
            "</div>"
        )
    return f"<h3>Rejected candidates ({len(rejected)})</h3>" + "".join(blocks)


def _render_baseline_comparison(comparison: BaselineComparison) -> str:
    verdict = comparison.report.verdict
    panel_class = {
        DoctorVerdict.IMPROVEMENT: "panel-good",
        DoctorVerdict.REGRESSION: "panel-bad",
        DoctorVerdict.NO_SIGNIFICANT_DIFFERENCE: "panel-warn",
        DoctorVerdict.INCONCLUSIVE: "panel-warn",
    }.get(verdict, "panel-warn")
    baseline_ms = _fmt_number(comparison.report.baseline_mean_total_ms, digits=2)
    speculative_ms = _fmt_number(comparison.report.speculative_mean_total_ms, digits=2)
    delta_ms = _fmt_number(comparison.report.delta_ms, digits=2)
    delta_pct = _fmt_number(comparison.report.delta_pct, digits=2)
    return (
        f'<h3>Speculative baseline comparison</h3><div class="panel {panel_class}">'
        f"<p><strong>Verdict:</strong> {_esc(verdict.value)} &mdash; {_esc(comparison.report.reason)}</p>"
        f"<p>Baseline <code>{_esc(comparison.baseline_candidate_key.label())}</code>: "
        f"{baseline_ms} ms mean total (run(s): {_esc(', '.join(comparison.report.baseline_run_ids))})</p>"
        f"<p>Speculative <code>{_esc(comparison.speculative_candidate_key.label())}</code>: "
        f"{speculative_ms} ms mean total (run(s): {_esc(', '.join(comparison.report.speculative_run_ids))})</p>"
        f"<p>Delta: {delta_ms} ms ({delta_pct}%)</p>"
        "</div>"
    )


def _render_group(group: GroupReport, *, redact_paths: bool) -> str:
    parts = [f"<h2>Group: {_esc(group.group_key.label())}</h2>"]
    parts.append(_render_group_identity(group))

    if group.outcome == GroupOutcome.RECOMMENDED and group.recommended is not None:
        winner = group.recommended
        parts.append(
            '<div class="panel panel-good">'
            '<span class="badge badge-recommended">RECOMMENDED</span> '
            f"<strong>{_esc(winner.candidate_key.label())}</strong>"
            f"<p>{_render_recommendation_rationale(group)}</p>"
            "</div>"
        )
    else:
        parts.append(
            '<div class="panel panel-warn">'
            '<span class="badge badge-inconclusive">INCONCLUSIVE</span> '
            f"<p>{_esc(group.inconclusive_reason or 'no reason recorded')}</p>"
            "</div>"
        )

    recommended_rank = group.recommended.rank if group.recommended is not None else None
    parts.append("<h3>Accepted candidate ranking</h3>")
    parts.append(
        _render_accepted_table(
            group.accepted,
            recommended_rank=recommended_rank,
            redact_paths=redact_paths,
        )
    )

    parts.append(_render_rejected(group.rejected, redact_paths=redact_paths))

    if group.baseline_comparison is not None:
        parts.append(_render_baseline_comparison(group.baseline_comparison))

    return "".join(parts)


def _render_excluded_runs(report: TuneReport, *, redact_paths: bool) -> str:
    if not report.excluded_runs:
        return ""
    header = "<tr><th>Run ID</th><th>Source results directory</th><th>Reason</th></tr>"
    rows = "".join(
        "<tr>"
        f"<td><code>{_esc(run.run_id)}</code></td>"
        f"<td><code>{_esc(_path_label(run.source_results_dir, redact_paths=redact_paths))}</code></td>"
        f"<td>{_esc(run.reason)}</td>"
        "</tr>"
        for run in report.excluded_runs
    )
    return (
        f"<h2>Excluded runs ({len(report.excluded_runs)})</h2>"
        '<p class="muted">Runs whose evidence could not be trusted at all '
        "(never even considered as a rejected candidate).</p>"
        f"<table>{header}{rows}</table>"
    )


def render_tune_report_html(report: TuneReport, *, redact_paths: bool = True) -> str:
    """Render ``report`` as a single, self-contained, portable HTML page.

    ``redact_paths`` (default ``True``) replaces every local artifact path
    (results directories, verification/final-record paths) with a stable,
    non-identifying label instead of the full path, so the report is safe
    to share without leaking a user's home directory layout. Pass
    ``redact_paths=False`` to include full paths as plain text.
    """
    title = report.policy.name or "Tune report"
    body_parts = [
        "<main>",
        f"<h1>{_esc(title)}</h1>",
        f'<p class="muted">Generated {_esc(report.generated_at)} '
        f"&middot; schema version {_esc(report.schema_version)}</p>",
        "<h2>Source results directories</h2>",
        _list_items(
            _path_label(d, redact_paths=redact_paths) for d in report.results_dirs
        ),
        _render_policy(report.policy),
        _render_summary(report),
        _render_excluded_runs(report, redact_paths=redact_paths),
    ]

    if not report.groups:
        body_parts.append(
            '<p class="muted">No comparable groups were found in the '
            "provided results directories.</p>"
        )
    else:
        for group in report.groups:
            body_parts.append(_render_group(group, redact_paths=redact_paths))

    body_parts.append(
        "<footer>Generated by <code>llmtracefx-optimizer tune-report</code>. "
        "This file is static HTML with no external references; open it "
        "directly in a browser.</footer>"
    )
    body_parts.append("</main>")

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{_esc(title)} &mdash; LLMTraceFX tune report</title>\n"
        f"<style>{_style()}</style>\n"
        "</head>\n"
        "<body>\n" + "".join(body_parts) + "\n</body>\n</html>\n"
    )
