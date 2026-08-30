"""Self-contained static HTML rendering of a ``CompareReport``.

A product surface over already-computed comparison evidence, not a second
scoring system: every value rendered here comes straight from a validated
``CompareReport`` (see ``report.py``) and nothing is recomputed, re-ranked or
estimated at render time.

The design system is *reused*, not reimplemented. The stylesheet, the
escaping helper, the path redaction rule and the routing ornaments are
imported from the tune report renderer, so a colour, a rule weight or a
redaction rule can never mean one thing in a tune report and another in a
comparison report. See ``DESIGN.md`` and ``llmtracefx.brand``.

Three properties are deliberate and are covered by tests:

* **Determinism.** Rendering the same report twice produces byte-identical
  HTML. The only clock in the document is the report's own ``generated_at``.
* **Escaping.** Every string that originates from the report JSON is passed
  through ``html.escape`` before it reaches the document.
* **No network.** Inline CSS, inline SVG, no JavaScript, no CDN, no external
  reference of any kind, so the file opens from disk and is safe to attach
  to an issue or a chat message.

Local artifact paths are redacted to stable, non-identifying labels by
default; pass ``redact_paths=False`` to include them as plain text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from itertools import zip_longest
from pathlib import PurePosixPath
from urllib.parse import urlsplit

from ...brand import LOCKUP_SVG
from ..tune.report_html import (
    DIRECTION_CONTRACT,
    ORNAMENT_CLOSED_SVG,
    ORNAMENT_OPEN_SVG,
    _esc,
    _fmt_bytes_mb,
    _fmt_number,
    _fmt_percent,
    _list_items,
    _path_label,
    _style,
)
from .policy import CompareConstraints
from .report import (
    CompareReport,
    CostSummary,
    FrontierEntry,
    RejectedSystemReport,
    StratumOutcome,
    StratumReport,
    SystemReport,
    TtftBasis,
    UsageTotals,
)

#: How a time-to-first-token figure is described in prose, so a reader never
#: has to know which collector produced it to know what it measures.
_TTFT_BASIS_LABEL: dict[TtftBasis, str] = {
    TtftBasis.LOCAL_PREFILL: "local prefill (host-observed prompt processing)",
    TtftBasis.CLIENT_OBSERVED_STREAM: (
        "client-observed stream offset (includes network transport and queueing)"
    ),
}

#: Measurement columns for the ranking table, in reading order. The system
#: identifier is deliberately not a column: it is the one value that must
#: never be clipped, so it heads its own row group instead.
_RANKED_COLUMNS: tuple[str, ...] = (
    "Objective",
    "Pass rate",
    "Quality",
    "Mean latency (ms)",
    "p50 / p95 (ms)",
    "Correct/min",
    "Cost per correct case",
    "Correct per unit",
    "Evidence",
)


#: The opening paragraph, kept as one named constant rather than seven
#: adjacent literals inside the ``body_parts`` list. Implicit concatenation
#: inside a list reads as a missing comma to a reader and to CodeQL alike,
#: which is the same reason the API collector spells its own joins out.
_LEDE = (
    '<p class="lede">Systems are only placed side by side when they were '
    "asked the identical question: same workload and version, same prompt "
    "hash, same context tier, same evaluator, same output cap and sampling. "
    "Anything else is reported as a separate unit. Nothing here is "
    "re-scored, no measurement is blended into an overall grade, and every "
    "value that was not recorded stays unavailable rather than becoming "
    "zero.</p>"
)


def _fmt_money(value: float | None, currency: str) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6g} {currency}"


def _fmt_tokens(value: int | None) -> str:
    return "n/a" if value is None else f"{value:,}"


def _missing_evidence_block(
    values: Iterable[str], *, empty: str, redact_paths: bool = True
) -> str:
    items = [
        f"<li>{_esc(_redact_paths_in_prose(value, redact_paths=redact_paths))}</li>"
        for value in values
    ]
    if not items:
        return f'<p class="muted">{_esc(empty)}</p>'
    return f'<ul class="violations">{"".join(items)}</ul>'


#: Matches the ``endpoint=<url>`` segment a hosted system carries in its
#: label. The host is deployment infrastructure and can name a private or
#: internal service, so it is redacted out of the rendered document by
#: default on exactly the same terms as a local artifact path.
_ENDPOINT_IN_LABEL = re.compile(r"(endpoint=)(\S+)")


def _system_label(label: str, *, redact_paths: bool) -> str:
    """Redact the deployment host from a system label.

    The path is kept: it is what distinguishes one deployment route from
    another (``/v1/chat/completions`` against a per-deployment path, say)
    without naming the host it lives on. Applied to every rendered label,
    including the frontier's ``dominated_by`` entries, so a host cannot
    survive in one corner of the page after being redacted elsewhere.
    """
    if not redact_paths:
        return label

    def replace(match: re.Match[str]) -> str:
        parts = urlsplit(match.group(2))
        if not parts.scheme or not parts.netloc:
            return match.group(0)
        return match.group(1) + (parts.path or "/")

    return _ENDPOINT_IN_LABEL.sub(replace, label)


#: Candidate tokens for path scrubbing: any run of non-space, non-quote
#: characters. Whether a candidate is actually a path is decided by
#: ``_looks_like_path`` rather than by the pattern, because a regex tight
#: enough to exclude ordinary prose like "read/parse" and loose enough to
#: catch "acme-client/eval/runs/x/final_record.json" is unreadable.
_PROSE_TOKEN = re.compile(r"[^\s'\"()<>]+")

#: A final component is treated as a filename when it carries a short
#: alphanumeric extension, which is what distinguishes ``a/b.json`` from the
#: prose slash in ``read/parse``.
_FILENAME_SUFFIX = re.compile(r"\.[A-Za-z0-9]{1,8}$")


def _looks_like_path(token: str) -> bool:
    """Whether a prose token is a filesystem path rather than ordinary words.

    Deliberately conservative in both directions. Absolute paths and Windows
    drive paths always count. A relative token counts when it has more than
    one separator, or exactly one separator and a filename-looking final
    component. That admits ``artifacts/run/runs/x/record.json`` and
    ``run/record.json`` while leaving ``read/parse`` alone, which a naive
    "contains a slash" rule mangles into ``readparse``.
    """
    if token.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", token):
        return True
    parts = re.split(r"[/\\]", token)
    if len(parts) > 2:
        return True
    return len(parts) == 2 and bool(_FILENAME_SUFFIX.search(parts[-1]))


def _redact_paths_in_prose(text: str, *, redact_paths: bool) -> str:
    """Reduce any filesystem path quoted inside prose to its final component.

    Exclusion and rejection reasons are assembled where the failure happened,
    including inside shared loaders this package does not own, so despite
    care at each site one can still quote a directory. The rendered document
    is meant to be shareable, so rather than trusting every message to be
    written correctly forever, the renderer scrubs them on the way out.

    Relative paths matter as much as absolute ones here: the documented
    workflow uses a relative ``--output-dir``, so a leaked ancestor is
    typically a relative one such as a client or project directory name.
    """
    if not redact_paths:
        return text

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if not _looks_like_path(token):
            return token
        name = PurePosixPath(token.replace("\\", "/")).name
        if not name:
            return "<path>"
        # A path containing whitespace does not survive tokenisation:
        # ``/Users/secret client/results/runs/x/record.json`` splits at the
        # space, and the first half's basename is the private ancestor
        # ``secret`` rather than a filename, so reducing it to its basename
        # would publish exactly the directory name redaction exists to hide.
        # The same is true of a bare directory path. So an anchored path is
        # kept only when its final component looks like a file; otherwise it
        # becomes opaque. Nothing is lost by that: a directory that is
        # genuinely worth naming is carried as a structured field and
        # labelled by ``_redacted_label``, not recovered from prose.
        anchored = (
            token.startswith("/")
            or token.startswith("\\\\")
            or bool(re.match(r"^[A-Za-z]:[\\/]", token))
        )
        if anchored and not _FILENAME_SUFFIX.search(name):
            return "<path>"
        return name

    return _PROSE_TOKEN.sub(replace, text)


def _redacted_label(raw: str, *, redact_paths: bool, run_id: str | None = None) -> str:
    """Redact a path that has no verified run id to its basename alone.

    The shared ``_redact_path`` falls back to the last ``runs`` segment when
    it is given no run id. For a per-run artifact that is what you want. For
    a results directory or a pricing manifest it is not: those paths have no
    run id to anchor on, and if such a path happens to contain a ``runs``
    segment anywhere -- ``/home/someone/secret-client/runs/private-eval`` --
    the fallback keeps everything from that segment onward and publishes
    directory names that have nothing to do with this report.

    So when no run id is available, the label is the final path component and
    nothing else. That is enough to tell two inputs apart in a report while
    carrying no ancestor names at all.
    """
    if not redact_paths:
        return raw
    if run_id:
        return _path_label(raw, redact_paths=True, run_id=run_id)
    return PurePosixPath(raw.replace("\\", "/")).name or raw


def _render_masthead(report: CompareReport, *, redact_paths: bool) -> str:
    redaction = "paths redacted" if redact_paths else "full paths included"
    return (
        '<header class="masthead">'
        f"{LOCKUP_SVG}"
        '<p class="stamp">Cross-system comparison'
        f"<br><b>{_esc(report.generated_at)}</b>"
        f"<br>Schema {_esc(report.schema_version)} &middot; {redaction}</p>"
        "</header>"
    )


def _counts(report: CompareReport) -> tuple[tuple[str, int, str], ...]:
    recommended = sum(
        1 for stratum in report.strata if stratum.outcome == StratumOutcome.RECOMMENDED
    )
    inconclusive = sum(
        1 for stratum in report.strata if stratum.outcome == StratumOutcome.INCONCLUSIVE
    )
    ranked = sum(len(stratum.ranked) for stratum in report.strata)
    rejected = sum(len(stratum.rejected) for stratum in report.strata)
    excluded = len(report.excluded_runs)
    return (
        ("Comparable units", len(report.strata), ""),
        ("Recommendations", recommended, "v-rec" if recommended else ""),
        ("Inconclusive", inconclusive, "v-hold" if inconclusive else ""),
        ("Ranked systems", ranked, ""),
        ("Rejected systems", rejected, ""),
        ("Excluded runs", excluded, "v-breach" if excluded else ""),
    )


def _render_summary(report: CompareReport) -> str:
    divisions = "".join(
        '<div class="division">'
        f'<span class="v {state}">{value}</span>'
        f'<span class="k">{_esc(label)}</span>'
        "</div>"
        for label, value, state in _counts(report)
    )
    return (
        '<section id="summary" aria-labelledby="summary-h">'
        '<div class="rule"><h2 id="summary-h">Summary</h2></div>'
        f'<div class="readout">{divisions}</div>'
        "</section>"
    )


def _render_unit_identity(stratum: StratumReport) -> str:
    key = stratum.unit_key
    rows = (
        ("Workload", f"{key.workload_id} (v{key.workload_version})"),
        ("Context tier", key.context_tier),
        ("Prompt hash", key.workload_prompt_hash),
        ("Evaluator / quality metric", key.quality_metric or "unrecorded"),
        (
            "Max output tokens",
            (
                "unrecorded"
                if key.max_output_tokens is None
                else str(key.max_output_tokens)
            ),
        ),
        (
            "Sampling",
            "temperature="
            + ("unrecorded" if key.temperature is None else f"{key.temperature:g}")
            + ", top_p="
            + ("unrecorded" if key.top_p is None else f"{key.top_p:g}"),
        ),
        ("Ranking objective", stratum.objective_name),
    )
    items = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in rows)
    return f'<dl class="spec">{items}</dl>'


def _artifact_lists(
    *,
    run_ids: tuple[str, ...],
    verification_paths: tuple[str, ...],
    record_paths: tuple[str, ...],
    redact_paths: bool,
) -> str:
    verification = _list_items(
        _path_label(path, redact_paths=redact_paths, run_id=run_id)
        for path, run_id in zip_longest(verification_paths, run_ids, fillvalue=None)
        if path is not None
    )
    records = _list_items(
        _path_label(path, redact_paths=redact_paths, run_id=run_id)
        for path, run_id in zip_longest(record_paths, run_ids, fillvalue=None)
        if path is not None
    )
    return (
        f"<h4>Run IDs</h4>{_list_items(run_ids)}"
        f"<h4>Verification artifacts</h4>{verification}"
        f"<h4>Final record artifacts</h4>{records}"
    )


def _render_usage(usage: UsageTotals | None) -> str:
    if usage is None:
        return (
            '<p class="muted">No provider-reported token usage. This system was '
            "not executed by a hosted API, or the provider returned no usage "
            "block.</p>"
        )
    rows = (
        ("Input tokens", _fmt_tokens(usage.input_tokens)),
        ("Output tokens", _fmt_tokens(usage.output_tokens)),
        ("Cached input tokens", _fmt_tokens(usage.cached_input_tokens)),
        ("Reasoning tokens", _fmt_tokens(usage.reasoning_tokens)),
        (
            "Runs reporting usage",
            f"{usage.runs_reporting_usage} of {usage.runs_total}",
        ),
    )
    items = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in rows)
    caveat = (
        ""
        if usage.complete
        else (
            '<p class="muted">Incomplete: at least one ranked run carried no '
            "usage block, so any total above that depends on it is withheld "
            "rather than partially summed.</p>"
        )
    )
    return (
        '<p class="muted">Reported by the provider, not measured by this client.'
        "</p>"
        f'<dl class="spec">{items}</dl>{caveat}'
    )


def _render_cost(cost: CostSummary | None) -> str:
    if cost is None:
        return (
            '<p class="muted">No cost estimate. Either no pricing entry matched '
            "this system, or no pricing manifest was supplied.</p>"
        )
    rows = (
        ("Total", _fmt_money(cost.total_amount, cost.currency)),
        ("Per case", _fmt_money(cost.cost_per_case, cost.currency)),
        ("Per correct case", _fmt_money(cost.cost_per_correct_case, cost.currency)),
        (
            f"Correct cases per {cost.currency}",
            _fmt_number(cost.correct_cases_per_currency_unit, digits=3),
        ),
        ("Pricing entry", cost.pricing_entry_id),
        ("Pricing entry hash", cost.pricing_entry_sha256),
    )
    items = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in rows)
    marker = (
        '<span class="state state-hold">Illustrative rates</span>'
        if cost.rates_are_illustrative
        else '<span class="state state-hold">Estimated</span>'
    )
    caveat = (
        '<p class="muted">Every figure above is derived from provider-reported '
        "token usage priced with supplied rates. None of it was measured, and "
        "none of it is a quotation."
        + (
            " The manifest declares its rates to be illustrative examples, so "
            "these numbers demonstrate the arithmetic and are not prices."
            if cost.rates_are_illustrative
            else ""
        )
        + "</p>"
    )
    reasons = (
        f"<h4>Cost caveats</h4>{_missing_evidence_block(cost.reasons, empty='none')}"
        if cost.reasons
        else ""
    )
    return f'<p>{marker}</p>{caveat}<dl class="spec">{items}</dl>{reasons}'


def _render_system_detail(system: SystemReport, *, redact_paths: bool) -> str:
    ttft = (
        "n/a"
        if system.mean_ttft_ms is None or system.ttft_basis is None
        else (
            f"{_fmt_number(system.mean_ttft_ms, digits=2)} ms "
            f"({_TTFT_BASIS_LABEL[system.ttft_basis]})"
        )
    )
    memory = (
        "local-only measurement, not available for a hosted system"
        if not system.system_key.is_local
        else (
            f"{_fmt_bytes_mb(system.mean_peak_memory_bytes)} mean / "
            f"{_fmt_bytes_mb(system.max_peak_memory_bytes)} max"
        )
    )
    rows = (
        ("Time to first token", ttft),
        (
            "Latency spread (stdev)",
            _fmt_number(system.stdev_total_latency_ms, digits=2),
        ),
        ("Coefficient of variation", _fmt_number(system.coefficient_of_variation)),
        ("Local peak memory", memory),
    )
    items = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in rows)
    artifacts = _artifact_lists(
        run_ids=system.run_ids,
        verification_paths=system.verification_paths,
        record_paths=system.record_paths,
        redact_paths=redact_paths,
    )
    return (
        "<details><summary>Measurements, usage, cost and source artifacts"
        "</summary>"
        f'<dl class="spec">{items}</dl>'
        "<h4>Provider-reported usage</h4>"
        f"{_render_usage(system.usage)}"
        "<h4>Estimated cost</h4>"
        f"{_render_cost(system.cost)}"
        "<h4>Missing evidence</h4>"
        f"{_missing_evidence_block(system.missing_evidence, empty='Nothing missing.')}"
        f"{artifacts}"
        "</details>"
    )


def _render_ranked_table(stratum: StratumReport, *, redact_paths: bool) -> str:
    if not stratum.ranked:
        return (
            '<p class="empty">No system cleared every constraint, so there is '
            "no ranking to show.</p>"
        )
    header = "".join(
        f'<th scope="col" role="columnheader" class="num">{_esc(label)}</th>'
        for label in _RANKED_COLUMNS
    )
    recommended_rank = (
        stratum.recommended.rank if stratum.recommended is not None else None
    )
    groups: list[str] = []
    for system in stratum.ranked:
        is_recommended = system.rank == recommended_rank
        pad = "pad" if is_recommended else "pad-open"
        marker = (
            '<span class="state state-rec">Recommended</span>' if is_recommended else ""
        )
        quality = (
            "n/a"
            if system.quality_metric is None
            else (
                f"{system.quality_metric} = "
                f"{_fmt_number(system.mean_quality_score, digits=3)}"
            )
        )
        currency = system.cost.currency if system.cost is not None else ""
        values = (
            _fmt_number(system.objective_value),
            _fmt_percent(system.pass_rate),
            quality,
            _fmt_number(system.mean_total_latency_ms, digits=2),
            f"{_fmt_number(system.p50_total_latency_ms, digits=2)} / "
            + f"{_fmt_number(system.p95_total_latency_ms, digits=2)}",
            _fmt_number(system.correct_cases_per_minute, digits=3),
            (
                "n/a"
                if system.cost is None
                else _fmt_money(system.cost.cost_per_correct_case, currency)
            ),
            (
                "n/a"
                if system.cost is None
                else _fmt_number(system.cost.correct_cases_per_currency_unit, digits=3)
            ),
            str(system.evidence_count),
        )
        # Every cell value is escaped here, at the single boundary where a
        # value enters the document, rather than each formatter escaping its
        # own inputs. A formatter that forgot (``_fmt_money`` interpolates the
        # currency string straight from the report) would otherwise be the one
        # unescaped path on the page.
        cells = "".join(
            f'<td role="cell" class="num" data-label="{_esc(label)}">'
            f"{_esc(value)}</td>"
            for label, value in zip(_RANKED_COLUMNS, values, strict=True)
        )
        span = len(_RANKED_COLUMNS)
        groups.append(
            f'<tbody role="rowgroup" class="cand{" is-rec" if is_recommended else ""}">'
            '<tr role="row" class="cand-head">'
            f'<th scope="rowgroup" role="rowheader" colspan="{span}">'
            '<span class="cand-id"><span class="cand-rank">'
            f'<span class="{pad}" aria-hidden="true"></span>#{system.rank}</span>'
            f'<span class="cand-label">'
            f"{_esc(_system_label(system.system_key.label(), redact_paths=redact_paths))}"
            "</span>"
            f"{marker}</span></th></tr>"
            f'<tr role="row" class="cand-metrics">{cells}</tr>'
            '<tr role="row" class="cand-evidence">'
            f'<td role="cell" colspan="{span}">'
            + _render_system_detail(system, redact_paths=redact_paths)
            + "</td></tr></tbody>"
        )
    caption = (
        f"{len(stratum.ranked)} system(s) cleared every constraint, ranked by "
        f"{stratum.objective_name}. Quality columns are shown beside the "
        "objective so a faster system is never read as a better one."
    )
    return (
        '<div class="scroller"><table class="reflow" role="table">'
        f"<caption>{_esc(caption)}</caption>"
        f'<thead role="rowgroup"><tr role="row">{header}</tr></thead>'
        f"{''.join(groups)}"
        "</table></div>"
    )


def _render_frontier(stratum: StratumReport, *, redact_paths: bool) -> str:
    if not stratum.frontier or not stratum.frontier_axes:
        return (
            "<h3>Evidence frontier</h3>"
            '<p class="empty">No frontier could be computed: the ranked systems '
            "share no axis on which all of them carry evidence.</p>"
        )
    axes = ", ".join(axis.value for axis in stratum.frontier_axes)
    rows = "".join(
        _render_frontier_row(entry, redact_paths=redact_paths)
        for entry in stratum.frontier
    )
    return (
        "<h3>Evidence frontier</h3>"
        '<p class="muted">Systems that nothing else beats on every axis at once. '
        f"Axes: <code>{_esc(axes)}</code>. Dominance is computed on point "
        "estimates, so a system that leads only inside its noise band still "
        "shows here as dominating; the ranking above applies the noise test and "
        "this table does not. There is no universal winner on this page.</p>"
        '<div class="scroller"><table class="reflow" role="table">'
        "<caption>Each system is either on the frontier or named with every "
        "system that dominates it.</caption>"
        '<thead role="rowgroup"><tr role="row">'
        '<th scope="col" role="columnheader">System</th>'
        '<th scope="col" role="columnheader">Position</th>'
        '<th scope="col" role="columnheader">Dominated by</th></tr></thead>'
        f'<tbody role="rowgroup">{rows}</tbody></table></div>'
    )


def _render_frontier_row(entry: FrontierEntry, *, redact_paths: bool) -> str:
    state = (
        '<span class="state state-bad">Dominated</span>'
        if entry.dominated
        else '<span class="state state-ok">On frontier</span>'
    )
    dominated_by = (
        _list_items(
            _system_label(name, redact_paths=redact_paths)
            for name in entry.dominated_by
        )
        if entry.dominated_by
        else '<span class="nil">none</span>'
    )
    return (
        '<tr role="row">'
        f'<th scope="row" role="rowheader" data-label="System">'
        f"<code>{_esc(_system_label(entry.system_key.label(), redact_paths=redact_paths))}</code></th>"
        f'<td role="cell" data-label="Position">{state}</td>'
        f'<td role="cell" data-label="Dominated by">{dominated_by}</td>'
        "</tr>"
    )


def _render_rejected(
    rejected: tuple[RejectedSystemReport, ...], *, redact_paths: bool
) -> str:
    if not rejected:
        return (
            "<h3>Rejected systems</h3>"
            '<p class="empty">None. Every system in this unit cleared the '
            "policy constraints.</p>"
        )
    blocks = []
    for system in rejected:
        reasons = "".join(
            f"<li>{_esc(_redact_paths_in_prose(reason, redact_paths=redact_paths))}</li>"
            for reason in system.reasons
        )
        artifacts = _artifact_lists(
            run_ids=system.run_ids,
            verification_paths=system.verification_paths,
            record_paths=system.record_paths,
            redact_paths=redact_paths,
        )
        blocks.append(
            '<div class="reject">'
            '<span class="state state-bad">Rejected</span>'
            f'<div class="subject">'
            f"{_esc(_system_label(system.system_key.label(), redact_paths=redact_paths))}"
            "</div>"
            f'<ul class="violations">{reasons}</ul>'
            "<details><summary>Source run IDs and artifact paths</summary>"
            f"{artifacts}</details>"
            "</div>"
        )
    return (
        f"<h3>Rejected systems ({len(rejected)})</h3>"
        '<p class="muted">Each system below is listed with every constraint it '
        "breached.</p>" + "".join(blocks)
    )


def _stratum_anchor(index: int) -> str:
    return f"unit-{index}"


def _render_constraints(constraints: CompareConstraints) -> str:
    """The bar a system had to clear, stated with the verdict.

    Without it "cleared every constraint" is unreadable: there is no way to
    tell a demanding comparison from one that constrained nothing.
    """
    items = "".join(f"<li>{_esc(item)}</li>" for item in constraints.active_summary())
    return (
        '<div class="constraints">'
        '<p class="muted">Constraints in force for this verdict:</p>'
        f'<ul class="constraint-list">{items}</ul>'
        "</div>"
    )


def _render_recommendation(
    stratum: StratumReport, *, constraints: CompareConstraints, redact_paths: bool
) -> str:
    """The recommendation, always stated with the question it answers."""
    key = stratum.unit_key
    scope = (
        f"workload <code>{_esc(key.workload_id)}</code> v{_esc(key.workload_version)} "
        f"at context tier <code>{_esc(key.context_tier)}</code>, evaluated by "
        f"<code>{_esc(key.quality_metric or 'unrecorded')}</code>, ranked on "
        f"<code>{_esc(stratum.objective_name)}</code>"
    )
    if stratum.outcome != StratumOutcome.RECOMMENDED or stratum.recommended is None:
        return (
            '<div class="verdict">'
            f"<p>No system could be recommended for {scope}.</p>"
            f"<p>{_esc(stratum.inconclusive_reason or 'no reason recorded')}</p>"
            f"{_render_constraints(constraints)}"
            "</div>"
        )
    winner = stratum.recommended
    runner_up = (
        f"Ranked ahead of <code>"
        f"{_esc(_system_label(stratum.ranked[1].system_key.label(), redact_paths=redact_paths))}"
        "</code> "
        f"(objective value <code>{_fmt_number(stratum.ranked[1].objective_value)}"
        "</code>)."
        if len(stratum.ranked) > 1
        else "It was the only system accepted for this unit."
    )
    return (
        '<div class="verdict is-rec">'
        f"<p>For {scope}, and only for that:</p>"
        f'<span class="subject">'
        f"{_esc(_system_label(winner.system_key.label(), redact_paths=redact_paths))}"
        "</span>"
        f"<p>Won with objective value "
        f"<code>{_fmt_number(winner.objective_value)}</code>, backed by "
        f"{winner.evidence_count} run(s) with usable timing evidence. "
        f"{runner_up}</p>"
        f"{_render_constraints(constraints)}"
        "</div>"
    )


def _render_stratum(
    stratum: StratumReport,
    *,
    index: int,
    constraints: CompareConstraints,
    redact_paths: bool,
) -> str:
    anchor = _stratum_anchor(index)
    state = (
        '<span class="state state-rec">RECOMMENDED</span>'
        if stratum.outcome == StratumOutcome.RECOMMENDED
        else '<span class="state state-hold">INCONCLUSIVE</span>'
    )
    return "".join(
        (
            f'<section id="{anchor}" aria-labelledby="{anchor}-h">',
            f'<div class="rule"><h2 id="{anchor}-h">'
            + '<span class="qualifier">Comparable unit</span>'
            + f'<span class="subject-id">{_esc(stratum.unit_key.label())}</span>'
            + f"</h2>{state}</div>",
            _render_unit_identity(stratum),
            _render_recommendation(
                stratum, constraints=constraints, redact_paths=redact_paths
            ),
            "<h3>Ranked systems</h3>",
            _render_ranked_table(stratum, redact_paths=redact_paths),
            _render_frontier(stratum, redact_paths=redact_paths),
            _render_rejected(stratum.rejected, redact_paths=redact_paths),
            "<h3>Missing evidence for this unit</h3>",
            _missing_evidence_block(
                stratum.missing_evidence,
                empty="Nothing missing: every ranked system carried evidence for "
                "every axis this unit was compared on.",
            ),
            "</section>",
        )
    )


def _render_pricing(report: CompareReport, *, redact_paths: bool) -> str:
    pricing = report.pricing
    if pricing is None:
        return (
            '<section id="pricing" aria-labelledby="pricing-h">'
            '<div class="rule"><h2 id="pricing-h">Pricing input</h2></div>'
            '<p class="empty">No pricing manifest was supplied, so this report '
            "contains no monetary values at all.</p>"
            f"{ORNAMENT_OPEN_SVG}"
            "</section>"
        )
    rows = (
        (
            "Manifest",
            _redacted_label(pricing.manifest_path, redact_paths=redact_paths),
        ),
        ("Manifest hash", pricing.manifest_sha256),
        ("Currency", pricing.currency),
        (
            "Entries used",
            ", ".join(pricing.entry_ids_used) if pricing.entry_ids_used else "none",
        ),
    )
    items = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in rows)
    warning = (
        '<p><span class="state state-hold">Illustrative rates</span></p>'
        "<p>This manifest declares its rates to be examples rather than prices. "
        "Every monetary figure in this report therefore demonstrates the "
        "arithmetic on real usage; none of it is a quotation, and none of it "
        "was fetched from a provider.</p>"
        if pricing.rates_are_illustrative
        else "<p>Rates were supplied by the operator with a source and an "
        "effective date. They are still applied to provider-reported usage "
        "rather than measured, so every figure derived from them is an "
        "estimate.</p>"
    )
    return (
        '<section id="pricing" aria-labelledby="pricing-h">'
        '<div class="rule"><h2 id="pricing-h">Pricing input</h2></div>'
        f"{warning}"
        f'<dl class="spec">{items}</dl>'
        "</section>"
    )


def _render_provenance(report: CompareReport, *, redact_paths: bool) -> str:
    sources = _list_items(
        _redacted_label(path, redact_paths=redact_paths) for path in report.results_dirs
    )
    parts = [
        '<section id="provenance" aria-labelledby="provenance-h">',
        '<div class="rule"><h2 id="provenance-h">Provenance</h2>'
        + '<span class="muted">'
        + f'{"paths redacted" if redact_paths else "full paths included"}</span></div>',
        "<h3>Source results directories</h3>",
        sources,
    ]
    if report.tune_report_paths:
        parts.append("<h3>Corroborating tune reports</h3>")
        parts.append(
            _list_items(
                _redacted_label(path, redact_paths=redact_paths)
                for path in report.tune_report_paths
            )
        )
    if report.excluded_runs:
        rows = "".join(
            '<tr role="row">'
            '<th scope="row" role="rowheader" data-label="Run ID">'
            f"<code>{_esc(run.run_id)}</code></th>"
            '<td role="cell" data-label="Source results directory"><code>'
            f"{_esc(_redacted_label(run.source_results_dir, redact_paths=redact_paths))}"
            "</code></td>"
            f'<td role="cell" data-label="Reason">'
            f"{_esc(_redact_paths_in_prose(run.reason, redact_paths=redact_paths))}"
            "</td>"
            "</tr>"
            for run in report.excluded_runs
        )
        parts.append(f"<h3>Excluded runs ({len(report.excluded_runs)})</h3>")
        parts.append(
            '<p class="muted">Runs whose evidence could not be trusted at all '
            "(never even considered as a rejected system).</p>"
        )
        parts.append(
            '<div class="scroller"><table class="reflow hazard" role="table">'
            "<caption>Every run listed here was set aside before any system was "
            "compared, so none of it reached the tables above.</caption>"
            '<thead role="rowgroup"><tr role="row">'
            '<th scope="col" role="columnheader">Run ID</th>'
            '<th scope="col" role="columnheader">Source results directory</th>'
            '<th scope="col" role="columnheader">Reason</th></tr></thead>'
            f'<tbody role="rowgroup">{rows}</tbody></table></div>'
        )
    else:
        parts.append("<h3>Excluded runs</h3>")
        parts.append(
            '<p class="empty">None. Every run found in the source directories '
            "carried enough evidence to be compared.</p>"
        )
    parts.append("</section>")
    return "".join(parts)


def _render_transect(report: CompareReport) -> str:
    stations = [
        '<li><a href="#summary"><span class="station" aria-hidden="true"></span>'
        + '<span class="label">Summary</span></a></li>'
    ]
    for index, stratum in enumerate(report.strata, start=1):
        recommended = stratum.outcome == StratumOutcome.RECOMMENDED
        station_class = "station station-rec" if recommended else "station station-hold"
        note_class = "note" if recommended else "note note-hold"
        note = "RECOMMENDED" if recommended else "INCONCLUSIVE"
        stations.append(
            f'<li><a href="#{_stratum_anchor(index)}">'
            f'<span class="{station_class}" aria-hidden="true"></span>'
            f'<span class="label">{_esc(stratum.unit_key.label())}</span>'
            f'<span class="{note_class}">{note}</span></a></li>'
        )
    for anchor, label in (
        ("#pricing", "Pricing input"),
        ("#provenance", "Provenance"),
    ):
        stations.append(
            f'<li><a href="{anchor}">'
            '<span class="station" aria-hidden="true"></span>'
            f'<span class="label">{label}</span></a></li>'
        )
    return (
        '<nav class="transect" aria-label="Report contents">'
        f"<ol>{''.join(stations)}</ol></nav>"
    )


def _meta_description(report: CompareReport) -> str:
    counts = {label: value for label, value, _ in _counts(report)}
    return (
        f"Cross-system comparison for policy {report.policy.name or 'unnamed'}: "
        f"{counts['Comparable units']} comparable unit(s), "
        f"{counts['Recommendations']} recommendation(s), "
        f"{counts['Inconclusive']} inconclusive, "
        f"{counts['Rejected systems']} rejected system(s), "
        f"{counts['Excluded runs']} excluded run(s). "
        f"Generated {report.generated_at}."
    )


def render_compare_report_html(
    report: CompareReport, *, redact_paths: bool = True
) -> str:
    """Render ``report`` as a single, self-contained, portable HTML page.

    ``redact_paths`` (default ``True``) replaces every local artifact path
    with a stable, non-identifying label, so the report is safe to share
    without leaking a user's directory layout. Pass ``redact_paths=False``
    to include full paths as plain text.
    """
    title = report.policy.name or "Cross-system comparison"
    body_parts = [
        '<a class="skip" href="#report">Skip to the report</a>',
        '<div class="sheet">',
        _render_masthead(report, redact_paths=redact_paths),
        '<main id="report">',
        f"<h1>{_esc(title)}</h1>",
        _LEDE,
        _render_transect(report),
        _render_summary(report),
    ]

    if not report.strata:
        body_parts.append(
            '<section id="units" aria-labelledby="units-h">'
            '<div class="rule"><h2 id="units-h">Comparable units</h2></div>'
            '<p class="empty">No comparable units were found in the provided '
            "results directories. Nothing could be compared, so nothing is "
            "recommended.</p>"
            '<div class="empty-spec">'
            + "".join(
                f'<div><span>{label}</span><span class="nil">none</span></div>'
                for label in (
                    "Comparable units",
                    "Recommendations",
                    "Ranked systems",
                    "Rejected systems",
                    "Frontier entries",
                )
            )
            + "</div>"
            f"{ORNAMENT_OPEN_SVG}"
            "</section>"
        )
    else:
        for index, stratum in enumerate(report.strata, start=1):
            body_parts.append(
                _render_stratum(
                    stratum,
                    index=index,
                    constraints=report.policy.constraints,
                    redact_paths=redact_paths,
                )
            )

    body_parts.append(_render_pricing(report, redact_paths=redact_paths))
    body_parts.append(_render_provenance(report, redact_paths=redact_paths))
    body_parts.append("</main>")
    body_parts.append(
        f"{ORNAMENT_CLOSED_SVG}"
        "<footer><span>Generated by <code>llmtracefx-optimizer compare-report"
        "</code></span><span>Static HTML, no external references, safe to open "
        "from disk</span></footer>"
    )
    body_parts.append("</div>")

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="color-scheme" content="light">\n'
        '<meta name="generator" content="llmtracefx-optimizer compare-report">\n'
        '<meta name="robots" content="noindex, nofollow">\n'
        f'<meta name="description" content="{_esc(_meta_description(report))}">\n'
        f"<title>{_esc(title)} &middot; LLMTraceFX cross-system comparison</title>\n"
        f"<!--{DIRECTION_CONTRACT}-->\n"
        f"<style>{_style()}</style>\n"
        "</head>\n"
        "<body>\n" + "".join(body_parts) + "\n</body>\n</html>\n"
    )


__all__ = ["render_compare_report_html"]
