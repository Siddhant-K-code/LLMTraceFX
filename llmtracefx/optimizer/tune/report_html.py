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
from itertools import zip_longest
from pathlib import PurePosixPath

from ...brand import LOCKUP_SVG, TOKENS_CSS
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


def _redact_path(raw: str, *, run_id: str | None = None) -> str:
    """Replace a local artifact path with a stable, non-identifying label.

    When ``run_id`` is known (the run this path belongs to), the label is
    built by locating that exact run id as a path segment -- matching the
    *last* occurrence, in case some unrelated ancestor directory happens to
    share the run id string -- and keeping only that segment onward (plus
    one leading ``runs`` segment when the run id is nested directly under
    one, matching this project's own ``<results_dir>/runs/<run_id>/...``
    layout). This never depends on guessing which occurrence of a literal
    ``runs`` directory in the path is the canonical one, so it cannot leak
    an intervening private project/experiment directory name even when the
    path happens to contain more than one ``runs`` segment (e.g. because a
    user's own directory tree is itself named ``runs`` somewhere above the
    real results directory).

    When no run id is given, or it is not found as an exact path segment,
    falls back to the last ``runs`` segment (never the first, for the same
    reason) or, failing that, to the final path component alone -- never
    preserving any ancestor directory name.
    """
    posix_raw = raw.replace("\\", "/")
    parts = tuple(p for p in PurePosixPath(posix_raw).parts if p not in ("/", ""))

    if run_id:
        matches = [index for index, part in enumerate(parts) if part == run_id]
        if matches:
            start = matches[-1]
            if start > 0 and parts[start - 1] == "runs":
                start -= 1
            return "/".join(parts[start:])

    if "runs" in parts:
        last_runs_index = len(parts) - 1 - parts[::-1].index("runs")
        return "/".join(parts[last_runs_index:])

    name = PurePosixPath(posix_raw).name
    return name or raw


def _path_label(raw: str, *, redact_paths: bool, run_id: str | None = None) -> str:
    return raw if not redact_paths else _redact_path(raw, run_id=run_id)


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


# The mark's own geometry at diagram scale. Two measured traces converge on a
# pad: open when nothing was decided, filled when the sheet is complete. Used
# to anchor the empty Groups block and to terminate every report, so the
# signature routing recurs in the document rather than only in the masthead.
_ORNAMENT_TRACES = (
    '<path d="M0 6H72L88 14H120" stroke-width="2.5"/>'
    '<path d="M0 26H72L88 18H120" stroke-width="1.25"/>'
    '<path d="M24 3V9M48 3V9M64 3V9" stroke-width="1.25"/>'
)

ORNAMENT_OPEN_SVG = (
    '<svg class="ornament" viewBox="0 0 160 32" aria-hidden="true">'
    '<g fill="none" stroke="currentColor">'
    f"{_ORNAMENT_TRACES}"
    '<rect x="119.75" y="6.75" width="18.5" height="18.5" stroke-width="1.5"/>'
    "</g></svg>"
)

ORNAMENT_CLOSED_SVG = (
    '<svg class="ornament terminus" viewBox="0 0 160 32" aria-hidden="true">'
    '<g fill="none" stroke="currentColor">'
    f"{_ORNAMENT_TRACES}"
    "</g>"
    '<rect x="119" y="6" width="20" height="20" fill="currentColor"/>'
    "</svg>"
)

# The direction this document is built in, recorded in the document itself so a
# later change can be checked against the world it was designed for rather than
# against taste. Emitted as a comment in every rendered report.
DIRECTION_CONTRACT = """
THESIS: an instrument readout for a decision that has to survive an audit.
OWN-WORLD: oscilloscope graticule and measurement cursors (seed 3bd5d052).
Bone field, paper sheet lifted on one shadow, near black ink, one signal orange.
STORY: unmeasured runs enter, the policy sets thresholds, one candidate clears
them, and everything rejected stays on the page with the limit it breached.
FIRST VIEWPORT: lockup and document stamp, the policy under test, then a
divisional readout of totals, so the answer and its cost are legible before any
table is read.
FORM: hairline divisions, no cards, no corner radius, tabular mono for every
measurement and identifier, sans for prose, status carried by a word plus a
mark and never by color alone.
"""


def _style() -> str:
    return (
        TOKENS_CSS
        + """
    *, *::before, *::after { box-sizing: border-box; }

    html { -webkit-text-size-adjust: 100%; }

    body {
      margin: 0;
      padding: clamp(16px, 3vw, 44px) clamp(12px, 3vw, 40px) 72px;
      background-color: var(--field);
      background-image:
        repeating-linear-gradient(to right, var(--graticule) 0 1px, transparent 1px 48px),
        repeating-linear-gradient(to bottom, var(--graticule) 0 1px, transparent 1px 48px);
      color: var(--ink);
      font-family: var(--sans);
      font-size: 15px;
      line-height: 1.55;
      -webkit-font-smoothing: antialiased;
    }

    .sheet {
      max-width: 1180px;
      margin: 0 auto;
      background: var(--sheet);
      border: 1px solid var(--rule);
      box-shadow: 0 1px 1px #16181a0f, 0 26px 52px -30px #16181a5c;
      padding: clamp(20px, 4vw, 56px);
    }

    /* Masthead: brand on the left, document stamp on the right. */
    .masthead {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 20px 32px;
      flex-wrap: wrap;
      border-bottom: 1px solid var(--ink);
      padding-bottom: 14px;
      margin-bottom: clamp(28px, 4vw, 44px);
    }
    .lockup { display: block; height: 19px; width: auto; color: var(--ink); }
    .stamp {
      font-family: var(--mono);
      font-size: 10.5px;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      color: var(--muted);
      text-align: right;
      line-height: 1.75;
      margin: 0;
    }
    .stamp b { color: var(--ink); font-weight: 600; }

    h1 {
      font-size: clamp(1.55rem, 1.05rem + 1.9vw, 2.3rem);
      line-height: 1.13;
      letter-spacing: -0.022em;
      font-weight: 600;
      margin: 0 0 12px;
      max-width: 24ch;
      text-wrap: balance;
    }
    .lede {
      margin: 0 0 6px;
      max-width: 68ch;
      font-size: 1.0625rem;
      line-height: 1.5;
      color: var(--muted);
    }
    p { max-width: 72ch; }

    section { margin-top: clamp(34px, 4.5vw, 52px); }
    /* Section rule with a division tick, the way a graticule marks a division. */
    .rule {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px 24px;
      flex-wrap: wrap;
      position: relative;
      border-top: 1px solid var(--ink);
      padding-top: 13px;
      margin-bottom: 18px;
    }
    .rule::before {
      content: "";
      position: absolute;
      top: 0; left: 0;
      width: 2px; height: 7px;
      background: var(--signal);
    }
    h2 {
      font-size: 1.0625rem;
      font-weight: 600;
      letter-spacing: -0.01em;
      margin: 0;
    }
    h2 .qualifier {
      display: block;
      font-size: 10.5px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: var(--muted);
      margin-bottom: 5px;
    }
    h2 .subject-id {
      font-family: var(--mono);
      font-size: 0.9375rem;
      font-weight: 400;
      overflow-wrap: anywhere;
    }
    .rule > h2 { flex: 1 1 24rem; min-width: 0; }
    h3 {
      font-size: 0.8125rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin: 30px 0 12px;
    }
    h4 {
      font-size: 10.5px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: var(--muted);
      margin: 14px 0 4px;
    }
    .muted { color: var(--muted); }

    a { color: var(--ink); text-decoration-color: var(--rule); text-underline-offset: 3px; }
    a:hover { text-decoration-color: var(--signal); }
    a:focus-visible, summary:focus-visible {
      outline: 2px solid var(--signal);
      outline-offset: 3px;
    }
    .skip { position: absolute; left: -9999px; }
    .skip:focus {
      position: fixed; left: 16px; top: 16px; z-index: 9;
      background: var(--ink); color: var(--sheet);
      padding: 10px 14px; font-size: 13px;
    }

    /* Contents: the transect through the evidence, station by station. */
    .transect {
      border-top: 1px solid var(--rule);
      border-bottom: 1px solid var(--rule);
      padding: 6px 0;
      margin-top: 48px;
    }
    .transect ol { list-style: none; margin: 0; padding: 0; }
    .transect li { border-top: 1px solid var(--rule-soft); }
    .transect li:first-child { border-top: 0; }
    .transect a {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr) auto;
      gap: 4px 12px;
      align-items: baseline;
      padding: 7px 0;
      font-size: 13px;
      line-height: 1.45;
      text-decoration: none;
      overflow-wrap: anywhere;
    }
    .transect .label { min-width: 0; }
    .transect a:hover { text-decoration: underline; text-underline-offset: 3px; }
    .transect .station {
      flex: 0 0 auto;
      width: 7px; height: 7px;
      background: var(--ink);
      transform: translateY(-1px);
    }
    .transect .station-rec { background: var(--signal); }
    .transect .station-hold { background: none; border: 1.5px solid var(--hold); }
    .transect .note {
      font-family: var(--mono);
      font-size: 10.5px;
      letter-spacing: 0.09em;
      text-transform: uppercase;
      color: var(--ink);
      white-space: nowrap;
    }
    .transect .note-hold { color: var(--hold); }

    /* Divisional readout: counts on one ruled strip, no cards, no hierarchy
       between the numbers, because none of them outranks another. */
    .readout {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(132px, 1fr));
      border-top: 1px solid var(--ink);
      border-bottom: 1px solid var(--rule);
    }
    .division {
      position: relative;
      padding: 15px 16px 15px 15px;
      border-left: 1px solid var(--rule-soft);
    }
    .division:first-child { border-left: 0; padding-left: 0; }
    .division::before {
      content: "";
      position: absolute;
      top: 0; left: -1px;
      width: 1px; height: 6px;
      background: var(--ink);
    }
    .division:first-child::before { display: none; }
    .division .v {
      display: block;
      font-family: var(--mono);
      font-size: 1.375rem;
      font-variant-numeric: tabular-nums;
      line-height: 1.1;
      letter-spacing: -0.015em;
    }
    .division .k {
      display: block;
      margin-top: 7px;
      font-size: 10.5px;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: var(--muted);
    }
    .v-rec { color: var(--signal); }
    .v-hold { color: var(--hold); }
    .v-breach { color: var(--breach); }

    /* Status is a word plus a mark. The color only ever repeats what the
       word already says, so it is safe to lose it. */
    .state {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      font-family: var(--mono);
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.09em;
      text-transform: uppercase;
      white-space: nowrap;
    }
    .state::before { content: ""; width: 8px; height: 8px; flex: 0 0 auto; }
    .state-rec { color: var(--signal); }
    .state-rec::before { background: var(--signal); }
    .state-ok { color: var(--verify); }
    .state-ok::before { background: var(--verify); }
    .state-bad { color: var(--breach); }
    .state-bad::before { background: var(--breach); }
    .state-hold { color: var(--hold); }
    .state-hold::before { border: 1.5px solid var(--hold); }

    /* Verdict block: ruled, not carded. */
    .verdict {
      border-top: 1px solid var(--ink);
      border-bottom: 1px solid var(--rule);
      padding: 16px 0 18px;
      margin: 0;
    }
    .verdict .subject {
      display: block;
      font-family: var(--mono);
      font-size: 1rem;
      margin: 10px 0 0;
      word-break: break-word;
    }
    .verdict p { margin: 8px 0 0; max-width: 72ch; }

    /* Name/value specification lists. */
    .spec {
      display: grid;
      grid-template-columns: minmax(11rem, auto) 1fr;
      margin: 0;
      border-top: 1px solid var(--rule);
    }
    .spec dt {
      padding: 9px 24px 9px 0;
      font-size: 12.5px;
      color: var(--muted);
      border-bottom: 1px solid var(--rule-soft);
    }
    .spec dd {
      margin: 0;
      padding: 9px 0;
      font-family: var(--mono);
      font-size: 12.5px;
      font-variant-numeric: tabular-nums;
      border-bottom: 1px solid var(--rule-soft);
      word-break: break-word;
    }

    /* Evidence tables. */
    .scroller { overflow-x: auto; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13.5px;
    }
    caption {
      caption-side: top;
      text-align: left;
      font-size: 12.5px;
      color: var(--muted);
      padding-bottom: 12px;
      max-width: 72ch;
    }
    thead th {
      font-size: 10.5px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      text-align: left;
      vertical-align: bottom;
      padding: 0 14px 8px 0;
      border-bottom: 1px solid var(--ink);
      white-space: nowrap;
    }
    tbody th, tbody td {
      text-align: left;
      font-weight: 400;
      vertical-align: top;
      padding: 11px 14px 11px 0;
      border-bottom: 1px solid var(--rule-soft);
    }
    tbody th { font-family: var(--mono); word-break: break-word; }
    .num {
      font-family: var(--mono);
      font-variant-numeric: tabular-nums;
      text-align: right;
      padding-right: 20px;
      white-space: nowrap;
    }
    thead th.num { text-align: right; padding-right: 20px; }
    /* Each candidate is a row group: its identifier spans the full width so
       the measurement columns stay narrow enough to read across, and the
       identifier is never clipped or broken into a one-word column. */
    tbody.cand { border-top: 1px solid var(--rule-soft); }
    tbody.cand:first-of-type { border-top: 0; }
    .cand-head > th {
      padding: 14px 0 6px;
      border-bottom: 0;
      font-family: var(--sans);
      font-weight: 400;
    }
    .cand-id {
      display: flex;
      align-items: baseline;
      gap: 10px 14px;
      flex-wrap: wrap;
    }
    .cand-rank {
      font-family: var(--mono);
      font-size: 13.5px;
      color: var(--muted);
      white-space: nowrap;
    }
    .cand-label {
      font-family: var(--mono);
      font-size: 13.5px;
      overflow-wrap: anywhere;
      flex: 1 1 18rem;
      min-width: 0;
    }
    .cand-metrics > td { border-bottom: 0; padding-top: 4px; }
    .cand-evidence > td { border-bottom: 0; padding: 0 0 12px; }
    /* The recommended row is marked in the gutter with the same filled pad the
       logo ends on, not with a colored stripe down its side. */
    .pad, .pad-open {
      display: inline-block;
      width: 8px; height: 8px;
      margin-right: 9px;
      transform: translateY(-1px);
    }
    .pad { background: var(--signal); }
    .pad-open { border: 1px solid var(--rule); }
    tbody.is-rec > tr > * { background: var(--signal-tint); }
    tbody.is-rec .cand-rank { color: var(--ink); }

    /* Rejected candidates: each one keeps the limit it breached. */
    .reject { border-top: 1px solid var(--rule); padding: 15px 0; }
    .reject:first-of-type { border-top: 1px solid var(--ink); }
    .reject .subject {
      font-family: var(--mono);
      font-size: 13.5px;
      word-break: break-word;
    }
    .violations { list-style: none; margin: 9px 0 0; padding: 0; max-width: 72ch; }
    .violations li { position: relative; padding-left: 20px; margin: 5px 0; }
    .violations li::before {
      content: "";
      position: absolute;
      left: 0; top: 0.62em;
      width: 11px; height: 2px;
      background: var(--breach);
    }

    /* Cursor pair: two measurements against a shared baseline, then the delta,
       the way two cursors on a scope produce one reading. */
    .cursors { border-top: 1px solid var(--rule); margin-top: 14px; }
    .cursor {
      display: grid;
      grid-template-columns: minmax(9rem, auto) 1fr auto;
      gap: 8px 18px;
      align-items: baseline;
      padding: 10px 0;
      border-bottom: 1px solid var(--rule-soft);
    }
    .cursor .who { font-size: 12.5px; color: var(--muted); }
    .cursor .what {
      font-family: var(--mono);
      font-size: 12.5px;
      word-break: break-word;
      color: var(--ink);
    }
    .cursor .reading {
      font-family: var(--mono);
      font-variant-numeric: tabular-nums;
      text-align: right;
      white-space: nowrap;
    }
    .cursor-delta { border-bottom: 1px solid var(--ink); }
    .cursor-delta .who { color: var(--ink); }
    .runs { font-size: 12px; color: var(--muted); font-family: var(--mono); }

    /* Excluded material carries a diagonal ruling, the way set aside samples
       are struck through on a lab sheet. No hue: this is not a warning. */
    .hazard tbody th { white-space: nowrap; }
    .hazard tbody td { overflow-wrap: anywhere; }
    .hazard tbody tr {
      background-image: repeating-linear-gradient(
        135deg, #16181a0d 0 5px, transparent 5px 11px);
    }

    /* The routed motif at diagram scale. Quiet by default: it is a mark of
       where a measurement ended, not an illustration. */
    .ornament {
      display: block;
      width: 232px;
      height: auto;
      color: var(--ink);
      opacity: 0.3;
      margin: 26px 0 0;
    }
    .ornament.terminus { width: 148px; opacity: 0.18; margin: 44px 0 0; }
    .empty-spec { max-width: 68ch; }
    .empty-spec div {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 4px 24px;
      padding: 9px 0;
      border-bottom: 1px solid var(--rule-soft);
      font-size: 13px;
    }
    .empty-spec .nil {
      font-family: var(--mono);
      font-size: 12.5px;
      color: var(--muted);
      font-variant-numeric: tabular-nums;
    }

    .evidence-list { list-style: none; margin: 5px 0 12px; padding: 0; }
    .evidence-list li { padding: 3px 0; }
    code {
      font-family: var(--mono);
      font-size: 12.5px;
      word-break: break-all;
    }

    details { margin-top: 4px; }
    summary {
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 9px;
      padding: 6px 0;
      font-size: 10.5px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: var(--muted);
      list-style: none;
    }
    summary::-webkit-details-marker { display: none; }
    summary::before {
      content: "";
      width: 6px; height: 6px;
      border-right: 1.5px solid currentColor;
      border-bottom: 1.5px solid currentColor;
      transform: rotate(-45deg);
      transition: transform 180ms cubic-bezier(0.2, 0.8, 0.2, 1);
      margin-bottom: 2px;
    }
    details[open] > summary::before { transform: rotate(45deg); }
    summary:hover { color: var(--ink); }

    .empty {
      border-top: 1px solid var(--rule);
      padding: 22px 0 18px;
      color: var(--muted);
      max-width: 68ch;
    }

    footer {
      margin-top: clamp(40px, 5vw, 60px);
      border-top: 1px solid var(--rule);
      padding-top: 14px;
      display: flex;
      justify-content: space-between;
      gap: 10px 32px;
      flex-wrap: wrap;
      font-family: var(--mono);
      font-size: 10.5px;
      letter-spacing: 0.07em;
      text-transform: uppercase;
      color: var(--muted);
    }

    @media (prefers-reduced-motion: reduce) {
      * { transition-duration: 0.01ms !important; }
    }

    /* Narrow screens read one evidence chain at a time: the table stops being
       a grid and becomes labelled rows, so no measurement is ever clipped. */
    @media (max-width: 760px) {
      .reflow thead {
        position: absolute;
        width: 1px; height: 1px;
        margin: -1px; padding: 0;
        overflow: hidden;
        clip: rect(0 0 0 0);
        white-space: nowrap;
        border: 0;
      }
      .reflow, .reflow tbody, .reflow tbody tr { display: block; width: 100%; }
      /* A table-caption inside a block-ified table generates an anonymous
         shrink-to-fit table box and collapses to one word per line. */
      .reflow caption { display: block; width: 100%; }
      .reflow tbody th, .reflow tbody td {
        display: grid;
        grid-template-columns: minmax(0, 38%) minmax(0, 1fr);
        gap: 6px 14px;
        padding: 5px 0;
        border-bottom: 0;
        text-align: left;
        min-width: 0;
        white-space: normal;
      }
      .reflow tbody th::before, .reflow tbody td::before {
        content: attr(data-label);
        font-family: var(--sans);
        font-size: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.07em;
        color: var(--muted);
        padding-top: 3px;
        overflow-wrap: anywhere;
      }
      /* Values are the point of the document: they get the room, and they
         wrap rather than clip, whatever the identifier length. */
      .reflow tbody th > *, .reflow tbody td > * { min-width: 0; overflow-wrap: anywhere; }
      .reflow tbody code { word-break: break-all; }
      .reflow .num { text-align: left; padding-right: 0; }
      .reflow tbody td, .reflow tbody th { overflow-wrap: anywhere; }
      .reflow .cand-head > th, .reflow .cand-evidence > td { display: block; }
      .reflow .cand-head > th::before, .reflow .cand-evidence > td::before { content: none; }
      .reflow tbody.cand { display: block; border-top: 1px solid var(--rule); padding-bottom: 8px; }
      .reflow tbody.cand:first-of-type { border-top: 1px solid var(--ink); }
      /* Stacked records need their own seam; without it two consecutive
         rows read as one long record. */
      .reflow tbody:not(.cand) tr + tr {
        border-top: 1px solid var(--rule);
        margin-top: 8px;
        padding-top: 8px;
      }
      .scroller { overflow-x: visible; }
      .spec { grid-template-columns: 1fr; }
      .spec dt { padding-bottom: 0; border-bottom: 0; }
      .spec dd { padding-top: 2px; }
      .cursor { grid-template-columns: 1fr; }
      .cursor .reading { text-align: left; }
      .transect a { grid-template-columns: auto minmax(0, 1fr); }
      .transect .note { grid-column: 2; }
      .stamp { text-align: left; }
    }

    @media print {
      body {
        background: #fff;
        padding: 0;
        font-size: 10.5pt;
      }
      .sheet {
        max-width: none;
        border: 0;
        box-shadow: none;
        padding: 0;
        background: #fff;
      }
      .skip, .transect { display: none; }
      a { text-decoration: none; }
      tr, .reject, .cursor, .division, tbody.cand { break-inside: avoid; }
      h1, h2, h3, .rule { break-after: avoid; }
      section { break-inside: auto; }
      /* Printed evidence is not hidden evidence: expand disclosures for paper
         in engines that support it, and label them where it is unsupported. */
      details::details-content { content-visibility: visible; }
      details:not([open]) > summary::after { content: " (collapsed on screen)"; }
    }
    """
    ).strip()


def _render_constraints(constraints: TuneConstraints) -> str:
    """Render the policy constraints as the threshold list they are.

    Every row here is a limit a candidate had to clear; the accepted table
    and the rejection reasons both refer back to these values, so they are
    rendered as measurements (mono, tabular) rather than as prose.
    """
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
    return f'<dl class="spec">{items}</dl>'


def _render_policy(policy: TunePolicy) -> str:
    name = policy.name or "(unnamed policy)"
    description = (
        f'<p class="muted">{_esc(policy.description)}</p>' if policy.description else ""
    )
    return (
        '<section id="policy" aria-labelledby="policy-h">'
        '<div class="rule"><h2 id="policy-h">Policy and constraints</h2>'
        f'<span class="muted">{_esc(name)}</span></div>'
        f"{description}"
        f'<dl class="spec"><dt>Objective</dt><dd>{_esc(policy.objective.value)}</dd></dl>'
        "<h3>Constraints applied to every candidate</h3>"
        f"{_render_constraints(policy.constraints)}"
        "</section>"
    )


def _counts(report: TuneReport) -> tuple[tuple[str, int, str], ...]:
    """Summary totals, in reading order, with the state class for each value."""
    recommended_groups = sum(
        1 for group in report.groups if group.outcome == GroupOutcome.RECOMMENDED
    )
    inconclusive_groups = sum(
        1 for group in report.groups if group.outcome == GroupOutcome.INCONCLUSIVE
    )
    accepted_total = sum(len(group.accepted) for group in report.groups)
    rejected_total = sum(len(group.rejected) for group in report.groups)
    excluded_total = len(report.excluded_runs)
    return (
        ("Groups", len(report.groups), ""),
        ("Recommendations", recommended_groups, "v-rec" if recommended_groups else ""),
        ("Inconclusive", inconclusive_groups, "v-hold" if inconclusive_groups else ""),
        ("Accepted candidates", accepted_total, ""),
        ("Rejected candidates", rejected_total, ""),
        ("Excluded runs", excluded_total, "v-breach" if excluded_total else ""),
    )


def _render_summary(report: TuneReport) -> str:
    divisions = "".join(
        f'<div class="division">'
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
    return f'<dl class="spec">{items}</dl>'


def _artifact_lists(
    *,
    run_ids: tuple[str, ...],
    verification_paths: tuple[str, ...],
    final_record_paths: tuple[str, ...],
    redact_paths: bool,
) -> str:
    verification = _list_items(
        _path_label(path, redact_paths=redact_paths, run_id=run_id)
        for path, run_id in zip_longest(verification_paths, run_ids, fillvalue=None)
        if path is not None
    )
    final_records = _list_items(
        _path_label(path, redact_paths=redact_paths, run_id=run_id)
        for path, run_id in zip_longest(final_record_paths, run_ids, fillvalue=None)
        if path is not None
    )
    return (
        f"<h4>Run IDs</h4>{_list_items(run_ids)}"
        f"<h4>Verification artifacts</h4>{verification}"
        f"<h4>Final record artifacts</h4>{final_records}"
    )


def _candidate_evidence(candidate: CandidateReport, *, redact_paths: bool) -> str:
    body = _artifact_lists(
        run_ids=candidate.run_ids,
        verification_paths=candidate.verification_paths,
        final_record_paths=candidate.final_record_paths,
        redact_paths=redact_paths,
    )
    return (
        "<details><summary>Source run IDs and artifact paths</summary>"
        f"{body}</details>"
    )


def _render_recommendation_rationale(
    group: GroupReport, *, ranked_accepted: tuple[CandidateReport, ...]
) -> str:
    winner = group.recommended
    assert winner is not None
    parts = [
        f"Won on objective <code>{_esc(winner.objective_name)}</code> with value "
        f"<code>{_fmt_number(winner.objective_value)}</code>, backed by "
        f"{winner.evidence_count} accepted run(s)."
    ]
    if len(ranked_accepted) > 1:
        # ``ranked_accepted`` is sorted by rank, never by JSON/source list
        # order, so index 1 is always the true rank-2 runner-up.
        runner_up = ranked_accepted[1]
        parts.append(
            "Ranked ahead of runner-up "
            f"<code>{_esc(runner_up.candidate_key.label())}</code> "
            f"(objective value <code>{_fmt_number(runner_up.objective_value)}</code>)."
        )
    else:
        parts.append("It was the only candidate accepted in this group.")
    return " ".join(parts)


#: Measurement columns for the accepted ranking table. The candidate
#: identifier is deliberately not a column: it is long, it is the one value
#: that must never be clipped, and giving it a column squeezes every
#: measurement beside it. It heads its own row group instead.
_ACCEPTED_COLUMNS: tuple[str, ...] = (
    "Objective",
    "Mean latency (ms)",
    "Pass rate",
    "Quality metric / score",
    "Peak memory (mean / max)",
    "Evidence",
    "CV",
)


def _render_accepted_table(
    accepted: tuple[CandidateReport, ...],
    *,
    recommended_rank: int | None,
    redact_paths: bool,
    objective_name: str,
) -> str:
    if not accepted:
        return (
            '<p class="empty">No accepted candidates. Nothing in this group '
            "cleared every constraint, so there is no ranking to show.</p>"
        )
    header = "".join(
        f'<th scope="col" role="columnheader" class="num">{_esc(label)}</th>'
        for label in _ACCEPTED_COLUMNS
    )
    groups: list[str] = []
    for candidate in sorted(accepted, key=lambda item: item.rank):
        is_recommended = candidate.rank == recommended_rank
        # The recommended candidate is marked in the rank gutter with the same
        # filled pad the logo terminates on, and with the word itself, so the
        # marking survives greyscale, print, and colour vision deficiency.
        pad = "pad" if is_recommended else "pad-open"
        marker = (
            '<span class="state state-rec">Recommended</span>' if is_recommended else ""
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
        values = (
            _fmt_number(candidate.objective_value),
            _fmt_number(candidate.mean_total_latency_ms, digits=2),
            _fmt_percent(candidate.pass_rate),
            quality,
            memory,
            str(candidate.evidence_count),
            _fmt_number(candidate.coefficient_of_variation, digits=4),
        )
        cells = "".join(
            f'<td role="cell" class="num" data-label="{_esc(label)}">{value}</td>'
            for label, value in zip(_ACCEPTED_COLUMNS, values, strict=True)
        )
        span = len(_ACCEPTED_COLUMNS)
        groups.append(
            f'<tbody role="rowgroup" class="cand{" is-rec" if is_recommended else ""}">'
            f'<tr role="row" class="cand-head">'
            f'<th scope="rowgroup" role="rowheader" colspan="{span}"><span class="cand-id">'
            f'<span class="cand-rank">'
            f'<span class="{pad}" aria-hidden="true"></span>#{candidate.rank}</span>'
            f'<span class="cand-label">{_esc(candidate.candidate_key.label())}</span>'
            f"{marker}</span></th></tr>"
            f'<tr role="row" class="cand-metrics">{cells}</tr>'
            f'<tr role="row" class="cand-evidence"><td role="cell" colspan="{span}">'
            + _candidate_evidence(candidate, redact_paths=redact_paths)
            + "</td></tr></tbody>"
        )
    caption = (
        f"{len(accepted)} candidate(s) cleared every constraint, ranked by "
        f"{objective_name}. The recommended row is marked in the rank gutter."
    )
    return (
        '<div class="scroller"><table class="reflow" role="table">'
        f"<caption>{_esc(caption)}</caption>"
        f'<thead role="rowgroup"><tr role="row">{header}</tr></thead>'
        f"{''.join(groups)}"
        "</table></div>"
    )


def _render_rejected(
    rejected: tuple[RejectedCandidateReport, ...], *, redact_paths: bool
) -> str:
    if not rejected:
        return (
            "<h3>Rejected candidates</h3>"
            '<p class="empty">None. Every candidate in this group cleared '
            "the policy constraints.</p>"
        )
    blocks = []
    for candidate in rejected:
        reasons = "".join(f"<li>{_esc(reason)}</li>" for reason in candidate.reasons)
        artifacts = _artifact_lists(
            run_ids=candidate.run_ids,
            verification_paths=candidate.verification_paths,
            final_record_paths=candidate.final_record_paths,
            redact_paths=redact_paths,
        )
        blocks.append(
            '<div class="reject">'
            '<span class="state state-bad">Rejected</span>'
            f'<div class="subject">{_esc(candidate.candidate_key.label())}</div>'
            f'<ul class="violations">{reasons}</ul>'
            "<details><summary>Source run IDs and artifact paths</summary>"
            f"{artifacts}</details>"
            "</div>"
        )
    return (
        f"<h3>Rejected candidates ({len(rejected)})</h3>"
        '<p class="muted">Each candidate below is listed with every constraint '
        "it breached.</p>" + "".join(blocks)
    )


def _render_baseline_comparison(comparison: BaselineComparison) -> str:
    verdict = comparison.report.verdict
    state_class = {
        DoctorVerdict.IMPROVEMENT: "state-ok",
        DoctorVerdict.REGRESSION: "state-bad",
        DoctorVerdict.NO_SIGNIFICANT_DIFFERENCE: "state-hold",
        DoctorVerdict.INCONCLUSIVE: "state-hold",
    }.get(verdict, "state-hold")
    baseline_ms = _fmt_number(comparison.report.baseline_mean_total_ms, digits=2)
    speculative_ms = _fmt_number(comparison.report.speculative_mean_total_ms, digits=2)
    delta_ms = _fmt_number(comparison.report.delta_ms, digits=2)
    delta_pct = _fmt_percent(comparison.report.delta_pct)
    baseline_runs = _esc(", ".join(comparison.report.baseline_run_ids))
    speculative_runs = _esc(", ".join(comparison.report.speculative_run_ids))
    return (
        "<h3>Speculative baseline comparison</h3>"
        f'<p><span class="state {state_class}">{_esc(verdict.value)}</span></p>'
        f"<p>{_esc(comparison.report.reason)}</p>"
        '<div class="cursors">'
        '<div class="cursor"><span class="who">Baseline</span>'
        f'<span class="what">{_esc(comparison.baseline_candidate_key.label())}'
        f'<br><span class="runs">run(s): {baseline_runs}</span></span>'
        f'<span class="reading">{baseline_ms} ms</span></div>'
        '<div class="cursor"><span class="who">Speculative</span>'
        f'<span class="what">{_esc(comparison.speculative_candidate_key.label())}'
        f'<br><span class="runs">run(s): {speculative_runs}</span></span>'
        f'<span class="reading">{speculative_ms} ms</span></div>'
        '<div class="cursor cursor-delta"><span class="who">Delta</span>'
        '<span class="what">Speculative measured against baseline</span>'
        f'<span class="reading">{delta_ms} ms / {delta_pct}</span></div>'
        "</div>"
    )


def _group_anchor(index: int) -> str:
    return f"group-{index}"


def _render_group(group: GroupReport, *, index: int, redact_paths: bool) -> str:
    anchor = _group_anchor(index)
    label = _esc(group.group_key.label())
    is_recommended = (
        group.outcome == GroupOutcome.RECOMMENDED and group.recommended is not None
    )
    state = (
        '<span class="state state-rec">RECOMMENDED</span>'
        if is_recommended
        else '<span class="state state-hold">INCONCLUSIVE</span>'
    )
    parts = [
        f'<section id="{anchor}" aria-labelledby="{anchor}-h">',
        f'<div class="rule"><h2 id="{anchor}-h">'
        f'<span class="qualifier">Comparable group</span>'
        f'<span class="subject-id">{label}</span></h2>{state}</div>',
    ]

    # Rendering must always follow rank order, never whatever order the
    # source JSON/list happened to be in -- computed once here and reused
    # by both the recommendation rationale and the accepted table so a
    # runner-up mention and its table row can never disagree.
    ranked_accepted = tuple(sorted(group.accepted, key=lambda item: item.rank))

    if is_recommended:
        winner = group.recommended
        assert winner is not None
        parts.append(
            '<div class="verdict is-rec">'
            "<p>Recommended configuration for this group:</p>"
            f'<span class="subject">{_esc(winner.candidate_key.label())}</span>'
            f"<p>{_render_recommendation_rationale(group, ranked_accepted=ranked_accepted)}</p>"
            "</div>"
        )
    else:
        parts.append(
            '<div class="verdict">'
            "<p>No candidate in this group could be recommended on the "
            "available evidence.</p>"
            f"<p>{_esc(group.inconclusive_reason or 'no reason recorded')}</p>"
            "</div>"
        )

    recommended_rank = group.recommended.rank if group.recommended is not None else None
    parts.append("<h3>Accepted candidate ranking</h3>")
    parts.append(
        _render_accepted_table(
            ranked_accepted,
            recommended_rank=recommended_rank,
            redact_paths=redact_paths,
            objective_name=(
                group.recommended.objective_name
                if group.recommended is not None
                else (ranked_accepted[0].objective_name if ranked_accepted else "rank")
            ),
        )
    )

    parts.append(_render_rejected(group.rejected, redact_paths=redact_paths))

    if group.baseline_comparison is not None:
        parts.append(_render_baseline_comparison(group.baseline_comparison))

    parts.append("</section>")
    return "".join(parts)


def _render_provenance(report: TuneReport, *, redact_paths: bool) -> str:
    """Where the evidence came from, and what was thrown out before scoring."""
    sources = _list_items(
        _path_label(d, redact_paths=redact_paths) for d in report.results_dirs
    )
    parts = [
        '<section id="provenance" aria-labelledby="provenance-h">',
        '<div class="rule"><h2 id="provenance-h">Provenance</h2>'
        f'<span class="muted">{"paths redacted" if redact_paths else "full paths included"}</span></div>',
        "<h3>Source results directories</h3>",
        sources,
    ]
    if report.excluded_runs:
        rows = "".join(
            '<tr role="row">'
            f'<th scope="row" role="rowheader" data-label="Run ID">'
            f"<code>{_esc(run.run_id)}</code></th>"
            f'<td role="cell" data-label="Source results directory"><code>'
            f"{_esc(_path_label(run.source_results_dir, redact_paths=redact_paths, run_id=run.run_id))}"
            "</code></td>"
            f'<td role="cell" data-label="Reason">{_esc(run.reason)}</td>'
            "</tr>"
            for run in report.excluded_runs
        )
        parts.append(f"<h3>Excluded runs ({len(report.excluded_runs)})</h3>")
        parts.append(
            '<p class="muted">Runs whose evidence could not be trusted at all '
            "(never even considered as a rejected candidate).</p>"
        )
        parts.append(
            '<div class="scroller"><table class="reflow hazard" role="table">'
            "<caption>Every run listed here was set aside before any candidate "
            "was scored, so none of it reached the ranking above.</caption>"
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
            "carried enough evidence to be scored.</p>"
        )
    parts.append("</section>")
    return "".join(parts)


def _render_transect(report: TuneReport) -> str:
    """In-page contents: one station per section, in reading order."""
    stations: list[str] = [
        '<li><a href="#summary"><span class="station" aria-hidden="true"></span>'
        '<span class="label">Summary</span></a></li>'
    ]
    for index, group in enumerate(report.groups, start=1):
        recommended = (
            group.outcome == GroupOutcome.RECOMMENDED and group.recommended is not None
        )
        station_class = "station station-rec" if recommended else "station station-hold"
        # The station square is the mark, so the note only has to supply the
        # word -- same grammar as the state chips, one scale down.
        note_class = "note" if recommended else "note note-hold"
        note = "RECOMMENDED" if recommended else "INCONCLUSIVE"
        stations.append(
            f'<li><a href="#{_group_anchor(index)}">'
            f'<span class="{station_class}" aria-hidden="true"></span>'
            f'<span class="label">{_esc(group.group_key.label())}</span>'
            f'<span class="{note_class}">{note}</span></a></li>'
        )
    stations.append(
        '<li><a href="#policy"><span class="station" aria-hidden="true"></span>'
        '<span class="label">Policy and constraints</span></a></li>'
    )
    stations.append(
        '<li><a href="#provenance"><span class="station" aria-hidden="true"></span>'
        '<span class="label">Provenance</span></a></li>'
    )
    return (
        '<nav class="transect" aria-label="Report contents">'
        f"<ol>{''.join(stations)}</ol></nav>"
    )


def _render_masthead(report: TuneReport, *, redact_paths: bool) -> str:
    redaction = "paths redacted" if redact_paths else "full paths included"
    return (
        '<header class="masthead">'
        f"{LOCKUP_SVG}"
        '<p class="stamp">Tune report'
        f"<br><b>{_esc(report.generated_at)}</b>"
        f"<br>Schema {_esc(report.schema_version)} &middot; {redaction}</p>"
        "</header>"
    )


def _meta_description(report: TuneReport) -> str:
    counts = {label: value for label, value, _ in _counts(report)}
    return (
        f"Tune report for policy {report.policy.name or 'unnamed'}: "
        f"{counts['Groups']} comparable group(s), "
        f"{counts['Recommendations']} recommendation(s), "
        f"{counts['Inconclusive']} inconclusive, "
        f"{counts['Rejected candidates']} rejected candidate(s), "
        f"{counts['Excluded runs']} excluded run(s). "
        f"Generated {report.generated_at}."
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
        '<a class="skip" href="#report">Skip to the report</a>',
        '<div class="sheet">',
        _render_masthead(report, redact_paths=redact_paths),
        '<main id="report">',
        f"<h1>{_esc(title)}</h1>",
        '<p class="lede">Every value below comes from completed runs that were '
        "already recorded and validated. Nothing here is re-scored or estimated: "
        "candidates are ranked only on evidence that satisfied the policy "
        "constraints, and anything that failed is kept on the page with the "
        "limit it breached.</p>",
        _render_transect(report),
        _render_summary(report),
    ]

    if not report.groups:
        body_parts.append(
            '<section id="groups" aria-labelledby="groups-h">'
            '<div class="rule"><h2 id="groups-h">Groups</h2></div>'
            '<p class="empty">No comparable groups were found in the '
            "provided results directories. Nothing could be compared, so "
            "nothing is recommended.</p>"
            '<div class="empty-spec">'
            + "".join(
                f'<div><span>{label}</span><span class="nil">none</span></div>'
                for label in (
                    "Comparable groups",
                    "Recommendations",
                    "Accepted candidates",
                    "Rejected candidates",
                    "Speculative baseline comparisons",
                )
            )
            + "</div>"
            f"{ORNAMENT_OPEN_SVG}"
            "</section>"
        )
    else:
        for index, group in enumerate(report.groups, start=1):
            body_parts.append(
                _render_group(group, index=index, redact_paths=redact_paths)
            )

    body_parts.append(_render_policy(report.policy))
    body_parts.append(_render_provenance(report, redact_paths=redact_paths))
    body_parts.append("</main>")
    body_parts.append(
        f"{ORNAMENT_CLOSED_SVG}"
        "<footer><span>Generated by <code>llmtracefx-optimizer tune-report</code>"
        "</span><span>Static HTML, no external references, safe to open from "
        "disk</span></footer>"
    )
    body_parts.append("</div>")

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<meta name="color-scheme" content="light">\n'
        '<meta name="generator" content="llmtracefx-optimizer tune-report">\n'
        '<meta name="robots" content="noindex, nofollow">\n'
        f'<meta name="description" content="{_esc(_meta_description(report))}">\n'
        f"<title>{_esc(title)} &middot; LLMTraceFX tune report</title>\n"
        f"<!--{DIRECTION_CONTRACT}-->\n"
        f"<style>{_style()}</style>\n"
        "</head>\n"
        "<body>\n" + "".join(body_parts) + "\n</body>\n</html>\n"
    )
