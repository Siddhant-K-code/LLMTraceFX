"""Brand constants shared by every HTML surface LLMTraceFX generates.

These are the single source of truth for the identity: the token block and
the lockup live here, not copied into each renderer, so a colour or a mark
can never mean one thing in the tune report and another in the dashboard.
See ``DESIGN.md`` for what each token is for and when it may be used.

This module is standard library only, and deliberately so: the offline tune
report renderer imports it, and that renderer must stay dependency free.
"""

from __future__ import annotations

#: Design tokens, as a CSS custom property block.
#:
#: Contrast is checked against ``--sheet``: ``--signal`` is 5.06:1,
#: ``--muted`` 6.00:1, ``--verify`` 8.85:1, ``--breach`` 8.67:1 and
#: ``--hold`` 7.72:1, so every one of them clears WCAG AA for body text
#: rather than only for large text.
TOKENS_CSS = """
    :root {
      color-scheme: light;
      /* Ground. A bone field carrying a faint graticule, and a lighter paper
         sheet that the document is printed on. */
      --field: #f4f1ea;
      --sheet: #fbfaf7;
      --graticule: #16181a0f;
      /* Ink and rules. */
      --ink: #16181a;
      --muted: #5b6167;
      --rule: #16181a26;
      --rule-soft: #16181a14;
      /* One accent, used for structure and for the single marked thing in a
         region: 5.06:1 on the sheet, so it is safe as text as well as fill. */
      --signal: #c23d16;
      --signal-tint: #c23d1612;
      /* Data states. Each one is always paired with a word, never used alone.
         "hold" is graphite rather than amber on purpose: refusing to draw a
         conclusion is a null reading, not a warning. */
      --verify: #17513a;
      --breach: #8c1d28;
      --hold: #4a5157;
      --sans: -apple-system, BlinkMacSystemFont, "Segoe UI Variable Text",
        "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas,
        "Liberation Mono", "DejaVu Sans Mono", monospace;
    }
""".strip()

#: The routed wordmark path, drawn on the same 45 degree grid as the symbol.
_WORDMARK_PATH = (
    "M1.5 7.5V24.5H11.5 M17.5 7.5V24.5H27.5 M33.5 24.5V7.5L40 16L46.5 7.5V24.5 "
    "M52.5 7.5H62.5M57.5 7.5V24.5 "
    "M68.5 24.5V7.5H75.5L78.5 10.5V13L75.5 15.5H68.5M75 15.5L78.5 24.5 "
    "M84.5 24.5V11L88 7.5H91L94.5 11V24.5M84.5 17H94.5 "
    "M110.5 10.5L107.5 7.5H103.5L100.5 10.5V21.5L103.5 24.5H107.5L110.5 21.5 "
    "M126.5 7.5H116.5V24.5H126.5M116.5 16H124.5 "
    "M142.5 7.5H132.5V24.5M132.5 16H140.5 "
    "M148.5 7.5L158.5 24.5M158.5 7.5L148.5 24.5"
)

#: The two converging traces and the decision pad, shared by mark and lockup.
_TRACES = (
    '<g fill="none" stroke="currentColor" stroke-linecap="butt" '
    'stroke-linejoin="miter">'
    '<path d="M3 8H10L16 14H19" stroke-width="3.5"/>'
    '<path d="M3 24H10L16 18H19" stroke-width="1.75"/>'
)
_PAD = '<rect x="18" y="10.5" width="11" height="11" fill="var(--signal, #c23d16)"/>'

#: Symbol only, for tight spaces. Inline SVG with no ``xmlns`` attribute: HTML
#: parsing does not need one, and leaving it out keeps generated documents free
#: of anything that looks like an external reference.
MARK_SVG = f'<svg class="mark" viewBox="0 0 32 32" role="img" aria-label="LLMTraceFX">{_TRACES}</g>{_PAD}</svg>'

#: Horizontal lockup: symbol plus routed wordmark.
LOCKUP_SVG = (
    '<svg class="lockup" viewBox="0 0 206 32" role="img" aria-label="LLMTraceFX">'
    f"{_TRACES}"
    f'<path transform="translate(43 0)" stroke-width="3" d="{_WORDMARK_PATH}"/>'
    f"</g>{_PAD}</svg>"
)

__all__ = [
    "CHART_SEQUENCE",
    "HEATMAP_SCALE",
    "LOCKUP_SVG",
    "MARK_SVG",
    "PLOT_ANNOTATION",
    "PLOT_LAYOUT",
    "TOKENS_CSS",
]

# --- Charts ---------------------------------------------------------------

_INK = "#16181a"
_SHEET = "#fbfaf7"
_MUTED = "#5b6167"
_RULE = "#e2ded5"

#: Categorical series colours, for charts that have to separate operations or
#: bottleneck classes. Every one of them clears 5.2:1 against white and 5.0:1
#: against ``--sheet``, so a label sits legibly inside a filled bar as well as
#: beside a line, and the set stays inside the ink-and-earth world rather than
#: reaching for saturated defaults.
CHART_SEQUENCE = (
    "#c23d16",
    "#2f5d7c",
    "#17513a",
    "#8c1d28",
    "#6f6230",
    "#4a5157",
    "#9a4b1f",
    "#35625c",
    "#5c4470",
    "#7a3f52",
)

#: Sequential ramp for intensity, running from paper to the accent and on into
#: a burnt end, so "hot" reads as the accent rather than as an unrelated hue.
HEATMAP_SCALE = (
    (0.0, "#fbfaf7"),
    (0.25, "#efe2d5"),
    (0.5, "#dfae90"),
    (0.75, "#c23d16"),
    (1.0, "#6b1f08"),
)

#: Layout defaults so a Plotly figure sits on the sheet instead of importing
#: its own theme. Applied with ``fig.update_layout(**PLOT_LAYOUT)``.
PLOT_LAYOUT: dict[str, object] = {
    "paper_bgcolor": _SHEET,
    "plot_bgcolor": _SHEET,
    "font": {
        "family": 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
        "size": 12,
        "color": _INK,
    },
    "title": {
        "font": {
            "family": (
                "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
                "'Helvetica Neue', Arial, sans-serif"
            ),
            "size": 15,
            "color": _INK,
        },
        "x": 0,
        "xanchor": "left",
    },
    "colorway": list(CHART_SEQUENCE),
    "xaxis": {
        "gridcolor": _RULE,
        "zerolinecolor": _RULE,
        "linecolor": _INK,
        "ticks": "outside",
        "tickcolor": _RULE,
        "title": {"font": {"size": 11, "color": _MUTED}},
    },
    "yaxis": {
        "gridcolor": _RULE,
        "zerolinecolor": _RULE,
        "linecolor": _INK,
        "ticks": "outside",
        "tickcolor": _RULE,
        "title": {"font": {"size": 11, "color": _MUTED}},
    },
    # A legend boxed on the right eats a third of the plot area in a narrow
    # column. Anchoring it to the figure container, flush left above the
    # chart, matches the eyebrow grammar used elsewhere and stops a wide
    # legend from reserving margin and squeezing the data.
    "legend": {
        "font": {"size": 11},
        "bgcolor": "rgba(0,0,0,0)",
        "orientation": "h",
        "xref": "container",
        "x": 0,
        "xanchor": "left",
        "yref": "container",
        "y": 1,
        "yanchor": "top",
    },
    "hoverlabel": {"bgcolor": _INK, "font": {"color": _SHEET, "size": 12}},
    # Charts are titled by the panel heading above them, so the 100px band
    # Plotly reserves for a chart title is dead space. The base margins are
    # deliberately tight; autoexpand still grows them to fit tick labels and
    # axis titles, so wide layouts are unchanged while a narrow column gets
    # the space back as plot area instead of blank paper.
    "margin": {"t": 56, "l": 48, "r": 24, "b": 48},
}

#: Subplot titles arrive from Plotly as 16px paper annotations, which read as a
#: second heading competing with the panel heading above the chart. Demote them
#: to the same small muted eyebrow the rest of the page uses for labels.
PLOT_ANNOTATION: dict[str, object] = {
    "font": {
        "family": 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace',
        "size": 11,
        "color": _MUTED,
    },
}
