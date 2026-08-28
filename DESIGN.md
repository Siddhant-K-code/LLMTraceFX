---
name: LLMTraceFX
description: Evidence first local LLM and GPU inference optimizer, profiler, and offline report system.
colors:
  field: "#f4f1ea"
  sheet: "#fbfaf7"
  graticule: "#16181a0f"
  ink: "#16181a"
  muted: "#5b6167"
  rule: "#16181a26"
  rule-soft: "#16181a14"
  signal: "#c23d16"
  signal-tint: "#c23d1612"
  verify: "#17513a"
  breach: "#8c1d28"
  hold: "#4a5157"
typography:
  display:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI Variable Text", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    fontSize: "clamp(1.55rem, 1.05rem + 1.9vw, 2.3rem)"
    fontWeight: 600
    lineHeight: 1.13
    letterSpacing: "-0.022em"
  headline:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI Variable Text", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    fontSize: "1.0625rem"
    fontWeight: 600
    lineHeight: 1.55
    letterSpacing: "-0.01em"
  title:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI Variable Text", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    fontSize: "0.8125rem"
    fontWeight: 600
    lineHeight: 1.55
    letterSpacing: "0.08em"
  body:
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI Variable Text", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.55
  label:
    fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", "DejaVu Sans Mono", monospace'
    fontSize: "10.5px"
    fontWeight: 600
    lineHeight: 1.55
    letterSpacing: "0.09em"
  evidence:
    fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", "DejaVu Sans Mono", monospace'
    fontSize: "12.5px"
    fontWeight: 400
    lineHeight: 1.55
rounded:
  none: "0"
spacing:
  graticule: "48px"
  body-x-min: "12px"
  body-y-min: "16px"
  sheet-max: "1180px"
  sheet-pad: "clamp(20px, 4vw, 56px)"
  section-gap: "clamp(34px, 4.5vw, 52px)"
  transect-gap: "48px"
  rule-pad-top: "13px"
  cell-y: "11px"
  cell-x: "14px"
components:
  report-sheet:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "{spacing.sheet-pad}"
  section-rule:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "13px 0 0"
  state-recommended:
    textColor: "{colors.signal}"
    typography: "{typography.label}"
  state-accepted:
    textColor: "{colors.verify}"
    typography: "{typography.label}"
  state-rejected:
    textColor: "{colors.breach}"
    typography: "{typography.label}"
  state-inconclusive:
    textColor: "{colors.hold}"
    typography: "{typography.label}"
  evidence-cell:
    textColor: "{colors.ink}"
    typography: "{typography.evidence}"
    padding: "{spacing.cell-y} {spacing.cell-x} {spacing.cell-y} 0"
---

# Design System: LLMTraceFX

## Overview

**Creative North Star: "Oscilloscope measurement sheet"**

LLMTraceFX records evidence for local LLM and GPU inference work. The visual system must therefore read as measurement, not as promotion. The built world is a laboratory sheet over a bone coloured field. A faint graticule sits behind a paper sheet, data is set in monospace with tabular figures, and structure is made from hairline rules rather than cards.

The primary surface is `llmtracefx/optimizer/tune/report_html.py`, where `DIRECTION_CONTRACT` names the world: oscilloscope graticule and measurement cursors, seed `3bd5d052`. The report is a static audit artifact. It opens from disk, carries all evidence in one HTML file, and keeps accepted, rejected, inconclusive, and excluded material visible. The secondary shell in `llmtracefx/visualize/flame.py` reuses the same field, sheet, lockup, ruled readout, and Plotly theme. The API index in `llmtracefx/api/serve.py` uses the same grammar for a compact route list.

**Key Characteristics:**

- Bone field with a 48px graticule behind a lighter paper sheet.
- Near black ink, grey mute, translucent hairlines, and exactly one authored accent hue.
- System type only: sans for prose, mono for evidence, route paths, identifiers, metrics, and stamps.
- Zero radius on sheets, tables, state marks, pads, and route rows.
- Status is a word plus a mark, never colour alone.
- Offline tune reports have no JavaScript, no web fonts, no external images, and no CDN references.

## Colors

The palette is locked in `llmtracefx/brand.py` as `TOKENS_CSS`. `tests/test_brand_assets.py::test_token_block_defines_every_documented_token` checks that every documented token remains present, and `tests/test_brand_assets.py::test_token_block_fetches_no_fonts` checks that the token block fetches nothing.

### Primary

| Token | Value | Measured contrast | Use |
| --- | --- | --- | --- |
| `--signal` | `#c23d16` | 5.06:1 on sheet, 4.69:1 on field, 5.29:1 with white text | The single accent. Use for the decision pad, section division tick, recommended state, method labels, hover underline, focus outline, and the single marked thing in a region. |
| `--signal-tint` | `#c23d1612` | Transparent tint, not text | Recommended row wash only. It marks a row without adding a second accent. |

### Neutral

| Token | Value | Measured contrast | Use |
| --- | --- | --- | --- |
| `--field` | `#f4f1ea` | 1.08:1 against sheet | Bone page field. It carries the graticule and sits outside the report sheet. |
| `--sheet` | `#fbfaf7` | 17.05:1 under ink | Paper surface for reports, API index, social preview card, and Plotly paper and plot backgrounds. |
| `--graticule` | `#16181a0f` | Transparent line, not text | One pixel graticule line on the field at 48px. |
| `--ink` | `#16181a` | 17.05:1 on sheet, 15.78:1 on field | Body text, main rules, lockup, chart axes, and hover labels. |
| `--muted` | `#5b6167` | 6.00:1 on sheet, 5.56:1 on field | Stamps, captions, labels, ledes, table headers, notes, route descriptions, and secondary text. |
| `--rule` | `#16181a26` | Transparent rule, not text | Main soft borders, sheet border, section seams, table separators, and link underline colour. |
| `--rule-soft` | `#16181a14` | Transparent rule, not text | Secondary seams inside readouts, rows, specs, and route lists. |

### Tertiary

| Token | Value | Measured contrast | Use |
| --- | --- | --- | --- |
| `--verify` | `#17513a` | 8.85:1 on sheet, 8.19:1 on field | Accepted or improvement state. It appears only with a word or label. |
| `--breach` | `#8c1d28` | 8.67:1 on sheet, 8.02:1 on field | Rejected, breached, and excluded count state. It appears with labels and violation text. |
| `--hold` | `#4a5157` | 7.72:1 on sheet, 7.14:1 on field | Inconclusive or no significant difference state. This is graphite by design, not amber. |

### Named Rules

**The One Signal Rule.** Use `--signal` as the only accent hue. Do not add purple AI gradients, neon cyberpunk colours, glass tints, or a second callout hue.

**The Null Reading Rule.** Inconclusive states use `--hold`, a graphite colour, because refusing to guess is a null reading rather than a warning.

**The Tested Contrast Rule.** Text colours must keep the measured contrast ratios above. The chart sequence is covered by `tests/test_brand_assets.py::test_every_series_colour_carries_a_readable_label`, which requires every series colour to hold white text and sheet contrast.

Chart sequence from `CHART_SEQUENCE`:

| Index | Value | White text contrast | Sheet contrast |
| --- | --- | --- | --- |
| 1 | `#c23d16` | 5.29:1 | 5.06:1 |
| 2 | `#2f5d7c` | 7.05:1 | 6.76:1 |
| 3 | `#17513a` | 9.23:1 | 8.85:1 |
| 4 | `#8c1d28` | 9.05:1 | 8.67:1 |
| 5 | `#6f6230` | 6.06:1 | 5.81:1 |
| 6 | `#4a5157` | 8.06:1 | 7.72:1 |
| 7 | `#9a4b1f` | 6.18:1 | 5.92:1 |
| 8 | `#35625c` | 6.88:1 | 6.59:1 |
| 9 | `#5c4470` | 8.35:1 | 8.00:1 |
| 10 | `#7a3f52` | 7.89:1 | 7.56:1 |

Heatmap scale from `HEATMAP_SCALE`: `0.0 #fbfaf7`, `0.25 #efe2d5`, `0.5 #dfae90`, `0.75 #c23d16`, `1.0 #6b1f08`.

## Typography

**Display Font:** system sans via `--sans`.
**Body Font:** system sans via `--sans`.
**Label/Mono Font:** `ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", "DejaVu Sans Mono", monospace` via `--mono`.

**Character:** Prose is quiet and system native. Evidence is mechanical: mono, tabular, wrapped rather than clipped, and never converted into decorative numerals.

### Hierarchy

- **Display** (600, `clamp(1.55rem, 1.05rem + 1.9vw, 2.3rem)`, 1.13, `-0.022em`): report title in the tune report. Secondary shells use the close value `clamp(1.5rem, 1.1rem + 1.6vw, 2.1rem)`.
- **Headline** (600, `1.0625rem`, 1.55, `-0.01em`): `h2` section titles and panel titles.
- **Title** (600, `0.8125rem`, uppercase, `0.08em`): `h3` labels for local subsections.
- **Small label** (600, `10.5px`, uppercase, `0.09em`): stamps, state chips, metric labels, table headers, details summaries, and footer text.
- **Body** (400, `15px`, 1.55): explanatory prose. Paragraphs max at 72ch, ledes max at 68ch.
- **Evidence** (400, `12.5px` to `13.5px`, tabular figures): specs, code, paths, run IDs, metric cells, route paths, and cursor readings.
- **Readout value** (mono, `1.375rem`, 1.1, `-0.015em`): the divisional summary values in report and performance shells.

### Named Rules

**The Evidence Mono Rule.** Every identifier, artifact path, metric, route path, candidate label, and cursor reading uses `--mono` with tabular figures where numbers are compared.

**The System Type Rule.** Do not fetch web fonts. The offline report rule and `tests/test_brand_assets.py::test_token_block_fetches_no_fonts` require system stacks only.

## Layout

The page model is a sheet on a field. The body uses `background-color: var(--field)` plus two repeating linear gradients at 48px to form a one pixel graticule. The tune report body padding is `clamp(16px, 3vw, 44px) clamp(12px, 3vw, 40px) 72px`. The sheet is `max-width: 1180px`, centered, with `padding: clamp(20px, 4vw, 56px)`. The API index uses a narrower `max-width: 760px`. The social preview uses a 64px graticule for the fixed 1280 by 640 image only.

Sections are stations on an evidence transect. `section` spacing is `clamp(34px, 4.5vw, 52px)`. The in-page contents nav appears after `48px`, uses ruled rows, and marks recommended and inconclusive groups with the same square grammar as state chips. Summary counts sit in one ruled strip, `repeat(auto-fit, minmax(132px, 1fr))`, with vertical division ticks.

At `max-width: 760px`, reflowing tables become stacked labelled records. The CSS sets table parts to `display: block`, so the HTML declares explicit `role="table"`, `role="rowgroup"`, `role="row"`, `role="columnheader"`, `role="rowheader"`, and `role="cell"`. Each cell carries `data-label`. This is enforced by `tests/optimizer/test_tune_report_html.py::test_tables_declare_explicit_aria_roles` and `tests/optimizer/test_tune_report_html.py::test_measurement_cells_carry_their_label_for_the_narrow_reflow`.

At `max-width: 520px`, the API route index collapses endpoint rows from a method column plus description column to one column.

## Elevation & Depth

Depth is minimal. The system is ruled and layered, not carded. The only persistent shadow is the sheet lift: `0 1px 1px #16181a0f, 0 26px 52px -30px #16181a5c` in the report, API index, and GPU performance shell. The social preview uses the same idea at image scale: `0 1px 1px #16181a0f, 0 30px 60px -34px #16181a66`. Plotly figures are themed onto the sheet with `PLOT_LAYOUT`, using `paper_bgcolor` and `plot_bgcolor` set to `#fbfaf7`.

### Chart Titling and Frame

Charts carry no figure title of their own. The panel heading in the surrounding page names the chart, so `PLOT_LAYOUT` reclaims the roughly 100px band the library reserves for a title it will never draw. The base margins are deliberately tight at `t: 56, l: 48, r: 24, b: 48`. Plotly's `autoexpand` still grows any side that genuinely needs room, so a wide layout keeps the space its tick labels and axis titles require while a narrow column spends the difference on plot area instead of blank paper.

The legend runs horizontally, flush left, anchored to the figure container rather than the plot area. A legend boxed on the right takes a third of the width out of a narrow column, and even a top legend anchored to the plot reserves margin when it is wider than the plot itself. Container anchoring matches the flush-left eyebrow grammar used elsewhere and leaves the data the full frame.

Subplot titles are the one titling exception, because they distinguish stacked panes inside a single figure. They do not arrive through `layout`, so the layout theme does not reach them and they would render at 16px, reading as a second heading competing with the panel heading above. `PLOT_ANNOTATION` demotes them to the same 11px muted mono eyebrow the rest of the system uses for labels, and the strings themselves are written uppercase to match that grammar. Apply it wherever `PLOT_LAYOUT` is applied.

### Named Rules

**The Ruled Not Carded Rule.** Use hairline borders, table rules, division ticks, and sheet lift. Do not wrap every statistic or table in separate cards.

**The No Paint Gradient Rule.** Gradients are structural only: graticules and excluded-run diagonal ruling. Do not use gradients as brand paint.

## Shapes

The form language is square, routed, and measured. Radius is zero across the system. Sheets, tables, state marks, decision pads, route rows, and ornaments all use hard corners. This keeps the report in the world of lab paper, routed traces, and instrument readouts rather than app cards.

Borders are hairlines. Main section rules use `1px solid var(--ink)`, secondary separators use `1px solid var(--rule)`, and internal seams use `1px solid var(--rule-soft)`. Section rules include a `2px` by `7px` signal tick at the start. Readout divisions include a `1px` by `6px` ink tick. Rejected violations use an `11px` by `2px` breach mark.

### Named Rules

**The Zero Radius Rule.** Do not add rounded corners. A radius would contradict the measurement sheet, decision pad, route geometry, and test-backed mark system.

## Components

### Logo system

The logo is two converging routed traces, one heavy and one hairline, terminating in a solid decision pad. It means trace, evidence, and resolved decision. It is not a brain, sparkle, robot, or waveform.

- `assets/brand/llmtracefx-lockup.svg`: primary horizontal lockup for light grounds. Use on sheet, README header, and report mastheads.
- `assets/brand/llmtracefx-lockup-inverse.svg`: inverse horizontal lockup for ink or other dark grounds. It draws in bone and keeps the signal pad.
- `assets/brand/llmtracefx-lockup-mono.svg`: one colour lockup for stamping, print, or contexts that must inherit `currentColor`.
- `assets/brand/llmtracefx-mark.svg`: symbol only for tight spaces at normal size.
- `assets/brand/llmtracefx-mark-mono.svg`: one colour symbol that inherits `currentColor`.
- `assets/brand/llmtracefx-wordmark.svg`: routed wordmark alone, use only when the symbol is already present nearby.
- `assets/brand/llmtracefx-icon.svg`: favicon and 16px chrome reduction. It drops the hairline trace, as enforced by `tests/test_brand_assets.py::test_icon_drops_the_hairline_trace`.

Committed SVG assets keep `xmlns` because standalone SVG files need it. Inline report constants `MARK_SVG` and `LOCKUP_SVG` omit `xmlns` so the generated HTML contains no `http://` substring. This is enforced by `tests/test_brand_assets.py::test_inline_marks_reach_nothing_outside_the_document` and `tests/optimizer/test_tune_report_html.py::test_inline_lockup_carries_an_accessible_name`.

### Ornament

`ORNAMENT_OPEN_SVG` and `ORNAMENT_CLOSED_SVG` in `report_html.py` reuse the mark at diagram scale. The open pad appears when no comparable groups exist. The filled pad terminates every report footer. It may anchor an empty measurement or close a completed sheet. Do not use it as decoration between unrelated blocks.

### Report sheet

The sheet is a single paper surface on the graticule field. It has one border, one shadow, no radius, and one masthead rule. The masthead pairs the lockup with a mono stamp showing report type, generated time, schema, and path redaction state.

### Ruled section

Each section starts with a top ink rule and a signal tick. This is the main rhythm for policy, summary, comparable groups, provenance, and secondary shells. It replaces panels or cards.

### Divisional readout

The readout is a single strip with equal divisions. Values are mono, tabular, and large. Labels are uppercase small sans. Counts use state text colour only when the label already states the meaning.

### State chip

State chips are inline flex labels in mono uppercase text with a square or open square mark. `state-rec` uses signal, `state-ok` uses verify, `state-bad` uses breach, and `state-hold` uses hold. `tests/optimizer/test_tune_report_html.py::test_states_are_never_signalled_by_colour_alone` requires the word to be present.

### Evidence table

Accepted candidate tables are reflowing evidence records. Candidate identity heads each row group so long labels are not squeezed into a narrow column. Measurement cells are right aligned on wide screens, stacked and labelled on narrow screens, and always preserve `data-label`. Captions and scoped headers are enforced by `tests/optimizer/test_tune_report_html.py::test_tables_have_captions_and_scoped_headers`.

### Details disclosure

The only interactive affordance in the tune report is native `<details>` and `<summary>` for source run IDs and artifact paths. The chevron rotates from `-45deg` to `45deg` over `180ms cubic-bezier(0.2, 0.8, 0.2, 1)`. `@media (prefers-reduced-motion: reduce)` sets transitions to `0.01ms`.

### API route index

The API root is a ruled route list, not a stack of boxes. Methods use mono uppercase text in signal. Paths use mono at `0.9375rem`. Descriptions sit in muted prose.

### Print behavior

The `@media print` block removes the field graticule by setting the body background to white, removes body padding, sets body type to `10.5pt`, removes sheet border, shadow, and padding, hides the skip link and transect nav, removes link underlines, and applies break control to rows, rejects, cursors, divisions, candidate groups, headings, and rules. It also tries to expand details content with `details::details-content { content-visibility: visible; }` and labels closed disclosures with ` (collapsed on screen)` where expansion is unsupported.

### Offline, escaping, and determinism

The tune report is a binding static artifact. It must remain one self-contained HTML file with no network and no JavaScript. `tests/optimizer/test_tune_report_html.py::test_no_network_or_cdn_references_in_output` forbids `http://`, `https://`, `<script`, `cdn.`, and `googleapis` in report output. That bans web fonts, CDN assets, JSON-LD, and SVG data URI favicons.

Rendering must be byte identical for the same `TuneReport`, enforced by `tests/optimizer/test_tune_report_html.py::test_render_is_byte_identical_across_calls`. The only clock is `report.generated_at`, covered by `tests/optimizer/test_tune_report_html.py::test_render_has_no_new_timestamp_beyond_generated_at`. Every report-derived string passes through `html.escape(..., quote=True)`, covered by the escaping tests in `tests/optimizer/test_tune_report_html.py`. Paths are redacted by default, covered by the path redaction tests in that file.

The older Plotly visualization methods in `llmtracefx/visualize/flame.py` call `to_html(include_plotlyjs='cdn')`. Treat that shell as a secondary visualization surface, not as the offline tune report artifact. It already runs the library's own JavaScript, so it is the one surface allowed a script: charts are plotted with `{"responsive": True}` and the shell fires a single resize on load, because a chart inside `.chart-grid` can otherwise be measured before the grid resolves its track width and stay drawn wider than the column that holds it. The offline tune report remains scriptless.

## Do's and Don'ts

### Do:

- **Do** use `llmtracefx/brand.py` as the single source of truth for tokens, inline marks, chart sequence, heatmap scale, and Plotly layout.
- **Do** keep the report in the oscilloscope and logic analyzer measurement sheet world recorded in `DIRECTION_CONTRACT`.
- **Do** use the 48px graticule on the field and the paper sheet on top of it.
- **Do** use mono with tabular figures for all evidence, identifiers, paths, routes, metrics, stamps, and cursor readings.
- **Do** keep rejected, inconclusive, excluded, and provenance material visible and labelled.
- **Do** preserve explicit ARIA table roles and `data-label` attributes on any table that reflows at `max-width: 760px`.
- **Do** pair every status colour with an uppercase word or visible label.
- **Do** use native `<details>` for disclosure in the offline report.
- **Do** use the primary lockup on light sheets, the inverse lockup on dark grounds, mono lockups for one colour reproduction, and the icon only at favicon scale.
- **Do** keep report output deterministic, escaped, path redacted by default, and free of network references.

### Don't:

- **Don't** add rounded corners, pill chips, card stacks, glass effects, or paint gradients.
- **Don't** introduce a second accent hue or generic AI purple.
- **Don't** fetch fonts or assets in the tune report.
- **Don't** add JavaScript to the tune report. The only interaction is native disclosure.
- **Don't** encode status by colour alone.
- **Don't** add `xmlns` to inline report SVG constants, because that introduces a forbidden `http://` string.
- **Don't** clip candidate labels, run IDs, paths, or metric cells. Wrap them and provide labels on narrow screens.
- **Don't** use the diagram ornament as filler decoration. Use open for no decision and filled for a completed sheet terminus.
