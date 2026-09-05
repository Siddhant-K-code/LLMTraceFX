"""Offline deterministic HTML/SVG rendering for cache-audit evidence."""

from __future__ import annotations

import html
from collections.abc import Sequence

from .report import build_claim_matrix
from .schema import AuditManifest, RequestEvidence


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def render_reuse_alignment_svg(records: Sequence[RequestEvidence]) -> str:
    width = 960
    row_height = 34
    height = 52 + row_height * len(records)
    bars: list[str] = []
    max_tokens = max((record.spec.input_token_count for record in records), default=1)
    scale = 640 / max_tokens
    for index, record in enumerate(records):
        y = 34 + index * row_height
        expected = record.reuse.policy_reusable_tokens.value
        attested = record.reuse.engine_cached_tokens.value
        expected_width = (
            0 if not isinstance(expected, int) else max(0, expected) * scale
        )
        attested_width = (
            0 if not isinstance(attested, int) else max(0, attested) * scale
        )
        bars.extend(
            (
                f'<text x="8" y="{y + 13}" font-size="12">'
                f"{_esc(record.spec.request_id)}</text>",
                f'<rect x="260" y="{y}" width="{expected_width:.2f}" height="10" '
                'fill="#2563eb"/>',
                f'<rect x="260" y="{y + 13}" width="{attested_width:.2f}" '
                'height="10" fill="#0f766e"/>',
            )
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Expected and engine-attested reusable tokens">'
        '<rect width="100%" height="100%" fill="#ffffff"/>'
        '<text x="260" y="16" font-size="12" fill="#2563eb">expected</text>'
        '<text x="340" y="16" font-size="12" fill="#0f766e">engine-attested</text>'
        + "".join(bars)
        + "</svg>\n"
    )


def render_html(
    manifest: AuditManifest,
    records: Sequence[RequestEvidence],
) -> str:
    matrix = build_claim_matrix(records)
    rows = []
    statements = []
    for row in matrix["rows"]:
        rows.append(
            "<tr>"
            f"<td>{_esc(row['request_id'])}</td>"
            f"<td>{_esc(row['scenario'])}</td>"
            f"<td>{_esc(row['expected_policy_reusable_tokens'])}</td>"
            f"<td>{_esc(row['engine_cached_tokens'])}</td>"
            f"<td>{_esc(row['observed_prompt_tokens'])}</td>"
            f"<td>{_esc(row['correctness'])}</td>"
            f"<td>{_esc(row['verdict'])}</td>"
            "</tr>"
        )
        statements.append(f"<li>{_esc(row['statement'])}</li>")
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>LLMTraceFX cache audit</title>"
        "<style>"
        "body{font:15px system-ui,sans-serif;max-width:1100px;margin:32px auto;"
        "padding:0 20px;color:#172033;background:#fff}"
        "h1{font-size:30px}table{width:100%;border-collapse:collapse}"
        "th,td{padding:8px;border-bottom:1px solid #d8dee9;text-align:left}"
        "th{background:#f3f6fa}.meta{color:#526070}.claim{line-height:1.5}"
        "</style></head><body>"
        "<h1>KV-cache truth audit</h1>"
        f'<p class="meta">Run {_esc(manifest.run_id)} · backend '
        f"{_esc(manifest.backend)} {_esc(manifest.backend_version)} · "
        f"publication {_esc(manifest.publication_mode.value)}</p>"
        "<p>A cache counter is evidence, not proof of skipped work, latency, "
        "memory savings, or correctness.</p>"
        "<h2>Per-request truth table</h2>"
        "<table><thead><tr><th>Request</th><th>Scenario</th><th>Expected reuse</th>"
        "<th>Engine cached</th><th>Prompt work</th><th>Correct</th>"
        "<th>Verdict</th></tr></thead><tbody>"
        + "".join(rows)
        + '</tbody></table><h2>Claims</h2><ol class="claim">'
        + "".join(statements)
        + "</ol></body></html>\n"
    )
