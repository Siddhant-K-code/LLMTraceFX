"""Offline deterministic HTML/SVG rendering for cache-audit evidence."""

from __future__ import annotations

import html
from collections.abc import Sequence

from .report import build_claim_matrix
from .schema import AuditManifest, RequestEvidence


def _esc(value: object) -> str:
    return html.escape("unavailable" if value is None else str(value), quote=True)


def _fact_cell(fact: dict[str, object]) -> str:
    limitations = fact["limitations"]
    assert isinstance(limitations, list)
    detail = (
        f"basis={_esc(fact['basis'])}; source={_esc(fact['source'])}; "
        f"scope={_esc(fact['scope'])}"
    )
    if limitations:
        detail += f"; limitations={_esc(', '.join(str(item) for item in limitations))}"
    return f"{_esc(fact['value'])}<br><small>{detail}</small>"


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
        expected_available = isinstance(expected, int)
        attested_available = isinstance(attested, int)
        expected_width = max(0, expected or 0) * scale if expected_available else 640
        attested_width = max(0, attested or 0) * scale if attested_available else 640
        bars.extend(
            (
                f'<text x="8" y="{y + 13}" font-size="12">'
                f"{_esc(record.spec.request_id)}</text>",
                f'<rect x="260" y="{y}" width="{expected_width:.2f}" height="10" '
                f'fill="{"#2563eb" if expected_available else "url(#unavailable)"}"/>',
                f'<rect x="260" y="{y + 13}" width="{attested_width:.2f}" '
                f'height="10" fill="{"#0f766e" if attested_available else "url(#unavailable)"}"/>',
                (
                    ""
                    if expected_available and attested_available
                    else f'<text x="905" y="{y + 13}" font-size="11">unavailable</text>'
                ),
            )
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Expected and engine-attested reusable tokens">'
        '<defs><pattern id="unavailable" width="8" height="8" '
        'patternUnits="userSpaceOnUse"><path d="M0 8L8 0" '
        'stroke="#94a3b8" stroke-width="2"/></pattern></defs>'
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
    if manifest.publication_mode.value == "public_redacted":
        evidence_label = "Redacted cache-audit evidence"
    elif manifest.backend == "synthetic_reference":
        evidence_label = "Synthetic reference cache evidence"
    else:
        evidence_label = f"{manifest.backend} runtime cache evidence"
    rows = []
    statements = []
    evidence_details = []
    for row in matrix["rows"]:
        rows.append(
            "<tr>"
            f"<td>{_esc(row['request_id'])}</td>"
            f"<td>{_esc(row['scenario'])}</td>"
            f"<td>{_fact_cell(row['expected_policy_reuse'])}</td>"
            f"<td>{_fact_cell(row['engine_cached'])}</td>"
            f"<td>{_fact_cell(row['observed_prompt_work'])}</td>"
            f"<td>{_esc(row['claim_eligibility']['output_equivalence'])}</td>"
            f"<td>{_esc(row['claim_eligibility']['quality'])}</td>"
            f"<td>{_esc(row['cache_reuse_verdict'])}<br>"
            f"<small>{_esc(', '.join(row['verdict_reasons']))}</small></td>"
            "</tr>"
        )
        statements.append(f"<li>{_esc(row['statement'])}</li>")
        fact_rows = []
        for label, key in (
            ("Semantic prefix", "semantic_prefix"),
            ("Expected policy reuse", "expected_policy_reuse"),
            ("Engine cached", "engine_cached"),
            ("Observed prompt work", "observed_prompt_work"),
            ("Unexpected recomputation", "unexpected_recomputed"),
            ("Logical cache memory", "logical_cache_memory"),
            ("Runtime peak memory", "runtime_peak_memory"),
            ("Output identity", "output_identity"),
            ("Deterministic correctness", "correctness"),
        ):
            fact_rows.append(
                f"<tr><th>{_esc(label)}</th><td>{_fact_cell(row[key])}</td></tr>"
            )
        record_limitations = "; ".join(
            f"{item['code']}: {item['message']}" for item in row["limitations"]
        )
        eligibility_reasons = ", ".join(row["claim_eligibility"]["reasons"])
        timing = row["timing_evidence"]
        evidence_details.append(
            f"<details><summary>{_esc(row['request_id'])} evidence details</summary>"
            "<table><tbody>"
            + "".join(fact_rows)
            + f"<tr><th>Timing basis</th><td>scope={_esc(row['timing_scope'])}; "
            f"exclusions={_esc(', '.join(row['timing_exclusions']))}; "
            f"client TTFT={_esc(timing['client_ttft']['value_ms'])} ms "
            f"({_esc(timing['client_ttft']['basis'])}); "
            "in-process first token="
            f"{_esc(timing['in_process_first_token']['value_ms'])} ms "
            f"({_esc(timing['in_process_first_token']['basis'])})</td></tr>"
            f"<tr><th>Verdict reasons</th><td>{_esc(', '.join(row['verdict_reasons']))}</td></tr>"
            f"<tr><th>Eligibility reasons</th><td>{_esc(eligibility_reasons)}</td></tr>"
            f"<tr><th>Record limitations</th><td>{_esc(record_limitations)}</td></tr>"
            "</tbody></table></details>"
        )
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>LLMTraceFX {_esc(evidence_label)}</title>"
        "<style>"
        "body{font:15px system-ui,sans-serif;max-width:1100px;margin:32px auto;"
        "padding:0 20px;color:#172033;background:#fff}"
        "h1{font-size:30px}table{width:100%;border-collapse:collapse}"
        "th,td{padding:8px;border-bottom:1px solid #d8dee9;text-align:left}"
        "th{background:#f3f6fa}.meta{color:#526070}.claim{line-height:1.5}"
        "</style></head><body>"
        f"<h1>{_esc(evidence_label)}</h1>"
        f'<p class="meta">Run {_esc(manifest.run_id)} · backend '
        f"{_esc(manifest.backend)} {_esc(manifest.backend_version)} · "
        f"privacy mode {_esc(manifest.publication_mode.value)}</p>"
        f'<p class="meta">Evidence captured {_esc(manifest.created_at)} · '
        f"implementation bound/generated {_esc(manifest.generated_at)}</p>"
        f'<p class="meta">Evidence label: {_esc(evidence_label)}. '
        + (
            "Synthetic evidence validates arithmetic and verdict wiring only; "
            "it does not establish runtime compute avoidance, model quality, "
            "latency, or memory effects. "
            if manifest.backend == "synthetic_reference"
            else "Evidence claims are limited to the displayed bases and scopes. "
        )
        + "Unavailable values are hatched or labeled unavailable, never rendered as zero.</p>"
        "<p>A cache counter is evidence, not proof of skipped work, latency, "
        "memory savings, or correctness.</p>"
        "<h2>Per-request truth table</h2>"
        "<table><thead><tr><th>Request</th><th>Scenario</th><th>Expected reuse</th>"
        "<th>Engine cached</th><th>Prompt work</th><th>Output equivalence</th>"
        "<th>Model quality</th>"
        "<th>Verdict</th></tr></thead><tbody>"
        + "".join(rows)
        + '</tbody></table><h2>Claims</h2><ol class="claim">'
        + "".join(statements)
        + "</ol><h2>Evidence basis, scope, and limitations</h2>"
        + "".join(evidence_details)
        + "<h2>Limitations and eligibility</h2><p>"
        + _esc(
            "; ".join(
                sorted(
                    {
                        reason
                        for row in matrix["rows"]
                        for reason in row["claim_eligibility"]["reasons"]
                    }
                )
            )
        )
        + "</p></body></html>\n"
    )
