"""Tests for the ``compare`` HTML renderer.

The renderer is a read-only surface over a validated report, so the tests
here are about the properties that make a rendered file safe to share:
determinism, escaping, path redaction, no network references, and enough
structure for a screen reader and a narrow viewport.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
from _compare_fixtures import write_api_run, write_run

from llmtracefx.optimizer.compare.compare import compare
from llmtracefx.optimizer.compare.identity import ComparableUnitKey, SystemKey
from llmtracefx.optimizer.compare.policy import (
    CompareConstraints,
    CompareObjective,
    ComparePolicy,
)
from llmtracefx.optimizer.compare.pricing import PricingManifest
from llmtracefx.optimizer.compare.report import (
    COMPARE_REPORT_SCHEMA_VERSION,
    CompareReport,
    CostSummary,
    PricingProvenance,
    StratumOutcome,
    StratumReport,
    SystemReport,
    UsageTotals,
)
from llmtracefx.optimizer.compare.report_html import (
    _system_label,
    render_compare_report_html,
)

_PRICING = PricingManifest.from_dict(
    {
        "schema_version": "1",
        "currency": "USD",
        "entries": [
            {
                "entry_id": "glm-5.3",
                "provider": "z-ai",
                "model_id": "glm-5.3",
                "currency": "USD",
                "effective_at": "2026-01-01",
                "source": "illustrative example",
                "rates_are_illustrative": True,
                "input_per_million": 1.0,
                "output_per_million": 2.0,
            },
            {
                "entry_id": "glm-5.3-flash",
                "provider": "z-ai",
                "model_id": "glm-5.3-flash",
                "currency": "USD",
                "effective_at": "2026-01-01",
                "source": "illustrative example",
                "rates_are_illustrative": True,
                "input_per_million": 0.1,
                "output_per_million": 0.2,
            },
        ],
    }
)


def _policy(**overrides: Any) -> ComparePolicy:
    return ComparePolicy(
        objective=CompareObjective.MIN_MEAN_TOTAL_LATENCY_MS,
        name="Local vs frontier vs flash",
        constraints=CompareConstraints(**overrides),
    )


def _built_report(tmp_path: Path, **compare_kwargs: Any) -> CompareReport:
    write_run(tmp_path, "local-1", total_ms=8000.0)
    write_api_run(tmp_path, "frontier-1", model_id="glm-5.3", total_ms=3000.0)
    write_api_run(
        tmp_path,
        "flash-1",
        model_id="glm-5.3-flash",
        reasoning_effort="low",
        total_ms=1200.0,
    )
    return compare(results_dirs=(tmp_path,), policy=_policy(), **compare_kwargs)


# --- Determinism ----------------------------------------------------------


def test_rendering_the_same_report_twice_is_byte_identical(tmp_path: Path) -> None:
    report = _built_report(tmp_path)
    assert render_compare_report_html(report) == render_compare_report_html(report)


def test_rendering_does_not_take_its_own_timestamp(tmp_path: Path) -> None:
    report = _built_report(tmp_path)
    document = render_compare_report_html(report)
    assert report.generated_at in document
    # The report's own clock is the only timestamp in the document.
    timestamps = set(re.findall(r"\d{4}-\d{2}-\d{2}T[\d:.]+Z", document))
    assert timestamps == {report.generated_at}


def test_a_reloaded_report_renders_identically(tmp_path: Path) -> None:
    report = _built_report(tmp_path)
    reloaded = CompareReport.from_json(report.to_json())
    assert render_compare_report_html(reloaded) == render_compare_report_html(report)


# --- No network -----------------------------------------------------------


def test_the_document_has_no_script_or_external_reference(tmp_path: Path) -> None:
    document = render_compare_report_html(_built_report(tmp_path))
    lowered = document.lower()
    for forbidden in (
        "<script",
        "javascript:",
        "<iframe",
        "<link ",
        "@import",
        "srcset=",
    ):
        assert forbidden not in lowered
    assert "http://" not in lowered
    assert "https://" not in lowered


def test_the_document_declares_itself_offline_and_unindexed(tmp_path: Path) -> None:
    document = render_compare_report_html(_built_report(tmp_path))
    assert '<meta name="robots" content="noindex, nofollow">' in document
    assert "no external references" in document


# --- Escaping -------------------------------------------------------------


def _hostile_report() -> CompareReport:
    hostile = "<script>alert('x')</script>"
    system_key = SystemKey(
        model_id=hostile,
        model_revision=None,
        provider="z-ai",
        runtime_name=hostile,
        runtime_backend=None,
        accelerator=None,
        quantization=hostile,
        reasoning_effort=hostile,
        decode_mode=hostile,
    )
    system = SystemReport(
        system_key=system_key,
        rank=1,
        run_ids=(hostile,),
        verification_paths=(f"/tmp/{hostile}/verification.json",),
        record_paths=(f"/tmp/{hostile}/final_record.json",),
        evidence_count=1,
        objective_name=CompareObjective.MIN_MEAN_TOTAL_LATENCY_MS.value,
        objective_value=1.0,
        quality_metric=hostile,
        missing_evidence=(hostile,),
        usage=UsageTotals(runs_reporting_usage=1, runs_total=1, input_tokens=1),
        cost=CostSummary(
            currency="USD",
            pricing_entry_id=hostile,
            pricing_entry_sha256=hostile,
            rates_are_illustrative=True,
            total_amount=1.0,
            reasons=(hostile,),
        ),
    )
    return CompareReport(
        schema_version=COMPARE_REPORT_SCHEMA_VERSION,
        generated_at="2026-01-01T00:00:00.000000Z",
        results_dirs=(f"/tmp/{hostile}",),
        policy=ComparePolicy(
            objective=CompareObjective.MIN_MEAN_TOTAL_LATENCY_MS, name=hostile
        ),
        pricing=PricingProvenance(
            manifest_path=f"/tmp/{hostile}/rates.json",
            manifest_sha256=hostile,
            currency="USD",
            rates_are_illustrative=True,
            entry_ids_used=(hostile,),
        ),
        strata=(
            StratumReport(
                unit_key=ComparableUnitKey(
                    workload_id=hostile,
                    workload_version="1",
                    workload_prompt_hash=hostile,
                    context_tier=hostile,
                    quality_metric=hostile,
                    max_output_tokens=None,
                    temperature=None,
                    top_p=None,
                ),
                outcome=StratumOutcome.RECOMMENDED,
                objective_name=CompareObjective.MIN_MEAN_TOTAL_LATENCY_MS.value,
                ranked=(system,),
                recommended=system,
                missing_evidence=(hostile,),
            ),
        ),
    )


def test_hostile_strings_never_reach_the_document_unescaped() -> None:
    document = render_compare_report_html(_hostile_report(), redact_paths=False)
    assert "<script>alert" not in document
    assert "&lt;script&gt;alert" in document


def test_a_hostile_currency_cannot_inject_markup_through_a_money_cell() -> None:
    """The money formatter interpolates the currency, so the cell must escape.

    ``_fmt_money`` builds ``"<amount> <currency>"`` from report JSON, and that
    string lands in the ranked table. Escaping only at the formatters would
    leave this one path open, so the table escapes at the cell boundary.
    """
    hostile = "<img src=x onerror=alert(1)>"
    system = SystemReport(
        system_key=SystemKey(
            model_id="glm-5.3",
            model_revision=None,
            provider="z-ai",
            runtime_name="openai-compatible-stream",
            runtime_backend=None,
            accelerator=None,
            quantization=None,
            reasoning_effort=None,
            decode_mode="autoregressive",
        ),
        rank=1,
        run_ids=("r1",),
        verification_paths=(),
        record_paths=(),
        evidence_count=1,
        objective_name=CompareObjective.MIN_MEAN_TOTAL_LATENCY_MS.value,
        objective_value=1.0,
        quality_metric=hostile,
        cost=CostSummary(
            currency=hostile,
            pricing_entry_id="e",
            pricing_entry_sha256="h",
            rates_are_illustrative=True,
            total_amount=1.0,
            cost_per_correct_case=1.0,
        ),
    )
    report = CompareReport(
        schema_version=COMPARE_REPORT_SCHEMA_VERSION,
        generated_at="2026-01-01T00:00:00.000000Z",
        results_dirs=("/tmp/results",),
        policy=ComparePolicy(
            objective=CompareObjective.MIN_MEAN_TOTAL_LATENCY_MS, name="p"
        ),
        pricing=PricingProvenance(
            manifest_path="rates.json",
            manifest_sha256="h",
            currency=hostile,
            rates_are_illustrative=True,
        ),
        strata=(
            StratumReport(
                unit_key=ComparableUnitKey(
                    workload_id="w",
                    workload_version="1",
                    workload_prompt_hash="sha256:abc",
                    context_tier="2k",
                    quality_metric=hostile,
                    max_output_tokens=None,
                    temperature=None,
                    top_p=None,
                ),
                outcome=StratumOutcome.RECOMMENDED,
                objective_name=CompareObjective.MIN_MEAN_TOTAL_LATENCY_MS.value,
                ranked=(system,),
                recommended=system,
            ),
        ),
    )
    document = render_compare_report_html(report)
    assert "<img src=x" not in document
    assert "&lt;img src=x onerror=alert(1)&gt;" in document


def test_the_meta_description_is_escaped() -> None:
    document = render_compare_report_html(_hostile_report())
    description = re.search(r'<meta name="description" content="([^"]*)"', document)
    assert description is not None
    assert "<" not in description.group(1)


def test_the_title_is_escaped() -> None:
    document = render_compare_report_html(_hostile_report())
    title = re.search(r"<title>(.*?)</title>", document, re.S)
    assert title is not None
    assert "<script" not in title.group(1)


# --- Path redaction -------------------------------------------------------


def test_paths_are_redacted_by_default(tmp_path: Path) -> None:
    report = _built_report(tmp_path)
    document = render_compare_report_html(report)
    assert str(tmp_path) not in document
    assert "runs/local-1/verification.json" in document
    assert "paths redacted" in document


def test_include_paths_emits_the_full_path(tmp_path: Path) -> None:
    report = _built_report(tmp_path)
    document = render_compare_report_html(report, redact_paths=False)
    assert str(tmp_path / "runs" / "local-1" / "verification.json") in document
    assert "full paths included" in document


def test_the_pricing_manifest_path_is_redacted_too(tmp_path: Path) -> None:
    manifest_path = str(tmp_path / "private" / "rates.json")
    report = _built_report(
        tmp_path, pricing=_PRICING, pricing_manifest_path=manifest_path
    )
    document = render_compare_report_html(report)
    assert manifest_path not in document
    assert "rates.json" in document


def test_no_prompt_or_reasoning_text_can_appear(tmp_path: Path) -> None:
    """The report schema simply has no field that could carry one."""
    report = _built_report(tmp_path)
    payload = report.to_dict()
    forbidden = ("prompt_text", "response", "reasoning_content", "messages")
    assert not any(key in str(payload) for key in forbidden)


# --- Content --------------------------------------------------------------


def test_every_system_label_appears(tmp_path: Path) -> None:
    report = _built_report(tmp_path)
    document = render_compare_report_html(report)
    for system in report.strata[0].ranked:
        rendered = _system_label(system.system_key.label(), redact_paths=True)
        assert rendered.replace("&", "&amp;") in document


def test_the_deployment_host_is_redacted_by_default(tmp_path: Path) -> None:
    """An endpoint host can name a private service, so it is not shared."""
    document = render_compare_report_html(_built_report(tmp_path))
    assert "example.invalid" not in document
    # The route is kept: it is what tells two deployments apart, and it
    # names no host.
    assert "endpoint=/v1/chat/completions" in document


def test_include_paths_shows_the_full_endpoint(tmp_path: Path) -> None:
    document = render_compare_report_html(_built_report(tmp_path), redact_paths=False)
    assert "https://example.invalid/v1/chat/completions" in document


def test_endpoint_redaction_reaches_the_frontier_dominated_by_entries() -> None:
    """A host must not survive in one corner after redaction elsewhere."""
    label = (
        "m via p [rt/unknown] quant=unrecorded reasoning=high decode=autoregressive "
        "endpoint=https://private.internal/v1/chat"
    )
    assert _system_label(label, redact_paths=True).endswith("endpoint=/v1/chat")
    assert "private.internal" not in _system_label(label, redact_paths=True)
    assert _system_label(label, redact_paths=False) == label


def test_a_label_without_an_endpoint_is_untouched() -> None:
    label = "local/qwen3-8b via Apple M5 Pro [mlx-lm/Metal] decode=autoregressive"
    assert _system_label(label, redact_paths=True) == label


def test_the_recommendation_states_its_scope(tmp_path: Path) -> None:
    report = _built_report(tmp_path)
    document = render_compare_report_html(report)
    assert "at context tier" in document
    assert "and only for that" in document
    assert "no universal winner" in document


def test_the_ttft_basis_is_spelled_out(tmp_path: Path) -> None:
    report = _built_report(tmp_path)
    document = render_compare_report_html(report)
    assert "local prefill" in document
    assert "includes network transport" in document


def test_illustrative_rates_are_labeled_as_examples(tmp_path: Path) -> None:
    report = _built_report(
        tmp_path, pricing=_PRICING, pricing_manifest_path="rates.json"
    )
    document = render_compare_report_html(report)
    assert "Illustrative rates" in document
    assert "none of it is a quotation" in document


def test_a_report_without_pricing_says_so(tmp_path: Path) -> None:
    document = render_compare_report_html(_built_report(tmp_path))
    assert "no monetary values" in document


def test_an_empty_report_renders_an_explicit_nothing(tmp_path: Path) -> None:
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    document = render_compare_report_html(report)
    assert "No comparable units were found" in document
    assert "<table" not in document


def test_excluded_runs_are_shown_with_their_reason(tmp_path: Path) -> None:
    write_run(tmp_path, "good")
    write_api_run(tmp_path, "other")
    write_run(tmp_path, "broken", corrupt_final_record=True)
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    document = render_compare_report_html(report)
    assert "Excluded runs (1)" in document
    assert "broken" in document


# --- Structure, accessibility and responsiveness --------------------------


def test_the_document_is_well_formed_html(tmp_path: Path) -> None:
    document = render_compare_report_html(_built_report(tmp_path))
    assert document.startswith("<!DOCTYPE html>\n")
    assert document.rstrip().endswith("</html>")
    assert document.count("<body>") == 1
    assert document.count("</body>") == 1


def test_every_section_is_labeled_for_a_screen_reader(tmp_path: Path) -> None:
    document = render_compare_report_html(_built_report(tmp_path))
    sections = re.findall(
        r"<section id=\"([^\"]+)\" aria-labelledby=\"([^\"]+)\"", document
    )
    assert sections
    for _section_id, labelled_by in sections:
        assert f'id="{labelled_by}"' in document


def test_the_page_offers_a_skip_link_and_a_contents_nav(tmp_path: Path) -> None:
    document = render_compare_report_html(_built_report(tmp_path))
    assert '<a class="skip" href="#report">' in document
    assert '<nav class="transect" aria-label="Report contents">' in document
    assert '<main id="report">' in document


def test_every_table_has_a_caption_and_scoped_headers(tmp_path: Path) -> None:
    document = render_compare_report_html(
        _built_report(tmp_path, pricing=_PRICING, pricing_manifest_path="rates.json")
    )
    tables = document.count("<table")
    assert tables >= 2
    assert document.count("<caption>") == tables
    assert 'scope="col"' in document
    assert 'scope="rowgroup"' in document


def test_measurement_cells_carry_their_column_label_for_narrow_viewports(
    tmp_path: Path,
) -> None:
    document = render_compare_report_html(_built_report(tmp_path))
    # The stylesheet reflows the table into stacked rows below 760px, where the
    # only remaining column identification is the data-label attribute.
    assert 'data-label="Mean latency (ms)"' in document
    assert "@media (max-width: 760px)" in document


def test_the_document_adapts_to_reduced_motion_and_print(tmp_path: Path) -> None:
    document = render_compare_report_html(_built_report(tmp_path))
    assert "@media (prefers-reduced-motion: reduce)" in document
    assert "@media print" in document


def test_status_is_never_carried_by_colour_alone(tmp_path: Path) -> None:
    document = render_compare_report_html(_built_report(tmp_path))
    assert ">RECOMMENDED<" in document or ">Recommended<" in document
    assert "viewport" in document


def test_the_brand_lockup_is_inline_with_an_accessible_name(tmp_path: Path) -> None:
    document = render_compare_report_html(_built_report(tmp_path))
    assert 'role="img" aria-label="LLMTraceFX"' in document
    assert "<svg" in document


@pytest.mark.parametrize("redact", [True, False])
def test_rendering_never_raises_on_a_minimal_report(
    tmp_path: Path, redact: bool
) -> None:
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    assert render_compare_report_html(report, redact_paths=redact)
