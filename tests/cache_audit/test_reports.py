from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from llmtracefx.cache_audit.adapters.reference import ReferenceCacheAdapter
from llmtracefx.cache_audit.report import build_summary
from llmtracefx.cache_audit.report_html import (
    render_html,
    render_reuse_alignment_svg,
)
from llmtracefx.cache_audit.runner import run_audit
from llmtracefx.cache_audit.schema import CacheConfig, TimingEvidence, unavailable
from llmtracefx.cache_audit.workloads import (
    adversarial_requests,
    gated_extension_requests,
)
from llmtracefx.optimizer.schema import Measurement, MetricProvenance


def _measurement(value: float) -> Measurement:
    return Measurement(
        value=value,
        unit="ms",
        provenance=MetricProvenance.MEASURED_WALL_CLOCK,
    )


def test_nulls_render_as_unavailable_and_not_zero(tmp_path: Path) -> None:
    manifest, records = run_audit(
        adapter=ReferenceCacheAdapter(),
        requests=adversarial_requests(),
        cache_config=CacheConfig(namespace_id="synthetic", cache_type="token_trie"),
        output_dir=tmp_path / "bundle",
        backend_version="1",
        model_id="synthetic-tiny-model",
        tokenizer_id="integer-tokenizer-v1",
        created_at="2026-01-01T00:00:00Z",
    )
    html = render_html(manifest, records)
    unavailable_record = ReferenceCacheAdapter().run(gated_extension_requests()[:1])[0]
    unavailable_record = replace(
        unavailable_record,
        reuse=replace(
            unavailable_record.reuse,
            policy_reusable_tokens=unavailable("test", "not_available"),
            engine_cached_tokens=unavailable("test", "not_available"),
        ),
    )
    unavailable_svg = render_reuse_alignment_svg([unavailable_record])
    assert "None" not in html
    assert "unavailable" in html
    assert "url(#unavailable)" in unavailable_svg


def test_pair_delta_requires_matching_scope_and_records_noncausal_basis(
    tmp_path: Path,
) -> None:
    manifest, records = run_audit(
        adapter=ReferenceCacheAdapter(),
        requests=adversarial_requests()[:2],
        cache_config=CacheConfig(namespace_id="synthetic", cache_type="token_trie"),
        output_dir=tmp_path / "bundle",
        backend_version="1",
        model_id="synthetic-tiny-model",
        tokenizer_id="integer-tokenizer-v1",
        created_at="2026-01-01T00:00:00Z",
    )
    control = replace(
        records[0],
        timing=TimingEvidence(
            in_process_first_token=_measurement(2.0),
            scope="scope-a",
        ),
    )
    treatment = replace(
        records[1],
        timing=TimingEvidence(
            in_process_first_token=_measurement(1.0),
            scope="scope-b",
        ),
    )
    pair = build_summary([control, treatment], manifest)["paired_deltas"][0]
    assert pair["compatible"] is False
    assert pair["first_token_delta_ms"] is None
    assert pair["sample_count"] == 1
    assert pair["uncertainty_available"] is False
    assert pair["causal_interpretation_eligible"] is False
