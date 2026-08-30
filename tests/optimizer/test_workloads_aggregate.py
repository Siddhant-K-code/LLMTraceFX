"""Tests for aggregation/reporting across executed verify-pipeline rows."""

from __future__ import annotations

import json
from pathlib import Path

from llmtracefx.optimizer.workloads.aggregate import (
    AGGREGATE_SCHEMA_VERSION,
    summarize_results,
    write_summary,
)
from llmtracefx.optimizer.workloads.verify import (
    BACKEND_MLX,
    BACKEND_OPENAI_API,
    RowStatus,
    RowVerification,
)

SCHEMA_VERSION = "1"


def _write_verification(
    results_dir: Path,
    run_id: str,
    *,
    status: RowStatus,
    decode_mode: str = "autoregressive",
    context_tier: str = "2k",
    outcome_success: bool | None = None,
    quality_score: float | None = None,
    total_ms: float | None = None,
    backend: str = BACKEND_MLX,
    provider: str | None = None,
) -> None:
    verification = RowVerification(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        workload_id="structured-json-profile-extraction",
        workload_version="1",
        category="structured_json",
        context_tier=context_tier,
        decode_mode=decode_mode,
        status=status,
        reason=None,
        recorded_prompt_hash="sha256:abc",
        verified_prompt_hash="sha256:abc",
        run_binding_hash="sha256:def",
        resumed=False,
        outcome_success=outcome_success,
        quality_score=quality_score,
        total_ms=total_ms,
        started_at="2024-01-01T00:00:00.000000Z",
        ended_at="2024-01-01T00:00:01.000000Z",
        final_record_path=str(results_dir / "runs" / run_id / "final_record.json"),
        collection_dir=str(results_dir / "runs" / run_id / "collection"),
        backend=backend,
        provider=provider,
    )
    run_dir = results_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "verification.json").write_text(verification.to_json(), encoding="utf-8")


def test_summarize_results_with_no_runs_dir_is_empty(tmp_path):
    summary = summarize_results(tmp_path / "does-not-exist")
    assert summary.overall.total == 0
    assert summary.overall.pass_rate is None
    assert summary.by_decode_mode == ()
    assert summary.by_context_tier == ()


def test_summarize_results_counts_by_status(tmp_path):
    results_dir = tmp_path / "results"
    _write_verification(
        results_dir,
        "r1",
        status=RowStatus.COMPLETED,
        outcome_success=True,
        quality_score=1.0,
        total_ms=1000,
    )
    _write_verification(
        results_dir,
        "r2",
        status=RowStatus.COMPLETED,
        outcome_success=False,
        quality_score=0.0,
        total_ms=1000,
    )
    _write_verification(results_dir, "r3", status=RowStatus.FAILED)
    _write_verification(
        results_dir, "r4", status=RowStatus.UNSUPPORTED, decode_mode="native-mtp"
    )
    _write_verification(
        results_dir, "r5", status=RowStatus.INCONCLUSIVE, outcome_success=True
    )
    _write_verification(
        results_dir,
        "r6",
        status=RowStatus.SKIPPED,
        outcome_success=True,
        quality_score=1.0,
        total_ms=500,
    )

    summary = summarize_results(results_dir)

    assert summary.overall.total == 6
    assert summary.overall.completed == 2
    assert summary.overall.failed == 1
    assert summary.overall.unsupported == 1
    assert summary.overall.inconclusive == 1
    assert summary.overall.skipped == 1


def test_summarize_results_pass_rate_excludes_failed_unsupported_inconclusive(tmp_path):
    results_dir = tmp_path / "results"
    _write_verification(
        results_dir,
        "pass1",
        status=RowStatus.COMPLETED,
        outcome_success=True,
        quality_score=1.0,
        total_ms=1000,
    )
    _write_verification(
        results_dir,
        "fail1",
        status=RowStatus.COMPLETED,
        outcome_success=False,
        quality_score=0.0,
        total_ms=1000,
    )
    _write_verification(results_dir, "unsupported1", status=RowStatus.UNSUPPORTED)
    _write_verification(results_dir, "failed-runtime", status=RowStatus.FAILED)
    _write_verification(
        results_dir,
        "inconclusive1",
        status=RowStatus.INCONCLUSIVE,
        outcome_success=True,
    )

    summary = summarize_results(results_dir)

    # Only the two COMPLETED rows with a quality_score count toward pass rate.
    assert summary.overall.evaluated_total == 2
    assert summary.overall.evaluated_pass == 1
    assert summary.overall.pass_rate == 0.5


def test_correct_cases_per_minute_uses_only_passing_timed_rows(tmp_path):
    results_dir = tmp_path / "results"
    # Two correct cases each taking 30s (30_000 ms) => 60s total => 2 cases/min.
    _write_verification(
        results_dir,
        "r1",
        status=RowStatus.COMPLETED,
        outcome_success=True,
        quality_score=1.0,
        total_ms=30_000,
    )
    _write_verification(
        results_dir,
        "r2",
        status=RowStatus.COMPLETED,
        outcome_success=True,
        quality_score=1.0,
        total_ms=30_000,
    )
    # An incorrect (but timed) case must not count toward throughput.
    _write_verification(
        results_dir,
        "r3",
        status=RowStatus.COMPLETED,
        outcome_success=False,
        quality_score=0.0,
        total_ms=5_000,
    )

    summary = summarize_results(results_dir)

    assert summary.overall.correct_cases_per_minute == 2.0


def test_correct_cases_per_minute_is_none_without_timing(tmp_path):
    results_dir = tmp_path / "results"
    _write_verification(
        results_dir,
        "r1",
        status=RowStatus.COMPLETED,
        outcome_success=True,
        quality_score=1.0,
        total_ms=None,
    )

    summary = summarize_results(results_dir)

    assert summary.overall.correct_cases_per_minute is None


def test_summary_groups_by_decode_mode_and_context_tier(tmp_path):
    results_dir = tmp_path / "results"
    _write_verification(
        results_dir,
        "ar-2k",
        status=RowStatus.COMPLETED,
        decode_mode="autoregressive",
        context_tier="2k",
        outcome_success=True,
        quality_score=1.0,
        total_ms=1000,
    )
    _write_verification(
        results_dir,
        "mtp-2k",
        status=RowStatus.UNSUPPORTED,
        decode_mode="native-mtp",
        context_tier="2k",
    )
    _write_verification(
        results_dir,
        "ar-8k",
        status=RowStatus.COMPLETED,
        decode_mode="autoregressive",
        context_tier="8k",
        outcome_success=False,
        quality_score=0.0,
        total_ms=1000,
    )

    summary = summarize_results(results_dir)

    modes = {group.key: group for group in summary.by_decode_mode}
    assert modes["autoregressive"].total == 2
    assert modes["native-mtp"].total == 1
    assert modes["native-mtp"].unsupported == 1

    tiers = {group.key: group for group in summary.by_context_tier}
    assert tiers["2k"].total == 2
    assert tiers["8k"].total == 1
    assert tiers["8k"].evaluated_pass == 0


def test_summarize_results_skips_corrupt_verification_files(tmp_path):
    results_dir = tmp_path / "results"
    _write_verification(
        results_dir,
        "good",
        status=RowStatus.COMPLETED,
        outcome_success=True,
        quality_score=1.0,
        total_ms=1000,
    )
    corrupt_dir = results_dir / "runs" / "corrupt"
    corrupt_dir.mkdir(parents=True)
    (corrupt_dir / "verification.json").write_text("not json", encoding="utf-8")

    summary = summarize_results(results_dir)

    assert summary.overall.total == 1


def test_write_summary_persists_json(tmp_path):
    results_dir = tmp_path / "results"
    _write_verification(
        results_dir,
        "r1",
        status=RowStatus.COMPLETED,
        outcome_success=True,
        quality_score=1.0,
        total_ms=1000,
    )
    summary = summarize_results(results_dir)

    output_path = tmp_path / "summary.json"
    write_summary(summary, output_path)

    payload = json.loads(output_path.read_text())
    assert payload["overall"]["total"] == 1
    assert payload["schema_version"] == AGGREGATE_SCHEMA_VERSION
    # v2 is a strict superset of v1: both new axes are always present.
    assert "by_backend" in payload
    assert "by_provider" in payload


def test_backend_axis_keeps_local_and_api_rows_apart(tmp_path):
    """A local row and a hosted-API row are never blended into one figure."""
    results_dir = tmp_path / "results"
    _write_verification(
        results_dir,
        "local",
        status=RowStatus.COMPLETED,
        outcome_success=True,
        quality_score=1.0,
        total_ms=1000,
        backend=BACKEND_MLX,
    )
    _write_verification(
        results_dir,
        "remote",
        status=RowStatus.COMPLETED,
        outcome_success=False,
        quality_score=0.0,
        total_ms=4000,
        backend=BACKEND_OPENAI_API,
        provider="openrouter",
    )
    summary = summarize_results(results_dir)

    by_backend = {group.key: group for group in summary.by_backend}
    assert set(by_backend) == {BACKEND_MLX, BACKEND_OPENAI_API}
    assert by_backend[BACKEND_MLX].pass_rate == 1.0
    assert by_backend[BACKEND_OPENAI_API].pass_rate == 0.0
    # The blended overall figure still exists, but each backend keeps its own.
    assert summary.overall.pass_rate == 0.5


def test_provider_axis_only_covers_rows_that_have_a_provider(tmp_path):
    """Local rows are absent rather than gathered under a fake provider."""
    results_dir = tmp_path / "results"
    _write_verification(
        results_dir,
        "local",
        status=RowStatus.COMPLETED,
        outcome_success=True,
        quality_score=1.0,
        total_ms=1000,
    )
    _write_verification(
        results_dir,
        "router",
        status=RowStatus.COMPLETED,
        outcome_success=True,
        quality_score=1.0,
        total_ms=2000,
        backend=BACKEND_OPENAI_API,
        provider="openrouter",
    )
    _write_verification(
        results_dir,
        "direct",
        status=RowStatus.COMPLETED,
        outcome_success=True,
        quality_score=1.0,
        total_ms=3000,
        backend=BACKEND_OPENAI_API,
        provider="z.ai",
    )
    summary = summarize_results(results_dir)

    by_provider = {group.key: group for group in summary.by_provider}
    assert set(by_provider) == {"openrouter", "z.ai"}
    assert by_provider["openrouter"].total == 1
    assert by_provider["z.ai"].total == 1


def test_group_with_no_evaluated_rows_reports_null_not_zero(tmp_path):
    """An unavailable metric is never rendered as a measured zero."""
    results_dir = tmp_path / "results"
    _write_verification(
        results_dir,
        "failed-remote",
        status=RowStatus.FAILED,
        backend=BACKEND_OPENAI_API,
        provider="openrouter",
    )
    summary = summarize_results(results_dir)

    group = next(g for g in summary.by_provider if g.key == "openrouter")
    assert group.evaluated_total == 0
    assert group.pass_rate is None
    assert group.correct_cases_per_minute is None
    payload = json.loads(summary.to_json())
    assert payload["by_provider"][0]["pass_rate"] is None


def test_a_mixed_backend_group_withholds_its_timing_metric(tmp_path):
    """A local duration and an API duration are not the same quantity.

    Blending them yields a throughput figure describing no system that
    exists, and the wider the aggregate the more authoritative it looks.
    """
    results_dir = tmp_path / "results"
    _write_verification(
        results_dir,
        "local",
        status=RowStatus.COMPLETED,
        outcome_success=True,
        quality_score=1.0,
        total_ms=1000,
        backend=BACKEND_MLX,
    )
    _write_verification(
        results_dir,
        "remote",
        status=RowStatus.COMPLETED,
        outcome_success=True,
        quality_score=1.0,
        total_ms=9000,
        backend=BACKEND_OPENAI_API,
        provider="openrouter",
    )
    summary = summarize_results(results_dir)

    # Quality survives: a pass is a pass wherever it was produced.
    assert summary.overall.pass_rate == 1.0
    assert summary.overall.evaluated_pass == 2
    # The timing-derived figure does not, and says why.
    assert summary.overall.correct_cases_per_minute is None
    assert summary.overall.timing_comparable is False
    assert "not the same quantity" in (summary.overall.timing_unavailable_reason or "")
    assert summary.overall.measurement_contexts == ("mlx", "openai-api/openrouter")

    # The homogeneous per-backend groups keep theirs.
    by_backend = {group.key: group for group in summary.by_backend}
    assert by_backend[BACKEND_MLX].correct_cases_per_minute is not None
    assert by_backend[BACKEND_MLX].timing_comparable is True
    assert by_backend[BACKEND_OPENAI_API].correct_cases_per_minute is not None


def test_two_providers_in_one_group_also_withhold_timing(tmp_path):
    """Two hosted endpoints are not one measurement context either."""
    results_dir = tmp_path / "results"
    for run_id, provider in (("a", "openrouter"), ("b", "z.ai")):
        _write_verification(
            results_dir,
            run_id,
            status=RowStatus.COMPLETED,
            outcome_success=True,
            quality_score=1.0,
            total_ms=2000,
            backend=BACKEND_OPENAI_API,
            provider=provider,
        )
    summary = summarize_results(results_dir)

    assert summary.overall.correct_cases_per_minute is None
    assert summary.overall.timing_comparable is False
    by_backend = {group.key: group for group in summary.by_backend}
    # Same backend, still two providers, so still not comparable.
    assert by_backend[BACKEND_OPENAI_API].correct_cases_per_minute is None
    by_provider = {group.key: group for group in summary.by_provider}
    assert by_provider["openrouter"].correct_cases_per_minute is not None
    assert by_provider["z.ai"].correct_cases_per_minute is not None


def test_a_homogeneous_group_is_unaffected(tmp_path):
    """The suppression is narrow: one context keeps every metric."""
    results_dir = tmp_path / "results"
    for run_id in ("a", "b"):
        _write_verification(
            results_dir,
            run_id,
            status=RowStatus.COMPLETED,
            outcome_success=True,
            quality_score=1.0,
            total_ms=1000,
            backend=BACKEND_OPENAI_API,
            provider="openrouter",
        )
    summary = summarize_results(results_dir)

    assert summary.overall.timing_comparable is True
    assert summary.overall.timing_unavailable_reason is None
    assert summary.overall.correct_cases_per_minute is not None


def test_a_failing_row_from_another_backend_does_not_suppress_timing(tmp_path):
    """Only rows that contribute a duration define the context.

    A failed row has no timing to blend, so its backend must not withhold
    a figure the passing rows can legitimately support.
    """
    results_dir = tmp_path / "results"
    _write_verification(
        results_dir,
        "remote-pass",
        status=RowStatus.COMPLETED,
        outcome_success=True,
        quality_score=1.0,
        total_ms=2000,
        backend=BACKEND_OPENAI_API,
        provider="openrouter",
    )
    _write_verification(
        results_dir, "local-fail", status=RowStatus.FAILED, backend=BACKEND_MLX
    )
    summary = summarize_results(results_dir)

    assert summary.overall.timing_comparable is True
    assert summary.overall.correct_cases_per_minute is not None
