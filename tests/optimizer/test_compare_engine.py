"""Tests for the comparison engine: grouping, metrics, ranking and frontier.

Every artifact tree in this module is synthetic. The numbers are chosen to
exercise a specific code path and are not measurements of any real system.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _compare_fixtures import write_api_run, write_run

from llmtracefx.optimizer.compare.compare import compare
from llmtracefx.optimizer.compare.evidence import CompareEvidenceError
from llmtracefx.optimizer.compare.policy import (
    CompareConstraints,
    CompareObjective,
    ComparePolicy,
)
from llmtracefx.optimizer.compare.pricing import PricingManifest
from llmtracefx.optimizer.compare.report import (
    CompareReport,
    StratumOutcome,
    TtftBasis,
)
from llmtracefx.optimizer.schema import MetricProvenance
from llmtracefx.optimizer.workloads.verify import RowStatus

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


def _policy(
    objective: CompareObjective = CompareObjective.MIN_MEAN_TOTAL_LATENCY_MS,
    **constraint_overrides: object,
) -> ComparePolicy:
    return ComparePolicy(
        objective=objective,
        name="synthetic policy",
        constraints=CompareConstraints(**constraint_overrides),  # type: ignore[arg-type]
    )


def _three_systems(results: Path) -> None:
    """A local model, a frontier API and a flash API on the identical unit."""
    write_run(results, "local-1", total_ms=8000.0, max_tokens_argv=512)
    write_api_run(
        results,
        "frontier-1",
        model_id="glm-5.3",
        total_ms=3000.0,
        prompt_tokens=1000,
        completion_tokens=400,
    )
    write_api_run(
        results,
        "flash-1",
        model_id="glm-5.3-flash",
        reasoning_effort="low",
        total_ms=1200.0,
        prompt_tokens=1000,
        completion_tokens=400,
    )


# --- Grouping isolation ---------------------------------------------------


def test_one_stratum_per_comparable_unit(tmp_path: Path) -> None:
    _three_systems(tmp_path)
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    assert len(report.strata) == 1
    assert len(report.strata[0].ranked) == 3


def test_different_workloads_never_share_a_stratum(tmp_path: Path) -> None:
    write_run(tmp_path, "a", workload_id="one")
    write_run(tmp_path, "b", workload_id="two")
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    assert len(report.strata) == 2


def test_different_context_tiers_never_share_a_stratum(tmp_path: Path) -> None:
    write_run(tmp_path, "a", context_tier="2k")
    write_run(tmp_path, "b", context_tier="32k")
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    assert len(report.strata) == 2


def test_different_prompt_hashes_never_share_a_stratum(tmp_path: Path) -> None:
    write_run(tmp_path, "a", prompt_hash="sha256:one")
    write_run(tmp_path, "b", prompt_hash="sha256:two")
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    assert len(report.strata) == 2


def test_different_evaluators_never_share_a_stratum(tmp_path: Path) -> None:
    write_run(tmp_path, "a", quality_metric="exact_match")
    write_run(tmp_path, "b", quality_metric="fuzzy_match")
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    assert len(report.strata) == 2


def test_unlike_systems_are_never_averaged_together(tmp_path: Path) -> None:
    write_api_run(tmp_path, "high-1", reasoning_effort="high", total_ms=3000.0)
    write_api_run(tmp_path, "low-1", reasoning_effort="low", total_ms=1000.0)
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    stratum = report.strata[0]
    assert len(stratum.ranked) == 2
    latencies = sorted(system.mean_total_latency_ms for system in stratum.ranked)
    assert latencies == pytest.approx([1000.0, 3000.0])


def test_runs_of_one_system_are_pooled(tmp_path: Path) -> None:
    write_api_run(tmp_path, "a", total_ms=1000.0)
    write_api_run(tmp_path, "b", total_ms=3000.0)
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    system = report.strata[0].ranked[0]
    assert system.evidence_count == 2
    assert system.mean_total_latency_ms == pytest.approx(2000.0)
    assert system.p50_total_latency_ms == pytest.approx(2000.0)
    assert system.p95_total_latency_ms == pytest.approx(3000.0)


# --- Metrics --------------------------------------------------------------


def test_pass_rate_and_throughput_use_the_shared_formulas(tmp_path: Path) -> None:
    write_api_run(tmp_path, "a", total_ms=30_000.0, success=True)
    write_api_run(tmp_path, "b", total_ms=30_000.0, success=False, quality_score=0.0)
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    system = report.strata[0].ranked[0]
    assert system.pass_rate == pytest.approx(0.5)
    # One passing case in 30 s of measured time is two correct cases a minute.
    assert system.correct_cases_per_minute == pytest.approx(2.0)


def test_local_ttft_is_labeled_as_a_local_prefill(tmp_path: Path) -> None:
    write_run(tmp_path, "local-1", prefill_ms=310.0)
    write_api_run(tmp_path, "api-1")
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    by_local = {
        system.system_key.is_local: system for system in report.strata[0].ranked
    }
    assert by_local[True].ttft_basis == TtftBasis.LOCAL_PREFILL
    assert by_local[True].mean_ttft_ms == pytest.approx(310.0)
    assert by_local[False].ttft_basis == TtftBasis.CLIENT_OBSERVED_STREAM
    assert by_local[False].mean_ttft_ms == pytest.approx(220.0)


def test_local_peak_memory_stays_local_only(tmp_path: Path) -> None:
    write_run(tmp_path, "local-1")
    write_api_run(tmp_path, "api-1")
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    for system in report.strata[0].ranked:
        if system.system_key.is_local:
            assert system.mean_peak_memory_bytes is not None
        else:
            assert system.mean_peak_memory_bytes is None
            assert any(
                "local-only measurement" in note for note in system.missing_evidence
            )


def test_missing_measurements_stay_unavailable_never_zero(tmp_path: Path) -> None:
    write_api_run(tmp_path, "api-1", first_content_token_offset_ms=None)
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    system = report.strata[0].ranked[0]
    assert system.mean_ttft_ms is None
    assert system.ttft_basis is None
    assert any("time-to-first-token" in note for note in system.missing_evidence)


def test_partial_usage_reporting_withholds_the_total(tmp_path: Path) -> None:
    write_api_run(tmp_path, "a", prompt_tokens=1000, completion_tokens=400)
    write_api_run(tmp_path, "b", prompt_tokens=None, completion_tokens=400)
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    usage = report.strata[0].ranked[0].usage
    assert usage is not None
    assert usage.input_tokens is None
    assert usage.output_tokens == 800


def test_usage_absent_entirely_is_reported_as_none(tmp_path: Path) -> None:
    write_run(tmp_path, "local-1")
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    assert report.strata[0].ranked[0].usage is None


# --- Cost -----------------------------------------------------------------


def test_cost_is_estimated_from_provider_usage_and_manifest_rates(
    tmp_path: Path,
) -> None:
    write_api_run(
        tmp_path, "api-1", prompt_tokens=1_000_000, completion_tokens=1_000_000
    )
    report = compare(
        results_dirs=(tmp_path,),
        policy=_policy(),
        pricing=_PRICING,
        pricing_manifest_path="synthetic-rates.json",
    )
    cost = report.strata[0].ranked[0].cost
    assert cost is not None
    assert cost.total_amount == pytest.approx(3.0)
    assert cost.cost_per_correct_case == pytest.approx(3.0)
    assert cost.rates_are_illustrative is True
    assert report.pricing is not None
    assert report.pricing.entry_ids_used == ("glm-5.3",)


def test_an_unpriced_system_gets_no_cost_and_says_so(tmp_path: Path) -> None:
    write_run(tmp_path, "local-1")
    report = compare(
        results_dirs=(tmp_path,),
        policy=_policy(),
        pricing=_PRICING,
        pricing_manifest_path="synthetic-rates.json",
    )
    system = report.strata[0].ranked[0]
    assert system.cost is None
    assert any("no entry for provider" in note for note in system.missing_evidence)


def test_a_run_without_usage_blocks_the_system_total(tmp_path: Path) -> None:
    write_api_run(tmp_path, "a", prompt_tokens=1000, completion_tokens=400)
    write_api_run(tmp_path, "b", usage_reported=False)
    report = compare(
        results_dirs=(tmp_path,),
        policy=_policy(),
        pricing=_PRICING,
        pricing_manifest_path="synthetic-rates.json",
    )
    cost = report.strata[0].ranked[0].cost
    assert cost is not None
    assert cost.total_amount is None
    assert any("could not be priced" in reason for reason in cost.reasons)


def test_no_pricing_manifest_means_no_monetary_values_at_all(tmp_path: Path) -> None:
    _three_systems(tmp_path)
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    assert report.pricing is None
    assert all(
        system.cost is None for stratum in report.strata for system in stratum.ranked
    )


def test_a_cost_objective_without_pricing_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CompareEvidenceError, match="ranks on money"):
        compare(
            results_dirs=(tmp_path,),
            policy=_policy(CompareObjective.MIN_COST_PER_CORRECT_CASE),
        )


def test_pricing_without_its_path_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CompareEvidenceError, match="without its path"):
        compare(results_dirs=(tmp_path,), policy=_policy(), pricing=_PRICING)


def test_cost_objective_ranks_the_cheaper_system_first(tmp_path: Path) -> None:
    write_api_run(
        tmp_path,
        "frontier-1",
        model_id="glm-5.3",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    write_api_run(
        tmp_path,
        "flash-1",
        model_id="glm-5.3-flash",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    report = compare(
        results_dirs=(tmp_path,),
        policy=_policy(CompareObjective.MIN_COST_PER_CORRECT_CASE),
        pricing=_PRICING,
        pricing_manifest_path="synthetic-rates.json",
    )
    stratum = report.strata[0]
    assert stratum.outcome == StratumOutcome.RECOMMENDED
    assert stratum.recommended is not None
    assert stratum.recommended.system_key.model_id == "glm-5.3-flash"


def test_cost_constraint_rejects_an_expensive_system(tmp_path: Path) -> None:
    write_api_run(
        tmp_path,
        "frontier-1",
        model_id="glm-5.3",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    write_api_run(
        tmp_path,
        "flash-1",
        model_id="glm-5.3-flash",
        prompt_tokens=1_000_000,
        completion_tokens=1_000_000,
    )
    report = compare(
        results_dirs=(tmp_path,),
        policy=_policy(max_cost_per_correct_case=1.0),
        pricing=_PRICING,
        pricing_manifest_path="synthetic-rates.json",
    )
    stratum = report.strata[0]
    assert [system.system_key.model_id for system in stratum.ranked] == [
        "glm-5.3-flash"
    ]
    assert len(stratum.rejected) == 1
    assert any(
        "exceeds the maximum" in reason for reason in stratum.rejected[0].reasons
    )


def test_an_unpriced_system_cannot_slip_past_a_cost_ceiling(tmp_path: Path) -> None:
    """A ceiling that cannot be evaluated is a rejection, never a free pass.

    A provider-keyed manifest can never match a local system, so nesting the
    ceiling inside "an entry resolved" would exempt exactly the system with no
    cost evidence at all while holding every priced system to the limit.
    """
    write_run(tmp_path, "local-1", total_ms=8000.0)
    write_api_run(
        tmp_path,
        "flash-1",
        model_id="glm-5.3-flash",
        prompt_tokens=1_000,
        completion_tokens=1_000,
    )
    report = compare(
        results_dirs=(tmp_path,),
        policy=_policy(max_cost_per_correct_case=1.0),
        pricing=_PRICING,
        pricing_manifest_path="synthetic-rates.json",
    )
    stratum = report.strata[0]
    assert [system.system_key.model_id for system in stratum.ranked] == [
        "glm-5.3-flash"
    ]
    rejected = {system.system_key.model_id for system in stratum.rejected}
    assert rejected == {"local/qwen3-8b"}
    assert any(
        "no cost per correct case could be estimated" in reason
        for system in stratum.rejected
        for reason in system.reasons
    )


def test_a_cost_ceiling_without_pricing_is_refused_up_front(tmp_path: Path) -> None:
    with pytest.raises(CompareEvidenceError, match="could never be evaluated"):
        compare(
            results_dirs=(tmp_path,),
            policy=_policy(max_cost_per_correct_case=1.0),
        )


# --- Ranking, ties and noise ----------------------------------------------


def test_lowest_latency_wins_the_latency_objective(tmp_path: Path) -> None:
    _three_systems(tmp_path)
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    stratum = report.strata[0]
    assert stratum.outcome == StratumOutcome.RECOMMENDED
    assert stratum.recommended is not None
    assert stratum.recommended.system_key.model_id == "glm-5.3-flash"
    assert [system.rank for system in stratum.ranked] == [1, 2, 3]


def test_highest_throughput_wins_the_throughput_objective(tmp_path: Path) -> None:
    _three_systems(tmp_path)
    report = compare(
        results_dirs=(tmp_path,),
        policy=_policy(CompareObjective.MAX_CORRECT_CASES_PER_MINUTE),
    )
    stratum = report.strata[0]
    assert stratum.recommended is not None
    assert stratum.recommended.system_key.model_id == "glm-5.3-flash"


def test_an_exact_tie_is_inconclusive(tmp_path: Path) -> None:
    write_api_run(tmp_path, "a", model_id="glm-5.3", total_ms=1000.0)
    write_api_run(tmp_path, "b", model_id="glm-5.3-flash", total_ms=1000.0)
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    stratum = report.strata[0]
    assert stratum.outcome == StratumOutcome.INCONCLUSIVE
    assert stratum.recommended is None
    assert "exactly tied" in (stratum.inconclusive_reason or "")


def test_a_difference_inside_the_noise_band_is_inconclusive(tmp_path: Path) -> None:
    write_api_run(tmp_path, "a1", model_id="glm-5.3", total_ms=800.0)
    write_api_run(tmp_path, "a2", model_id="glm-5.3", total_ms=1200.0)
    write_api_run(tmp_path, "b1", model_id="glm-5.3-flash", total_ms=850.0)
    write_api_run(tmp_path, "b2", model_id="glm-5.3-flash", total_ms=1250.0)
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    stratum = report.strata[0]
    assert stratum.outcome == StratumOutcome.INCONCLUSIVE
    assert "measurement noise" in (stratum.inconclusive_reason or "")


def test_a_difference_beyond_the_noise_band_is_conclusive(tmp_path: Path) -> None:
    write_api_run(tmp_path, "a1", model_id="glm-5.3", total_ms=5000.0)
    write_api_run(tmp_path, "a2", model_id="glm-5.3", total_ms=5010.0)
    write_api_run(tmp_path, "b1", model_id="glm-5.3-flash", total_ms=1000.0)
    write_api_run(tmp_path, "b2", model_id="glm-5.3-flash", total_ms=1010.0)
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    assert report.strata[0].outcome == StratumOutcome.RECOMMENDED


def test_a_single_system_never_yields_a_comparison(tmp_path: Path) -> None:
    write_run(tmp_path, "local-1")
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    stratum = report.strata[0]
    assert stratum.outcome == StratumOutcome.INCONCLUSIVE
    assert "nothing was compared" in (stratum.inconclusive_reason or "")
    assert any("at least two" in note for note in stratum.missing_evidence)


def test_nothing_clearing_the_constraints_is_inconclusive(tmp_path: Path) -> None:
    write_run(tmp_path, "a", success=False, quality_score=0.0)
    write_api_run(tmp_path, "b", success=False, quality_score=0.0)
    report = compare(results_dirs=(tmp_path,), policy=_policy(min_pass_rate=1.0))
    stratum = report.strata[0]
    assert stratum.outcome == StratumOutcome.INCONCLUSIVE
    assert stratum.ranked == ()
    assert len(stratum.rejected) == 2


def test_every_violated_constraint_is_reported_not_just_the_first(
    tmp_path: Path,
) -> None:
    write_run(tmp_path, "a", success=False, quality_score=0.1, total_ms=90_000.0)
    write_api_run(tmp_path, "b", total_ms=1000.0)
    report = compare(
        results_dirs=(tmp_path,),
        policy=_policy(
            min_pass_rate=1.0,
            max_mean_total_latency_ms=1000.0,
            min_quality_score=0.9,
            required_quality_metric="structured_json_exact_field_match",
        ),
    )
    rejected = report.strata[0].rejected
    assert len(rejected) == 1
    assert len(rejected[0].reasons) >= 2


def test_a_disallowed_status_rejects_the_system(tmp_path: Path) -> None:
    write_run(tmp_path, "a", status=RowStatus.FAILED, success=False)
    write_api_run(tmp_path, "b")
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    reasons = [
        reason for system in report.strata[0].rejected for reason in system.reasons
    ]
    assert any("not one of the required statuses" in reason for reason in reasons)


def test_a_disallowed_provenance_rejects_the_timing_evidence(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "a",
        total_provenance=MetricProvenance.ESTIMATED,
    )
    write_api_run(tmp_path, "b")
    report = compare(
        results_dirs=(tmp_path,),
        policy=_policy(
            allowed_provenances=frozenset({MetricProvenance.MEASURED_WALL_CLOCK})
        ),
    )
    reasons = [
        reason for system in report.strata[0].rejected for reason in system.reasons
    ]
    assert any("not in the allowed provenance set" in reason for reason in reasons)


def test_a_missing_timing_measurement_rejects_the_system(tmp_path: Path) -> None:
    write_run(tmp_path, "a", total_ms=None)
    write_api_run(tmp_path, "b")
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    reasons = [
        reason for system in report.strata[0].rejected for reason in system.reasons
    ]
    assert any("missing timing.total" in reason for reason in reasons)


def test_ranking_is_deterministic_for_equal_objective_values(tmp_path: Path) -> None:
    write_api_run(tmp_path, "a", model_id="glm-5.3", total_ms=1000.0)
    write_api_run(tmp_path, "b", model_id="glm-5.3-flash", total_ms=1000.0)
    first = compare(results_dirs=(tmp_path,), policy=_policy())
    second = compare(results_dirs=(tmp_path,), policy=_policy())
    assert [system.system_key.label() for system in first.strata[0].ranked] == [
        system.system_key.label() for system in second.strata[0].ranked
    ]


# --- Frontier -------------------------------------------------------------


def test_the_frontier_names_dominating_systems(tmp_path: Path) -> None:
    write_api_run(
        tmp_path, "slow", model_id="glm-5.3", total_ms=9000.0, quality_score=1.0
    )
    write_api_run(
        tmp_path,
        "fast",
        model_id="glm-5.3-flash",
        total_ms=1000.0,
        quality_score=1.0,
    )
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    stratum = report.strata[0]
    by_model = {entry.system_key.model_id: entry for entry in stratum.frontier}
    assert by_model["glm-5.3-flash"].dominated is False
    assert by_model["glm-5.3"].dominated is True
    assert by_model["glm-5.3"].dominated_by


def test_a_speed_versus_quality_tradeoff_leaves_both_on_the_frontier(
    tmp_path: Path,
) -> None:
    write_api_run(
        tmp_path,
        "accurate",
        model_id="glm-5.3",
        total_ms=9000.0,
        success=True,
        quality_score=1.0,
    )
    write_api_run(
        tmp_path,
        "quick-1",
        model_id="glm-5.3-flash",
        total_ms=1000.0,
        success=True,
        quality_score=1.0,
    )
    write_api_run(
        tmp_path,
        "quick-2",
        model_id="glm-5.3-flash",
        total_ms=1000.0,
        success=False,
        quality_score=0.0,
    )
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    stratum = report.strata[0]
    assert all(entry.dominated is False for entry in stratum.frontier)


def test_axes_without_evidence_for_everyone_are_dropped(tmp_path: Path) -> None:
    write_run(tmp_path, "local-1")
    write_api_run(tmp_path, "api-1")
    report = compare(
        results_dirs=(tmp_path,),
        policy=_policy(),
        pricing=_PRICING,
        pricing_manifest_path="synthetic-rates.json",
    )
    stratum = report.strata[0]
    axis_values = [axis.value for axis in stratum.frontier_axes]
    assert "min_cost_per_correct_case" not in axis_values
    assert any("were dropped" in note for note in stratum.missing_evidence)


# --- Report round trip ----------------------------------------------------


def test_the_report_round_trips_through_json(tmp_path: Path) -> None:
    _three_systems(tmp_path)
    report = compare(
        results_dirs=(tmp_path,),
        policy=_policy(),
        pricing=_PRICING,
        pricing_manifest_path="synthetic-rates.json",
        tune_report_paths=("tune.json",),
    )
    reloaded = CompareReport.from_json(report.to_json())
    assert reloaded.to_dict() == report.to_dict()


def test_excluded_runs_are_carried_into_the_report(tmp_path: Path) -> None:
    write_run(tmp_path, "good")
    write_api_run(tmp_path, "other")
    write_run(tmp_path, "broken", corrupt_final_record=True)
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    assert [run.run_id for run in report.excluded_runs] == ["broken"]


def test_no_evidence_produces_an_empty_report(tmp_path: Path) -> None:
    report = compare(results_dirs=(tmp_path,), policy=_policy())
    assert report.strata == ()
    assert report.has_recommendation is False
