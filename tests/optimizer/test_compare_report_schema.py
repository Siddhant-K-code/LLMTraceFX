"""Strict-loading and invariant tests for the ``compare`` report schema.

A persisted report is untrusted input. Every test here feeds it something
malformed, inconsistent, or quietly dishonest and requires an explicit
refusal rather than a silently degraded load.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer.compare.identity import ComparableUnitKey, SystemKey
from llmtracefx.optimizer.compare.policy import (
    CompareConstraints,
    CompareObjective,
    ComparePolicy,
    ComparePolicyError,
)
from llmtracefx.optimizer.compare.report import (
    COMPARE_REPORT_SCHEMA_VERSION,
    CompareReport,
    CompareReportValidationError,
    CostSummary,
    FrontierEntry,
    ParetoAxis,
    PricingProvenance,
    RejectedSystemReport,
    StratumOutcome,
    StratumReport,
    SystemReport,
    TtftBasis,
    UsageTotals,
)

_UNIT = ComparableUnitKey(
    workload_id="w",
    workload_version="1",
    workload_prompt_hash="sha256:abc",
    context_tier="2k",
    quality_metric="exact",
    max_output_tokens=512,
    temperature=0.0,
    top_p=1.0,
)

_LOCAL = SystemKey(
    model_id="local/qwen3-8b",
    model_revision=None,
    provider=None,
    runtime_name="mlx-lm",
    runtime_backend="Metal",
    accelerator="Apple M5 Pro",
    quantization="Q4",
    reasoning_effort=None,
    decode_mode="autoregressive",
)

_HOSTED = SystemKey(
    model_id="glm-5.3",
    model_revision=None,
    provider="z-ai",
    runtime_name="openai-compatible-stream",
    runtime_backend=None,
    accelerator=None,
    quantization=None,
    reasoning_effort="high",
    decode_mode="autoregressive",
)


def _system(system_key: SystemKey = _LOCAL, **overrides: Any) -> SystemReport:
    payload: dict[str, Any] = {
        "system_key": system_key,
        "rank": 1,
        "run_ids": ("r1",),
        "verification_paths": ("/tmp/results/runs/r1/verification.json",),
        "record_paths": ("/tmp/results/runs/r1/final_record.json",),
        "evidence_count": 1,
        "objective_name": CompareObjective.MIN_MEAN_TOTAL_LATENCY_MS.value,
        "objective_value": 1000.0,
        "pass_rate": 1.0,
        "mean_total_latency_ms": 1000.0,
    }
    payload.update(overrides)
    return SystemReport(**payload)


def _stratum(**overrides: Any) -> StratumReport:
    system = overrides.pop("system", _system())
    payload: dict[str, Any] = {
        "unit_key": _UNIT,
        "outcome": StratumOutcome.RECOMMENDED,
        "objective_name": CompareObjective.MIN_MEAN_TOTAL_LATENCY_MS.value,
        "ranked": (system,),
        "recommended": system,
    }
    payload.update(overrides)
    return StratumReport(**payload)


def _report(**overrides: Any) -> CompareReport:
    payload: dict[str, Any] = {
        "schema_version": COMPARE_REPORT_SCHEMA_VERSION,
        "generated_at": "2026-01-01T00:00:00.000000Z",
        "results_dirs": ("/tmp/results",),
        "policy": ComparePolicy(
            objective=CompareObjective.MIN_MEAN_TOTAL_LATENCY_MS, name="p"
        ),
        "strata": (_stratum(),),
    }
    payload.update(overrides)
    return CompareReport(**payload)


def _mutate(report: CompareReport, mutator: Any) -> dict[str, Any]:
    payload = json.loads(report.to_json())
    mutator(payload)
    return payload


# --- Round trip -----------------------------------------------------------


def test_a_full_report_round_trips(tmp_path: Path) -> None:
    report = _report(
        pricing=PricingProvenance(
            manifest_path="rates.json",
            manifest_sha256="deadbeef",
            currency="USD",
            rates_are_illustrative=True,
            entry_ids_used=("glm-5.3",),
        ),
        strata=(
            _stratum(
                system=_system(
                    _HOSTED,
                    mean_ttft_ms=220.0,
                    ttft_basis=TtftBasis.CLIENT_OBSERVED_STREAM,
                    usage=UsageTotals(
                        runs_reporting_usage=1,
                        runs_total=1,
                        input_tokens=1000,
                        output_tokens=400,
                    ),
                    cost=CostSummary(
                        currency="USD",
                        pricing_entry_id="glm-5.3",
                        pricing_entry_sha256="cafebabe",
                        rates_are_illustrative=True,
                        total_amount=1.4,
                        cost_per_case=1.4,
                        cost_per_correct_case=1.4,
                        correct_cases_per_currency_unit=0.71,
                    ),
                ),
                frontier_axes=(ParetoAxis.MAX_PASS_RATE,),
                frontier=(FrontierEntry(system_key=_HOSTED, dominated=False),),
                rejected=(
                    RejectedSystemReport(
                        system_key=_LOCAL,
                        run_ids=("r2",),
                        verification_paths=(),
                        record_paths=(),
                        reasons=("pass rate 0.0 is below the required minimum 1.0",),
                    ),
                ),
            ),
        ),
    )
    path = tmp_path / "compare.json"
    path.write_text(report.to_json() + "\n", encoding="utf-8")
    assert CompareReport.read_json(path).to_dict() == report.to_dict()


# --- Version and shape ----------------------------------------------------


def test_unknown_schema_version_is_refused() -> None:
    payload = _mutate(_report(), lambda p: p.update(schema_version="99"))
    with pytest.raises(CompareReportValidationError, match="unsupported"):
        CompareReport.from_dict(payload)


def test_a_non_object_report_is_refused() -> None:
    with pytest.raises(CompareReportValidationError, match="must be a JSON object"):
        CompareReport.from_dict([1, 2, 3])


def test_invalid_json_is_refused() -> None:
    with pytest.raises(CompareReportValidationError, match="invalid JSON"):
        CompareReport.from_json("{not json")


def test_a_missing_generated_at_is_refused() -> None:
    payload = _mutate(_report(), lambda p: p.pop("generated_at"))
    with pytest.raises(CompareReportValidationError, match="generated_at"):
        CompareReport.from_dict(payload)


# --- Numeric strictness ---------------------------------------------------


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity"])
def test_non_finite_numbers_are_refused(bad: str) -> None:
    raw = (
        _report()
        .to_json()
        .replace('"objective_value": 1000.0', f'"objective_value": {bad}')
    )
    with pytest.raises(CompareReportValidationError, match="finite"):
        CompareReport.from_json(raw)


def test_to_json_refuses_to_emit_a_non_finite_value() -> None:
    report = _report(strata=(_stratum(system=_system(objective_value=float("inf"))),))
    with pytest.raises(ValueError, match="Out of range float"):
        report.to_json()


def test_a_string_where_a_number_belongs_is_refused() -> None:
    payload = _mutate(
        _report(),
        lambda p: p["strata"][0]["ranked"][0].update(mean_total_latency_ms="fast"),
    )
    with pytest.raises(CompareReportValidationError, match="must be a number"):
        CompareReport.from_dict(payload)


def test_a_boolean_where_a_number_belongs_is_refused() -> None:
    payload = _mutate(
        _report(),
        lambda p: p["strata"][0]["ranked"][0].update(pass_rate=True),
    )
    with pytest.raises(CompareReportValidationError, match="must be a number"):
        CompareReport.from_dict(payload)


def test_a_negative_evidence_count_is_refused() -> None:
    payload = _mutate(
        _report(), lambda p: p["strata"][0]["ranked"][0].update(evidence_count=-1)
    )
    with pytest.raises(CompareReportValidationError, match=">= 0"):
        CompareReport.from_dict(payload)


# --- Invariants -----------------------------------------------------------


def test_non_contiguous_ranks_are_refused() -> None:
    payload = _mutate(_report(), lambda p: p["strata"][0]["ranked"][0].update(rank=3))
    with pytest.raises(CompareReportValidationError, match="contiguous"):
        CompareReport.from_dict(payload)


def test_a_recommended_stratum_must_carry_a_recommendation() -> None:
    payload = _mutate(_report(), lambda p: p["strata"][0].update(recommended=None))
    with pytest.raises(CompareReportValidationError, match="'recommended' is null"):
        CompareReport.from_dict(payload)


def test_the_recommendation_must_be_the_rank_one_system() -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["strata"][0]["recommended"]["objective_value"] = 5.0

    with pytest.raises(CompareReportValidationError, match="must equal the rank 1"):
        CompareReport.from_dict(_mutate(_report(), mutate))


def test_an_inconclusive_stratum_must_carry_a_reason() -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["strata"][0]["outcome"] = "inconclusive"
        payload["strata"][0]["recommended"] = None

    with pytest.raises(CompareReportValidationError, match="inconclusive_reason"):
        CompareReport.from_dict(_mutate(_report(), mutate))


def test_an_inconclusive_stratum_may_not_also_recommend() -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["strata"][0]["outcome"] = "inconclusive"
        payload["strata"][0]["inconclusive_reason"] = "tied"

    with pytest.raises(CompareReportValidationError, match="recommended is set"):
        CompareReport.from_dict(_mutate(_report(), mutate))


def test_a_stratum_may_not_mix_objectives() -> None:
    def mutate(payload: dict[str, Any]) -> None:
        for system in (
            payload["strata"][0]["ranked"][0],
            payload["strata"][0]["recommended"],
        ):
            system["objective_name"] = "max_correct_cases_per_minute"

    with pytest.raises(CompareReportValidationError, match="mixes objectives"):
        CompareReport.from_dict(_mutate(_report(), mutate))


def test_a_stratum_objective_must_match_the_policy_objective() -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["strata"][0]["objective_name"] = "max_correct_cases_per_minute"
        payload["strata"][0]["ranked"][0][
            "objective_name"
        ] = "max_correct_cases_per_minute"
        payload["strata"][0]["recommended"][
            "objective_name"
        ] = "max_correct_cases_per_minute"

    with pytest.raises(CompareReportValidationError, match="one objective"):
        CompareReport.from_dict(_mutate(_report(), mutate))


def test_a_hosted_system_may_not_report_local_peak_memory() -> None:
    report = _report(
        strata=(_stratum(system=_system(_HOSTED, mean_peak_memory_bytes=1.0)),)
    )
    with pytest.raises(CompareReportValidationError, match="local-only measurement"):
        CompareReport.from_json(report.to_json())


def test_a_local_system_may_not_report_a_client_observed_ttft() -> None:
    """Being local and having a network-inclusive TTFT are contradictory."""
    report = _report(
        strata=(
            _stratum(
                system=_system(
                    _LOCAL,
                    mean_ttft_ms=220.0,
                    ttft_basis=TtftBasis.CLIENT_OBSERVED_STREAM,
                )
            ),
        )
    )
    with pytest.raises(CompareReportValidationError, match="contradict each other"):
        CompareReport.from_json(report.to_json())


def test_a_local_system_may_report_a_local_prefill_ttft() -> None:
    report = _report(
        strata=(
            _stratum(
                system=_system(
                    _LOCAL,
                    mean_ttft_ms=310.0,
                    ttft_basis=TtftBasis.LOCAL_PREFILL,
                )
            ),
        )
    )
    assert CompareReport.from_json(report.to_json()).to_dict() == report.to_dict()


def test_a_ttft_without_a_basis_is_refused() -> None:
    payload = _mutate(
        _report(), lambda p: p["strata"][0]["ranked"][0].update(mean_ttft_ms=220.0)
    )
    with pytest.raises(CompareReportValidationError, match="which measurement it is"):
        CompareReport.from_dict(payload)


def test_an_unknown_ttft_basis_is_refused() -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["strata"][0]["ranked"][0]["mean_ttft_ms"] = 220.0
        payload["strata"][0]["ranked"][0]["ttft_basis"] = "vibes"

    with pytest.raises(CompareReportValidationError, match="ttft_basis"):
        CompareReport.from_dict(_mutate(_report(), mutate))


def test_usage_may_not_claim_a_client_measurement() -> None:
    with pytest.raises(CompareReportValidationError, match="never a client"):
        UsageTotals.from_dict(
            {
                "provenance": "measured_wall_clock",
                "runs_reporting_usage": 1,
                "runs_total": 1,
            }
        )


def test_usage_reporting_more_runs_than_exist_is_refused() -> None:
    with pytest.raises(CompareReportValidationError, match="cannot exceed runs_total"):
        UsageTotals.from_dict({"runs_reporting_usage": 3, "runs_total": 1})


def test_cost_must_declare_itself_estimated() -> None:
    with pytest.raises(CompareReportValidationError, match="must be true"):
        CostSummary.from_dict(
            {
                "currency": "USD",
                "estimated": False,
                "pricing_entry_id": "e",
                "pricing_entry_sha256": "h",
                "rates_are_illustrative": True,
            }
        )


def test_a_persisted_report_must_carry_an_iso_currency_code() -> None:
    """Loading applies the manifest's own currency rule to a saved report."""
    with pytest.raises(CompareReportValidationError, match="ISO 4217"):
        CostSummary.from_dict(
            {
                "currency": "<script>alert(1)</script>",
                "estimated": True,
                "pricing_entry_id": "e",
                "pricing_entry_sha256": "h",
                "rates_are_illustrative": True,
            }
        )
    with pytest.raises(CompareReportValidationError, match="ISO 4217"):
        PricingProvenance.from_dict(
            {
                "manifest_path": "rates.json",
                "manifest_sha256": "h",
                "currency": "usd",
                "rates_are_illustrative": True,
            }
        )


def test_cost_must_declare_the_derivation_it_came_from() -> None:
    with pytest.raises(CompareReportValidationError, match="monetary_basis"):
        CostSummary.from_dict(
            {
                "currency": "USD",
                "estimated": True,
                "monetary_basis": "measured",
                "pricing_entry_id": "e",
                "pricing_entry_sha256": "h",
                "rates_are_illustrative": True,
            }
        )


def test_cost_without_pricing_provenance_is_refused() -> None:
    report = _report(
        strata=(
            _stratum(
                system=_system(
                    _HOSTED,
                    cost=CostSummary(
                        currency="USD",
                        pricing_entry_id="e",
                        pricing_entry_sha256="h",
                        rates_are_illustrative=True,
                        total_amount=1.0,
                    ),
                )
            ),
        )
    )
    with pytest.raises(CompareReportValidationError, match="unattributable"):
        CompareReport.from_json(report.to_json())


def test_currency_mixing_between_cost_and_manifest_is_refused() -> None:
    report = _report(
        pricing=PricingProvenance(
            manifest_path="rates.json",
            manifest_sha256="h",
            currency="USD",
            rates_are_illustrative=True,
        ),
        strata=(
            _stratum(
                system=_system(
                    _HOSTED,
                    cost=CostSummary(
                        currency="EUR",
                        pricing_entry_id="e",
                        pricing_entry_sha256="h",
                        rates_are_illustrative=True,
                        total_amount=1.0,
                    ),
                )
            ),
        ),
    )
    with pytest.raises(CompareReportValidationError, match="mixes currencies"):
        CompareReport.from_json(report.to_json())


def test_a_cost_objective_report_without_pricing_is_refused() -> None:
    report = _report(
        policy=ComparePolicy(objective=CompareObjective.MIN_COST_PER_CORRECT_CASE),
        strata=(
            _stratum(
                objective_name=CompareObjective.MIN_COST_PER_CORRECT_CASE.value,
                system=_system(
                    objective_name=CompareObjective.MIN_COST_PER_CORRECT_CASE.value
                ),
            ),
        ),
    )
    with pytest.raises(CompareReportValidationError, match="ranks on money"):
        CompareReport.from_json(report.to_json())


def test_a_rejected_system_must_record_a_reason() -> None:
    with pytest.raises(CompareReportValidationError, match="at least one rejection"):
        RejectedSystemReport.from_dict(
            {"system_key": _LOCAL.to_dict(), "run_ids": [], "reasons": []}
        )


def test_a_dominated_frontier_entry_must_name_its_dominator() -> None:
    with pytest.raises(CompareReportValidationError, match="names nothing"):
        FrontierEntry.from_dict(
            {"system_key": _LOCAL.to_dict(), "dominated": True, "dominated_by": []}
        )


def test_an_undominated_entry_may_not_name_dominators() -> None:
    with pytest.raises(CompareReportValidationError, match="yet names dominating"):
        FrontierEntry.from_dict(
            {
                "system_key": _LOCAL.to_dict(),
                "dominated": False,
                "dominated_by": ["something"],
            }
        )


def test_a_frontier_without_declared_axes_is_refused() -> None:
    def mutate(payload: dict[str, Any]) -> None:
        payload["strata"][0]["frontier"] = [
            {"system_key": _LOCAL.to_dict(), "dominated": False, "dominated_by": []}
        ]
        payload["strata"][0]["frontier_axes"] = []

    with pytest.raises(CompareReportValidationError, match="without naming the axes"):
        CompareReport.from_dict(_mutate(_report(), mutate))


def test_an_unknown_frontier_axis_is_refused() -> None:
    payload = _mutate(
        _report(), lambda p: p["strata"][0].update(frontier_axes=["max_vibes"])
    )
    with pytest.raises(CompareReportValidationError, match="unknown axis"):
        CompareReport.from_dict(payload)


# --- Policy ---------------------------------------------------------------


def test_policy_requires_an_objective() -> None:
    with pytest.raises(ComparePolicyError, match="'objective'"):
        ComparePolicy.from_dict({"constraints": {}})


def test_policy_rejects_an_unknown_objective() -> None:
    with pytest.raises(ComparePolicyError, match="invalid objective"):
        ComparePolicy.from_dict({"objective": "be_the_best"})


def test_policy_rejects_a_quality_floor_without_a_named_metric() -> None:
    with pytest.raises(ComparePolicyError, match="required_quality_metric"):
        CompareConstraints(min_quality_score=0.9)


def test_policy_rejects_a_disallowed_required_status() -> None:
    with pytest.raises(ComparePolicyError, match="may only contain"):
        CompareConstraints.from_dict({"required_statuses": ["failed"]})


def test_policy_rejects_a_non_finite_threshold() -> None:
    with pytest.raises(ComparePolicyError, match="must be > 0"):
        CompareConstraints.from_dict({"max_mean_total_latency_ms": float("inf")})


def test_policy_rejects_a_pass_rate_outside_the_unit_interval() -> None:
    with pytest.raises(ComparePolicyError, match=r"within \[0, 1\]"):
        CompareConstraints.from_dict({"min_pass_rate": 1.5})


def test_policy_rejects_zero_measured_repetitions() -> None:
    with pytest.raises(ComparePolicyError, match=">= 1"):
        CompareConstraints(min_measured_repetitions=0)


def test_policy_from_file_rejects_an_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "policy.txt"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ComparePolicyError, match="unsupported compare policy"):
        ComparePolicy.from_file(path)


def test_policy_round_trips_through_a_file(tmp_path: Path) -> None:
    policy = ComparePolicy(
        objective=CompareObjective.MAX_CORRECT_CASES_PER_CURRENCY_UNIT,
        name="round trip",
        constraints=CompareConstraints(min_pass_rate=0.8),
    )
    path = tmp_path / "policy.json"
    path.write_text(policy.to_json(), encoding="utf-8")
    assert ComparePolicy.from_file(path).to_dict() == policy.to_dict()


def test_cost_objectives_declare_that_they_need_money() -> None:
    assert CompareObjective.MIN_COST_PER_CORRECT_CASE.requires_cost is True
    assert CompareObjective.MAX_CORRECT_CASES_PER_CURRENCY_UNIT.requires_cost is True
    assert CompareObjective.MIN_MEAN_TOTAL_LATENCY_MS.requires_cost is False


def test_minimising_objectives_declare_their_direction() -> None:
    assert CompareObjective.MIN_MEAN_TOTAL_LATENCY_MS.prefers_lower is True
    assert CompareObjective.MAX_CORRECT_CASES_PER_MINUTE.prefers_lower is False
