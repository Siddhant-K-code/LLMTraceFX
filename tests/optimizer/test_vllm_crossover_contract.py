"""Strict offline contract tests for the Qwen3-8B vLLM crossover core."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from llmtracefx.optimizer.lab.qwen3_8b import vllm_compile as crossover


def _base_cumulative() -> list[float]:
    return [
        float((index + 1) * 10)
        for index in range(crossover.CONTROLLED_REQUESTS_PER_CELL)
    ]


def _make_pair(
    pair_id: str,
    order: str,
    difference_curve: Iterable[float],
) -> crossover.PairCurve:
    eager = _base_cumulative()
    differences = list(difference_curve)
    compiled = [
        eager_value + delta
        for eager_value, delta in zip(eager, differences, strict=True)
    ]
    return crossover.PairCurve(
        pair_id=pair_id,
        order=order,
        eager_cumulative=tuple(eager),
        compiled_cumulative=tuple(compiled),
    )


def _sustained_crossing_curve() -> list[float]:
    return [8.0, 6.0, 4.0, 2.0, -1.0] + [-1.0] * (
        crossover.CONTROLLED_REQUESTS_PER_CELL - 5
    )


def _transient_then_sustained_curve() -> list[float]:
    return [8.0, 4.0, -1.0, 3.0, 1.0, -2.0] + [-2.0] * (
        crossover.CONTROLLED_REQUESTS_PER_CELL - 6
    )


def _transient_censored_curve() -> list[float]:
    return [8.0, 4.0, -1.0, 3.0, 1.0, 2.0] + [2.0] * (
        crossover.CONTROLLED_REQUESTS_PER_CELL - 6
    )


def _no_crossing_curve() -> list[float]:
    return [8.0] * crossover.CONTROLLED_REQUESTS_PER_CELL


def _default_ledger(tmp_path: Path) -> tuple[crossover.LifecycleBudgetLedger, Path]:
    path = tmp_path / "ledger.json"
    ledger = crossover.LifecycleBudgetLedger.initialize(
        path,
        plan=crossover.build_default_plan(),
        git_head="a" * 40,
        workspace_path=tmp_path,
    )
    return ledger, path


def test_sampling_contracts_are_exact_and_canonical() -> None:
    assert crossover.PROTOCOL_ID == "qwen3-8b-vllm-crossover-v2"
    assert crossover.SCHEDULE_SEED == 20260904
    assert crossover.PAIRS_PER_LANE == 8
    assert crossover.LANES == ("controlled", "natural")
    assert crossover.CONTROLLED_REQUESTS_PER_CELL == 144
    assert crossover.NATURAL_REQUESTS_PER_CELL == 12
    assert crossover.REQUESTS_PER_CELL == 12
    plan = crossover.build_default_plan().to_dict()
    assert plan["execution_modes"] == crossover.EXECUTION_MODES
    assert plan["reproducibility"]["environment"] == crossover.DETERMINISTIC_ENVIRONMENT
    assert plan["lifecycle_controls"]["hidden_generation_warmups"] == 0
    assert plan["measurement_contract"]["gpu_memory"]["target_interval_ms"] == 200
    assert plan["measurement_contract"]["cuda_graph_capture_time"]["value"] is None
    assert crossover.CONTROLLED_SAMPLING.to_dict() == {
        "temperature": 0,
        "top_p": 1,
        "seed": 20260831,
        "n": 1,
        "best_of": 1,
        "max_tokens": 96,
        "min_tokens": 96,
        "ignore_eos": True,
        "stop": [],
        "detokenize": False,
    }
    assert crossover.NATURAL_SAMPLING.to_dict() == {
        "temperature": 0,
        "top_p": 1,
        "seed": 20260831,
        "n": 1,
        "best_of": 1,
        "max_tokens": 96,
        "min_tokens": 0,
        "ignore_eos": False,
        "stop": [],
        "detokenize": True,
    }
    assert crossover.canonical_json(crossover.CONTROLLED_SAMPLING.to_dict()) == (
        '{"best_of":1,"detokenize":false,"ignore_eos":true,"max_tokens":96,'
        '"min_tokens":96,"n":1,"seed":20260831,"stop":[],"temperature":0,'
        '"top_p":1}'
    )
    assert (
        crossover.SamplingContract.from_dict(
            crossover.CONTROLLED_SAMPLING.to_dict(),
            expected=crossover.CONTROLLED_SAMPLING,
        )
        == crossover.CONTROLLED_SAMPLING
    )
    with pytest.raises(crossover.VLLMCompileContractError, match="frozen canonical"):
        crossover.SamplingContract.from_dict(
            {**crossover.CONTROLLED_SAMPLING.to_dict(), "min_tokens": 95},
            expected=crossover.CONTROLLED_SAMPLING,
        )


def test_schedule_is_materialized_balanced_adjacent_and_roundtrips() -> None:
    schedule = crossover.crossover_schedule()
    assert schedule is crossover.CROSSOVER_SCHEDULE
    assert len(schedule) == 32
    assert "".join(crossover.lane_first_mode_symbols("controlled")) == "ABBABAAB"
    assert "".join(crossover.lane_first_mode_symbols("natural")) == "ABBABAAB"
    assert crossover.lane_pair_orders("controlled") == (
        "eager-compiled",
        "compiled-eager",
        "compiled-eager",
        "eager-compiled",
        "compiled-eager",
        "eager-compiled",
        "eager-compiled",
        "compiled-eager",
    )
    assert [cell.lane for cell in schedule[::2]] == [
        "natural",
        "controlled",
        "natural",
        "controlled",
        "natural",
        "controlled",
        "controlled",
        "natural",
        "natural",
        "controlled",
        "controlled",
        "natural",
        "controlled",
        "natural",
        "controlled",
        "natural",
    ]
    pattern = re.compile(
        rf"^{crossover.PROTOCOL_ID}-(controlled|natural)-pair-\d{{2}}-"
        r"period-0[12]-(eager|compiled)$"
    )
    for first, second in zip(schedule[0::2], schedule[1::2], strict=True):
        assert pattern.fullmatch(first.cell_id)
        assert pattern.fullmatch(second.cell_id)
        assert first.pair_id == second.pair_id
        assert first.lane == second.lane
        assert first.period_index == 1
        assert second.period_index == 2
        assert {first.mode, second.mode} == {"eager", "compiled"}
        assert crossover.ScheduleCell.from_dict(first.to_dict()) == first
        assert crossover.ScheduleCell.from_dict(second.to_dict()) == second
    with pytest.raises(crossover.VLLMCompileContractError, match="mode"):
        crossover.ScheduleCell.from_dict(
            {
                **schedule[0].to_dict(),
                "mode": "compiled",
                "cell_id": schedule[0].cell_id,
            }
        )


def test_default_plan_is_strict_offline_and_rejects_drift() -> None:
    plan = crossover.build_default_plan()
    roundtrip = crossover.VLLMCompilePlan.from_json(plan.to_json())
    assert roundtrip.to_dict() == plan.to_dict()
    assert plan.to_dict()["runtime"]["runtime_pins"] == crossover.RUNTIME_PINS
    assert (
        plan.to_dict()["runtime"]["vllm_source_commit"]
        == "2cf0a6915ce544dc493a0990f2ea38d81601128a"
    )
    assert plan.to_dict()["budget"]["summary"] == {
        "anticipated_rate_usd_per_hour": "0.39",
        "hard_cap_usd": "3",
        "active_planned_seconds": 19680,
        "active_planned_usd": "2.132",
        "untouched_margin_seconds": 8012,
        "untouched_margin_usd": "0.8679666666666666666666666667",
        "absolute_ceiling_seconds": 27692,
        "absolute_ceiling_usd": "2.999966666666666666666666667",
    }
    assert plan.to_dict()["quality_preservation"] == {
        "lane": "natural",
        "evaluator": "evaluate_workload",
        "independent_unit": "adjacent eager-compiled lifecycle pair",
        "effect": "compiled_minus_eager_request_success_rate",
        "noninferiority_margin": "0",
        "inference_method": (
            "deterministic whole-pair percentile bootstrap unless every pair "
            "effect is identical; identical effects are reported as a "
            "deterministic observed-workload fact without CI endpoints"
        ),
        "confidence_level": "0.95",
        "resamples": 20_000,
        "support_rule": (
            "lower confidence endpoint >= negative margin; when all pair effects "
            "are identical, the shared deterministic effect >= negative margin"
        ),
    }

    extra = plan.to_dict()
    extra["extra"] = True
    with pytest.raises(
        crossover.VLLMCompileContractError,
        match="keys must match exactly",
    ):
        crossover.VLLMCompilePlan.from_dict(extra)

    drift = plan.to_dict()
    drift["runtime"]["runtime_pins"]["python_version"] = "main"
    with pytest.raises(crossover.VLLMCompileContractError, match="frozen canonical"):
        crossover.VLLMCompilePlan.from_dict(drift)

    nonfinite = plan.to_dict()
    nonfinite["analysis_seed"] = float("nan")
    with pytest.raises(crossover.VLLMCompileContractError, match="invalid plan JSON"):
        crossover.VLLMCompilePlan.from_json(json.dumps(nonfinite))


def test_budget_math_and_historical_workload_contract_remain_exact() -> None:
    lines = {line.line_id: line.to_dict() for line in crossover.BUDGET_LINES}
    assert lines["preflight"]["amount_usd"] == "0.2925"
    assert lines["controlled-cell"]["amount_usd"] == "0.832"
    assert lines["natural-cell"]["amount_usd"] == "0.416"
    assert lines["reset"]["amount_usd"] == "0.2015"
    assert lines["export"]["amount_usd"] == "0.0975"
    assert lines["teardown"]["amount_usd"] == "0.2925"
    assert lines["untouched-margin"]["amount_usd"] == "0.8679666666666666666666666667"
    assert (
        sum(line.total_seconds for line in crossover.BUDGET_LINES if line.reservable)
        == 19680
    )
    assert sum(line.total_seconds for line in crossover.BUDGET_LINES) == 27692
    assert len(crossover.BUDGET_LIFECYCLES) == 66
    assert (
        len(
            [
                item
                for item in crossover.BUDGET_LIFECYCLES
                if item.line_id == "controlled-cell"
            ]
        )
        == 16
    )
    assert (
        len(
            [
                item
                for item in crossover.BUDGET_LIFECYCLES
                if item.line_id == "natural-cell"
            ]
        )
        == 16
    )
    assert (
        len([item for item in crossover.BUDGET_LIFECYCLES if item.line_id == "reset"])
        == 31
    )

    base = crossover.workload_descriptors()
    controlled = crossover.lane_request_descriptors("controlled")
    natural = crossover.lane_request_descriptors("natural")
    assert len(base) == 12
    assert base[0].request_id == "2k-structured-json-profile-extraction-rep-01"
    assert base[-1].request_id == "16k-prose-reasoning-two-train-gap-rep-02"
    assert len(controlled) == 144
    assert len(natural) == 12
    assert natural == base
    for offset in range(
        0,
        crossover.CONTROLLED_REQUESTS_PER_CELL,
        crossover.REQUESTS_PER_CELL,
    ):
        assert controlled[offset : offset + crossover.REQUESTS_PER_CELL] == base


def test_lifecycle_ledger_roundtrips_and_rejects_reset_duplicate_and_unknown(
    tmp_path: Path,
) -> None:
    ledger, _ = _default_ledger(tmp_path)
    preflight = crossover.BUDGET_LIFECYCLES[0]
    first_cell = next(
        lifecycle
        for lifecycle in crossover.BUDGET_LIFECYCLES
        if lifecycle.line_id in {"controlled-cell", "natural-cell"}
    )

    ledger.reserve(
        "preflight-run",
        line_id=preflight.line_id,
        lifecycle_id=preflight.lifecycle_id,
        ceiling_usd=preflight.ceiling_usd,
        argv=("timeout", "2700", "python3"),
        reserved_at="2026-09-04T00:00:00Z",
    )
    ledger.complete(
        "preflight-run",
        completed_at="2026-09-04T00:01:00Z",
        actual_seconds=60,
    )
    ledger.reserve(
        "cell-run",
        line_id=first_cell.line_id,
        lifecycle_id=first_cell.lifecycle_id,
        ceiling_usd=first_cell.ceiling_usd,
        argv=("timeout", str(first_cell.planned_seconds), "python3"),
        reserved_at="2026-09-04T00:02:00Z",
    )
    ledger.abort(
        "cell-run",
        aborted_at="2026-09-04T00:02:30Z",
        reason="manual-stop",
    )

    snapshot = ledger.snapshot()
    entries = {entry["lifecycle_id"]: entry for entry in snapshot["entries"]}
    assert entries[preflight.lifecycle_id]["status"] == "completed"
    assert entries[preflight.lifecycle_id]["actual_cost_usd"] == "0.0065"
    assert entries[first_cell.lifecycle_id]["status"] == "aborted"
    assert snapshot["reserved_usd"] == crossover.canonical_decimal(
        preflight.ceiling_usd + first_cell.ceiling_usd
    )

    with pytest.raises(crossover.VLLMCompileContractError, match="cannot be reset"):
        crossover.LifecycleBudgetLedger.initialize(
            tmp_path / "ledger.json",
            plan=crossover.build_default_plan(),
            git_head="a" * 40,
            workspace_path=tmp_path,
        )
    with pytest.raises(crossover.VLLMCompileContractError, match="already reserved"):
        ledger.reserve(
            "preflight-run",
            line_id=preflight.line_id,
            lifecycle_id=preflight.lifecycle_id,
            ceiling_usd=preflight.ceiling_usd,
            argv=("timeout", "2700"),
            reserved_at="2026-09-04T00:03:00Z",
        )
    with pytest.raises(crossover.VLLMCompileContractError, match="not reservable"):
        ledger.complete(
            "preflight-run",
            completed_at="2026-09-04T00:03:00Z",
            actual_seconds=1,
        )
    with pytest.raises(crossover.VLLMCompileContractError, match="unknown"):
        ledger.abort(
            "unknown-command",
            aborted_at="2026-09-04T00:03:00Z",
            reason="missing",
        )
    with pytest.raises(crossover.VLLMCompileContractError, match="timezone"):
        ledger.reserve(
            "naive-time",
            line_id=preflight.line_id,
            lifecycle_id=preflight.lifecycle_id,
            ceiling_usd=preflight.ceiling_usd,
            argv=("noop",),
            reserved_at="2026-09-04T00:03:00",
        )


def test_lifecycle_ledger_detects_tamper_and_rollback(tmp_path: Path) -> None:
    ledger, path = _default_ledger(tmp_path)
    preflight = crossover.BUDGET_LIFECYCLES[0]
    ledger.reserve(
        "preflight-run",
        line_id=preflight.line_id,
        lifecycle_id=preflight.lifecycle_id,
        ceiling_usd=preflight.ceiling_usd,
        argv=("timeout", "2700", "python3"),
        reserved_at="2026-09-04T00:00:00Z",
    )
    rollback_text = path.read_text(encoding="utf-8")
    ledger.complete(
        "preflight-run",
        completed_at="2026-09-04T00:01:00Z",
        actual_seconds=60,
    )

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["reserved_usd"] = "9"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(crossover.VLLMCompileContractError, match="integrity seal"):
        ledger.snapshot()

    path.write_text(rollback_text, encoding="utf-8")
    with pytest.raises(crossover.VLLMCompileContractError, match="rollback"):
        ledger.snapshot()


def test_lifecycle_ledger_rejects_over_cap(tmp_path: Path) -> None:
    huge_plan = SimpleNamespace(
        content_sha256="sha256:" + ("b" * 64),
        budget_lifecycles=(
            crossover.BudgetLifecycle(
                lifecycle_id="big-01",
                line_id="big",
                kind="cell",
                ordinal=1,
                planned_seconds=1,
                ceiling_usd=Decimal("2"),
            ),
            crossover.BudgetLifecycle(
                lifecycle_id="big-02",
                line_id="big",
                kind="cell",
                ordinal=2,
                planned_seconds=1,
                ceiling_usd=Decimal("2"),
            ),
        ),
    )
    ledger = crossover.LifecycleBudgetLedger.initialize(
        tmp_path / "over-cap-ledger.json",
        plan=huge_plan,
        git_head="a" * 40,
        workspace_path=tmp_path,
    )
    ledger.reserve(
        "big-01",
        line_id="big",
        lifecycle_id="big-01",
        ceiling_usd=Decimal("2"),
        argv=("python3",),
        reserved_at="2026-09-04T00:00:00Z",
    )
    with pytest.raises(crossover.VLLMCompileContractError, match="hard cap"):
        ledger.reserve(
            "big-02",
            line_id="big",
            lifecycle_id="big-02",
            ceiling_usd=Decimal("2"),
            argv=("python3",),
            reserved_at="2026-09-04T00:00:01Z",
        )


def test_paired_analysis_reports_crossings_censoring_and_deterministic_bootstrap() -> (
    None
):
    pairs = tuple(
        _make_pair(f"pair-{index:02d}", "eager-compiled", _sustained_crossing_curve())
        for index in range(1, crossover.PAIRS_PER_LANE + 1)
    )
    analysis = crossover.analyze_pair_curves(
        pairs,
        resample_count=64,
        analysis_seed=20260915,
    )
    assert (
        analysis.to_dict()
        == crossover.analyze_pair_curves(
            pairs,
            resample_count=64,
            analysis_seed=20260915,
        ).to_dict()
    )
    assert analysis.aggregate_first_crossing_request_count == 5
    assert analysis.aggregate_sustained_crossing_request_count == 5
    assert analysis.bootstrap_uncensored_resamples == 64
    assert analysis.bootstrap_censored_resamples == 0
    assert analysis.bootstrap_sustained_crossing_median_request_count == 5
    assert analysis.simultaneous_band_lower == analysis.mean_difference_curve
    assert analysis.simultaneous_band_upper == analysis.mean_difference_curve
    assert analysis.simultaneous_band_sustained_crossing_request_count == 5
    assert analysis.bootstrap_sustained_crossing_lower_is_open is False
    assert analysis.bootstrap_sustained_crossing_upper_is_open is False
    assert analysis.terminal_effect_sign_flip_p_value == pytest.approx(1 / 128)
    assert (
        json.loads(crossover.canonical_json(analysis.to_dict()))["resample_count"] == 64
    )

    mixed = (
        _make_pair("pair-a", "eager-compiled", _sustained_crossing_curve()),
        _make_pair("pair-b", "compiled-eager", _transient_then_sustained_curve()),
        _make_pair("pair-c", "eager-compiled", _transient_censored_curve()),
        _make_pair("pair-d", "compiled-eager", _no_crossing_curve()),
        _make_pair("pair-e", "eager-compiled", _sustained_crossing_curve()),
        _make_pair("pair-f", "compiled-eager", _transient_then_sustained_curve()),
        _make_pair("pair-g", "eager-compiled", _transient_censored_curve()),
        _make_pair("pair-h", "compiled-eager", _no_crossing_curve()),
    )
    effects = {
        effect.pair_id: effect
        for effect in crossover.analyze_pair_curves(
            mixed,
            resample_count=32,
            analysis_seed=20260916,
        ).pair_effects
    }
    assert effects["pair-a"].first_crossing_request_count == 5
    assert effects["pair-a"].sustained_crossing_request_count == 5
    assert effects["pair-b"].first_crossing_request_count == 3
    assert effects["pair-b"].sustained_crossing_request_count == 6
    assert effects["pair-c"].first_crossing_request_count == 3
    assert effects["pair-c"].sustained_crossing_request_count is None
    assert effects["pair-c"].right_censored is True
    assert effects["pair-d"].first_crossing_request_count is None
    assert effects["pair-d"].sustained_crossing_request_count is None
    assert effects["pair-d"].right_censored is True

    censored = tuple(
        _make_pair(
            f"censored-{index}",
            "eager-compiled" if index % 2 else "compiled-eager",
            _no_crossing_curve(),
        )
        for index in range(1, 9)
    )
    censored_analysis = crossover.analyze_pair_curves(
        censored,
        resample_count=32,
        analysis_seed=20260917,
    )
    assert censored_analysis.bootstrap_uncensored_resamples == 0
    assert censored_analysis.bootstrap_censored_resamples == 32
    assert censored_analysis.bootstrap_sustained_crossing_lower_request_count is None
    assert censored_analysis.bootstrap_sustained_crossing_upper_request_count is None
    assert censored_analysis.bootstrap_sustained_crossing_lower_is_open is True
    assert censored_analysis.bootstrap_sustained_crossing_upper_is_open is True


def test_claim_gate_is_mechanical_and_blocks_forward_pass_identity() -> None:
    gate = crossover.ClaimGate(
        terminal=True,
        completeness=True,
        fixed_count=True,
        controlled_supported_crossing=True,
        controlled_output_identity=True,
        controlled_numeric_reproducibility=True,
        natural_output_identity=True,
        natural_numeric_reproducibility=True,
        natural_absolute_correctness=True,
        natural_supported_speedup=True,
        component_observability=False,
    )
    states = {decision.claim_id: decision.state for decision in gate.matrix()}
    assert states["fixed-token-count-crossover"] == "supported"
    assert states["output-identical-generation-crossover"] == "supported"
    assert states["numerically-reproducible-generation-crossover"] == "supported"
    assert states["natural-output-quality-preserved"] == "supported"
    assert states["natural-end-to-end-causal-speedup"] == "supported"
    assert states["compile-cuda-graph-component-timing"] == "unsupported"
    assert states["forward-pass-identical"] == "not_applicable"
    assert gate.evaluate("compile-cuda-graph-component-timing").blockers == (
        "component_observability",
    )

    degraded = crossover.ClaimGate(
        terminal=False,
        completeness=False,
        fixed_count=False,
        controlled_supported_crossing=False,
        controlled_output_identity=False,
        controlled_numeric_reproducibility=False,
        natural_output_identity=False,
        natural_numeric_reproducibility=False,
        natural_absolute_correctness=False,
        natural_supported_speedup=False,
        component_observability=False,
    )
    assert degraded.evaluate("forward-pass-identical").state == "not_applicable"
    assert degraded.evaluate("natural-end-to-end-causal-speedup").blockers == (
        "terminal",
        "completeness",
        "natural_output_identity",
        "natural_numeric_reproducibility",
        "natural_absolute_correctness",
        "natural_supported_speedup",
    )
    with pytest.raises(crossover.VLLMCompileContractError, match="unknown claim_id"):
        degraded.evaluate("impossible-claim")
