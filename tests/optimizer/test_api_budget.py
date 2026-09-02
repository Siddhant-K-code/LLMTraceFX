"""Fail-closed tests for the hosted API lifetime budget ledger."""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

from llmtracefx.optimizer.collectors.openai_api import ProviderUsage
from llmtracefx.optimizer.workloads.api_budget import (
    BudgetError,
    BudgetGate,
    BudgetPlan,
)


def request(
    request_id: str,
    *,
    model_id: str = "z-ai/glm-5.3",
    prompt_rate: str = "0.0000014",
    completion_rate: str = "0.0000044",
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "model_id": model_id,
        "workload_id": "structured-json-profile-extraction",
        "workload_version": "1",
        "prompt_sha256": "sha256:prompt",
        "request_config_sha256": "sha256:config",
        "endpoint_origin": "https://openrouter.ai",
        "endpoint_path": "/api/v1/chat/completions",
        "route_providers": ["z-ai/fp8"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "max_provider_prompt_price_per_million": "1.4",
        "max_provider_completion_price_per_million": "4.4",
        "reasoning_effort": "low",
        "input_token_ceiling": 10_000,
        "max_output_tokens": 96,
        "prompt_usd_per_token": prompt_rate,
        "completion_usd_per_token": completion_rate,
        "cached_prompt_usd_per_token": "0.00000026",
        "cache_write_billing": "included_in_uncached_prompt_rate",
        "reasoning_billing": "included_in_completion_tokens",
    }


def write_plan(
    path: Path,
    requests: list[dict[str, object]],
    *,
    authorized: str = "5.00",
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "2",
                "experiment_id": "hosted-budget-test",
                "ledger_file_name": "ledger.json",
                "authorized_total_usd": authorized,
                "requests": requests,
            }
        ),
        encoding="utf-8",
    )
    return path


def claim(
    gate: BudgetGate,
    request_id: str,
    *,
    model_id: str = "z-ai/glm-5.3",
    workload_id: str = "structured-json-profile-extraction",
    workload_version: str = "1",
    prompt_sha256: str = "sha256:prompt",
    request_config_sha256: str = "sha256:config",
    endpoint_origin: str = "https://openrouter.ai",
    endpoint_path: str = "/api/v1/chat/completions",
    route_providers: tuple[str, ...] = ("z-ai/fp8",),
    allow_fallbacks: bool = False,
    require_parameters: bool = True,
    max_provider_prompt_price_per_million: Decimal = Decimal("1.4"),
    max_provider_completion_price_per_million: Decimal = Decimal("4.4"),
    reasoning_effort: str = "low",
    input_token_upper_bound: int = 9_000,
    max_output_tokens: int = 96,
) -> None:
    gate.claim(
        request_id,
        model_id=model_id,
        workload_id=workload_id,
        workload_version=workload_version,
        prompt_sha256=prompt_sha256,
        request_config_sha256=request_config_sha256,
        endpoint_origin=endpoint_origin,
        endpoint_path=endpoint_path,
        route_providers=route_providers,
        allow_fallbacks=allow_fallbacks,
        require_parameters=require_parameters,
        max_provider_prompt_price_per_million=max_provider_prompt_price_per_million,
        max_provider_completion_price_per_million=(
            max_provider_completion_price_per_million
        ),
        reasoning_effort=reasoning_effort,
        input_token_upper_bound=input_token_upper_bound,
        max_output_tokens=max_output_tokens,
    )


def test_five_dollar_lifetime_cap_is_hard(tmp_path: Path) -> None:
    plan = write_plan(tmp_path / "plan.json", [request("one")], authorized="5.01")

    with pytest.raises(BudgetError, match="hard lifetime cap"):
        BudgetPlan.read(plan)


def test_cumulative_request_ceilings_must_fit_authorization(tmp_path: Path) -> None:
    expensive = request("one", prompt_rate="0.0005", completion_rate="0.001")
    plan = write_plan(tmp_path / "plan.json", [expensive], authorized="5.00")

    with pytest.raises(BudgetError, match="planned worst-case cost"):
        BudgetPlan.read(plan)


def test_claim_is_atomic_sealed_and_cannot_be_retried(tmp_path: Path) -> None:
    plan = write_plan(tmp_path / "plan.json", [request("one")])
    ledger = tmp_path / "ledger.json"
    gate = BudgetGate.initialize(plan, ledger)

    claim(gate, "one")

    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["ledger_sha256"].startswith("sha256:")
    assert payload["entries"][0]["status"] == "attempted"
    assert payload["events"][0]["stage"] == "pre_request"
    with pytest.raises(BudgetError, match="already attempted"):
        claim(gate, "one")


def test_failed_request_keeps_ceiling_and_stops_the_model(tmp_path: Path) -> None:
    requests = [request("first"), request("second")]
    gate = BudgetGate.initialize(
        write_plan(tmp_path / "plan.json", requests), tmp_path / "ledger.json"
    )
    claim(gate, "first")
    gate.settle(
        "first",
        provider_success=False,
        usage=ProviderUsage(reported=False),
        failure="timeout",
    )

    snapshot = gate.snapshot()
    first = snapshot["entries"][0]
    assert first["status"] == "failed"
    assert first["accounted_cost_usd"] == first["ceiling_usd"]
    with pytest.raises(BudgetError, match="stopped this model"):
        claim(gate, "second")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("cache_write_billing", None, "cache_write_billing"),
        ("cache_write_billing", "free", "unknown cache-write billing"),
        ("reasoning_billing", None, "reasoning_billing"),
        ("reasoning_billing", "free", "unknown reasoning billing"),
    ],
)
def test_missing_or_ambiguous_billing_categories_refuse(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    planned = request("one")
    if value is None:
        planned.pop(field)
    else:
        planned[field] = value
    plan = write_plan(tmp_path / "plan.json", [planned])

    with pytest.raises(BudgetError, match=message):
        BudgetPlan.read(plan)


def test_tampered_ledger_seal_refuses_before_another_claim(tmp_path: Path) -> None:
    plan = write_plan(tmp_path / "plan.json", [request("one"), request("two")])
    ledger = tmp_path / "ledger.json"
    gate = BudgetGate.initialize(plan, ledger)
    claim(gate, "one")
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["remaining_authorized_usd"] = "5.000000000000"
    ledger.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BudgetError, match="integrity seal"):
        claim(gate, "two")


def test_request_binding_must_match_exactly(tmp_path: Path) -> None:
    gate = BudgetGate.initialize(
        write_plan(tmp_path / "plan.json", [request("one")]),
        tmp_path / "ledger.json",
    )

    with pytest.raises(BudgetError, match="plan binding"):
        claim(gate, "one", prompt_sha256="sha256:different")
    with pytest.raises(BudgetError, match="plan binding"):
        claim(gate, "one", request_config_sha256="sha256:different")
    with pytest.raises(BudgetError, match="plan binding"):
        claim(gate, "one", route_providers=("unrestricted",))
    with pytest.raises(BudgetError, match="token ceiling"):
        claim(gate, "one", input_token_upper_bound=10_001)


def test_success_settles_to_provider_cost_but_keeps_computed_cost_separate(
    tmp_path: Path,
) -> None:
    gate = BudgetGate.initialize(
        write_plan(tmp_path / "plan.json", [request("one")]),
        tmp_path / "ledger.json",
    )
    claim(gate, "one")
    gate.settle(
        "one",
        provider_success=True,
        usage=ProviderUsage(
            reported=True,
            prompt_tokens=1_000,
            completion_tokens=50,
            cached_prompt_tokens=200,
            reasoning_tokens=20,
            cost_usd=0.002,
        ),
        failure=None,
    )

    entry = gate.snapshot()["entries"][0]
    assert entry["status"] == "completed"
    assert entry["provider_reported_cost_usd_credits"] == "0.002000000000"
    assert entry["computed_observed_cost_usd"] == "0.001392000000"
    assert entry["accounted_cost_usd"] == "0.002000000000"


def test_missing_cached_usage_makes_computed_cost_unavailable_not_zero(
    tmp_path: Path,
) -> None:
    gate = BudgetGate.initialize(
        write_plan(tmp_path / "plan.json", [request("one")]),
        tmp_path / "ledger.json",
    )
    claim(gate, "one")
    gate.settle(
        "one",
        provider_success=True,
        usage=ProviderUsage(
            reported=True,
            prompt_tokens=1_000,
            completion_tokens=50,
            cached_prompt_tokens=None,
            cost_usd=0.002,
        ),
        failure=None,
    )

    assert gate.snapshot()["entries"][0]["computed_observed_cost_usd"] is None


def test_execution_refuses_to_initialize_or_reset_a_missing_ledger(
    tmp_path: Path,
) -> None:
    plan = write_plan(tmp_path / "plan.json", [request("one")])
    ledger = tmp_path / "ledger.json"

    with pytest.raises(BudgetError, match="never initializes or resets"):
        BudgetGate(plan, ledger)
    initialized = BudgetGate.initialize(plan, ledger)
    claim(initialized, "one")
    ledger.unlink()
    with pytest.raises(BudgetError, match="never initializes or resets"):
        BudgetGate(plan, ledger)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "max_provider_prompt_price_per_million",
            "1.41",
            "prompt price cap exceeds",
        ),
        (
            "max_provider_completion_price_per_million",
            "4.41",
            "completion price cap exceeds",
        ),
    ],
)
def test_provider_price_caps_cannot_exceed_planned_rates(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    planned = request("one")
    planned[field] = value

    with pytest.raises(BudgetError, match=message):
        BudgetPlan.read(write_plan(tmp_path / "plan.json", [planned]))
