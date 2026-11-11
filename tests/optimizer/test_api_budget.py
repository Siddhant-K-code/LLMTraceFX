"""Fail-closed tests for the hosted API lifetime budget ledger."""

from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer.collectors.openai_api import ProviderUsage
from llmtracefx.optimizer.workloads import api_budget
from llmtracefx.optimizer.workloads.api_budget import (
    BudgetError,
    BudgetGate,
    BudgetPlan,
)


@pytest.fixture(autouse=True)
def isolate_authorization_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        api_budget,
        "_default_authorization_state_dir",
        lambda: tmp_path / "state",
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
    ledger_path = (path.parent / "ledger.json").resolve()
    payload = {
        "schema_version": "3",
        "experiment_id": "hosted-budget-test",
        "ledger_identity": "hosted-budget-test-ledger",
        "ledger_file_name": ledger_path.name,
        "ledger_path_sha256": (
            "sha256:" + hashlib.sha256(str(ledger_path).encode()).hexdigest()
        ),
        "authorized_total_usd": authorized,
        "requests": requests,
    }
    payload["plan_sha256"] = (
        "sha256:"
        + hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
    )
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    return path


def initialize_gate(plan: Path, ledger: Path) -> BudgetGate:
    return BudgetGate.initialize(plan, ledger)


def open_gate(plan: Path, ledger: Path) -> BudgetGate:
    return BudgetGate(plan, ledger)


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
    gate = initialize_gate(plan, ledger)

    claim(gate, "one")

    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["ledger_sha256"].startswith("sha256:")
    assert payload["entries"][0]["status"] == "attempted"
    assert payload["events"][0]["stage"] == "pre_request"
    with pytest.raises(BudgetError, match="already attempted"):
        claim(gate, "one")


def test_failed_request_keeps_ceiling_and_stops_the_model(tmp_path: Path) -> None:
    requests = [request("first"), request("second")]
    gate = initialize_gate(
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
    gate = initialize_gate(plan, ledger)
    claim(gate, "one")
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["remaining_authorized_usd"] = "5.000000000000"
    ledger.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(BudgetError, match="integrity seal"):
        claim(gate, "two")


def test_request_binding_must_match_exactly(tmp_path: Path) -> None:
    gate = initialize_gate(
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


@pytest.mark.parametrize(
    "override",
    [
        {"route_providers": ()},
        {"route_providers": ("z-ai",)},
        {"allow_fallbacks": True},
        {"endpoint_origin": "https://example.invalid"},
        {"endpoint_path": "/different"},
        {"reasoning_effort": "max"},
        {"max_provider_prompt_price_per_million": Decimal("1.39")},
        {"max_provider_completion_price_per_million": Decimal("4.39")},
    ],
)
def test_pricing_critical_runtime_mutations_refuse(
    tmp_path: Path, override: dict[str, Any]
) -> None:
    plan = write_plan(tmp_path / "plan.json", [request("one")])
    gate = initialize_gate(plan, tmp_path / "ledger.json")

    with pytest.raises(BudgetError, match="plan binding"):
        claim(gate, "one", **override)


def test_success_settles_to_provider_cost_but_keeps_computed_cost_separate(
    tmp_path: Path,
) -> None:
    gate = initialize_gate(
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
    gate = initialize_gate(
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
        open_gate(plan, ledger)
    initialized = initialize_gate(plan, ledger)
    claim(initialized, "one")
    ledger.unlink()
    with pytest.raises(BudgetError, match="never initializes or resets"):
        open_gate(plan, ledger)
    with pytest.raises(BudgetError, match="cannot be reset"):
        initialize_gate(plan, ledger)


def test_plan_seal_binds_the_only_ledger_path(tmp_path: Path) -> None:
    plan = write_plan(tmp_path / "plan.json", [request("one")])
    ledger = tmp_path / "ledger.json"
    initialize_gate(plan, ledger)

    alternate = tmp_path / "alternate" / "ledger.json"
    with pytest.raises(BudgetError, match="path does not match"):
        BudgetGate(plan, alternate)

    copied_plan = tmp_path / "alternate" / "plan.json"
    copied_plan.parent.mkdir()
    copied_plan.write_bytes(plan.read_bytes())
    with pytest.raises(BudgetError, match="path does not match"):
        BudgetGate(copied_plan, alternate)


def test_rollback_to_an_earlier_valid_ledger_is_refused(tmp_path: Path) -> None:
    plan = write_plan(tmp_path / "plan.json", [request("one")])
    ledger = tmp_path / "ledger.json"
    gate = initialize_gate(plan, ledger)
    initial = ledger.read_bytes()
    claim(gate, "one")
    ledger.write_bytes(initial)

    with pytest.raises(BudgetError, match="monotonic anchor"):
        open_gate(plan, ledger)


def test_over_ceiling_charge_is_persisted_and_globally_terminal(
    tmp_path: Path,
) -> None:
    plan = write_plan(tmp_path / "plan.json", [request("one"), request("two")])
    ledger = tmp_path / "ledger.json"
    gate = initialize_gate(plan, ledger)
    claim(gate, "one")

    with pytest.raises(BudgetError, match="provider-reported cost exceeded"):
        gate.settle(
            "one",
            provider_success=True,
            usage=ProviderUsage(
                reported=True,
                prompt_tokens=1,
                completion_tokens=1,
                cached_prompt_tokens=0,
                reasoning_tokens=0,
                cost_usd=6.0,
            ),
            failure=None,
        )

    snapshot = gate.snapshot()
    assert snapshot["entries"][0]["status"] == "failed"
    assert snapshot["terminal_failure"]["request_id"] == "one"
    assert snapshot["remaining_authorized_usd"].startswith("-")
    with pytest.raises(BudgetError, match="terminally failed"):
        claim(gate, "two")


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


def _invoke_cli(argv: list[str]) -> int:
    from llmtracefx.optimizer import cli

    parser = cli.build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def test_unsettled_attempt_blocks_its_model_but_not_other_models(
    tmp_path: Path,
) -> None:
    requests = [
        request("glm-first"),
        request("glm-second"),
        request("flash-first", model_id="z-ai/glm-5.3-flash"),
    ]
    plan = write_plan(tmp_path / "plan.json", requests)
    ledger = tmp_path / "ledger.json"
    claim(initialize_gate(plan, ledger), "glm-first")

    # A fresh gate stands in for the next process after a crash between
    # claim and settle.
    restarted = open_gate(plan, ledger)
    with pytest.raises(BudgetError, match="never settled") as excinfo:
        claim(restarted, "glm-second")
    assert "glm-first" in str(excinfo.value)
    assert "abandon-budget-request" in str(excinfo.value)
    assert restarted.snapshot()["entries"][1]["status"] == "planned"

    claim(restarted, "flash-first", model_id="z-ai/glm-5.3-flash")
    assert restarted.snapshot()["entries"][2]["status"] == "attempted"


def test_operator_abandon_unblocks_the_model_and_keeps_the_charge(
    tmp_path: Path,
) -> None:
    plan = write_plan(tmp_path / "plan.json", [request("first"), request("second")])
    ledger = tmp_path / "ledger.json"
    gate = initialize_gate(plan, ledger)
    claim(gate, "first")
    before = gate.snapshot()

    gate.abandon("first", reason="  runner was killed by OOM  ")

    after = gate.snapshot()
    first = after["entries"][0]
    assert first["status"] == "abandoned"
    assert first["accounted_cost_usd"] == first["ceiling_usd"]
    assert first["abandonment"]["reason"] == "runner was killed by OOM"
    assert first["settled_at"] is None
    assert after["cumulative_accounted_usd"] == before["cumulative_accounted_usd"]
    assert after["remaining_authorized_usd"] == before["remaining_authorized_usd"]
    assert after["revision"] == before["revision"] + 1
    event = after["events"][-1]
    assert event["stage"] == "operator_abandon"
    assert event["at"] == first["abandonment"]["at"]
    assert event["request_id"] == "first"
    assert event["model_id"] == "z-ai/glm-5.3"
    assert event["reason"] == "runner was killed by OOM"

    with pytest.raises(BudgetError, match="can never be retried"):
        claim(gate, "first")
    with pytest.raises(BudgetError, match="this request is abandoned"):
        gate.abandon("first", reason="again")

    claim(open_gate(plan, ledger), "second")
    assert gate.snapshot()["entries"][1]["status"] == "attempted"


def test_operator_abandon_refuses_entries_that_are_not_attempted(
    tmp_path: Path,
) -> None:
    plan = write_plan(
        tmp_path / "plan.json",
        [request("planned"), request("done"), request("broken", model_id="m/x")],
    )
    ledger = tmp_path / "ledger.json"
    gate = initialize_gate(plan, ledger)
    claim(gate, "done")
    gate.settle(
        "done",
        provider_success=True,
        usage=ProviderUsage(reported=True, cost_usd=0.001),
        failure=None,
    )
    claim(gate, "broken", model_id="m/x")
    gate.settle(
        "broken",
        provider_success=False,
        usage=None,
        failure="timeout",
    )
    revision = gate.snapshot()["revision"]

    with pytest.raises(BudgetError, match="this request is planned"):
        gate.abandon("planned", reason="no")
    with pytest.raises(BudgetError, match="this request is completed"):
        gate.abandon("done", reason="no")
    with pytest.raises(BudgetError, match="this request is failed"):
        gate.abandon("broken", reason="no")
    with pytest.raises(BudgetError, match="not authorized"):
        gate.abandon("unknown", reason="no")
    with pytest.raises(BudgetError, match="non-empty reason"):
        gate.abandon("planned", reason="   ")
    assert gate.snapshot()["revision"] == revision


def test_failed_request_still_blocks_after_an_abandon_for_that_model(
    tmp_path: Path,
) -> None:
    plan = write_plan(
        tmp_path / "plan.json",
        [request("crashed"), request("failed"), request("later")],
    )
    gate = initialize_gate(plan, tmp_path / "ledger.json")
    claim(gate, "crashed")
    gate.abandon("crashed", reason="process died")
    claim(gate, "failed")
    gate.settle("failed", provider_success=False, usage=None, failure="timeout")

    with pytest.raises(BudgetError, match="stopped this model"):
        claim(gate, "later")


def test_abandoned_entry_with_a_refund_is_refused_on_read(tmp_path: Path) -> None:
    plan = write_plan(tmp_path / "plan.json", [request("one")])
    ledger = tmp_path / "ledger.json"
    gate = initialize_gate(plan, ledger)
    claim(gate, "one")
    gate.abandon("one", reason="process died")
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    payload["entries"][0]["accounted_cost_usd"] = "0.000000000000"
    payload.pop("ledger_sha256")
    sealed = api_budget._sealed(payload)
    ledger.write_text(json.dumps(sealed), encoding="utf-8")
    anchor = gate._anchor_payload(sealed)
    gate.anchor_path.write_text(
        json.dumps(api_budget._sealed(anchor, seal_field="anchor_sha256")),
        encoding="utf-8",
    )

    with pytest.raises(BudgetError, match="must keep its full ceiling"):
        open_gate(plan, ledger)


def test_ledger_written_before_the_abandoned_status_still_loads(
    tmp_path: Path,
) -> None:
    plan = write_plan(
        tmp_path / "plan.json",
        [request("done"), request("broken", model_id="m/x"), request("open")],
    )
    ledger = tmp_path / "ledger.json"
    gate = initialize_gate(plan, ledger)
    claim(gate, "done")
    gate.settle(
        "done",
        provider_success=True,
        usage=ProviderUsage(reported=True, cost_usd=0.001),
        failure=None,
    )
    claim(gate, "broken", model_id="m/x")
    gate.settle("broken", provider_success=False, usage=None, failure="timeout")
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert {entry["status"] for entry in payload["entries"]} == {
        "completed",
        "failed",
        "planned",
    }
    assert all("abandonment" not in entry for entry in payload["entries"])
    assert all(event["stage"] != "operator_abandon" for event in payload["events"])

    reopened = open_gate(plan, ledger)
    assert reopened.snapshot() == payload
    claim(reopened, "open")


def test_abandon_budget_request_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = write_plan(tmp_path / "plan.json", [request("first"), request("second")])
    ledger = tmp_path / "ledger.json"
    claim(initialize_gate(plan, ledger), "first")
    argv = [
        "workloads",
        "abandon-budget-request",
        "--budget-plan",
        str(plan),
        "--budget-ledger",
        str(ledger),
        "--budget-request-id",
        "first",
        "--reason",
        "runner host was rebooted",
    ]

    assert _invoke_cli(argv) == 0
    out = capsys.readouterr().out
    assert "Abandoned first for z-ai/glm-5.3" in out
    assert "can never be retried" in out
    payload = json.loads(ledger.read_text(encoding="utf-8"))
    assert payload["entries"][0]["status"] == "abandoned"
    assert payload["events"][-1]["reason"] == "runner host was rebooted"

    assert _invoke_cli(argv) == 1
    assert "this request is abandoned" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _invoke_cli(argv[:-2])
