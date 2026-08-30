"""Plan adjudication: what blocks a deployment, and what that withholds."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from _fakes import PINNED_IMAGE, TAG_ONLY_IMAGE, VALID_REVISION

from llmtracefx.deploy.budget import BudgetRequest
from llmtracefx.deploy.endpoint import EndpointConfig
from llmtracefx.deploy.errors import DeploymentPlanError
from llmtracefx.deploy.plan import (
    MAX_SCALEDOWN_WINDOW_SECONDS,
    RuntimeControls,
    assert_executable,
    build_plan,
)
from llmtracefx.deploy.pricing import ComputeQuote, GpuPriceQuote, StorageQuote
from llmtracefx.deploy.recipe import SGLANG, VLLM, build_recipe

AS_OF = date(2026, 8, 30)


def _compute(effective_date: str = "2026-08-01") -> ComputeQuote:
    return ComputeQuote(
        usd_per_cpu_core_hour=1e-9,
        usd_per_gib_memory_hour=1e-9,
        effective_date=effective_date,
        source="https://modal.com/pricing",
    )


def _storage(effective_date: str = "2026-08-01") -> StorageQuote:
    return StorageQuote(
        usd_per_gib_month=1e-9,
        effective_date=effective_date,
        source="https://modal.com/pricing",
    )


def make_recipe(**overrides: object):  # type: ignore[no-untyped-def]
    kwargs: dict[str, object] = {
        "framework": VLLM,
        "framework_version": "0.5.6",
        "image_reference": PINNED_IMAGE,
        "model_revision": VALID_REVISION,
        "gpu_type": "H200",
        "gpu_count": 4,
        "context_length": 131072,
        "weights_mount_path": "/weights",
        "port": 30000,
    }
    kwargs.update(overrides)
    return build_recipe(**kwargs)  # type: ignore[arg-type]


def make_plan(
    *,
    recipe_overrides: dict[str, object] | None = None,
    max_usd: float = 10.0,
    rate: float = 1.0,
    price_date: str = "2026-08-01",
    timeout: int = 1800,
    max_containers: int = 1,
    controls_overrides: dict[str, object] | None = None,
    **plan_kwargs: object,
):  # type: ignore[no-untyped-def]
    recipe = make_recipe(**(recipe_overrides or {}))
    controls_kwargs: dict[str, object] = {
        "timeout_seconds": timeout,
        "deployment_seconds": timeout,
        "startup_timeout_seconds": 900,
        "max_containers": max_containers,
    }
    controls_kwargs.update(controls_overrides or {})
    budget = BudgetRequest(
        max_usd=max_usd,
        gpu_type=recipe.gpu_type,
        gpu_count=recipe.gpu_count,
        max_runtime_seconds=timeout,
        deployment_seconds=controls_kwargs["deployment_seconds"],  # type: ignore[arg-type]
        price=GpuPriceQuote(
            gpu_type=recipe.gpu_type,
            usd_per_gpu_hour=rate,
            effective_date=price_date,
            source="https://modal.com/pricing",
        ),
        compute=_compute(price_date),
        storage=_storage(price_date),
        stored_gib=306.0,
        storage_retention_days=0.0,
        max_containers=max_containers,
    )
    return build_plan(
        recipe=recipe,
        controls=RuntimeControls(**controls_kwargs),  # type: ignore[arg-type]
        budget=budget,
        endpoint=EndpointConfig(),
        as_of=AS_OF,
        **plan_kwargs,  # type: ignore[arg-type]
    )


def test_a_sound_plan_is_approved_and_every_step_is_executable() -> None:
    plan = make_plan()
    assert plan.approved is True
    assert plan.blockers == ()
    assert len(plan.executable_steps) == len(plan.steps)
    assert_executable(plan)


def test_the_plan_asserts_it_touched_nothing() -> None:
    payload = make_plan().to_dict()
    assert payload["network_request_performed"] is False
    assert payload["gpu_allocated"] is False
    assert payload["modal_authentication_used"] is False


def test_default_collect_step_uses_a_committed_prompt() -> None:
    collect = next(step for step in make_plan().steps if step.name == "collect")
    prompt = collect.argv[collect.argv.index("--prompt-file") + 1]
    assert isinstance(prompt, str)
    assert Path(prompt).is_file()


def test_over_budget_withholds_every_paid_step() -> None:
    plan = make_plan(max_usd=1.0, rate=10.0)
    assert plan.approved is False
    assert any("exceeds the authorised budget" in b for b in plan.blockers)

    withheld = {step.name for step in plan.steps} - {
        step.name for step in plan.executable_steps
    }
    assert {"stage-weights", "deploy", "smoke", "collect"} <= withheld
    assert all(not step.spends_money for step in plan.executable_steps)
    with pytest.raises(DeploymentPlanError, match="refused"):
        assert_executable(plan)


def test_teardown_stays_available_while_a_plan_is_refused() -> None:
    """Stopping and deleting must never be gated on approval.

    They are the actions that make spending stop, so a plan that
    withheld them would be at its most restrictive exactly when an
    operator most needs to shut something down.
    """
    plan = make_plan(max_usd=1.0, rate=10.0)
    available = {step.name for step in plan.executable_steps}
    assert {"stop", "delete-volume"} <= available


def test_insufficient_accelerators_block_the_plan() -> None:
    plan = make_plan(recipe_overrides={"gpu_count": 2}, max_usd=100.0)
    assert plan.approved is False
    assert any("KV cache and activations" in b for b in plan.blockers)


def test_a_stale_price_blocks_until_it_is_explicitly_accepted() -> None:
    stale = make_plan(price_date="2025-01-01")
    assert stale.approved is False
    assert any("days old" in b for b in stale.blockers)

    accepted = make_plan(price_date="2025-01-01", accept_stale_price=True)
    assert accepted.approved is True
    assert any("Accepted explicitly" in w for w in accepted.warnings)


def test_a_future_dated_price_blocks_and_cannot_be_accepted_away() -> None:
    plan = make_plan(price_date="2027-01-01", accept_stale_price=True)
    assert plan.approved is False
    assert any("in the future" in b for b in plan.blockers)


def test_a_tighter_freshness_limit_can_be_requested() -> None:
    plan = make_plan(price_date="2026-08-01", max_price_age_days=7)
    assert plan.approved is False
    assert any("limit 7 days" in b for b in plan.blockers)


def test_a_tag_only_image_warns_but_does_not_block() -> None:
    plan = make_plan(
        recipe_overrides={
            "image_reference": TAG_ONLY_IMAGE,
            "accept_mutable_image": True,
        }
    )
    assert plan.approved is True
    assert any("pinned by tag only" in w for w in plan.warnings)


def test_storage_is_a_mandatory_part_of_the_total() -> None:
    """It cannot be omitted, and the envelope says what it covers."""
    envelope = make_plan().envelope.to_dict()
    assert any(line["name"] == "storage" for line in envelope["lines"])
    assert "after deletion" in envelope["covers"]


def test_budget_and_recipe_must_describe_the_same_deployment() -> None:
    recipe = make_recipe()
    controls = RuntimeControls(
        timeout_seconds=1800, deployment_seconds=1800, startup_timeout_seconds=900
    )
    price = GpuPriceQuote(
        gpu_type="H200",
        usd_per_gpu_hour=1.0,
        effective_date="2026-08-01",
        source="pricing",
    )
    mismatched = BudgetRequest(
        max_usd=100.0,
        gpu_type="H200",
        gpu_count=2,
        max_runtime_seconds=1800,
        deployment_seconds=1800,
        price=price,
        compute=_compute(),
        storage=_storage(),
        stored_gib=306.0,
        storage_retention_days=0.0,
    )
    with pytest.raises(DeploymentPlanError, match="budget covers 2 GPU"):
        build_plan(
            recipe=recipe,
            controls=controls,
            budget=mismatched,
            endpoint=EndpointConfig(),
            as_of=AS_OF,
        )


def test_budget_runtime_must_equal_the_configured_window() -> None:
    recipe = make_recipe()
    price = GpuPriceQuote(
        gpu_type="H200",
        usd_per_gpu_hour=1.0,
        effective_date="2026-08-01",
        source="pricing",
    )
    budget = BudgetRequest(
        max_usd=100.0,
        gpu_type="H200",
        gpu_count=4,
        max_runtime_seconds=600,
        deployment_seconds=1800,
        price=price,
        compute=_compute(),
        storage=_storage(),
        stored_gib=306.0,
        storage_retention_days=0.0,
    )
    with pytest.raises(DeploymentPlanError, match="same number"):
        build_plan(
            recipe=recipe,
            controls=RuntimeControls(
                timeout_seconds=1800,
                deployment_seconds=1800,
                startup_timeout_seconds=900,
            ),
            budget=budget,
            endpoint=EndpointConfig(),
            as_of=AS_OF,
        )


def test_default_controls_do_not_keep_a_container_warm() -> None:
    controls = RuntimeControls(timeout_seconds=1800, deployment_seconds=1800)
    assert controls.min_containers == 0
    assert controls.max_containers == 1
    assert controls.max_concurrent_inputs == 1


def test_warm_containers_require_an_explicit_acknowledgement() -> None:
    with pytest.raises(DeploymentPlanError, match="bills continuously"):
        RuntimeControls(timeout_seconds=1800, deployment_seconds=1800, min_containers=1)
    allowed = RuntimeControls(
        timeout_seconds=1800,
        deployment_seconds=1800,
        min_containers=1,
        allow_warm_containers=True,
    )
    assert allowed.min_containers == 1


def test_warm_containers_block_because_the_window_cannot_price_them() -> None:
    """A warm container ignores the priced window entirely.

    Modal keeps min_containers running while the function is idle, so the
    accelerators bill from deploy until the app is stopped with no
    relation to max_runtime_seconds. Approving that on the strength of a
    window it ignores would turn the envelope into an estimate.
    """
    plan = make_plan(
        controls_overrides={"min_containers": 1, "allow_warm_containers": True}
    )
    assert plan.approved is False
    assert any("bill continuously" in b for b in plan.blockers)
    assert all(not step.spends_money for step in plan.executable_steps)


def test_the_cold_start_residual_is_stated_plainly() -> None:
    """Auth is at the edge now, but an authenticated caller still churns.

    A Python-side refusal runs after the container is scheduled, so it
    bounds how long one serves rather than whether one starts. Proxy auth
    is what keeps that from being a public, unbounded cost; the residual
    that remains is worth naming rather than rounding away.
    """
    plan = make_plan()
    assert plan.controls.require_proxy_auth is True
    assert plan.envelope.bounded is True
    assert any(
        "rejects unauthenticated requests at its edge" in w for w in plan.warnings
    )
    assert any("is not priced above" in w for w in plan.warnings)


def test_sglang_blocks_because_it_takes_the_key_on_its_command_line() -> None:
    blocked = make_plan(recipe_overrides={"framework": SGLANG})
    assert blocked.approved is False
    assert any("command line argument" in b for b in blocked.blockers)

    accepted = make_plan(
        recipe_overrides={"framework": SGLANG},
        accept_argv_credential_exposure=True,
    )
    assert accepted.approved is True
    assert any("Accepted explicitly" in w for w in accepted.warnings)


def test_the_default_framework_keeps_the_key_out_of_the_command_line() -> None:
    plan = make_plan()
    assert plan.recipe.framework == VLLM
    assert plan.recipe.credential_transport == "environment"
    assert plan.approved is True


def test_startup_timeout_cannot_outlast_the_container() -> None:
    with pytest.raises(DeploymentPlanError, match="killed while still starting"):
        RuntimeControls(
            timeout_seconds=600, deployment_seconds=600, startup_timeout_seconds=1200
        )


@pytest.mark.parametrize("bad", [0, -1, MAX_SCALEDOWN_WINDOW_SECONDS + 1])
def test_scaledown_window_bounds(bad: int) -> None:
    with pytest.raises(DeploymentPlanError, match="scaledown_window_seconds"):
        RuntimeControls(
            timeout_seconds=1800,
            deployment_seconds=1800,
            scaledown_window_seconds=bad,
        )


def test_min_containers_cannot_exceed_max_containers() -> None:
    with pytest.raises(DeploymentPlanError, match="must not exceed max_containers"):
        RuntimeControls(
            timeout_seconds=1800,
            deployment_seconds=1800,
            max_containers=1,
            min_containers=2,
            allow_warm_containers=True,
        )


def test_concurrency_above_one_is_warned_about_as_measurement_interference() -> None:
    plan = make_plan(controls_overrides={"max_concurrent_inputs": 8})
    assert any("interfere" in w for w in plan.warnings)
