"""The environment contract, and what happens when it is incomplete."""

from __future__ import annotations

import json
from datetime import date

import pytest
from _fakes import PINNED_IMAGE, VALID_REVISION, valid_environ

from llmtracefx.deploy.environment import (
    REQUIRED_ENV_VARS,
    missing_required_env_vars,
    plan_from_environ,
)
from llmtracefx.deploy.errors import DeploymentPlanError

AS_OF = date(2026, 8, 30)

PLANTED_SECRET = "sk-live-do-not-persist-4f9a2c7e1b"


def test_a_complete_environment_produces_an_approved_plan() -> None:
    plan = plan_from_environ(valid_environ(), as_of=AS_OF)
    assert plan.approved is True
    assert plan.recipe.model_revision == VALID_REVISION
    assert plan.recipe.image.reference == PINNED_IMAGE
    assert plan.controls.max_containers == 1
    assert plan.controls.min_containers == 0


def test_an_empty_environment_refuses_and_names_everything_missing() -> None:
    with pytest.raises(DeploymentPlanError) as excinfo:
        plan_from_environ({}, as_of=AS_OF)
    message = str(excinfo.value)
    assert "refusing to configure a paid deployment" in message
    for name in REQUIRED_ENV_VARS:
        assert name in message


@pytest.mark.parametrize("name", REQUIRED_ENV_VARS)
def test_every_required_variable_is_individually_required(name: str) -> None:
    environ = valid_environ()
    del environ[name]
    assert missing_required_env_vars(environ) == (name,)
    with pytest.raises(DeploymentPlanError, match=name):
        plan_from_environ(environ, as_of=AS_OF)


@pytest.mark.parametrize("name", REQUIRED_ENV_VARS)
def test_a_blank_variable_counts_as_missing(name: str) -> None:
    environ = valid_environ(**{name: "   "})
    assert missing_required_env_vars(environ) == (name,)


def test_a_refused_plan_raises_rather_than_being_returned() -> None:
    """At deploy time nobody reads a summary, so a refusal must be fatal."""
    environ = valid_environ(
        LLMTRACEFX_GLM_MAX_USD="0.01", LLMTRACEFX_GLM_USD_PER_GPU_HOUR="50"
    )
    with pytest.raises(DeploymentPlanError, match="exceeds the planning threshold"):
        plan_from_environ(environ, as_of=AS_OF)


def test_an_unpinned_revision_is_refused_at_deploy_time() -> None:
    environ = valid_environ(LLMTRACEFX_GLM_MODEL_REVISION="main")
    with pytest.raises(DeploymentPlanError, match="40-character commit SHA"):
        plan_from_environ(environ, as_of=AS_OF)


def test_a_latest_image_is_refused_at_deploy_time() -> None:
    environ = valid_environ(LLMTRACEFX_GLM_IMAGE="lmsysorg/sglang:latest")
    with pytest.raises(DeploymentPlanError, match="never reproducible"):
        plan_from_environ(environ, as_of=AS_OF)


def test_warm_containers_are_refused_at_deploy_time() -> None:
    """Refused twice, for two different reasons, and neither is skippable.

    RuntimeControls refuses an unacknowledged warm floor outright; the
    plan then refuses an acknowledged one as well, because a container
    that bills while idle is not bounded by the window the envelope
    prices.
    """
    with pytest.raises(DeploymentPlanError, match="bills continuously"):
        plan_from_environ(valid_environ(LLMTRACEFX_GLM_MIN_CONTAINERS="1"), as_of=AS_OF)
    with pytest.raises(DeploymentPlanError, match="bill continuously"):
        plan_from_environ(
            valid_environ(
                LLMTRACEFX_GLM_MIN_CONTAINERS="1",
                LLMTRACEFX_GLM_ALLOW_WARM_CONTAINERS="true",
                LLMTRACEFX_GLM_MAX_USD="20.00",
            ),
            as_of=AS_OF,
        )


def test_the_default_framework_needs_no_credential_exposure_waiver() -> None:
    plan = plan_from_environ(valid_environ(), as_of=AS_OF)
    assert plan.recipe.framework == "vllm"
    assert plan.approved is True


def test_sglang_is_refused_at_deploy_time_unless_acknowledged() -> None:
    with pytest.raises(DeploymentPlanError, match="command line argument"):
        plan_from_environ(valid_environ(LLMTRACEFX_GLM_FRAMEWORK="sglang"), as_of=AS_OF)
    approved = plan_from_environ(
        valid_environ(
            LLMTRACEFX_GLM_FRAMEWORK="sglang",
            LLMTRACEFX_GLM_ACCEPT_ARGV_CREDENTIAL_EXPOSURE="true",
        ),
        as_of=AS_OF,
    )
    assert approved.recipe.framework == "sglang"


def test_extra_containers_multiply_the_serving_terms() -> None:
    one = plan_from_environ(valid_environ(), as_of=AS_OF)
    four = plan_from_environ(
        valid_environ(
            LLMTRACEFX_GLM_MAX_CONTAINERS="4", LLMTRACEFX_GLM_MAX_USD="200.00"
        ),
        as_of=AS_OF,
    )
    for name in ("serving-gpu", "serving-compute"):
        single = next(line for line in one.envelope.lines if line.name == name).usd
        quadruple = next(line for line in four.envelope.lines if line.name == name).usd
        assert quadruple == pytest.approx(single * 4)
    with pytest.raises(DeploymentPlanError, match="exceeds the planning threshold"):
        plan_from_environ(
            valid_environ(
                LLMTRACEFX_GLM_MAX_CONTAINERS="8", LLMTRACEFX_GLM_MAX_USD="10.00"
            ),
            as_of=AS_OF,
        )


@pytest.mark.parametrize(
    "name",
    [
        "LLMTRACEFX_GLM_MAX_USD",
        "LLMTRACEFX_GLM_USD_PER_GPU_HOUR",
    ],
)
def test_unparseable_numbers_are_refused_without_echoing_the_value(name: str) -> None:
    environ = valid_environ(**{name: "not-a-number-but-secret-ish"})
    with pytest.raises(DeploymentPlanError) as excinfo:
        plan_from_environ(environ, as_of=AS_OF)
    assert "not-a-number-but-secret-ish" not in str(excinfo.value)
    assert name in str(excinfo.value)


def test_unparseable_integers_are_refused() -> None:
    environ = valid_environ(LLMTRACEFX_GLM_GPU_COUNT="four")
    with pytest.raises(DeploymentPlanError, match="must be an integer"):
        plan_from_environ(environ, as_of=AS_OF)


def test_a_bad_boolean_is_refused_rather_than_read_as_false() -> None:
    environ = valid_environ(LLMTRACEFX_GLM_ACCEPT_MUTABLE_IMAGE="maybe")
    with pytest.raises(DeploymentPlanError, match="ACCEPT_MUTABLE_IMAGE"):
        plan_from_environ(environ, as_of=AS_OF)


def test_a_zero_freshness_limit_is_honoured_and_not_read_as_absent() -> None:
    """Zero means "only a quote read today", which is the strictest setting.

    Defaulting it away would hand back the lenient ninety day limit to
    the one operator who explicitly asked for the opposite.
    """
    environ = valid_environ(LLMTRACEFX_GLM_MAX_PRICE_AGE_DAYS="0")
    with pytest.raises(DeploymentPlanError, match="limit 0 days"):
        plan_from_environ(environ, as_of=AS_OF)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("LLMTRACEFX_GLM_MAX_CONTAINERS", "max_containers"),
        ("LLMTRACEFX_GLM_SCALEDOWN_WINDOW_SECONDS", "scaledown_window_seconds"),
        ("LLMTRACEFX_GLM_STARTUP_TIMEOUT_SECONDS", "startup_timeout_seconds"),
        ("LLMTRACEFX_GLM_MAX_CONCURRENT_INPUTS", "max_concurrent_inputs"),
    ],
)
def test_an_explicit_zero_is_validated_rather_than_replaced(
    name: str, expected: str
) -> None:
    """A present value must reach its validator, not the default.

    `_int(...) or default` cannot tell "unset" from "set to zero", so it
    rewrites every zero into a plausible default and the validator that
    exists to reject zero never runs.
    """
    with pytest.raises(DeploymentPlanError, match=expected):
        plan_from_environ(valid_environ(**{name: "0"}), as_of=AS_OF)


def test_an_explicit_port_of_zero_is_refused() -> None:
    with pytest.raises(DeploymentPlanError, match="port"):
        plan_from_environ(valid_environ(LLMTRACEFX_GLM_PORT="0"), as_of=AS_OF)


def test_storage_size_defaults_to_the_checkpoint() -> None:
    plan = plan_from_environ(valid_environ(), as_of=AS_OF)
    assert plan.envelope.stored_gib == pytest.approx(306.0)
    assert plan.envelope.storage_billed_days == pytest.approx(5.0)


def test_no_credential_value_reaches_the_serialised_plan() -> None:
    """A key sitting in the deploy environment must not be written down.

    The planner reads its own prefixed variables and nothing else, so a
    credential exported alongside them has no route into the document.
    This asserts that end to end rather than by inspection.
    """
    environ = valid_environ()
    environ["GLM_SELFHOST_API_KEY"] = PLANTED_SECRET
    environ["HF_TOKEN"] = PLANTED_SECRET
    plan = plan_from_environ(environ, as_of=AS_OF)
    document = json.dumps(plan.to_dict())
    assert PLANTED_SECRET not in document
    assert "GLM_SELFHOST_API_KEY" in document


def test_the_plan_still_reports_that_it_touched_nothing() -> None:
    payload = plan_from_environ(valid_environ(), as_of=AS_OF).to_dict()
    assert payload["network_request_performed"] is False
    assert payload["gpu_allocated"] is False
