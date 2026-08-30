"""Budget-guarded, pinned self-hosting of GLM-5.3-Flash.

The public surface is deliberately small and entirely offline:

``plan``        adjudicate a deployment and render the commands for it.
``budget``      worst-case cost arithmetic and the fail-closed gate.
``recipe``      the pinned model facts and the serving configuration.
``manifest``    what was staged, and what served.

Importing this package does not import the Modal SDK. Modal is needed
only by ``llmtracefx.deploy.modal_glm_app``, which is an entrypoint for
``modal run`` and ``modal deploy`` rather than a library module, and is
never imported by the planner, the CLI or the tests.
"""

from __future__ import annotations

from .budget import (
    BudgetRequest,
    CostEnvelope,
    assert_within_budget,
    evaluate_budget,
    recommended_session_budget_usd,
)
from .commands import CommandStep
from .endpoint import EndpointConfig, collector_argv
from .environment import REQUIRED_ENV_VARS, plan_from_environ
from .errors import DeploymentPlanError
from .manifest import ServerManifest, StagedFile, WeightStagingManifest
from .plan import DeploymentPlan, RuntimeControls, assert_executable, build_plan
from .pricing import GpuPriceQuote, StorageQuote
from .recipe import GLM_53_FLASH, ServingRecipe, build_recipe, check_memory_fit

__all__ = [
    "GLM_53_FLASH",
    "REQUIRED_ENV_VARS",
    "BudgetRequest",
    "CommandStep",
    "CostEnvelope",
    "DeploymentPlan",
    "DeploymentPlanError",
    "EndpointConfig",
    "GpuPriceQuote",
    "RuntimeControls",
    "ServerManifest",
    "ServingRecipe",
    "StagedFile",
    "StorageQuote",
    "WeightStagingManifest",
    "assert_executable",
    "assert_within_budget",
    "build_plan",
    "build_recipe",
    "check_memory_fit",
    "collector_argv",
    "evaluate_budget",
    "plan_from_environ",
    "recommended_session_budget_usd",
]
