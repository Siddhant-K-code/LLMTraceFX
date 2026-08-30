"""Reconstructing a plan from environment variables, and failing closed.

The Modal app cannot take command line flags: it is imported by ``modal
deploy``, which decides GPU count, timeout and autoscaling at build time
from whatever the module says at import. So the same parameters that
``deploy plan`` takes as flags have to reach the app some other way, and
that way is the environment.

The important property is what happens when a variable is missing. It is
not "use a sensible default": a sensible default for a GPU count is still
a GPU count, and the app would deploy. It is a refusal that names every
missing variable at once, raised while the module is being imported, so
``modal deploy`` fails before it registers anything.

This module imports no Modal SDK. It is ordinary parsing and validation
over a mapping, which is what lets the whole fail-closed contract be
tested without Modal installed and without a network.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

from .budget import BudgetRequest
from .commands import DEFAULT_APP_NAME, DEFAULT_VOLUME_NAME
from .endpoint import (
    DEFAULT_API_KEY_ENV_VAR,
    DEFAULT_MODAL_SECRET_NAME,
    EndpointConfig,
)
from .errors import DeploymentPlanError
from .plan import (
    DEFAULT_SCALEDOWN_WINDOW_SECONDS,
    DeploymentPlan,
    RuntimeControls,
    build_plan,
)
from .pricing import (
    DEFAULT_MAX_PRICE_AGE_DAYS,
    ComputeQuote,
    GpuPriceQuote,
    StorageQuote,
)
from .recipe import (
    DEFAULT_FRAMEWORK,
    DEFAULT_SERVER_PORT,
    DEFAULT_WEIGHTS_MOUNT_PATH,
    GLM_53_FLASH,
    SUPPORTED_REPO_ID,
    build_recipe,
)

PREFIX = "LLMTRACEFX_GLM_"

MAX_USD = f"{PREFIX}MAX_USD"
GPU_TYPE = f"{PREFIX}GPU_TYPE"
GPU_COUNT = f"{PREFIX}GPU_COUNT"
MAX_RUNTIME_SECONDS = f"{PREFIX}MAX_RUNTIME_SECONDS"
USD_PER_GPU_HOUR = f"{PREFIX}USD_PER_GPU_HOUR"
PRICE_EFFECTIVE_DATE = f"{PREFIX}PRICE_EFFECTIVE_DATE"
PRICE_SOURCE = f"{PREFIX}PRICE_SOURCE"
MODEL_REVISION = f"{PREFIX}MODEL_REVISION"
IMAGE = f"{PREFIX}IMAGE"
FRAMEWORK_VERSION = f"{PREFIX}FRAMEWORK_VERSION"
CONTEXT_LENGTH = f"{PREFIX}CONTEXT_LENGTH"
MAX_DEPLOYMENT_SECONDS = f"{PREFIX}MAX_DEPLOYMENT_SECONDS"
USD_PER_CPU_CORE_HOUR = f"{PREFIX}USD_PER_CPU_CORE_HOUR"
USD_PER_GIB_MEMORY_HOUR = f"{PREFIX}USD_PER_GIB_MEMORY_HOUR"
STORAGE_RETENTION_DAYS = f"{PREFIX}STORAGE_RETENTION_DAYS"
STORAGE_USD_PER_GIB_MONTH = f"{PREFIX}STORAGE_USD_PER_GIB_MONTH"

# Every one of these has to be present. There is no default for any of
# them because each is either a spending authority, a price, or the pin
# that makes the deployment reproducible.
REQUIRED_ENV_VARS: tuple[str, ...] = (
    MAX_USD,
    GPU_TYPE,
    GPU_COUNT,
    MAX_RUNTIME_SECONDS,
    USD_PER_GPU_HOUR,
    PRICE_EFFECTIVE_DATE,
    PRICE_SOURCE,
    MODEL_REVISION,
    IMAGE,
    FRAMEWORK_VERSION,
    CONTEXT_LENGTH,
    MAX_DEPLOYMENT_SECONDS,
    USD_PER_CPU_CORE_HOUR,
    USD_PER_GIB_MEMORY_HOUR,
    STORAGE_USD_PER_GIB_MONTH,
    STORAGE_RETENTION_DAYS,
)

FRAMEWORK = f"{PREFIX}FRAMEWORK"
TENSOR_PARALLEL_SIZE = f"{PREFIX}TENSOR_PARALLEL_SIZE"
SERVED_MODEL_NAME = f"{PREFIX}SERVED_MODEL_NAME"
PORT = f"{PREFIX}PORT"
WEIGHTS_MOUNT_PATH = f"{PREFIX}WEIGHTS_MOUNT_PATH"
MAX_CONTAINERS = f"{PREFIX}MAX_CONTAINERS"
MIN_CONTAINERS = f"{PREFIX}MIN_CONTAINERS"
ALLOW_WARM_CONTAINERS = f"{PREFIX}ALLOW_WARM_CONTAINERS"
SCALEDOWN_WINDOW_SECONDS = f"{PREFIX}SCALEDOWN_WINDOW_SECONDS"
STARTUP_TIMEOUT_SECONDS = f"{PREFIX}STARTUP_TIMEOUT_SECONDS"
MAX_CONCURRENT_INPUTS = f"{PREFIX}MAX_CONCURRENT_INPUTS"
ACCEPT_MUTABLE_IMAGE = f"{PREFIX}ACCEPT_MUTABLE_IMAGE"
ACCEPT_STALE_PRICE = f"{PREFIX}ACCEPT_STALE_PRICE"
ACCEPT_ARGV_CREDENTIAL_EXPOSURE = f"{PREFIX}ACCEPT_ARGV_CREDENTIAL_EXPOSURE"
MAX_PRICE_AGE_DAYS = f"{PREFIX}MAX_PRICE_AGE_DAYS"
AS_OF = f"{PREFIX}AS_OF"
APP_NAME = f"{PREFIX}APP_NAME"
VOLUME_NAME = f"{PREFIX}VOLUME_NAME"
API_KEY_ENV = f"{PREFIX}API_KEY_ENV"
MODAL_SECRET_NAME = f"{PREFIX}MODAL_SECRET_NAME"
ENDPOINT_BASE_URL = f"{PREFIX}ENDPOINT_BASE_URL"
STORED_GIB = f"{PREFIX}STORED_GIB"

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})


def _text(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _float(
    environ: Mapping[str, str], name: str, default: float | None
) -> float | None:
    raw = _text(environ, name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        # The offending value is not echoed. These variables are set
        # alongside credentials in the same deploy environment, and a
        # message that quotes "the value of X" is one copy-paste mistake
        # away from quoting a key.
        raise DeploymentPlanError(f"{name} must be a number") from exc


def _int(environ: Mapping[str, str], name: str, default: int | None) -> int | None:
    raw = _text(environ, name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise DeploymentPlanError(f"{name} must be an integer") from exc


def _int_with_default(environ: Mapping[str, str], name: str, default: int) -> int:
    """Parse an optional integer, defaulting only when it is *absent*.

    Deliberately not ``_int(...) or default``. That idiom cannot tell an
    unset variable from a variable set to zero, so it silently rewrites
    every zero into the default. For most of these settings that turns a
    nonsense value into a plausible one instead of an error, and for
    ``MAX_PRICE_AGE_DAYS`` it is worse than that: zero is the operator
    asking for maximum strictness, and swallowing it hands back the
    lenient default. Absence is the only thing that may produce a
    default; a value that is present is passed through and validated.
    """
    value = _int(environ, name, default)
    if value is None:  # pragma: no cover - default is not None
        return default
    return value


def _bool(environ: Mapping[str, str], name: str, default: bool = False) -> bool:
    raw = environ.get(name)
    if raw is None:
        return default
    folded = raw.strip().casefold()
    if folded in _TRUE:
        return True
    if folded in _FALSE:
        return False
    raise DeploymentPlanError(
        f"{name} must be one of: " + ", ".join(sorted(_TRUE | {"false"}))
    )


def missing_required_env_vars(environ: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(name for name in REQUIRED_ENV_VARS if _text(environ, name) is None)


def plan_from_environ(
    environ: Mapping[str, str], *, as_of: date | None = None
) -> DeploymentPlan:
    """Rebuild the adjudicated plan the app must obey, or refuse to build.

    Raises rather than returning an unapproved plan. At import time
    inside ``modal deploy`` there is no operator to read a summary, so
    the only useful outcome of a refused plan is a failed deploy.
    """
    missing = missing_required_env_vars(environ)
    if missing:
        raise DeploymentPlanError(
            "refusing to configure a paid deployment: "
            + ", ".join(missing)
            + " must be set. Run `llmtracefx-deploy plan` first; it prints "
            "the exact values to export."
        )

    max_usd = _float(environ, MAX_USD, None)
    usd_per_gpu_hour = _float(environ, USD_PER_GPU_HOUR, None)
    usd_per_cpu = _float(environ, USD_PER_CPU_CORE_HOUR, None)
    usd_per_memory = _float(environ, USD_PER_GIB_MEMORY_HOUR, None)
    storage_rate = _float(environ, STORAGE_USD_PER_GIB_MONTH, None)
    retention_days = _float(environ, STORAGE_RETENTION_DAYS, None)
    gpu_count = _int(environ, GPU_COUNT, None)
    max_runtime = _int(environ, MAX_RUNTIME_SECONDS, None)
    deployment_seconds = _int(environ, MAX_DEPLOYMENT_SECONDS, None)
    context_length = _int(environ, CONTEXT_LENGTH, None)
    # Narrowing for the type checker; the presence check above already
    # guarantees each of these parsed from a non-empty string.
    assert max_usd is not None
    assert usd_per_gpu_hour is not None
    assert usd_per_cpu is not None
    assert usd_per_memory is not None
    assert storage_rate is not None
    assert retention_days is not None
    assert gpu_count is not None
    assert max_runtime is not None
    assert deployment_seconds is not None
    assert context_length is not None

    gpu_type = environ[GPU_TYPE].strip()
    effective_date = environ[PRICE_EFFECTIVE_DATE].strip()
    price_source = environ[PRICE_SOURCE].strip()
    max_containers = _int_with_default(environ, MAX_CONTAINERS, 1)
    min_containers = _int_with_default(environ, MIN_CONTAINERS, 0)
    stored_gib = _float(environ, STORED_GIB, None)
    if stored_gib is None:
        stored_gib = GLM_53_FLASH.approximate_checkpoint_gib

    budget = BudgetRequest(
        max_usd=max_usd,
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        max_runtime_seconds=max_runtime,
        deployment_seconds=deployment_seconds,
        price=GpuPriceQuote(
            gpu_type=gpu_type,
            usd_per_gpu_hour=usd_per_gpu_hour,
            effective_date=effective_date,
            source=price_source,
        ),
        compute=ComputeQuote(
            usd_per_cpu_core_hour=usd_per_cpu,
            usd_per_gib_memory_hour=usd_per_memory,
            effective_date=effective_date,
            source=price_source,
        ),
        storage=StorageQuote(
            usd_per_gib_month=storage_rate,
            effective_date=effective_date,
            source=price_source,
        ),
        stored_gib=stored_gib,
        storage_retention_days=retention_days,
        max_containers=max_containers,
    )
    recipe = build_recipe(
        framework=_text(environ, FRAMEWORK) or DEFAULT_FRAMEWORK,
        framework_version=environ[FRAMEWORK_VERSION].strip(),
        image_reference=environ[IMAGE].strip(),
        model_revision=environ[MODEL_REVISION].strip(),
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        context_length=context_length,
        weights_mount_path=_text(environ, WEIGHTS_MOUNT_PATH)
        or DEFAULT_WEIGHTS_MOUNT_PATH,
        port=_int_with_default(environ, PORT, DEFAULT_SERVER_PORT),
        served_model_name=_text(environ, SERVED_MODEL_NAME) or SUPPORTED_REPO_ID,
        tensor_parallel_size=_int(environ, TENSOR_PARALLEL_SIZE, None),
        accept_mutable_image=_bool(environ, ACCEPT_MUTABLE_IMAGE),
    )
    controls = RuntimeControls(
        timeout_seconds=max_runtime,
        deployment_seconds=deployment_seconds,
        scaledown_window_seconds=_int_with_default(
            environ, SCALEDOWN_WINDOW_SECONDS, DEFAULT_SCALEDOWN_WINDOW_SECONDS
        ),
        startup_timeout_seconds=_int_with_default(
            environ, STARTUP_TIMEOUT_SECONDS, 1800
        ),
        max_containers=max_containers,
        min_containers=min_containers,
        max_concurrent_inputs=_int_with_default(environ, MAX_CONCURRENT_INPUTS, 1),
        allow_warm_containers=_bool(environ, ALLOW_WARM_CONTAINERS),
    )
    endpoint = EndpointConfig(
        api_key_env_var=_text(environ, API_KEY_ENV) or DEFAULT_API_KEY_ENV_VAR,
        modal_secret_name=_text(environ, MODAL_SECRET_NAME)
        or DEFAULT_MODAL_SECRET_NAME,
        served_model_name=recipe.served_model_name,
        base_url=_text(environ, ENDPOINT_BASE_URL),
    )

    as_of_value = as_of
    if as_of_value is None:
        raw_as_of = _text(environ, AS_OF)
        if raw_as_of is None:
            as_of_value = date.today()
        else:
            try:
                as_of_value = date.fromisoformat(raw_as_of)
            except ValueError as exc:
                raise DeploymentPlanError(f"{AS_OF} must be an ISO date") from exc

    plan = build_plan(
        recipe=recipe,
        controls=controls,
        budget=budget,
        endpoint=endpoint,
        as_of=as_of_value,
        app_name=_text(environ, APP_NAME) or DEFAULT_APP_NAME,
        volume_name=_text(environ, VOLUME_NAME) or DEFAULT_VOLUME_NAME,
        max_price_age_days=_int_with_default(
            environ, MAX_PRICE_AGE_DAYS, DEFAULT_MAX_PRICE_AGE_DAYS
        ),
        accept_stale_price=_bool(environ, ACCEPT_STALE_PRICE),
        accept_argv_credential_exposure=_bool(environ, ACCEPT_ARGV_CREDENTIAL_EXPOSURE),
    )
    if not plan.approved:
        raise DeploymentPlanError(
            "refusing to configure a paid deployment:\n"
            + "\n".join(f"  - {reason}" for reason in plan.blockers)
        )
    return plan


def plan_environment(plan: DeploymentPlan, *, as_of: date) -> dict[str, str]:
    """The environment a remote container needs to rebuild ``plan``.

    This exists because the Modal entrypoint is imported twice: once
    locally, where the operator has exported these variables, and once
    inside every container, where nothing has. Baking the adjudicated
    values into the container images is what makes the second import
    reconstruct the same plan instead of failing on eleven missing
    variables.

    It is derived from the validated plan rather than copied from the
    caller's environment, so what the container obeys is what the planner
    approved, not whatever happened to be exported alongside it. Nothing
    here is a secret: names, prices, revisions and limits only.
    """
    envelope = plan.envelope
    values = {
        MAX_USD: f"{envelope.budget_usd:.6f}",
        GPU_TYPE: plan.recipe.gpu_type,
        GPU_COUNT: str(plan.recipe.gpu_count),
        MAX_RUNTIME_SECONDS: str(plan.controls.timeout_seconds),
        MAX_DEPLOYMENT_SECONDS: str(plan.controls.deployment_seconds),
        USD_PER_GPU_HOUR: f"{envelope.usd_per_gpu_hour:.6f}",
        USD_PER_CPU_CORE_HOUR: f"{plan.budget.compute.usd_per_cpu_core_hour:.6f}",
        USD_PER_GIB_MEMORY_HOUR: (f"{plan.budget.compute.usd_per_gib_memory_hour:.6f}"),
        STORAGE_USD_PER_GIB_MONTH: f"{envelope.storage_usd_per_gib_month:.6f}",
        STORAGE_RETENTION_DAYS: f"{envelope.storage_retention_days:.6f}",
        STORED_GIB: f"{envelope.stored_gib:.6f}",
        PRICE_EFFECTIVE_DATE: envelope.price_effective_date,
        PRICE_SOURCE: envelope.price_source,
        MODEL_REVISION: plan.recipe.model_revision,
        IMAGE: plan.recipe.image.reference,
        FRAMEWORK: plan.recipe.framework,
        FRAMEWORK_VERSION: plan.recipe.framework_version,
        CONTEXT_LENGTH: str(plan.recipe.context_length),
        TENSOR_PARALLEL_SIZE: str(plan.recipe.tensor_parallel_size),
        SERVED_MODEL_NAME: plan.recipe.served_model_name,
        PORT: str(plan.recipe.port),
        WEIGHTS_MOUNT_PATH: plan.recipe.weights_mount_path,
        MAX_CONTAINERS: str(plan.controls.max_containers),
        MIN_CONTAINERS: str(plan.controls.min_containers),
        SCALEDOWN_WINDOW_SECONDS: str(plan.controls.scaledown_window_seconds),
        STARTUP_TIMEOUT_SECONDS: str(plan.controls.startup_timeout_seconds),
        MAX_CONCURRENT_INPUTS: str(plan.controls.max_concurrent_inputs),
        APP_NAME: plan.app_name,
        VOLUME_NAME: plan.volume_name,
        API_KEY_ENV: plan.endpoint.api_key_env_var,
        MODAL_SECRET_NAME: plan.endpoint.modal_secret_name,
        ACCEPT_MUTABLE_IMAGE: str(not plan.recipe.image.is_digest_pinned).lower(),
        ACCEPT_ARGV_CREDENTIAL_EXPOSURE: str(
            plan.recipe.exposes_credential_on_argv
        ).lower(),
        ALLOW_WARM_CONTAINERS: str(plan.controls.allow_warm_containers).lower(),
        # Pinned so the remote rebuild ages the quotes against the same
        # day the local planner did. Without it a container starting after
        # the freshness limit elapsed would refuse to boot on a price that
        # was fresh when the deployment was approved.
        AS_OF: as_of.isoformat(),
        MAX_PRICE_AGE_DAYS: str(plan.max_price_age_days),
        ACCEPT_STALE_PRICE: str(plan.accept_stale_price).lower(),
    }
    return values
