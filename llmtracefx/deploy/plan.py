"""The deployment plan: one pure function from parameters to a decision.

Planning is offline by construction. It reads no configuration file, opens
no socket, imports no Modal SDK and consults no credential. Given the same
inputs it produces the same document, which is what makes it safe to run
before authenticating and useful to diff in review.

The plan is also the gate. It does not merely describe what would happen;
it decides whether the paid steps are allowed to be handed to the operator
at all. When something is wrong (over budget, a stale price, an unpinned
revision, accelerators that cannot hold the checkpoint) the plan still
renders in full, because an operator needs to see why, but every step that
can spend money is withheld from the executable set.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date
from typing import Any

from . import commands
from .budget import BudgetRequest, CostEnvelope, evaluate_budget
from .commands import CommandStep
from .endpoint import EndpointConfig, collector_argv
from .errors import DeploymentPlanError
from .pricing import DEFAULT_MAX_PRICE_AGE_DAYS
from .recipe import (
    VLLM,
    VLLM_API_KEY_ENV_VAR,
    MemoryFit,
    ServingRecipe,
    check_memory_fit,
)

PLAN_SCHEMA_VERSION = "1"

# Five minutes of idle before releasing the GPUs. Long enough that a
# health check followed by a smoke request followed by a collector run
# does not pay three cold starts, short enough that walking away from the
# terminal does not quietly burn the budget.
DEFAULT_SCALEDOWN_WINDOW_SECONDS = 300

# An idle window beyond half an hour stops being "avoid a cold start" and
# becomes "rent an idle accelerator", so it is refused.
MAX_SCALEDOWN_WINDOW_SECONDS = 1800


@dataclass(frozen=True)
class RuntimeControls:
    """The autoscaling and lifetime limits, all stated, none inherited.

    Every field here is a multiplier or a duration on a billed resource.
    Platform defaults are not used for any of them: a default that is
    right for a cheap CPU function is the wrong default for four H200s,
    and the difference is not visible until the invoice.
    """

    timeout_seconds: int
    deployment_seconds: int
    scaledown_window_seconds: int = DEFAULT_SCALEDOWN_WINDOW_SECONDS
    startup_timeout_seconds: int = 1800
    max_containers: int = 1
    min_containers: int = 0
    max_concurrent_inputs: int = 1
    allow_warm_containers: bool = False
    require_proxy_auth: bool = True

    def __post_init__(self) -> None:
        if self.timeout_seconds < 1:
            raise DeploymentPlanError("timeout_seconds must be at least 1")
        if self.deployment_seconds < 1:
            raise DeploymentPlanError("deployment_seconds must be at least 1")
        if self.deployment_seconds < self.timeout_seconds:
            raise DeploymentPlanError(
                f"deployment_seconds {self.deployment_seconds} is shorter than "
                f"timeout_seconds {self.timeout_seconds}; the window the "
                "deployment may serve for cannot be shorter than one "
                "container's own lifetime"
            )
        if self.startup_timeout_seconds < 1:
            raise DeploymentPlanError("startup_timeout_seconds must be at least 1")
        if self.startup_timeout_seconds > self.timeout_seconds:
            raise DeploymentPlanError(
                f"startup_timeout_seconds {self.startup_timeout_seconds} "
                f"exceeds timeout_seconds {self.timeout_seconds}; the "
                "container would be killed while still starting"
            )
        if not (1 <= self.scaledown_window_seconds <= MAX_SCALEDOWN_WINDOW_SECONDS):
            raise DeploymentPlanError(
                "scaledown_window_seconds must be in "
                f"1..{MAX_SCALEDOWN_WINDOW_SECONDS}, got "
                f"{self.scaledown_window_seconds}"
            )
        if self.max_containers < 1:
            raise DeploymentPlanError("max_containers must be at least 1")
        if self.min_containers < 0:
            raise DeploymentPlanError("min_containers must not be negative")
        if self.min_containers > self.max_containers:
            raise DeploymentPlanError("min_containers must not exceed max_containers")
        if self.min_containers > 0 and not self.allow_warm_containers:
            raise DeploymentPlanError(
                "min_containers greater than zero keeps accelerators "
                "allocated with no request in flight, which bills "
                "continuously. Pass allow_warm_containers to state that you "
                "mean it."
            )
        if self.max_concurrent_inputs < 1:
            raise DeploymentPlanError("max_concurrent_inputs must be at least 1")
        if not self.require_proxy_auth:
            # Refused outright, with no flag to override it, because there
            # is no honest number to put next to a public endpoint. A web
            # server container is scheduled by the platform before any of
            # this project's code runs, so on a public URL any request
            # from anyone allocates accelerators first and is refused
            # second, without limit and including after the deployment
            # expiry. Nothing in the cost envelope bounds that, and this
            # harness will not approve spending it cannot bound.
            #
            # Nothing is lost by requiring it. Modal's proxy token pair
            # can be sent as a single `Authorization: Bearer` value, the
            # same scheme the OpenAI API uses
            # (https://modal.com/docs/guide/webhook-proxy-auth, read
            # 2026-08-30), so the endpoint stays OpenAI-compatible and the
            # collector needs no change.
            raise DeploymentPlanError(
                "require_proxy_auth cannot be turned off. Without it the "
                "endpoint is public, and a web server container is "
                "scheduled before any of its code runs, so every request "
                "from anyone allocates accelerators and is refused "
                "afterwards, without limit and including after the "
                "expiry. That cost is unbounded, so it cannot be "
                "approved. Modal's proxy token works as an ordinary "
                "`Authorization: Bearer` value, so keeping it on costs "
                "you nothing."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeout_seconds": self.timeout_seconds,
            "deployment_seconds": self.deployment_seconds,
            "scaledown_window_seconds": self.scaledown_window_seconds,
            "startup_timeout_seconds": self.startup_timeout_seconds,
            "max_containers": self.max_containers,
            "min_containers": self.min_containers,
            "max_concurrent_inputs": self.max_concurrent_inputs,
            "allow_warm_containers": self.allow_warm_containers,
            "require_proxy_auth": self.require_proxy_auth,
        }


@dataclass(frozen=True)
class DeploymentPlan:
    """A rendered decision: what would run, what it could cost, and whether."""

    schema_version: str
    recipe: ServingRecipe
    controls: RuntimeControls
    budget: BudgetRequest
    envelope: CostEnvelope
    memory: MemoryFit
    endpoint: EndpointConfig
    app_name: str
    volume_name: str
    steps: tuple[CommandStep, ...]
    max_price_age_days: int = DEFAULT_MAX_PRICE_AGE_DAYS
    accept_stale_price: bool = False
    blockers: tuple[str, ...] = field(default_factory=tuple)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def approved(self) -> bool:
        return not self.blockers

    @property
    def executable_steps(self) -> tuple[CommandStep, ...]:
        """Steps the operator may run given the current decision.

        When the plan is not approved this collapses to the steps that
        cannot spend anything, so an operator who ignores the summary and
        copies the command list still cannot start a paid resource.
        """
        if self.approved:
            return self.steps
        return tuple(step for step in self.steps if not step.spends_money)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "llmtracefx.deploy.plan",
            "approved": self.approved,
            # Named the same way the API collector names its dry-run
            # assertions, so "did this touch anything" reads identically
            # across the two tools.
            "network_request_performed": False,
            "gpu_allocated": False,
            "modal_authentication_used": False,
            "app_name": self.app_name,
            "volume_name": self.volume_name,
            "recipe": self.recipe.to_dict(),
            "server_argv": list(self.recipe.launch_argv()),
            "runtime_controls": self.controls.to_dict(),
            "cost_envelope": self.envelope.to_dict(),
            "memory_fit": self.memory.to_dict(),
            "endpoint": self.endpoint.to_dict(),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "steps": [step.to_dict() for step in self.steps],
            "executable_steps": [step.name for step in self.executable_steps],
        }


def build_plan(
    *,
    recipe: ServingRecipe,
    controls: RuntimeControls,
    budget: BudgetRequest,
    endpoint: EndpointConfig,
    as_of: date,
    app_name: str = commands.DEFAULT_APP_NAME,
    volume_name: str = commands.DEFAULT_VOLUME_NAME,
    smoke_max_output_tokens: int = 32,
    collector_run_id: str = "glm53flash-selfhost-smoke",
    collector_prompt_file: str = "examples/optimizer/api-smoke-prompt.txt",
    collector_output_dir: str = "output/glm53flash-selfhost",
    max_price_age_days: int = DEFAULT_MAX_PRICE_AGE_DAYS,
    accept_stale_price: bool = False,
    accept_argv_credential_exposure: bool = False,
) -> DeploymentPlan:
    """Assemble and adjudicate a plan. Pure; performs no I/O."""
    if budget.gpu_type.casefold() != recipe.gpu_type.casefold():
        raise DeploymentPlanError(
            f"budget prices {budget.gpu_type!r} but the recipe serves on "
            f"{recipe.gpu_type!r}"
        )
    if budget.gpu_count != recipe.gpu_count:
        raise DeploymentPlanError(
            f"budget covers {budget.gpu_count} GPU(s) but the recipe "
            f"requests {recipe.gpu_count}"
        )
    if budget.max_containers != controls.max_containers:
        raise DeploymentPlanError(
            f"budget prices {budget.max_containers} container(s) but the "
            f"runtime controls allow {controls.max_containers}"
        )
    if budget.deployment_seconds != controls.deployment_seconds:
        raise DeploymentPlanError(
            f"budget prices a {budget.deployment_seconds}s deployment window "
            f"but the controls declare {controls.deployment_seconds}s; the "
            "window that is priced must be the window that is enforced"
        )
    if budget.max_runtime_seconds != controls.timeout_seconds:
        raise DeploymentPlanError(
            f"budget prices {budget.max_runtime_seconds}s but the container "
            f"timeout is {controls.timeout_seconds}s; the priced window and "
            "the configured window must be the same number or the envelope "
            "describes a deployment other than the one being planned"
        )

    envelope = evaluate_budget(budget)
    # The arithmetic cannot see the endpoint's auth posture, so the
    # boundedness flag is set from the controls that decide it.
    envelope = replace(envelope, bounded=controls.require_proxy_auth)
    memory = check_memory_fit(
        gpu_type=recipe.gpu_type, gpu_count=recipe.gpu_count, facts=recipe.facts
    )

    blockers: list[str] = []
    warnings: list[str] = []

    if not envelope.within_budget:
        blockers.append(
            f"Modeled cost ${envelope.worst_case_usd:.2f} exceeds the "
            f"planning threshold ${envelope.budget_usd:.2f}."
        )
    if not memory.fits:
        blockers.append(
            f"{recipe.gpu_count} x {recipe.gpu_type} provides "
            f"{memory.total_vram_gib:.0f} GiB against roughly "
            f"{memory.weights_gib:.0f} GiB of weights, leaving "
            f"{memory.residual_fraction:.0%} for KV cache and activations; "
            f"at least {memory.required_headroom_fraction:.0%} is required."
        )

    price_age = budget.price.age_days(as_of=as_of)
    if price_age < 0:
        blockers.append(
            f"GPU price is dated {budget.price.effective_date}, which is in "
            "the future relative to the planning date; correct the quote."
        )
    elif budget.price.is_stale(as_of=as_of, max_age_days=max_price_age_days):
        message = (
            f"GPU price quote is {price_age} days old (effective "
            f"{budget.price.effective_date}, limit {max_price_age_days} "
            "days). Cloud GPU prices change; re-read the current rate."
        )
        if accept_stale_price:
            warnings.append(message + " Accepted explicitly.")
        else:
            blockers.append(message)

    for label, quote in (
        ("CPU and memory", budget.compute),
        ("Storage", budget.storage),
    ):
        age = quote.age_days(as_of=as_of)
        if age < 0:
            blockers.append(
                f"{label} price is dated {quote.effective_date}, which is in "
                "the future relative to the planning date; correct the quote."
            )
        elif quote.is_stale(as_of=as_of, max_age_days=max_price_age_days):
            message = (
                f"{label} price quote is {age} days old (effective "
                f"{quote.effective_date}, limit {max_price_age_days} days). "
                "It is a mandatory part of the total, so a stale rate makes "
                "the total wrong."
            )
            if accept_stale_price:
                warnings.append(message + " Accepted explicitly.")
            else:
                blockers.append(message)

    if not recipe.image.is_digest_pinned:
        warnings.append(
            f"Serving image {recipe.image.reference} is pinned by tag only. "
            "A tag can be repointed, so this deployment is reproducible only "
            "as long as the tag is not moved."
        )

    if recipe.exposes_credential_on_argv:
        message = (
            f"{recipe.framework} accepts the endpoint key only as a command "
            "line argument, and its engine logs its own resolved server "
            "configuration without redacting it, so the key will appear in "
            "the container's logs and in /proc/<pid>/cmdline. "
            f"{VLLM} reads {VLLM_API_KEY_ENV_VAR} from the environment "
            "instead and does not have this problem."
        )
        if accept_argv_credential_exposure:
            warnings.append(message + " Accepted explicitly.")
        else:
            blockers.append(message)

    if controls.min_containers > 0:
        # Not a warning. The envelope prices a window that begins at
        # deploy and is closed by the expiry, but a warm container bills
        # from deploy regardless of whether anyone calls it, so the whole
        # window is spent by definition rather than in the worst case.
        # This harness exists for a short validation; paying for idle
        # accelerators is not one.
        blockers.append(
            f"min_containers is {controls.min_containers}: Modal keeps that "
            "many containers running while the function is idle, so the "
            "accelerators bill continuously from deploy whether or not "
            "anything calls them. Set min_containers to 0 and accept the "
            "cold start."
        )
    if controls.max_concurrent_inputs > 1:
        warnings.append(
            f"max_concurrent_inputs is {controls.max_concurrent_inputs}; "
            "concurrent requests share one container and their measurements "
            "will interfere with each other."
        )

    # The expiry stops a container from *serving* past the window, but it
    # cannot stop one from being *started*: a web server function is
    # scheduled by the platform before any of this module's code runs, so
    # the refusal happens on an accelerator that has already been
    # allocated and billed. On a public endpoint that is an unbounded
    # cost, because anyone can keep triggering cold starts forever and
    # each one bills. Proxy auth is the only thing that closes it, since
    # Modal rejects an unauthenticated request at its edge with a 401
    # before scheduling anything
    # (https://modal.com/docs/guide/webhook-proxy-auth, read 2026-08-30).
    warnings.append(
        "Proxy auth is on, so Modal rejects unauthenticated requests at its "
        "edge before scheduling a container and they cannot allocate "
        "accelerators. A request bearing a valid workspace token still "
        "cold-starts a container, including after the expiry, where it is "
        "refused within seconds. That residual is operator-controlled and "
        "is not priced above."
    )

    steps = (
        commands.setup_step(),
        commands.secret_step(endpoint),
        commands.volume_step(volume_name),
        commands.stage_weights_step(
            revision=recipe.model_revision,
            volume_name=volume_name,
            repo_id=recipe.model_repo_id,
        ),
        commands.verify_weights_step(
            volume_name=volume_name, revision=recipe.model_revision
        ),
        commands.manifest_step(volume_name=volume_name, revision=recipe.model_revision),
        commands.deploy_step(),
        commands.health_step(endpoint),
        commands.readiness_step(endpoint),
        commands.smoke_step(
            endpoint=endpoint, max_output_tokens=smoke_max_output_tokens
        ),
        commands.collect_step(
            collector_argv(
                endpoint=endpoint,
                run_id=collector_run_id,
                prompt_file=collector_prompt_file,
                output_dir=collector_output_dir,
                model_revision=recipe.model_revision,
            )
        ),
        commands.stop_step(app_name=app_name),
        commands.delete_volume_step(volume_name=volume_name),
    )

    return DeploymentPlan(
        schema_version=PLAN_SCHEMA_VERSION,
        recipe=recipe,
        controls=controls,
        budget=budget,
        envelope=envelope,
        memory=memory,
        endpoint=endpoint,
        app_name=app_name,
        volume_name=volume_name,
        steps=steps,
        max_price_age_days=max_price_age_days,
        accept_stale_price=accept_stale_price,
        blockers=tuple(blockers),
        warnings=tuple(warnings),
    )


def assert_executable(plan: DeploymentPlan) -> None:
    """Raise unless every gate passed."""
    if plan.approved:
        return
    raise DeploymentPlanError(
        "deployment plan refused:\n"
        + "\n".join(f"  - {reason}" for reason in plan.blockers)
    )
