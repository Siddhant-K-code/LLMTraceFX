"""``llmtracefx-deploy``: plan a self-host without being able to run one.

Three commands, none of which can spend anything:

``recipe``   print the pinned model facts and their sources.
``budget``   turn an available credit balance into a session cap.
``plan``     adjudicate a full deployment and print the commands.

``plan`` is the dry run. It needs no Modal authentication, opens no socket
and allocates nothing, so it can be run before the operator has even
installed the Modal CLI. What it prints is a decision plus the exact
commands that would carry it out, and when the decision is negative the
money-spending commands are withheld from the executable set rather than
merely annotated.

Every parameter that governs spending is required and has no default.
Omitting one is an error, never an assumption.

This is a separate console script rather than a subcommand of
``llmtracefx-optimizer``, so that adding it changes no existing module.
It does reuse that CLI's credential defences, which are the part worth
sharing: the parser that never echoes a caller-supplied value back into a
diagnostic, and the guard that refuses a credential-shaped flag before
argparse can format its value into a message.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

from ..optimizer.cli import (
    _CREDENTIAL_ARGUMENT_STEMS,
    SecureArgumentParser,
    _argument_scrub_scope,
    _option_stem,
)
from ..optimizer.collectors._shared import atomic_write_text
from ..optimizer.collectors.openai_api import redact_text_for_dry_run
from .budget import (
    MAX_GPU_COUNT_CEILING,
    MAX_RUNTIME_SECONDS_CEILING,
    BudgetRequest,
    recommended_session_budget_usd,
)
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
    SUPPORTED_FRAMEWORKS,
    SUPPORTED_REPO_ID,
    build_recipe,
)

PROG = "llmtracefx-deploy"


def _parse_as_of(value: str | None) -> date:
    if value is None:
        return date.today()
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise DeploymentPlanError("--as-of must be an ISO date (YYYY-MM-DD)") from exc


def render_plan_text(plan: DeploymentPlan) -> str:
    """A human summary that leads with the decision and the money."""
    envelope = plan.envelope
    memory = plan.memory
    recipe = plan.recipe
    lines: list[str] = []

    verdict = "APPROVED" if plan.approved else "REFUSED"
    lines.append(f"Deployment plan: {verdict}")
    lines.append("")
    lines.append("This command allocated nothing, authenticated nowhere and")
    lines.append("made no network request.")
    lines.append("")

    lines.append("Model")
    lines.append(f"  repository        {recipe.model_repo_id}")
    lines.append(f"  revision          {recipe.model_revision}")
    lines.append(
        f"  quantization      {recipe.facts.quantization} "
        f"({recipe.facts.quantization_format}, "
        f"{recipe.facts.activation_scheme} activation scaling)"
    )
    lines.append(
        f"  parameters        {recipe.facts.total_parameters_b:.0f}B total, "
        f"{recipe.facts.active_parameters_b:.0f}B active"
    )
    lines.append(f"  context cap       {recipe.context_length} tokens")
    lines.append("")

    lines.append("Serving")
    lines.append(f"  framework         {recipe.framework} {recipe.framework_version}")
    lines.append(f"  image             {recipe.image.reference}")
    lines.append(
        "  image pinned      "
        + ("by digest" if recipe.image.is_digest_pinned else "by tag only")
    )
    lines.append(f"  accelerators      {recipe.gpu_count} x {recipe.gpu_type}")
    lines.append(f"  tensor parallel   {recipe.tensor_parallel_size}")
    lines.append(f"  server argv       {' '.join(recipe.launch_argv())}")
    lines.append("")

    lines.append("Limits")
    lines.append(
        f"  deployment expiry {plan.controls.deployment_seconds}s "
        "after deploy (enforced: the container refuses to start)"
    )
    lines.append(f"  container timeout {plan.controls.timeout_seconds}s")
    lines.append(f"  startup timeout   {plan.controls.startup_timeout_seconds}s")
    lines.append(f"  scaledown window  {plan.controls.scaledown_window_seconds}s")
    lines.append(
        f"  containers        max {plan.controls.max_containers}, "
        f"min {plan.controls.min_containers}"
    )
    lines.append(f"  concurrency       {plan.controls.max_concurrent_inputs}")
    lines.append(
        "  endpoint auth     "
        + (
            "Modal proxy auth (unauthenticated requests are rejected at the "
            "edge, before a container is scheduled)"
            if plan.controls.require_proxy_auth
            else "PUBLIC (any request can allocate accelerators)"
        )
    )
    lines.append("")

    lines.append("Modeled cost envelope (planning estimate from supplied inputs)")
    lines.append(
        f"  billable window   {envelope.billable_seconds}s "
        f"({envelope.deployment_seconds}s deployment expiry plus one "
        f"{envelope.max_runtime_seconds}s container lifetime)"
    )
    for line in envelope.lines:
        lines.append(
            f"  {line.name:<17} ${line.usd:>8.2f}  "
            f"({line.hours:.2f} h at ${line.usd_per_hour:.4f}/h: {line.detail})"
        )
    lines.append(
        f"  rates read        {envelope.price_effective_date} from "
        f"{envelope.price_source}"
    )
    lines.append(f"  modeled total     ${envelope.worst_case_usd:.2f}")
    lines.append(
        "  request access    "
        + (
            "authenticated at the edge"
            if envelope.bounded
            else "PUBLIC: traffic can allocate unmodeled resources"
        )
    )
    lines.append(f"  planning threshold ${envelope.budget_usd:.2f}")
    lines.append(f"  modeled headroom   ${envelope.headroom_usd:.2f}")
    lines.append("")

    lines.append("Capacity check")
    lines.append(
        f"  {memory.gpu_count} x {memory.gpu_type} = "
        f"{memory.total_vram_gib:.0f} GiB against roughly "
        f"{memory.weights_gib:.0f} GiB of weights"
    )
    lines.append(
        f"  residual          {memory.residual_gib:.0f} GiB "
        f"({memory.residual_fraction:.0%}, minimum "
        f"{memory.required_headroom_fraction:.0%})"
    )
    lines.append(f"  caveat            {memory.caveat}")
    lines.append("")

    lines.append("Endpoint")
    lines.append(f"  chat completions  {plan.endpoint.chat_completions_url}")
    lines.append(f"  credential from   Modal Secret {plan.endpoint.modal_secret_name}")
    lines.append(
        f"  injected as       ${plan.endpoint.api_key_env_var} "
        "(name only; no value is ever recorded)"
    )
    lines.append("")

    if plan.blockers:
        lines.append("Blockers (paid steps withheld until every one is cleared)")
        for blocker in plan.blockers:
            lines.append(f"  - {blocker}")
        lines.append("")
    if plan.warnings:
        lines.append("Warnings")
        for warning in plan.warnings:
            lines.append(f"  - {warning}")
        lines.append("")

    lines.append("Steps")
    executable = {step.name for step in plan.executable_steps}
    for step in plan.steps:
        marker = "  " if step.name in executable else "x "
        cost = "$" if step.spends_money else " "
        lines.append(f"{marker}{cost} {step.name}: {step.purpose}")
        lines.append(f"      {step.rendered()}")
    lines.append("")
    lines.append("Legend: 'x' withheld by a blocker, '$' can spend money.")
    if not plan.approved:
        lines.append("")
        lines.append("No paid step may be run while this plan is refused.")
    return "\n".join(lines)


def _cmd_deploy_recipe(args: argparse.Namespace) -> int:
    facts = GLM_53_FLASH
    if args.format == "json":
        print(json.dumps(facts.to_dict(), indent=2, allow_nan=False))
        return 0
    print(f"Model            {facts.repo_id}")
    print(
        f"Parameters       {facts.total_parameters_b:.0f}B total, "
        f"{facts.active_parameters_b:.0f}B active"
    )
    print(f"Layers           {facts.num_hidden_layers}")
    print(
        f"Experts          {facts.num_routed_experts} routed, "
        f"{facts.num_experts_per_token} per token"
    )
    print(f"Attention        {facts.attention}")
    print(
        f"Quantization     {facts.quantization} ({facts.quantization_format}, "
        f"{facts.activation_scheme})"
    )
    print(f"Max context      {facts.max_position_embeddings} tokens")
    print(f"Multimodal       {'yes' if facts.multimodal else 'no'}")
    print(
        f"Checkpoint size  roughly {facts.approximate_checkpoint_gib:.0f} GiB "
        "(approximate; the staging manifest records what actually landed)"
    )
    print(f"Config source    {facts.config_source}")
    print(f"Model card       {facts.model_card}")
    return 0


def _cmd_deploy_budget(args: argparse.Namespace) -> int:
    try:
        recommended = recommended_session_budget_usd(args.credit_usd)
    except (DeploymentPlanError, ValueError, OverflowError) as exc:
        print(f"{PROG}: error: {exc}", file=sys.stderr)
        return 1
    reserve = args.credit_usd - recommended
    payload = {
        "kind": "llmtracefx.deploy.budget_recommendation",
        "available_credit_usd": round(args.credit_usd, 2),
        "recommended_session_budget_usd": recommended,
        "reserve_usd": round(reserve, 2),
        "rationale": (
            "One session is capped at a third of the balance so a failed "
            "start up, one retry and the volume storage that outlives the "
            "run all remain affordable."
        ),
    }
    if args.format == "json":
        print(json.dumps(payload, indent=2, allow_nan=False))
        return 0
    print(f"Available credit           ${args.credit_usd:.2f}")
    print(f"Recommended session cap    ${recommended:.2f}")
    print(f"Held in reserve            ${reserve:.2f}")
    print()
    print(payload["rationale"])
    print()
    print(f"Pass --max-usd {recommended:.2f} to `deploy plan`.")
    return 0


def _configured_credential(name: str | None) -> str | None:
    """The value held by the variable named, if it is set.

    Read so it can be scrubbed, never so it can be used. The planner has
    no use for the credential itself; resolving it is what lets the
    rendered document be checked against it before anything is printed.

    The name is stripped, because ``require_env_var_name`` strips before
    it validates and returns the stripped form. Looking up the raw
    argument instead would mean a padded ``--api-key-env " NAME "`` was
    accepted as a valid name everywhere else while this lookup missed and
    silently returned ``None``, turning the scrub into a no-op with
    nothing in the output to say so.
    """
    if not name:
        return None
    return os.environ.get(name.strip(), "").strip() or None


def _build_plan_from_args(args: argparse.Namespace) -> DeploymentPlan:
    price = GpuPriceQuote(
        gpu_type=args.gpu_type,
        usd_per_gpu_hour=args.usd_per_gpu_hour,
        effective_date=args.price_effective_date,
        source=args.price_source,
    )
    compute = ComputeQuote(
        usd_per_cpu_core_hour=args.usd_per_cpu_core_hour,
        usd_per_gib_memory_hour=args.usd_per_gib_memory_hour,
        effective_date=args.price_effective_date,
        source=args.price_source,
    )
    storage = StorageQuote(
        usd_per_gib_month=args.storage_usd_per_gib_month,
        effective_date=args.price_effective_date,
        source=args.price_source,
    )
    stored_gib = (
        GLM_53_FLASH.approximate_checkpoint_gib
        if args.stored_gib is None
        else args.stored_gib
    )
    budget = BudgetRequest(
        max_usd=args.max_usd,
        gpu_type=args.gpu_type,
        gpu_count=args.gpu_count,
        max_runtime_seconds=args.max_runtime_seconds,
        deployment_seconds=args.max_deployment_seconds,
        price=price,
        compute=compute,
        storage=storage,
        stored_gib=stored_gib,
        storage_retention_days=args.storage_retention_days,
        max_containers=args.max_containers,
    )
    recipe = build_recipe(
        framework=args.framework,
        framework_version=args.framework_version,
        image_reference=args.image,
        model_revision=args.model_revision,
        gpu_type=args.gpu_type,
        gpu_count=args.gpu_count,
        context_length=args.context_length,
        weights_mount_path=args.weights_mount_path,
        port=args.port,
        served_model_name=args.served_model_name,
        tensor_parallel_size=args.tensor_parallel_size,
        accept_mutable_image=args.accept_mutable_image,
    )
    controls = RuntimeControls(
        timeout_seconds=args.max_runtime_seconds,
        deployment_seconds=args.max_deployment_seconds,
        scaledown_window_seconds=args.scaledown_window_seconds,
        startup_timeout_seconds=args.startup_timeout_seconds,
        max_containers=args.max_containers,
        min_containers=args.min_containers,
        max_concurrent_inputs=args.max_concurrent_inputs,
        allow_warm_containers=args.allow_warm_containers,
    )
    endpoint = EndpointConfig(
        api_key_env_var=args.api_key_env,
        modal_secret_name=args.modal_secret_name,
        served_model_name=recipe.served_model_name,
        base_url=args.endpoint_base_url,
    )
    return build_plan(
        recipe=recipe,
        controls=controls,
        budget=budget,
        endpoint=endpoint,
        as_of=_parse_as_of(args.as_of),
        app_name=args.app_name,
        volume_name=args.volume_name,
        smoke_max_output_tokens=args.smoke_max_output_tokens,
        max_price_age_days=args.max_price_age_days,
        accept_stale_price=args.accept_stale_price,
        accept_argv_credential_exposure=args.accept_argv_credential_exposure,
    )


def _cmd_deploy_plan(args: argparse.Namespace) -> int:
    # Resolved before the plan is built, so the scrub is available on the
    # path where building the plan raised and there is no plan to read the
    # validated name from.
    credential = _configured_credential(getattr(args, "api_key_env", None))
    try:
        plan = _build_plan_from_args(args)
    except (DeploymentPlanError, ValueError, OverflowError) as exc:
        # Broader than DeploymentPlanError on purpose. Validation is
        # thorough but not total, and an escaping exception would reach
        # stderr as a traceback, which is the one output path that does
        # not go through the scrub below.
        print(
            redact_text_for_dry_run(f"{PROG}: error: {exc}", credential),
            file=sys.stderr,
        )
        return 1

    # Re-resolved from the name the plan actually validated, so the value
    # being scrubbed for cannot drift from the value the deployment will
    # use. Falls back to the pre-plan resolution rather than replacing it.
    credential = _configured_credential(plan.endpoint.api_key_env_var) or credential

    document = json.dumps(plan.to_dict(), indent=2, allow_nan=False)
    rendered = document if args.format == "json" else render_plan_text(plan)

    # Defence in depth on the way to stdout and to disk. Every field in
    # the plan is a name by construction and each is validated as one,
    # but this is the same barrier `collect-api --dry-run` puts in front
    # of its rendered plan: the document is checked against the resolved
    # credential rather than each field being trusted individually.
    text = redact_text_for_dry_run(rendered, credential)
    print(text)

    if args.output:
        try:
            atomic_write_text(
                Path(args.output),
                redact_text_for_dry_run(document, credential) + "\n",
            )
        except OSError as exc:
            print(
                redact_text_for_dry_run(f"Failed to write plan: {exc}", credential),
                file=sys.stderr,
            )
            return 1

    if not plan.approved:
        print(
            f"\n{PROG}: deployment refused; no paid step may be run.",
            file=sys.stderr,
        )
        return 1
    return 0


def _reject_credential_arguments(raw_argv: list[str]) -> None:
    """Refuse a credential-bearing flag before argparse can echo its value.

    Shares the optimizer CLI's stem list and normalizer, which are the
    parts worth reusing, but deliberately does not reuse its message.
    That one names the offending flag to be helpful, which means a token
    the caller typed reaches stderr. Here the diagnostic is a fixed
    string: nothing derived from ``raw_argv`` reaches the output at all,
    so there is no argument about whether the part that got echoed was
    only the name.
    """
    for token in raw_argv:
        if token == "--":
            return
        if not token.startswith("-"):
            continue
        if _option_stem(token) not in _CREDENTIAL_ARGUMENT_STEMS:
            continue
        print(
            f"{PROG}: error: that option is not supported, and a credential "
            "must never appear in a command line. Export the credential to "
            "an environment variable and name that variable with "
            "--api-key-env. The offending option is not repeated here so it "
            "cannot reach your shell history or CI log through this message.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``llmtracefx-deploy`` parser.

    ``SecureArgumentParser`` rather than a plain one, so a value the
    caller typed can never be quoted back into a diagnostic. Several of
    these flags sit next to a credential in an operator's shell history.
    """
    parser = SecureArgumentParser(
        prog=PROG,
        description=(
            "Plan a budget-guarded, pinned self-host of GLM-5.3-Flash on "
            "Modal. Never deploys, never authenticates, never spends."
        ),
    )
    deploy_subparsers = parser.add_subparsers(dest="deploy_command")
    deploy_subparsers.required = True

    recipe_parser = deploy_subparsers.add_parser(
        "recipe",
        help="Print the pinned GLM-5.3-Flash facts and where they came from",
    )
    recipe_parser.add_argument("--format", choices=("text", "json"), default="text")
    recipe_parser.set_defaults(func=_cmd_deploy_recipe)

    budget_parser = deploy_subparsers.add_parser(
        "budget",
        help=(
            "Recommend a conservative per-session spending cap from an "
            "available credit balance"
        ),
    )
    budget_parser.add_argument(
        "--credit-usd",
        type=float,
        required=True,
        help="Credit currently available, in US dollars",
    )
    budget_parser.add_argument("--format", choices=("text", "json"), default="text")
    budget_parser.set_defaults(func=_cmd_deploy_budget)

    plan_parser = deploy_subparsers.add_parser(
        "plan",
        help=(
            "Dry run: adjudicate a deployment and print the exact commands. "
            "Requires no Modal authentication and makes no network request; "
            "refuses the paid steps unless every budget and pinning gate "
            "passes."
        ),
        # Prefix matching would let a mistyped flag resolve to a different
        # one, and several flags here are the difference between a ten
        # dollar run and a hundred dollar one.
        allow_abbrev=False,
    )

    money = plan_parser.add_argument_group(
        "spending authority (all required; there are no defaults)"
    )
    money.add_argument(
        "--max-usd",
        type=float,
        required=True,
        help=(
            "Operator planning threshold for the modeled costs calculated "
            "from the supplied inputs; not a provider billing cap"
        ),
    )
    money.add_argument(
        "--gpu-type", required=True, help="Accelerator model, for example H200"
    )
    money.add_argument(
        "--gpu-count",
        type=int,
        required=True,
        help=f"Accelerators per container (1..{MAX_GPU_COUNT_CEILING})",
    )
    money.add_argument(
        "--max-runtime-seconds",
        type=int,
        required=True,
        help=(
            "Container timeout, and the runtime the modeled serving term is priced "
            f"against (1..{MAX_RUNTIME_SECONDS_CEILING})"
        ),
    )
    money.add_argument(
        "--usd-per-gpu-hour",
        type=float,
        required=True,
        help="Price you read for this accelerator, per GPU per hour",
    )
    money.add_argument(
        "--max-deployment-seconds",
        type=int,
        required=True,
        help=(
            "Wall-clock window the deployment may serve at all. The serving "
            "container refuses to start once it has passed, and this is what "
            "the accelerator cost is priced against"
        ),
    )
    money.add_argument(
        "--usd-per-cpu-core-hour",
        type=float,
        required=True,
        help="CPU rate; bills on every container, including the CPU-only ones",
    )
    money.add_argument(
        "--usd-per-gib-memory-hour",
        type=float,
        required=True,
        help="Memory rate; bills on every container",
    )
    money.add_argument(
        "--storage-usd-per-gib-month",
        type=float,
        required=True,
        help="Volume storage rate; the weights outlive the run",
    )
    money.add_argument(
        "--storage-retention-days",
        type=float,
        required=True,
        help=(
            "How long you will keep the weights on the volume. Modal bills "
            "deleted data for up to four more days, which is added for you"
        ),
    )
    money.add_argument(
        "--price-effective-date",
        required=True,
        help="ISO date (YYYY-MM-DD) on which you read these prices",
    )
    money.add_argument(
        "--price-source",
        required=True,
        help="Where you read it, normally the pricing page URL",
    )

    pinning = plan_parser.add_argument_group("pinning")
    pinning.add_argument(
        "--model-revision",
        required=True,
        help=(
            "Full 40-character commit SHA of the model repository. Branch "
            "names are refused"
        ),
    )
    pinning.add_argument(
        "--image",
        required=True,
        help=(
            "Serving container reference, ideally name:tag@sha256:<digest>. "
            "'latest' is always refused"
        ),
    )
    pinning.add_argument(
        "--framework",
        choices=SUPPORTED_FRAMEWORKS,
        default=DEFAULT_FRAMEWORK,
        help=(
            "Serving framework. Default vllm, because it reads the endpoint "
            "key from the environment; sglang takes it on the command line "
            "and logs its own configuration unredacted (default: %(default)s)"
        ),
    )
    pinning.add_argument(
        "--framework-version",
        required=True,
        help="Version of the serving framework inside that image",
    )
    pinning.add_argument(
        "--accept-mutable-image",
        action="store_true",
        help="Allow a tag-only image reference and record that you chose to",
    )

    serving = plan_parser.add_argument_group("serving")
    serving.add_argument(
        "--context-length",
        type=int,
        required=True,
        help="Context cap to serve with; must not exceed the model maximum",
    )
    serving.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="Tensor parallel degree (default: --gpu-count)",
    )
    serving.add_argument(
        "--served-model-name",
        default=SUPPORTED_REPO_ID,
        help="Model id clients send (default: %(default)s)",
    )
    serving.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT)
    serving.add_argument("--weights-mount-path", default=DEFAULT_WEIGHTS_MOUNT_PATH)

    limits = plan_parser.add_argument_group("runtime limits")
    limits.add_argument(
        "--max-containers",
        type=int,
        default=1,
        help=(
            "Autoscaling ceiling. Default 1: every extra container "
            "multiplies the modeled serving term (default: %(default)s)"
        ),
    )
    limits.add_argument(
        "--min-containers",
        type=int,
        default=0,
        help=(
            "Warm container floor. Default 0 to avoid intentionally keeping "
            "an idle serving container warm "
            "(default: %(default)s)"
        ),
    )
    limits.add_argument(
        "--allow-warm-containers",
        action="store_true",
        help="Required before --min-containers may exceed zero",
    )
    limits.add_argument(
        "--scaledown-window-seconds",
        type=int,
        default=DEFAULT_SCALEDOWN_WINDOW_SECONDS,
        help="Idle seconds before accelerators are released (default: %(default)s)",
    )
    limits.add_argument(
        "--startup-timeout-seconds",
        type=int,
        default=1800,
        help="How long the server may take to become ready (default: %(default)s)",
    )
    limits.add_argument(
        "--max-concurrent-inputs",
        type=int,
        default=1,
        help=(
            "Requests served concurrently per container. Default 1 so "
            "measurements do not interfere (default: %(default)s)"
        ),
    )

    storage = plan_parser.add_argument_group("volume storage")
    storage.add_argument(
        "--stored-gib",
        type=float,
        default=None,
        help="GiB kept on the volume (default: the checkpoint size)",
    )

    freshness = plan_parser.add_argument_group("price freshness")
    freshness.add_argument(
        "--max-price-age-days",
        type=int,
        default=DEFAULT_MAX_PRICE_AGE_DAYS,
        help="Refuse quotes older than this (default: %(default)s)",
    )
    freshness.add_argument(
        "--accept-argv-credential-exposure",
        action="store_true",
        help=(
            "Allow a framework that takes the endpoint key on its command "
            "line and logs it. Only sglang needs this"
        ),
    )
    freshness.add_argument(
        "--accept-stale-price",
        action="store_true",
        help="Downgrade a stale GPU quote from a blocker to a warning",
    )
    freshness.add_argument(
        "--as-of",
        default=None,
        help="ISO date to age quotes against (default: today)",
    )

    wiring = plan_parser.add_argument_group("endpoint and naming")
    wiring.add_argument(
        "--api-key-env",
        default=DEFAULT_API_KEY_ENV_VAR,
        help=(
            "Name of the environment variable carrying the endpoint key. "
            "Only the name is used or recorded (default: %(default)s)"
        ),
    )
    wiring.add_argument(
        "--modal-secret-name",
        default=DEFAULT_MODAL_SECRET_NAME,
        help="Modal Secret holding that variable (default: %(default)s)",
    )
    wiring.add_argument(
        "--endpoint-base-url",
        default=None,
        help=(
            "https base URL once deployed; omit before the first deploy and "
            "the plan renders a placeholder"
        ),
    )
    wiring.add_argument("--app-name", default=DEFAULT_APP_NAME)
    wiring.add_argument("--volume-name", default=DEFAULT_VOLUME_NAME)
    wiring.add_argument(
        "--smoke-max-output-tokens",
        type=int,
        default=32,
        help="Output cap on the single smoke request (default: %(default)s)",
    )

    plan_parser.add_argument("--format", choices=("text", "json"), default="text")
    plan_parser.add_argument(
        "--output",
        default=None,
        help="Atomically write the plan JSON to this path as well",
    )
    plan_parser.set_defaults(func=_cmd_deploy_plan)

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point. Mirrors the optimizer CLI's credential handling.

    The scrub scope wraps the credential guard as well as the parse, so
    the one diagnostic argparse emits for an unrecognized argument cannot
    echo a caller-supplied token either.
    """
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    with _argument_scrub_scope(parser, raw_argv):
        _reject_credential_arguments(raw_argv)
        args = parser.parse_args(raw_argv)
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
