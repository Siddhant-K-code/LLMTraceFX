"""Command generation. Nothing in this module executes anything.

The harness deliberately stops at printing the command an operator would
run. Two reasons, and the second is the important one.

The obvious reason is safety: a planner that can also launch is one
mis-parsed flag away from launching, and the resource it would launch
costs money per second.

The less obvious reason is that a printed command is reviewable. The
operator sees the GPU count, the timeout and the revision before anything
starts, can paste the line into a review, and can re-run exactly it later.
A function that internally calls the same API leaves no such artifact.

Every step carries whether it can spend money and whether it needs
authentication, so a plan can assert, rather than assume, that its
default path does neither.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from typing import Any

from .endpoint import EndpointConfig, require_env_var_name
from .errors import DeploymentPlanError

APP_MODULE_PATH = "llmtracefx/deploy/modal_glm_app.py"
DEFAULT_APP_NAME = "llmtracefx-glm53flash"
DEFAULT_VOLUME_NAME = "llmtracefx-glm53flash-weights"

STAGE_FUNCTION = "stage_weights"
VERIFY_FUNCTION = "verify_weights"
MANIFEST_FUNCTION = "read_manifest"

# Characters that keep their meaning inside a double-quoted shell word.
# Everything else is literal there, so escaping exactly these is what
# makes an interpolated prefix or suffix inert.
_DOUBLE_QUOTE_ESCAPES = str.maketrans(
    {"\\": "\\\\", '"': '\\"', "$": "\\$", "`": "\\`"}
)


@dataclass(frozen=True)
class EnvRef:
    """A deliberate shell expansion, and the only one the renderer emits.

    Every other token is quoted literally, so the single place a command
    can still reach the shell is here. That is the point of making it a
    type rather than a string: the name is validated as an environment
    variable name, which excludes every metacharacter, and the literal
    text around it is escaped for the double-quoted context it lands in.
    An operator-supplied value therefore cannot become an expansion by
    accident, and an expansion cannot carry an operator-supplied value.
    """

    name: str
    prefix: str = ""
    suffix: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", require_env_var_name(self.name, field="EnvRef name")
        )

    def display(self) -> str:
        """The form recorded in artifacts: readable, and not a shell word."""
        return f"{self.prefix}${self.name}{self.suffix}"

    def rendered(self) -> str:
        """The form pasted into a shell, with the literals made inert."""
        prefix = self.prefix.translate(_DOUBLE_QUOTE_ESCAPES)
        suffix = self.suffix.translate(_DOUBLE_QUOTE_ESCAPES)
        return f'"{prefix}${{{self.name}}}{suffix}"'


Token = str | EnvRef


@dataclass(frozen=True)
class CommandStep:
    """One operator action, described rather than performed."""

    name: str
    purpose: str
    argv: tuple[Token, ...]
    requires_auth: bool
    spends_money: bool
    notes: tuple[str, ...] = ()

    def rendered(self) -> str:
        """A copy-pasteable rendering that cannot smuggle in a command.

        Every literal token goes through ``shlex.quote``. The previous
        rendering quoted only tokens containing whitespace, a quote or a
        backslash, which left ``;``, ``|``, ``&``, ``$(...)`` and
        backticks untouched: an app name or volume name chosen by whoever
        supplied the plan would then execute when the operator pasted the
        line. Quoting unconditionally costs some readability on unusual
        tokens and removes that entirely.
        """
        return " ".join(
            token.rendered() if isinstance(token, EnvRef) else shlex.quote(token)
            for token in self.argv
        )

    def display_argv(self) -> tuple[str, ...]:
        return tuple(
            token.display() if isinstance(token, EnvRef) else token
            for token in self.argv
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "purpose": self.purpose,
            "argv": list(self.display_argv()),
            "command": self.rendered(),
            "requires_auth": self.requires_auth,
            "spends_money": self.spends_money,
            "notes": list(self.notes),
        }


def setup_step() -> CommandStep:
    return CommandStep(
        name="setup",
        purpose="Authenticate the local Modal CLI against your workspace.",
        argv=("modal", "setup"),
        requires_auth=False,
        spends_money=False,
        notes=("Opens a browser and writes a token to ~/.modal.toml.",),
    )


def secret_step(endpoint: EndpointConfig) -> CommandStep:
    """Create the Modal Secret without ever writing the value down.

    The value is referenced as a shell variable, so the generated text
    contains the variable's name and not its contents. The command can
    therefore be printed, saved to a plan file and pasted into a review
    with nothing to redact.
    """
    return CommandStep(
        name="secret",
        purpose=(
            "Store the endpoint API key as a Modal Secret so the serving "
            "container receives it as an environment variable."
        ),
        argv=(
            "modal",
            "secret",
            "create",
            endpoint.modal_secret_name,
            EnvRef(endpoint.api_key_env_var, prefix=f"{endpoint.api_key_env_var}="),
        ),
        requires_auth=True,
        spends_money=False,
        notes=(
            f"Export {endpoint.api_key_env_var} in your shell first; the "
            "generated command references the variable and never contains "
            "the value.",
            "Creating the secret in the Modal dashboard instead keeps it out "
            "of shell history entirely.",
        ),
    )


def volume_step(volume_name: str) -> CommandStep:
    return CommandStep(
        name="volume",
        purpose="Create the persistent volume that will hold the weights.",
        argv=("modal", "volume", "create", volume_name),
        requires_auth=True,
        spends_money=False,
        notes=(
            "Creating an empty volume is free; storing weights on it is not. "
            "Storage bills until the volume is deleted.",
        ),
    )


def stage_weights_step(
    *,
    revision: str,
    volume_name: str,
    app_module: str = APP_MODULE_PATH,
    repo_id: str = "zai-org/GLM-5.3-Flash",
) -> CommandStep:
    """Download weights on CPU only, with the revision stated twice.

    ``--confirm`` repeats ``--revision`` rather than being a bare yes/no
    flag. A yes/no flag can be carried unchanged from an older command
    line and confirms whatever that line now says; restating the SHA
    means confirmation is specific to the revision being fetched, so a
    stale copy-paste fails instead of silently downloading something
    else.
    """
    return CommandStep(
        name="stage-weights",
        purpose=(
            "Download the pinned model revision onto the volume using CPU "
            "and network only, then write a manifest of what landed."
        ),
        argv=(
            "modal",
            "run",
            f"{app_module}::{STAGE_FUNCTION}",
            "--repo-id",
            repo_id,
            "--revision",
            revision,
            "--confirm",
            revision,
            "--volume-name",
            volume_name,
        ),
        requires_auth=True,
        spends_money=True,
        notes=(
            "No GPU is attached to this step. Weights are never downloaded "
            "on an accelerator.",
            "This transfers hundreds of GiB and takes a long time; the cost "
            "is CPU container time plus the storage it leaves behind.",
            "Re-running with the same revision is a no-op once the manifest "
            "matches.",
        ),
    )


def verify_weights_step(
    *, volume_name: str, revision: str, app_module: str = APP_MODULE_PATH
) -> CommandStep:
    """Check the staged inventory on CPU, before any accelerator exists.

    A manifest that names the right revision does not prove the bytes
    are there. A download interrupted near the end leaves a manifest and
    a short file, and the only place that discrepancy is cheap to find
    is a CPU container: discovering it after the serving container has
    allocated four accelerators means paying for the discovery.
    """
    return CommandStep(
        name="verify-weights",
        purpose=(
            "Re-check every staged file against the manifest, on CPU, and "
            "record the result the serving container requires."
        ),
        argv=(
            "modal",
            "run",
            f"{app_module}::{VERIFY_FUNCTION}",
            "--volume-name",
            volume_name,
            "--revision",
            revision,
        ),
        requires_auth=True,
        spends_money=True,
        notes=(
            "CPU only. No accelerator is allocated to find a truncated " "download.",
            "The serving container refuses to start until this has passed "
            "for the revision it is configured with.",
        ),
    )


def manifest_step(
    *, volume_name: str, revision: str, app_module: str = APP_MODULE_PATH
) -> CommandStep:
    return CommandStep(
        name="manifest",
        purpose="Print the staging manifest recorded on the volume.",
        argv=(
            "modal",
            "run",
            f"{app_module}::{MANIFEST_FUNCTION}",
            "--volume-name",
            volume_name,
            "--revision",
            revision,
        ),
        requires_auth=True,
        spends_money=True,
        notes=("CPU only, seconds of container time.",),
    )


def deploy_step(*, app_module: str = APP_MODULE_PATH) -> CommandStep:
    return CommandStep(
        name="deploy",
        purpose="Deploy the serving app and obtain its https URL.",
        argv=("modal", "deploy", app_module),
        requires_auth=True,
        spends_money=True,
        notes=(
            "Deploying registers the app. GPUs are allocated on the first "
            "request and released after the scaledown window.",
            "Every budget parameter is read from the environment at deploy "
            "time; the app refuses to build if any is missing.",
        ),
    )


def health_step(endpoint: EndpointConfig) -> CommandStep:
    return CommandStep(
        name="health",
        purpose="Liveness probe. Cheap, bounded, no model work.",
        argv=(
            "curl",
            "-fsS",
            "--max-time",
            "30",
            "-H",
            # Carried even on the liveness probe, because proxy auth is
            # enforced at the edge for every path. Without it this gets a
            # 401 and never reaches the server, which would read as a
            # broken deployment rather than a working one.
            EnvRef(endpoint.api_key_env_var, prefix="Authorization: Bearer "),
            endpoint.health_url,
        ),
        requires_auth=False,
        spends_money=True,
        notes=(
            "The first authenticated call cold-starts the container and "
            "therefore allocates GPUs.",
        ),
    )


def readiness_step(endpoint: EndpointConfig) -> CommandStep:
    return CommandStep(
        name="readiness",
        purpose=(
            "Readiness probe: the model list is only served once weights " "are loaded."
        ),
        argv=(
            "curl",
            "-fsS",
            "--max-time",
            "60",
            "-H",
            EnvRef(endpoint.api_key_env_var, prefix="Authorization: Bearer "),
            endpoint.readiness_url,
        ),
        requires_auth=False,
        spends_money=True,
        notes=(
            "The header references a shell variable, so the command text "
            "never contains the key.",
        ),
    )


def smoke_step(
    *,
    endpoint: EndpointConfig,
    max_output_tokens: int,
) -> CommandStep:
    """One bounded generation, sized so a hung server cannot run away."""
    if max_output_tokens < 1:
        raise DeploymentPlanError("max_output_tokens must be at least 1")
    payload = json.dumps(
        {
            "model": endpoint.served_model_name,
            "max_tokens": max_output_tokens,
            "stream": False,
            "messages": [{"role": "user", "content": "Reply with the word: ready"}],
        },
        separators=(",", ":"),
    )
    return CommandStep(
        name="smoke",
        purpose="One bounded chat completion proving the endpoint serves.",
        argv=(
            "curl",
            "-fsS",
            "--max-time",
            "120",
            "-H",
            "Content-Type: application/json",
            "-H",
            EnvRef(endpoint.api_key_env_var, prefix="Authorization: Bearer "),
            "-d",
            payload,
            endpoint.chat_completions_url,
        ),
        requires_auth=False,
        spends_money=True,
        notes=(
            f"Capped at {max_output_tokens} output tokens so a misbehaving "
            "server cannot generate indefinitely.",
            "This proves the endpoint works. It is not a benchmark: timing "
            "evidence comes from the collector step.",
        ),
    )


def collect_step(argv: tuple[str, ...]) -> CommandStep:
    return CommandStep(
        name="collect",
        purpose=(
            "Measure the endpoint with the existing LLMTraceFX API "
            "collector, which owns all timing evidence."
        ),
        argv=argv,
        requires_auth=False,
        spends_money=True,
        notes=(
            "Metrics are produced outside the server on purpose: the server "
            "does not measure or grade itself.",
        ),
    )


def stop_step(*, app_name: str) -> CommandStep:
    return CommandStep(
        name="stop",
        purpose="Stop the app so no container can be started by a request.",
        argv=("modal", "app", "stop", app_name),
        requires_auth=True,
        spends_money=False,
        notes=("Do this first during teardown; it is what stops GPU spend.",),
    )


def delete_volume_step(*, volume_name: str) -> CommandStep:
    return CommandStep(
        name="delete-volume",
        purpose="Delete the weights volume so storage stops accruing.",
        argv=("modal", "volume", "delete", volume_name),
        requires_auth=True,
        spends_money=False,
        notes=(
            "Storage is the charge that outlives the experiment. Modal "
            "bills deleted data for up to four days after deletion.",
            "Skip this only if you intend to serve again soon; re-staging "
            "means transferring the whole checkpoint again.",
        ),
    )
