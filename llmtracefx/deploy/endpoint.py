"""Wiring the deployed endpoint into the existing API collector.

Two rules shape this module.

The first is that a credential is referred to by the *name* of the
environment variable that carries it and never by its value. That is the
convention the API collector already follows, and it is what lets a plan,
a manifest and a generated command all be written to disk and pasted into
an issue without redacting anything.

The second is that this harness does not measure anything itself. The
collector already streams an OpenAI-compatible completion and records
normalized timing evidence; duplicating that inside the server would
produce a second set of numbers with no provenance and a server that has
opinions about its own performance. So what this module produces is the
collector invocation, not a benchmark.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .errors import DeploymentPlanError

# POSIX-ish environment variable name. Deliberately narrow: it excludes
# every character an API key is likely to contain, so a key pasted into
# this field is rejected on shape alone rather than being written down as
# if it were a name.
_ENV_VAR_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")

# Modal secret names are lowercase, dash separated in practice; keep the
# check loose enough to accept what Modal accepts and tight enough to
# reject a pasted credential.
_SECRET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

DEFAULT_API_KEY_ENV_VAR = "GLM_SELFHOST_API_KEY"
DEFAULT_MODAL_SECRET_NAME = "glm-selfhost-api-key"

HEALTH_PATH = "/health"
READINESS_PATH = "/v1/models"
CHAT_COMPLETIONS_PATH = "/v1/chat/completions"


def require_env_var_name(name: str, *, field: str = "api_key_env_var") -> str:
    """Accept an environment variable name, reject anything key-shaped."""
    if not isinstance(name, str) or not _ENV_VAR_PATTERN.match(name.strip()):
        raise DeploymentPlanError(
            f"{field} must be an environment variable NAME such as "
            f"{DEFAULT_API_KEY_ENV_VAR}, not a credential value. Names match "
            "[A-Za-z_][A-Za-z0-9_]* and are at most 64 characters."
        )
    return name.strip()


def require_secret_name(name: str) -> str:
    if not isinstance(name, str) or not _SECRET_NAME_PATTERN.match(name.strip()):
        raise DeploymentPlanError(
            "modal_secret_name must be a Modal Secret name such as "
            f"{DEFAULT_MODAL_SECRET_NAME}, never the secret's value"
        )
    return name.strip()


def require_endpoint_base_url(url: str) -> str:
    """Require an https base URL with no credential embedded in it.

    Userinfo in a URL is the classic way a key ends up in a log line, and
    the collector refuses it downstream. Refusing it here as well means
    the plan that is written to disk never contains it either.
    """
    if not isinstance(url, str) or not url.strip():
        raise DeploymentPlanError("endpoint base URL must be a non-empty string")
    candidate = url.strip().rstrip("/")
    try:
        parsed = urlparse(candidate)
        # The netloc is parsed lazily, so a malformed bracketed host does
        # not raise until something reads it. Forcing that here keeps the
        # failure inside this guard, where it becomes a refusal, instead
        # of surfacing later as a bare ValueError.
        _ = parsed.netloc
    except ValueError as exc:
        raise DeploymentPlanError(f"endpoint base URL is not parseable: {exc}") from exc
    if parsed.scheme != "https":
        raise DeploymentPlanError(
            f"endpoint base URL must use https, got {parsed.scheme or 'no scheme'}"
        )
    if not parsed.netloc:
        raise DeploymentPlanError("endpoint base URL must include a host")
    if "@" in parsed.netloc:
        raise DeploymentPlanError(
            "endpoint base URL must not embed credentials; pass the key "
            "through an environment variable instead"
        )
    if parsed.query or parsed.fragment:
        raise DeploymentPlanError(
            "endpoint base URL must not carry a query string or fragment; "
            "those are where a token gets smuggled in"
        )
    return candidate


@dataclass(frozen=True)
class EndpointConfig:
    """How a client reaches the served model, names only.

    ``base_url`` is optional because it does not exist until the app has
    been deployed and Modal has assigned a URL. Planning therefore has to
    be able to describe the endpoint before it can address it, which is
    why the plan renders a placeholder rather than inventing a hostname.
    """

    api_key_env_var: str = DEFAULT_API_KEY_ENV_VAR
    modal_secret_name: str = DEFAULT_MODAL_SECRET_NAME
    served_model_name: str = "zai-org/GLM-5.3-Flash"
    base_url: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "api_key_env_var", require_env_var_name(self.api_key_env_var)
        )
        object.__setattr__(
            self, "modal_secret_name", require_secret_name(self.modal_secret_name)
        )
        if not isinstance(self.served_model_name, str) or not self.served_model_name:
            raise DeploymentPlanError("served_model_name must be a non-empty string")
        if self.base_url is not None:
            object.__setattr__(
                self, "base_url", require_endpoint_base_url(self.base_url)
            )

    @property
    def is_resolved(self) -> bool:
        return self.base_url is not None

    def _url(self, path: str) -> str:
        if self.base_url is None:
            return f"<deployed-url>{path}"
        return f"{self.base_url}{path}"

    @property
    def health_url(self) -> str:
        return self._url(HEALTH_PATH)

    @property
    def readiness_url(self) -> str:
        return self._url(READINESS_PATH)

    @property
    def chat_completions_url(self) -> str:
        return self._url(CHAT_COMPLETIONS_PATH)

    def to_dict(self) -> dict[str, Any]:
        return {
            "api_key_env_var": self.api_key_env_var,
            "modal_secret_name": self.modal_secret_name,
            "served_model_name": self.served_model_name,
            "base_url": self.base_url,
            "resolved": self.is_resolved,
            "health_url": self.health_url,
            "readiness_url": self.readiness_url,
            "chat_completions_url": self.chat_completions_url,
            "credential_handling": (
                "The API key is stored in a Modal Secret and injected into "
                "the serving container as an environment variable. Only the "
                "secret name and the variable name are ever recorded."
            ),
        }


def collector_argv(
    *,
    endpoint: EndpointConfig,
    run_id: str,
    prompt_file: str,
    output_dir: str,
    provider: str = "modal-selfhost",
    max_output_tokens: int | None = None,
    request_timeout_seconds: float = 120.0,
    model_revision: str | None = None,
) -> tuple[str, ...]:
    """The exact ``collect-api`` invocation for the deployed endpoint.

    Returned as a tuple so it is executed, if the operator chooses to
    execute it, without a shell in between. There is no credential in it
    by construction: the key is named, not passed.
    """
    if not run_id.strip():
        raise DeploymentPlanError("run_id must be a non-empty string")
    argv = [
        "llmtracefx-optimizer",
        "collect-api",
        "--run-id",
        run_id.strip(),
        "--provider",
        provider,
        "--endpoint",
        endpoint.chat_completions_url,
        "--model-id",
        endpoint.served_model_name,
        "--prompt-file",
        prompt_file,
        "--output-dir",
        output_dir,
        "--api-key-env",
        endpoint.api_key_env_var,
        "--request-timeout",
        str(request_timeout_seconds),
    ]
    if max_output_tokens is not None:
        argv.extend(("--max-output-tokens", str(max_output_tokens)))
    if model_revision is not None:
        argv.extend(("--model-revision", model_revision))
    return tuple(argv)
