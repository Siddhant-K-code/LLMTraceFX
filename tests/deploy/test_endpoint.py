"""Endpoint wiring: names, never values."""

from __future__ import annotations

import pytest

from llmtracefx.deploy.endpoint import (
    DEFAULT_API_KEY_ENV_VAR,
    EndpointConfig,
    require_endpoint_base_url,
    require_env_var_name,
)
from llmtracefx.deploy.errors import DeploymentPlanError


def test_an_unresolved_endpoint_renders_a_placeholder_rather_than_a_guess() -> None:
    endpoint = EndpointConfig()
    assert endpoint.is_resolved is False
    assert endpoint.chat_completions_url == "<deployed-url>/v1/chat/completions"
    assert endpoint.health_url == "<deployed-url>/health"


def test_a_resolved_endpoint_builds_absolute_paths() -> None:
    endpoint = EndpointConfig(base_url="https://ws--app-serve.modal.run/")
    assert endpoint.base_url == "https://ws--app-serve.modal.run"
    assert endpoint.readiness_url == "https://ws--app-serve.modal.run/v1/models"


@pytest.mark.parametrize(
    "candidate",
    [
        "sk-abcdef0123456789abcdef0123456789",
        "Bearer abc",
        "my key",
        "",
        "a" * 65,
        "1LEADING_DIGIT",
        "has-dashes",
    ],
)
def test_key_shaped_values_are_refused_where_a_name_belongs(candidate: str) -> None:
    with pytest.raises(DeploymentPlanError, match="NAME"):
        require_env_var_name(candidate)


def test_ordinary_variable_names_are_accepted() -> None:
    assert require_env_var_name(DEFAULT_API_KEY_ENV_VAR) == DEFAULT_API_KEY_ENV_VAR
    assert require_env_var_name("_PRIVATE") == "_PRIVATE"


@pytest.mark.parametrize(
    "url",
    [
        "http://ws--app.modal.run",
        "ftp://ws--app.modal.run",
        "https://user:secret@ws--app.modal.run",
        "https://ws--app.modal.run/v1?api_key=abc",
        "https://ws--app.modal.run/v1#token",
        "https://",
        "",
    ],
)
def test_unsafe_base_urls_are_refused(url: str) -> None:
    with pytest.raises(DeploymentPlanError):
        require_endpoint_base_url(url)


def test_a_secret_name_is_not_allowed_to_be_a_secret_value() -> None:
    with pytest.raises(DeploymentPlanError, match="never the secret's value"):
        EndpointConfig(modal_secret_name="sk-live-" + "x" * 80)


def test_serialised_endpoint_carries_names_and_an_explanation_only() -> None:
    payload = EndpointConfig().to_dict()
    assert payload["api_key_env_var"] == DEFAULT_API_KEY_ENV_VAR
    assert payload["modal_secret_name"] == "glm-selfhost-api-key"
    assert "Only the secret name" in payload["credential_handling"]
    assert set(payload) == {
        "api_key_env_var",
        "modal_secret_name",
        "served_model_name",
        "base_url",
        "resolved",
        "health_url",
        "readiness_url",
        "chat_completions_url",
        "credential_handling",
    }


def test_a_malformed_bracketed_host_is_refused_not_raised() -> None:
    """urlsplit defers netloc parsing, so the ValueError arrives late.

    Left unguarded it escapes as a bare ValueError rather than the
    refusal every other bad URL produces.
    """
    with pytest.raises(DeploymentPlanError, match="not parseable"):
        require_endpoint_base_url("https://[bad")
