"""Generated commands, and what they are allowed to contain."""

from __future__ import annotations

import json
import shlex

import pytest
from _fakes import VALID_REVISION

from llmtracefx.deploy import commands
from llmtracefx.deploy.commands import EnvRef
from llmtracefx.deploy.endpoint import EndpointConfig, collector_argv
from llmtracefx.deploy.errors import DeploymentPlanError

ENDPOINT = EndpointConfig(base_url="https://ws--app-serve.modal.run")


def test_staging_states_the_revision_twice() -> None:
    step = commands.stage_weights_step(revision=VALID_REVISION, volume_name="vol")
    argv = step.argv
    assert argv[argv.index("--revision") + 1] == VALID_REVISION
    assert argv[argv.index("--confirm") + 1] == VALID_REVISION
    assert "stage_weights" in argv[2]


def test_staging_never_requests_an_accelerator_on_the_command_line() -> None:
    step = commands.stage_weights_step(revision=VALID_REVISION, volume_name="vol")
    joined = " ".join(step.argv).casefold()
    assert "gpu" not in joined
    assert any("No GPU is attached" in note for note in step.notes)


def test_secret_creation_references_a_variable_and_not_a_value() -> None:
    step = commands.secret_step(ENDPOINT)
    assignment = step.argv[-1]
    assert isinstance(assignment, EnvRef)
    assert assignment.display() == "GLM_SELFHOST_API_KEY=$GLM_SELFHOST_API_KEY"
    assert step.rendered().endswith('"GLM_SELFHOST_API_KEY=${GLM_SELFHOST_API_KEY}"')
    assert step.spends_money is False


def test_probes_pass_the_key_by_shell_reference() -> None:
    header = next(
        token
        for token in commands.readiness_step(ENDPOINT).argv
        if isinstance(token, EnvRef)
    )
    assert header.display() == "Authorization: Bearer $GLM_SELFHOST_API_KEY"
    assert header.rendered() == '"Authorization: Bearer ${GLM_SELFHOST_API_KEY}"'


@pytest.mark.parametrize(
    "hostile",
    [
        "app; rm -rf /",
        "app$(id)",
        "app`id`",
        "app|cat /etc/passwd",
        "app > /tmp/pwned",
        "app && curl evil.example",
        "app'''; id; '''",
        'app"; id; "',
        "app\nid",
    ],
)
def test_a_hostile_name_cannot_escape_the_rendered_command(hostile: str) -> None:
    """The rendered line is what an operator pastes into a shell.

    Quoting only tokens containing whitespace left ``;``, ``|``, ``$()``
    and backticks bare, so a name chosen by whoever supplied the plan
    executed on paste. Every literal token is quoted now.
    """
    rendered = commands.stop_step(app_name=hostile).rendered()
    assert shlex.split(rendered) == ["modal", "app", "stop", hostile]

    rendered = commands.delete_volume_step(volume_name=hostile).rendered()
    assert shlex.split(rendered) == ["modal", "volume", "delete", hostile]


def test_a_hostile_endpoint_url_cannot_escape_either() -> None:
    endpoint = EndpointConfig(base_url="https://ws--app-serve.modal.run")
    step = commands.smoke_step(endpoint=endpoint, max_output_tokens=8)
    # shlex round-trips it, so nothing in the payload is a shell operator.
    assert shlex.split(step.rendered())[-1] == endpoint.chat_completions_url


def test_smoke_request_is_bounded() -> None:
    step = commands.smoke_step(endpoint=ENDPOINT, max_output_tokens=16)
    payload = step.argv[step.argv.index("-d") + 1]
    assert '"max_tokens":16' in payload
    assert '"stream":false' in payload
    assert "--max-time" in step.argv


def test_smoke_request_rejects_an_unbounded_token_budget() -> None:
    with pytest.raises(DeploymentPlanError, match="at least 1"):
        commands.smoke_step(endpoint=ENDPOINT, max_output_tokens=0)


def test_teardown_steps_cannot_spend() -> None:
    assert commands.stop_step(app_name="app").spends_money is False
    assert commands.delete_volume_step(volume_name="vol").spends_money is False


def test_steps_declare_whether_they_can_spend() -> None:
    assert commands.setup_step().spends_money is False
    assert commands.volume_step("vol").spends_money is False
    assert commands.deploy_step().spends_money is True
    assert (
        commands.stage_weights_step(
            revision=VALID_REVISION, volume_name="vol"
        ).spends_money
        is True
    )


def test_rendering_round_trips_through_a_shell_lexer() -> None:
    step = commands.smoke_step(endpoint=ENDPOINT, max_output_tokens=8)
    tokens = shlex.split(step.rendered())
    assert tokens[0] == "curl"
    payload = json.loads(tokens[tokens.index("-d") + 1])
    assert payload["max_tokens"] == 8


def test_collector_command_names_the_variable_holding_the_key() -> None:
    argv = collector_argv(
        endpoint=ENDPOINT,
        run_id="smoke",
        prompt_file="prompts/smoke.txt",
        output_dir="output/smoke",
        model_revision=VALID_REVISION,
    )
    assert argv[:2] == ("llmtracefx-optimizer", "collect-api")
    assert argv[argv.index("--api-key-env") + 1] == "GLM_SELFHOST_API_KEY"
    assert argv[argv.index("--model-revision") + 1] == VALID_REVISION
    assert "--api-key" not in argv


def test_collector_targets_the_chat_completions_path() -> None:
    argv = collector_argv(
        endpoint=ENDPOINT,
        run_id="smoke",
        prompt_file="p.txt",
        output_dir="out",
    )
    endpoint = argv[argv.index("--endpoint") + 1]
    assert endpoint == "https://ws--app-serve.modal.run/v1/chat/completions"


def test_collector_refuses_an_empty_run_id() -> None:
    with pytest.raises(DeploymentPlanError, match="run_id"):
        collector_argv(
            endpoint=ENDPOINT, run_id="  ", prompt_file="p.txt", output_dir="out"
        )
