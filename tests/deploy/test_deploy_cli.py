"""The ``deploy`` CLI: exit codes, artifacts, and what it refuses to touch."""

from __future__ import annotations

import json
import socket
import sys
from pathlib import Path

import pytest
from _fakes import PINNED_IMAGE, TAG_ONLY_IMAGE, VALID_REVISION

from llmtracefx.deploy.cli import main

BASE_ARGS = [
    "plan",
    "--gpu-type",
    "H200",
    "--gpu-count",
    "4",
    "--max-runtime-seconds",
    "1800",
    "--max-deployment-seconds",
    "3600",
    "--usd-per-cpu-core-hour",
    "0.01",
    "--usd-per-gib-memory-hour",
    "0.005",
    "--storage-usd-per-gib-month",
    "0.02",
    "--storage-retention-days",
    "1",
    "--price-effective-date",
    "2026-08-01",
    "--price-source",
    "https://modal.com/pricing",
    "--model-revision",
    VALID_REVISION,
    "--image",
    PINNED_IMAGE,
    "--framework-version",
    "0.5.6",
    "--context-length",
    "131072",
    "--startup-timeout-seconds",
    "900",
    "--as-of",
    "2026-08-30",
]


def plan_args(*extra: str, max_usd: str = "40.00", rate: str = "1.00") -> list[str]:
    return [*BASE_ARGS, "--max-usd", max_usd, "--usd-per-gpu-hour", rate, *extra]


def run(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    return int(excinfo.value.code or 0)


def test_a_sound_plan_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(plan_args()) == 0
    assert "Deployment plan: APPROVED" in capsys.readouterr().out


def test_an_over_budget_plan_exits_non_zero(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run(plan_args(max_usd="1.00", rate="10.00")) == 1
    captured = capsys.readouterr()
    assert "Deployment plan: REFUSED" in captured.out
    assert "no paid step may be run" in captured.err


def test_refusal_marks_the_paid_steps_as_withheld(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run(plan_args("--format", "json", max_usd="1.00", rate="10.00"))
    payload = json.loads(capsys.readouterr().out)
    assert payload["approved"] is False
    assert "deploy" not in payload["executable_steps"]
    assert "stop" in payload["executable_steps"]


@pytest.mark.parametrize(
    "omit",
    [
        "--max-usd",
        "--gpu-type",
        "--gpu-count",
        "--max-runtime-seconds",
        "--max-deployment-seconds",
        "--usd-per-gpu-hour",
        "--usd-per-cpu-core-hour",
        "--usd-per-gib-memory-hour",
        "--storage-usd-per-gib-month",
        "--storage-retention-days",
        "--price-effective-date",
        "--price-source",
        "--model-revision",
        "--image",
        "--framework-version",
        "--context-length",
    ],
)
def test_every_spending_and_pinning_flag_is_required(
    omit: str, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = plan_args()
    index = argv.index(omit)
    del argv[index : index + 2]
    assert run(argv) == 2
    assert omit in capsys.readouterr().err


def test_the_plan_command_makes_no_network_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail loudly if planning ever grows a network dependency."""

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("planning must not open a socket")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    assert run(plan_args()) == 0


def test_the_plan_command_never_imports_the_modal_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "modal", raising=False)
    assert run(plan_args()) == 0
    assert "modal" not in sys.modules


def test_the_plan_command_does_not_read_modal_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A missing Modal config must be irrelevant to planning."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)
    assert run(plan_args()) == 0


def test_plan_json_can_be_written_atomically(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "nested" / "plan.json"
    assert run(plan_args("--output", str(destination))) == 0
    capsys.readouterr()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["kind"] == "llmtracefx.deploy.plan"
    assert payload["approved"] is True
    assert payload["gpu_allocated"] is False
    assert not list(destination.parent.glob(".*tmp*"))


def test_generated_commands_are_present_and_credential_free(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "plan.json"
    run(plan_args("--output", str(destination)))
    capsys.readouterr()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    names = [step["name"] for step in payload["steps"]]
    assert names == [
        "setup",
        "secret",
        "volume",
        "stage-weights",
        "verify-weights",
        "manifest",
        "deploy",
        "health",
        "readiness",
        "smoke",
        "collect",
        "stop",
        "delete-volume",
    ]
    document = json.dumps(payload)
    assert "--api-key " not in document
    assert "Bearer $GLM_SELFHOST_API_KEY" in document


def test_endpoint_url_is_substituted_once_it_is_known(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    destination = tmp_path / "plan.json"
    run(
        plan_args(
            "--endpoint-base-url",
            "https://ws--llmtracefx-glm53flash-serve.modal.run",
            "--output",
            str(destination),
        )
    )
    capsys.readouterr()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["endpoint"]["resolved"] is True
    collect = next(s for s in payload["steps"] if s["name"] == "collect")
    assert (
        "https://ws--llmtracefx-glm53flash-serve.modal.run/v1/chat/completions"
        in collect["argv"]
    )


def test_an_http_endpoint_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    assert run(plan_args("--endpoint-base-url", "http://insecure.example")) == 1
    assert "must use https" in capsys.readouterr().err


def test_a_tag_only_image_needs_the_flag(capsys: pytest.CaptureFixture[str]) -> None:
    argv = plan_args()
    argv[argv.index("--image") + 1] = TAG_ONLY_IMAGE
    assert run(argv) == 1
    assert "accept-mutable-image" in capsys.readouterr().err
    assert run([*argv, "--accept-mutable-image"]) == 0


def test_a_credential_flag_is_rejected_before_argparse_sees_its_value(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The existing credential guard covers the new subcommand too."""
    secret = "sk-live-should-never-be-echoed"
    with pytest.raises(SystemExit) as excinfo:
        main([*plan_args(), "--api-key", secret])
    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert secret not in captured.err + captured.out
    # Not even the flag name is repeated, so there is nothing derived
    # from the caller's argv in the diagnostic at all.
    assert "--api-key" not in captured.err.replace("--api-key-env", "")
    assert "--api-key-env" in captured.err


def test_budget_command_recommends_a_reserve(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run(["budget", "--credit-usd", "30", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recommended_session_budget_usd"] == pytest.approx(10.0)
    assert payload["reserve_usd"] == pytest.approx(20.0)


def test_recipe_command_prints_the_pinned_facts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run(["recipe", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["repo_id"] == "zai-org/GLM-5.3-Flash"
    assert payload["num_hidden_layers"] == 45


def test_a_deployment_window_shorter_than_a_container_is_refused(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = plan_args()
    argv[argv.index("--max-deployment-seconds") + 1] = "600"
    assert run(argv) == 1
    assert "shorter than" in capsys.readouterr().err


# A value that is not a plausible env var name, so any appearance of it in
# output means a credential value escaped rather than a name being echoed.
SENTINEL = "sk-live-SENTINEL-6f2b91c4e7a0d3"


def _no_sentinel(*streams: str) -> None:
    for stream in streams:
        assert SENTINEL not in stream


def test_success_path_never_emits_the_credential_value(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """The key is in the environment where the harness expects it to be."""
    monkeypatch.setenv("GLM_SELFHOST_API_KEY", SENTINEL)
    destination = tmp_path / "plan.json"
    assert run(plan_args("--output", str(destination))) == 0
    captured = capsys.readouterr()
    _no_sentinel(captured.out, captured.err, destination.read_text(encoding="utf-8"))
    assert "GLM_SELFHOST_API_KEY" in captured.out


def test_json_format_never_emits_the_credential_value(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GLM_SELFHOST_API_KEY", SENTINEL)
    assert run(plan_args("--format", "json")) == 0
    captured = capsys.readouterr()
    _no_sentinel(captured.out, captured.err)


def test_refusal_path_never_emits_the_credential_value(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("GLM_SELFHOST_API_KEY", SENTINEL)
    assert run(plan_args(max_usd="1.00", rate="10.00")) == 1
    captured = capsys.readouterr()
    _no_sentinel(captured.out, captured.err)


def test_validation_error_path_never_emits_the_credential_value(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An error message that genuinely quotes the offending value.

    The image parser reports the reference it refused, so pasting the key
    there puts it directly into the diagnostic. That is what makes this
    test load-bearing: remove the scrub and it fails. A flag whose
    validator never quotes its input would pass either way and prove
    nothing.
    """
    monkeypatch.setenv("GLM_SELFHOST_API_KEY", SENTINEL)
    argv = plan_args()
    argv[argv.index("--image") + 1] = SENTINEL
    assert run(argv) == 1
    captured = capsys.readouterr()
    _no_sentinel(captured.out, captured.err)
    assert "neither a tag nor a digest" in captured.err


def test_a_padded_variable_name_still_resolves_the_credential_to_scrub(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Padding must not silently disable the scrub.

    `require_env_var_name` strips before validating, so a padded name is
    accepted everywhere else and appears stripped in the plan. If the
    lookup that resolves the value to scrub for used the raw argument it
    would miss, return None, and turn the redaction into a no-op with
    nothing in the output to indicate it.
    """
    monkeypatch.setenv("GLM_SELFHOST_API_KEY", SENTINEL)
    assert (
        run(
            plan_args(
                "--api-key-env",
                "  GLM_SELFHOST_API_KEY  ",
                "--price-source",
                SENTINEL,
            )
        )
        == 0
    )
    captured = capsys.readouterr()
    _no_sentinel(captured.out, captured.err)
    assert "GLM_SELFHOST_API_KEY" in captured.out


def test_an_absurd_container_count_is_refused_not_raised(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A typo must produce a refusal, not a traceback.

    Multiplying a very large integer container count reaches an
    int-to-float conversion that raises OverflowError, which would escape
    as a traceback on the one output path the design routes through the
    scrub.
    """
    assert run(plan_args("--max-containers", str(10**308))) == 1
    assert "max_containers" in capsys.readouterr().err


def test_a_malformed_endpoint_host_is_refused_not_raised(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """urlsplit raises on a malformed bracketed host; that must be caught."""
    assert run(plan_args("--endpoint-base-url", "https://[bad")) == 1
    captured = capsys.readouterr()
    assert "endpoint base URL" in captured.err
    assert "Traceback" not in captured.err


def test_a_credential_pasted_where_a_secret_name_belongs_is_scrubbed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A credential can be a well-formed Modal Secret name.

    Shape validation cannot save us here: this sentinel is alphanumeric
    with dashes and inside the length limit, so it passes as a name and
    the plan is built. What keeps it out of the output is the scrub,
    which compares the rendered document against the resolved value of
    the variable the caller named. The plan is therefore approved and the
    value still never appears.
    """
    monkeypatch.setenv("GLM_SELFHOST_API_KEY", SENTINEL)
    assert run(plan_args("--modal-secret-name", SENTINEL)) == 0
    captured = capsys.readouterr()
    _no_sentinel(captured.out, captured.err)


def test_the_plan_never_contains_a_bearer_value_only_a_variable_reference(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    monkeypatch.setenv("GLM_SELFHOST_API_KEY", SENTINEL)
    destination = tmp_path / "plan.json"
    run(plan_args("--output", str(destination)))
    capsys.readouterr()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    for step in payload["steps"]:
        for token in step["argv"]:
            assert SENTINEL not in token
            if "Authorization" in token:
                assert token.endswith("$GLM_SELFHOST_API_KEY")


def test_budget_command_refuses_an_unscalable_credit(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert run(["budget", "--credit-usd", "1e307"]) == 1
    captured = capsys.readouterr()
    assert "too large" in captured.err
    assert "Traceback" not in captured.err
