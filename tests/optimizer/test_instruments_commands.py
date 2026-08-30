"""Tests for shell-free xctrace argv construction and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmtracefx.optimizer.instruments.commands import (
    REDACTED,
    AttachTarget,
    EnvironmentAssignment,
    ExportPlan,
    InstrumentsCommandError,
    LaunchTarget,
    RecordPlan,
    build_list_templates_argv,
    build_version_argv,
    duration_to_seconds,
    redact_argv,
    table_xpath,
    validate_schema_name,
    validate_time_limit,
    validate_window,
)


def make_plan(**overrides) -> RecordPlan:
    values = {
        "xctrace_path": "/usr/bin/xctrace",
        "template": "Metal System Trace",
        "output_trace": Path("/tmp/out/run.trace"),
        "target": LaunchTarget(argv=("/bin/infer", "--tokens", "128")),
        "time_limit": "30s",
    }
    values.update(overrides)
    return RecordPlan(**values)


# --- Durations --------------------------------------------------------


@pytest.mark.parametrize("value", ["500ms", "1s", "30s", "2m", "1h"])
def test_valid_time_limits(value):
    assert validate_time_limit(value) == value


@pytest.mark.parametrize(
    "value", ["30", "30 s", "s30", "-5s", "1.5s", "30sec", "", "30S", "1d"]
)
def test_invalid_time_limits_are_refused(value):
    with pytest.raises(InstrumentsCommandError, match="invalid --time-limit"):
        validate_time_limit(value)


def test_window_rejects_hours_because_xctrace_does():
    assert validate_window("5s") == "5s"
    with pytest.raises(InstrumentsCommandError, match="not for --window"):
        validate_window("1h")


@pytest.mark.parametrize(
    "value,seconds", [("500ms", 0.5), ("30s", 30.0), ("2m", 120.0), ("1h", 3600.0)]
)
def test_duration_to_seconds(value, seconds):
    assert duration_to_seconds(value) == seconds


# --- XPath construction and injection ---------------------------------


def test_table_xpath_matches_the_documented_form():
    assert table_xpath("metal-gpu-intervals") == (
        '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]'
    )


@pytest.mark.parametrize(
    "name",
    [
        'a"] | //*[@x="',
        "schema name",
        "",
        "-leading-dash",
        "1starts-with-digit",
        "has/slash",
        "has'quote",
        "has]bracket",
    ],
)
def test_schema_names_that_could_break_out_of_an_xpath_are_refused(name):
    with pytest.raises(InstrumentsCommandError, match="invalid trace table schema"):
        validate_schema_name(name)


@pytest.mark.parametrize("run_number", [0, -1, True])
def test_invalid_run_numbers_are_refused(run_number):
    with pytest.raises(InstrumentsCommandError, match="invalid trace run number"):
        table_xpath("time-profile", run_number=run_number)


# --- Record argv ------------------------------------------------------


def test_record_argv_shape_and_launch_is_last():
    argv = make_plan().to_argv()
    assert argv[:2] == ("/usr/bin/xctrace", "record")
    assert "--template" in argv and argv[argv.index("--template") + 1] == (
        "Metal System Trace"
    )
    assert argv[argv.index("--time-limit") + 1] == "30s"
    assert "--no-prompt" in argv
    # --launch and everything after it must be the tail of the argv.
    assert argv[-5:] == ("--launch", "--", "/bin/infer", "--tokens", "128")


def test_record_argv_never_targets_another_device():
    """Omitting --device is what makes the host the target."""
    argv = make_plan().to_argv()
    assert "--device" not in argv
    assert "--device-name" not in argv


def test_record_argv_never_appends_to_an_existing_trace():
    assert "--append-run" not in make_plan().to_argv()


def test_record_argv_never_redirects_target_streams():
    """Prompt and completion text must not flow into captured logs."""
    argv = make_plan().to_argv()
    assert "--target-stdout" not in argv
    assert "--target-stdin" not in argv


def test_record_argv_never_records_all_processes():
    assert "--all-processes" not in make_plan().to_argv()


def test_record_argv_is_deterministic():
    assert make_plan().to_argv() == make_plan().to_argv()


def test_attach_target_uses_pid():
    argv = make_plan(target=AttachTarget(pid=4242)).to_argv()
    assert argv[argv.index("--attach") + 1] == "4242"
    assert "--launch" not in argv


@pytest.mark.parametrize("pid", [0, -1, True, "1234"])
def test_attach_by_anything_other_than_a_positive_int_is_refused(pid):
    with pytest.raises(InstrumentsCommandError, match="attach pid"):
        AttachTarget(pid=pid)


def test_launch_target_requires_a_non_empty_command():
    with pytest.raises(InstrumentsCommandError, match="non-empty"):
        LaunchTarget(argv=())
    with pytest.raises(InstrumentsCommandError, match="non-empty"):
        LaunchTarget(argv=("",))


def test_output_must_be_a_trace_bundle():
    with pytest.raises(InstrumentsCommandError, match="must end in '.trace'"):
        make_plan(output_trace=Path("/tmp/out.txt"))


def test_window_flag_is_emitted_when_requested():
    argv = make_plan(window="5s").to_argv()
    assert argv[argv.index("--window") + 1] == "5s"


def test_run_name_flag_is_emitted_when_requested():
    argv = make_plan(run_name="baseline").to_argv()
    assert argv[argv.index("--run-name") + 1] == "baseline"


# --- Timeout ----------------------------------------------------------


def test_host_timeout_exceeds_the_recording_window():
    """xctrace keeps writing after the window closes.

    A host deadline equal to --time-limit would kill every recording
    during finalization.
    """
    plan = make_plan(time_limit="30s", grace_seconds=90.0)
    assert plan.timeout_seconds == 120.0
    assert plan.timeout_seconds > duration_to_seconds(plan.time_limit)


def test_grace_must_be_positive():
    with pytest.raises(InstrumentsCommandError, match="grace_seconds"):
        make_plan(grace_seconds=0)


# --- Credential redaction ---------------------------------------------


def test_env_values_that_look_like_credentials_are_redacted():
    plan = make_plan(
        environment=(
            EnvironmentAssignment(name="HF_TOKEN", value="hf_secret_value"),
            EnvironmentAssignment(name="BATCH_SIZE", value="8"),
        )
    )
    real = plan.to_argv()
    stored = plan.to_redacted_argv()

    assert "HF_TOKEN=hf_secret_value" in real
    assert "hf_secret_value" not in " ".join(stored)
    assert f"HF_TOKEN={REDACTED}" in stored
    # A non-credential variable stays readable, so reproduction works.
    assert "BATCH_SIZE=8" in stored


@pytest.mark.parametrize(
    "name",
    [
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "MY_PASSWORD",
        "db_credential",
        "Service_Token",
        "PRIVATE_KEY",
    ],
)
def test_credential_marker_matching_is_case_insensitive(name):
    plan = make_plan(environment=(EnvironmentAssignment(name=name, value="s3cr3t"),))
    assert "s3cr3t" not in " ".join(plan.to_redacted_argv())


def test_redacted_argv_is_deterministic():
    plan = make_plan(environment=(EnvironmentAssignment(name="HF_TOKEN", value="a"),))
    assert plan.to_redacted_argv() == plan.to_redacted_argv()


def test_redact_argv_handles_loose_argv():
    assert redact_argv(("run", "API_KEY=abc", "SIZE=4")) == (
        "run",
        f"API_KEY={REDACTED}",
        "SIZE=4",
    )


def test_environment_name_must_not_contain_equals():
    with pytest.raises(InstrumentsCommandError, match="must not contain"):
        EnvironmentAssignment(name="A=B", value="c")


def test_env_is_refused_when_attaching():
    """xctrace documents --env as launch only."""
    with pytest.raises(InstrumentsCommandError, match="only supported when launching"):
        make_plan(
            target=AttachTarget(pid=1),
            environment=(EnvironmentAssignment(name="A", value="b"),),
        )


# --- Export argv ------------------------------------------------------


def test_export_toc_argv():
    plan = ExportPlan(
        xctrace_path="/usr/bin/xctrace",
        input_trace=Path("/tmp/run.trace"),
        output_path=Path("/tmp/toc.xml"),
        toc=True,
    )
    argv = plan.to_argv()
    assert argv[:2] == ("/usr/bin/xctrace", "export")
    assert "--toc" in argv
    assert "--xpath" not in argv


def test_export_table_argv_uses_xpath():
    plan = ExportPlan(
        xctrace_path="/usr/bin/xctrace",
        input_trace=Path("/tmp/run.trace"),
        output_path=Path("/tmp/t.xml"),
        schema_name="metal-gpu-intervals",
        run_number=2,
    )
    argv = plan.to_argv()
    assert "--toc" not in argv
    assert argv[argv.index("--xpath") + 1] == (
        '/trace-toc/run[@number="2"]/data/table[@schema="metal-gpu-intervals"]'
    )


def test_export_refuses_to_combine_toc_and_xpath():
    with pytest.raises(InstrumentsCommandError, match="cannot combine"):
        ExportPlan(
            xctrace_path="/usr/bin/xctrace",
            input_trace=Path("/tmp/run.trace"),
            output_path=Path("/tmp/t.xml"),
            toc=True,
            schema_name="time-profile",
        )


def test_export_requires_a_mode():
    with pytest.raises(InstrumentsCommandError, match="needs either"):
        ExportPlan(
            xctrace_path="/usr/bin/xctrace",
            input_trace=Path("/tmp/run.trace"),
            output_path=Path("/tmp/t.xml"),
        )


def test_export_rejects_injected_schema_name():
    with pytest.raises(InstrumentsCommandError, match="invalid trace table schema"):
        ExportPlan(
            xctrace_path="/usr/bin/xctrace",
            input_trace=Path("/tmp/run.trace"),
            output_path=Path("/tmp/t.xml"),
            schema_name='x"] | //*[@a="',
        )


def test_simple_argv_builders():
    assert build_list_templates_argv("/usr/bin/xctrace") == (
        "/usr/bin/xctrace",
        "list",
        "templates",
    )
    assert build_version_argv("/usr/bin/xctrace") == ("/usr/bin/xctrace", "version")
    with pytest.raises(InstrumentsCommandError):
        build_version_argv("")


# --- Regressions found in independent review --------------------------


def test_secrets_in_the_profiled_command_are_redacted():
    """The profiled command is user supplied and can carry a secret."""
    plan = make_plan(
        target=LaunchTarget(
            argv=(
                "/usr/bin/env",
                "HF_TOKEN=hf_supersecret",
                "python",
                "bench.py",
                "OPENAI_API_KEY=sk-live-123",
            )
        )
    )
    stored = " ".join(plan.to_redacted_argv())
    assert "hf_supersecret" not in stored
    assert "sk-live-123" not in stored
    assert f"HF_TOKEN={REDACTED}" in plan.to_redacted_argv()
    # The real invocation still receives the true values.
    assert "HF_TOKEN=hf_supersecret" in plan.to_argv()
    # Ordinary arguments survive so the run stays reproducible.
    assert "bench.py" in plan.to_redacted_argv()


def test_redaction_does_not_disturb_launch_being_last():
    plan = make_plan(
        target=LaunchTarget(argv=("/bin/infer", "TOKEN=abc", "--tokens", "8"))
    )
    argv = plan.to_redacted_argv()
    separator = argv.index("--launch")
    assert argv[separator + 1] == "--"
    assert argv[separator + 2 :] == (
        "/bin/infer",
        f"TOKEN={REDACTED}",
        "--tokens",
        "8",
    )


def test_output_trace_tilde_is_expanded_once(tmp_path, monkeypatch):
    """The checked path and the recorded path must be identical.

    An unexpanded `~` would make the collision check and mkdir target
    $HOME while xctrace created a literal './~' directory.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    plan = make_plan(output_trace=Path("~/traces/run.trace"))
    argv = plan.to_argv()
    recorded = argv[argv.index("--output") + 1]
    assert "~" not in recorded
    assert recorded == str(plan.output_trace)
    assert Path(recorded).is_absolute()


# --- Split and attached credential arguments --------------------------
#
# Redaction used to handle only NAME=value, so `--api-key sk-live` kept
# the secret in every artifact this project writes.

SENTINEL = "sk-live-DO-NOT-LEAK"


@pytest.mark.parametrize(
    "argv",
    [
        ("bench", "--api-key", SENTINEL),
        ("bench", "--api_key", SENTINEL),
        ("bench", "--API-KEY", SENTINEL),
        ("bench", f"--api-key={SENTINEL}"),
        ("bench", f"API_KEY={SENTINEL}"),
        ("bench", "--hf-token", SENTINEL),
        ("bench", "--auth-token", SENTINEL),
        ("bench", "--password", SENTINEL),
        ("bench", "--secret", SENTINEL),
        ("bench", "--private-key", SENTINEL),
        ("bench", "--credential", SENTINEL),
        ("bench", "--bearer", SENTINEL),
        # A value that itself looks like an option must still go.
        ("bench", "--api-key", f"-{SENTINEL}"),
        ("bench", "--token", f"--{SENTINEL}"),
    ],
)
def test_every_credential_shape_is_redacted(argv):
    plan = make_plan(target=LaunchTarget(argv=argv))
    assert SENTINEL not in " ".join(plan.to_redacted_argv())
    # The real invocation is untouched, or the program would break.
    assert SENTINEL in " ".join(plan.to_argv())


@pytest.mark.parametrize(
    "argv",
    [
        ("bench", "--tokens", "128"),
        ("bench", "--max-tokens", "512"),
        ("bench", "--num-tokens", "64"),
        ("bench", "--token-count", "9"),
        ("bench", "--max_new_tokens", "32"),
        ("bench", "--input-tokens", "10"),
        ("bench", "--sort-key", "name"),
        ("bench", "--cache-key", "abc"),
        ("bench", "--temperature", "0.7"),
        ("bench", "--keyword", "value"),
    ],
)
def test_legitimate_options_are_not_corrupted(argv):
    """Redaction must not eat ordinary inference parameters.

    An LLM command is full of `--max-tokens` and `--num-tokens`. A naive
    substring match on "token" would redact all of them and destroy the
    reproducibility the recorded argv exists to provide.
    """
    plan = make_plan(target=LaunchTarget(argv=argv))
    redacted = plan.to_redacted_argv()
    assert REDACTED not in redacted
    assert redacted[-len(argv) :] == argv


def test_a_credential_flag_at_the_end_does_not_crash():
    plan = make_plan(target=LaunchTarget(argv=("bench", "--api-key")))
    assert plan.to_redacted_argv()[-1] == "--api-key"


def test_only_the_value_immediately_after_the_flag_is_redacted():
    plan = make_plan(
        target=LaunchTarget(argv=("bench", "--api-key", SENTINEL, "--tokens", "128"))
    )
    redacted = plan.to_redacted_argv()
    assert redacted[-4:] == ("--api-key", REDACTED, "--tokens", "128")


def test_single_dash_attached_values_are_a_documented_limit():
    """`-ksk-live` cannot be told from a cluster of short flags.

    Guessing would corrupt legitimate arguments, so the boundary is
    documented rather than papered over. This pins it so a future change
    is a deliberate one.
    """
    plan = make_plan(target=LaunchTarget(argv=("bench", f"-k{SENTINEL}")))
    assert SENTINEL in " ".join(plan.to_redacted_argv())


@pytest.mark.parametrize(
    "flag",
    [
        "--Authorization",
        "--authorization",
        "--auth",
        "--client-secret",
        "--refresh-token",
        "--x-api-key",
        "--aws-secret-access-key",
        "--apiKey",
        "-api-key",
    ],
)
def test_further_credential_spellings_are_redacted(flag):
    plan = make_plan(target=LaunchTarget(argv=("bench", flag, SENTINEL)))
    assert SENTINEL not in " ".join(plan.to_redacted_argv())


@pytest.mark.parametrize(
    "flag",
    [
        "--auth-mode",
        "--auth-type",
        "--private-key-path",
        "--api-key-file",
        "--token-file",
        "--credential-dir",
        "--secret-name",
        "--tokenizer-path",
    ],
)
def test_location_names_are_not_secrets(flag):
    """A name that points at where a secret lives is not the secret.

    Redacting `--private-key-path /etc/k.pem` hides a filename, gains no
    privacy, and loses the reproducibility the recorded argv exists for.
    """
    plan = make_plan(target=LaunchTarget(argv=("bench", flag, "/some/value")))
    assert REDACTED not in plan.to_redacted_argv()
    assert plan.to_redacted_argv()[-1] == "/some/value"


@pytest.mark.parametrize(
    "flag", ["--jwt", "--ssh-key", "--deploy-key", "--session-key", "--signing-key"]
)
def test_unrecognized_key_style_options_default_to_secret(flag):
    """An unfamiliar `*-key` option is treated as sensitive.

    Redacting a lookup key costs a little reproducibility; persisting a
    private one is unrecoverable, so the default leans the safe way.
    """
    plan = make_plan(target=LaunchTarget(argv=("bench", flag, SENTINEL)))
    assert SENTINEL not in " ".join(plan.to_redacted_argv())


@pytest.mark.parametrize(
    "flag", ["--sort-key", "--cache-key", "--partition-key", "--primary-key"]
)
def test_lookup_keys_are_still_readable(flag):
    plan = make_plan(target=LaunchTarget(argv=("bench", flag, "user_id")))
    assert plan.to_redacted_argv()[-1] == "user_id"
