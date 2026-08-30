"""Tests for the ``collect-api`` CLI surface.

These tests never perform a network request. The dry-run tests replace the
real transport with one that fails if it is ever used, and the collection
tests inject a fake transport that replays recorded byte chunks.
"""

from __future__ import annotations

import contextlib
import io
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer import cli
from llmtracefx.optimizer.collectors.openai_api import (
    HTTPRequest,
    redact_text_for_dry_run,
)

ENDPOINT = "https://api.z.ai/api/paas/v4/chat/completions"
API_KEY = "cli-test-key-not-a-real-credential"


class ExplodingTransport:
    def open_stream(self, request: HTTPRequest) -> Any:
        raise AssertionError("dry-run must not open a stream")


class FakeResponse:
    def __init__(self, chunks: list[bytes], status_code: int = 200) -> None:
        self._chunks = chunks
        self._status_code = status_code

    @property
    def status_code(self) -> int:
        return self._status_code

    @property
    def headers(self) -> Mapping[str, str]:
        return {}

    def iter_bytes(self) -> Iterator[bytes]:
        yield from self._chunks

    def close(self) -> None:
        return None


class FakeTransport:
    requests: list[HTTPRequest] = []

    def __init__(self, chunks: list[bytes], status_code: int = 200) -> None:
        self._chunks = chunks
        self._status_code = status_code

    def open_stream(self, request: HTTPRequest) -> FakeResponse:
        FakeTransport.requests.append(request)
        return FakeResponse(self._chunks, self._status_code)


def sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


SUCCESS_STREAM = [
    sse(
        {
            "id": "chatcmpl-1",
            "model": "glm-5.3",
            "choices": [{"index": 0, "delta": {"content": "hi"}}],
        }
    ),
    sse(
        {
            "id": "chatcmpl-1",
            "choices": [
                {"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        }
    ),
    b"data: [DONE]\n\n",
]


@pytest.fixture(autouse=True)
def _reset_recorded_requests() -> Iterator[None]:
    FakeTransport.requests = []
    yield
    FakeTransport.requests = []


def base_argv(tmp_path: Path, **extra: str) -> list[str]:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Explain a stack in one sentence.", encoding="utf-8")
    argv = [
        "collect-api",
        "--run-id",
        "cli-run",
        "--endpoint",
        ENDPOINT,
        "--model-id",
        "glm-5.3",
        "--prompt-file",
        str(prompt),
        "--output-dir",
        str(tmp_path / "artifacts"),
    ]
    for flag, value in extra.items():
        argv.extend((f"--{flag.replace('_', '-')}", value))
    return argv


def invoke(argv: list[str]) -> int:
    parser = cli.build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


def invoke_captured(argv: list[str], *, known: bool = False) -> tuple[int, str, str]:
    """Run a handler obtained directly from either public parser API."""
    stdout, stderr = io.StringIO(), io.StringIO()
    code = 0
    try:
        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            parser = cli.build_parser()
            if known:
                args, remaining = parser.parse_known_args(argv)
                assert remaining == []
            else:
                args = parser.parse_args(argv)
            code = int(args.func(args))
    except SystemExit as exit_error:
        code = int(exit_error.code or 0)
    return code, stdout.getvalue(), stderr.getvalue()


# --- Dry run -----------------------------------------------------------------


def test_dry_run_performs_no_network_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    exit_code = invoke(base_argv(tmp_path) + ["--dry-run"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert payload["network_request_performed"] is False
    # The variable is unset here, so nothing proved this string is a
    # variable name rather than a credential pasted into the name slot.
    assert payload["credential_env_var"] == "[REDACTED]"
    assert payload["credential_env_var_present"] is False
    assert payload["plan"]["model_id"] == "glm-5.3"
    assert payload["plan"]["endpoint_origin"] == "https://api.z.ai"

    artifacts = tmp_path / "artifacts"
    assert sorted(path.name for path in artifacts.iterdir()) == ["request_plan.json"]
    assert (
        json.loads((artifacts / "request_plan.json").read_text(encoding="utf-8"))
        == payload
    )


def test_dry_run_reports_presence_without_revealing_the_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)
    monkeypatch.setenv("ZAI_API_KEY", API_KEY)

    exit_code = invoke(base_argv(tmp_path) + ["--dry-run"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert json.loads(output)["credential_env_var_present"] is True
    assert API_KEY not in output
    assert API_KEY not in (tmp_path / "artifacts" / "request_plan.json").read_text(
        encoding="utf-8"
    )


def test_dry_run_command_reconstruction_names_the_env_var_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)
    monkeypatch.setenv("ZAI_API_KEY", API_KEY)
    argv = base_argv(tmp_path, reasoning_effort="high", clear_thinking="false") + [
        "--dry-run"
    ]

    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)

    assert excinfo.value.code == 0
    command = json.loads(capsys.readouterr().out)["plan"]["command"]
    assert command[0] == "llmtracefx-optimizer"
    assert command[1] == "collect-api"
    assert "--api-key-env" in command
    assert command[command.index("--api-key-env") + 1] == "ZAI_API_KEY"
    assert API_KEY not in command
    assert not any(part.startswith("--api-key=") for part in command)


def test_reconstructed_command_round_trips_through_the_parser(tmp_path: Path) -> None:
    argv = base_argv(
        tmp_path,
        provider="z.ai",
        api_key_env="ZAI_API_KEY",
        max_output_tokens="128",
        temperature="0.2",
        top_p="0.9",
        seed="11",
        request_timeout="45.0",
        reasoning_effort="max",
        thinking="enabled",
        clear_thinking="true",
        model_revision="unavailable-from-provider",
    )
    parser = cli.build_parser()
    args = parser.parse_args(argv + ["--dry-run"])

    reconstructed = cli._collect_api_argv(args)

    assert reconstructed[0] == "llmtracefx-optimizer"
    reparsed = parser.parse_args(list(reconstructed[1:]))
    assert reparsed.model_id == args.model_id
    assert reparsed.reasoning_effort == "max"
    assert reparsed.thinking == "enabled"
    assert reparsed.clear_thinking == "true"
    assert reparsed.max_output_tokens == 128
    assert reparsed.dry_run is True


def test_dry_run_rejects_an_invalid_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)
    argv = base_argv(tmp_path) + ["--dry-run"]
    argv[argv.index("--endpoint") + 1] = "http://api.z.ai/v4/chat/completions"

    exit_code = invoke(argv)

    assert exit_code == 1
    assert "must use https" in capsys.readouterr().err


def test_missing_prompt_file_is_reported_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = base_argv(tmp_path)
    argv[argv.index("--prompt-file") + 1] = str(tmp_path / "absent.txt")

    exit_code = invoke(argv + ["--dry-run"])

    assert exit_code == 1
    assert "Failed to configure API collection" in capsys.readouterr().err


# --- Argument surface --------------------------------------------------------


def test_the_cli_never_accepts_a_key_argument(tmp_path: Path) -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(base_argv(tmp_path) + ["--api-key", API_KEY])


@pytest.mark.parametrize("value", ["medium", "MAX", "none"])
def test_reasoning_effort_choices_are_restricted(tmp_path: Path, value: str) -> None:
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(base_argv(tmp_path, reasoning_effort=value))


@pytest.mark.parametrize("value", ["low", "high", "max"])
def test_documented_reasoning_effort_levels_are_accepted(
    tmp_path: Path, value: str
) -> None:
    parser = cli.build_parser()

    args = parser.parse_args(base_argv(tmp_path, reasoning_effort=value))

    assert args.reasoning_effort == value


def test_defaults_match_the_documented_zai_usage(tmp_path: Path) -> None:
    parser = cli.build_parser()

    args = parser.parse_args(base_argv(tmp_path))

    assert args.api_key_env == "ZAI_API_KEY"
    assert args.provider == "z.ai"
    assert args.request_timeout == 120.0
    assert args.reasoning_effort is None
    assert args.thinking is None
    assert args.clear_thinking is None
    assert args.dry_run is False


# --- Collection --------------------------------------------------------------


def test_successful_collection_writes_artifacts_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", API_KEY)
    monkeypatch.setattr(
        cli, "UrllibStreamingTransport", lambda: FakeTransport(SUCCESS_STREAM)
    )

    exit_code = invoke(
        base_argv(
            tmp_path,
            model_id="glm-5.3-flash",
            reasoning_effort="low",
            clear_thinking="true",
        )
    )

    assert exit_code == 0
    assert "record.json" in capsys.readouterr().out
    artifacts = tmp_path / "artifacts"
    assert sorted(path.name for path in artifacts.iterdir()) == [
        "api_evidence.json",
        "artifacts.json",
        "environment.json",
        "record.json",
        "response.txt",
    ]
    body = json.loads(FakeTransport.requests[0].body.decode("utf-8"))
    assert body["model"] == "glm-5.3-flash"
    assert body["reasoning_effort"] == "low"
    assert body["thinking"] == {"clear_thinking": True}
    for artifact in artifacts.iterdir():
        assert API_KEY not in artifact.read_text(encoding="utf-8")


def test_failed_collection_exits_non_zero_and_still_writes_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", API_KEY)
    error_body = json.dumps({"code": 1113, "message": "insufficient balance"}).encode()
    monkeypatch.setattr(
        cli, "UrllibStreamingTransport", lambda: FakeTransport([error_body], 402)
    )

    exit_code = invoke(base_argv(tmp_path))

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "http_status" in captured.err
    assert "insufficient balance" in captured.err
    evidence = json.loads(
        (tmp_path / "artifacts" / "api_evidence.json").read_text(encoding="utf-8")
    )
    assert evidence["success"] is False
    assert evidence["failure"]["status_code"] == 402


def test_missing_credential_exits_non_zero_without_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.setattr(
        cli, "UrllibStreamingTransport", lambda: FakeTransport(SUCCESS_STREAM)
    )

    exit_code = invoke(base_argv(tmp_path))

    assert exit_code == 1
    error = capsys.readouterr().err
    assert "named by --api-key-env is not set" in error
    assert "ZAI_API_KEY" not in error
    assert not (tmp_path / "artifacts").exists()


def test_custom_credential_env_var_is_honoured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.setenv("OTHER_PROVIDER_KEY", API_KEY)
    monkeypatch.setattr(
        cli, "UrllibStreamingTransport", lambda: FakeTransport(SUCCESS_STREAM)
    )

    exit_code = invoke(
        base_argv(tmp_path, provider="other-provider", api_key_env="OTHER_PROVIDER_KEY")
    )

    assert exit_code == 0
    evidence = json.loads(
        (tmp_path / "artifacts" / "api_evidence.json").read_text(encoding="utf-8")
    )
    assert evidence["plan"]["credential_env_var"] == "OTHER_PROVIDER_KEY"
    assert evidence["plan"]["provider"] == "other-provider"


# --- Dry-run credential containment ------------------------------------------


def test_dry_run_refuses_a_credential_pasted_into_the_endpoint_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dry run still handles a configured key.

    If the operator pasted it into the endpoint, it is inside the plan and
    the reconstructed command that are about to be printed and written.
    """
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)
    monkeypatch.setenv("ZAI_API_KEY", API_KEY)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("hello", encoding="utf-8")
    argv = [
        "collect-api",
        "--run-id",
        "cli-run",
        "--endpoint",
        f"https://example.test/v1/chat?deployment={API_KEY}",
        "--model-id",
        "glm-5.3",
        "--prompt-file",
        str(prompt),
        "--output-dir",
        str(tmp_path / "artifacts"),
        "--dry-run",
    ]

    exit_code = invoke(argv)

    # A real run refuses this configuration, so the pre-flight check must
    # refuse it too rather than reporting a plan that could never be used.
    assert exit_code == 1
    captured = capsys.readouterr()
    assert API_KEY not in captured.out
    assert API_KEY not in captured.err
    assert "appears in endpoint" in captured.err
    assert not (tmp_path / "artifacts" / "request_plan.json").exists()


def test_dry_run_refuses_a_credential_pasted_into_the_endpoint_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)
    monkeypatch.setenv("ZAI_API_KEY", API_KEY)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("hello", encoding="utf-8")
    argv = [
        "collect-api",
        "--run-id",
        "cli-run",
        "--endpoint",
        f"https://example.test/v1/{API_KEY}/chat",
        "--model-id",
        "glm-5.3",
        "--prompt-file",
        str(prompt),
        "--output-dir",
        str(tmp_path / "artifacts"),
        "--dry-run",
    ]

    exit_code = invoke(argv)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert API_KEY not in captured.out
    assert API_KEY not in captured.err
    assert "appears in endpoint" in captured.err
    assert not (tmp_path / "artifacts" / "request_plan.json").exists()


def test_an_invalid_endpoint_is_never_echoed_back_verbatim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Config failure is where a pasted key is most likely to surface."""
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)
    monkeypatch.setenv("ZAI_API_KEY", API_KEY)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("hello", encoding="utf-8")
    argv = [
        "collect-api",
        "--run-id",
        "cli-run",
        "--endpoint",
        f"ftp://example.test/v1/{API_KEY}/chat?token={API_KEY}",
        "--model-id",
        "glm-5.3",
        "--prompt-file",
        str(prompt),
        "--output-dir",
        str(tmp_path / "artifacts"),
        "--dry-run",
    ]

    exit_code = invoke(argv)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert API_KEY not in captured.err
    assert API_KEY not in captured.out
    assert "http or https" in captured.err
    assert not (tmp_path / "artifacts" / "request_plan.json").exists()


def test_a_plain_http_endpoint_error_does_not_echo_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)
    monkeypatch.setenv("ZAI_API_KEY", API_KEY)
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("hello", encoding="utf-8")
    argv = [
        "collect-api",
        "--run-id",
        "cli-run",
        "--endpoint",
        "http://example.test/v1/super-secret-path/chat",
        "--model-id",
        "glm-5.3",
        "--prompt-file",
        str(prompt),
        "--output-dir",
        str(tmp_path / "artifacts"),
        "--dry-run",
    ]

    exit_code = invoke(argv)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "super-secret-path" not in captured.err
    assert "example.test" not in captured.err
    assert "must use https for non-local hosts" in captured.err


def test_dry_run_still_performs_no_network_request_when_a_key_is_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The pre-flight check must not turn into a request."""
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)
    monkeypatch.setenv("ZAI_API_KEY", API_KEY)

    exit_code = invoke(base_argv(tmp_path) + ["--dry-run"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert API_KEY not in captured.out
    assert API_KEY not in captured.err
    payload = json.loads(captured.out)
    assert payload["network_request_performed"] is False
    assert payload["credential_env_var_present"] is True
    written = (tmp_path / "artifacts" / "request_plan.json").read_text(encoding="utf-8")
    assert API_KEY not in written


def test_the_dry_run_scrubber_removes_a_credential_and_bearer_shapes() -> None:
    """Defence in depth behind the pre-flight refusal.

    The refusal covers the fields the collector knows about. This scrubber
    is the last gate on the rendered document, so it is tested directly.
    """
    text = f'{{"a": "{API_KEY}", "b": "Bearer {API_KEY}", "c": "Bearer other"}}'

    cleaned = redact_text_for_dry_run(text, API_KEY)

    assert API_KEY not in cleaned
    assert "Bearer other" not in cleaned
    assert cleaned.count("[REDACTED]") == 3


def test_the_dry_run_scrubber_is_safe_without_a_configured_credential() -> None:
    cleaned = redact_text_for_dry_run('{"a": "Bearer sk-abc123"}', None)

    assert "sk-abc123" not in cleaned
    assert "[REDACTED]" in cleaned


# --- Malformed endpoints through the CLI --------------------------------------


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.z.ai:99999/v1/chat/completions",
        "https://api.z.ai:notaport/v1/chat/completions",
        "https://[unclosed/v1/chat/completions",
    ],
    ids=["port-out-of-range", "port-not-an-integer", "malformed-ipv6"],
)
@pytest.mark.parametrize("dry_run", [False, True], ids=["real", "dry-run"])
def test_a_malformed_endpoint_fails_cleanly_without_a_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    endpoint: str,
    dry_run: bool,
) -> None:
    """A parse failure must be a diagnostic, not an escaping ValueError."""
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)
    monkeypatch.setenv("ZAI_API_KEY", API_KEY)
    argv = base_argv(tmp_path)
    argv[argv.index("--endpoint") + 1] = endpoint
    if dry_run:
        argv.append("--dry-run")

    exit_code = invoke(argv)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "ValueError" not in captured.err
    assert "notaport" not in captured.err
    assert "unclosed" not in captured.err
    assert "/v1/chat/completions" not in captured.err
    assert not (tmp_path / "artifacts" / "request_plan.json").exists()


@pytest.mark.parametrize("dry_run", [False, True], ids=["real", "dry-run"])
def test_a_malformed_endpoint_holding_a_secret_leaks_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dry_run: bool,
) -> None:
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)
    monkeypatch.setenv("ZAI_API_KEY", API_KEY)
    argv = base_argv(tmp_path)
    argv[argv.index("--endpoint") + 1] = f"https://api.z.ai:99999/v1/{API_KEY}/chat"
    if dry_run:
        argv.append("--dry-run")

    exit_code = invoke(argv)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert API_KEY not in captured.err
    assert API_KEY not in captured.out


@pytest.mark.parametrize("dry_run", [False, True], ids=["real", "dry-run"])
def test_a_percent_encoded_credential_is_refused_through_the_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    dry_run: bool,
) -> None:
    """The encoded form must be refused on both paths, not just the real run."""
    credential = "sk-slash/credential"
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)
    monkeypatch.setenv("ZAI_API_KEY", credential)
    argv = base_argv(tmp_path)
    argv[argv.index("--endpoint") + 1] = (
        "https://api.z.ai/v1/sk-slash%2Fcredential/completions"
    )
    if dry_run:
        argv.append("--dry-run")

    exit_code = invoke(argv)

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "appears in endpoint" in captured.err
    assert credential not in captured.err
    assert credential not in captured.out
    assert not (tmp_path / "artifacts" / "request_plan.json").exists()


# --- Sixth review pass: parse diagnostics never repeat a value ---------------

_SENTINEL = "sentinel-not-a-real-credential-9137"


def run_main(argv: list[str]) -> tuple[int, str, str]:
    """Drive ``cli.main`` the way a shell does, capturing both streams."""
    stdout, stderr = io.StringIO(), io.StringIO()
    code = 0
    try:
        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            cli.main(argv)
    except SystemExit as exit_error:
        code = int(exit_error.code or 0)
    return code, stdout.getvalue(), stderr.getvalue()


@pytest.mark.parametrize(
    "flag",
    [
        "--api-key",
        "--api_key",
        "--apikey",
        "--API-KEY",
        "--token",
        "--access-token",
        "--bearer-token",
        "--secret",
        "--password",
        "--credential",
    ],
)
@pytest.mark.parametrize("attached", [False, True])
def test_a_credential_flag_is_rejected_without_repeating_its_value(
    tmp_path: Path, flag: str, attached: bool
) -> None:
    """argparse quotes an unrecognized argument back, value and all.

    The credential is read from the environment and no flag accepts one,
    so the flag is refused before argparse can format it into a message.
    """
    extra = [f"{flag}={_SENTINEL}"] if attached else [flag, _SENTINEL]

    code, out, err = run_main(base_argv(tmp_path) + extra)

    assert code == 2
    assert _SENTINEL not in err
    assert _SENTINEL not in out
    assert flag.split("=")[0] in err
    assert "--api-key-env" in err
    assert not (tmp_path / "artifacts").exists()


def test_the_env_var_flag_is_not_mistaken_for_a_credential_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--api-key-env`` names a variable, so it must keep working."""
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    code, out, err = run_main(
        base_argv(tmp_path, api_key_env="ZAI_API_KEY") + ["--dry-run"]
    )

    assert code == 0
    assert err == ""
    assert json.loads(out)["credential_env_var"] == "[REDACTED]"


@pytest.mark.parametrize(
    "extra",
    [
        ["--unknown-flag", _SENTINEL],
        [f"--unknown-flag={_SENTINEL}"],
        ["--temperature", _SENTINEL],
        ["--max-output-tokens", _SENTINEL],
        ["--reasoning-effort", _SENTINEL],
    ],
)
def test_no_parse_diagnostic_repeats_a_supplied_value(
    tmp_path: Path, extra: list[str]
) -> None:
    """A value typed into the wrong option is still a value worth containing."""
    code, out, err = run_main(base_argv(tmp_path) + extra)

    assert code == 2
    assert _SENTINEL not in err
    assert _SENTINEL not in out
    assert "error" in err


def test_an_unknown_subcommand_value_is_not_echoed(tmp_path: Path) -> None:
    code, out, err = run_main([_SENTINEL])

    assert code == 2
    assert _SENTINEL not in err
    assert _SENTINEL not in out


def test_a_valid_invocation_is_unaffected_by_the_rejection_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    code, out, _ = run_main(base_argv(tmp_path) + ["--dry-run"])

    assert code == 0
    assert json.loads(out)["network_request_performed"] is False


def test_a_secret_in_the_endpoint_query_is_not_echoed_by_a_parse_error(
    tmp_path: Path,
) -> None:
    """The endpoint is a value like any other, so its query stays contained."""
    endpoint = f"{ENDPOINT}?deployment={_SENTINEL}"
    argv = base_argv(tmp_path)
    argv[argv.index(ENDPOINT)] = endpoint

    code, out, err = run_main(argv + ["--unknown-flag", "x"])

    assert code == 2
    assert _SENTINEL not in err
    assert _SENTINEL not in out


# Seventh review pass: attached short clusters, choice literals, and REMAINDER.


@pytest.mark.parametrize("flag", ["-p", "-k", "-H", "-u", "-1"])
def test_an_attached_short_cluster_value_is_not_echoed(
    tmp_path: Path, flag: str
) -> None:
    """``-p<password>`` is the attached form, so the tail is a value."""
    code, out, err = run_main(base_argv(tmp_path) + [f"{flag}{_SENTINEL}"])

    assert code == 2
    assert _SENTINEL not in err
    assert _SENTINEL not in out


def test_an_attached_short_cluster_keeps_its_flag_letter(tmp_path: Path) -> None:
    """Redacting the value must not cost the caller the option they typed."""
    _, _, err = run_main(base_argv(tmp_path) + [f"-p{_SENTINEL}"])

    assert "-p[REDACTED]" in err


def test_a_mistyped_subcommand_leaves_the_valid_choices_readable() -> None:
    """A typo is a prefix of the real name, and the real name must survive."""
    code, _, err = run_main(["collect-ap"])

    assert code == 2
    assert "'collect-api'" in err
    assert "[REDACTED]i" not in err


def test_a_value_containing_an_option_name_is_still_replaced_whole(
    tmp_path: Path,
) -> None:
    """Protecting this program's vocabulary must not carve a hole in the scrub."""
    embedded = f"x--api-key-env{_SENTINEL}"
    code, out, err = run_main(base_argv(tmp_path) + ["--unknown-flag", embedded])

    assert code == 2
    assert embedded not in err + out
    assert _SENTINEL not in err + out


def test_a_credential_flag_after_a_bare_separator_is_not_rejected(
    tmp_path: Path,
) -> None:
    """Everything after ``--`` belongs to a recorded command, not to us."""
    stdout_file = tmp_path / "llama.txt"
    stdout_file.write_text("llama_perf_context_print: eval time = 1.0 ms\n")

    code, out, err = run_main(
        [
            "parse-llama-cpp",
            "--run-id",
            "r1",
            "--model-id",
            "local.gguf",
            "--stdout-file",
            str(stdout_file),
            "--",
            "llama-server",
            "--api-key",
            _SENTINEL,
        ]
    )

    assert code == 0
    assert err == ""
    assert _SENTINEL not in out
    assert json.loads(out)["command"]["argv"] == [
        "llama-server",
        "--api-key",
        "[REDACTED]",
    ]


def test_an_attached_credential_flag_in_a_recorded_command_is_redacted(
    tmp_path: Path,
) -> None:
    stdout_file = tmp_path / "llama.txt"
    stdout_file.write_text("llama_perf_context_print: eval time = 1.0 ms\n")

    code, out, _ = run_main(
        [
            "parse-llama-cpp",
            "--run-id",
            "r1",
            "--model-id",
            "local.gguf",
            "--stdout-file",
            str(stdout_file),
            "--",
            "llama-server",
            f"--api-key={_SENTINEL}",
            "--port",
            "8080",
        ]
    )

    assert code == 0
    assert _SENTINEL not in out
    assert json.loads(out)["command"]["argv"] == [
        "llama-server",
        "--api-key=[REDACTED]",
        "--port",
        "8080",
    ]


def test_a_credential_flag_before_the_separator_is_still_rejected(
    tmp_path: Path,
) -> None:
    """The separator relaxes the scan, it does not disable it."""
    code, _, err = run_main(
        ["parse-llama-cpp", "--api-key", _SENTINEL, "--", "llama-server"]
    )

    assert code == 2
    assert _SENTINEL not in err
    assert "--api-key is not a supported option" in err


# Eighth review pass: a name is a name only when this program defined it.


@pytest.mark.parametrize("suffix", ["=", "==", ""])
@pytest.mark.parametrize("flag", ["-p", "-k"])
def test_a_base64_padded_attached_cluster_is_not_echoed(
    tmp_path: Path, flag: str, suffix: str
) -> None:
    """Splitting on ``=`` first leaves nothing to redact in a padded key."""
    secret = f"{_SENTINEL}{suffix}"
    code, out, err = run_main(base_argv(tmp_path) + [f"{flag}{secret}"])

    assert code == 2
    assert secret not in err + out
    assert _SENTINEL not in err + out


@pytest.mark.parametrize("flag", ["--api-key", "--token", "--unknown-flag"])
def test_a_long_option_with_a_dropped_space_is_not_echoed(
    tmp_path: Path, flag: str
) -> None:
    """``--api-key$KEY`` is caller data wearing a long option's clothes."""
    code, out, err = run_main(base_argv(tmp_path) + [f"{flag}{_SENTINEL}"])

    assert code == 2
    assert _SENTINEL not in err + out


def test_a_dropped_space_after_a_real_option_keeps_the_option_readable(
    tmp_path: Path,
) -> None:
    """A defined option is a name, so only the tail past it is a value."""
    code, _, err = run_main(base_argv(tmp_path) + [f"--dry-run{_SENTINEL}"])

    assert code == 2
    assert _SENTINEL not in err
    assert "--dry-run[REDACTED]" in err


def test_a_defined_option_is_never_treated_as_a_value() -> None:
    """The scrub must not swallow the names that make a diagnostic useful."""
    code, _, err = run_main(["collect-api", "--run-id", "r1"])

    assert code == 2
    assert "--endpoint" in err
    assert "--model-id" in err


def test_a_recorded_credential_value_that_looks_like_a_flag_is_redacted(
    tmp_path: Path,
) -> None:
    """base64url values start with ``-`` often enough to matter."""
    stdout_file = tmp_path / "llama.txt"
    stdout_file.write_text("llama_perf_context_print: eval time = 1.0 ms\n")

    code, out, _ = run_main(
        [
            "parse-llama-cpp",
            "--run-id",
            "r1",
            "--model-id",
            "local.gguf",
            "--stdout-file",
            str(stdout_file),
            "--",
            "llama-server",
            "--api-key",
            f"-{_SENTINEL}",
            "--port",
            "8080",
        ]
    )

    assert code == 0
    assert _SENTINEL not in out
    assert json.loads(out)["command"]["argv"] == [
        "llama-server",
        "--api-key",
        "[REDACTED]",
        "--port",
        "8080",
    ]


def test_a_trailing_credential_flag_in_a_recorded_command_has_nothing_to_eat(
    tmp_path: Path,
) -> None:
    stdout_file = tmp_path / "llama.txt"
    stdout_file.write_text("llama_perf_context_print: eval time = 1.0 ms\n")

    code, out, _ = run_main(
        [
            "parse-llama-cpp",
            "--run-id",
            "r1",
            "--model-id",
            "local.gguf",
            "--stdout-file",
            str(stdout_file),
            "--",
            "llama-server",
            "--api-key",
        ]
    )

    assert code == 0
    assert json.loads(out)["command"]["argv"] == ["llama-server", "--api-key"]


# --- Ninth review pass -------------------------------------------------------


SENTINEL = "Sentinel0123456789abcdefXYZ"

# Values whose ``repr`` differs from the value itself. argparse formats the
# offending token with ``%r`` in several messages, so a scrub that only
# looks for the raw spelling matches nothing and prints the value in full.
ESCAPING_SUFFIXES = ["\n", "\r", "\t", "\u200b", "\u00a0", "\\x", "'\"", "\x01"]


@pytest.mark.parametrize("suffix", ESCAPING_SUFFIXES)
def test_a_value_repr_escapes_is_not_echoed_by_the_subcommand_diagnostic(
    suffix: str,
) -> None:
    """``invalid choice: %r`` renders the token, it does not print it."""
    code, out, err = run_main([SENTINEL + suffix])

    assert code == 2
    assert SENTINEL not in out
    assert SENTINEL not in err
    assert "[REDACTED]" in err


@pytest.mark.parametrize("suffix", ESCAPING_SUFFIXES)
def test_a_value_repr_escapes_is_not_echoed_by_a_type_conversion_error(
    suffix: str, tmp_path: Path
) -> None:
    code, out, err = run_main(
        [
            "collect-api",
            "--endpoint",
            "https://api.example.test/v1/chat/completions",
            "--model-id",
            "glm-5.3",
            "--run-id",
            "r1",
            "--output-dir",
            str(tmp_path),
            "--seed",
            SENTINEL + suffix,
        ]
    )

    assert code == 2
    assert SENTINEL not in out
    assert SENTINEL not in err


@pytest.mark.parametrize("suffix", ESCAPING_SUFFIXES)
def test_a_value_repr_escapes_is_not_echoed_by_an_ignored_explicit_argument(
    suffix: str, tmp_path: Path
) -> None:
    code, out, err = run_main(
        [
            "collect-api",
            "--endpoint",
            "https://api.example.test/v1/chat/completions",
            "--model-id",
            "glm-5.3",
            "--run-id",
            "r1",
            "--output-dir",
            str(tmp_path),
            f"--dry-run={SENTINEL}{suffix}",
        ]
    )

    assert code == 2
    assert SENTINEL not in out
    assert SENTINEL not in err


def test_the_scrub_still_leaves_the_valid_choices_readable() -> None:
    """Rendering-aware scrubbing must not eat this program's vocabulary."""
    code, _, err = run_main([SENTINEL + "\n"])

    assert code == 2
    assert "collect-api" in err


@pytest.mark.parametrize(
    "flags",
    [
        ("--api-key", "--token"),
        ("--api-key", "--api-key"),
        ("--token", "--api_key"),
        ("--api-key", "--APIKEY"),
    ],
)
def test_a_credential_flag_does_not_consume_another_credential_flag(
    tmp_path: Path, flags: tuple[str, str]
) -> None:
    """The one token after a credential flag that is not its value.

    Swallowing the second flag as the first flag's value means the second
    flag's own handler never runs and the real credential lands in
    ``record.command.argv`` verbatim.
    """
    stdout_file = tmp_path / "llama.txt"
    stdout_file.write_text("llama_perf_context_print: eval time = 1.0 ms\n")

    code, out, _ = run_main(
        [
            "parse-llama-cpp",
            "--run-id",
            "r1",
            "--model-id",
            "local.gguf",
            "--stdout-file",
            str(stdout_file),
            "--",
            "llama-server",
            flags[0],
            flags[1],
            SENTINEL,
            "--port",
            "8080",
        ]
    )

    assert code == 0
    argv = json.loads(out)["command"]["argv"]
    assert SENTINEL not in argv
    assert argv == [
        "llama-server",
        flags[0],
        flags[1],
        "[REDACTED]",
        "--port",
        "8080",
    ]


def test_a_credential_flag_followed_by_an_ordinary_flag_still_redacts_it(
    tmp_path: Path,
) -> None:
    """Only another credential flag is exempt, not any flag-shaped token."""
    stdout_file = tmp_path / "llama.txt"
    stdout_file.write_text("llama_perf_context_print: eval time = 1.0 ms\n")

    code, out, _ = run_main(
        [
            "parse-llama-cpp",
            "--run-id",
            "r1",
            "--model-id",
            "local.gguf",
            "--stdout-file",
            str(stdout_file),
            "--",
            "llama-server",
            "--api-key",
            "-Ab9x-secret-value",
            "--port",
            "8080",
        ]
    )

    assert code == 0
    argv = json.loads(out)["command"]["argv"]
    assert "-Ab9x-secret-value" not in argv
    assert argv == ["llama-server", "--api-key", "[REDACTED]", "--port", "8080"]


# --- Tenth review pass -------------------------------------------------------


# Uppercase, so it satisfies the conventional-name rule and only the
# presence rule can stop it. That is the harder of the two vectors.
NAME_SLOT_KEY = "AKIA1234567890ABCDEF"
# The shapes real keys take, all rejected by the name rule.
KEY_SHAPED_NAMES = [
    "sk-3f0a1c2b-9d8e-7f6a-5b4c",
    "sk_live_9f2b7d41ca6e4b8f",
    "ghp_16C7e42F292c6912E7710c838347Ae178B4a",
]


def _artifact_text(root: Path) -> str:
    if not root.exists():
        return ""
    return "".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(root.rglob("*"))
        if path.is_file()
    )


@pytest.mark.parametrize("name", KEY_SHAPED_NAMES)
@pytest.mark.parametrize("dry_run", [False, True])
def test_a_credential_in_the_name_slot_never_reaches_a_stream_or_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, dry_run: bool
) -> None:
    """Swapping --api-key for --api-key-env must not become the leak.

    The refusal for ``--api-key`` tells the caller to name a variable
    instead. The mechanical response is to keep the value and change the
    flag, which puts the credential where the name goes.
    """
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)
    argv = base_argv(tmp_path, api_key_env=name)
    if dry_run:
        argv = argv + ["--dry-run"]

    code, out, err = run_main(argv)

    assert code == 1
    assert name not in out
    assert name not in err
    assert name not in _artifact_text(tmp_path / "artifacts")


@pytest.mark.parametrize("dry_run", [False, True])
def test_an_uppercase_credential_in_the_name_slot_is_contained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, dry_run: bool
) -> None:
    """The shape rule cannot catch this one, so presence must."""
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)
    argv = base_argv(tmp_path, api_key_env=NAME_SLOT_KEY)
    if dry_run:
        argv = argv + ["--dry-run"]

    _, out, err = run_main(argv)

    assert NAME_SLOT_KEY not in out
    assert NAME_SLOT_KEY not in err
    assert NAME_SLOT_KEY not in _artifact_text(tmp_path / "artifacts")


def test_the_dry_run_plan_masks_an_unproven_name_in_the_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ZAI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)

    code, out, _ = run_main(
        base_argv(tmp_path, api_key_env=NAME_SLOT_KEY) + ["--dry-run"]
    )

    assert code == 0
    payload = json.loads(out)
    assert payload["credential_env_var"] == "[REDACTED]"
    assert payload["credential_env_var_present"] is False
    assert NAME_SLOT_KEY not in json.dumps(payload)


def test_a_set_variable_is_still_named_in_the_dry_run_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Containment must not cost the ordinary case its readability."""
    monkeypatch.setenv("ZAI_API_KEY", "sentinel-value")
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)

    code, out, _ = run_main(base_argv(tmp_path) + ["--dry-run"])

    assert code == 0
    payload = json.loads(out)
    assert payload["credential_env_var"] == "ZAI_API_KEY"
    assert payload["credential_env_var_present"] is True
    assert "sentinel-value" not in json.dumps(payload)


# --- Eleventh review pass ----------------------------------------------------


def run_parser(argv: list[str]) -> tuple[int, str, str, str]:
    """Drive the public parser directly, as an embedding caller would.

    ``main`` is not the only entry point. ``build_parser().parse_args(...)``
    is public and is what a test or another program reaches for, so it has
    to be as safe as the path that goes through ``main``.
    """
    stdout, stderr = io.StringIO(), io.StringIO()
    code = 0
    detail = ""
    try:
        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            cli.build_parser().parse_args(list(argv))
    except SystemExit as exit_error:
        code = int(exit_error.code or 0)
        detail = str(exit_error)
    return code, stdout.getvalue(), stderr.getvalue(), detail


@pytest.fixture(autouse=True)
def _scrub_state_is_not_inherited() -> Iterator[None]:
    """Every case must start and end with no scrub state in force.

    The leak this guards appears when the state left over from an earlier
    parse is mistaken for one still in force, which suppresses
    installation for the next parse and scrubs it against the wrong argv.
    Asserting on both sides means a case that leaves state behind is
    reported here rather than by whichever unrelated case runs next.
    """
    assert cli._scrub_state.get() is None
    yield
    assert cli._scrub_state.get() is None


_PARSER_SECRET = "sk-live-CANARY-4f9a2b7c1d8e"


@pytest.mark.parametrize(
    "argv",
    [
        # An unsupported credential flag, separate and attached.
        ["collect-api", "--api-key", _PARSER_SECRET],
        ["collect-api", f"--api-key={_PARSER_SECRET}"],
        # Left over after an otherwise complete command line, which is the
        # path argparse reports itself rather than through parse_known_args.
        # Every required option is present, so parsing gets far enough for
        # the unrecognized-argument report to be what fails.
        [
            "collect-api",
            "--run-id",
            "cli-run",
            "--endpoint",
            "https://api.z.ai/api/paas/v4/chat/completions",
            "--model-id",
            "glm-5.3",
            "--prompt-file",
            "prompt.txt",
            "--output-dir",
            "out",
            "--api-key",
            _PARSER_SECRET,
        ],
        # An arbitrary rejected value, with no credential flag involved.
        ["collect-api", "--reasoning-effort", _PARSER_SECRET],
        ["collect-api", "--thinking-type", _PARSER_SECRET],
        [_PARSER_SECRET],
        ["--", _PARSER_SECRET],
        [f"--endpoint={_PARSER_SECRET}"],
    ],
)
def test_direct_parser_use_never_echoes_a_value(argv: list[str]) -> None:
    """No stream, and no exit detail, repeats what the caller supplied."""
    code, out, err, detail = run_parser(argv)

    assert code != 0
    for stream in (out, err, detail):
        assert _PARSER_SECRET not in stream
        assert _PARSER_SECRET.lower() not in stream.lower()


def test_direct_parser_use_still_names_the_option() -> None:
    """Scrubbing values must not take the actionable part with it."""
    _, _, err, _ = run_parser(["collect-api", "--reasoning-effort", _PARSER_SECRET])

    assert "--reasoning-effort" in err
    assert _PARSER_SECRET not in err


@pytest.mark.parametrize(
    "argv",
    [
        ["collect-api", "--reasoning-effort", _PARSER_SECRET],
        ["collect-api", f"--reasoning-effort={_PARSER_SECRET}"],
        ["collect-api", "--thinking-type", _PARSER_SECRET],
        [_PARSER_SECRET],
    ],
)
def test_direct_parse_known_args_never_echoes_a_value(argv: list[str]) -> None:
    """``parse_known_args`` is public too, and is reached on its own.

    Callers that want to forward unrecognized tokens use it directly, so
    it cannot rely on ``parse_args`` having installed the scrub state for
    it. Its own diagnostics have to be safe when it is the entry point.
    """
    stdout, stderr = io.StringIO(), io.StringIO()
    detail = ""
    try:
        with (
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            cli.build_parser().parse_known_args(list(argv))
    except SystemExit as exit_error:
        detail = str(exit_error)

    for stream in (stdout.getvalue(), stderr.getvalue(), detail):
        assert _PARSER_SECRET not in stream
        assert _PARSER_SECRET.lower() not in stream.lower()


def test_parser_scrub_state_is_restored_after_a_direct_parse() -> None:
    """A parse must not leave process-wide state behind it.

    The scope exists for the duration of one parse. Leaking it would make
    a later, unrelated diagnostic scrub tokens from a command line that
    has nothing to do with it.
    """
    run_parser(["collect-api", "--reasoning-effort", _PARSER_SECRET])

    assert cli._scrub_state.get() is None


def test_a_valid_direct_parse_returns_the_parsed_arguments() -> None:
    """The scope is a wrapper, not a change in behaviour."""
    args = cli.build_parser().parse_args(
        [
            "collect-api",
            "--endpoint",
            "https://api.z.ai/api/paas/v4/chat/completions",
            "--model-id",
            "glm-5.3",
            "--prompt-file",
            "prompt.txt",
            "--run-id",
            "r1",
            "--output-dir",
            "out",
            "--api-key-env",
            "ZAI_API_KEY",
        ]
    )

    assert args.model_id == "glm-5.3"
    assert args.api_key_env == "ZAI_API_KEY"


# --- Twelfth review pass -----------------------------------------------------


def test_a_direct_parse_after_main_is_still_scrubbed() -> None:
    """State left standing by an earlier call must not disarm the next one.

    ``main`` used to assign the scrub globals and never restore them. The
    scope then treated a populated state as an enclosing scope still in
    force and skipped installation, so the second parse was scrubbed
    against the first command line and echoed its own value in full. Both
    now go through the scope, and nesting is detected from the context
    variable being set rather than from it being non-empty.
    """
    first = "FIRSTSECRET-4f9a2b7c"
    second = "SECONDSECRET-1d8e3a6b"

    with contextlib.suppress(SystemExit):
        run_main([first])
    _, stdout, stderr, detail = run_parser([second])

    for stream in (stdout, stderr, detail):
        assert second not in stream
        assert first not in stream


def test_main_restores_the_scrub_state_it_installed() -> None:
    """The scope must unwind through the SystemExit argparse raises."""
    with contextlib.suppress(SystemExit):
        run_main(["definitely-not-a-command"])

    assert cli._scrub_state.get() is None


@pytest.mark.parametrize("value", ["abc", "ab", "a", "xy"])
def test_a_short_argument_value_is_not_echoed_either(value: str) -> None:
    """The guarantee was absolute, but the implementation had a floor.

    Values below four characters were never collected, so argparse
    printed them verbatim. They are collected now; only their bare
    spelling is withheld from replacement, because a two character string
    occurs inside ordinary words and blanking every occurrence would
    destroy the message.
    """
    _, stdout, stderr, detail = run_parser(["collect-api", "--reasoning-effort", value])

    for stream in (stdout, stderr, detail):
        assert f"'{value}'" not in stream
        assert f'"{value}"' not in stream


def test_scrubbing_a_short_value_keeps_the_message_readable() -> None:
    """Replacing a two character value as a bare substring would not."""
    _, _, stderr, detail = run_parser(["collect-api", "--reasoning-effort", "ax"])

    message = stderr + detail
    assert "--reasoning-effort" in message
    assert "invalid choice" in message
    # The valid spellings are what make the error actionable, and each
    # contains one of the value's characters.
    for choice in ("low", "high", "max"):
        assert choice in message


# --- Fourteenth review pass ---------------------------------------------------


@pytest.mark.parametrize("slot", ["prompt-file", "system-prompt-file"])
def test_a_runtime_diagnostic_scrubs_values_the_caller_supplied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, slot: str
) -> None:
    """The argv scrub has to stay installed while the command runs.

    Parsing succeeding is not the end of the risk. A path that does not open
    surfaces in ``OSError`` text verbatim, and that happens inside
    ``args.func(args)``. With the scrub scope closed at the end of parsing,
    the scrub that exists to remove caller-supplied values had nothing
    installed and silently did nothing.
    """
    monkeypatch.setenv("ZAI_API_KEY", "test-key-not-a-real-credential")
    secret_shaped = "sk-caller-supplied-value-that-must-not-echo"
    argv = base_argv(tmp_path)
    if slot == "prompt-file":
        argv[argv.index("--prompt-file") + 1] = secret_shaped
    else:
        argv.extend(["--system-prompt-file", secret_shaped])
    argv.append("--dry-run")

    code, out, err = run_main(argv)

    assert code != 0
    assert secret_shaped not in out
    assert secret_shaped not in err
    assert "[REDACTED]" in out + err


def test_the_scrub_scope_is_still_undone_after_the_command_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running inside the scope must not leave state behind for the next parse.

    The scope was widened to cover execution, so the property it was
    introduced for, that nothing survives into a later parse in the same
    process, is re-checked at the wider boundary.
    """
    monkeypatch.setenv("ZAI_API_KEY", "test-key-not-a-real-credential")
    first = base_argv(tmp_path)
    first[first.index("--prompt-file") + 1] = "sk-first-invocation-value"
    first.append("--dry-run")
    run_main(first)

    # A second, unrelated parse must not be scrubbed against the first argv.
    code, out, err = run_main(["collect-api", "--help"])

    assert code == 0
    assert "[REDACTED]" not in out
    assert "sk-first-invocation-value" not in out + err


# --- Fifteenth review pass ---------------------------------------------------


@pytest.mark.parametrize(
    "token,expected",
    [
        (f"--api-key{_SENTINEL}", "--api-key[REDACTED]"),
        (f"--apikey{_SENTINEL}", "--apikey[REDACTED]"),
        (f"--token{_SENTINEL}", "--token[REDACTED]"),
        (f"--bearer{_SENTINEL}", "--bearer[REDACTED]"),
        (f"--password{_SENTINEL}", "--password[REDACTED]"),
        (f"-key{_SENTINEL}", "-key[REDACTED]"),
        (f"-p{_SENTINEL}", "-p[REDACTED]"),
        ("--api-key-sk_live_example", "--api-key[REDACTED]"),
        ("--api-key_SECRET", "--api-key[REDACTED]"),
        (f"--api-key{_SENTINEL}=tail", "--api-key[REDACTED]"),
    ],
)
def test_a_glued_credential_flag_in_a_recorded_command_is_redacted(
    tmp_path: Path, token: str, expected: str
) -> None:
    """A dropped space is the most common way a key ends up in a command.

    ``--api-keySECRET`` is a single token, so splitting on whitespace or on
    ``=`` finds nothing to redact and the whole thing is written into the
    record verbatim. The recorded command is the artifact people paste into
    issues, which is exactly the wrong place for a working key.
    """
    stdout_file = tmp_path / "llama.txt"
    stdout_file.write_text("llama_perf_context_print: eval time = 1.0 ms\n")

    code, out, _ = run_main(
        [
            "parse-llama-cpp",
            "--run-id",
            "r1",
            "--model-id",
            "local.gguf",
            "--stdout-file",
            str(stdout_file),
            "--",
            "llama-server",
            token,
        ]
    )

    assert code == 0
    assert _SENTINEL not in out
    assert json.loads(out)["command"]["argv"] == ["llama-server", expected]


@pytest.mark.parametrize(
    "token",
    [
        "--api-key-file",
        "--tokenizer",
        "--keyfile",
        "--authors",
        "--secrets",
        "--token-limit",
        "--n-keep",
        "--no-secret",
        "--verbose",
        "--authentication-method",
        "--authorization-policy",
        "--tokenizer-model",
        "--api-key-file=/keys/key.txt",
        "--authentication-method=oauth",
        "--authorization-policy=deny",
        "--tokenizer-model=model.json",
        "--password-file",
        "--password-env",
        "--access-token-lifetime=3600",
        "--password-policy",
        "--credentials-file",
        "--apikeys-file",
    ],
)
def test_an_option_that_only_looks_glued_is_left_intact(
    tmp_path: Path, token: str
) -> None:
    """Redacting a real option name corrupts the record it is protecting.

    ``--tokenizer`` starts with a credential stem and is an ordinary
    option, so a prefix match with no length floor would rewrite it into
    ``--token[REDACTED]`` and destroy the reproduction the record exists
    to provide.
    """
    stdout_file = tmp_path / "llama.txt"
    stdout_file.write_text("llama_perf_context_print: eval time = 1.0 ms\n")

    code, out, _ = run_main(
        [
            "parse-llama-cpp",
            "--run-id",
            "r1",
            "--model-id",
            "local.gguf",
            "--stdout-file",
            str(stdout_file),
            "--",
            "llama-server",
            token,
        ]
    )

    assert code == 0
    assert json.loads(out)["command"]["argv"] == ["llama-server", token]


@pytest.mark.parametrize(
    "token",
    [
        "--api-keylowercasesecret",
        "--accesskeylowercasesecret",
        "--access_keylowercasesecret",
        "--accesstokenlowercasesecret",
        "--access_tokenlowercasesecret",
        "--apisecretlowercasesecret",
        "--api_secretlowercasesecret",
        "--apitokenlowercasesecret",
        "--api_tokenlowercasesecret",
        "--authhuntertwo",
        "--auth_tokenlowercasesecret",
        "--bearerhuntertwo",
        "--bearer_tokenlowercasesecret",
        "--clientsecretlowercasesecret",
        "--client_secretlowercasesecret",
        "--credentialhuntertwo",
        "--passwordhuntertwo",
        "--secret_keylowercasesecret",
        "--tokenhuntertwo",
    ],
)
def test_lowercase_value_glued_to_unambiguous_flag_is_redacted(
    tmp_path: Path, token: str
) -> None:
    stdout_file = tmp_path / "llama.txt"
    stdout_file.write_text("llama_perf_context_print: eval time = 1.0 ms\n")

    code, out, err = run_main(
        [
            "parse-llama-cpp",
            "--run-id",
            "r1",
            "--model-id",
            "local.gguf",
            "--stdout-file",
            str(stdout_file),
            "--",
            "llama-server",
            token,
        ]
    )

    assert code == 0
    assert err == ""
    assert "lowercasesecret" not in out
    assert "huntertwo" not in out
    assert "[REDACTED]" in out


def test_the_retained_event_limit_is_reconstructed_and_validated(
    tmp_path: Path,
) -> None:
    """The bound shapes the evidence, so it belongs in the reproduction.

    A run that kept a different number of event rows produced different
    artifacts, so leaving the bound out of the recorded command would make
    two different runs look identical.
    """
    argv = base_argv(tmp_path) + ["--retained-event-limit", "7", "--dry-run"]
    code, out, err = run_main(argv)

    assert code == 0, err
    plan = json.loads(out)
    assert "--retained-event-limit" in plan["plan"]["command"]
    index = plan["plan"]["command"].index("--retained-event-limit")
    assert plan["plan"]["command"][index + 1] == "7"


@pytest.mark.parametrize("limit", ["0", "-1"])
def test_a_non_positive_retained_event_limit_is_refused(
    tmp_path: Path, limit: str
) -> None:
    """A bound of zero or less would silently discard the whole timeline."""
    code, _, err = run_main(
        base_argv(tmp_path) + ["--retained-event-limit", limit, "--dry-run"]
    )

    assert code == 1
    assert "retained_event_limit must be a positive integer" in err


# ---------------------------------------------------------------------------
# Sixteenth review pass: argument iterables are read exactly once
# ---------------------------------------------------------------------------


def test_parse_args_accepts_a_one_shot_iterator_without_losing_tokens(
    tmp_path: Path,
) -> None:
    """argparse documents an iterable, and the scrub scope also reads it.

    Two consumers reading the same generator would leave one of them with
    an empty command line. Whichever lost would either parse nothing or
    scrub nothing, and the second failure mode is the dangerous one.
    """
    argv = base_argv(tmp_path) + ["--dry-run"]
    parser = cli.build_parser()

    args = parser.parse_args(token for token in argv)

    assert args.model_id == "glm-5.3"
    assert args.dry_run is True


def test_a_one_shot_iterator_still_gets_its_values_scrubbed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The scrub state has to be built from the same tokens argparse parses."""
    secret = "GENERATOR_CANARY_SECRET"
    argv = base_argv(tmp_path) + ["--reasoning-effort", secret]
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(token for token in argv)

    captured = capsys.readouterr()
    assert secret not in captured.out
    assert secret not in captured.err
    assert "--reasoning-effort" in captured.err


def test_parse_known_args_also_reads_a_one_shot_iterator_once(
    tmp_path: Path,
) -> None:
    argv = base_argv(tmp_path) + ["--dry-run", "--not-a-flag"]
    parser = cli.build_parser()

    args, leftover = parser.parse_known_args(token for token in argv)

    assert args.model_id == "glm-5.3"
    assert leftover == ["--not-a-flag"]


# ---------------------------------------------------------------------------
# Final consolidated regressions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("entrypoint", ["main", "parse-args", "parse-known"])
def test_dry_run_write_errors_scrub_the_output_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", API_KEY)
    canary = "output-CANARY-secret-path"
    argv = base_argv(tmp_path) + ["--dry-run"]
    argv[argv.index("--output-dir") + 1] = canary

    def fail_write(path: Path, _content: str) -> None:
        raise OSError(f"cannot write {path}")

    monkeypatch.setattr(cli, "atomic_write_text", fail_write)
    if entrypoint == "main":
        code, out, err = run_main(argv)
    else:
        code, out, err = invoke_captured(argv, known=entrypoint == "parse-known")

    assert code == 1
    assert canary not in out + err
    assert "[REDACTED]" in err


@pytest.mark.parametrize("entrypoint", ["main", "parse-args", "parse-known"])
@pytest.mark.parametrize("dry_run", [False, True], ids=["real", "dry-run"])
def test_handler_errors_scrub_values_for_every_parser_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    dry_run: bool,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", API_KEY)
    canary = "caller-model-CANARY-that-must-not-echo"
    argv = base_argv(tmp_path)
    argv[argv.index("--model-id") + 1] = canary
    if dry_run:
        argv.append("--dry-run")

        def fail_write(_path: Path, _content: str) -> None:
            raise OSError(f"write failed for {canary}")

        monkeypatch.setattr(cli, "atomic_write_text", fail_write)
    else:

        def fail_collection(*_args: Any, **_kwargs: Any) -> Any:
            raise OSError(f"provider rejected {canary}")

        monkeypatch.setattr(cli, "collect_openai_stream", fail_collection)

    if entrypoint == "main":
        code, out, err = run_main(argv)
    else:
        code, out, err = invoke_captured(argv, known=entrypoint == "parse-known")

    assert code == 1
    assert canary not in out + err
    assert "[REDACTED]" in err


@pytest.mark.parametrize("entrypoint", ["main", "parse-args", "parse-known"])
def test_returned_failures_scrub_values_for_every_parser_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    monkeypatch.setenv("ZAI_API_KEY", API_KEY)
    canary = "caller-model-CANARY-that-must-not-echo"
    provider_error = sse(
        {
            "error": {
                "code": "model_not_found",
                "message": f"model {canary} was not found",
            }
        }
    )
    monkeypatch.setattr(
        cli,
        "UrllibStreamingTransport",
        lambda: FakeTransport([provider_error]),
    )
    argv = base_argv(tmp_path)
    argv[argv.index("--model-id") + 1] = canary

    if entrypoint == "main":
        code, out, err = run_main(argv)
    else:
        code, out, err = invoke_captured(argv, known=entrypoint == "parse-known")

    assert code == 1
    assert canary not in out + err
    assert "[REDACTED]" in err


def test_mixed_case_glued_credential_flag_is_redacted(
    tmp_path: Path,
) -> None:
    stdout_file = tmp_path / "llama.txt"
    stdout_file.write_text("llama_perf_context_print: eval time = 1.0 ms\n")

    code, out, err = run_main(
        [
            "parse-llama-cpp",
            "--run-id",
            "r1",
            "--model-id",
            "local.gguf",
            "--stdout-file",
            str(stdout_file),
            "--",
            "llama-server",
            "--API-KEYSecret123",
        ]
    )

    assert code == 0
    assert err == ""
    assert "Secret123" not in out
    assert json.loads(out)["command"]["argv"] == [
        "llama-server",
        "--API-KEY[REDACTED]",
    ]


@pytest.mark.parametrize("value", ["a", "xy", "123"])
def test_short_bare_unrecognized_arguments_are_scrubbed(
    tmp_path: Path, value: str
) -> None:
    code, out, err = run_main(base_argv(tmp_path) + [value])

    assert code == 2
    assert out == ""
    assert f"unrecognized arguments: {value}" not in err
    assert "unrecognized arguments: [REDACTED]" in err
