"""Tests for the ``workloads run-api`` CLI surface.

No test here performs a network request. The dry-run tests install a
transport that fails if it is ever opened, and the execution tests inject
one that replays recorded byte chunks. No real API key is used and neither
OpenRouter nor Z.ai is ever contacted; both appear only as configuration.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer import cli
from llmtracefx.optimizer.collectors.openai_api import HTTPRequest
from llmtracefx.optimizer.workloads.catalog import (
    STRUCTURED_JSON_PROFILE_EXTRACTION,
)
from llmtracefx.optimizer.workloads.matrix import (
    DECODE_MODE_AUTOREGRESSIVE,
    DECODE_MODE_NATIVE_MTP,
    MatrixManifest,
    generate_matrix,
    write_matrix,
)
from llmtracefx.optimizer.workloads.schema import ContextTier

API_KEY = "run-api-cli-test-key-not-a-real-credential"
ENV_VAR = "LLMTRACEFX_TEST_API_KEY"
GOOD_JSON_ANSWER = '{"name": "Priya", "age": 34, "is_active": true}'


class ExplodingTransport:
    def open_stream(self, request: HTTPRequest) -> Any:
        raise AssertionError("no network request should have been attempted")


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
    """Class-level request log, so the CLI can construct it itself."""

    requests: list[HTTPRequest] = []
    chunks: list[bytes] = []
    status_code: int = 200

    def open_stream(self, request: HTTPRequest) -> FakeResponse:
        FakeTransport.requests.append(request)
        return FakeResponse(FakeTransport.chunks, FakeTransport.status_code)


def sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def answer_stream(answer: str) -> list[bytes]:
    return [
        sse(
            {
                "id": "chatcmpl-1",
                "model": "glm-5.3",
                "choices": [{"index": 0, "delta": {"content": answer}}],
            }
        ),
        sse(
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                },
            }
        ),
        b"data: [DONE]\n\n",
    ]


@pytest.fixture(autouse=True)
def _reset_fake_transport() -> Iterator[None]:
    FakeTransport.requests = []
    FakeTransport.chunks = answer_stream(GOOD_JSON_ANSWER)
    FakeTransport.status_code = 200
    yield
    FakeTransport.requests = []


@pytest.fixture(autouse=True)
def _no_real_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts unable to reach the network.

    Tests that need a response opt in by swapping in ``FakeTransport``.
    """
    monkeypatch.setattr(cli, "UrllibStreamingTransport", ExplodingTransport)


def build_matrix(tmp_path: Path) -> Path:
    output_dir = tmp_path / "matrix"
    manifest = generate_matrix(
        model_id="local/test-model",
        model_family="qwen3_next",
        output_dir=str(output_dir),
        workloads=(STRUCTURED_JSON_PROFILE_EXTRACTION,),
        context_tiers=(ContextTier.TIER_2K,),
        mtp_depths=(2,),
    )
    write_matrix(manifest)
    return output_dir / "manifest.json"


def base_argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "workloads",
        "run-api",
        "--matrix",
        str(build_matrix(tmp_path)),
        "--output-dir",
        str(tmp_path / "results"),
        "--model-id",
        "z-ai/glm-5.3",
        *extra,
    ]


def invoke(argv: list[str]) -> int:
    parser = cli.build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


# --- Profiles ----------------------------------------------------------------


def test_list_api_profiles_prints_both_documented_profiles(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert invoke(["workloads", "list-api-profiles"]) == 0
    payload = json.loads(capsys.readouterr().out)
    by_name = {profile["name"]: profile for profile in payload["profiles"]}
    assert by_name["openrouter"]["endpoint"] == (
        "https://openrouter.ai/api/v1/chat/completions"
    )
    assert by_name["openrouter"]["credential_env_var"] == "OPENROUTER_API_KEY"
    assert "z-ai/glm-5.3" in by_name["openrouter"]["documented_model_ids"]
    assert by_name["z.ai"]["endpoint"] == (
        "https://api.z.ai/api/paas/v4/chat/completions"
    )


@pytest.mark.parametrize(
    ("profile", "endpoint", "env_var"),
    [
        ("openrouter", "https://openrouter.ai", "OPENROUTER_API_KEY"),
        ("z.ai", "https://api.z.ai", "ZAI_API_KEY"),
    ],
)
def test_profile_supplies_endpoint_and_credential_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    profile: str,
    endpoint: str,
    env_var: str,
) -> None:
    monkeypatch.setenv(env_var, API_KEY)
    code = invoke(
        base_argv(
            tmp_path,
            "--profile",
            profile,
            "--mode",
            DECODE_MODE_AUTOREGRESSIVE,
            "--dry-run",
        )
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    plan = payload["rows"][0]["request_plan"]
    assert plan["endpoint_origin"] == endpoint
    assert plan["credential_env_var"] == env_var


def test_explicit_flags_override_the_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV_VAR, API_KEY)
    code = invoke(
        base_argv(
            tmp_path,
            "--profile",
            "openrouter",
            "--provider",
            "self-hosted",
            "--endpoint",
            "https://vllm.internal.example/v1/chat/completions",
            "--api-key-env",
            ENV_VAR,
            "--mode",
            DECODE_MODE_AUTOREGRESSIVE,
            "--dry-run",
        )
    )
    assert code == 0
    plan = json.loads(capsys.readouterr().out)["rows"][0]["request_plan"]
    assert plan["provider"] == "self-hosted"
    assert plan["endpoint_origin"] == "https://vllm.internal.example"


def test_unlisted_provider_works_without_a_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV_VAR, API_KEY)
    code = invoke(
        base_argv(
            tmp_path,
            "--provider",
            "self-hosted",
            "--endpoint",
            "https://vllm.internal.example/v1/chat/completions",
            "--api-key-env",
            ENV_VAR,
            "--mode",
            DECODE_MODE_AUTOREGRESSIVE,
            "--dry-run",
        )
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["rows"][0]["status"] == "ready"


def test_missing_binding_fields_without_a_profile_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert invoke(base_argv(tmp_path, "--dry-run")) == 1
    error = capsys.readouterr().err
    assert "--provider" in error
    assert "--endpoint" in error
    assert "--api-key-env" in error


# --- Dry run -----------------------------------------------------------------


def test_dry_run_writes_a_plan_and_performs_no_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", API_KEY)
    code = invoke(
        base_argv(
            tmp_path,
            "--profile",
            "openrouter",
            "--mode",
            DECODE_MODE_AUTOREGRESSIVE,
            "--dry-run",
        )
    )
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["dry_run"] is True
    assert payload["network_request_performed"] is False
    assert payload["credential_env_var_present"] is True
    assert FakeTransport.requests == []

    written = (tmp_path / "results" / "api_request_plan.json").read_text(
        encoding="utf-8"
    )
    assert json.loads(written) == payload


def test_dry_run_reports_unsupported_and_ready_rows_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", API_KEY)
    assert invoke(base_argv(tmp_path, "--profile", "openrouter", "--dry-run")) == 0
    rows = json.loads(capsys.readouterr().out)["rows"]
    statuses = {row["decode_mode"]: row["status"] for row in rows}
    assert statuses[DECODE_MODE_AUTOREGRESSIVE] == "ready"
    assert statuses[DECODE_MODE_NATIVE_MTP] == "unsupported"


def test_dry_run_returns_two_when_a_row_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    code = invoke(
        base_argv(
            tmp_path,
            "--profile",
            "openrouter",
            "--mode",
            DECODE_MODE_AUTOREGRESSIVE,
            "--dry-run",
        )
    )
    assert code == 2
    assert json.loads(capsys.readouterr().out)["rows"][0]["status"] == "blocked"


def test_dry_run_with_no_matching_rows_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", API_KEY)
    code = invoke(
        base_argv(
            tmp_path,
            "--profile",
            "openrouter",
            "--run-id",
            "no-such-row",
            "--dry-run",
        )
    )
    assert code == 1
    assert "No matrix rows matched" in capsys.readouterr().err


def test_dry_run_rejects_a_plain_http_remote_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV_VAR, API_KEY)
    code = invoke(
        base_argv(
            tmp_path,
            "--provider",
            "insecure",
            "--endpoint",
            "http://openrouter.ai/api/v1/chat/completions",
            "--api-key-env",
            ENV_VAR,
            "--dry-run",
        )
    )
    assert code == 1
    assert "must use https" in capsys.readouterr().err


# --- Execution ---------------------------------------------------------------


def test_successful_run_writes_artifacts_and_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", API_KEY)
    monkeypatch.setattr(cli, "UrllibStreamingTransport", FakeTransport)

    code = invoke(
        base_argv(
            tmp_path, "--profile", "openrouter", "--mode", DECODE_MODE_AUTOREGRESSIVE
        )
    )
    assert code == 0
    assert len(FakeTransport.requests) == 1
    out = capsys.readouterr().out
    assert "[COMPLETED]" in out

    verification = json.loads(
        next((tmp_path / "results" / "runs").glob("*/verification.json")).read_text(
            encoding="utf-8"
        )
    )
    assert verification["backend"] == "openai-api"
    assert verification["provider"] == "openrouter"
    assert verification["api_model_id"] == "z-ai/glm-5.3"
    assert verification["artifacts_verified"] is True


def test_provider_failure_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", API_KEY)
    monkeypatch.setattr(cli, "UrllibStreamingTransport", FakeTransport)
    FakeTransport.status_code = 503
    FakeTransport.chunks = [b'{"error": {"message": "upstream is down"}}']

    code = invoke(
        base_argv(
            tmp_path, "--profile", "openrouter", "--mode", DECODE_MODE_AUTOREGRESSIVE
        )
    )
    assert code == 1
    assert "[FAILED]" in capsys.readouterr().out


def test_native_mtp_only_selection_never_opens_a_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", API_KEY)
    # The transport stays ExplodingTransport from the autouse fixture, so
    # this asserts by construction that nothing was sent.
    code = invoke(
        base_argv(
            tmp_path,
            "--profile",
            "openrouter",
            "--mode",
            DECODE_MODE_NATIVE_MTP,
            "--reasoning-effort",
            "max",
        )
    )
    assert code == 0
    assert "[UNSUPPORTED]" in capsys.readouterr().out


def test_rerun_resumes_without_a_second_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", API_KEY)
    monkeypatch.setattr(cli, "UrllibStreamingTransport", FakeTransport)
    matrix = str(build_matrix(tmp_path))
    argv = [
        "workloads",
        "run-api",
        "--matrix",
        matrix,
        "--output-dir",
        str(tmp_path / "results"),
        "--model-id",
        "z-ai/glm-5.3",
        "--profile",
        "openrouter",
        "--mode",
        DECODE_MODE_AUTOREGRESSIVE,
    ]
    assert invoke(argv) == 0
    assert len(FakeTransport.requests) == 1

    capsys.readouterr()
    assert invoke(argv) == 0
    assert len(FakeTransport.requests) == 1, "resume must not re-issue a request"
    assert "[SKIPPED]" in capsys.readouterr().out


def test_no_resume_forces_a_second_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", API_KEY)
    monkeypatch.setattr(cli, "UrllibStreamingTransport", FakeTransport)
    matrix = str(build_matrix(tmp_path))
    argv = [
        "workloads",
        "run-api",
        "--matrix",
        matrix,
        "--output-dir",
        str(tmp_path / "results"),
        "--model-id",
        "z-ai/glm-5.3",
        "--profile",
        "openrouter",
        "--mode",
        DECODE_MODE_AUTOREGRESSIVE,
    ]
    assert invoke(argv) == 0
    assert invoke([*argv, "--no-resume"]) == 0
    assert len(FakeTransport.requests) == 2


def test_reasoning_settings_reach_the_request_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", API_KEY)
    monkeypatch.setattr(cli, "UrllibStreamingTransport", FakeTransport)

    assert (
        invoke(
            base_argv(
                tmp_path,
                "--profile",
                "openrouter",
                "--mode",
                DECODE_MODE_AUTOREGRESSIVE,
                "--reasoning-effort",
                "high",
                "--thinking",
                "enabled",
                "--clear-thinking",
                "false",
            )
        )
        == 0
    )
    body = json.loads(FakeTransport.requests[0].body.decode("utf-8"))
    assert body["reasoning_effort"] == "high"
    assert body["thinking"] == {"type": "enabled", "clear_thinking": False}


def test_missing_matrix_manifest_is_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = invoke(
        [
            "workloads",
            "run-api",
            "--matrix",
            str(tmp_path / "nope.json"),
            "--output-dir",
            str(tmp_path / "results"),
            "--model-id",
            "z-ai/glm-5.3",
            "--profile",
            "openrouter",
        ]
    )
    assert code == 1
    assert "Failed to load matrix manifest" in capsys.readouterr().err


# --- Secret containment ------------------------------------------------------


def test_there_is_no_api_key_flag(tmp_path: Path) -> None:
    """A credential must never be accepted as a command argument."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(
            [
                "workloads",
                "run-api",
                "--matrix",
                str(tmp_path / "manifest.json"),
                "--output-dir",
                str(tmp_path / "results"),
                "--model-id",
                "z-ai/glm-5.3",
                "--api-key",
                API_KEY,
            ]
        )
    assert excinfo.value.code == 2


def test_an_endpoint_query_holding_the_key_is_refused_before_it_is_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A secret-shaped query key is refused, and nothing derived is written.

    Only query *keys* are recorded, but every query *value* is folded into
    the plan's config hash as its sha256. A redactor cannot undo a hash, so
    the pre-flight has to refuse before the plan is built rather than
    relying on the value being withheld.
    """
    monkeypatch.setenv(ENV_VAR, API_KEY)
    code = invoke(
        base_argv(
            tmp_path,
            "--provider",
            "leaky",
            "--endpoint",
            f"https://openrouter.ai/api/v1/chat/completions?token={API_KEY}",
            "--api-key-env",
            ENV_VAR,
            "--mode",
            DECODE_MODE_AUTOREGRESSIVE,
            "--dry-run",
        )
    )
    assert code == 1
    captured = capsys.readouterr()
    assert "looks like a credential" in captured.err
    assert API_KEY not in captured.out
    assert API_KEY not in captured.err


def test_an_innocuous_query_key_holding_the_key_is_also_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The key check does not depend on the query key looking suspicious.

    A dry run that green-lights what the real run refuses is worse than no
    pre-flight at all, so this must be blocked rather than ready.
    """
    monkeypatch.setenv(ENV_VAR, API_KEY)
    code = invoke(
        base_argv(
            tmp_path,
            "--provider",
            "azure-like",
            "--endpoint",
            f"https://openrouter.ai/api/v1/chat/completions?deployment={API_KEY}",
            "--api-key-env",
            ENV_VAR,
            "--mode",
            DECODE_MODE_AUTOREGRESSIVE,
            "--dry-run",
        )
    )
    assert code == 2
    captured = capsys.readouterr()
    assert API_KEY not in captured.out
    assert API_KEY not in captured.err

    written = (tmp_path / "results" / "api_request_plan.json").read_text(
        encoding="utf-8"
    )
    assert API_KEY not in written
    row = json.loads(written)["rows"][0]
    assert row["status"] == "blocked"
    assert row["api_binding_hash"] is None
    assert any("refusing to run" in blocker for blocker in row["blockers"])


def test_an_innocuous_endpoint_query_still_records_only_its_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(ENV_VAR, API_KEY)
    code = invoke(
        base_argv(
            tmp_path,
            "--provider",
            "azure-like",
            "--endpoint",
            "https://openrouter.ai/api/v1/chat/completions?deployment=prod-eu",
            "--api-key-env",
            ENV_VAR,
            "--mode",
            DECODE_MODE_AUTOREGRESSIVE,
            "--dry-run",
        )
    )
    assert code == 0
    plan = json.loads(capsys.readouterr().out)["rows"][0]["request_plan"]
    assert plan["endpoint_query_keys"] == ["deployment"]
    assert "prod-eu" not in json.dumps(plan)


def test_an_invalid_provider_extension_is_reported_not_raised(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``ProviderExtensions`` raises the collector's error type directly.

    It validates in its own constructor, before ``binding.validate()`` can
    translate the failure, and its message quotes the rejected value. An
    uncaught traceback here would print that value unscrubbed.
    """
    code = invoke(
        base_argv(
            tmp_path,
            "--profile",
            "openrouter",
            "--provider-request-id",
            "   ",
            "--dry-run",
        )
    )
    assert code == 1
    assert "Failed to configure API workload execution" in capsys.readouterr().err


def test_no_run_artifact_contains_the_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", API_KEY)
    monkeypatch.setattr(cli, "UrllibStreamingTransport", FakeTransport)
    FakeTransport.chunks = answer_stream(f"the key is {API_KEY}")

    assert (
        invoke(
            base_argv(
                tmp_path,
                "--profile",
                "openrouter",
                "--mode",
                DECODE_MODE_AUTOREGRESSIVE,
            )
        )
        == 0
    )
    for path in sorted((tmp_path / "results").rglob("*")):
        if path.is_file():
            assert API_KEY not in path.read_text(encoding="utf-8", errors="replace")


def test_manifest_is_never_mutated_by_a_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", API_KEY)
    monkeypatch.setattr(cli, "UrllibStreamingTransport", FakeTransport)
    matrix_path = build_matrix(tmp_path)
    before = matrix_path.read_text(encoding="utf-8")

    assert (
        invoke(
            [
                "workloads",
                "run-api",
                "--matrix",
                str(matrix_path),
                "--output-dir",
                str(tmp_path / "results"),
                "--model-id",
                "z-ai/glm-5.3",
                "--profile",
                "openrouter",
                "--mode",
                DECODE_MODE_AUTOREGRESSIVE,
            ]
        )
        == 0
    )
    assert matrix_path.read_text(encoding="utf-8") == before
    assert MatrixManifest.read_json(matrix_path).entries
