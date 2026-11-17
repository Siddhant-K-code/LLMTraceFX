"""``replay run`` against a local fake OpenAI-compatible server."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
from _load_fakes import API_KEY, API_KEY_ENV, ENVIRON, fake_openai_server
from _tokenizer_fakes import ByteLevel, fake_tokenizer

from llmtracefx.optimizer import cli
from llmtracefx.optimizer.collectors.openai_api import (
    APICollectionConfig,
    UrllibStreamingTransport,
)
from llmtracefx.optimizer.replay import cli as replay_cli
from llmtracefx.optimizer.replay.artifact import (
    REQUESTS_FILE,
    RESULT_FILE,
    SCHEDULE_FILE,
)
from llmtracefx.optimizer.replay.formats import parse_trace
from llmtracefx.optimizer.replay.run import (
    INCOMPLETE,
    SERVER_METRICS_FILE,
    ReplayOptions,
    ReplayRunError,
    check_target,
    dispatch_schedule,
    fidelity,
    read_result,
)
from llmtracefx.optimizer.replay.schedule import (
    Selection,
    build_schedule,
    read_schedule,
)

FIXTURES = Path(__file__).parent / "fixtures" / "replay"
MOONCAKE = str(FIXTURES / "mooncake_small.jsonl")
TARGET_OUTPUTS = [20, 15, 10, 30]


@pytest.fixture(autouse=True)
def _credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, API_KEY)


def _run(argv: list[str]) -> object:
    with pytest.raises(SystemExit) as info:
        cli.main(argv)
    return info.value.code


def _argv(endpoint: str, output_dir: Path, *extra: str) -> list[str]:
    return [
        "replay",
        "run",
        MOONCAKE,
        "--format",
        "mooncake",
        "--time-scale",
        "0.01",
        "--endpoint",
        endpoint,
        "--model-id",
        "test-model",
        "--api-key-env",
        API_KEY_ENV,
        "--output-dir",
        str(output_dir),
        "--max-dispatch-lag-s",
        "5",
        *extra,
    ]


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_faithful_run_records_every_request(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with fake_openai_server() as server:
        assert _run(_argv(server.endpoint, out)) == 0
        bodies = list(server.bodies)

    assert sorted(body["max_tokens"] for body in bodies) == sorted(TARGET_OUTPUTS)
    assert all("ignore_eos" not in body for body in bodies)
    assert all(body["stream_options"] == {"include_usage": True} for body in bodies)

    result = read_result(out / RESULT_FILE)
    assert result["fidelity"]["verdict"] == "faithful"
    assert result["counts"]["sent"] == 4
    assert result["counts"]["succeeded"] == 4
    assert result["counts"]["prompt_sha256_mismatches"] == 0
    assert result["stopped_by_operator"] is False
    assert result["tokens"]["cached_prompt_tokens"] == {
        "provenance": "provider_reported",
        "requests_reporting": 4,
        "total": 16,
    }
    assert result["tokens"]["input_vs_trace"]["requests_compared"] == 4
    assert "LOAD_TEST_KEY" in json.dumps(result["target"])
    assert API_KEY not in json.dumps(result)

    schedule = read_schedule(out / SCHEDULE_FILE)
    assert result["schedule"]["schedule_sha256"] == schedule["schedule_sha256"]
    planned = {row["request_id"]: row["prompt_sha256"] for row in schedule["requests"]}
    rows = _lines(out / REQUESTS_FILE)
    assert len(rows) == 4
    for row in rows:
        assert row["sent"] is True
        assert row["prompt_sha256"] == "sha256:" + planned[row["request_id"]]
        assert row["planned_prompt_sha256"] == planned[row["request_id"]]
        assert row["dispatch_lag_s"] >= 0

    # Prompts are rebuilt on demand and never written to disk.
    prompts = {body["messages"][-1]["content"] for body in bodies}
    for path in out.iterdir():
        text = path.read_text()
        assert not any(prompt in text for prompt in prompts)


def test_ignore_eos_is_sent_only_when_asked(tmp_path: Path) -> None:
    with fake_openai_server() as server:
        argv = _argv(server.endpoint, tmp_path / "out", "--ignore-eos")
        assert _run([*argv, "--no-stream-usage"]) == 0
        bodies = list(server.bodies)
    assert all(body["ignore_eos"] is True for body in bodies)
    assert all("stream_options" not in body for body in bodies)
    result = read_result(tmp_path / "out" / RESULT_FILE)
    assert result["options"]["ignore_eos"] is True


def test_a_tight_lag_limit_gives_lagged_and_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    with fake_openai_server() as server:
        argv = _argv(server.endpoint, out)
        argv[argv.index("--max-dispatch-lag-s") + 1] = "0"
        assert _run(argv) == 2
    assert read_result(out / RESULT_FILE)["fidelity"]["verdict"] == "lagged"
    assert "fidelity: lagged" in capsys.readouterr().out


def test_missing_usage_stays_null(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with fake_openai_server() as server:
        server.behaviour.usage = None
        assert _run(_argv(server.endpoint, out)) == 0
    result = read_result(out / RESULT_FILE)
    assert result["tokens"]["input_vs_trace"]["requests_compared"] == 0
    assert result["tokens"]["input_vs_trace"]["requests_without_provider_count"] == 4
    assert result["tokens"]["cached_prompt_tokens"]["total"] is None
    assert all(row["prompt_tokens"] is None for row in _lines(out / REQUESTS_FILE))


def test_all_requests_failing_exits_1(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with fake_openai_server() as server:
        server.behaviour.status_code = 500
        assert _run(_argv(server.endpoint, out)) == 1
    result = read_result(out / RESULT_FILE)
    assert result["counts"]["succeeded"] == 0
    assert sum(result["counts"]["failed_by_category"].values()) == 4
    assert all(row["http_status"] == 500 for row in _lines(out / REQUESTS_FILE))


def test_server_metrics_bracket_the_replay(tmp_path: Path) -> None:
    out = tmp_path / "out"
    labels = 'engine="0",model_name="test-model"'
    with fake_openai_server() as server:
        server.behaviour.metrics_text = (
            "# TYPE vllm:num_requests_running gauge\n"
            f"vllm:num_requests_running{{{labels}}} 0\n"
            "# TYPE vllm:e2e_request_latency_seconds histogram\n"
            f'vllm:e2e_request_latency_seconds_bucket{{{labels},le="+Inf"}} @COMPLETED@\n'
            f"vllm:e2e_request_latency_seconds_count{{{labels}}} @COMPLETED@\n"
            f"vllm:e2e_request_latency_seconds_sum{{{labels}}} @COMPLETED@\n"
        )
        argv = _argv(server.endpoint, out, "--server-metrics-url", server.metrics_url)
        assert _run([*argv, "--server-metrics-runtime", "vllm"]) == 0
    result = read_result(out / RESULT_FILE)
    record = result["server_metrics"]
    assert record["file"] == SERVER_METRICS_FILE
    evidence = json.loads((out / SERVER_METRICS_FILE).read_text())
    assert record["attribution"] == evidence["attribution"]


def test_remote_endpoint_needs_explicit_opt_in(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    argv = _argv("https://api.example.com/v1/chat/completions", out)
    assert _run(argv) == 1
    assert "--allow-remote-load" in capsys.readouterr().err
    assert not out.exists()
    check_target(
        "https://api.example.com/v1/chat/completions",
        ReplayOptions(allow_remote_load=True),
    )
    with pytest.raises(ReplayRunError):
        check_target("https://api.example.com/v1", ReplayOptions())


def test_missing_credential_fails_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(API_KEY_ENV)
    out = tmp_path / "out"
    with fake_openai_server() as server:
        assert _run(_argv(server.endpoint, out)) == 1
        assert server.bodies == []
    assert not out.exists()


def test_earlier_replay_evidence_is_never_overwritten(tmp_path: Path) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / RESULT_FILE).write_text("earlier")
    with fake_openai_server() as server:
        assert _run(_argv(server.endpoint, out)) == 1
        assert server.bodies == []
    assert sorted(path.name for path in out.iterdir()) == [RESULT_FILE]
    assert (out / RESULT_FILE).read_text() == "earlier"


def test_a_stop_before_dispatch_is_incomplete(tmp_path: Path) -> None:
    trace = parse_trace(Path(MOONCAKE), "mooncake", block_size=512)
    schedule = build_schedule(trace, Selection(time_scale=0.01), salt="a" * 32)
    stop = threading.Event()
    stop.set()
    with fake_openai_server() as server:
        base = APICollectionConfig(
            run_id="replay",
            provider="openai-compatible",
            endpoint=server.endpoint,
            model_id="test-model",
            prompt="replay",
            output_dir=tmp_path,
            command_argv=("llmtracefx-optimizer", "replay", "run"),
            credential_env_var=API_KEY_ENV,
        )
        replay = dispatch_schedule(
            schedule,
            base,
            ReplayOptions(),
            transport=UrllibStreamingTransport(),
            environ=ENVIRON,
            stop=stop,
        )
    assert replay.report.unsent_count == 4
    assert server.bodies == []
    assert fidelity(replay.report, ReplayOptions())["verdict"] == INCOMPLETE


def test_a_tokenizer_run_reports_exact_counts_and_overhead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        replay_cli,
        "load_tokenizer",
        lambda spec, revision=None: fake_tokenizer(ByteLevel(), name=spec),
    )
    out = tmp_path / "out"
    with fake_openai_server() as server:
        argv = _argv(server.endpoint, out, "--tokenizer", "org/model")
        assert _run(argv) == 0

    rows = _lines(out / REQUESTS_FILE)
    assert sorted(row["tokenizer_prompt_tokens"] for row in rows) == [
        600,
        1024,
        1100,
        1300,
    ]
    result = read_result(out / RESULT_FILE)
    overhead = result["tokens"]["input_minus_tokenizer"]
    # The fake server always reports 11 prompt tokens.
    assert overhead["requests_compared"] == 4
    assert overhead["min"] == 11 - 1300
    assert overhead["max"] == 11 - 600
    assert overhead["provenance"] == "derived"
    assert "--tokenizer" in result["target"]["command"]
    assert result["field_provenance"]["tokenizer_prompt_tokens"] == "tokenizer_exact"


def test_a_plain_run_has_no_tokenizer_overhead(tmp_path: Path) -> None:
    out = tmp_path / "out"
    with fake_openai_server() as server:
        assert _run(_argv(server.endpoint, out)) == 0
    result = read_result(out / RESULT_FILE)
    assert result["tokens"]["input_minus_tokenizer"] is None
    assert all(
        row["tokenizer_prompt_tokens"] is None for row in _lines(out / REQUESTS_FILE)
    )
