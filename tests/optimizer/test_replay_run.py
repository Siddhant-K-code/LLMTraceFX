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


def test_trtllm_metrics_opt_in_is_sent_and_recorded(tmp_path: Path) -> None:
    trailer = (
        b"event: trtllm.perf_metrics\n"
        b'data: {"Server-Timing": "server_queue;dur=2, server_ttft;dur=30"}\n\n'
    )
    with fake_openai_server() as server:
        assert _run(_argv(server.endpoint, tmp_path / "plain")) == 0
        server.behaviour.after_done = trailer
        argv = _argv(server.endpoint, tmp_path / "out", "--trtllm-return-metrics")
        assert _run(argv) == 0
        headers = [{k.lower(): v for k, v in h.items()} for h in server.request_headers]
    assert [h.get("x-trtllm-return-metrics") for h in headers] == [None] * 4 + ["1"] * 4

    plain = read_result(tmp_path / "plain" / RESULT_FILE)
    assert "trtllm_return_metrics" not in plain["target"]
    result = read_result(tmp_path / "out" / RESULT_FILE)
    assert result["target"]["trtllm_return_metrics"] is True
    assert "X-TRTLLM-return-metrics" in result["target"]["header_names"]
    assert "--trtllm-return-metrics" in result["target"]["command"]
    rows = _lines(tmp_path / "out" / REQUESTS_FILE)
    assert all(row["server_metrics_source"] == "trtllm_perf_metrics" for row in rows)
    assert all(row["server_queue_ms"] == 2.0 for row in rows)
    assert all(row["server_ttft_ms"] == 30.0 for row in rows)


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


MULTI = str(FIXTURES / "aiperf_multi_turn.jsonl")


def test_multi_turn_run_sends_history_after_each_turn_completes(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    with fake_openai_server() as server:
        argv = _argv(server.endpoint, out)
        argv[2:5] = [MULTI, "--format", "aiperf-multi-turn"]
        assert _run(argv) == 0
        bodies = list(server.bodies)

    roles = sorted(
        (tuple(m["role"] for m in body["messages"]) for body in bodies), key=len
    )
    assert roles[-1] == ("user", "assistant", "user", "assistant", "user")
    longest = max(bodies, key=lambda body: len(body["messages"]))
    assert [m["content"] for m in longest["messages"][::2]] == [
        "Plan a short trip.",
        "Add a budget.",
        "Make it shorter.",
    ]
    lines = _lines(out / REQUESTS_FILE)
    trip = [line for line in lines if line["session"] == 0]
    assert [line["turn"] for line in trip] == [0, 1, 2]
    assert trip[1]["dispatched_offset_s"] >= trip[0]["completed_offset_s"]
    assert trip[2]["dispatched_offset_s"] >= trip[1]["completed_offset_s"]
    result = read_result(out / RESULT_FILE)
    assert result["counts"]["prompt_sha256_mismatches"] == 0
    assert result["sessions"]["sessions"] == 3
    assert result["sessions"]["all_turns_succeeded"] == 3
    assert result["dispatch"]["policy"]["kind"] == "session_arrivals"
    for name in (REQUESTS_FILE, RESULT_FILE, SCHEDULE_FILE):
        assert "Make it shorter" not in (out / name).read_text()


def test_multi_turn_run_with_a_tokenizer_reports_exact_counts(
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
        argv[2:5] = [MULTI, "--format", "aiperf-multi-turn"]
        assert _run(argv) == 0
    result = read_result(out / RESULT_FILE)
    assert result["counts"]["prompt_sha256_mismatches"] == 0
    assert result["sessions"]["all_turns_succeeded"] == 3
    rows = _lines(out / REQUESTS_FILE)
    assert all(row["tokenizer_prompt_tokens"] for row in rows)
    assert result["tokens"]["input_minus_tokenizer"]["requests_compared"] == 5


def test_multi_turn_session_continues_after_a_failed_turn(tmp_path: Path) -> None:
    trace = parse_trace(MULTI, "aiperf-multi-turn")
    schedule = build_schedule(trace, Selection(max_sessions=1), salt="s")
    with fake_openai_server() as server:
        server.behaviour.fail_every = 1
        base = APICollectionConfig(
            run_id="replay",
            provider="openai-compatible",
            endpoint=server.endpoint,
            model_id="test-model",
            prompt="unused",
            output_dir=tmp_path,
            command_argv=("llmtracefx-optimizer",),
            credential_env_var=API_KEY_ENV,
        )
        replay = dispatch_schedule(
            schedule,
            base,
            ReplayOptions(),
            transport=UrllibStreamingTransport(),
            environ=ENVIRON,
        )
    assert [row.sent for row in replay.rows] == [True, True, True]
    assert not any(row.success for row in replay.rows)


WEKA = str(FIXTURES / "weka_small.jsonl")


def test_weka_run_sends_one_prompt_per_request_in_conversation_order(
    tmp_path: Path,
) -> None:
    out = tmp_path / "out"
    with fake_openai_server() as server:
        argv = _argv(server.endpoint, out)
        argv[2:5] = [WEKA, "--format", "weka"]
        assert _run(argv) == 0
        bodies = list(server.bodies)

    assert len(bodies) == 5
    assert {tuple(m["role"] for m in body["messages"]) for body in bodies} == {
        ("user",)
    }
    lines = _lines(out / REQUESTS_FILE)
    first = sorted(
        (line for line in lines if line["session"] == 0), key=lambda r: r["turn"]
    )
    # At time scale 0.01 the 0.5 s gap after request 0 ends is 0.005 s; the
    # overlapping request 2 waits only for its own start, 1.8 s scaled.
    assert first[1]["dispatched_offset_s"] >= first[0]["completed_offset_s"] + 0.005
    assert first[2]["dispatched_offset_s"] >= 0.018
    result = read_result(out / RESULT_FILE)
    assert result["counts"]["prompt_sha256_mismatches"] == 0
    assert result["sessions"]["sessions"] == 2
    assert result["dispatch"]["policy"]["kind"] == "session_arrivals"
    assert "planned prompt" in result["sessions"]["note"]
    for name in (REQUESTS_FILE, RESULT_FILE, SCHEDULE_FILE):
        assert "conv-secret" not in (out / name).read_text()
