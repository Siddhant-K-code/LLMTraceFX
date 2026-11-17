"""sweep-llama-cpp against a fake native llama-server. Nothing leaves the machine."""

from __future__ import annotations

import contextlib
import io
import json
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer import cli
from llmtracefx.optimizer.collectors.openai_api import UrllibStreamingTransport
from llmtracefx.optimizer.load.api_worker import ApiRequestSpec
from llmtracefx.optimizer.load.artifact import SWEEP_KIND, SWEEP_PLAN_KIND
from llmtracefx.optimizer.load.dispatcher import ClosedLoop, dispatch
from llmtracefx.optimizer.load.llama_cpp_worker import (
    LlamaCppConfig,
    LlamaCppLoadError,
    make_llama_cpp_sender,
)

SERVER_PROMPT_ECHO = "detokenized prompt echoed by the server"
MODEL_PATH = "/models/private/secret-model.gguf"
KEY_ENV = "LLAMA_TEST_KEY"
KEY = "llama-test-key-not-a-real-credential"


@dataclass
class Behaviour:
    total_slots: Any = 2
    props_status: int = 200
    pieces: tuple[str, ...] = ("Hel", "lo", " you")
    send_tokens: bool = True
    status_code: int = 200
    stop_type: str = "limit"
    truncated: bool = False
    mode: str = "ok"
    """ok, error_payload, truncate, bad_json"""
    delay_s: float = 0.005
    seen_auth: list[str | None] = field(default_factory=list)


class FakeLlamaServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.behaviour = Behaviour()
        self.bodies: list[dict[str, Any]] = []
        self.props_calls = 0
        self.active = 0
        self.peak_active = 0
        self.lock = threading.Lock()

    @property
    def root(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


def _sse(payload: Any) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: FakeLlamaServer

    def log_message(self, *args: Any) -> None:
        pass

    def _json(self, status: int, payload: Any) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        behaviour = self.server.behaviour
        if self.path != "/props":
            self.send_error(404)
            return
        with self.server.lock:
            self.server.props_calls += 1
        if behaviour.props_status != 200:
            self._json(behaviour.props_status, {"error": {"message": "nope"}})
            return
        self._json(
            200,
            {
                "total_slots": behaviour.total_slots,
                "build_info": "b9999-ebc9fb93",
                "model_path": MODEL_PATH,
                "default_generation_settings": {"n_ctx": 4096},
            },
        )

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        server = self.server
        behaviour = server.behaviour
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        behaviour.seen_auth.append(self.headers.get("Authorization"))
        if self.path != "/completion":
            self.send_error(404)
            return
        with server.lock:
            server.bodies.append(body)
            slot = server.active
            server.active += 1
            server.peak_active = max(server.peak_active, server.active)
        try:
            if behaviour.status_code != 200:
                self._json(behaviour.status_code, {"error": {"message": "busy"}})
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(b":\n\n")
            if behaviour.mode == "bad_json":
                self.wfile.write(b"data: {not json\n\n")
                return
            for number, piece in enumerate(behaviour.pieces, start=1):
                time.sleep(behaviour.delay_s)
                self.wfile.write(
                    _sse(
                        {
                            "index": 0,
                            "content": piece,
                            "tokens": [100 + number] if behaviour.send_tokens else [],
                            "stop": False,
                            "id_slot": slot,
                            "tokens_predicted": number,
                            "tokens_evaluated": 5,
                        }
                    )
                )
                self.wfile.flush()
            if behaviour.mode == "truncate":
                return
            if behaviour.mode == "error_payload":
                self.wfile.write(
                    _sse({"error": {"code": 500, "message": "slot crashed"}})
                )
                return
            self.wfile.write(
                _sse(
                    {
                        "index": 0,
                        "content": "",
                        "tokens": [],
                        "id_slot": slot,
                        "stop": True,
                        "model": MODEL_PATH,
                        "tokens_predicted": len(behaviour.pieces),
                        "tokens_evaluated": 5,
                        "prompt": SERVER_PROMPT_ECHO,
                        "truncated": behaviour.truncated,
                        "stop_type": behaviour.stop_type,
                        "stopping_word": "",
                        "tokens_cached": 7,
                        "timings": {
                            "cache_n": 1,
                            "prompt_n": 4,
                            "prompt_ms": 12.5,
                            "predicted_n": len(behaviour.pieces),
                            "predicted_ms": 30.0,
                        },
                    }
                )
            )
        finally:
            with server.lock:
                server.active -= 1


@pytest.fixture
def server() -> Iterator[FakeLlamaServer]:
    fake = FakeLlamaServer()
    thread = threading.Thread(target=fake.serve_forever, daemon=True)
    thread.start()
    try:
        yield fake
    finally:
        fake.shutdown()
        fake.server_close()


def run_main(argv: list[str]) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            cli.main(argv)
    except SystemExit as exit_error:
        code = int(exit_error.code or 0)
    return code, stdout.getvalue(), stderr.getvalue()


def sweep_argv(tmp_path: Path, endpoint: str, *extra: str) -> list[str]:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Once upon a time", encoding="utf-8")
    return [
        "sweep-llama-cpp",
        "--run-id",
        "llama-load",
        "--endpoint",
        endpoint,
        "--output-dir",
        str(tmp_path / "out"),
        "--prompt-file",
        str(prompt_file),
        "--max-output-tokens",
        "3",
        "--requests-per-point",
        "4",
        *extra,
    ]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_sweep_records_server_accounting_and_goodput(
    tmp_path: Path, server: FakeLlamaServer
) -> None:
    server.behaviour.delay_s = 0.03
    code, out, err = run_main(
        sweep_argv(
            tmp_path,
            server.root,
            "--concurrency",
            "1,2",
            "--slo-ttft-ms",
            "5000",
            "--slo-e2e-ms",
            "10000",
            "--max-dispatch-lag-ms",
            "1000",
        )
    )
    assert code == 0, err
    assert "concurrency=2: 4/4 succeeded" in out
    assert server.props_calls == 1
    assert server.peak_active == 2
    assert len(server.bodies) == 8
    assert server.bodies[0] == {
        "prompt": "Once upon a time",
        "n_predict": 3,
        "stream": True,
        "return_tokens": True,
        "cache_prompt": False,
    }
    assert server.behaviour.seen_auth == [None] * 8

    text = (tmp_path / "out" / "sweep.json").read_text(encoding="utf-8")
    assert SERVER_PROMPT_ECHO not in text
    assert MODEL_PATH not in text
    assert "Once upon a time" not in text
    sweep = json.loads(text)
    assert sweep["kind"] == SWEEP_KIND
    assert sweep["summary"]["clean"] is True
    backend = sweep["attribution"]["backend"]
    assert backend["name"] == "llama.cpp"
    assert backend["server"]["total_slots"] == 2
    assert backend["server"]["provenance"] == "provider_reported"
    assert backend["server"]["build_info"] == "b9999-ebc9fb93"
    assert backend["max_supported_in_flight"] == 2
    assert backend["server_queueing_allowed"] is False
    assert backend["upstream_reference"]["commit"] == (
        "ebc9fb933e774bb4ac0ba497a67948e16b7e5656"
    )
    assert sweep["field_provenance"]["server_prompt_ms"] == "measured_native"
    assert sweep["field_provenance"]["output_tokens"] == "provider_reported"
    assert any("chat template" in item for item in sweep["attribution"]["cannot_claim"])
    assert sweep["request_plan"]["command_argv"][1] == "sweep-llama-cpp"

    point = sweep["points"][1]
    assert point["scheduling"]["max_observed_in_flight"] == 2
    assert point["latency"]["tpot_ms"]["count"] == 4
    assert point["goodput"]["request_goodput_fraction"]["value"] == 1.0
    row = point["requests"][0]
    assert row["success"] is True
    assert row["http_status"] == 200
    assert row["ttft_source"] == "first_token_chunk"
    assert row["output_tokens"] == 3
    assert row["prompt_tokens"] == 5
    assert row["cached_prompt_tokens"] == 1
    assert row["content_delta_count"] == 3
    assert row["finish_reason"] == "limit"
    assert row["finish_reason_code"] == "length"
    assert row["truncated"] is False
    assert row["server_prompt_n"] == 4
    assert row["server_prompt_ms"] == 12.5
    assert row["server_predicted_ms"] == 30.0
    assert row["tpot_status"] == "derived"
    assert row["ttft_ms"] <= row["first_content_ms"] <= row["e2e_ms"]


def test_load_above_total_slots_is_refused_before_sending(
    tmp_path: Path, server: FakeLlamaServer
) -> None:
    code, _, err = run_main(sweep_argv(tmp_path, server.root, "--concurrency", "1,4"))
    assert code == 1
    # Numbers that also appear in argv are scrubbed from diagnostics.
    assert "requests in flight but the server has 2 slots" in err
    assert "--allow-server-queue" in err
    assert server.bodies == []
    assert not (tmp_path / "out" / "sweep.json").exists()


def test_rate_mode_needs_in_flight_cap_within_slots(
    tmp_path: Path, server: FakeLlamaServer
) -> None:
    rate = ("--request-rate", "50", "--arrival", "constant")
    code, _, err = run_main(sweep_argv(tmp_path, server.root, *rate))
    assert code == 1
    assert "256 requests in flight" in err
    assert "--max-in-flight" in err

    code, _, err = run_main(
        sweep_argv(
            tmp_path,
            server.root,
            *rate,
            "--max-in-flight",
            "2",
            "--max-dispatch-lag-ms",
            "100000",
        )
    )
    assert code == 0, err
    assert server.peak_active <= 2


def test_allow_server_queue_records_the_choice(
    tmp_path: Path, server: FakeLlamaServer
) -> None:
    server.behaviour.total_slots = 1
    code, _, err = run_main(
        sweep_argv(
            tmp_path,
            server.root,
            "--concurrency",
            "2",
            "--allow-server-queue",
            "--max-dispatch-lag-ms",
            "1000",
        )
    )
    assert code == 0, err
    backend = read_json(tmp_path / "out" / "sweep.json")["attribution"]["backend"]
    assert backend["server_queueing_allowed"] is True
    assert backend["max_supported_in_flight"] == 1


@pytest.mark.parametrize(
    ("props", "message"),
    [
        ({"total_slots": None}, "no positive integer total_slots"),
        ({"total_slots": 0}, "no positive integer total_slots"),
        ({"props_status": 404}, "returned HTTP 404"),
    ],
)
def test_props_probe_failure_refuses_the_sweep(
    tmp_path: Path, server: FakeLlamaServer, props: dict[str, Any], message: str
) -> None:
    for key, value in props.items():
        setattr(server.behaviour, key, value)
    code, _, err = run_main(sweep_argv(tmp_path, server.root, "--concurrency", "1"))
    assert code == 1
    assert message in err
    assert server.bodies == []


def test_unreachable_server_is_refused(tmp_path: Path) -> None:
    code, _, err = run_main(
        sweep_argv(tmp_path, "http://127.0.0.1:9", "--concurrency", "1")
    )
    assert code == 1
    assert "GET /props failed" in err


def test_dry_run_probes_and_sends_nothing(
    tmp_path: Path, server: FakeLlamaServer
) -> None:
    code, out, err = run_main(
        sweep_argv(tmp_path, server.root, "--concurrency", "1,8", "--dry-run")
    )
    assert code == 0, err
    assert "nothing sent" in out
    assert server.props_calls == 0
    assert server.bodies == []
    plan = read_json(tmp_path / "out" / "sweep_plan.json")
    assert plan["kind"] == SWEEP_PLAN_KIND
    assert plan["network_request_performed"] is False
    backend = plan["attribution"]["backend"]
    assert backend["server"]["status"] == "not probed"
    assert backend["max_supported_in_flight"] is None
    assert plan["request_plan"]["body_without_prompt"]["n_predict"] == 3
    assert "prompt" not in plan["request_plan"]["body_without_prompt"]


def test_remote_server_needs_allow_remote_load(tmp_path: Path) -> None:
    remote = "https://llama.example.invalid"
    code, _, err = run_main(sweep_argv(tmp_path, remote, "--concurrency", "1"))
    assert code == 1
    assert "--allow-remote-load" in err

    code, _, err = run_main(
        sweep_argv(
            tmp_path, remote, "--concurrency", "1", "--allow-remote-load", "--dry-run"
        )
    )
    assert code == 0, err

    code, _, err = run_main(
        sweep_argv(
            tmp_path,
            "http://llama.example.invalid",
            "--concurrency",
            "1",
            "--allow-remote-load",
            "--dry-run",
        )
    )
    assert code == 1
    assert "https" in err


def test_completion_url_and_query_are_refused(tmp_path: Path) -> None:
    code, _, err = run_main(
        sweep_argv(tmp_path, "http://127.0.0.1:8080/completion", "--concurrency", "1")
    )
    assert code == 1
    assert "server root" in err
    code, _, err = run_main(
        sweep_argv(tmp_path, "http://127.0.0.1:8080/?x=1", "--concurrency", "1")
    )
    assert code == 1
    assert "query string" in err


def test_api_key_is_sent_and_never_written(
    tmp_path: Path, server: FakeLlamaServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_ENV, KEY)
    code, _, err = run_main(
        sweep_argv(
            tmp_path,
            server.root,
            "--concurrency",
            "1",
            "--api-key-env",
            KEY_ENV,
            "--temperature",
            "0.5",
            "--seed",
            "3",
            "--cache-prompt",
        )
    )
    assert code == 0, err
    assert server.behaviour.seen_auth == [f"Bearer {KEY}"] * 4
    body = server.bodies[0]
    assert body["temperature"] == 0.5
    assert body["seed"] == 3
    assert body["cache_prompt"] is True
    assert "top_k" not in body
    sweep = read_json(tmp_path / "out" / "sweep.json")
    assert KEY not in json.dumps(sweep)
    assert sweep["attribution"]["backend"]["cache_prompt"] is True


def test_missing_api_key_variable_is_refused(
    tmp_path: Path, server: FakeLlamaServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)
    code, _, err = run_main(
        sweep_argv(
            tmp_path, server.root, "--concurrency", "1", "--api-key-env", KEY_ENV
        )
    )
    assert code == 1
    assert "not set" in err
    assert server.props_calls == 0


def _one_row(server: FakeLlamaServer) -> dict[str, Any]:
    config = LlamaCppConfig(run_id="r", base_url=server.root, n_predict=3)
    send = make_llama_cpp_sender(
        config, transport=UrllibStreamingTransport(), environ={}
    )
    report = dispatch([ApiRequestSpec(prompt="x")], send, ClosedLoop(1))
    result = report.outcomes[0].result
    assert result is not None
    return result.to_dict()


@pytest.mark.parametrize(
    ("change", "category"),
    [
        ({"mode": "error_payload"}, "provider_error_payload"),
        ({"mode": "truncate"}, "stream_truncated"),
        ({"mode": "bad_json"}, "stream_decode"),
        ({"status_code": 503}, "http_status"),
        ({"stop_type": "none"}, "unrecognized_stop_type"),
    ],
)
def test_failure_categories(
    server: FakeLlamaServer, change: dict[str, Any], category: str
) -> None:
    for key, value in change.items():
        setattr(server.behaviour, key, value)
    row = _one_row(server)
    assert row["success"] is False
    assert row["failure_category"] == category
    if category != "unrecognized_stop_type":
        assert row["output_tokens"] is None


def test_connection_failure_category() -> None:
    config = LlamaCppConfig(run_id="r", base_url="http://127.0.0.1:9", n_predict=3)
    send = make_llama_cpp_sender(
        config, transport=UrllibStreamingTransport(), environ={}
    )
    report = dispatch([ApiRequestSpec(prompt="x")], send, ClosedLoop(1))
    result = report.outcomes[0].result
    assert result is not None
    assert result.failure_category == "connection"
    assert result.http_status is None


def test_ttft_falls_back_to_first_content_without_tokens(
    server: FakeLlamaServer,
) -> None:
    server.behaviour.send_tokens = False
    row = _one_row(server)
    assert row["success"] is True
    assert row["ttft_source"] == "first_content_chunk"
    assert row["ttft_ms"] == row["first_content_ms"]


@pytest.mark.parametrize(
    ("stop_type", "truncated", "code"),
    [("eos", False, "stop"), ("word", False, "stop"), ("limit", True, "length")],
)
def test_stop_types_map_to_finish_codes(
    server: FakeLlamaServer, stop_type: str, truncated: bool, code: str
) -> None:
    server.behaviour.stop_type = stop_type
    server.behaviour.truncated = truncated
    row = _one_row(server)
    assert row["success"] is True
    assert row["finish_reason"] == stop_type
    assert row["finish_reason_code"] == code
    assert row["truncated"] is truncated


def test_specs_with_chat_fields_are_refused(server: FakeLlamaServer) -> None:
    config = LlamaCppConfig(run_id="r", base_url=server.root, n_predict=3)
    send = make_llama_cpp_sender(
        config, transport=UrllibStreamingTransport(), environ={}
    )
    report = dispatch(
        [ApiRequestSpec(prompt="x", system_prompt="be brief")], send, ClosedLoop(1)
    )
    assert report.outcomes[0].client_error == "LlamaCppLoadError"
    assert server.bodies == []


def test_n_predict_is_required_positive() -> None:
    with pytest.raises(LlamaCppLoadError, match="positive integer"):
        LlamaCppConfig(run_id="r", base_url="http://127.0.0.1:8080", n_predict=0)
