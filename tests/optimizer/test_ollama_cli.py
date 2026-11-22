"""collect-ollama and sweep-ollama against a fake Ollama server.

Nothing leaves the machine.
"""

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
from llmtracefx.optimizer.collectors.ollama import OllamaSettings
from llmtracefx.optimizer.collectors.openai_api import UrllibStreamingTransport
from llmtracefx.optimizer.load.api_worker import ApiRequestSpec
from llmtracefx.optimizer.load.artifact import SWEEP_KIND, SWEEP_PLAN_KIND
from llmtracefx.optimizer.load.dispatcher import ClosedLoop, dispatch
from llmtracefx.optimizer.load.ollama_worker import make_ollama_sender

KEY_ENV = "OLLAMA_TEST_KEY"
KEY = "ollama-test-key-not-a-real-credential"
PROMPT = "Once upon a time"


@dataclass
class Behaviour:
    pieces: tuple[str, ...] = ("Hel", "lo", " you")
    version_status: int = 200
    status_code: int = 200
    mode: str = "ok"
    """ok, error_line, truncate"""
    delay_s: float = 0.005
    cached: int | None = 2
    seen_auth: list[str | None] = field(default_factory=list)


class FakeOllama(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.behaviour = Behaviour()
        self.bodies: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self.version_calls = 0
        self.lock = threading.Lock()

    @property
    def root(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: FakeOllama

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
        if self.path != "/api/version":
            self.send_error(404)
            return
        with self.server.lock:
            self.server.version_calls += 1
        status = self.server.behaviour.version_status
        self._json(status, {"version": "0.12.3"} if status == 200 else {})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        server = self.server
        behaviour = server.behaviour
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        behaviour.seen_auth.append(self.headers.get("Authorization"))
        if self.path not in ("/api/chat", "/api/generate"):
            self.send_error(404)
            return
        with server.lock:
            server.bodies.append(body)
            server.paths.append(self.path)
        if behaviour.status_code != 200:
            self._json(behaviour.status_code, {"error": "model not found"})
            return
        chat = self.path == "/api/chat"
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def chunk(text: str, **extra: Any) -> bytes:
            payload: dict[str, Any] = {"model": body["model"], **extra}
            if chat:
                payload["message"] = {"role": "assistant", "content": text}
            else:
                payload["response"] = text
            return json.dumps(payload).encode() + b"\n"

        for piece in behaviour.pieces:
            time.sleep(behaviour.delay_s)
            self.wfile.write(chunk(piece, done=False))
            self.wfile.flush()
        if behaviour.mode == "truncate":
            return
        if behaviour.mode == "error_line":
            self.wfile.write(b'{"error":"runner crashed"}\n')
            return
        metrics: dict[str, Any] = {
            "total_duration": 90_000_000,
            "load_duration": 1_500_000,
            "prompt_eval_count": 6,
            "prompt_eval_duration": 8_000_000,
            "eval_count": len(behaviour.pieces),
            "eval_duration": 30_000_000,
        }
        if behaviour.cached is not None:
            metrics["prompt_eval_cached_count"] = behaviour.cached
        extra = {} if chat else {"context": [1, 2, 3]}
        self.wfile.write(chunk("", done=True, done_reason="length", **metrics, **extra))


@pytest.fixture
def server() -> Iterator[FakeOllama]:
    fake = FakeOllama()
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
            result = cli.main(argv)
            code = int(result or 0)
    except SystemExit as exit_error:
        code = int(exit_error.code or 0)
    return code, stdout.getvalue(), stderr.getvalue()


def prompt_file(tmp_path: Path) -> Path:
    path = tmp_path / "prompt.txt"
    path.write_text(PROMPT, encoding="utf-8")
    return path


def collect_argv(tmp_path: Path, endpoint: str, *extra: str) -> list[str]:
    return [
        "collect-ollama",
        "--run-id",
        "ollama-one",
        "--endpoint",
        endpoint,
        "--model",
        "qwen3:0.6b",
        "--prompt-file",
        str(prompt_file(tmp_path)),
        "--output-dir",
        str(tmp_path / "out"),
        "--max-output-tokens",
        "3",
        *extra,
    ]


def sweep_argv(tmp_path: Path, endpoint: str, *extra: str) -> list[str]:
    return [
        "sweep-ollama",
        "--run-id",
        "ollama-load",
        "--endpoint",
        endpoint,
        "--model",
        "qwen3:0.6b",
        "--output-dir",
        str(tmp_path / "out"),
        "--prompt-file",
        str(prompt_file(tmp_path)),
        "--max-output-tokens",
        "3",
        "--requests-per-point",
        "4",
        *extra,
    ]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_collect_ollama_writes_evidence(tmp_path: Path, server: FakeOllama) -> None:
    code, out, err = run_main(
        collect_argv(tmp_path, server.root, "--api", "generate", "--seed", "4")
    )
    assert code == 0, err
    assert "record.json" in out
    assert server.paths == ["/api/generate"]
    assert server.bodies[0] == {
        "model": "qwen3:0.6b",
        "stream": True,
        "prompt": PROMPT,
        "options": {"num_predict": 3, "seed": 4},
    }
    record = read_json(tmp_path / "out" / "record.json")
    assert record["runtime"]["name"] == "ollama"
    assert record["runtime"]["version"] == "0.12.3"
    assert record["timing"]["model_load"]["value"] == 1.5
    assert record["timing"]["prefill"]["value"] == 8.0
    assert record["command"]["argv"][1] == "collect-ollama"
    evidence = read_json(tmp_path / "out" / "ollama_evidence.json")
    assert evidence["response"]["finish_reason_code"] == "length"
    assert evidence["server_reported"]["raw"]["prompt_eval_cached_count"] == 2
    assert PROMPT not in json.dumps(evidence)


def test_collect_ollama_reports_a_failed_request(
    tmp_path: Path, server: FakeOllama
) -> None:
    server.behaviour.status_code = 404
    code, _, err = run_main(collect_argv(tmp_path, server.root))
    assert code == 1
    assert "http_status" in err
    assert "model not found" in err
    record = read_json(tmp_path / "out" / "record.json")
    assert record["outcome"]["success"] is False


def test_collect_ollama_dry_run_sends_nothing(
    tmp_path: Path, server: FakeOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_ENV, KEY)
    code, _, err = run_main(
        collect_argv(
            tmp_path,
            server.root,
            "--dry-run",
            "--api-key-env",
            KEY_ENV,
            "--think",
            "true",
        )
    )
    assert code == 0, err
    assert server.version_calls == 0
    assert server.bodies == []
    plan = read_json(tmp_path / "out" / "request_plan.json")
    assert plan["network_request_performed"] is False
    assert plan["credential_env_var_present"] is True
    assert plan["plan"]["body_without_prompt"]["think"] is True
    assert "messages" not in plan["plan"]["body_without_prompt"]
    assert KEY not in json.dumps(plan)


def test_sweep_records_server_timers_and_goodput(
    tmp_path: Path, server: FakeOllama, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(KEY_ENV, KEY)
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
            "--api-key-env",
            KEY_ENV,
        )
    )
    assert code == 0, err
    assert "concurrency=2: 4/4 succeeded" in out
    assert server.version_calls == 1
    assert len(server.bodies) == 8
    assert server.behaviour.seen_auth == [f"Bearer {KEY}"] * 8

    text = (tmp_path / "out" / "sweep.json").read_text(encoding="utf-8")
    assert PROMPT not in text
    assert KEY not in text
    sweep = json.loads(text)
    assert sweep["kind"] == SWEEP_KIND
    assert sweep["summary"]["clean"] is True
    backend = sweep["attribution"]["backend"]
    assert backend["name"] == "ollama"
    assert backend["server"]["version"] == "0.12.3"
    assert backend["max_supported_in_flight"] is None
    assert any(
        "OLLAMA_NUM_PARALLEL" in item for item in sweep["attribution"]["cannot_claim"]
    )
    provenance = sweep["field_provenance"]
    assert provenance["server_load_ms"] == "provider_reported"
    assert provenance["server_e2e_ms"] == "provider_reported"
    assert "server_queue_ms" not in provenance

    point = sweep["points"][1]
    assert point["goodput"]["request_goodput_fraction"]["value"] == 1.0
    assert point["server_reported"]["server_e2e_ms"]["count"] == 4
    row = point["requests"][0]
    assert row["success"] is True
    assert row["ttft_source"] == "first_output_chunk"
    assert row["output_tokens"] == 3
    assert row["prompt_tokens"] == 6
    assert row["cached_prompt_tokens"] == 2
    assert row["finish_reason"] == "length"
    assert row["finish_reason_code"] == "length"
    assert row["server_metrics_source"] == "ollama_metrics"
    assert row["server_e2e_ms"] == 90.0
    assert row["server_load_ms"] == 1.5
    assert row["server_prompt_eval_ms"] == 8.0
    assert row["server_eval_ms"] == 30.0
    assert row["content_delta_count"] == 3
    assert row["tpot_status"] == "derived"
    assert row["ttft_ms"] <= row["first_content_ms"] <= row["e2e_ms"]


def test_sweep_refuses_when_version_probe_fails(
    tmp_path: Path, server: FakeOllama
) -> None:
    server.behaviour.version_status = 500
    code, _, err = run_main(sweep_argv(tmp_path, server.root, "--concurrency", "1"))
    assert code == 1
    assert "GET /api/version failed" in err
    assert server.bodies == []


def test_sweep_dry_run_and_remote_refusal(tmp_path: Path, server: FakeOllama) -> None:
    code, out, err = run_main(
        sweep_argv(tmp_path, server.root, "--concurrency", "1,8", "--dry-run")
    )
    assert code == 0, err
    assert "nothing sent" in out
    assert server.version_calls == 0
    plan = read_json(tmp_path / "out" / "sweep_plan.json")
    assert plan["kind"] == SWEEP_PLAN_KIND
    assert plan["attribution"]["backend"]["server"]["status"] == "not probed"
    assert plan["request_plan"]["body_without_prompt"]["options"] == {"num_predict": 3}

    code, _, err = run_main(
        sweep_argv(tmp_path, "https://ollama.example.invalid", "--concurrency", "1")
    )
    assert code == 1
    assert "--allow-remote-load" in err


def _one_row(server: FakeOllama) -> dict[str, Any]:
    send = make_ollama_sender(
        OllamaSettings(base_url=server.root, model="m", num_predict=3),
        transport=UrllibStreamingTransport(),
        environ={},
        run_id="r",
    )
    report = dispatch([ApiRequestSpec(prompt="x")], send, ClosedLoop(1))
    result = report.outcomes[0].result
    assert result is not None
    return result.to_dict()


@pytest.mark.parametrize(
    ("change", "category"),
    [
        ({"mode": "truncate"}, "stream_truncated"),
        ({"mode": "error_line"}, "provider_error_payload"),
        ({"status_code": 500}, "http_status"),
    ],
)
def test_sweep_row_failure_categories(
    server: FakeOllama, change: dict[str, Any], category: str
) -> None:
    for key, value in change.items():
        setattr(server.behaviour, key, value)
    row = _one_row(server)
    assert row["success"] is False
    assert row["failure_category"] == category
    assert row["output_tokens"] is None
    assert row["server_metrics_source"] is None


def test_absent_cached_count_stays_absent(server: FakeOllama) -> None:
    server.behaviour.cached = None
    row = _one_row(server)
    assert row["success"] is True
    assert row["cached_prompt_tokens"] is None
    assert row["prompt_tokens"] == 6


def test_specs_with_chat_fields_are_refused(server: FakeOllama) -> None:
    send = make_ollama_sender(
        OllamaSettings(base_url=server.root, model="m", num_predict=3),
        transport=UrllibStreamingTransport(),
        environ={},
        run_id="r",
    )
    report = dispatch(
        [ApiRequestSpec(prompt="x", system_prompt="be brief")], send, ClosedLoop(1)
    )
    assert report.outcomes[0].client_error == "OllamaError"
    assert server.bodies == []
