"""A local OpenAI-compatible streaming server for load tests.

It binds to 127.0.0.1 on a free port, streams a short chat completion per
request, and records the request bodies and the peak number of requests
it was serving at once. Nothing leaves the machine.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

API_KEY_ENV = "LOAD_TEST_KEY"
API_KEY = "load-test-key-not-a-real-credential"
ENVIRON = {API_KEY_ENV: API_KEY}


@dataclass
class ServerBehaviour:
    """What the fake server does. Tests mutate it between runs."""

    content_parts: tuple[str, ...] = ("Hello", " there", " friend")
    chunk_delay_s: float = 0.005
    status_code: int = 200
    usage: dict[str, Any] | None = field(
        default_factory=lambda: {
            "prompt_tokens": 11,
            "completion_tokens": 3,
            "total_tokens": 14,
            "prompt_tokens_details": {"cached_tokens": 4},
        }
    )
    finish_reason: str = "stop"
    metrics_text: str = ""


class FakeOpenAIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.behaviour = ServerBehaviour()
        self.bodies: list[dict[str, Any]] = []
        self.active = 0
        self.peak_active = 0
        self.completed = 0
        self.lock = threading.Lock()

    @property
    def endpoint(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        return f"http://{host}:{port}/v1/chat/completions"

    @property
    def metrics_url(self) -> str:
        host, port = self.server_address[0], self.server_address[1]
        return f"http://{host}:{port}/metrics"


def _sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: FakeOpenAIServer

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/metrics":
            self.send_error(404)
            return
        with self.server.lock:
            # Prometheus labels use braces, so a plain token stands in for
            # the live completed-request count.
            body = self.server.behaviour.metrics_text.replace(
                "@COMPLETED@", str(self.server.completed)
            ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8"))
        server = self.server
        behaviour = server.behaviour
        with server.lock:
            server.bodies.append(body)
            server.active += 1
            server.peak_active = max(server.peak_active, server.active)
        try:
            if behaviour.status_code != 200:
                payload = json.dumps({"error": {"message": "overloaded"}}).encode()
                self.send_response(behaviour.status_code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            for part in behaviour.content_parts:
                time.sleep(behaviour.chunk_delay_s)
                self.wfile.write(
                    _sse(
                        {
                            "id": "chatcmpl-load",
                            "model": body.get("model"),
                            "choices": [{"index": 0, "delta": {"content": part}}],
                        }
                    )
                )
                self.wfile.flush()
            final: dict[str, Any] = {
                "id": "chatcmpl-load",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": behaviour.finish_reason,
                    }
                ],
            }
            if behaviour.usage is not None:
                final["usage"] = behaviour.usage
            self.wfile.write(_sse(final))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        finally:
            with server.lock:
                server.active -= 1
                server.completed += 1

    def log_message(self, format: str, *args: Any) -> None:
        return None


@contextmanager
def fake_openai_server() -> Iterator[FakeOpenAIServer]:
    server = FakeOpenAIServer()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
