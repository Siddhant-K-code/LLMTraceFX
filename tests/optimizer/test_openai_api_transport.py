"""End-to-end tests for the standard-library streaming transport.

These tests run a deterministic HTTP server bound to 127.0.0.1 inside the
test process. No external host is contacted and no real credential is
used, but the real ``urllib`` transport, socket reads and SSE framing are
exercised together.
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer.collectors.openai_api import (
    FAILURE_CONNECTION,
    FAILURE_HTTP_STATUS,
    APICollectionConfig,
    HTTPRequest,
    TransportConnectionError,
    UrllibStreamingTransport,
    collect_openai_stream,
)

API_KEY = "transport-test-key-not-a-real-credential"
ENVIRON = {"ZAI_API_KEY": API_KEY}

STREAM_BODY = (
    b": keepalive\n\n"
    b'data: {"id": "chatcmpl-local", "model": "glm-5.3", '
    b'"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]}\n\n'
    b'data: {"choices": [{"index": 0, "delta": {"content": "Hello"}}]}\n\n'
    b'data: {"choices": [{"index": 0, "delta": {"content": " \xe4\xb8\x96\xe7\x95\x8c"}}]}\n\n'
    b'data: {"choices": [{"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}], '
    b'"usage": {"prompt_tokens": 6, "completion_tokens": 2, "total_tokens": 8}}\n\n'
    b"data: [DONE]\n\n"
)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    mode = "stream"
    seen: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        type(self).seen.append(
            {
                "path": self.path,
                "authorization": self.headers.get("Authorization"),
                "accept": self.headers.get("Accept"),
                "body": json.loads(body.decode("utf-8")),
            }
        )
        if self.mode == "stream":
            self._respond_stream()
        elif self.mode == "error":
            self._respond_error()
        else:
            self._respond_redirect()

    def _respond_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("X-Request-Id", "local-req-1")
        self.send_header("X-RateLimit-Remaining-Requests", "58")
        self.send_header("Content-Length", str(len(STREAM_BODY)))
        self.end_headers()
        # Write in small pieces so event boundaries land mid-socket-read.
        for start in range(0, len(STREAM_BODY), 17):
            self.wfile.write(STREAM_BODY[start : start + 17])
            self.wfile.flush()

    def _respond_error(self) -> None:
        payload = json.dumps(
            {"code": 1210, "message": "model is not available"}
        ).encode()
        self.send_response(404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _respond_redirect(self) -> None:
        self.send_response(302)
        self.send_header("Location", "https://attacker.example/steal")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        return None


class RawSocketServer:
    """A TCP server that sends fixed bytes and then closes the connection.

    ``BaseHTTPRequestHandler`` cannot produce a malformed status line or a
    body that stops short of its ``Content-Length``, so these protocol
    level failures need a socket the test controls directly.
    """

    def __init__(self, reply: bytes) -> None:
        self._reply = reply
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = int(self._sock.getsockname()[1])
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            conn, _ = self._sock.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(5.0)
            try:
                self._drain_request(conn)
                conn.sendall(self._reply)
            except OSError:
                return

    @staticmethod
    def _drain_request(conn: socket.socket) -> None:
        """Consume the whole request so the client never sees a reset."""
        buffer = b""
        while b"\r\n\r\n" not in buffer:
            chunk = conn.recv(65536)
            if not chunk:
                return
            buffer += chunk
        head, _, body = buffer.partition(b"\r\n\r\n")
        expected = 0
        for line in head.split(b"\r\n"):
            name, _, value = line.partition(b":")
            if name.strip().lower() == b"content-length":
                expected = int(value.strip())
        while len(body) < expected:
            chunk = conn.recv(65536)
            if not chunk:
                return
            body += chunk

    def close(self) -> None:
        self._sock.close()
        self._thread.join(timeout=5)


@pytest.fixture
def server() -> Iterator[ThreadingHTTPServer]:
    _Handler.seen = []
    _Handler.mode = "stream"
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def endpoint_for(httpd: ThreadingHTTPServer) -> str:
    host, port = httpd.server_address[0], httpd.server_address[1]
    return f"http://{host}:{port}/v1/chat/completions"


def make_config(tmp_path: Path, endpoint: str) -> APICollectionConfig:
    return APICollectionConfig(
        run_id="transport-run",
        provider="local-test-server",
        endpoint=endpoint,
        model_id="glm-5.3",
        prompt="Say hello.",
        output_dir=tmp_path / "artifacts",
        command_argv=(
            "llmtracefx-optimizer",
            "collect-api",
            "--run-id",
            "transport-run",
        ),
        credential_env_var="ZAI_API_KEY",
        request_timeout_seconds=10.0,
    )


def test_real_transport_streams_and_measures(
    tmp_path: Path, server: ThreadingHTTPServer
) -> None:
    config = make_config(tmp_path, endpoint_for(server))

    result = collect_openai_stream(
        config, transport=UrllibStreamingTransport(), environ=ENVIRON
    )

    assert result.record.outcome.success is True
    assert result.response_text == "Hello 世界"
    evidence = result.evidence
    assert evidence.finish_reason == "stop"
    assert evidence.response_id == "chatcmpl-local"
    assert evidence.usage.completion_tokens == 2
    assert evidence.rate_limit_headers == {"x-ratelimit-remaining-requests": "58"}
    assert evidence.provider_request_id == "local-req-1"
    assert evidence.stream_terminated_with_done is True

    timeline = evidence.timeline
    assert timeline.response_headers_offset_ms is not None
    assert timeline.first_content_token_offset_ms is not None
    assert timeline.completed_offset_ms >= timeline.first_content_token_offset_ms

    request = _Handler.seen[0]
    assert request["authorization"] == f"Bearer {API_KEY}"
    assert request["accept"] == "text/event-stream"
    assert request["body"]["stream"] is True
    for artifact in sorted(config.output_dir.iterdir()):
        assert API_KEY not in artifact.read_text(encoding="utf-8")


def test_real_transport_records_http_error_bodies(
    tmp_path: Path, server: ThreadingHTTPServer
) -> None:
    _Handler.mode = "error"
    config = make_config(tmp_path, endpoint_for(server))

    result = collect_openai_stream(
        config, transport=UrllibStreamingTransport(), environ=ENVIRON
    )

    failure = result.evidence.failure
    assert failure is not None
    assert failure.category == FAILURE_HTTP_STATUS
    assert failure.status_code == 404
    assert failure.provider_error_code == "1210"
    assert "model is not available" in failure.message


def test_real_transport_refuses_to_follow_redirects(
    tmp_path: Path, server: ThreadingHTTPServer
) -> None:
    _Handler.mode = "redirect"
    config = make_config(tmp_path, endpoint_for(server))

    result = collect_openai_stream(
        config, transport=UrllibStreamingTransport(), environ=ENVIRON
    )

    failure = result.evidence.failure
    assert failure is not None
    assert failure.category == FAILURE_HTTP_STATUS
    assert failure.status_code == 302
    # Only one request was made: the credential was never replayed elsewhere.
    assert len(_Handler.seen) == 1


def test_connection_refused_becomes_a_transport_connection_error(
    tmp_path: Path, server: ThreadingHTTPServer
) -> None:
    port = server.server_address[1]
    server.shutdown()
    server.server_close()
    transport = UrllibStreamingTransport()
    request = HTTPRequest(
        url=f"http://127.0.0.1:{port}/v1/chat/completions",
        method="POST",
        headers={"Content-Type": "application/json"},
        body=b"{}",
        timeout_seconds=5.0,
    )

    with pytest.raises(TransportConnectionError):
        transport.open_stream(request)


def test_a_stream_cut_short_mid_chunk_is_failure_evidence(tmp_path: Path) -> None:
    """``IncompleteRead`` is an ``HTTPException``, not an ``OSError``.

    SSE responses are chunked, so a proxy or load balancer hanging up mid
    body is the most likely real failure for a long lived stream. It has
    to become a failure shaped record instead of an unhandled traceback.
    """
    complete = b'data: {"choices": [{"index": 0, "delta": {"content": "Hi"}}]}\n\n'
    truncated = b'data: {"choices": [{"index"'
    body = (
        f"{len(complete):x}\r\n".encode()
        + complete
        + b"\r\n"
        # A chunk header promising far more than the bytes that follow.
        + f"{len(truncated) + 512:x}\r\n".encode()
        + truncated
    )
    raw = RawSocketServer(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n" + body
    )
    try:
        config = make_config(
            tmp_path, f"http://127.0.0.1:{raw.port}/v1/chat/completions"
        )
        result = collect_openai_stream(
            config, transport=UrllibStreamingTransport(), environ=ENVIRON
        )
    finally:
        raw.close()

    failure = result.evidence.failure
    assert failure is not None
    assert failure.category == FAILURE_CONNECTION
    assert "IncompleteRead" in failure.message
    assert result.record.outcome.success is False
    assert (config.output_dir / "api_evidence.json").is_file()
    for artifact in sorted(config.output_dir.iterdir()):
        assert API_KEY not in artifact.read_text(encoding="utf-8")


def test_a_malformed_status_line_is_failure_evidence(tmp_path: Path) -> None:
    """``BadStatusLine`` escapes ``urllib``'s ``OSError`` wrapping."""
    raw = RawSocketServer(b"NOT-HTTP GARBAGE LINE\r\n\r\n")
    try:
        config = make_config(
            tmp_path, f"http://127.0.0.1:{raw.port}/v1/chat/completions"
        )
        result = collect_openai_stream(
            config, transport=UrllibStreamingTransport(), environ=ENVIRON
        )
    finally:
        raw.close()

    failure = result.evidence.failure
    assert failure is not None
    assert failure.category == FAILURE_CONNECTION
    assert failure.status_code is None
    assert result.record.outcome.success is False


def test_an_unencodable_header_never_reaches_the_error_message(
    tmp_path: Path, server: ThreadingHTTPServer
) -> None:
    """Defence in depth behind the credential validation.

    ``http.client.putheader`` reports a rejected header by embedding the
    whole value in the ``ValueError``. That value can be the credential,
    so the transport must not let the original message or its traceback
    escape.
    """
    secret = "sk-zai-DEFENCE-IN-DEPTH\nInjected: yes"
    transport = UrllibStreamingTransport()
    request = HTTPRequest(
        url=endpoint_for(server),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": secret},
        body=b"{}",
        timeout_seconds=5.0,
    )

    with pytest.raises(TransportConnectionError) as excinfo:
        transport.open_stream(request)

    assert "sk-zai-DEFENCE-IN-DEPTH" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__ is True


def test_a_credential_with_a_trailing_newline_is_stripped_not_leaked(
    tmp_path: Path, server: ThreadingHTTPServer
) -> None:
    """A key sourced from a file usually ends in a newline.

    Sent unstripped, ``http.client.putheader`` raises a ``ValueError``
    whose message embeds the entire header value, which would print the
    credential to stderr.
    """
    config = make_config(tmp_path, endpoint_for(server))

    result = collect_openai_stream(
        config,
        transport=UrllibStreamingTransport(),
        environ={"ZAI_API_KEY": f"{API_KEY}\n"},
    )

    assert result.record.outcome.success is True
    assert _Handler.seen[0]["authorization"] == f"Bearer {API_KEY}"
