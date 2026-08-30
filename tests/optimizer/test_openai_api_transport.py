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
import time
import urllib.request
from collections.abc import Iterator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer.collectors.openai_api import (
    FAILURE_CONNECTION,
    FAILURE_HTTP_STATUS,
    FAILURE_TIMEOUT,
    APICollectionConfig,
    HTTPRequest,
    TransportConnectionError,
    TransportTimeout,
    UrllibStreamingTransport,
    _UrllibResponse,
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
    assert isinstance(host, str)
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
    assert timeline.completed_offset_ms is not None
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


class SlowChunkedServer:
    """Emits chunked SSE frames on demand so read timing is observable.

    Each frame is released only when the test says so, which is how a real
    provider behaves: tokens trickle out over seconds. A transport that
    waits for a full buffer before yielding cannot pass this.
    """

    def __init__(self, frames: list[bytes]) -> None:
        self._frames = frames
        self._release = [threading.Event() for _ in frames]
        self.finished = threading.Event()
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = int(self._sock.getsockname()[1])
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def release(self, index: int) -> None:
        self._release[index].set()

    def _serve(self) -> None:
        try:
            conn, _ = self._sock.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(10.0)
            buffer = b""
            try:
                while b"\r\n\r\n" not in buffer:
                    part = conn.recv(65536)
                    if not part:
                        return
                    buffer += part
                head, _, body = buffer.partition(b"\r\n\r\n")
                declared = 0
                for line in head.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        declared = int(line.split(b":", 1)[1])
                while len(body) < declared:
                    part = conn.recv(65536)
                    if not part:
                        break
                    body += part
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/event-stream\r\n"
                    b"Transfer-Encoding: chunked\r\n\r\n"
                )
                for event, frame in zip(self._release, self._frames, strict=True):
                    event.wait(10.0)
                    conn.sendall(b"%x\r\n" % len(frame) + frame + b"\r\n")
                conn.sendall(b"0\r\n\r\n")
            except OSError:
                return
            finally:
                self.finished.set()

    def close(self) -> None:
        self._sock.close()


def test_first_content_is_observed_before_the_stream_completes() -> None:
    """A blocking read would hold every early delta until the buffer filled.

    ``read(8192)`` blocks until it has 8192 bytes or the body ends, so on a
    real SSE stream the first token would only surface once the whole
    answer had arrived and every timing measurement would be worthless.
    """
    frames = [
        b'data: {"id": "c1", "choices": [{"index": 0, "delta": {"content": "A"}}]}\n\n',
        b'data: {"choices": [{"index": 0, "delta": {"content": "B"}},'
        + b'{"index": 0}]}\n\n',
        b'data: {"choices": [{"index": 0, "delta": {"content": ""},'
        + b'"finish_reason": "stop"}]}\n\ndata: [DONE]\n\n',
    ]
    server = SlowChunkedServer(frames)
    try:
        transport = UrllibStreamingTransport()
        response = transport.open_stream(
            HTTPRequest(
                url=f"http://127.0.0.1:{server.port}/v1/chat/completions",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=b"{}",
                timeout_seconds=10.0,
            )
        )
        stream = response.iter_bytes()
        server.release(0)
        first = next(stream)

        # The decisive assertion: bytes arrived while the server is still
        # holding back the rest of the response.
        assert b'"A"' in first
        assert not server.finished.is_set()

        server.release(1)
        server.release(2)
        rest = b"".join(stream)
        assert b'"B"' in rest
        response.close()
    finally:
        server.close()


def test_a_body_shorter_than_its_content_length_is_a_connection_failure(
    tmp_path: Path,
) -> None:
    """CPython returns a clean EOF here, so the shortfall must be detected.

    A truncated fixed-length body previously looked identical to a
    complete one and was published as a successful run.
    """
    body = b'data: {"choices": [{"index": 0, "delta": {"content": "half"}}]}\n\n'
    server = RawSocketServer(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Content-Length: 4096\r\n\r\n" + body
    )
    transport = UrllibStreamingTransport()
    response = transport.open_stream(
        HTTPRequest(
            url=f"http://127.0.0.1:{server.port}/v1/chat/completions",
            method="POST",
            headers={"Content-Type": "application/json"},
            body=b"{}",
            timeout_seconds=5.0,
        )
    )

    with pytest.raises(TransportConnectionError, match="of 4096 declared bytes"):
        list(response.iter_bytes())


def test_a_complete_fixed_length_body_is_not_flagged_as_truncated() -> None:
    body = b'data: {"choices": [{"index": 0, "delta": {"content": "x"}}]}\n\n'
    server = RawSocketServer(
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/event-stream\r\n"
        b"Content-Length: %d\r\n\r\n" % len(body) + body
    )
    transport = UrllibStreamingTransport()
    response = transport.open_stream(
        HTTPRequest(
            url=f"http://127.0.0.1:{server.port}/v1/chat/completions",
            method="POST",
            headers={"Content-Type": "application/json"},
            body=b"{}",
            timeout_seconds=5.0,
        )
    )

    assert b"".join(response.iter_bytes()) == body


def test_the_authorization_header_reaches_the_wire_verbatim() -> None:
    """Redaction applies to artifacts and logs, never to the live request.

    The sentinel is deliberately not key-shaped so the assertion stays
    readable, and the comparison is exact: a transport that sent a
    placeholder, a truncated value or a re-encoded value would fail here.
    """
    sentinel = "sentinel-not-a-real-key-0123456789"
    captured: dict[str, bytes] = {}
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])

    def serve() -> None:
        conn, _ = listener.accept()
        with conn:
            buffer = b""
            while b"\r\n\r\n" not in buffer:
                part = conn.recv(65536)
                if not part:
                    return
                buffer += part
            captured["head"] = buffer
            payload = b"data: [DONE]\n\n"
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Content-Length: %d\r\n\r\n" % len(payload) + payload
            )

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        transport = UrllibStreamingTransport()
        response = transport.open_stream(
            HTTPRequest(
                url=f"http://127.0.0.1:{port}/v1/chat/completions",
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {sentinel}",
                },
                body=b"{}",
                timeout_seconds=5.0,
            )
        )
        list(response.iter_bytes())
        response.close()
        thread.join(timeout=5.0)
    finally:
        listener.close()

    lines = captured["head"].decode("latin-1").split("\r\n")
    headers = [line for line in lines if line.lower().startswith("authorization:")]
    assert len(headers) == 1
    assert headers[0].split(":", 1)[1].strip() == "Bearer " + sentinel


def test_the_collector_sends_the_resolved_credential_to_a_real_socket(
    tmp_path: Path,
) -> None:
    """Same guarantee, exercised through the full collector rather than the
    transport alone, so a redaction applied one layer up would be caught."""
    captured: dict[str, bytes] = {}
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])

    def serve() -> None:
        conn, _ = listener.accept()
        with conn:
            buffer = b""
            while b"\r\n\r\n" not in buffer:
                part = conn.recv(65536)
                if not part:
                    return
                buffer += part
            head, _, body = buffer.partition(b"\r\n\r\n")
            declared = 0
            for line in head.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    declared = int(line.split(b":", 1)[1])
            while len(body) < declared:
                part = conn.recv(65536)
                if not part:
                    break
                body += part
            captured["head"] = head
            conn.sendall(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Type: text/event-stream\r\n"
                b"Content-Length: %d\r\n\r\n" % len(STREAM_BODY) + STREAM_BODY
            )

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    sentinel = "sentinel-not-a-real-key-0123456789"
    config = APICollectionConfig(
        run_id="wire",
        provider="local",
        endpoint=f"http://127.0.0.1:{port}/v1/chat/completions",
        model_id="glm-5.3-flash",
        prompt="hello",
        output_dir=tmp_path / "artifacts",
        command_argv=("llmtracefx-optimizer", "collect-api"),
        credential_env_var="LOCAL_TEST_KEY",
    )
    try:
        result = collect_openai_stream(
            config,
            transport=UrllibStreamingTransport(),
            environ={"LOCAL_TEST_KEY": sentinel},
        )
        thread.join(timeout=5.0)
    finally:
        listener.close()

    assert result.record.outcome.success is True
    lines = captured["head"].decode("latin-1").split("\r\n")
    headers = [line for line in lines if line.lower().startswith("authorization:")]
    assert headers[0].split(":", 1)[1].strip() == "Bearer " + sentinel
    # ...and the same value never reaches an artifact.
    for path in sorted(config.output_dir.iterdir()):
        assert sentinel not in path.read_text(encoding="utf-8"), path.name


# --- Fourteenth review pass ---------------------------------------------------


class _FakeSocket:
    """Records every timeout the response asks for."""

    def __init__(self) -> None:
        self.timeouts: list[float] = []

    def settimeout(self, value: float) -> None:
        self.timeouts.append(value)


class _FakeRaw:
    """An ``HTTPResponse`` stub exposing the documented ``fp.raw._sock`` chain."""

    def __init__(self, chunks: list[bytes], sock: _FakeSocket) -> None:
        self._chunks = list(chunks)
        self.fp = type("_FP", (), {"raw": type("_Raw", (), {"_sock": sock})()})()

    def read1(self, _size: int) -> bytes:
        return self._chunks.pop(0) if self._chunks else b""

    def close(self) -> None:
        return None


def test_each_read_is_bounded_by_what_is_left_of_the_budget() -> None:
    """The socket timeout must shrink toward the deadline, not reset per read.

    The timeout handed to ``urlopen`` bounds one blocking operation. Without
    this, a read starting just inside the budget can block for another full
    timeout, so the advertised whole-response bound is close to twice what
    was configured.
    """
    sock = _FakeSocket()
    now = [0.0]
    response = _UrllibResponse(
        _FakeRaw([b"a", b"b", b"c"], sock),
        200,
        {},
        deadline=10.0,
        clock=lambda: now[0],
    )

    consumed = []
    for chunk in response.iter_bytes():
        consumed.append(chunk)
        now[0] += 3.0

    assert consumed == [b"a", b"b", b"c"]
    assert sock.timeouts == [10.0, 7.0, 4.0, 1.0]
    assert sock.timeouts == sorted(sock.timeouts, reverse=True)


def test_a_read_that_starts_past_the_deadline_is_a_timeout() -> None:
    """Once the budget is gone the stream fails rather than blocking again."""
    sock = _FakeSocket()
    now = [0.0]
    response = _UrllibResponse(
        _FakeRaw([b"a", b"b"], sock), 200, {}, deadline=5.0, clock=lambda: now[0]
    )

    with pytest.raises(TransportTimeout):
        for _ in response.iter_bytes():
            now[0] += 6.0


def test_a_response_without_a_deadline_is_unchanged() -> None:
    """No deadline configured means no socket adjustment at all."""
    sock = _FakeSocket()
    response = _UrllibResponse(_FakeRaw([b"a"], sock), 200, {})

    chunks = list(response.iter_bytes())

    assert chunks == [b"a"]
    assert sock.timeouts == []


class StallingServer:
    """Sends headers and one chunk, then stops sending without closing.

    This is the shape the whole-response deadline exists for: the provider
    stays reachable and the connection stays open, so nothing errors out.
    Only the clock ends the run, and how long that takes is exactly what
    the per-read budget decides.
    """

    def __init__(self, first_chunk_after: float) -> None:
        self._first_chunk_after = first_chunk_after
        self._sock = socket.socket()
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = int(self._sock.getsockname()[1])
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            conn, _ = self._sock.accept()
        except OSError:
            return
        with conn:
            conn.settimeout(10.0)
            try:
                RawSocketServer._drain_request(conn)
                # No Content-Length, so the client reads until the peer
                # closes and cannot decide the body is complete on its own.
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/event-stream\r\n"
                    b"\r\n"
                    b'data: {"id": "stall", "choices": [{"index": 0, '
                    b'"delta": {"role": "assistant"}}]}\n\n'
                )
                # One real chunk lands inside the budget, so the failing
                # read is the one that starts with only part of the budget
                # left rather than the first read of the stream.
                self._stop.wait(self._first_chunk_after)
                conn.sendall(
                    b'data: {"choices": [{"index": 0, '
                    b'"delta": {"content": "Hi"}}]}\n\n'
                )
                self._stop.wait(30.0)
            except OSError:
                return

    def close(self) -> None:
        self._stop.set()
        self._sock.close()
        self._thread.join(timeout=10)


def test_a_stalled_stream_stops_at_the_budget_not_at_twice_it(
    tmp_path: Path,
) -> None:
    """A real socket must honour the whole-response budget, not per read.

    The last chunk arrives at 70 percent of the budget, so a per-read
    timeout would let the following read block for another full timeout and
    finish at about 1.7 times the configured bound. The assertion is
    deliberately loose: it only has to separate one budget from nearly two.
    """
    budget = 2.0
    server = StallingServer(first_chunk_after=budget * 0.7)
    try:
        config = replace(
            make_config(
                tmp_path, f"http://127.0.0.1:{server.port}/v1/chat/completions"
            ),
            request_timeout_seconds=budget,
        )

        started = time.monotonic()
        result = collect_openai_stream(
            config, transport=UrllibStreamingTransport(), environ=ENVIRON
        )
        elapsed = time.monotonic() - started
    finally:
        server.close()

    assert result.record.outcome.success is False
    assert result.evidence.failure is not None
    assert result.evidence.failure.category == FAILURE_TIMEOUT
    # It really did wait for the budget rather than giving up early.
    assert elapsed >= budget * 0.9
    # Without the per-read tightening this lands near budget * 1.7.
    assert elapsed < budget * 1.35, f"stalled stream ran for {elapsed:.2f}s"


class DelayedRequestAndHeadersServer:
    """Delay request-body draining, then delay response headers."""

    def __init__(self, phase_delay: float) -> None:
        self._phase_delay = phase_delay
        self._stop = threading.Event()
        self.request_body_read_started = threading.Event()
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
            received = b""
            try:
                while b"\r\n\r\n" not in received:
                    part = conn.recv(65536)
                    if not part:
                        return
                    received += part
                head, _, body = received.partition(b"\r\n\r\n")
                declared = 0
                for line in head.split(b"\r\n"):
                    if line.lower().startswith(b"content-length:"):
                        declared = int(line.split(b":", 1)[1])
                        break
                if self._stop.wait(self._phase_delay):
                    return
                self.request_body_read_started.set()
                while len(body) < declared:
                    part = conn.recv(65536)
                    if not part:
                        return
                    body += part
                if self._stop.wait(self._phase_delay):
                    return
                payload = b"data: [DONE]\n\n"
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    + b"Content-Type: text/event-stream\r\n"
                    + b"Content-Length: %d\r\n\r\n" % len(payload)
                    + payload
                )
            except OSError:
                return

    def close(self) -> None:
        self._stop.set()
        self._sock.close()
        self._thread.join(timeout=5.0)


class SlowDripStatusServer:
    """Drip an unterminated status line while the absolute budget expires."""

    def __init__(self, byte_interval: float) -> None:
        self._byte_interval = byte_interval
        self._stop = threading.Event()
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
                RawSocketServer._drain_request(conn)
                for byte in b"HTTP/1.1 200 OK\r\n":
                    if self._stop.wait(self._byte_interval):
                        return
                    conn.sendall(bytes((byte,)))
                self._stop.wait(5.0)
            except OSError:
                return

    def close(self) -> None:
        self._stop.set()
        self._sock.close()
        self._thread.join(timeout=5.0)


class SlowDripChunkSizeServer:
    """Send response headers, then drip a never-complete chunk-size line."""

    def __init__(self, byte_interval: float) -> None:
        self._byte_interval = byte_interval
        self._stop = threading.Event()
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
                RawSocketServer._drain_request(conn)
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    + b"Content-Type: text/event-stream\r\n"
                    + b"Transfer-Encoding: chunked\r\n\r\n"
                )
                for byte in b"1;slow-extension-never-ends":
                    if self._stop.wait(self._byte_interval):
                        return
                    conn.sendall(bytes((byte,)))
                self._stop.wait(5.0)
            except OSError:
                return

    def close(self) -> None:
        self._stop.set()
        self._sock.close()
        self._thread.join(timeout=5.0)


class TerminalThenStallServer:
    """Send a terminal finish reason, then keep the response open."""

    def __init__(self) -> None:
        self._stop = threading.Event()
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
                RawSocketServer._drain_request(conn)
                conn.sendall(
                    b"HTTP/1.1 200 OK\r\n"
                    + b"Content-Type: text/event-stream\r\n\r\n"
                    + b'data: {"choices": [{"index": 0, "delta": '
                    + b'{"content": "done"}, "finish_reason": "stop"}]}\n\n'
                )
                self._stop.wait(5.0)
            except OSError:
                return

    def close(self) -> None:
        self._stop.set()
        self._sock.close()
        self._thread.join(timeout=5.0)


def test_connect_upload_and_headers_share_one_deadline() -> None:
    budget = 1.2
    server = DelayedRequestAndHeadersServer(phase_delay=0.65)
    started = time.monotonic()
    try:
        with pytest.raises(TransportTimeout):
            UrllibStreamingTransport().open_stream(
                HTTPRequest(
                    url=f"http://127.0.0.1:{server.port}/v1/chat/completions",
                    method="POST",
                    headers={"Content-Type": "application/json"},
                    body=b"x" * (8 * 1024 * 1024),
                    timeout_seconds=budget,
                )
            )
    finally:
        server.close()
    elapsed = time.monotonic() - started

    assert server.request_body_read_started.is_set()
    assert elapsed >= budget * 0.75
    assert elapsed < budget + 0.4


def test_slow_drip_status_bytes_cannot_reset_the_deadline() -> None:
    budget = 0.2
    server = SlowDripStatusServer(byte_interval=0.04)
    started = time.monotonic()
    try:
        with pytest.raises(TransportTimeout):
            UrllibStreamingTransport().open_stream(
                HTTPRequest(
                    url=f"http://127.0.0.1:{server.port}/v1/chat/completions",
                    method="POST",
                    headers={"Content-Type": "application/json"},
                    body=b"{}",
                    timeout_seconds=budget,
                )
            )
    finally:
        server.close()
    elapsed = time.monotonic() - started

    assert elapsed >= budget * 0.75
    assert elapsed < budget + 0.3


def test_slow_drip_chunk_framing_cannot_reset_the_deadline() -> None:
    budget = 0.2
    server = SlowDripChunkSizeServer(byte_interval=0.04)
    response = None
    started = time.monotonic()
    try:
        response = UrllibStreamingTransport().open_stream(
            HTTPRequest(
                url=f"http://127.0.0.1:{server.port}/v1/chat/completions",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=b"{}",
                timeout_seconds=budget,
            )
        )
        with pytest.raises(TransportTimeout):
            list(response.iter_bytes())
    finally:
        if response is not None:
            response.close()
        server.close()
    elapsed = time.monotonic() - started

    assert elapsed >= budget * 0.75
    assert elapsed < budget + 0.3


def test_dns_resolution_cannot_outlive_the_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_getaddrinfo = socket.getaddrinfo

    def slow_resolution(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.3)
        return original_getaddrinfo(*args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", slow_resolution)
    started = time.monotonic()

    with pytest.raises(TransportTimeout):
        UrllibStreamingTransport().open_stream(
            HTTPRequest(
                url="http://localhost:9/v1/chat/completions",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=b"{}",
                timeout_seconds=0.08,
            )
        )

    assert time.monotonic() - started < 0.25


def test_loopback_http_explicitly_disables_environment_proxies(
    tmp_path: Path,
    server: ThreadingHTTPServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_build_opener = urllib.request.build_opener
    installed_handlers: list[Any] = []

    def capture_handlers(*handlers: Any) -> Any:
        installed_handlers.extend(handlers)
        return original_build_opener(*handlers)

    monkeypatch.setenv("http_proxy", "http://proxy.invalid:8080")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:8080")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.setattr(urllib.request, "build_opener", capture_handlers)

    result = collect_openai_stream(
        make_config(tmp_path, endpoint_for(server)),
        transport=UrllibStreamingTransport(),
        environ=ENVIRON,
    )

    assert result.record.outcome.success is True
    proxy_handlers = [
        handler
        for handler in installed_handlers
        if isinstance(handler, urllib.request.ProxyHandler)
    ]
    assert len(proxy_handlers) == 1
    assert proxy_handlers[0].proxies == {}


def test_deadline_eof_cannot_turn_a_terminal_stall_into_success(
    tmp_path: Path,
) -> None:
    budget = 0.25
    server = TerminalThenStallServer()
    try:
        config = replace(
            make_config(
                tmp_path, f"http://127.0.0.1:{server.port}/v1/chat/completions"
            ),
            request_timeout_seconds=budget,
        )

        result = collect_openai_stream(
            config, transport=UrllibStreamingTransport(), environ=ENVIRON
        )
    finally:
        server.close()

    assert result.evidence.failure is not None
    assert result.evidence.failure.category == FAILURE_TIMEOUT
    assert result.record.outcome.success is False


def test_dns_unicode_errors_are_sanitized_transport_failures(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def invalid_idna(*_args: Any, **_kwargs: Any) -> Any:
        raise UnicodeEncodeError("idna", "x" * 64, 0, 64, "label too long")

    monkeypatch.setattr(socket, "getaddrinfo", invalid_idna)

    with pytest.raises(TransportConnectionError, match="could not be encoded"):
        UrllibStreamingTransport().open_stream(
            HTTPRequest(
                url="http://invalid.example/v1/chat/completions",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=b"{}",
                timeout_seconds=1.0,
            )
        )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
