"""Native Ollama collector for ``/api/chat`` and ``/api/generate``.

Ollama also serves an OpenAI-compatible ``/v1`` API, which ``collect-api``
already drives. That route drops the server's own timers. The native API
sends them on the final stream chunk, so this collector reads it directly
and records them next to the client's own TTFT and inter-chunk timing.

Wire format checked against ollama/ollama at :data:`OLLAMA_COMMIT`
(``api/types.go``, ``server/routes.go``, ``llm/llama_server.go`` and
``docs/api.md``):

* ``POST /api/chat`` and ``POST /api/generate`` with ``stream: true``
  answer with NDJSON (``application/x-ndjson``): one JSON object per line.
  Every chunk carries ``model``, ``created_at`` and ``done``. Chat chunks
  hold text in ``message.content`` and ``message.thinking``; generate
  chunks in ``response`` and ``thinking``.
* The last chunk has ``done: true``, a ``done_reason`` (``stop`` or
  ``length``) and the ``Metrics`` fields: ``total_duration``,
  ``load_duration``, ``prompt_eval_count``, ``prompt_eval_cached_count``,
  ``prompt_eval_duration``, ``eval_count`` and ``eval_duration``.
  Durations are nanoseconds. Most fields are ``omitempty``, so a field the
  server left at zero is absent; it is recorded as absent, never as zero.
  ``prompt_eval_cached_count`` is a pointer, so a reported zero survives.
* There is no ``[DONE]`` sentinel. A stream that ends before the
  ``done: true`` chunk was cut short.
* An error after the stream started is a line ``{"error": "..."}`` with
  the HTTP status still 200. An error before it is a non-200 status with a
  JSON ``{"error": "..."}`` body.
* ``GET /api/version`` returns ``{"version": "..."}``.

What the server timers cover, from ``server/routes.go``: the clock starts
when the handler starts. ``load_duration`` runs until the model is
scheduled and loaded. A request that then waits for a parallel slot
(``OLLAMA_NUM_PARALLEL``) waits inside the runner, so that wait is in
``total_duration`` but in no other field. Nothing here derives a queue time
from the difference.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..manifest import collect_environment_manifest
from ..schema import (
    CommandInfo,
    ErrorInfo,
    ExperimentRecord,
    Measurement,
    MemoryMetrics,
    MetricProvenance,
    ModelInfo,
    OutcomeInfo,
    RepetitionInfo,
    RuntimeInfo,
    TimingMetrics,
    TokenCounts,
    utc_now_iso,
)
from ._shared import (
    atomic_write_text,
    config_hash,
    milliseconds,
    record_platform,
    sha256_bytes,
    sha256_text,
)
from .openai_api import (
    FAILURE_CONNECTION,
    FAILURE_HTTP_STATUS,
    FAILURE_PROVIDER_ERROR,
    FAILURE_STREAM_DECODE,
    FAILURE_STREAM_TRUNCATED,
    FAILURE_TIMEOUT,
    HTTPRequest,
    OpenAIStreamCollectorError,
    StreamingTransport,
    TransportConnectionError,
    TransportTimeout,
    _assert_header_safe_credential,
    _latency_distribution,
    _validate_endpoint,
    parse_endpoint,
    redact_text_for_dry_run,
)
from .server_reported import PROVENANCE, SOURCE_OLLAMA

OLLAMA_REPOSITORY = "ollama/ollama"
OLLAMA_COMMIT = "42ac2c0c1e132c0b4f0d9cfcd4e7950f0c5470bc"
DEFAULT_BASE_URL = "http://127.0.0.1:11434"
RUNTIME_NAME = "ollama"

API_CHAT = "chat"
API_GENERATE = "generate"
API_PATHS: Mapping[str, str] = {API_CHAT: "/api/chat", API_GENERATE: "/api/generate"}
VERSION_PATH = "/api/version"

EVIDENCE_KIND = "llmtracefx.ollama_evidence"
EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_FILE_NAME = "ollama_evidence.json"

DURATION_FIELDS: tuple[str, ...] = (
    "total_duration",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
)
COUNT_FIELDS: tuple[str, ...] = (
    "prompt_eval_count",
    "prompt_eval_cached_count",
    "eval_count",
)
FIELD_DEFINITIONS: Mapping[str, str] = {
    "total_duration": (
        "from the start of request handling to the final chunk; includes "
        "load_duration, any wait for a parallel slot, prompt evaluation and "
        "generation"
    ),
    "load_duration": (
        "from the start of request handling until the model was scheduled "
        "and loaded; small when the model was already loaded"
    ),
    "prompt_eval_count": "prompt tokens, including any read from the prompt cache",
    "prompt_eval_cached_count": "prompt tokens read from the prompt cache",
    "prompt_eval_duration": "time evaluating the prompt tokens that were not cached",
    "eval_count": "generated tokens",
    "eval_duration": "time generating the response tokens",
}

FINISH_CODES: Mapping[str, str] = {"stop": "stop", "length": "length"}
"""``done_reason`` values that end a generation normally."""

FAILURE_UNRECOGNIZED_DONE = "unrecognized_done_reason"

TTFT_FIRST_OUTPUT_CHUNK = "first_output_chunk"
"""The first chunk with content, thinking or a tool call, matching the
OpenAI-compatible collector's earliest output event."""

_NS_PER_MS = 1_000_000
_MAX_LINE_BYTES = 16 * 1024 * 1024
"""Generate's final chunk carries a ``context`` token array that grows with
the prompt. It is parsed and dropped, never persisted."""
_MAX_ERROR_BODY_BYTES = 64 * 1024
_MAX_VERSION_BYTES = 64 * 1024
_MAX_ERROR_CHARS = 300
_MAX_LABEL_CHARS = 128
_MAX_EXACT_INT = 2**63 - 1


class OllamaError(ValueError):
    """An Ollama collection or sweep cannot be configured or started."""


class NDJSONDecodeError(ValueError):
    """A stream line was longer than the decoder accepts."""


def _number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


@dataclass(frozen=True)
class OllamaSettings:
    """What every request to one Ollama server sends.

    ``base_url`` is the server root; the collector appends the API path.
    Sampling values left as ``None`` are not sent, so Ollama's and the
    model's own defaults apply and are not guessed. ``num_predict`` is
    always sent because Ollama's default is no limit.
    """

    base_url: str
    model: str
    num_predict: int
    api: str = API_CHAT
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    num_ctx: int | None = None
    think: bool | None = None
    request_timeout_seconds: float = 300.0
    credential_env_var: str | None = None

    def __post_init__(self) -> None:
        try:
            _validate_endpoint(self.base_url)
        except (ValueError, OpenAIStreamCollectorError) as exc:
            raise OllamaError(str(exc).replace("endpoint", "server URL")) from None
        parts = parse_endpoint(self.base_url)
        if parts.query or parts.fragment:
            raise OllamaError("server URL must not contain a query or fragment")
        path = parts.path.rstrip("/")
        if path.endswith("/api") or any(
            path.endswith(suffix) for suffix in (*API_PATHS.values(), "/v1")
        ):
            raise OllamaError(
                "give the server root, for example http://127.0.0.1:11434, not "
                "an API path; the collector adds /api/chat or /api/generate"
            )
        if not isinstance(self.model, str) or not self.model.strip():
            raise OllamaError("--model must name an Ollama model")
        if self.model != self.model.strip() or any(
            ord(char) < 0x20 or ord(char) == 0x7F for char in self.model
        ):
            raise OllamaError(
                "--model must not have surrounding spaces or control characters"
            )
        if self.api not in API_PATHS:
            raise OllamaError(f"--api must be one of {sorted(API_PATHS)}")
        if (
            isinstance(self.num_predict, bool)
            or not isinstance(self.num_predict, int)
            or self.num_predict < 1
        ):
            raise OllamaError(
                "--max-output-tokens must be a positive integer. Ollama "
                "defaults num_predict to -1 (no limit), so it is always sent"
            )
        if self.temperature is not None and (
            not _number(self.temperature)
            or not math.isfinite(self.temperature)
            or self.temperature < 0
        ):
            raise OllamaError("--temperature must be a finite number of at least 0")
        if self.top_p is not None and (
            not _number(self.top_p)
            or not math.isfinite(self.top_p)
            or not 0 < self.top_p <= 1
        ):
            raise OllamaError("--top-p must be within (0, 1]")
        for flag, value, minimum in (
            ("--top-k", self.top_k, 0),
            ("--num-ctx", self.num_ctx, 1),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < minimum
            ):
                raise OllamaError(f"{flag} must be an integer of at least {minimum}")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise OllamaError("--seed must be an integer")
        if self.think is not None and not isinstance(self.think, bool):
            raise OllamaError("think must be true, false or unset")
        if (
            not _number(self.request_timeout_seconds)
            or not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
        ):
            raise OllamaError("--request-timeout must be a positive number")
        if self.credential_env_var is not None and not self.credential_env_var:
            raise OllamaError("--api-key-env must name an environment variable")

    def url(self, path: str) -> str:
        return self.base_url.rstrip("/") + path

    @property
    def api_url(self) -> str:
        return self.url(API_PATHS[self.api])

    @property
    def version_url(self) -> str:
        return self.url(VERSION_PATH)

    def options(self) -> dict[str, Any]:
        options: dict[str, Any] = {"num_predict": self.num_predict}
        for key, value in (
            ("temperature", self.temperature),
            ("top_p", self.top_p),
            ("top_k", self.top_k),
            ("seed", self.seed),
            ("num_ctx", self.num_ctx),
        ):
            if value is not None:
                options[key] = value
        return options


def request_body(
    settings: OllamaSettings, prompt: str, system_prompt: str | None = None
) -> dict[str, Any]:
    """The request body. Ollama applies the model's template to both APIs."""
    body: dict[str, Any] = {"model": settings.model, "stream": True}
    if settings.api == API_CHAT:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        body["messages"] = messages
    else:
        body["prompt"] = prompt
        if system_prompt is not None:
            body["system"] = system_prompt
    body["options"] = settings.options()
    if settings.think is not None:
        body["think"] = settings.think
    return body


def encode_body(body: Mapping[str, Any]) -> bytes:
    return json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")


def upstream_reference() -> dict[str, str]:
    return {"repository": OLLAMA_REPOSITORY, "commit": OLLAMA_COMMIT}


def request_plan(settings: OllamaSettings) -> dict[str, Any]:
    """What every request sends, without the prompt or any credential."""
    body = request_body(settings, "")
    if settings.api == API_CHAT:
        body.pop("messages")
        prompt_field = (
            "messages: an optional system message, then the prompt as one user "
            "message; Ollama applies the model's chat template"
        )
    else:
        body.pop("prompt")
        prompt_field = (
            "prompt, with system when a system prompt is given; Ollama applies "
            "the model's template"
        )
    return {
        "endpoint": settings.api_url,
        "method": "POST",
        "api": f"Ollama native {API_PATHS[settings.api]} (not OpenAI-compatible)",
        "body_without_prompt": body,
        "prompt_field": prompt_field,
        "header_names": sorted(
            ["Accept", "Content-Type"]
            + ([] if settings.credential_env_var is None else ["Authorization"])
        ),
        "credential_env_var": settings.credential_env_var,
        "request_timeout_seconds": settings.request_timeout_seconds,
        "upstream_reference": upstream_reference(),
    }


def settings_hash(settings: OllamaSettings) -> str:
    """Hash every setting that shapes the request, excluding the prompt."""
    plan = request_plan(settings)
    return config_hash(
        {
            "endpoint": plan["endpoint"],
            "body_without_prompt": plan["body_without_prompt"],
            "request_timeout_seconds": settings.request_timeout_seconds,
        }
    )


def resolve_credential(
    settings: OllamaSettings, environ: Mapping[str, str]
) -> str | None:
    """The bearer token from the named environment variable, if one is named.

    Local Ollama needs none. A proxy in front of it may.
    """
    if settings.credential_env_var is None:
        return None
    value = environ.get(settings.credential_env_var)
    if value is None:
        raise OllamaError(
            "the environment variable named by --api-key-env is not set in "
            "this environment"
        )
    credential = value.strip()
    if not credential:
        raise OllamaError("the environment variable named by --api-key-env is empty")
    try:
        _assert_header_safe_credential(credential, settings.credential_env_var)
    except OpenAIStreamCollectorError as exc:
        raise OllamaError(str(exc)) from None
    return credential


def _headers(credential: str | None, *, accept: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": accept}
    if credential is not None:
        headers["Authorization"] = f"Bearer {credential}"
    return headers


def _read_body(response: Any, limit: int) -> bytes:
    data = bytearray()
    for chunk in response.iter_bytes():
        data.extend(chunk)
        if len(data) > limit:
            break
    return bytes(data[:limit])


def _label(value: Any, limit: int = _MAX_LABEL_CHARS) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value[:limit]


@dataclass(frozen=True)
class VersionProbe:
    """What ``GET /api/version`` said, or why it said nothing."""

    version: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": f"GET {VERSION_PATH}",
            "provenance": PROVENANCE,
            "error": self.error,
        }


def probe_version(
    settings: OllamaSettings,
    *,
    transport: StreamingTransport,
    credential: str | None,
) -> VersionProbe:
    """Ask the server for its version. Never raises; a failure is described."""
    request = HTTPRequest(
        url=settings.version_url,
        method="GET",
        headers=_headers(credential, accept="application/json"),
        body=b"",
        timeout_seconds=min(settings.request_timeout_seconds, 30.0),
    )
    try:
        response = transport.open_stream(request)
    except (TransportTimeout, TransportConnectionError) as exc:
        return VersionProbe(None, f"{type(exc).__name__}; is Ollama running?")
    try:
        status = response.status_code
        body = _read_body(response, _MAX_VERSION_BYTES)
    except (TransportTimeout, TransportConnectionError) as exc:
        return VersionProbe(None, f"{type(exc).__name__} while reading")
    finally:
        response.close()
    if status != 200:
        return VersionProbe(None, f"HTTP {status}")
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return VersionProbe(None, "response was not JSON")
    version = payload.get("version") if isinstance(payload, dict) else None
    label = _label(version, 64)
    if label is None or not all(char.isalnum() or char in "._+-" for char in label):
        return VersionProbe(None, "response had no usable version string")
    return VersionProbe(label, None)


class NDJSONDecoder:
    """Splits a byte stream into non-empty lines, refusing oversized ones."""

    def __init__(self, max_line_bytes: int = _MAX_LINE_BYTES) -> None:
        self._buffer = bytearray()
        self._scanned = 0
        self._max = max_line_bytes

    def feed(self, chunk: bytes) -> list[bytes]:
        self._buffer.extend(chunk)
        lines: list[bytes] = []
        while (index := self._buffer.find(b"\n", self._scanned)) >= 0:
            line = bytes(self._buffer[:index])
            del self._buffer[: index + 1]
            self._scanned = 0
            self._take(line, lines)
        self._scanned = len(self._buffer)
        if self._scanned > self._max:
            raise NDJSONDecodeError("a stream line exceeded the size limit")
        return lines

    def close(self) -> list[bytes]:
        lines: list[bytes] = []
        self._take(bytes(self._buffer), lines)
        self._buffer.clear()
        self._scanned = 0
        return lines

    def _take(self, line: bytes, lines: list[bytes]) -> None:
        if len(line) > self._max:
            raise NDJSONDecodeError("a stream line exceeded the size limit")
        stripped = line.strip()
        if stripped:
            lines.append(stripped)


def _error_text(value: Any, credential: str | None) -> str:
    if isinstance(value, str):
        text = value
    elif isinstance(value, dict) and isinstance(value.get("message"), str):
        text = value["message"]
    else:
        text = "the server sent an error without a message"
    text = redact_text_for_dry_run(text, credential)
    return text[:_MAX_ERROR_CHARS]


@dataclass
class OllamaStream:
    """One request's stream as the client saw it.

    Times are on the caller's clock. ``content_parts`` stays empty unless
    the caller asked to keep the text.
    """

    api: str
    start: float
    keep_text: bool = False
    credential: str | None = None
    status: int | None = None
    first_output_at: float | None = None
    first_content_at: float | None = None
    first_thinking_at: float | None = None
    output_chunk_times: list[float] = field(default_factory=list)
    chunks: int = 0
    content_chunks: int = 0
    thinking_chunks: int = 0
    tool_call_chunks: int = 0
    content_characters: int = 0
    thinking_characters: int = 0
    content_parts: list[str] = field(default_factory=list)
    final: dict[str, Any] | None = None
    final_at: float | None = None
    end: float | None = None
    failure: str | None = None
    failure_message: str | None = None

    @property
    def finished(self) -> bool:
        return self.failure is not None or self.final is not None

    def fail(self, category: str, message: str) -> None:
        if self.failure is None:
            self.failure = category
            self.failure_message = message

    def _pieces(self, chunk: Mapping[str, Any]) -> tuple[Any, Any, Any] | None:
        if self.api == API_CHAT:
            message = chunk.get("message")
            if message is None:
                return None, None, None
            if not isinstance(message, dict):
                return None
            return (
                message.get("content"),
                message.get("thinking"),
                message.get("tool_calls"),
            )
        return chunk.get("response"), chunk.get("thinking"), None

    def handle(self, line: bytes, now: float) -> None:
        try:
            chunk = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.fail(FAILURE_STREAM_DECODE, "a stream line was not JSON")
            return
        if not isinstance(chunk, dict):
            self.fail(FAILURE_STREAM_DECODE, "a stream line was not a JSON object")
            return
        if "error" in chunk:
            self.fail(
                FAILURE_PROVIDER_ERROR, _error_text(chunk["error"], self.credential)
            )
            return
        pieces = self._pieces(chunk)
        done = chunk.get("done", False)
        if pieces is None or not isinstance(done, bool):
            self.fail(FAILURE_STREAM_DECODE, "a stream chunk had an unexpected shape")
            return
        content, thinking, tool_calls = pieces
        if any(
            value is not None and not isinstance(value, kind)
            for value, kind in ((content, str), (thinking, str), (tool_calls, list))
        ):
            self.fail(FAILURE_STREAM_DECODE, "a stream chunk had an unexpected shape")
            return
        self.chunks += 1
        output = False
        if content:
            output = True
            self.content_chunks += 1
            self.content_characters += len(content)
            if self.first_content_at is None:
                self.first_content_at = now
            if self.keep_text:
                self.content_parts.append(content)
        if thinking:
            output = True
            self.thinking_chunks += 1
            self.thinking_characters += len(thinking)
            if self.first_thinking_at is None:
                self.first_thinking_at = now
        if tool_calls:
            output = True
            self.tool_call_chunks += 1
        if output:
            self.output_chunk_times.append(now)
            if self.first_output_at is None:
                self.first_output_at = now
        if done:
            self.final = chunk
            self.final_at = now

    def done_reason(self) -> str | None:
        return None if self.final is None else _label(self.final.get("done_reason"))

    def finish_code(self) -> str | None:
        reason = self.done_reason()
        return None if reason is None else FINISH_CODES.get(reason)

    def ms(self, at: float | None) -> float | None:
        return None if at is None else max(0.0, at - self.start) * 1000.0

    def response_text(self) -> str:
        return "".join(self.content_parts)


def stream_request(
    settings: OllamaSettings,
    *,
    prompt: str,
    system_prompt: str | None,
    transport: StreamingTransport,
    credential: str | None,
    clock: Callable[[], float],
    keep_text: bool,
) -> tuple[OllamaStream, bytes]:
    """Send one streaming request and read it to its final chunk.

    Returns the stream and the encoded body. Transport and protocol
    failures are recorded on the stream; nothing raises.
    """
    body = encode_body(request_body(settings, prompt, system_prompt))
    request = HTTPRequest(
        url=settings.api_url,
        method="POST",
        headers=_headers(credential, accept="application/x-ndjson"),
        body=body,
        timeout_seconds=settings.request_timeout_seconds,
    )
    stream = OllamaStream(
        api=settings.api, start=clock(), keep_text=keep_text, credential=credential
    )
    try:
        response = transport.open_stream(request)
    except TransportTimeout:
        stream.fail(FAILURE_TIMEOUT, "the request timed out before a response")
    except TransportConnectionError:
        stream.fail(FAILURE_CONNECTION, "the server could not be reached")
    else:
        try:
            stream.status = response.status_code
            if stream.status != 200:
                _record_http_error(stream, _read_body(response, _MAX_ERROR_BODY_BYTES))
            else:
                _read_stream(response, stream, clock)
        except TransportTimeout:
            stream.fail(FAILURE_TIMEOUT, "the stream exceeded the request timeout")
        except TransportConnectionError:
            stream.fail(FAILURE_CONNECTION, "the connection failed mid-stream")
        except NDJSONDecodeError as exc:
            stream.fail(FAILURE_STREAM_DECODE, str(exc))
        finally:
            response.close()
    stream.end = clock()
    if stream.failure is None and stream.final is None:
        stream.fail(FAILURE_STREAM_TRUNCATED, "the stream ended before the done chunk")
    if stream.failure is None and stream.finish_code() is None:
        stream.fail(
            FAILURE_UNRECOGNIZED_DONE,
            f"done_reason {stream.done_reason()!r} is not stop or length",
        )
    return stream, body


def _record_http_error(stream: OllamaStream, body: bytes) -> None:
    detail = "no error message"
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        payload = None
    if isinstance(payload, dict) and "error" in payload:
        detail = _error_text(payload["error"], stream.credential)
    stream.fail(FAILURE_HTTP_STATUS, f"HTTP {stream.status}: {detail}")


def _read_stream(
    response: Any, stream: OllamaStream, clock: Callable[[], float]
) -> None:
    decoder = NDJSONDecoder()
    for chunk in response.iter_bytes():
        for line in decoder.feed(chunk):
            stream.handle(line, clock())
            if stream.finished:
                return
    for line in decoder.close():
        stream.handle(line, clock())
        if stream.finished:
            return


@dataclass(frozen=True)
class OllamaServerMetrics:
    """The final chunk's ``Metrics`` fields, as sent.

    ``raw`` holds every field the server sent in a usable shape, durations
    in nanoseconds. An absent field is left out, never filled with zero. A
    field in an unusable shape is dropped and named in
    ``malformed_fields``.
    """

    raw: Mapping[str, int]
    malformed_fields: tuple[str, ...] = ()

    def count(self, name: str) -> int | None:
        return self.raw.get(name)

    def ms(self, name: str) -> float | None:
        value = self.raw.get(name)
        return None if value is None else value / _NS_PER_MS

    @property
    def reported(self) -> bool:
        return bool(self.raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": SOURCE_OLLAMA,
            "provenance": PROVENANCE,
            "raw": {
                name: self.raw.get(name) for name in (*DURATION_FIELDS, *COUNT_FIELDS)
            },
            "raw_duration_unit": "ns",
            "milliseconds": {name: self.ms(name) for name in DURATION_FIELDS},
            "absent_fields": [
                name
                for name in (*DURATION_FIELDS, *COUNT_FIELDS)
                if name not in self.raw and name not in self.malformed_fields
            ],
            "malformed_fields": list(self.malformed_fields),
            "definitions": dict(FIELD_DEFINITIONS),
        }


def parse_server_metrics(final: Mapping[str, Any]) -> OllamaServerMetrics:
    raw: dict[str, int] = {}
    malformed: list[str] = []
    for name in (*DURATION_FIELDS, *COUNT_FIELDS):
        if name not in final:
            continue
        value = final[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= _MAX_EXACT_INT
        ):
            malformed.append(name)
            continue
        raw[name] = value
    return OllamaServerMetrics(raw=raw, malformed_fields=tuple(malformed))


def _provider_ms(value: float | None) -> Measurement | None:
    if value is None:
        return None
    return Measurement(
        value=value, provenance=MetricProvenance.PROVIDER_REPORTED, unit="ms"
    )


@dataclass(frozen=True)
class OllamaCollectionConfig:
    """One request to collect, and where to write it."""

    run_id: str
    settings: OllamaSettings
    prompt: str
    output_dir: Path
    command_argv: tuple[str, ...]
    system_prompt: str | None = None
    model_revision: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise OllamaError("run_id must be non-empty")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise OllamaError("the prompt must be non-empty text")
        if self.system_prompt is not None and not isinstance(self.system_prompt, str):
            raise OllamaError("the system prompt must be text")
        if not self.command_argv or not all(self.command_argv):
            raise OllamaError("command_argv must contain non-empty argument strings")


def workload_hash(config: OllamaCollectionConfig) -> str:
    return sha256_text(
        json.dumps(
            {
                "api": config.settings.api,
                "system_prompt": config.system_prompt,
                "prompt": config.prompt,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
    )


@dataclass(frozen=True)
class OllamaCollectionResult:
    record: ExperimentRecord
    evidence: dict[str, Any]
    response_text: str


def _gaps_ms(times: Iterable[float]) -> list[float]:
    ordered = list(times)
    return [
        max(0.0, later - earlier) * 1000.0
        for earlier, later in zip(ordered, ordered[1:], strict=False)
    ]


def build_evidence(
    config: OllamaCollectionConfig,
    stream: OllamaStream,
    metrics: OllamaServerMetrics | None,
    version: VersionProbe,
    *,
    body: bytes,
) -> dict[str, Any]:
    gaps = _latency_distribution(_gaps_ms(stream.output_chunk_times))
    final = stream.final or {}
    return {
        "kind": EVIDENCE_KIND,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "run_id": config.run_id,
        "request": {
            **request_plan(config.settings),
            "body_sha256": sha256_bytes(body),
        },
        "server": version.to_dict(),
        "response": {
            "http_status": stream.status,
            "model": _label(final.get("model")),
            "done_reason": stream.done_reason(),
            "finish_reason_code": stream.finish_code(),
        },
        "client_timing": {
            "provenance": MetricProvenance.MEASURED_WALL_CLOCK.value,
            "origin": "just before the request was sent",
            "ttft_ms": stream.ms(stream.first_output_at),
            "ttft_source": (
                None if stream.first_output_at is None else TTFT_FIRST_OUTPUT_CHUNK
            ),
            "first_content_ms": stream.ms(stream.first_content_at),
            "first_thinking_ms": stream.ms(stream.first_thinking_at),
            "done_chunk_ms": stream.ms(stream.final_at),
            "e2e_ms": stream.ms(stream.end),
            "inter_output_chunk_gap_ms": None if gaps is None else gaps.to_dict(),
            "note": (
                "gaps are between NDJSON chunks that carried output; Ollama "
                "usually sends one token per chunk but that is not guaranteed"
            ),
        },
        "stream": {
            "chunks": stream.chunks,
            "output_chunks": len(stream.output_chunk_times),
            "content_chunks": stream.content_chunks,
            "thinking_chunks": stream.thinking_chunks,
            "tool_call_chunks": stream.tool_call_chunks,
            "content_characters": stream.content_characters,
            "thinking_characters": stream.thinking_characters,
        },
        "server_reported": None if metrics is None else metrics.to_dict(),
        "failure": (
            None
            if stream.failure is None
            else {"category": stream.failure, "message": stream.failure_message}
        ),
    }


def build_record(
    config: OllamaCollectionConfig,
    stream: OllamaStream,
    metrics: OllamaServerMetrics | None,
    version: VersionProbe,
    *,
    started_at: str,
) -> ExperimentRecord:
    settings = config.settings
    prompt_tokens = None if metrics is None else metrics.count("prompt_eval_count")
    generated = None if metrics is None else metrics.count("eval_count")
    record = ExperimentRecord(
        run_id=config.run_id,
        started_at=started_at,
        ended_at=utc_now_iso(),
        # The client host. Ollama may run elsewhere, and its hardware is not
        # observable from here, so no accelerator is claimed.
        platform=record_platform(accelerator=None, extra_packages=()),
        model=ModelInfo(model_id=settings.model, model_revision=config.model_revision),
        runtime=RuntimeInfo(
            name=RUNTIME_NAME, version=version.version, backend=None, git_revision=None
        ),
        command=CommandInfo(
            argv=config.command_argv,
            config_hash=settings_hash(settings),
            workload_hash=workload_hash(config),
        ),
        repetition=RepetitionInfo(
            warmup_repetitions=0,
            measured_repetitions=1,
            repetition_index=0,
            seed=settings.seed,
        ),
        tokens=TokenCounts(
            input_tokens=prompt_tokens,
            context_tokens=prompt_tokens,
            generated_tokens=generated,
            provenance=(
                MetricProvenance.PROVIDER_REPORTED
                if prompt_tokens is not None or generated is not None
                else None
            ),
        ),
        timing=TimingMetrics(
            # The server's own phase timers. prefill covers only the prompt
            # tokens that were not cached.
            model_load=_provider_ms(
                None if metrics is None else metrics.ms("load_duration")
            ),
            tokenize=None,
            prefill=_provider_ms(
                None if metrics is None else metrics.ms("prompt_eval_duration")
            ),
            decode=_provider_ms(
                None if metrics is None else metrics.ms("eval_duration")
            ),
            total=milliseconds(stream.start, stream.end),
        ),
        memory=MemoryMetrics(),
        outcome=OutcomeInfo(success=stream.failure is None),
        error=(
            None
            if stream.failure is None
            else ErrorInfo(
                category=stream.failure, message=stream.failure_message or ""
            )
        ),
    )
    record.validate()
    return record


def collect_ollama(
    config: OllamaCollectionConfig,
    *,
    transport: StreamingTransport,
    environ: Mapping[str, str],
    clock: Callable[[], float] = time.perf_counter,
) -> OllamaCollectionResult:
    """Stream one request and write the evidence.

    Request failures become a failed record. A missing credential raises
    before anything is sent. ``record.json`` is written last, so its
    presence means the other files of the run are in place.
    """
    settings = config.settings
    credential = resolve_credential(settings, environ)
    version = probe_version(settings, transport=transport, credential=credential)
    started_at = utc_now_iso()
    stream, body = stream_request(
        settings,
        prompt=config.prompt,
        system_prompt=config.system_prompt,
        transport=transport,
        credential=credential,
        clock=clock,
        keep_text=True,
    )
    metrics = None if stream.final is None else parse_server_metrics(stream.final)
    record = build_record(config, stream, metrics, version, started_at=started_at)
    evidence = build_evidence(config, stream, metrics, version, body=body)
    response_text = stream.response_text()

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "record.json").unlink(missing_ok=True)
    atomic_write_text(output_dir / "response.txt", response_text)
    atomic_write_text(
        output_dir / EVIDENCE_FILE_NAME,
        json.dumps(evidence, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    manifest = collect_environment_manifest(extra_packages=())
    atomic_write_text(output_dir / "environment.json", manifest.to_json() + "\n")
    record.write_json(output_dir / "record.json")
    return OllamaCollectionResult(
        record=record, evidence=evidence, response_text=response_text
    )


__all__ = [
    "API_CHAT",
    "API_GENERATE",
    "API_PATHS",
    "DEFAULT_BASE_URL",
    "EVIDENCE_FILE_NAME",
    "EVIDENCE_KIND",
    "FAILURE_UNRECOGNIZED_DONE",
    "FINISH_CODES",
    "NDJSONDecoder",
    "OLLAMA_COMMIT",
    "OllamaCollectionConfig",
    "OllamaCollectionResult",
    "OllamaError",
    "OllamaServerMetrics",
    "OllamaSettings",
    "OllamaStream",
    "TTFT_FIRST_OUTPUT_CHUNK",
    "VersionProbe",
    "collect_ollama",
    "parse_server_metrics",
    "probe_version",
    "request_body",
    "request_plan",
    "resolve_credential",
    "stream_request",
]
