"""Load worker for llama.cpp's native ``/completion`` endpoint.

``llama-server`` also serves an OpenAI-compatible API, which ``sweep-api``
already drives. This worker uses the native endpoint instead, because its
final stream chunk carries the server's own prompt and decode timers, its
slot id and the reason generation stopped. Rows have the same base fields
as the OpenAI-compatible worker, so goodput and the scheduling diagnostics
read them unchanged.

Wire format checked against ggml-org/llama.cpp at :data:`LLAMA_CPP_COMMIT`
(``tools/server/server-context.cpp``, ``server-task.cpp``,
``server-common.cpp`` and ``README.md``):

* ``POST /completion`` with ``stream: true`` answers with SSE ``data:``
  lines, one JSON object each. Partial chunks have ``stop: false``. The
  last chunk has ``stop: true`` with ``tokens_predicted``,
  ``tokens_evaluated``, ``stop_type`` (``eos``, ``word``, ``limit`` or
  ``none``), ``truncated``, ``id_slot`` and ``timings``.
* The native stream sends no ``[DONE]``. It ends after the ``stop: true``
  chunk, so a stream that closes before that chunk was cut short.
* Keepalive pings are SSE comments. Errors after the stream has started
  arrive as ``data: {"error": {...}}``.
* ``GET /props`` reports ``total_slots`` (the ``--parallel`` setting) and
  ``build_info``.

``llama-bench`` runs its own fixed workload and cannot accept dispatched
requests, so it is not a load worker.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..collectors._shared import sha256_text
from ..collectors.openai_api import (
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
    _validate_endpoint,
    parse_endpoint,
)
from ..collectors.server_reported import SOURCE_LLAMA_CPP
from ..collectors.sse import SSEDecodeError, SSEDecoder
from ..schema import MetricProvenance
from .api_worker import (
    ROW_FIELD_PROVENANCE,
    SERVER_REPORTED_ROW_FIELDS,
    ApiRequestSpec,
    RequestRow,
    _tpot,
)
from .dispatcher import DispatchContext
from .sweep import ATTRIBUTION, SweepPoint

LLAMA_CPP_REPOSITORY = "ggml-org/llama.cpp"
LLAMA_CPP_COMMIT = "f8caa5ea30b5775512906ae3b7ee53b21a7a6e1d"
COMPLETION_PATH = "/completion"
PROPS_PATH = "/props"

FINISH_CODES: Mapping[str, str] = {"eos": "stop", "word": "stop", "limit": "length"}
"""``stop_type`` values that end a generation normally, mapped to the
OpenAI-style codes the other worker reports. ``limit`` covers both
``n_predict`` and a full context; ``truncated`` tells them apart."""

FAILURE_UNRECOGNIZED_STOP = "unrecognized_stop_type"

TTFT_FIRST_TOKEN_CHUNK = "first_token_chunk"
TTFT_FIRST_CONTENT_CHUNK = "first_content_chunk"

_MAX_ERROR_BODY_BYTES = 64 * 1024
_MAX_PROPS_BYTES = 1024 * 1024
_MAX_LABEL_CHARS = 64
_MAX_BUILD_INFO_CHARS = 200

_LLAMA_CPP_SERVER_REPORTED = ("server_draft_tokens", "server_accepted_draft_tokens")

LLAMA_CPP_FIELD_PROVENANCE: dict[str, str] = {
    **{
        k: v
        for k, v in ROW_FIELD_PROVENANCE.items()
        if k not in SERVER_REPORTED_ROW_FIELDS or k in _LLAMA_CPP_SERVER_REPORTED
    },
    "server_prompt_n": MetricProvenance.PROVIDER_REPORTED.value,
    "server_prompt_ms": MetricProvenance.MEASURED_NATIVE.value,
    "server_predicted_ms": MetricProvenance.MEASURED_NATIVE.value,
}
"""Base row fields keep the HTTP worker's labels. ``output_tokens`` is
``tokens_predicted``, ``prompt_tokens`` is ``tokens_evaluated`` and
``cached_prompt_tokens`` is ``timings.cache_n``, all reported by the
server. ``server_prompt_ms`` and ``server_predicted_ms`` are the server's
own prompt and decode timers, the same values the llama.cpp log parser
labels ``measured_native``. ``server_draft_tokens`` and
``server_accepted_draft_tokens`` are ``timings.draft_n`` and
``timings.draft_n_accepted``, which the server sends only when speculative
decoding drafted at least one token (``server_slot_stats::to_json`` in
``tools/server/server-common.cpp`` L146-167 at ``LLAMA_CPP_COMMIT``). The
other ``server_*`` fields of the HTTP worker are never filled here and
carry no label."""


class LlamaCppLoadError(ValueError):
    """A llama.cpp load sweep cannot be configured or the server cannot serve it."""


@dataclass(frozen=True)
class LlamaCppConfig:
    """One sweep's requests to a ``llama-server``.

    ``base_url`` is the server root, with any ``--api-prefix``; the worker
    appends ``/completion`` and ``/props``. Sampling values left as ``None``
    are not sent, so the server's own defaults apply and are not guessed.
    """

    run_id: str
    base_url: str
    n_predict: int
    cache_prompt: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    seed: int | None = None
    request_timeout_seconds: float = 120.0
    credential_env_var: str | None = None

    def __post_init__(self) -> None:
        try:
            _validate_endpoint(self.base_url)
        except (ValueError, OpenAIStreamCollectorError) as exc:
            raise LlamaCppLoadError(
                str(exc).replace("endpoint", "server URL")
            ) from None
        parts = parse_endpoint(self.base_url)
        if parts.query:
            raise LlamaCppLoadError("server URL must not contain a query string")
        if parts.path.rstrip("/").endswith(COMPLETION_PATH):
            raise LlamaCppLoadError(
                "give the server root, for example http://127.0.0.1:8080, not "
                "the /completion URL; the worker adds the path"
            )
        if (
            isinstance(self.n_predict, bool)
            or not isinstance(self.n_predict, int)
            or self.n_predict < 1
        ):
            raise LlamaCppLoadError(
                "--max-output-tokens must be a positive integer. llama.cpp "
                "defaults n_predict to -1 (no limit), so it is always sent"
            )
        if self.top_k is not None and self.top_k < 0:
            raise LlamaCppLoadError("--top-k must be at least 0")
        if not self.request_timeout_seconds > 0:
            raise LlamaCppLoadError("--request-timeout must be positive")

    def url(self, path: str) -> str:
        return self.base_url.rstrip("/") + path

    @property
    def completion_url(self) -> str:
        return self.url(COMPLETION_PATH)

    @property
    def props_url(self) -> str:
        return self.url(PROPS_PATH)


def request_body(config: LlamaCppConfig, prompt: str) -> dict[str, Any]:
    """The ``/completion`` body. The prompt is sent raw, with no chat template."""
    body: dict[str, Any] = {
        "prompt": prompt,
        "n_predict": config.n_predict,
        "stream": True,
        "return_tokens": True,
        "cache_prompt": config.cache_prompt,
    }
    for key, value in (
        ("temperature", config.temperature),
        ("top_p", config.top_p),
        ("top_k", config.top_k),
        ("seed", config.seed),
    ):
        if value is not None:
            body[key] = value
    return body


def resolve_credential(
    config: LlamaCppConfig, environ: Mapping[str, str]
) -> str | None:
    """The API key from the named environment variable, or ``None`` when none is named."""
    if config.credential_env_var is None:
        return None
    value = environ.get(config.credential_env_var)
    if value is None:
        raise LlamaCppLoadError(
            "the environment variable named by --api-key-env is not set in "
            "this environment"
        )
    credential = value.strip()
    if not credential:
        raise LlamaCppLoadError(
            "the environment variable named by --api-key-env is empty"
        )
    try:
        _assert_header_safe_credential(credential, config.credential_env_var)
    except OpenAIStreamCollectorError as exc:
        raise LlamaCppLoadError(str(exc)) from None
    return credential


def _headers(credential: str | None, *, accept: str) -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": accept}
    if credential is not None:
        headers["Authorization"] = f"Bearer {credential}"
    return headers


def request_plan(config: LlamaCppConfig) -> dict[str, Any]:
    """What every request sends, without the prompt or any credential."""
    body = request_body(config, "")
    body.pop("prompt")
    return {
        "endpoint": config.completion_url,
        "method": "POST",
        "api": "llama.cpp native /completion (not OpenAI-compatible)",
        "body_without_prompt": body,
        "prompt_field": "prompt, raw text with no chat template applied",
        "header_names": sorted(
            ["Accept", "Content-Type"]
            + ([] if config.credential_env_var is None else ["Authorization"])
        ),
        "credential_env_var": config.credential_env_var,
        "request_timeout_seconds": config.request_timeout_seconds,
        "upstream_reference": upstream_reference(),
    }


def upstream_reference() -> dict[str, str]:
    return {"repository": LLAMA_CPP_REPOSITORY, "commit": LLAMA_CPP_COMMIT}


@dataclass(frozen=True)
class LlamaCppServerProps:
    """What ``GET /props`` said about the server, before the sweep."""

    total_slots: int
    build_info: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_slots": self.total_slots,
            "build_info": self.build_info,
            "source": f"GET {PROPS_PATH}",
            "provenance": MetricProvenance.PROVIDER_REPORTED.value,
        }


def _read_body(response: Any, limit: int) -> bytes:
    data = bytearray()
    for chunk in response.iter_bytes():
        data.extend(chunk)
        if len(data) > limit:
            break
    return bytes(data[:limit])


def probe_props(
    config: LlamaCppConfig,
    *,
    transport: StreamingTransport,
    environ: Mapping[str, str],
) -> LlamaCppServerProps:
    """Read ``total_slots`` and ``build_info``. Refuses when they are not there."""
    request = HTTPRequest(
        url=config.props_url,
        method="GET",
        headers=_headers(
            resolve_credential(config, environ), accept="application/json"
        ),
        body=b"",
        timeout_seconds=config.request_timeout_seconds,
    )
    try:
        response = transport.open_stream(request)
    except (TransportTimeout, TransportConnectionError) as exc:
        raise LlamaCppLoadError(
            f"GET {PROPS_PATH} failed ({type(exc).__name__}); is llama-server "
            "running at the server URL?"
        ) from None
    try:
        status = response.status_code
        body = _read_body(response, _MAX_PROPS_BYTES)
    except (TransportTimeout, TransportConnectionError) as exc:
        raise LlamaCppLoadError(
            f"GET {PROPS_PATH} failed while reading ({type(exc).__name__})"
        ) from None
    finally:
        response.close()
    if status != 200:
        raise LlamaCppLoadError(
            f"GET {PROPS_PATH} returned HTTP {status}. The slot count is needed "
            "to check the load against what the server can run at once"
        )
    try:
        props = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise LlamaCppLoadError(f"GET {PROPS_PATH} did not return JSON") from None
    if not isinstance(props, dict):
        raise LlamaCppLoadError(f"GET {PROPS_PATH} did not return a JSON object")
    slots = props.get("total_slots")
    if isinstance(slots, bool) or not isinstance(slots, int) or slots < 1:
        raise LlamaCppLoadError(
            f"GET {PROPS_PATH} has no positive integer total_slots, so the "
            "server's parallel capacity is unknown; this does not look like "
            "llama-server"
        )
    build = props.get("build_info")
    return LlamaCppServerProps(
        total_slots=slots,
        build_info=(
            build[:_MAX_BUILD_INFO_CHARS] if isinstance(build, str) and build else None
        ),
    )


def max_point_in_flight(point: SweepPoint) -> int:
    """The most requests a point can have in flight at once."""
    return point.policy.max_in_flight


def check_slots(
    points: Sequence[SweepPoint], total_slots: int, *, allow_server_queue: bool
) -> None:
    """Refuse load that would queue inside the server unless that is asked for.

    With more requests in flight than slots, ``llama-server`` holds the extra
    ones in its queue and serves them one after another. Their latency then
    includes that wait, and nothing in the client rows shows it.
    """
    if allow_server_queue:
        return
    for point in points:
        in_flight = max_point_in_flight(point)
        if in_flight > total_slots:
            raise LlamaCppLoadError(
                f"point {point.label} can have {in_flight} requests in flight "
                f"but the server has {total_slots} slots (total_slots from GET "
                f"{PROPS_PATH}). The extra requests would wait in the server "
                "queue and their latency would include that wait. Lower the "
                "load or --max-in-flight, start llama-server with a larger "
                "--parallel, or pass --allow-server-queue to measure queueing "
                "on purpose"
            )


def backend_block(
    config: LlamaCppConfig,
    props: LlamaCppServerProps | None,
    *,
    allow_server_queue: bool,
) -> dict[str, Any]:
    return {
        "name": "llama.cpp",
        "api": "native /completion, not OpenAI-compatible",
        "upstream_reference": upstream_reference(),
        "server": (
            props.to_dict()
            if props is not None
            else {"status": "not probed", "reason": "dry run; nothing was sent"}
        ),
        "max_supported_in_flight": None if props is None else props.total_slots,
        "server_queueing_allowed": allow_server_queue,
        "concurrency_limit": (
            "requests in flight are refused above total_slots unless "
            "--allow-server-queue is given"
        ),
        "chat_template_applied": False,
        "cache_prompt": config.cache_prompt,
    }


def llama_cpp_attribution(backend: Mapping[str, Any]) -> dict[str, Any]:
    attribution = dict(ATTRIBUTION)
    attribution["cannot_claim"] = [
        *ATTRIBUTION["cannot_claim"],
        (
            "comparability with sweep-api rows from the same server: prompts "
            "go to /completion raw, without the chat template"
        ),
    ]
    attribution["backend"] = dict(backend)
    return attribution


@dataclass(frozen=True)
class LlamaCppRequestRow(RequestRow):
    """A ``RequestRow`` with the server's own accounting from the final chunk.

    ``finish_reason`` holds the raw ``stop_type``. ``server_prompt_ms`` and
    ``server_predicted_ms`` are the server's timers; they are not the
    client's TTFT and decode time, which also include queueing and network.
    """

    truncated: bool | None = None
    id_slot: int | None = None
    server_prompt_n: int | None = None
    server_prompt_ms: float | None = None
    server_predicted_ms: float | None = None


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _millis(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def _label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value[:_MAX_LABEL_CHARS]


@dataclass
class _Stream:
    """Reads one ``/completion`` stream. Holds no generated text."""

    start: float
    first_token_at: float | None = None
    first_content_at: float | None = None
    content_chunks: int = 0
    final: dict[str, Any] | None = None
    final_at: float | None = None
    failure: str | None = None

    def handle(self, data: str, now: float) -> None:
        if data == "[DONE]":
            return
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            self.failure = FAILURE_STREAM_DECODE
            return
        if not isinstance(chunk, dict):
            self.failure = FAILURE_STREAM_DECODE
            return
        if "error" in chunk:
            self.failure = FAILURE_PROVIDER_ERROR
            return
        tokens = chunk.get("tokens")
        if isinstance(tokens, list) and tokens and self.first_token_at is None:
            self.first_token_at = now
        content = chunk.get("content")
        if isinstance(content, str) and content:
            self.content_chunks += 1
            if self.first_content_at is None:
                self.first_content_at = now
        if chunk.get("stop") is True:
            self.final = chunk
            self.final_at = now


def make_llama_cpp_sender(
    config: LlamaCppConfig,
    *,
    transport: StreamingTransport,
    environ: Mapping[str, str],
) -> Callable[[ApiRequestSpec, DispatchContext], LlamaCppRequestRow]:
    """A thread-safe ``send`` for ``dispatch`` that streams one ``/completion``.

    ``transport`` must be safe to use from several threads at once.
    ``UrllibStreamingTransport`` is: it opens a new connection per request.
    Specs with a system prompt, ``max_tokens`` or ``extra_body`` are refused;
    the body is fixed by ``config`` so every request of a sweep is alike.
    """
    credential = resolve_credential(config, environ)

    def send(spec: ApiRequestSpec, context: DispatchContext) -> LlamaCppRequestRow:
        if (
            spec.system_prompt is not None
            or spec.max_tokens is not None
            or spec.extra_body is not None
        ):
            raise LlamaCppLoadError(
                "llama.cpp /completion specs take only a prompt and a run id suffix"
            )
        suffix = (
            spec.run_id_suffix
            if spec.run_id_suffix is not None
            else f"{context.index:06d}"
        )
        body = request_body(config, spec.prompt)
        encoded = json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")
        request = HTTPRequest(
            url=config.completion_url,
            method="POST",
            headers=_headers(credential, accept="text/event-stream"),
            body=encoded,
            timeout_seconds=config.request_timeout_seconds,
        )
        start = context.clock()
        stream = _Stream(start=start)
        status: int | None = None
        try:
            response = transport.open_stream(request)
        except TransportTimeout:
            stream.failure = FAILURE_TIMEOUT
        except TransportConnectionError:
            stream.failure = FAILURE_CONNECTION
        else:
            try:
                status = response.status_code
                if status != 200:
                    _read_body(response, _MAX_ERROR_BODY_BYTES)
                    stream.failure = FAILURE_HTTP_STATUS
                else:
                    _read_stream(response, stream, context.clock)
            except TransportTimeout:
                stream.failure = FAILURE_TIMEOUT
            except TransportConnectionError:
                stream.failure = FAILURE_CONNECTION
            except SSEDecodeError:
                stream.failure = FAILURE_STREAM_DECODE
            finally:
                response.close()
        end = context.clock()
        if stream.failure is None and stream.final is None:
            stream.failure = FAILURE_STREAM_TRUNCATED
        return _row(
            stream,
            context,
            run_id=f"{config.run_id}-{suffix}",
            status=status,
            end=stream.final_at if stream.final_at is not None else end,
            prompt=spec.prompt,
            workload_hash=hashlib.sha256(encoded).hexdigest(),
        )

    return send


def _read_stream(response: Any, stream: _Stream, clock: Callable[[], float]) -> None:
    decoder = SSEDecoder()
    for chunk in response.iter_bytes():
        for event in decoder.feed(chunk):
            stream.handle(event.data, clock())
            if stream.failure is not None or stream.final is not None:
                return
    for event in decoder.close():
        stream.handle(event.data, clock())
        if stream.failure is not None or stream.final is not None:
            return


def _row(
    stream: _Stream,
    context: DispatchContext,
    *,
    run_id: str,
    status: int | None,
    end: float,
    prompt: str,
    workload_hash: str,
) -> LlamaCppRequestRow:
    final = stream.final if stream.failure is None else None
    stop_type = None if final is None else _label(final.get("stop_type"))
    finish_code = None if stop_type is None else FINISH_CODES.get(stop_type)
    failure = stream.failure
    if failure is None and finish_code is None:
        failure = FAILURE_UNRECOGNIZED_STOP
    timings = {} if final is None else final.get("timings")
    if not isinstance(timings, dict):
        timings = {}

    def ms(at: float | None) -> float | None:
        return None if at is None else (at - stream.start) * 1000.0

    if stream.first_token_at is not None:
        ttft_ms, ttft_source = ms(stream.first_token_at), TTFT_FIRST_TOKEN_CHUNK
    elif stream.first_content_at is not None:
        ttft_ms, ttft_source = ms(stream.first_content_at), TTFT_FIRST_CONTENT_CHUNK
    else:
        ttft_ms, ttft_source = None, None
    e2e_ms = ms(end)
    output_tokens = None if final is None else _count(final.get("tokens_predicted"))
    tpot_ms, tpot_status = _tpot(ttft_ms, e2e_ms, output_tokens)
    truncated = None if final is None else final.get("truncated")
    return LlamaCppRequestRow(
        index=context.index,
        sent=True,
        run_id=run_id,
        scheduled_offset_s=context.scheduled_offset_s,
        dispatched_offset_s=context.dispatched_offset_s,
        completed_offset_s=end - context.clock_origin,
        in_flight_at_dispatch=context.in_flight_at_dispatch,
        success=failure is None,
        failure_category=failure,
        http_status=status,
        finish_reason=stop_type,
        finish_reason_code=finish_code,
        ttft_ms=ttft_ms,
        ttft_source=ttft_source,
        first_content_ms=ms(stream.first_content_at),
        e2e_ms=e2e_ms,
        output_tokens=output_tokens,
        prompt_tokens=None if final is None else _count(final.get("tokens_evaluated")),
        cached_prompt_tokens=_count(timings.get("cache_n")),
        tpot_ms=tpot_ms,
        tpot_status=tpot_status,
        content_delta_count=stream.content_chunks,
        prompt_sha256=sha256_text(prompt),
        workload_hash=workload_hash,
        truncated=truncated if isinstance(truncated, bool) else None,
        id_slot=None if final is None else _count(final.get("id_slot")),
        server_prompt_n=_count(timings.get("prompt_n")),
        server_prompt_ms=_millis(timings.get("prompt_ms")),
        server_predicted_ms=_millis(timings.get("predicted_ms")),
        **_draft_fields(timings),
    )


def _draft_fields(timings: Mapping[str, Any]) -> dict[str, Any]:
    drafted = _count(timings.get("draft_n"))
    accepted = _count(timings.get("draft_n_accepted"))
    if drafted is None and accepted is None:
        return {}
    # No acceptance rate is derived: the server reports the two counts and
    # this worker does not divide them for it.
    return {
        "server_metrics_source": SOURCE_LLAMA_CPP,
        "server_draft_tokens": drafted,
        "server_accepted_draft_tokens": accepted,
    }


__all__ = [
    "FINISH_CODES",
    "LLAMA_CPP_COMMIT",
    "LLAMA_CPP_FIELD_PROVENANCE",
    "LlamaCppConfig",
    "LlamaCppLoadError",
    "LlamaCppRequestRow",
    "LlamaCppServerProps",
    "backend_block",
    "check_slots",
    "llama_cpp_attribution",
    "make_llama_cpp_sender",
    "probe_props",
    "request_body",
    "request_plan",
]
