"""Collect streaming evidence from one OpenAI-compatible chat-completions call.

This collector is provider-neutral. It speaks the OpenAI chat-completions
wire format over an explicit endpoint URL and an explicit model ID, and it
carries provider-specific request fields in a typed
:class:`ProviderExtensions` block rather than pretending they are part of
the OpenAI contract. Z.ai's GLM models are the first profile it is
exercised against; nothing about Z.ai is hardcoded into the request path.

What it measures, and what it only repeats
------------------------------------------
Two different kinds of number come out of a hosted API, and they are kept
apart on purpose:

* **Client-observed timing.** Every timestamp here comes from a monotonic
  clock in this process, around the HTTP request. It therefore includes
  network transport, TLS, provider queueing and server compute, and it
  cannot decompose them. ``timing.prefill`` is time-to-first-*content*
  token as observed by this client, not a server-side prefill
  measurement. The response-header and first-body-chunk offsets are
  persisted separately so a reader can see roughly how much of that
  window was spent before generation could have started, instead of
  attributing all of it to the model.
* **Provider-reported usage.** Token counts come from the provider's
  ``usage`` object. They are recorded under
  ``MetricProvenance.PROVIDER_REPORTED`` and are never relabeled as a
  local measurement. When the provider omits a field it stays ``None``;
  a missing count is never read as zero.

Because SSE deltas are not guaranteed to carry exactly one token each,
this module never calls a delta a token. It reports content *delta*
counts and a content *delta* rate, and it computes a token rate only from
provider-reported completion tokens, labeled as such.

Privacy
-------
The credential is read from an explicitly named environment variable, is
never written to any artifact, is never placed in a command
reconstruction, and is never echoed back through provider messages (every
persisted string is redacted against it). Prompt text is hashed, not
copied. Reasoning/"thinking" text is never persisted: only whether it was
returned, how many deltas arrived and how many characters they held.

Failures
--------
HTTP errors, timeouts, connection failures, malformed SSE or JSON,
provider error payloads and streams that produced no content are all
persisted as failure-shaped canonical records plus failure-shaped API
evidence. There is no success-shaped fallback and no retry: one
invocation is one attempt, so that a recorded attempt always corresponds
to exactly one request.
"""

from __future__ import annotations

import base64
import http.client
import json
import math
import os
import re
import time
import unicodedata
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from statistics import median
from typing import Any, Protocol
from urllib.parse import (
    SplitResult,
    parse_qsl,
    unquote,
    unquote_plus,
    urlsplit,
    urlunsplit,
)

from ..manifest import collect_environment_manifest
from ..schema import (
    CommandInfo,
    ErrorInfo,
    ExperimentRecord,
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
from .sse import SSEDecodeError, SSEDecoder, SSEEvent

API_EVIDENCE_SCHEMA_VERSION = "1"
ARTIFACT_MANIFEST_NAME = "artifacts.json"
ARTIFACT_MANIFEST_SCHEMA_VERSION = "1"

RUNTIME_NAME = "openai-compatible-api"
"""``RuntimeInfo.name`` for every record this collector writes."""

#: Documented ``reasoning_effort`` levels for Z.ai's GLM-5.3 and
#: GLM-5.3-Flash. Z.ai's chat-completions reference accepts a wider enum
#: for older models, but states that for these two "only the `low` /
#: `high` / `max` levels are supported" (any other value is an error) and
#: that the default is ``max``. Surfaced for CLI/docs validation only; the
#: collector itself stays provider-neutral and forwards whatever string it
#: is given.
GLM_REASONING_EFFORT_LEVELS: tuple[str, ...] = ("low", "high", "max")

#: Documented ``thinking.type`` values. Z.ai documents that GLM-5.3 and
#: GLM-5.3-Flash accept only ``enabled``.
THINKING_TYPES: tuple[str, ...] = ("enabled", "disabled")

FAILURE_HTTP_STATUS = "http_status"
FAILURE_TIMEOUT = "timeout"
FAILURE_CONNECTION = "connection"
FAILURE_STREAM_DECODE = "stream_decode"
FAILURE_PROVIDER_ERROR = "provider_error_payload"
FAILURE_MISSING_CONTENT = "missing_content"
FAILURE_STREAM_TRUNCATED = "stream_truncated"

# Portable ``finish_reason`` values from the OpenAI chat-completions API.
# A stream that stops without a terminal reason and without ``[DONE]`` was
# cut short, however much content it already delivered.
# https://platform.openai.com/docs/api-reference/chat/streaming
OPENAI_TERMINAL_FINISH_REASONS = frozenset(
    {"stop", "length", "content_filter", "tool_calls", "function_call"}
)

# Z.ai's documented additions for GLM. ``sensitive`` is its analogue of
# OpenAI's ``content_filter``: generation ended by a filter, which is a
# real ending rather than a lost stream. ``network_error`` and
# ``model_context_window_exceeded`` are listed alongside the successful
# reasons, so a run can carry content, one of them, and ``[DONE]`` all at
# once. The sentinel must not be allowed to outrank them: the text on the
# wire is not a completed answer, and publishing it as a success would put
# a truncated or aborted generation into the evidence set as though it
# were whole.
# https://docs.z.ai/api-reference/llm/chat-completion
ZAI_TERMINAL_FINISH_REASONS = frozenset({"sensitive"})
ZAI_FAILURE_FINISH_REASONS = frozenset(
    {"network_error", "model_context_window_exceeded"}
)

# How a ``finish_reason`` is classified. The provider's raw string is
# classified before redaction can touch it, because redaction rewrites
# provider-controlled text and a credential that happens to contain
# ``error`` would turn ``network_error`` into ``network_[REDACTED]`` and
# erase the failure. The classification is derived from a value this
# collector defines, so it stays meaningful whatever the redactor does to
# the text that is persisted alongside it.
_FINISH_TERMINAL = "terminal"
_FINISH_FAILURE = "failure"
_FINISH_UNRECOGNIZED = "unrecognized"

_EVENT_KIND_CONTENT = "content"
_EVENT_KIND_REASONING = "reasoning"
_EVENT_KIND_METADATA = "metadata"

_MAX_ERROR_BODY_BYTES = 64 * 1024
_MAX_PERSISTED_MESSAGE_CHARS = 600
_MAX_PERSISTED_HEADER_CHARS = 128
_REDACTED = "[REDACTED]"
_ENV_VAR_OPTION = "--api-key-env"
_MIN_ENCODED_CREDENTIAL_CHARS = 6

# The one-character escapes JSON and Python emit instead of a numeric one.
# A JSON encoder is free to write "/" as "\/", and every encoder writes a
# backslash and a double quote that way, so these are as much a spelling of
# the credential as \u002F is. The whitespace short escapes (\t, \n, \v, \f,
# \r) are handled by the whitespace run rather than here.
_SHORT_ESCAPES = {'"': '\\"', "\\": "\\\\", "/": "\\/", "'": "\\'"}

# Upper bound on retained per-event timing rows. One row per SSE event with
# no cap lets a chatty provider grow the timeline without limit, and JSON
# serialization amplifies it again. Past the cap the rows stop accumulating
# and the timeline says so; the counters and the first/last offsets every
# derived metric actually reads stay exact, so capping costs per-event
# detail and not a single published number.
DEFAULT_RETAINED_EVENT_LIMIT = 20_000

_PROVIDER_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
#: An exported credential variable name. Deliberately narrower than POSIX,
#: which also permits lowercase. The mechanical response to the ``--api-key``
#: refusal is to swap the flag for ``--api-key-env`` and keep the value, which
#: puts the credential in the name slot. Requiring the universal uppercase
#: convention rejects the shapes keys actually take, ``sk-...`` and
#: ``sk_live_...``, before any of them can reach a diagnostic or an artifact.
_ENV_VAR_PATTERN = re.compile(r"^[A-Z_][A-Z0-9_]*$")
#: Nouns that name a credential outright. Matched as whole components of a
#: parameter name, never as a bare substring: ``sig`` inside ``design`` and
#: ``key`` inside ``monkey`` are not credentials, and refusing those names
#: blocks legitimate endpoints while the deliberately scrubbed diagnostic
#: gives the operator no way to tell which parameter was at fault.
_CREDENTIAL_QUERY_TERMS = frozenset(
    {
        "auth",
        "authorization",
        "bearer",
        "credential",
        "jwt",
        "key",
        "passphrase",
        "passwd",
        "password",
        "pwd",
        "secret",
        "sid",
        "sig",
        "signature",
        "token",
    }
)

#: Words that qualify a credential noun rather than naming one. They matter
#: only for glued compounds such as ``apikey``, where no separator and no
#: case change marks the boundary the tokenizer would otherwise split on.
_CREDENTIAL_QUERY_QUALIFIERS = frozenset(
    {
        "access",
        "account",
        "admin",
        "api",
        "app",
        "application",
        "client",
        "id",
        "master",
        "ocp",
        "primary",
        "private",
        "public",
        "refresh",
        "sas",
        "secondary",
        "service",
        # ``session`` describes a neighbouring noun rather than naming a
        # credential on its own. It cannot be a term: parameter names are
        # tokenized on separators, so ``session_timeout`` yields ``session``
        # as a whole component, and a term there would refuse an ordinary
        # timeout setting with a diagnostic that names nothing.
        "session",
        "shared",
        "subscription",
        "user",
        "x",
    }
)

#: The longest word either table spells. A cover step never has to look
#: further ahead than this, and bounding the lookahead is what keeps the
#: search linear in the length of the component. Unbounded, every reachable
#: offset rescans every remaining substring, so a key made of a few
#: thousand repeated qualifier characters costs seconds of work before the
#: endpoint it belongs to is even rejected.
_MAX_CREDENTIAL_WORD = max(
    len(word) for word in _CREDENTIAL_QUERY_TERMS | _CREDENTIAL_QUERY_QUALIFIERS
)

#: Split on separators, on camel-case boundaries and between letters and
#: digits, so ``x-api-key``, ``apiKey`` and ``key2`` all yield ``key``.
_QUERY_KEY_COMPONENT = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]*|[a-z]+|[0-9]+")
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})

#: Response headers safe to persist verbatim: rate-limit accounting only.
#: Z.ai does not document response headers at all, so this is best effort
#: and frequently yields nothing.
_RATE_LIMIT_HEADER_PREFIXES: tuple[str, ...] = ("x-ratelimit-", "ratelimit-")
_RATE_LIMIT_HEADER_NAMES: frozenset[str] = frozenset({"retry-after"})

#: Response headers checked, in order, for a provider request identifier.
_REQUEST_ID_HEADER_NAMES: tuple[str, ...] = (
    "x-request-id",
    "request-id",
    "x-requestid",
)


class OpenAIStreamCollectorError(RuntimeError):
    """Raised when API collection cannot be configured or started."""


class TransportError(RuntimeError):
    """Base class for transport failures that never reached a response."""


class TransportTimeout(TransportError):
    """The request or the stream exceeded the configured timeout."""


class TransportConnectionError(TransportError):
    """DNS, TCP, TLS or socket failure; no usable HTTP response."""


# --- Transport boundary ------------------------------------------------------


@dataclass(frozen=True)
class HTTPRequest:
    """One outbound request.

    ``headers`` carries the ``Authorization`` value and is therefore
    never persisted, logged or hashed. Only header *names* reach any
    artifact. ``repr`` is overridden so an accidental log line, traceback
    or debugger dump cannot surface the credential.
    """

    url: str
    method: str
    headers: Mapping[str, str] = field(repr=False)
    body: bytes = field(repr=False)
    timeout_seconds: float

    def header_names(self) -> tuple[str, ...]:
        """Header names only. Safe to persist."""
        return tuple(sorted(self.headers))

    def __repr__(self) -> str:
        # The URL is stripped of query *values* for the same reason the
        # persisted command is: a value there is user or provider supplied
        # and the rest of the collector never lets one reach an artifact,
        # so a traceback must not be the one place that does.
        return (
            f"HTTPRequest(url={_endpoint_for_command(self.url)!r}, "
            f"method={self.method!r}, "
            f"header_names={self.header_names()!r}, "
            f"body_bytes={len(self.body)}, "
            f"timeout_seconds={self.timeout_seconds!r})"
        )


class StreamingResponse(Protocol):
    """A response whose body is consumed incrementally."""

    @property
    def status_code(self) -> int: ...

    @property
    def headers(self) -> Mapping[str, str]: ...

    def iter_bytes(self) -> Iterator[bytes]: ...

    def close(self) -> None: ...


class StreamingTransport(Protocol):
    """Injectable HTTP boundary used by the collector and its tests."""

    def open_stream(self, request: HTTPRequest) -> StreamingResponse: ...


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so the ``Authorization`` header is never forwarded.

    ``urllib`` replays request headers on a redirect, which would hand the
    credential to whatever host the response points at. Returning ``None``
    here leaves the 3xx to surface as an ordinary HTTP response, which the
    collector records as a failure with its status code.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> urllib.request.Request | None:
        return None


#: How far to walk a response's wrapper chain looking for its socket. A
#: success is an ``HTTPResponse`` one level above the buffered socket
#: reader; a non-2xx response is an ``HTTPError`` wrapping that response,
#: so it sits one level further out again.
_MAX_RESPONSE_WRAPPER_DEPTH = 4


def _response_socket(raw: object) -> object | None:
    """The socket underneath a response, whichever wrapper shape it has.

    ``urlopen`` returns an ``HTTPResponse`` for a success and raises an
    ``HTTPError`` wrapping one for every other status, so the socket sits
    at a different depth depending on the status. Naming one depth would
    silently do nothing on the other, which is how an error body can keep
    the original per-read timeout while the caller believes the whole
    response is bounded. Walking the chain covers both shapes without
    naming either class, and returns ``None`` rather than raising when the
    object is a stub with no socket at all.
    """
    node: object | None = raw
    for _ in range(_MAX_RESPONSE_WRAPPER_DEPTH):
        if node is None:
            return None
        sock: object | None = getattr(getattr(node, "raw", None), "_sock", None)
        if sock is not None:
            return sock
        node = getattr(node, "fp", None)
    return None


class _UrllibResponse:
    """Adapter over ``http.client.HTTPResponse`` (or an ``HTTPError``)."""

    _CHUNK_SIZE = 8192

    def __init__(
        self,
        raw: Any,
        status_code: int,
        headers: Mapping[str, str],
        *,
        deadline: float | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._raw = raw
        self._status_code = status_code
        self._headers = headers
        self._bytes_read = 0
        self._deadline = deadline
        self._clock = clock

    def _tighten_socket_timeout(self) -> float | None:
        """Shrink the socket timeout to what is left of the whole budget.

        The timeout handed to ``urlopen`` bounds one blocking operation, so
        a read that starts just inside the budget can still block for a
        further full timeout and overshoot the advertised total by close to
        two times. Lowering the socket timeout to the remaining budget makes
        the per-operation bound and the whole-response bound the same
        deadline. Returns the remaining seconds, or ``None`` when no
        deadline is configured.

        The socket is reached through ``_response_socket``, which covers
        both the success and the error wrapper shape, and the adjustment is
        skipped when no socket is reachable, which is what a stubbed
        response in a test looks like. Skipping it costs the tighter bound,
        not correctness: the accumulator still fails the run once the
        deadline passes.
        """
        if self._deadline is None:
            return None
        remaining = self._deadline - self._clock()
        sock = _response_socket(self._raw)
        setter = getattr(sock, "settimeout", None)
        if callable(setter) and remaining > 0:
            setter(remaining)
        return remaining

    @property
    def status_code(self) -> int:
        return self._status_code

    @property
    def headers(self) -> Mapping[str, str]:
        return self._headers

    def _declared_length(self) -> int | None:
        """``Content-Length`` when the body is fixed-length, else ``None``."""
        raw_value = self._headers.get("content-length")
        if raw_value is None:
            return None
        try:
            length = int(str(raw_value).strip())
        except ValueError:
            return None
        return length if length >= 0 else None

    def iter_bytes(self) -> Iterator[bytes]:
        # ``read(n)`` blocks until it has n bytes or the stream ends, so on a
        # long-lived SSE body it would hold every early delta hostage until
        # 8 KiB accumulated and destroy time-to-first-token. ``read1`` returns
        # whatever one underlying socket read produced, which is what makes
        # incremental timing observable at all.
        reader = getattr(self._raw, "read1", None)
        if not callable(reader):
            reader = self._raw.read
        while True:
            remaining = self._tighten_socket_timeout()
            if remaining is not None and remaining <= 0:
                raise TransportTimeout(
                    "request timed out: the response did not finish within "
                    "the configured request timeout"
                )
            try:
                chunk = reader(self._CHUNK_SIZE)
            except TimeoutError as exc:
                raise TransportTimeout(f"stream read timed out: {exc}") from exc
            except http.client.HTTPException as exc:
                # ``HTTPException`` is not an ``OSError``. ``IncompleteRead``
                # in particular is the expected shape when a proxy or load
                # balancer hangs up mid-chunk on a long-lived SSE stream, and
                # it must become failure evidence rather than a traceback.
                raise TransportConnectionError(
                    f"stream read failed: {type(exc).__name__}"
                ) from exc
            except OSError as exc:
                raise TransportConnectionError(f"stream read failed: {exc}") from exc
            if not chunk:
                break
            self._bytes_read += len(chunk)
            yield chunk

        # A fixed-length body that ends early is a truncated response, not a
        # short one. CPython returns a clean EOF here rather than raising,
        # so the shortfall has to be detected explicitly.
        declared = self._declared_length()
        if declared is not None and self._bytes_read < declared:
            detail = f"{self._bytes_read} of {declared} declared bytes"
            raise TransportConnectionError(f"stream ended after {detail}")

    def close(self) -> None:
        self._raw.close()


class UrllibStreamingTransport:
    """Standard-library transport. No third-party HTTP dependency is needed.

    TLS verification is left at ``urllib``'s default (enabled) and
    redirects are refused, so a credential is never replayed to a host the
    caller did not name.
    """

    def open_stream(self, request: HTTPRequest) -> StreamingResponse:
        # One deadline for the whole exchange, fixed before the connection
        # is attempted so connect, TLS and every read draw on the same
        # budget rather than each getting a fresh one.
        deadline = time.monotonic() + float(request.timeout_seconds)
        opener = urllib.request.build_opener(_NoRedirectHandler)
        urllib_request = urllib.request.Request(
            request.url,
            data=request.body,
            headers=dict(request.headers),
            method=request.method,
        )
        try:
            raw = opener.open(urllib_request, timeout=request.timeout_seconds)
        except urllib.error.HTTPError as exc:
            # A non-2xx response is still a response: keep it so the error
            # body and status can be recorded as evidence.
            return _UrllibResponse(
                exc,
                int(exc.code),
                _normalize_headers(exc.headers),
                deadline=deadline,
            )
        except urllib.error.URLError as exc:
            reason = exc.reason
            if isinstance(reason, TimeoutError):
                raise TransportTimeout(f"request timed out: {reason}") from exc
            raise TransportConnectionError(f"request failed: {reason}") from exc
        except TimeoutError as exc:
            raise TransportTimeout(f"request timed out: {exc}") from exc
        except http.client.HTTPException as exc:
            # ``getresponse`` raises these (``BadStatusLine``, ``LineTooLong``)
            # for a garbled status line, and they bypass the ``URLError``
            # wrapping that ``urllib`` applies to ``OSError``.
            raise TransportConnectionError(
                f"request failed: {type(exc).__name__}"
            ) from exc
        except (ValueError, UnicodeEncodeError) as exc:
            # ``http.client.putheader`` reports an unencodable header by
            # embedding the whole value in the message. Never let that text
            # reach a traceback: the credential is one of those values.
            # ``UnicodeEncodeError`` is a ``ValueError`` subclass; both are
            # named so the intent survives a future refactor.
            raise TransportConnectionError(
                f"request could not be encoded: {type(exc).__name__}"
            ) from None
        except OSError as exc:
            raise TransportConnectionError(f"request failed: {exc}") from exc
        return _UrllibResponse(
            raw, int(raw.status), _normalize_headers(raw.headers), deadline=deadline
        )


def _normalize_headers(headers: Any) -> dict[str, str]:
    items = getattr(headers, "items", None)
    if items is None:
        return {}
    return {str(name).lower(): str(value) for name, value in items()}


# --- Configuration -----------------------------------------------------------


@dataclass(frozen=True)
class FinishReasonVocabulary:
    """Which ``finish_reason`` strings mean "ended" and which mean "failed".

    ``finish_reason`` is not fully standardized. OpenAI documents one set
    and providers add their own, so the meaning of a reason is a property
    of the endpoint being measured rather than of this collector. Holding
    the vocabulary here keeps provider-specific semantics in typed
    configuration: pointing the collector at a provider with different
    reasons is a configuration change, not a code change.

    Anything in neither set is classified ``unrecognized``. That is
    deliberately not the same as terminal: an unknown reason is not
    evidence that generation completed, so a stream ending on one without
    ``[DONE]`` is still reported as truncated rather than published as a
    whole answer.

    The default is the union of the OpenAI reasons and Z.ai's documented
    additions, which is the configuration this collector was validated
    against. The Z.ai strings are absent from every other OpenAI-compatible
    vocabulary known at the time of writing, so the union classifies a
    non-Z.ai stream exactly as the OpenAI set alone would. Use
    :meth:`openai_only` for an endpoint that is known to reuse one of those
    strings with a different meaning.
    """

    terminal: frozenset[str] = OPENAI_TERMINAL_FINISH_REASONS | (
        ZAI_TERMINAL_FINISH_REASONS
    )
    failure: frozenset[str] = ZAI_FAILURE_FINISH_REASONS

    def __post_init__(self) -> None:
        for name in ("terminal", "failure"):
            value = getattr(self, name)
            if not isinstance(value, (frozenset, set)):
                raise OpenAIStreamCollectorError(
                    f"{name} finish reasons must be a set of strings, got {value!r}"
                )
            for reason in value:
                if not isinstance(reason, str) or reason != reason.strip().lower():
                    raise OpenAIStreamCollectorError(
                        f"{name} finish reasons must be stripped lowercase "
                        f"strings, got {reason!r}"
                    )
                if not reason:
                    raise OpenAIStreamCollectorError(
                        f"{name} finish reasons must be non-empty strings"
                    )
            object.__setattr__(self, name, frozenset(value))
        # A reason that means both "finished" and "failed" has no defined
        # classification, and silently letting one set win would decide a
        # run's outcome by the order of two branches.
        overlap = sorted(self.terminal & self.failure)
        if overlap:
            raise OpenAIStreamCollectorError(
                "a finish reason cannot be both terminal and a failure, got "
                f"{overlap!r}"
            )
        if not self.terminal:
            raise OpenAIStreamCollectorError(
                "at least one terminal finish reason is required, otherwise "
                "no stream can ever be classified as complete"
            )

    @classmethod
    def openai_only(cls) -> FinishReasonVocabulary:
        """Just the portable OpenAI reasons, with no provider additions."""
        return cls(terminal=OPENAI_TERMINAL_FINISH_REASONS, failure=frozenset())

    def identity(self) -> dict[str, list[str]]:
        """The vocabulary as sorted lists, for hashing and for evidence."""
        return {
            "terminal": sorted(self.terminal),
            "failure": sorted(self.failure),
        }


@dataclass(frozen=True)
class ProviderExtensions:
    """Provider-specific request fields, kept out of the OpenAI core.

    These are *not* portable OpenAI chat-completions parameters. Z.ai
    documents them for GLM models:

    * ``reasoning_effort`` (``low`` / ``high`` / ``max`` for GLM-5.3 and
      GLM-5.3-Flash, default ``max``);
    * ``thinking.type`` (``enabled`` / ``disabled``; GLM-5.3 and
      GLM-5.3-Flash accept only ``enabled``);
    * ``thinking.clear_thinking`` (boolean, default ``true``), which
      controls whether ``reasoning_content`` from *previous* turns is
      cleared. It does not change whether the current turn thinks.

    ``provider_request_id`` maps to Z.ai's optional body-level
    ``request_id``. Leave it unset to let the provider generate one.
    """

    reasoning_effort: str | None = None
    thinking_type: str | None = None
    clear_thinking: bool | None = None
    provider_request_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("reasoning_effort", "thinking_type", "provider_request_id"):
            value = getattr(self, name)
            if value is None:
                continue
            if not isinstance(value, str) or not value.strip():
                raise OpenAIStreamCollectorError(
                    f"{name} must be a non-empty string when set, got {value!r}"
                )
        if self.clear_thinking is not None and not isinstance(
            self.clear_thinking, bool
        ):
            raise OpenAIStreamCollectorError(
                f"clear_thinking must be a boolean when set, got {self.clear_thinking!r}"
            )

    def to_request_fields(self) -> dict[str, Any]:
        """Render these extensions as chat-completions body fields."""
        fields: dict[str, Any] = {}
        if self.reasoning_effort is not None:
            fields["reasoning_effort"] = self.reasoning_effort
        thinking: dict[str, Any] = {}
        if self.thinking_type is not None:
            thinking["type"] = self.thinking_type
        if self.clear_thinking is not None:
            thinking["clear_thinking"] = self.clear_thinking
        if thinking:
            fields["thinking"] = thinking
        if self.provider_request_id is not None:
            fields["request_id"] = self.provider_request_id
        return fields


def _reasoning_ruled_out(extensions: ProviderExtensions) -> bool:
    """True only when this request cannot have produced hidden reasoning.

    Not having asked for reasoning is not the same as reasoning being off.
    Omitting ``reasoning_effort`` leaves the provider free to apply its own
    default, and for ``glm-5.3`` and ``glm-5.3-flash`` that default is
    ``max``: the common invocation, which sets nothing, is a thinking one.
    Treating silence as "no reasoning" would publish a completion rate whose
    numerator counts tokens generated before the visible window opened.

    So the only thing that rules hidden reasoning out on the request side is
    turning thinking off explicitly. Everything else is settled by evidence
    at the end of the stream, where an explicit zero reasoning token count
    or an observed reasoning delta answers the question directly.

    Derived from what this collector sent rather than from the provider's
    identity, so it stays meaningful for any OpenAI-compatible endpoint.
    """
    return extensions.thinking_type == "disabled"


@dataclass(frozen=True)
class APICollectionConfig:
    """Inputs and reproducibility metadata for one streaming API call."""

    run_id: str
    provider: str
    """Short, sanitized provider label recorded in evidence (e.g. ``z.ai``)."""
    endpoint: str
    """Full chat-completions URL, e.g.
    ``https://api.z.ai/api/paas/v4/chat/completions``."""
    model_id: str
    prompt: str
    output_dir: Path
    command_argv: tuple[str, ...]
    credential_env_var: str
    """Name of the environment variable holding the API key. The name is
    persisted; the value never is."""
    system_prompt: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    request_timeout_seconds: float = 120.0
    extensions: ProviderExtensions = field(default_factory=ProviderExtensions)
    finish_reasons: FinishReasonVocabulary = field(
        default_factory=FinishReasonVocabulary
    )
    """How this endpoint's ``finish_reason`` strings are classified."""
    model_revision: str | None = None
    """Provider-side model build, when the provider exposes one. Hosted
    APIs generally do not, in which case this stays ``None`` rather than
    being guessed from the model ID."""
    retained_event_limit: int = DEFAULT_RETAINED_EVENT_LIMIT
    """How many per-event timing rows the timeline keeps.

    A stream has no declared length, so an unbounded timeline makes the
    artifact size a function of provider behaviour. Past this many events
    the rows stop being retained while the counters keep counting, so the
    totals, the rates and the inter-token distribution stay exact and the
    record says plainly that the per-event rows were cut. Each row is four
    fixed-width fields and never any generated text, so this bound is the
    only thing the timeline's size depends on."""

    def __post_init__(self) -> None:
        if not self.run_id:
            raise OpenAIStreamCollectorError("run_id must be non-empty")
        if not _PROVIDER_LABEL_PATTERN.match(self.provider or ""):
            raise OpenAIStreamCollectorError(
                "provider must be a short label matching "
                f"[A-Za-z0-9][A-Za-z0-9._-]{{0,63}}, got {self.provider!r}"
            )
        if not self.model_id or not self.model_id.strip():
            raise OpenAIStreamCollectorError("model_id must be non-empty")
        if not self.prompt:
            raise OpenAIStreamCollectorError("prompt must be non-empty")
        if (
            not isinstance(self.retained_event_limit, int)
            or isinstance(self.retained_event_limit, bool)
            or self.retained_event_limit < 1
        ):
            raise OpenAIStreamCollectorError(
                "retained_event_limit must be a positive integer, got "
                f"{self.retained_event_limit!r}"
            )
        if self.system_prompt is not None and not self.system_prompt:
            raise OpenAIStreamCollectorError(
                "system_prompt must be non-empty when provided"
            )
        if not _ENV_VAR_PATTERN.match(self.credential_env_var or ""):
            # The rejected value is never echoed. It reached this branch by
            # not being a variable name, and the reason it usually is not is
            # that it is the credential itself.
            raise OpenAIStreamCollectorError(
                "the value of --api-key-env must be an exported environment "
                "variable name in the conventional uppercase form, such as "
                "ZAI_API_KEY, and must not be the credential itself; the "
                "rejected value is not repeated here"
            )
        if not self.command_argv or not all(
            isinstance(item, str) and item for item in self.command_argv
        ):
            raise OpenAIStreamCollectorError(
                "command_argv must contain non-empty argument strings"
            )
        _validate_endpoint(self.endpoint)

        if self.max_output_tokens is not None and (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens < 1
        ):
            raise OpenAIStreamCollectorError(
                "max_output_tokens must be a positive integer when set"
            )
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise OpenAIStreamCollectorError("seed must be an integer when set")
        _validate_optional_ratio("temperature", self.temperature, maximum=2.0)
        _validate_optional_ratio("top_p", self.top_p, maximum=1.0)
        timeout = self.request_timeout_seconds
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(float(timeout))
            or float(timeout) <= 0
        ):
            raise OpenAIStreamCollectorError(
                "request_timeout_seconds must be a positive finite number"
            )
        if not isinstance(self.extensions, ProviderExtensions):
            raise OpenAIStreamCollectorError(
                "extensions must be a ProviderExtensions instance"
            )


def _validate_optional_ratio(name: str, value: Any, *, maximum: float) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise OpenAIStreamCollectorError(f"{name} must be a finite number when set")
    if not 0.0 <= float(value) <= maximum:
        raise OpenAIStreamCollectorError(
            f"{name} must be between 0 and {maximum} when set, got {value!r}"
        )


def _try_urlsplit(endpoint: str) -> SplitResult | None:
    """``urlsplit`` that reports failure instead of raising.

    ``urlsplit`` raises ``ValueError`` on a malformed authority such as an
    unclosed IPv6 bracket, and its message can quote the netloc.
    """
    try:
        return urlsplit(endpoint)
    except ValueError:
        return None


def _names_a_credential(key: str) -> bool:
    """True when a query parameter name reads as a credential.

    The name is tokenized first and each component is judged whole. An
    unanchored substring search rejects ``design``, ``assignment``,
    ``monkey`` and ``insignia``, which are ordinary parameter names, and
    the caller cannot learn why because the diagnostic deliberately echoes
    nothing. A glued compound such as ``apikey`` carries no separator and no
    case change, so a component also counts when recognized words cover it
    end to end and at least one of them names a credential outright.
    """
    for raw in _QUERY_KEY_COMPONENT.findall(key):
        component = raw.lower()
        # One trailing plural, so ``keys`` is read as ``key``.
        if component not in _CREDENTIAL_QUERY_TERMS and component.endswith("s"):
            component = component[:-1]
        if _covers_a_credential(component) or _spans_a_credential_phrase(component):
            return True
    return False


def _spans_a_credential_phrase(component: str) -> bool:
    """True when a multiword credential phrase sits inside the component.

    The complete-cover rule cannot see ``openaiapikey`` or ``myapikey``,
    because ``openai`` and ``my`` are not words this recognizes and the
    cover never finishes. Those are plainly credential names, and the
    substring search that predated the cover caught them.

    Requiring *two* recognized words, at least one of them a credential
    noun, is what makes the surrounding text safe to ignore. A lone
    ambiguous noun proves nothing, which is why ``keyword`` and ``monkey``
    stay accepted: ``key`` is one word and neither ``word`` nor ``mon``
    extends it. ``design``, ``signal``, ``insignia`` and ``assignment``
    stay accepted for the same reason around ``sig``.

    States are capped at two words, so each offset carries at most six of
    them and the scan stays linear in the length of the component.
    """
    states: list[set[tuple[int, bool]]] = [set() for _ in range(len(component) + 1)]
    for start in range(len(component)):
        # A phrase may begin at any offset, so every offset seeds one.
        states[start].add((0, False))
        for words, seen_term in states[start]:
            limit = min(len(component), start + _MAX_CREDENTIAL_WORD)
            for end in range(start + 1, limit + 1):
                word = component[start:end]
                is_term = word in _CREDENTIAL_QUERY_TERMS
                if not is_term and word not in _CREDENTIAL_QUERY_QUALIFIERS:
                    continue
                reached = (min(words + 1, 2), seen_term or is_term)
                if reached[0] >= 2 and reached[1]:
                    return True
                states[end].add(reached)
    return False


def _covers_a_credential(component: str) -> bool:
    """True when recognized words cover the component end to end.

    Splitting once into two parts is not enough: ``xapikey`` is three
    words, and stopping at two accepted it while the substring search this
    replaced had caught it. Requiring a complete cover instead makes the
    number of glued words irrelevant, and it is the completeness that
    separates ``apikey`` from ``keyword`` and ``monkey``, whose leftover
    ``word`` and ``mon`` are not words this recognizes.

    ``reached`` maps each offset a cover can reach to whether some cover
    reaching it has used a credential noun, so a cover built only from
    qualifiers, such as ``appid``, does not count.

    The lookahead stops at the longest word either table spells. No cover
    step can use a longer word, so nothing is missed, and it is what keeps
    the cost linear rather than quadratic in the length of the component.
    """
    reached: dict[int, bool] = {0: False}
    for start in range(len(component)):
        if start not in reached:
            continue
        seen_term = reached[start]
        limit = min(len(component), start + _MAX_CREDENTIAL_WORD)
        for end in range(start + 1, limit + 1):
            word = component[start:end]
            is_term = word in _CREDENTIAL_QUERY_TERMS
            if not is_term and word not in _CREDENTIAL_QUERY_QUALIFIERS:
                continue
            reached[end] = reached.get(end, False) or seen_term or is_term
    return reached.get(len(component), False)


def _try_parse_qsl(query: str) -> list[tuple[str, str]] | None:
    try:
        return parse_qsl(query, keep_blank_values=True)
    except ValueError:
        return None


def _force_netloc_component(read: Callable[[], object], message: str) -> object:
    """Read one lazily parsed netloc component, mapping failure to our error.

    ``SplitResult.hostname`` and ``SplitResult.port`` only parse the netloc
    when they are read, and they raise ``ValueError`` with the offending
    text in the message. Reading them behind this barrier is what keeps a
    malformed netloc from escaping as an exception that quotes what the
    operator typed.
    """
    try:
        return read()
    except ValueError:
        raise OpenAIStreamCollectorError(message) from None


def parse_endpoint(endpoint: str) -> SplitResult:
    """Split an endpoint, turning any parse failure into a safe error.

    Every caller that needs the parsed form goes through this, so a
    malformed URL cannot escape as a raw ``ValueError`` whose message
    quotes the netloc, the port or anything else the operator typed. The
    raised message describes the shape only.
    """
    parts = _try_urlsplit(endpoint)
    if parts is None:
        raise OpenAIStreamCollectorError(
            "endpoint could not be parsed as a URL; check the host and any "
            "IPv6 brackets"
        )
    # Both are lazily parsed properties, so reading them here is what
    # forces the netloc to be validated inside this guard. The values are
    # deliberately discarded: this call is a validation barrier, and the
    # caller reads whichever component it needs from the returned split.
    _force_netloc_component(
        lambda: parts.hostname,
        "endpoint has a host that could not be parsed",
    )
    _force_netloc_component(
        lambda: parts.port,
        "endpoint has a port that is not an integer in the range 0 to 65535",
    )
    return parts


def _safe_endpoint_for_message(endpoint: str) -> str:
    """Render an endpoint for an error message without leaking secrets.

    A misconfigured endpoint is exactly the case where a key is most
    likely to have been pasted into the URL, so the raw string must never
    reach stderr or an artifact. Only the scheme, host, port and the
    *shape* of the path and query survive.
    """
    parts = _try_urlsplit(endpoint)
    if parts is None:
        return "<unparsable endpoint>"
    if not parts.scheme and not parts.netloc:
        return "<endpoint>"
    # ``hostname`` and ``port`` are properties that parse the netloc lazily,
    # so both can raise even though ``urlsplit`` itself succeeded. This
    # function builds error messages, so it must never raise.
    try:
        host = parts.hostname or "<no-host>"
    except ValueError:
        host = "<invalid-host>"
    try:
        port = parts.port
    except ValueError:
        authority = f"{host}:<invalid-port>"
    else:
        authority = f"{host}:{port}" if port else host
    rendered = f"{parts.scheme or '<no-scheme>'}://{authority}"
    if parts.path:
        segments = [segment for segment in parts.path.split("/") if segment]
        rendered += "/" + "/".join(_REDACTED for _ in segments) if segments else "/"
    if parts.query:
        pairs = _try_parse_qsl(parts.query)
        if pairs is None:
            rendered += "?" + _REDACTED
        else:
            # A key whose name looks like a credential is replaced along with
            # the values, so no message built from this rendering can echo
            # it. Every other key is kept: the shape of the query is what
            # makes a misconfiguration diagnosable.
            keys = sorted(
                {_REDACTED if _names_a_credential(key) else key for key, _ in pairs}
            )
            rendered += "?" + "&".join(f"{key}={_REDACTED}" for key in keys)
    return rendered


def _validate_endpoint(endpoint: str) -> None:
    if not endpoint:
        raise OpenAIStreamCollectorError("endpoint must be non-empty")
    safe = _safe_endpoint_for_message(endpoint)
    parts = parse_endpoint(endpoint)
    if parts.scheme not in ("http", "https"):
        raise OpenAIStreamCollectorError(f"endpoint must use http or https, got {safe}")
    if not parts.hostname:
        raise OpenAIStreamCollectorError(f"endpoint must include a host: {safe}")
    if parts.username is not None or parts.password is not None:
        raise OpenAIStreamCollectorError(
            "endpoint must not embed credentials; pass the API key through the "
            "environment variable named by --api-key-env"
        )
    if parts.fragment:
        raise OpenAIStreamCollectorError("endpoint must not contain a fragment")
    if parts.scheme == "http" and parts.hostname not in _LOCAL_HOSTS:
        raise OpenAIStreamCollectorError(
            "endpoint must use https for non-local hosts so the Authorization "
            f"header is not sent in clear text, got {safe}"
        )
    pairs = _try_parse_qsl(parts.query)
    if pairs is None:
        raise OpenAIStreamCollectorError(f"endpoint query could not be parsed: {safe}")
    for key, _value in pairs:
        if _names_a_credential(key):
            raise OpenAIStreamCollectorError(
                # The parameter name is not echoed. A name that looks like a
                # credential is exactly the name most likely to *carry* one,
                # and this diagnostic reaches stderr and a failure record.
                # Nothing the caller supplied appears here: not the name, the
                # host, the path or any other part of the query.
                "the endpoint query string names a parameter that looks like "
                "a credential; credentials must come from the environment "
                "variable named by --api-key-env instead"
            )


# --- Redaction ---------------------------------------------------------------


#: The largest token count that survives a round trip through a float.
#: Above this, ``float(value)`` either overflows outright or loses integer
#: precision, so any rate derived from the count would be wrong without
#: saying so. Real counts are many orders of magnitude below it.
_MAX_EXACT_TOKEN_COUNT = 2**53


#: A whitespace run is matched up to this many characters. Bounded so a
#: pathological run of separators cannot make matching walk indefinitely.
_MAX_WHITESPACE_RUN = 64

#: One whitespace character: the step a whitespace run repeats.
_WHITESPACE_STEP = re.compile(r"\s")


@cache
def _form_pattern(form: str) -> re.Pattern[str]:
    """One credential spelling, compiled once and shared by every matcher.

    Spellings repeat heavily: every ``a`` in a key yields the same tuple of
    forms, and every whole-value spelling of the credential re-uses them.
    Caching keeps building a redactor cheap even though it now prepares one
    matcher per spelling rather than one pattern per credential.
    """
    return re.compile(re.escape(form), re.IGNORECASE)


@dataclass(frozen=True)
class _CredentialVariant:
    """One spelling of the credential, prepared for bounded matching.

    ``forms`` holds the accepted spellings of each element in order and
    ``runs`` marks the elements that are whitespace runs rather than single
    characters. ``head`` is a zero-width prefilter over the first element's
    spellings: it yields the only positions a match can begin at, so a scan
    stays linear in the length of the text being scrubbed.
    """

    forms: tuple[tuple[str, ...], ...]
    runs: tuple[bool, ...]
    head: re.Pattern[str]
    #: The most characters a match beginning at one position can consume.
    #: Every element contributes its longest spelling, and a whitespace run
    #: contributes that spelling repeated up to the run cap, so this is a
    #: true upper bound rather than an estimate.
    span: int
    #: How many characters one whitespace run element may consume. Normally
    #: ``_MAX_WHITESPACE_RUN``, but never fewer than the longest run of
    #: whitespace the credential itself contains: a matcher that cannot
    #: consume its own literal spelling would leave the exact credential in
    #: the artifact, which is the one thing redaction must never do.
    whitespace_bound: int


def _longest_whitespace_run(value: str) -> int:
    longest = 0
    current = 0
    for character in value.strip():
        if character.isspace():
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def _transport_spellings(body: str) -> list[str]:
    """Whole-value re-encodings of the credential into another alphabet.

    ``_character_forms`` covers the spellings that keep the credential's own
    characters: percent-encoding, backslash escapes, case and whitespace
    variance. A value re-encoded into a different alphabet keeps none of
    those characters, so it needs a spelling of its own. Base64 and hex are
    covered because each is one mechanical decode from the key and each is
    what an intermediary reaches for when an opaque value has to be carried
    somewhere byte-safe.

    Base64 is generated at all three byte alignments, because the credential
    may sit inside a larger encoded blob rather than at its start. The
    characters whose bits depend on bytes outside the credential are
    dropped, so what remains is fixed by the credential alone; at most one
    leading and one trailing character of an embedded encoding is left
    outside the redacted span.
    """
    raw = body.encode("utf-8")
    spellings = [raw.hex()]
    for encode in (base64.b64encode, base64.urlsafe_b64encode):
        for offset in range(3):
            encoded = encode(b"\0" * offset + raw).decode("ascii").rstrip("=")
            if (offset + len(raw)) % 3:
                # The final character encodes trailing bits padded with
                # zeros, which a longer message would fill differently.
                encoded = encoded[:-1]
            trimmed = encoded[(0, 2, 3)[offset] :]
            if len(trimmed) >= _MIN_ENCODED_CREDENTIAL_CHARS:
                spellings.append(trimmed)
    return spellings


class _Redactor:
    """Scrub a known credential (and bearer-token shapes) out of any string.

    Every provider-controlled string that reaches an artifact goes through
    this, not just error messages. A server that echoes the key back in a
    response id, a model name, a finish reason or the generated text
    itself would otherwise write it straight to disk.
    """

    _BEARER = re.compile(r"(?i)\b(bearer)(?:\s|%20|%2520)+[A-Za-z0-9._~+/=%-]+")

    # A whitespace run, in any rendering a provider might echo it in: real
    # whitespace, the form-encoded ``+``, the percent-encoded byte, or a
    # backslash escape from a JSON or Python repr. Generated from
    # ``_character_forms`` so the run matcher and the truncation repair
    # cannot disagree about what spells a space.
    # Bounded repetition so a pathological run cannot make matching blow up.
    _WHITESPACE_CHARACTERS = "\t\n\v\f\r "

    @staticmethod
    def _character_forms(character: str) -> tuple[str, ...]:
        """Every spelling of one character the scrub treats as equivalent.

        Returned unescaped so ``boundary`` can ask whether a truncated tail
        is a proper prefix of one of them. ``_character_element`` compiles
        the same list into a pattern, so the matcher and the truncation
        repair can never disagree about what counts as a spelling.

        Percent-encoding covers a provider that echoes the key through a
        URL builder. Backslash escapes cover the other mechanical rendering
        a provider produces: a JSON encoder emitting ``\\u0073``, or a
        Python ``repr`` emitting ``\\x73``. Both are one mechanical decode
        away from the key, so an artifact holding either is holding the
        credential. This matters most where the text is *not* re-parsed:
        a non-JSON error body is persisted as it arrived, so an escape in
        it stays an escape rather than being decoded back into characters
        the literal matcher would catch.
        """
        codepoint = ord(character)
        encoded = "".join(f"%{byte:02X}" for byte in character.encode("utf-8"))
        forms = [character, encoded, encoded.replace("%", "%25")]
        short = _SHORT_ESCAPES.get(character)
        if short is not None:
            forms.append(short)
        if codepoint < 0x100:
            forms.append(f"\\x{codepoint:02X}")
            # The octal escape a C, Java or Python source renderer emits.
            # Both the zero-padded and the minimal spelling are accepted,
            # because either decodes back to the same character.
            forms.append(f"\\{codepoint:03o}")
            forms.append(f"\\{codepoint:o}")
        if codepoint <= 0xFFFF:
            forms.append(f"\\u{codepoint:04X}")
        else:
            # Outside the BMP a JSON encoder emits a UTF-16 surrogate pair.
            high, low = (
                int.from_bytes(character.encode("utf-16-be")[i : i + 2], "big")
                for i in (0, 2)
            )
            forms.append(f"\\u{high:04X}\\u{low:04X}")
        forms.append(f"\\U{codepoint:08X}")
        return tuple(dict.fromkeys(forms))

    @classmethod
    def _whitespace_forms(cls) -> tuple[str, ...]:
        """Every spelling of a single whitespace character.

        The literal one-character spellings are dropped: they cannot be cut
        short, and the run step already accepts any one whitespace
        character. What remains is the multi-character renderings, which
        truncation *can* cut through, leaving ``%2``, ``%25`` or ``\\u00``
        where a space belonged. ``+`` is the form encoding of a space and
        is included as a whole spelling.
        """
        forms: list[str] = ["+", "\\t", "\\n", "\\v", "\\f", "\\r"]
        for character in cls._WHITESPACE_CHARACTERS:
            forms.extend(
                form for form in cls._character_forms(character) if len(form) > 1
            )
        return tuple(dict.fromkeys(forms))

    @classmethod
    def _elements_with_forms(cls, fragment: str) -> list[tuple[tuple[str, ...], bool]]:
        """Split a fragment into elements, each as the spellings it accepts.

        One element per character, except that a run of whitespace becomes
        a single run element flagged ``True``. Surrounding whitespace is
        dropped: a credential read from a file or an environment value may
        carry it, it is not part of the secret, and keeping it would make
        the matcher demand whitespace the provider never echoed. A fragment
        with no non-whitespace character is not a credential and yields no
        elements at all, so it cannot become a matcher that hits every
        space in every artifact.

        The truncation repair reads the same table, so a spelling can never
        be added to the matcher without also being known to the repair, or
        the other way around.
        """
        elements: list[tuple[tuple[str, ...], bool]] = []
        stripped = fragment.strip()
        index = 0
        while index < len(stripped):
            if stripped[index].isspace():
                while index < len(stripped) and stripped[index].isspace():
                    index += 1
                elements.append((cls._whitespace_forms(), True))
                continue
            elements.append((cls._character_forms(stripped[index]), False))
            index += 1
        return elements

    @classmethod
    def _spellings(cls, credential: str) -> tuple[str, ...]:
        """Every whole-value rendering of the credential worth matching.

        Unicode normalization is included because a normalizing
        intermediary can rewrite an accented key into a different sequence
        of codepoints that renders identically to anyone reading the
        artifact and decodes back to the key. The re-encodings come from
        ``_transport_spellings``. Each is prepared as its own matcher, so
        the per-character spellings, the truncation repair and the
        preflight check all apply to them unchanged.
        """
        spellings = [credential]
        for form in ("NFC", "NFD"):
            normalized = unicodedata.normalize(form, credential)
            if normalized != credential:
                spellings.append(normalized)
        # Re-encode every literal spelling, not just the one that arrived.
        # Normalization and transport encoding compose in both orders: an
        # intermediary that normalizes an accented key and then base64s it
        # produces bytes that decode straight back to the credential, and
        # encoding only the original spelling left exactly that shape
        # unmatched.
        encoded: list[str] = []
        for spelling in spellings:
            body = spelling.strip()
            if len(body) >= _MIN_ENCODED_CREDENTIAL_CHARS:
                # Short values are left to the literal spellings. An encoding
                # of two or three characters collides with ordinary text, and
                # a redaction that fires on noise destroys the artifact it
                # guards.
                encoded.extend(_transport_spellings(body))
        spellings.extend(encoded)
        return tuple(dict.fromkeys(spellings))

    @classmethod
    def _variant(cls, spelling: str) -> _CredentialVariant | None:
        elements = cls._elements_with_forms(spelling)
        if not elements:
            return None
        forms = tuple(element for element, _ in elements)
        runs = tuple(run for _, run in elements)
        whitespace_bound = max(_MAX_WHITESPACE_RUN, _longest_whitespace_run(spelling))
        span = 0
        for index, element in enumerate(forms):
            longest = max(len(form) for form in element)
            span += whitespace_bound * max(longest, 1) if runs[index] else longest
        return _CredentialVariant(
            forms=forms,
            runs=runs,
            head=re.compile(
                "(?=" + "|".join(re.escape(form) for form in forms[0]) + ")",
                re.IGNORECASE,
            ),
            span=span,
            whitespace_bound=whitespace_bound,
        )

    def __init__(self, credential: str | None) -> None:
        self._credential = credential or None
        variants: list[_CredentialVariant] = []
        if self._credential:
            for spelling in self._spellings(self._credential):
                variant = self._variant(spelling)
                if variant is not None:
                    variants.append(variant)
        self._variants = tuple(variants)

    @staticmethod
    def _advance(
        variant: _CredentialVariant, index: int, text: str, positions: set[int]
    ) -> set[int]:
        """Every position reachable by matching one more element.

        Carrying a *set* of positions rather than one cursor is what keeps
        matching both exact and bounded. Spellings of the same character
        can have different lengths where one is a prefix of another: a
        literal backslash is a prefix of the escaped ``\\\\``, and a literal
        ``%`` is a prefix of the double-encoded ``%25``. A backtracking
        matcher explores every combination of those choices and takes
        exponential time to *fail*, so provider-controlled text could stall
        redaction while the request timeout is not running. Advancing all
        choices together visits each element once and still finds a match
        whenever one exists, so neither exactness nor termination is
        traded away.
        """
        forms = variant.forms[index]
        reached: set[int] = set()
        if not variant.runs[index]:
            for position in positions:
                for form in forms:
                    found = _form_pattern(form).match(text, position)
                    if found is not None:
                        reached.add(found.end())
            return reached
        frontier = positions
        for _ in range(variant.whitespace_bound):
            step: set[int] = set()
            for position in frontier:
                single = _WHITESPACE_STEP.match(text, position)
                if single is not None:
                    step.add(single.end())
                for form in forms:
                    found = _form_pattern(form).match(text, position)
                    if found is not None:
                        step.add(found.end())
            step -= reached
            if not step:
                break
            reached |= step
            frontier = step
        return reached

    @classmethod
    def _match_end(
        cls, variant: _CredentialVariant, text: str, start: int
    ) -> int | None:
        """The furthest end of a whole match beginning at ``start``."""
        positions = {start}
        for index in range(len(variant.forms)):
            positions = cls._advance(variant, index, text, positions)
            if not positions:
                return None
        return max(positions)

    @staticmethod
    def _is_truncated_tail(
        variant: _CredentialVariant, index: int, remainder: str
    ) -> bool:
        """True when ``remainder`` is element ``index`` cut short.

        Truncation can land inside a percent escape or a backslash escape,
        leaving ``%``, ``%2`` or ``\\u00`` where a whole spelling belonged.
        The element matcher rejects a partial escape, so without this
        repair the scrub stops at the cut and every credential character
        before it survives. Only the multi-character spellings can be cut
        this way, since a literal character is one character long and a
        non-empty remainder can never be shorter.
        """
        if not remainder or index >= len(variant.forms):
            return False
        lowered = remainder.lower()
        return any(
            len(remainder) < len(form) and form.lower().startswith(lowered)
            for form in variant.forms[index]
        )

    def _redact_variant(self, variant: _CredentialVariant, text: str) -> str:
        """Replace every non-overlapping match of one spelling."""
        pieces: list[str] = []
        cursor = 0
        for found in variant.head.finditer(text):
            start = found.start()
            if start < cursor:
                continue
            ending = self._match_end(variant, text, start)
            if ending is None:
                continue
            pieces.append(text[cursor:start])
            pieces.append(_REDACTED)
            cursor = ending
        if not pieces:
            return text
        pieces.append(text[cursor:])
        return "".join(pieces)

    def _scrub(self, text: str) -> str:
        cleaned = text
        for variant in self._variants:
            cleaned = self._redact_variant(variant, cleaned)
        return self._BEARER.sub(rf"\1 {_REDACTED}", cleaned)

    def search(self, text: str) -> bool:
        """True when any spelling of the credential occurs in ``text``.

        The preflight that decides whether a provider-controlled value may
        be persisted at all asks this, so it is answered by the matcher
        that would have to scrub the value rather than by a second,
        weaker comparison.
        """
        return any(
            self._match_end(variant, text, found.start()) is not None
            for variant in self._variants
            for found in variant.head.finditer(text)
        )

    def __call__(self, text: str, *, limit: int = _MAX_PERSISTED_MESSAGE_CHARS) -> str:
        # Normalize before scrubbing as well as after. Whitespace collapse
        # can reassemble a credential that the pre-collapse form hid.
        cleaned = " ".join(self._scrub(" ".join(text.split())).split())
        if len(cleaned) > limit:
            cleaned = cleaned[: limit - 3] + "..."
        return cleaned

    def _truncated_start(self, text: str) -> int | None:
        """Where a credential cut short by truncation begins, if it is there.

        Every candidate start is walked element by element. A candidate
        wins either by consuming the text exactly, or by ending in a
        remainder that is one element cut short. Both require enough whole
        elements to have matched first, so a couple of coincidental
        characters at the end of a body cannot blank it.

        Only a suffix window is searched. Truncation removes the tail, so
        both ways of winning end at the last character of the text, and a
        match consumes at most ``variant.span`` characters. A start further
        back than that cannot reach the end and so cannot win. Scanning the
        whole body instead would be quadratic: provider-controlled text can
        put a candidate start every few characters, and each walk reads to
        the end. That is reachable from the network, since this runs after
        the read loop where the request deadline no longer applies.
        """
        ending = len(text)
        best: int | None = None
        for variant in self._variants:
            window = max(0, ending - variant.span)
            for found in variant.head.finditer(text, window):
                start = found.start()
                if best is not None and start >= best:
                    break
                positions = {start}
                for index in range(len(variant.forms)):
                    if index >= _MIN_ENCODED_CREDENTIAL_CHARS and any(
                        self._is_truncated_tail(variant, index, text[position:])
                        for position in positions
                        if position < ending
                    ):
                        best = start
                        break
                    positions = self._advance(variant, index, text, positions)
                    if not positions:
                        break
                    if (
                        ending in positions
                        and index + 1 >= _MIN_ENCODED_CREDENTIAL_CHARS
                    ):
                        best = start
                        break
        return best

    def boundary(self, text: str) -> str:
        """Scrub a string that truncation may have cut through.

        Truncation always removes the tail, so a credential split by a byte
        cap or a closed connection survives as a trailing proper prefix that
        the whole-credential scrub cannot see. Whitespace collapse then
        pulls that tail back into the persisted window, which is how a 64
        KiB body can leak all but the last character of a key.
        """
        cleaned = text.rstrip("\ufffd")
        start = self._truncated_start(cleaned)
        if start is None:
            return cleaned
        return cleaned[:start] + _REDACTED

    def text(self, value: str) -> str:
        """Scrub generated text, preserving whitespace, newlines and length.

        ``__call__`` collapses whitespace and truncates, which is right for
        a one-line diagnostic and wrong for the model's answer.
        """
        return self._scrub(value)

    def identifier(self, value: str | None) -> str | None:
        """Scrub a provider-supplied identifier such as an id or model name."""
        if value is None:
            return None
        return self(value, limit=_MAX_PERSISTED_HEADER_CHARS)


def redact_text_for_dry_run(text: str, credential: str | None) -> str:
    """Scrub a rendered document that is about to be printed or written.

    Used where no request is made and therefore no per-field redactor is
    in play, so a configured key cannot escape through a plan, a
    reconstructed command or a diagnostic.
    """
    return _Redactor(credential).text(text)


# --- Request plan ------------------------------------------------------------


@dataclass(frozen=True)
class MessageDigest:
    """A message's role and content fingerprint. Never the content itself."""

    role: str
    characters: int
    content_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "characters": self.characters,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class RequestPlan:
    """Exactly what would be sent, with nothing sensitive in it.

    This is what ``--dry-run`` prints and what real runs persist. It has
    no credential value, no prompt text and no header values.
    """

    schema_version: str
    provider: str
    method: str
    endpoint_origin: str
    endpoint_path: str
    endpoint_query_keys: tuple[str, ...]
    model_id: str
    model_revision: str | None
    credential_env_var: str
    credential_header_name: str
    header_names: tuple[str, ...]
    messages: tuple[MessageDigest, ...]
    request_parameters: dict[str, Any]
    provider_extensions: dict[str, Any]
    finish_reasons: dict[str, list[str]]
    request_timeout_seconds: float
    command: tuple[str, ...]
    config_hash: str
    workload_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provider": self.provider,
            "method": self.method,
            "endpoint_origin": self.endpoint_origin,
            "endpoint_path": self.endpoint_path,
            "endpoint_query_keys": list(self.endpoint_query_keys),
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "credential_env_var": self.credential_env_var,
            "credential_header_name": self.credential_header_name,
            "header_names": list(self.header_names),
            "messages": [message.to_dict() for message in self.messages],
            "request_parameters": dict(sorted(self.request_parameters.items())),
            "provider_extensions": dict(sorted(self.provider_extensions.items())),
            "finish_reasons": {
                key: list(value) for key, value in sorted(self.finish_reasons.items())
            },
            "request_timeout_seconds": self.request_timeout_seconds,
            "command": list(self.command),
            "config_hash": self.config_hash,
            "workload_hash": self.workload_hash,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


def _build_messages(config: APICollectionConfig) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if config.system_prompt is not None:
        messages.append({"role": "system", "content": config.system_prompt})
    messages.append({"role": "user", "content": config.prompt})
    return messages


def _core_request_parameters(config: APICollectionConfig) -> dict[str, Any]:
    """Portable OpenAI chat-completions fields, excluding ``messages``."""
    parameters: dict[str, Any] = {"model": config.model_id, "stream": True}
    if config.max_output_tokens is not None:
        parameters["max_tokens"] = config.max_output_tokens
    if config.temperature is not None:
        parameters["temperature"] = config.temperature
    if config.top_p is not None:
        parameters["top_p"] = config.top_p
    if config.seed is not None:
        parameters["seed"] = config.seed
    return parameters


def _request_body(config: APICollectionConfig) -> dict[str, Any]:
    body = _core_request_parameters(config)
    body.update(config.extensions.to_request_fields())
    body["messages"] = _build_messages(config)
    return body


def _normalized_query_identity(query: str) -> list[list[Any]]:
    """Query keys in a stable order, values hashed, per-key order preserved.

    Dropping the values entirely made ``?api-version=2024-01`` and
    ``?api-version=2025-06`` share one identity, which silently merges two
    different deployments in any downstream comparison. Hashing keeps the
    values identity-bearing without ever persisting them, which matters
    because a value here is provider-controlled and may be sensitive even
    when its key does not look like a credential.

    Keys are sorted so that ``?a=1&b=2`` and ``?b=2&a=1`` agree, since the
    order of distinct keys is not significant in a query string. Values
    under a *repeated* key keep their original order, because there the
    order can carry meaning and sorting the flat pair list would make
    ``?a=1&a=2`` and ``?a=2&a=1`` collide.
    """
    grouped: dict[str, list[str]] = {}
    for key, value in parse_qsl(query, keep_blank_values=True):
        grouped.setdefault(key, []).append(sha256_text(value))
    return [[key, grouped[key]] for key in sorted(grouped)]


def _config_identity_hash(config: APICollectionConfig) -> str:
    parts = parse_endpoint(config.endpoint)
    identity: str = config_hash(
        {
            "provider": config.provider,
            "endpoint_origin": f"{parts.scheme}://{parts.netloc}",
            "endpoint_path": parts.path,
            "endpoint_query_keys": sorted(
                key for key, _ in parse_qsl(parts.query, keep_blank_values=True)
            ),
            "endpoint_query_identity": _normalized_query_identity(parts.query),
            "model_id": config.model_id,
            "model_revision": config.model_revision,
            # The variable name is deployment plumbing, not request identity.
            # Two runs differing only in which variable held the key issue
            # byte-identical requests, and hashing a value that may be the
            # credential itself would persist a derivation of a secret this
            # collector promises never to hash.
            "request_parameters": _core_request_parameters(config),
            "provider_extensions": config.extensions.to_request_fields(),
            # Two runs that classify the same finish reason differently are
            # not the same measurement configuration, even though they
            # issue byte-identical requests. The difference is in what the
            # response is taken to mean, which is exactly what a config
            # identity is for.
            "finish_reasons": config.finish_reasons.identity(),
            "request_timeout_seconds": config.request_timeout_seconds,
            # Two runs that kept different numbers of event rows did not
            # produce the same evidence, even though they issued the same
            # request, so they are not the same collector configuration.
            "retained_event_limit": config.retained_event_limit,
            "system_prompt_sha256": (
                None
                if config.system_prompt is None
                else sha256_text(config.system_prompt)
            ),
        }
    )
    return identity


def _endpoint_for_command(endpoint: str) -> str:
    """The endpoint as it may appear in a persisted command.

    Query *values* are stripped. The plan only ever records query keys, so
    keeping raw values in the reconstructed command would reintroduce
    exactly what the rest of the plan is careful to leave out, and a value
    is provider or user supplied and may hold a token.
    """
    parts = _try_urlsplit(endpoint)
    if parts is None:
        return "<unparsable endpoint>"
    if not parts.query:
        return endpoint
    pairs = _try_parse_qsl(parts.query)
    if pairs is None:
        return _safe_endpoint_for_message(endpoint)
    query = "&".join(f"{key}={_REDACTED}" for key, _ in pairs)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, parts.fragment))


def _persistable_env_var(name: str, environ: Mapping[str, str] | None) -> str:
    """Persist the variable name only once the environment proves it is one.

    ``--api-key-env`` takes a name, but the mechanical response to the
    ``--api-key`` refusal is to swap the flag and keep the value, which puts
    the credential in the name slot. The uppercase shape rule rejects the
    common key spellings, and an all-uppercase key such as an AWS access key
    id still passes it. What cannot be faked is presence: a name the
    environment defines is a real exported variable, so it is safe to write
    down. A name the environment does not define was never proven to be a
    name, so it is treated as caller-supplied untrusted text, exactly as an
    argument token is treated as a value until the parser vouches for it.
    """
    source = os.environ if environ is None else environ
    return name if name in source else _REDACTED


def _mask_option_value(
    arguments: Sequence[str], option: str, replacement: str
) -> list[str]:
    """Replace the value of one option in both of its spellings.

    Once a known option consumes the next token that token is its value,
    whatever it looks like, so it is replaced unconditionally rather than
    being matched against the raw string.
    """
    attached = f"{option}="
    masked: list[str] = []
    expect_value = False
    for argument in arguments:
        if expect_value:
            expect_value = False
            masked.append(replacement)
            continue
        if argument == option:
            expect_value = True
            masked.append(argument)
            continue
        if argument.startswith(attached):
            masked.append(f"{attached}{replacement}")
            continue
        masked.append(argument)
    return masked


def _sanitized_command(
    config: APICollectionConfig, *, env_var_display: str
) -> tuple[str, ...]:
    """Replace the raw endpoint wherever it appears in the reconstruction.

    Matching whole arguments only would sanitize ``--endpoint <url>`` and
    miss the equally ordinary ``--endpoint=<url>``, leaving the raw query
    values in ``plan.command`` and in ``record.command.argv``. Substring
    replacement covers both spellings and any other reconstruction format
    without having to enumerate them, and the replacement is the same URL
    with its query values masked, so it stays a faithful command.

    The credential variable name is masked on the same terms the plan uses:
    if the environment did not vouch for it, it may be the credential, and
    the reconstructed command is an artifact like any other.
    """
    arguments = list(config.command_argv)
    safe_endpoint = _endpoint_for_command(config.endpoint)
    if safe_endpoint != config.endpoint:
        arguments = [
            argument.replace(config.endpoint, safe_endpoint) for argument in arguments
        ]
    if env_var_display != config.credential_env_var:
        arguments = _mask_option_value(arguments, _ENV_VAR_OPTION, env_var_display)
    return tuple(arguments)


def build_request_plan(
    config: APICollectionConfig, *, environ: Mapping[str, str] | None = None
) -> RequestPlan:
    """Describe the request without sending it and without any secret."""
    parts = parse_endpoint(config.endpoint)
    env_var_display = _persistable_env_var(config.credential_env_var, environ)
    messages = tuple(
        MessageDigest(
            role=message["role"],
            characters=len(message["content"]),
            content_sha256=sha256_text(message["content"]),
        )
        for message in _build_messages(config)
    )
    return RequestPlan(
        schema_version=API_EVIDENCE_SCHEMA_VERSION,
        provider=config.provider,
        method="POST",
        endpoint_origin=f"{parts.scheme}://{parts.netloc}",
        endpoint_path=parts.path,
        endpoint_query_keys=tuple(
            sorted(key for key, _ in parse_qsl(parts.query, keep_blank_values=True))
        ),
        model_id=config.model_id,
        model_revision=config.model_revision,
        credential_env_var=env_var_display,
        credential_header_name="Authorization",
        header_names=("Accept", "Authorization", "Content-Type"),
        messages=messages,
        request_parameters=_core_request_parameters(config),
        provider_extensions=config.extensions.to_request_fields(),
        finish_reasons=config.finish_reasons.identity(),
        request_timeout_seconds=float(config.request_timeout_seconds),
        command=_sanitized_command(config, env_var_display=env_var_display),
        config_hash=_config_identity_hash(config),
        workload_hash=sha256_text(config.prompt),
    )


# --- Evidence ----------------------------------------------------------------


@dataclass(frozen=True)
class ProviderUsage:
    """Token accounting exactly as the provider reported it."""

    reported: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_prompt_tokens: int | None = None
    """``usage.prompt_tokens_details.cached_tokens`` where exposed."""
    reasoning_tokens: int | None = None
    """``usage.completion_tokens_details.reasoning_tokens`` where exposed.
    Z.ai does not document this field for GLM models, so it is usually
    absent and stays ``None`` rather than being derived."""
    malformed_fields: tuple[str, ...] = ()
    """Usage fields the provider returned in an unusable shape. Recorded
    so a dropped value is visible instead of silently becoming ``None``."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reported": self.reported,
            "provenance": MetricProvenance.PROVIDER_REPORTED.value,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_prompt_tokens": self.cached_prompt_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "malformed_fields": list(self.malformed_fields),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ProviderUsage:
        malformed: list[str] = []

        def read(container: Any, key: str, label: str) -> int | None:
            if not isinstance(container, Mapping) or key not in container:
                return None
            value = container[key]
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int):
                malformed.append(label)
                return None
            if value < 0:
                malformed.append(label)
                return None
            if value > _MAX_EXACT_TOKEN_COUNT:
                # Beyond this a count cannot be converted to a float at all,
                # or converts with silent precision loss, so every rate
                # derived from it would be fiction. It is recorded as
                # malformed and dropped rather than carried, because
                # inventing a usable number here is the inference this
                # collector refuses to make everywhere else.
                malformed.append(label)
                return None
            return int(value)

        details = payload.get("prompt_tokens_details")
        completion_details = payload.get("completion_tokens_details")
        for label, container in (
            ("prompt_tokens_details", details),
            ("completion_tokens_details", completion_details),
        ):
            if container is not None and not isinstance(container, Mapping):
                malformed.append(label)

        return cls(
            reported=True,
            prompt_tokens=read(payload, "prompt_tokens", "prompt_tokens"),
            completion_tokens=read(payload, "completion_tokens", "completion_tokens"),
            total_tokens=read(payload, "total_tokens", "total_tokens"),
            cached_prompt_tokens=read(
                details, "cached_tokens", "prompt_tokens_details.cached_tokens"
            ),
            reasoning_tokens=read(
                completion_details,
                "reasoning_tokens",
                "completion_tokens_details.reasoning_tokens",
            ),
            malformed_fields=tuple(malformed),
        )


@dataclass(frozen=True)
class StreamEventTiming:
    """When one SSE event arrived, what kind it was, and how big it was."""

    index: int
    offset_ms: float
    kind: str
    characters: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "offset_ms": self.offset_ms,
            "kind": self.kind,
            "characters": self.characters,
        }


@dataclass(frozen=True)
class StreamTimeline:
    """Client-observed offsets, in milliseconds from the request start.

    Every offset is measured by this process against a monotonic clock and
    therefore includes network transport. ``response_headers_offset_ms``
    and ``first_body_chunk_offset_ms`` exist so a reader can see the
    transport-dominated part of the window instead of attributing all of
    time-to-first-token to model compute.
    """

    response_headers_offset_ms: float | None = None
    first_body_chunk_offset_ms: float | None = None
    first_content_token_offset_ms: float | None = None
    last_event_offset_ms: float | None = None
    completed_offset_ms: float | None = None
    events: tuple[StreamEventTiming, ...] = ()
    total_event_count: int = 0
    events_truncated: bool = False
    retained_event_limit: int = DEFAULT_RETAINED_EVENT_LIMIT

    def to_dict(self) -> dict[str, Any]:
        return {
            "clock": "monotonic_client_perf_counter",
            "provenance": MetricProvenance.MEASURED_WALL_CLOCK.value,
            "response_headers_offset_ms": self.response_headers_offset_ms,
            "first_body_chunk_offset_ms": self.first_body_chunk_offset_ms,
            "first_content_token_offset_ms": self.first_content_token_offset_ms,
            "last_event_offset_ms": self.last_event_offset_ms,
            "completed_offset_ms": self.completed_offset_ms,
            "events": [event.to_dict() for event in self.events],
            "total_event_count": self.total_event_count,
            "events_truncated": self.events_truncated,
            "retained_event_limit": self.retained_event_limit,
        }


@dataclass(frozen=True)
class LatencyDistribution:
    """Summary of inter-arrival gaps between successive content deltas."""

    count: int
    mean_ms: float
    min_ms: float
    median_ms: float
    p95_ms: float
    max_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms,
            "median_ms": self.median_ms,
            "p95_ms": self.p95_ms,
            "max_ms": self.max_ms,
        }


@dataclass(frozen=True)
class StreamStatistics:
    """Counts and rates derived from the timeline.

    A streaming delta is not guaranteed to be one token, so nothing here
    calls a delta a token. ``content_delta_rate_per_second`` is a delta
    rate. ``provider_completion_tokens_per_second`` mixes a
    provider-reported token count with a client-measured window and is
    labeled accordingly.

    Both rates are computed over one window, ``content_window_ms``, which
    runs from the first content delta arrival to the last. That window
    deliberately excludes the request, the response headers, any leading
    metadata chunk, and the trailing usage, finish-reason and ``[DONE]``
    events, none of which carry generated content. It is persisted so a
    consumer can see how wide the measurement actually was: a rate over a
    window of a few microseconds with two samples is arithmetic, not
    evidence, and only the window makes that visible.

    ``generation_window_ms`` is the wider window that starts at the first
    generated event of any kind, reasoning or content, and ends at the
    last. It exists because a provider-reported ``completion_tokens``
    counts reasoning tokens as well as visible ones, so the content window
    is the wrong denominator for it whenever reasoning was streamed first.
    With no reasoning deltas the two windows are identical.
    """

    content_delta_count: int = 0
    content_characters: int = 0
    reasoning_delta_count: int = 0
    reasoning_characters: int = 0
    metadata_event_count: int = 0
    comment_count: int = 0
    inter_content_delta: LatencyDistribution | None = None
    content_window_ms: float | None = None
    generation_window_ms: float | None = None
    content_delta_rate_per_second: float | None = None
    provider_completion_tokens_per_second: float | None = None
    provider_completion_tokens_per_second_unavailable_reason: str | None = None
    provider_visible_completion_tokens_per_second: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "content_delta_count": self.content_delta_count,
            "content_characters": self.content_characters,
            "reasoning_delta_count": self.reasoning_delta_count,
            "reasoning_characters": self.reasoning_characters,
            "metadata_event_count": self.metadata_event_count,
            "comment_count": self.comment_count,
            "inter_content_delta": (
                None
                if self.inter_content_delta is None
                else self.inter_content_delta.to_dict()
            ),
            "content_window_ms": self.content_window_ms,
            "content_window_definition": (
                "first content delta arrival to last content delta arrival, "
                "client-observed; excludes request, headers, leading metadata "
                "and the trailing usage/finish-reason/[DONE] events"
            ),
            "content_delta_rate_per_second": self.content_delta_rate_per_second,
            "content_delta_rate_provenance": MetricProvenance.DERIVED.value,
            "generation_window_ms": self.generation_window_ms,
            "generation_window_definition": (
                "first generated event arrival, reasoning or content, to the "
                "last, client-observed; equals content_window_ms when no "
                "reasoning delta was streamed"
            ),
            "provider_completion_tokens_per_second": (
                self.provider_completion_tokens_per_second
            ),
            "provider_completion_tokens_per_second_note": (
                "provider-reported completion tokens divided by "
                "generation_window_ms; mixed provenance. completion_tokens "
                "includes reasoning tokens where the provider counts them, "
                "so the wider generation window is the matching denominator. "
                "The window starts at the first generated delta, so that "
                "delta's own generation time is excluded, and when the delta "
                "count is far below completion_tokens the window endpoints "
                "are delta boundaries rather than token boundaries. Treat "
                "this as a coarse estimate, not a measured per-token rate."
            ),
            "provider_completion_tokens_per_second_unavailable_reason": (
                self.provider_completion_tokens_per_second_unavailable_reason
            ),
            "provider_visible_completion_tokens_per_second": (
                self.provider_visible_completion_tokens_per_second
            ),
            "provider_visible_completion_tokens_per_second_note": (
                "(completion tokens minus provider-reported reasoning "
                "tokens) divided by content_window_ms; mixed provenance. It "
                "is null whenever the provider did not report a reasoning "
                "token count, because a missing count is not zero and "
                "assuming it were would inflate the visible rate."
            ),
        }


@dataclass(frozen=True)
class APIFailure:
    """A failure, described without leaking anything sensitive."""

    category: str
    message: str
    status_code: int | None = None
    provider_error_code: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "message": self.message,
            "status_code": self.status_code,
            "provider_error_code": self.provider_error_code,
        }


@dataclass(frozen=True)
class APIEvidence:
    """Everything observed about one streaming call, safe to persist."""

    schema_version: str
    run_id: str
    collected_at: str
    plan: RequestPlan
    success: bool
    response_id: str | None = None
    provider_request_id: str | None = None
    response_model: str | None = None
    finish_reason: str | None = None
    finish_reason_classification: str | None = None
    """How the raw ``finish_reason`` was classified, decided before
    redaction could rewrite the text it was read from."""
    finish_reason_code: str | None = None
    """The documented spelling this collector recognized, or null when the
    provider sent something outside the documented set. Drawn from a value
    this collector defines rather than from provider bytes, so unlike
    ``finish_reason`` it carries no provider-controlled text."""
    usage: ProviderUsage = field(default_factory=ProviderUsage)
    timeline: StreamTimeline = field(default_factory=StreamTimeline)
    statistics: StreamStatistics = field(default_factory=StreamStatistics)
    rate_limit_headers: dict[str, str] = field(default_factory=dict)
    stream_terminated_with_done: bool = False
    stream_had_unterminated_event: bool = False
    reasoning_content_returned: bool = False
    """Whether the provider returned reasoning deltas. The reasoning text
    itself is never persisted."""
    failure: APIFailure | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "collected_at": self.collected_at,
            "plan": self.plan.to_dict(),
            "success": self.success,
            "response_id": self.response_id,
            "provider_request_id": self.provider_request_id,
            "response_model": self.response_model,
            "finish_reason": self.finish_reason,
            "finish_reason_classification": self.finish_reason_classification,
            "finish_reason_code": self.finish_reason_code,
            "usage": self.usage.to_dict(),
            "timeline": self.timeline.to_dict(),
            "statistics": self.statistics.to_dict(),
            "rate_limit_headers": dict(sorted(self.rate_limit_headers.items())),
            "stream_terminated_with_done": self.stream_terminated_with_done,
            "stream_had_unterminated_event": self.stream_had_unterminated_event,
            "reasoning_content_returned": self.reasoning_content_returned,
            "reasoning_text_persisted": False,
            "failure": None if self.failure is None else self.failure.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


@dataclass(frozen=True)
class APICollectionResult:
    """Canonical record, API evidence and the final answer text."""

    record: ExperimentRecord
    evidence: APIEvidence
    response_text: str


# --- Streaming accumulation --------------------------------------------------


class _StreamAccumulator:
    """Consume SSE events, timing each one, without keeping any secret text."""

    def __init__(
        self,
        *,
        clock: Callable[[], float],
        started: float,
        redactor: _Redactor,
        finish_reasons: FinishReasonVocabulary | None = None,
        retained_event_limit: int = DEFAULT_RETAINED_EVENT_LIMIT,
    ) -> None:
        self._clock = clock
        self._started = started
        self._retained_event_limit = retained_event_limit
        self.finish_reasons = finish_reasons or FinishReasonVocabulary()
        self.response_headers_at: float | None = None
        self.first_body_chunk_at: float | None = None
        self.first_content_token_at: float | None = None
        self.last_content_at: float | None = None
        self.last_event_at: float | None = None
        self.content_parts: list[str] = []
        self._redactor = redactor
        self.content_delta_count = 0
        self.reasoning_delta_count = 0
        self.reasoning_characters = 0
        self.metadata_event_count = 0
        self.comment_count = 0
        self.events: list[StreamEventTiming] = []
        self.total_event_count = 0
        self.content_arrival_times: list[float] = []
        self.response_id: str | None = None
        self.provider_request_id: str | None = None
        self.response_model: str | None = None
        self.finish_reason: str | None = None
        self.finish_outcome: str | None = None
        self.finish_reason_code: str | None = None
        self.first_generated_token_at: float | None = None
        self.last_generated_token_at: float | None = None
        self.usage: ProviderUsage = ProviderUsage()
        self.terminated_with_done = False
        self.incomplete_event_discarded = False

    def offset_ms(self, moment: float | None) -> float | None:
        if moment is None:
            return None
        return max(0.0, moment - self._started) * 1000

    def consume(
        self,
        response: StreamingResponse,
        *,
        redact: _Redactor,
        deadline: float | None = None,
    ) -> APIFailure | None:
        """Drain the body. Returns a failure, or ``None`` when it streamed.

        ``deadline`` is an absolute monotonic time past which the stream is
        abandoned. The transport timeout is a per-socket-operation timeout,
        so it only fires when the connection goes quiet; a server that emits
        a keepalive comment before each socket timeout expires resets it
        forever, and the run neither completes nor fails. That also bounds
        the memory a stream can consume, since every event appends a timing
        row.
        """
        decoder = SSEDecoder()
        observed_at: float | None = None
        try:
            for chunk in response.iter_bytes():
                # One clock read per network chunk, shared by every event it
                # completes. Reading the clock per event would time the
                # parser instead of the network and turn microseconds of
                # local CPU into inter-token latency that no observer could
                # have seen: several deltas can arrive in a single chunk.
                observed_at = self._clock()
                if self.first_body_chunk_at is None:
                    self.first_body_chunk_at = observed_at
                if deadline is not None and observed_at >= deadline:
                    return APIFailure(
                        category=FAILURE_TIMEOUT,
                        message=(
                            "the response did not finish within the configured "
                            f"{self._budget_text(deadline)} request timeout; "
                            "the stream was still open and is abandoned "
                            "incomplete"
                        ),
                    )
                for event in decoder.feed(chunk):
                    failure = self._handle_event(event, observed_at, redact=redact)
                    if failure is not None or self.terminated_with_done:
                        return failure
            final_at = self._clock() if observed_at is None else observed_at
            for event in decoder.close():
                failure = self._handle_event(event, final_at, redact=redact)
                if failure is not None or self.terminated_with_done:
                    return failure
        except SSEDecodeError as exc:
            return APIFailure(category=FAILURE_STREAM_DECODE, message=redact(str(exc)))
        except TransportTimeout as exc:
            return APIFailure(category=FAILURE_TIMEOUT, message=redact(str(exc)))
        except TransportConnectionError as exc:
            return APIFailure(category=FAILURE_CONNECTION, message=redact(str(exc)))
        finally:
            self.comment_count = decoder.comment_count
            self.incomplete_event_discarded = decoder.incomplete_event_discarded
        return None

    def _budget_text(self, deadline: float) -> str:
        """The configured budget, rendered from the deadline and start."""
        return f"{deadline - self._started:g}s"

    def _handle_event(
        self, event: SSEEvent, observed_at: float, *, redact: _Redactor
    ) -> APIFailure | None:
        now = observed_at
        self.last_event_at = now
        data = event.data.strip()

        # A named ``event: error`` frame is an error regardless of what its
        # data carries. The event name is therefore resolved before the data
        # is interpreted at all: checking for ``[DONE]`` first would let a
        # provider close a failed stream as though it had finished cleanly.
        named_error = (event.event or "").strip().lower() == "error"
        if named_error and data == "[DONE]":
            return APIFailure(
                category=FAILURE_PROVIDER_ERROR,
                message=(
                    "provider sent an 'error' event carrying the [DONE] "
                    "sentinel; the stream ended in error, not cleanly"
                ),
            )
        if named_error and not data:
            return APIFailure(
                category=FAILURE_PROVIDER_ERROR,
                message="provider sent an 'error' event with an empty payload",
            )

        if data == "[DONE]":
            self.terminated_with_done = True
            return None
        if not data:
            self._record_event(_EVENT_KIND_METADATA, 0, now)
            return None

        try:
            payload = json.loads(data)
        # ``json`` raises past its own limits with exceptions that are not
        # ``JSONDecodeError``: an integer literal over the interpreter's
        # digit cap raises a plain ``ValueError``, and deep nesting raises
        # ``RecursionError``. Both are reachable from provider-controlled
        # bytes, and both used to escape as an unhandled crash with no
        # failure-shaped evidence written at all.
        except (ValueError, RecursionError) as exc:
            if named_error:
                # ``data`` reaches here mainly when the payload was cut in
                # half, since a complete line would have parsed. The cut can
                # fall inside an echoed credential, so repair the boundary
                # before the raw line is interpolated.
                return APIFailure(
                    category=FAILURE_PROVIDER_ERROR,
                    message=redact(f"provider 'error' event: {redact.boundary(data)}"),
                )
            return APIFailure(
                category=FAILURE_STREAM_DECODE,
                message=redact(f"stream chunk is not valid JSON: {exc}"),
            )
        if not isinstance(payload, dict):
            if named_error:
                return APIFailure(
                    category=FAILURE_PROVIDER_ERROR,
                    message=redact(f"provider 'error' event: {payload}"),
                )
            return APIFailure(
                category=FAILURE_STREAM_DECODE,
                message="stream chunk is not a JSON object",
            )

        provider_failure = _provider_error_from_payload(payload, redact=redact)
        if provider_failure is not None:
            self._absorb_identity(payload, redact=redact)
            return provider_failure

        self._absorb_identity(payload, redact=redact)

        if named_error:
            return APIFailure(
                category=FAILURE_PROVIDER_ERROR,
                message=redact(_error_message(payload)),
                provider_error_code=redact.identifier(_error_code(payload)),
            )

        usage_payload = payload.get("usage")
        if usage_payload is not None:
            if not isinstance(usage_payload, dict):
                return APIFailure(
                    category=FAILURE_STREAM_DECODE,
                    message="stream chunk 'usage' is not a JSON object",
                )
            self.usage = ProviderUsage.from_payload(usage_payload)

        choice, choices_malformed = _first_choice(payload)
        if choices_malformed:
            # ``{"choices": {}}`` or ``{"choices": [7]}`` is a broken frame,
            # not a metadata frame. Treating it as metadata would let a
            # malformed tail be published as a successful run.
            return APIFailure(
                category=FAILURE_STREAM_DECODE,
                message="stream chunk 'choices' is present but malformed",
            )
        if choice is None:
            self._record_event(_EVENT_KIND_METADATA, 0, now)
            return None

        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            if not isinstance(finish_reason, str):
                return APIFailure(
                    category=FAILURE_STREAM_DECODE,
                    message="stream chunk 'finish_reason' is not a string",
                )
            # Classify what the provider actually sent, before redaction
            # rewrites it. Redaction is a text transform on an untrusted
            # string, so it can dissolve a documented reason: a credential
            # containing "error" turns "network_error" into
            # "network_[REDACTED]", the failure is no longer recognized and
            # a following [DONE] publishes an aborted generation as a
            # success. Meaning is decided first, text is redacted second.
            normalized = finish_reason.strip().lower()
            if normalized in self.finish_reasons.failure:
                self.finish_outcome = _FINISH_FAILURE
                self.finish_reason_code = normalized
                self.finish_reason = redact.identifier(finish_reason)
            elif self.finish_outcome == _FINISH_FAILURE:
                # A failure the provider already reported is not undone by a
                # later chunk. Streams carry one finish reason per choice in
                # practice, but nothing in the wire format prevents a second,
                # and last-write-wins would let a trailing "stop" erase
                # "network_error" and publish an aborted generation as a
                # success with a full latency timeline. The first failure is
                # the outcome, so it is kept for both the code and the
                # persisted text, which therefore always agree.
                pass
            elif normalized in self.finish_reasons.terminal:
                self.finish_outcome = _FINISH_TERMINAL
                self.finish_reason_code = normalized
                self.finish_reason = redact.identifier(finish_reason)
            else:
                self.finish_outcome = _FINISH_UNRECOGNIZED
                self.finish_reason_code = None
                self.finish_reason = redact.identifier(finish_reason)

        delta = choice.get("delta")
        if delta is not None and not isinstance(delta, dict):
            return APIFailure(
                category=FAILURE_STREAM_DECODE,
                message="stream chunk 'delta' is not a JSON object",
            )
        delta = delta or {}

        content = delta.get("content")
        if content is not None and not isinstance(content, str):
            return APIFailure(
                category=FAILURE_STREAM_DECODE,
                message=(
                    "stream chunk 'delta.content' is not a string; this "
                    "collector only records text completions"
                ),
            )
        reasoning = _reasoning_delta(delta)
        if reasoning is not None and not isinstance(reasoning, str):
            return APIFailure(
                category=FAILURE_STREAM_DECODE,
                message="stream chunk reasoning delta is not a string",
            )

        if content:
            # Scrubbed before it is stored, so a provider that echoes the
            # credential back inside generated text cannot reach response.txt.
            content = redact.text(content)
        if content:
            if self.first_content_token_at is None:
                self.first_content_token_at = now
            if self.first_generated_token_at is None:
                self.first_generated_token_at = now
            self.content_arrival_times.append(now)
            self.last_content_at = now
            self.last_generated_token_at = now
            self.content_parts.append(content)
            self.content_delta_count += 1
            self._record_event(_EVENT_KIND_CONTENT, len(content), now)
            if reasoning:
                self.reasoning_delta_count += 1
                self.reasoning_characters += len(reasoning)
            return None

        if reasoning:
            if self.first_generated_token_at is None:
                self.first_generated_token_at = now
            self.last_generated_token_at = now
            self.reasoning_delta_count += 1
            self.reasoning_characters += len(reasoning)
            self._record_event(_EVENT_KIND_REASONING, len(reasoning), now)
            return None

        self._record_event(_EVENT_KIND_METADATA, 0, now)
        return None

    def _absorb_identity(
        self, payload: Mapping[str, Any], *, redact: _Redactor
    ) -> None:
        for key, attribute in (
            ("id", "response_id"),
            ("request_id", "provider_request_id"),
            ("model", "response_model"),
        ):
            value = payload.get(key)
            if isinstance(value, str) and value and getattr(self, attribute) is None:
                # Provider-controlled, so it is scrubbed like any other
                # untrusted string before it can reach an artifact.
                setattr(self, attribute, redact.identifier(value))

    def _record_event(self, kind: str, characters: int, moment: float) -> None:
        if kind == _EVENT_KIND_METADATA:
            self.metadata_event_count += 1
        offset = self.offset_ms(moment)
        index = self.total_event_count
        self.total_event_count += 1
        if index >= self._retained_event_limit:
            # The counters above already ran, so dropping the row costs the
            # per-event detail and nothing else.
            return
        self.events.append(
            StreamEventTiming(
                index=index,
                offset_ms=0.0 if offset is None else offset,
                kind=kind,
                characters=characters,
            )
        )

    def terminated_cleanly(self) -> bool:
        """True when the provider signalled a real, successful end of stream.

        Either the ``[DONE]`` sentinel or a terminal ``finish_reason`` is
        accepted, because both are documented endings and not every
        OpenAI-compatible provider sends both.

        A documented failure ``finish_reason`` and a frame the stream left
        pending are both checked first. Either one means bytes were lost or
        the generation was aborted, and neither is cancelled out by a
        ``[DONE]`` that happens to follow.
        """
        if self.failed_finish_reason() is not None:
            return False
        if self.incomplete_event_discarded:
            return False
        if self.terminated_with_done:
            return True
        return self.finish_outcome == _FINISH_TERMINAL

    def failed_finish_reason(self) -> str | None:
        """The documented failure reason the provider reported, if any.

        The value returned is the documented spelling this collector
        recognized, not the provider's bytes, so it is safe to persist as
        an error code and cannot carry an echoed credential.
        """
        if self.finish_outcome != _FINISH_FAILURE:
            return None
        return self.finish_reason_code

    def content_text(self) -> str:
        """The final answer, scrubbed as one string.

        Per-delta redaction is not sufficient on its own. A provider that
        wants the key back can dribble it across delta boundaries, five
        characters at a time, and no individual delta ever contains it.
        The assembled text is therefore the authoritative scrub, and every
        persisted length is derived from this value so the record cannot
        disagree with ``response.txt``.

        A truncated stream is still persisted, so a cut that landed inside
        an echoed credential leaves a trailing fragment the substring scrub
        cannot see. That boundary is repaired only when the stream did not
        end cleanly, so a complete answer is never altered.
        """
        joined = self._redactor.text("".join(self.content_parts))
        if self.terminated_cleanly():
            return joined
        return self._redactor.boundary(joined)

    def statistics(self, *, reasoning_ruled_out: bool = False) -> StreamStatistics:
        gaps = [
            (later - earlier) * 1000
            for earlier, later in zip(
                self.content_arrival_times, self.content_arrival_times[1:], strict=False
            )
        ]
        distribution = _latency_distribution(gaps)

        # One window for both rates: first content arrival to last content
        # arrival. Anchoring the end on ``last_event_at`` instead would
        # fold the trailing usage/finish-reason/[DONE] events, which carry
        # no generated content, into a decode denominator.
        # A second window, ``generation_window_ms``, runs from the first
        # generated event of any kind to the last. When the provider streams
        # reasoning deltas before the answer, that window is the one the
        # provider's completion-token count was produced over: Z.ai counts
        # reasoning tokens inside ``completion_tokens``, so dividing them by
        # the content window would credit a long silent reasoning phase to a
        # short visible one and overstate throughput.
        window_seconds: float | None = None
        if (
            len(self.content_arrival_times) > 1
            and self.content_arrival_times[-1] > self.content_arrival_times[0]
        ):
            window_seconds = (
                self.content_arrival_times[-1] - self.content_arrival_times[0]
            )

        generation_seconds: float | None = None
        first_generated = self.first_generated_token_at
        last_generated = self.last_generated_token_at
        if (
            first_generated is not None
            and last_generated is not None
            and last_generated > first_generated
        ):
            generation_seconds = last_generated - first_generated

        delta_rate = None
        token_rate = None
        visible_token_rate = None
        rate_unavailable: str | None = None
        if window_seconds is not None:
            # ``n`` arrivals bound ``n - 1`` intervals, so the delta rate is
            # measured over the gaps it actually observed.
            delta_rate = (len(self.content_arrival_times) - 1) / window_seconds
            completion_tokens = self.usage.completion_tokens
            reasoning_tokens = self.usage.reasoning_tokens
            if (
                completion_tokens is not None
                and reasoning_tokens is not None
                and completion_tokens >= reasoning_tokens
            ):
                visible_token_rate = (
                    completion_tokens - reasoning_tokens
                ) / window_seconds
        if generation_seconds is not None:
            completion_tokens = self.usage.completion_tokens
            reasoning_tokens = self.usage.reasoning_tokens
            if (
                reasoning_tokens is not None
                and reasoning_tokens > 0
                and self.reasoning_delta_count == 0
            ):
                # The provider counted reasoning tokens but streamed no
                # reasoning delta, so the generation window collapses onto
                # the visible answer while the numerator still carries
                # tokens produced before it. There is no window here that
                # spans the numerator, so no honest rate exists.
                rate_unavailable = (
                    "provider reported reasoning tokens but streamed no "
                    "reasoning delta, so the period those tokens were "
                    "generated in was never observed and any rate over the "
                    "visible window would overstate throughput"
                )
            elif not (
                reasoning_ruled_out
                or reasoning_tokens == 0
                or self.reasoning_delta_count > 0
            ):
                # The provider accounted for reasoning in neither of the two
                # ways it can: no reasoning delta on the wire and no
                # reasoning token count in usage. Silence is not evidence of
                # absence here, because a provider that defaults to thinking
                # produces exactly this shape, and treating the absent count
                # as zero is the inference this collector refuses to make
                # everywhere else. Ruling it out takes either an explicit
                # ``thinking.type=disabled`` on the request, an explicit zero
                # reasoning token count, or an observed reasoning delta.
                rate_unavailable = (
                    "the provider reported neither reasoning deltas nor a "
                    "reasoning token count, and thinking was not explicitly "
                    "disabled, so a hidden reasoning phase cannot be ruled "
                    "out and the observed window may not span every counted "
                    "token"
                )
            elif completion_tokens is not None:
                token_rate = completion_tokens / generation_seconds
            else:
                # The window is fine and reasoning is accounted for, but the
                # provider sent no completion token count to divide by it.
                # The rate is null for a reason that has nothing to do with
                # this client's measurement, and saying so is what keeps it
                # from reading as a window we failed to observe.
                rate_unavailable = (
                    "the provider reported no completion token count, so "
                    "there is no number to divide the measured generation "
                    "window by"
                )
        elif self.usage.completion_tokens is not None:
            # There is a completion count but no window with any width to
            # divide it by: every generated token arrived inside one network
            # chunk, or only one arrived. The rate is null either way, and
            # without this it was null with nothing recorded to say why,
            # which reads as an absent metric rather than an unmeasurable
            # one.
            rate_unavailable = (
                "the generated tokens did not arrive far enough apart to "
                "measure a window, so no period exists to divide the "
                "provider's completion token count by"
            )

        return StreamStatistics(
            content_delta_count=self.content_delta_count,
            content_characters=len(self.content_text()),
            reasoning_delta_count=self.reasoning_delta_count,
            reasoning_characters=self.reasoning_characters,
            metadata_event_count=self.metadata_event_count,
            comment_count=self.comment_count,
            inter_content_delta=distribution,
            content_window_ms=(
                None if window_seconds is None else window_seconds * 1000
            ),
            generation_window_ms=(
                None if generation_seconds is None else generation_seconds * 1000
            ),
            content_delta_rate_per_second=delta_rate,
            provider_completion_tokens_per_second=token_rate,
            provider_completion_tokens_per_second_unavailable_reason=rate_unavailable,
            provider_visible_completion_tokens_per_second=visible_token_rate,
        )

    def timeline(self, completed_at: float) -> StreamTimeline:
        return StreamTimeline(
            response_headers_offset_ms=self.offset_ms(self.response_headers_at),
            first_body_chunk_offset_ms=self.offset_ms(self.first_body_chunk_at),
            first_content_token_offset_ms=self.offset_ms(self.first_content_token_at),
            last_event_offset_ms=self.offset_ms(self.last_event_at),
            completed_offset_ms=self.offset_ms(completed_at),
            events=tuple(self.events),
            total_event_count=self.total_event_count,
            events_truncated=self.total_event_count > len(self.events),
            retained_event_limit=self._retained_event_limit,
        )


def _latency_distribution(gaps: list[float]) -> LatencyDistribution | None:
    if not gaps:
        return None
    ordered = sorted(gaps)
    rank = max(1, -(-95 * len(ordered) // 100))
    return LatencyDistribution(
        count=len(ordered),
        mean_ms=sum(ordered) / len(ordered),
        min_ms=ordered[0],
        median_ms=float(median(ordered)),
        p95_ms=ordered[rank - 1],
        max_ms=ordered[-1],
    )


def _first_choice(payload: Mapping[str, Any]) -> tuple[dict[str, Any] | None, bool]:
    """Return ``(choice, malformed)`` for the first choice in a chunk.

    An absent or empty ``choices`` is normal: GLM's usage-only final chunk
    has none. A ``choices`` that is present with the wrong type, or whose
    first element is not an object, is a broken frame and is reported as
    malformed so it cannot be mistaken for a metadata frame.
    """
    if "choices" not in payload:
        return None, False
    choices = payload.get("choices")
    if choices is None:
        return None, False
    if not isinstance(choices, list):
        return None, True
    if not choices:
        return None, False
    first = choices[0]
    if not isinstance(first, dict):
        return None, True
    return first, False


def _reasoning_delta(delta: Mapping[str, Any]) -> Any:
    """Return the reasoning delta under whichever key the provider used.

    Z.ai documents ``reasoning_content``. ``reasoning`` is accepted too
    because other OpenAI-compatible providers use it; the text is only
    ever counted, never stored.
    """
    for key in ("reasoning_content", "reasoning"):
        if key in delta and delta[key] is not None:
            return delta[key]
    return None


def _provider_error_from_payload(
    payload: Mapping[str, Any], *, redact: _Redactor
) -> APIFailure | None:
    """Detect an error object delivered inside an otherwise 200 stream."""
    error = payload.get("error")
    if isinstance(error, Mapping):
        return APIFailure(
            category=FAILURE_PROVIDER_ERROR,
            message=redact(_error_message(error)),
            provider_error_code=redact.identifier(_error_code(error)),
        )
    if error is not None:
        return APIFailure(
            category=FAILURE_PROVIDER_ERROR,
            message=redact(str(error)),
        )
    # Z.ai returns a bare {"code": ..., "message": ...} object rather than
    # OpenAI's {"error": {...}} wrapper.
    if "choices" not in payload and "message" in payload and "code" in payload:
        return APIFailure(
            category=FAILURE_PROVIDER_ERROR,
            message=redact(_error_message(payload)),
            provider_error_code=redact.identifier(_error_code(payload)),
        )
    return None


def _error_message(error: Mapping[str, Any]) -> str:
    message = error.get("message")
    if isinstance(message, str) and message.strip():
        return message
    return "provider returned an error without a message"


def _error_code(error: Mapping[str, Any]) -> str | None:
    code = error.get("code")
    if isinstance(code, bool):
        return None
    if isinstance(code, (str, int)):
        text = str(code).strip()
        return text or None
    return None


# --- Collection --------------------------------------------------------------


def _resolve_credential(config: APICollectionConfig, environ: Mapping[str, str]) -> str:
    value = environ.get(config.credential_env_var)
    if value is None:
        # The name is not repeated. A name absent from the environment was
        # never proven to be a name, and the likeliest reason it is absent is
        # that the caller passed the credential in the name slot.
        raise OpenAIStreamCollectorError(
            "the environment variable named by --api-key-env is not set in "
            "this environment; export the API key there (it is never accepted "
            "as a command argument and never written to any artifact)"
        )
    # Strip first. A key read from a file or a ``.env`` routinely carries a
    # trailing newline, and a newline in a header value makes
    # ``http.client.putheader`` raise a ``ValueError`` whose message embeds
    # the whole value, which would print the secret in a traceback.
    credential = value.strip()
    if not credential:
        raise OpenAIStreamCollectorError(
            f"environment variable {_persistable_env_var(config.credential_env_var, environ)} "
            "is empty"
        )
    _assert_header_safe_credential(
        credential, _persistable_env_var(config.credential_env_var, environ)
    )
    return credential


def _assert_header_safe_credential(credential: str, env_var: str) -> None:
    """Reject a credential that cannot be sent as an HTTP header value.

    The diagnostic names the environment variable and the offending
    character class only. It never reproduces the value.
    """
    for index, char in enumerate(credential):
        if ord(char) < 0x20 or ord(char) == 0x7F:
            raise OpenAIStreamCollectorError(
                f"the value of {env_var} contains a control character at "
                f"offset {index} and cannot be sent as an HTTP header value"
            )
    try:
        credential.encode("latin-1")
    except UnicodeEncodeError:
        raise OpenAIStreamCollectorError(
            f"the value of {env_var} contains a non latin-1 character and "
            "cannot be sent as an HTTP header value"
        ) from None


_MAX_PERCENT_DECODE_ROUNDS = 3


def _percent_decodings(value: str) -> list[str]:
    """``value`` plus the forms it decodes to, up to a bounded depth.

    A credential pasted into a URL is usually percent encoded, so a literal
    containment test misses it: ``abc/def`` is written ``abc%2Fdef`` and
    survives the check while remaining trivially reversible once
    persisted. Decoding the haystack rather than enumerating encodings of
    the needle handles case variants (``%2F`` and ``%2f``), ``+`` for
    space, and mixed encodings for free. A few rounds are applied because
    ``%252F`` decodes to ``%2F`` before it decodes to ``/``.
    """
    seen = [value]
    current = value
    for _ in range(_MAX_PERCENT_DECODE_ROUNDS):
        if "%" not in current and "+" not in current:
            break
        candidates = []
        for decoder in (unquote, unquote_plus):
            try:
                decoded = decoder(current)
            except (UnicodeDecodeError, ValueError):
                continue
            if decoded not in seen:
                candidates.append(decoded)
                seen.append(decoded)
        if not candidates:
            break
        current = candidates[0]
    return seen


def _contains_credential(value: str, credential: str) -> bool:
    """True when ``credential`` is present in any spelling the scrub knows.

    This guards the same threat as ``_Redactor`` and therefore uses the
    same matcher: case-insensitive, whitespace-flexible and encoding
    aware. A preflight that were merely literal would pass a provider
    identifier echoing the key in lower case, or an extension value with a
    tab where the key has a space, and the refusal that exists to stop the
    value being persisted would not fire on the exact shapes redaction was
    built to catch.

    Short credentials skip the extra decoding rounds only. A one or two
    character value would otherwise match a decoded byte by coincidence,
    and a refusal that fires on noise trains people to work around it. The
    flexible match itself runs at every length, because the redactor runs
    at every length: a value the redactor would scrub must never pass the
    check that decides whether it may be persisted at all.
    """
    if credential in value:
        return True
    matcher = _Redactor(credential)
    if matcher.search(value):
        return True
    if len(credential.strip()) < _MIN_ENCODED_CREDENTIAL_CHARS:
        return False
    # ``_percent_decodings`` yields the value itself first, so this covers
    # the direct search as well as the ``+``-for-space and multi-round
    # decodings that a per-character matcher cannot express.
    return any(matcher.search(decoded) for decoded in _percent_decodings(value))


def _assert_credential_not_embedded(
    credential: str, config: APICollectionConfig
) -> None:
    """Refuse to run if the secret is anywhere it would then be persisted."""
    haystacks: list[tuple[str, str]] = [
        ("run_id", config.run_id),
        ("endpoint", config.endpoint),
        ("provider", config.provider),
        ("model_id", config.model_id),
        ("prompt", config.prompt),
        # The variable name is persisted in the plan whenever it resolves,
        # so a value that is somehow both the name and the secret has to be
        # refused here rather than relied on to be masked downstream.
        ("credential_env_var", config.credential_env_var),
        # The output directory becomes a pathname on disk. A credential
        # there is written into the filesystem itself, where no downstream
        # redactor can reach it.
        ("output_dir", str(config.output_dir)),
    ]
    optional: tuple[tuple[str, str | None], ...] = (
        ("model_revision", config.model_revision),
        ("system_prompt", config.system_prompt),
        ("reasoning_effort", config.extensions.reasoning_effort),
        ("thinking_type", config.extensions.thinking_type),
        ("provider_request_id", config.extensions.provider_request_id),
    )
    haystacks.extend((label, value) for label, value in optional if value is not None)
    haystacks.extend(
        (f"command_argv[{index}]", value)
        for index, value in enumerate(config.command_argv)
    )
    for label, value in haystacks:
        if _contains_credential(value, credential):
            raise OpenAIStreamCollectorError(
                f"the value named by {_ENV_VAR_OPTION} appears in {label}; "
                "refusing to run because that value would be persisted"
            )


def assert_credential_not_embedded(
    config: APICollectionConfig, environ: Mapping[str, str]
) -> None:
    """Public pre-flight check used by ``--dry-run``.

    A real run refuses to start when the key is sitting in the endpoint or
    the command, so a dry run has to apply the same rule. Otherwise the
    validation step would report a plan as fine and the real call would
    then refuse, which is the opposite of what a pre-flight check is for.
    """
    credential = environ.get(config.credential_env_var)
    if credential is None or not credential.strip():
        return
    _assert_credential_not_embedded(credential.strip(), config)


def _rate_limit_headers(
    headers: Mapping[str, str], *, redact: _Redactor
) -> dict[str, str]:
    collected: dict[str, str] = {}
    for name, value in headers.items():
        lowered = name.lower()
        if lowered in _RATE_LIMIT_HEADER_NAMES or lowered.startswith(
            _RATE_LIMIT_HEADER_PREFIXES
        ):
            # The name is provider controlled too. Header name tokens allow
            # the same alphabet most API keys use, so a server can echo the
            # credential in the name and have it persisted as a dict key.
            safe_name = redact(lowered, limit=_MAX_PERSISTED_HEADER_CHARS)
            collected[safe_name] = redact(str(value), limit=_MAX_PERSISTED_HEADER_CHARS)
    return collected


def _request_id_from_headers(headers: Mapping[str, str]) -> str | None:
    # Deliberately untruncated. Redaction matches the credential as an exact
    # substring, so cutting the value here first would slice through an
    # echoed credential and let the surviving prefix pass the scrub. The
    # caller bounds the value through ``redact.identifier``, which scrubs
    # before it truncates, the same order ``_rate_limit_headers`` uses.
    lowered = {name.lower(): value for name, value in headers.items()}
    for name in _REQUEST_ID_HEADER_NAMES:
        value = lowered.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _read_error_body(response: StreamingResponse) -> bytes:
    collected = bytearray()
    try:
        for chunk in response.iter_bytes():
            collected.extend(chunk)
            if len(collected) >= _MAX_ERROR_BODY_BYTES:
                break
    except (TransportTimeout, TransportConnectionError):
        # The status code is the evidence that matters; a truncated or
        # unreadable error body must not replace it.
        return bytes(collected[:_MAX_ERROR_BODY_BYTES])
    return bytes(collected[:_MAX_ERROR_BODY_BYTES])


def _terminal_condition_failure(
    accumulator: _StreamAccumulator, response_text: str
) -> APIFailure | None:
    """Decide whether a stream that raised nothing actually succeeded.

    The three ways a stream can end badly without erroring are ordered by
    how much they explain. A documented failure ``finish_reason`` is the
    provider telling us directly why generation stopped, so it is reported
    even when ``[DONE]`` followed and even when some content arrived. A
    frame the stream left pending means the connection was cut mid event.
    Only then is the generic "no terminal condition at all" case reached.
    """
    failed_reason = accumulator.failed_finish_reason()
    if failed_reason is not None:
        return APIFailure(
            category=FAILURE_PROVIDER_ERROR,
            message=(
                "provider reported a failure finish_reason; the stream did "
                "not carry a completed generation"
            ),
            provider_error_code=failed_reason,
        )

    if accumulator.incomplete_event_discarded:
        return APIFailure(
            category=FAILURE_STREAM_TRUNCATED,
            message=(
                "stream ended in the middle of an event; the pending frame "
                "was discarded and the response is truncated"
            ),
            provider_error_code=None,
        )

    if not response_text:
        return APIFailure(
            category=FAILURE_MISSING_CONTENT,
            message=(
                "stream completed without any content delta; finish_reason="
                f"{accumulator.finish_reason!r}"
            ),
            provider_error_code=None,
        )

    if not accumulator.terminated_cleanly():
        # Content alone does not mean the answer is whole. Without ``[DONE]``
        # or a terminal finish_reason the body was cut short, and publishing
        # a truncated answer as a success would corrupt every downstream
        # comparison that reads it.
        return APIFailure(
            category=FAILURE_STREAM_TRUNCATED,
            message=(
                "stream ended without a terminal condition: no [DONE] "
                "sentinel and finish_reason="
                f"{accumulator.finish_reason!r}; the response is truncated"
            ),
            provider_error_code=None,
        )

    return None


def _http_status_failure(
    response: StreamingResponse,
    *,
    redact: _Redactor,
    accumulator: _StreamAccumulator,
) -> APIFailure:
    body = _read_error_body(response).decode("utf-8", errors="replace")
    # The byte cap in ``_read_error_body`` can cut through an echoed
    # credential, so repair that boundary before the body is parsed or used.
    body = redact.boundary(body)
    message = f"HTTP {response.status_code}"
    code: str | None = None
    try:
        payload = json.loads(body) if body.strip() else None
    except (ValueError, RecursionError):
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        source = error if isinstance(error, Mapping) else payload
        message = f"HTTP {response.status_code}: {_error_message(source)}"
        code = _error_code(source)
        # Z.ai returns its request id in the body, and an error response
        # need not carry the header form. Discarding it would throw away
        # the one identifier that makes a failed call traceable with the
        # provider, which is exactly when it is most needed. Only the id
        # is taken: an error body's other fields do not describe a
        # response that was produced.
        for candidate in (payload, error):
            if not isinstance(candidate, Mapping):
                continue
            found = candidate.get("request_id")
            if isinstance(found, str) and found:
                accumulator.provider_request_id = redact.identifier(found)
                break
    elif body.strip():
        message = f"HTTP {response.status_code}: {body}"
    return APIFailure(
        category=FAILURE_HTTP_STATUS,
        message=redact(message),
        status_code=response.status_code,
        provider_error_code=redact.identifier(code),
    )


def collect_openai_stream(
    config: APICollectionConfig,
    *,
    transport: StreamingTransport,
    environ: Mapping[str, str] | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> APICollectionResult:
    """Stream one chat completion and persist normalized evidence.

    Transport, HTTP and protocol failures become failure-shaped records.
    Invalid configuration, a missing credential and artifact write
    failures stay explicit exceptions, because none of them describes a
    request that was actually attempted.
    """
    resolved_environ = os.environ if environ is None else environ
    credential = _resolve_credential(config, resolved_environ)
    _assert_credential_not_embedded(credential, config)
    redact = _Redactor(credential)
    plan = build_request_plan(config, environ=resolved_environ)

    request = HTTPRequest(
        url=config.endpoint,
        method="POST",
        headers={
            "Authorization": f"Bearer {credential}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        body=json.dumps(_request_body(config), allow_nan=False).encode("utf-8"),
        timeout_seconds=float(config.request_timeout_seconds),
    )

    started_at = utc_now_iso()
    started = clock()
    accumulator = _StreamAccumulator(
        clock=clock,
        started=started,
        redactor=redact,
        finish_reasons=config.finish_reasons,
        retained_event_limit=config.retained_event_limit,
    )
    failure: APIFailure | None = None
    rate_limit_headers: dict[str, str] = {}
    header_request_id: str | None = None

    try:
        response = transport.open_stream(request)
    except TransportTimeout as exc:
        failure = APIFailure(category=FAILURE_TIMEOUT, message=redact(str(exc)))
    except TransportConnectionError as exc:
        failure = APIFailure(category=FAILURE_CONNECTION, message=redact(str(exc)))
    else:
        accumulator.response_headers_at = clock()
        try:
            headers = response.headers
            rate_limit_headers = _rate_limit_headers(headers, redact=redact)
            header_request_id = _request_id_from_headers(headers)
            header_request_id = redact.identifier(header_request_id)
            if response.status_code != 200:
                failure = _http_status_failure(
                    response, redact=redact, accumulator=accumulator
                )
            else:
                # The same budget the transport gets per socket operation
                # is also the whole-response budget. A stream that stays
                # chatty past it is abandoned as a timeout rather than
                # followed indefinitely.
                failure = accumulator.consume(
                    response,
                    redact=redact,
                    deadline=started + float(config.request_timeout_seconds),
                )
        finally:
            try:
                response.close()
            except (OSError, http.client.HTTPException):
                # Releasing the socket must never discard collected evidence.
                pass

    completed = clock()
    response_text = accumulator.content_text()

    if failure is None:
        failure = _terminal_condition_failure(accumulator, response_text)

    provider_request_id = accumulator.provider_request_id or header_request_id

    evidence = APIEvidence(
        schema_version=API_EVIDENCE_SCHEMA_VERSION,
        run_id=config.run_id,
        collected_at=started_at,
        plan=plan,
        success=failure is None,
        response_id=accumulator.response_id,
        provider_request_id=provider_request_id,
        response_model=accumulator.response_model,
        finish_reason=accumulator.finish_reason,
        finish_reason_classification=accumulator.finish_outcome,
        finish_reason_code=accumulator.finish_reason_code,
        usage=accumulator.usage,
        timeline=accumulator.timeline(completed),
        statistics=accumulator.statistics(
            reasoning_ruled_out=_reasoning_ruled_out(config.extensions)
        ),
        rate_limit_headers=rate_limit_headers,
        stream_terminated_with_done=accumulator.terminated_with_done,
        stream_had_unterminated_event=accumulator.incomplete_event_discarded,
        reasoning_content_returned=accumulator.reasoning_delta_count > 0,
        failure=failure,
    )

    record = _build_record(
        config,
        plan=plan,
        accumulator=accumulator,
        started_at=started_at,
        started=started,
        completed=completed,
        failure=failure,
    )

    _assert_finite_evidence(record, evidence)

    _publish_artifacts(
        config.output_dir,
        run_id=config.run_id,
        record=record,
        evidence=evidence,
        response_text=response_text,
    )
    return APICollectionResult(
        record=record, evidence=evidence, response_text=response_text
    )


def _publish_artifacts(
    output_dir: Path,
    *,
    run_id: str,
    record: ExperimentRecord,
    evidence: APIEvidence,
    response_text: str,
) -> None:
    """Write the artifact set, marking it complete only once it all landed.

    Each individual write is atomic, but a run produces four files and a
    crash between them used to leave a brand new record.json beside a
    stale api_evidence.json from an earlier run of the same id. That set
    reads as successful and is silently wrong. The marker is removed
    first and written last, so any interruption leaves a set that
    ``artifact_set_is_complete`` rejects instead of one that lies.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    marker_path = output_dir / ARTIFACT_MANIFEST_NAME
    marker_path.unlink(missing_ok=True)

    payloads: list[tuple[str, str]] = [
        ("record.json", record.to_json() + "\n"),
        ("response.txt", response_text),
        ("api_evidence.json", evidence.to_json() + "\n"),
        ("environment.json", collect_environment_manifest().to_json() + "\n"),
    ]
    for name, text in payloads:
        atomic_write_text(output_dir / name, text)

    marker = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "run_id": run_id,
        "artifacts": [
            # Hashed as bytes, and verified as bytes, so a response holding
            # CRLF or a lone CR cannot make a correctly written set look
            # tampered with.
            {"name": name, "sha256": sha256_bytes(text.encode("utf-8"))}
            for name, text in payloads
        ],
    }
    atomic_write_text(marker_path, json.dumps(marker, indent=2, allow_nan=False) + "\n")


def artifact_set_is_complete(output_dir: Path) -> bool:
    """True when ``output_dir`` holds a complete, self-consistent set.

    A consumer must call this before trusting a run directory. A missing
    marker means an interrupted write; a hash mismatch means a file was
    replaced independently of the set it belongs to.
    """
    marker_path = output_dir / ARTIFACT_MANIFEST_NAME
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        return False
    if not isinstance(marker, dict):
        return False
    if marker.get("schema_version") != ARTIFACT_MANIFEST_SCHEMA_VERSION:
        return False
    artifacts = marker.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False
    for entry in artifacts:
        if not isinstance(entry, dict):
            return False
        name = entry.get("name")
        digest = entry.get("sha256")
        if not isinstance(name, str) or not isinstance(digest, str):
            return False
        try:
            raw = (output_dir / name).read_bytes()
        except OSError:
            return False
        if sha256_bytes(raw) != digest:
            return False
    return True


def _assert_finite_evidence(record: ExperimentRecord, evidence: APIEvidence) -> None:
    """Refuse to persist anything carrying ``NaN`` or an infinity.

    ``json`` emits those as the non-standard ``NaN``/``Infinity`` tokens,
    and they survive every ``>= 0`` range check downstream. Nothing here
    can produce one from a monotonic clock, so hitting this means the
    clock or a provider value is broken. The check runs before the first
    write so a broken run leaves no artifacts at all rather than a
    half-written set.
    """
    for label, payload in (
        ("record", record.to_dict()),
        ("api_evidence", evidence.to_dict()),
    ):
        offender = _find_non_finite(payload, label)
        if offender is not None:
            path, value = offender
            raise OpenAIStreamCollectorError(
                f"refusing to persist a non-finite measurement: {path}={value!r}"
            )


def _find_non_finite(value: Any, path: str) -> tuple[str, float] | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return path, value
    if isinstance(value, Mapping):
        for key, item in value.items():
            found = _find_non_finite(item, f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _find_non_finite(item, f"{path}[{index}]")
            if found is not None:
                return found
    return None


def _build_record(
    config: APICollectionConfig,
    *,
    plan: RequestPlan,
    accumulator: _StreamAccumulator,
    started_at: str,
    started: float,
    completed: float,
    failure: APIFailure | None,
) -> ExperimentRecord:
    usage = accumulator.usage
    tokens = TokenCounts(
        input_tokens=usage.prompt_tokens,
        context_tokens=usage.prompt_tokens,
        generated_tokens=usage.completion_tokens,
        provenance=(
            MetricProvenance.PROVIDER_REPORTED
            if usage.reported
            and (usage.prompt_tokens is not None or usage.completion_tokens is not None)
            else None
        ),
    )
    record = ExperimentRecord(
        run_id=config.run_id,
        started_at=started_at,
        ended_at=utc_now_iso(),
        # This describes the client host that measured the timings, not the
        # provider's hardware, which is not observable. No accelerator is
        # claimed and no memory is recorded for the same reason.
        platform=record_platform(accelerator=None, extra_packages=()),
        model=ModelInfo(
            model_id=config.model_id,
            model_revision=config.model_revision,
        ),
        runtime=RuntimeInfo(
            name=RUNTIME_NAME,
            version=None,
            # No local compute backend executed this run. ``backend`` is
            # documented as the local one ('Metal', 'CUDA', 'CPU'), and
            # ``provider`` exists precisely so a hosted run is recorded
            # without overloading it. Writing a transport there would put
            # "remote-http" everywhere a reader expects hardware.
            backend=None,
            git_revision=None,
            provider=config.provider,
        ),
        command=CommandInfo(
            # The plan already holds the sanitized argv. Rebuilding it from
            # the raw config here would put endpoint query values straight
            # into record.json, which is exactly what the plan avoids.
            argv=tuple(plan.command),
            config_hash=plan.config_hash,
            workload_hash=plan.workload_hash,
        ),
        repetition=RepetitionInfo(
            warmup_repetitions=0,
            measured_repetitions=1,
            repetition_index=0,
            seed=config.seed,
        ),
        tokens=tokens,
        timing=TimingMetrics(
            model_load=None,
            tokenize=None,
            # ``prefill`` and ``decode`` name model phases: prompt
            # processing and generation. Neither is observable from
            # outside a hosted API. The client-side interval before the
            # first content token also contains DNS, connection setup,
            # TLS, request transfer and any server-side queueing, and the
            # interval after it runs to the last SSE event, which can be
            # a usage chunk or [DONE] sent long after generation
            # finished. Publishing those two numbers under these names
            # would state a decomposition the evidence does not support,
            # so they are left unset and the client-observed offsets are
            # kept, under their own names and with transport called out,
            # in the API evidence timeline.
            prefill=None,
            decode=None,
            # End to end wall clock is genuinely measurable from here.
            total=milliseconds(started, completed),
        ),
        memory=MemoryMetrics(),
        outcome=OutcomeInfo(success=failure is None),
        error=(
            None
            if failure is None
            else ErrorInfo(category=failure.category, message=failure.message)
        ),
    )
    record.validate()
    return record
