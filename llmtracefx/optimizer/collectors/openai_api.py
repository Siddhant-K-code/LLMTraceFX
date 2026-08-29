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

import http.client
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
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

# Terminal ``finish_reason`` values. A stream that stops without one of
# these and without ``[DONE]`` was cut short, however much content it
# already delivered.
_TERMINAL_FINISH_REASONS = frozenset(
    {"stop", "length", "content_filter", "tool_calls", "function_call"}
)

_EVENT_KIND_CONTENT = "content"
_EVENT_KIND_REASONING = "reasoning"
_EVENT_KIND_METADATA = "metadata"

_MAX_ERROR_BODY_BYTES = 64 * 1024
_MAX_PERSISTED_MESSAGE_CHARS = 600
_MAX_PERSISTED_HEADER_CHARS = 128
_REDACTED = "[REDACTED]"
_MIN_ENCODED_CREDENTIAL_CHARS = 6

_PROVIDER_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_ENV_VAR_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRETISH_QUERY_KEY = re.compile(
    r"(?i)(key|token|secret|password|passwd|credential|signature|sig|auth)"
)
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


class _UrllibResponse:
    """Adapter over ``http.client.HTTPResponse`` (or an ``HTTPError``)."""

    _CHUNK_SIZE = 8192

    def __init__(self, raw: Any, status_code: int, headers: Mapping[str, str]) -> None:
        self._raw = raw
        self._status_code = status_code
        self._headers = headers
        self._bytes_read = 0

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
            return _UrllibResponse(exc, int(exc.code), _normalize_headers(exc.headers))
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
        return _UrllibResponse(raw, int(raw.status), _normalize_headers(raw.headers))


def _normalize_headers(headers: Any) -> dict[str, str]:
    items = getattr(headers, "items", None)
    if items is None:
        return {}
    return {str(name).lower(): str(value) for name, value in items()}


# --- Configuration -----------------------------------------------------------


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
    model_revision: str | None = None
    """Provider-side model build, when the provider exposes one. Hosted
    APIs generally do not, in which case this stays ``None`` rather than
    being guessed from the model ID."""

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
        if self.system_prompt is not None and not self.system_prompt:
            raise OpenAIStreamCollectorError(
                "system_prompt must be non-empty when provided"
            )
        if not _ENV_VAR_PATTERN.match(self.credential_env_var or ""):
            raise OpenAIStreamCollectorError(
                "credential_env_var must be a valid environment variable name, "
                f"got {self.credential_env_var!r}"
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


def _try_parse_qsl(query: str) -> list[tuple[str, str]] | None:
    try:
        return parse_qsl(query, keep_blank_values=True)
    except ValueError:
        return None


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
    # forces the netloc to be validated inside this guard.
    try:
        _host = parts.hostname
    except ValueError:
        raise OpenAIStreamCollectorError(
            "endpoint has a host that could not be parsed"
        ) from None
    try:
        _port = parts.port
    except ValueError:
        raise OpenAIStreamCollectorError(
            "endpoint has a port that is not an integer in the range 0 to 65535"
        ) from None
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
            keys = sorted({key for key, _ in pairs})
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
        if _SECRETISH_QUERY_KEY.search(key):
            raise OpenAIStreamCollectorError(
                f"endpoint query parameter {key!r} looks like a credential; "
                "credentials must come from the environment variable instead"
            )


# --- Redaction ---------------------------------------------------------------


class _Redactor:
    """Scrub a known credential (and bearer-token shapes) out of any string.

    Every provider-controlled string that reaches an artifact goes through
    this, not just error messages. A server that echoes the key back in a
    response id, a model name, a finish reason or the generated text
    itself would otherwise write it straight to disk.
    """

    _BEARER = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]+")

    def __init__(self, credential: str | None) -> None:
        self._credential = credential or None
        # Match more than the literal value. A credential may legally contain
        # spaces, and different sinks treat whitespace differently: ``text``
        # must preserve the answer's own spacing, while ``__call__``
        # collapses it. Matching each internal whitespace run as ``\s+``, case
        # insensitively, gives every sink the same coverage without any sink
        # having to alter what it persists. Case insensitivity also covers
        # header names, which are lowercased before they are persisted.
        self._pattern: re.Pattern[str] | None = None
        if self._credential:
            parts = [re.escape(part) for part in self._credential.split()]
            joined = r"\s+".join(parts) if len(parts) > 1 else parts[0]
            self._pattern = re.compile(joined, re.IGNORECASE)
        # Prefix forms used only by ``boundary``, which compares against a
        # truncated tail and so cannot use the pattern.
        variants: list[str] = []
        if self._credential:
            for variant in (
                " ".join(self._credential.split()),
                self._credential.lower(),
                " ".join(self._credential.lower().split()),
            ):
                if variant and variant != self._credential and variant not in variants:
                    variants.append(variant)
        self._variants = tuple(sorted(variants, key=len, reverse=True))

    def _scrub(self, text: str) -> str:
        cleaned = text
        if self._pattern is not None:
            cleaned = self._pattern.sub(_REDACTED, cleaned)
        return self._BEARER.sub(rf"\1 {_REDACTED}", cleaned)

    def __call__(self, text: str, *, limit: int = _MAX_PERSISTED_MESSAGE_CHARS) -> str:
        # Normalize before scrubbing as well as after. Whitespace collapse
        # can reassemble a credential that the pre-collapse form hid.
        cleaned = " ".join(self._scrub(" ".join(text.split())).split())
        if len(cleaned) > limit:
            cleaned = cleaned[: limit - 3] + "..."
        return cleaned

    def boundary(self, text: str) -> str:
        """Scrub a string that a byte cap may have cut through.

        Truncation always removes the tail, so a credential split by the cap
        survives as a trailing proper prefix that the exact-substring scrub
        cannot see. Whitespace collapse then pulls that tail back into the
        persisted window, which is how a 64 KiB body can leak all but the
        last character of a key.
        """
        cleaned = text.rstrip("\ufffd")
        forms = [self._credential, *self._variants] if self._credential else []
        for credential in forms:
            if len(credential) < _MIN_ENCODED_CREDENTIAL_CHARS:
                continue
            longest = min(len(cleaned), len(credential) - 1)
            for length in range(longest, _MIN_ENCODED_CREDENTIAL_CHARS - 1, -1):
                if cleaned.endswith(credential[:length]):
                    return cleaned[:-length] + _REDACTED
        return cleaned

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
    return config_hash(
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
            "credential_env_var": config.credential_env_var,
            "request_parameters": _core_request_parameters(config),
            "provider_extensions": config.extensions.to_request_fields(),
            "request_timeout_seconds": config.request_timeout_seconds,
            "system_prompt_sha256": (
                None
                if config.system_prompt is None
                else sha256_text(config.system_prompt)
            ),
        }
    )


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


def _sanitized_command(config: APICollectionConfig) -> tuple[str, ...]:
    safe_endpoint = _endpoint_for_command(config.endpoint)
    if safe_endpoint == config.endpoint:
        return tuple(config.command_argv)
    return tuple(
        safe_endpoint if argument == config.endpoint else argument
        for argument in config.command_argv
    )


def build_request_plan(config: APICollectionConfig) -> RequestPlan:
    """Describe the request without sending it and without any secret."""
    parts = parse_endpoint(config.endpoint)
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
        credential_env_var=config.credential_env_var,
        credential_header_name="Authorization",
        header_names=("Accept", "Authorization", "Content-Type"),
        messages=messages,
        request_parameters=_core_request_parameters(config),
        provider_extensions=config.extensions.to_request_fields(),
        request_timeout_seconds=float(config.request_timeout_seconds),
        command=_sanitized_command(config),
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
    """

    content_delta_count: int = 0
    content_characters: int = 0
    reasoning_delta_count: int = 0
    reasoning_characters: int = 0
    metadata_event_count: int = 0
    comment_count: int = 0
    inter_content_delta: LatencyDistribution | None = None
    content_window_ms: float | None = None
    content_delta_rate_per_second: float | None = None
    provider_completion_tokens_per_second: float | None = None

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
            "provider_completion_tokens_per_second": (
                self.provider_completion_tokens_per_second
            ),
            "provider_completion_tokens_per_second_note": (
                "provider-reported completion tokens divided by "
                "content_window_ms; mixed provenance. The window starts at "
                "the first content delta, so the first delta's own "
                "generation time is excluded, and when content_delta_count "
                "is far below completion_tokens the window endpoints are "
                "delta boundaries rather than token boundaries. Treat this "
                "as a coarse estimate, not a measured per-token rate."
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
    ) -> None:
        self._clock = clock
        self._started = started
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
        self.content_arrival_times: list[float] = []
        self.response_id: str | None = None
        self.provider_request_id: str | None = None
        self.response_model: str | None = None
        self.finish_reason: str | None = None
        self.usage: ProviderUsage = ProviderUsage()
        self.terminated_with_done = False
        self.had_unterminated_event = False

    def offset_ms(self, moment: float | None) -> float | None:
        if moment is None:
            return None
        return max(0.0, moment - self._started) * 1000

    def consume(
        self, response: StreamingResponse, *, redact: _Redactor
    ) -> APIFailure | None:
        """Drain the body. Returns a failure, or ``None`` when it streamed."""
        decoder = SSEDecoder()
        try:
            for chunk in response.iter_bytes():
                now = self._clock()
                if self.first_body_chunk_at is None:
                    self.first_body_chunk_at = now
                for event in decoder.feed(chunk):
                    failure = self._handle_event(event, redact=redact)
                    if failure is not None or self.terminated_with_done:
                        return failure
            for event in decoder.close():
                failure = self._handle_event(event, redact=redact)
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
            self.had_unterminated_event = decoder.dispatched_unterminated_event
        return None

    def _handle_event(self, event: SSEEvent, *, redact: _Redactor) -> APIFailure | None:
        now = self._clock()
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
        except json.JSONDecodeError as exc:
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
            self.content_arrival_times.append(now)
            self.last_content_at = now
            self.content_parts.append(content)
            self.content_delta_count += 1
            self._record_event(_EVENT_KIND_CONTENT, len(content), now)
            if reasoning:
                self.reasoning_delta_count += 1
                self.reasoning_characters += len(reasoning)
            return None

        if reasoning:
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
        self.events.append(
            StreamEventTiming(
                index=len(self.events),
                offset_ms=0.0 if offset is None else offset,
                kind=kind,
                characters=characters,
            )
        )

    def terminated_cleanly(self) -> bool:
        """True when the provider signalled a real end of stream.

        Either the ``[DONE]`` sentinel or a terminal ``finish_reason`` is
        accepted, because both are documented endings and not every
        OpenAI-compatible provider sends both.
        """
        if self.terminated_with_done:
            return True
        reason = self.finish_reason
        return reason is not None and reason in _TERMINAL_FINISH_REASONS

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

    def statistics(self) -> StreamStatistics:
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
        window_seconds: float | None = None
        if (
            len(self.content_arrival_times) > 1
            and self.content_arrival_times[-1] > self.content_arrival_times[0]
        ):
            window_seconds = (
                self.content_arrival_times[-1] - self.content_arrival_times[0]
            )

        delta_rate = None
        token_rate = None
        if window_seconds is not None:
            # ``n`` arrivals bound ``n - 1`` intervals, so the delta rate is
            # measured over the gaps it actually observed.
            delta_rate = (len(self.content_arrival_times) - 1) / window_seconds
            completion_tokens = self.usage.completion_tokens
            if completion_tokens is not None:
                token_rate = completion_tokens / window_seconds

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
            content_delta_rate_per_second=delta_rate,
            provider_completion_tokens_per_second=token_rate,
        )

    def timeline(self, completed_at: float) -> StreamTimeline:
        return StreamTimeline(
            response_headers_offset_ms=self.offset_ms(self.response_headers_at),
            first_body_chunk_offset_ms=self.offset_ms(self.first_body_chunk_at),
            first_content_token_offset_ms=self.offset_ms(self.first_content_token_at),
            last_event_offset_ms=self.offset_ms(self.last_event_at),
            completed_offset_ms=self.offset_ms(completed_at),
            events=tuple(self.events),
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
        raise OpenAIStreamCollectorError(
            f"environment variable {config.credential_env_var} is not set; "
            "export the API key there (it is never accepted as a command "
            "argument and never written to any artifact)"
        )
    # Strip first. A key read from a file or a ``.env`` routinely carries a
    # trailing newline, and a newline in a header value makes
    # ``http.client.putheader`` raise a ``ValueError`` whose message embeds
    # the whole value, which would print the secret in a traceback.
    credential = value.strip()
    if not credential:
        raise OpenAIStreamCollectorError(
            f"environment variable {config.credential_env_var} is empty"
        )
    _assert_header_safe_credential(credential, config.credential_env_var)
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
    """True when ``credential`` is present literally or percent encoded.

    Short credentials are compared literally only. A one or two character
    value would otherwise match a decoded byte by coincidence, and a
    refusal that fires on noise trains people to work around it.
    """
    if credential in value:
        return True
    if len(credential) < _MIN_ENCODED_CREDENTIAL_CHARS:
        return False
    return any(credential in decoded for decoded in _percent_decodings(value))


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
                f"the value of {config.credential_env_var} appears in {label}; "
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


def _http_status_failure(
    response: StreamingResponse, *, redact: _Redactor
) -> APIFailure:
    body = _read_error_body(response).decode("utf-8", errors="replace")
    # The byte cap in ``_read_error_body`` can cut through an echoed
    # credential, so repair that boundary before the body is parsed or used.
    body = redact.boundary(body)
    message = f"HTTP {response.status_code}"
    code: str | None = None
    try:
        payload = json.loads(body) if body.strip() else None
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error")
        source = error if isinstance(error, Mapping) else payload
        message = f"HTTP {response.status_code}: {_error_message(source)}"
        code = _error_code(source)
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
    plan = build_request_plan(config)

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
    accumulator = _StreamAccumulator(clock=clock, started=started, redactor=redact)
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
                failure = _http_status_failure(response, redact=redact)
            else:
                failure = accumulator.consume(response, redact=redact)
        finally:
            try:
                response.close()
            except (OSError, http.client.HTTPException):
                # Releasing the socket must never discard collected evidence.
                pass

    completed = clock()
    response_text = accumulator.content_text()

    if failure is None and not response_text:
        failure = APIFailure(
            category=FAILURE_MISSING_CONTENT,
            message=(
                "stream completed without any content delta; finish_reason="
                f"{accumulator.finish_reason!r}"
            ),
            provider_error_code=None,
        )

    if failure is None and not accumulator.terminated_cleanly():
        # Content alone does not mean the answer is whole. Without ``[DONE]``
        # or a terminal finish_reason the body was cut short, and publishing
        # a truncated answer as a success would corrupt every downstream
        # comparison that reads it.
        failure = APIFailure(
            category=FAILURE_STREAM_TRUNCATED,
            message=(
                "stream ended without a terminal condition: no [DONE] "
                "sentinel and finish_reason="
                f"{accumulator.finish_reason!r}; the response is truncated"
            ),
            provider_error_code=None,
        )

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
        usage=accumulator.usage,
        timeline=accumulator.timeline(completed),
        statistics=accumulator.statistics(),
        rate_limit_headers=rate_limit_headers,
        stream_terminated_with_done=accumulator.terminated_with_done,
        stream_had_unterminated_event=accumulator.had_unterminated_event,
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
            {"name": name, "sha256": sha256_text(text)} for name, text in payloads
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
    except (OSError, json.JSONDecodeError):
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
            text = (output_dir / name).read_text(encoding="utf-8")
        except OSError:
            return False
        if sha256_text(text) != digest:
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
            backend="remote-http",
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
            # Client-observed time to first content token, transport included.
            prefill=milliseconds(started, accumulator.first_content_token_at),
            decode=milliseconds(
                accumulator.first_content_token_at, accumulator.last_event_at
            ),
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
