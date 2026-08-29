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
from urllib.parse import parse_qsl, urlsplit

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

_EVENT_KIND_CONTENT = "content"
_EVENT_KIND_REASONING = "reasoning"
_EVENT_KIND_METADATA = "metadata"

_MAX_ERROR_BODY_BYTES = 64 * 1024
_MAX_PERSISTED_MESSAGE_CHARS = 600
_MAX_PERSISTED_HEADER_CHARS = 128
_REDACTED = "[REDACTED]"

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
        return (
            f"HTTPRequest(url={self.url!r}, method={self.method!r}, "
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

    @property
    def status_code(self) -> int:
        return self._status_code

    @property
    def headers(self) -> Mapping[str, str]:
        return self._headers

    def iter_bytes(self) -> Iterator[bytes]:
        while True:
            try:
                chunk = self._raw.read(self._CHUNK_SIZE)
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
                return
            yield chunk

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


def _validate_endpoint(endpoint: str) -> None:
    if not endpoint:
        raise OpenAIStreamCollectorError("endpoint must be non-empty")
    parts = urlsplit(endpoint)
    if parts.scheme not in ("http", "https"):
        raise OpenAIStreamCollectorError(
            f"endpoint must use http or https, got {endpoint!r}"
        )
    if not parts.hostname:
        raise OpenAIStreamCollectorError(f"endpoint must include a host: {endpoint!r}")
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
            f"header is not sent in clear text, got {endpoint!r}"
        )
    for key, _value in parse_qsl(parts.query, keep_blank_values=True):
        if _SECRETISH_QUERY_KEY.search(key):
            raise OpenAIStreamCollectorError(
                f"endpoint query parameter {key!r} looks like a credential; "
                "credentials must come from the environment variable instead"
            )


# --- Redaction ---------------------------------------------------------------


class _Redactor:
    """Scrub a known credential (and bearer-token shapes) out of any string."""

    _BEARER = re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._~+/=-]+")

    def __init__(self, credential: str | None) -> None:
        self._credential = credential or None

    def __call__(self, text: str, *, limit: int = _MAX_PERSISTED_MESSAGE_CHARS) -> str:
        cleaned = text
        if self._credential:
            cleaned = cleaned.replace(self._credential, _REDACTED)
        cleaned = self._BEARER.sub(rf"\1 {_REDACTED}", cleaned)
        cleaned = " ".join(cleaned.split())
        if len(cleaned) > limit:
            cleaned = cleaned[: limit - 3] + "..."
        return cleaned


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


def _config_identity_hash(config: APICollectionConfig) -> str:
    parts = urlsplit(config.endpoint)
    return config_hash(
        {
            "provider": config.provider,
            "endpoint_origin": f"{parts.scheme}://{parts.netloc}",
            "endpoint_path": parts.path,
            "endpoint_query_keys": sorted(
                key for key, _ in parse_qsl(parts.query, keep_blank_values=True)
            ),
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


def build_request_plan(config: APICollectionConfig) -> RequestPlan:
    """Describe the request without sending it and without any secret."""
    parts = urlsplit(config.endpoint)
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
        command=tuple(config.command_argv),
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
    """

    content_delta_count: int = 0
    content_characters: int = 0
    reasoning_delta_count: int = 0
    reasoning_characters: int = 0
    metadata_event_count: int = 0
    comment_count: int = 0
    inter_content_delta: LatencyDistribution | None = None
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
            "content_delta_rate_per_second": self.content_delta_rate_per_second,
            "content_delta_rate_provenance": MetricProvenance.DERIVED.value,
            "provider_completion_tokens_per_second": (
                self.provider_completion_tokens_per_second
            ),
            "provider_completion_tokens_per_second_note": (
                "provider-reported completion tokens divided by a "
                "client-measured decode window; mixed provenance"
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

    def __init__(self, *, clock: Callable[[], float], started: float) -> None:
        self._clock = clock
        self._started = started
        self.response_headers_at: float | None = None
        self.first_body_chunk_at: float | None = None
        self.first_content_token_at: float | None = None
        self.last_content_at: float | None = None
        self.last_event_at: float | None = None
        self.content_parts: list[str] = []
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
        if data == "[DONE]":
            self.terminated_with_done = True
            return None
        if not data:
            self._record_event(_EVENT_KIND_METADATA, 0, now)
            return None

        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            return APIFailure(
                category=FAILURE_STREAM_DECODE,
                message=redact(f"stream chunk is not valid JSON: {exc}"),
            )
        if not isinstance(payload, dict):
            return APIFailure(
                category=FAILURE_STREAM_DECODE,
                message="stream chunk is not a JSON object",
            )

        provider_failure = _provider_error_from_payload(payload, redact=redact)
        if provider_failure is not None:
            self._absorb_identity(payload)
            return provider_failure

        self._absorb_identity(payload)

        usage_payload = payload.get("usage")
        if usage_payload is not None:
            if not isinstance(usage_payload, dict):
                return APIFailure(
                    category=FAILURE_STREAM_DECODE,
                    message="stream chunk 'usage' is not a JSON object",
                )
            self.usage = ProviderUsage.from_payload(usage_payload)

        choice = _first_choice(payload)
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
            self.finish_reason = finish_reason

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

    def _absorb_identity(self, payload: Mapping[str, Any]) -> None:
        for key, attribute in (
            ("id", "response_id"),
            ("request_id", "provider_request_id"),
            ("model", "response_model"),
        ):
            value = payload.get(key)
            if isinstance(value, str) and value and getattr(self, attribute) is None:
                setattr(self, attribute, value)

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

    def statistics(self) -> StreamStatistics:
        gaps = [
            (later - earlier) * 1000
            for earlier, later in zip(
                self.content_arrival_times, self.content_arrival_times[1:], strict=False
            )
        ]
        distribution = _latency_distribution(gaps)
        delta_rate = None
        if (
            len(self.content_arrival_times) > 1
            and self.content_arrival_times[-1] > self.content_arrival_times[0]
        ):
            window = self.content_arrival_times[-1] - self.content_arrival_times[0]
            delta_rate = (len(self.content_arrival_times) - 1) / window

        token_rate = None
        completion_tokens = self.usage.completion_tokens
        if (
            completion_tokens is not None
            and self.first_content_token_at is not None
            and self.last_event_at is not None
            and self.last_event_at > self.first_content_token_at
        ):
            token_rate = completion_tokens / (
                self.last_event_at - self.first_content_token_at
            )

        return StreamStatistics(
            content_delta_count=self.content_delta_count,
            content_characters=sum(len(part) for part in self.content_parts),
            reasoning_delta_count=self.reasoning_delta_count,
            reasoning_characters=self.reasoning_characters,
            metadata_event_count=self.metadata_event_count,
            comment_count=self.comment_count,
            inter_content_delta=distribution,
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


def _first_choice(payload: Mapping[str, Any]) -> dict[str, Any] | None:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    return first if isinstance(first, dict) else None


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
            provider_error_code=_error_code(error),
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
            provider_error_code=_error_code(payload),
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
        if credential in value:
            raise OpenAIStreamCollectorError(
                f"the value of {config.credential_env_var} appears in {label}; "
                "refusing to run because that value would be persisted"
            )


def _rate_limit_headers(
    headers: Mapping[str, str], *, redact: _Redactor
) -> dict[str, str]:
    collected: dict[str, str] = {}
    for name, value in headers.items():
        lowered = name.lower()
        if lowered in _RATE_LIMIT_HEADER_NAMES or lowered.startswith(
            _RATE_LIMIT_HEADER_PREFIXES
        ):
            collected[lowered] = redact(str(value), limit=_MAX_PERSISTED_HEADER_CHARS)
    return collected


def _request_id_from_headers(headers: Mapping[str, str]) -> str | None:
    lowered = {name.lower(): value for name, value in headers.items()}
    for name in _REQUEST_ID_HEADER_NAMES:
        value = lowered.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()[:_MAX_PERSISTED_HEADER_CHARS]
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
        provider_error_code=code,
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
    accumulator = _StreamAccumulator(clock=clock, started=started)
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
    response_text = "".join(accumulator.content_parts)

    if failure is None and not response_text:
        failure = APIFailure(
            category=FAILURE_MISSING_CONTENT,
            message=(
                "stream completed without any content delta; finish_reason="
                f"{accumulator.finish_reason!r}"
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

    config.output_dir.mkdir(parents=True, exist_ok=True)
    record.write_json(config.output_dir / "record.json")
    atomic_write_text(config.output_dir / "response.txt", response_text)
    atomic_write_text(
        config.output_dir / "api_evidence.json", evidence.to_json() + "\n"
    )
    manifest = collect_environment_manifest()
    atomic_write_text(config.output_dir / "environment.json", manifest.to_json() + "\n")
    return APICollectionResult(
        record=record, evidence=evidence, response_text=response_text
    )


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
            argv=tuple(config.command_argv),
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
