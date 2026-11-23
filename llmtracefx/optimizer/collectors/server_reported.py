"""Per-request numbers a serving engine reports about itself in the response.

A Prometheus window covers every request the server handled while it was
open, so under load it cannot say which request was slow. Several engines
also return their own timing and speculative-decoding counters inside the
response to one request. Those survive under concurrency because they are
already attributed. This module reads them, and only them:

* vLLM's top-level ``metrics`` object, on the final usage chunk of a stream
  (``vllm/entrypoints/generate/base/protocol.py`` L105-138 and
  ``docs/features/per_request_metrics.md`` at vllm-project/vllm@1ae1b64).
  The server opts in with ``--enable-per-request-metrics`` and, for the
  ``speculative_decoding`` block, ``--per-request-spec-decode-metrics``.
* SGLang's response-level ``sglext`` object, sent on a ``choices: []`` chunk
  when the request body sets ``return_spec_tokens_details`` or
  ``return_cached_tokens_details`` (``entrypoints/openai/serving_chat.py``
  L2690-2761 and ``protocol.py`` L475-499 at sgl-project/sglang@b6d4389).
* TensorRT-LLM's ``trtllm.perf_metrics`` SSE event, sent after ``[DONE]``
  when the request carries ``X-TRTLLM-return-metrics: 1`` and the server runs
  with ``return_perf_metrics``. Its data holds the ``Server-Timing`` and
  ``X-TRTLLM-*`` header values a non-streaming response would carry
  (``tensorrt_llm/serve/perf_metrics.py`` L15-47, L322-399 and L805-910 at
  NVIDIA/TensorRT-LLM@e6ecd28).

Everything here is ``provider_reported``. The numbers are the server's
account of itself, measured on the server's clocks with the server's own
phase boundaries, and they are kept apart from the client's measurements
rather than replacing any of them. A server that sends none of these leaves
no block at all; nothing is filled with zeros.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..schema import MetricProvenance

SOURCE_VLLM = "vllm_metrics"
SOURCE_SGLANG = "sglang_sglext"
SOURCE_TRTLLM = "trtllm_perf_metrics"
SOURCE_LLAMA_CPP = "llama_cpp_timings"
SOURCE_OLLAMA = "ollama_metrics"

TRTLLM_RETURN_METRICS_HEADER = "X-TRTLLM-return-metrics"
TRTLLM_PERF_METRICS_EVENT = "trtllm.perf_metrics"
SGLANG_DETAILS_REQUEST_FIELDS: Mapping[str, bool] = {
    "return_cached_tokens_details": True,
    "return_spec_tokens_details": True,
}
"""Body fields that make SGLang send ``sglext`` (``ChatCompletionRequest``
in ``python/sglang/srt/entrypoints/openai/protocol.py`` L1038-1039 at
sgl-project/sglang@b6d4389). Sent through ``extra_body``, so they are part
of the request identity like any other body field."""

PROVENANCE = MetricProvenance.PROVIDER_REPORTED.value

NORMALIZED_FIELDS: tuple[str, ...] = (
    "queue_ms",
    "ttft_ms",
    "scheduled_ttft_ms",
    "decode_ms",
    "e2e_ms",
    "mean_itl_ms",
    "tokens_per_second",
    "draft_tokens",
    "accepted_draft_tokens",
    "draft_acceptance_rate",
    "mean_acceptance_length",
    "verify_steps",
)
"""Engine-neutral names, each with one definition.

``queue_ms`` runs from arrival to first scheduling. ``ttft_ms`` runs from
arrival to the first token, queue included. ``scheduled_ttft_ms`` runs from
first scheduling to the first token, queue excluded; vLLM calls this
``time_to_first_token_ms``. ``decode_ms`` runs from the first token to the
last. ``e2e_ms`` runs from arrival to the last token. A value an engine does
not report under that definition stays ``None``; none is derived from
another."""

_MAX_EXACT_COUNT = 2**53
_MAX_LIST_ITEMS = 65_536
_MAX_TIMING_ENTRIES = 4_096
_ENTRY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

_TRTLLM_HEADERS: tuple[tuple[str, str, str], ...] = (
    ("server-timing", "server_timing", "dur"),
    ("x-trtllm-start-end-time", "start_end_time", "ts"),
    ("x-trtllm-step-metrics", "step_metrics", "dur"),
    ("x-trtllm-ctx-chunk-metrics", "ctx_chunk_metrics", "dur"),
)
_TRTLLM_NORMALIZED: tuple[tuple[str, str], ...] = (
    # Phase ``server`` is a non-disaggregated worker; perf_metrics.py
    # L351-361 at e6ecd28 computes queue, ttft and e2e from ``arrival_time``.
    ("server_queue", "queue_ms"),
    ("server_ttft", "ttft_ms"),
    ("server_e2e", "e2e_ms"),
)


@dataclass(frozen=True)
class ServerReportedMetrics:
    """What the server said about one request, by source, never merged with client timing."""

    sources: tuple[str, ...]
    values: Mapping[str, float | int] = field(default_factory=dict)
    """Normalized values that were reported, keyed by ``NORMALIZED_FIELDS``."""
    raw: Mapping[str, Any] = field(default_factory=dict)
    """The validated fields per source, under the source's own names."""
    malformed_fields: tuple[str, ...] = ()
    """Fields sent in an unusable shape; dropped and named, never coerced."""

    def value(self, name: str) -> Any:
        return self.values.get(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": PROVENANCE,
            "sources": list(self.sources),
            "normalized": {name: self.values.get(name) for name in NORMALIZED_FIELDS},
            "raw": {source: self.raw[source] for source in sorted(self.raw)},
            "malformed_fields": list(self.malformed_fields),
        }


class ServerReportedAccumulator:
    """Collects server-reported blocks as a stream delivers them."""

    def __init__(self, *, unsafe_text: Callable[[str], bool] | None = None) -> None:
        # Entry names and labels are provider-controlled text. A name the
        # caller's redactor flags (an echoed credential) is dropped as
        # malformed rather than persisted.
        self._unsafe_text = unsafe_text or (lambda _text: False)
        self._sources: list[str] = []
        self._values: dict[str, float | int] = {}
        self._raw: dict[str, Any] = {}
        self._malformed: list[str] = []

    def note_malformed(self, label: str) -> None:
        if label not in self._malformed:
            self._malformed.append(label)

    def result(self) -> ServerReportedMetrics | None:
        if not self._raw and not self._malformed:
            return None
        return ServerReportedMetrics(
            sources=tuple(self._sources),
            values=dict(self._values),
            raw=dict(self._raw),
            malformed_fields=tuple(self._malformed),
        )

    def _record(self, source: str, raw: dict[str, Any]) -> None:
        if not raw:
            return
        self._raw[source] = raw
        if source not in self._sources:
            self._sources.append(source)

    def absorb_vllm_metrics(self, payload: Any) -> None:
        """vLLM ``metrics``: ``PerRequestMetrics`` with an optional spec block."""
        if payload is None:
            return
        if not isinstance(payload, Mapping):
            self.note_malformed("metrics")
            return
        raw: dict[str, Any] = {}
        values: dict[str, float | int] = {}
        for key, normalized in (
            ("queue_time_ms", "queue_ms"),
            ("time_to_first_token_ms", "scheduled_ttft_ms"),
            ("generation_time_ms", "decode_ms"),
            ("mean_itl_ms", "mean_itl_ms"),
            ("tokens_per_second", "tokens_per_second"),
        ):
            value = _number(payload.get(key), f"metrics.{key}", self)
            if value is not None:
                raw[key] = values[normalized] = value
        spec = payload.get("speculative_decoding")
        if spec is not None:
            spec_raw = self._vllm_spec(spec, values)
            if spec_raw:
                raw["speculative_decoding"] = spec_raw
        if raw:
            self._values.update(values)
            self._record(SOURCE_VLLM, raw)

    def _vllm_spec(self, spec: Any, values: dict[str, float | int]) -> dict[str, Any]:
        prefix = "metrics.speculative_decoding"
        if not isinstance(spec, Mapping):
            self.note_malformed(prefix)
            return {}
        raw: dict[str, Any] = {}
        for key, normalized, maximum in (
            ("mean_acceptance_length", "mean_acceptance_length", None),
            ("draft_acceptance_rate", "draft_acceptance_rate", 1.0),
        ):
            value = _number(spec.get(key), f"{prefix}.{key}", self, maximum=maximum)
            if value is not None:
                raw[key] = values[normalized] = value
        count_fields: tuple[tuple[str, str | None], ...] = (
            ("num_spec_steps", "verify_steps"),
            ("num_accepted_draft_tokens", "accepted_draft_tokens"),
            ("num_draft_tokens", "draft_tokens"),
            ("num_spec_tokens", None),
            ("num_verified_draft_tokens", None),
        )
        for key, count_name in count_fields:
            count = _count(spec.get(key), f"{prefix}.{key}", self)
            if count is not None:
                raw[key] = count
                if count_name is not None:
                    values[count_name] = count
        for key in (
            "acceptance_histogram",
            "per_step_accepted",
            "per_step_drafted",
            "per_step_verified",
        ):
            counts = _counts(spec.get(key), f"{prefix}.{key}", self)
            if counts is not None:
                raw[key] = counts
        return raw

    def absorb_sglext(self, payload: Any) -> None:
        """SGLang ``sglext``: spec and cached-token details, never token ids."""
        if payload is None:
            return
        if not isinstance(payload, Mapping):
            self.note_malformed("sglext")
            return
        raw: dict[str, Any] = {}
        values: dict[str, float | int] = {}
        spec = payload.get("spec_tokens_details")
        if isinstance(spec, list):
            # A list is one entry per choice when n > 1. This collector never
            # sets n, so anything but a single entry cannot be attributed.
            if len(spec) == 1:
                spec = spec[0]
            else:
                self.note_malformed("sglext.spec_tokens_details")
                spec = None
        if spec is not None:
            spec_raw = self._sglang_spec(spec, values)
            if spec_raw:
                raw["spec_tokens_details"] = spec_raw
        cached = payload.get("cached_tokens_details")
        if cached is not None:
            cached_raw = self._sglang_cached(cached)
            if cached_raw:
                raw["cached_tokens_details"] = cached_raw
        if raw:
            self._values.update(values)
            self._record(SOURCE_SGLANG, raw)

    def _sglang_spec(self, spec: Any, values: dict[str, float | int]) -> dict[str, Any]:
        prefix = "sglext.spec_tokens_details"
        if not isinstance(spec, Mapping):
            self.note_malformed(prefix)
            return {}
        raw: dict[str, Any] = {}
        # SGLang serializes a pydantic model whose numeric fields default to
        # 0, and it sets the real values only for a request that ran a verify
        # step (spec_decoding_metrics.py L40-65, utils.py L314-345 at
        # b6d4389). Without one the zeros are defaults, not measurements.
        verify_ct = _count(spec.get("spec_verify_ct"), f"{prefix}.spec_verify_ct", self)
        if verify_ct:
            raw["spec_verify_ct"] = values["verify_steps"] = verify_ct
            for key, normalized, maximum in (
                ("spec_accept_rate", "draft_acceptance_rate", 1.0),
                ("spec_accept_length", "mean_acceptance_length", None),
            ):
                value = _number(spec.get(key), f"{prefix}.{key}", self, maximum=maximum)
                if value is not None:
                    raw[key] = values[normalized] = value
            for key, normalized in (
                ("spec_num_correct_drafts", "accepted_draft_tokens"),
                ("spec_num_proposed_drafts", "draft_tokens"),
            ):
                count = _count(spec.get(key), f"{prefix}.{key}", self)
                if count is not None:
                    raw[key] = values[normalized] = count
            for key in ("spec_cap_length", "spec_block_accept_length"):
                value = _number(spec.get(key), f"{prefix}.{key}", self)
                if value:
                    raw[key] = value
        for key in ("spec_correct_drafts_histogram", "spec_cap_lens_histogram"):
            counts = _counts(spec.get(key), f"{prefix}.{key}", self)
            if counts:
                raw[key] = counts
        return raw

    def _sglang_cached(self, cached: Any) -> dict[str, Any]:
        prefix = "sglext.cached_tokens_details"
        if not isinstance(cached, Mapping):
            self.note_malformed(prefix)
            return {}
        raw: dict[str, Any] = {}
        for key in ("device", "host", "storage"):
            count = _count(cached.get(key), f"{prefix}.{key}", self)
            if count is not None:
                raw[key] = count
        backend = cached.get("storage_backend")
        if backend is not None:
            if (
                isinstance(backend, str)
                and _LABEL.match(backend)
                and not self._unsafe_text(backend)
            ):
                raw["storage_backend"] = backend
            else:
                self.note_malformed(f"{prefix}.storage_backend")
        return raw

    def absorb_trtllm_perf_metrics(self, data: str) -> None:
        """The ``trtllm.perf_metrics`` event: header name to Server-Timing list."""
        label = TRTLLM_PERF_METRICS_EVENT
        try:
            payload = json.loads(data)
        except (ValueError, RecursionError):
            self.note_malformed(label)
            return
        if not isinstance(payload, Mapping):
            self.note_malformed(label)
            return
        headers = {str(name).lower(): value for name, value in payload.items()}
        raw: dict[str, Any] = {}
        for header, key, parameter in _TRTLLM_HEADERS:
            if header not in headers:
                continue
            entries = _timing_entries(
                headers[header], parameter, f"{label}.{header}", self
            )
            if entries:
                raw[key] = entries
        timing = raw.get("server_timing", {})
        for name, normalized in _TRTLLM_NORMALIZED:
            if name in timing:
                self._values[normalized] = timing[name]
        self._record(SOURCE_TRTLLM, raw)


def _number(
    value: Any,
    label: str,
    sink: ServerReportedAccumulator,
    *,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        sink.note_malformed(label)
        return None
    try:
        numeric = float(value)
    except OverflowError:
        sink.note_malformed(label)
        return None
    if (
        not math.isfinite(numeric)
        or numeric < 0
        or (maximum is not None and numeric > maximum)
    ):
        sink.note_malformed(label)
        return None
    return numeric


def _count(value: Any, label: str, sink: ServerReportedAccumulator) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > _MAX_EXACT_COUNT
    ):
        sink.note_malformed(label)
        return None
    return int(value)


def _counts(
    value: Any, label: str, sink: ServerReportedAccumulator
) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) > _MAX_LIST_ITEMS:
        sink.note_malformed(label)
        return None
    counts: list[int] = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 0
            or item > _MAX_EXACT_COUNT
        ):
            sink.note_malformed(label)
            return None
        counts.append(int(item))
    return counts


def _timing_entries(
    value: Any, parameter: str, label: str, sink: ServerReportedAccumulator
) -> dict[str, float]:
    """``name;dur=1.25, other;dur=2`` as ``{name: value}``; bad entries are dropped."""
    if not isinstance(value, str):
        sink.note_malformed(label)
        return {}
    items = [item.strip() for item in value.split(",") if item.strip()]
    if len(items) > _MAX_TIMING_ENTRIES:
        sink.note_malformed(label)
        return {}
    entries: dict[str, float] = {}
    for item in items:
        name, *parameters = (part.strip() for part in item.split(";"))
        numeric: float | None = None
        for candidate in parameters:
            key, separator, text = candidate.partition("=")
            if separator and key.strip() == parameter:
                try:
                    numeric = float(text.strip())
                except ValueError:
                    numeric = None
                break
        if (
            not _ENTRY_NAME.match(name)
            or sink._unsafe_text(name)
            or numeric is None
            or not math.isfinite(numeric)
            or numeric < 0
        ):
            sink.note_malformed(label)
            continue
        entries[name] = numeric
    return entries


__all__ = [
    "NORMALIZED_FIELDS",
    "PROVENANCE",
    "SGLANG_DETAILS_REQUEST_FIELDS",
    "SOURCE_LLAMA_CPP",
    "SOURCE_OLLAMA",
    "SOURCE_SGLANG",
    "SOURCE_TRTLLM",
    "SOURCE_VLLM",
    "TRTLLM_PERF_METRICS_EVENT",
    "TRTLLM_RETURN_METRICS_HEADER",
    "ServerReportedAccumulator",
    "ServerReportedMetrics",
]
