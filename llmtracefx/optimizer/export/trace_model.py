"""Build a vendor-neutral span tree from LLMTraceFX evidence.

The OTLP and Perfetto writers both render this model, so they agree on
every span boundary. Nothing here reads a clock: the same inputs always
produce the same trace.

Two kinds of child span exist and are labelled differently:

* API runs carry real client-observed offsets (response headers, first
  body chunk, first output, last event, completion). Their children are
  placed at those offsets (``measured_offsets``).
* Local runs only record how long each phase took, not when it started.
  Their children are laid back to back from the start of the run
  (``sequential_from_durations``). That placement is an assumption and
  the span says so.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any

from .._artifact_io import (
    MAX_EVIDENCE_ARTIFACT_BYTES,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from ..schema import ExperimentRecord, Measurement, MetricProvenance

AttributeValue = str | bool | int | float | list[str]

PLACEMENT_MEASURED = "measured_offsets"
PLACEMENT_SEQUENTIAL = "sequential_from_durations"
API_RUNTIME_NAME = "openai-compatible-api"
SERVER_EVIDENCE_KIND = "llmtracefx.server_metrics.evidence"
_PHASES = ("model_load", "tokenize", "prefill", "decode")
_OUTPUT_EVENT_KINDS = frozenset({"content", "reasoning"})
_WINDOWS_PATH = re.compile(r"^[A-Za-z]:[\\/]")


class ExportError(ValueError):
    """Raised when evidence cannot be exported without inventing data."""


@dataclass(frozen=True)
class TraceEvent:
    name: str
    time_ns: int
    attributes: dict[str, AttributeValue] = field(default_factory=dict)


@dataclass
class TraceSpan:
    key: str
    name: str
    start_ns: int
    end_ns: int
    parent_key: str | None = None
    client: bool = False
    attributes: dict[str, AttributeValue] = field(default_factory=dict)
    events: list[TraceEvent] = field(default_factory=list)
    error_message: str | None = None


@dataclass(frozen=True)
class TraceModel:
    run_id: str
    started_at: str
    resource: dict[str, AttributeValue]
    spans: tuple[TraceSpan, ...]
    """Root first, then children in start order."""

    @property
    def root(self) -> TraceSpan:
        return self.spans[0]


def llmtracefx_version() -> str | None:
    try:
        return metadata.version("llmtracefx")
    except metadata.PackageNotFoundError:
        return None


def read_json_artifact(path: Path, *, expected_kind: str | None = None) -> Any:
    try:
        text = read_bounded_regular_text(path, MAX_EVIDENCE_ARTIFACT_BYTES)
        data = json.loads(text, parse_constant=reject_non_finite_json_constant)
    except (OSError, ValueError) as exc:
        raise ExportError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ExportError(f"{path} must contain a JSON object")
    if expected_kind is not None and data.get("kind") != expected_kind:
        raise ExportError(f"{path} is not a {expected_kind} document")
    return data


def _epoch_ns(timestamp: str) -> int:
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ExportError(f"invalid timestamp {timestamp!r}") from exc
    if parsed.tzinfo is None:
        raise ExportError(f"timestamp {timestamp!r} has no timezone")
    delta = parsed - datetime(1970, 1, 1, tzinfo=UTC)
    return (
        delta.days * 86_400 + delta.seconds
    ) * 1_000_000_000 + delta.microseconds * 1_000


def _ms_to_ns(value: float) -> int:
    return int(round(value * 1_000_000))


def _model_label(model_id: str, *, include_paths: bool) -> str:
    if include_paths:
        return model_id
    looks_local = (
        model_id.startswith(("/", "~", "./", "../", "\\"))
        or _WINDOWS_PATH.match(model_id) is not None
    )
    if not looks_local:
        return model_id
    name = PurePosixPath(model_id.replace("\\", "/")).name
    return name or "local-model"


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _put(target: dict[str, AttributeValue], key: str, value: Any) -> None:
    """Set ``key`` only when ``value`` is present and representable."""
    if value is None:
        return
    if isinstance(value, float) and not math.isfinite(value):
        return
    if isinstance(value, (str, bool, int, float)):
        target[key] = value
    elif isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        target[key] = list(value)


def _put_measurement(
    target: dict[str, AttributeValue], prefix: str, measurement: Measurement | None
) -> None:
    if measurement is None or not math.isfinite(measurement.value):
        return
    target[f"{prefix}.value"] = float(measurement.value)
    target[f"{prefix}.unit"] = measurement.unit
    target[f"{prefix}.provenance"] = measurement.provenance.value


def _first_output_offset_ms(evidence: Mapping[str, Any]) -> tuple[float, str] | None:
    """Time to the first chunk carrying new output, reasoning included.

    The GenAI convention for ``gen_ai.response.time_to_first_chunk`` counts
    reasoning output. The record's first-content offset does not, so it is
    only used when the provider streamed no reasoning at all.
    """
    timeline = evidence.get("timeline")
    statistics = evidence.get("statistics")
    if not isinstance(timeline, Mapping):
        return None
    events = timeline.get("events")
    offsets = [
        offset
        for event in (events if isinstance(events, list) else [])
        if isinstance(event, Mapping)
        and event.get("kind") in _OUTPUT_EVENT_KINDS
        and (_integer(event.get("characters")) or 0) > 0
        and (offset := _number(event.get("offset_ms"))) is not None
    ]
    if offsets:
        return min(offsets), "earliest_output_event"
    reasoning = (
        _integer(statistics.get("reasoning_delta_count"))
        if isinstance(statistics, Mapping)
        else None
    )
    first_content = _number(timeline.get("first_content_token_offset_ms"))
    if reasoning == 0 and first_content is not None:
        return first_content, "first_content_token_offset"
    return None


def _api_children(evidence: Mapping[str, Any], root: TraceSpan) -> list[TraceSpan]:
    timeline = evidence.get("timeline")
    if not isinstance(timeline, Mapping):
        return []
    first_output = _first_output_offset_ms(evidence)
    marks: list[tuple[str, float | None]] = [
        ("start", 0.0),
        ("headers", _number(timeline.get("response_headers_offset_ms"))),
        ("first_body_chunk", _number(timeline.get("first_body_chunk_offset_ms"))),
        ("first_output", None if first_output is None else first_output[0]),
        ("last_event", _number(timeline.get("last_event_offset_ms"))),
        ("completed", _number(timeline.get("completed_offset_ms"))),
    ]
    names = {
        ("start", "headers"): "wait response_headers",
        ("headers", "first_body_chunk"): "wait first_body_chunk",
        ("first_body_chunk", "first_output"): "wait first_output",
        ("first_output", "last_event"): "stream output",
        ("last_event", "completed"): "stream tail",
    }
    present = [(label, value) for label, value in marks if value is not None]
    children: list[TraceSpan] = []
    for (start_label, start), (end_label, end) in zip(
        present, present[1:], strict=False
    ):
        name = names.get((start_label, end_label))
        if name is None or end < start:
            continue
        start_ns = root.start_ns + _ms_to_ns(start)
        end_ns = min(root.start_ns + _ms_to_ns(end), root.end_ns)
        if end_ns < start_ns:
            continue
        children.append(
            TraceSpan(
                key=f"api:{start_label}:{end_label}",
                name=name,
                start_ns=start_ns,
                end_ns=end_ns,
                parent_key=root.key,
                attributes={
                    "llmtracefx.span.placement": PLACEMENT_MEASURED,
                    "llmtracefx.span.provenance": (
                        MetricProvenance.MEASURED_WALL_CLOCK.value
                    ),
                    "llmtracefx.span.clock": "monotonic_client_perf_counter",
                    "llmtracefx.span.from": start_label,
                    "llmtracefx.span.to": end_label,
                },
            )
        )
    return children


def _api_events(evidence: Mapping[str, Any], root: TraceSpan) -> list[TraceEvent]:
    timeline = evidence.get("timeline")
    if not isinstance(timeline, Mapping):
        return []
    raw_events = timeline.get("events")
    events: list[TraceEvent] = []
    for raw in raw_events if isinstance(raw_events, list) else []:
        if not isinstance(raw, Mapping):
            continue
        offset = _number(raw.get("offset_ms"))
        kind = raw.get("kind")
        if offset is None or not isinstance(kind, str):
            continue
        attributes: dict[str, AttributeValue] = {"llmtracefx.sse.kind": kind}
        _put(attributes, "llmtracefx.sse.index", _integer(raw.get("index")))
        _put(attributes, "llmtracefx.sse.characters", _integer(raw.get("characters")))
        events.append(
            TraceEvent(
                name="llmtracefx.sse_event",
                time_ns=min(root.start_ns + _ms_to_ns(offset), root.end_ns),
                attributes=attributes,
            )
        )
    return events


def _local_children(record: ExperimentRecord, root: TraceSpan) -> list[TraceSpan]:
    children: list[TraceSpan] = []
    cursor = root.start_ns
    for phase in _PHASES:
        measurement = getattr(record.timing, phase)
        if measurement is None or measurement.unit != "ms":
            continue
        if not math.isfinite(measurement.value) or measurement.value < 0:
            continue
        duration = _ms_to_ns(measurement.value)
        children.append(
            TraceSpan(
                key=f"phase:{phase}",
                name=phase,
                start_ns=cursor,
                end_ns=cursor + duration,
                parent_key=root.key,
                attributes={
                    "llmtracefx.span.placement": PLACEMENT_SEQUENTIAL,
                    "llmtracefx.span.provenance": measurement.provenance.value,
                    "llmtracefx.span.duration_ms": float(measurement.value),
                },
            )
        )
        cursor += duration
    if children and cursor > root.end_ns:
        # The phases do not fit inside the measured total, so any layout
        # would contradict a measurement. Report durations only.
        root.attributes["llmtracefx.span.phase_layout"] = "omitted_phases_exceed_total"
        return []
    return children


def _server_attributes(
    target: dict[str, AttributeValue], server_evidence: Mapping[str, Any]
) -> None:
    attribution = server_evidence.get("attribution")
    if not isinstance(attribution, Mapping):
        raise ExportError("server metrics evidence has no attribution block")
    _put(target, "llmtracefx.server.runtime", server_evidence.get("runtime"))
    _put(target, "llmtracefx.server.attribution.status", attribution.get("status"))
    _put(target, "llmtracefx.server.attribution.reason", attribution.get("reason"))
    if attribution.get("attributable") is not True:
        return
    highlights = server_evidence.get("highlights")
    if not isinstance(highlights, Mapping):
        return
    for key in sorted(highlights):
        entry = highlights[key]
        if not isinstance(entry, Mapping) or not re.fullmatch(r"[a-z0-9_]+", key):
            continue
        value = _number(entry.get("value"))
        if value is None:
            continue
        prefix = f"llmtracefx.server.{key}"
        target[f"{prefix}.value"] = value
        _put(target, f"{prefix}.unit", entry.get("unit"))
        _put(target, f"{prefix}.provenance", entry.get("provenance"))
        _put(target, f"{prefix}.source", entry.get("source"))


def _require_window_brackets_run(
    server_evidence: Mapping[str, Any], start_ns: int, end_ns: int
) -> None:
    """Refuse server evidence whose scrape window does not contain the run.

    A window that misses any part of the run cannot describe it, and a stale
    evidence file from an earlier run would otherwise be attached silently.
    """
    bounds: list[int] = []
    for side in ("before", "after"):
        meta = server_evidence.get(side)
        captured = meta.get("captured_at") if isinstance(meta, Mapping) else None
        if not isinstance(captured, str):
            raise ExportError(f"server metrics evidence has no {side}.captured_at")
        bounds.append(_epoch_ns(captured))
    if bounds[0] > start_ns or bounds[1] < end_ns:
        raise ExportError(
            "server metrics window does not bracket the run: the before snapshot "
            "must precede the run start and the after snapshot must follow its end"
        )


def _request_parameters(
    target: dict[str, AttributeValue], parameters: Mapping[str, Any]
) -> None:
    max_tokens = _integer(parameters.get("max_tokens"))
    if max_tokens is None:
        max_tokens = _integer(parameters.get("max_completion_tokens"))
    _put(target, "gen_ai.request.max_tokens", max_tokens)
    _put(target, "gen_ai.request.seed", _integer(parameters.get("seed")))
    _put(target, "gen_ai.request.temperature", _number(parameters.get("temperature")))
    _put(target, "gen_ai.request.top_p", _number(parameters.get("top_p")))


def _server_address(target: dict[str, AttributeValue], origin: Any) -> None:
    if not isinstance(origin, str):
        return
    try:
        parts = urllib.parse.urlsplit(origin)
        port = parts.port
    except ValueError:
        return
    _put(target, "server.address", parts.hostname)
    if port is None and parts.scheme in ("http", "https"):
        port = 443 if parts.scheme == "https" else 80
    _put(target, "server.port", port)


def _trace_hex(seed: str, length: int) -> str:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:length]
    return digest if digest.strip("0") else "1".rjust(length, "0")


def trace_id(run_id: str) -> str:
    return _trace_hex(f"llmtracefx/trace/{run_id}", 32)


def span_id(run_id: str, key: str) -> str:
    return _trace_hex(f"llmtracefx/span/{run_id}/{key}", 16)


def build_trace(
    record: ExperimentRecord,
    *,
    api_evidence: Mapping[str, Any] | None = None,
    server_evidence: Mapping[str, Any] | None = None,
    include_paths: bool = False,
) -> TraceModel:
    is_api = record.runtime.name == API_RUNTIME_NAME
    if api_evidence is not None:
        if not is_api:
            raise ExportError(
                "API evidence was given for a record that is not an API run"
            )
        if api_evidence.get("run_id") != record.run_id:
            raise ExportError("API evidence run_id does not match the record run_id")
    if server_evidence is not None and server_evidence.get("kind") != (
        SERVER_EVIDENCE_KIND
    ):
        raise ExportError(f"server evidence is not a {SERVER_EVIDENCE_KIND} document")

    start_ns = _epoch_ns(record.started_at)
    total = record.timing.total
    if total is not None and total.unit == "ms" and math.isfinite(total.value):
        end_ns = start_ns + _ms_to_ns(max(0.0, total.value))
        duration_source = "timing.total"
    elif record.ended_at is not None:
        end_ns = max(start_ns, _epoch_ns(record.ended_at))
        duration_source = "ended_at"
    else:
        end_ns = start_ns
        duration_source = "unknown"
    if server_evidence is not None:
        _require_window_brackets_run(server_evidence, start_ns, end_ns)

    operation = "chat" if is_api else "text_completion"
    model = _model_label(record.model.model_id, include_paths=include_paths)
    provider = record.runtime.provider if is_api else None
    root_attributes: dict[str, AttributeValue] = {}
    _put(root_attributes, "gen_ai.operation.name", operation)
    _put(root_attributes, "gen_ai.provider.name", provider or record.runtime.name)
    _put(root_attributes, "gen_ai.request.model", model)
    _put(root_attributes, "gen_ai.usage.input_tokens", record.tokens.input_tokens)
    _put(root_attributes, "gen_ai.usage.output_tokens", record.tokens.generated_tokens)
    if record.tokens.provenance is not None:
        root_attributes["llmtracefx.tokens.provenance"] = record.tokens.provenance.value
    _put(root_attributes, "llmtracefx.run_id", record.run_id)
    _put(root_attributes, "llmtracefx.runtime.name", record.runtime.name)
    _put(root_attributes, "llmtracefx.runtime.version", record.runtime.version)
    _put(root_attributes, "llmtracefx.runtime.backend", record.runtime.backend)
    _put(root_attributes, "llmtracefx.model.revision", record.model.model_revision)
    _put(root_attributes, "llmtracefx.model.quantization", record.model.quantization)
    _put(root_attributes, "llmtracefx.command.config_hash", record.command.config_hash)
    _put(
        root_attributes,
        "llmtracefx.command.workload_hash",
        record.command.workload_hash,
    )
    _put(
        root_attributes,
        "llmtracefx.repetition.index",
        record.repetition.repetition_index,
    )
    _put(root_attributes, "llmtracefx.repetition.seed", record.repetition.seed)
    _put(root_attributes, "llmtracefx.outcome.success", record.outcome.success)
    root_attributes["llmtracefx.span.duration_source"] = duration_source
    for phase in (*_PHASES, "total"):
        _put_measurement(
            root_attributes, f"llmtracefx.timing.{phase}", getattr(record.timing, phase)
        )
    for slot in ("active", "cache", "peak", "wired"):
        _put_measurement(
            root_attributes, f"llmtracefx.memory.{slot}", getattr(record.memory, slot)
        )
    _put_measurement(
        root_attributes, "llmtracefx.power.average", record.power.average_power
    )
    _put_measurement(root_attributes, "llmtracefx.power.energy", record.power.energy)

    error_message = None
    if record.error is not None:
        root_attributes["error.type"] = record.error.category
        error_message = record.error.message
    elif not record.outcome.success:
        root_attributes["error.type"] = "_OTHER"
        error_message = record.outcome.notes or "run did not succeed"

    if api_evidence is not None:
        _put(root_attributes, "gen_ai.response.id", api_evidence.get("response_id"))
        _put(
            root_attributes,
            "gen_ai.response.model",
            api_evidence.get("response_model"),
        )
        finish_reason = api_evidence.get("finish_reason")
        if isinstance(finish_reason, str):
            root_attributes["gen_ai.response.finish_reasons"] = [finish_reason]
        usage = api_evidence.get("usage")
        if isinstance(usage, Mapping):
            _put(
                root_attributes,
                "gen_ai.usage.reasoning.output_tokens",
                _integer(usage.get("reasoning_tokens")),
            )
            _put(
                root_attributes,
                "gen_ai.usage.cache_read.input_tokens",
                _integer(usage.get("cached_prompt_tokens")),
            )
            _put(
                root_attributes,
                "gen_ai.usage.cache_write.input_tokens",
                _integer(usage.get("cache_write_tokens")),
            )
        plan = api_evidence.get("plan")
        if isinstance(plan, Mapping):
            parameters = plan.get("request_parameters")
            if isinstance(parameters, Mapping):
                _request_parameters(root_attributes, parameters)
            _server_address(root_attributes, plan.get("endpoint_origin"))
        first_output = _first_output_offset_ms(api_evidence)
        if first_output is not None:
            root_attributes["gen_ai.response.time_to_first_chunk"] = (
                first_output[0] / 1000.0
            )
            root_attributes["llmtracefx.time_to_first_chunk.source"] = first_output[1]
            root_attributes["llmtracefx.time_to_first_chunk.provenance"] = (
                MetricProvenance.MEASURED_WALL_CLOCK.value
            )
        timeline = api_evidence.get("timeline")
        if isinstance(timeline, Mapping):
            for name in (
                "response_headers_offset_ms",
                "first_body_chunk_offset_ms",
                "first_content_token_offset_ms",
            ):
                _put(
                    root_attributes,
                    f"llmtracefx.api.{name}",
                    _number(timeline.get(name)),
                )
            _put(
                root_attributes,
                "llmtracefx.api.events_truncated",
                timeline.get("events_truncated"),
            )

    if server_evidence is not None:
        _server_attributes(root_attributes, server_evidence)

    root = TraceSpan(
        key="root",
        name=f"{operation} {model}",
        start_ns=start_ns,
        end_ns=end_ns,
        client=is_api,
        attributes=root_attributes,
        error_message=error_message,
    )
    if api_evidence is not None:
        children = _api_children(api_evidence, root)
        root.events.extend(_api_events(api_evidence, root))
    elif is_api:
        children = []
    else:
        children = _local_children(record, root)

    resource: dict[str, AttributeValue] = {"service.name": "llmtracefx"}
    _put(resource, "service.version", llmtracefx_version())
    os_type = record.platform.os_name.lower()
    if os_type in ("darwin", "linux", "windows", "freebsd"):
        resource["os.type"] = os_type
    _put(resource, "host.arch", record.platform.architecture or None)
    _put(resource, "llmtracefx.platform.accelerator", record.platform.accelerator)

    return TraceModel(
        run_id=record.run_id,
        started_at=record.started_at,
        resource=resource,
        spans=(root, *sorted(children, key=lambda span: (span.start_ns, span.key))),
    )
