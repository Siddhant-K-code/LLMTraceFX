"""Aggregate LLMTraceFX runs into OTLP/JSON GenAI histograms.

The output is one ``MetricsData`` object per line, the format of the
OpenTelemetry file exporter, so the Collector's ``otlpjsonfile`` receiver
reads it and the object can be POSTed to any OTLP/HTTP ``/v1/metrics``
endpoint. Metric names, units, bucket boundaries and attributes follow the
GenAI semantic conventions in ``open-telemetry/semantic-conventions-genai``
(see :data:`SEMCONV_REFERENCE`).
The conventions define no energy or power metric, so GPU energy and
average power from a ``gpu_energy_evidence.json`` sidecar use the
``llmtracefx.gpu.*`` names instead.

Only measured values are recorded. A run that lacks the evidence for a
metric contributes nothing to it, and the reason is reported in
:attr:`MetricsExport.notes`. Every data point carries the provenance and
source of its values, so values obtained in different ways never share a
histogram.

All runs in one export form a single cumulative stream: the start time is
the earliest run start and the point time is the latest run end.
"""

from __future__ import annotations

import bisect
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

from ..gpu_energy.record_power import (
    GPU_ENERGY_EVIDENCE_FILE,
    GpuEnergyEvidenceError,
    apply_sidecar,
    shared_window_refusals,
)
from ..schema import ExperimentRecord, Measurement, MetricProvenance
from ..server_metrics.diff import PROFILES
from .otlp import SCOPE_NAME, _attributes
from .trace_model import (
    API_RUNTIME_NAME,
    SERVER_EVIDENCE_KIND,
    AttributeValue,
    ExportError,
    TraceModel,
    _first_output_offset_ms,
    _integer,
    _ms_to_ns,
    _number,
    build_trace,
    llmtracefx_version,
    read_json_artifact,
    span_id,
    trace_id,
)

SEMCONV_REFERENCE = "open-telemetry/semantic-conventions-genai@e6658f1"
AGGREGATION_TEMPORALITY_CUMULATIVE = 2
API_EVIDENCE_FILE = "api_evidence.json"
SERVER_METRICS_EVIDENCE_FILE = "server_metrics_evidence.json"
PROVENANCE_ATTRIBUTE = "llmtracefx.provenance"
SOURCE_ATTRIBUTE = "llmtracefx.source"
GPU_ATTRIBUTION_ATTRIBUTE = "llmtracefx.gpu.attribution_status"
GPU_DEVICE_COUNT_ATTRIBUTE = "llmtracefx.gpu.device_count"

_SECONDS_BOUNDS = (
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
)
_TOKEN_BOUNDS = (
    1.0,
    4.0,
    16.0,
    64.0,
    256.0,
    1024.0,
    4096.0,
    16384.0,
    65536.0,
    262144.0,
    1048576.0,
    4194304.0,
    16777216.0,
    67108864.0,
)
_SERVER_TTFT_BOUNDS = (
    0.001,
    0.005,
    0.01,
    0.02,
    0.04,
    0.06,
    0.08,
    0.1,
    0.25,
    0.5,
    0.75,
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
)
_SERVER_TPOT_BOUNDS = (
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1.0,
    2.5,
)
_ENERGY_BOUNDS = tuple(float(4**power) for power in range(13))
_POWER_BOUNDS = (
    10.0,
    25.0,
    50.0,
    75.0,
    100.0,
    150.0,
    200.0,
    300.0,
    400.0,
    500.0,
    700.0,
    1000.0,
    1500.0,
    2000.0,
    3000.0,
    5000.0,
)
_BASE_ATTRIBUTES = (
    "gen_ai.operation.name",
    "gen_ai.provider.name",
    "gen_ai.request.model",
    "gen_ai.response.model",
    "server.address",
    "server.port",
)
_TOOL_CALL_FINISH_REASONS = frozenset({"tool_calls", "function_call"})
_OUTPUT_EVENT_KINDS = frozenset({"content", "reasoning"})


@dataclass(frozen=True)
class MetricSpec:
    name: str
    unit: str
    description: str
    bounds: tuple[float, ...]
    exemplars: bool = True
    error_type: bool = False


CLIENT_DURATION = MetricSpec(
    "gen_ai.client.inference.duration",
    "s",
    "GenAI client inference operation duration.",
    _SECONDS_BOUNDS,
    error_type=True,
)
CLIENT_TIME_TO_FIRST_CHUNK = MetricSpec(
    "gen_ai.client.inference.time_to_first_chunk",
    "s",
    "Time from the start of a logical inference operation to the first chunk "
    "in its streaming response.",
    _SECONDS_BOUNDS,
)
CLIENT_TIME_PER_OUTPUT_CHUNK = MetricSpec(
    "gen_ai.client.inference.time_per_output_chunk",
    "s",
    "Time per output chunk, recorded for each chunk received after the first "
    "one, measured as the time elapsed from the end of the previous chunk to "
    "the end of the current chunk.",
    _SECONDS_BOUNDS,
    exemplars=False,
)
CLIENT_INPUT_TOKENS = MetricSpec(
    "gen_ai.client.inference.operation.input_tokens",
    "{token}",
    "The number of input (prompt) tokens used per inference operation.",
    _TOKEN_BOUNDS,
    error_type=True,
)
CLIENT_OUTPUT_TOKENS = MetricSpec(
    "gen_ai.client.inference.operation.output_tokens",
    "{token}",
    "The number of output (completion) tokens used per inference operation.",
    _TOKEN_BOUNDS,
    error_type=True,
)
SERVER_TIME_TO_FIRST_TOKEN = MetricSpec(
    "gen_ai.server.time_to_first_token",
    "s",
    "Time to generate first token for successful responses.",
    _SERVER_TTFT_BOUNDS,
)
SERVER_TIME_PER_OUTPUT_TOKEN = MetricSpec(
    "gen_ai.server.time_per_output_token",
    "s",
    "Time per output token generated after the first token for successful "
    "responses.",
    _SERVER_TPOT_BOUNDS,
)
SERVER_REQUEST_DURATION = MetricSpec(
    "gen_ai.server.request.duration",
    "s",
    "Generative AI server request duration such as time-to-last byte or last "
    "output token.",
    _SECONDS_BOUNDS,
    error_type=True,
)
# The GenAI conventions define no energy or power metric, and the hardware
# conventions' hw.energy and hw.power describe one component over its
# lifetime, not a window around one run. So these use the llmtracefx
# namespace rather than borrow a name with a different meaning.
GPU_ENERGY = MetricSpec(
    "llmtracefx.gpu.energy",
    "J",
    "Device wide GPU energy over the gpu-energy window that bracketed the "
    "run, from the NVML energy counter. Includes idle power and command "
    "start up.",
    _ENERGY_BOUNDS,
)
GPU_AVERAGE_POWER = MetricSpec(
    "llmtracefx.gpu.power.average",
    "W",
    "Device wide average GPU power over the gpu-energy window that bracketed "
    "the run: counter energy over the counter read interval.",
    _POWER_BOUNDS,
)
METRICS = (
    CLIENT_DURATION,
    CLIENT_TIME_TO_FIRST_CHUNK,
    CLIENT_TIME_PER_OUTPUT_CHUNK,
    CLIENT_INPUT_TOKENS,
    CLIENT_OUTPUT_TOKENS,
    SERVER_TIME_TO_FIRST_TOKEN,
    SERVER_TIME_PER_OUTPUT_TOKEN,
    SERVER_REQUEST_DURATION,
    GPU_ENERGY,
    GPU_AVERAGE_POWER,
)

# Server highlights whose runtime definition was checked against the
# convention. vLLM and SGLang measure TTFT and e2e latency from request
# arrival and TPOT as (last token - first token) / (tokens - 1). TGI's
# mean time per token divides the whole inference time, prefill included,
# by every generated token, so only its request duration is mapped.
_SERVER_MAPPING: dict[str, dict[str, MetricSpec]] = {
    "vllm": {
        "time_to_first_token": SERVER_TIME_TO_FIRST_TOKEN,
        "time_per_output_token": SERVER_TIME_PER_OUTPUT_TOKEN,
        "e2e_latency": SERVER_REQUEST_DURATION,
    },
    "sglang": {
        "time_to_first_token": SERVER_TIME_TO_FIRST_TOKEN,
        "time_per_output_token": SERVER_TIME_PER_OUTPUT_TOKEN,
        "e2e_latency": SERVER_REQUEST_DURATION,
    },
    "tgi": {"e2e_latency": SERVER_REQUEST_DURATION},
}
# vLLM records a TPOT of 0 when only one token was generated.
_ZERO_IS_NO_VALUE = frozenset({("vllm", "time_per_output_token")})


@dataclass(frozen=True)
class MetricsRun:
    """One record with the evidence collected next to it."""

    record: ExperimentRecord
    api_evidence: Mapping[str, Any] | None = None
    server_evidence: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class MetricsExport:
    payload: dict[str, Any]
    observation_count: int
    exported_run_ids: tuple[str, ...]
    notes: tuple[str, ...]
    """Why a run, or one metric of a run, was left out."""


@dataclass(frozen=True)
class _Exemplar:
    time_ns: int
    value: float
    run_id: str


@dataclass
class _Point:
    bounds: tuple[float, ...]
    values: list[float] = field(default_factory=list)
    exemplars: list[_Exemplar] = field(default_factory=list)


_AttributeKey = tuple[tuple[str, AttributeValue], ...]


class _Collector:
    def __init__(self) -> None:
        self.resources: dict[_AttributeKey, dict[str, dict[_AttributeKey, _Point]]] = {}
        self.count = 0

    def observe(
        self,
        resource: _AttributeKey,
        spec: MetricSpec,
        attributes: dict[str, AttributeValue],
        value: float,
        *,
        exemplar: _Exemplar | None = None,
    ) -> None:
        metrics = self.resources.setdefault(resource, {})
        points = metrics.setdefault(spec.name, {})
        key = tuple(sorted(attributes.items()))
        point = points.setdefault(key, _Point(spec.bounds))
        point.values.append(value)
        if exemplar is not None and spec.exemplars:
            point.exemplars.append(exemplar)
        self.count += 1


def _bucket_counts(bounds: tuple[float, ...], values: list[float]) -> list[str]:
    """Bucket ``i`` holds ``bounds[i-1] < v <= bounds[i]``; the last is open."""
    counts = [0] * (len(bounds) + 1)
    for value in values:
        counts[bisect.bisect_left(bounds, value)] += 1
    return [str(count) for count in counts]


def _data_point(
    attributes: _AttributeKey, point: _Point, start_ns: int, end_ns: int
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "attributes": _attributes(dict(attributes)),
        "startTimeUnixNano": str(start_ns),
        "timeUnixNano": str(end_ns),
        "count": str(len(point.values)),
        "sum": math.fsum(point.values),
        "bucketCounts": _bucket_counts(point.bounds, point.values),
        "explicitBounds": list(point.bounds),
    }
    if point.exemplars:
        payload["exemplars"] = [
            {
                "filteredAttributes": _attributes({"llmtracefx.run_id": item.run_id}),
                "timeUnixNano": str(item.time_ns),
                "asDouble": item.value,
                "spanId": span_id(item.run_id, "root"),
                "traceId": trace_id(item.run_id),
            }
            for item in sorted(point.exemplars, key=lambda e: (e.time_ns, e.run_id))
        ]
    payload["min"] = min(point.values)
    payload["max"] = max(point.values)
    return payload


def _sort_key(attributes: _AttributeKey) -> str:
    return json.dumps(attributes, sort_keys=True)


def _render(collector: _Collector, start_ns: int, end_ns: int) -> dict[str, Any]:
    scope: dict[str, Any] = {"name": SCOPE_NAME}
    version = llmtracefx_version()
    if version is not None:
        scope["version"] = version
    resource_metrics = []
    for resource in sorted(collector.resources, key=_sort_key):
        metrics = collector.resources[resource]
        rendered = []
        for spec in METRICS:
            points = metrics.get(spec.name)
            if not points:
                continue
            rendered.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "unit": spec.unit,
                    "histogram": {
                        "dataPoints": [
                            _data_point(key, points[key], start_ns, end_ns)
                            for key in sorted(points, key=_sort_key)
                        ],
                        "aggregationTemporality": AGGREGATION_TEMPORALITY_CUMULATIVE,
                    },
                }
            )
        resource_metrics.append(
            {
                "resource": {"attributes": _attributes(dict(resource))},
                "scopeMetrics": [{"scope": scope, "metrics": rendered}],
            }
        )
    return {"resourceMetrics": resource_metrics}


def _with_provenance(
    base: dict[str, AttributeValue], provenance: str, source: str
) -> dict[str, AttributeValue]:
    return {**base, PROVENANCE_ATTRIBUTE: provenance, SOURCE_ATTRIBUTE: source}


def _is_streaming(api_evidence: Mapping[str, Any]) -> bool:
    plan = api_evidence.get("plan")
    parameters = plan.get("request_parameters") if isinstance(plan, Mapping) else None
    return isinstance(parameters, Mapping) and parameters.get("stream") is True


def _chunk_gaps_ms(api_evidence: Mapping[str, Any]) -> list[float] | str:
    """Gaps between consecutive output chunks, or why there are none."""
    timeline = api_evidence.get("timeline")
    if not isinstance(timeline, Mapping):
        return "API evidence has no timeline"
    if timeline.get("events_truncated") is not False:
        return "the event timeline was truncated, so some gaps are unknown"
    events = timeline.get("events")
    if not isinstance(events, list):
        return "API evidence has no event list"
    chunks: list[tuple[int, float]] = []
    for event in events:
        if not isinstance(event, Mapping) or event.get("kind") not in (
            _OUTPUT_EVENT_KINDS
        ):
            continue
        index = _integer(event.get("index"))
        offset = _number(event.get("offset_ms"))
        if (_integer(event.get("characters")) or 0) <= 0:
            continue
        if index is None or offset is None:
            return "an output event has no index or offset"
        chunks.append((index, offset))
    chunks.sort()
    if len(chunks) < 2:
        return "fewer than two output chunks arrived"
    gaps = [later[1] - earlier[1] for earlier, later in pairwise(chunks)]
    if min(gaps) < 0:
        return "output event offsets are not monotonic"
    return gaps


class _RunMapper:
    """Maps the evidence of one API run onto the collector."""

    def __init__(
        self,
        collector: _Collector,
        notes: list[str],
        record: ExperimentRecord,
        model: TraceModel,
        power_refusal: str | None = None,
    ) -> None:
        root = model.root
        self.power_refusal = power_refusal
        self.collector = collector
        self.notes = notes
        self.run_id = record.run_id
        self.resource: _AttributeKey = tuple(sorted(model.resource.items()))
        self.base = {
            key: root.attributes[key]
            for key in _BASE_ATTRIBUTES
            if key in root.attributes
        }
        self.error_type = root.attributes.get("error.type")
        self.start_ns = root.start_ns
        self.end_ns = root.end_ns

    def map(self, run: MetricsRun) -> bool:
        """Record every metric the run measured; True if any was recorded."""
        record = run.record
        before = self.collector.count
        self._duration(record)
        self._tokens(record)
        if run.api_evidence is None:
            self.note(record, f"chunk metrics omitted: no {API_EVIDENCE_FILE}")
        else:
            self._chunks(record, run.api_evidence)
        if run.server_evidence is not None:
            self._server(record, run.server_evidence)
        self._power(record)
        return self.collector.count > before

    def note(self, record: ExperimentRecord, message: str) -> None:
        self.notes.append(f"{record.run_id}: {message}")

    def observe(
        self,
        spec: MetricSpec,
        value: float,
        provenance: str,
        source: str,
        *,
        time_ns: int | None = None,
        extra: Mapping[str, AttributeValue] | None = None,
    ) -> None:
        attributes = {
            **_with_provenance(self.base, provenance, source),
            **(extra or {}),
        }
        if spec.error_type and self.error_type is not None:
            attributes["error.type"] = self.error_type
        exemplar = _Exemplar(
            self.end_ns if time_ns is None else time_ns, value, self.run_id
        )
        self.collector.observe(
            self.resource, spec, attributes, value, exemplar=exemplar
        )

    def _duration(self, record: ExperimentRecord) -> None:
        total = record.timing.total
        if (
            total is None
            or total.unit != "ms"
            or not math.isfinite(total.value)
            or total.value < 0
        ):
            self.note(record, "duration omitted: no measured timing.total in ms")
            return
        self.observe(
            CLIENT_DURATION,
            total.value / 1000.0,
            total.provenance.value,
            "record.timing.total",
        )

    def _tokens(self, record: ExperimentRecord) -> None:
        provenance = record.tokens.provenance
        if provenance is None:
            self.note(record, "token metrics omitted: token provenance not recorded")
            return
        if provenance is MetricProvenance.ESTIMATED:
            self.note(record, "token metrics omitted: token counts are estimated")
            return
        for spec, field_name in (
            (CLIENT_INPUT_TOKENS, "input_tokens"),
            (CLIENT_OUTPUT_TOKENS, "generated_tokens"),
        ):
            count = getattr(record.tokens, field_name)
            if count is None or count < 0:
                self.note(record, f"{spec.name} omitted: no {field_name}")
                continue
            self.observe(
                spec, float(count), provenance.value, f"record.tokens.{field_name}"
            )

    def _chunks(self, record: ExperimentRecord, evidence: Mapping[str, Any]) -> None:
        if not _is_streaming(evidence):
            self.note(record, "chunk metrics omitted: the request did not stream")
            return
        if evidence.get("finish_reason") in _TOOL_CALL_FINISH_REASONS:
            self.note(
                record,
                "chunk metrics omitted: tool call chunks are not timed by the "
                "collector",
            )
            return
        wall_clock = MetricProvenance.MEASURED_WALL_CLOCK.value
        first = _first_output_offset_ms(evidence)
        if first is None:
            self.note(record, "time to first chunk omitted: no output chunk arrived")
        else:
            offset_ms, how = first
            source = (
                "api_evidence.timeline.events"
                if how == "earliest_output_event"
                else "api_evidence.timeline.first_content_token_offset_ms"
            )
            self.observe(
                CLIENT_TIME_TO_FIRST_CHUNK,
                offset_ms / 1000.0,
                wall_clock,
                source,
                time_ns=self.start_ns + _ms_to_ns(offset_ms),
            )
        gaps = _chunk_gaps_ms(evidence)
        if isinstance(gaps, str):
            self.note(record, f"time per output chunk omitted: {gaps}")
            return
        for gap in gaps:
            self.observe(
                CLIENT_TIME_PER_OUTPUT_CHUNK,
                gap / 1000.0,
                wall_clock,
                "api_evidence.timeline.events",
            )

    def _power(self, record: ExperimentRecord) -> None:
        power = record.power
        source = power.source
        if source is None:
            if power.energy is not None or power.average_power is not None:
                self.note(record, "GPU energy omitted: the power values have no source")
            return
        reason = source.refused_reason or self.power_refusal
        if reason is None and (not record.outcome.success or record.error is not None):
            reason = "the run did not succeed"
        if reason is not None:
            self.note(record, f"GPU energy omitted: {reason}")
            return
        extra: dict[str, AttributeValue] = {
            GPU_ATTRIBUTION_ATTRIBUTE: source.attribution_status or "unknown",
            GPU_DEVICE_COUNT_ATTRIBUTE: source.device_count or 0,
        }
        measurement: Measurement | None
        for spec, measurement, method, unit in (
            (GPU_ENERGY, power.energy, source.energy_method, "J"),
            (GPU_AVERAGE_POWER, power.average_power, source.average_power_method, "W"),
        ):
            if measurement is None:
                self.note(record, f"{spec.name} omitted: no value in the evidence")
                continue
            if measurement.unit != unit or not math.isfinite(measurement.value):
                self.note(record, f"{spec.name} omitted: not a finite value in {unit}")
                continue
            self.observe(
                spec,
                measurement.value,
                measurement.provenance.value,
                f"{GPU_ENERGY_EVIDENCE_FILE}:{method}",
                extra=extra,
            )

    def _server(self, record: ExperimentRecord, evidence: Mapping[str, Any]) -> None:
        attribution = evidence.get("attribution")
        if (
            not isinstance(attribution, Mapping)
            or attribution.get("attributable") is not True
        ):
            status = (
                attribution.get("status") if isinstance(attribution, Mapping) else None
            )
            self.note(
                record,
                f"server metrics omitted: the window is not attributable ({status})",
            )
            return
        if not record.outcome.success or record.error is not None:
            self.note(record, "server metrics omitted: the run did not succeed")
            return
        runtime = evidence.get("runtime")
        mapping = _SERVER_MAPPING.get(runtime) if isinstance(runtime, str) else None
        if mapping is None or not isinstance(runtime, str):
            self.note(
                record,
                f"server metrics omitted: no checked mapping for runtime {runtime!r}",
            )
            return
        families = {spec.key: spec.family for spec in PROFILES[runtime].histograms}
        highlights = evidence.get("highlights")
        if not isinstance(highlights, Mapping):
            highlights = {}
        for key, spec in mapping.items():
            entry = highlights.get(key)
            if not isinstance(entry, Mapping):
                self.note(record, f"{spec.name} omitted: no {key} highlight")
                continue
            if entry.get("statistic") != "single_observation":
                self.note(
                    record,
                    f"{spec.name} omitted: the window holds more than one "
                    "observation, so only a mean is known",
                )
                continue
            value = _number(entry.get("value"))
            if (
                value is None
                or value < 0
                or entry.get("unit") != "ms"
                or entry.get("provenance") != MetricProvenance.MEASURED_NATIVE.value
                or entry.get("source") != families.get(key)
            ):
                self.note(record, f"{spec.name} omitted: unexpected {key} highlight")
                continue
            if value == 0 and (runtime, key) in _ZERO_IS_NO_VALUE:
                self.note(
                    record,
                    f"{spec.name} omitted: {runtime} records 0 when only one "
                    "token was generated",
                )
                continue
            self.observe(
                spec,
                value / 1000.0,
                MetricProvenance.MEASURED_NATIVE.value,
                str(entry["source"]),
            )


def build_otlp_metrics(
    runs: Sequence[MetricsRun], *, include_paths: bool = False
) -> MetricsExport:
    """Aggregate ``runs`` into one cumulative OTLP ``MetricsData`` payload.

    Each run is first passed through :func:`build_trace`, so every refusal
    of the trace export (mismatched evidence, a server window that does not
    bracket the run) applies here too.
    """
    seen: set[str] = set()
    for run in runs:
        if run.record.run_id in seen:
            raise ExportError(f"run_id {run.record.run_id} appears more than once")
        seen.add(run.record.run_id)
    collector = _Collector()
    notes: list[str] = []
    windows: list[tuple[int, int]] = []
    exported: list[str] = []
    shared = shared_window_refusals(run.record for run in runs)
    for run in sorted(runs, key=lambda item: item.record.run_id):
        record = run.record
        model = build_trace(
            record,
            api_evidence=run.api_evidence,
            server_evidence=run.server_evidence,
            include_paths=include_paths,
        )
        if record.runtime.name != API_RUNTIME_NAME:
            notes.append(
                f"{record.run_id}: skipped: not an API run, so there is no "
                "client to provider boundary to measure"
            )
            continue
        mapper = _RunMapper(
            collector, notes, record, model, power_refusal=shared.get(record.run_id)
        )
        if mapper.map(run):
            windows.append((model.root.start_ns, model.root.end_ns))
            exported.append(record.run_id)
    if not windows:
        return MetricsExport({"resourceMetrics": []}, 0, (), tuple(notes))
    start_ns = min(start for start, _ in windows)
    end_ns = max(end for _, end in windows)
    return MetricsExport(
        _render(collector, start_ns, end_ns),
        collector.count,
        tuple(exported),
        tuple(notes),
    )


def render_otlp_metrics_json_line(export: MetricsExport) -> str:
    """One JSON Lines record, including the trailing newline."""
    return json.dumps(export.payload, separators=(",", ":"), allow_nan=False) + "\n"


def load_metrics_runs(paths: Sequence[str | Path]) -> list[MetricsRun]:
    """Read records and the evidence written next to each one.

    A directory expands to every ``record.json`` below it. For each record,
    ``api_evidence.json`` and ``server_metrics_evidence.json`` in the same
    directory are attached when present, as ``collect-api`` writes them.
    A ``gpu_energy_evidence.json`` there fills the record's power in memory,
    or records why it was refused.
    """
    files: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found = sorted(path.rglob("record.json"))
            if not found:
                raise ExportError(f"no record.json found under {path}")
            files.extend(found)
        else:
            files.append(path)
    runs: list[MetricsRun] = []
    seen: set[Path] = set()
    for file in files:
        resolved = file.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            record = apply_sidecar(ExperimentRecord.read_json(file), file)
        except GpuEnergyEvidenceError as exc:
            raise ExportError(str(exc)) from exc
        api_path = file.parent / API_EVIDENCE_FILE
        server_path = file.parent / SERVER_METRICS_EVIDENCE_FILE
        runs.append(
            MetricsRun(
                record,
                read_json_artifact(api_path) if api_path.is_file() else None,
                (
                    read_json_artifact(server_path, expected_kind=SERVER_EVIDENCE_KIND)
                    if server_path.is_file()
                    else None
                ),
            )
        )
    return runs
