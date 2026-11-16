"""The imported benchmark artifact.

An imported benchmark holds summary statistics that another benchmark tool
computed and saved. LLMTraceFX did not measure these values and holds no
per-request samples, so they are not an ``ExperimentRecord``. Every value
keeps the key it came from in the source file, and the artifact records the
source tool, the tool version when the file states one, and the sha256 of the
exact source bytes.

Provenance labels are scoped to this artifact:

``imported_derived``
    A statistic (mean, median, percentile, standard deviation, minimum,
    maximum or rate) that the source tool computed from its own client-side
    timings.
``imported_reported``
    A count, total or duration that the source tool reported as is.

This module imports only the standard library so the perf gate can read
imported artifacts without installing extras.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .._artifact_io import (
    ArtifactReadError,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)

IMPORTED_BENCHMARK_KIND = "llmtracefx.imported_benchmark"
IMPORTED_BENCHMARK_SCHEMA_VERSION = "1"
MAX_IMPORTED_ARTIFACT_BYTES = 4 * 1024 * 1024

IMPORTED_DERIVED = "imported_derived"
IMPORTED_REPORTED = "imported_reported"
LOWER_IS_BETTER = "lower_is_better"
HIGHER_IS_BETTER = "higher_is_better"

VLLM_BENCH_TOOL = "vllm-bench-serve"
AIPERF_TOOL = "aiperf"
SUPPORTED_TOOLS = (VLLM_BENCH_TOOL, AIPERF_TOOL)

TOTAL_NAMES = (
    "completed_requests",
    "failed_requests",
    "total_input_tokens",
    "total_output_tokens",
    "duration",
)

_STAT_NAME = re.compile(r"^[a-z][a-z0-9_.]{0,31}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# Metric direction is fixed per tool and metric. The gate reads it from here,
# never from the artifact, and the reader refuses artifacts that disagree.
METRIC_DIRECTIONS: dict[str, dict[str, str]] = {
    VLLM_BENCH_TOOL: {
        "ttft_ms": LOWER_IS_BETTER,
        "tpot_ms": LOWER_IS_BETTER,
        "itl_ms": LOWER_IS_BETTER,
        "e2el_ms": LOWER_IS_BETTER,
        "client_queue_time_ms": LOWER_IS_BETTER,
        "e2el_including_client_queue_ms": LOWER_IS_BETTER,
        "request_throughput": HIGHER_IS_BETTER,
        "request_goodput": HIGHER_IS_BETTER,
        "output_throughput": HIGHER_IS_BETTER,
        "total_token_throughput": HIGHER_IS_BETTER,
    },
    AIPERF_TOOL: {
        "time_to_first_token": LOWER_IS_BETTER,
        "time_to_second_token": LOWER_IS_BETTER,
        "time_to_first_output_token": LOWER_IS_BETTER,
        "inter_token_latency": LOWER_IS_BETTER,
        "inter_chunk_latency": LOWER_IS_BETTER,
        "request_latency": LOWER_IS_BETTER,
        "request_throughput": HIGHER_IS_BETTER,
        "output_token_throughput": HIGHER_IS_BETTER,
        "output_token_throughput_per_user": HIGHER_IS_BETTER,
        "prefill_throughput": HIGHER_IS_BETTER,
        "goodput": HIGHER_IS_BETTER,
    },
}


class BenchmarkImportError(ValueError):
    """A benchmark file was refused: malformed, unsupported or mixed runs."""


class ImportedArtifactError(ValueError):
    """An imported benchmark artifact is unreadable or invalid."""


@dataclass(frozen=True)
class ImportedValue:
    value: float
    source_key: str


@dataclass(frozen=True)
class ImportedTotal:
    value: float
    unit: str
    source_key: str
    provenance: str = IMPORTED_REPORTED


@dataclass(frozen=True)
class ImportedMetric:
    unit: str
    direction: str
    stats: Mapping[str, ImportedValue]
    provenance: str = IMPORTED_DERIVED
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ImportedSource:
    tool: str
    sha256: str
    filename: str
    tool_version: str | None = None
    schema_version: str | None = None
    user_metadata: Mapping[str, str] = field(default_factory=dict)


LoadValue = str | int | float | None
EndpointValue = str | bool | None


@dataclass(frozen=True)
class ImportedBenchmark:
    source: ImportedSource
    model_id: str
    endpoint: Mapping[str, EndpointValue]
    load: Mapping[str, LoadValue]
    totals: Mapping[str, ImportedTotal]
    metrics: Mapping[str, ImportedMetric]
    run: Mapping[str, str | None] = field(default_factory=dict)
    load_unresolved: tuple[str, ...] = ()
    unmapped_keys: tuple[str, ...] = ()
    importer_version: str | None = None

    @property
    def run_id(self) -> str:
        return f"{self.source.filename}#{self.source.sha256[:12]}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": IMPORTED_BENCHMARK_KIND,
            "schema_version": IMPORTED_BENCHMARK_SCHEMA_VERSION,
            "importer": {"name": "llmtracefx", "version": self.importer_version},
            "source": {
                "tool": self.source.tool,
                "tool_version": self.source.tool_version,
                "schema_version": self.source.schema_version,
                "sha256": self.source.sha256,
                "filename": self.source.filename,
                "user_metadata": dict(sorted(self.source.user_metadata.items())),
            },
            "run": dict(self.run),
            "model_id": self.model_id,
            "endpoint": dict(self.endpoint),
            "load": dict(self.load),
            "load_unresolved": list(self.load_unresolved),
            "totals": {
                name: {
                    "value": total.value,
                    "unit": total.unit,
                    "provenance": total.provenance,
                    "source_key": total.source_key,
                }
                for name, total in self.totals.items()
            },
            "metrics": {
                name: {
                    "unit": metric.unit,
                    "direction": metric.direction,
                    "provenance": metric.provenance,
                    "stats": {
                        stat: {"value": item.value, "source_key": item.source_key}
                        for stat, item in metric.stats.items()
                    },
                    "notes": list(metric.notes),
                }
                for name, metric in self.metrics.items()
            },
            "unmapped_keys": list(self.unmapped_keys),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_dict(cls, data: Any) -> ImportedBenchmark:
        return _parse_artifact(data)


def finite_number(value: Any) -> float | None:
    """A finite JSON number as float; ``None`` for bools, strings and NaN."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def is_imported_benchmark(data: Any) -> bool:
    return isinstance(data, dict) and data.get("kind") == IMPORTED_BENCHMARK_KIND


def parse_json_document(text: str, label: str) -> Any:
    """Parse exactly one strict JSON document, refusing NaN and concatenations."""
    decoder = json.JSONDecoder(parse_constant=reject_non_finite_json_constant)
    stripped = text.lstrip()
    try:
        value, end = decoder.raw_decode(stripped)
    except ValueError as exc:
        raise BenchmarkImportError(f"{label} is not valid JSON: {exc}") from exc
    except RecursionError as exc:
        raise BenchmarkImportError(f"{label} is nested too deeply") from exc
    if stripped[end:].strip():
        raise BenchmarkImportError(
            f"{label} holds more than one JSON document (for example a "
            "`vllm bench serve --append-result` file); results from several "
            "runs are not mixed, so split it into one file per run"
        )
    return value


def read_imported_benchmark(path: str | Path) -> ImportedBenchmark:
    try:
        text = read_bounded_regular_text(path, MAX_IMPORTED_ARTIFACT_BYTES)
    except (OSError, ArtifactReadError) as exc:
        raise ImportedArtifactError(f"{path}: {exc}") from exc
    try:
        data = parse_json_document(text, str(path))
    except BenchmarkImportError as exc:
        raise ImportedArtifactError(str(exc)) from exc
    try:
        return _parse_artifact(data)
    except ImportedArtifactError as exc:
        raise ImportedArtifactError(f"{path}: {exc}") from exc


def _fail(message: str) -> ImportedArtifactError:
    return ImportedArtifactError(message)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _fail(f"{name} must be an object")
    return value


def _optional_str(value: Any, name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise _fail(f"{name} must be a string or null")
    return value


def _required_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise _fail(f"{name} must be a non-empty string")
    return value


def _string_list(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise _fail(f"{name} must be a list of strings")
    return tuple(value)


def _load_value(value: Any, name: str) -> LoadValue:
    if value is None or isinstance(value, str):
        return value
    if finite_number(value) is None:
        raise _fail(f"{name} must be a finite number, string or null")
    return int(value) if isinstance(value, int) else float(value)


def _parse_source(data: Any) -> ImportedSource:
    source = _mapping(data, "source")
    tool = source.get("tool")
    if tool not in SUPPORTED_TOOLS:
        raise _fail(f"unsupported source tool {tool!r}")
    sha256 = source.get("sha256")
    if not isinstance(sha256, str) or not _SHA256.fullmatch(sha256):
        raise _fail("source.sha256 must be a lowercase hex sha256")
    filename = _required_str(source.get("filename"), "source.filename")
    if "/" in filename or "\\" in filename:
        raise _fail("source.filename must be a file name, not a path")
    user_metadata = _mapping(source.get("user_metadata", {}), "source.user_metadata")
    if not all(isinstance(value, str) for value in user_metadata.values()):
        raise _fail("source.user_metadata values must be strings")
    return ImportedSource(
        tool=tool,
        sha256=sha256,
        filename=filename,
        tool_version=_optional_str(source.get("tool_version"), "tool_version"),
        schema_version=_optional_str(
            source.get("schema_version"), "source.schema_version"
        ),
        user_metadata=dict(user_metadata),
    )


def _parse_totals(data: Any) -> dict[str, ImportedTotal]:
    totals: dict[str, ImportedTotal] = {}
    for name, item in _mapping(data, "totals").items():
        if name not in TOTAL_NAMES:
            raise _fail(f"unknown total {name!r}")
        entry = _mapping(item, f"totals.{name}")
        value = finite_number(entry.get("value"))
        if value is None or value < 0:
            raise _fail(f"totals.{name}.value must be a finite number >= 0")
        if entry.get("provenance") != IMPORTED_REPORTED:
            raise _fail(f"totals.{name}.provenance must be {IMPORTED_REPORTED!r}")
        totals[name] = ImportedTotal(
            value=value,
            unit=_required_str(entry.get("unit"), f"totals.{name}.unit"),
            source_key=_required_str(
                entry.get("source_key"), f"totals.{name}.source_key"
            ),
        )
    if "completed_requests" not in totals:
        raise _fail("totals.completed_requests is required")
    return totals


def _parse_metrics(tool: str, data: Any) -> dict[str, ImportedMetric]:
    directions = METRIC_DIRECTIONS[tool]
    metrics: dict[str, ImportedMetric] = {}
    for name, item in _mapping(data, "metrics").items():
        if name not in directions:
            raise _fail(f"unknown {tool} metric {name!r}")
        entry = _mapping(item, f"metrics.{name}")
        if entry.get("direction") != directions[name]:
            raise _fail(f"metrics.{name}.direction must be {directions[name]!r}")
        if entry.get("provenance") != IMPORTED_DERIVED:
            raise _fail(f"metrics.{name}.provenance must be {IMPORTED_DERIVED!r}")
        stats: dict[str, ImportedValue] = {}
        for stat, stat_item in _mapping(entry.get("stats"), f"{name}.stats").items():
            if not _STAT_NAME.fullmatch(stat):
                raise _fail(f"metrics.{name} has a bad stat name {stat!r}")
            stat_entry = _mapping(stat_item, f"metrics.{name}.stats.{stat}")
            value = finite_number(stat_entry.get("value"))
            if value is None:
                raise _fail(f"metrics.{name}.stats.{stat}.value must be finite")
            stats[stat] = ImportedValue(
                value=value,
                source_key=_required_str(
                    stat_entry.get("source_key"),
                    f"metrics.{name}.stats.{stat}.source_key",
                ),
            )
        if not stats:
            raise _fail(f"metrics.{name} has no stats")
        metrics[name] = ImportedMetric(
            unit=_required_str(entry.get("unit"), f"metrics.{name}.unit"),
            direction=directions[name],
            stats=stats,
            notes=_string_list(entry.get("notes", []), f"metrics.{name}.notes"),
        )
    return metrics


def _parse_artifact(data: Any) -> ImportedBenchmark:
    data = _mapping(data, "artifact")
    if data.get("kind") != IMPORTED_BENCHMARK_KIND:
        raise _fail(f"kind must be {IMPORTED_BENCHMARK_KIND!r}")
    if data.get("schema_version") != IMPORTED_BENCHMARK_SCHEMA_VERSION:
        raise _fail(f"unsupported schema_version {data.get('schema_version')!r}")
    importer = _mapping(data.get("importer"), "importer")
    source = _parse_source(data.get("source"))
    run = {
        key: _optional_str(value, f"run.{key}")
        for key, value in _mapping(data.get("run", {}), "run").items()
    }
    endpoint: dict[str, EndpointValue] = {}
    for key, value in _mapping(data.get("endpoint"), "endpoint").items():
        if value is not None and not isinstance(value, (str, bool)):
            raise _fail(f"endpoint.{key} must be a string, boolean or null")
        endpoint[key] = value
    load = {
        key: _load_value(value, f"load.{key}")
        for key, value in _mapping(data.get("load"), "load").items()
    }
    return ImportedBenchmark(
        source=source,
        model_id=_required_str(data.get("model_id"), "model_id"),
        endpoint=endpoint,
        load=load,
        totals=_parse_totals(data.get("totals")),
        metrics=_parse_metrics(source.tool, data.get("metrics")),
        run=run,
        load_unresolved=_string_list(
            data.get("load_unresolved", []), "load_unresolved"
        ),
        unmapped_keys=_string_list(data.get("unmapped_keys", []), "unmapped_keys"),
        importer_version=_optional_str(importer.get("version"), "importer.version"),
    )
