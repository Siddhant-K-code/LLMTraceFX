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

The ``aiperf-records`` tool is different: LLMTraceFX reads AIPerf's
per-request records and computes the statistics itself. Its metrics and
totals are ``derived``, each distribution states its ``sample_count`` and
``sample_provenance`` (``imported_reported``, the per-request values AIPerf
wrote) and carries order statistic confidence intervals. ``source.companion``
names the AIPerf summary file that supplied the model, endpoint and load.

The optional ``speculative`` section holds the speculative decoding
acceptance a tool read from the server's own counters (``vllm bench serve``
does this). Those values are server side and cover every request the server
handled while the benchmark ran, so they never enter ``metrics`` and the
gate does not compare them.

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
DERIVED = "derived"
NEAREST_RANK = "nearest_rank"
LOWER_IS_BETTER = "lower_is_better"
HIGHER_IS_BETTER = "higher_is_better"

VLLM_BENCH_TOOL = "vllm-bench-serve"
AIPERF_TOOL = "aiperf"
AIPERF_RECORDS_TOOL = "aiperf-records"
SUPPORTED_TOOLS = (VLLM_BENCH_TOOL, AIPERF_TOOL, AIPERF_RECORDS_TOOL)
SLO_TARGETS = ("ttft_ms", "tpot_ms", "e2e_ms")

TOTAL_NAMES = (
    "completed_requests",
    "failed_requests",
    "total_input_tokens",
    "total_output_tokens",
    "duration",
)

SPECULATIVE_REPORTED = "reported"
SPECULATIVE_NOT_REPORTED = "not_reported"

# name -> (unit, provenance). Fixed here so the reader refuses relabelled
# values.
SPECULATIVE_VALUES: dict[str, tuple[str, str]] = {
    "acceptance_rate": ("percent", IMPORTED_DERIVED),
    "acceptance_length": ("tokens/step", IMPORTED_DERIVED),
    "drafts": ("count", IMPORTED_REPORTED),
    "draft_tokens": ("tokens", IMPORTED_REPORTED),
    "accepted_tokens": ("tokens", IMPORTED_REPORTED),
    "verified_tokens": ("tokens", IMPORTED_REPORTED),
}

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
    AIPERF_RECORDS_TOOL: {
        "time_to_first_token": LOWER_IS_BETTER,
        "time_to_second_token": LOWER_IS_BETTER,
        "time_to_first_output_token": LOWER_IS_BETTER,
        "inter_token_latency": LOWER_IS_BETTER,
        "request_latency": LOWER_IS_BETTER,
        "output_token_throughput_per_user": HIGHER_IS_BETTER,
        "request_throughput": HIGHER_IS_BETTER,
        "output_token_throughput": HIGHER_IS_BETTER,
        "goodput": HIGHER_IS_BETTER,
        "goodput_fraction": HIGHER_IS_BETTER,
    },
}

# Statistics LLMTraceFX computes itself are ``derived``; everything else in
# an imported artifact was computed or reported by the source tool.
_METRIC_PROVENANCE = {AIPERF_RECORDS_TOOL: DERIVED}
_TOTAL_PROVENANCE = {AIPERF_RECORDS_TOOL: DERIVED}


def metric_provenance(tool: str) -> str:
    return _METRIC_PROVENANCE.get(tool, IMPORTED_DERIVED)


def total_provenance(tool: str) -> str:
    return _TOTAL_PROVENANCE.get(tool, IMPORTED_REPORTED)


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
class ImportedInterval:
    """A distribution-free confidence interval for one percentile.

    ``lower`` and ``upper`` are order statistics at 1-based ranks; both are
    ``None`` with a ``withheld_reason`` when there are too few values.
    """

    lower: float | None
    upper: float | None
    lower_rank: int | None = None
    upper_rank: int | None = None
    coverage: float | None = None
    withheld_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.withheld_reason is not None:
            return {
                "lower": None,
                "upper": None,
                "withheld_reason": self.withheld_reason,
            }
        return {
            "lower": self.lower,
            "upper": self.upper,
            "lower_rank": self.lower_rank,
            "upper_rank": self.upper_rank,
            "coverage": self.coverage,
            "withheld_reason": None,
        }


@dataclass(frozen=True)
class ImportedMetric:
    unit: str
    direction: str
    stats: Mapping[str, ImportedValue]
    provenance: str = IMPORTED_DERIVED
    notes: tuple[str, ...] = ()
    sample_count: int | None = None
    """Per-request values the statistics were computed from (``aiperf-records``)."""
    sample_provenance: str | None = None
    statistic: str | None = None
    intervals: Mapping[str, ImportedInterval] = field(default_factory=dict)
    interval_level: float | None = None
    interval_method: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "unit": self.unit,
            "direction": self.direction,
            "provenance": self.provenance,
            "stats": {
                stat: {"value": item.value, "source_key": item.source_key}
                for stat, item in self.stats.items()
            },
        }
        if self.sample_count is not None:
            result["sample_count"] = self.sample_count
            result["sample_provenance"] = self.sample_provenance
            result["statistic"] = self.statistic
        if self.intervals:
            result["confidence_intervals"] = {
                "level": self.interval_level,
                "method": self.interval_method,
                "provenance": DERIVED,
                **{stat: item.to_dict() for stat, item in self.intervals.items()},
            }
        result["notes"] = list(self.notes)
        return result


@dataclass(frozen=True)
class ImportedSource:
    tool: str
    sha256: str
    filename: str
    tool_version: str | None = None
    schema_version: str | None = None
    user_metadata: Mapping[str, str] = field(default_factory=dict)
    companion_filename: str | None = None
    """The summary file read alongside the source (``aiperf-records``)."""
    companion_sha256: str | None = None


@dataclass(frozen=True)
class ImportedSpeculative:
    """Speculative decoding acceptance the source tool scraped from the server."""

    status: str
    """``reported`` or ``not_reported``."""
    reason: str
    values: Mapping[str, ImportedTotal] = field(default_factory=dict)
    per_position_acceptance_rate: tuple[float, ...] = ()
    """Accepted drafts at each position / drafts, as fractions; index is the
    draft position."""
    per_position_source_key: str | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "reason": self.reason,
            "values": {
                name: {
                    "value": item.value,
                    "unit": item.unit,
                    "provenance": item.provenance,
                    "source_key": item.source_key,
                }
                for name, item in self.values.items()
            },
            "per_position_acceptance_rate": [
                {
                    "position": position,
                    "value": value,
                    "unit": "ratio",
                    "provenance": IMPORTED_DERIVED,
                    "source_key": f"{self.per_position_source_key}[{position}]",
                }
                for position, value in enumerate(self.per_position_acceptance_rate)
            ],
            "notes": list(self.notes),
        }


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
    speculative: ImportedSpeculative | None = None
    slo: Mapping[str, float] = field(default_factory=dict)
    """SLO targets in ms that ``goodput`` was computed against (``<=``)."""

    @property
    def run_id(self) -> str:
        return f"{self.source.filename}#{self.source.sha256[:12]}"

    def to_dict(self) -> dict[str, Any]:
        source: dict[str, Any] = {
            "tool": self.source.tool,
            "tool_version": self.source.tool_version,
            "schema_version": self.source.schema_version,
            "sha256": self.source.sha256,
            "filename": self.source.filename,
            "user_metadata": dict(sorted(self.source.user_metadata.items())),
        }
        if self.source.companion_filename is not None:
            source["companion"] = {
                "filename": self.source.companion_filename,
                "sha256": self.source.companion_sha256,
            }
        return {
            "kind": IMPORTED_BENCHMARK_KIND,
            "schema_version": IMPORTED_BENCHMARK_SCHEMA_VERSION,
            "importer": {"name": "llmtracefx", "version": self.importer_version},
            "source": source,
            "run": dict(self.run),
            "model_id": self.model_id,
            "endpoint": dict(self.endpoint),
            "load": dict(self.load),
            "load_unresolved": list(self.load_unresolved),
            **({"slo": {**self.slo, "comparison": "<="}} if self.slo else {}),
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
                name: metric.to_dict() for name, metric in self.metrics.items()
            },
            **(
                {}
                if self.speculative is None
                else {"speculative": self.speculative.to_dict()}
            ),
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
    companion_filename: str | None = None
    companion_sha256: str | None = None
    if "companion" in source:
        if tool != AIPERF_RECORDS_TOOL:
            raise _fail(f"source.companion is not used by {tool} imports")
        companion = _mapping(source["companion"], "source.companion")
        companion_filename = _required_str(
            companion.get("filename"), "source.companion.filename"
        )
        if "/" in companion_filename or "\\" in companion_filename:
            raise _fail("source.companion.filename must be a file name, not a path")
        companion_sha256 = companion.get("sha256")
        if not isinstance(companion_sha256, str) or not _SHA256.fullmatch(
            companion_sha256
        ):
            raise _fail("source.companion.sha256 must be a lowercase hex sha256")
    elif tool == AIPERF_RECORDS_TOOL:
        raise _fail("source.companion is required for aiperf-records imports")
    return ImportedSource(
        tool=tool,
        sha256=sha256,
        filename=filename,
        tool_version=_optional_str(source.get("tool_version"), "tool_version"),
        schema_version=_optional_str(
            source.get("schema_version"), "source.schema_version"
        ),
        user_metadata=dict(user_metadata),
        companion_filename=companion_filename,
        companion_sha256=companion_sha256,
    )


def _parse_totals(tool: str, data: Any) -> dict[str, ImportedTotal]:
    provenance = total_provenance(tool)
    totals: dict[str, ImportedTotal] = {}
    for name, item in _mapping(data, "totals").items():
        if name not in TOTAL_NAMES:
            raise _fail(f"unknown total {name!r}")
        entry = _mapping(item, f"totals.{name}")
        value = finite_number(entry.get("value"))
        if value is None or value < 0:
            raise _fail(f"totals.{name}.value must be a finite number >= 0")
        if entry.get("provenance") != provenance:
            raise _fail(f"totals.{name}.provenance must be {provenance!r}")
        totals[name] = ImportedTotal(
            value=value,
            unit=_required_str(entry.get("unit"), f"totals.{name}.unit"),
            source_key=_required_str(
                entry.get("source_key"), f"totals.{name}.source_key"
            ),
            provenance=provenance,
        )
    if "completed_requests" not in totals:
        raise _fail("totals.completed_requests is required")
    return totals


def _optional_rank(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _fail(f"{name} must be an integer >= 1")
    return value


def _parse_interval(data: Any, name: str, count: int) -> ImportedInterval:
    entry = _mapping(data, name)
    reason = _optional_str(entry.get("withheld_reason"), f"{name}.withheld_reason")
    if reason is not None:
        if entry.get("lower") is not None or entry.get("upper") is not None:
            raise _fail(f"{name} is withheld and must hold no bounds")
        return ImportedInterval(lower=None, upper=None, withheld_reason=reason)
    lower = finite_number(entry.get("lower"))
    upper = finite_number(entry.get("upper"))
    coverage = finite_number(entry.get("coverage"))
    if lower is None or upper is None or lower > upper:
        raise _fail(f"{name} needs finite lower <= upper or a withheld_reason")
    if coverage is None or not 0 < coverage <= 1:
        raise _fail(f"{name}.coverage must be in (0, 1]")
    lower_rank = _optional_rank(entry.get("lower_rank"), f"{name}.lower_rank")
    upper_rank = _optional_rank(entry.get("upper_rank"), f"{name}.upper_rank")
    if not lower_rank <= upper_rank <= count:
        raise _fail(f"{name} ranks must satisfy lower <= upper <= sample_count")
    return ImportedInterval(
        lower=lower,
        upper=upper,
        lower_rank=lower_rank,
        upper_rank=upper_rank,
        coverage=coverage,
    )


def _parse_samples(
    tool: str, name: str, entry: dict[str, Any], stats: Mapping[str, ImportedValue]
) -> dict[str, Any]:
    """The optional sample count and intervals; only ``aiperf-records`` has them."""
    has_samples = "sample_count" in entry or "confidence_intervals" in entry
    if not has_samples:
        return {}
    if tool != AIPERF_RECORDS_TOOL:
        raise _fail(f"metrics.{name} holds sample fields that {tool} imports lack")
    count = entry.get("sample_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise _fail(f"metrics.{name}.sample_count must be an integer >= 1")
    if entry.get("sample_provenance") != IMPORTED_REPORTED:
        raise _fail(f"metrics.{name}.sample_provenance must be {IMPORTED_REPORTED!r}")
    if entry.get("statistic") != NEAREST_RANK:
        raise _fail(f"metrics.{name}.statistic must be {NEAREST_RANK!r}")
    result: dict[str, Any] = {
        "sample_count": count,
        "sample_provenance": IMPORTED_REPORTED,
        "statistic": NEAREST_RANK,
    }
    if "confidence_intervals" not in entry:
        return result
    block = _mapping(entry["confidence_intervals"], f"metrics.{name}.ci")
    level = finite_number(block.get("level"))
    if level is None or not 0 < level < 1:
        raise _fail(f"metrics.{name}.confidence_intervals.level must be in (0, 1)")
    if block.get("provenance") != DERIVED:
        raise _fail(f"metrics.{name}.confidence_intervals.provenance must be derived")
    intervals: dict[str, ImportedInterval] = {}
    for stat, item in block.items():
        if stat in ("level", "method", "provenance"):
            continue
        if stat not in stats:
            raise _fail(f"metrics.{name} has an interval for missing stat {stat!r}")
        intervals[stat] = _parse_interval(
            item, f"metrics.{name}.confidence_intervals.{stat}", count
        )
    result["intervals"] = intervals
    result["interval_level"] = level
    result["interval_method"] = _required_str(
        block.get("method"), f"metrics.{name}.confidence_intervals.method"
    )
    return result


def _parse_metrics(tool: str, data: Any) -> dict[str, ImportedMetric]:
    directions = METRIC_DIRECTIONS[tool]
    provenance = metric_provenance(tool)
    metrics: dict[str, ImportedMetric] = {}
    for name, item in _mapping(data, "metrics").items():
        if name not in directions:
            raise _fail(f"unknown {tool} metric {name!r}")
        entry = _mapping(item, f"metrics.{name}")
        if entry.get("direction") != directions[name]:
            raise _fail(f"metrics.{name}.direction must be {directions[name]!r}")
        if entry.get("provenance") != provenance:
            raise _fail(f"metrics.{name}.provenance must be {provenance!r}")
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
            provenance=provenance,
            notes=_string_list(entry.get("notes", []), f"metrics.{name}.notes"),
            **_parse_samples(tool, name, entry, stats),
        )
    return metrics


def _parse_slo(tool: str, data: Any, metrics: Mapping[str, Any]) -> dict[str, float]:
    if data is None:
        if tool == AIPERF_RECORDS_TOOL and {"goodput", "goodput_fraction"} & set(
            metrics
        ):
            raise _fail("goodput metrics need the slo they were computed against")
        return {}
    if tool != AIPERF_RECORDS_TOOL:
        raise _fail(f"slo is not used by {tool} imports")
    entry = _mapping(data, "slo")
    if entry.get("comparison") != "<=":
        raise _fail("slo.comparison must be '<='")
    slo: dict[str, float] = {}
    for name, value in entry.items():
        if name == "comparison":
            continue
        number = finite_number(value)
        if name not in SLO_TARGETS or number is None or number <= 0:
            raise _fail(f"slo.{name} must be one of {', '.join(SLO_TARGETS)} > 0")
        slo[name] = number
    if not slo:
        raise _fail("slo states no target")
    return slo


def _parse_speculative(data: Any) -> ImportedSpeculative | None:
    if data is None:
        return None
    entry = _mapping(data, "speculative")
    status = entry.get("status")
    if status not in (SPECULATIVE_REPORTED, SPECULATIVE_NOT_REPORTED):
        raise _fail(f"speculative.status {status!r} is not recognised")
    values: dict[str, ImportedTotal] = {}
    for name, item in _mapping(entry.get("values", {}), "speculative.values").items():
        if name not in SPECULATIVE_VALUES:
            raise _fail(f"unknown speculative value {name!r}")
        unit, provenance = SPECULATIVE_VALUES[name]
        value_entry = _mapping(item, f"speculative.values.{name}")
        value = finite_number(value_entry.get("value"))
        if value is None or value < 0:
            raise _fail(f"speculative.values.{name}.value must be finite and >= 0")
        if value_entry.get("unit") != unit:
            raise _fail(f"speculative.values.{name}.unit must be {unit!r}")
        if value_entry.get("provenance") != provenance:
            raise _fail(f"speculative.values.{name}.provenance must be {provenance!r}")
        values[name] = ImportedTotal(
            value=value,
            unit=unit,
            source_key=_required_str(
                value_entry.get("source_key"), f"speculative.values.{name}.source_key"
            ),
            provenance=provenance,
        )
    rates: list[float] = []
    source_key: str | None = None
    raw_positions = entry.get("per_position_acceptance_rate", [])
    if not isinstance(raw_positions, list):
        raise _fail("speculative.per_position_acceptance_rate must be a list")
    for position, item in enumerate(raw_positions):
        name = f"speculative.per_position_acceptance_rate[{position}]"
        rate_entry = _mapping(item, name)
        value = finite_number(rate_entry.get("value"))
        if rate_entry.get("position") != position or value is None or value < 0:
            raise _fail(f"{name} must hold position {position} and a value >= 0")
        if (
            rate_entry.get("unit") != "ratio"
            or rate_entry.get("provenance") != IMPORTED_DERIVED
        ):
            raise _fail(f"{name} must be an {IMPORTED_DERIVED} ratio")
        key = _required_str(rate_entry.get("source_key"), f"{name}.source_key")
        prefix = key.removesuffix(f"[{position}]")
        if prefix == key or (source_key is not None and prefix != source_key):
            raise _fail(f"{name}.source_key {key!r} does not index one source list")
        source_key = prefix
        rates.append(value)
    if status == SPECULATIVE_NOT_REPORTED and (values or rates):
        raise _fail("a not_reported speculative section must hold no values")
    return ImportedSpeculative(
        status=status,
        reason=_required_str(entry.get("reason"), "speculative.reason"),
        values=values,
        per_position_acceptance_rate=tuple(rates),
        per_position_source_key=source_key,
        notes=_string_list(entry.get("notes", []), "speculative.notes"),
    )


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
    metrics = _parse_metrics(source.tool, data.get("metrics"))
    return ImportedBenchmark(
        source=source,
        model_id=_required_str(data.get("model_id"), "model_id"),
        endpoint=endpoint,
        load=load,
        totals=_parse_totals(source.tool, data.get("totals")),
        metrics=metrics,
        run=run,
        load_unresolved=_string_list(
            data.get("load_unresolved", []), "load_unresolved"
        ),
        unmapped_keys=_string_list(data.get("unmapped_keys", []), "unmapped_keys"),
        importer_version=_optional_str(importer.get("version"), "importer.version"),
        speculative=_parse_speculative(data.get("speculative")),
        slo=_parse_slo(source.tool, data.get("slo"), metrics),
    )
