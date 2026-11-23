"""Map AIPerf per-request records (``profile_export.jsonl``) to an imported benchmark.

AIPerf writes one ``MetricRecordInfo`` per line when run with
``--export-level records`` or ``raw`` (``src/aiperf/common/models/record_models.py``,
written by ``post_processors/record_export_jsonl_writer.py``). Each line has
``metadata``, ``metrics`` (tag to ``{"value", "unit"}`` in display units) and
``error`` (absent or null on success). The fields read here are the same from
AIPerf 0.2.0 to 0.14.0; later versions only added optional keys.

The records name no model, endpoint, load or AIPerf version, so the
``profile_export_aiperf.json`` from the same run is required. It goes through
the summary importer's checks, and its request count, error count and token
totals must equal what the records add up to.

LLMTraceFX computes the statistics itself over successful profiling records:
nearest-rank percentiles with distribution-free confidence intervals (the
method ``load sweep`` uses), throughput over the records' own window and,
when SLO targets are given, goodput. AIPerf's summary interpolates
percentiles and counts failed requests in ``request_latency``, so these
values are kept under their own tool and never compared with summary
imports.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .._artifact_io import reject_non_finite_json_constant
from ._distribution import nearest_rank_metric
from .artifact import (
    AIPERF_RECORDS_TOOL,
    DERIVED,
    IMPORTED_REPORTED,
    METRIC_DIRECTIONS,
    SLO_TARGETS,
    BenchmarkImportError,
    ImportedBenchmark,
    ImportedMetric,
    ImportedSource,
    ImportedTotal,
    ImportedValue,
    finite_number,
)

_DISTRIBUTIONS = {
    "time_to_first_token": "ms",
    "time_to_second_token": "ms",
    "time_to_first_output_token": "ms",
    "inter_token_latency": "ms",
    "request_latency": "ms",
    "output_token_throughput_per_user": "tokens/sec/user",
}
_TOKENS = {
    "input_sequence_length": "total_input_tokens",
    "output_sequence_length": "total_output_tokens",
}
_REQUIRED_METADATA: dict[str, type] = {
    "session_num": int,
    "request_start_ns": int,
    "request_end_ns": int,
    "worker_id": str,
    "record_processor_id": str,
    "benchmark_phase": str,
}
_PHASES = ("warmup", "profiling")
_SLO_METRICS = {
    "ttft_ms": "time_to_first_token",
    "tpot_ms": "inter_token_latency",
    "e2e_ms": "request_latency",
}
_WINDOW_KEY = "metadata.request_start_ns, metadata.request_end_ns"
_COUNT_KEY = "metadata.benchmark_phase, metadata.was_cancelled, error"
_COMMON_NOTES = (
    "Computed by LLMTraceFX from AIPerf's per-request values over successful "
    "profiling records; warmup and failed records are left out.",
    "Percentiles are nearest rank. AIPerf's own summary interpolates, so "
    "values can differ slightly from profile_export_aiperf.json.",
)
_METRIC_NOTES = {
    "inter_token_latency": (
        "Per request (request latency - TTFT) / (output tokens - 1); vLLM bench "
        "calls this TPOT, not ITL.",
    ),
    "request_latency": (
        "AIPerf's summary counts failed requests in request_latency; this "
        "import does not.",
    ),
}
_WINDOW_NOTE = (
    "Over the records' window: first profiling request_start_ns to last "
    "request_end_ns. AIPerf's benchmark_duration is measured differently."
)


@dataclass(frozen=True)
class _Record:
    line: int
    phase: str
    phase_id: tuple[Any, Any]
    request_id: str | None
    start_ns: int
    end_ns: int
    failed: bool
    values: Mapping[str, float]


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _parse_line(line: str, number: int) -> Any:
    try:
        return json.loads(
            line,
            parse_constant=reject_non_finite_json_constant,
            object_pairs_hook=_no_duplicate_keys,
        )
    except RecursionError as exc:
        raise BenchmarkImportError(f"line {number} is nested too deeply") from exc
    except ValueError as exc:
        raise BenchmarkImportError(f"line {number} is not valid JSON: {exc}") from exc


def looks_like_aiperf_record(data: Any) -> bool:
    metadata = data.get("metadata") if isinstance(data, dict) else None
    return (
        isinstance(metadata, dict)
        and "request_start_ns" in metadata
        and "benchmark_phase" in metadata
    )


def looks_like_aiperf_records(text: str) -> bool:
    """Whether the first non-empty line is an AIPerf per-request record."""
    for line in text.splitlines():
        if line.strip():
            try:
                return looks_like_aiperf_record(_parse_line(line, 1))
            except BenchmarkImportError:
                return False
    return False


def _metadata(metadata: dict[str, Any], number: int) -> None:
    for key, kind in _REQUIRED_METADATA.items():
        value = metadata.get(key)
        if isinstance(value, bool) or not isinstance(value, kind):
            article = "an integer" if kind is int else "a string"
            raise BenchmarkImportError(
                f"line {number}: metadata.{key} is missing or not {article}"
            )


def _values(raw: Any, number: int, unmapped: set[str]) -> dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise BenchmarkImportError(f"line {number}: metrics must be an object")
    values: dict[str, float] = {}
    for tag, item in raw.items():
        if not isinstance(item, dict) or "value" not in item:
            raise BenchmarkImportError(
                f"line {number}: metrics.{tag} must be an object with a value"
            )
        if tag not in _DISTRIBUTIONS and tag not in _TOKENS:
            unmapped.add(f"metrics.{tag}")
            continue
        unit = _DISTRIBUTIONS.get(tag, "tokens")
        if item.get("unit") != unit:
            raise BenchmarkImportError(
                f"line {number}: metrics.{tag} unit {item.get('unit')!r} is not {unit!r}"
            )
        value = finite_number(item["value"])
        if value is None or value < 0:
            raise BenchmarkImportError(
                f"line {number}: metrics.{tag}.value must be a finite number >= 0"
            )
        values[tag] = value
    return values


def _record(data: Any, number: int, unmapped: set[str]) -> _Record:
    if not isinstance(data, dict) or not isinstance(data.get("metadata"), dict):
        raise BenchmarkImportError(
            f"line {number} is not an AIPerf record (no metadata object)"
        )
    metadata: dict[str, Any] = data["metadata"]
    _metadata(metadata, number)
    phase = metadata["benchmark_phase"]
    if phase not in _PHASES:
        raise BenchmarkImportError(
            f"line {number}: benchmark_phase {phase!r} is not one of {', '.join(_PHASES)}"
        )
    start, end = metadata["request_start_ns"], metadata["request_end_ns"]
    if start < 0 or end < start:
        raise BenchmarkImportError(
            f"line {number}: request_end_ns must not be before request_start_ns"
        )
    cancelled = metadata.get("was_cancelled", False)
    if not isinstance(cancelled, bool):
        raise BenchmarkImportError(
            f"line {number}: metadata.was_cancelled must be a bool"
        )
    error = data.get("error")
    if error is not None and not isinstance(error, dict):
        raise BenchmarkImportError(f"line {number}: error must be an object or null")
    values = _values(data.get("metrics"), number, unmapped)
    failed = error is not None or cancelled
    if not failed and "request_latency" not in values:
        raise BenchmarkImportError(
            f"line {number}: a successful record has no request_latency"
        )
    unmapped.update(key for key in data if key not in ("metadata", "metrics", "error"))
    request_id = metadata.get("x_request_id")
    return _Record(
        line=number,
        phase=phase,
        phase_id=(metadata.get("profiling_index"), metadata.get("phase_name")),
        request_id=request_id if isinstance(request_id, str) else None,
        start_ns=start,
        end_ns=end,
        failed=failed,
        values=values,
    )


def parse_records(text: str) -> tuple[list[_Record], set[str]]:
    records: list[_Record] = []
    unmapped: set[str] = set()
    seen: dict[str, int] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        record = _record(_parse_line(line, number), number, unmapped)
        if record.request_id is not None:
            if record.request_id in seen:
                raise BenchmarkImportError(
                    f"line {number} repeats x_request_id from line "
                    f"{seen[record.request_id]}; records from several runs are "
                    "not mixed"
                )
            seen[record.request_id] = number
        records.append(record)
    return records, unmapped


def _total_tokens(records: list[_Record], tag: str) -> int | None:
    values = [record.values.get(tag) for record in records]
    if any(value is None for value in values):
        return None
    return int(math.fsum(value for value in values if value is not None))


def _check_summary(
    summary: ImportedBenchmark, counted: dict[str, float | None]
) -> None:
    for name, value in counted.items():
        reported = summary.totals.get(name)
        if value is None or reported is None or reported.value == value:
            continue
        raise BenchmarkImportError(
            f"the records add up to {name} {value:g} but the summary reports "
            f"{reported.source_key} {reported.value:g}; the two files are not "
            "from the same run"
        )


def _distribution(tag: str, values: list[float]) -> ImportedMetric:
    return nearest_rank_metric(
        values,
        source_key=f"metrics.{tag}.value",
        unit=_DISTRIBUTIONS[tag],
        direction=METRIC_DIRECTIONS[AIPERF_RECORDS_TOOL][tag],
        sample_provenance=IMPORTED_REPORTED,
        mean_stat="avg",
        notes=_COMMON_NOTES + _METRIC_NOTES.get(tag, ()),
    )


def _rate(
    name: str, unit: str, value: float, key: str, notes: tuple[str, ...]
) -> ImportedMetric:
    return ImportedMetric(
        unit=unit,
        direction=METRIC_DIRECTIONS[AIPERF_RECORDS_TOOL][name],
        stats={"value": ImportedValue(value, key)},
        provenance=DERIVED,
        notes=notes,
    )


def check_slo(slo: Mapping[str, float]) -> dict[str, float]:
    checked: dict[str, float] = {}
    for name, value in slo.items():
        number = finite_number(value)
        if name not in SLO_TARGETS:
            raise BenchmarkImportError(f"unknown SLO target {name!r}")
        if number is None or number <= 0:
            raise BenchmarkImportError(
                f"SLO target {name} must be a finite number above zero"
            )
        checked[name] = number
    return checked


def _goodput_met(profiling: list[_Record], slo: Mapping[str, float]) -> int:
    """Successful records that meet every target (``<=``); refuses unverifiable ones."""
    met = 0
    unverifiable: list[tuple[int, str]] = []
    for record in profiling:
        if record.failed:
            continue
        missed = False
        missing = []
        for target, limit in slo.items():
            value = record.values.get(_SLO_METRICS[target])
            output_tokens = record.values.get("output_sequence_length")
            if value is None:
                # One output token has no inter-token gap, so TPOT does not apply.
                if not (
                    target == "tpot_ms"
                    and output_tokens is not None
                    and output_tokens <= 1
                ):
                    missing.append(target)
            elif value > limit:
                missed = True
        if missed:
            continue
        if missing:
            unverifiable.append((record.line, missing[0]))
        else:
            met += 1
    if unverifiable:
        line, target = unverifiable[0]
        raise BenchmarkImportError(
            f"{len(unverifiable)} successful profiling record(s) have no "
            f"{_SLO_METRICS[target]} for the {target} target (first on line "
            f"{line}); goodput is not computed from incomplete evidence. TTFT "
            "needs a streaming run; drop the target otherwise"
        )
    return met


def import_aiperf_records(
    text: str,
    *,
    summary: ImportedBenchmark,
    sha256: str,
    filename: str,
    slo: Mapping[str, float] | None = None,
    importer_version: str | None,
) -> ImportedBenchmark:
    slo = check_slo(slo or {})
    records, unmapped = parse_records(text)
    profiling = [record for record in records if record.phase == "profiling"]
    if not profiling:
        raise BenchmarkImportError(
            "no profiling records (AIPerf writes them with --export-level records)"
        )
    phases = {record.phase_id for record in profiling}
    if len(phases) != 1:
        raise BenchmarkImportError(
            f"the records span {len(phases)} profiling phases; only runs with "
            "exactly one profiling phase are imported"
        )
    ok = [record for record in profiling if not record.failed]
    if not ok:
        raise BenchmarkImportError("the records hold no successful profiling request")
    failed = len(profiling) - len(ok)
    input_tokens = _total_tokens(ok, "input_sequence_length")
    output_tokens = _total_tokens(ok, "output_sequence_length")
    _check_summary(
        summary,
        {
            "completed_requests": len(ok),
            "failed_requests": failed,
            "total_input_tokens": input_tokens,
            "total_output_tokens": output_tokens,
        },
    )

    totals = {
        "completed_requests": ImportedTotal(len(ok), "requests", _COUNT_KEY, DERIVED),
        "failed_requests": ImportedTotal(failed, "requests", _COUNT_KEY, DERIVED),
    }
    for tag, name in _TOKENS.items():
        tokens = input_tokens if tag == "input_sequence_length" else output_tokens
        if tokens is not None:
            totals[name] = ImportedTotal(
                tokens, "tokens", f"metrics.{tag}.value", DERIVED
            )
    window = (
        max(record.end_ns for record in profiling)
        - min(record.start_ns for record in profiling)
    ) / 1e9

    metrics: dict[str, ImportedMetric] = {}
    for tag in _DISTRIBUTIONS:
        values = [record.values[tag] for record in ok if tag in record.values]
        if values:
            metrics[tag] = _distribution(tag, values)
    if window > 0:
        totals["duration"] = ImportedTotal(window, "sec", _WINDOW_KEY, DERIVED)
        metrics["request_throughput"] = _rate(
            "request_throughput",
            "requests/sec",
            len(ok) / window,
            _WINDOW_KEY,
            ("Successful profiling requests per second.", _WINDOW_NOTE),
        )
        if output_tokens is not None:
            metrics["output_token_throughput"] = _rate(
                "output_token_throughput",
                "tokens/sec",
                output_tokens / window,
                "metrics.output_sequence_length.value",
                ("Output tokens of successful requests per second.", _WINDOW_NOTE),
            )
    if slo:
        met = _goodput_met(profiling, slo)
        stated = ", ".join(f"{name} <= {value:g}" for name, value in slo.items())
        slo_keys = ", ".join(f"metrics.{_SLO_METRICS[name]}.value" for name in slo)
        if window > 0:
            metrics["goodput"] = _rate(
                "goodput",
                "requests/sec",
                met / window,
                slo_keys,
                (
                    f"Requests that met every SLO target ({stated}) per second.",
                    _WINDOW_NOTE,
                ),
            )
        metrics["goodput_fraction"] = _rate(
            "goodput_fraction",
            "ratio",
            met / len(profiling),
            slo_keys,
            (
                f"Share of profiling requests that met every SLO target ({stated}).",
                "Failed requests stay in the denominator and count as missed.",
            ),
        )

    return ImportedBenchmark(
        source=ImportedSource(
            tool=AIPERF_RECORDS_TOOL,
            sha256=sha256,
            filename=filename,
            tool_version=summary.source.tool_version,
            schema_version=summary.source.schema_version,
            companion_filename=summary.source.filename,
            companion_sha256=summary.source.sha256,
        ),
        model_id=summary.model_id,
        endpoint=summary.endpoint,
        load=summary.load,
        totals=totals,
        metrics=metrics,
        run=summary.run,
        load_unresolved=summary.load_unresolved,
        unmapped_keys=tuple(sorted(unmapped)),
        importer_version=importer_version,
        slo=slo,
    )
