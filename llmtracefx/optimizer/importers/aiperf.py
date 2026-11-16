"""Map an AIPerf ``profile_export_aiperf.json`` file to an imported benchmark.

Field names follow AIPerf's ``JsonExportData`` and ``JsonMetricResult``
(``src/aiperf/common/models/export_models.py``) and
``docs/reference/json-export-schema.md``. Each metric block carries a display
``unit`` and, depending on the metric type, ``avg``, ``min``, ``max``,
``p1`` to ``p99`` and ``std``. Files written before ``schema_version`` existed
are accepted; versions newer than the ones listed here are refused.

``input_config`` is read only from the documented paths of the two layouts
AIPerf has shipped: the older ``endpoint.model_names`` and ``loadgen`` layout
and the ``BenchmarkConfig`` layout with ``models.items`` and ``phases``.
"""

from __future__ import annotations

from typing import Any

from .artifact import (
    AIPERF_TOOL,
    METRIC_DIRECTIONS,
    BenchmarkImportError,
    EndpointValue,
    ImportedBenchmark,
    ImportedMetric,
    ImportedSource,
    ImportedTotal,
    ImportedValue,
    LoadValue,
    finite_number,
)

SUPPORTED_SCHEMA_VERSIONS = ("1.0", "1.1", "1.2", "1.3", "1.4", "1.5", "1.6", "1.7")
_STATS = (
    "avg",
    "min",
    "max",
    "p1",
    "p5",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "p95",
    "p99",
    "std",
)
_TOTALS = {
    "request_count": ("completed_requests", "requests"),
    "error_request_count": ("failed_requests", "requests"),
    "total_isl": ("total_input_tokens", "tokens"),
    "total_osl": ("total_output_tokens", "tokens"),
    "benchmark_duration": ("duration", None),
}
_RUN_KEYS = ("benchmark_id", "start_time", "end_time")
_ENVELOPE_KEYS = (
    "schema_version",
    "aiperf_version",
    "input_config",
    "was_cancelled",
    "is_complete",
    "incomplete_reason",
    "error_summary",
    *_RUN_KEYS,
)
_RATE_TYPES = frozenset({"poisson", "gamma", "constant", "user_centric"})
_COMMON_NOTES = (
    "Computed by AIPerf from its client-side timings.",
    "std is the sample standard deviation.",
)
_METRIC_NOTES = {
    "inter_token_latency": (
        "Per request (request latency - TTFT) / (output tokens - 1); vLLM bench "
        "calls this TPOT, not ITL.",
    ),
    "request_latency": (
        "AIPerf includes failed requests in request_latency percentiles.",
    ),
}


def looks_like_aiperf_profile(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and isinstance(data.get("input_config"), dict)
        and "timeslices" not in data
        and any(
            isinstance(data.get(name), dict) and "unit" in data[name]
            for name in ("request_count", "request_throughput", "request_latency")
        )
    )


def looks_like_other_aiperf_export(data: Any) -> bool:
    return isinstance(data, dict) and (
        "timeslices" in data
        or (
            ("aiperf_version" in data or "schema_version" in data)
            and "input_config" not in data
        )
    )


def _str(value: Any, name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise BenchmarkImportError(f"{name} must be a string")
    return value


def _block(data: dict[str, Any], key: str) -> dict[str, Any] | None:
    block = data.get(key)
    if block is None:
        return None
    if not isinstance(block, dict) or not isinstance(block.get("unit"), str):
        raise BenchmarkImportError(f"{key} must be a metric block with a unit")
    return block


def _scalar(data: dict[str, Any], key: str) -> tuple[float, str] | None:
    block = _block(data, key)
    if block is None:
        return None
    value = finite_number(block.get("avg"))
    if value is None or value < 0:
        raise BenchmarkImportError(f"{key}.avg must be a finite number >= 0")
    return value, block["unit"]


def _models(config: dict[str, Any]) -> list[str]:
    models = config.get("models")
    if isinstance(models, dict) and isinstance(models.get("items"), list):
        names = [item.get("name") for item in models["items"] if isinstance(item, dict)]
        if len(names) != len(models["items"]):
            raise BenchmarkImportError("input_config.models.items is malformed")
    elif isinstance(config.get("endpoint"), dict) and isinstance(
        config["endpoint"].get("model_names"), list
    ):
        names = list(config["endpoint"]["model_names"])
    else:
        raise BenchmarkImportError(
            "input_config names no model (expected models.items or "
            "endpoint.model_names)"
        )
    checked = [name for name in names if isinstance(name, str) and name]
    if len(checked) != len(names):
        raise BenchmarkImportError("input_config model names must be strings")
    return checked


def _endpoint(config: dict[str, Any]) -> dict[str, EndpointValue]:
    endpoint = config.get("endpoint")
    if not isinstance(endpoint, dict):
        return {"type": None, "streaming": None}
    result: dict[str, EndpointValue] = {"type": _str(endpoint.get("type"), "type")}
    streaming = endpoint.get("streaming")
    if streaming is not None and not isinstance(streaming, bool):
        raise BenchmarkImportError("input_config.endpoint.streaming must be a bool")
    result["streaming"] = streaming
    for key in ("path", "custom_endpoint"):
        if key in endpoint:
            result[key] = _str(endpoint[key], f"endpoint.{key}")
    return result


def _load_number(value: Any, name: str) -> LoadValue:
    if value is None or isinstance(value, str):
        return value
    number = finite_number(value)
    if number is None or number < 0:
        raise BenchmarkImportError(f"{name} must be a finite number >= 0")
    return int(value) if isinstance(value, int) else number


def _counts_toward_results(phase: dict[str, Any]) -> bool:
    if phase.get("exclude_from_results") is True:
        return False
    kind = phase.get("kind")
    if kind is not None:
        return bool(kind == "profiling")
    return phase.get("name") != "warmup"


def _phase_load(phases: Any) -> tuple[dict[str, LoadValue], list[str]]:
    if not isinstance(phases, list) or not all(isinstance(p, dict) for p in phases):
        raise BenchmarkImportError("input_config.phases must be a list of objects")
    included = [phase for phase in phases if _counts_toward_results(phase)]
    if len(included) != 1:
        raise BenchmarkImportError(
            f"input_config has {len(included)} phases that count toward results; "
            "the root export then mixes load settings, so only runs with exactly "
            "one profiling phase are imported"
        )
    phase = included[0]
    load: dict[str, LoadValue] = {"layout": "phases"}
    unresolved: list[str] = []
    for key in ("type", "concurrency", "rate", "requests", "duration", "sessions"):
        if key in phase:
            load[key] = _load_number(phase[key], f"phases.{key}")
    phase_type = load.get("type")
    if not isinstance(phase_type, str):
        unresolved.append("type")
    elif phase_type == "concurrency" and "concurrency" not in load:
        unresolved.append("concurrency")
    elif phase_type in _RATE_TYPES and "rate" not in load:
        unresolved.append("rate")
    stops = {"requests", "duration", "sessions"}
    if phase_type != "fixed_schedule" and not stops & set(load):
        unresolved.append("requests, duration or sessions")
    return load, unresolved


def _loadgen_load(loadgen: Any) -> tuple[dict[str, LoadValue], list[str]]:
    if not isinstance(loadgen, dict):
        raise BenchmarkImportError("input_config.loadgen must be an object")
    load: dict[str, LoadValue] = {"layout": "loadgen"}
    for key in (
        "concurrency",
        "request_rate",
        "request_count",
        "benchmark_duration",
        "arrival_pattern",
    ):
        if key in loadgen:
            load[key] = _load_number(loadgen[key], f"loadgen.{key}")
    unresolved: list[str] = []
    if not {"concurrency", "request_rate"} & set(load):
        unresolved.append("concurrency or request_rate")
    if not {"request_count", "benchmark_duration"} & set(load):
        unresolved.append("request_count or benchmark_duration")
    return load, unresolved


def _load(config: dict[str, Any]) -> tuple[dict[str, LoadValue], list[str]]:
    if "phases" in config:
        return _phase_load(config["phases"])
    if "loadgen" in config:
        return _loadgen_load(config["loadgen"])
    return {}, ["input_config.phases or input_config.loadgen"]


def _totals(data: dict[str, Any]) -> dict[str, ImportedTotal]:
    totals: dict[str, ImportedTotal] = {}
    for key, (name, unit) in _TOTALS.items():
        found = _scalar(data, key)
        if found is None:
            continue
        value, source_unit = found
        totals[name] = ImportedTotal(
            value=value, unit=unit or source_unit, source_key=f"{key}.avg"
        )
    if "failed_requests" not in totals:
        # AIPerf omits error_request_count when it has no observations. Zero is
        # stated only when the error summary is present and empty.
        if data.get("error_summary") == []:
            totals["failed_requests"] = ImportedTotal(
                value=0, unit="requests", source_key="error_summary"
            )
    return totals


def _metrics(data: dict[str, Any]) -> dict[str, ImportedMetric]:
    metrics: dict[str, ImportedMetric] = {}
    for name, direction in METRIC_DIRECTIONS[AIPERF_TOOL].items():
        block = _block(data, name)
        if block is None:
            continue
        stats: dict[str, ImportedValue] = {}
        for stat in _STATS:
            if block.get(stat) is None:
                continue  # AIPerf writes null when a statistic is not finite.
            value = finite_number(block[stat])
            if value is None or value < 0:
                raise BenchmarkImportError(
                    f"{name}.{stat} must be a finite number >= 0"
                )
            stats[stat] = ImportedValue(value, f"{name}.{stat}")
        if not stats:
            continue
        metrics[name] = ImportedMetric(
            unit=block["unit"],
            direction=direction,
            stats=stats,
            notes=_COMMON_NOTES + _METRIC_NOTES.get(name, ()),
        )
    return metrics


def import_aiperf(
    data: Any, *, sha256: str, filename: str, importer_version: str | None
) -> ImportedBenchmark:
    if not isinstance(data, dict):
        raise BenchmarkImportError("an AIPerf export must be a JSON object")
    if looks_like_other_aiperf_export(data) or not looks_like_aiperf_profile(data):
        raise BenchmarkImportError(
            "not an AIPerf profile_export_aiperf.json file (aggregate, collated, "
            "timeslice, outputs and server metrics exports are not supported)"
        )
    schema_version = _str(data.get("schema_version"), "schema_version")
    if schema_version is not None and schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise BenchmarkImportError(
            f"AIPerf schema_version {schema_version!r} is not supported "
            f"(supported: {', '.join(SUPPORTED_SCHEMA_VERSIONS)} or absent)"
        )
    if data.get("was_cancelled") is True:
        raise BenchmarkImportError("the AIPerf run was cancelled")
    if data.get("is_complete") is False:
        reason = data.get("incomplete_reason")
        raise BenchmarkImportError(
            "the AIPerf run is incomplete" + (f": {reason}" if reason else "")
        )

    config: dict[str, Any] = data["input_config"]
    models = _models(config)
    if len(models) != 1:
        raise BenchmarkImportError(
            f"the AIPerf run targets {len(models)} models; results that mix "
            "models are not imported"
        )
    totals = _totals(data)
    completed = totals.get("completed_requests")
    if completed is None or completed.value < 1:
        raise BenchmarkImportError("the AIPerf run has no successful requests")
    for total in totals.values():
        if total.unit == "requests" and not float(total.value).is_integer():
            raise BenchmarkImportError(f"{total.source_key} must be a whole number")
    duration = totals.get("duration")
    if duration is not None and duration.unit not in ("sec", "s"):
        raise BenchmarkImportError(
            f"benchmark_duration unit {duration.unit!r} is not seconds"
        )

    load, unresolved = _load(config)
    metrics = _metrics(data)
    consumed = {*_ENVELOPE_KEYS, *_TOTALS, *metrics}
    return ImportedBenchmark(
        source=ImportedSource(
            tool=AIPERF_TOOL,
            sha256=sha256,
            filename=filename,
            tool_version=_str(data.get("aiperf_version"), "aiperf_version"),
            schema_version=schema_version,
        ),
        model_id=models[0],
        endpoint=_endpoint(config),
        load=load,
        totals=totals,
        metrics=metrics,
        run={key: _str(data.get(key), key) for key in _RUN_KEYS},
        load_unresolved=tuple(unresolved),
        unmapped_keys=tuple(sorted(key for key in data if key not in consumed)),
        importer_version=importer_version,
    )
