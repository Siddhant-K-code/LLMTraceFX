"""Map a ``vllm bench serve --save-result`` JSON file to an imported benchmark.

Keys and units follow ``vllm/benchmarks/serve.py`` (``benchmark`` and
``main``): latency statistics are ``{mean,median,std,p<N>}_<metric>_ms`` in
milliseconds, throughputs are per second, ``duration`` is in seconds, and
``request_rate`` is a number or the string ``"inf"``. vLLM writes no tool or
schema version, so ``tool_version`` stays null.
"""

from __future__ import annotations

import re
from typing import Any

from .artifact import (
    METRIC_DIRECTIONS,
    VLLM_BENCH_TOOL,
    BenchmarkImportError,
    ImportedBenchmark,
    ImportedMetric,
    ImportedSource,
    ImportedTotal,
    ImportedValue,
    LoadValue,
    finite_number,
)

REQUIRED_KEYS = (
    "date",
    "model_id",
    "num_prompts",
    "duration",
    "completed",
    "request_throughput",
)

_LATENCY_METRICS = (
    "e2el_including_client_queue",
    "client_queue_time",
    "ttft",
    "tpot",
    "itl",
    "e2el",
)
_LATENCY_KEY = re.compile(
    r"^(?P<stat>mean|median|std|p\d+(?:\.\d+)?)_"
    rf"(?P<metric>{'|'.join(_LATENCY_METRICS)})_ms$"
)
_RATES = {
    "request_throughput": "requests/s",
    "request_goodput": "requests/s",
    "output_throughput": "tokens/s",
    "total_token_throughput": "tokens/s",
}
_TOTALS = {
    "completed": ("completed_requests", "requests"),
    "failed": ("failed_requests", "requests"),
    "total_input_tokens": ("total_input_tokens", "tokens"),
    "total_output_tokens": ("total_output_tokens", "tokens"),
    "duration": ("duration", "s"),
}
_LOAD_KEYS = ("request_rate", "burstiness", "max_concurrency")
_RAMP_KEYS = ("ramp_up_strategy", "ramp_up_start_rps", "ramp_up_end_rps")
_RUN_KEYS = ("date", "label", "tokenizer_id")
_ENDPOINT_KEYS = ("backend", "endpoint_type")
_KNOWN_STRING_KEYS = frozenset(
    {"model_id", "request_rate", *_RUN_KEYS, *_ENDPOINT_KEYS, *_RAMP_KEYS}
)
_MAX_USER_METADATA = 64
_MAX_USER_METADATA_CHARS = 256

_COMMON_NOTES = (
    "Computed by vllm bench serve over successful requests only.",
    "std is the population standard deviation (numpy std).",
)
_METRIC_NOTES = {
    "tpot": (
        "Per request (latency - TTFT) / (output tokens - 1), then summarised "
        "over requests.",
    ),
    "itl": ("Gaps between streamed chunks, pooled over all requests.",),
}


def looks_like_vllm_bench(data: Any) -> bool:
    return (
        isinstance(data, dict)
        and all(key in data for key in REQUIRED_KEYS)
        and ("backend" in data or "endpoint_type" in data)
    )


def _count(data: dict[str, Any], key: str) -> int:
    value = data[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or finite_number(value) is None
    ):
        raise BenchmarkImportError(f"{key} must be a non-negative integer")
    return value


def _non_negative(value: Any, key: str) -> float:
    number = finite_number(value)
    if number is None or number < 0:
        raise BenchmarkImportError(f"{key} must be a finite number >= 0")
    return number


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise BenchmarkImportError(f"{key} must be a string")
    return value


def _load(data: dict[str, Any]) -> tuple[dict[str, LoadValue], list[str]]:
    num_prompts = _count(data, "num_prompts")
    if num_prompts < 1:
        raise BenchmarkImportError("num_prompts must be at least 1")
    load: dict[str, LoadValue] = {"num_prompts": num_prompts}
    unresolved: list[str] = []
    for key in _LOAD_KEYS:
        if key not in data:
            unresolved.append(key)
            continue
        value = data[key]
        if key == "request_rate" and value == "inf":
            load[key] = "inf"
        elif key == "max_concurrency" and value is None:
            load[key] = None  # vLLM writes null when concurrency is unlimited.
        elif key == "max_concurrency":
            load[key] = _count(data, key)
        else:
            number = finite_number(value)
            if number is None or number <= 0:
                raise BenchmarkImportError(f"{key} must be a positive number")
            load[key] = number
    if "ramp_up_strategy" in data:
        strategy = _optional_str(data, "ramp_up_strategy")
        load["ramp_up_strategy"] = strategy
        for key in _RAMP_KEYS[1:]:
            if key not in data:
                unresolved.append(key)
            else:
                load[key] = _non_negative(data[key], key)
    return load, unresolved


def _totals(data: dict[str, Any]) -> dict[str, ImportedTotal]:
    totals: dict[str, ImportedTotal] = {}
    for key, (name, unit) in _TOTALS.items():
        if key not in data:
            continue
        value: float = (
            _non_negative(data[key], key) if key == "duration" else _count(data, key)
        )
        totals[name] = ImportedTotal(value=value, unit=unit, source_key=key)
    return totals


def _metrics(data: dict[str, Any], consumed: set[str]) -> dict[str, ImportedMetric]:
    directions = METRIC_DIRECTIONS[VLLM_BENCH_TOOL]
    stats: dict[str, dict[str, ImportedValue]] = {}
    for key in sorted(data):
        match = _LATENCY_KEY.fullmatch(key)
        if match is None:
            continue
        name = f"{match['metric']}_ms"
        stats.setdefault(name, {})[match["stat"]] = ImportedValue(
            _non_negative(data[key], key), key
        )
        consumed.add(key)
    metrics: dict[str, ImportedMetric] = {}
    for name in sorted(stats):
        short = name.removesuffix("_ms")
        metrics[name] = ImportedMetric(
            unit="ms",
            direction=directions[name],
            stats=stats[name],
            notes=_COMMON_NOTES + _METRIC_NOTES.get(short, ()),
        )
    for key, unit in _RATES.items():
        if key not in data:
            continue
        consumed.add(key)
        if data[key] is None:
            continue  # request_goodput is null unless --goodput was set.
        metrics[key] = ImportedMetric(
            unit=unit,
            direction=directions[key],
            stats={"value": ImportedValue(_non_negative(data[key], key), key)},
            notes=("Computed by vllm bench serve over the benchmark duration.",),
        )
    return metrics


def _user_metadata(data: dict[str, Any], consumed: set[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key in sorted(data):
        value = data[key]
        if (
            key in consumed
            or key in _KNOWN_STRING_KEYS
            or not isinstance(value, str)
            or len(key) > _MAX_USER_METADATA_CHARS
            or len(value) > _MAX_USER_METADATA_CHARS
            or len(metadata) >= _MAX_USER_METADATA
        ):
            continue
        metadata[key] = value
    return metadata


def import_vllm_bench(
    data: Any, *, sha256: str, filename: str, importer_version: str | None
) -> ImportedBenchmark:
    if not isinstance(data, dict):
        raise BenchmarkImportError("a vLLM bench result must be a JSON object")
    if "total_input_sequences" in data or "input_sequence_throughput" in data:
        raise BenchmarkImportError(
            "this is a vLLM pooling or embedding benchmark; only generation "
            "benchmarks are supported"
        )
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing or not ("backend" in data or "endpoint_type" in data):
        raise BenchmarkImportError(
            "not a `vllm bench serve --save-result` file; missing "
            + ", ".join(missing or ["backend or endpoint_type"])
        )
    model_id = data["model_id"]
    if not isinstance(model_id, str) or not model_id:
        raise BenchmarkImportError("model_id must be a non-empty string")
    completed = _count(data, "completed")
    if completed == 0:
        raise BenchmarkImportError(
            "no request completed; vLLM writes placeholder zeros for every "
            "statistic in that case, so nothing is imported"
        )
    duration = _non_negative(data["duration"], "duration")
    if duration <= 0:
        raise BenchmarkImportError("duration must be positive")

    load, unresolved = _load(data)
    consumed = {
        *REQUIRED_KEYS,
        *_TOTALS,
        *_LOAD_KEYS,
        *_RAMP_KEYS,
        *_RUN_KEYS,
        *_ENDPOINT_KEYS,
    }
    metrics = _metrics(data, consumed)
    endpoint = {key: _optional_str(data, key) for key in _ENDPOINT_KEYS}
    run = {
        "started_at": _optional_str(data, "date"),
        "label": _optional_str(data, "label"),
        "tokenizer_id": _optional_str(data, "tokenizer_id"),
    }
    user_metadata = _user_metadata(data, consumed)
    unmapped = sorted(
        key for key in data if key not in consumed and key not in user_metadata
    )
    return ImportedBenchmark(
        source=ImportedSource(
            tool=VLLM_BENCH_TOOL,
            sha256=sha256,
            filename=filename,
            tool_version=None,
            schema_version=None,
            user_metadata=user_metadata,
        ),
        model_id=model_id,
        endpoint=endpoint,
        load=load,
        totals=_totals(data),
        metrics=metrics,
        run=run,
        load_unresolved=tuple(unresolved),
        unmapped_keys=tuple(unmapped),
        importer_version=importer_version,
    )
