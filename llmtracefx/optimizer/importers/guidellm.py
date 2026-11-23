"""Map one benchmark from a GuideLLM ``benchmarks.json`` report.

Field names follow GuideLLM's ``GenerativeBenchmarksReport``,
``GenerativeBenchmark``, ``BenchmarkConfig`` and ``GenerativeMetrics``
(``src/guidellm/benchmark/schemas/``). Report version 1 (GuideLLM 0.5 and
0.6) and version 2 (GuideLLM 0.7 and later) are read. Any other
``metadata.version`` is refused, as are reports from GuideLLM 0.4 and
earlier, which state no version.

A report can hold several benchmarks (a sweep, or several rates or stream
counts). Each artifact holds exactly one, chosen with ``benchmark``.

Every GuideLLM metric is a ``StatusDistributionSummary``. Only its
``successful`` distribution is read, and only when it has observations;
GuideLLM writes zero filled placeholders for empty statuses. The per-request
``requests`` lists are never read or copied (they hold prompt and output
text). Endpoint URLs and API keys are never copied either.
"""

from __future__ import annotations

import json
from typing import Any

from .artifact import (
    GUIDELLM_TOOL,
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

SUPPORTED_REPORT_VERSIONS = (1, 2)
# GuideLLM's DEFAULT_API and LEGACY_API_ALIASES (backends/openai/http.py).
_DEFAULT_REQUEST_FORMAT = "/v1/chat/completions"
_LEGACY_REQUEST_FORMATS = {
    "text_completions": "/v1/completions",
    "chat_completions": "/v1/chat/completions",
    "audio_transcriptions": "/v1/audio/transcriptions",
    "audio_translations": "/v1/audio/translations",
}
_UNITS = {
    "time_to_first_token_ms": "ms",
    "time_to_first_output_token_ms": "ms",
    "time_per_output_token_ms": "ms",
    "inter_token_latency_ms": "ms",
    "request_latency": "s",
    "requests_per_second": "requests/s",
    "output_tokens_per_second": "tokens/s",
    "prompt_tokens_per_second": "tokens/s",
    "tokens_per_second": "tokens/s",
}
_RATES = frozenset(
    {
        "requests_per_second",
        "output_tokens_per_second",
        "prompt_tokens_per_second",
        "tokens_per_second",
    }
)
# (LLMTraceFX stat name, GuideLLM DistributionSummary key)
_STATS = (
    ("mean", "mean"),
    ("median", "median"),
    ("std", "std_dev"),
    ("min", "min"),
    ("max", "max"),
)
_PERCENTILES = (
    ("p0.1", "p001"),
    ("p1", "p01"),
    ("p5", "p05"),
    ("p10", "p10"),
    ("p25", "p25"),
    ("p50", "p50"),
    ("p75", "p75"),
    ("p90", "p90"),
    ("p95", "p95"),
    ("p99", "p99"),
    ("p99.9", "p999"),
)
_TOTAL_COUNTS = {
    "successful": "completed_requests",
    "errored": "failed_requests",
    "incomplete": "incomplete_requests",
}
_TOKEN_TOTALS = {
    "prompt_token_count": "total_input_tokens",
    "output_token_count": "total_output_tokens",
}
_STRATEGY_NEEDS = {
    "concurrent": "streams",
    "constant": "rate",
    "poisson": "rate",
}
# Process count on the client machine, not load on the server.
_STRATEGY_SKIP = frozenset({"type_", "worker_count"})
_COMMON_NOTES = (
    "Computed by GuideLLM from its client-side timings over the successful "
    "requests in the measurement window (warmup and cooldown excluded).",
)
_LATENCY_NOTES = (
    "std is GuideLLM's std_dev (population). GuideLLM counts a request with "
    "no value for this metric as 0.",
)
_RATE_NOTES = (
    "mean is the count in the measurement window divided by its duration. "
    "GuideLLM's percentiles of a rate are instantaneous rates between events "
    "and are not imported.",
)
_METRIC_NOTES = {
    "time_per_output_token_ms": (
        "Per request (last token - request start) / output tokens, so it "
        "includes TTFT. Statistics are weighted by each request's output tokens.",
    ),
    "inter_token_latency_ms": (
        "Per request (last token - first token) / (output tokens - 1); vLLM "
        "bench calls this TPOT. Statistics are weighted by output tokens - 1, "
        "so long responses count more than in vLLM bench or AIPerf.",
    ),
    "time_to_first_output_token_ms": (
        "Time to the first content token, excluding reasoning tokens.",
    ),
}


def looks_like_guidellm_report(data: Any) -> bool:
    if not isinstance(data, dict) or not isinstance(data.get("benchmarks"), list):
        return False
    metadata = data.get("metadata")
    if isinstance(metadata, dict) and "guidellm_version" in metadata:
        return True
    return any(
        isinstance(item, dict) and item.get("type_") == "generative_benchmark"
        for item in data["benchmarks"]
    )


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkImportError(f"{name} must be an object")
    return value


def _str(value: Any, name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise BenchmarkImportError(f"{name} must be a string")
    return value


def _count(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BenchmarkImportError(f"{name} must be a whole number >= 0")
    return value


def _number(value: Any, name: str) -> float:
    number = finite_number(value)
    if number is None or number < 0:
        raise BenchmarkImportError(f"{name} must be a finite number >= 0")
    return number


def _strategy_label(benchmark: Any) -> str:
    try:
        strategy = benchmark["config"]["strategy"]
        label = str(strategy["type_"])
    except (KeyError, TypeError):
        return "unknown strategy"
    for key in ("streams", "rate"):
        if isinstance(strategy.get(key), (int, float)):
            return f"{label} {key}={strategy[key]:g}"
    return label


def _select(benchmarks: list[Any], index: int | None) -> int:
    if not benchmarks:
        raise BenchmarkImportError("the GuideLLM report holds no benchmarks")
    if index is None:
        if len(benchmarks) == 1:
            return 0
        listed = "; ".join(
            f"{i}: {_strategy_label(item)}" for i, item in enumerate(benchmarks)
        )
        raise BenchmarkImportError(
            f"the GuideLLM report holds {len(benchmarks)} benchmarks at different "
            f"loads; choose one with --benchmark INDEX ({listed})"
        )
    if not 0 <= index < len(benchmarks):
        raise BenchmarkImportError(
            f"--benchmark {index} is out of range; the report holds "
            f"{len(benchmarks)} benchmark(s), indexed from 0"
        )
    return index


def _load_scalar(value: Any) -> LoadValue | dict[str, Any] | list[Any]:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None or isinstance(value, (str, dict, list)):
        return value
    number = finite_number(value)
    if number is None:
        raise BenchmarkImportError(f"load value {value!r} is not a finite number")
    return int(value) if isinstance(value, int) else number


def _pick(value: Any, index: int) -> Any:
    if isinstance(value, list):
        if not value:
            raise ValueError("empty list")
        return _pick(value[min(index, len(value) - 1)], index)
    if isinstance(value, dict):
        return {k: _pick(v, index) for k, v in value.items()}
    return value


def _constraint(value: Any) -> str | None:
    """Canonical JSON for one constraint, or None when it cannot be resolved.

    A multi-benchmark run stores each limit as a list, one entry per
    benchmark, and ``current_index`` says which entry applied. GuideLLM
    resolves it as ``value[min(max(0, current_index), len(value) - 1)]``.
    The same is done here and ``current_index`` is dropped, so a limit of 20
    keys the same whether it came from a single run or from a sweep.
    """
    if not isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    index = value.get("current_index", -1)
    if isinstance(index, bool) or not isinstance(index, int):
        return None
    body = {k: v for k, v in value.items() if k != "current_index"}
    try:
        body = _pick(body, max(0, index))
    except ValueError:
        return None
    return json.dumps(body, sort_keys=True)


def _load(
    config: dict[str, Any], prefix: str
) -> tuple[dict[str, LoadValue], list[str]]:
    load: dict[str, LoadValue] = {}
    unresolved: list[str] = []
    strategy = _mapping(config.get("strategy"), f"{prefix}.config.strategy")
    strategy_type = strategy.get("type_")
    if not isinstance(strategy_type, str):
        unresolved.append("strategy type")
    load["strategy"] = strategy_type if isinstance(strategy_type, str) else None
    for key, value in sorted(strategy.items()):
        if key in _STRATEGY_SKIP:
            continue
        scalar = _load_scalar(value)
        if not isinstance(scalar, (dict, list)):
            load[f"strategy.{key}"] = scalar
    needed = _STRATEGY_NEEDS.get(str(strategy_type))
    if needed is not None and load.get(f"strategy.{needed}") is None:
        unresolved.append(f"strategy.{needed}")
    # Constraint shapes differ across GuideLLM releases, so each is kept as
    # canonical JSON and runs compare only when it matches exactly.
    constraints = _mapping(
        config.get("constraints", {}), f"{prefix}.config.constraints"
    )
    for name, value in sorted(constraints.items()):
        resolved = _constraint(value)
        if resolved is None:
            unresolved.append(f"constraint.{name}")
        load[f"constraint.{name}"] = resolved
    for phase in ("warmup", "cooldown"):
        settings = config.get(phase)
        if not isinstance(settings, dict):
            continue
        for key, value in sorted(settings.items()):
            scalar = _load_scalar(value)
            if not isinstance(scalar, (dict, list)):
                load[f"{phase}.{key}"] = scalar
    return load, unresolved


def _stream(value: Any, where: str) -> bool | None:
    if value is not None and not isinstance(value, bool):
        raise BenchmarkImportError(f"{where} must be a bool")
    return value


def _request_format(value: Any, where: str) -> str | None:
    found = _str(value, where)
    return None if found is None else _LEGACY_REQUEST_FORMATS.get(found, found)


def _endpoint_v2(backend: dict[str, Any], prefix: str) -> dict[str, EndpointValue]:
    # GuideLLM 0.7 and later record the backend's resolved settings.
    where = f"{prefix}.config.backend"
    return {
        "backend": _str(backend.get("kind"), f"{where}.kind"),
        "request_format": _request_format(
            backend.get("request_format"), f"{where}.request_format"
        ),
        "stream": _stream(backend.get("stream"), f"{where}.stream"),
    }


def _endpoint_v1(data: dict[str, Any]) -> dict[str, EndpointValue]:
    # GuideLLM 0.5 and 0.6 record only what the user passed, under the
    # top-level args, so GuideLLM's own defaults are applied here.
    args = _mapping(data.get("args", {}), "args")
    request_format: Any
    stream: Any
    if "data_request_formatter" in args:
        # 0.5: the request formatter sets the API route and streaming.
        formatter = args["data_request_formatter"]
        if isinstance(formatter, str):
            request_format, stream = formatter, True
        elif isinstance(formatter, dict):
            # Without request_type GuideLLM falls back to an environment
            # setting the report does not record, so it stays unresolved.
            request_format = formatter.get("request_type")
            stream = formatter.get("stream", True)
        else:
            request_format, stream = None, None
        where = "args.data_request_formatter"
    else:
        # 0.6: the backend arguments carry them, with these defaults.
        kwargs = _mapping(args.get("backend_kwargs") or {}, "args.backend_kwargs")
        request_format = kwargs.get("request_format") or _DEFAULT_REQUEST_FORMAT
        stream = kwargs.get("stream", True)
        where = "args.backend_kwargs"
    return {
        "backend": _str(args.get("backend"), "args.backend"),
        "request_format": _request_format(request_format, f"{where}.request_format"),
        "stream": _stream(stream, f"{where}.stream"),
    }


def _successful(metrics: dict[str, Any], name: str, prefix: str) -> dict | None:
    summary = metrics.get(name)
    if summary is None:
        return None
    summary = _mapping(summary, f"{prefix}.{name}")
    successful = summary.get("successful")
    if successful is None:
        return None
    successful = _mapping(successful, f"{prefix}.{name}.successful")
    if _count(successful.get("count"), f"{prefix}.{name}.successful.count") == 0:
        return None  # GuideLLM's zero filled placeholder for an empty status.
    return successful


def _totals(
    benchmark: dict[str, Any], metrics: dict[str, Any], prefix: str
) -> dict[str, ImportedTotal]:
    request_totals = _mapping(
        metrics.get("request_totals"), f"{prefix}.metrics.request_totals"
    )
    totals: dict[str, ImportedTotal] = {}
    for key, name in _TOTAL_COUNTS.items():
        source_key = f"{prefix}.metrics.request_totals.{key}"
        totals[name] = ImportedTotal(
            value=_count(request_totals.get(key), source_key),
            unit="requests",
            source_key=source_key,
        )
    for metric, name in _TOKEN_TOTALS.items():
        successful = _successful(metrics, metric, f"{prefix}.metrics")
        if successful is None:
            continue
        source_key = f"{prefix}.metrics.{metric}.successful.total_sum"
        tokens = _number(successful.get("total_sum"), source_key)
        totals[name] = ImportedTotal(
            value=int(tokens) if tokens.is_integer() else tokens,
            unit="tokens",
            source_key=source_key,
        )
    if "duration" in benchmark:
        source_key = f"{prefix}.duration"
        totals["duration"] = ImportedTotal(
            value=_number(benchmark["duration"], source_key),
            unit="s",
            source_key=source_key,
        )
    return totals


def _metrics(metrics: dict[str, Any], prefix: str) -> dict[str, ImportedMetric]:
    imported: dict[str, ImportedMetric] = {}
    for name, direction in METRIC_DIRECTIONS[GUIDELLM_TOOL].items():
        successful = _successful(metrics, name, f"{prefix}.metrics")
        if successful is None:
            continue
        base = f"{prefix}.metrics.{name}.successful"
        fields = [(stat, successful, key, base) for stat, key in _STATS]
        if name in _RATES:
            fields = fields[:1]
        else:
            where = f"{base}.percentiles"
            percentiles = _mapping(successful.get("percentiles"), where)
            fields += [(stat, percentiles, key, where) for stat, key in _PERCENTILES]
        stats: dict[str, ImportedValue] = {}
        for stat, holder, key, where in fields:
            if holder.get(key) is None:
                continue
            source_key = f"{where}.{key}"
            stats[stat] = ImportedValue(_number(holder[key], source_key), source_key)
        if not stats:
            continue
        notes = _COMMON_NOTES + (_RATE_NOTES if name in _RATES else _LATENCY_NOTES)
        imported[name] = ImportedMetric(
            unit=_UNITS[name],
            direction=direction,
            stats=stats,
            notes=notes + _METRIC_NOTES.get(name, ()),
        )
    return imported


def import_guidellm(
    data: Any,
    *,
    sha256: str,
    filename: str,
    importer_version: str | None,
    benchmark: int | None = None,
) -> ImportedBenchmark:
    if not isinstance(data, dict):
        raise BenchmarkImportError("a GuideLLM report must be a JSON object")
    if not looks_like_guidellm_report(data):
        raise BenchmarkImportError(
            "not a GuideLLM benchmarks.json report (expected metadata and a "
            "benchmarks list)"
        )
    metadata = data.get("metadata")
    supported = ", ".join(str(v) for v in SUPPORTED_REPORT_VERSIONS)
    if not isinstance(metadata, dict) or "version" not in metadata:
        raise BenchmarkImportError(
            "the GuideLLM report states no metadata.version; reports from "
            f"GuideLLM 0.4 and earlier are not supported (supported: {supported})"
        )
    version = metadata["version"]
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version not in SUPPORTED_REPORT_VERSIONS
    ):
        raise BenchmarkImportError(
            f"GuideLLM report version {version!r} is not supported "
            f"(supported: {supported})"
        )

    index = _select(data["benchmarks"], benchmark)
    prefix = f"benchmarks[{index}]"
    item = _mapping(data["benchmarks"][index], prefix)
    if item.get("type_") != "generative_benchmark":
        raise BenchmarkImportError(
            f"{prefix}.type_ is {item.get('type_')!r}; only generative "
            "benchmarks are imported"
        )
    config = _mapping(item.get("config"), f"{prefix}.config")
    backend = _mapping(config.get("backend"), f"{prefix}.config.backend")
    model = backend.get("model")
    if not isinstance(model, str) or not model:
        raise BenchmarkImportError(
            f"{prefix}.config.backend.model names no model; results without "
            "a model are not imported"
        )
    metrics = _mapping(item.get("metrics"), f"{prefix}.metrics")
    totals = _totals(item, metrics, prefix)
    if totals["completed_requests"].value < 1:
        raise BenchmarkImportError(
            f"{prefix} has no successful requests in its measurement window"
        )

    load, unresolved = _load(config, prefix)
    endpoint = _endpoint_v1(data) if version == 1 else _endpoint_v2(backend, prefix)
    # Runs only compare with the same API route and streaming mode, so an
    # endpoint setting the report does not state blocks the gate.
    unresolved.extend(f"endpoint.{k}" for k, v in endpoint.items() if v is None)
    imported = _metrics(metrics, prefix)
    consumed = {"request_totals", *_TOKEN_TOTALS, *imported}
    return ImportedBenchmark(
        source=ImportedSource(
            tool=GUIDELLM_TOOL,
            sha256=sha256,
            filename=filename,
            tool_version=_str(
                metadata.get("guidellm_version"), "metadata.guidellm_version"
            ),
            schema_version=str(version),
        ),
        model_id=model,
        endpoint=endpoint,
        load=load,
        totals=totals,
        metrics=imported,
        run={
            "benchmark_index": str(index),
            "benchmark_count": str(len(data["benchmarks"])),
            "benchmark_id": _str(config.get("id_"), f"{prefix}.config.id_"),
            "run_id": _str(config.get("run_id"), f"{prefix}.config.run_id"),
        },
        load_unresolved=tuple(unresolved),
        unmapped_keys=tuple(
            sorted(f"metrics.{key}" for key in metrics if key not in consumed)
        ),
        importer_version=importer_version,
    )
