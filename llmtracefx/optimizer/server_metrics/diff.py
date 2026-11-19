"""Turn two server metrics snapshots into attributable evidence.

A Prometheus scrape is server wide. A delta between two scrapes counts
every request the server handled in the window, not only the one that was
measured. This module therefore never presents a delta as belonging to a
request until the server's own request counter agrees with the number of
requests the caller says it sent. When it cannot check, it says so.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..collectors._shared import atomic_write_text
from ..schema import MetricProvenance
from .prometheus import MetricFamily, Sample, encode_value, parse_number
from .snapshot import (
    Fetcher,
    ServerMetricsError,
    ServerMetricsSnapshot,
    capture_snapshot,
    urllib_fetcher,
)
from .speculative import SpeculativeCounters, speculative_evidence

EVIDENCE_KIND = "llmtracefx.server_metrics.evidence"
EVIDENCE_SCHEMA_VERSION = "1"

ATTRIBUTED = "attributed"
OPERATOR_ASSERTED = "operator_asserted"
CONTAMINATED = "contaminated"
INCOMPLETE = "incomplete"
COUNTER_RESET = "counter_reset"
UNVERIFIABLE = "unverifiable"
STATS_NOT_ENABLED = "stats_not_enabled"
ATTRIBUTABLE_STATUSES = frozenset({ATTRIBUTED, OPERATOR_ASSERTED})

# Labels kept when series are collapsed. Everything else (model names that
# may be local paths, engine indexes, pod names) is summed away so the
# evidence is safe to share.
KEPT_LABELS = frozenset(
    {
        "le",
        "quantile",
        "finished_reason",
        "is_streaming",
        "cache_source",
        "reason",
        "position",
        "source",
        "transfer_type",
        "cache_level",
        "cache_tier",
    }
)
_PROCESS_START = "process_start_time_seconds"
_QUANTILES = (0.5, 0.9, 0.99)

SeriesKey = tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class HistogramHighlight:
    key: str
    family: str
    unit: str = "seconds"
    """``seconds`` values are reported in ms; anything else as is."""


@dataclass(frozen=True)
class CounterHighlight:
    key: str
    family: str
    unit: str
    plus: tuple[str, ...] = ()
    """Further families added to ``family``. Every one must be exposed."""
    split_label: str | None = None
    """Also report the delta for each value of this label."""
    match: tuple[tuple[str, str], ...] = ()
    """Only series carrying all of these label values count."""
    one_rank_labels: tuple[str, ...] = ()
    """Rank labels whose series each report the same logical count. The
    lowest rank's delta is used instead of the sum over those ranks."""
    fallback: CounterHighlight | None = None
    """Read instead when ``family`` has no samples in the after snapshot."""


@dataclass(frozen=True)
class RatioHighlight:
    key: str
    numerator: str | tuple[str, ...]
    denominator: str | tuple[str, ...]
    """A tuple is the sum of its families, all of which must be exposed."""
    unit: str


@dataclass(frozen=True)
class PointGauge:
    """A gauge read as a before and an after point sample, never diffed."""

    key: str
    family: str
    unit: str


@dataclass(frozen=True)
class StatsRequirement:
    """Families a server only exposes when a setting is on."""

    families: tuple[str, ...]
    setting: str
    consequence: str
    required: bool = True
    """When False, ``--assume-isolated`` may stand in for the families."""


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    family_prefix: str | None
    request_counters: tuple[tuple[str, str], ...] = ()
    """``(family, kind)`` in preference order. ``kind`` is ``histogram`` (use
    the ``_count`` delta) or ``counter``. The first present family is reported
    as the observed request count, and every present family must count
    exactly the expected requests. Some runtimes count a request when it
    starts and record its duration and outcome when it finishes, so a
    foreign request already running at the before scrape can leave the start
    counter exact while the finish-time families count it in the window."""
    in_flight_gauges: tuple[str, ...] = ()
    histograms: tuple[HistogramHighlight, ...] = ()
    counters: tuple[CounterHighlight, ...] = ()
    ratios: tuple[RatioHighlight, ...] = ()
    speculative: SpeculativeCounters | None = None
    point_gauges: tuple[PointGauge, ...] = ()
    stats_requirements: tuple[StatsRequirement, ...] = ()
    stats_hint: str = ""
    """Appended to a ``stats_not_enabled`` reason."""
    notes: tuple[str, ...] = field(default_factory=tuple)


SGLANG_RANK_LABELS = ("tp_rank", "pp_rank")

PROFILES: dict[str, RuntimeProfile] = {
    "vllm": RuntimeProfile(
        name="vllm",
        family_prefix="vllm:",
        request_counters=(
            ("vllm:e2e_request_latency_seconds", "histogram"),
            ("vllm:request_success", "counter"),
        ),
        in_flight_gauges=("vllm:num_requests_running", "vllm:num_requests_waiting"),
        histograms=(
            HistogramHighlight(
                "time_to_first_token", "vllm:time_to_first_token_seconds"
            ),
            HistogramHighlight("e2e_latency", "vllm:e2e_request_latency_seconds"),
            HistogramHighlight("queue_time", "vllm:request_queue_time_seconds"),
            HistogramHighlight("prefill_time", "vllm:request_prefill_time_seconds"),
            HistogramHighlight("decode_time", "vllm:request_decode_time_seconds"),
            HistogramHighlight("inference_time", "vllm:request_inference_time_seconds"),
            HistogramHighlight(
                "inter_token_latency", "vllm:inter_token_latency_seconds"
            ),
            HistogramHighlight(
                "time_per_output_token", "vllm:request_time_per_output_token_seconds"
            ),
        ),
        counters=(
            CounterHighlight("prompt_tokens", "vllm:prompt_tokens", "tokens"),
            CounterHighlight("generation_tokens", "vllm:generation_tokens", "tokens"),
            CounterHighlight(
                "prefix_cache_queries", "vllm:prefix_cache_queries", "tokens"
            ),
            CounterHighlight("prefix_cache_hits", "vllm:prefix_cache_hits", "tokens"),
            CounterHighlight("preemptions", "vllm:num_preemptions", "count"),
            # Prompt tokens by where they came from and cached prompt tokens
            # by tier: vllm/v1/metrics/loggers.py L762-810 at vLLM 1ae1b648.
            CounterHighlight(
                "cached_prompt_tokens", "vllm:prompt_tokens_cached", "tokens"
            ),
            CounterHighlight(
                "prompt_tokens_by_source",
                "vllm:prompt_tokens_by_source",
                "tokens",
                split_label="source",
            ),
            CounterHighlight(
                "cached_prompt_tokens_by_tier",
                "vllm:prompt_tokens_cached_by_source",
                "tokens",
                split_label="source",
            ),
            # KV connector cache (loggers.py L700-712).
            CounterHighlight(
                "external_prefix_cache_queries",
                "vllm:external_prefix_cache_queries",
                "tokens",
            ),
            CounterHighlight(
                "external_prefix_cache_hits",
                "vllm:external_prefix_cache_hits",
                "tokens",
            ),
            # CPU offloading connector: per-direction counters, falling back
            # to the deprecated transfer_type-labelled totals
            # (vllm/distributed/kv_transfer/kv_connector/v1/offloading/
            # metrics.py L42-62 and L149-151 at vLLM 1ae1b648).
            CounterHighlight(
                "kv_offload_load_bytes",
                "vllm:kv_offload_load_bytes",
                "bytes",
                fallback=CounterHighlight(
                    "kv_offload_load_bytes",
                    "vllm:kv_offload_total_bytes",
                    "bytes",
                    match=(("transfer_type", "CPU_to_GPU"),),
                ),
            ),
            CounterHighlight(
                "kv_offload_load_time",
                "vllm:kv_offload_load_time",
                "seconds",
                fallback=CounterHighlight(
                    "kv_offload_load_time",
                    "vllm:kv_offload_total_time",
                    "seconds",
                    match=(("transfer_type", "CPU_to_GPU"),),
                ),
            ),
            CounterHighlight(
                "kv_offload_store_bytes",
                "vllm:kv_offload_store_bytes",
                "bytes",
                fallback=CounterHighlight(
                    "kv_offload_store_bytes",
                    "vllm:kv_offload_total_bytes",
                    "bytes",
                    match=(("transfer_type", "GPU_to_CPU"),),
                ),
            ),
            CounterHighlight(
                "kv_offload_store_time",
                "vllm:kv_offload_store_time",
                "seconds",
                fallback=CounterHighlight(
                    "kv_offload_store_time",
                    "vllm:kv_offload_total_time",
                    "seconds",
                    match=(("transfer_type", "GPU_to_CPU"),),
                ),
            ),
        ),
        ratios=(
            RatioHighlight(
                "prefix_cache_hit_rate",
                "vllm:prefix_cache_hits",
                "vllm:prefix_cache_queries",
                "ratio",
            ),
            RatioHighlight(
                "cached_prompt_token_ratio",
                "vllm:prompt_tokens_cached",
                "vllm:prompt_tokens",
                "ratio",
            ),
            RatioHighlight(
                "external_prefix_cache_hit_rate",
                "vllm:external_prefix_cache_hits",
                "vllm:external_prefix_cache_queries",
                "ratio",
            ),
        ),
        # loggers.py L634: fraction of KV blocks in use, 0 to 1.
        point_gauges=(
            PointGauge("kv_cache_usage", "vllm:kv_cache_usage_perc", "ratio"),
        ),
        # vllm/v1/spec_decode/metrics.py registers these only when the server
        # runs with a speculative config.
        speculative=SpeculativeCounters(
            verify_steps="vllm:spec_decode_num_drafts",
            draft_tokens="vllm:spec_decode_num_draft_tokens",
            accepted_tokens="vllm:spec_decode_num_accepted_tokens",
            accepted_per_position="vllm:spec_decode_num_accepted_tokens_per_pos",
        ),
    ),
    "sglang": RuntimeProfile(
        name="sglang",
        family_prefix="sglang:",
        request_counters=(
            ("sglang:num_requests", "counter"),
            ("sglang:e2e_request_latency_seconds", "histogram"),
        ),
        in_flight_gauges=("sglang:num_running_reqs", "sglang:num_queue_reqs"),
        histograms=(
            HistogramHighlight(
                "time_to_first_token", "sglang:time_to_first_token_seconds"
            ),
            HistogramHighlight("e2e_latency", "sglang:e2e_request_latency_seconds"),
            HistogramHighlight("queue_time", "sglang:queue_time_seconds"),
            HistogramHighlight(
                "inter_token_latency", "sglang:inter_token_latency_seconds"
            ),
            HistogramHighlight(
                "time_per_output_token",
                "sglang:request_time_per_output_token_seconds",
            ),
        ),
        counters=(
            CounterHighlight("prompt_tokens", "sglang:prompt_tokens", "tokens"),
            CounterHighlight("generation_tokens", "sglang:generation_tokens", "tokens"),
            CounterHighlight("cached_tokens", "sglang:cached_tokens", "tokens"),
            CounterHighlight(
                "aborted_requests", "sglang:num_aborted_requests", "count"
            ),
            # cache_source is device, host or storage, or total on builds
            # without the breakdown (python/sglang/srt/observability/
            # metrics_collector.py L2013 and L2193-2213 at SGLang b6d4389a).
            CounterHighlight(
                "cached_prompt_tokens_by_tier",
                "sglang:cached_tokens",
                "tokens",
                split_label="cache_source",
            ),
            # HiCache counters. Every TP rank reports the same logical count
            # under its own rank labels, so summing ranks gives TP times the
            # count (metrics_collector.py L2515-2530 and L2650-2700). The L2
            # counters are split by host pool; the kv pool is the full KV
            # cache (mem_cache/hicache_storage.py L109 and
            # managers/cache_controller.py L1137-1138), and the others repeat
            # the same tokens for other state.
            CounterHighlight(
                "hicache_load_back_tokens",
                "sglang:load_back_tokens",
                "tokens",
                match=(("pool", "kv"),),
                one_rank_labels=SGLANG_RANK_LABELS,
            ),
            CounterHighlight(
                "hicache_backup_tokens",
                "sglang:hicache_backup_tokens",
                "tokens",
                match=(("pool", "kv"),),
                one_rank_labels=SGLANG_RANK_LABELS,
            ),
            # L3 storage counters carry the same rank keys
            # (mem_cache/unified_cache/storage_attachment.py L333-341) and are
            # read the same way; ranks_agree says whether the ranks did agree.
            CounterHighlight(
                "hicache_storage_prefetched_tokens",
                "sglang:prefetched_tokens",
                "tokens",
                one_rank_labels=SGLANG_RANK_LABELS,
            ),
            CounterHighlight(
                "hicache_storage_backup_tokens",
                "sglang:backuped_tokens",
                "tokens",
                one_rank_labels=SGLANG_RANK_LABELS,
            ),
            CounterHighlight(
                "hicache_storage_prefetch_hit_tokens",
                "sglang:storage_prefetch_hit_tokens",
                "tokens",
                one_rank_labels=SGLANG_RANK_LABELS,
            ),
        ),
        ratios=(
            RatioHighlight(
                "cached_prompt_token_ratio",
                "sglang:cached_tokens",
                "sglang:prompt_tokens",
                "ratio",
            ),
        ),
        # SGLang counts verify calls but not accepted drafts, so only the
        # acceptance length can come from counters. Its spec_accept_* series
        # are last-value gauges and are never differenced.
        speculative=SpeculativeCounters(
            verify_steps="sglang:spec_verify_calls",
            emitted_tokens="sglang:generation_tokens",
            ignored_gauges=(
                "sglang:spec_accept_length",
                "sglang:spec_accept_rate",
                "sglang:spec_block_accept_length",
                "sglang:spec_cap_length",
                "sglang:spec_num_draft_tokens",
                "sglang:spec_num_steps",
            ),
        ),
        # metrics_collector.py L418 and L458-470; the host gauges exist only
        # with HiCache (L819-830).
        point_gauges=(
            PointGauge("token_usage", "sglang:token_usage", "ratio"),
            PointGauge("kv_available_tokens", "sglang:kv_available_tokens", "tokens"),
            PointGauge("kv_evictable_tokens", "sglang:kv_evictable_tokens", "tokens"),
            PointGauge("kv_used_tokens", "sglang:kv_used_tokens", "tokens"),
            PointGauge(
                "hicache_host_used_tokens", "sglang:hicache_host_used_tokens", "tokens"
            ),
            PointGauge(
                "hicache_host_total_tokens",
                "sglang:hicache_host_total_tokens",
                "tokens",
            ),
        ),
    ),
    "llamacpp": RuntimeProfile(
        name="llamacpp",
        family_prefix="llamacpp:",
        in_flight_gauges=("llamacpp:requests_processing", "llamacpp:requests_deferred"),
        counters=(
            # prompt_tokens_total excludes cached tokens, which have their own
            # counter (tools/server/server-task.cpp L1566-1580 at llama.cpp
            # f8caa5ea, split in decaf508). Prompt work is their sum.
            CounterHighlight(
                "prompt_tokens",
                "llamacpp:prompt_tokens",
                "tokens",
                plus=("llamacpp:prompt_tokens_cached",),
            ),
            CounterHighlight(
                "computed_prompt_tokens", "llamacpp:prompt_tokens", "tokens"
            ),
            CounterHighlight(
                "cached_prompt_tokens", "llamacpp:prompt_tokens_cached", "tokens"
            ),
            CounterHighlight("prefill_time", "llamacpp:prompt_seconds", "seconds"),
            CounterHighlight(
                "generation_tokens", "llamacpp:tokens_predicted", "tokens"
            ),
            CounterHighlight(
                "decode_time", "llamacpp:tokens_predicted_seconds", "seconds"
            ),
            CounterHighlight("decode_calls", "llamacpp:n_decode", "count"),
        ),
        ratios=(
            RatioHighlight(
                "cached_prompt_token_ratio",
                "llamacpp:prompt_tokens_cached",
                ("llamacpp:prompt_tokens", "llamacpp:prompt_tokens_cached"),
                "ratio",
            ),
            RatioHighlight(
                "prompt_tokens_per_second",
                "llamacpp:prompt_tokens",
                "llamacpp:prompt_seconds",
                "tokens/s",
            ),
            RatioHighlight(
                "generation_tokens_per_second",
                "llamacpp:tokens_predicted",
                "llamacpp:tokens_predicted_seconds",
                "tokens/s",
            ),
        ),
        # Exported at zero when speculation is off; spec_decode_num_drafts
        # counts verification steps.
        speculative=SpeculativeCounters(
            verify_steps="llamacpp:spec_decode_num_drafts",
            draft_tokens="llamacpp:spec_decode_num_draft_tokens",
            accepted_tokens="llamacpp:spec_decode_num_accepted_tokens",
            accepted_per_position="llamacpp:spec_decode_num_accepted_tokens_per_pos",
        ),
        notes=(
            "llama.cpp exposes no request counter, so a window can only be "
            "attributed when the operator asserts the server was isolated.",
        ),
    ),
    "tgi": RuntimeProfile(
        name="tgi",
        family_prefix="tgi_",
        request_counters=(
            ("tgi_request_count", "counter"),
            ("tgi_request_duration", "histogram"),
            ("tgi_request_success", "counter"),
        ),
        # tgi_queue_size counts only queued requests; tgi_batch_current_size
        # counts the ones already running.
        in_flight_gauges=("tgi_queue_size", "tgi_batch_current_size"),
        histograms=(
            HistogramHighlight("e2e_latency", "tgi_request_duration"),
            HistogramHighlight("queue_time", "tgi_request_queue_duration"),
            HistogramHighlight("inference_time", "tgi_request_inference_duration"),
            HistogramHighlight(
                "time_per_output_token", "tgi_request_mean_time_per_token_duration"
            ),
            HistogramHighlight(
                "generated_tokens", "tgi_request_generated_tokens", "tokens"
            ),
        ),
        counters=(
            CounterHighlight("successful_requests", "tgi_request_success", "count"),
            CounterHighlight("failed_requests", "tgi_request_failure", "count"),
        ),
    ),
    # trtllm-serve exposes Prometheus text on /prometheus/metrics, and only
    # with return_perf_metrics: true; /metrics serves JSON iteration stats
    # (tensorrt_llm/serve/openai_server.py L1093-1100 and L1408-1431 at
    # TensorRT-LLM e6ecd280). Names are in tensorrt_llm/metrics/collector.py.
    "trtllm": RuntimeProfile(
        name="trtllm",
        family_prefix="trtllm_",
        # Counted when a finished request's perf metrics arrive (collector.py
        # L236-239 and L1013-1017). The latency histograms are skipped when
        # their value is zero, so they are not request counters.
        request_counters=(("trtllm_request_success", "counter"),),
        # Set from rank 0 iteration stats (collector.py L404-412 and
        # L1153-1158); an idle engine publishes a heartbeat every second
        # (tensorrt_llm/_torch/pyexecutor/py_executor.py L454-456, L9462-9500).
        in_flight_gauges=("trtllm_num_requests_running", "trtllm_num_requests_waiting"),
        histograms=(
            HistogramHighlight(
                "time_to_first_token", "trtllm_time_to_first_token_seconds"
            ),
            HistogramHighlight("e2e_latency", "trtllm_e2e_request_latency_seconds"),
            HistogramHighlight("queue_time", "trtllm_request_queue_time_seconds"),
            HistogramHighlight("prefill_time", "trtllm_request_prefill_time_seconds"),
            HistogramHighlight("decode_time", "trtllm_request_decode_time_seconds"),
            HistogramHighlight(
                "inference_time", "trtllm_request_inference_time_seconds"
            ),
            HistogramHighlight(
                "time_per_output_token", "trtllm_time_per_output_token_seconds"
            ),
        ),
        counters=(
            CounterHighlight("prompt_tokens", "trtllm_prompt_tokens", "tokens"),
            CounterHighlight("generation_tokens", "trtllm_generation_tokens", "tokens"),
            CounterHighlight(
                "cached_prompt_tokens", "trtllm_prompt_cached_tokens", "tokens"
            ),
            # cache_tier is gpu, host, disk, or connector for tokens a KV
            # connector loaded, per dp_rank (collector.py L579-603, L858-899).
            CounterHighlight(
                "cached_prompt_tokens_by_tier",
                "trtllm_prompt_cache_hit_tokens",
                "tokens",
                split_label="cache_tier",
            ),
            CounterHighlight(
                "kv_cache_onboard_bytes", "trtllm_kv_cache_onboard_bytes", "bytes"
            ),
            CounterHighlight(
                "kv_cache_offload_bytes", "trtllm_kv_cache_offload_bytes", "bytes"
            ),
        ),
        # collector.py L337-348; the fraction of blocks in use, 0 to 1
        # (tensorrt_llm/metrics/perf.py L78).
        point_gauges=(
            PointGauge("kv_cache_utilization", "trtllm_kv_cache_utilization", "ratio"),
            PointGauge(
                "kv_cache_host_utilization", "trtllm_kv_cache_host_utilization", "ratio"
            ),
        ),
        # Logged from rank 0 iteration stats, so they also need
        # enable_iter_perf_stats, and only once a step drafts (collector.py
        # L550-567 and L1200-1217). There is no verification step counter.
        # The per-position counters are labelled token_position and are
        # not read.
        speculative=SpeculativeCounters(
            draft_tokens="trtllm_spec_decode_num_draft_tokens",
            accepted_tokens="trtllm_spec_decode_num_accepted_tokens",
            ignored_gauges=(
                "trtllm_spec_decode_acceptance_length",
                "trtllm_spec_decode_draft_overhead",
            ),
        ),
        stats_requirements=(
            StatsRequirement(
                families=("trtllm_request_success",),
                setting="return_perf_metrics: true",
                consequence="no request counter, so the window cannot be checked",
            ),
            StatsRequirement(
                families=("trtllm_num_requests_running", "trtllm_num_requests_waiting"),
                setting="enable_iter_perf_stats: true",
                consequence="no in-flight gauges, so overlapping traffic "
                "cannot be ruled out",
                required=False,
            ),
        ),
        stats_hint=(
            "Set both in the trtllm-serve --config YAML and scrape "
            "/prometheus/metrics."
        ),
        notes=(
            "trtllm-serve exposes Prometheus text on /prometheus/metrics, "
            "not /metrics, and only with return_perf_metrics: true.",
        ),
    ),
    "generic": RuntimeProfile(name="generic", family_prefix=None),
}
RUNTIME_CHOICES = ("auto", *PROFILES)


def detect_runtime(families: Iterable[str]) -> str:
    names = list(families)
    for profile in PROFILES.values():
        prefix = profile.family_prefix
        if prefix is not None and any(name.startswith(prefix) for name in names):
            return profile.name
    return "generic"


def _is_value_sample(family: MetricFamily, sample: Sample) -> bool:
    return not sample.name.endswith("_created") and not (
        family.type == "summary" and any(k == "quantile" for k, _ in sample.labels)
    )


def _collapse(
    samples: Iterable[Sample], *, suffix: str | None = None
) -> dict[SeriesKey, float]:
    """Sum series that differ only in labels outside ``KEPT_LABELS``."""
    totals: dict[SeriesKey, float] = {}
    for sample in samples:
        if suffix is not None and not sample.name.endswith(suffix):
            continue
        key = tuple((k, v) for k, v in sample.labels if k in KEPT_LABELS)
        totals[key] = totals.get(key, 0.0) + sample.value
    return totals


def _family_total(family: MetricFamily | None, suffix: str | None) -> float | None:
    if family is None:
        return None
    values = [
        s.value
        for s in family.samples
        if _is_value_sample(family, s) and (suffix is None or s.name.endswith(suffix))
    ]
    return math.fsum(values) if values else 0.0


def _as_count(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def _requests(count: int) -> str:
    return "1 request" if count == 1 else f"{count} requests"


def _measurement(
    value: float, provenance: MetricProvenance, unit: str
) -> dict[str, Any]:
    return {"value": value, "provenance": provenance.value, "unit": unit}


def _scaled(value: float, unit: str) -> tuple[float, str]:
    return (value * 1000.0, "ms") if unit == "seconds" else (value, unit)


def _parse_iso(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ServerMetricsError(f"invalid timestamp {value!r}") from exc
    if parsed.tzinfo is None:
        raise ServerMetricsError(f"timestamp {value!r} has no timezone")
    return parsed


@dataclass
class _DiffState:
    resets: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _series_delta(
    name: str,
    before: Mapping[SeriesKey, float],
    after: Mapping[SeriesKey, float],
    state: _DiffState,
) -> dict[SeriesKey, float]:
    deltas: dict[SeriesKey, float] = {}
    for key in sorted(set(before) | set(after)):
        if key not in after:
            state.resets.append(f"{name}{dict(key)} disappeared")
            continue
        start = before.get(key, 0.0)
        delta = after[key] - start
        if not math.isfinite(delta):
            state.notes.append(f"{name}{dict(key)} has a non-finite value; skipped")
            continue
        if delta < 0:
            state.resets.append(f"{name}{dict(key)} decreased")
            continue
        deltas[key] = delta
    return deltas


def _counter_rows(
    name: str, before: MetricFamily | None, after: MetricFamily, state: _DiffState
) -> list[dict[str, Any]]:
    before_samples = (
        []
        if before is None
        else [s for s in before.samples if _is_value_sample(before, s)]
    )
    b = _collapse(before_samples)
    a = _collapse(s for s in after.samples if _is_value_sample(after, s))
    deltas = _series_delta(name, b, a, state)
    return [
        {
            "family": name,
            "labels": dict(key),
            "before": encode_value(b.get(key, 0.0)),
            "after": encode_value(a[key]),
            "delta": deltas[key],
        }
        for key in sorted(deltas)
        if deltas[key] != 0
    ]


def _quantile_bounds(
    buckets: list[tuple[float, float]], count: float
) -> list[dict[str, Any]]:
    bounds: list[dict[str, Any]] = []
    for quantile in _QUANTILES:
        target = quantile * count
        lower = None
        for upper, cumulative in buckets:
            if cumulative >= target and cumulative > 0:
                bounds.append(
                    {
                        "quantile": quantile,
                        "lower_bound": lower,
                        "upper_bound": None if math.isinf(upper) else upper,
                        "provenance": MetricProvenance.ESTIMATED.value,
                    }
                )
                break
            lower = upper
    return bounds


def _histogram_rows(
    name: str, before: MetricFamily | None, after: MetricFamily, state: _DiffState
) -> list[dict[str, Any]]:
    def grouped(family: MetricFamily | None) -> dict[SeriesKey, dict[str, Any]]:
        groups: dict[SeriesKey, dict[str, Any]] = {}
        if family is None:
            return groups
        for sample in family.samples:
            if not _is_value_sample(family, sample):
                continue
            le = None
            kept = []
            for k, v in sample.labels:
                if k == "le":
                    le = v
                elif k in KEPT_LABELS:
                    kept.append((k, v))
            group = groups.setdefault(
                tuple(kept), {"count": 0.0, "sum": 0.0, "buckets": {}}
            )
            if sample.name.endswith("_bucket") and le is not None:
                bound = parse_number(le)
                group["buckets"][bound] = (
                    group["buckets"].get(bound, 0.0) + sample.value
                )
            elif sample.name.endswith("_count"):
                group["count"] += sample.value
            elif sample.name.endswith("_sum"):
                group["sum"] += sample.value
        return groups

    b, a = grouped(before), grouped(after)
    rows: list[dict[str, Any]] = []
    for key in sorted(set(b) | set(a)):
        if key not in a:
            state.resets.append(f"{name}{dict(key)} disappeared")
            continue
        start = b.get(key, {"count": 0.0, "sum": 0.0, "buckets": {}})
        end = a[key]
        count = end["count"] - start["count"]
        total = end["sum"] - start["sum"]
        if not (math.isfinite(count) and math.isfinite(total)):
            state.notes.append(f"{name}{dict(key)} has a non-finite value; skipped")
            continue
        if count < 0:
            state.resets.append(f"{name}{dict(key)} count decreased")
            continue
        if count == 0:
            continue
        buckets: list[tuple[float, float]] = []
        for bound in sorted(end["buckets"]):
            delta = end["buckets"][bound] - start["buckets"].get(bound, 0.0)
            if delta < 0:
                state.resets.append(f"{name}{dict(key)} bucket le={bound} decreased")
            buckets.append((bound, delta))
        row: dict[str, Any] = {
            "family": name,
            "type": after.type,
            "labels": dict(key),
            "count_delta": count,
            "sum_delta": total,
        }
        if buckets:
            row["bucket_deltas"] = [
                {"le": encode_value(bound), "cumulative_delta": delta}
                for bound, delta in buckets
            ]
            row["quantile_bounds"] = _quantile_bounds(buckets, count)
        rows.append(row)
    return rows


def _gauge_summary(family: MetricFamily | None) -> dict[str, Any] | None:
    if family is None:
        return None
    values = [s.value for s in family.samples if _is_value_sample(family, s)]
    if not values:
        return {"series": 0}
    finite = [v for v in values if math.isfinite(v)]
    return {
        "series": len(values),
        "min": encode_value(min(values)) if finite else "NaN",
        "max": encode_value(max(values)) if finite else "NaN",
        "sum": encode_value(math.fsum(finite)) if len(finite) == len(values) else "NaN",
    }


def _highlight_histogram(
    spec: HistogramHighlight, rows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    matching = [row for row in rows if row["family"] == spec.family]
    if not matching:
        return None
    count = math.fsum(row["count_delta"] for row in matching)
    total = math.fsum(row["sum_delta"] for row in matching)
    if count <= 0:
        return None
    if count == 1:
        value, unit = _scaled(total, spec.unit)
        result = _measurement(value, MetricProvenance.MEASURED_NATIVE, unit)
        result["statistic"] = "single_observation"
    else:
        value, unit = _scaled(total / count, spec.unit)
        result = _measurement(value, MetricProvenance.DERIVED, unit)
        result["statistic"] = "mean"
    result["source"] = spec.family
    result["observations"] = _as_count(count)
    if count > 1 and len(matching) == 1 and "quantile_bounds" in matching[0]:
        result["quantile_bounds_native_unit"] = matching[0]["quantile_bounds"]
    return result


def _has_samples(family: MetricFamily | None) -> bool:
    return family is not None and any(
        _is_value_sample(family, s) for s in family.samples
    )


def _rank_order(labels: SeriesKey, rank_labels: tuple[str, ...]) -> tuple[Any, ...]:
    values = dict(labels)
    order: list[Any] = []
    for name in rank_labels:
        value = values.get(name, "")
        order.append((0, int(value), "") if value.isdigit() else (1, 0, value))
    return tuple(order)


def _counter_series_deltas(
    spec: CounterHighlight,
    before: ServerMetricsSnapshot,
    after: ServerMetricsSnapshot,
    notes: list[str],
) -> tuple[dict[SeriesKey, float], bool | None] | None:
    """Per series deltas for one highlight, before any label is summed away.

    Series are keyed by their full label set, so ``one_rank_labels`` can be
    told apart. The result is still only ever summed or split by a kept
    label, so no other label value reaches the evidence."""
    deltas: dict[SeriesKey, float] = {}
    for name in (spec.family, *spec.plus):
        family = after.families[name]
        previous = before.families.get(name)
        start = (
            {}
            if previous is None
            else {
                s.labels: s.value
                for s in previous.samples
                if _is_value_sample(previous, s)
            }
        )
        for sample in family.samples:
            if not _is_value_sample(family, sample):
                continue
            if any(dict(sample.labels).get(k) != v for k, v in spec.match):
                continue
            delta = sample.value - start.get(sample.labels, 0.0)
            if not math.isfinite(delta) or delta < 0:
                notes.append(f"{spec.key} skipped: a {name} series moved by {delta!r}")
                return None
            deltas[sample.labels] = deltas.get(sample.labels, 0.0) + delta
    if not spec.one_rank_labels:
        return deltas, None
    groups: dict[SeriesKey, dict[SeriesKey, float]] = {}
    for labels, delta in deltas.items():
        rest = tuple((k, v) for k, v in labels if k not in spec.one_rank_labels)
        groups.setdefault(rest, {})[labels] = delta
    picked: dict[SeriesKey, float] = {}
    agree = True
    for rest, ranks in groups.items():
        lowest = min(ranks, key=lambda key: _rank_order(key, spec.one_rank_labels))
        picked[rest] = ranks[lowest]
        agree = agree and len(set(ranks.values())) == 1
    return picked, agree


def _counter_highlight(
    spec: CounterHighlight,
    before: ServerMetricsSnapshot,
    after: ServerMetricsSnapshot,
    notes: list[str],
) -> dict[str, Any] | None:
    if (
        spec.fallback is not None
        and not _has_samples(after.families.get(spec.family))
        and _has_samples(after.families.get(spec.fallback.family))
    ):
        spec = spec.fallback
    if any(name not in after.families for name in (spec.family, *spec.plus)):
        return None
    computed = _counter_series_deltas(spec, before, after, notes)
    if computed is None:
        return None
    deltas, ranks_agree = computed
    value, unit = _scaled(math.fsum(deltas.values()), spec.unit)
    entry = _measurement(value, MetricProvenance.MEASURED_NATIVE, unit)
    entry.update(
        statistic="delta" if ranks_agree is None else "delta_one_rank",
        source=" + ".join((spec.family, *spec.plus)),
    )
    if spec.match:
        entry["match"] = dict(spec.match)
    if ranks_agree is not None:
        entry["ranks_agree"] = ranks_agree
    if spec.split_label is not None:
        split: dict[str, float] = {}
        for labels, delta in deltas.items():
            label_value = dict(labels).get(spec.split_label, "")
            split[label_value] = split.get(label_value, 0.0) + delta
        entry["breakdown"] = {
            spec.split_label: {
                k: _scaled(split[k], spec.unit)[0] for k in sorted(split)
            }
        }
    return entry


def _family_sum(
    names: str | tuple[str, ...],
    after: ServerMetricsSnapshot,
    deltas: Mapping[str, float],
) -> tuple[float, str] | None:
    names = (names,) if isinstance(names, str) else names
    if any(name not in after.families for name in names):
        return None
    return math.fsum(deltas.get(name, 0.0) for name in names), " + ".join(names)


def _point_sample(family: MetricFamily | None) -> dict[str, Any] | None:
    if family is None:
        return None
    values = [s.value for s in family.samples if _is_value_sample(family, s)]
    sample: dict[str, Any] = {"series": len(values)}
    if values:
        finite = [v for v in values if math.isfinite(v)]
        sample["min"] = encode_value(min(finite)) if finite else "NaN"
        sample["max"] = encode_value(max(finite)) if finite else "NaN"
    if len(values) == 1:
        sample["value"] = encode_value(values[0])
    return sample


def _kv_usage(
    profile: RuntimeProfile,
    before: ServerMetricsSnapshot,
    after: ServerMetricsSnapshot,
) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    for spec in profile.point_gauges:
        start = _point_sample(before.families.get(spec.family))
        end = _point_sample(after.families.get(spec.family))
        if start is None and end is None:
            continue
        usage[spec.key] = {
            "family": spec.family,
            "unit": spec.unit,
            "provenance": MetricProvenance.MEASURED_NATIVE.value,
            "statistic": "point_sample",
            "before": start,
            "after": end,
        }
    return usage


def diff_snapshots(
    before: ServerMetricsSnapshot,
    after: ServerMetricsSnapshot,
    *,
    expected_requests: int,
    runtime: str = "auto",
    assume_isolated: bool = False,
) -> dict[str, Any]:
    """Build server metrics evidence for the window between two scrapes."""
    if isinstance(expected_requests, bool) or expected_requests < 1:
        raise ServerMetricsError("expected_requests must be a positive integer")
    if before.endpoint != after.endpoint:
        raise ServerMetricsError(
            "snapshots come from different endpoints; a diff across servers "
            "is meaningless"
        )
    window_seconds = (
        _parse_iso(after.captured_at) - _parse_iso(before.captured_at)
    ).total_seconds()
    if window_seconds < 0:
        raise ServerMetricsError("the after snapshot was captured before the before")
    if runtime == "auto":
        runtime_name, detection = detect_runtime(after.families), "auto"
    elif runtime in PROFILES:
        runtime_name, detection = runtime, "explicit"
    else:
        raise ServerMetricsError(f"unknown runtime {runtime!r}")
    profile = PROFILES[runtime_name]
    state = _DiffState()

    for family_name, family in before.families.items():
        if family.type in ("counter", "histogram", "summary") and (
            family_name not in after.families
        ):
            state.resets.append(f"{family_name} disappeared")
    start_before = _family_total(before.families.get(_PROCESS_START), None)
    start_after = _family_total(after.families.get(_PROCESS_START), None)
    if (
        start_before is not None
        and start_after is not None
        and (start_before != start_after)
    ):
        state.resets.append(f"{_PROCESS_START} changed; the server restarted")

    counters: list[dict[str, Any]] = []
    histograms: list[dict[str, Any]] = []
    gauges: list[dict[str, Any]] = []
    for name, family in after.families.items():
        previous = before.families.get(name)
        if previous is not None and previous.type != family.type:
            state.resets.append(f"{name} changed type")
            continue
        if family.type == "counter":
            counters.extend(_counter_rows(name, previous, family, state))
        elif family.type in ("histogram", "summary"):
            histograms.extend(_histogram_rows(name, previous, family, state))
        elif name.endswith("_created"):
            # A series that first appears in the window is fine. One whose
            # creation time moved was recreated, so its counter restarted.
            created = {(s.name, s.labels): s.value for s in family.samples}
            for sample in previous.samples if previous is not None else ():
                if created.get((sample.name, sample.labels), sample.value) != (
                    sample.value
                ):
                    state.resets.append(f"{name} changed; a series was recreated")
                    break
        else:
            before_summary = _gauge_summary(previous)
            after_summary = _gauge_summary(family)
            if before_summary != after_summary or name in profile.in_flight_gauges:
                gauges.append(
                    {
                        "family": name,
                        "type": family.type,
                        "before": before_summary,
                        "after": after_summary,
                    }
                )

    request_checks: list[tuple[str, float]] = []
    for family_name, kind in profile.request_counters:
        if family_name not in after.families:
            continue
        if kind == "histogram":
            count = math.fsum(
                row["count_delta"] for row in histograms if row["family"] == family_name
            )
        else:
            count = math.fsum(
                row["delta"] for row in counters if row["family"] == family_name
            )
        request_checks.append((family_name, count))
    request_counter, observed = request_checks[0] if request_checks else (None, None)
    over = [(f, c) for f, c in request_checks if c > expected_requests]
    under = [(f, c) for f, c in request_checks if c < expected_requests]

    in_flight_before = math.fsum(
        _family_total(before.families.get(g), None) or 0.0
        for g in profile.in_flight_gauges
    )
    in_flight_after = math.fsum(
        _family_total(after.families.get(g), None) or 0.0
        for g in profile.in_flight_gauges
    )
    missing = [
        requirement
        for requirement in profile.stats_requirements
        if not any(_has_samples(after.families.get(f)) for f in requirement.families)
    ]
    blocking = [r for r in missing if r.required or not assume_isolated]
    in_flight_unknown = any(
        set(r.families) & set(profile.in_flight_gauges) for r in missing
    )

    if state.resets:
        status = COUNTER_RESET
        reason = "a counter went backwards or a series vanished: " + "; ".join(
            state.resets[:5]
        )
    elif blocking:
        status = STATS_NOT_ENABLED
        reason = "; ".join(
            f"{' and '.join(r.families)} not exposed, which needs {r.setting}: "
            f"{r.consequence}"
            for r in blocking
        )
        if any(r.required for r in blocking):
            reason += " (or no request has finished on this server yet)"
        if profile.stats_hint:
            reason += f". {profile.stats_hint}"
    elif over:
        status = CONTAMINATED
        reason = (
            f"{over[0][0]} counted {over[0][1]:g} requests in the window, "
            f"more than the {expected_requests} expected"
        )
    elif not in_flight_unknown and in_flight_before > 0:
        status = CONTAMINATED
        reason = (
            f"{in_flight_before:g} requests were already in flight at the before "
            "snapshot, so their work may land in this window"
        )
    elif under:
        status = INCOMPLETE
        reason = (
            f"{under[0][0]} counted {under[0][1]:g} requests in the window, "
            f"fewer than the {expected_requests} expected"
        )
    elif not in_flight_unknown and in_flight_after > 0:
        status = INCOMPLETE
        reason = (
            f"{in_flight_after:g} requests were still in flight at the after "
            "snapshot, so the window did not close cleanly"
        )
    elif request_checks and in_flight_unknown:
        status = OPERATOR_ASSERTED
        reason = (
            f"{', '.join(f for f, _ in request_checks)} each counted exactly "
            f"{_requests(expected_requests)} in the window; the server exposes "
            "no in-flight gauges, and the operator asserted that no other "
            "traffic overlapped the window"
        )
    elif request_checks:
        status = ATTRIBUTED
        reason = (
            f"{', '.join(f for f, _ in request_checks)} each counted exactly "
            f"{_requests(expected_requests)} in the window and nothing was in "
            "flight at either edge"
        )
    elif assume_isolated:
        status = OPERATOR_ASSERTED
        reason = (
            "the server exposes no request counter; the operator asserted that "
            "no other traffic reached it during the window"
        )
    else:
        status = UNVERIFIABLE
        reason = (
            "the server exposes no request counter, so other traffic in the "
            "window cannot be ruled out; pass --assume-isolated to assert it"
        )

    highlights: dict[str, Any] = {}
    if status in ATTRIBUTABLE_STATUSES:
        for spec in profile.histograms:
            entry = _highlight_histogram(spec, histograms)
            if entry is not None:
                highlights[spec.key] = entry
        deltas = {
            name: math.fsum(r["delta"] for r in counters if r["family"] == name)
            for name in {r["family"] for r in counters}
        }
        for counter_spec in profile.counters:
            entry = _counter_highlight(counter_spec, before, after, state.notes)
            if entry is not None:
                highlights[counter_spec.key] = entry
        for ratio in profile.ratios:
            numerator = _family_sum(ratio.numerator, after, deltas)
            denominator = _family_sum(ratio.denominator, after, deltas)
            if numerator is None or denominator is None or denominator[0] <= 0:
                continue
            entry = _measurement(
                numerator[0] / denominator[0], MetricProvenance.DERIVED, ratio.unit
            )
            entry.update(
                statistic="ratio_of_deltas",
                source=(
                    f"{numerator[1]} / ({denominator[1]})"
                    if " + " in denominator[1]
                    else f"{numerator[1]} / {denominator[1]}"
                ),
            )
            highlights[ratio.key] = entry

    notes = list(profile.notes) + state.notes
    if any(row["type"] == "summary" for row in histograms):
        notes.append(
            "summary quantiles are not differenced; only count and sum deltas "
            "are reported for summary families"
        )
    return {
        "kind": EVIDENCE_KIND,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "scope": "server_wide_window",
        "endpoint": after.endpoint,
        "runtime": runtime_name,
        "runtime_detection": detection,
        "before": {
            "captured_at": before.captured_at,
            "body_sha256": before.body_sha256,
            "scrape_duration_ms": before.scrape_duration_ms,
        },
        "after": {
            "captured_at": after.captured_at,
            "body_sha256": after.body_sha256,
            "scrape_duration_ms": after.scrape_duration_ms,
        },
        "window": _measurement(
            window_seconds * 1000.0, MetricProvenance.MEASURED_WALL_CLOCK, "ms"
        ),
        "attribution": {
            "status": status,
            "attributable": status in ATTRIBUTABLE_STATUSES,
            "reason": reason,
            "expected_requests": expected_requests,
            "observed_requests": None if observed is None else _as_count(observed),
            "request_counter": request_counter,
            "request_checks": [
                {"family": f, "observed": _as_count(c)} for f, c in request_checks
            ],
            "in_flight_before": (
                in_flight_before
                if profile.in_flight_gauges and not in_flight_unknown
                else None
            ),
            "in_flight_after": (
                in_flight_after
                if profile.in_flight_gauges and not in_flight_unknown
                else None
            ),
            "assume_isolated": assume_isolated,
            "missing_stats": [
                {
                    "setting": r.setting,
                    "families": list(r.families),
                    "required": r.required,
                }
                for r in missing
            ],
        },
        "highlights": highlights,
        "highlights_withheld_reason": (
            None if status in ATTRIBUTABLE_STATUSES else reason
        ),
        "speculative": speculative_evidence(
            profile.speculative,
            runtime_name,
            before.families,
            after.families,
            attributable=status in ATTRIBUTABLE_STATUSES,
            withheld_reason=reason,
        ),
        "kv_usage": _kv_usage(profile, before, after),
        "kept_labels": sorted(KEPT_LABELS),
        "counters": counters,
        "histograms": histograms,
        "gauges": gauges,
        "notes": notes,
    }


def capture_settled_after(
    before: ServerMetricsSnapshot,
    url: str,
    *,
    expected_requests: int,
    runtime: str = "auto",
    settle_timeout_seconds: float = 2.0,
    poll_interval_seconds: float = 0.2,
    fetcher: Fetcher = urllib_fetcher,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> ServerMetricsSnapshot:
    """Scrape until the window looks closed, or until the timeout.

    A server may record a finished request a moment after the client sees
    the end of the stream. Re-scraping briefly avoids reporting such a
    window as incomplete. The timeout only bounds the waiting; whatever the
    last scrape shows is still judged by :func:`diff_snapshots`.
    """
    deadline = clock() + max(0.0, settle_timeout_seconds)
    while True:
        after = capture_snapshot(url, fetcher=fetcher)
        evidence = diff_snapshots(
            before, after, expected_requests=expected_requests, runtime=runtime
        )
        attribution = evidence["attribution"]
        settled = (
            not attribution["in_flight_after"]
            and not any(m["required"] for m in attribution["missing_stats"])
            and all(
                check["observed"] >= expected_requests
                for check in attribution["request_checks"]
            )
        )
        if settled or clock() >= deadline:
            return after
        sleep(poll_interval_seconds)


def write_evidence(evidence: Mapping[str, Any], path: Path) -> None:
    atomic_write_text(path, json.dumps(evidence, indent=2, allow_nan=False) + "\n")
