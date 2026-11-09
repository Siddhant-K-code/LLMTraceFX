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

EVIDENCE_KIND = "llmtracefx.server_metrics.evidence"
EVIDENCE_SCHEMA_VERSION = "1"

ATTRIBUTED = "attributed"
OPERATOR_ASSERTED = "operator_asserted"
CONTAMINATED = "contaminated"
INCOMPLETE = "incomplete"
COUNTER_RESET = "counter_reset"
UNVERIFIABLE = "unverifiable"
ATTRIBUTABLE_STATUSES = frozenset({ATTRIBUTED, OPERATOR_ASSERTED})

# Labels kept when series are collapsed. Everything else (model names that
# may be local paths, engine indexes, pod names) is summed away so the
# evidence is safe to share.
KEPT_LABELS = frozenset(
    {"le", "quantile", "finished_reason", "is_streaming", "cache_source", "reason"}
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


@dataclass(frozen=True)
class RatioHighlight:
    key: str
    numerator: str
    denominator: str
    unit: str


@dataclass(frozen=True)
class RuntimeProfile:
    name: str
    family_prefix: str | None
    request_counters: tuple[tuple[str, str], ...] = ()
    """``(family, kind)`` in preference order. ``kind`` is ``histogram`` (use
    the ``_count`` delta) or ``counter``."""
    in_flight_gauges: tuple[str, ...] = ()
    histograms: tuple[HistogramHighlight, ...] = ()
    counters: tuple[CounterHighlight, ...] = ()
    ratios: tuple[RatioHighlight, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)


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
        ),
        ratios=(
            RatioHighlight(
                "prefix_cache_hit_rate",
                "vllm:prefix_cache_hits",
                "vllm:prefix_cache_queries",
                "ratio",
            ),
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
        ),
        ratios=(
            RatioHighlight(
                "cached_prompt_token_ratio",
                "sglang:cached_tokens",
                "sglang:prompt_tokens",
                "ratio",
            ),
        ),
    ),
    "llamacpp": RuntimeProfile(
        name="llamacpp",
        family_prefix="llamacpp:",
        in_flight_gauges=("llamacpp:requests_processing", "llamacpp:requests_deferred"),
        counters=(
            CounterHighlight("prompt_tokens", "llamacpp:prompt_tokens", "tokens"),
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
        ),
        in_flight_gauges=("tgi_queue_size",),
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

    observed: float | None = None
    request_counter: str | None = None
    for family_name, kind in profile.request_counters:
        if family_name not in after.families:
            continue
        request_counter = family_name
        if kind == "histogram":
            observed = math.fsum(
                row["count_delta"] for row in histograms if row["family"] == family_name
            )
        else:
            observed = math.fsum(
                row["delta"] for row in counters if row["family"] == family_name
            )
        break

    in_flight_before = math.fsum(
        _family_total(before.families.get(g), None) or 0.0
        for g in profile.in_flight_gauges
    )
    in_flight_after = math.fsum(
        _family_total(after.families.get(g), None) or 0.0
        for g in profile.in_flight_gauges
    )

    if state.resets:
        status = COUNTER_RESET
        reason = "a counter went backwards or a series vanished: " + "; ".join(
            state.resets[:5]
        )
    elif observed is not None and observed > expected_requests:
        status = CONTAMINATED
        reason = (
            f"{request_counter} counted {observed:g} requests in the window, "
            f"more than the {expected_requests} expected"
        )
    elif in_flight_before > 0:
        status = CONTAMINATED
        reason = (
            f"{in_flight_before:g} requests were already in flight at the before "
            "snapshot, so their work may land in this window"
        )
    elif observed is not None and observed < expected_requests:
        status = INCOMPLETE
        reason = (
            f"{request_counter} counted {observed:g} requests in the window, "
            f"fewer than the {expected_requests} expected"
        )
    elif in_flight_after > 0:
        status = INCOMPLETE
        reason = (
            f"{in_flight_after:g} requests were still in flight at the after "
            "snapshot, so the window did not close cleanly"
        )
    elif observed is not None:
        status = ATTRIBUTED
        reason = (
            f"{request_counter} counted exactly {_requests(expected_requests)} "
            "in the window and nothing was in flight at either edge"
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
            if counter_spec.family not in after.families:
                continue
            value, unit = _scaled(
                deltas.get(counter_spec.family, 0.0), counter_spec.unit
            )
            entry = _measurement(value, MetricProvenance.MEASURED_NATIVE, unit)
            entry.update(statistic="delta", source=counter_spec.family)
            highlights[counter_spec.key] = entry
        for ratio in profile.ratios:
            if (
                ratio.numerator not in after.families
                or ratio.denominator not in after.families
            ):
                continue
            denominator = deltas.get(ratio.denominator, 0.0)
            if denominator <= 0:
                continue
            entry = _measurement(
                deltas.get(ratio.numerator, 0.0) / denominator,
                MetricProvenance.DERIVED,
                ratio.unit,
            )
            entry.update(
                statistic="ratio_of_deltas",
                source=f"{ratio.numerator} / {ratio.denominator}",
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
            "in_flight_before": in_flight_before if profile.in_flight_gauges else None,
            "in_flight_after": in_flight_after if profile.in_flight_gauges else None,
            "assume_isolated": assume_isolated,
        },
        "highlights": highlights,
        "highlights_withheld_reason": (
            None if status in ATTRIBUTABLE_STATUSES else reason
        ),
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
        observed = attribution["observed_requests"]
        settled = not attribution["in_flight_after"] and (
            observed is None or observed >= expected_requests
        )
        if settled or clock() >= deadline:
            return after
        sleep(poll_interval_seconds)


def write_evidence(evidence: Mapping[str, Any], path: Path) -> None:
    atomic_write_text(path, json.dumps(evidence, indent=2, allow_nan=False) + "\n")
