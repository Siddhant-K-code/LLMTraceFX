"""Concurrency and request rate sweeps: one dispatch per load point.

A sweep runs the same request mix at several offered loads and summarises
each point: client-observed latency distributions, throughput, goodput
against an optional SLO, the dispatcher's scheduling diagnostics and,
when asked for, the server's own Prometheus window.

The result is a single ``sweep.json`` of kind ``llmtracefx.load_sweep``.
Its ``attribution`` block states what a sweep can and cannot claim.
No per-request ``record.json`` is written, so sweep requests never reach a
reader that expects isolated single-request evidence.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..schema import MetricProvenance, utc_now_iso
from ..server_metrics import (
    ServerMetricsSnapshot,
    capture_settled_after,
    capture_snapshot,
    diff_snapshots,
    write_evidence,
)
from .api_worker import (
    ROW_FIELD_PROVENANCE,
    TPOT_DERIVED,
    ApiRequestSpec,
    RequestRow,
    request_rows,
)
from .artifact import SWEEP_KIND, SWEEP_PLAN_KIND, SWEEP_SCHEMA_VERSION
from .dispatcher import (
    DEFAULT_MAX_DISPATCH_LAG_MS,
    DISPATCHER_NAME,
    DISPATCHER_VERSION,
    ClosedLoop,
    DispatchContext,
    DispatchReport,
    LoadPolicy,
    OpenLoopRate,
    SessionArrivals,
    _nearest_rank,
    dispatch,
    planned_offsets,
)
from .goodput import SloTargets, measured_window_seconds, summarize_goodput

MODE_CONCURRENCY = "concurrency"
MODE_REQUEST_RATE = "request_rate"

POINT_COMPLETE = "complete"
POINT_FAILURES = "failures"
POINT_INCOMPLETE = "incomplete"

ATTRIBUTION: dict[str, Any] = {
    "measurement_mode": "concurrent_load",
    "requests_isolated": False,
    "can_claim": [
        (
            "the latency distribution and goodput this client observed for this "
            "request mix, at this offered load, against this endpoint, from this "
            "client host"
        ),
        (
            "each per-request value is a real wall clock reading for that request "
            "while it shared the server with the other requests of its point"
        ),
        (
            "server-side counters and histograms for a point's window, when "
            "server metrics were paired and the window is attributed"
        ),
    ],
    "cannot_claim": [
        "what any single request would cost on an idle server",
        (
            "how a request's time splits into queueing, prefill and decode, from "
            "client timings alone"
        ),
        (
            "that a latency change between points is caused by the server rather "
            "than the client or the network; the scheduling diagnostics are there "
            "to rule out the client"
        ),
        (
            "comparability with single-request records; gate, compare and export "
            "refuse sweep artifacts"
        ),
    ],
    "provenance_note": (
        "Every value keeps an existing MetricProvenance label. What separates "
        "a sweep from single-request evidence is the artifact kind and the "
        "load_context on each point, not a new provenance value."
    ),
}

_LATENCY_FIELDS = ("ttft_ms", "first_content_ms", "tpot_ms", "e2e_ms")
_PERCENTILES = (50, 90, 95, 99)
_DERIVED = MetricProvenance.DERIVED.value


@dataclass(frozen=True)
class SweepPoint:
    """One offered load in a sweep."""

    index: int
    policy: LoadPolicy

    @property
    def label(self) -> str:
        if isinstance(self.policy, ClosedLoop):
            return f"concurrency={self.policy.concurrency}"
        if isinstance(self.policy, OpenLoopRate):
            return f"rate={self.policy.rate_per_second:g}/s"
        if isinstance(self.policy, SessionArrivals):
            return f"sessions={len(self.policy.turns)}"
        return f"timed={len(self.policy.offsets_seconds)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "policy": self.policy.to_dict(),
            "max_in_flight": self.policy.max_in_flight,
        }


def build_points(
    *,
    concurrency: Sequence[int] | None = None,
    request_rates: Sequence[float] | None = None,
    arrival: str = "poisson",
    seed: int | None = None,
    max_in_flight: int | None = None,
) -> list[SweepPoint]:
    """Points in the order given. Exactly one of the two load lists is used.

    Every rate point uses the same seed, so a Poisson schedule differs
    between points only by its rate.
    """
    if (concurrency is None) == (request_rates is None):
        raise ValueError("give either concurrency levels or request rates")
    if concurrency is not None:
        if not concurrency:
            raise ValueError("give at least one concurrency level")
        return [
            SweepPoint(index, ClosedLoop(level))
            for index, level in enumerate(concurrency)
        ]
    assert request_rates is not None
    if not request_rates:
        raise ValueError("give at least one request rate")
    extra: dict[str, Any] = {}
    if max_in_flight is not None:
        extra["max_in_flight"] = max_in_flight
    return [
        SweepPoint(
            index,
            OpenLoopRate(
                rate,
                distribution=arrival,
                seed=seed if arrival == "poisson" else None,
                **extra,
            ),
        )
        for index, rate in enumerate(request_rates)
    ]


def point_specs(
    prompts: Sequence[str],
    count: int,
    *,
    point_index: int,
    system_prompt: str | None = None,
) -> list[ApiRequestSpec]:
    """``count`` specs cycling through ``prompts`` in order, with sweep-unique run ids."""
    if not prompts:
        raise ValueError("give at least one prompt")
    return [
        ApiRequestSpec(
            prompt=prompts[index % len(prompts)],
            system_prompt=system_prompt,
            run_id_suffix=f"p{point_index:02d}-{index:06d}",
        )
        for index in range(count)
    ]


def distribution(values: Sequence[float], sample_provenance: str) -> dict[str, Any]:
    """Nearest-rank percentiles, mean, min and max; ``None`` values when empty."""
    ordered = sorted(values)
    summary: dict[str, Any] = {
        "count": len(ordered),
        "provenance": _DERIVED,
        "sample_provenance": sample_provenance,
        "statistic": "nearest_rank",
    }
    summary["mean"] = math.fsum(ordered) / len(ordered) if ordered else None
    summary["min"] = ordered[0] if ordered else None
    for percentile in _PERCENTILES:
        summary[f"p{percentile}"] = (
            _nearest_rank(ordered, percentile) if ordered else None
        )
    summary["max"] = ordered[-1] if ordered else None
    return summary


def _succeeded(row: RequestRow) -> bool:
    return row.sent and row.client_error is None and row.success is True


def latency_summary(rows: Sequence[RequestRow]) -> dict[str, Any]:
    """Distributions over succeeded requests only; failed requests have no comparable latency."""
    succeeded = [row for row in rows if _succeeded(row)]
    result: dict[str, Any] = {"covers": "succeeded requests with a value"}
    for name in _LATENCY_FIELDS:
        if name == "tpot_ms":
            values = [
                row.tpot_ms
                for row in succeeded
                if row.tpot_status == TPOT_DERIVED and row.tpot_ms is not None
            ]
        else:
            values = [
                value for row in succeeded if (value := getattr(row, name)) is not None
            ]
        result[name] = distribution(values, ROW_FIELD_PROVENANCE[name])
    return result


def _withheld(value: float | None, reason: str, provenance: str) -> dict[str, Any]:
    return {
        "value": value,
        "provenance": provenance,
        "withheld_reason": None if value is not None else reason,
    }


def throughput_summary(rows: Sequence[RequestRow]) -> dict[str, Any]:
    window = measured_window_seconds(rows)
    sent = [row for row in rows if row.sent]
    succeeded = [row for row in rows if _succeeded(row)]
    tokens = [row.output_tokens for row in succeeded]
    no_window = "no measured window"
    rate = None if window is None else len(sent) / window
    succeeded_rate = None if window is None else len(succeeded) / window
    token_rate: float | None = None
    token_reason = no_window
    if window is not None:
        if any(count is None for count in tokens):
            token_reason = (
                "the provider did not report output tokens for every succeeded request"
            )
        else:
            token_rate = sum(count for count in tokens if count is not None) / window
    return {
        "window_seconds": _withheld(
            window, no_window, MetricProvenance.MEASURED_WALL_CLOCK.value
        ),
        "completed_requests_per_second": _withheld(rate, no_window, _DERIVED),
        "succeeded_requests_per_second": _withheld(succeeded_rate, no_window, _DERIVED),
        "succeeded_output_tokens_per_second": _withheld(
            token_rate, token_reason, _DERIVED
        ),
    }


def _span_rate(offsets: Sequence[float]) -> float | None:
    """Arrivals per second between the first and last offset."""
    if len(offsets) < 2 or offsets[-1] <= offsets[0]:
        return None
    return (len(offsets) - 1) / (offsets[-1] - offsets[0])


def _dispatch_rate(
    point: SweepPoint, report: DispatchReport[RequestRow], rows: Sequence[RequestRow]
) -> dict[str, Any] | None:
    if not isinstance(point.policy, OpenLoopRate):
        return None
    starts = sorted(
        row.dispatched_offset_s
        for row in rows
        if row.sent and row.dispatched_offset_s is not None
    )
    planned = sorted(report.planned_offsets_s or ())
    return {
        "target_rate_per_second": point.policy.rate_per_second,
        "planned_schedule_rate_per_second": _withheld(
            _span_rate(planned), "fewer than two planned arrivals", _DERIVED
        ),
        "achieved_dispatch_rate_per_second": _withheld(
            _span_rate(starts), "fewer than two dispatches", _DERIVED
        ),
        "note": (
            "a Poisson schedule's own rate over a finite run differs from the "
            "target; compare the achieved rate with the planned schedule's rate"
        ),
    }


def summarize_point(
    point: SweepPoint,
    report: DispatchReport[RequestRow],
    *,
    slo: SloTargets | None,
) -> dict[str, Any]:
    rows = request_rows(report)
    sent = [row for row in rows if row.sent]
    failed = [row for row in sent if not _succeeded(row)]
    unsent = len(rows) - len(sent)
    categories: Counter[str] = Counter()
    for row in failed:
        categories[
            (
                "client_error"
                if row.client_error is not None
                else row.failure_category or "unknown"
            )
        ] += 1
    finish_codes = Counter(
        row.finish_reason_code or "unreported" for row in rows if _succeeded(row)
    )
    status = (
        POINT_INCOMPLETE if unsent else POINT_FAILURES if failed else POINT_COMPLETE
    )
    dispatch_dict = report.to_dict()
    diagnostics = dispatch_dict.pop("diagnostics")
    scheduling = {**diagnostics, "dispatch_rate": _dispatch_rate(point, report, rows)}
    return {
        **point.to_dict(),
        "status": status,
        "load_context": {
            "measurement_mode": ATTRIBUTION["measurement_mode"],
            "requests_isolated": False,
            "policy": point.policy.to_dict(),
            "max_in_flight": report.max_in_flight,
            "requests": len(rows),
            "started_at_utc": report.started_at_utc,
        },
        "counts": {
            "requests": len(rows),
            "sent": len(sent),
            "unsent": unsent,
            "succeeded": len(sent) - len(failed),
            "failed": len(failed),
            "failure_categories": dict(sorted(categories.items())),
            "finish_reason_codes": dict(sorted(finish_codes.items())),
        },
        "latency": latency_summary(rows),
        "throughput": throughput_summary(rows),
        "goodput": None if slo is None else summarize_goodput(rows, slo).to_dict(),
        "scheduling": scheduling,
        "dispatch": dispatch_dict,
        "requests": [row.to_dict() for row in rows],
    }


class ServerMetricsPairing:
    """Opens and closes one Prometheus window around each point.

    ``open`` returns a handle for ``close``. ``close`` returns the summary to
    embed in the point, which names the evidence file and its sha256.
    Implementations raise on failure; the sweep records the error and goes on.
    """

    def open(self) -> Any:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self, handle: Any, *, succeeded: int, path: Path) -> dict[str, Any]:
        raise NotImplementedError  # pragma: no cover - interface


class PrometheusPairing(ServerMetricsPairing):
    """Server metrics windows from a Prometheus ``/metrics`` endpoint.

    The window is judged by ``diff_snapshots`` with ``expected_requests``
    set to the requests that succeeded. Its attribution status is reported
    as is. A failed request that the server still counted makes the window
    ``contaminated``, which is the honest outcome.
    """

    def __init__(
        self,
        url: str,
        *,
        runtime: str = "auto",
        settle_timeout_seconds: float = 2.0,
        assume_isolated: bool = False,
    ) -> None:
        self.url = url
        self.runtime = runtime
        self.settle_timeout_seconds = settle_timeout_seconds
        self.assume_isolated = assume_isolated

    def open(self) -> ServerMetricsSnapshot:
        return capture_snapshot(self.url)

    def close(self, handle: Any, *, succeeded: int, path: Path) -> dict[str, Any]:
        after = capture_settled_after(
            handle,
            self.url,
            expected_requests=succeeded,
            runtime=self.runtime,
            settle_timeout_seconds=self.settle_timeout_seconds,
        )
        evidence = diff_snapshots(
            handle,
            after,
            expected_requests=succeeded,
            runtime=self.runtime,
            assume_isolated=self.assume_isolated,
        )
        write_evidence(evidence, path)
        attribution = evidence["attribution"]
        return {
            "status": "recorded",
            "file": path.name,
            "sha256": file_sha256(path),
            "runtime": evidence["runtime"],
            "attribution_status": attribution["status"],
            "attributable": attribution["attributable"],
            "reason": attribution["reason"],
            "expected_requests": attribution["expected_requests"],
            "observed_requests": attribution["observed_requests"],
        }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def server_metrics_file_name(point_index: int) -> str:
    return f"server_metrics_point_{point_index}.json"


def _exception_type_name(exc: BaseException) -> str:
    return type(exc).__name__


@dataclass
class SweepRun:
    """Everything one sweep needs besides the request mix."""

    points: Sequence[SweepPoint]
    send: Callable[[ApiRequestSpec, DispatchContext], RequestRow]
    specs_for: Callable[[SweepPoint], Sequence[ApiRequestSpec]]
    output_dir: Path
    slo: SloTargets | None = None
    cooldown_seconds: float = 0.0
    max_dispatch_lag_ms: float = DEFAULT_MAX_DISPATCH_LAG_MS
    server_metrics: ServerMetricsPairing | None = None
    describe_error: Callable[[BaseException], str] = field(default=_exception_type_name)
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.perf_counter
    on_point: Callable[[dict[str, Any]], None] | None = None
    client_cpu_withheld_reason: str | None = None


def run_points(run: SweepRun) -> list[dict[str, Any]]:
    """Dispatch each point in order and return the point summaries.

    A server metrics error is recorded on its point and does not stop the
    sweep. ``KeyboardInterrupt`` propagates after the dispatcher has joined
    its workers; points finished before it are passed to ``on_point``.
    """
    results: list[dict[str, Any]] = []
    for position, point in enumerate(run.points):
        if position and run.cooldown_seconds > 0:
            run.sleep(run.cooldown_seconds)
        specs = list(run.specs_for(point))
        handle = None
        metrics_error: str | None = None
        if run.server_metrics is not None:
            try:
                handle = run.server_metrics.open()
            except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                metrics_error = f"before snapshot failed: {run.describe_error(exc)}"
        report = dispatch(
            specs,
            run.send,
            point.policy,
            clock=run.clock,
            max_dispatch_lag_ms=run.max_dispatch_lag_ms,
            client_cpu_withheld_reason=run.client_cpu_withheld_reason,
        )
        summary = summarize_point(point, report, slo=run.slo)
        if run.server_metrics is not None:
            summary["server_metrics"] = _close_metrics(
                run, point, handle, metrics_error, summary["counts"]["succeeded"]
            )
        results.append(summary)
        if run.on_point is not None:
            run.on_point(summary)
    return results


def _close_metrics(
    run: SweepRun,
    point: SweepPoint,
    handle: Any,
    error: str | None,
    succeeded: int,
) -> dict[str, Any]:
    assert run.server_metrics is not None
    if error is not None:
        return {"status": "error", "error": error}
    if succeeded < 1:
        return {
            "status": "skipped",
            "reason": "no request succeeded, so the window has nothing to attribute",
        }
    path = run.output_dir / server_metrics_file_name(point.index)
    try:
        return run.server_metrics.close(handle, succeeded=succeeded, path=path)
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        return {
            "status": "error",
            "error": f"after snapshot failed: {run.describe_error(exc)}",
        }


def sweep_summary(points: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def labels(predicate: Callable[[Mapping[str, Any]], bool]) -> list[str]:
        return [point["label"] for point in points if predicate(point)]

    return {
        "points": len(points),
        "incomplete_points": labels(lambda p: p["status"] == POINT_INCOMPLETE),
        "points_with_failures": labels(lambda p: p["status"] == POINT_FAILURES),
        "degraded_points": labels(
            lambda p: p["scheduling"]["status"] != "ok",
        ),
        "server_metrics_errors": labels(
            lambda p: (p.get("server_metrics") or {}).get("status") == "error"
        ),
    }


def sweep_document(
    *,
    mode: str,
    load: Mapping[str, Any],
    slo: SloTargets | None,
    request_plan: Mapping[str, Any],
    workload: Mapping[str, Any],
    points: Sequence[Mapping[str, Any]],
    planned_points: int,
    finished: bool,
    interrupted: bool = False,
    started_at_utc: str,
    attribution: Mapping[str, Any] = ATTRIBUTION,
    field_provenance: Mapping[str, str] = ROW_FIELD_PROVENANCE,
) -> dict[str, Any]:
    """The ``sweep.json`` document.

    ``attribution`` and ``field_provenance`` default to the OpenAI-compatible
    HTTP worker's. A backend whose rows differ (another provenance for token
    counts, a concurrency limit) passes its own.
    """
    summary = sweep_summary(points)
    summary["planned_points"] = planned_points
    summary["finished"] = finished
    summary["interrupted"] = interrupted
    summary["clean"] = (
        finished
        and not interrupted
        and len(points) == planned_points
        and not summary["incomplete_points"]
        and not summary["points_with_failures"]
        and not summary["server_metrics_errors"]
    )
    return {
        "kind": SWEEP_KIND,
        "schema_version": SWEEP_SCHEMA_VERSION,
        "started_at_utc": started_at_utc,
        "written_at_utc": utc_now_iso(),
        "attribution": dict(attribution),
        "dispatcher": {"name": DISPATCHER_NAME, "version": DISPATCHER_VERSION},
        "mode": mode,
        "load": dict(load),
        "slo": None if slo is None else slo.to_dict(),
        "workload": dict(workload),
        "request_plan": dict(request_plan),
        "field_provenance": dict(field_provenance),
        "summary": summary,
        "points": list(points),
    }


def sweep_plan_document(
    *,
    mode: str,
    load: Mapping[str, Any],
    slo: SloTargets | None,
    request_plan: Mapping[str, Any],
    workload: Mapping[str, Any],
    points: Sequence[SweepPoint],
    requests_per_point: int,
    attribution: Mapping[str, Any] = ATTRIBUTION,
) -> dict[str, Any]:
    planned = []
    for point in points:
        offsets = planned_offsets(point.policy, requests_per_point)
        planned.append(
            {
                **point.to_dict(),
                "requests": requests_per_point,
                "planned_offsets_s": None if offsets is None else list(offsets),
                "planned_last_arrival_s": None if offsets is None else offsets[-1],
            }
        )
    return {
        "kind": SWEEP_PLAN_KIND,
        "schema_version": SWEEP_SCHEMA_VERSION,
        "dry_run": True,
        "network_request_performed": False,
        "attribution": dict(attribution),
        "dispatcher": {"name": DISPATCHER_NAME, "version": DISPATCHER_VERSION},
        "mode": mode,
        "load": dict(load),
        "slo": None if slo is None else slo.to_dict(),
        "workload": dict(workload),
        "request_plan": dict(request_plan),
        "total_requests": requests_per_point * len(points),
        "points": planned,
    }
