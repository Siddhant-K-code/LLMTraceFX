"""The sweep run loop and point summaries, driven by an in-process sender."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer.load.api_worker import (
    TPOT_DERIVED,
    ApiRequestSpec,
    RequestRow,
)
from llmtracefx.optimizer.load.dispatcher import DispatchContext
from llmtracefx.optimizer.load.goodput import SloTargets
from llmtracefx.optimizer.load.sweep import (
    POINT_COMPLETE,
    POINT_FAILURES,
    ServerMetricsPairing,
    SweepRun,
    build_points,
    distribution,
    point_specs,
    run_points,
    sweep_summary,
)


def _sender(fail: set[int] | None = None, raise_on: set[int] | None = None) -> Any:
    fail = fail or set()
    raise_on = raise_on or set()

    def send(spec: ApiRequestSpec, context: DispatchContext) -> RequestRow:
        if context.index in raise_on:
            raise ConnectionResetError("peer went away")
        ok = context.index not in fail
        now = context.clock() - context.clock_origin
        return RequestRow(
            index=context.index,
            sent=True,
            dispatched_offset_s=context.dispatched_offset_s,
            completed_offset_s=max(now, context.dispatched_offset_s + 0.01),
            success=ok,
            failure_category=None if ok else "http_status",
            http_status=200 if ok else 503,
            ttft_ms=50.0 if ok else None,
            e2e_ms=100.0 if ok else None,
            tpot_ms=5.0 if ok else None,
            tpot_status=TPOT_DERIVED if ok else None,
            output_tokens=10 if ok else None,
        )

    return send


class _Metrics(ServerMetricsPairing):
    def __init__(self, fail_open_at: int | None = None) -> None:
        self.opened = 0
        self.closed: list[int] = []
        self.fail_open_at = fail_open_at

    def open(self) -> Any:
        self.opened += 1
        if self.opened - 1 == self.fail_open_at:
            raise OSError("connection refused")
        return "before"

    def close(self, handle: Any, *, succeeded: int, path: Path) -> dict[str, Any]:
        assert handle == "before"
        self.closed.append(succeeded)
        return {"status": "recorded", "file": path.name}


def _run(tmp_path: Path, send: Any, **kwargs: Any) -> SweepRun:
    points = kwargs.pop("points", build_points(concurrency=[1, 2, 3]))
    return SweepRun(
        points=points,
        send=send,
        specs_for=lambda point: point_specs(["p"], 4, point_index=point.index),
        output_dir=tmp_path,
        max_dispatch_lag_ms=10_000,
        **kwargs,
    )


def test_cooldown_runs_between_points_only(tmp_path: Path) -> None:
    sleeps: list[float] = []
    seen: list[str] = []
    run = _run(
        tmp_path,
        _sender(),
        cooldown_seconds=0.5,
        sleep=sleeps.append,
        on_point=lambda point: seen.append(point["label"]),
    )
    points = run_points(run)
    assert sleeps == [0.5, 0.5]
    assert seen == ["concurrency=1", "concurrency=2", "concurrency=3"]
    assert all(point["status"] == POINT_COMPLETE for point in points)
    assert sweep_summary(points)["points_with_failures"] == []


def test_client_errors_and_http_failures_are_counted_separately(
    tmp_path: Path,
) -> None:
    run = _run(
        tmp_path,
        _sender(fail={0}, raise_on={1}),
        points=build_points(concurrency=[2]),
        slo=SloTargets(e2e_ms=1000.0),
    )
    (point,) = run_points(run)
    assert point["status"] == POINT_FAILURES
    assert point["counts"]["failure_categories"] == {
        "client_error": 1,
        "http_status": 1,
    }
    assert point["latency"]["e2e_ms"]["count"] == 2
    assert point["goodput"]["counts"]["failed"] == 2
    assert point["goodput"]["request_goodput_fraction"]["value"] == 0.5
    client_row = point["requests"][1]
    assert client_row["client_error"] == "ConnectionResetError"
    assert client_row["success"] is None


def test_server_metrics_errors_are_recorded_and_the_sweep_continues(
    tmp_path: Path,
) -> None:
    metrics = _Metrics(fail_open_at=1)
    run = _run(
        tmp_path,
        _sender(),
        server_metrics=metrics,
        describe_error=lambda exc: type(exc).__name__,
    )
    points = run_points(run)
    assert [point["server_metrics"]["status"] for point in points] == [
        "recorded",
        "error",
        "recorded",
    ]
    assert points[1]["server_metrics"]["error"] == "before snapshot failed: OSError"
    assert metrics.closed == [4, 4]
    assert points[2]["server_metrics"]["file"] == "server_metrics_point_2.json"
    assert sweep_summary(points)["server_metrics_errors"] == ["concurrency=2"]


def test_server_metrics_window_is_skipped_when_nothing_succeeded(
    tmp_path: Path,
) -> None:
    metrics = _Metrics()
    run = _run(
        tmp_path,
        _sender(fail={0, 1, 2, 3}),
        points=build_points(concurrency=[1]),
        server_metrics=metrics,
    )
    (point,) = run_points(run)
    assert point["server_metrics"]["status"] == "skipped"
    assert metrics.closed == []


def test_rate_points_share_one_seed_and_constant_drops_it() -> None:
    poisson = build_points(request_rates=[1.0, 2.0], arrival="poisson", seed=9)
    assert [point.policy.seed for point in poisson] == [9, 9]  # type: ignore[attr-defined]
    constant = build_points(request_rates=[1.0], arrival="constant", seed=9)
    assert constant[0].policy.seed is None  # type: ignore[attr-defined]
    assert constant[0].label == "rate=1/s"
    with pytest.raises(ValueError):
        build_points(concurrency=[1], request_rates=[1.0])


def test_distribution_uses_nearest_rank() -> None:
    summary = distribution([float(value) for value in range(1, 11)], "measured")
    assert summary["p50"] == 5.0
    assert summary["p90"] == 9.0
    assert summary["p99"] == 10.0
    assert summary["mean"] == 5.5
    empty = distribution([], "measured")
    assert empty["count"] == 0
    assert empty["p50"] is None
