"""Tests for the transport-free load dispatcher.

``send`` is always a local function, so nothing leaves the process. Timing
assertions use events and barriers rather than tight wall-clock bounds.
"""

from __future__ import annotations

import itertools
import json
import statistics
import threading
from typing import Any

import pytest

from llmtracefx.optimizer.load.dispatcher import (
    DISPATCHER_NAME,
    DISPATCHER_VERSION,
    MAX_CONCURRENCY,
    REASON_CAP_DEFERRED,
    REASON_CPU,
    STATUS_DEGRADED,
    STATUS_OK,
    ClosedLoop,
    DispatchConfigError,
    DispatchContext,
    LagSummary,
    OpenLoopRate,
    TimedArrivals,
    dispatch,
    planned_offsets,
)

_TIMEOUT = 5.0


def _echo(item: Any, context: DispatchContext) -> Any:
    return item


# --- Policies and schedules --------------------------------------------------


def test_constant_rate_spaces_arrivals_evenly() -> None:
    offsets = planned_offsets(OpenLoopRate(4.0, "constant"), 5)
    assert offsets == (0.0, 0.25, 0.5, 0.75, 1.0)


def test_poisson_schedule_is_seeded_and_has_the_requested_mean_gap() -> None:
    policy = OpenLoopRate(10.0, "poisson", seed=7)
    first = planned_offsets(policy, 2000)
    assert first == planned_offsets(OpenLoopRate(10.0, "poisson", seed=7), 2000)
    assert first != planned_offsets(OpenLoopRate(10.0, "poisson", seed=8), 2000)
    assert first is not None and first[0] == 0.0
    gaps = [b - a for a, b in itertools.pairwise(first)]
    assert all(gap >= 0 for gap in gaps)
    assert statistics.fmean(gaps) == pytest.approx(0.1, rel=0.1)


def test_closed_loop_has_no_schedule_and_timed_must_match_the_count() -> None:
    assert planned_offsets(ClosedLoop(2), 10) is None
    assert planned_offsets(TimedArrivals((0, 0, 1.5)), 3) == (0.0, 0.0, 1.5)
    with pytest.raises(DispatchConfigError, match="3 offsets for 2 requests"):
        planned_offsets(TimedArrivals((0, 0, 1)), 2)


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: ClosedLoop(0), "concurrency"),
        (lambda: ClosedLoop(MAX_CONCURRENCY + 1), "concurrency"),
        (lambda: ClosedLoop(True), "concurrency must be an integer"),
        (lambda: OpenLoopRate(0), "rate_per_second"),
        (lambda: OpenLoopRate(float("inf"), seed=1), "rate_per_second"),
        (lambda: OpenLoopRate(1.0), "need a seed"),
        (lambda: OpenLoopRate(1.0, "constant", seed=1), "only applies to poisson"),
        (lambda: OpenLoopRate(1.0, "bursty", seed=1), "distribution"),
        (lambda: OpenLoopRate(1.0, "constant", max_in_flight=0), "max_in_flight"),
        (lambda: TimedArrivals(()), "must not be empty"),
        (lambda: TimedArrivals((0, 2, 1)), "non-decreasing"),
        (lambda: TimedArrivals((0, float("nan"))), "finite"),
        (lambda: TimedArrivals((-1,)), "at least 0"),
        (lambda: TimedArrivals("012"), "sequence of numbers"),  # type: ignore[arg-type]
    ],
)
def test_invalid_policies_are_refused(build: Any, match: str) -> None:
    with pytest.raises(DispatchConfigError, match=match):
        build()


def test_dispatch_refuses_empty_work_and_a_bad_threshold() -> None:
    with pytest.raises(DispatchConfigError, match="request count"):
        dispatch([], _echo, ClosedLoop(1))
    with pytest.raises(DispatchConfigError, match="max_dispatch_lag_ms"):
        dispatch([1], _echo, ClosedLoop(1), max_dispatch_lag_ms=-1)


# --- Closed loop -------------------------------------------------------------


def test_closed_loop_holds_concurrency_and_records_slot_refill() -> None:
    barrier = threading.Barrier(3, timeout=_TIMEOUT)

    def send(item: int, context: DispatchContext) -> int:
        if item < 3:
            # Fails with BrokenBarrierError unless the first three start together.
            barrier.wait()
        return item * 10

    report = dispatch(list(range(9)), send, ClosedLoop(3))

    assert [outcome.result for outcome in report.outcomes] == [i * 10 for i in range(9)]
    assert all(outcome.client_error is None for outcome in report.outcomes)
    diagnostics = report.diagnostics
    assert diagnostics.mode == "closed_loop"
    assert diagnostics.max_observed_in_flight == 3
    assert report.max_in_flight == 3
    assert diagnostics.dispatch_lag is None
    assert diagnostics.slot_refill is not None
    assert diagnostics.slot_refill.count == 6
    assert report.planned_offsets_s is None
    assert all(outcome.scheduled_offset_s is None for outcome in report.outcomes)


def test_send_errors_are_recorded_by_type_and_do_not_end_the_run() -> None:
    def send(item: int, context: DispatchContext) -> int:
        if item == 1:
            raise ValueError("prompt text that must not be kept")
        return item

    report = dispatch([0, 1, 2], send, ClosedLoop(1))

    assert [outcome.client_error for outcome in report.outcomes] == [
        None,
        "ValueError",
        None,
    ]
    assert report.outcomes[1].result is None
    assert "prompt text" not in json.dumps(
        [outcome.timing_dict() for outcome in report.outcomes]
    )


# --- Scheduled arrivals ------------------------------------------------------


def test_equal_offsets_are_released_together() -> None:
    barrier = threading.Barrier(3, timeout=_TIMEOUT)

    def send(item: int, context: DispatchContext) -> int:
        if item < 3:
            barrier.wait()
        return context.in_flight_at_dispatch

    report = dispatch([0, 1, 2, 3], send, TimedArrivals((0, 0, 0, 0.02)))

    assert all(outcome.client_error is None for outcome in report.outcomes)
    assert report.diagnostics.deferred_by_in_flight_cap == 0
    assert report.diagnostics.max_observed_in_flight >= 3
    assert [o.scheduled_offset_s for o in report.outcomes] == [0, 0, 0, 0.02]


def test_in_flight_cap_defers_and_the_wait_becomes_dispatch_lag() -> None:
    release = threading.Event()
    both_in_flight = threading.Barrier(2, timeout=_TIMEOUT)

    def send(item: int, context: DispatchContext) -> int:
        if item < 2:
            both_in_flight.wait()
            release.wait(_TIMEOUT)
        return item

    timer = threading.Timer(0.1, release.set)
    timer.start()
    try:
        report = dispatch([0, 1, 2], send, TimedArrivals((0, 0, 0), max_in_flight=2))
    finally:
        timer.cancel()

    diagnostics = report.diagnostics
    assert diagnostics.deferred_by_in_flight_cap == 1
    assert diagnostics.max_observed_in_flight == 2
    assert diagnostics.status == STATUS_DEGRADED
    assert REASON_CAP_DEFERRED in diagnostics.reasons
    deferred_lag = report.outcomes[2].dispatch_lag_ms
    assert deferred_lag is not None and deferred_lag >= 50.0
    assert diagnostics.dispatch_lag is not None
    assert diagnostics.dispatch_lag.max_ms == pytest.approx(deferred_lag)


def test_open_loop_never_dispatches_before_its_scheduled_time() -> None:
    report = dispatch(
        list(range(5)),
        _echo,
        OpenLoopRate(100.0, "constant"),
        max_dispatch_lag_ms=1000,
    )
    assert report.planned_offsets_s == (0.0, 0.01, 0.02, 0.03, 0.04)
    lags = [outcome.dispatch_lag_ms for outcome in report.outcomes]
    assert all(lag is not None and lag >= 0 for lag in lags)
    assert report.diagnostics.status in (STATUS_OK, STATUS_DEGRADED)
    assert report.diagnostics.dispatch_lag is not None
    assert report.diagnostics.dispatch_lag.count == 5


def test_stop_prevents_further_dispatches_and_marks_the_rest_unsent() -> None:
    stop = threading.Event()

    def send(item: int, context: DispatchContext) -> int:
        stop.set()
        return item

    report = dispatch(list(range(4)), send, OpenLoopRate(5.0, "constant"), stop=stop)

    assert report.outcomes[0].sent
    assert report.stopped_early
    assert report.unsent_count == 3
    unsent = report.outcomes[3]
    assert not unsent.sent and unsent.dispatched_offset_s is None
    assert unsent.scheduled_offset_s == pytest.approx(0.6)


# --- Diagnostics and the report ----------------------------------------------


def test_lag_summary_uses_nearest_rank_percentiles() -> None:
    summary = LagSummary.from_samples([float(i) for i in range(1, 101)], 95.0)
    assert summary is not None
    assert (summary.p50_ms, summary.p90_ms, summary.p99_ms, summary.max_ms) == (
        50.0,
        90.0,
        99.0,
        100.0,
    )
    assert summary.over_threshold_count == 5
    assert LagSummary.from_samples([], 1.0) is None


def test_client_cpu_saturation_marks_the_run_degraded() -> None:
    readings = iter([0.0, 1_000.0])
    report = dispatch([1], _echo, ClosedLoop(1), cpu_clock=lambda: next(readings))
    assert REASON_CPU in report.diagnostics.reasons
    assert report.diagnostics.status == STATUS_DEGRADED


def test_report_identifies_the_dispatcher_and_serialises() -> None:
    report = dispatch(
        [1, 2], _echo, OpenLoopRate(50.0, "poisson", seed=3, max_in_flight=4)
    )
    document = report.to_dict()
    json.dumps(document, allow_nan=False)
    assert document["dispatcher"] == {
        "name": DISPATCHER_NAME,
        "version": DISPATCHER_VERSION,
    }
    assert document["max_in_flight"] == 4
    assert document["started_at_utc"].endswith("Z")
    assert document["policy"]["seed"] == 3
    assert document["sent_count"] == 2 and not document["stopped_early"]
