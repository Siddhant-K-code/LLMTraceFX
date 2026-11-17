"""A reusable, transport-free concurrent request dispatcher.

The dispatcher decides *when* each request is issued and records how well
it kept to that plan. It never looks inside a request or a result, so the
same scheduler serves a load sweep against an OpenAI-compatible endpoint,
a trace replay, or a test with a fake ``send`` function.

Three load policies are supported:

``ClosedLoop(concurrency)``
    ``concurrency`` workers each issue the next request as soon as their
    previous one completes. Throughput is an outcome, not an input.

``OpenLoopRate(rate_per_second, distribution, seed, max_in_flight)``
    Requests arrive on a schedule fixed in advance: evenly spaced
    (``constant``) or with exponential gaps (``poisson``, seeded so the
    schedule is reproducible). Arrivals do not wait for completions.

``TimedArrivals(offsets_seconds, max_in_flight)``
    Requests arrive at caller-supplied offsets from the start, as in a
    recorded production trace. Requests that share an offset are released
    together, with no artificial gap between them.

Open-loop and timed policies bound the number of requests in flight with
``max_in_flight``. A request that reaches its arrival time while that cap
is full waits for a free slot. It is counted in
``deferred_by_in_flight_cap`` and its wait shows up as dispatch lag, so a
schedule the client could not honour is never reported as if it had been.

Every offset is measured with one monotonic clock from one origin, the
moment ``dispatch`` starts. The scheduling diagnostics summarise the gap
between the planned and the actual dispatch time, the slot refill delay
in closed loop, the observed concurrency and the client's own CPU use,
and mark the run ``degraded`` when any of them suggests the client, not
the server, shaped the result.
"""

from __future__ import annotations

import math
import queue
import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, Protocol, TypeVar

from ..schema import utc_now_iso

DISPATCHER_NAME = "llmtracefx.load.dispatcher"
DISPATCHER_VERSION = "1"

MAX_CONCURRENCY = 256
"""Upper bound on workers and on requests in flight. Each in-flight
request holds one thread, so this also bounds the client's thread count."""

MAX_REQUESTS = 1_000_000
DEFAULT_MAX_DISPATCH_LAG_MS = 10.0
CLIENT_CPU_SATURATION = 0.9
"""Client CPU seconds per wall second above which the run is marked
degraded. CPython threads share one interpreter lock, so a load generator
near one full core is likely delaying its own dispatches and reads."""

ARRIVAL_POISSON = "poisson"
ARRIVAL_CONSTANT = "constant"
ARRIVAL_DISTRIBUTIONS = (ARRIVAL_POISSON, ARRIVAL_CONSTANT)

STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"

REASON_DISPATCH_LAG = "dispatch_lag_p99_over_threshold"
REASON_SLOT_REFILL = "slot_refill_p99_over_threshold"
REASON_CAP_DEFERRED = "requests_deferred_by_in_flight_cap"
REASON_CPU = "client_cpu_saturated"

_POLL_SECONDS = 0.05

T = TypeVar("T")
R = TypeVar("R")


class DispatchConfigError(ValueError):
    """A load policy or a dispatch argument is invalid."""


class _StopSignal(Protocol):
    def is_set(self) -> bool: ...


def _check_int(name: str, value: Any, *, minimum: int, maximum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DispatchConfigError(f"{name} must be an integer, got {value!r}")
    if not minimum <= value <= maximum:
        raise DispatchConfigError(
            f"{name} must be between {minimum} and {maximum}, got {value}"
        )


def _check_finite(name: str, value: Any, *, positive: bool) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DispatchConfigError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (positive and number == 0):
        bound = "greater than 0" if positive else "at least 0"
        raise DispatchConfigError(f"{name} must be finite and {bound}, got {value}")
    return number


# --- Load policies -----------------------------------------------------------


@dataclass(frozen=True)
class ClosedLoop:
    """Keep ``concurrency`` requests in flight until every request is sent."""

    concurrency: int
    kind: ClassVar[str] = "closed_loop"

    def __post_init__(self) -> None:
        _check_int("concurrency", self.concurrency, minimum=1, maximum=MAX_CONCURRENCY)

    @property
    def max_in_flight(self) -> int:
        return self.concurrency

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "concurrency": self.concurrency}


@dataclass(frozen=True)
class OpenLoopRate:
    """Issue requests at ``rate_per_second`` regardless of completions."""

    rate_per_second: float
    distribution: str = ARRIVAL_POISSON
    seed: int | None = None
    max_in_flight: int = MAX_CONCURRENCY
    kind: ClassVar[str] = "open_loop_rate"

    def __post_init__(self) -> None:
        rate = _check_finite("rate_per_second", self.rate_per_second, positive=True)
        object.__setattr__(self, "rate_per_second", rate)
        if self.distribution not in ARRIVAL_DISTRIBUTIONS:
            raise DispatchConfigError(
                f"distribution must be one of {', '.join(ARRIVAL_DISTRIBUTIONS)}, "
                f"got {self.distribution!r}"
            )
        if self.distribution == ARRIVAL_POISSON:
            if self.seed is None:
                raise DispatchConfigError(
                    "poisson arrivals need a seed so the schedule is reproducible"
                )
            _check_int("seed", self.seed, minimum=0, maximum=2**63 - 1)
        elif self.seed is not None:
            raise DispatchConfigError("seed only applies to poisson arrivals")
        _check_int(
            "max_in_flight", self.max_in_flight, minimum=1, maximum=MAX_CONCURRENCY
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "rate_per_second": self.rate_per_second,
            "distribution": self.distribution,
            "seed": self.seed,
            "max_in_flight": self.max_in_flight,
        }


@dataclass(frozen=True)
class TimedArrivals:
    """Issue request ``i`` at ``offsets_seconds[i]`` after the start."""

    offsets_seconds: tuple[float, ...]
    max_in_flight: int = MAX_CONCURRENCY
    kind: ClassVar[str] = "timed_arrivals"

    def __post_init__(self) -> None:
        raw: Any = self.offsets_seconds
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise DispatchConfigError("offsets_seconds must be a sequence of numbers")
        offsets = tuple(
            _check_finite(f"offsets_seconds[{i}]", value, positive=False)
            for i, value in enumerate(raw)
        )
        if not offsets:
            raise DispatchConfigError("offsets_seconds must not be empty")
        if len(offsets) > MAX_REQUESTS:
            raise DispatchConfigError(
                f"at most {MAX_REQUESTS} arrivals are supported, got {len(offsets)}"
            )
        for i in range(1, len(offsets)):
            if offsets[i] < offsets[i - 1]:
                raise DispatchConfigError(
                    f"offsets_seconds must be non-decreasing; index {i} is "
                    f"{offsets[i]} after {offsets[i - 1]}"
                )
        object.__setattr__(self, "offsets_seconds", offsets)
        _check_int(
            "max_in_flight", self.max_in_flight, minimum=1, maximum=MAX_CONCURRENCY
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "arrival_count": len(self.offsets_seconds),
            "first_offset_s": self.offsets_seconds[0],
            "last_offset_s": self.offsets_seconds[-1],
            "max_in_flight": self.max_in_flight,
        }


LoadPolicy = ClosedLoop | OpenLoopRate | TimedArrivals


def planned_offsets(policy: LoadPolicy, count: int) -> tuple[float, ...] | None:
    """The arrival schedule for ``count`` requests, in seconds from the start.

    Closed loop has no schedule, since each dispatch waits on a completion,
    so it returns ``None``. The other policies are fully determined by
    their parameters: the same policy and count always give the same
    offsets. The first arrival is at 0.
    """
    _check_int("count", count, minimum=1, maximum=MAX_REQUESTS)
    if isinstance(policy, ClosedLoop):
        return None
    if isinstance(policy, OpenLoopRate):
        if policy.distribution == ARRIVAL_CONSTANT:
            return tuple(index / policy.rate_per_second for index in range(count))
        rng = random.Random(policy.seed)
        offsets = [0.0]
        elapsed = 0.0
        for _ in range(count - 1):
            elapsed += rng.expovariate(policy.rate_per_second)
            offsets.append(elapsed)
        return tuple(offsets)
    if isinstance(policy, TimedArrivals):
        if len(policy.offsets_seconds) != count:
            raise DispatchConfigError(
                f"timed arrivals list {len(policy.offsets_seconds)} offsets "
                f"for {count} requests"
            )
        return policy.offsets_seconds
    raise DispatchConfigError(f"unsupported load policy {type(policy).__name__}")


# --- Results -----------------------------------------------------------------


@dataclass(frozen=True)
class DispatchContext:
    """What ``send`` knows about its own dispatch.

    ``clock`` and ``clock_origin`` let a sender place its own clock readings
    on the dispatch timeline: ``offset = clock() - clock_origin``.
    """

    index: int
    scheduled_offset_s: float | None
    dispatched_offset_s: float
    in_flight_at_dispatch: int
    clock: Callable[[], float]
    clock_origin: float


@dataclass(frozen=True)
class DispatchOutcome(Generic[R]):
    """One request's placement on the timeline, plus whatever ``send`` returned.

    ``sent`` is False when the run stopped before the request was issued;
    every other timing field is then ``None``. ``client_error`` names the
    exception type when ``send`` raised. The message is not kept, since it
    may carry request content.
    """

    index: int
    sent: bool
    scheduled_offset_s: float | None
    dispatched_offset_s: float | None = None
    completed_offset_s: float | None = None
    in_flight_at_dispatch: int | None = None
    result: R | None = None
    client_error: str | None = None

    @property
    def dispatch_lag_ms(self) -> float | None:
        if self.scheduled_offset_s is None or self.dispatched_offset_s is None:
            return None
        return (self.dispatched_offset_s - self.scheduled_offset_s) * 1000.0

    def timing_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "sent": self.sent,
            "scheduled_offset_s": self.scheduled_offset_s,
            "dispatched_offset_s": self.dispatched_offset_s,
            "completed_offset_s": self.completed_offset_s,
            "dispatch_lag_ms": self.dispatch_lag_ms,
            "in_flight_at_dispatch": self.in_flight_at_dispatch,
            "client_error": self.client_error,
        }


def _nearest_rank(ordered: Sequence[float], percentile: float) -> float:
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return ordered[rank - 1]


@dataclass(frozen=True)
class LagSummary:
    """Nearest-rank percentiles of a set of client-side delays."""

    count: int
    p50_ms: float
    p90_ms: float
    p99_ms: float
    max_ms: float
    over_threshold_count: int

    @classmethod
    def from_samples(
        cls, samples_ms: Sequence[float], threshold_ms: float
    ) -> LagSummary | None:
        if not samples_ms:
            return None
        ordered = sorted(samples_ms)
        return cls(
            count=len(ordered),
            p50_ms=_nearest_rank(ordered, 50),
            p90_ms=_nearest_rank(ordered, 90),
            p99_ms=_nearest_rank(ordered, 99),
            max_ms=ordered[-1],
            over_threshold_count=sum(1 for value in ordered if value > threshold_ms),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
            "over_threshold_count": self.over_threshold_count,
        }


@dataclass(frozen=True)
class SchedulingDiagnostics:
    """How faithfully the client kept to its load policy.

    ``dispatch_lag`` (open loop and timed) is actual minus planned dispatch
    time. ``slot_refill`` (closed loop) is the gap between a worker's
    completion and its next dispatch. Both are client-side delays that the
    server never saw, so a high value means the offered load was lower or
    burstier than requested.
    """

    mode: str
    max_dispatch_lag_ms: float
    dispatch_lag: LagSummary | None
    slot_refill: LagSummary | None
    max_in_flight: int
    max_observed_in_flight: int
    deferred_by_in_flight_cap: int
    client_cpu_seconds: float
    window_seconds: float
    client_cpu_utilization: float | None
    status: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "max_dispatch_lag_ms": self.max_dispatch_lag_ms,
            "dispatch_lag": (
                None if self.dispatch_lag is None else self.dispatch_lag.to_dict()
            ),
            "slot_refill": (
                None if self.slot_refill is None else self.slot_refill.to_dict()
            ),
            "max_in_flight": self.max_in_flight,
            "max_observed_in_flight": self.max_observed_in_flight,
            "deferred_by_in_flight_cap": self.deferred_by_in_flight_cap,
            "client_cpu_seconds": self.client_cpu_seconds,
            "window_seconds": self.window_seconds,
            "client_cpu_utilization": self.client_cpu_utilization,
            "client_cpu_saturation_threshold": CLIENT_CPU_SATURATION,
            "status": self.status,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class DispatchReport(Generic[R]):
    """The full outcome of one ``dispatch`` call.

    ``outcomes`` is in request order and has one entry per request,
    including requests that were never sent because the run stopped.
    """

    policy: LoadPolicy
    max_in_flight: int
    started_at_utc: str
    planned_offsets_s: tuple[float, ...] | None
    outcomes: tuple[DispatchOutcome[R], ...]
    diagnostics: SchedulingDiagnostics
    dispatcher_name: str = DISPATCHER_NAME
    dispatcher_version: str = DISPATCHER_VERSION

    @property
    def sent_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.sent)

    @property
    def unsent_count(self) -> int:
        return len(self.outcomes) - self.sent_count

    @property
    def stopped_early(self) -> bool:
        return self.unsent_count > 0

    def to_dict(self) -> dict[str, Any]:
        """Policy, timing and diagnostics. Results are left to the caller."""
        return {
            "dispatcher": {
                "name": self.dispatcher_name,
                "version": self.dispatcher_version,
            },
            "policy": self.policy.to_dict(),
            "max_in_flight": self.max_in_flight,
            "started_at_utc": self.started_at_utc,
            "request_count": len(self.outcomes),
            "sent_count": self.sent_count,
            "unsent_count": self.unsent_count,
            "stopped_early": self.stopped_early,
            "diagnostics": self.diagnostics.to_dict(),
        }


# --- Dispatch ----------------------------------------------------------------


class _Run(Generic[T, R]):
    def __init__(
        self,
        items: Sequence[T],
        send: Callable[[T, DispatchContext], R],
        *,
        clock: Callable[[], float],
        stop: _StopSignal | None,
    ) -> None:
        self.items = items
        self.send = send
        self.clock = clock
        self.external_stop = stop
        self.halt = threading.Event()
        self.lock = threading.Lock()
        self.in_flight = 0
        self.max_observed = 0
        self.outcomes: list[DispatchOutcome[R] | None] = [None] * len(items)
        self.origin = 0.0

    def stopped(self) -> bool:
        if self.halt.is_set():
            return True
        return self.external_stop is not None and self.external_stop.is_set()

    def execute(self, index: int, scheduled: float | None) -> tuple[float, float]:
        """Send request ``index`` now. Returns its dispatch and completion offsets."""
        with self.lock:
            self.in_flight += 1
            in_flight = self.in_flight
            self.max_observed = max(self.max_observed, in_flight)
        dispatched = self.clock() - self.origin
        context = DispatchContext(
            index=index,
            scheduled_offset_s=scheduled,
            dispatched_offset_s=dispatched,
            in_flight_at_dispatch=in_flight,
            clock=self.clock,
            clock_origin=self.origin,
        )
        result: R | None = None
        error: str | None = None
        try:
            result = self.send(self.items[index], context)
        except Exception as exc:  # noqa: BLE001 - one request must not end the run
            error = type(exc).__name__
        completed = self.clock() - self.origin
        with self.lock:
            self.in_flight -= 1
        self.outcomes[index] = DispatchOutcome(
            index=index,
            sent=True,
            scheduled_offset_s=scheduled,
            dispatched_offset_s=dispatched,
            completed_offset_s=completed,
            in_flight_at_dispatch=in_flight,
            result=result,
            client_error=error,
        )
        return dispatched, completed


def _join_all(threads: Sequence[threading.Thread], run: _Run[Any, Any]) -> None:
    try:
        for thread in threads:
            thread.join()
    except BaseException:
        # Ctrl-C lands here. Stop issuing new work and let requests already
        # in flight finish under their own timeouts before re-raising.
        run.halt.set()
        for thread in threads:
            thread.join()
        raise


def _run_closed_loop(run: _Run[Any, Any], concurrency: int) -> list[float]:
    count = len(run.items)
    workers = min(concurrency, count)
    next_index = iter(range(count))
    index_lock = threading.Lock()
    refill_ms: list[float] = []
    refill_lock = threading.Lock()
    start = threading.Barrier(workers)

    def worker() -> None:
        try:
            start.wait()
        except threading.BrokenBarrierError:
            return
        last_completed: float | None = None
        while not run.stopped():
            with index_lock:
                index = next(next_index, None)
            if index is None:
                return
            dispatched, completed = run.execute(index, None)
            if last_completed is not None:
                with refill_lock:
                    refill_ms.append((dispatched - last_completed) * 1000.0)
            last_completed = completed

    threads = [
        threading.Thread(target=worker, name=f"llmtracefx-load-{k}", daemon=True)
        for k in range(workers)
    ]
    for thread in threads:
        thread.start()
    _join_all(threads, run)
    return refill_ms


def _run_scheduled(
    run: _Run[Any, Any],
    offsets: Sequence[float],
    cap: int,
    *,
    sleep: Callable[[float], None],
) -> int:
    count = len(run.items)
    slots = threading.Semaphore(cap)
    work: queue.SimpleQueue[tuple[int, float] | None] = queue.SimpleQueue()
    deferred = 0

    def worker() -> None:
        while True:
            job = work.get()
            if job is None:
                return
            try:
                run.execute(*job)
            finally:
                slots.release()

    threads = [
        threading.Thread(target=worker, name=f"llmtracefx-load-{k}", daemon=True)
        for k in range(min(cap, count))
    ]
    for thread in threads:
        thread.start()
    try:
        for index, offset in enumerate(offsets):
            target = run.origin + offset
            while not run.stopped():
                remaining = target - run.clock()
                if remaining <= 0:
                    break
                sleep(min(remaining, _POLL_SECONDS))
            if run.stopped():
                break
            if not slots.acquire(blocking=False):
                # The cap is full at this request's arrival time. It waits
                # for a slot, and the wait becomes its dispatch lag.
                deferred += 1
                acquired = False
                while not acquired and not run.stopped():
                    acquired = slots.acquire(timeout=_POLL_SECONDS)
                if not acquired:
                    break
            work.put((index, offset))
    except BaseException:
        run.halt.set()
        raise
    finally:
        for _ in threads:
            work.put(None)
        _join_all(threads, run)
    return deferred


def dispatch(
    requests: Sequence[T],
    send: Callable[[T, DispatchContext], R],
    policy: LoadPolicy,
    *,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
    cpu_clock: Callable[[], float] = time.process_time,
    stop: _StopSignal | None = None,
    max_dispatch_lag_ms: float = DEFAULT_MAX_DISPATCH_LAG_MS,
) -> DispatchReport[R]:
    """Issue ``requests`` through ``send`` under ``policy`` and time each one.

    ``send(request, context)`` runs on a worker thread and must be safe to
    call concurrently. Its return value is kept as the outcome's
    ``result``. An exception from ``send`` is recorded by type name and
    does not stop the run. Setting ``stop`` prevents further dispatches;
    requests already in flight are allowed to finish.
    """
    items = list(requests)
    _check_int("request count", len(items), minimum=1, maximum=MAX_REQUESTS)
    threshold = _check_finite(
        "max_dispatch_lag_ms", max_dispatch_lag_ms, positive=False
    )
    if not isinstance(policy, (ClosedLoop, OpenLoopRate, TimedArrivals)):
        raise DispatchConfigError(f"unsupported load policy {type(policy).__name__}")
    offsets = planned_offsets(policy, len(items))
    cap = policy.max_in_flight

    run: _Run[T, R] = _Run(items, send, clock=clock, stop=stop)
    started_at_utc = utc_now_iso()
    cpu_started = cpu_clock()
    run.origin = clock()

    refill_ms: list[float] = []
    deferred = 0
    if offsets is None:
        refill_ms = _run_closed_loop(run, cap)
    else:
        deferred = _run_scheduled(run, offsets, cap, sleep=sleep)

    window = clock() - run.origin
    cpu_seconds = max(0.0, cpu_clock() - cpu_started)

    outcomes: list[DispatchOutcome[R]] = []
    for index, outcome in enumerate(run.outcomes):
        if outcome is None:
            outcome = DispatchOutcome(
                index=index,
                sent=False,
                scheduled_offset_s=None if offsets is None else offsets[index],
            )
        outcomes.append(outcome)

    lag_samples = [
        lag for outcome in outcomes if (lag := outcome.dispatch_lag_ms) is not None
    ]
    dispatch_lag = LagSummary.from_samples(lag_samples, threshold)
    slot_refill = LagSummary.from_samples(refill_ms, threshold)
    utilization = cpu_seconds / window if window > 0 else None

    reasons: list[str] = []
    if dispatch_lag is not None and dispatch_lag.p99_ms > threshold:
        reasons.append(REASON_DISPATCH_LAG)
    if slot_refill is not None and slot_refill.p99_ms > threshold:
        reasons.append(REASON_SLOT_REFILL)
    if deferred:
        reasons.append(REASON_CAP_DEFERRED)
    if utilization is not None and utilization > CLIENT_CPU_SATURATION:
        reasons.append(REASON_CPU)

    diagnostics = SchedulingDiagnostics(
        mode=policy.kind,
        max_dispatch_lag_ms=threshold,
        dispatch_lag=dispatch_lag,
        slot_refill=slot_refill,
        max_in_flight=cap,
        max_observed_in_flight=run.max_observed,
        deferred_by_in_flight_cap=deferred,
        client_cpu_seconds=cpu_seconds,
        window_seconds=window,
        client_cpu_utilization=utilization,
        status=STATUS_DEGRADED if reasons else STATUS_OK,
        reasons=tuple(reasons),
    )
    return DispatchReport(
        policy=policy,
        max_in_flight=cap,
        started_at_utc=started_at_utc,
        planned_offsets_s=offsets,
        outcomes=tuple(outcomes),
        diagnostics=diagnostics,
    )
