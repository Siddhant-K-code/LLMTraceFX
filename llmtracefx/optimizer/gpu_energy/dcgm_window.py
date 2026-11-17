"""Run one command while scraping dcgm-exporter, and bound the window it measured.

dcgm-exporter output has no timestamps, so a counter value is only known
to have been sampled before the scrape that returned it. A change in the
value is used as an edge: the new value was sampled after the start of the
last scrape that still showed the old value and before the end of the
scrape that showed the new one. The loop waits for such an edge on every
selected GPU, starts the command at once, and after the command exits
waits for the next edge whose earlier scrape began after the exit. The
energy delta then covers a span that contains the whole command, with both
ends bounded by host clock reads. No exporter or DCGM clock is used.

As in :mod:`.window`, every side effect is injected so tests drive the loop
with a fake exporter and a fake clock. Nothing here interprets readings.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .dcgm_exporter import (
    GPU_MISSING,
    METRIC_MISSING,
    POWER_LIMIT_METRICS,
    POWER_METRICS,
    EnergyMetric,
    GpuReading,
    ScrapeError,
    Scraper,
    scrape_gpus,
)
from .window import ChildProcess, _stop, default_launcher

DEFAULT_SCRAPE_INTERVAL_MS = 500
DEFAULT_EDGE_TIMEOUT_MS = 120_000
DEFAULT_MIN_RUN_SHARE = 0.9


class DcgmExporterUnavailable(RuntimeError):
    """The exporter cannot bracket a window. Raised before the command starts."""


@dataclass(frozen=True)
class Edge:
    """A counter change seen between two scrapes, on the host monotonic clock."""

    #: Start of the last scrape that still showed the previous value.
    after_ms: float
    #: End of the scrape that first showed ``value``.
    by_ms: float
    value: float


@dataclass
class DcgmDeviceReadings:
    index: int
    metric: EnergyMetric
    name: str | None = None
    uuid_sha256: str | None = None
    mig_seen: bool = False
    enforced_power_limit_w: float | None = None
    start: Edge | None = None
    end: Edge | None = None
    counter_reset: bool = False
    #: ``by_ms`` of every counter change seen, for the observed update gap.
    changes: list[float] = field(default_factory=list)
    read_errors: dict[str, int] = field(default_factory=dict)
    power_metric: str | None = None
    #: ``(end of scrape in ms, watts)``. When DCGM sampled it is unknown.
    samples: list[tuple[float, float]] = field(default_factory=list)
    sample_errors: dict[str, int] = field(default_factory=dict)
    capped: bool = False
    last_value: float | None = None
    last_ms: float | None = None


@dataclass
class DcgmWindowRecord:
    devices: list[DcgmDeviceReadings]
    scrape_interval_ms: float
    max_samples: int
    edge_timeout_ms: float
    scrapes: int
    scrape_errors: dict[str, int]
    command_start_ms: float
    command_end_ms: float
    window_ms: float
    child_pid: int
    exit_code: int


class _Recorder:
    def __init__(
        self,
        scraper: Scraper,
        *,
        max_samples: int,
        clock: Callable[[], float],
    ) -> None:
        self.scraper = scraper
        self.clock = clock
        self.t0 = clock()
        self.max_samples = max_samples
        self.scrapes = 0
        self.scrape_errors: dict[str, int] = {}
        self.started = False
        self.command_end_ms: float | None = None
        self.readings: list[DcgmDeviceReadings] = []

    def offset_ms(self) -> float:
        return (self.clock() - self.t0) * 1000.0

    def scrape(self) -> tuple[dict[int, GpuReading], float, float]:
        """Scrape once and count it. ``ScrapeError`` propagates."""
        start_ms = self.offset_ms()
        self.scrapes += 1
        try:
            gpus = scrape_gpus(self.scraper)
        except ScrapeError as exc:
            _count(self.scrape_errors, exc.reason)
            raise
        return gpus, start_ms, self.offset_ms()

    def tick(self) -> None:
        try:
            gpus, start_ms, end_ms = self.scrape()
        except ScrapeError:
            return
        self.observe(gpus, start_ms, end_ms)

    def observe(
        self, gpus: dict[int, GpuReading], start_ms: float, end_ms: float
    ) -> None:
        for device in self.readings:
            gpu = gpus.get(device.index)
            if gpu is None:
                _count(device.read_errors, GPU_MISSING)
                continue
            self._describe(device, gpu)
            self._sample_power(device, gpu, end_ms)
            value, error = gpu.energy(device.metric)
            if error is not None:
                _count(device.read_errors, error)
                continue
            assert value is not None
            previous = device.last_value
            if previous is not None and value != previous:
                assert device.last_ms is not None
                self._on_change(device, Edge(device.last_ms, end_ms, value), previous)
            device.last_value = value
            device.last_ms = start_ms

    def _describe(self, device: DcgmDeviceReadings, gpu: GpuReading) -> None:
        device.name = device.name or gpu.name
        device.uuid_sha256 = device.uuid_sha256 or gpu.uuid_sha256
        device.mig_seen = device.mig_seen or gpu.mig_seen
        if device.enforced_power_limit_w is None:
            _, device.enforced_power_limit_w = gpu.first(POWER_LIMIT_METRICS)

    def _sample_power(
        self, device: DcgmDeviceReadings, gpu: GpuReading, end_ms: float
    ) -> None:
        if device.capped:
            return
        name, watts = gpu.first(POWER_METRICS)
        if name is None or watts is None:
            _count(device.sample_errors, METRIC_MISSING)
            return
        if device.power_metric is None:
            device.power_metric = name
        elif name != device.power_metric:
            _count(device.sample_errors, "metric_changed")
            return
        if len(device.samples) >= self.max_samples:
            device.capped = True
            return
        device.samples.append((end_ms, watts))

    def _on_change(
        self, device: DcgmDeviceReadings, edge: Edge, previous: float
    ) -> None:
        device.changes.append(edge.by_ms)
        if not self.started:
            # Before the command starts the newest edge is the tightest.
            device.start = edge
            return
        if device.end is not None:
            return
        if edge.value < previous:
            device.counter_reset = True
        if self.command_end_ms is not None and edge.after_ms >= self.command_end_ms:
            device.end = edge


def _count(counts: dict[str, int], key: str) -> None:
    counts[key] = counts.get(key, 0) + 1


def _ticker(
    clock: Callable[[], float], sleep: Callable[[float], None], interval_s: float
) -> Callable[[], None]:
    next_tick = clock() + interval_s

    def wait() -> None:
        nonlocal next_tick
        delay = next_tick - clock()
        if delay > 0:
            sleep(delay)
        next_tick += interval_s
        if next_tick < clock():
            next_tick = clock() + interval_s

    return wait


def record_dcgm_window(
    scraper: Scraper,
    devices: Sequence[int],
    argv: Sequence[str],
    *,
    scrape_interval_ms: float = DEFAULT_SCRAPE_INTERVAL_MS,
    edge_timeout_ms: float = DEFAULT_EDGE_TIMEOUT_MS,
    max_samples: int = 100_000,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    launcher: Callable[[Sequence[str]], ChildProcess] = default_launcher,
) -> DcgmWindowRecord:
    """Scrape until every GPU shows a counter edge, run ``argv``, then wait
    for an edge that starts after the command exits.

    Raises :class:`DcgmExporterUnavailable` before starting the command when
    the first scrape fails, a selected GPU has no energy counter, or no edge
    appears within ``edge_timeout_ms``. ``OSError`` from the launcher and
    ``KeyboardInterrupt`` propagate; on an interrupt the child is stopped.
    """
    recorder = _Recorder(scraper, max_samples=max_samples, clock=clock)
    try:
        baseline, start_ms, end_ms = recorder.scrape()
    except ScrapeError as exc:
        raise DcgmExporterUnavailable(
            f"the first scrape failed ({exc.reason}): {exc}"
        ) from exc
    missing = [
        index
        for index in devices
        if index not in baseline or baseline[index].energy_metric() is None
    ]
    if missing:
        raise DcgmExporterUnavailable(
            "no unambiguous energy counter for GPU "
            f"{', '.join(map(str, missing))} in the first scrape"
        )
    for index in devices:
        metric = baseline[index].energy_metric()
        assert metric is not None
        recorder.readings.append(DcgmDeviceReadings(index, metric))
    recorder.observe(baseline, start_ms, end_ms)

    interval_s = scrape_interval_ms / 1000.0
    wait = _ticker(clock, sleep, interval_s)
    while any(device.start is None for device in recorder.readings):
        if recorder.offset_ms() >= edge_timeout_ms:
            waiting = [d.index for d in recorder.readings if d.start is None]
            raise DcgmExporterUnavailable(
                "the energy counter of GPU "
                f"{', '.join(map(str, waiting))} did not change within "
                f"{edge_timeout_ms:.0f} ms, so the start of the window "
                "cannot be bounded. The command was not run"
            )
        wait()
        recorder.tick()

    recorder.started = True
    command_start_ms = recorder.offset_ms()
    child = launcher(argv)
    wait = _ticker(clock, sleep, interval_s)
    try:
        while child.poll() is None:
            wait()
            recorder.tick()
    except KeyboardInterrupt:
        _stop(child)
        raise
    exit_code = child.wait()
    command_end_ms = recorder.offset_ms()
    recorder.command_end_ms = command_end_ms

    deadline_ms = command_end_ms + edge_timeout_ms
    recorder.tick()
    wait = _ticker(clock, sleep, interval_s)
    while any(device.end is None for device in recorder.readings):
        if recorder.offset_ms() >= deadline_ms:
            break
        wait()
        recorder.tick()

    return DcgmWindowRecord(
        devices=recorder.readings,
        scrape_interval_ms=scrape_interval_ms,
        max_samples=max_samples,
        edge_timeout_ms=edge_timeout_ms,
        scrapes=recorder.scrapes,
        scrape_errors=recorder.scrape_errors,
        command_start_ms=command_start_ms,
        command_end_ms=command_end_ms,
        window_ms=recorder.offset_ms(),
        child_pid=child.pid,
        exit_code=exit_code,
    )
