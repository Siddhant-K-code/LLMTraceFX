"""Run one command while reading NVML, and keep every reading as observed.

The loop is single threaded and every side effect (clock, sleep, process
launch, the local process table) is injected, so tests drive it with a fake
NVML and a fake clock. Nothing here interprets the readings; that is
:mod:`.evidence`.
"""

from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .nvml import NvmlBackend, NvmlCallError

DEFAULT_INTERVAL_MS = 100
DEFAULT_MAX_SAMPLES = 100_000

#: How a GPU process relates to the command being measured.
ALLOWED = "allowed"
FOREIGN = "foreign"
NOT_VISIBLE = "not_visible"
ANCESTRY_UNKNOWN = "ancestry_unknown"

_TERMINATE_GRACE_S = 5.0


class ChildProcess(Protocol):
    pid: int

    def poll(self) -> int | None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


class ProcessTable(Protocol):
    def classify(self, pid: int, roots: frozenset[int]) -> str:
        """Return ``ALLOWED``, ``FOREIGN``, ``NOT_VISIBLE`` or ``ANCESTRY_UNKNOWN``."""
        ...


class ProcProcessTable:
    """Walk parent PIDs through ``/proc/<pid>/stat`` (Linux only)."""

    def __init__(self, root: Path = Path("/proc")) -> None:
        self._root = root

    def _parent(self, pid: int) -> int | None:
        try:
            stat = (self._root / str(pid) / "stat").read_text(encoding="utf-8")
        except OSError:
            return None
        # The command name is in parentheses and may itself contain ")",
        # so the fields after it are found from the last one.
        fields = stat[stat.rfind(")") + 2 :].split()
        try:
            return int(fields[1])
        except (IndexError, ValueError):
            return None

    def classify(self, pid: int, roots: frozenset[int]) -> str:
        if pid in roots:
            return ALLOWED
        if not self._root.is_dir():
            return ANCESTRY_UNKNOWN
        current = pid
        seen: set[int] = set()
        while current not in seen:
            seen.add(current)
            parent = self._parent(current)
            if parent is None:
                # An ancestor that vanished mid-walk leaves the chain unknown.
                return NOT_VISIBLE if current == pid else ANCESTRY_UNKNOWN
            if parent in roots:
                return ALLOWED
            if parent <= 1:
                return FOREIGN
            current = parent
        return FOREIGN


class UnknownProcessTable:
    """Used where process ancestry cannot be read."""

    def classify(self, pid: int, roots: frozenset[int]) -> str:
        return ALLOWED if pid in roots else ANCESTRY_UNKNOWN


def default_process_table() -> ProcessTable:
    if sys.platform.startswith("linux"):
        return ProcProcessTable()
    return UnknownProcessTable()


def default_launcher(argv: Sequence[str]) -> ChildProcess:
    # Same process group and inherited stdio, so Ctrl-C reaches both.
    return subprocess.Popen(list(argv))


@dataclass
class CounterRead:
    value_mj: int | None
    error: str | None
    offset_ms: float


@dataclass
class DeviceReadings:
    index: int
    energy_start: CounterRead | None = None
    energy_end: CounterRead | None = None
    samples: list[tuple[float, int]] = field(default_factory=list)
    sample_errors: dict[str, int] = field(default_factory=dict)
    capped: bool = False


@dataclass
class ObservedProcess:
    pid: int
    classification: str
    devices: set[int]
    first_seen_ms: float


@dataclass
class ProcessChecks:
    checks: int = 0
    listing_errors: dict[str, int] = field(default_factory=dict)
    processes: dict[int, ObservedProcess] = field(default_factory=dict)


@dataclass
class WindowRecord:
    devices: list[DeviceReadings]
    processes: ProcessChecks
    interval_ms: float
    max_samples: int
    window_ms: float
    child_pid: int
    exit_code: int
    allowed_roots: frozenset[int]


_RANK = {ALLOWED: 3, FOREIGN: 2, NOT_VISIBLE: 1, ANCESTRY_UNKNOWN: 0}


class _Recorder:
    def __init__(
        self,
        backend: NvmlBackend,
        devices: Sequence[int],
        *,
        interval_ms: float,
        max_samples: int,
        clock: Callable[[], float],
        process_table: ProcessTable,
        gpu_pids: Sequence[int],
    ) -> None:
        self.backend = backend
        self.readings = [DeviceReadings(index) for index in devices]
        self.processes = ProcessChecks()
        self.interval_ms = interval_ms
        self.max_samples = max_samples
        self.clock = clock
        self.process_table = process_table
        self.roots = frozenset(gpu_pids)
        self.t0 = clock()

    def offset_ms(self) -> float:
        return (self.clock() - self.t0) * 1000.0

    def read_counter(self, index: int) -> CounterRead:
        try:
            value: int | None = self.backend.total_energy_mj(index)
            error = None
        except NvmlCallError as exc:
            value, error = None, exc.reason
        return CounterRead(value, error, self.offset_ms())

    def sample_power(self) -> None:
        for device in self.readings:
            if device.capped:
                continue
            if len(device.samples) >= self.max_samples:
                device.capped = True
                continue
            try:
                milliwatts = self.backend.power_mw(device.index)
            except NvmlCallError as exc:
                device.sample_errors[exc.reason] = (
                    device.sample_errors.get(exc.reason, 0) + 1
                )
                continue
            device.samples.append((self.offset_ms(), milliwatts))

    def check_processes(self) -> None:
        self.processes.checks += 1
        for device in self.readings:
            try:
                pids = self.backend.running_pids(device.index)
            except NvmlCallError as exc:
                errors = self.processes.listing_errors
                errors[exc.reason] = errors.get(exc.reason, 0) + 1
                continue
            for pid in pids:
                classification = self.process_table.classify(pid, self.roots)
                seen = self.processes.processes.get(pid)
                if seen is None:
                    self.processes.processes[pid] = ObservedProcess(
                        pid, classification, {device.index}, self.offset_ms()
                    )
                    continue
                seen.devices.add(device.index)
                # A process seen once as a descendant of the command stays
                # attributed even if it is orphaned before a later check.
                if _RANK[classification] > _RANK[seen.classification]:
                    seen.classification = classification


def _stop(child: ChildProcess) -> None:
    try:
        child.wait(timeout=_TERMINATE_GRACE_S)
        return
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        pass
    child.terminate()
    try:
        child.wait(timeout=_TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def record_window(
    backend: NvmlBackend,
    devices: Sequence[int],
    argv: Sequence[str],
    *,
    interval_ms: float = DEFAULT_INTERVAL_MS,
    max_samples: int = DEFAULT_MAX_SAMPLES,
    gpu_pids: Sequence[int] = (),
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    launcher: Callable[[Sequence[str]], ChildProcess] = default_launcher,
    process_table: ProcessTable | None = None,
) -> WindowRecord:
    """Bracket ``argv`` with energy counter reads and sample power during it.

    Order: start counters, a process check and a power sample, then the
    child starts. Each tick samples power and checks processes until the
    child exits, then a final sample, a final check and the end counters.
    ``OSError`` from the launcher and ``KeyboardInterrupt`` propagate; on
    an interrupt the child is stopped first.
    """
    recorder = _Recorder(
        backend,
        devices,
        interval_ms=interval_ms,
        max_samples=max_samples,
        clock=clock,
        process_table=process_table or default_process_table(),
        gpu_pids=gpu_pids,
    )
    for device in recorder.readings:
        device.energy_start = recorder.read_counter(device.index)
    recorder.check_processes()
    recorder.sample_power()

    child = launcher(argv)
    recorder.roots = recorder.roots | {child.pid}
    interval_s = interval_ms / 1000.0
    try:
        next_tick = clock() + interval_s
        while child.poll() is None:
            delay = next_tick - clock()
            if delay > 0:
                sleep(delay)
            next_tick += interval_s
            if next_tick < clock():
                next_tick = clock() + interval_s
            recorder.sample_power()
            recorder.check_processes()
    except KeyboardInterrupt:
        _stop(child)
        raise
    exit_code = child.wait()

    recorder.sample_power()
    recorder.check_processes()
    for device in recorder.readings:
        device.energy_end = recorder.read_counter(device.index)
    return WindowRecord(
        devices=recorder.readings,
        processes=recorder.processes,
        interval_ms=interval_ms,
        max_samples=max_samples,
        window_ms=recorder.offset_ms(),
        child_pid=child.pid,
        exit_code=exit_code,
        allowed_roots=recorder.roots,
    )
