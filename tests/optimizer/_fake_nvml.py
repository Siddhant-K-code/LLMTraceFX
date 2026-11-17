"""A scripted NVML, clock, child process and process table for gpu-energy tests.

Each fake device draws a constant power, so its energy counter is a linear
function of the fake clock. Every reading can be replaced by a script or an
NVML error reason, which is how the refusal paths are reached.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from llmtracefx.optimizer.gpu_energy.nvml import NvmlCallError, NvmlUnavailable
from llmtracefx.optimizer.gpu_energy.window import ALLOWED


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@dataclass
class FakeDevice:
    name: str = "NVIDIA L4"
    uuid: str = "GPU-11111111-2222-3333-4444-555555555555"
    power_w: float = 50.0
    energy_base_mj: int = 9_000_000
    limit_mw: int = 72_000
    mig: bool = False
    #: Values returned by successive energy counter reads (the capability
    #: probe reads once before the window), instead of the
    #: clock driven counter.
    energy_script: list[int] | None = None
    energy_error: str | None = None
    power_error: str | None = None
    #: Fail only these power reads (0 based call numbers).
    power_error_calls: frozenset[int] = frozenset()
    pids: Callable[[float], tuple[int, ...]] = lambda _t: ()
    pids_error: str | None = None
    power_calls: int = field(default=0, init=False)


class FakeNvml:
    def __init__(
        self,
        devices: Sequence[FakeDevice],
        clock: FakeClock,
        *,
        init_error: NvmlUnavailable | None = None,
        count_error: str | None = None,
    ) -> None:
        self.devices = list(devices)
        self.clock = clock
        self.init_error = init_error
        self.count_error = count_error
        self.initialised = False
        self.shutdowns = 0

    def init(self) -> None:
        if self.init_error is not None:
            raise self.init_error
        self.initialised = True

    def shutdown(self) -> None:
        self.shutdowns += 1

    def versions(self) -> dict[str, str | None]:
        return {
            "nvml_version": "12.560.35.03",
            "driver_version": "560.35.03",
            "cuda_driver_version": "12.6",
            "nvidia_ml_py_version": "13.615.71",
        }

    def device_count(self) -> int:
        if self.count_error is not None:
            raise NvmlCallError(self.count_error, "count failed")
        return len(self.devices)

    def _device(self, index: int) -> FakeDevice:
        return self.devices[index]

    def device_name(self, index: int) -> str:
        return self._device(index).name

    def device_uuid(self, index: int) -> str:
        return self._device(index).uuid

    def total_energy_mj(self, index: int) -> int:
        device = self._device(index)
        if device.energy_error is not None:
            raise NvmlCallError(device.energy_error, "energy failed")
        if device.energy_script is not None:
            return device.energy_script.pop(0)
        return device.energy_base_mj + round(device.power_w * self.clock.now * 1000)

    def power_mw(self, index: int) -> int:
        device = self._device(index)
        call = device.power_calls
        device.power_calls += 1
        if device.power_error is not None or call in device.power_error_calls:
            raise NvmlCallError(device.power_error or "call_failed", "power failed")
        return round(device.power_w * 1000)

    def enforced_power_limit_mw(self, index: int) -> int:
        return self._device(index).limit_mw

    def mig_mode_enabled(self, index: int) -> bool:
        return self._device(index).mig

    def running_pids(self, index: int) -> tuple[int, ...]:
        device = self._device(index)
        if device.pids_error is not None:
            raise NvmlCallError(device.pids_error, "listing failed")
        return device.pids(self.clock.now)


class FakeChild:
    def __init__(self, clock: FakeClock, pid: int, runtime_s: float, code: int):
        self.clock = clock
        self.pid = pid
        self.ends_at = clock.now + runtime_s
        self.code = code
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        if self.terminated or self.clock.now >= self.ends_at:
            return self.code
        return None

    def wait(self, timeout: float | None = None) -> int:
        result = self.poll()
        if result is None:
            if timeout is not None:
                raise subprocess.TimeoutExpired("fake", timeout)
            self.clock.now = self.ends_at
            return self.code
        return result

    def terminate(self) -> None:
        self.terminated = True
        self.code = -15

    def kill(self) -> None:
        self.killed = True


class FakeLauncher:
    def __init__(
        self,
        clock: FakeClock,
        *,
        pid: int = 4242,
        runtime_s: float = 1.0,
        exit_code: int = 0,
        error: OSError | None = None,
    ) -> None:
        self.clock = clock
        self.pid = pid
        self.runtime_s = runtime_s
        self.exit_code = exit_code
        self.error = error
        self.calls: list[list[str]] = []
        self.children: list[FakeChild] = []

    def __call__(self, argv: Sequence[str]) -> FakeChild:
        self.calls.append(list(argv))
        if self.error is not None:
            raise self.error
        child = FakeChild(self.clock, self.pid, self.runtime_s, self.exit_code)
        self.children.append(child)
        return child


class FakeProcessTable:
    """Classify by a fixed map. Unlisted pids are allowed only as roots."""

    def __init__(self, classes: dict[int, str] | None = None) -> None:
        self.classes = classes or {}

    def classify(self, pid: int, roots: frozenset[int]) -> str:
        if pid in roots:
            return ALLOWED
        return self.classes.get(pid, "foreign")
