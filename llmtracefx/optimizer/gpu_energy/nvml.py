"""The only module that talks to NVML.

NVML is reached through ``pynvml`` from the optional ``nvidia-ml-py``
package (``llmtracefx[nvml]``). It is imported when a backend is created,
never at module import, so the rest of LLMTraceFX and the whole test suite
run without it. Everything above this module works against the small
:class:`NvmlBackend` protocol, which tests replace with a fake.

Every NVML failure is translated here into one of two typed errors, so no
``pynvml`` exception type leaks out and no failure is mistaken for a value.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from ...extras import extra_install_hint

#: Capability states :class:`NvmlUnavailable` can carry.
EXTRA_MISSING = "extra_missing"
LIBRARY_UNAVAILABLE = "library_unavailable"
DRIVER_UNAVAILABLE = "driver_unavailable"
NVML_ERROR = "nvml_error"

#: Reasons :class:`NvmlCallError` can carry.
NOT_SUPPORTED = "not_supported"
NO_PERMISSION = "no_permission"
GPU_LOST = "gpu_lost"
CALL_FAILED = "call_failed"
#: The installed ``nvidia-ml-py`` lacks the function or field id.
BINDING_UNSUPPORTED = "binding_unsupported"
#: The driver returned a field value without a timestamp.
NO_TIMESTAMP = "no_timestamp"

#: Instantaneous GPU power through ``nvmlDeviceGetFieldValues``. The value is
#: an unsigned int in milliwatts (DCGM divides it by 1000 for watts) and
#: carries the driver's CPU timestamp in microseconds since 1970. Scope id 0
#: is ``NVML_POWER_SCOPE_GPU``.
POWER_INSTANT_FIELD = "NVML_FI_DEV_POWER_INSTANT"
POWER_INSTANT_FIELD_ID = 186
POWER_INSTANT_SCOPE_ID = 0

_NVML_SUCCESS = 0
_NVML_ERROR_NOT_FOUND = 6
#: ``nvmlValue_t`` members for the integer ``nvmlValueType_t`` values.
_INTEGER_VALUE_MEMBERS = {1: "uiVal", 2: "ulVal", 3: "ullVal", 4: "sllVal", 5: "siVal"}


@dataclass(frozen=True)
class InstantPower:
    milliwatts: int
    #: NVML CPU timestamp of the reading, microseconds since 1970.
    timestamp_us: int


@dataclass(frozen=True)
class ProcessUtilization:
    """One ``nvmlDeviceGetProcessUtilization`` sample, percentages as given."""

    pid: int
    #: NVML CPU timestamp of the sample, microseconds since 1970.
    timestamp_us: int
    sm_pct: int
    mem_pct: int
    enc_pct: int
    dec_pct: int


class NvmlUnavailable(RuntimeError):
    """NVML cannot be used at all on this machine."""

    def __init__(self, state: str, message: str) -> None:
        super().__init__(message)
        self.state = state


class NvmlCallError(RuntimeError):
    """One NVML query failed. The value it would have returned is unknown."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


class NvmlBackend(Protocol):
    """The NVML calls this package uses, keyed by device index."""

    def init(self) -> None:
        """Initialise NVML. Raises :class:`NvmlUnavailable`."""
        ...

    def shutdown(self) -> None: ...

    def versions(self) -> dict[str, str | None]:
        """NVML library, driver, CUDA driver and binding versions."""
        ...

    def device_count(self) -> int: ...

    def device_name(self, index: int) -> str: ...

    def device_uuid(self, index: int) -> str: ...

    def total_energy_mj(self, index: int) -> int:
        """Energy counter in millijoules since the driver was loaded."""
        ...

    def power_mw(self, index: int) -> int:
        """Current power reading in milliwatts."""
        ...

    def power_instant(self, index: int) -> InstantPower:
        """Timestamped instantaneous power from ``NVML_FI_DEV_POWER_INSTANT``."""
        ...

    def enforced_power_limit_mw(self, index: int) -> int: ...

    def mig_mode_enabled(self, index: int) -> bool: ...

    def running_pids(self, index: int) -> tuple[int, ...]:
        """PIDs NVML lists as compute or graphics processes on the device."""
        ...

    def process_utilization(
        self, index: int, since_us: int
    ) -> tuple[ProcessUtilization, ...]:
        """Per process samples newer than ``since_us``. Empty when there are none."""
        ...


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


class PynvmlBackend:
    """:class:`NvmlBackend` backed by the real ``pynvml`` module."""

    def __init__(self, module: Any) -> None:
        self._nvml = module
        self._handles: dict[int, Any] = {}

    def _reason(self, value: Any) -> str:
        nvml = self._nvml
        if value == nvml.NVML_ERROR_NOT_SUPPORTED:
            return NOT_SUPPORTED
        if value == nvml.NVML_ERROR_NO_PERMISSION:
            return NO_PERMISSION
        if value == nvml.NVML_ERROR_GPU_IS_LOST:
            return GPU_LOST
        return CALL_FAILED

    def _call_error(self, exc: Exception) -> NvmlCallError:
        return NvmlCallError(
            self._reason(getattr(exc, "value", None)), f"NVML error: {exc}"
        )

    def _binding(self, name: str) -> Any:
        found = getattr(self._nvml, name, None)
        if found is None:
            raise NvmlCallError(
                BINDING_UNSUPPORTED, f"the installed nvidia-ml-py has no {name}"
            )
        return found

    def _query(self, function: Callable[..., Any], *args: Any) -> Any:
        try:
            return function(*args)
        except self._nvml.NVMLError as exc:
            raise self._call_error(exc) from exc

    def _handle(self, index: int) -> Any:
        if index not in self._handles:
            self._handles[index] = self._query(
                self._nvml.nvmlDeviceGetHandleByIndex, index
            )
        return self._handles[index]

    def init(self) -> None:
        nvml = self._nvml
        try:
            nvml.nvmlInit()
        except nvml.NVMLError as exc:
            value = getattr(exc, "value", None)
            if value == nvml.NVML_ERROR_LIBRARY_NOT_FOUND:
                raise NvmlUnavailable(
                    LIBRARY_UNAVAILABLE,
                    "the NVML shared library (libnvidia-ml) was not found, "
                    "so no NVIDIA driver is installed here",
                ) from exc
            if value == nvml.NVML_ERROR_DRIVER_NOT_LOADED:
                raise NvmlUnavailable(
                    DRIVER_UNAVAILABLE, "the NVIDIA kernel driver is not loaded"
                ) from exc
            raise NvmlUnavailable(NVML_ERROR, f"nvmlInit failed: {exc}") from exc

    def shutdown(self) -> None:
        self._handles.clear()
        try:
            self._nvml.nvmlShutdown()
        except self._nvml.NVMLError:
            pass

    def versions(self) -> dict[str, str | None]:
        def read(function: Callable[[], Any]) -> str | None:
            try:
                return _text(function())
            except self._nvml.NVMLError:
                return None

        cuda = read(self._nvml.nvmlSystemGetCudaDriverVersion)
        if cuda is not None and cuda.isdigit():
            # NVML encodes CUDA 12.4 as 12040.
            cuda = f"{int(cuda) // 1000}.{int(cuda) % 1000 // 10}"
        try:
            binding: str | None = importlib.metadata.version("nvidia-ml-py")
        except importlib.metadata.PackageNotFoundError:
            binding = None
        return {
            "nvml_version": read(self._nvml.nvmlSystemGetNVMLVersion),
            "driver_version": read(self._nvml.nvmlSystemGetDriverVersion),
            "cuda_driver_version": cuda,
            "nvidia_ml_py_version": binding,
        }

    def device_count(self) -> int:
        return int(self._query(self._nvml.nvmlDeviceGetCount))

    def device_name(self, index: int) -> str:
        return _text(self._query(self._nvml.nvmlDeviceGetName, self._handle(index)))

    def device_uuid(self, index: int) -> str:
        return _text(self._query(self._nvml.nvmlDeviceGetUUID, self._handle(index)))

    def total_energy_mj(self, index: int) -> int:
        return int(
            self._query(
                self._nvml.nvmlDeviceGetTotalEnergyConsumption, self._handle(index)
            )
        )

    def power_mw(self, index: int) -> int:
        return int(self._query(self._nvml.nvmlDeviceGetPowerUsage, self._handle(index)))

    def power_instant(self, index: int) -> InstantPower:
        function = self._binding("nvmlDeviceGetFieldValues")
        field_id = self._binding(POWER_INSTANT_FIELD)
        values = self._query(
            function, self._handle(index), [(field_id, POWER_INSTANT_SCOPE_ID)]
        )
        value = values[0]
        code = getattr(self._nvml, "NVML_SUCCESS", _NVML_SUCCESS)
        if value.nvmlReturn != code:
            raise NvmlCallError(
                self._reason(value.nvmlReturn),
                f"{POWER_INSTANT_FIELD} returned NVML code {value.nvmlReturn}",
            )
        member = _INTEGER_VALUE_MEMBERS.get(int(value.valueType))
        if member is None:
            raise NvmlCallError(
                CALL_FAILED,
                f"{POWER_INSTANT_FIELD} has value type {value.valueType}, "
                "not an integer",
            )
        timestamp_us = int(value.timestamp)
        if timestamp_us <= 0:
            raise NvmlCallError(NO_TIMESTAMP, f"{POWER_INSTANT_FIELD} has no timestamp")
        return InstantPower(int(getattr(value.value, member)), timestamp_us)

    def enforced_power_limit_mw(self, index: int) -> int:
        return int(
            self._query(self._nvml.nvmlDeviceGetEnforcedPowerLimit, self._handle(index))
        )

    def mig_mode_enabled(self, index: int) -> bool:
        current, _pending = self._query(
            self._nvml.nvmlDeviceGetMigMode, self._handle(index)
        )
        return bool(current == self._nvml.NVML_DEVICE_MIG_ENABLE)

    def running_pids(self, index: int) -> tuple[int, ...]:
        handle = self._handle(index)
        pids: set[int] = set()
        for function in (
            self._nvml.nvmlDeviceGetComputeRunningProcesses,
            self._nvml.nvmlDeviceGetGraphicsRunningProcesses,
        ):
            for process in self._query(function, handle):
                pids.add(int(process.pid))
        return tuple(sorted(pids))

    def process_utilization(
        self, index: int, since_us: int
    ) -> tuple[ProcessUtilization, ...]:
        nvml = self._nvml
        function = self._binding("nvmlDeviceGetProcessUtilization")
        handle = self._handle(index)
        try:
            samples = function(handle, since_us)
        except nvml.NVMLError as exc:
            # NOT_FOUND means no sample is newer than since_us. The binding
            # raises with NVML_SUCCESS when the sizing call finds none.
            value = getattr(exc, "value", None)
            not_found = getattr(nvml, "NVML_ERROR_NOT_FOUND", _NVML_ERROR_NOT_FOUND)
            if value in (not_found, getattr(nvml, "NVML_SUCCESS", _NVML_SUCCESS)):
                return ()
            raise self._call_error(exc) from exc
        return tuple(
            ProcessUtilization(
                pid=int(sample.pid),
                timestamp_us=int(sample.timeStamp),
                sm_pct=int(sample.smUtil),
                mem_pct=int(sample.memUtil),
                enc_pct=int(sample.encUtil),
                dec_pct=int(sample.decUtil),
            )
            for sample in samples
        )


def load_backend() -> NvmlBackend:
    """Return the real backend. Raises :class:`NvmlUnavailable` without it."""
    try:
        module = importlib.import_module("pynvml")
    except ImportError as exc:
        raise NvmlUnavailable(
            EXTRA_MISSING,
            "the pynvml module is not installed. Install the optional extra "
            f"with {extra_install_hint('nvml')}",
        ) from exc
    return PynvmlBackend(module)
