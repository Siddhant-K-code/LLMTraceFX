"""Report what NVML can measure on this machine, without running anything."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .nvml import (
    NO_PERMISSION,
    NOT_SUPPORTED,
    NvmlBackend,
    NvmlCallError,
    NvmlUnavailable,
)

CAPABILITY_KIND = "llmtracefx.gpu_energy.capability"

SUPPORTED = "supported"
NO_DEVICES = "no_devices"
NO_POWER_READINGS = "no_power_readings"


def uuid_sha256(uuid: str) -> str:
    """Hash a GPU UUID so evidence can match devices without exposing it."""
    return hashlib.sha256(uuid.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Probe:
    """Whether one NVML query worked when it was tried."""

    supported: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"supported": self.supported, "reason": self.reason}


def _probe(call: Any) -> Probe:
    try:
        call()
    except NvmlCallError as exc:
        return Probe(False, exc.reason)
    return Probe(True)


def _optional(call: Any) -> Any:
    try:
        return call()
    except NvmlCallError:
        return None


@dataclass(frozen=True)
class DeviceCapability:
    index: int
    name: str | None
    uuid_sha256: str | None
    mig_enabled: bool | None
    enforced_power_limit_w: float | None
    energy_counter: Probe
    power_reading: Probe
    process_listing: Probe

    @property
    def can_measure(self) -> bool:
        return self.energy_counter.supported or self.power_reading.supported

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "uuid_sha256": self.uuid_sha256,
            "mig_enabled": self.mig_enabled,
            "enforced_power_limit_w": self.enforced_power_limit_w,
            "energy_counter": self.energy_counter.as_dict(),
            "power_reading": self.power_reading.as_dict(),
            "process_listing": self.process_listing.as_dict(),
        }


@dataclass(frozen=True)
class NvmlCapability:
    state: str
    message: str
    versions: dict[str, str | None] = field(default_factory=dict)
    devices: tuple[DeviceCapability, ...] = ()

    @property
    def is_supported(self) -> bool:
        return self.state == SUPPORTED

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": CAPABILITY_KIND,
            "schema_version": "1",
            "source": "nvml",
            "state": self.state,
            "supported": self.is_supported,
            "message": self.message,
            "collector": dict(self.versions),
            "devices": [device.as_dict() for device in self.devices],
        }


def probe_device(backend: NvmlBackend, index: int) -> DeviceCapability:
    uuid = _optional(lambda: backend.device_uuid(index))
    limit = _optional(lambda: backend.enforced_power_limit_mw(index))
    return DeviceCapability(
        index=index,
        name=_optional(lambda: backend.device_name(index)),
        uuid_sha256=None if uuid is None else uuid_sha256(uuid),
        mig_enabled=_optional(lambda: backend.mig_mode_enabled(index)),
        enforced_power_limit_w=None if limit is None else limit / 1000.0,
        energy_counter=_probe(lambda: backend.total_energy_mj(index)),
        power_reading=_probe(lambda: backend.power_mw(index)),
        process_listing=_probe(lambda: backend.running_pids(index)),
    )


def probe_capability(backend: NvmlBackend) -> NvmlCapability:
    """Probe an initialised backend. Every device query is actually tried."""
    versions = backend.versions()
    try:
        count = backend.device_count()
    except NvmlCallError as exc:
        return NvmlCapability(
            state="nvml_error",
            message=f"NVML could not count devices: {exc}",
            versions=versions,
        )
    if count == 0:
        return NvmlCapability(
            state=NO_DEVICES, message="NVML reports no GPUs", versions=versions
        )
    devices = tuple(probe_device(backend, index) for index in range(count))
    if not any(device.can_measure for device in devices):
        reasons = sorted(
            {
                str(probe.reason)
                for device in devices
                for probe in (device.energy_counter, device.power_reading)
            }
        )
        hint = ""
        if NO_PERMISSION in reasons:
            hint = " Some queries need more privileges."
        elif reasons == [NOT_SUPPORTED]:
            hint = " The energy counter needs a Volta or newer GPU."
        return NvmlCapability(
            state=NO_POWER_READINGS,
            message=(
                "no GPU supports the energy counter or a power reading "
                f"({', '.join(reasons)}).{hint}"
            ),
            versions=versions,
            devices=devices,
        )
    return NvmlCapability(
        state=SUPPORTED,
        message=f"NVML can read power or energy on {count} GPU(s)",
        versions=versions,
        devices=devices,
    )


def check_capability(backend_factory: Any) -> NvmlCapability:
    """Load, initialise, probe and shut down. Never raises for NVML causes."""
    try:
        backend = backend_factory()
        backend.init()
    except NvmlUnavailable as exc:
        return NvmlCapability(state=exc.state, message=str(exc))
    try:
        return probe_capability(backend)
    finally:
        backend.shutdown()


def capability_exit_code(capability: NvmlCapability) -> int:
    """0 when supported, 3 when a known cause blocks it."""
    return 0 if capability.is_supported else 3
