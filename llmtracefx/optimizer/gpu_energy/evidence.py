"""Turn one recorded window into labelled evidence.

Two different energy figures can come out of a window, and they are never
merged or given the same name:

``energy``
    The delta of NVML's total energy counter between the start and end
    reads. This is a hardware counter, labelled ``measured_native``.
``sampled_power_integral``
    A trapezoid over the polled power readings. This is an estimate,
    labelled ``estimated``, and it only covers the span between the first
    and last sample.

When a value cannot be computed honestly it is left ``null`` with a reason.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

from ..collectors._shared import atomic_write_text
from ..instruments.commands import redact_argv
from ..schema import MetricProvenance
from .capability import DeviceCapability
from .window import (
    ALLOWED,
    ANCESTRY_UNKNOWN,
    FOREIGN,
    NOT_VISIBLE,
    DeviceReadings,
    WindowRecord,
)

EVIDENCE_KIND = "llmtracefx.gpu_energy.evidence"
EVIDENCE_SCHEMA_VERSION = "1"

NO_FOREIGN_PROCESS_OBSERVED = "no_foreign_process_observed"
OPERATOR_ASSERTED = "operator_asserted"
UNVERIFIABLE = "unverifiable"
CONTAMINATED = "contaminated"
ATTRIBUTABLE_STATUSES = frozenset({NO_FOREIGN_PROCESS_OBSERVED, OPERATOR_ASSERTED})

NOTES = (
    "Energy and power are device wide. They include idle power and the "
    "command's own start up and shut down, not only inference.",
    "On Ampere (except GA100) and newer GPUs, NVML power usage is a one "
    "second average, so sampled power smooths and lags short changes.",
    "The energy counter updates at a driver defined interval, so a very "
    "short window can read a coarse or zero delta.",
    "sampled_power_integral is estimated from the power samples and covers "
    "only the span between the first and last sample. It is not the "
    "energy counter.",
)


def _measurement(
    value: float, unit: str, provenance: MetricProvenance, method: str
) -> dict[str, Any]:
    return {
        "value": value,
        "unit": unit,
        "provenance": provenance.value,
        "method": method,
    }


def _ms(value: float) -> float:
    return round(value, 3)


def _counter(device: DeviceReadings) -> dict[str, Any]:
    start, end = device.energy_start, device.energy_end
    return {
        "unit": "mJ",
        "start": None if start is None else start.value_mj,
        "end": None if end is None else end.value_mj,
        "start_offset_ms": None if start is None else _ms(start.offset_ms),
        "end_offset_ms": None if end is None else _ms(end.offset_ms),
        "start_error": None if start is None else start.error,
        "end_error": None if end is None else end.error,
    }


def _energy(
    device: DeviceReadings,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    """Return ``(energy, average_power, refused_reason)``."""
    start, end = device.energy_start, device.energy_end
    if start is None or end is None:
        return None, None, "counter_not_read"
    for read in (start, end):
        if read.error is not None:
            return None, None, f"counter_{read.error}"
    assert start.value_mj is not None and end.value_mj is not None
    delta_mj = end.value_mj - start.value_mj
    if delta_mj < 0:
        return None, None, "counter_reset"
    if delta_mj == 0:
        return None, None, "stale_counter"
    joules = delta_mj / 1000.0
    energy = _measurement(
        joules,
        "J",
        MetricProvenance.MEASURED_NATIVE,
        "total_energy_counter_delta",
    )
    seconds = (end.offset_ms - start.offset_ms) / 1000.0
    if seconds <= 0:
        return energy, None, None
    power = _measurement(
        joules / seconds,
        "W",
        MetricProvenance.DERIVED,
        "energy_over_counter_read_interval",
    )
    return energy, power, None


def _sampled(device: DeviceReadings) -> dict[str, Any]:
    samples = device.samples
    out: dict[str, Any] = {
        "sampled_power_mean": None,
        "sampled_power_max": None,
        "sampled_power_integral": None,
        "sampled_power_refused_reason": None,
    }
    if device.capped:
        out["sampled_power_refused_reason"] = "sample_cap_reached"
        return out
    if device.sample_errors:
        out["sampled_power_refused_reason"] = "sample_errors"
        return out
    if not samples:
        out["sampled_power_refused_reason"] = "no_samples"
        return out
    watts = [milliwatts / 1000.0 for _, milliwatts in samples]
    out["sampled_power_mean"] = _measurement(
        sum(watts) / len(watts),
        "W",
        MetricProvenance.DERIVED,
        "arithmetic_mean_of_samples",
    )
    out["sampled_power_max"] = _measurement(
        max(watts), "W", MetricProvenance.DERIVED, "max_of_samples"
    )
    if len(samples) < 2:
        out["sampled_power_refused_reason"] = "fewer_than_two_samples"
        return out
    joules = 0.0
    for (t_a, p_a), (t_b, p_b) in pairwise(samples):
        joules += (p_a + p_b) / 2000.0 * (t_b - t_a) / 1000.0
    out["sampled_power_integral"] = _measurement(
        joules,
        "J",
        MetricProvenance.ESTIMATED,
        "trapezoid_over_sampled_power",
    )
    return out


def _device_entry(
    capability: DeviceCapability, readings: DeviceReadings
) -> dict[str, Any]:
    energy, average_power, refused = _energy(readings)
    entry: dict[str, Any] = {
        "index": capability.index,
        "name": capability.name,
        "uuid_sha256": capability.uuid_sha256,
        "mig_enabled": capability.mig_enabled,
        "enforced_power_limit_w": capability.enforced_power_limit_w,
        "energy_counter": _counter(readings),
        "energy": energy,
        "average_power": average_power,
        "energy_refused_reason": refused,
    }
    entry.update(_sampled(readings))
    entry["power_samples"] = {
        "count": len(readings.samples),
        "capped": readings.capped,
        "errors": dict(sorted(readings.sample_errors.items())),
        "columns": ["offset_ms", "power_mw"],
        "provenance": MetricProvenance.MEASURED_NATIVE.value,
        "values": [[_ms(t), mw] for t, mw in readings.samples],
    }
    return entry


def _sum(
    entries: Sequence[dict[str, Any]],
    key: str,
    provenance: MetricProvenance,
    method: str,
) -> dict[str, Any] | None:
    values = [entry[key] for entry in entries]
    if not values or any(value is None for value in values):
        return None
    return _measurement(
        sum(value["value"] for value in values), values[0]["unit"], provenance, method
    )


def _totals(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    maxima = [entry["sampled_power_max"] for entry in entries]
    peak = None
    if maxima and all(value is not None for value in maxima):
        peak = _measurement(
            max(value["value"] for value in maxima),
            "W",
            MetricProvenance.DERIVED,
            "max_of_sampled_device_power",
        )
    return {
        "device_count": len(entries),
        "energy": _sum(
            entries, "energy", MetricProvenance.DERIVED, "sum_of_device_energy"
        ),
        "average_power": _sum(
            entries,
            "average_power",
            MetricProvenance.DERIVED,
            "sum_of_device_average_power",
        ),
        "sampled_power_integral": _sum(
            entries,
            "sampled_power_integral",
            MetricProvenance.ESTIMATED,
            "sum_of_device_sampled_power_integral",
        ),
        "sampled_power_max": peak,
    }


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _attribution(
    record: WindowRecord,
    devices: Sequence[DeviceCapability],
    *,
    assume_isolated: bool,
    gpu_pids: Sequence[int],
) -> dict[str, Any]:
    checks = record.processes
    observed = sorted(checks.processes.values(), key=lambda p: p.pid)
    by_class: dict[str, list[int]] = {}
    for process in observed:
        by_class.setdefault(process.classification, []).append(process.pid)

    status: str
    if by_class.get(FOREIGN):
        pids = by_class[FOREIGN]
        status = CONTAMINATED
        reason = (
            f"{_plural(len(pids), 'process')} outside the command used a "
            f"selected GPU during the window (pid {', '.join(map(str, pids))})"
        )
    else:
        problems: list[str] = []
        if checks.listing_errors:
            errors = ", ".join(sorted(checks.listing_errors))
            problems.append(f"NVML process listing failed ({errors})")
        if by_class.get(NOT_VISIBLE):
            problems.append(
                f"{_plural(len(by_class[NOT_VISIBLE]), 'listed process')} "
                "not visible here, for example from another PID namespace"
            )
        if by_class.get(ANCESTRY_UNKNOWN):
            problems.append("process ancestry cannot be read on this platform")
        mig = [device.index for device in devices if device.mig_enabled]
        if mig:
            problems.append(
                f"MIG is enabled on GPU {', '.join(map(str, mig))}, so other "
                "instances share the device wide readings"
            )
        if not by_class.get(ALLOWED):
            problems.append(
                "NVML never listed the command, its children or a --gpu-pid "
                "as a GPU process"
            )
        if problems:
            status = OPERATOR_ASSERTED if assume_isolated else UNVERIFIABLE
            reason = "; ".join(problems)
            if assume_isolated:
                reason = f"--assume-isolated accepted the window although: {reason}"
        else:
            status = NO_FOREIGN_PROCESS_OBSERVED
            reason = (
                f"{_plural(checks.checks, 'process check')} listed only the "
                "command, its children or a --gpu-pid on the selected GPUs"
            )
    return {
        "status": status,
        "attributable": status in ATTRIBUTABLE_STATUSES,
        "reason": reason,
        "process_checks": checks.checks,
        "listing_errors": dict(sorted(checks.listing_errors.items())),
        "allowed_roots": sorted(record.allowed_roots),
        "gpu_pids": sorted(gpu_pids),
        "assume_isolated": assume_isolated,
        "processes": [
            {
                "pid": process.pid,
                "classification": process.classification,
                "devices": sorted(process.devices),
                "first_seen_offset_ms": _ms(process.first_seen_ms),
            }
            for process in observed
        ],
    }


HIGHLIGHT_KEYS = (
    ("gpu_energy", "energy"),
    ("gpu_average_power", "average_power"),
    ("gpu_sampled_power_integral", "sampled_power_integral"),
    ("gpu_sampled_power_max", "sampled_power_max"),
)


def build_evidence(
    record: WindowRecord,
    devices: Sequence[DeviceCapability],
    *,
    versions: dict[str, str | None],
    argv: Sequence[str],
    started_at: str,
    ended_at: str,
    assume_isolated: bool = False,
    gpu_pids: Sequence[int] = (),
) -> dict[str, Any]:
    by_index = {readings.index: readings for readings in record.devices}
    entries = [_device_entry(device, by_index[device.index]) for device in devices]
    totals = _totals(entries)
    attribution = _attribution(
        record, devices, assume_isolated=assume_isolated, gpu_pids=gpu_pids
    )

    withheld: str | None = None
    if not attribution["attributable"]:
        withheld = attribution["reason"]
    elif record.exit_code != 0:
        withheld = f"the command exited with code {record.exit_code}"
    highlights: dict[str, Any] = {}
    if withheld is None:
        for key, total in HIGHLIGHT_KEYS:
            if totals[total] is not None:
                highlights[key] = totals[total]

    return {
        "kind": EVIDENCE_KIND,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "scope": "device_wide_window",
        "collector": {"name": "nvml", **versions},
        "started_at": started_at,
        "ended_at": ended_at,
        "window": _measurement(
            _ms(record.window_ms),
            "ms",
            MetricProvenance.MEASURED_WALL_CLOCK,
            "host_monotonic_clock",
        ),
        "command": {
            "argv": list(redact_argv(argv)),
            "pid": record.child_pid,
            "exit_code": record.exit_code,
        },
        "power_sampling": {
            "source": "nvmlDeviceGetPowerUsage",
            "interval_ms": record.interval_ms,
            "max_samples_per_device": record.max_samples,
        },
        "devices": entries,
        "totals": totals,
        "attribution": attribution,
        "highlights": highlights,
        "highlights_withheld_reason": withheld,
        "notes": list(NOTES),
    }


def write_evidence(path: Path, evidence: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(evidence, indent=2, allow_nan=False) + "\n")
