"""Turn one dcgm-exporter window into evidence in the ``gpu-energy`` format.

The keys match :mod:`.evidence` so readers handle both sources the same
way. What differs is stated, not hidden:

* ``energy`` is the delta of the exporter's energy counter between two
  counter edges. It is ``measured_native``, but it covers a span that
  contains the command and is longer than it. ``bracket`` records the
  bounds of that span, and the energy is refused when the command is less
  than ``required_min_run_share`` of the longest span it could cover.
* ``average_power`` and every sampled power statistic are refused, because
  the exporter does not say when DCGM took each reading.
* attribution is never verified: the exporter does not list GPU processes.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any

from ..instruments.commands import redact_argv
from ..schema import MetricProvenance
from .dcgm_exporter import PROCESS_LISTING_UNAVAILABLE, SOURCE
from .dcgm_window import DcgmDeviceReadings, DcgmWindowRecord
from .evidence import (
    EVIDENCE_KIND,
    EVIDENCE_SCHEMA_VERSION,
    HIGHLIGHT_KEYS,
    OPERATOR_ASSERTED,
    UNVERIFIABLE,
    _measurement,
    _ms,
    _totals,
)

BRACKET_TOO_WIDE = "bracket_too_wide"
NO_COUNTER_UPDATE_AFTER_RUN = "no_counter_update_after_run"
COUNTER_RESET = "counter_reset"
SAMPLE_TIMES_UNKNOWN = "sample_times_unknown"

NOTES = (
    "Energy and power are device wide. They include idle power and the "
    "command's own start up and shut down, not only inference.",
    "dcgm-exporter output has no timestamps and DCGM refreshes it every "
    "collect interval (default 30000 ms). The energy delta therefore covers "
    "a span that starts before and ends after the command. bracket records "
    "the host clock bounds of that span.",
    "DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL is truncated to whole joules by "
    "DCGM. DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION has 1 mJ resolution.",
    "Power samples are the exporter's latest cached value at each scrape. "
    "When DCGM read them is unknown, so no mean, maximum or integral is "
    "computed from them.",
    "dcgm-exporter does not list GPU processes, so the window cannot be "
    "verified as free of other work.",
)


def _bracket(
    device: DcgmDeviceReadings, record: DcgmWindowRecord, min_run_share: float
) -> dict[str, Any] | None:
    start, end = device.start, device.end
    if start is None or end is None:
        return None
    run_ms = record.command_end_ms - record.command_start_ms
    max_span_ms = end.by_ms - start.after_ms
    min_span_ms = max(0.0, end.after_ms - start.by_ms)
    share = run_ms / max_span_ms if max_span_ms > 0 else 0.0
    return {
        "method": "counter_change_edges_on_host_monotonic_clock",
        "start_sampled_after_offset_ms": _ms(start.after_ms),
        "start_sampled_by_offset_ms": _ms(start.by_ms),
        "end_sampled_after_offset_ms": _ms(end.after_ms),
        "end_sampled_by_offset_ms": _ms(end.by_ms),
        "command_ms": _ms(run_ms),
        "min_span_ms": _ms(min_span_ms),
        "max_span_ms": _ms(max_span_ms),
        "min_command_share": round(share, 6),
        "required_min_run_share": min_run_share,
    }


def _energy(
    device: DcgmDeviceReadings, bracket: dict[str, Any] | None
) -> tuple[dict[str, Any] | None, str | None]:
    if device.counter_reset:
        return None, COUNTER_RESET
    if device.start is None or device.end is None or bracket is None:
        return None, NO_COUNTER_UPDATE_AFTER_RUN
    delta = device.end.value - device.start.value
    if delta < 0:
        return None, COUNTER_RESET
    if bracket["min_command_share"] < bracket["required_min_run_share"]:
        return None, BRACKET_TOO_WIDE
    return (
        _measurement(
            delta * device.metric.joules_per_unit,
            "J",
            MetricProvenance.MEASURED_NATIVE,
            "dcgm_exporter_counter_delta_over_bracket",
        ),
        None,
    )


def _counter(device: DcgmDeviceReadings) -> dict[str, Any]:
    start, end = device.start, device.end
    return {
        "metric": device.metric.name,
        "unit": device.metric.unit,
        "resolution": device.metric.resolution,
        "start": None if start is None else start.value,
        "end": None if end is None else end.value,
        "changes_seen": len(device.changes),
        "read_errors": dict(sorted(device.read_errors.items())),
    }


def _device_entry(
    device: DcgmDeviceReadings, record: DcgmWindowRecord, min_run_share: float
) -> dict[str, Any]:
    bracket = _bracket(device, record, min_run_share)
    energy, refused = _energy(device, bracket)
    return {
        "index": device.index,
        "name": device.name,
        "uuid_sha256": device.uuid_sha256,
        "mig_enabled": True if device.mig_seen else None,
        "enforced_power_limit_w": device.enforced_power_limit_w,
        "energy_counter": _counter(device),
        "bracket": bracket,
        "energy": energy,
        "average_power": None,
        "average_power_refused_reason": SAMPLE_TIMES_UNKNOWN,
        "energy_refused_reason": refused,
        "sampled_power_mean": None,
        "sampled_power_max": None,
        "sampled_power_integral": None,
        "sampled_power_refused_reason": SAMPLE_TIMES_UNKNOWN,
        "power_samples": {
            "metric": device.power_metric,
            "count": len(device.samples),
            "capped": device.capped,
            "errors": dict(sorted(device.sample_errors.items())),
            "columns": ["scrape_end_offset_ms", "power_w"],
            "provenance": MetricProvenance.MEASURED_NATIVE.value,
            "values": [[_ms(t), watts] for t, watts in device.samples],
        },
    }


def _observed_update_gap(
    devices: Sequence[DcgmDeviceReadings], scrape_interval_ms: float
) -> dict[str, Any] | None:
    gaps = [
        b - a
        for device in devices
        for a, b in zip(device.changes, device.changes[1:], strict=False)
    ]
    if not gaps:
        return None
    out = _measurement(
        _ms(statistics.median(gaps)),
        "ms",
        MetricProvenance.DERIVED,
        "median_gap_between_counter_changes_seen_by_scrapes",
    )
    out["uncertainty_ms"] = scrape_interval_ms
    return out


def _attribution(
    devices: Sequence[DcgmDeviceReadings],
    record: DcgmWindowRecord,
    *,
    assume_isolated: bool,
) -> dict[str, Any]:
    problems = ["dcgm-exporter does not list GPU processes"]
    mig = [device.index for device in devices if device.mig_seen]
    if mig:
        problems.append(
            f"MIG labels were seen on GPU {', '.join(map(str, mig))}, so other "
            "instances share the device wide readings"
        )
    reason = "; ".join(problems)
    status = OPERATOR_ASSERTED if assume_isolated else UNVERIFIABLE
    if assume_isolated:
        reason = f"--assume-isolated accepted the window although: {reason}"
    return {
        "status": status,
        "attributable": assume_isolated,
        "reason": reason,
        "process_checks": 0,
        "process_listing": PROCESS_LISTING_UNAVAILABLE,
        "listing_errors": {},
        "allowed_roots": [record.child_pid],
        "gpu_pids": [],
        "assume_isolated": assume_isolated,
        "processes": [],
    }


def build_dcgm_evidence(
    record: DcgmWindowRecord,
    *,
    endpoint: str,
    argv: Sequence[str],
    started_at: str,
    ended_at: str,
    min_run_share: float,
    stated_collect_interval_ms: float | None = None,
    assume_isolated: bool = False,
) -> dict[str, Any]:
    entries = [
        _device_entry(device, record, min_run_share) for device in record.devices
    ]
    totals = _totals(entries)
    attribution = _attribution(record.devices, record, assume_isolated=assume_isolated)

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

    stated = None
    if stated_collect_interval_ms is not None:
        stated = {
            "value": stated_collect_interval_ms,
            "unit": "ms",
            "stated_by": "operator",
            "verified": False,
        }
    power_metrics = sorted({d.power_metric for d in record.devices if d.power_metric})
    return {
        "kind": EVIDENCE_KIND,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "scope": "device_wide_window",
        "source": SOURCE,
        "collector": {
            "name": SOURCE,
            "endpoint": endpoint,
            "energy_metrics": sorted({d.metric.name for d in record.devices}),
            "power_metrics": power_metrics,
        },
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
            "start_offset_ms": _ms(record.command_start_ms),
            "end_offset_ms": _ms(record.command_end_ms),
        },
        "power_sampling": {
            "source": f"{SOURCE}:{','.join(power_metrics) or 'none'}",
            "interval_ms": record.scrape_interval_ms,
            "max_samples_per_device": record.max_samples,
            "scrapes": record.scrapes,
            "scrape_errors": dict(sorted(record.scrape_errors.items())),
            "edge_timeout_ms": record.edge_timeout_ms,
            "collect_interval_stated": stated,
            "collect_interval_observed": _observed_update_gap(
                record.devices, record.scrape_interval_ms
            ),
        },
        "devices": entries,
        "totals": totals,
        "attribution": attribution,
        "highlights": highlights,
        "highlights_withheld_reason": withheld,
        "notes": list(NOTES),
    }
