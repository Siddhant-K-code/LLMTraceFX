"""Fill ``ExperimentRecord.power`` from ``gpu-energy record`` evidence.

The evidence is a sidecar file, :data:`GPU_ENERGY_EVIDENCE_FILE`, next to
``record.json``. The record file is never rewritten: ``collect-api`` seals
its artifacts with hashes, so power is filled on a copy in memory.

Only an attributable window fills power, and only from the energy counter:

* one device: its ``energy`` (``measured_native``) and ``average_power``
  (``derived``);
* several devices: the ``totals`` sums, both ``derived``.

The sampled power integral is an estimate and is never used. When the
evidence cannot be used the values stay empty and
``power.source.refused_reason`` says why. Malformed evidence, or a record
that already carries power, is an error rather than a refusal.

Standard library only: the CI gate imports this without extras.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .._artifact_io import (
    MAX_EVIDENCE_ARTIFACT_BYTES,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from ..schema import (
    ExperimentRecord,
    Measurement,
    MetricProvenance,
    PowerMetrics,
    PowerSource,
)
from .evidence import (
    ATTRIBUTABLE_STATUSES,
    CONTAMINATED,
    EVIDENCE_KIND,
    EVIDENCE_SCHEMA_VERSION,
    NO_FOREIGN_PROCESS_OBSERVED,
    OPERATOR_ASSERTED,
    UNVERIFIABLE,
)

GPU_ENERGY_EVIDENCE_FILE = "gpu_energy_evidence.json"
EVIDENCE_SCOPE = "device_wide_window"
ENERGY_PER_OUTPUT_TOKEN_UNIT = "J/{token}"
ENERGY_PER_OUTPUT_TOKEN_BASIS = (
    "device wide energy of the gpu-energy window divided by this record's "
    "generated tokens; includes idle power and command start up"
)
_STATUSES = frozenset(
    {NO_FOREIGN_PROCESS_OBSERVED, OPERATOR_ASSERTED, UNVERIFIABLE, CONTAMINATED}
)


class GpuEnergyEvidenceError(ValueError):
    """The evidence is malformed or cannot be applied to this record."""


def read_gpu_energy_evidence(path: Path) -> tuple[dict[str, Any], str]:
    """Read and check one evidence file; return it and its sha256."""
    try:
        text = read_bounded_regular_text(path, MAX_EVIDENCE_ARTIFACT_BYTES)
        data = json.loads(text, parse_constant=reject_non_finite_json_constant)
    except (OSError, ValueError) as exc:
        raise GpuEnergyEvidenceError(f"cannot read {path}: {exc}") from exc
    try:
        _check(data)
    except GpuEnergyEvidenceError as exc:
        raise GpuEnergyEvidenceError(f"{path}: {exc}") from exc
    return data, hashlib.sha256(text.encode("utf-8")).hexdigest()


def _timestamp(value: Any, name: str) -> datetime:
    if not isinstance(value, str):
        raise GpuEnergyEvidenceError(f"{name} must be a timestamp string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise GpuEnergyEvidenceError(f"{name} is not a timestamp: {value!r}") from exc
    if parsed.tzinfo is None:
        raise GpuEnergyEvidenceError(f"{name} has no timezone: {value!r}")
    return parsed


def _object(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise GpuEnergyEvidenceError(f"{name} must be an object")
    return value


def _check_measurement(
    value: Any, name: str, unit: str, provenance: MetricProvenance
) -> None:
    if value is None:
        return
    entry = _object(value, name)
    number = entry.get("value")
    if (
        isinstance(number, bool)
        or not isinstance(number, (int, float))
        or not math.isfinite(number)
        or number < 0
    ):
        raise GpuEnergyEvidenceError(f"{name}.value must be a finite number >= 0")
    if entry.get("unit") != unit:
        raise GpuEnergyEvidenceError(f"{name}.unit must be {unit!r}")
    if entry.get("provenance") != provenance.value:
        raise GpuEnergyEvidenceError(f"{name}.provenance must be {provenance.value!r}")
    if not isinstance(entry.get("method"), str):
        raise GpuEnergyEvidenceError(f"{name}.method must be a string")


def _check(data: Any) -> None:
    evidence = _object(data, "evidence")
    if evidence.get("kind") != EVIDENCE_KIND:
        raise GpuEnergyEvidenceError(f"not a {EVIDENCE_KIND} document")
    if evidence.get("schema_version") != EVIDENCE_SCHEMA_VERSION:
        raise GpuEnergyEvidenceError(
            f"unsupported schema_version {evidence.get('schema_version')!r}"
        )
    if evidence.get("scope") != EVIDENCE_SCOPE:
        raise GpuEnergyEvidenceError(f"scope must be {EVIDENCE_SCOPE!r}")
    if not isinstance(_object(evidence.get("collector"), "collector").get("name"), str):
        raise GpuEnergyEvidenceError("collector.name must be a string")
    if _timestamp(evidence.get("started_at"), "started_at") > _timestamp(
        evidence.get("ended_at"), "ended_at"
    ):
        raise GpuEnergyEvidenceError("started_at is after ended_at")
    exit_code = _object(evidence.get("command"), "command").get("exit_code")
    if exit_code is not None and (
        isinstance(exit_code, bool) or not isinstance(exit_code, int)
    ):
        raise GpuEnergyEvidenceError("command.exit_code must be an integer or null")
    attribution = _object(evidence.get("attribution"), "attribution")
    status = attribution.get("status")
    if status not in _STATUSES:
        raise GpuEnergyEvidenceError(f"unknown attribution.status {status!r}")
    if attribution.get("attributable") is not (status in ATTRIBUTABLE_STATUSES):
        raise GpuEnergyEvidenceError(
            "attribution.attributable does not match attribution.status"
        )
    withheld = evidence.get("highlights_withheld_reason")
    if withheld is not None and not isinstance(withheld, str):
        raise GpuEnergyEvidenceError("highlights_withheld_reason must be a string")
    devices = evidence.get("devices")
    if not isinstance(devices, list) or not devices:
        raise GpuEnergyEvidenceError("devices must be a non-empty list")
    for index, raw in enumerate(devices):
        device = _object(raw, f"devices[{index}]")
        if not isinstance(device.get("name"), str) or not device["name"]:
            raise GpuEnergyEvidenceError(f"devices[{index}].name must be a string")
        _check_measurement(
            device.get("energy"),
            f"devices[{index}].energy",
            "J",
            MetricProvenance.MEASURED_NATIVE,
        )
        _check_measurement(
            device.get("average_power"),
            f"devices[{index}].average_power",
            "W",
            MetricProvenance.DERIVED,
        )
    totals = _object(evidence.get("totals"), "totals")
    if totals.get("device_count") != len(devices):
        raise GpuEnergyEvidenceError("totals.device_count does not match devices")
    _check_measurement(
        totals.get("energy"), "totals.energy", "J", MetricProvenance.DERIVED
    )
    _check_measurement(
        totals.get("average_power"),
        "totals.average_power",
        "W",
        MetricProvenance.DERIVED,
    )


def _as_measurement(entry: Mapping[str, Any] | None) -> Measurement | None:
    if entry is None:
        return None
    return Measurement(
        value=float(entry["value"]),
        provenance=MetricProvenance(entry["provenance"]),
        unit=entry["unit"],
    )


def _refusal(record: ExperimentRecord, evidence: Mapping[str, Any]) -> str | None:
    attribution = evidence["attribution"]
    if not attribution["attributable"]:
        return (
            f"the window is not attributable ({attribution['status']}): "
            f"{attribution.get('reason')}"
        )
    exit_code = evidence["command"].get("exit_code")
    if exit_code is None:
        return "the measured command exit code is not recorded"
    if exit_code != 0:
        return f"the measured command exited with code {exit_code}"
    if evidence.get("highlights_withheld_reason") is not None:
        return str(evidence["highlights_withheld_reason"])
    if record.ended_at is None:
        return "the record has no ended_at, so its run cannot be placed in the window"
    try:
        run_start = _timestamp(record.started_at, "record started_at")
        run_end = _timestamp(record.ended_at, "record ended_at")
    except GpuEnergyEvidenceError as exc:
        return str(exc)
    window_start = _timestamp(evidence["started_at"], "started_at")
    window_end = _timestamp(evidence["ended_at"], "ended_at")
    if run_start < window_start or run_end > window_end:
        return (
            f"the run ({record.started_at} to {record.ended_at}) is not inside the "
            f"evidence window ({evidence['started_at']} to {evidence['ended_at']})"
        )
    devices = evidence["devices"]
    if len(devices) == 1:
        energy = devices[0].get("energy")
    else:
        energy = evidence["totals"].get("energy")
    if energy is None:
        reasons = sorted(
            {
                str(device.get("energy_refused_reason"))
                for device in devices
                if device.get("energy") is None
            }
        )
        return (
            f"the energy counter was refused ({', '.join(reasons)}); the sampled "
            "power integral is an estimate and is not used"
        )
    return None


def power_from_evidence(
    record: ExperimentRecord, evidence: Mapping[str, Any], sha256: str
) -> PowerMetrics:
    """Power for ``record`` from checked evidence, or an empty refusal."""
    devices = evidence["devices"]
    chosen = devices[0] if len(devices) == 1 else evidence["totals"]
    refused = _refusal(record, evidence)
    energy = None if refused else chosen.get("energy")
    power = None if refused else chosen.get("average_power")
    source = PowerSource(
        collector=str(evidence["collector"]["name"]),
        evidence_kind=EVIDENCE_KIND,
        evidence_sha256=sha256,
        scope=EVIDENCE_SCOPE,
        attribution_status=str(evidence["attribution"]["status"]),
        window_started_at=str(evidence["started_at"]),
        window_ended_at=str(evidence["ended_at"]),
        device_count=len(devices),
        device_names=tuple(str(device["name"]) for device in devices),
        energy_method=None if energy is None else str(energy["method"]),
        average_power_method=None if power is None else str(power["method"]),
        refused_reason=refused,
    )
    return PowerMetrics(
        average_power=_as_measurement(power),
        energy=_as_measurement(energy),
        source=source,
    )


def apply_gpu_energy(record: ExperimentRecord, path: Path) -> ExperimentRecord:
    """A copy of ``record`` with power filled, or refused, from ``path``."""
    current = record.power
    if (
        current.energy is not None
        or current.average_power is not None
        or current.source is not None
    ):
        raise GpuEnergyEvidenceError(
            f"{record.run_id} already carries power values; {path} is not mixed in"
        )
    evidence, sha256 = read_gpu_energy_evidence(path)
    filled = dataclasses.replace(
        record, power=power_from_evidence(record, evidence, sha256)
    )
    filled.validate()
    return filled


def apply_sidecar(record: ExperimentRecord, record_path: Path) -> ExperimentRecord:
    """Apply the sidecar next to ``record_path`` when there is one."""
    sidecar = record_path.parent / GPU_ENERGY_EVIDENCE_FILE
    if not sidecar.is_file():
        return record
    return apply_gpu_energy(record, sidecar)


def _span(start: str, end: str | None) -> tuple[datetime, datetime] | None:
    try:
        begin = _timestamp(start, "started_at")
        return begin, begin if end is None else _timestamp(end, "ended_at")
    except GpuEnergyEvidenceError:
        return None


def shared_window_refusals(records: Iterable[ExperimentRecord]) -> dict[str, str]:
    """Runs whose power cannot be told apart from another loaded run.

    A device wide window covers everything that ran inside it. When two
    loaded runs use the same evidence file, or one window also contains
    another loaded run, the energy is not this run's alone.
    """
    records = list(records)
    by_sha: dict[str, list[str]] = {}
    for record in records:
        source = record.power.source
        if source is not None and source.refused_reason is None:
            by_sha.setdefault(source.evidence_sha256, []).append(record.run_id)
    refusals: dict[str, str] = {}
    for run_ids in by_sha.values():
        if len(run_ids) > 1:
            for run_id in run_ids:
                others = ", ".join(sorted(set(run_ids) - {run_id}))
                refusals[run_id] = f"the same gpu-energy evidence is used by {others}"
    for record in records:
        source = record.power.source
        if (
            record.run_id in refusals
            or source is None
            or source.refused_reason is not None
            or source.window_started_at is None
            or source.window_ended_at is None
        ):
            continue
        window = _span(source.window_started_at, source.window_ended_at)
        if window is None:
            continue
        inside = []
        for other in records:
            if other.run_id == record.run_id:
                continue
            span = _span(other.started_at, other.ended_at)
            if span is not None and span[0] <= window[1] and span[1] >= window[0]:
                inside.append(other.run_id)
        if inside:
            refusals[record.run_id] = (
                "the gpu-energy window also covers "
                f"{', '.join(sorted(inside))}, so its energy is not this run's alone"
            )
    return refusals


def energy_per_output_token(record: ExperimentRecord) -> Measurement | str:
    """``derived`` energy per generated token, or why there is none.

    The basis is :data:`ENERGY_PER_OUTPUT_TOKEN_BASIS`.
    """
    energy = record.power.energy
    if energy is None:
        return "no power.energy"
    if energy.unit != "J" or not math.isfinite(energy.value):
        return "power.energy is not a finite value in J"
    tokens = record.tokens.generated_tokens
    provenance = record.tokens.provenance
    if provenance is None:
        return "token provenance is not recorded"
    if provenance is MetricProvenance.ESTIMATED:
        return "generated token count is estimated"
    if tokens is None or tokens <= 0:
        return "no positive generated token count"
    return Measurement(
        value=energy.value / tokens,
        provenance=MetricProvenance.DERIVED,
        unit=ENERGY_PER_OUTPUT_TOKEN_UNIT,
    )
