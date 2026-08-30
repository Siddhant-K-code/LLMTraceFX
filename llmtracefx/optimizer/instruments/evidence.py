"""Turn parsed trace artifacts into canonical ``InstrumentsEvidence``.

This is the only place that decides what counts as a claimable metric.
The rule it enforces: a number is emitted only when a strict parser
produced it from an exported table, and it is named for exactly what it
measures.

What is deliberately never produced here, despite Metal System Trace
advertising tables whose names gesture at them: GPU utilization, GPU
busy percentage, kernel time, memory bandwidth, occupancy, GPU power and
GPU memory footprint. Each would need modelling assumptions this project
has not validated against ground truth, so each stays absent instead of
being approximated.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..schema import InstrumentsEvidence, Measurement, MetricProvenance
from .capability import XctraceCapabilityReport
from .export import (
    SUPPORTED_TABLE_SCHEMAS,
    ExportedTable,
    InstrumentsExportError,
    MetalGpuIntervalSummary,
    ProcessGpuIntervals,
    summarize_metal_gpu_intervals,
)

#: Nanoseconds per millisecond. Trace durations are integer nanoseconds.
_NS_PER_MS = 1_000_000.0


@dataclass(frozen=True)
class TraceEvidenceInputs:
    """Everything needed to build evidence for one recorded trace."""

    capability: XctraceCapabilityReport
    trace_bundle_name: str
    template: str
    available_schemas: tuple[str, ...]
    table: ExportedTable | None = None
    """A parsed table, when one was exported and understood."""
    target_pid: int | None = None
    """Which process the metrics should describe.

    Metal System Trace captures GPU work system wide. Without a target
    pid there is no honest way to attribute intervals to the workload
    under test, so no scalar metric is emitted."""


def _partition_schemas(
    available: tuple[str, ...], parsed: tuple[str, ...]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    parsed_set = set(parsed)
    unsupported = tuple(schema for schema in available if schema not in parsed_set)
    return parsed, unsupported


def metrics_for_process(
    summary: MetalGpuIntervalSummary, entry: ProcessGpuIntervals
) -> dict[str, Measurement]:
    """Build the metric set for one process's GPU intervals.

    Names are chosen so that no reader can mistake them for utilization
    or throughput. ``duration_sum`` in particular sums intervals that
    Metal runs concurrently across channels, so it exceeds wall-clock
    GPU busy time and is never presented as a fraction of anything.
    """
    return {
        "metal_gpu_interval_count": Measurement(
            value=float(entry.interval_count),
            provenance=MetricProvenance.MEASURED_NATIVE,
            unit="intervals",
        ),
        "metal_gpu_interval_duration_sum": Measurement(
            value=entry.duration_sum_ns / _NS_PER_MS,
            provenance=MetricProvenance.MEASURED_NATIVE,
            unit="ms",
        ),
        "metal_gpu_interval_wall_span": Measurement(
            value=entry.wall_span_ns / _NS_PER_MS,
            provenance=MetricProvenance.MEASURED_NATIVE,
            unit="ms",
        ),
        "metal_gpu_interval_count_all_processes": Measurement(
            value=float(summary.total_interval_count),
            provenance=MetricProvenance.MEASURED_NATIVE,
            unit="intervals",
        ),
    }


def build_instruments_evidence(inputs: TraceEvidenceInputs) -> InstrumentsEvidence:
    """Assemble evidence, leaving unmeasurable quantities absent."""
    capability = inputs.capability
    parsed_schemas: tuple[str, ...] = ()
    metrics: dict[str, Measurement] = {}
    notes: list[str] = []

    table = inputs.table
    if table is not None:
        if table.schema_name not in SUPPORTED_TABLE_SCHEMAS:
            notes.append(
                f"table {table.schema_name!r} was exported but this project "
                "has no strict summarizer for it, so no metric was derived"
            )
        else:
            parsed_schemas = (table.schema_name,)
            summary = summarize_metal_gpu_intervals(table)
            if inputs.target_pid is None:
                notes.append(
                    "no target pid was supplied, so the "
                    f"{summary.total_interval_count} parsed GPU intervals "
                    "were left unattributed. Metal System Trace records "
                    "every process on the system, so a system-wide total "
                    "would misattribute other processes' GPU work."
                )
            else:
                entry = None
                try:
                    entry = summary.for_process(inputs.target_pid)
                except InstrumentsExportError as exc:
                    # An ambiguous pid is a refusal, not a crash: the
                    # trace is still valid evidence, there is just no
                    # honest way to attribute a scalar to one process.
                    notes.append(str(exc))
                else:
                    if entry is None:
                        notes.append(
                            f"pid {inputs.target_pid} contributed no GPU "
                            "intervals to this trace, so no metric was "
                            "derived"
                        )
                if entry is not None:
                    metrics = metrics_for_process(summary, entry)
                    notes.append(
                        f"metrics describe pid {inputs.target_pid} only "
                        f"({entry.interval_count} of "
                        f"{summary.total_interval_count} intervals in the "
                        "trace). duration_sum adds concurrent Metal "
                        "channels together and is not GPU busy time or "
                        "utilization."
                    )

    parsed, unsupported = _partition_schemas(inputs.available_schemas, parsed_schemas)
    return InstrumentsEvidence(
        tool="xctrace",
        tool_version=capability.xctrace_version,
        capability=capability.capability.value,
        template=inputs.template,
        trace_bundle_name=inputs.trace_bundle_name,
        available_schemas=inputs.available_schemas,
        parsed_schemas=parsed,
        unsupported_schemas=unsupported,
        metrics=metrics,
        notes=" ".join(notes) if notes else None,
    )


def failed_recording_evidence(
    capability: XctraceCapabilityReport, *, template: str, reason: str
) -> InstrumentsEvidence:
    """Evidence recorded when a supported toolchain still failed to record.

    Distinct from :func:`unsupported_evidence`: capability succeeded, so
    reusing the capability reason here would persist an affirmatively
    misleading note ("xctrace provides the template") on an artifact
    that represents a failed recording. The recorder's own message is
    carried instead.
    """
    return InstrumentsEvidence(
        tool="xctrace",
        tool_version=capability.xctrace_version,
        capability=capability.capability.value,
        template=template,
        trace_bundle_name=None,
        available_schemas=(),
        parsed_schemas=(),
        unsupported_schemas=(),
        metrics={},
        notes=f"no trace was produced: {reason}",
    )


def unsupported_evidence(
    capability: XctraceCapabilityReport, *, template: str
) -> InstrumentsEvidence:
    """Evidence recorded when no trace could be taken at all.

    Carries the capability state and its reason so a downstream reader
    sees why the measurement is missing instead of finding a silent gap.
    """
    return InstrumentsEvidence(
        tool="xctrace",
        tool_version=capability.xctrace_version,
        capability=capability.capability.value,
        template=template,
        trace_bundle_name=None,
        available_schemas=(),
        parsed_schemas=(),
        unsupported_schemas=(),
        metrics={},
        notes=capability.reason,
    )
