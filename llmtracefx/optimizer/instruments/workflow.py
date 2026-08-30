"""End-to-end workflows composing capability, record and export.

Kept out of the CLI module so each workflow is testable without
argparse, and so the CLI handlers stay thin.

The three workflows mirror the three CLI subcommands:

:func:`plan_trace`
    Resolve and validate everything, execute nothing. Reports the exact
    argv, the exact output paths, and any unmet prerequisite.
:func:`record_trace`
    Everything :func:`plan_trace` does, then actually record.
:func:`import_trace`
    Read an existing ``.trace`` bundle, export its table of contents and
    optionally one table, and build canonical evidence from it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..collectors._shared import atomic_write_text
from ..schema import InstrumentsEvidence
from .capability import (
    METAL_SYSTEM_TRACE_TEMPLATE,
    XctraceCapability,
    XctraceCapabilityReport,
    detect_xctrace_capability,
)
from .commands import (
    ExportPlan,
    InstrumentsCommandError,
    LaunchTarget,
    RecordPlan,
)
from .evidence import (
    TraceEvidenceInputs,
    build_instruments_evidence,
    unsupported_evidence,
)
from .export import (
    SUPPORTED_TABLE_SCHEMAS,
    ExportedTable,
    InstrumentsExportError,
    TraceTableOfContents,
    parse_exported_table,
    parse_table_of_contents,
    read_export_text,
)
from .process import CommandRunner, ProcessLauncher
from .recorder import (
    InstrumentsRecordError,
    RecordResult,
    RecordStatus,
    check_output_collision,
    run_record,
)

#: Default table exported after a recording. The only Metal schema this
#: project has a strict summarizer for.
DEFAULT_TABLE_SCHEMA = "metal-gpu-intervals"


@dataclass(frozen=True)
class TracePlan:
    """A validated, unexecuted recording plan plus its prerequisites."""

    capability: XctraceCapabilityReport
    record_plan: RecordPlan | None
    prerequisites: tuple[str, ...]
    """Unmet conditions. Empty means the plan is ready to run."""
    output_paths: dict[str, str]
    error: str | None = None

    @property
    def ready(self) -> bool:
        return not self.prerequisites and self.record_plan is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "capability": self.capability.capability.value,
            "capability_reason": self.capability.reason,
            "remediation": self.capability.remediation,
            "prerequisites": list(self.prerequisites),
            "argv": (
                None
                if self.record_plan is None
                else list(self.record_plan.to_redacted_argv())
            ),
            "host_timeout_seconds": (
                None if self.record_plan is None else self.record_plan.timeout_seconds
            ),
            "output_paths": dict(self.output_paths),
            "error": self.error,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


def _artifact_paths(output_dir: Path, trace: Path) -> dict[str, str]:
    return {
        "trace_bundle": str(trace),
        "record_metadata": str(output_dir / "xctrace_record.json"),
        "record_stdout": str(output_dir / "xctrace_record_stdout.txt"),
        "record_stderr": str(output_dir / "xctrace_record_stderr.txt"),
        "capability_report": str(output_dir / "capability_report.json"),
        "toc_xml": str(output_dir / "trace_toc.xml"),
        "table_xml": str(output_dir / "trace_table.xml"),
        "evidence": str(output_dir / "instruments_evidence.json"),
    }


def plan_trace(
    *,
    runner: CommandRunner,
    command: tuple[str, ...],
    output_trace: Path,
    output_dir: Path,
    template: str = METAL_SYSTEM_TRACE_TEMPLATE,
    time_limit: str = "60s",
) -> TracePlan:
    """Validate a recording without executing anything.

    Never spawns the target program and never creates a trace. The only
    subprocesses are the capability probes, which are read-only metadata
    queries against xctrace itself.
    """
    capability = detect_xctrace_capability(runner=runner, template=template)
    prerequisites: list[str] = []
    if not capability.supported:
        detail = capability.reason
        if capability.remediation:
            detail += f" Fix: {capability.remediation}"
        prerequisites.append(detail)

    paths = _artifact_paths(output_dir, output_trace)

    # A collision is a prerequisite, not a hard error: the plan should
    # still show what it would have run.
    try:
        check_output_collision(output_trace)
    except InstrumentsRecordError as exc:
        prerequisites.append(str(exc))

    record_plan: RecordPlan | None = None
    error: str | None = None
    try:
        record_plan = RecordPlan(
            xctrace_path=capability.xctrace_path or "xctrace",
            template=template,
            output_trace=output_trace,
            target=LaunchTarget(argv=command),
            time_limit=time_limit,
        )
    except InstrumentsCommandError as exc:
        error = str(exc)
        prerequisites.append(f"invalid recording request: {exc}")

    return TracePlan(
        capability=capability,
        record_plan=record_plan,
        prerequisites=tuple(prerequisites),
        output_paths=paths,
        error=error,
    )


@dataclass(frozen=True)
class TraceCollection:
    """Result of recording and importing one trace."""

    capability: XctraceCapabilityReport
    record: RecordResult | None
    evidence: InstrumentsEvidence
    toc: TraceTableOfContents | None = None
    table: ExportedTable | None = None
    message: str = ""

    @property
    def succeeded(self) -> bool:
        return self.record is not None and self.record.succeeded


def record_trace(
    *,
    runner: CommandRunner,
    launcher: ProcessLauncher,
    command: tuple[str, ...],
    output_trace: Path,
    output_dir: Path,
    template: str = METAL_SYSTEM_TRACE_TEMPLATE,
    time_limit: str = "60s",
    table_schema: str | None = DEFAULT_TABLE_SCHEMA,
) -> TraceCollection:
    """Record a trace, then import it into canonical evidence.

    When capability detection says no, this records an explicit
    unsupported evidence artifact rather than attempting a recording
    that is known to fail.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    capability = detect_xctrace_capability(runner=runner, template=template)
    capability.write_json(output_dir / "capability_report.json")

    if not capability.supported:
        evidence = unsupported_evidence(capability, template=template)
        _write_evidence(output_dir, evidence)
        return TraceCollection(
            capability=capability,
            record=None,
            evidence=evidence,
            message=(
                f"xctrace is not usable here ({capability.capability.value}): "
                f"{capability.reason}"
            ),
        )

    plan = RecordPlan(
        xctrace_path=capability.xctrace_path or "xctrace",
        template=template,
        output_trace=output_trace,
        target=LaunchTarget(argv=command),
        time_limit=time_limit,
    )
    result = run_record(plan, launcher=launcher, artifacts_dir=output_dir)

    if result.status is not RecordStatus.COMPLETED:
        evidence = unsupported_evidence(capability, template=template)
        _write_evidence(output_dir, evidence)
        return TraceCollection(
            capability=capability,
            record=result,
            evidence=evidence,
            message=result.message,
        )

    collection = import_trace(
        runner=runner,
        capability=capability,
        trace_path=result.trace_path,
        output_dir=output_dir,
        template=template,
        table_schema=table_schema,
    )
    return TraceCollection(
        capability=capability,
        record=result,
        evidence=collection.evidence,
        toc=collection.toc,
        table=collection.table,
        message=collection.message or result.message,
    )


def _write_evidence(output_dir: Path, evidence: InstrumentsEvidence) -> None:
    atomic_write_text(
        output_dir / "instruments_evidence.json",
        json.dumps(evidence.to_dict(), indent=2) + "\n",
    )


def import_trace(
    *,
    runner: CommandRunner,
    trace_path: Path,
    output_dir: Path,
    capability: XctraceCapabilityReport | None = None,
    template: str = METAL_SYSTEM_TRACE_TEMPLATE,
    table_schema: str | None = DEFAULT_TABLE_SCHEMA,
) -> TraceCollection:
    """Export and parse an existing ``.trace`` bundle into evidence."""
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_capability = (
        detect_xctrace_capability(runner=runner, template=template)
        if capability is None
        else capability
    )

    if not resolved_capability.supported:
        evidence = unsupported_evidence(resolved_capability, template=template)
        _write_evidence(output_dir, evidence)
        return TraceCollection(
            capability=resolved_capability,
            record=None,
            evidence=evidence,
            message=resolved_capability.reason,
        )

    if not trace_path.exists():
        raise InstrumentsExportError(f"trace bundle does not exist: {trace_path}")

    xctrace = resolved_capability.xctrace_path or "xctrace"
    toc_path = output_dir / "trace_toc.xml"
    toc_plan = ExportPlan(
        xctrace_path=xctrace,
        input_trace=trace_path,
        output_path=toc_path,
        toc=True,
    )
    toc_result = runner.run(
        toc_plan.to_argv(), timeout_seconds=toc_plan.timeout_seconds
    )
    if toc_result.returncode != 0 or toc_result.timed_out:
        raise InstrumentsExportError(
            f"`xctrace export --toc` failed (exit {toc_result.returncode}) for "
            f"{trace_path.name}"
        )

    toc = parse_table_of_contents(read_export_text(toc_path))
    run = toc.runs[0]

    table: ExportedTable | None = None
    messages: list[str] = []
    if table_schema is not None:
        if table_schema not in run.schemas:
            messages.append(
                f"table {table_schema!r} is not present in this trace's "
                f"table of contents ({len(run.schemas)} schemas available), "
                "so no metric was derived"
            )
        else:
            table_path = output_dir / "trace_table.xml"
            table_plan = ExportPlan(
                xctrace_path=xctrace,
                input_trace=trace_path,
                output_path=table_path,
                schema_name=table_schema,
                run_number=run.number,
            )
            table_result = runner.run(
                table_plan.to_argv(), timeout_seconds=table_plan.timeout_seconds
            )
            if table_result.returncode != 0 or table_result.timed_out:
                raise InstrumentsExportError(
                    f"`xctrace export --xpath` failed (exit "
                    f"{table_result.returncode}) for schema {table_schema!r}"
                )
            table = parse_exported_table(
                read_export_text(table_path), expected_schema=table_schema
            )
            if table_schema not in SUPPORTED_TABLE_SCHEMAS:
                messages.append(
                    f"table {table_schema!r} was exported and parsed but has "
                    "no strict summarizer, so no metric was derived"
                )

    evidence = build_instruments_evidence(
        TraceEvidenceInputs(
            capability=resolved_capability,
            trace_bundle_name=trace_path.name,
            template=run.template_name or template,
            available_schemas=run.schemas,
            table=table,
            target_pid=run.target_pid,
        )
    )
    _write_evidence(output_dir, evidence)
    atomic_write_text(
        output_dir / "trace_toc.json", json.dumps(toc.to_dict(), indent=2) + "\n"
    )
    return TraceCollection(
        capability=resolved_capability,
        record=None,
        evidence=evidence,
        toc=toc,
        table=table,
        message=" ".join(messages),
    )


def capability_exit_code(capability: XctraceCapability) -> int:
    """0 when supported, 3 when a known cause blocks it."""
    return 0 if capability.is_supported else 3
