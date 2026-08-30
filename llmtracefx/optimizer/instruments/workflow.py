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

import contextlib
import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..collectors._shared import atomic_write_text
from ..schema import InstrumentsEvidence, utc_now_iso
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
    failed_export_evidence,
    failed_recording_evidence,
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
    sanitize_table_of_contents,
)
from .process import CommandRunner, InstrumentsProcessError, ProcessLauncher
from .recorder import (
    InstrumentsRecordError,
    RecordResult,
    RecordStatus,
    check_output_collision,
    reserve_output_dir,
    reserve_trace_path,
    run_record,
)

#: Default table exported after a recording. The only Metal schema this
#: project has a strict summarizer for.
DEFAULT_TABLE_SCHEMA = "metal-gpu-intervals"

#: Seconds allowed after each stop signal before escalating. Mirrors
#: RecordPlan's own default; exposed so a caller can tighten it.
DEFAULT_STOP_GRACE_SECONDS = 30.0

#: Scratch directory used while exporting, removed on every exit.
STAGING_DIR_NAME = ".import-staging"


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
    stop_grace_seconds: float = DEFAULT_STOP_GRACE_SECONDS,
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
            stop_grace_seconds=stop_grace_seconds,
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
    wrote_artifacts: bool = True
    """False when the run was refused before touching the filesystem."""
    export_failed: bool = False
    """True when the trace recorded but could not be exported.

    Kept separate from ``record.status``, which correctly says the
    recording completed. The run as a whole did not succeed, because it
    produced no measurement, and an exit code that said otherwise would
    let a CI job record a green run with zero GPU metrics."""

    @property
    def succeeded(self) -> bool:
        return (
            self.record is not None and self.record.succeeded and not self.export_failed
        )


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
    stop_grace_seconds: float = DEFAULT_STOP_GRACE_SECONDS,
) -> TraceCollection:
    """Record a trace, then import it into canonical evidence.

    When capability detection says no, this records an explicit
    unsupported evidence artifact rather than attempting a recording
    that is known to fail.
    """
    # The trace path is reserved before anything is written, and the
    # reservation is held for the whole run. Doing this inside
    # run_record (after the capability report had already landed) meant
    # a rerun into a directory that still held a trace refused the
    # recording but had by then overwritten capability_report.json and
    # instruments_evidence.json, leaving one run's trace beside another
    # run's metadata. A bare check would also still race a concurrent
    # writer, so the claim is atomic and outlives the check.
    try:
        # Trace first, then the artifact directory. A trace collision is
        # the common refusal and must not create the artifact directory
        # on its way to being refused.
        with (
            reserve_trace_path(output_trace) as reserved_trace,
            reserve_output_dir(output_dir),
        ):
            return _record_reserved(
                runner=runner,
                launcher=launcher,
                command=command,
                reserved_trace=reserved_trace,
                output_dir=output_dir,
                template=template,
                time_limit=time_limit,
                table_schema=table_schema,
                stop_grace_seconds=stop_grace_seconds,
            )
    except InstrumentsRecordError as exc:
        capability = detect_xctrace_capability(runner=runner, template=template)
        return TraceCollection(
            capability=capability,
            record=RecordResult(
                status=RecordStatus.REFUSED,
                argv=(),
                trace_path=output_trace,
                message=str(exc),
                started_at=utc_now_iso(),
                ended_at=utc_now_iso(),
            ),
            evidence=failed_recording_evidence(
                capability, template=template, reason=str(exc)
            ),
            message=str(exc),
            wrote_artifacts=False,
        )


def _record_reserved(
    *,
    runner: CommandRunner,
    launcher: ProcessLauncher,
    command: tuple[str, ...],
    reserved_trace: Path,
    output_dir: Path,
    template: str,
    time_limit: str,
    table_schema: str | None,
    stop_grace_seconds: float = DEFAULT_STOP_GRACE_SECONDS,
) -> TraceCollection:
    """Record into a trace path whose reservation is already held."""
    output_trace = reserved_trace
    capability = detect_xctrace_capability(runner=runner, template=template)

    # Everything that can be rejected is rejected before the first byte
    # is written. Building the plan validates the trace suffix, the time
    # limit and the target command, and it used to happen after
    # capability_report.json had already landed, so an invalid request
    # left an artifact behind describing a run that never started.
    plan: RecordPlan | None = None
    if capability.supported:
        plan = RecordPlan(
            xctrace_path=capability.xctrace_path or "xctrace",
            template=template,
            output_trace=output_trace,
            target=LaunchTarget(argv=command),
            time_limit=time_limit,
            stop_grace_seconds=stop_grace_seconds,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    capability.write_json(output_dir / "capability_report.json")

    if not capability.supported or plan is None:
        evidence = unsupported_evidence(capability, template=template)
        _clear_stale_exports(output_dir)
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

    result = run_record(
        plan,
        launcher=launcher,
        artifacts_dir=output_dir,
        reserved_trace=reserved_trace,
    )

    if result.status is not RecordStatus.COMPLETED:
        evidence = failed_recording_evidence(
            capability, template=template, reason=result.message
        )
        _clear_stale_exports(output_dir)
        _write_evidence(output_dir, evidence)
        return TraceCollection(
            capability=capability,
            record=result,
            evidence=evidence,
            message=result.message,
        )

    try:
        collection = import_trace(
            runner=runner,
            capability=capability,
            trace_path=result.trace_path,
            output_dir=output_dir,
            template=template,
            table_schema=table_schema,
        )
    except (
        InstrumentsExportError,
        InstrumentsCommandError,
        InstrumentsProcessError,
        ArithmeticError,
        OSError,
    ) as exc:
        # The recording succeeded, so this run has already written its
        # capability report and record metadata into output_dir. Letting
        # any of these propagate would leave those beside a previous
        # run's evidence and GPU metrics, which is exactly the mixing
        # the staging directory prevents on the export side. Writing
        # this run's evidence replaces them.
        #
        # All three are reachable: the export tool can fail (export
        # error), become unexecutable between the capability probe and
        # the export (process error), or the staging and evidence writes
        # can fail (OSError).
        reason = f"the trace was recorded but could not be exported: {exc}"
        evidence = failed_export_evidence(
            capability,
            template=template,
            trace_bundle_name=result.trace_path.name,
            reason=str(exc),
        )
        _clear_stale_exports(output_dir)
        _write_evidence(output_dir, evidence)
        return TraceCollection(
            capability=capability,
            record=result,
            evidence=evidence,
            message=reason,
            export_failed=True,
        )
    return TraceCollection(
        capability=capability,
        record=result,
        evidence=collection.evidence,
        toc=collection.toc,
        table=collection.table,
        message=collection.message or result.message,
    )


@contextmanager
def _staging_dir(output_dir: Path) -> Iterator[Path]:
    """A scratch directory that never outlives this call.

    Without the ``finally``, a failed export left the partial staging
    set sitting in the user's output directory until the next run
    happened to clear it.
    """
    staging = output_dir / STAGING_DIR_NAME
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        yield staging
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _promote_staged(staging: Path, output_dir: Path) -> None:
    """Move a completed staging set into place.

    Every file is written before any is promoted, so a failed export
    cannot leave one run's table of contents beside another run's
    evidence. Each individual move is an atomic replace within the same
    filesystem. The loop itself is not atomic as a whole: an error
    partway leaves some files promoted. That is deliberate and is still
    strictly better than the alternative, because the promoted files all
    belong to this run, whereas exporting in place mixed two runs.
    """
    for staged in sorted(
        staging.iterdir(),
        # instruments_evidence.json is the artifact a reader trusts, so
        # it is promoted last. If a tear ever did happen mid-loop, the
        # evidence left behind would be the stale one, rather than a new
        # one pointing at a table that had not landed yet.
        key=lambda path: (path.name == "instruments_evidence.json", path.name),
    ):
        if staged.is_dir():
            raise InstrumentsExportError(
                f"unexpected directory in the export staging set: " f"{staged.name}"
            )
        os.replace(staged, output_dir / staged.name)


#: The set _promote_staged writes. A run that fails before promoting
#: must clear these, or its metadata sits beside a previous run's.
EXPORT_ARTIFACT_NAMES: tuple[str, ...] = (
    "trace_toc.xml",
    "trace_toc.json",
    "trace_table.xml",
)


def _clear_stale_exports(output_dir: Path) -> None:
    """Remove exported artifacts left by an earlier run."""
    for name in EXPORT_ARTIFACT_NAMES:
        with contextlib.suppress(FileNotFoundError, IsADirectoryError):
            (output_dir / name).unlink()


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
        _clear_stale_exports(output_dir)
        _write_evidence(output_dir, evidence)
        return TraceCollection(
            capability=resolved_capability,
            record=None,
            evidence=evidence,
            message=resolved_capability.reason,
        )

    if not trace_path.exists():
        raise InstrumentsExportError(f"trace bundle does not exist: {trace_path}")

    # Exports land in a staging directory and are promoted together
    # only once evidence has been built. xctrace writes trace_toc.xml
    # and trace_table.xml itself, mid-flight, so exporting straight into
    # output_dir meant a failure between the two left this run's TOC
    # sitting beside a previous run's evidence.
    xctrace = resolved_capability.xctrace_path or "xctrace"
    with _staging_dir(output_dir) as staging:
        return _export_and_build(
            runner=runner,
            xctrace=xctrace,
            trace_path=trace_path,
            output_dir=output_dir,
            staging=staging,
            resolved_capability=resolved_capability,
            template=template,
            table_schema=table_schema,
        )


def _export_and_build(
    *,
    runner: CommandRunner,
    xctrace: str,
    trace_path: Path,
    output_dir: Path,
    staging: Path,
    resolved_capability: XctraceCapabilityReport,
    template: str,
    table_schema: str | None,
) -> TraceCollection:
    """Export into ``staging`` and promote once evidence is built."""
    toc_path = staging / "trace_toc.xml"
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

    # xctrace wrote the raw TOC into the staging directory. Read it,
    # then immediately replace it with a sanitized copy: the raw file
    # names the machine's owner, its hardware UUID and the profiled
    # command's full argument list, and only the sanitized version is
    # ever promoted into the output directory.
    raw_toc = read_export_text(toc_path)
    toc = parse_table_of_contents(raw_toc)
    atomic_write_text(toc_path, sanitize_table_of_contents(raw_toc))
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
            table_path = staging / "trace_table.xml"
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
    atomic_write_text(
        staging / "instruments_evidence.json",
        json.dumps(evidence.to_dict(), indent=2) + "\n",
    )
    atomic_write_text(
        staging / "trace_toc.json", json.dumps(toc.to_dict(), indent=2) + "\n"
    )
    _promote_staged(staging, output_dir)
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
