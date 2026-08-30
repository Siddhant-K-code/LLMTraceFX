"""Apple Instruments (``xctrace``) capability, recording and export.

Public surface for the Metal/Instruments evidence path. See the module
docstrings for the verified behavior each layer relies on:

* :mod:`.capability` distinguishes every reason a trace cannot be taken.
* :mod:`.commands` builds shell-free, validated argv.
* :mod:`.recorder` runs a recording under a deadline and preserves what
  it produced.
* :mod:`.export` parses ``xctrace export`` output strictly.
* :mod:`.evidence` decides what may be claimed as a metric.
"""

from __future__ import annotations

from .capability import (
    CAPABILITY_SCHEMA_VERSION,
    METAL_SYSTEM_TRACE_TEMPLATE,
    InstrumentsCapabilityError,
    XctraceCapability,
    XctraceCapabilityReport,
    classify_xctrace_failure,
    default_xctrace_path,
    detect_xctrace_capability,
    parse_templates,
    parse_version,
)
from .commands import (
    AttachTarget,
    EnvironmentAssignment,
    ExportPlan,
    InstrumentsCommandError,
    LaunchTarget,
    RecordPlan,
    RecordTarget,
    build_list_templates_argv,
    build_version_argv,
    duration_to_seconds,
    redact_argv,
    table_xpath,
    validate_schema_name,
    validate_time_limit,
    validate_window,
)
from .evidence import (
    TraceEvidenceInputs,
    build_instruments_evidence,
    unsupported_evidence,
)
from .export import (
    FORBIDDEN_METRIC_NAMES,
    MAX_EXPORT_BYTES,
    SUPPORTED_TABLE_SCHEMAS,
    ExportedRow,
    ExportedTable,
    InstrumentsExportError,
    MetalGpuIntervalSummary,
    ProcessGpuIntervals,
    TableColumn,
    TraceRun,
    TraceTableOfContents,
    parse_exported_table,
    parse_table_of_contents,
    read_export_text,
    summarize_metal_gpu_intervals,
)
from .process import (
    CommandResult,
    CommandRunner,
    InstrumentsProcessError,
    ManagedProcess,
    ProcessLauncher,
    SubprocessCommandRunner,
    SubprocessProcessLauncher,
)
from .recorder import (
    InstrumentsRecordError,
    RecordResult,
    RecordStatus,
    check_output_collision,
    run_record,
)

__all__ = [
    "AttachTarget",
    "CAPABILITY_SCHEMA_VERSION",
    "CommandResult",
    "CommandRunner",
    "EnvironmentAssignment",
    "ExportPlan",
    "ExportedRow",
    "ExportedTable",
    "FORBIDDEN_METRIC_NAMES",
    "InstrumentsCapabilityError",
    "InstrumentsCommandError",
    "InstrumentsExportError",
    "InstrumentsProcessError",
    "InstrumentsRecordError",
    "LaunchTarget",
    "MAX_EXPORT_BYTES",
    "METAL_SYSTEM_TRACE_TEMPLATE",
    "ManagedProcess",
    "MetalGpuIntervalSummary",
    "ProcessGpuIntervals",
    "ProcessLauncher",
    "RecordPlan",
    "RecordResult",
    "RecordStatus",
    "RecordTarget",
    "SUPPORTED_TABLE_SCHEMAS",
    "SubprocessCommandRunner",
    "SubprocessProcessLauncher",
    "TableColumn",
    "TraceEvidenceInputs",
    "TraceRun",
    "TraceTableOfContents",
    "XctraceCapability",
    "XctraceCapabilityReport",
    "build_instruments_evidence",
    "build_list_templates_argv",
    "build_version_argv",
    "check_output_collision",
    "classify_xctrace_failure",
    "default_xctrace_path",
    "detect_xctrace_capability",
    "duration_to_seconds",
    "parse_exported_table",
    "parse_table_of_contents",
    "parse_templates",
    "parse_version",
    "read_export_text",
    "redact_argv",
    "run_record",
    "summarize_metal_gpu_intervals",
    "table_xpath",
    "unsupported_evidence",
    "validate_schema_name",
    "validate_time_limit",
    "validate_window",
]
