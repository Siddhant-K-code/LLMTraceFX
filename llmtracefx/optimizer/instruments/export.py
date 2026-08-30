"""Strict parsers for ``xctrace export`` output.

Two formats are handled, both verified against real output from
``xctrace version 16.0 (17F113)`` on macOS 26.6.2 (Apple M5 Pro):

Table of contents (``xctrace export --toc``)
    ``<trace-toc>`` containing ``<run number="N">`` elements. Each run
    carries ``<info><summary>`` metadata and a ``<data>`` list of
    ``<table schema="..."/>`` entries.

Table data (``xctrace export --xpath ...``)
    ``<trace-query-result>`` containing ``<node>`` elements. Each node
    holds one ``<schema name="...">`` with ordered ``<col>`` definitions,
    followed by ``<row>`` elements.

Two properties of the real format drive the whole design:

1. **Rows are positional.** A row's Nth direct child corresponds to the
   schema's Nth ``<col>``, and the child's tag equals that column's
   ``<engineering-type>``. Verified across every row of a real
   ``metal-gpu-intervals`` export (2460 rows, 18 of 18 children each,
   tags matching the declared engineering types in order). The parser
   enforces both invariants and fails loudly rather than mapping values
   onto the wrong columns.

2. **Values are reference-deduplicated.** A repeated value is emitted
   once with an ``id`` and afterwards referenced as ``<tag ref="id"/>``.
   The same real export contained 2460 rows but 43818 ``ref``
   attributes, so a parser that ignored references would silently drop
   the overwhelming majority of every row's content.

Privacy: the TOC also contains the device's display name (which
routinely embeds a person's name), its hardware UUID, and the launched
process's full argument list. None of those are read by
:func:`parse_table_of_contents`, so they cannot reach an experiment
record or a log line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

#: Refuse inputs above this size rather than loading them into memory.
#: A real 0.87 second Metal System Trace exported a 2.7 MB
#: ``metal-gpu-intervals`` table, so this leaves ample headroom while
#: still bounding a runaway export.
MAX_EXPORT_BYTES = 256 * 1024 * 1024

#: Metal System Trace table schemas this project knows how to summarize.
#: Everything else found in a trace is reported as present but
#: unsupported rather than guessed at.
SUPPORTED_TABLE_SCHEMAS: tuple[str, ...] = ("metal-gpu-intervals",)

#: Metric names this project must never emit. Metal System Trace does
#: expose tables whose names gesture at these ideas, but none of them
#: can be derived from an exported table without modelling assumptions
#: this project has not validated. Asserted by the test suite.
FORBIDDEN_METRIC_NAMES: tuple[str, ...] = (
    "gpu_utilization",
    "gpu_busy_percent",
    "gpu_kernel_time",
    "memory_bandwidth",
    "bandwidth_gb_s",
    "occupancy",
    "gpu_power",
    "gpu_energy",
    "gpu_memory_bytes",
)

_DOCTYPE_RE = re.compile(r"<!\s*(DOCTYPE|ENTITY)", re.IGNORECASE)


class InstrumentsExportError(ValueError):
    """Raised when exported XML is missing, malformed or unsupported."""


def _parse_xml(text: str, *, source: str) -> ElementTree.Element:
    """Parse XML with entity-expansion inputs refused up front.

    :mod:`xml.etree.ElementTree` does not resolve external entities, but
    it will happily expand internal ones, which is the "billion laughs"
    denial of service. Real xctrace output contains no doctype at all,
    so refusing any input that declares one costs nothing and removes
    the class of attack entirely.
    """
    if _DOCTYPE_RE.search(text):
        raise InstrumentsExportError(
            f"{source} declares a DOCTYPE or ENTITY. Real xctrace output "
            "never does, and entity expansion is refused."
        )
    try:
        return ElementTree.fromstring(text)
    except ElementTree.ParseError as exc:
        raise InstrumentsExportError(f"{source} is not valid XML: {exc}") from exc


def read_export_text(path: str | Path) -> str:
    """Read an export file, refusing absent or oversized inputs."""
    target = Path(path)
    if not target.exists():
        raise InstrumentsExportError(f"export file does not exist: {target}")
    if not target.is_file():
        raise InstrumentsExportError(f"export path is not a file: {target}")
    size = target.stat().st_size
    if size > MAX_EXPORT_BYTES:
        raise InstrumentsExportError(
            f"export file is {size} bytes, above the {MAX_EXPORT_BYTES} byte "
            "limit. Re-record with a shorter --time-limit."
        )
    try:
        return target.read_text(encoding="utf-8")
    except UnicodeError as exc:
        raise InstrumentsExportError(
            f"export file is not valid UTF-8: {target}: {exc}"
        ) from exc


# --- Table of contents ------------------------------------------------


@dataclass(frozen=True)
class TraceRun:
    """One recorded run inside a ``.trace`` bundle.

    Only allowlisted fields are captured. ``target_pid`` and
    ``target_process_name`` describe the process this project asked
    xctrace to launch, and are read because metric attribution is
    impossible without them. The device display name (which routinely
    embeds a person's name), the device hardware UUID, and the target
    process's full argument list also live in the TOC and are
    deliberately not read.
    """

    number: int
    template_name: str | None = None
    instruments_version: str | None = None
    duration_seconds: float | None = None
    end_reason: str | None = None
    target_pid: int | None = None
    target_process_name: str | None = None
    schemas: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "template_name": self.template_name,
            "instruments_version": self.instruments_version,
            "duration_seconds": self.duration_seconds,
            "end_reason": self.end_reason,
            "target_pid": self.target_pid,
            "target_process_name": self.target_process_name,
            "schemas": list(self.schemas),
        }


@dataclass(frozen=True)
class TraceTableOfContents:
    """Parsed ``xctrace export --toc`` output."""

    runs: tuple[TraceRun, ...]

    @property
    def schema_names(self) -> tuple[str, ...]:
        """Every distinct table schema across all runs, sorted."""
        names: set[str] = set()
        for run in self.runs:
            names.update(run.schemas)
        return tuple(sorted(names))

    def run_by_number(self, number: int) -> TraceRun | None:
        for run in self.runs:
            if run.number == number:
                return run
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"runs": [run.to_dict() for run in self.runs]}


def _optional_float(value: str | None, *, field: str) -> float | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError as exc:
        raise InstrumentsExportError(
            f"trace TOC {field} is not a number: {value!r}"
        ) from exc


def _text_or_none(parent: ElementTree.Element | None, path: str) -> str | None:
    if parent is None:
        return None
    found = parent.find(path)
    if found is None or found.text is None:
        return None
    stripped = found.text.strip()
    return stripped or None


def parse_table_of_contents(xml_text: str) -> TraceTableOfContents:
    """Parse ``xctrace export --toc`` XML into a schema inventory."""
    root = _parse_xml(xml_text, source="trace table of contents")
    if root.tag != "trace-toc":
        raise InstrumentsExportError(
            f"expected a <trace-toc> document, got <{root.tag}>. This does "
            "not look like `xctrace export --toc` output."
        )

    runs: list[TraceRun] = []
    for run_element in root.findall("run"):
        raw_number = run_element.get("number")
        if raw_number is None:
            raise InstrumentsExportError("trace TOC <run> is missing @number")
        try:
            number = int(raw_number)
        except ValueError as exc:
            raise InstrumentsExportError(
                f"trace TOC <run> @number is not an integer: {raw_number!r}"
            ) from exc

        summary = run_element.find("info/summary")
        target_process = run_element.find("info/target/process")
        target_pid: int | None = None
        target_process_name: str | None = None
        if target_process is not None:
            raw_pid = target_process.get("pid")
            if raw_pid is not None:
                try:
                    target_pid = int(raw_pid)
                except ValueError as exc:
                    raise InstrumentsExportError(
                        f"trace TOC target process @pid is not an integer: "
                        f"{raw_pid!r}"
                    ) from exc
            target_process_name = target_process.get("name")

        schemas = tuple(
            dict.fromkeys(
                schema
                for schema in (
                    table.get("schema") for table in run_element.findall("data/table")
                )
                if schema
            )
        )
        runs.append(
            TraceRun(
                number=number,
                template_name=_text_or_none(summary, "template-name"),
                instruments_version=_text_or_none(summary, "instruments-version"),
                duration_seconds=_optional_float(
                    _text_or_none(summary, "duration"), field="duration"
                ),
                end_reason=_text_or_none(summary, "end-reason"),
                target_pid=target_pid,
                target_process_name=target_process_name,
                schemas=schemas,
            )
        )

    if not runs:
        raise InstrumentsExportError(
            "trace TOC contains no <run> elements; the trace bundle has no "
            "recorded runs to export"
        )
    return TraceTableOfContents(runs=tuple(runs))


# --- Table data -------------------------------------------------------


@dataclass(frozen=True)
class TableColumn:
    """One ``<col>`` definition from an exported table's schema."""

    mnemonic: str
    name: str | None
    engineering_type: str


@dataclass(frozen=True)
class CellValue:
    """One resolved cell, after any ``ref`` indirection was followed."""

    tag: str
    text: str | None
    fmt: str | None

    def as_int(self, *, field: str) -> int:
        if self.text is None:
            raise InstrumentsExportError(
                f"exported value for {field!r} has no text content"
            )
        try:
            return int(self.text.strip())
        except ValueError as exc:
            raise InstrumentsExportError(
                f"exported value for {field!r} is not an integer: " f"{self.text!r}"
            ) from exc


@dataclass(frozen=True)
class ExportedRow:
    """One table row, keyed by column mnemonic."""

    values: dict[str, CellValue]

    def get(self, mnemonic: str) -> CellValue | None:
        return self.values.get(mnemonic)

    def require(self, mnemonic: str) -> CellValue:
        value = self.values.get(mnemonic)
        if value is None:
            raise InstrumentsExportError(f"exported row has no column {mnemonic!r}")
        return value


@dataclass(frozen=True)
class ExportedTable:
    """A fully parsed exported table."""

    schema_name: str
    columns: tuple[TableColumn, ...]
    rows: tuple[ExportedRow, ...]

    @property
    def row_count(self) -> int:
        return len(self.rows)

    @property
    def column_mnemonics(self) -> tuple[str, ...]:
        return tuple(column.mnemonic for column in self.columns)


def _build_id_index(root: ElementTree.Element) -> dict[str, ElementTree.Element]:
    """Index every element that defines an ``id``.

    Definitions can appear anywhere, including nested inside a composite
    value such as ``formatted-label``, so the whole document is walked.
    """
    index: dict[str, ElementTree.Element] = {}
    for element in root.iter():
        identifier = element.get("id")
        if identifier is not None:
            index.setdefault(identifier, element)
    return index


def _resolve(
    element: ElementTree.Element,
    index: dict[str, ElementTree.Element],
) -> ElementTree.Element:
    """Follow ``ref`` indirection to the element that defines the value.

    Chains are followed, and a cycle or dangling reference is an error
    rather than a silently empty value.
    """
    seen: set[str] = set()
    current = element
    while True:
        reference = current.get("ref")
        if reference is None:
            return current
        if reference in seen:
            raise InstrumentsExportError(
                f"exported table has a cyclic ref chain at id {reference!r}"
            )
        seen.add(reference)
        target = index.get(reference)
        if target is None:
            raise InstrumentsExportError(
                f"exported table references undefined id {reference!r}"
            )
        current = target


def _parse_columns(schema_element: ElementTree.Element) -> tuple[TableColumn, ...]:
    columns: list[TableColumn] = []
    for col in schema_element.findall("col"):
        mnemonic = _text_or_none(col, "mnemonic")
        engineering_type = _text_or_none(col, "engineering-type")
        if mnemonic is None or engineering_type is None:
            raise InstrumentsExportError(
                "exported table column is missing <mnemonic> or " "<engineering-type>"
            )
        columns.append(
            TableColumn(
                mnemonic=mnemonic,
                name=_text_or_none(col, "name"),
                engineering_type=engineering_type,
            )
        )
    if not columns:
        raise InstrumentsExportError("exported table schema declares no <col> elements")
    return tuple(columns)


def parse_exported_table(
    xml_text: str, *, expected_schema: str | None = None
) -> ExportedTable:
    """Parse ``xctrace export --xpath ...`` XML into typed rows.

    ``expected_schema``, when given, must match the schema the document
    declares. A mismatch means the export did not return the table that
    was asked for, which is treated as an error rather than parsed
    anyway.
    """
    root = _parse_xml(xml_text, source="trace table export")
    if root.tag != "trace-query-result":
        raise InstrumentsExportError(
            f"expected a <trace-query-result> document, got <{root.tag}>. "
            "This does not look like `xctrace export --xpath` output."
        )

    node = root.find("node")
    if node is None:
        raise InstrumentsExportError(
            "trace export contains no <node>; the XPath selected nothing. "
            "Check the schema name against `xctrace export --toc`."
        )

    schema_element = node.find("schema")
    if schema_element is None:
        raise InstrumentsExportError(
            "trace export <node> has no <schema>; this table is not "
            "exportable in the row/column form this project parses"
        )
    schema_name = schema_element.get("name")
    if not schema_name:
        raise InstrumentsExportError("trace export <schema> is missing @name")
    if expected_schema is not None and schema_name != expected_schema:
        raise InstrumentsExportError(
            f"trace export returned schema {schema_name!r} but "
            f"{expected_schema!r} was requested"
        )

    columns = _parse_columns(schema_element)
    index = _build_id_index(root)

    rows: list[ExportedRow] = []
    for row_number, row_element in enumerate(node.findall("row"), start=1):
        children = list(row_element)
        if len(children) != len(columns):
            raise InstrumentsExportError(
                f"row {row_number} of table {schema_name!r} has "
                f"{len(children)} values but the schema declares "
                f"{len(columns)} columns. Refusing to map values onto "
                "columns positionally when the counts disagree."
            )
        values: dict[str, CellValue] = {}
        # strict=True is belt and braces: the length check above
        # already guarantees the two sequences match.
        for column, child in zip(columns, children, strict=True):
            if child.tag != column.engineering_type:
                raise InstrumentsExportError(
                    f"row {row_number} of table {schema_name!r} has "
                    f"<{child.tag}> where column {column.mnemonic!r} "
                    f"declares engineering type "
                    f"{column.engineering_type!r}"
                )
            resolved = _resolve(child, index)
            values[column.mnemonic] = CellValue(
                tag=resolved.tag,
                text=resolved.text,
                fmt=resolved.get("fmt"),
            )
        rows.append(ExportedRow(values=values))

    return ExportedTable(schema_name=schema_name, columns=columns, rows=tuple(rows))


# --- Summaries derived only from parsed rows --------------------------


@dataclass(frozen=True)
class ProcessGpuIntervals:
    """GPU interval statistics for one process in a trace.

    Every field is a direct consequence of parsed rows. Note what is
    absent: there is no utilization, occupancy, bandwidth or power
    figure, because none of those can be derived from this table without
    assumptions this project has not validated.
    """

    process_label: str
    """The trace's own label for the process, e.g. ``probe (14717)``."""
    pid: int | None
    interval_count: int
    duration_sum_ns: int
    """Sum of every interval's duration.

    Metal runs several channels (Vertex, Fragment, Compute) that overlap
    in time, so this sum counts concurrent work more than once. It is
    therefore *not* GPU busy time and *not* a utilization numerator."""
    wall_span_ns: int
    """Last interval end minus first interval start, for this process."""


@dataclass(frozen=True)
class MetalGpuIntervalSummary:
    """Per-process breakdown of a ``metal-gpu-intervals`` table.

    Metal System Trace records GPU work for every process on the system,
    not only the one that was launched. A real capture of a trivial
    local Metal program also contained intervals belonging to
    ``WindowServer`` and ``com.apple.WebKit.GPU``. Attribution is
    therefore kept per process, and a caller must name the process it
    means rather than receive a system-wide total labelled as its own.
    """

    total_interval_count: int
    per_process: tuple[ProcessGpuIntervals, ...]

    def for_process(self, pid: int) -> ProcessGpuIntervals | None:
        for entry in self.per_process:
            if entry.pid == pid:
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_interval_count": self.total_interval_count,
            "per_process": [
                {
                    "process_label": entry.process_label,
                    "pid": entry.pid,
                    "interval_count": entry.interval_count,
                    "duration_sum_ns": entry.duration_sum_ns,
                    "wall_span_ns": entry.wall_span_ns,
                }
                for entry in self.per_process
            ],
        }


def _process_pid(value: CellValue) -> int | None:
    """Extract a pid from a resolved ``process`` cell's ``fmt`` label.

    The label has the shape ``name (pid)``. Returning ``None`` when it
    does not match keeps an unparsable label from becoming a wrong pid.
    """
    if value.fmt is None:
        return None
    match = re.search(r"\((\d+)\)\s*$", value.fmt)
    if match is None:
        return None
    return int(match.group(1))


def summarize_metal_gpu_intervals(table: ExportedTable) -> MetalGpuIntervalSummary:
    """Aggregate a parsed ``metal-gpu-intervals`` table by process."""
    if table.schema_name != "metal-gpu-intervals":
        raise InstrumentsExportError(
            f"expected a 'metal-gpu-intervals' table, got " f"{table.schema_name!r}"
        )
    required = {"start", "duration", "process"}
    missing = required.difference(table.column_mnemonics)
    if missing:
        raise InstrumentsExportError(
            "metal-gpu-intervals table is missing required columns: "
            + ", ".join(sorted(missing))
        )

    counts: dict[str, int] = {}
    duration_sums: dict[str, int] = {}
    first_start: dict[str, int] = {}
    last_end: dict[str, int] = {}
    pids: dict[str, int | None] = {}

    for row in table.rows:
        process = row.require("process")
        label = process.fmt or "<unknown process>"
        start = row.require("start").as_int(field="start")
        duration = row.require("duration").as_int(field="duration")
        if duration < 0:
            raise InstrumentsExportError(
                f"metal-gpu-intervals row has a negative duration: {duration}"
            )
        end = start + duration

        counts[label] = counts.get(label, 0) + 1
        duration_sums[label] = duration_sums.get(label, 0) + duration
        pids.setdefault(label, _process_pid(process))
        if label not in first_start or start < first_start[label]:
            first_start[label] = start
        if label not in last_end or end > last_end[label]:
            last_end[label] = end

    per_process = tuple(
        ProcessGpuIntervals(
            process_label=label,
            pid=pids[label],
            interval_count=counts[label],
            duration_sum_ns=duration_sums[label],
            wall_span_ns=last_end[label] - first_start[label],
        )
        for label in sorted(counts, key=lambda key: (-counts[key], key))
    )
    return MetalGpuIntervalSummary(
        total_interval_count=sum(counts.values()), per_process=per_process
    )
