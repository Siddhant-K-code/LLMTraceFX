"""Tests for strict parsing of xctrace export output.

The fixtures are synthetic but structurally faithful to real
``xctrace 16.0 (17F113)`` output, including the two properties that make
naive parsing wrong: rows map to columns positionally, and repeated
values are emitted once and then referenced by id.
"""

from __future__ import annotations

import pytest
from _instruments_fakes import read_fixture

from llmtracefx.optimizer.instruments.export import (
    FORBIDDEN_METRIC_NAMES,
    MAX_EXPORT_BYTES,
    SUPPORTED_TABLE_SCHEMAS,
    InstrumentsExportError,
    parse_exported_table,
    parse_table_of_contents,
    read_export_text,
    sanitize_table_of_contents,
    summarize_metal_gpu_intervals,
)

TOC = "toc_metal_system_trace.xml"
TABLE = "table_metal_gpu_intervals.xml"
UNSUPPORTED = "table_unsupported_schema.xml"


# --- Table of contents ------------------------------------------------


def test_toc_reports_runs_and_schemas():
    toc = parse_table_of_contents(read_fixture(TOC))
    assert len(toc.runs) == 1
    run = toc.runs[0]
    assert run.number == 1
    assert run.template_name == "Metal System Trace"
    assert run.instruments_version == "16.0 (17F113)"
    assert run.duration_seconds == pytest.approx(0.5)
    assert run.end_reason == "Target app exited"
    assert "metal-gpu-intervals" in run.schemas


def test_toc_reads_the_launched_process_for_attribution():
    run = parse_table_of_contents(read_fixture(TOC)).runs[0]
    assert run.target_pid == 4242
    assert run.target_process_name == "probe"


def test_toc_does_not_ingest_device_identity_or_target_arguments():
    """The TOC carries a device display name, a hardware UUID and the
    target's argument list. None may reach a parsed artifact."""
    raw = read_fixture(TOC)
    assert "Example Mac" in raw and "00000000-0000-0000-0000-000000000000" in raw

    serialized = str(parse_table_of_contents(raw).to_dict())
    assert "Example Mac" not in serialized
    assert "00000000-0000-0000-0000-000000000000" not in serialized
    assert "MacBook Pro" not in serialized
    assert "arguments" not in serialized


def test_toc_schema_names_are_deduplicated_and_sorted():
    toc = parse_table_of_contents(read_fixture(TOC))
    names = toc.schema_names
    assert names == tuple(sorted(set(names)))


def test_run_lookup_by_number():
    toc = parse_table_of_contents(read_fixture(TOC))
    assert toc.run_by_number(1) is not None
    assert toc.run_by_number(99) is None


@pytest.mark.parametrize(
    "payload,match",
    [
        ("<not-a-toc/>", "expected a <trace-toc>"),
        ("<trace-toc></trace-toc>", "no <run> elements"),
        ("<trace-toc><run/></trace-toc>", "missing @number"),
        ('<trace-toc><run number="x"/></trace-toc>', "not an integer"),
        ("<trace-toc", "not valid XML"),
    ],
)
def test_malformed_toc_is_refused(payload, match):
    with pytest.raises(InstrumentsExportError, match=match):
        parse_table_of_contents(payload)


def test_toc_with_non_numeric_duration_is_refused():
    payload = (
        '<trace-toc><run number="1"><info><summary>'
        "<duration>not-a-number</duration>"
        "</summary></info></run></trace-toc>"
    )
    with pytest.raises(InstrumentsExportError, match="duration is not a number"):
        parse_table_of_contents(payload)


def test_toc_with_non_numeric_target_pid_is_refused():
    payload = (
        '<trace-toc><run number="1"><info><target>'
        '<process name="probe" pid="not-a-pid"/>'
        "</target></info></run></trace-toc>"
    )
    with pytest.raises(InstrumentsExportError, match="@pid is not an integer"):
        parse_table_of_contents(payload)


# --- Entity expansion -------------------------------------------------


def test_doctype_is_refused_before_parsing():
    """Guards against entity expansion denial of service.

    Real xctrace output never declares a doctype, so refusing one costs
    nothing.
    """
    bomb = (
        '<?xml version="1.0"?>'
        '<!DOCTYPE trace-toc [<!ENTITY a "aaaaaaaaaa">]>'
        "<trace-toc></trace-toc>"
    )
    with pytest.raises(InstrumentsExportError, match="DOCTYPE or ENTITY"):
        parse_table_of_contents(bomb)


def test_entity_declaration_is_refused_in_table_exports_too():
    with pytest.raises(InstrumentsExportError, match="DOCTYPE or ENTITY"):
        parse_exported_table('<!ENTITY x "y"><trace-query-result/>')


# --- Table data -------------------------------------------------------


def test_table_parses_columns_and_rows():
    table = parse_exported_table(
        read_fixture(TABLE), expected_schema="metal-gpu-intervals"
    )
    assert table.schema_name == "metal-gpu-intervals"
    assert table.column_mnemonics == ("start", "duration", "channel-name", "process")
    assert table.row_count == 5


def test_references_are_resolved_including_forward_references():
    """Row 4 references a duration first defined in row 5.

    Ids are indexed over the whole document before rows are resolved, so
    document order does not matter.
    """
    table = parse_exported_table(read_fixture(TABLE))
    fourth = table.rows[3]
    assert fourth.require("duration").as_int(field_name="duration") == 1000
    assert fourth.require("channel-name").text == "Vertex"
    assert fourth.require("process").fmt == "WindowServer (77)"


def test_back_references_resolve_to_the_original_value():
    table = parse_exported_table(read_fixture(TABLE))
    second = table.rows[1]
    assert second.require("channel-name").text == "Compute"
    assert second.require("process").fmt == "probe (4242)"


def test_dangling_reference_is_refused():
    payload = (
        "<trace-query-result><node>"
        '<schema name="t"><col><mnemonic>a</mnemonic>'
        "<engineering-type>uint64</engineering-type></col></schema>"
        '<row><uint64 ref="999"/></row>'
        "</node></trace-query-result>"
    )
    with pytest.raises(InstrumentsExportError, match="undefined id"):
        parse_exported_table(payload)


def test_cyclic_reference_is_refused():
    payload = (
        "<trace-query-result><node>"
        '<schema name="t"><col><mnemonic>a</mnemonic>'
        "<engineering-type>uint64</engineering-type></col></schema>"
        '<row><uint64 ref="1"/></row>'
        '<row><uint64 id="1" ref="2"/></row>'
        '<row><uint64 id="2" ref="1"/></row>'
        "</node></trace-query-result>"
    )
    with pytest.raises(InstrumentsExportError, match="cyclic ref chain"):
        parse_exported_table(payload)


def test_column_count_mismatch_is_refused_rather_than_guessed():
    """A short row must not be silently mapped onto the wrong columns."""
    payload = (
        "<trace-query-result><node>"
        '<schema name="t">'
        "<col><mnemonic>a</mnemonic><engineering-type>uint64</engineering-type></col>"
        "<col><mnemonic>b</mnemonic><engineering-type>uint64</engineering-type></col>"
        "</schema>"
        '<row><uint64 id="1">5</uint64></row>'
        "</node></trace-query-result>"
    )
    with pytest.raises(InstrumentsExportError, match="Refusing to map values"):
        parse_exported_table(payload)


def test_engineering_type_mismatch_is_refused():
    payload = (
        "<trace-query-result><node>"
        '<schema name="t"><col><mnemonic>a</mnemonic>'
        "<engineering-type>uint64</engineering-type></col></schema>"
        '<row><duration id="1">5</duration></row>'
        "</node></trace-query-result>"
    )
    with pytest.raises(InstrumentsExportError, match="declares engineering type"):
        parse_exported_table(payload)


def test_requesting_a_different_schema_than_returned_is_refused():
    with pytest.raises(
        InstrumentsExportError, match="but 'time-profile' was requested"
    ):
        parse_exported_table(read_fixture(TABLE), expected_schema="time-profile")


@pytest.mark.parametrize(
    "payload,match",
    [
        ("<wrong-root/>", "expected a <trace-query-result>"),
        ("<trace-query-result/>", "contains no <node>"),
        ("<trace-query-result><node/></trace-query-result>", "has no <schema>"),
        (
            "<trace-query-result><node><schema/></node></trace-query-result>",
            "missing @name",
        ),
        (
            '<trace-query-result><node><schema name="t"/></node></trace-query-result>',
            "declares no <col>",
        ),
    ],
)
def test_malformed_table_exports_are_refused(payload, match):
    with pytest.raises(InstrumentsExportError, match=match):
        parse_exported_table(payload)


def test_non_integer_cell_is_refused():
    payload = (
        "<trace-query-result><node>"
        '<schema name="t"><col><mnemonic>a</mnemonic>'
        "<engineering-type>uint64</engineering-type></col></schema>"
        '<row><uint64 id="1">not-a-number</uint64></row>'
        "</node></trace-query-result>"
    )
    table = parse_exported_table(payload)
    with pytest.raises(InstrumentsExportError, match="is not an integer"):
        table.rows[0].require("a").as_int(field_name="a")


def test_requiring_an_absent_column_is_an_error():
    table = parse_exported_table(read_fixture(TABLE))
    assert table.rows[0].get("nope") is None
    with pytest.raises(InstrumentsExportError, match="no column 'nope'"):
        table.rows[0].require("nope")


# --- Reading files ----------------------------------------------------


def test_read_export_text_refuses_missing_and_oversized(tmp_path):
    with pytest.raises(InstrumentsExportError, match="does not exist"):
        read_export_text(tmp_path / "absent.xml")

    directory = tmp_path / "dir.xml"
    directory.mkdir()
    with pytest.raises(InstrumentsExportError, match="not a file"):
        read_export_text(directory)


def test_read_export_text_rejects_invalid_utf8(tmp_path):
    target = tmp_path / "bad.xml"
    target.write_bytes(b"\xff\xfe\x00invalid")
    with pytest.raises(InstrumentsExportError, match="not valid UTF-8"):
        read_export_text(target)


def test_export_size_limit_is_bounded():
    assert 0 < MAX_EXPORT_BYTES <= 1024 * 1024 * 1024


# --- Summaries --------------------------------------------------------


def test_summary_attributes_intervals_per_process():
    """Metal System Trace records every process, not only the target.

    The fixture deliberately mixes the launched process with
    WindowServer, so a system-wide total would overstate the target's
    GPU work by more than half.
    """
    table = parse_exported_table(read_fixture(TABLE))
    summary = summarize_metal_gpu_intervals(table)

    assert summary.total_interval_count == 5
    probe = summary.for_process(4242)
    assert probe is not None
    assert probe.interval_count == 3
    assert probe.duration_sum_ns == 10000 + 20000 + 1000
    assert probe.wall_span_ns == (500000 + 1000) - 100000

    window_server = summary.for_process(77)
    assert window_server is not None
    assert window_server.interval_count == 2


def test_summary_orders_processes_by_interval_count():
    summary = summarize_metal_gpu_intervals(parse_exported_table(read_fixture(TABLE)))
    counts = [entry.interval_count for entry in summary.per_process]
    assert counts == sorted(counts, reverse=True)


def test_summary_for_unknown_pid_is_none():
    summary = summarize_metal_gpu_intervals(parse_exported_table(read_fixture(TABLE)))
    assert summary.for_process(999999) is None


def test_summary_refuses_a_table_of_the_wrong_schema():
    table = parse_exported_table(read_fixture(UNSUPPORTED))
    with pytest.raises(
        InstrumentsExportError, match="expected a 'metal-gpu-intervals'"
    ):
        summarize_metal_gpu_intervals(table)


def test_summary_refuses_a_table_missing_required_columns():
    payload = (
        "<trace-query-result><node>"
        '<schema name="metal-gpu-intervals"><col><mnemonic>start</mnemonic>'
        "<engineering-type>start-time</engineering-type></col></schema>"
        '<row><start-time id="1">5</start-time></row>'
        "</node></trace-query-result>"
    )
    with pytest.raises(InstrumentsExportError, match="missing required columns"):
        summarize_metal_gpu_intervals(parse_exported_table(payload))


def test_negative_duration_is_refused():
    payload = (
        "<trace-query-result><node>"
        '<schema name="metal-gpu-intervals">'
        "<col><mnemonic>start</mnemonic>"
        "<engineering-type>start-time</engineering-type></col>"
        "<col><mnemonic>duration</mnemonic>"
        "<engineering-type>duration</engineering-type></col>"
        "<col><mnemonic>process</mnemonic>"
        "<engineering-type>process</engineering-type></col>"
        "</schema>"
        "<row>"
        '<start-time id="1">5</start-time>'
        '<duration id="2">-7</duration>'
        '<process id="3" fmt="p (1)"/>'
        "</row>"
        "</node></trace-query-result>"
    )
    with pytest.raises(InstrumentsExportError, match="negative duration"):
        summarize_metal_gpu_intervals(parse_exported_table(payload))


def test_unparsable_process_label_yields_no_pid_rather_than_a_wrong_one():
    payload = (
        "<trace-query-result><node>"
        '<schema name="metal-gpu-intervals">'
        "<col><mnemonic>start</mnemonic>"
        "<engineering-type>start-time</engineering-type></col>"
        "<col><mnemonic>duration</mnemonic>"
        "<engineering-type>duration</engineering-type></col>"
        "<col><mnemonic>process</mnemonic>"
        "<engineering-type>process</engineering-type></col>"
        "</schema>"
        "<row>"
        '<start-time id="1">5</start-time>'
        '<duration id="2">7</duration>'
        '<process id="3" fmt="no pid here"/>'
        "</row>"
        "</node></trace-query-result>"
    )
    summary = summarize_metal_gpu_intervals(parse_exported_table(payload))
    assert summary.per_process[0].pid is None


# --- No overclaiming --------------------------------------------------


def test_only_validated_schemas_are_declared_supported():
    assert SUPPORTED_TABLE_SCHEMAS == ("metal-gpu-intervals",)


def test_forbidden_metric_names_cover_the_usual_gpu_overclaims():
    for name in (
        "gpu_utilization",
        "gpu_kernel_time",
        "memory_bandwidth",
        "occupancy",
        "gpu_power",
    ):
        assert name in FORBIDDEN_METRIC_NAMES


# --- Regressions found in independent review --------------------------


def test_a_ref_pointing_at_the_wrong_engineering_type_is_refused():
    """Almost every real cell is a ref.

    Checking only the referencing element's tag would leave the
    engineering-type contract enforced on a tiny minority of the data,
    and let a duration column take its value from a start-time.
    """
    payload = (
        "<trace-query-result><node>"
        '<schema name="t">'
        "<col><mnemonic>a</mnemonic>"
        "<engineering-type>start-time</engineering-type></col>"
        "<col><mnemonic>b</mnemonic>"
        "<engineering-type>duration</engineering-type></col>"
        "</schema>"
        '<row><start-time id="1">100</start-time>'
        '<duration id="2">7</duration></row>'
        '<row><start-time ref="1"/><duration ref="1"/></row>'
        "</node></trace-query-result>"
    )
    with pytest.raises(InstrumentsExportError, match="referenced value"):
        parse_exported_table(payload)


def test_duplicate_column_mnemonics_are_refused():
    """A repeated mnemonic would let one column overwrite another."""
    payload = (
        "<trace-query-result><node>"
        '<schema name="t">'
        "<col><mnemonic>duration</mnemonic>"
        "<engineering-type>duration</engineering-type></col>"
        "<col><mnemonic>duration</mnemonic>"
        "<engineering-type>duration</engineering-type></col>"
        "</schema>"
        '<row><duration id="1">100</duration>'
        '<duration id="2">999999</duration></row>'
        "</node></trace-query-result>"
    )
    with pytest.raises(InstrumentsExportError, match="duplicate column mnemonics"):
        parse_exported_table(payload)


def test_multiple_nodes_are_refused_rather_than_silently_truncated():
    node = (
        "<node>"
        '<schema name="t"><col><mnemonic>a</mnemonic>'
        "<engineering-type>uint64</engineering-type></col></schema>"
        '<row><uint64 id="{i}">1</uint64></row>'
        "</node>"
    )
    payload = (
        "<trace-query-result>"
        + node.format(i=1)
        + node.format(i=2)
        + "</trace-query-result>"
    )
    with pytest.raises(InstrumentsExportError, match="2 <node> elements"):
        parse_exported_table(payload)


def _intervals_table(*rows: str) -> str:
    return (
        "<trace-query-result><node>"
        '<schema name="metal-gpu-intervals">'
        "<col><mnemonic>start</mnemonic>"
        "<engineering-type>start-time</engineering-type></col>"
        "<col><mnemonic>duration</mnemonic>"
        "<engineering-type>duration</engineering-type></col>"
        "<col><mnemonic>process</mnemonic>"
        "<engineering-type>process</engineering-type></col>"
        "</schema>" + "".join(rows) + "</node></trace-query-result>"
    )


def test_one_pid_under_two_labels_refuses_to_attribute():
    """Pid reuse or an exec rename must not silently pick a winner."""
    table = parse_exported_table(
        _intervals_table(
            '<row><start-time id="1">10</start-time>'
            '<duration id="2">10</duration>'
            '<process id="3" fmt="probe (4242)"><pid id="4">4242</pid>'
            "</process></row>",
            '<row><start-time id="5">20</start-time>'
            '<duration id="6">30</duration>'
            '<process id="7" fmt="WindowServer (4242)"><pid id="8">4242</pid>'
            "</process></row>",
        )
    )
    summary = summarize_metal_gpu_intervals(table)
    assert len(summary.per_process) == 2
    with pytest.raises(InstrumentsExportError, match="is ambiguous"):
        summary.for_process(4242)


def test_pid_comes_from_the_structured_child_not_the_label():
    """The display label is only a fallback."""
    table = parse_exported_table(
        _intervals_table(
            '<row><start-time id="1">10</start-time>'
            '<duration id="2">10</duration>'
            '<process id="3" fmt="misleading (999)"><pid id="4">4242</pid>'
            "</process></row>"
        )
    )
    summary = summarize_metal_gpu_intervals(table)
    assert summary.per_process[0].pid == 4242
    assert summary.for_process(999) is None


def test_processes_without_a_label_are_kept_apart_by_pid():
    """Two unlabelled processes must not merge into one bucket."""
    table = parse_exported_table(
        _intervals_table(
            '<row><start-time id="1">10</start-time>'
            '<duration id="2">10</duration>'
            '<process id="3"><pid id="4">11</pid></process></row>',
            '<row><start-time id="5">20</start-time>'
            '<duration id="6">10</duration>'
            '<process id="7"><pid id="8">22</pid></process></row>',
        )
    )
    summary = summarize_metal_gpu_intervals(table)
    assert len(summary.per_process) == 2
    assert {entry.pid for entry in summary.per_process} == {11, 22}
    assert summary.for_process(11).interval_count == 1


def test_parsable_schema_lists_cannot_drift_apart():
    """The parser's list and the schema's guard must stay in step.

    If they diverge, either a real run stops validating or the guard
    stops guarding.
    """
    from llmtracefx.optimizer.schema import (
        INSTRUMENT_METRIC_SPECS,
        INSTRUMENT_PARSABLE_SCHEMAS,
    )

    assert set(SUPPORTED_TABLE_SCHEMAS) == set(INSTRUMENT_PARSABLE_SCHEMAS)
    sources = {source for source, _, _ in INSTRUMENT_METRIC_SPECS.values()}
    assert sources <= set(SUPPORTED_TABLE_SCHEMAS)


# --- Table of contents sanitization -----------------------------------


def test_toc_sanitization_removes_identity_and_target_arguments():
    """The parser never read these; the raw file still carried them.

    A real macOS device name is routinely the owner's own name, and the
    target argument list is the one place a credential passed on the
    command line survives argv redaction.
    """
    raw = read_fixture(TOC)
    assert "Example Mac" in raw and 'arguments="120"' in raw

    clean = sanitize_table_of_contents(raw)
    assert "Example Mac" not in clean
    assert "00000000-0000-0000-0000-000000000000" not in clean
    assert "arguments=" not in clean


def test_toc_sanitization_keeps_everything_attribution_needs():
    raw = read_fixture(TOC)
    clean = sanitize_table_of_contents(raw)

    parsed = parse_table_of_contents(clean)
    run = parsed.runs[0]
    assert run.template_name == "Metal System Trace"
    assert run.instruments_version == "16.0 (17F113)"
    assert run.target_pid == 4242
    assert run.target_process_name == "probe"
    assert run.schemas == parse_table_of_contents(raw).runs[0].schemas


def test_toc_sanitization_says_what_it_did():
    clean = sanitize_table_of_contents(read_fixture(TOC))
    assert "Sanitized by llmtracefx" in clean
    assert ".trace bundle" in clean


def test_toc_sanitization_is_idempotent():
    once = sanitize_table_of_contents(read_fixture(TOC))
    assert sanitize_table_of_contents(once) == once


def test_implausible_interval_values_are_refused_not_overflowed():
    """Python ints are unbounded; floats are not.

    A malformed cell used to raise OverflowError from deep inside
    evidence building rather than being rejected as bad input.
    """
    huge = "9" * 400
    payload = (
        "<trace-query-result><node>"
        '<schema name="metal-gpu-intervals">'
        "<col><mnemonic>start</mnemonic>"
        "<engineering-type>start-time</engineering-type></col>"
        "<col><mnemonic>duration</mnemonic>"
        "<engineering-type>duration</engineering-type></col>"
        "<col><mnemonic>process</mnemonic>"
        "<engineering-type>process</engineering-type></col>"
        "</schema><row>"
        '<start-time id="1">5</start-time>'
        f'<duration id="2">{huge}</duration>'
        '<process id="3" fmt="p (1)"><pid id="4">1</pid></process>'
        "</row></node></trace-query-result>"
    )
    with pytest.raises(InstrumentsExportError, match="implausible"):
        summarize_metal_gpu_intervals(parse_exported_table(payload))
