"""Tests for the plan / record / import workflows and their CLI surfaces."""

from __future__ import annotations

import json

import pytest
from _instruments_fakes import (
    COMMAND_LINE_TOOLS_STDERR,
    FakeCommandRunner,
    FakeLauncher,
    FakeProcess,
    fail,
    ok,
    read_fixture,
)

from llmtracefx.optimizer.cli import main
from llmtracefx.optimizer.instruments.capability import detect_xctrace_capability
from llmtracefx.optimizer.instruments.evidence import (
    TraceEvidenceInputs,
    build_instruments_evidence,
    unsupported_evidence,
)
from llmtracefx.optimizer.instruments.export import (
    FORBIDDEN_METRIC_NAMES,
    parse_exported_table,
)
from llmtracefx.optimizer.instruments.workflow import (
    import_trace,
    plan_trace,
    record_trace,
)
from llmtracefx.optimizer.schema import (
    CommandInfo,
    ExperimentRecord,
    ModelInfo,
    PlatformInfo,
    RepetitionInfo,
    RuntimeInfo,
    SchemaValidationError,
    utc_now_iso,
)

TOC = "toc_metal_system_trace.xml"
TABLE = "table_metal_gpu_intervals.xml"
UNSUPPORTED = "table_unsupported_schema.xml"


def exporting_runner(**overrides) -> FakeCommandRunner:
    exports = {
        "toc": ok((), read_fixture(TOC)),
        "metal-gpu-intervals": ok((), read_fixture(TABLE)),
        "displayed-surfaces-per-second": ok((), read_fixture(UNSUPPORTED)),
    }
    exports.update(overrides)
    return FakeCommandRunner(exports=exports)


def capability(runner: FakeCommandRunner):
    return detect_xctrace_capability(
        runner=runner,
        os_name="Darwin",
        architecture="arm64",
        path_resolver=lambda: "/usr/bin/xctrace",
    )


# --- Dry-run plan -----------------------------------------------------


def test_plan_executes_nothing_and_creates_nothing(tmp_path):
    runner = exporting_runner()
    trace = tmp_path / "run.trace"
    out = tmp_path / "artifacts"

    plan = plan_trace(
        runner=runner,
        command=("/bin/infer", "--tokens", "8"),
        output_trace=trace,
        output_dir=out,
    )

    assert plan.ready is True
    assert not trace.exists()
    assert not out.exists()
    # Only read-only capability probes ran.
    assert all(
        argv[1:] in (("version",), ("list", "templates")) for argv in runner.calls
    )


def test_plan_lists_exact_output_paths(tmp_path):
    plan = plan_trace(
        runner=exporting_runner(),
        command=("/bin/infer",),
        output_trace=tmp_path / "run.trace",
        output_dir=tmp_path / "artifacts",
    )
    paths = plan.output_paths
    assert paths["trace_bundle"] == str(tmp_path / "run.trace")
    assert paths["evidence"].endswith("instruments_evidence.json")
    assert paths["capability_report"].endswith("capability_report.json")


def test_plan_reports_the_exact_argv_it_would_run(tmp_path):
    plan = plan_trace(
        runner=exporting_runner(),
        command=("/bin/infer", "--tokens", "8"),
        output_trace=tmp_path / "run.trace",
        output_dir=tmp_path / "artifacts",
        time_limit="45s",
    )
    argv = plan.record_plan.to_redacted_argv()
    assert argv[-4:] == ("--launch", "--", "/bin/infer", "--tokens", "8")[-4:]
    assert "45s" in argv


def test_plan_surfaces_unmet_capability_as_a_prerequisite(tmp_path):
    runner = FakeCommandRunner(
        version=fail((), returncode=1, stderr=COMMAND_LINE_TOOLS_STDERR)
    )
    plan = plan_trace(
        runner=runner,
        command=("/bin/infer",),
        output_trace=tmp_path / "run.trace",
        output_dir=tmp_path / "artifacts",
    )
    assert plan.ready is False
    assert any("full Xcode" in item for item in plan.prerequisites)


def test_plan_surfaces_a_collision_as_a_prerequisite(tmp_path):
    trace = tmp_path / "run.trace"
    trace.mkdir()
    plan = plan_trace(
        runner=exporting_runner(),
        command=("/bin/infer",),
        output_trace=trace,
        output_dir=tmp_path / "artifacts",
    )
    assert plan.ready is False
    assert any("never overwritten" in item for item in plan.prerequisites)


def test_plan_serializes_to_json(tmp_path):
    plan = plan_trace(
        runner=exporting_runner(),
        command=("/bin/infer",),
        output_trace=tmp_path / "run.trace",
        output_dir=tmp_path / "artifacts",
    )
    payload = json.loads(plan.to_json())
    assert payload["ready"] is True
    assert payload["capability"] == "supported"


# --- Record -----------------------------------------------------------


def test_record_writes_evidence_with_attributed_metrics(tmp_path):
    trace = tmp_path / "run.trace"
    runner = exporting_runner()
    launcher = FakeLauncher(FakeProcess(returncode=0), creates_trace=trace)

    collection = record_trace(
        runner=runner,
        launcher=launcher,
        command=("/bin/infer",),
        output_trace=trace,
        output_dir=tmp_path / "artifacts",
    )

    assert collection.succeeded is True
    evidence = collection.evidence
    assert evidence.parsed_schemas == ("metal-gpu-intervals",)
    # The fixture's launched process is pid 4242 with 3 of 5 intervals.
    assert evidence.metrics["metal_gpu_interval_count"].value == 3.0
    assert evidence.metrics["metal_gpu_interval_count_all_processes"].value == 5.0


def test_record_notes_explain_the_attribution(tmp_path):
    trace = tmp_path / "run.trace"
    collection = record_trace(
        runner=exporting_runner(),
        launcher=FakeLauncher(FakeProcess(returncode=0), creates_trace=trace),
        command=("/bin/infer",),
        output_trace=trace,
        output_dir=tmp_path / "artifacts",
    )
    notes = collection.evidence.notes or ""
    assert "pid 4242 only" in notes
    assert "not GPU busy time or utilization" in notes


def test_record_on_unsupported_capability_does_not_launch_anything(tmp_path):
    runner = FakeCommandRunner(
        version=fail((), returncode=1, stderr=COMMAND_LINE_TOOLS_STDERR)
    )
    launcher = FakeLauncher(FakeProcess())

    collection = record_trace(
        runner=runner,
        launcher=launcher,
        command=("/bin/infer",),
        output_trace=tmp_path / "run.trace",
        output_dir=tmp_path / "artifacts",
    )

    assert launcher.spawned == []
    assert collection.succeeded is False
    assert collection.evidence.capability == "command_line_tools_only"
    assert collection.evidence.metrics == {}


def test_record_failure_still_writes_evidence(tmp_path):
    trace = tmp_path / "run.trace"
    out = tmp_path / "artifacts"
    collection = record_trace(
        runner=exporting_runner(),
        launcher=FakeLauncher(FakeProcess(returncode=2)),
        command=("/bin/infer",),
        output_trace=trace,
        output_dir=out,
    )
    assert collection.succeeded is False
    assert (out / "instruments_evidence.json").exists()
    assert collection.evidence.metrics == {}


# --- Import -----------------------------------------------------------


def test_import_of_an_existing_bundle(tmp_path):
    trace = tmp_path / "run.trace"
    trace.mkdir()
    out = tmp_path / "artifacts"

    collection = import_trace(
        runner=exporting_runner(), trace_path=trace, output_dir=out
    )

    assert collection.evidence.parsed_schemas == ("metal-gpu-intervals",)
    assert (out / "trace_toc.json").exists()
    assert (out / "instruments_evidence.json").exists()


def test_import_records_unsupported_schemas_explicitly(tmp_path):
    trace = tmp_path / "run.trace"
    trace.mkdir()
    collection = import_trace(
        runner=exporting_runner(), trace_path=trace, output_dir=tmp_path / "a"
    )
    evidence = collection.evidence
    assert "displayed-surfaces-per-second" in evidence.unsupported_schemas
    assert "time-profile" in evidence.unsupported_schemas
    assert set(evidence.parsed_schemas).isdisjoint(evidence.unsupported_schemas)


def test_import_of_a_schema_without_a_summarizer_derives_no_metric(tmp_path):
    trace = tmp_path / "run.trace"
    trace.mkdir()
    collection = import_trace(
        runner=exporting_runner(),
        trace_path=trace,
        output_dir=tmp_path / "a",
        table_schema="displayed-surfaces-per-second",
    )
    assert collection.evidence.metrics == {}
    assert "no strict summarizer" in (collection.message or "")


def test_import_of_a_schema_absent_from_the_toc_is_reported(tmp_path):
    trace = tmp_path / "run.trace"
    trace.mkdir()
    collection = import_trace(
        runner=exporting_runner(),
        trace_path=trace,
        output_dir=tmp_path / "a",
        table_schema="os-signpost",
    )
    assert collection.evidence.metrics == {}
    assert "not present in this trace" in (collection.message or "")


def test_import_with_no_export_reads_only_the_toc(tmp_path):
    trace = tmp_path / "run.trace"
    trace.mkdir()
    collection = import_trace(
        runner=exporting_runner(),
        trace_path=trace,
        output_dir=tmp_path / "a",
        table_schema=None,
    )
    assert collection.table is None
    assert collection.evidence.metrics == {}
    assert collection.evidence.available_schemas


def test_import_of_a_missing_bundle_raises(tmp_path):
    from llmtracefx.optimizer.instruments.export import InstrumentsExportError

    with pytest.raises(InstrumentsExportError, match="does not exist"):
        import_trace(
            runner=exporting_runner(),
            trace_path=tmp_path / "absent.trace",
            output_dir=tmp_path / "a",
        )


def test_import_evidence_carries_no_absolute_path(tmp_path):
    trace = tmp_path / "run.trace"
    trace.mkdir()
    out = tmp_path / "artifacts"
    import_trace(runner=exporting_runner(), trace_path=trace, output_dir=out)

    raw = (out / "instruments_evidence.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in raw
    assert json.loads(raw)["trace_bundle_name"] == "run.trace"


# --- Evidence and no overclaiming -------------------------------------


def test_evidence_without_a_target_pid_emits_no_metric(tmp_path):
    table = parse_exported_table(read_fixture(TABLE))
    evidence = build_instruments_evidence(
        TraceEvidenceInputs(
            capability=capability(exporting_runner()),
            trace_bundle_name="run.trace",
            template="Metal System Trace",
            available_schemas=("metal-gpu-intervals",),
            table=table,
            target_pid=None,
        )
    )
    assert evidence.metrics == {}
    assert "would misattribute" in (evidence.notes or "")


def test_evidence_for_a_pid_with_no_intervals_emits_no_metric():
    table = parse_exported_table(read_fixture(TABLE))
    evidence = build_instruments_evidence(
        TraceEvidenceInputs(
            capability=capability(exporting_runner()),
            trace_bundle_name="run.trace",
            template="Metal System Trace",
            available_schemas=("metal-gpu-intervals",),
            table=table,
            target_pid=999999,
        )
    )
    assert evidence.metrics == {}
    assert "contributed no GPU intervals" in (evidence.notes or "")


def test_no_emitted_metric_name_is_an_overclaim():
    table = parse_exported_table(read_fixture(TABLE))
    evidence = build_instruments_evidence(
        TraceEvidenceInputs(
            capability=capability(exporting_runner()),
            trace_bundle_name="run.trace",
            template="Metal System Trace",
            available_schemas=("metal-gpu-intervals",),
            table=table,
            target_pid=4242,
        )
    )
    assert evidence.metrics
    for name in evidence.metrics:
        assert name not in FORBIDDEN_METRIC_NAMES
    joined = " ".join(evidence.metrics)
    for forbidden in ("utilization", "occupancy", "bandwidth", "power"):
        assert forbidden not in joined


def test_unsupported_evidence_keeps_the_reason(tmp_path):
    runner = FakeCommandRunner(
        version=fail((), returncode=1, stderr=COMMAND_LINE_TOOLS_STDERR)
    )
    report = capability(runner)
    evidence = unsupported_evidence(report, template="Metal System Trace")
    assert evidence.capability == "command_line_tools_only"
    assert evidence.metrics == {}
    assert evidence.notes == report.reason


# --- Canonical schema integration -------------------------------------


def base_record(**overrides) -> ExperimentRecord:
    values = {
        "run_id": "r1",
        "started_at": utc_now_iso(),
        "platform": PlatformInfo(
            os_name="Darwin", os_version="26.6.2", architecture="arm64"
        ),
        "model": ModelInfo(model_id="m"),
        "runtime": RuntimeInfo(name="mlx"),
        "command": CommandInfo(argv=("infer",)),
        "repetition": RepetitionInfo(
            warmup_repetitions=0, measured_repetitions=1, repetition_index=0
        ),
    }
    values.update(overrides)
    return ExperimentRecord(**values)


def test_instruments_evidence_round_trips_in_an_experiment_record():
    table = parse_exported_table(read_fixture(TABLE))
    evidence = build_instruments_evidence(
        TraceEvidenceInputs(
            capability=capability(exporting_runner()),
            trace_bundle_name="run.trace",
            template="Metal System Trace",
            available_schemas=("metal-gpu-intervals", "time-profile"),
            table=table,
            target_pid=4242,
        )
    )
    record = base_record(instruments=evidence)
    record.validate()
    restored = ExperimentRecord.from_dict(record.to_dict())
    assert restored.instruments == evidence


def test_records_without_instruments_still_parse():
    """The field is additive: older records must load unchanged."""
    payload = base_record().to_dict()
    del payload["instruments"]
    assert ExperimentRecord.from_dict(payload).instruments is None


def test_instruments_memory_stays_separate_from_allocator_memory():
    """MLX allocator bytes and Instruments values must not merge."""
    from llmtracefx.optimizer.schema import Measurement, MemoryMetrics, MetricProvenance

    record = base_record(
        memory=MemoryMetrics(
            active=Measurement(
                value=1024.0,
                provenance=MetricProvenance.MEASURED_NATIVE,
                unit="bytes",
            )
        ),
        instruments=build_instruments_evidence(
            TraceEvidenceInputs(
                capability=capability(exporting_runner()),
                trace_bundle_name="run.trace",
                template="Metal System Trace",
                available_schemas=("metal-gpu-intervals",),
                table=parse_exported_table(read_fixture(TABLE)),
                target_pid=4242,
            )
        ),
    )
    record.validate()
    payload = record.to_dict()
    assert payload["memory"]["active"]["value"] == 1024.0
    assert "active" not in payload["instruments"]["metrics"]


def test_native_provenance_without_a_parsed_schema_is_rejected():
    """Structurally prevents a fabricated hardware measurement."""
    from llmtracefx.optimizer.schema import (
        InstrumentsEvidence,
        Measurement,
        MetricProvenance,
    )

    evidence = InstrumentsEvidence(
        parsed_schemas=(),
        metrics={
            "metal_gpu_interval_count": Measurement(
                value=1.0,
                provenance=MetricProvenance.MEASURED_NATIVE,
                unit="intervals",
            )
        },
    )
    with pytest.raises(SchemaValidationError, match="parsed_schemas is"):
        base_record(instruments=evidence).validate()


def test_a_schema_cannot_be_both_parsed_and_unsupported():
    from llmtracefx.optimizer.schema import InstrumentsEvidence

    evidence = InstrumentsEvidence(
        parsed_schemas=("metal-gpu-intervals",),
        unsupported_schemas=("metal-gpu-intervals",),
    )
    with pytest.raises(SchemaValidationError, match="both parsed and unsupported"):
        base_record(instruments=evidence).validate()


def test_negative_instrument_metric_is_rejected():
    from llmtracefx.optimizer.schema import (
        InstrumentsEvidence,
        Measurement,
        MetricProvenance,
    )

    evidence = InstrumentsEvidence(
        parsed_schemas=("metal-gpu-intervals",),
        metrics={
            "metal_gpu_interval_count": Measurement(
                value=-1.0,
                provenance=MetricProvenance.MEASURED_NATIVE,
                unit="intervals",
            )
        },
    )
    with pytest.raises(SchemaValidationError, match="must be >= 0"):
        base_record(instruments=evidence).validate()


def test_instruments_evidence_rejects_a_non_object_metrics_field():
    from llmtracefx.optimizer.schema import InstrumentsEvidence

    with pytest.raises(SchemaValidationError, match="must be an object"):
        InstrumentsEvidence.from_dict({"metrics": ["not", "an", "object"]})


def test_instruments_evidence_rejects_a_bare_string_schema_list():
    """A bare string is a Sequence, so tuple() would shred it."""
    from llmtracefx.optimizer.schema import InstrumentsEvidence

    with pytest.raises(SchemaValidationError, match="must be a list of strings"):
        InstrumentsEvidence.from_dict({"parsed_schemas": "metal-gpu-intervals"})


# --- CLI --------------------------------------------------------------


def run_cli(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    return excinfo.value.code


def test_cli_plan_requires_a_target_command(tmp_path, capsys):
    code = run_cli(
        [
            "instruments",
            "plan",
            "--output-trace",
            str(tmp_path / "run.trace"),
            "--output-dir",
            str(tmp_path / "a"),
        ]
    )
    assert code == 1
    assert "no program to profile" in capsys.readouterr().err


def test_cli_rejects_an_output_path_that_is_not_a_trace(tmp_path, capsys):
    code = run_cli(
        [
            "instruments",
            "plan",
            "--output-trace",
            str(tmp_path / "run.txt"),
            "--output-dir",
            str(tmp_path / "a"),
            "--",
            "/bin/echo",
            "hi",
        ]
    )
    assert code in (1, 3)
    combined = capsys.readouterr()
    assert ".trace" in (combined.out + combined.err)


def test_cli_import_of_a_missing_trace_exits_one(tmp_path, capsys):
    code = run_cli(
        [
            "instruments",
            "import",
            "--trace",
            str(tmp_path / "absent.trace"),
            "--output-dir",
            str(tmp_path / "a"),
        ]
    )
    assert code == 1
    assert "Failed to import the trace" in capsys.readouterr().err


def test_cli_exposes_the_four_documented_subcommands():
    from llmtracefx.optimizer.cli import build_parser

    parser = build_parser()
    instruments = next(
        action
        for action in parser._subparsers._group_actions[0].choices.items()
        if action[0] == "instruments"
    )[1]
    names = set(instruments._subparsers._group_actions[0].choices)
    assert names == {"capability", "plan", "record", "import"}
