"""Tests for the plan / record / import workflows and their CLI surfaces."""

from __future__ import annotations

import json
import os
import platform
import shutil
import tempfile
from pathlib import Path
from unittest import mock

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

from llmtracefx.optimizer import cli as cli_module
from llmtracefx.optimizer.cli import main
from llmtracefx.optimizer.instruments import workflow as workflow_module
from llmtracefx.optimizer.instruments.capability import (
    METAL_SYSTEM_TRACE_TEMPLATE,
    XctraceCapability,
    detect_xctrace_capability,
)
from llmtracefx.optimizer.instruments.commands import InstrumentsCommandError
from llmtracefx.optimizer.instruments.evidence import (
    TraceEvidenceInputs,
    build_instruments_evidence,
    unsupported_evidence,
)
from llmtracefx.optimizer.instruments.export import (
    FORBIDDEN_METRIC_NAMES,
    InstrumentsExportError,
    parse_exported_table,
)
from llmtracefx.optimizer.instruments.process import InstrumentsProcessError
from llmtracefx.optimizer.instruments.recorder import RecordStatus
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


#: Host identity the workflow tests pin themselves to. Chosen because
#: the Metal path only claims support on Apple Silicon, so this is the
#: identity under which the interesting branches exist at all.
PINNED_OS = "Darwin"
PINNED_ARCHITECTURE = "arm64"
PINNED_XCTRACE_PATH = "/usr/bin/xctrace"


@pytest.fixture(autouse=True)
def pinned_host_identity(monkeypatch):
    """Pin the host identity that the workflow layer detects against.

    ``plan_trace``/``record_trace``/``import_trace`` call
    ``detect_xctrace_capability`` with no platform overrides, so it reads
    the *real* host. That made these tests host-dependent in both
    directions: on a Linux runner every one of them short-circuited to
    ``unsupported_os`` before the injected fake was ever consulted, and
    on a macOS runner they silently depended on whether Xcode happened
    to be installed.

    Only host identity and path discovery are overridden. The real
    detector still runs, and still runs against whatever
    ``FakeCommandRunner`` the individual test supplied, so the Command
    Line Tools, license, template-missing and probe-failure branches are
    all still genuinely exercised rather than stubbed out. A test that
    wants to assert the platform gating itself passes its own
    ``os_name``/``architecture``, which this wrapper preserves.
    """
    real_detector = workflow_module.detect_xctrace_capability

    def pinned(*, runner, template=METAL_SYSTEM_TRACE_TEMPLATE, **overrides):
        # Filled when the key is absent *or* explicitly None, because
        # None is the real API's "resolve this from the host" sentinel.
        # A plain setdefault would let `os_name=None` reach the detector
        # unchanged and quietly reintroduce the host dependence this
        # fixture exists to remove, in the same shape as before: green
        # on macOS with Xcode, red on Linux.
        for key, value in (
            ("os_name", PINNED_OS),
            ("architecture", PINNED_ARCHITECTURE),
            ("path_resolver", lambda: PINNED_XCTRACE_PATH),
        ):
            if overrides.get(key) is None:
                overrides[key] = value
        return real_detector(runner=runner, template=template, **overrides)

    monkeypatch.setattr(workflow_module, "detect_xctrace_capability", pinned)
    # `instruments capability` resolves the detector from the CLI
    # module's own namespace, which is a separate binding. Pinning only
    # the workflow one would leave that subcommand reading the real host.
    monkeypatch.setattr(cli_module, "detect_xctrace_capability", pinned)


@pytest.fixture
def cli_launcher(monkeypatch, tmp_path):
    """Give the CLI a fake process launcher.

    `instruments record` spawns for real, so without this a CLI test
    would depend on the profiled command existing on the machine.
    """
    created = tmp_path / "run.trace"
    launcher = FakeLauncher(FakeProcess(returncode=0), creates_trace=created)
    monkeypatch.setattr(cli_module, "SubprocessProcessLauncher", lambda: launcher)
    return launcher


@pytest.fixture
def cli_command_runner(monkeypatch):
    """Give the CLI a fake xctrace instead of the real subprocess one.

    The CLI constructs its own ``SubprocessCommandRunner``, so pinning
    the host identity alone is not enough: on a machine without xctrace
    the probe would still fail for real. Replacing the constructor keeps
    the whole cli -> workflow -> capability -> export path under test
    while making it independent of what is installed.
    """
    runner = exporting_runner()
    monkeypatch.setattr(cli_module, "SubprocessCommandRunner", lambda: runner)
    return runner


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
    with pytest.raises(
        SchemaValidationError, match="not in instruments.parsed_schemas"
    ):
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
        available_schemas=("metal-gpu-intervals",),
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


def test_cli_rejects_an_output_path_that_is_not_a_trace(
    tmp_path, capsys, cli_command_runner
):
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


def test_cli_import_of_a_missing_trace_exits_one(tmp_path, capsys, cli_command_runner):
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


# --- Regressions found in independent review --------------------------


def test_failed_recording_evidence_does_not_claim_success(tmp_path):
    """Capability succeeded, but no trace was produced.

    Reusing the capability reason here would persist a note saying
    xctrace provides the template, on an artifact that represents a
    failed recording.
    """
    trace = tmp_path / "run.trace"
    out = tmp_path / "artifacts"
    record_trace(
        runner=exporting_runner(),
        launcher=FakeLauncher(FakeProcess(returncode=5)),
        command=("/bin/infer",),
        output_trace=trace,
        output_dir=out,
    )

    payload = json.loads((out / "instruments_evidence.json").read_text("utf-8"))
    assert payload["metrics"] == {}
    assert payload["trace_bundle_name"] is None
    notes = payload["notes"]
    assert notes.startswith("no trace was produced:")
    assert "provides the template" not in notes


def test_timed_out_recording_evidence_says_so(tmp_path):
    trace = tmp_path / "run.trace"
    out = tmp_path / "artifacts"
    record_trace(
        runner=exporting_runner(),
        launcher=FakeLauncher(FakeProcess(timeout_waits=1)),
        command=("/bin/infer",),
        output_trace=trace,
        output_dir=out,
    )
    notes = json.loads((out / "instruments_evidence.json").read_text("utf-8"))["notes"]
    assert "host deadline" in notes


def test_ambiguous_target_pid_yields_no_metric():
    """A pid under two labels must not get one of them attributed."""
    table = parse_exported_table(
        "<trace-query-result><node>"
        '<schema name="metal-gpu-intervals">'
        "<col><mnemonic>start</mnemonic>"
        "<engineering-type>start-time</engineering-type></col>"
        "<col><mnemonic>duration</mnemonic>"
        "<engineering-type>duration</engineering-type></col>"
        "<col><mnemonic>process</mnemonic>"
        "<engineering-type>process</engineering-type></col>"
        "</schema>"
        '<row><start-time id="1">10</start-time>'
        '<duration id="2">10</duration>'
        '<process id="3" fmt="probe (7)"><pid id="4">7</pid></process></row>'
        '<row><start-time id="5">20</start-time>'
        '<duration id="6">50</duration>'
        '<process id="7" fmt="other (7)"><pid id="8">7</pid></process></row>'
        "</node></trace-query-result>"
    )
    evidence = build_instruments_evidence(
        TraceEvidenceInputs(
            capability=capability(exporting_runner()),
            trace_bundle_name="run.trace",
            template="Metal System Trace",
            available_schemas=("metal-gpu-intervals",),
            table=table,
            target_pid=7,
        )
    )
    assert evidence.metrics == {}
    assert "is ambiguous" in (evidence.notes or "")


@pytest.mark.parametrize(
    "name",
    [
        "gpu_utilization",
        "metal_gpu_occupancy",
        "memory_bandwidth",
        "gpu_busy_percent",
        "metal_kernel_time",
        "gpu_power_watts",
        "gpu_memory_bytes",
    ],
)
def test_schema_structurally_rejects_overclaiming_metric_names(name):
    """The rule is enforced by validate(), not only by convention."""
    from llmtracefx.optimizer.schema import (
        InstrumentsEvidence,
        Measurement,
        MetricProvenance,
    )

    evidence = InstrumentsEvidence(
        available_schemas=("metal-gpu-intervals",),
        parsed_schemas=("metal-gpu-intervals",),
        metrics={
            name: Measurement(
                value=1.0, provenance=MetricProvenance.MEASURED_NATIVE, unit="x"
            )
        },
    )
    with pytest.raises(SchemaValidationError, match="forbidden marker"):
        base_record(instruments=evidence).validate()


def test_the_metrics_this_project_emits_survive_that_rule():
    """The guard must not be so broad that real metrics are rejected."""
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
    base_record(instruments=evidence).validate()


# --- Host independence of this test module ----------------------------
#
# These tests exist because every workflow test above once passed only
# because the machine running them happened to be an Apple Silicon Mac
# with Xcode installed. On the Linux CI runners the same tests failed,
# since plan/record/import detect against the real host and correctly
# short-circuit to unsupported_os there.


def test_platform_gating_still_rejects_linux_when_linux_is_declared():
    """The fixture pins identity; it does not disable the gate.

    Run inside this module, so the autouse fixture is active. Supplying
    an explicit non-Darwin identity must still produce ``unsupported_os``
    through the very same call path the workflows use. If the fixture
    had stubbed detection out, or the implementation had stopped gating,
    this would come back supported.
    """
    report = workflow_module.detect_xctrace_capability(
        runner=exporting_runner(), os_name="Linux"
    )
    assert report.capability is XctraceCapability.UNSUPPORTED_OS
    assert report.supported is False


def test_platform_gating_still_rejects_non_arm64_when_declared():
    report = workflow_module.detect_xctrace_capability(
        runner=exporting_runner(), architecture="x86_64"
    )
    assert report.capability is XctraceCapability.UNSUPPORTED_ARCHITECTURE


def test_workflows_do_not_read_the_real_host(monkeypatch):
    """Simulate the Linux CI runner and re-run a full workflow.

    ``platform.system``/``platform.machine`` are forced to the values a
    Linux runner reports, and ``shutil.which`` is made to deny xctrace.
    The workflow still succeeds, because this module supplies the
    platform identity explicitly rather than inheriting the host's.
    """
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        shutil,
        "which",
        lambda command, *args, **kwargs: None,
    )

    trace = Path(tempfile.mkdtemp()) / "run.trace"
    collection = record_trace(
        runner=exporting_runner(),
        launcher=FakeLauncher(FakeProcess(returncode=0), creates_trace=trace),
        command=("/bin/infer",),
        output_trace=trace,
        output_dir=trace.parent / "artifacts",
    )

    assert collection.succeeded is True
    assert collection.evidence.parsed_schemas == ("metal-gpu-intervals",)
    assert collection.evidence.metrics["metal_gpu_interval_count"].value == 3.0


def test_the_real_detector_would_have_failed_under_that_simulation(monkeypatch):
    """Confirms the simulation above is a real one.

    Without the pinned identity, the same simulated Linux process makes
    the production detector report ``unsupported_os``. That is what
    proves the previous test passes because identity is injected, and
    not because the simulation was ineffective.
    """
    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")

    report = detect_xctrace_capability(runner=exporting_runner())
    assert report.capability is XctraceCapability.UNSUPPORTED_OS


def test_cli_uses_the_injected_runner_rather_than_a_real_xctrace(
    tmp_path, cli_command_runner
):
    """Asserts the CLI observably went through the fake.

    An earlier version of this test asserted that the patched factory
    returned the fixture's object, which is true by construction one
    frame after the fixture sets it and stays true even when the CLI
    obtains its runner some other way. This drives `main` instead and
    checks the fake was actually consulted.
    """
    run_cli(
        [
            "instruments",
            "import",
            "--trace",
            str(tmp_path / "absent.trace"),
            "--output-dir",
            str(tmp_path / "a"),
        ]
    )
    assert cli_command_runner.calls, "the CLI never consulted the injected runner"
    assert all(argv[0] == PINNED_XCTRACE_PATH for argv in cli_command_runner.calls)


def test_cli_capability_subcommand_is_host_independent(capsys, cli_command_runner):
    """`instruments capability` resolves the detector from cli.py.

    That is a different binding from the workflow one, so it needs its
    own pinning. Without it this subcommand would report unsupported_os
    on a Linux runner and depend on a local Xcode on a macOS one.
    """
    code = run_cli(["instruments", "capability"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["capability"] == "supported"
    assert payload["os_name"] == PINNED_OS
    assert payload["architecture"] == PINNED_ARCHITECTURE
    assert METAL_SYSTEM_TRACE_TEMPLATE in payload["available_templates"]


def test_pinning_survives_an_explicit_none_identity():
    """None is the real API's "read the host" sentinel.

    Passing it must not defeat the pinning, or the host dependence comes
    straight back in its original shape.
    """
    report = workflow_module.detect_xctrace_capability(
        runner=exporting_runner(),
        os_name=None,
        architecture=None,
        path_resolver=None,
    )
    assert report.capability is XctraceCapability.SUPPORTED
    assert report.os_name == PINNED_OS


# --- Collisions are resolved before anything is written ---------------


def test_rerun_into_an_occupied_directory_touches_no_existing_byte(tmp_path):
    """A refused rerun must not leave one run's trace beside another's
    metadata.

    record_trace used to write capability_report.json immediately, and
    only then reach the trace collision check inside run_record. The
    trace was correctly refused, but the metadata beside it had already
    been replaced.
    """
    out = tmp_path / "artifacts"
    out.mkdir()
    trace = out / "run.trace"
    trace.mkdir()

    existing = {
        out / "capability_report.json": b'{"from": "the first run"}',
        out / "instruments_evidence.json": b'{"from": "the first run"}',
        out / "xctrace_record.json": b'{"from": "the first run"}',
        out / "trace_toc.xml": b"<trace-toc/>",
    }
    for path, payload in existing.items():
        path.write_bytes(payload)
    before = {path: path.read_bytes() for path in existing}

    launcher = FakeLauncher(FakeProcess(returncode=0), creates_trace=trace)
    collection = record_trace(
        runner=exporting_runner(),
        launcher=launcher,
        command=("/bin/infer",),
        output_trace=trace,
        output_dir=out,
    )

    assert collection.succeeded is False
    assert collection.wrote_artifacts is False
    assert launcher.spawned == []
    assert "never overwritten" in collection.message
    for path, payload in before.items():
        assert path.read_bytes() == payload, f"{path.name} was rewritten"


def test_refused_rerun_creates_no_new_files(tmp_path):
    out = tmp_path / "artifacts"
    trace = tmp_path / "run.trace"
    trace.mkdir()

    record_trace(
        runner=exporting_runner(),
        launcher=FakeLauncher(FakeProcess(returncode=0)),
        command=("/bin/infer",),
        output_trace=trace,
        output_dir=out,
    )

    assert not out.exists()


def test_a_clean_run_still_writes_every_artifact(tmp_path):
    """The ordering fix must not stop a normal run from writing."""
    out = tmp_path / "artifacts"
    trace = tmp_path / "run.trace"
    collection = record_trace(
        runner=exporting_runner(),
        launcher=FakeLauncher(FakeProcess(returncode=0), creates_trace=trace),
        command=("/bin/infer",),
        output_trace=trace,
        output_dir=out,
    )

    assert collection.succeeded is True
    assert collection.wrote_artifacts is True
    for name in (
        "capability_report.json",
        "instruments_evidence.json",
        "xctrace_record.json",
        "trace_toc.xml",
        "trace_toc.json",
    ):
        assert (out / name).exists(), name


# --- Metric allowlist -------------------------------------------------


def _evidence(**overrides):
    from llmtracefx.optimizer.schema import InstrumentsEvidence

    values = {
        "available_schemas": ("metal-gpu-intervals",),
        "parsed_schemas": ("metal-gpu-intervals",),
    }
    values.update(overrides)
    return InstrumentsEvidence(**values)


def _native(value=1.0, unit="intervals"):
    from llmtracefx.optimizer.schema import Measurement, MetricProvenance

    return Measurement(
        value=value, provenance=MetricProvenance.MEASURED_NATIVE, unit=unit
    )


@pytest.mark.parametrize(
    "name",
    [
        "metal_power_watts",
        "metal_energy_joules",
        "metal_memory_footprint",
        "gpu_temperature",
        "metal_gpu_interval_count_v2",
        "interval_count",
        "anything_at_all",
    ],
)
def test_metrics_outside_the_allowlist_are_rejected(name):
    """A denylist only blocks what somebody enumerated.

    `metal_power_watts` passed the substring denylist cleanly, because
    the entry was `gpu_power`. An allowlist has no such gap.
    """
    evidence = _evidence(metrics={name: _native()})
    with pytest.raises(SchemaValidationError, match="not a metric this project"):
        base_record(instruments=evidence).validate()


@pytest.mark.parametrize(
    "name,wrong_unit",
    [
        ("metal_gpu_interval_count", "ms"),
        ("metal_gpu_interval_duration_sum", "intervals"),
        ("metal_gpu_interval_wall_span", "ns"),
        ("metal_gpu_interval_count_all_processes", ""),
    ],
)
def test_allowlisted_metrics_must_carry_their_declared_unit(name, wrong_unit):
    evidence = _evidence(metrics={name: _native(unit=wrong_unit)})
    with pytest.raises(SchemaValidationError, match="must be in"):
        base_record(instruments=evidence).validate()


@pytest.mark.parametrize(
    "provenance_name", ["MEASURED_WALL_CLOCK", "DERIVED", "ESTIMATED"]
)
def test_allowlisted_metrics_must_carry_native_provenance(provenance_name):
    from llmtracefx.optimizer.schema import Measurement, MetricProvenance

    evidence = _evidence(
        metrics={
            "metal_gpu_interval_count": Measurement(
                value=1.0,
                provenance=getattr(MetricProvenance, provenance_name),
                unit="intervals",
            )
        }
    )
    with pytest.raises(SchemaValidationError, match="must have provenance"):
        base_record(instruments=evidence).validate()


def test_a_metric_requires_its_own_source_table_to_have_been_parsed():
    evidence = _evidence(
        parsed_schemas=(), metrics={"metal_gpu_interval_count": _native()}
    )
    with pytest.raises(
        SchemaValidationError, match="not in instruments.parsed_schemas"
    ):
        base_record(instruments=evidence).validate()


def test_a_phantom_parsed_schema_cannot_unlock_a_metric():
    """parsed_schemas must be a subset of what the trace advertised."""
    evidence = _evidence(
        available_schemas=("time-profile",),
        parsed_schemas=("metal-gpu-intervals",),
        metrics={"metal_gpu_interval_count": _native()},
    )
    with pytest.raises(SchemaValidationError, match="never advertised"):
        base_record(instruments=evidence).validate()


def test_parsing_cannot_be_claimed_when_the_trace_advertised_nothing():
    evidence = _evidence(available_schemas=())
    with pytest.raises(SchemaValidationError, match="advertised nothing"):
        base_record(instruments=evidence).validate()


def test_parsed_schemas_must_name_a_table_this_project_can_parse():
    evidence = _evidence(
        available_schemas=("os-signpost",), parsed_schemas=("os-signpost",)
    )
    with pytest.raises(SchemaValidationError, match="no parser for"):
        base_record(instruments=evidence).validate()


def test_the_four_real_metrics_pass_every_rule(tmp_path):
    """The guard must not be so strict that a real run is rejected."""
    trace = tmp_path / "run.trace"
    collection = record_trace(
        runner=exporting_runner(),
        launcher=FakeLauncher(FakeProcess(returncode=0), creates_trace=trace),
        command=("/bin/infer",),
        output_trace=trace,
        output_dir=tmp_path / "artifacts",
    )
    from llmtracefx.optimizer.schema import INSTRUMENT_METRIC_SPECS

    assert set(collection.evidence.metrics) == set(INSTRUMENT_METRIC_SPECS)
    base_record(instruments=collection.evidence).validate()


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_metrics_are_rejected(bad):
    """NaN and infinity both pass a `>= 0` check silently.

    A NaN interval count is not a measurement, and would propagate into
    every downstream comparison as though it were one.
    """
    from llmtracefx.optimizer.schema import Measurement, MetricProvenance

    evidence = _evidence(
        metrics={
            "metal_gpu_interval_count": Measurement(
                value=bad,
                provenance=MetricProvenance.MEASURED_NATIVE,
                unit="intervals",
            )
        }
    )
    with pytest.raises(SchemaValidationError, match="finite number|must be >= 0"):
        base_record(instruments=evidence).validate()


# --- Failures must not leave two runs' artifacts mixed ----------------


def test_a_failed_import_leaves_every_previous_artifact_intact(tmp_path):
    """xctrace writes the TOC and the table itself, mid-flight.

    Exporting straight into the output directory meant a failure between
    the two left this run's trace_toc.xml sitting beside a previous
    run's evidence, table and toc json.
    """
    trace = tmp_path / "run.trace"
    trace.mkdir()
    out = tmp_path / "artifacts"
    out.mkdir()

    seeded = {
        out / "trace_toc.xml": b"PREVIOUS toc xml",
        out / "trace_toc.json": b"PREVIOUS toc json",
        out / "trace_table.xml": b"PREVIOUS table",
        out / "instruments_evidence.json": b"PREVIOUS evidence",
    }
    for path, payload in seeded.items():
        path.write_bytes(payload)

    # The TOC export succeeds; the table export fails.
    runner = exporting_runner(**{"metal-gpu-intervals": fail((), returncode=1)})
    with pytest.raises(InstrumentsExportError, match="failed"):
        import_trace(runner=runner, trace_path=trace, output_dir=out)

    for path, payload in seeded.items():
        assert path.read_bytes() == payload, f"{path.name} was overwritten"


def test_a_successful_import_replaces_every_artifact_together(tmp_path):
    trace = tmp_path / "run.trace"
    trace.mkdir()
    out = tmp_path / "artifacts"
    out.mkdir()
    for name in ("trace_toc.xml", "trace_toc.json", "instruments_evidence.json"):
        (out / name).write_bytes(b"PREVIOUS")

    import_trace(runner=exporting_runner(), trace_path=trace, output_dir=out)

    for name in ("trace_toc.xml", "trace_toc.json", "instruments_evidence.json"):
        assert (out / name).read_bytes() != b"PREVIOUS", name
    assert not (out / ".import-staging").exists()


def test_import_staging_is_cleaned_up_on_success(tmp_path):
    trace = tmp_path / "run.trace"
    trace.mkdir()
    out = tmp_path / "artifacts"
    import_trace(runner=exporting_runner(), trace_path=trace, output_dir=out)
    assert sorted(p.name for p in out.iterdir()) == [
        "instruments_evidence.json",
        "trace_table.xml",
        "trace_toc.json",
        "trace_toc.xml",
    ]


def test_cli_does_not_announce_evidence_it_did_not_write(
    tmp_path, capsys, cli_command_runner
):
    """A refusal must not point the reader at an earlier run's metrics.

    The filesystem was already correct, but the CLI still printed
    "Instruments evidence written to ...", naming a file that either did
    not exist or belonged to a different run.
    """
    out = tmp_path / "artifacts"
    out.mkdir()
    stale = out / "instruments_evidence.json"
    stale.write_bytes(b"EVIDENCE FROM AN EARLIER RUN")
    trace = out / "run.trace"
    trace.mkdir()

    code = run_cli(
        [
            "instruments",
            "record",
            "--output-trace",
            str(trace),
            "--output-dir",
            str(out),
            "--",
            "/bin/infer",
        ]
    )
    captured = capsys.readouterr()

    assert code == 1
    assert "evidence written to" not in captured.out.casefold()
    assert "Nothing was written" in captured.err
    assert stale.read_bytes() == b"EVIDENCE FROM AN EARLIER RUN"


def test_cli_still_announces_evidence_on_a_real_run(
    tmp_path, capsys, cli_command_runner
):
    out = tmp_path / "artifacts"
    code = run_cli(
        [
            "instruments",
            "plan",
            "--output-trace",
            str(tmp_path / "run.trace"),
            "--output-dir",
            str(out),
            "--",
            "/bin/infer",
        ]
    )
    assert code == 0
    assert "instruments_evidence.json" in capsys.readouterr().out


def test_a_concurrent_record_is_refused_before_touching_artifacts(tmp_path):
    """The reservation is held across the artifact writes, not just the
    recording.

    A second run must not overwrite the first run's capability report
    and evidence on its way to discovering the conflict.
    """
    from llmtracefx.optimizer.instruments.recorder import reserve_trace_path

    out = tmp_path / "artifacts"
    out.mkdir()
    seeded = out / "capability_report.json"
    seeded.write_bytes(b"FIRST RUN")
    trace = tmp_path / "run.trace"

    with reserve_trace_path(trace):
        launcher = FakeLauncher(FakeProcess(returncode=0))
        collection = record_trace(
            runner=exporting_runner(),
            launcher=launcher,
            command=("/bin/infer",),
            output_trace=trace,
            output_dir=out,
        )

    assert collection.succeeded is False
    assert collection.wrote_artifacts is False
    assert "already reserved" in collection.message
    assert launcher.spawned == []
    assert seeded.read_bytes() == b"FIRST RUN"


def test_a_normal_record_still_works_with_the_reservation(tmp_path):
    trace = tmp_path / "run.trace"
    collection = record_trace(
        runner=exporting_runner(),
        launcher=FakeLauncher(FakeProcess(returncode=0), creates_trace=trace),
        command=("/bin/infer",),
        output_trace=trace,
        output_dir=tmp_path / "artifacts",
    )
    assert collection.succeeded is True
    # The marker must not survive the run.
    assert not list(tmp_path.glob(".*reservation"))


def test_a_failed_import_leaves_no_staging_directory_behind(tmp_path):
    """Scratch state must not outlive the call that created it.

    Without guaranteed cleanup the partial staging set sat in the user's
    output directory until some later run happened to clear it.
    """
    from llmtracefx.optimizer.instruments.workflow import STAGING_DIR_NAME

    trace = tmp_path / "run.trace"
    trace.mkdir()
    out = tmp_path / "artifacts"

    runner = exporting_runner(**{"metal-gpu-intervals": fail((), returncode=1)})
    with pytest.raises(InstrumentsExportError):
        import_trace(runner=runner, trace_path=trace, output_dir=out)

    assert not (out / STAGING_DIR_NAME).exists()
    assert list(out.iterdir()) == []


def test_a_successful_import_leaves_no_staging_directory_behind(tmp_path):
    from llmtracefx.optimizer.instruments.workflow import STAGING_DIR_NAME

    trace = tmp_path / "run.trace"
    trace.mkdir()
    out = tmp_path / "artifacts"
    import_trace(runner=exporting_runner(), trace_path=trace, output_dir=out)
    assert not (out / STAGING_DIR_NAME).exists()


def test_a_failed_export_after_a_good_recording_replaces_stale_evidence(tmp_path):
    """`record` writes its metadata before exporting.

    An export failure used to propagate out of record_trace, leaving
    this run's capability report and record metadata sitting beside a
    previous run's evidence and GPU metrics. --output-trace and
    --output-dir are independent flags, so a fixed artifacts directory
    with a per-run trace name is the natural invocation and the trace
    collision check does not catch it.
    """
    out = tmp_path / "artifacts"

    first = tmp_path / "run1.trace"
    record_trace(
        runner=exporting_runner(),
        launcher=FakeLauncher(FakeProcess(returncode=0), creates_trace=first),
        command=("/bin/infer",),
        output_trace=first,
        output_dir=out,
    )
    stale = json.loads((out / "instruments_evidence.json").read_text("utf-8"))
    assert stale["trace_bundle_name"] == "run1.trace"
    assert stale["metrics"]

    second = tmp_path / "run2.trace"
    collection = record_trace(
        runner=exporting_runner(**{"toc": fail((), returncode=1)}),
        launcher=FakeLauncher(FakeProcess(returncode=0), creates_trace=second),
        command=("/bin/infer",),
        output_trace=second,
        output_dir=out,
    )

    # The recording completed, but the run measured nothing, so it is
    # not a success and must not exit 0.
    assert collection.succeeded is False
    assert collection.export_failed is True
    assert collection.record.status is RecordStatus.COMPLETED

    fresh = json.loads((out / "instruments_evidence.json").read_text("utf-8"))
    # The bundle exists and is re-exportable, so its name is kept.
    assert fresh["trace_bundle_name"] == "run2.trace"
    assert fresh["metrics"] == {}
    assert "could not be exported" in fresh["notes"]
    assert "instruments import" in fresh["notes"]
    assert "no trace was produced" not in fresh["notes"]
    record_meta = json.loads((out / "xctrace_record.json").read_text("utf-8"))
    assert record_meta["trace_name"] == "run2.trace"


def test_evidence_is_promoted_last_so_a_tear_leaves_it_stale(tmp_path):
    """Promotion order matters if the loop is ever interrupted."""
    from llmtracefx.optimizer.instruments.workflow import STAGING_DIR_NAME

    trace = tmp_path / "run.trace"
    trace.mkdir()
    out = tmp_path / "artifacts"
    out.mkdir()
    (out / STAGING_DIR_NAME).mkdir()
    staging = out / STAGING_DIR_NAME
    for name in ("trace_toc.xml", "instruments_evidence.json", "trace_table.xml"):
        (staging / name).write_text(name, encoding="utf-8")

    promoted: list[str] = []
    real_replace = os.replace

    def recording_replace(src, dst):
        promoted.append(Path(dst).name)
        real_replace(src, dst)

    with mock.patch.object(workflow_module.os, "replace", recording_replace):
        workflow_module._promote_staged(staging, out)

    assert promoted[-1] == "instruments_evidence.json"


def test_cli_exits_nonzero_when_the_export_failed(
    tmp_path, capsys, cli_command_runner, cli_launcher
):
    """A run that measured nothing must not report success.

    `record.status` is COMPLETED, which is true of the recording, but
    letting that stand in for the run would let a CI job record a green,
    "measured" run whose evidence has zero GPU metrics.
    """
    cli_command_runner.exports["toc"] = fail((), returncode=1)
    out = tmp_path / "artifacts"

    code = run_cli(
        [
            "instruments",
            "record",
            "--output-trace",
            str(tmp_path / "run.trace"),
            "--output-dir",
            str(out),
            "--",
            "/bin/infer",
        ]
    )
    captured = capsys.readouterr()

    assert code == 1
    assert "Recorded, but the export failed" in captured.err
    assert "instruments import" in captured.err
    assert "Recording did not complete" not in captured.err


def test_cli_exits_zero_on_a_fully_successful_record(
    tmp_path, capsys, cli_command_runner, cli_launcher
):
    """The nonzero path must not swallow real successes."""
    out = tmp_path / "artifacts"
    code = run_cli(
        [
            "instruments",
            "record",
            "--output-trace",
            str(tmp_path / "run.trace"),
            "--output-dir",
            str(out),
            "--",
            "/bin/infer",
        ]
    )
    assert code == 0
    assert "evidence written to" in capsys.readouterr().out.casefold()


@pytest.mark.parametrize(
    "error",
    [
        InstrumentsProcessError("could not execute xctrace"),
        OSError("disk full"),
    ],
)
def test_any_export_failure_replaces_stale_evidence(tmp_path, error):
    """Not only InstrumentsExportError reaches this path.

    xctrace can become unexecutable between the capability probe and the
    export, and the staging or evidence writes can fail. Each used to
    propagate past the handler and leave the previous run's GPU metrics
    attributed to this run's command.
    """
    out = tmp_path / "artifacts"
    first = tmp_path / "run1.trace"
    record_trace(
        runner=exporting_runner(),
        launcher=FakeLauncher(FakeProcess(returncode=0), creates_trace=first),
        command=("/bin/infer",),
        output_trace=first,
        output_dir=out,
    )
    assert json.loads((out / "instruments_evidence.json").read_text("utf-8"))["metrics"]

    class ExplodingRunner(FakeCommandRunner):
        def run(self, argv, *, timeout_seconds):
            if len(argv) > 1 and argv[1] == "export":
                raise error
            return super().run(argv, timeout_seconds=timeout_seconds)

    second = tmp_path / "run2.trace"
    collection = record_trace(
        runner=ExplodingRunner(),
        launcher=FakeLauncher(FakeProcess(returncode=0), creates_trace=second),
        command=("/bin/infer",),
        output_trace=second,
        output_dir=out,
    )

    assert collection.export_failed is True
    assert collection.succeeded is False
    evidence = json.loads((out / "instruments_evidence.json").read_text("utf-8"))
    assert evidence["metrics"] == {}
    assert evidence["trace_bundle_name"] == "run2.trace"


def test_the_promoted_toc_carries_no_identity(tmp_path):
    """The file written to disk, not just the parsed object."""
    trace = tmp_path / "run.trace"
    trace.mkdir()
    out = tmp_path / "artifacts"
    import_trace(runner=exporting_runner(), trace_path=trace, output_dir=out)

    promoted = (out / "trace_toc.xml").read_text("utf-8")
    assert "Example Mac" not in promoted
    assert "00000000-0000-0000-0000-000000000000" not in promoted
    assert "arguments=" not in promoted
    assert "Metal System Trace" in promoted


def test_a_failed_run_clears_a_previous_run_exported_tables(tmp_path):
    """Replacing only the evidence still leaves a stale GPU table.

    A directory holding run 2's metadata beside run 1's exported
    intervals invites exactly the misattribution this project exists to
    prevent.
    """
    out = tmp_path / "artifacts"
    first = tmp_path / "run1.trace"
    record_trace(
        runner=exporting_runner(),
        launcher=FakeLauncher(FakeProcess(returncode=0), creates_trace=first),
        command=("/bin/infer",),
        output_trace=first,
        output_dir=out,
    )
    for name in ("trace_toc.xml", "trace_toc.json", "trace_table.xml"):
        assert (out / name).exists(), name

    second = tmp_path / "run2.trace"
    record_trace(
        runner=exporting_runner(**{"toc": fail((), returncode=1)}),
        launcher=FakeLauncher(FakeProcess(returncode=0), creates_trace=second),
        command=("/bin/infer",),
        output_trace=second,
        output_dir=out,
    )

    for name in ("trace_toc.xml", "trace_toc.json", "trace_table.xml"):
        assert not (out / name).exists(), f"{name} survived from the earlier run"


def test_an_invalid_request_writes_nothing(tmp_path):
    """Validation must precede the first byte written.

    capability_report.json used to land before the plan was built, so a
    bad time limit left an artifact describing a run that never started.
    """
    out = tmp_path / "artifacts"
    with pytest.raises(InstrumentsCommandError, match="time-limit"):
        record_trace(
            runner=exporting_runner(),
            launcher=FakeLauncher(FakeProcess(returncode=0)),
            command=("/bin/infer",),
            output_trace=tmp_path / "run.trace",
            output_dir=out,
            time_limit="not-a-duration",
        )
    assert not out.exists() or list(out.iterdir()) == []


def test_two_runs_cannot_share_an_artifact_directory(tmp_path):
    """Different traces, same --output-dir, still race on the metadata."""
    from llmtracefx.optimizer.instruments.recorder import reserve_output_dir

    out = tmp_path / "artifacts"
    with reserve_output_dir(out):
        launcher = FakeLauncher(FakeProcess(returncode=0))
        collection = record_trace(
            runner=exporting_runner(),
            launcher=launcher,
            command=("/bin/infer",),
            output_trace=tmp_path / "other.trace",
            output_dir=out,
        )

    assert collection.succeeded is False
    assert collection.wrote_artifacts is False
    assert "already reserved the artifact directory" in collection.message
    assert launcher.spawned == []


def test_a_leaked_descendant_makes_the_run_unsuccessful(tmp_path):
    """A run that left processes running did not fully succeed.

    The trace is valid and the recording completed, both of which stay
    true in the metadata, but the exit code must not say success.
    """
    trace = tmp_path / "run.trace"
    collection = record_trace(
        runner=exporting_runner(),
        launcher=FakeLauncher(
            FakeProcess(returncode=0, group_dies_on=None), creates_trace=trace
        ),
        command=("/bin/infer",),
        output_trace=trace,
        output_dir=tmp_path / "artifacts",
        # The group never dies, so this bounds the full escalation.
        stop_grace_seconds=0.05,
    )

    assert collection.record.status is RecordStatus.COMPLETED
    assert collection.record.cleanup_failed is True
    assert collection.succeeded is False
    assert "could NOT be stopped" in collection.record.message


def test_a_clean_run_is_not_marked_as_leaking(tmp_path):
    trace = tmp_path / "run.trace"
    collection = record_trace(
        runner=exporting_runner(),
        launcher=FakeLauncher(FakeProcess(returncode=0), creates_trace=trace),
        command=("/bin/infer",),
        output_trace=trace,
        output_dir=tmp_path / "artifacts",
    )
    assert collection.record.cleanup_failed is False
    assert collection.succeeded is True


def test_a_bad_run_number_in_the_toc_does_not_mix_runs(tmp_path):
    """Trace-derived input is validated, and that raises a command error.

    It escaped the handler, leaving this run's metadata beside the
    previous run's live GPU metrics.
    """
    out = tmp_path / "artifacts"
    first = tmp_path / "run1.trace"
    record_trace(
        runner=exporting_runner(),
        launcher=FakeLauncher(FakeProcess(returncode=0), creates_trace=first),
        command=("/bin/infer",),
        output_trace=first,
        output_dir=out,
    )
    assert json.loads((out / "instruments_evidence.json").read_text("utf-8"))["metrics"]

    bad_toc = read_fixture(TOC).replace('<run number="1">', '<run number="0">')
    second = tmp_path / "run2.trace"
    collection = record_trace(
        runner=exporting_runner(**{"toc": ok((), bad_toc)}),
        launcher=FakeLauncher(FakeProcess(returncode=0), creates_trace=second),
        command=("/bin/infer",),
        output_trace=second,
        output_dir=out,
    )

    assert collection.export_failed is True
    evidence = json.loads((out / "instruments_evidence.json").read_text("utf-8"))
    assert evidence["metrics"] == {}
    assert evidence["trace_bundle_name"] == "run2.trace"


def test_derived_artifacts_never_name_another_process(tmp_path):
    """Only the raw export lists other processes, and it is documented.

    Metal System Trace records every process on the system, so the raw
    table names them. The artifacts a reader actually consumes must not:
    they carry this run's pid and counts only.
    """
    trace = tmp_path / "run.trace"
    trace.mkdir()
    out = tmp_path / "artifacts"
    import_trace(runner=exporting_runner(), trace_path=trace, output_dir=out)

    for name in ("instruments_evidence.json", "trace_toc.json", "trace_toc.xml"):
        assert "WindowServer" not in (out / name).read_text("utf-8"), name
    # The raw export does, by nature. Pinned so the boundary is explicit.
    assert "WindowServer" in (out / "trace_table.xml").read_text("utf-8")


def test_an_ambiguous_pid_note_does_not_name_other_processes(tmp_path):
    """The refusal must stay actionable without leaking a name.

    An operator needs the conflicting labels on stderr; a persisted
    artifact must not carry another process's name, which is what the
    README promises of instruments_evidence.json.
    """
    rows = "".join(
        f'<row><start-time id="{i * 4 + 1}">{i * 10}</start-time>'
        f'<duration id="{i * 4 + 2}">5</duration>'
        f'<process id="{i * 4 + 3}" fmt="{name} (4242)">'
        f'<pid id="{i * 4 + 4}">4242</pid></process></row>'
        for i, name in enumerate(["Signal", "probe"])
    )
    table = (
        '<trace-query-result><node><schema name="metal-gpu-intervals">'
        "<col><mnemonic>start</mnemonic>"
        "<engineering-type>start-time</engineering-type></col>"
        "<col><mnemonic>duration</mnemonic>"
        "<engineering-type>duration</engineering-type></col>"
        "<col><mnemonic>process</mnemonic>"
        "<engineering-type>process</engineering-type></col>"
        f"</schema>{rows}</node></trace-query-result>"
    )
    trace = tmp_path / "run.trace"
    trace.mkdir()
    out = tmp_path / "artifacts"
    import_trace(
        runner=exporting_runner(**{"metal-gpu-intervals": ok((), table)}),
        trace_path=trace,
        output_dir=out,
    )

    notes = json.loads((out / "instruments_evidence.json").read_text("utf-8"))["notes"]
    assert "is ambiguous" in notes
    assert "2 different process labels" in notes
    assert "Signal" not in notes


def test_import_clears_stale_exports_when_xctrace_is_unavailable(tmp_path):
    """The one evidence-writing path that was missed.

    `instruments import` is reachable directly, so an unavailable
    xctrace wrote fresh no-metrics evidence on top of a previous run's
    exported GPU table and left it in place.
    """
    out = tmp_path / "artifacts"
    out.mkdir()
    for name in ("trace_toc.xml", "trace_toc.json", "trace_table.xml"):
        (out / name).write_text("PREVIOUS RUN", encoding="utf-8")

    trace = tmp_path / "run.trace"
    trace.mkdir()
    runner = FakeCommandRunner(
        version=fail((), returncode=1, stderr=COMMAND_LINE_TOOLS_STDERR)
    )
    collection = import_trace(runner=runner, trace_path=trace, output_dir=out)

    assert collection.capability.supported is False
    for name in ("trace_toc.xml", "trace_toc.json", "trace_table.xml"):
        assert not (out / name).exists(), f"{name} survived from the earlier run"
    assert (out / "instruments_evidence.json").exists()
