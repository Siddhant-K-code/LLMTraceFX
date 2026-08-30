"""Tests for recording safety: collisions, timeouts, cleanup, artifacts."""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path

import pytest
from _instruments_fakes import COMMAND_LINE_TOOLS_STDERR, FakeLauncher, FakeProcess

from llmtracefx.optimizer.instruments.commands import LaunchTarget, RecordPlan
from llmtracefx.optimizer.instruments.process import InstrumentsProcessError
from llmtracefx.optimizer.instruments.recorder import (
    InstrumentsRecordError,
    RecordStatus,
    check_output_collision,
    run_record,
)


def make_plan(trace: Path, **overrides) -> RecordPlan:
    values = {
        "xctrace_path": "/usr/bin/xctrace",
        "template": "Metal System Trace",
        "output_trace": trace,
        "target": LaunchTarget(argv=("/bin/infer", "--tokens", "8")),
        "time_limit": "5s",
        "grace_seconds": 10.0,
    }
    values.update(overrides)
    return RecordPlan(**values)


# --- Collision refusal ------------------------------------------------


def test_existing_bundle_directory_is_refused(tmp_path):
    trace = tmp_path / "run.trace"
    trace.mkdir()
    with pytest.raises(InstrumentsRecordError, match="existing directory"):
        check_output_collision(trace)


def test_existing_file_is_refused(tmp_path):
    trace = tmp_path / "run.trace"
    trace.write_text("stale", encoding="utf-8")
    with pytest.raises(InstrumentsRecordError, match="existing file"):
        check_output_collision(trace)


def test_path_aliases_resolve_to_the_same_destination(tmp_path):
    """`a/../run.trace` must not be treated as a fresh destination."""
    trace = tmp_path / "run.trace"
    trace.mkdir()
    (tmp_path / "sub").mkdir()
    with pytest.raises(InstrumentsRecordError, match="existing directory"):
        check_output_collision(tmp_path / "sub" / ".." / "run.trace")


def test_symlinked_path_resolves_to_its_target(tmp_path):
    real = tmp_path / "real.trace"
    real.mkdir()
    link = tmp_path / "link.trace"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises(InstrumentsRecordError, match="existing directory"):
        check_output_collision(link)


def test_fresh_path_is_accepted_and_resolved(tmp_path):
    resolved = check_output_collision(tmp_path / "new.trace")
    assert resolved.name == "new.trace"
    assert resolved.is_absolute()


def test_collision_refusal_executes_nothing(tmp_path):
    trace = tmp_path / "run.trace"
    trace.mkdir()
    launcher = FakeLauncher(FakeProcess())
    artifacts = tmp_path / "artifacts"

    result = run_record(make_plan(trace), launcher=launcher, artifacts_dir=artifacts)

    assert result.status is RecordStatus.REFUSED
    assert launcher.spawned == []
    assert not artifacts.exists()
    assert "never overwritten" in result.message


# --- Success ----------------------------------------------------------


def test_successful_recording_writes_metadata_and_artifacts(tmp_path):
    trace = tmp_path / "run.trace"
    artifacts = tmp_path / "artifacts"
    launcher = FakeLauncher(
        FakeProcess(returncode=0, stdout_text="Recording completed."),
        creates_trace=trace,
    )

    result = run_record(make_plan(trace), launcher=launcher, artifacts_dir=artifacts)

    assert result.status is RecordStatus.COMPLETED
    assert result.succeeded is True
    assert result.returncode == 0
    assert result.trace_exists is True
    assert (artifacts / "xctrace_record_stdout.txt").exists()
    assert (artifacts / "xctrace_record_stderr.txt").exists()

    metadata = json.loads(
        (artifacts / "xctrace_record.json").read_text(encoding="utf-8")
    )
    assert metadata["status"] == "completed"
    assert metadata["trace_name"] == "run.trace"


def test_metadata_uses_basenames_for_derived_path_fields(tmp_path):
    """Derived fields are basenames; only argv keeps full paths.

    The argv has to record the exact command or the run is not
    reproducible, and every path in it was supplied by the caller. The
    derived name fields carry no directory, and the evidence that goes
    into a shared ``ExperimentRecord`` carries a bundle basename only
    (see ``test_instruments_workflow``).
    """
    trace = tmp_path / "run.trace"
    artifacts = tmp_path / "artifacts"
    launcher = FakeLauncher(FakeProcess(returncode=0), creates_trace=trace)

    run_record(make_plan(trace), launcher=launcher, artifacts_dir=artifacts)

    metadata = json.loads(
        (artifacts / "xctrace_record.json").read_text(encoding="utf-8")
    )
    assert metadata["trace_name"] == "run.trace"
    assert metadata["stdout_name"] == "xctrace_record_stdout.txt"
    assert metadata["stderr_name"] == "xctrace_record_stderr.txt"
    for key in ("trace_name", "stdout_name", "stderr_name"):
        assert "/" not in metadata[key]


def test_stored_argv_is_the_redacted_one(tmp_path):
    from llmtracefx.optimizer.instruments.commands import EnvironmentAssignment

    trace = tmp_path / "run.trace"
    plan = make_plan(
        trace,
        environment=(EnvironmentAssignment(name="HF_TOKEN", value="hf_leak"),),
    )
    launcher = FakeLauncher(FakeProcess(returncode=0), creates_trace=trace)

    result = run_record(plan, launcher=launcher, artifacts_dir=tmp_path / "artifacts")

    assert "hf_leak" not in " ".join(result.argv)
    # The real invocation still carries the true value.
    assert "HF_TOKEN=hf_leak" in launcher.spawned[0]


# --- Failure ----------------------------------------------------------


def test_nonzero_exit_is_failure_and_artifacts_are_kept(tmp_path):
    trace = tmp_path / "run.trace"
    artifacts = tmp_path / "artifacts"
    launcher = FakeLauncher(
        FakeProcess(returncode=3, stderr_text="something went wrong")
    )

    result = run_record(make_plan(trace), launcher=launcher, artifacts_dir=artifacts)

    assert result.status is RecordStatus.FAILED
    assert result.returncode == 3
    assert (artifacts / "xctrace_record_stderr.txt").read_text(
        encoding="utf-8"
    ) == "something went wrong"
    assert (artifacts / "xctrace_record.json").exists()


def test_failure_output_is_classified_but_not_persisted(tmp_path):
    trace = tmp_path / "run.trace"
    artifacts = tmp_path / "artifacts"
    launcher = FakeLauncher(
        FakeProcess(returncode=1, stderr_text=COMMAND_LINE_TOOLS_STDERR)
    )

    result = run_record(make_plan(trace), launcher=launcher, artifacts_dir=artifacts)

    assert result.failure_capability is not None
    assert result.failure_capability.value == "command_line_tools_only"
    # Classified, but the raw tool text stays out of the metadata file.
    metadata = (artifacts / "xctrace_record.json").read_text(encoding="utf-8")
    assert "CommandLineTools" not in metadata


def test_exit_zero_without_a_bundle_is_a_failure(tmp_path):
    """A clean exit that produced nothing must not read as success."""
    trace = tmp_path / "run.trace"
    launcher = FakeLauncher(FakeProcess(returncode=0), creates_trace=None)

    result = run_record(
        make_plan(trace), launcher=launcher, artifacts_dir=tmp_path / "artifacts"
    )

    assert result.status is RecordStatus.FAILED
    assert result.trace_exists is False
    assert "exited 0 but no trace bundle" in result.message


def test_spawn_failure_is_reported_without_raising(tmp_path):
    trace = tmp_path / "run.trace"
    launcher = FakeLauncher(
        FakeProcess(), spawn_error=InstrumentsProcessError("no such file")
    )

    result = run_record(
        make_plan(trace), launcher=launcher, artifacts_dir=tmp_path / "artifacts"
    )

    assert result.status is RecordStatus.FAILED
    assert "could not start xctrace" in result.message


# --- Timeout and process group cleanup --------------------------------


def test_timeout_stops_the_process_group_starting_with_sigint(tmp_path):
    trace = tmp_path / "run.trace"
    process = FakeProcess(returncode=0, timeout_waits=1)
    launcher = FakeLauncher(process, creates_trace=trace)

    result = run_record(
        make_plan(trace), launcher=launcher, artifacts_dir=tmp_path / "artifacts"
    )

    assert result.status is RecordStatus.TIMED_OUT
    # SIGINT first: it is how a recording is stopped with a valid bundle.
    assert process.signals[0] == signal.SIGINT
    assert len(process.signals) == 1


def test_timeout_escalates_when_sigint_is_ignored(tmp_path):
    trace = tmp_path / "run.trace"
    # One timeout for the main wait, then one per ignored stop signal.
    process = FakeProcess(returncode=0, timeout_waits=3)
    launcher = FakeLauncher(process)

    run_record(
        make_plan(trace), launcher=launcher, artifacts_dir=tmp_path / "artifacts"
    )

    assert process.signals == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]


def test_timeout_uses_the_plan_deadline_not_the_time_limit(tmp_path):
    trace = tmp_path / "run.trace"
    process = FakeProcess(returncode=0)
    launcher = FakeLauncher(process, creates_trace=trace)
    plan = make_plan(trace, time_limit="5s", grace_seconds=10.0)

    run_record(plan, launcher=launcher, artifacts_dir=tmp_path / "artifacts")

    assert process.wait_calls[0] == 15.0


def test_timeout_message_explains_the_budget(tmp_path):
    trace = tmp_path / "run.trace"
    launcher = FakeLauncher(FakeProcess(timeout_waits=1))

    result = run_record(
        make_plan(trace), launcher=launcher, artifacts_dir=tmp_path / "artifacts"
    )

    assert "host deadline" in result.message
    assert "Artifacts are preserved" in result.message


def test_timeout_preserves_artifacts(tmp_path):
    trace = tmp_path / "run.trace"
    artifacts = tmp_path / "artifacts"
    launcher = FakeLauncher(FakeProcess(timeout_waits=1, stderr_text="partial"))

    run_record(make_plan(trace), launcher=launcher, artifacts_dir=artifacts)

    assert (artifacts / "xctrace_record.json").exists()
    assert (artifacts / "xctrace_record_stderr.txt").read_text(
        encoding="utf-8"
    ) == "partial"


def test_unsignalable_group_does_not_loop_forever(tmp_path):
    trace = tmp_path / "run.trace"
    process = FakeProcess(
        timeout_waits=5, signal_error=InstrumentsProcessError("not permitted")
    )
    launcher = FakeLauncher(process)

    result = run_record(
        make_plan(trace), launcher=launcher, artifacts_dir=tmp_path / "artifacts"
    )

    assert result.status is RecordStatus.TIMED_OUT
    assert process.signals == [signal.SIGINT]


def test_non_timeout_wait_errors_propagate(tmp_path):
    """Only timeouts are handled; other failures must not be swallowed."""

    class Exploding(FakeProcess):
        def wait(self, timeout_seconds: float) -> int:
            self.wait_calls.append(timeout_seconds)
            raise subprocess.SubprocessError("unexpected")

    trace = tmp_path / "run.trace"
    process = Exploding()
    launcher = FakeLauncher(process)

    with pytest.raises(subprocess.SubprocessError, match="unexpected"):
        run_record(
            make_plan(trace),
            launcher=launcher,
            artifacts_dir=tmp_path / "artifacts",
        )

    # The recording lives in its own session, so nothing else would reap
    # it. It must be stopped even on an unexpected error.
    assert process.signals[0] == signal.SIGINT


def test_keyboard_interrupt_while_waiting_still_stops_the_recording(tmp_path):
    """KeyboardInterrupt is a BaseException, not an Exception."""

    class Interrupted(FakeProcess):
        def wait(self, timeout_seconds: float) -> int:
            self.wait_calls.append(timeout_seconds)
            raise KeyboardInterrupt

    trace = tmp_path / "run.trace"
    process = Interrupted()
    launcher = FakeLauncher(process)

    with pytest.raises(KeyboardInterrupt):
        run_record(
            make_plan(trace),
            launcher=launcher,
            artifacts_dir=tmp_path / "artifacts",
        )

    assert process.signals[0] == signal.SIGINT
