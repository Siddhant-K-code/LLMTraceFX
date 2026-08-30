"""Tests for recording safety: collisions, timeouts, cleanup, artifacts."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from unittest import mock

import pytest
from _instruments_fakes import COMMAND_LINE_TOOLS_STDERR, FakeLauncher, FakeProcess

from llmtracefx.optimizer.instruments import recorder as recorder_module
from llmtracefx.optimizer.instruments.commands import LaunchTarget, RecordPlan
from llmtracefx.optimizer.instruments.process import (
    InstrumentsProcessError,
    SubprocessProcessLauncher,
)
from llmtracefx.optimizer.instruments.recorder import (
    InstrumentsRecordError,
    RecordStatus,
    check_output_collision,
    reservation_path_for,
    reserve_trace_path,
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
        "stop_grace_seconds": 0.25,
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


def test_timeout_escalates_when_the_group_survives_sigint(tmp_path):
    """The leader exiting is not the stop condition.

    xctrace can exit on SIGINT while the program it launched keeps
    running in the same group. Escalation must continue until the group
    is empty, not stop the moment the leader is reaped.
    """
    trace = tmp_path / "run.trace"
    process = FakeProcess(returncode=0, timeout_waits=1, group_dies_on=signal.SIGKILL)
    launcher = FakeLauncher(process)

    run_record(
        make_plan(trace), launcher=launcher, artifacts_dir=tmp_path / "artifacts"
    )

    assert process.signals == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert process.group_alive() is False


def test_teardown_stops_once_the_group_is_actually_empty(tmp_path):
    trace = tmp_path / "run.trace"
    process = FakeProcess(returncode=0, timeout_waits=1, group_dies_on=signal.SIGTERM)
    launcher = FakeLauncher(process)

    run_record(
        make_plan(trace), launcher=launcher, artifacts_dir=tmp_path / "artifacts"
    )

    assert process.signals == [signal.SIGINT, signal.SIGTERM]


def test_teardown_is_bounded_when_the_group_never_dies(tmp_path):
    """A group that ignores everything must not hang the recorder."""
    trace = tmp_path / "run.trace"
    process = FakeProcess(returncode=0, timeout_waits=1, group_dies_on=None)
    launcher = FakeLauncher(process)

    result = run_record(
        make_plan(trace), launcher=launcher, artifacts_dir=tmp_path / "artifacts"
    )

    assert process.signals == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert result.status is RecordStatus.TIMED_OUT


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
        timeout_waits=5,
        signal_error=InstrumentsProcessError("not permitted"),
        group_dies_on=None,
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


# --- Real process group teardown --------------------------------------
#
# The fakes above model the group, but the property that matters is a
# real one: after run_record returns, nothing it started may still be
# running. These use actual processes.


#: A stand-in for xctrace: it accepts the same argv shape (a `record`
#: subcommand, flags, then `-- program args`), starts the program in its
#: own process group exactly as xctrace does, and exits on SIGINT while
#: leaving that program running. No shell is involved.
FAKE_XCTRACE_SOURCE = """\
import signal
import subprocess
import sys
import time

argv = sys.argv[1:]
target = argv[argv.index("--") + 1 :]
subprocess.Popen([sys.executable, *target])

# Exit cleanly on SIGINT, leaving the launched program behind. This is
# the exact shape that used to end teardown early.
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
time.sleep(600)
"""

#: A profiled program that refuses SIGINT and SIGTERM, like a wedged
#: inference process still holding the GPU.
STUBBORN_TARGET_SOURCE = """\
import os
import signal
import sys
import time

signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    handle.write(str(os.getpid()))
time.sleep(600)
"""


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _await_pid_exit(pid: int, timeout_seconds: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    return not _pid_alive(pid)


def _await_file(path: Path, timeout_seconds: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if path.exists() and path.read_text(encoding="utf-8").strip():
            return True
        time.sleep(0.05)
    return False


def test_descendant_is_killed_when_the_leader_exits_on_sigint(tmp_path):
    """The regression: leader exits on SIGINT, descendant survives it.

    Teardown used to return as soon as the leader was reaped, leaving
    the program xctrace launched running in a detached session with
    nothing left that knew how to reach it. Escalation now continues
    until the process group is actually empty.
    """
    fake_xctrace = tmp_path / "fake_xctrace.py"
    fake_xctrace.write_text(
        f"#!{sys.executable}\n" + FAKE_XCTRACE_SOURCE, encoding="utf-8"
    )
    fake_xctrace.chmod(0o755)

    target = tmp_path / "stubborn.py"
    target.write_text(STUBBORN_TARGET_SOURCE, encoding="utf-8")
    pid_file = tmp_path / "child.pid"

    plan = make_plan(
        tmp_path / "run.trace",
        xctrace_path=str(fake_xctrace),
        target=LaunchTarget(argv=(str(target), str(pid_file))),
        time_limit="1s",
        grace_seconds=1.0,
        stop_grace_seconds=3.0,
    )

    child_pid: int | None = None
    try:
        result = run_record(
            plan,
            launcher=SubprocessProcessLauncher(),
            artifacts_dir=tmp_path / "artifacts",
        )

        assert result.status is RecordStatus.TIMED_OUT
        assert _await_file(pid_file), "the profiled program never started"
        child_pid = int(pid_file.read_text(encoding="utf-8"))
        assert _await_pid_exit(child_pid), (
            f"the profiled program (pid {child_pid}) survived teardown; "
            "the process group was not cleaned up"
        )
    finally:
        if child_pid is not None and _pid_alive(child_pid):
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(child_pid, signal.SIGKILL)


def test_pgid_is_captured_at_spawn_and_survives_leader_exit(tmp_path):
    """`os.getpgid` stops working once the leader is reaped.

    Capturing the group at spawn is what keeps the survivors reachable.
    """
    script = tmp_path / "quick.py"
    script.write_text("import sys; sys.exit(0)\n", encoding="utf-8")

    launcher = SubprocessProcessLauncher()
    with (tmp_path / "o.txt").open("wb") as out, (tmp_path / "e.txt").open("wb") as err:
        process = launcher.spawn(
            (sys.executable, str(script)),
            stdout=out,
            stderr=err,
            cwd=None,
            env=None,
        )
        pgid = process.pgid
        assert pgid == process.pid  # start_new_session makes them equal
        process.wait(30.0)

    # The leader is reaped, so a lazy lookup would now fail outright.
    with pytest.raises(ProcessLookupError):
        os.getpgid(process.pid)
    # The captured group id is still usable, and reports the group empty.
    assert process.pgid == pgid
    assert process.group_alive() is False


# --- Spawn failure metadata -------------------------------------------


def test_spawn_failure_writes_metadata_like_every_other_failure(tmp_path):
    """A failure to start is still a failure worth recording.

    This path used to return before the metadata write, so the one
    outcome with no exit status and no tool output was also the one with
    no record of having happened.
    """
    trace = tmp_path / "run.trace"
    artifacts = tmp_path / "artifacts"
    launcher = FakeLauncher(
        FakeProcess(), spawn_error=InstrumentsProcessError("no such file")
    )

    result = run_record(make_plan(trace), launcher=launcher, artifacts_dir=artifacts)

    assert result.status is RecordStatus.FAILED
    metadata_path = artifacts / "xctrace_record.json"
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert "could not start xctrace" in metadata["message"]
    assert metadata["returncode"] is None
    assert metadata["trace_exists"] is False
    assert (artifacts / "xctrace_record_stdout.txt").exists()
    assert (artifacts / "xctrace_record_stderr.txt").exists()


def test_spawn_failure_replaces_stale_metadata_from_a_previous_run(tmp_path):
    """No mixing a previous run's metadata with this run's output.

    The stdout and stderr files are truncated on every attempt, so a
    surviving xctrace_record.json from an earlier success would have sat
    next to empty logs describing a run that did not happen.
    """
    trace = tmp_path / "run.trace"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True)
    stale = artifacts / "xctrace_record.json"
    stale.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "status": "completed",
                "message": "recorded an earlier run",
                "returncode": 0,
            }
        ),
        encoding="utf-8",
    )

    run_record(
        make_plan(trace),
        launcher=FakeLauncher(
            FakeProcess(), spawn_error=InstrumentsProcessError("boom")
        ),
        artifacts_dir=artifacts,
    )

    metadata = json.loads(stale.read_text(encoding="utf-8"))
    assert metadata["status"] == "failed"
    assert "recorded an earlier run" not in metadata["message"]
    assert metadata["returncode"] is None


def test_survivors_are_stopped_even_when_xctrace_exits_normally(tmp_path):
    """Adjacent to the timeout case, and just as leaky.

    xctrace can fail early while the program it launched keeps running.
    The leader exiting is not the timeout path, so nothing used to look
    at the group at all on that route.
    """
    trace = tmp_path / "run.trace"
    process = FakeProcess(returncode=1, group_dies_on=signal.SIGTERM)
    launcher = FakeLauncher(process)

    result = run_record(
        make_plan(trace), launcher=launcher, artifacts_dir=tmp_path / "artifacts"
    )

    assert result.status is RecordStatus.FAILED
    assert process.signals == [signal.SIGINT, signal.SIGTERM]
    assert "that process group was stopped" in result.message


def test_a_clean_exit_with_an_empty_group_signals_nothing(tmp_path):
    trace = tmp_path / "run.trace"
    process = FakeProcess(returncode=0)
    process._group_alive = False
    launcher = FakeLauncher(process, creates_trace=trace)

    result = run_record(
        make_plan(trace), launcher=launcher, artifacts_dir=tmp_path / "artifacts"
    )

    assert result.status is RecordStatus.COMPLETED
    assert process.signals == []
    assert "process group was stopped" not in result.message


def test_message_does_not_claim_a_stop_that_did_not_happen(tmp_path):
    """A group that survives SIGKILL must not be reported as stopped.

    The truth was already computed and then discarded, so the persisted
    artifact asserted a cleanup that demonstrably had not occurred.
    """
    trace = tmp_path / "run.trace"
    artifacts = tmp_path / "artifacts"
    process = FakeProcess(returncode=0, timeout_waits=1, group_dies_on=None)

    result = run_record(
        make_plan(trace),
        launcher=FakeLauncher(process),
        artifacts_dir=artifacts,
    )

    assert result.status is RecordStatus.TIMED_OUT
    assert "could NOT be stopped" in result.message
    assert str(process.pgid) in result.message
    persisted = json.loads(
        (artifacts / "xctrace_record.json").read_text(encoding="utf-8")
    )
    assert "could NOT be stopped" in persisted["message"]


def test_message_reports_a_real_stop_as_a_stop(tmp_path):
    trace = tmp_path / "run.trace"
    result = run_record(
        make_plan(trace),
        launcher=FakeLauncher(FakeProcess(returncode=0, timeout_waits=1)),
        artifacts_dir=tmp_path / "artifacts",
    )
    assert "the process group was stopped" in result.message
    assert "could NOT" not in result.message


def test_unsignalable_group_is_not_reported_as_stopped(tmp_path):
    trace = tmp_path / "run.trace"
    process = FakeProcess(
        timeout_waits=5,
        signal_error=InstrumentsProcessError("not permitted"),
        group_dies_on=None,
    )
    result = run_record(
        make_plan(trace),
        launcher=FakeLauncher(process),
        artifacts_dir=tmp_path / "artifacts",
    )
    assert "could NOT be stopped" in result.message


# --- Atomic reservation -----------------------------------------------
#
# Checking that a path is free and then recording into it is a
# check-then-use race. These pin the atomic claim that closes it.


def test_reservation_is_exclusive(tmp_path):
    trace = tmp_path / "run.trace"
    with reserve_trace_path(trace):
        with pytest.raises(InstrumentsRecordError, match="already reserved"):
            with reserve_trace_path(trace):
                pass


def test_reservation_is_released_on_success(tmp_path):
    trace = tmp_path / "run.trace"
    with reserve_trace_path(trace) as resolved:
        marker = reservation_path_for(resolved)
        assert marker.exists()
    assert not marker.exists()


def test_reservation_is_released_when_the_body_raises(tmp_path):
    trace = tmp_path / "run.trace"
    marker = reservation_path_for(trace.resolve())
    with pytest.raises(ValueError):
        with reserve_trace_path(trace):
            raise ValueError("boom")
    assert not marker.exists()


def test_reservation_refuses_a_bundle_that_appears_mid_start(tmp_path):
    """The window between checking and claiming is itself checked.

    A writer can create the bundle after the initial check passes.
    Winning the marker is what makes the recheck conclusive.
    """
    trace = tmp_path / "run.trace"
    real_check = recorder_module.check_output_collision

    def check_then_create(path):
        resolved = real_check(path)
        resolved.mkdir(parents=True)  # a concurrent writer wins the race
        return resolved

    with mock.patch.object(
        recorder_module, "check_output_collision", check_then_create
    ):
        with pytest.raises(InstrumentsRecordError, match="appeared while"):
            with reserve_trace_path(trace):
                pass
    assert not reservation_path_for(trace.resolve()).exists()


def test_only_one_of_many_concurrent_reservations_wins(tmp_path):
    """The claim is a single atomic O_CREAT|O_EXCL, so exactly one wins."""
    trace = tmp_path / "run.trace"
    barrier = threading.Barrier(8)
    won: list[int] = []
    refused: list[int] = []
    lock = threading.Lock()

    def attempt(index: int) -> None:
        barrier.wait(timeout=30)
        try:
            with reserve_trace_path(trace):
                with lock:
                    won.append(index)
                time.sleep(0.05)
        except InstrumentsRecordError:
            with lock:
                refused.append(index)

    threads = [threading.Thread(target=attempt, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert len(won) == 1, f"expected exactly one winner, got {won}"
    assert len(refused) == 7
    assert not reservation_path_for(trace.resolve()).exists()


def test_a_concurrent_run_is_refused_and_writes_nothing(tmp_path):
    """A second recorder must not touch the first one's artifacts."""
    trace = tmp_path / "run.trace"
    artifacts = tmp_path / "artifacts"

    with reserve_trace_path(trace):
        launcher = FakeLauncher(FakeProcess(returncode=0))
        result = run_record(
            make_plan(trace), launcher=launcher, artifacts_dir=artifacts
        )

    assert result.status is RecordStatus.REFUSED
    assert "already reserved" in result.message
    assert launcher.spawned == []
    assert not artifacts.exists()


def test_a_held_reservation_is_not_reacquired(tmp_path):
    """The caller's claim spans its artifact writes and the recording."""
    trace = tmp_path / "run.trace"
    with reserve_trace_path(trace) as resolved:
        result = run_record(
            make_plan(trace),
            launcher=FakeLauncher(FakeProcess(returncode=0), creates_trace=trace),
            artifacts_dir=tmp_path / "artifacts",
            reserved_trace=resolved,
        )
    assert result.status is RecordStatus.COMPLETED
