"""Run ``xctrace record`` safely and preserve what it produced.

Safety properties, each of which has a test:

Never overwrite
    The output path is resolved (symlinks and ``..`` collapsed) and
    refused if anything already exists there. xctrace itself also
    refuses, requiring ``--append-run``, but this pre-flight fails
    earlier, names the colliding path, and never lets ``--append-run``
    be passed implicitly.

Bounded
    Every recording has a host-side wall-clock deadline that is strictly
    greater than the requested ``--time-limit``, leaving room for
    xctrace to finalize the bundle.

Cleaned up
    The child is spawned in its own process group. ``xctrace record
    --launch`` starts the profiled program itself, so a timeout signals
    the whole group, escalating SIGINT then SIGTERM then SIGKILL. SIGINT
    comes first because it is how a recording is normally stopped with
    the bundle left valid.

Evidence preserving
    stdout, stderr and run metadata are always written, including on
    failure and timeout, and a partially written ``.trace`` bundle is
    left in place rather than cleaned away.
"""

from __future__ import annotations

import contextlib
import json
import os
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..collectors._shared import atomic_write_text
from ..schema import utc_now_iso
from .capability import XctraceCapability, classify_xctrace_failure
from .commands import RecordPlan
from .process import (
    STOP_SIGNAL_ESCALATION,
    InstrumentsProcessError,
    ManagedProcess,
    ProcessLauncher,
)

RECORD_METADATA_SCHEMA_VERSION = "1"

#: Seconds to wait after each stop signal before escalating.
SIGNAL_ESCALATION_GRACE_SECONDS = 30.0

#: How often to re-check whether the process group has emptied.
GROUP_POLL_INTERVAL_SECONDS = 0.05

#: Bytes of captured stderr read back purely to classify a failure.
#: The text is never persisted into a record; only the resulting
#: capability label is.
_CLASSIFY_TAIL_BYTES = 64 * 1024


class RecordStatus(str, Enum):
    """How a recording attempt ended."""

    COMPLETED = "completed"
    """xctrace exited 0 and the trace bundle exists."""

    FAILED = "failed"
    """xctrace exited non-zero, or exited 0 without leaving a bundle."""

    TIMED_OUT = "timed_out"
    """The host deadline elapsed and the process group was torn down."""

    REFUSED = "refused"
    """A pre-flight check failed. Nothing was executed."""

    @property
    def ran(self) -> bool:
        return self is not RecordStatus.REFUSED


class InstrumentsRecordError(RuntimeError):
    """Raised when a recording cannot be attempted at all."""


@dataclass(frozen=True)
class RecordResult:
    """Outcome of one recording attempt, plus where its artifacts are."""

    status: RecordStatus
    argv: tuple[str, ...]
    """The redacted argv. Safe to persist and print."""
    trace_path: Path
    message: str
    started_at: str
    ended_at: str | None = None
    duration_seconds: float | None = None
    returncode: int | None = None
    stdout_path: Path | None = None
    stderr_path: Path | None = None
    trace_exists: bool = False
    failure_capability: XctraceCapability | None = None
    """Set when a failure's output identified a specific cause."""

    @property
    def succeeded(self) -> bool:
        return self.status is RecordStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RECORD_METADATA_SCHEMA_VERSION,
            "status": self.status.value,
            "argv": list(self.argv),
            "trace_name": self.trace_path.name,
            "trace_exists": self.trace_exists,
            "message": self.message,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_seconds": self.duration_seconds,
            "returncode": self.returncode,
            "stdout_name": (
                None if self.stdout_path is None else self.stdout_path.name
            ),
            "stderr_name": (
                None if self.stderr_path is None else self.stderr_path.name
            ),
            "failure_capability": (
                None
                if self.failure_capability is None
                else self.failure_capability.value
            ),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def write_json(self, path: str | Path) -> None:
        atomic_write_text(Path(path), self.to_json() + "\n")


def check_output_collision(output_trace: Path) -> Path:
    """Resolve the trace path and refuse if anything is already there.

    ``strict=False`` resolution collapses symlinks and ``..`` for the
    part of the path that exists, so ``a/../b.trace`` and ``b.trace``
    cannot be treated as different destinations. On the case-insensitive
    filesystem macOS ships by default, ``Probe.trace`` and
    ``probe.trace`` also collide here, which is the intended behavior.

    This is a check, so on its own it is subject to a check-then-use
    race. Callers that go on to record must hold a
    :func:`reserve_trace_path` reservation, which closes it.
    """
    resolved = output_trace.expanduser().resolve()
    if resolved.exists():
        kind = "directory" if resolved.is_dir() else "file"
        raise InstrumentsRecordError(
            f"refusing to record over an existing {kind}: {resolved}. "
            "Traces are raw evidence and are never overwritten. Choose a "
            "different --output path, or move the existing bundle aside."
        )
    return resolved


def reservation_path_for(resolved_trace: Path) -> Path:
    """Where the exclusive claim for ``resolved_trace`` lives."""
    return resolved_trace.with_name(f".{resolved_trace.name}.reservation")


@contextmanager
def reserve_trace_path(output_trace: Path) -> Iterator[Path]:
    """Claim a trace output path exclusively for the whole run.

    Checking that a path is free and then recording into it is a
    check-then-use race: two runs can both pass the check, and a
    concurrent writer can create the bundle in between. The claim here
    is a single atomic ``O_CREAT | O_EXCL`` file creation, which the
    filesystem guarantees exactly one caller wins, and it is held until
    the recording finishes rather than released immediately.

    The reservation marker sits beside the bundle as a dotfile and is
    always removed on exit, including on failure. A leftover marker from
    a killed process is reported as a conflict naming its path, which is
    recoverable by deleting it, rather than being silently ignored.
    """
    resolved = check_output_collision(output_trace)
    # The parent has to exist to hold the marker, and it has to exist
    # for the bundle regardless. Creating it is not an artifact.
    resolved.parent.mkdir(parents=True, exist_ok=True)
    marker = reservation_path_for(resolved)

    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise InstrumentsRecordError(
            f"another recording already reserved {resolved.name} "
            f"({marker}). Wait for it to finish, or delete that marker if "
            "no recording is running."
        ) from exc
    except OSError as exc:
        raise InstrumentsRecordError(f"could not reserve {resolved}: {exc}") from exc

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()} started_at={utc_now_iso()}\n")
        # Re-check after the claim: a writer could have created the
        # bundle between the check above and the claim. Winning the
        # marker means nobody else can now, so this is the last window.
        if resolved.exists():
            kind = "directory" if resolved.is_dir() else "file"
            raise InstrumentsRecordError(
                f"refusing to record over an existing {kind}: {resolved}. "
                "It appeared while this run was starting."
            )
        yield resolved
    finally:
        with contextlib.suppress(FileNotFoundError):
            marker.unlink()


def _classify_failure_tail(stderr_path: Path) -> XctraceCapability | None:
    """Classify a failure from captured stderr without persisting it."""
    try:
        with stderr_path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - _CLASSIFY_TAIL_BYTES))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    return classify_xctrace_failure(tail)


def _wait_for_group_exit(process: ManagedProcess, deadline_seconds: float) -> bool:
    """Poll until the process group is empty or the deadline passes."""
    deadline = time.monotonic() + deadline_seconds
    while True:
        if not process.group_alive():
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(GROUP_POLL_INTERVAL_SECONDS)


def _stop_process_group(
    process: ManagedProcess, *, grace_seconds: float
) -> tuple[int | None, bool]:
    """Escalate signals until the whole group is gone.

    Returns the leader's exit status and whether the group actually
    emptied. That second value is not cosmetic: a group that survives
    SIGKILL, or one this process is not permitted to signal, must not be
    described in a persisted artifact as having been stopped.

    The leader exiting is deliberately *not* the stop condition.
    ``xctrace record --launch`` starts the profiled program itself, and
    that program can outlive xctrace: it may ignore SIGINT, or simply be
    slower to notice it. Returning as soon as the leader was reaped
    would leave the target and any of its descendants running, holding
    the pipes this recorder opened, with nothing left that knows how to
    reach them.
    """
    returncode = process.returncode
    for signal_number in STOP_SIGNAL_ESCALATION:
        try:
            process.signal_group(signal_number)
        except InstrumentsProcessError:
            # Not permitted to signal the group. Escalation cannot help,
            # so stop trying rather than looping on the same refusal.
            return returncode, not process.group_alive()

        # Reap the leader if it has not been reaped yet, so its exit
        # status is available, then wait on the group as a whole.
        try:
            returncode = process.wait(grace_seconds)
        except subprocess.TimeoutExpired:
            pass

        if _wait_for_group_exit(process, grace_seconds):
            return returncode, True
    return returncode, False


def run_record(
    plan: RecordPlan,
    *,
    launcher: ProcessLauncher,
    artifacts_dir: Path,
    reserved_trace: Path | None = None,
) -> RecordResult:
    """Execute ``plan``, preserving artifacts whatever the outcome.

    ``reserved_trace`` is the resolved path from an outer
    :func:`reserve_trace_path`. Callers that already hold the
    reservation pass it so the claim spans their artifact writes as well
    as the recording; callers that do not get one acquired here for the
    duration of the recording.
    """
    redacted_argv = plan.to_redacted_argv()
    started_at = utc_now_iso()

    if reserved_trace is not None:
        return _run_record_reserved(
            plan,
            launcher=launcher,
            artifacts_dir=artifacts_dir,
            resolved_trace=reserved_trace,
            redacted_argv=redacted_argv,
            started_at=started_at,
        )

    try:
        with reserve_trace_path(plan.output_trace) as resolved_trace:
            return _run_record_reserved(
                plan,
                launcher=launcher,
                artifacts_dir=artifacts_dir,
                resolved_trace=resolved_trace,
                redacted_argv=redacted_argv,
                started_at=started_at,
            )
    except InstrumentsRecordError as exc:
        return RecordResult(
            status=RecordStatus.REFUSED,
            argv=redacted_argv,
            trace_path=plan.output_trace,
            message=str(exc),
            started_at=started_at,
            ended_at=utc_now_iso(),
        )


def _run_record_reserved(
    plan: RecordPlan,
    *,
    launcher: ProcessLauncher,
    artifacts_dir: Path,
    resolved_trace: Path,
    redacted_argv: tuple[str, ...],
    started_at: str,
) -> RecordResult:
    """Record into a path whose reservation is already held."""

    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    resolved_trace.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = artifacts / "xctrace_record_stdout.txt"
    stderr_path = artifacts / "xctrace_record_stderr.txt"

    start = time.perf_counter()
    status = RecordStatus.FAILED
    returncode: int | None = None
    message = ""
    spawn_failed = False
    survivors_stopped = False
    survivors_cleared_ok = True

    with (
        stdout_path.open("wb") as stdout_handle,
        stderr_path.open("wb") as stderr_handle,
    ):
        try:
            process = launcher.spawn(
                plan.to_argv(),
                stdout=stdout_handle,
                stderr=stderr_handle,
                cwd=None,
                env=None,
            )
        except InstrumentsProcessError as exc:
            # Falls through to the shared persistence below rather than
            # returning here. Returning early skipped the metadata
            # write, which both lost the record of this failure and let
            # a previous run's xctrace_record.json survive next to this
            # run's freshly truncated stdout and stderr.
            spawn_failed = True
            message = f"could not start xctrace: {exc}"
        else:
            try:
                returncode = process.wait(plan.timeout_seconds)
            except subprocess.TimeoutExpired:
                status = RecordStatus.TIMED_OUT
                returncode, group_empty = _stop_process_group(
                    process, grace_seconds=plan.stop_grace_seconds
                )
                outcome = (
                    "the process group was stopped"
                    if group_empty
                    else (
                        "the process group could NOT be stopped and may "
                        f"still be running (pgid {process.pgid})"
                    )
                )
                message = (
                    f"recording exceeded its {plan.timeout_seconds:g}s host "
                    f"deadline (--time-limit {plan.time_limit} plus "
                    f"{plan.grace_seconds:g}s finalization grace) and "
                    f"{outcome}. Artifacts are preserved."
                )
            except BaseException:
                # Anything other than a timeout (an unexpected error, or
                # a KeyboardInterrupt while waiting) still propagates,
                # but the recording must not be left running detached.
                # It is in its own session, so nothing else would ever
                # reap it.
                _stop_process_group(process, grace_seconds=plan.stop_grace_seconds)
                raise
            else:
                # The leader exited on its own, which is not the same as
                # the group being empty. xctrace can fail early while the
                # program it launched keeps running, and that program is
                # in a session this process started, so nothing else
                # would clean it up.
                if process.group_alive():
                    survivors_stopped = True
                    _stop_process_group(process, grace_seconds=plan.stop_grace_seconds)

    ended_at = utc_now_iso()
    duration = time.perf_counter() - start
    trace_exists = resolved_trace.exists()
    failure_capability: XctraceCapability | None = None

    if spawn_failed:
        # Message already set. Nothing was executed, so there is no exit
        # status to classify and no bundle to look for.
        status = RecordStatus.FAILED
    elif status is not RecordStatus.TIMED_OUT:
        if returncode == 0 and trace_exists:
            status = RecordStatus.COMPLETED
            message = f"recorded {resolved_trace.name}"
        elif returncode == 0:
            status = RecordStatus.FAILED
            message = (
                "xctrace exited 0 but no trace bundle exists at "
                f"{resolved_trace}. Nothing was recorded."
            )
        else:
            status = RecordStatus.FAILED
            failure_capability = _classify_failure_tail(stderr_path)
            message = f"xctrace exited {returncode}"
            if failure_capability is not None:
                message += f" ({failure_capability.value})"
            message += f". See {stderr_path.name} for the tool's own output."
    elif returncode is not None and returncode != 0:
        failure_capability = _classify_failure_tail(stderr_path)

    if survivors_stopped:
        message += " xctrace exited while the program it launched was still " + (
            "running; that process group was stopped."
            if survivors_cleared_ok
            else "running, and that process group could NOT be stopped."
        )

    result = RecordResult(
        status=status,
        argv=redacted_argv,
        trace_path=resolved_trace,
        message=message,
        started_at=started_at,
        ended_at=ended_at,
        duration_seconds=duration,
        returncode=returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        trace_exists=trace_exists,
        failure_capability=failure_capability,
    )
    result.write_json(artifacts / "xctrace_record.json")
    return result
