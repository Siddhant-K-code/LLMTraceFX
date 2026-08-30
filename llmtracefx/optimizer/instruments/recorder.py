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

import json
import subprocess
import time
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


def _stop_process_group(process: ManagedProcess) -> int | None:
    """Signal the group, escalating until it exits."""
    for signal_number in STOP_SIGNAL_ESCALATION:
        try:
            process.signal_group(signal_number)
        except InstrumentsProcessError:
            # Not permitted to signal the group. Escalation cannot help,
            # so stop trying rather than looping on the same refusal.
            return process.returncode
        try:
            return process.wait(SIGNAL_ESCALATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            continue
    return process.returncode


def run_record(
    plan: RecordPlan,
    *,
    launcher: ProcessLauncher,
    artifacts_dir: Path,
) -> RecordResult:
    """Execute ``plan``, preserving artifacts whatever the outcome."""
    redacted_argv = plan.to_redacted_argv()
    started_at = utc_now_iso()

    try:
        resolved_trace = check_output_collision(plan.output_trace)
    except InstrumentsRecordError as exc:
        return RecordResult(
            status=RecordStatus.REFUSED,
            argv=redacted_argv,
            trace_path=plan.output_trace,
            message=str(exc),
            started_at=started_at,
            ended_at=utc_now_iso(),
        )

    artifacts = Path(artifacts_dir)
    artifacts.mkdir(parents=True, exist_ok=True)
    resolved_trace.parent.mkdir(parents=True, exist_ok=True)
    stdout_path = artifacts / "xctrace_record_stdout.txt"
    stderr_path = artifacts / "xctrace_record_stderr.txt"

    start = time.perf_counter()
    status = RecordStatus.FAILED
    returncode: int | None = None
    message = ""

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
            return RecordResult(
                status=RecordStatus.FAILED,
                argv=redacted_argv,
                trace_path=resolved_trace,
                message=f"could not start xctrace: {exc}",
                started_at=started_at,
                ended_at=utc_now_iso(),
                duration_seconds=time.perf_counter() - start,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )

        try:
            returncode = process.wait(plan.timeout_seconds)
        except subprocess.TimeoutExpired:
            status = RecordStatus.TIMED_OUT
            returncode = _stop_process_group(process)
            message = (
                f"recording exceeded its {plan.timeout_seconds:g}s host "
                f"deadline (--time-limit {plan.time_limit} plus "
                f"{plan.grace_seconds:g}s finalization grace) and the "
                "process group was stopped. Artifacts are preserved."
            )

    ended_at = utc_now_iso()
    duration = time.perf_counter() - start
    trace_exists = resolved_trace.exists()
    failure_capability: XctraceCapability | None = None

    if status is not RecordStatus.TIMED_OUT:
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
