"""Injectable subprocess boundaries for the Instruments/``xctrace`` layer.

Two separate boundaries, because the two use cases need different things:

``CommandRunner``
    Short, bounded, capture-everything probes such as ``xctrace version``
    and ``xctrace list templates``. Fully buffered, one shot.

``ProcessLauncher``
    A long-running ``xctrace record`` that writes a trace bundle, has to
    be stoppable by signal, and must not leave orphaned children behind.

Both are protocols so tests can drive the whole collector without Xcode
installed. The real implementations are the only place in this package
that touches :mod:`subprocess`, and neither one ever uses a shell: every
invocation is an argv list, so nothing a caller supplies can be
reinterpreted as shell syntax.
"""

from __future__ import annotations

import os
import signal
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Protocol


class InstrumentsProcessError(RuntimeError):
    """Raised when a probe or recording process cannot be started."""


@dataclass(frozen=True)
class CommandResult:
    """Captured result of one short, bounded command."""

    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def combined_output(self) -> str:
        """stdout and stderr joined, for pattern classification only.

        Callers classify known Apple error strings against this. It is
        never persisted verbatim into an experiment record.
        """
        return f"{self.stdout}\n{self.stderr}"


class CommandRunner(Protocol):
    """Runs one short command to completion and captures its output."""

    def run(self, argv: Sequence[str], *, timeout_seconds: float) -> CommandResult: ...


class ManagedProcess(Protocol):
    """A spawned recording process that can be waited on and signalled."""

    @property
    def pid(self) -> int: ...

    @property
    def returncode(self) -> int | None: ...

    @property
    def pgid(self) -> int:
        """Process group id, captured at spawn.

        Captured rather than looked up on demand: once the leader exits
        and is reaped, ``os.getpgid(pid)`` fails, but the group can still
        contain the program xctrace launched. Looking it up lazily would
        therefore lose the ability to signal exactly the survivors that
        matter most.
        """
        ...

    def wait(self, timeout_seconds: float) -> int:
        """Wait for exit. Raise ``subprocess.TimeoutExpired`` on timeout.

        Implementations must raise exactly that type, because it is what
        :mod:`subprocess` raises and what the recorder catches. Raising
        a different timeout type would escape the recorder's handler.
        """
        ...

    def signal_group(self, signal_number: int) -> None:
        """Signal the whole process group, not just the direct child.

        ``xctrace record --launch`` starts the target program itself, so
        signalling only the direct child can leave that target running.
        """
        ...

    def group_alive(self) -> bool:
        """Whether any process remains in the group.

        The leader exiting is not the same as the group being empty. A
        recording is only cleaned up when this returns ``False``.
        """
        ...


class ProcessLauncher(Protocol):
    """Spawns a recording process in its own process group."""

    def spawn(
        self,
        argv: Sequence[str],
        *,
        stdout: IO[bytes],
        stderr: IO[bytes],
        cwd: Path | None,
        env: dict[str, str] | None,
    ) -> ManagedProcess: ...


class SubprocessCommandRunner:
    """Real :mod:`subprocess` implementation of :class:`CommandRunner`."""

    def run(self, argv: Sequence[str], *, timeout_seconds: float) -> CommandResult:
        argv_tuple = tuple(argv)
        if not argv_tuple:
            raise InstrumentsProcessError("argv must not be empty")
        try:
            completed = subprocess.run(
                list(argv_tuple),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                argv=argv_tuple,
                returncode=-1,
                stdout=_as_text(exc.stdout),
                stderr=_as_text(exc.stderr),
                timed_out=True,
            )
        except OSError as exc:
            raise InstrumentsProcessError(
                f"could not execute {argv_tuple[0]!r}: {exc}"
            ) from exc
        return CommandResult(
            argv=argv_tuple,
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


class _PopenManagedProcess:
    """Adapts :class:`subprocess.Popen` to :class:`ManagedProcess`."""

    def __init__(self, popen: subprocess.Popen[bytes]) -> None:
        self._popen = popen
        # The child was spawned with start_new_session=True, so it leads
        # a new session and its process group id equals its pid. That is
        # captured here, while the leader is certainly still alive,
        # because os.getpgid stops working the moment it is reaped.
        try:
            self._pgid = os.getpgid(popen.pid)
        except (ProcessLookupError, PermissionError):
            # Already gone, or not visible. setsid guarantees pgid ==
            # pid, so that remains the correct group to signal.
            self._pgid = popen.pid

    @property
    def pid(self) -> int:
        return self._popen.pid

    @property
    def pgid(self) -> int:
        return self._pgid

    @property
    def returncode(self) -> int | None:
        return self._popen.returncode

    def wait(self, timeout_seconds: float) -> int:
        return self._popen.wait(timeout=timeout_seconds)

    def signal_group(self, signal_number: int) -> None:
        # Signals the group captured at spawn, which reaches the program
        # xctrace launched even after xctrace itself has exited.
        #
        # ProcessLookupError means the group is already gone, which is
        # the desired end state, so it is not an error here. It is
        # caught specifically rather than as a bare except.
        try:
            os.killpg(self._pgid, signal_number)
        except ProcessLookupError:
            return
        except PermissionError as exc:
            raise InstrumentsProcessError(
                f"not permitted to signal process group {self._pgid}: {exc}"
            ) from exc

    def group_alive(self) -> bool:
        """Whether the group still has members.

        Signal 0 performs the permission and existence checks without
        delivering anything. A PermissionError means the group exists
        but is not ours, which still counts as alive.
        """
        try:
            os.killpg(self._pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True


class SubprocessProcessLauncher:
    """Real :mod:`subprocess` implementation of :class:`ProcessLauncher`."""

    def spawn(
        self,
        argv: Sequence[str],
        *,
        stdout: IO[bytes],
        stderr: IO[bytes],
        cwd: Path | None,
        env: dict[str, str] | None,
    ) -> ManagedProcess:
        argv_tuple = tuple(argv)
        if not argv_tuple:
            raise InstrumentsProcessError("argv must not be empty")
        try:
            popen = subprocess.Popen(
                list(argv_tuple),
                stdout=stdout,
                stderr=stderr,
                stdin=subprocess.DEVNULL,
                cwd=cwd,
                env=env,
                shell=False,
                # Own session/process group, so a timeout can tear down
                # xctrace and the program it launched together.
                start_new_session=True,
            )
        except OSError as exc:
            raise InstrumentsProcessError(
                f"could not start {argv_tuple[0]!r}: {exc}"
            ) from exc
        return _PopenManagedProcess(popen)


#: Signal order used to stop a recording. SIGINT first because that is
#: the mechanism Instruments users and Chromium's profiling docs rely on
#: to end a recording with a valid trace bundle; escalation follows only
#: if the process ignores it.
STOP_SIGNAL_ESCALATION: tuple[int, ...] = (
    signal.SIGINT,
    signal.SIGTERM,
    signal.SIGKILL,
)
