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

    @property
    def pid(self) -> int:
        return self._popen.pid

    @property
    def returncode(self) -> int | None:
        return self._popen.returncode

    def wait(self, timeout_seconds: float) -> int:
        return self._popen.wait(timeout=timeout_seconds)

    def signal_group(self, signal_number: int) -> None:
        # The child was spawned with start_new_session=True, so it leads
        # its own process group whose id equals its pid. Signalling the
        # group reaches the program xctrace launched as well.
        #
        # ProcessLookupError means the group is already gone, which is
        # the desired end state, so it is not an error here. It is
        # caught specifically rather than as a bare except.
        try:
            os.killpg(os.getpgid(self._popen.pid), signal_number)
        except ProcessLookupError:
            return
        except PermissionError as exc:
            raise InstrumentsProcessError(
                f"not permitted to signal process group for pid "
                f"{self._popen.pid}: {exc}"
            ) from exc


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
