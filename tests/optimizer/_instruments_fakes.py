"""Injectable fakes so the Instruments tests never need Xcode.

Every boundary the collector touches (short probe commands, long
recording processes) is a protocol, so the whole capability, record,
export and evidence path can be exercised on any platform.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import IO

from llmtracefx.optimizer.instruments.process import (
    CommandResult,
    InstrumentsProcessError,
)

FIXTURES = Path(__file__).parent / "fixtures" / "instruments"

#: The real error macOS emits when only Command Line Tools are selected.
#: Captured verbatim from a machine in that state.
COMMAND_LINE_TOOLS_STDERR = (
    "xcode-select: error: tool 'xctrace' requires Xcode, but active "
    "developer directory '/Library/Developer/CommandLineTools' is a "
    "command line tools instance\n"
)

#: The real first line of `xctrace version` output on Xcode 16.0.
VERSION_STDOUT = "xctrace version 16.0 (17F113)\n"

#: Shape of real `xctrace list templates` output, trimmed.
TEMPLATES_STDOUT = (
    "== Standard Templates ==\n"
    "Activity Monitor\n"
    "Allocations\n"
    "CPU Counters\n"
    "Game Performance\n"
    "Metal System Trace\n"
    "System Trace\n"
    "Time Profiler\n"
    "== Custom Templates ==\n"
)


def read_fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def ok(argv: Sequence[str], stdout: str = "") -> CommandResult:
    return CommandResult(argv=tuple(argv), returncode=0, stdout=stdout, stderr="")


def fail(
    argv: Sequence[str], *, returncode: int = 1, stderr: str = ""
) -> CommandResult:
    return CommandResult(
        argv=tuple(argv), returncode=returncode, stdout="", stderr=stderr
    )


class FakeCommandRunner:
    """Answers probes from a table keyed by the argv tail.

    Keying on the tail (``('version',)``, ``('list', 'templates')``)
    keeps tests independent of where xctrace was found on PATH.
    """

    def __init__(
        self,
        *,
        version: CommandResult | None = None,
        templates: CommandResult | None = None,
        exports: dict[str, CommandResult] | None = None,
        raise_on: str | None = None,
    ) -> None:
        self.version = version
        self.templates = templates
        self.exports = exports or {}
        self.raise_on = raise_on
        self.calls: list[tuple[str, ...]] = []
        self.written: list[Path] = []

    def run(self, argv: Sequence[str], *, timeout_seconds: float) -> CommandResult:
        argv_tuple = tuple(argv)
        self.calls.append(argv_tuple)
        if self.raise_on is not None and self.raise_on in argv_tuple:
            raise InstrumentsProcessError(f"cannot execute {self.raise_on}")

        if argv_tuple[1:] == ("version",):
            if self.version is None:
                return ok(argv_tuple, VERSION_STDOUT)
            return self.version
        if argv_tuple[1:] == ("list", "templates"):
            if self.templates is None:
                return ok(argv_tuple, TEMPLATES_STDOUT)
            return self.templates
        if len(argv_tuple) > 1 and argv_tuple[1] == "export":
            return self._export(argv_tuple)
        raise AssertionError(f"unexpected command: {argv_tuple}")

    def _export(self, argv: tuple[str, ...]) -> CommandResult:
        output = Path(argv[argv.index("--output") + 1])
        if "--toc" in argv:
            key = "toc"
        else:
            xpath = argv[argv.index("--xpath") + 1]
            key = xpath.split('@schema="')[1].split('"')[0]
        result = self.exports.get(key)
        if result is None:
            return fail(argv, returncode=1, stderr=f"no such table: {key}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(result.stdout, encoding="utf-8")
        self.written.append(output)
        return ok(argv)


class FakeProcess:
    """A recording process whose exit behavior the test dictates."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        timeout_waits: int = 0,
        pid: int = 4242,
        stdout_text: str = "",
        stderr_text: str = "",
        signal_error: Exception | None = None,
    ) -> None:
        self._returncode: int | None = returncode
        self._final_returncode = returncode
        self.timeout_waits = timeout_waits
        self._pid = pid
        self.stdout_text = stdout_text
        self.stderr_text = stderr_text
        self.signal_error = signal_error
        self.signals: list[int] = []
        self.wait_calls: list[float] = []

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def returncode(self) -> int | None:
        return self._returncode

    def wait(self, timeout_seconds: float) -> int:
        self.wait_calls.append(timeout_seconds)
        if self.timeout_waits > 0:
            self.timeout_waits -= 1
            self._returncode = None
            raise subprocess.TimeoutExpired(cmd="xctrace", timeout=timeout_seconds)
        self._returncode = self._final_returncode
        return self._final_returncode

    def signal_group(self, signal_number: int) -> None:
        self.signals.append(signal_number)
        if self.signal_error is not None:
            raise self.signal_error


class FakeLauncher:
    """Spawns :class:`FakeProcess` and optionally creates the bundle."""

    def __init__(
        self,
        process: FakeProcess,
        *,
        creates_trace: Path | None = None,
        spawn_error: Exception | None = None,
    ) -> None:
        self.process = process
        self.creates_trace = creates_trace
        self.spawn_error = spawn_error
        self.spawned: list[tuple[str, ...]] = []

    def spawn(
        self,
        argv: Sequence[str],
        *,
        stdout: IO[bytes],
        stderr: IO[bytes],
        cwd: Path | None,
        env: dict[str, str] | None,
    ) -> FakeProcess:
        self.spawned.append(tuple(argv))
        if self.spawn_error is not None:
            raise self.spawn_error
        stdout.write(self.process.stdout_text.encode("utf-8"))
        stderr.write(self.process.stderr_text.encode("utf-8"))
        stdout.flush()
        stderr.flush()
        if self.creates_trace is not None:
            self.creates_trace.mkdir(parents=True, exist_ok=True)
        return self.process
