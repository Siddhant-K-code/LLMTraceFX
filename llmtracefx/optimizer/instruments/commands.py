"""Safe, shell-free ``xctrace`` command construction.

Everything here returns an argv tuple. Nothing in this package ever
builds a command string or passes ``shell=True``, so no template name,
file path or user-supplied inference argument can be reinterpreted as
shell syntax.

Flag spellings and semantics below were taken from the local
``xctrace help record`` and ``xctrace help export`` output on
``xctrace version 16.0 (17F113)``:

* ``record``: ``--output``, ``--append-run``, ``--run-name``,
  ``--template <path|name>``, ``--device <name|UDID>``, ``--instrument``,
  ``--time-limit <time[ms|s|m|h]>``, ``--window <duration[ms|s|m]>``,
  ``--package``, ``--all-processes``, ``--attach <pid|name>``,
  ``--launch -- command [arguments]``, ``--target-stdin``,
  ``--target-stdout``, ``--env <VAR=value>``, ``--notify-tracing-started``,
  ``--no-prompt``.
* ``export``: ``--input``, ``--output``, ``--toc``, ``--xpath``, ``--har``.
  The help text states TOC and XPath "cannot be specified together".
* On ``--output``: "If trace file already exists, then --append-run needs
  to be specified to add a run to it." xctrace therefore refuses to
  overwrite on its own; the collision check in :mod:`.recorder` is a
  pre-flight that fails earlier and more clearly.
* Omitting ``--device`` records on the host. This package never passes
  ``--device``, so it can never be pointed at someone else's machine.

Deliberate omissions, each a safety decision rather than an oversight:

``--all-processes``
    Captures every process on the system. Refused outright.
``--attach <name>``
    Attaching by name resolves ambiguously when several processes share
    a name. Only attaching by numeric pid is offered.
``--target-stdin`` / ``--target-stdout``
    Would route the profiled program's own input and output through
    xctrace and into this project's captured log files. For an LLM
    workload that is prompt and completion text, so neither flag is
    constructed here at all.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

#: ``--time-limit`` accepts ms, s, m or h per ``xctrace help record``.
TIME_LIMIT_UNITS: tuple[str, ...] = ("ms", "s", "m", "h")

#: ``--window`` accepts ms, s or m. It notably does not accept h.
WINDOW_UNITS: tuple[str, ...] = ("ms", "s", "m")

_TIME_LIMIT_RE = re.compile(r"^(?P<amount>\d+)(?P<unit>ms|s|m|h)$")
_WINDOW_RE = re.compile(r"^(?P<amount>\d+)(?P<unit>ms|s|m)$")

_UNIT_SECONDS: dict[str, float] = {
    "ms": 0.001,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}

#: A trace table schema name, as it appears in a TOC ``schema=``
#: attribute. Constrained so a schema name can never terminate the
#: quoted string in a generated XPath predicate and inject a new one.
_SCHEMA_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]*$")

#: Environment variable names whose values are redacted from any argv
#: this project stores or prints. Matched case-insensitively as
#: substrings, so ``HF_TOKEN`` and ``OPENAI_API_KEY`` both match.
_CREDENTIAL_NAME_MARKERS: tuple[str, ...] = (
    "token",
    "secret",
    "password",
    "passwd",
    "api_key",
    "apikey",
    "access_key",
    "private_key",
    "credential",
)

REDACTED = "<redacted>"


class InstrumentsCommandError(ValueError):
    """Raised when a requested xctrace invocation would be unsafe."""


def validate_time_limit(value: str) -> str:
    """Validate a ``--time-limit`` value such as ``10s`` or ``500ms``."""
    if _TIME_LIMIT_RE.match(value) is None:
        raise InstrumentsCommandError(
            f"invalid --time-limit {value!r}: expected a whole number "
            f"followed by one of {', '.join(TIME_LIMIT_UNITS)}, e.g. '30s'"
        )
    return value


def validate_window(value: str) -> str:
    """Validate a ``--window`` value such as ``5s``."""
    if _WINDOW_RE.match(value) is None:
        raise InstrumentsCommandError(
            f"invalid --window {value!r}: expected a whole number followed "
            f"by one of {', '.join(WINDOW_UNITS)}, e.g. '5s'. Note that "
            f"xctrace accepts 'h' for --time-limit but not for --window."
        )
    return value


def duration_to_seconds(value: str) -> float:
    """Convert a validated xctrace duration into seconds."""
    match = _TIME_LIMIT_RE.match(value)
    if match is None:
        raise InstrumentsCommandError(
            f"cannot convert {value!r} to seconds: not a valid xctrace duration"
        )
    return int(match.group("amount")) * _UNIT_SECONDS[match.group("unit")]


def validate_schema_name(value: str) -> str:
    """Validate a trace table schema name before it enters an XPath.

    Schema names reach this from a TOC or straight from the command
    line. Without this check a value such as ``a"] | //*[@x="`` would
    close the predicate's quoted string and append an attacker-chosen
    query, so anything outside the conservative character set is
    rejected instead of escaped.
    """
    if not _SCHEMA_NAME_RE.match(value):
        raise InstrumentsCommandError(
            f"invalid trace table schema name {value!r}: expected a name "
            "matching [A-Za-z][A-Za-z0-9._-]*, e.g. 'metal-gpu-intervals'"
        )
    return value


def validate_run_number(value: int) -> int:
    """Validate a trace run number before it enters an XPath."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InstrumentsCommandError(
            f"invalid trace run number {value!r}: expected an integer >= 1"
        )
    return value


def table_xpath(schema_name: str, *, run_number: int = 1) -> str:
    """Build the XPath that selects one table from a trace's TOC.

    Matches the form documented in ``xctrace help export``:
    ``/trace-toc/run[@number="1"]/data/table[@schema="my-table-schema"]``.
    """
    schema = validate_schema_name(schema_name)
    run = validate_run_number(run_number)
    return f'/trace-toc/run[@number="{run}"]/data/table[@schema="{schema}"]'


def _is_credential_name(name: str) -> bool:
    folded = name.casefold()
    return any(marker in folded for marker in _CREDENTIAL_NAME_MARKERS)


@dataclass(frozen=True)
class EnvironmentAssignment:
    """One ``--env VAR=value`` pair for the launched process."""

    name: str
    value: str

    def __post_init__(self) -> None:
        if not self.name or "=" in self.name:
            raise InstrumentsCommandError(
                f"invalid environment variable name {self.name!r}: must be "
                "non-empty and must not contain '='"
            )

    def as_argument(self) -> str:
        return f"{self.name}={self.value}"

    def as_redacted_argument(self) -> str:
        """The pair with a credential-looking value replaced.

        Applied whenever the argv is stored or printed, so a token
        passed through ``--env`` never reaches an experiment record.
        """
        if _is_credential_name(self.name):
            return f"{self.name}={REDACTED}"
        return self.as_argument()


@dataclass(frozen=True)
class LaunchTarget:
    """Launch and profile a program. ``argv`` is the program plus args."""

    argv: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.argv or not all(
            isinstance(item, str) and item for item in self.argv
        ):
            raise InstrumentsCommandError(
                "launch target argv must be a non-empty sequence of "
                "non-empty strings"
            )


@dataclass(frozen=True)
class AttachTarget:
    """Attach to an already running process by numeric pid."""

    pid: int

    def __post_init__(self) -> None:
        if isinstance(self.pid, bool) or not isinstance(self.pid, int):
            raise InstrumentsCommandError(
                f"attach pid must be an integer, got {self.pid!r}"
            )
        if self.pid <= 0:
            raise InstrumentsCommandError(
                f"attach pid must be a positive integer, got {self.pid}"
            )


RecordTarget = LaunchTarget | AttachTarget


@dataclass(frozen=True)
class RecordPlan:
    """A fully resolved, validated ``xctrace record`` invocation.

    Building one of these performs every check that does not need to
    touch the filesystem, so ``plan`` mode can surface problems without
    recording anything.
    """

    xctrace_path: str
    template: str
    output_trace: Path
    target: RecordTarget
    time_limit: str = "60s"
    window: str | None = None
    run_name: str | None = None
    environment: tuple[EnvironmentAssignment, ...] = ()
    no_prompt: bool = True
    grace_seconds: float = 90.0
    """Extra seconds allowed past ``time_limit`` before the host gives up.

    xctrace keeps running after the recording window closes while it
    writes and finalizes the ``.trace`` bundle, and that step is not
    instant. The host timeout must therefore be strictly greater than
    ``time_limit`` or every recording would be torn down mid-save."""

    def __post_init__(self) -> None:
        if not self.xctrace_path:
            raise InstrumentsCommandError("xctrace_path must be non-empty")
        if not self.template:
            raise InstrumentsCommandError("template must be non-empty")
        if self.output_trace.suffix != ".trace":
            raise InstrumentsCommandError(
                f"output trace path must end in '.trace', got "
                f"{self.output_trace.name!r}"
            )
        validate_time_limit(self.time_limit)
        if self.window is not None:
            validate_window(self.window)
        if self.run_name is not None and not self.run_name:
            raise InstrumentsCommandError("run_name must be non-empty when set")
        if self.grace_seconds <= 0:
            raise InstrumentsCommandError("grace_seconds must be > 0")
        if self.environment and not isinstance(self.target, LaunchTarget):
            # xctrace help record: "Specifying environment variables or
            # stream redirection is only available when using launch
            # option."
            raise InstrumentsCommandError(
                "--env is only supported when launching a process; it has "
                "no effect when attaching to an existing pid"
            )

    @property
    def timeout_seconds(self) -> float:
        """Host-side wall-clock bound for the whole recording."""
        return duration_to_seconds(self.time_limit) + self.grace_seconds

    def _argv(self, *, redacted: bool) -> tuple[str, ...]:
        argv: list[str] = [self.xctrace_path, "record"]
        argv.extend(("--template", self.template))
        argv.extend(("--output", str(self.output_trace)))
        argv.extend(("--time-limit", self.time_limit))
        if self.window is not None:
            argv.extend(("--window", self.window))
        if self.run_name is not None:
            argv.extend(("--run-name", self.run_name))
        if self.no_prompt:
            argv.append("--no-prompt")
        for assignment in self.environment:
            argv.append("--env")
            argv.append(
                assignment.as_redacted_argument()
                if redacted
                else assignment.as_argument()
            )

        # --device is never passed, so the host is always the target.
        # --append-run is never passed, so xctrace refuses to touch an
        # existing bundle rather than adding a run to it.
        if isinstance(self.target, AttachTarget):
            argv.extend(("--attach", str(self.target.pid)))
        else:
            # xctrace help record shows --launch as the trailing form,
            # and everything after `--` belongs to the target program.
            # This must stay last.
            argv.append("--launch")
            argv.append("--")
            argv.extend(self.target.argv)
        return tuple(argv)

    def to_argv(self) -> tuple[str, ...]:
        """The argv actually handed to the OS, with real env values."""
        return self._argv(redacted=False)

    def to_redacted_argv(self) -> tuple[str, ...]:
        """The argv safe to persist and print.

        Deterministic: the same plan always reconstructs the same
        sequence, so a record's stored command can be compared across
        runs. Credential-looking ``--env`` values are replaced.
        """
        return self._argv(redacted=True)


@dataclass(frozen=True)
class ExportPlan:
    """A validated ``xctrace export`` invocation.

    Exactly one of ``toc`` or ``schema_name`` is used, because the help
    text states the two modes cannot be combined.
    """

    xctrace_path: str
    input_trace: Path
    output_path: Path
    toc: bool = False
    schema_name: str | None = None
    run_number: int = 1
    timeout_seconds: float = 600.0
    extra_checks: tuple[str, ...] = field(default=(), repr=False)

    def __post_init__(self) -> None:
        if not self.xctrace_path:
            raise InstrumentsCommandError("xctrace_path must be non-empty")
        if self.toc and self.schema_name is not None:
            raise InstrumentsCommandError(
                "xctrace export cannot combine --toc with --xpath; request "
                "one mode or the other"
            )
        if not self.toc and self.schema_name is None:
            raise InstrumentsCommandError(
                "xctrace export needs either --toc or a table schema name"
            )
        if self.schema_name is not None:
            validate_schema_name(self.schema_name)
            validate_run_number(self.run_number)
        if self.timeout_seconds <= 0:
            raise InstrumentsCommandError("timeout_seconds must be > 0")

    def to_argv(self) -> tuple[str, ...]:
        argv: list[str] = [self.xctrace_path, "export"]
        argv.extend(("--input", str(self.input_trace)))
        argv.extend(("--output", str(self.output_path)))
        if self.toc:
            argv.append("--toc")
        else:
            assert self.schema_name is not None  # guarded in __post_init__
            argv.extend(
                ("--xpath", table_xpath(self.schema_name, run_number=self.run_number))
            )
        return tuple(argv)


def build_list_templates_argv(xctrace_path: str) -> tuple[str, ...]:
    """Argv for ``xctrace list templates``."""
    if not xctrace_path:
        raise InstrumentsCommandError("xctrace_path must be non-empty")
    return (xctrace_path, "list", "templates")


def build_version_argv(xctrace_path: str) -> tuple[str, ...]:
    """Argv for ``xctrace version``."""
    if not xctrace_path:
        raise InstrumentsCommandError("xctrace_path must be non-empty")
    return (xctrace_path, "version")


def redact_argv(argv: Sequence[str]) -> tuple[str, ...]:
    """Redact credential-looking ``NAME=VALUE`` tokens in an argv.

    Used for argv that did not come from a :class:`RecordPlan`, such as
    a reconstructed CLI invocation.
    """
    redacted: list[str] = []
    for item in argv:
        name, separator, _ = item.partition("=")
        if separator and name and _is_credential_name(name):
            redacted.append(f"{name}={REDACTED}")
        else:
            redacted.append(item)
    return tuple(redacted)
