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

#: Segments that mark a name as a credential on their own.
_CREDENTIAL_SEGMENTS: frozenset[str] = frozenset(
    {
        "secret",
        "secrets",
        "password",
        "passwd",
        "passphrase",
        "credential",
        "credentials",
        "apikey",
        "bearer",
        "authorization",
        "jwt",
        "totp",
        "otp",
    }
)

#: Segments that make a trailing ``key`` a lookup key rather than a
#: cryptographic one. ``--sort-key name`` is not a secret; ``--ssh-key``
#: is. Listing the benign qualifiers rather than the credential ones
#: means an unrecognized ``*-key`` option is treated as sensitive.
_BENIGN_KEY_QUALIFIERS: frozenset[str] = frozenset(
    {
        "sort",
        "cache",
        "group",
        "partition",
        "primary",
        "foreign",
        "index",
        "map",
        "lookup",
        "shard",
        "route",
        "order",
        "id",
        "column",
        "row",
        "object",
        "hash",
    }
)

#: Normalized phrases that mark a name as a credential.
_CREDENTIAL_PHRASES: tuple[str, ...] = (
    "api_key",
    "access_key",
    "secret_key",
    "private_key",
    "signing_key",
    "encryption_key",
    "auth_token",
    "access_token",
    "refresh_token",
    "session_token",
    "id_token",
)

#: Segments that mark a ``token`` name as a *quantity* rather than a
#: credential. This matters more here than in most projects: an LLM
#: inference command is full of ``--max-tokens``, ``--num-tokens`` and
#: ``--token-count``, and redacting those would destroy the
#: reproducibility the recorded argv exists to provide.
_TOKEN_QUANTITY_SEGMENTS: frozenset[str] = frozenset(
    {
        "max",
        "min",
        "num",
        "n",
        "count",
        "per",
        "budget",
        "new",
        "input",
        "output",
        "context",
        "prompt",
        "generated",
        "total",
        "limit",
        "size",
    }
)

#: Trailing segments that make a name refer to a *location* rather than
#: to a secret. ``--private-key-path /etc/k.pem`` names a file; the path
#: is not the credential, and redacting it would lose reproducibility
#: for no privacy gain.
_LOCATION_SUFFIXES: frozenset[str] = frozenset(
    {"path", "file", "dir", "directory", "url", "uri", "endpoint", "name"}
)

REDACTED = "<redacted>"


class InstrumentsCommandError(ValueError):
    """Raised when a requested xctrace invocation would be unsafe."""


def _normalize_option_name(name: str) -> str:
    """Reduce an option or variable name to a comparable form.

    ``--API-Key`` and ``api_key`` and ``HF_TOKEN`` all normalize into the
    lowercase underscore form the tables above are written in, so case
    and separator aliases do not need separate entries.
    """
    return name.lstrip("-").replace("-", "_").casefold()


def _is_credential_name(name: str) -> bool:
    """Whether a name's value should be treated as a secret.

    Deliberately not a plain substring match, in both directions.

    ``token`` appears in both ``--hf-token`` (a secret) and
    ``--max-tokens`` (a benign quantity), so the two are separated
    rather than redacting both and corrupting the second.

    Conversely a name ending in a location suffix refers to where a
    secret lives, not to the secret: ``--private-key-path`` is a
    filename worth keeping for reproducibility.
    """
    normalized = _normalize_option_name(name)
    if not normalized:
        return False
    segments = normalized.split("_")
    tail = segments[-1]
    if len(segments) > 1 and tail in _LOCATION_SUFFIXES:
        return False

    segment_set = set(segments)
    if segment_set & _CREDENTIAL_SEGMENTS:
        return True
    if any(phrase in normalized for phrase in _CREDENTIAL_PHRASES):
        return True
    if normalized in {"key", "token", "pat", "auth"}:
        return True
    if tail == "key":
        # `ssh_key`, `deploy_key`, `session_key`. Default to treating an
        # unrecognized `*-key` as sensitive, since the cost of redacting
        # a lookup key is far lower than the cost of persisting a
        # private one.
        return len(segments) < 2 or segments[-2] not in _BENIGN_KEY_QUALIFIERS
    if "token" in segment_set and not (segment_set & _TOKEN_QUANTITY_SEGMENTS):
        # `hf_token`, `service_token`. Plural `tokens` is a count, and
        # so is anything carrying a quantity qualifier.
        return True
    return False


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
    stop_grace_seconds: float = 30.0
    """Seconds allowed after each stop signal before escalating.

    Applies per signal in the SIGINT, SIGTERM, SIGKILL sequence, and
    also bounds how long the recorder waits for the whole process group
    to empty after each one."""

    def __post_init__(self) -> None:
        if not self.xctrace_path:
            raise InstrumentsCommandError("xctrace_path must be non-empty")
        if not self.template:
            raise InstrumentsCommandError("template must be non-empty")
        # Expand `~` once, here, so the path this plan validates is
        # byte-for-byte the path that reaches xctrace. Leaving it
        # unexpanded would let the collision check and mkdir target
        # $HOME while xctrace created a literal './~' directory.
        object.__setattr__(self, "output_trace", self.output_trace.expanduser())
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
        if self.stop_grace_seconds <= 0:
            raise InstrumentsCommandError("stop_grace_seconds must be > 0")
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
            # The profiled command is user supplied and can carry a
            # secret (for example `env HF_TOKEN=... python bench.py`),
            # so it goes through the same redaction as --env before
            # being persisted or printed.
            argv.extend(redact_argv(self.target.argv) if redacted else self.target.argv)
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
    """Redact credential-bearing tokens anywhere in an argv.

    Three shapes are recognized, because a secret reaches a command in
    all three and redacting only the first leaves the other two in
    plain text in every artifact this project writes:

    ``NAME=value``
        The environment style, with or without leading dashes.
    ``--api-key=value``
        A long option carrying its value after ``=``.
    ``--api-key value``
        A long option whose value is the *next* argv element. The next
        element is always replaced, including when it starts with a
        dash, since a credential can legitimately look like an option.

    Not recognized, and documented rather than guessed at: a value
    attached to a single-dash short option (``-ksk-live``). There is no
    way to tell that from a cluster of short flags without knowing the
    target program's own option grammar, and guessing would corrupt
    legitimate arguments.
    """
    redacted: list[str] = []
    redact_next = False
    for item in argv:
        if redact_next:
            redacted.append(REDACTED)
            redact_next = False
            continue

        name, separator, _ = item.partition("=")
        if separator and name and _is_credential_name(name):
            redacted.append(f"{name}={REDACTED}")
            continue

        if item.startswith("-") and _is_credential_name(item):
            redacted.append(item)
            redact_next = True
            continue

        redacted.append(item)
    return tuple(redacted)
