"""Capability detection for Apple's Instruments CLI (``xctrace``).

The point of this module is to answer, precisely, *why* a Metal trace
cannot be taken, instead of collapsing every failure into a single
"unavailable". Each distinguishable cause is its own
:class:`XctraceCapability` value with a fix the caller can act on.

There is deliberately no broad ``except Exception`` and no silent
fallback anywhere in this module. An unrecognized failure becomes
:attr:`XctraceCapability.PROBE_FAILED`, carrying the real exit status,
rather than being reported as some nearby state that happens to look
plausible.

Verified against a local install (recorded in ``checked_signals`` at
detection time, not hardcoded as a claim):

* ``xctrace version`` prints ``xctrace version 16.0 (17F113)`` and exits
  0 when a full Xcode is selected.
* With only Command Line Tools selected, the shim at ``/usr/bin/xctrace``
  exists and is executable, so a mere "is it on PATH" check passes while
  every real invocation fails with ``xcode-select: error: tool 'xctrace'
  requires Xcode, but active developer directory
  '/Library/Developer/CommandLineTools' is a command line tools
  instance``. That is why presence on PATH is never treated as support.
* ``xctrace list templates`` prints ``== Standard Templates ==`` followed
  by one template name per line, then an optional ``== Custom Templates
  ==`` section.
"""

from __future__ import annotations

import json
import platform
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..collectors._shared import atomic_write_text
from .process import CommandResult, CommandRunner, InstrumentsProcessError


def default_xctrace_path() -> str | None:
    """Locate xctrace on PATH, or ``None`` when it is absent.

    Finding it proves only that a file exists: on a machine with just
    Command Line Tools this returns ``/usr/bin/xctrace``, a shim that
    fails on every invocation. Callers must still probe.
    """
    return shutil.which("xctrace")


CAPABILITY_SCHEMA_VERSION = "1"

#: Instruments template this project records Metal evidence with.
#: Present in Xcode 16.0's ``xctrace list templates`` output.
METAL_SYSTEM_TRACE_TEMPLATE = "Metal System Trace"

#: Seconds allowed for a capability probe. These are metadata queries
#: that normally return in well under a second; the bound exists so a
#: wedged tool cannot hang a caller indefinitely.
PROBE_TIMEOUT_SECONDS = 60.0


class InstrumentsCapabilityError(RuntimeError):
    """Raised when a capability report cannot be built or parsed."""


class XctraceCapability(str, Enum):
    """Why Instruments evidence is or is not collectable here."""

    SUPPORTED = "supported"
    """xctrace ran, and the requested template exists."""

    UNSUPPORTED_OS = "unsupported_os"
    """Not macOS. Instruments and xctrace are macOS only."""

    UNSUPPORTED_ARCHITECTURE = "unsupported_architecture"
    """Not arm64. This project only validates Apple Silicon GPU
    evidence, so it refuses to imply support it has not exercised."""

    XCTRACE_NOT_FOUND = "xctrace_not_found"
    """No xctrace executable on PATH at all."""

    COMMAND_LINE_TOOLS_ONLY = "command_line_tools_only"
    """xctrace resolves to the Command Line Tools shim. The binary
    exists but refuses to run because no full Xcode is selected."""

    LICENSE_NOT_ACCEPTED = "license_not_accepted"
    """Xcode is installed but its license has not been agreed to."""

    FIRST_LAUNCH_REQUIRED = "first_launch_required"
    """Xcode's one time first launch setup has not completed."""

    TEMPLATE_UNAVAILABLE = "template_unavailable"
    """xctrace works, but the requested template is not installed."""

    PERMISSION_DENIED = "permission_denied"
    """The OS refused the recording (entitlements, hardened runtime,
    or profiling permission)."""

    PROBE_FAILED = "probe_failed"
    """xctrace failed for a reason this module does not recognize. The
    exit status and a classification hint are preserved rather than
    guessed at."""

    @property
    def is_supported(self) -> bool:
        return self is XctraceCapability.SUPPORTED


#: Substring patterns mapped to the capability they prove, matched
#: case-insensitively against combined stdout/stderr. Ordered: the
#: Command Line Tools check runs before the generic "requires Xcode"
#: check because its message contains both phrases.
_ERROR_PATTERNS: tuple[tuple[tuple[str, ...], XctraceCapability], ...] = (
    (
        ("command line tools instance",),
        XctraceCapability.COMMAND_LINE_TOOLS_ONLY,
    ),
    (
        ("requires xcode", "active developer directory"),
        XctraceCapability.COMMAND_LINE_TOOLS_ONLY,
    ),
    (
        ("xcode-select: error: unable to find utility",),
        XctraceCapability.COMMAND_LINE_TOOLS_ONLY,
    ),
    (
        ("license",),
        XctraceCapability.LICENSE_NOT_ACCEPTED,
    ),
    (
        ("runfirstlaunch",),
        XctraceCapability.FIRST_LAUNCH_REQUIRED,
    ),
    (
        ("first launch",),
        XctraceCapability.FIRST_LAUNCH_REQUIRED,
    ),
    (
        ("operation not permitted",),
        XctraceCapability.PERMISSION_DENIED,
    ),
    (
        ("not permitted to",),
        XctraceCapability.PERMISSION_DENIED,
    ),
    (
        ("permission denied",),
        XctraceCapability.PERMISSION_DENIED,
    ),
    (
        ("failed to attach",),
        XctraceCapability.PERMISSION_DENIED,
    ),
    (
        ("get-task-allow",),
        XctraceCapability.PERMISSION_DENIED,
    ),
)


def classify_xctrace_failure(text: str) -> XctraceCapability | None:
    """Map known Apple error text to a capability, or ``None``.

    ``None`` means "not recognized". Callers must escalate that to
    :attr:`XctraceCapability.PROBE_FAILED` rather than substituting a
    convenient guess.
    """
    haystack = text.casefold()
    for needles, capability in _ERROR_PATTERNS:
        if all(needle in haystack for needle in needles):
            return capability
    return None


def parse_templates(stdout: str) -> tuple[str, ...]:
    """Parse ``xctrace list templates`` output into template names.

    The real format is section headers delimited by ``==`` with one
    template name per line between them. Blank lines and headers are
    dropped; surrounding whitespace on names is stripped because the
    custom templates section indents its entries.
    """
    names: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("=="):
            continue
        names.append(line)
    # dict.fromkeys keeps first-seen order while removing duplicates,
    # which can occur when a custom template shadows a standard name.
    return tuple(dict.fromkeys(names))


def parse_version(stdout: str) -> str | None:
    """Extract the version string from ``xctrace version`` output.

    Returns the whole trimmed first non-empty line, for example
    ``xctrace version 16.0 (17F113)``. Kept verbatim rather than parsed
    into components so the evidence records exactly what the tool said.
    """
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if line:
            return line
    return None


@dataclass(frozen=True)
class XctraceCapabilityReport:
    """Whether this machine can produce Instruments evidence right now.

    A standalone artifact rather than an ``ExperimentRecord``: capability
    is a property of the installed toolchain, not of a measured run.
    """

    schema_version: str
    capability: XctraceCapability
    reason: str
    remediation: str | None
    os_name: str
    architecture: str
    xctrace_path: str | None
    xctrace_version: str | None
    requested_template: str | None
    available_templates: tuple[str, ...]
    checked_signals: tuple[str, ...]

    @property
    def supported(self) -> bool:
        return self.capability.is_supported

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "capability": self.capability.value,
            "supported": self.supported,
            "reason": self.reason,
            "remediation": self.remediation,
            "os_name": self.os_name,
            "architecture": self.architecture,
            "xctrace_path": self.xctrace_path,
            "xctrace_version": self.xctrace_version,
            "requested_template": self.requested_template,
            "available_templates": list(self.available_templates),
            "checked_signals": list(self.checked_signals),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    def write_json(self, path: str | Path) -> None:
        atomic_write_text(Path(path), self.to_json() + "\n")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> XctraceCapabilityReport:
        try:
            capability = XctraceCapability(data["capability"])
        except KeyError as exc:
            raise InstrumentsCapabilityError(
                f"XctraceCapabilityReport is missing required field: {exc}"
            ) from exc
        except ValueError as exc:
            raise InstrumentsCapabilityError(
                f"XctraceCapabilityReport has an unknown capability: {exc}"
            ) from exc
        try:
            return cls(
                schema_version=str(
                    data.get("schema_version", CAPABILITY_SCHEMA_VERSION)
                ),
                capability=capability,
                reason=data["reason"],
                remediation=data.get("remediation"),
                os_name=data["os_name"],
                architecture=data["architecture"],
                xctrace_path=data.get("xctrace_path"),
                xctrace_version=data.get("xctrace_version"),
                requested_template=data.get("requested_template"),
                available_templates=tuple(data.get("available_templates", ())),
                checked_signals=tuple(data.get("checked_signals", ())),
            )
        except KeyError as exc:
            raise InstrumentsCapabilityError(
                f"XctraceCapabilityReport is missing required field: {exc}"
            ) from exc


_REMEDIATION: dict[XctraceCapability, str] = {
    XctraceCapability.UNSUPPORTED_OS: (
        "Instruments and xctrace ship only with Xcode on macOS. There is "
        "no remediation on this platform."
    ),
    XctraceCapability.UNSUPPORTED_ARCHITECTURE: (
        "Run on an Apple Silicon (arm64) Mac. This project has only "
        "validated its Metal evidence path on Apple Silicon and will not "
        "imply support for hardware it has not exercised."
    ),
    XctraceCapability.XCTRACE_NOT_FOUND: (
        "Install Xcode from the Mac App Store, then select it with "
        "`sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`."
    ),
    XctraceCapability.COMMAND_LINE_TOOLS_ONLY: (
        "A full Xcode is required; Command Line Tools alone do not ship "
        "xctrace. Install Xcode, then run "
        "`sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`."
    ),
    XctraceCapability.LICENSE_NOT_ACCEPTED: (
        "Accept the Xcode license: `sudo xcodebuild -license accept`. "
        "This needs administrator rights, so run it yourself; this "
        "project never invokes sudo on your behalf."
    ),
    XctraceCapability.FIRST_LAUNCH_REQUIRED: (
        "Complete Xcode's first launch setup: "
        "`sudo xcodebuild -runFirstLaunch`. This needs administrator "
        "rights, so run it yourself; this project never invokes sudo on "
        "your behalf."
    ),
    XctraceCapability.PERMISSION_DENIED: (
        "macOS refused the recording. Profile a binary you built locally "
        "(hardened runtime binaries need the get-task-allow entitlement), "
        "and prefer launching the target rather than attaching to it."
    ),
}


def _report(
    capability: XctraceCapability,
    *,
    reason: str,
    os_name: str,
    architecture: str,
    checked: tuple[str, ...],
    xctrace_path: str | None = None,
    xctrace_version: str | None = None,
    requested_template: str | None = None,
    available_templates: tuple[str, ...] = (),
) -> XctraceCapabilityReport:
    return XctraceCapabilityReport(
        schema_version=CAPABILITY_SCHEMA_VERSION,
        capability=capability,
        reason=reason,
        remediation=_REMEDIATION.get(capability),
        os_name=os_name,
        architecture=architecture,
        xctrace_path=xctrace_path,
        xctrace_version=xctrace_version,
        requested_template=requested_template,
        available_templates=available_templates,
        checked_signals=checked,
    )


def _describe(result: CommandResult) -> str:
    """Summarize a probe result without echoing its output.

    Deliberately records only the exit status and whether it timed out.
    Instruments error text can contain device names and file paths, and
    this string is persisted into the capability report.
    """
    if result.timed_out:
        return "timed out"
    return f"exit {result.returncode}"


def detect_xctrace_capability(
    *,
    runner: CommandRunner,
    template: str | None = METAL_SYSTEM_TRACE_TEMPLATE,
    os_name: str | None = None,
    architecture: str | None = None,
    path_resolver: Callable[[], str | None] | None = None,
    timeout_seconds: float = PROBE_TIMEOUT_SECONDS,
) -> XctraceCapabilityReport:
    """Determine whether ``xctrace`` can record ``template`` here.

    ``path_resolver`` returns the xctrace path, or ``None`` when there
    is none. It is a callable rather than a plain string so that "not
    installed" is expressible: a ``None`` *path* argument would be
    ambiguous between "absent" and "look it up for me".

    Every argument beyond ``runner`` exists so tests can pin the
    environment. In production all of them are resolved from the host.
    """
    resolved_os = platform.system() if os_name is None else os_name
    resolved_arch = platform.machine() if architecture is None else architecture
    checked: list[str] = [
        f"os_name={resolved_os}",
        f"architecture={resolved_arch}",
    ]

    if resolved_os != "Darwin":
        return _report(
            XctraceCapability.UNSUPPORTED_OS,
            reason=(
                f"xctrace requires macOS; this host reports "
                f"os_name={resolved_os!r}."
            ),
            os_name=resolved_os,
            architecture=resolved_arch,
            checked=tuple(checked),
            requested_template=template,
        )

    if resolved_arch != "arm64":
        return _report(
            XctraceCapability.UNSUPPORTED_ARCHITECTURE,
            reason=(
                f"this project only collects Metal evidence on Apple "
                f"Silicon; this host reports architecture="
                f"{resolved_arch!r}."
            ),
            os_name=resolved_os,
            architecture=resolved_arch,
            checked=tuple(checked),
            requested_template=template,
        )

    resolve = default_xctrace_path if path_resolver is None else path_resolver
    resolved_path = resolve()
    checked.append(f"xctrace_on_path={resolved_path is not None}")
    if resolved_path is None:
        return _report(
            XctraceCapability.XCTRACE_NOT_FOUND,
            reason="no xctrace executable was found on PATH.",
            os_name=resolved_os,
            architecture=resolved_arch,
            checked=tuple(checked),
            requested_template=template,
        )

    # Presence on PATH proves nothing: the Command Line Tools shim at
    # /usr/bin/xctrace exists and is executable but refuses to run.
    # Only an actual invocation settles it.
    try:
        version_result = runner.run(
            [resolved_path, "version"], timeout_seconds=timeout_seconds
        )
    except InstrumentsProcessError as exc:
        return _report(
            XctraceCapability.PROBE_FAILED,
            reason=f"could not execute xctrace: {exc}",
            os_name=resolved_os,
            architecture=resolved_arch,
            checked=tuple(checked),
            xctrace_path=resolved_path,
            requested_template=template,
        )

    checked.append(f"xctrace_version_probe={_describe(version_result)}")
    version_text: str | None = None
    if version_result.returncode == 0 and not version_result.timed_out:
        version_text = parse_version(version_result.stdout)
    else:
        classified = classify_xctrace_failure(version_result.combined_output)
        if classified is not None:
            return _report(
                classified,
                reason=(
                    f"`xctrace version` failed ({_describe(version_result)}) "
                    f"and its output identifies this as {classified.value}."
                ),
                os_name=resolved_os,
                architecture=resolved_arch,
                checked=tuple(checked),
                xctrace_path=resolved_path,
                requested_template=template,
            )
        # `version` is not in the documented subcommand list, so a
        # failure here is not conclusive on its own. Fall through to the
        # documented `list templates` probe rather than guessing.

    try:
        templates_result = runner.run(
            [resolved_path, "list", "templates"], timeout_seconds=timeout_seconds
        )
    except InstrumentsProcessError as exc:
        return _report(
            XctraceCapability.PROBE_FAILED,
            reason=f"could not execute `xctrace list templates`: {exc}",
            os_name=resolved_os,
            architecture=resolved_arch,
            checked=tuple(checked),
            xctrace_path=resolved_path,
            xctrace_version=version_text,
            requested_template=template,
        )

    checked.append(f"xctrace_list_templates={_describe(templates_result)}")
    if templates_result.timed_out or templates_result.returncode != 0:
        classified = classify_xctrace_failure(templates_result.combined_output)
        capability = (
            XctraceCapability.PROBE_FAILED if classified is None else classified
        )
        reason = f"`xctrace list templates` failed ({_describe(templates_result)})"
        if classified is None:
            reason += (
                " and its output matched no known cause. The failure is "
                "reported as-is rather than guessed at."
            )
        else:
            reason += f" and its output identifies this as {classified.value}."
        return _report(
            capability,
            reason=reason + ".",
            os_name=resolved_os,
            architecture=resolved_arch,
            checked=tuple(checked),
            xctrace_path=resolved_path,
            xctrace_version=version_text,
            requested_template=template,
        )

    available = parse_templates(templates_result.stdout)
    checked.append(f"template_count={len(available)}")

    if template is not None and template not in available:
        return _report(
            XctraceCapability.TEMPLATE_UNAVAILABLE,
            reason=(
                f"xctrace works, but the template {template!r} is not "
                f"installed. Run `xctrace list templates` to see the "
                f"{len(available)} templates this Xcode provides."
            ),
            os_name=resolved_os,
            architecture=resolved_arch,
            checked=tuple(checked),
            xctrace_path=resolved_path,
            xctrace_version=version_text,
            requested_template=template,
            available_templates=available,
        )

    return _report(
        XctraceCapability.SUPPORTED,
        reason=(
            "xctrace responded to `list templates`"
            + (
                f" and provides the template {template!r}."
                if template is not None
                else "."
            )
        ),
        os_name=resolved_os,
        architecture=resolved_arch,
        checked=tuple(checked),
        xctrace_path=resolved_path,
        xctrace_version=version_text,
        requested_template=template,
        available_templates=available,
    )
