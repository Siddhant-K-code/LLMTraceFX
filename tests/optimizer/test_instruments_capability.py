"""Tests for xctrace capability detection.

Every distinguishable reason a Metal trace cannot be taken gets its own
test, because the whole point of the capability layer is that "it did
not work" is never an acceptable answer on its own.
"""

from __future__ import annotations

import pytest
from _instruments_fakes import (
    COMMAND_LINE_TOOLS_STDERR,
    TEMPLATES_STDOUT,
    VERSION_STDOUT,
    FakeCommandRunner,
    fail,
    ok,
)

from llmtracefx.optimizer.instruments.capability import (
    METAL_SYSTEM_TRACE_TEMPLATE,
    InstrumentsCapabilityError,
    XctraceCapability,
    XctraceCapabilityReport,
    classify_xctrace_failure,
    detect_xctrace_capability,
    parse_templates,
    parse_version,
)

DARWIN = {
    "os_name": "Darwin",
    "architecture": "arm64",
    "path_resolver": lambda: "/usr/bin/xctrace",
}


def detect(runner: FakeCommandRunner, **overrides):
    kwargs = {**DARWIN, **overrides}
    return detect_xctrace_capability(runner=runner, **kwargs)


# --- Platform gating --------------------------------------------------


@pytest.mark.parametrize("os_name", ["Linux", "Windows", "Java"])
def test_non_darwin_is_unsupported_os(os_name):
    report = detect(FakeCommandRunner(), os_name=os_name)
    assert report.capability is XctraceCapability.UNSUPPORTED_OS
    assert report.supported is False
    assert os_name in report.reason


def test_non_arm64_is_reported_separately_from_os():
    report = detect(FakeCommandRunner(), architecture="x86_64")
    assert report.capability is XctraceCapability.UNSUPPORTED_ARCHITECTURE
    assert "x86_64" in report.reason


def test_platform_checks_run_no_subprocess():
    runner = FakeCommandRunner()
    detect(runner, os_name="Linux")
    assert runner.calls == []


# --- Locating xctrace -------------------------------------------------


def test_missing_xctrace_is_not_found():
    report = detect(FakeCommandRunner(), path_resolver=lambda: None)
    assert report.capability is XctraceCapability.XCTRACE_NOT_FOUND
    assert "Install Xcode" in (report.remediation or "")


def test_command_line_tools_shim_is_not_treated_as_available():
    """The shim exists on PATH but refuses to run.

    This is the trap the whole layer exists to avoid: a plain "is
    xctrace on PATH" check passes on a Command Line Tools only machine.
    """
    runner = FakeCommandRunner(
        version=fail(
            ("/usr/bin/xctrace", "version"),
            returncode=1,
            stderr=COMMAND_LINE_TOOLS_STDERR,
        )
    )
    report = detect(runner)
    assert report.capability is XctraceCapability.COMMAND_LINE_TOOLS_ONLY
    assert report.supported is False
    assert "full Xcode" in (report.remediation or "")


def test_license_not_accepted_is_distinguished():
    runner = FakeCommandRunner(
        version=fail(
            ("/usr/bin/xctrace", "version"),
            returncode=69,
            stderr=(
                "You have not agreed to the Xcode license agreements, "
                "please run 'sudo xcodebuild -license'."
            ),
        )
    )
    report = detect(runner)
    assert report.capability is XctraceCapability.LICENSE_NOT_ACCEPTED
    assert "xcodebuild -license accept" in (report.remediation or "")


def test_first_launch_required_is_distinguished():
    runner = FakeCommandRunner(
        version=fail(
            ("/usr/bin/xctrace", "version"),
            returncode=1,
            stderr="Xcode needs to run runFirstLaunch before use.",
        )
    )
    report = detect(runner)
    assert report.capability is XctraceCapability.FIRST_LAUNCH_REQUIRED


def test_remediation_never_tells_the_tool_to_run_sudo_itself():
    runner = FakeCommandRunner(
        version=fail(
            ("/usr/bin/xctrace", "version"),
            returncode=1,
            stderr="Xcode needs to run runFirstLaunch before use.",
        )
    )
    report = detect(runner)
    assert "never invokes sudo on your behalf" in (report.remediation or "")


# --- Undocumented `version` must not be load bearing ------------------


def test_unclassifiable_version_failure_falls_through_to_list_templates():
    """`version` is not a documented subcommand.

    A failure there is inconclusive, so detection must continue to the
    documented `list templates` probe rather than declaring failure.
    """
    runner = FakeCommandRunner(
        version=fail(
            ("/usr/bin/xctrace", "version"), returncode=64, stderr="unknown command"
        )
    )
    report = detect(runner)
    assert report.capability is XctraceCapability.SUPPORTED
    assert report.xctrace_version is None
    assert ("/usr/bin/xctrace", "list", "templates") in runner.calls


def test_unclassifiable_list_templates_failure_is_probe_failed():
    runner = FakeCommandRunner(
        templates=fail(
            ("/usr/bin/xctrace", "list", "templates"),
            returncode=70,
            stderr="something nobody has seen before",
        )
    )
    report = detect(runner)
    assert report.capability is XctraceCapability.PROBE_FAILED
    assert "matched no known cause" in report.reason
    assert "70" in report.reason


def test_probe_failure_does_not_leak_tool_output_into_the_report():
    secret = "/Users/someone/private/path-that-should-not-leak"
    runner = FakeCommandRunner(
        templates=fail(
            ("/usr/bin/xctrace", "list", "templates"), returncode=70, stderr=secret
        )
    )
    report = detect(runner)
    serialized = report.to_json()
    assert secret not in serialized


def test_process_error_becomes_probe_failed():
    runner = FakeCommandRunner(raise_on="version")
    report = detect(runner)
    assert report.capability is XctraceCapability.PROBE_FAILED


def test_timeout_on_list_templates_is_not_reported_as_supported():
    runner = FakeCommandRunner(
        templates=fail(("/usr/bin/xctrace", "list", "templates"), returncode=-1)
    )
    runner.templates = runner.templates.__class__(
        argv=("/usr/bin/xctrace", "list", "templates"),
        returncode=-1,
        stdout="",
        stderr="",
        timed_out=True,
    )
    report = detect(runner)
    assert report.supported is False
    assert "timed out" in report.reason


# --- Templates --------------------------------------------------------


def test_missing_template_is_distinguished_from_broken_xctrace():
    runner = FakeCommandRunner(
        templates=ok(
            ("/usr/bin/xctrace", "list", "templates"),
            "== Standard Templates ==\nTime Profiler\n",
        )
    )
    report = detect(runner)
    assert report.capability is XctraceCapability.TEMPLATE_UNAVAILABLE
    assert METAL_SYSTEM_TRACE_TEMPLATE in report.reason
    assert report.available_templates == ("Time Profiler",)


def test_supported_when_template_present():
    report = detect(FakeCommandRunner())
    assert report.capability is XctraceCapability.SUPPORTED
    assert report.supported is True
    assert report.xctrace_version == "xctrace version 16.0 (17F113)"
    assert METAL_SYSTEM_TRACE_TEMPLATE in report.available_templates


def test_template_none_skips_the_template_requirement():
    runner = FakeCommandRunner(
        templates=ok(
            ("/usr/bin/xctrace", "list", "templates"),
            "== Standard Templates ==\nTime Profiler\n",
        )
    )
    report = detect(runner, template=None)
    assert report.capability is XctraceCapability.SUPPORTED


# --- Output parsing ---------------------------------------------------


def test_parse_templates_drops_headers_and_blank_lines():
    assert parse_templates(TEMPLATES_STDOUT) == (
        "Activity Monitor",
        "Allocations",
        "CPU Counters",
        "Game Performance",
        "Metal System Trace",
        "System Trace",
        "Time Profiler",
    )


def test_parse_templates_strips_indentation_and_deduplicates():
    parsed = parse_templates(
        "== Standard Templates ==\n  Blank\n\n== Custom Templates ==\n  Blank\n"
    )
    assert parsed == ("Blank",)


def test_parse_templates_of_empty_output_is_empty():
    assert parse_templates("") == ()


def test_parse_version_returns_first_non_empty_line():
    assert parse_version(VERSION_STDOUT) == "xctrace version 16.0 (17F113)"
    assert parse_version("\n\n") is None


@pytest.mark.parametrize(
    "text,expected",
    [
        (COMMAND_LINE_TOOLS_STDERR, XctraceCapability.COMMAND_LINE_TOOLS_ONLY),
        ("Operation not permitted", XctraceCapability.PERMISSION_DENIED),
        ("Failed to attach to process", XctraceCapability.PERMISSION_DENIED),
        (
            "requires the get-task-allow entitlement",
            XctraceCapability.PERMISSION_DENIED,
        ),
        ("total nonsense", None),
    ],
)
def test_classify_xctrace_failure(text, expected):
    assert classify_xctrace_failure(text) is expected


# --- Report serialization ---------------------------------------------


def test_report_round_trips_through_dict():
    report = detect(FakeCommandRunner())
    restored = XctraceCapabilityReport.from_dict(report.to_dict())
    assert restored == report


def test_report_rejects_unknown_capability():
    payload = detect(FakeCommandRunner()).to_dict()
    payload["capability"] = "definitely_not_a_state"
    with pytest.raises(InstrumentsCapabilityError, match="unknown capability"):
        XctraceCapabilityReport.from_dict(payload)


def test_report_rejects_missing_field():
    payload = detect(FakeCommandRunner()).to_dict()
    del payload["reason"]
    with pytest.raises(InstrumentsCapabilityError, match="missing required field"):
        XctraceCapabilityReport.from_dict(payload)


def test_report_writes_json_atomically(tmp_path):
    report = detect(FakeCommandRunner())
    target = tmp_path / "nested" / "capability.json"
    report.write_json(target)
    assert (
        XctraceCapabilityReport.from_dict(
            __import__("json").loads(target.read_text(encoding="utf-8"))
        )
        == report
    )
    assert list(target.parent.glob(".*tmp*")) == []
