"""``gpu-energy`` subcommands: ``capability`` and ``record``."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..collectors._shared import atomic_write_text
from ..schema import utc_now_iso
from .capability import (
    capability_exit_code,
    check_capability,
    probe_capability,
)
from .dcgm_evidence import build_dcgm_evidence
from .dcgm_exporter import (
    Scraper,
    check_exporter_capability,
    endpoint_label,
    http_scraper,
)
from .dcgm_window import (
    DEFAULT_EDGE_TIMEOUT_MS,
    DEFAULT_MIN_RUN_SHARE,
    DEFAULT_SCRAPE_INTERVAL_MS,
    DcgmExporterUnavailable,
    record_dcgm_window,
)
from .evidence import build_evidence, write_evidence
from .nvml import NvmlBackend, NvmlUnavailable, load_backend
from .window import DEFAULT_INTERVAL_MS, DEFAULT_MAX_SAMPLES, record_window

#: Indirections that tests replace with fakes and a fake clock.
backend_factory: Callable[[], NvmlBackend] = load_backend
run_window: Callable[..., Any] = record_window
scraper_factory: Callable[[str], Scraper] = http_scraper
run_dcgm_window: Callable[..., Any] = record_dcgm_window

EXIT_INTERRUPTED = 130

NVML = "nvml"
DCGM_EXPORTER = "dcgm-exporter"
SOURCES = (NVML, DCGM_EXPORTER)

#: ``record`` flags that only one source uses, as ``(dest, flag)``.
_NVML_ONLY = (
    ("power_sample_interval_ms", "--power-sample-interval-ms"),
    ("gpu_pid", "--gpu-pid"),
)
_DCGM_ONLY = (
    ("dcgm_exporter_url", "--dcgm-exporter-url"),
    ("scrape_interval_ms", "--scrape-interval-ms"),
    ("edge_timeout_ms", "--edge-timeout-ms"),
    ("min_run_share", "--min-run-share"),
    ("dcgm_collect_interval_ms", "--dcgm-collect-interval-ms"),
)


def _target_command(args: argparse.Namespace) -> list[str]:
    command = list(args.target_command or [])
    if command and command[0] == "--":
        command = command[1:]
    return command


def _exporter_endpoint(args: argparse.Namespace) -> str | None:
    if not args.dcgm_exporter_url:
        print(
            "--source dcgm-exporter needs --dcgm-exporter-url, for example "
            "http://localhost:9400/metrics",
            file=sys.stderr,
        )
        return None
    try:
        return endpoint_label(args.dcgm_exporter_url)
    except ValueError as exc:
        print(f"--dcgm-exporter-url: {exc}", file=sys.stderr)
        return None


def _cmd_capability(args: argparse.Namespace) -> int:
    capability: Any
    label = "NVML"
    if args.source == DCGM_EXPORTER:
        endpoint = _exporter_endpoint(args)
        if endpoint is None:
            return 1
        label = "dcgm-exporter"
        capability = check_exporter_capability(endpoint, scraper_factory(endpoint))
        code = 0 if capability.is_supported else 3
    else:
        if args.dcgm_exporter_url is not None:
            print(
                "--dcgm-exporter-url applies only to --source dcgm-exporter",
                file=sys.stderr,
            )
            return 1
        capability = check_capability(backend_factory)
        code = capability_exit_code(capability)
    text = json.dumps(capability.as_dict(), indent=2) + "\n"
    if args.output:
        try:
            atomic_write_text(Path(args.output), text)
        except OSError as exc:
            print(f"Failed to write {args.output}: {exc}", file=sys.stderr)
            return 1
        print(f"{label} capability report written to {args.output}")
    else:
        sys.stdout.write(text)
    if not capability.is_supported:
        print(
            f"{label} unavailable ({capability.state}): {capability.message}",
            file=sys.stderr,
        )
    return code


def _format(value: dict[str, Any] | None, refused: str | None = None) -> str:
    if value is None:
        return f"refused ({refused})" if refused else "refused"
    return f"{value['value']:.3f} {value['unit']} ({value['provenance']})"


def _summary(evidence: dict[str, Any]) -> list[str]:
    lines = [f"Window: {evidence['window']['value']:.1f} ms"]
    for device in evidence["devices"]:
        lines.append(
            f"GPU {device['index']} ({device['name']}): "
            f"energy {_format(device['energy'], device['energy_refused_reason'])}, "
            f"average power {_format(device['average_power'])}, "
            "sampled power integral "
            f"{_format(device['sampled_power_integral'], device['sampled_power_refused_reason'])}"
        )
        bracket = device.get("bracket")
        if bracket:
            lines.append(
                f"  The energy delta covers {bracket['min_span_ms'] / 1000:.1f} "
                f"to {bracket['max_span_ms'] / 1000:.1f} s around a "
                f"{bracket['command_ms'] / 1000:.1f} s command"
            )
        if device.get("energy_refused_detail"):
            lines.append(f"  Energy refused: {device['energy_refused_detail']}")
    attribution = evidence["attribution"]
    lines.append(f"Attribution: {attribution['status']}: {attribution['reason']}")
    if evidence["highlights_withheld_reason"]:
        lines.append(f"Highlights withheld: {evidence['highlights_withheld_reason']}")
    return lines


def _misplaced_flags(args: argparse.Namespace) -> str | None:
    flags: tuple[tuple[str, str], ...]
    if args.source == DCGM_EXPORTER:
        flags, owner = _NVML_ONLY, NVML
    else:
        flags, owner = _DCGM_ONLY, DCGM_EXPORTER
    used = [flag for dest, flag in flags if getattr(args, dest) not in (None, [])]
    if used:
        return f"{', '.join(used)} applies only to --source {owner}"
    return None


def _cmd_record(args: argparse.Namespace) -> int:
    command = _target_command(args)
    if not command:
        print(
            "no command to measure. Append it after '--', for example: "
            "gpu-energy record --output gpu.json -- python infer.py",
            file=sys.stderr,
        )
        return 1
    output = Path(args.output)
    if output.exists():
        print(f"refusing to overwrite existing {output}", file=sys.stderr)
        return 1
    misplaced = _misplaced_flags(args)
    if misplaced:
        print(misplaced, file=sys.stderr)
        return 1
    if args.max_samples < 2:
        print("--max-samples must be at least 2", file=sys.stderr)
        return 1
    if args.source == DCGM_EXPORTER:
        return _record_dcgm(args, command, output)
    if args.power_sample_interval_ms is None:
        args.power_sample_interval_ms = DEFAULT_INTERVAL_MS
    if args.power_sample_interval_ms <= 0:
        print("--power-sample-interval-ms must be positive", file=sys.stderr)
        return 1
    if any(pid <= 0 for pid in args.gpu_pid):
        print("--gpu-pid must be a positive process id", file=sys.stderr)
        return 1

    try:
        backend = backend_factory()
        backend.init()
    except NvmlUnavailable as exc:
        print(f"NVML unavailable ({exc.state}): {exc}", file=sys.stderr)
        return 3
    try:
        capability = probe_capability(backend)
        if not capability.is_supported:
            print(
                f"NVML unavailable ({capability.state}): {capability.message}",
                file=sys.stderr,
            )
            return 3
        known = {device.index: device for device in capability.devices}
        selected = sorted(set(args.device)) if args.device else sorted(known)
        unknown = [index for index in selected if index not in known]
        if unknown:
            print(
                f"no GPU with index {', '.join(map(str, unknown))}. "
                f"NVML lists {len(known)} GPU(s)",
                file=sys.stderr,
            )
            return 1
        started_at = utc_now_iso()
        try:
            record = run_window(
                backend,
                selected,
                command,
                interval_ms=args.power_sample_interval_ms,
                max_samples=args.max_samples,
                gpu_pids=args.gpu_pid,
            )
        except OSError as exc:
            print(f"Failed to start {command[0]}: {exc}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("Interrupted. No evidence was written.", file=sys.stderr)
            return EXIT_INTERRUPTED
        ended_at = utc_now_iso()
    finally:
        backend.shutdown()

    evidence = build_evidence(
        record,
        [known[index] for index in selected],
        versions=capability.versions,
        argv=command,
        started_at=started_at,
        ended_at=ended_at,
        assume_isolated=args.assume_isolated,
        gpu_pids=args.gpu_pid,
    )
    return _finish(output, evidence, record.exit_code)


def _finish(output: Path, evidence: dict[str, Any], exit_code: int) -> int:
    try:
        write_evidence(output, evidence)
    except (OSError, ValueError) as exc:
        print(f"Failed to write {output}: {exc}", file=sys.stderr)
        return 1
    for line in _summary(evidence):
        print(line)
    print(f"GPU energy evidence written to {output}")
    if exit_code != 0:
        print(f"The command exited with code {exit_code}", file=sys.stderr)
        return 1
    return 0 if evidence["attribution"]["attributable"] else 2


def _record_dcgm(args: argparse.Namespace, command: list[str], output: Path) -> int:
    endpoint = _exporter_endpoint(args)
    if endpoint is None:
        return 1
    scrape_interval_ms = (
        DEFAULT_SCRAPE_INTERVAL_MS
        if args.scrape_interval_ms is None
        else args.scrape_interval_ms
    )
    edge_timeout_ms = (
        DEFAULT_EDGE_TIMEOUT_MS
        if args.edge_timeout_ms is None
        else args.edge_timeout_ms
    )
    min_run_share = (
        DEFAULT_MIN_RUN_SHARE if args.min_run_share is None else args.min_run_share
    )
    stated = args.dcgm_collect_interval_ms
    for flag, value in (
        ("--scrape-interval-ms", scrape_interval_ms),
        ("--edge-timeout-ms", edge_timeout_ms),
        ("--dcgm-collect-interval-ms", stated),
    ):
        if value is not None and not (math.isfinite(value) and value > 0):
            print(f"{flag} must be a positive number", file=sys.stderr)
            return 1
    if not 0 < min_run_share <= 1:
        print("--min-run-share must be above 0 and at most 1", file=sys.stderr)
        return 1

    scraper = scraper_factory(endpoint)
    capability = check_exporter_capability(endpoint, scraper)
    if not capability.is_supported:
        print(
            f"dcgm-exporter unavailable ({capability.state}): {capability.message}",
            file=sys.stderr,
        )
        return 3
    known = {device["index"]: device for device in capability.devices}
    selected = sorted(set(args.device)) if args.device else sorted(known)
    unknown = [index for index in selected if index not in known]
    if unknown:
        print(
            f"no GPU with index {', '.join(map(str, unknown))}. "
            f"dcgm-exporter lists {len(known)} GPU(s)",
            file=sys.stderr,
        )
        return 1
    started_at = utc_now_iso()
    try:
        record = run_dcgm_window(
            scraper,
            selected,
            command,
            scrape_interval_ms=scrape_interval_ms,
            edge_timeout_ms=edge_timeout_ms,
            max_samples=args.max_samples,
        )
    except DcgmExporterUnavailable as exc:
        print(f"dcgm-exporter cannot bound a window: {exc}", file=sys.stderr)
        return 3
    except OSError as exc:
        print(f"Failed to start {command[0]}: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted. No evidence was written.", file=sys.stderr)
        return EXIT_INTERRUPTED
    ended_at = utc_now_iso()
    evidence = build_dcgm_evidence(
        record,
        endpoint=endpoint,
        argv=command,
        started_at=started_at,
        ended_at=ended_at,
        min_run_share=min_run_share,
        stated_collect_interval_ms=stated,
        assume_isolated=args.assume_isolated,
    )
    return _finish(output, evidence, record.exit_code)


def _add_source(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        choices=SOURCES,
        default=NVML,
        help=(
            "Where readings come from: NVML on this host, or an NVIDIA "
            "dcgm-exporter /metrics endpoint (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--dcgm-exporter-url",
        default=None,
        help="dcgm-exporter metrics URL. Required with --source dcgm-exporter.",
    )


def add_gpu_energy_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "gpu-energy",
        help=(
            "NVIDIA GPU energy and power through NVML or dcgm-exporter: "
            "capability report and a measured window around one command"
        ),
    )
    commands = parser.add_subparsers(dest="gpu_energy_command", required=True)

    capability = commands.add_parser(
        "capability",
        help=(
            "Report whether the source can read GPU energy or power "
            "(exit 0 if supported, 3 if not, 1 on error)"
        ),
    )
    _add_source(capability)
    capability.add_argument(
        "--output",
        default=None,
        help="Write the capability report JSON to this path instead of stdout",
    )
    capability.set_defaults(func=_cmd_capability)

    record = commands.add_parser(
        "record",
        help=(
            "Run one command and record device wide GPU energy and sampled "
            "power during it (exit 0 attributable, 2 not attributable, "
            "1 on error or a failed command, 3 if the source is unavailable)"
        ),
    )
    record.add_argument(
        "--output", required=True, help="Evidence JSON path. Must not exist."
    )
    _add_source(record)
    record.add_argument(
        "--device",
        type=int,
        action="append",
        default=[],
        help=(
            "GPU index to measure (NVML index, or the exporter's gpu label). "
            "Repeatable. Default: every GPU."
        ),
    )
    record.add_argument(
        "--power-sample-interval-ms",
        type=float,
        default=None,
        help=f"NVML power polling interval (default: {DEFAULT_INTERVAL_MS})",
    )
    record.add_argument(
        "--max-samples",
        type=int,
        default=DEFAULT_MAX_SAMPLES,
        help=(
            "Power samples kept per device. Reaching it refuses the sampled "
            "values (default: %(default)s)"
        ),
    )
    record.add_argument(
        "--scrape-interval-ms",
        type=float,
        default=None,
        help=(
            "dcgm-exporter scrape interval. Shorter tightens the window bounds "
            f"(default: {DEFAULT_SCRAPE_INTERVAL_MS})"
        ),
    )
    record.add_argument(
        "--edge-timeout-ms",
        type=float,
        default=None,
        help=(
            "How long to wait for the exporter's energy counter to change "
            "before the command and after it "
            f"(default: {DEFAULT_EDGE_TIMEOUT_MS})"
        ),
    )
    record.add_argument(
        "--min-run-share",
        type=float,
        default=None,
        help=(
            "Refuse the dcgm-exporter energy when the command is less than "
            "this share of the longest span the counter delta can cover "
            f"(default: {DEFAULT_MIN_RUN_SHARE})"
        ),
    )
    record.add_argument(
        "--dcgm-collect-interval-ms",
        type=float,
        default=None,
        help=(
            "The exporter's --collect-interval for the energy field, if "
            "known. Recorded as stated by the operator and checked against "
            "the gap between counter changes the scrapes observe. The value "
            "used for the bracket reasoning and its basis are recorded in "
            "power_sampling.collect_interval."
        ),
    )
    record.add_argument(
        "--gpu-pid",
        type=int,
        action="append",
        default=[],
        help=(
            "PID of a process the command drives, such as an inference "
            "server, whose GPU use belongs to this window. Repeatable."
        ),
    )
    record.add_argument(
        "--assume-isolated",
        action="store_true",
        help=(
            "Accept a window whose attribution cannot be verified. Never "
            "overrides a foreign GPU process that was observed."
        ),
    )
    record.add_argument(
        "target_command",
        nargs=argparse.REMAINDER,
        help="-- followed by the command to measure and its arguments",
    )
    record.set_defaults(func=_cmd_record)
