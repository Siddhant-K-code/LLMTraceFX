"""``gpu-energy`` subcommands: ``capability`` and ``record``."""

from __future__ import annotations

import argparse
import json
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
from .evidence import build_evidence, write_evidence
from .nvml import NvmlBackend, NvmlUnavailable, load_backend
from .window import DEFAULT_INTERVAL_MS, DEFAULT_MAX_SAMPLES, record_window

#: Indirections that tests replace with a fake NVML and a fake clock.
backend_factory: Callable[[], NvmlBackend] = load_backend
run_window: Callable[..., Any] = record_window

EXIT_INTERRUPTED = 130


def _target_command(args: argparse.Namespace) -> list[str]:
    command = list(args.target_command or [])
    if command and command[0] == "--":
        command = command[1:]
    return command


def _cmd_capability(args: argparse.Namespace) -> int:
    capability = check_capability(backend_factory)
    text = json.dumps(capability.as_dict(), indent=2) + "\n"
    if args.output:
        try:
            atomic_write_text(Path(args.output), text)
        except OSError as exc:
            print(f"Failed to write {args.output}: {exc}", file=sys.stderr)
            return 1
        print(f"NVML capability report written to {args.output}")
    else:
        sys.stdout.write(text)
    if not capability.is_supported:
        print(
            f"NVML unavailable ({capability.state}): {capability.message}",
            file=sys.stderr,
        )
    return capability_exit_code(capability)


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
    attribution = evidence["attribution"]
    lines.append(f"Attribution: {attribution['status']}: {attribution['reason']}")
    if evidence["highlights_withheld_reason"]:
        lines.append(f"Highlights withheld: {evidence['highlights_withheld_reason']}")
    return lines


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
    if args.power_sample_interval_ms <= 0:
        print("--power-sample-interval-ms must be positive", file=sys.stderr)
        return 1
    if args.max_samples < 2:
        print("--max-samples must be at least 2", file=sys.stderr)
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
    try:
        write_evidence(output, evidence)
    except (OSError, ValueError) as exc:
        print(f"Failed to write {output}: {exc}", file=sys.stderr)
        return 1
    for line in _summary(evidence):
        print(line)
    print(f"GPU energy evidence written to {output}")
    if record.exit_code != 0:
        print(f"The command exited with code {record.exit_code}", file=sys.stderr)
        return 1
    return 0 if evidence["attribution"]["attributable"] else 2


def add_gpu_energy_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "gpu-energy",
        help=(
            "NVIDIA GPU energy and power through NVML: capability report and "
            "a measured window around one command"
        ),
    )
    commands = parser.add_subparsers(dest="gpu_energy_command", required=True)

    capability = commands.add_parser(
        "capability",
        help=(
            "Report whether NVML can read GPU energy or power here "
            "(exit 0 if supported, 3 if not, 1 on error)"
        ),
    )
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
            "1 on error or a failed command, 3 if NVML is unavailable)"
        ),
    )
    record.add_argument(
        "--output", required=True, help="Evidence JSON path. Must not exist."
    )
    record.add_argument(
        "--device",
        type=int,
        action="append",
        default=[],
        help="NVML device index to measure. Repeatable. Default: every GPU.",
    )
    record.add_argument(
        "--power-sample-interval-ms",
        type=float,
        default=DEFAULT_INTERVAL_MS,
        help="Power polling interval (default: %(default)s)",
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
