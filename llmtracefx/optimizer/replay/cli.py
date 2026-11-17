"""``llmtracefx-optimizer replay``: validate, plan and run trace replays."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..schema import MetricProvenance
from .artifact import REQUESTS_FILE, RESULT_FILE, SCHEDULE_FILE, TRACE_DECLARED
from .formats import (
    MOONCAKE,
    MOONCAKE_BLOCK_SIZE,
    TRACE_FORMATS,
    ParsedTrace,
    TraceFormatError,
    parse_trace,
)
from .prompts import PromptSaltError, new_salt, validate_salt
from .schedule import (
    ReplaySchedule,
    ScheduleError,
    Selection,
    build_schedule,
    prefix_reuse,
    write_schedule,
)

REPLAY_ARTIFACTS = (SCHEDULE_FILE, REQUESTS_FILE, RESULT_FILE)


def _finite(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("must be a finite number")
    return value


def _add_trace_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("trace", help="Trace file to read")
    parser.add_argument(
        "--format",
        required=True,
        choices=TRACE_FORMATS,
        dest="trace_format",
        help="Trace format. Always explicit; the file is never sniffed.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=MOONCAKE_BLOCK_SIZE,
        help=(
            "Tokens per hash_ids block for mooncake traces (default "
            f"{MOONCAKE_BLOCK_SIZE}, as in the published Mooncake traces)"
        ),
    )
    parser.add_argument(
        "--exclude-failed",
        action="store_true",
        help=(
            "BurstGPT only: drop rows with 0 request or 0 response tokens and "
            "count them by reason, instead of refusing the trace"
        ),
    )


def _add_schedule_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--time-scale",
        type=_finite,
        default=1.0,
        help=(
            "Multiply every inter-arrival gap by this factor. 2.0 replays twice "
            "as slowly, 0.5 twice as fast (default 1.0)"
        ),
    )
    parser.add_argument(
        "--start-s",
        type=_finite,
        default=None,
        help="Window start, in source seconds after the trace's first row",
    )
    parser.add_argument(
        "--end-s",
        type=_finite,
        default=None,
        help="Window end (exclusive), in source seconds after the first row",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Keep only the first N requests after filters and window",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        dest="models",
        help="BurstGPT only: keep rows whose Model matches exactly. Repeatable.",
    )
    parser.add_argument(
        "--log-type",
        action="append",
        default=[],
        dest="log_types",
        help="BurstGPT only: keep rows whose Log Type matches exactly. Repeatable.",
    )
    parser.add_argument(
        "--prompt-salt",
        default=None,
        help=(
            "Seed for synthetic prompt text. Default: a fresh random salt per "
            "run, so a replay cannot hit cache entries left by an earlier run. "
            "The salt is always recorded; pass it back to reproduce prompts."
        ),
    )


def add_replay_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register ``replay`` and its subcommands on the main CLI."""
    replay = subparsers.add_parser(
        "replay",
        help="Replay Mooncake, BurstGPT or AIPerf single_turn arrival traces",
        allow_abbrev=False,
    )
    commands = replay.add_subparsers(dest="replay_command", required=True)

    validate = commands.add_parser(
        "validate",
        help="Check a trace file and print its shape. No network access.",
        allow_abbrev=False,
    )
    _add_trace_arguments(validate)
    validate.add_argument(
        "--json", action="store_true", help="Print the summary as JSON"
    )
    validate.set_defaults(func=_cmd_validate)

    plan = commands.add_parser(
        "plan",
        help=f"Write {SCHEDULE_FILE} without sending anything",
        allow_abbrev=False,
    )
    _add_trace_arguments(plan)
    _add_schedule_arguments(plan)
    plan.add_argument("--output-dir", required=True)
    plan.set_defaults(func=_cmd_plan)
    return replay


def _fail(message: object) -> int:
    print(f"replay: {message}", file=sys.stderr)
    return 1


def load_trace(args: argparse.Namespace) -> ParsedTrace:
    return parse_trace(
        Path(args.trace),
        args.trace_format,
        block_size=args.block_size,
        exclude_failed=args.exclude_failed,
    )


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "total": sum(ordered),
        "min": ordered[0],
        "median": ordered[(len(ordered) - 1) // 2],
        "max": ordered[-1],
    }


def trace_stats(trace: ParsedTrace) -> dict[str, Any]:
    entries = trace.entries
    span = entries[-1].arrival_s - entries[0].arrival_s
    stats: dict[str, Any] = {
        "source": trace.source_dict(),
        "span_s": span,
        "mean_arrival_rate_per_s": len(entries) / span if span > 0 else None,
        "rate_provenance": MetricProvenance.DERIVED.value,
        "output_tokens": _distribution([e.output_tokens for e in entries]),
        "token_provenance": TRACE_DECLARED,
    }
    declared = [e.input_tokens for e in entries if e.input_tokens is not None]
    if len(declared) == len(entries):
        stats["input_tokens"] = _distribution(declared)
    if trace.format == MOONCAKE and trace.block_size:
        reuse = prefix_reuse(entries, trace.block_size)
        blocks = sum(len(e.hash_ids or ()) for e in entries)
        reusable = sum(tokens for _, tokens in reuse)
        stats["prefix_reuse"] = {
            "provenance": MetricProvenance.DERIVED.value,
            "basis": "whole trace, unbounded cache, block-aligned reuse",
            "blocks_total": blocks,
            "unique_blocks": blocks - sum(count for count, _ in reuse),
            "reusable_fraction_upper_bound": reusable / sum(declared),
        }
    for name in ("model", "log_type"):
        values = [getattr(e, name) for e in entries if getattr(e, name) is not None]
        if values:
            counts: dict[str, int] = {}
            for value in values:
                counts[value] = counts.get(value, 0) + 1
            stats[f"{name}_counts"] = dict(sorted(counts.items()))
    return stats


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        trace = load_trace(args)
    except TraceFormatError as exc:
        return _fail(exc)
    stats = trace_stats(trace)
    if args.json:
        print(json.dumps(stats, indent=2, allow_nan=False))
        return 0
    source = stats["source"]
    print(f"{trace.file_name}: valid {trace.format} trace")
    print(f"  sha256: {trace.sha256}")
    print(
        f"  rows: {source['rows_total']} read, {source['rows_valid']} replayable, "
        f"span {stats['span_s']:.3f} s"
    )
    for reason, count in source["excluded_rows"].items():
        print(f"  excluded: {count} {reason}")
    if "input_tokens" in stats:
        inputs = stats["input_tokens"]
        print(
            f"  input tokens: median {inputs['median']}, max {inputs['max']}, "
            f"total {inputs['total']}"
        )
    outputs = stats["output_tokens"]
    print(
        f"  output tokens: median {outputs['median']}, max {outputs['max']}, "
        f"total {outputs['total']}"
    )
    if "prefix_reuse" in stats:
        reuse = stats["prefix_reuse"]
        print(
            f"  hash blocks: {reuse['unique_blocks']} unique of "
            f"{reuse['blocks_total']}; reusable prefix tokens at most "
            f"{reuse['reusable_fraction_upper_bound']:.1%} (derived)"
        )
    return 0


def schedule_from_args(args: argparse.Namespace, trace: ParsedTrace) -> ReplaySchedule:
    salt = new_salt() if args.prompt_salt is None else validate_salt(args.prompt_salt)
    selection = Selection(
        time_scale=args.time_scale,
        start_s=args.start_s,
        end_s=args.end_s,
        max_requests=args.max_requests,
        models=tuple(args.models),
        log_types=tuple(args.log_types),
    )
    return build_schedule(trace, selection, salt=salt)


def prepare_output_dir(output_dir: Path) -> None:
    """Refuse a directory that already holds replay evidence from another run."""
    existing = [name for name in REPLAY_ARTIFACTS if (output_dir / name).exists()]
    if existing:
        raise ScheduleError(
            f"{output_dir} already holds {', '.join(existing)}; use a new "
            "--output-dir so evidence from different runs is never mixed"
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def print_schedule_summary(schedule: ReplaySchedule) -> None:
    summary = schedule.summary()
    print(
        f"{summary['requests']} requests over {summary['planned_duration_s']:.3f} s "
        f"(source span {summary['source_span_s']:.3f} s, "
        f"time scale {schedule.selection.time_scale:g})"
    )
    print(f"prompt salt: {schedule.salt}")
    reuse = summary.get("prefix_reuse")
    if reuse is not None and reuse["reusable_fraction_upper_bound"] is not None:
        print(
            "planned prefix reuse: at most "
            f"{reuse['reusable_fraction_upper_bound']:.1%} of prompt tokens (derived)"
        )


def _cmd_plan(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    try:
        trace = load_trace(args)
        schedule = schedule_from_args(args, trace)
        prepare_output_dir(output_dir)
        write_schedule(output_dir / SCHEDULE_FILE, schedule)
    except (TraceFormatError, ScheduleError, PromptSaltError, OSError) as exc:
        return _fail(exc)
    print_schedule_summary(schedule)
    print(f"schedule written to {output_dir / SCHEDULE_FILE}")
    return 0
