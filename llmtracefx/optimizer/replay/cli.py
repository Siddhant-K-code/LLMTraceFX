"""``llmtracefx-optimizer replay``: validate, plan and run trace replays."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..collectors.openai_api import (
    APICollectionConfig,
    OpenAIStreamCollectorError,
    UrllibStreamingTransport,
)
from ..load.dispatcher import MAX_CONCURRENCY, DispatchConfigError
from ..schema import MetricProvenance
from ..server_metrics import (
    RUNTIME_CHOICES,
    ServerMetricsError,
    capture_settled_after,
    capture_snapshot,
    diff_snapshots,
    validate_metrics_url,
)
from .artifact import (
    REQUESTS_FILE,
    RESULT_FILE,
    SCHEDULE_FILE,
    TRACE_DECLARED,
    write_text,
)
from .formats import (
    MOONCAKE,
    MOONCAKE_BLOCK_SIZE,
    TRACE_FORMATS,
    ParsedTrace,
    TraceFormatError,
    parse_trace,
)
from .prompts import PromptSaltError, new_salt, validate_salt
from .run import (
    DEFAULT_MAX_DISPATCH_LAG_S,
    EXIT_ERROR,
    SERVER_METRICS_FILE,
    ReplayOptions,
    ReplayRunError,
    build_result,
    check_target,
    dispatch_schedule,
    exit_code,
    failed_server_metrics,
    preflight_schedule,
    render_requests,
    server_metrics_record,
    stop_on_interrupt,
    summary_lines,
    write_result,
    write_server_metrics,
)
from .schedule import (
    ReplaySchedule,
    ScheduleError,
    Selection,
    build_schedule,
    prefix_reuse,
    write_schedule,
)
from .tokenizer import TokenizerError, display_name, load_tokenizer

REPLAY_ARTIFACTS = (SCHEDULE_FILE, REQUESTS_FILE, RESULT_FILE, SERVER_METRICS_FILE)


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
        help=(
            "Keep only the first N requests after filters and window. Refused "
            "for a multi-turn trace, since it would cut a session"
        ),
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Multi-turn only: keep only the first N sessions after the window",
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
    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Make synthetic prompt lengths exact in this tokenizer: a "
            "Hugging Face repo id, a tokenizer.json file, or a directory "
            "holding one. Counts the prompt text only, not the chat template. "
            "Needs the optional extra: uv sync --extra tokenizer"
        ),
    )
    parser.add_argument(
        "--tokenizer-revision",
        default=None,
        help=(
            "Branch, tag or commit for a --tokenizer repo id. The resolved "
            "commit and the tokenizer.json sha256 are recorded either way."
        ),
    )


def add_replay_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Register ``replay`` and its subcommands on the main CLI."""
    replay: argparse.ArgumentParser = subparsers.add_parser(
        "replay",
        help=(
            "Replay Mooncake, BurstGPT or AIPerf single_turn and multi_turn "
            "arrival traces"
        ),
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

    run = commands.add_parser(
        "run",
        help="Send the schedule to an OpenAI-compatible endpoint at trace times",
        # Prefix matching would let "--api-key <secret>" resolve to
        # "--api-key-env" and record the secret as a variable name.
        allow_abbrev=False,
    )
    _add_trace_arguments(run)
    _add_schedule_arguments(run)
    _add_run_arguments(run)
    run.set_defaults(func=_cmd_run)
    return replay


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--endpoint",
        required=True,
        help="Full chat-completions URL, such as http://127.0.0.1:8000/v1/chat/completions",
    )
    parser.add_argument(
        "--model-id", required=True, help="Model ID sent in each request"
    )
    parser.add_argument(
        "--api-key-env",
        required=True,
        help=(
            "Name of the environment variable holding the API key. Only the "
            "name is recorded. For a local server without auth, export any "
            "placeholder value."
        ),
    )
    parser.add_argument(
        "--provider",
        default="openai-compatible",
        help="Short provider label recorded in the result (default openai-compatible)",
    )
    parser.add_argument(
        "--run-id",
        default="replay",
        help="Run id prefix; each request is <run-id>-<request_id>",
    )
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds; no retries are performed",
    )
    parser.add_argument(
        "--max-in-flight",
        type=int,
        default=MAX_CONCURRENCY,
        help=(
            "Most requests open at once (default and limit "
            f"{MAX_CONCURRENCY}). A request that arrives while the cap is full "
            "waits, and the wait is recorded as dispatch lag."
        ),
    )
    parser.add_argument(
        "--max-dispatch-lag-s",
        type=_finite,
        default=DEFAULT_MAX_DISPATCH_LAG_S,
        help=(
            "The replay is faithful only when the p99 gap between a request's "
            f"planned and actual send time is at most this (default "
            f"{DEFAULT_MAX_DISPATCH_LAG_S} s); otherwise it exits 2"
        ),
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help=(
            "Send ignore_eos: true so the server generates the trace's full "
            "output length. A vLLM and SGLang extension; other servers may "
            "reject it."
        ),
    )
    parser.add_argument(
        "--no-stream-usage",
        action="store_true",
        help=(
            "Do not send stream_options.include_usage. Without it vLLM and "
            "SGLang report no token counts, so token fidelity stays empty."
        ),
    )
    parser.add_argument(
        "--allow-remote-load",
        action="store_true",
        help=(
            "Allow an endpoint that is not on this machine. Off by default "
            "because a replay can send thousands of requests."
        ),
    )
    parser.add_argument(
        "--server-metrics-url",
        default=None,
        help=(
            "Optional Prometheus /metrics URL of the server. It is scraped "
            "before and after the replay and the window is written to "
            f"{SERVER_METRICS_FILE}, attributed to the requests sent."
        ),
    )
    parser.add_argument(
        "--server-metrics-runtime", choices=RUNTIME_CHOICES, default="auto"
    )
    parser.add_argument("--server-metrics-assume-isolated", action="store_true")
    parser.add_argument("--server-metrics-settle-timeout", type=float, default=2.0)


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
    if trace.sessions is not None:
        starts = trace.sessions
        turns: dict[int | None, int] = {}
        for entry in entries:
            turns[entry.session] = turns.get(entry.session, 0) + 1
        stats["mean_arrival_rate_per_s"] = None
        stats["sessions"] = starts
        stats["turns_per_session"] = _distribution(list(turns.values()))
        stats["mean_session_start_rate_per_s"] = starts / span if span > 0 else None
        stats["rate_basis"] = (
            "session starts; later turns wait for completions, so their send "
            "times are not in the trace"
        )
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
    if "sessions" in stats:
        turns = stats["turns_per_session"]
        print(
            f"  sessions: {stats['sessions']}, turns per session median "
            f"{turns['median']}, max {turns['max']}"
        )
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
        max_sessions=args.max_sessions,
    )
    tokenizer = None
    if args.tokenizer is not None:
        tokenizer = load_tokenizer(args.tokenizer, args.tokenizer_revision)
    elif args.tokenizer_revision is not None:
        raise TokenizerError("--tokenizer-revision needs --tokenizer")
    return build_schedule(trace, selection, salt=salt, tokenizer=tokenizer)


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
    if "sessions" in summary:
        print(
            f"{summary['sessions']} sessions; the duration covers session starts "
            "only, later turns wait for completions"
        )
    print(f"prompt salt: {schedule.salt}")
    if schedule.tokenizer is not None:
        identity = schedule.tokenizer.identity
        print(
            f"tokenizer: {identity.name}, tokenizer.json sha256 "
            f"{identity.file_sha256}; prompt text lengths are exact, chat "
            "template not counted"
        )
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
    except (
        TraceFormatError,
        ScheduleError,
        PromptSaltError,
        TokenizerError,
        OSError,
    ) as exc:
        return _fail(exc)
    print_schedule_summary(schedule)
    print(f"schedule written to {output_dir / SCHEDULE_FILE}")
    return 0


def _run_argv(args: argparse.Namespace) -> tuple[str, ...]:
    """A credential-free, fully explicit invocation for the record."""
    argv = [
        "llmtracefx-optimizer",
        "replay",
        "run",
        "--format",
        args.trace_format,
        Path(args.trace).name,
        "--endpoint",
        args.endpoint,
        "--model-id",
        args.model_id,
        "--api-key-env",
        args.api_key_env,
        "--provider",
        args.provider,
        "--run-id",
        args.run_id,
        "--block-size",
        str(args.block_size),
        "--time-scale",
        repr(args.time_scale),
        "--request-timeout",
        repr(args.request_timeout),
        "--max-in-flight",
        str(args.max_in_flight),
        "--max-dispatch-lag-s",
        repr(args.max_dispatch_lag_s),
    ]
    for flag, value in (
        ("--start-s", args.start_s),
        ("--end-s", args.end_s),
        ("--max-requests", args.max_requests),
        ("--max-sessions", args.max_sessions),
        ("--model-revision", args.model_revision),
        ("--temperature", args.temperature),
        ("--top-p", args.top_p),
        ("--seed", args.seed),
        (
            "--tokenizer",
            None if args.tokenizer is None else display_name(args.tokenizer),
        ),
        ("--tokenizer-revision", args.tokenizer_revision),
    ):
        if value is not None:
            argv.extend((flag, str(value)))
    for flag, values in (("--model", args.models), ("--log-type", args.log_types)):
        for value in values:
            argv.extend((flag, value))
    for flag, enabled in (
        ("--exclude-failed", args.exclude_failed),
        ("--ignore-eos", args.ignore_eos),
        ("--no-stream-usage", args.no_stream_usage),
        ("--allow-remote-load", args.allow_remote_load),
    ):
        if enabled:
            argv.append(flag)
    return tuple(argv)


def _base_config(args: argparse.Namespace) -> APICollectionConfig:
    return APICollectionConfig(
        run_id=args.run_id,
        provider=args.provider,
        endpoint=args.endpoint,
        model_id=args.model_id,
        # Every request replaces this with its own synthetic prompt.
        prompt="replay",
        output_dir=Path(args.output_dir),
        command_argv=_run_argv(args),
        credential_env_var=args.api_key_env,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        request_timeout_seconds=args.request_timeout,
        model_revision=args.model_revision,
        stream_include_usage=not args.no_stream_usage,
    )


def _detail(args: argparse.Namespace, exc: BaseException) -> str:
    # Imported here because the main CLI imports this module at load time.
    from ..cli import _scrubbed_detail

    return _scrubbed_detail(args.api_key_env, exc)


def _cmd_run(args: argparse.Namespace) -> int:
    environ = os.environ
    output_dir = Path(args.output_dir)
    options = ReplayOptions(
        max_in_flight=args.max_in_flight,
        max_dispatch_lag_s=args.max_dispatch_lag_s,
        ignore_eos=args.ignore_eos,
        allow_remote_load=args.allow_remote_load,
    )
    try:
        if args.max_dispatch_lag_s < 0:
            raise ReplayRunError("--max-dispatch-lag-s must not be negative")
        if not 1 <= args.max_in_flight <= MAX_CONCURRENCY:
            raise ReplayRunError(
                f"--max-in-flight must be between 1 and {MAX_CONCURRENCY}"
            )
        trace = load_trace(args)
        schedule = schedule_from_args(args, trace)
        base = _base_config(args)
        check_target(base.endpoint, options)
        metrics_url = (
            None
            if args.server_metrics_url is None
            else validate_metrics_url(args.server_metrics_url)
        )
        # Every request is built and checked before anything is written or sent.
        preflight_schedule(base, schedule, options, environ=environ)
        prepare_output_dir(output_dir)
        schedule_sha, schedule_document = write_schedule(
            output_dir / SCHEDULE_FILE, schedule
        )
        before = None if metrics_url is None else capture_snapshot(metrics_url)
    except (
        TraceFormatError,
        ScheduleError,
        PromptSaltError,
        TokenizerError,
        ReplayRunError,
        OpenAIStreamCollectorError,
        ServerMetricsError,
        OSError,
        UnicodeError,
    ) as exc:
        return _fail(_detail(args, exc))

    print_schedule_summary(schedule)
    print(f"replaying against {args.model_id}; Ctrl-C stops new requests")
    try:
        with stop_on_interrupt() as stop:
            replay = dispatch_schedule(
                schedule,
                base,
                options,
                transport=UrllibStreamingTransport(),
                environ=environ,
                stop=stop,
            )
    except DispatchConfigError as exc:
        return _fail(_detail(args, exc))

    metrics_failed = False
    server_metrics: dict[str, Any] | None = None
    sent = replay.report.sent_count
    if before is not None and metrics_url is not None:
        try:
            if not sent:
                raise ServerMetricsError("no request was sent, so there is no window")
            after = capture_settled_after(
                before,
                metrics_url,
                expected_requests=sent,
                runtime=args.server_metrics_runtime,
                settle_timeout_seconds=args.server_metrics_settle_timeout,
            )
            evidence = diff_snapshots(
                before,
                after,
                expected_requests=sent,
                runtime=args.server_metrics_runtime,
                assume_isolated=args.server_metrics_assume_isolated,
            )
            evidence_sha = write_server_metrics(
                output_dir / SERVER_METRICS_FILE, evidence
            )
            server_metrics = server_metrics_record(evidence, evidence_sha)
        except (OSError, ServerMetricsError, ValueError) as exc:
            metrics_failed = True
            message = _detail(args, exc)
            server_metrics = failed_server_metrics(message)
            print(f"replay: server metrics not recorded: {message}", file=sys.stderr)

    try:
        requests_text, _ = render_requests(schedule, replay)
        requests_sha = write_text(output_dir / REQUESTS_FILE, requests_text)
        document = build_result(
            schedule,
            replay,
            options,
            base=base,
            environ=environ,
            schedule_file_sha256=schedule_sha,
            schedule_document=schedule_document,
            requests_file_sha256=requests_sha,
            stopped_by_operator=stop.is_set(),
            server_metrics=server_metrics,
        )
        write_result(output_dir / RESULT_FILE, document)
    except (OSError, ValueError) as exc:
        return _fail(_detail(args, exc))

    for line in summary_lines(document):
        print(line)
    print(f"result written to {output_dir / RESULT_FILE}")
    code = exit_code(document)
    if code == 0 and metrics_failed:
        return EXIT_ERROR
    return code
