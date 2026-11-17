"""Send a replay schedule through the shared load dispatcher and record it.

The dispatcher (``load.dispatcher``) owns timing and concurrency, and the
API worker (``load.api_worker``) owns the request and its measurements.
This module only adapts a ``ReplaySchedule`` to them and reduces the
outcome to the replay artifacts:

* prompts are regenerated on the worker thread just before each request,
  so a long trace is never held in memory as text, and each prompt is
  checked against the sha256 the schedule recorded;
* the time spent building each prompt is recorded, because it falls
  between the dispatch timestamp and the moment the request is sent;
* a multi-turn trace goes through the session policy, so each later turn
  waits for the previous turn of its session, and its regenerated history
  is checked against the schedule's sha256 too;
* the verdict says whether the replay kept to the trace's arrival times.
"""

from __future__ import annotations

import json
import math
import signal
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import Any

from ..collectors.openai_api import (
    APICollectionConfig,
    StreamingTransport,
    build_request_plan,
)
from ..load.api_worker import (
    ROW_FIELD_PROVENANCE,
    ApiRequestSpec,
    RequestRow,
    make_api_sender,
    preflight,
    request_rows,
)
from ..load.cli import is_loopback_endpoint
from ..load.dispatcher import (
    MAX_CONCURRENCY,
    DispatchContext,
    DispatchReport,
    LoadPolicy,
    SessionArrivals,
    SessionTurn,
    TimedArrivals,
    dispatch,
)
from ..schema import MetricProvenance
from .artifact import (
    REQUESTS_FILE,
    RESULT_KIND,
    SCHEDULE_FILE,
    SCHEMA_VERSION,
    TOKENIZER_EXACT,
    TRACE_DECLARED,
    canonical_json,
    read_document,
    render,
    seal,
    sha256_text,
    tool_record,
    write_text,
)
from .schedule import History, ReplaySchedule, ScheduledRequest, history_sha256

DEFAULT_MAX_DISPATCH_LAG_S = 0.05
PREFLIGHT_CHUNK = 256
SERVER_METRICS_FILE = "server_metrics_evidence.json"

FAITHFUL = "faithful"
LAGGED = "lagged"
INCOMPLETE = "incomplete"

EXIT_FAITHFUL = 0
EXIT_ERROR = 1
EXIT_NOT_FAITHFUL = 2

REPLAY_FIELD_PROVENANCE: dict[str, str] = {
    **ROW_FIELD_PROVENANCE,
    "target_input_tokens": TRACE_DECLARED,
    "target_output_tokens": TRACE_DECLARED,
    "estimated_prompt_tokens": MetricProvenance.ESTIMATED.value,
    "tokenizer_prompt_tokens": TOKENIZER_EXACT,
    "dispatch_lag_s": MetricProvenance.DERIVED.value,
    "prompt_build_ms": MetricProvenance.MEASURED_WALL_CLOCK.value,
}


class ReplayRunError(ValueError):
    pass


class PromptMismatchError(RuntimeError):
    """A regenerated prompt or history did not hash to the schedule's value."""


@dataclass(frozen=True)
class ReplayOptions:
    max_in_flight: int = MAX_CONCURRENCY
    max_dispatch_lag_s: float = DEFAULT_MAX_DISPATCH_LAG_S
    ignore_eos: bool = False
    allow_remote_load: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_in_flight": self.max_in_flight,
            "max_dispatch_lag_s": self.max_dispatch_lag_s,
            "ignore_eos": self.ignore_eos,
            "allow_remote_load": self.allow_remote_load,
        }


def check_target(endpoint: str, options: ReplayOptions) -> None:
    """Refuse a non-loopback target unless the operator opted in."""
    if not options.allow_remote_load and not is_loopback_endpoint(endpoint):
        raise ReplayRunError(
            "the endpoint is not on this machine; a trace replay can send "
            "thousands of requests, so pass --allow-remote-load to replay "
            "against a remote or hosted API"
        )


def _spec(
    request: ScheduledRequest,
    prompt: str,
    options: ReplayOptions,
    history: History = (),
) -> ApiRequestSpec:
    return ApiRequestSpec(
        prompt=prompt,
        history=history,
        run_id_suffix=request.request_id,
        max_tokens=request.target_output_tokens,
        extra_body={"ignore_eos": True} if options.ignore_eos else None,
    )


def preflight_schedule(
    base: APICollectionConfig,
    schedule: ReplaySchedule,
    options: ReplayOptions,
    *,
    environ: Mapping[str, str],
    chunk: int = PREFLIGHT_CHUNK,
) -> None:
    """Run the API worker's preflight over every request, a chunk at a time.

    Run ids are unique by construction (one ``request_id`` per index), so
    checking them per chunk loses nothing, and only one chunk of prompt
    text is alive at once.
    """
    builder = schedule.prompt_builder()
    requests = schedule.requests
    for start in range(0, len(requests), chunk):
        specs = [
            _spec(
                request,
                schedule.prompt_at(builder, request.index),
                options,
                schedule.history_at(builder, request.index),
            )
            for request in requests[start : start + chunk]
        ]
        preflight(base, specs, environ=environ)


def load_policy(schedule: ReplaySchedule, options: ReplayOptions) -> LoadPolicy:
    """Timed arrivals, or the session policy for a multi-turn trace."""
    if not schedule.multi_turn:
        return TimedArrivals(schedule.offsets_s, max_in_flight=options.max_in_flight)
    turns = []
    for request in schedule.requests:
        if request.session is None:
            raise ReplayRunError(f"{request.request_id} has no session")
        offset = (
            request.arrival_offset_s
            if request.arrival_offset_s is not None
            else request.not_before_offset_s
        )
        turns.append(
            SessionTurn(
                session=request.session, offset_s=offset, delay_s=request.delay_s
            )
        )
    return SessionArrivals(tuple(turns), max_in_flight=options.max_in_flight)


@dataclass(frozen=True)
class DispatchedReplay:
    report: DispatchReport[RequestRow]
    rows: list[RequestRow]
    prompt_build_ms: dict[int, float]


def dispatch_schedule(
    schedule: ReplaySchedule,
    base: APICollectionConfig,
    options: ReplayOptions,
    *,
    transport: StreamingTransport,
    environ: Mapping[str, str],
    stop: Any = None,
    clock: Callable[[], float] = time.perf_counter,
    sleep: Callable[[float], None] = time.sleep,
) -> DispatchedReplay:
    """Send every scheduled request at its arrival offset.

    For a multi-turn trace, a later turn is sent once the previous turn of
    its session completes, as the schedule describes.
    """
    api_send = make_api_sender(base, transport=transport, environ=environ)
    builder = schedule.prompt_builder()
    build_ms: dict[int, float] = {}
    lock = threading.Lock()

    def send(request: ScheduledRequest, context: DispatchContext) -> RequestRow:
        started = context.clock()
        prompt = schedule.prompt_at(builder, request.index)
        history = schedule.history_at(builder, request.index)
        elapsed_ms = (context.clock() - started) * 1000.0
        with lock:
            build_ms[request.index] = elapsed_ms
        if sha256_text(prompt) != request.prompt_sha256:
            raise PromptMismatchError(request.request_id)
        if (history_sha256(history) if history else None) != request.history_sha256:
            raise PromptMismatchError(f"{request.request_id} history")
        return api_send(_spec(request, prompt, options, history), context)

    report = dispatch(
        schedule.requests,
        send,
        load_policy(schedule, options),
        clock=clock,
        sleep=sleep,
        stop=stop,
        max_dispatch_lag_ms=options.max_dispatch_lag_s * 1000.0,
    )
    return DispatchedReplay(
        report=report, rows=request_rows(report), prompt_build_ms=build_ms
    )


@contextmanager
def stop_on_interrupt() -> Iterator[threading.Event]:
    """The first Ctrl-C stops new dispatches; a second one aborts as usual."""
    stop = threading.Event()
    if threading.current_thread() is not threading.main_thread():
        yield stop
        return
    previous = signal.getsignal(signal.SIGINT)

    def handler(signum: int, frame: FrameType | None) -> None:
        stop.set()
        signal.signal(signal.SIGINT, previous)

    signal.signal(signal.SIGINT, handler)
    try:
        yield stop
    finally:
        signal.signal(signal.SIGINT, previous)


# --- Result ------------------------------------------------------------------


def request_line(
    request: ScheduledRequest, row: RequestRow, build_ms: float | None
) -> dict[str, Any]:
    """One ``replay_requests.jsonl`` line. Holds no prompt or output text."""
    lag = (
        None
        if row.dispatched_offset_s is None or row.scheduled_offset_s is None
        else row.dispatched_offset_s - row.scheduled_offset_s
    )
    return {
        "request_id": request.request_id,
        "source_line": request.source_line,
        **row.to_dict(),
        "dispatch_lag_s": lag,
        "prompt_build_ms": build_ms,
        "target_input_tokens": request.target_input_tokens,
        "target_output_tokens": request.target_output_tokens,
        "estimated_prompt_tokens": request.estimated_prompt_tokens,
        "tokenizer_prompt_tokens": request.tokenizer_prompt_tokens,
        "planned_prompt_sha256": request.prompt_sha256,
        **(
            {}
            if request.session is None
            else {
                "session": request.session,
                "turn": request.turn,
                "history_turns": request.history_turns,
            }
        ),
    }


def render_requests(
    schedule: ReplaySchedule, replay: DispatchedReplay
) -> tuple[str, int]:
    lines = [
        canonical_json(
            request_line(request, row, replay.prompt_build_ms.get(request.index))
        )
        for request, row in zip(schedule.requests, replay.rows, strict=True)
    ]
    return "".join(line + "\n" for line in lines), len(lines)


def _nearest_rank(ordered: Sequence[float], percentile: float) -> float:
    rank = max(1, math.ceil(percentile / 100.0 * len(ordered)))
    return ordered[rank - 1]


def distribution(values: Sequence[float]) -> dict[str, Any] | None:
    """Nearest-rank percentiles, the same method the dispatcher uses."""
    if not values:
        return None
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50": _nearest_rank(ordered, 50),
        "p90": _nearest_rank(ordered, 90),
        "p99": _nearest_rank(ordered, 99),
        "max": ordered[-1],
    }


def _ratio_summary(pairs: Sequence[tuple[int | None, int | None]]) -> dict[str, Any]:
    """Provider count over the trace's target, for rows that have both."""
    ratios = sorted(
        reported / target
        for reported, target in pairs
        if reported is not None and target
    )
    missing = sum(1 for reported, _ in pairs if reported is None)
    summary: dict[str, Any] = {
        "requests_compared": len(ratios),
        "requests_without_provider_count": missing,
        "provenance": MetricProvenance.DERIVED.value,
    }
    if ratios:
        summary.update(
            ratio_median=ratios[(len(ratios) - 1) // 2],
            ratio_min=ratios[0],
            ratio_max=ratios[-1],
        )
    return summary


def _overhead_summary(pairs: Sequence[tuple[int | None, int | None]]) -> dict[str, Any]:
    """Provider prompt_tokens minus the tokenizer's count of the prompt text."""
    gaps = sorted(
        reported - counted
        for reported, counted in pairs
        if reported is not None and counted is not None
    )
    summary: dict[str, Any] = {
        "basis": (
            "provider prompt_tokens minus the recorded tokenizer's count of the "
            "prompt text; when the server uses the same tokenizer this is the "
            "chat template and special tokens"
        ),
        "requests_compared": len(gaps),
        "provenance": MetricProvenance.DERIVED.value,
    }
    if gaps:
        summary.update(median=gaps[(len(gaps) - 1) // 2], min=gaps[0], max=gaps[-1])
    return summary


def _bare_sha256(value: str) -> str:
    # The API worker writes "sha256:<hex>"; the schedule stores bare hex.
    return value.removeprefix("sha256:")


def fidelity(
    report: DispatchReport[RequestRow], options: ReplayOptions
) -> dict[str, Any]:
    reasons: list[str] = []
    verdict = FAITHFUL
    if report.unsent_count:
        verdict = INCOMPLETE
        reasons.append(
            f"{report.unsent_count} of {len(report.outcomes)} requests were not sent"
        )
    lag = report.diagnostics.dispatch_lag
    threshold_ms = options.max_dispatch_lag_s * 1000.0
    if lag is not None and lag.p99_ms > threshold_ms:
        if verdict == FAITHFUL:
            verdict = LAGGED
        reasons.append(
            f"p99 dispatch lag {lag.p99_ms / 1000.0:.4f} s is over the "
            f"{options.max_dispatch_lag_s:g} s limit"
        )
    if not reasons:
        reasons.append(
            "every request was sent and p99 dispatch lag is within "
            f"{options.max_dispatch_lag_s:g} s"
        )
    return {
        "verdict": verdict,
        "reasons": reasons,
        "max_dispatch_lag_s": options.max_dispatch_lag_s,
        "provenance": MetricProvenance.DERIVED.value,
    }


def session_counts(
    schedule: ReplaySchedule, rows: Sequence[RequestRow]
) -> dict[str, Any] | None:
    """Per-session outcome counts for a multi-turn replay, else ``None``."""
    if not schedule.multi_turn:
        return None
    sessions: dict[int | None, list[RequestRow]] = {}
    for request, row in zip(schedule.requests, rows, strict=True):
        sessions.setdefault(request.session, []).append(row)
    complete = sum(1 for turns in sessions.values() if all(r.sent for r in turns))
    succeeded = sum(
        1 for turns in sessions.values() if all(r.success is True for r in turns)
    )
    after_failure = 0
    for turns in sessions.values():
        failed = False
        for row in turns:
            if failed and row.sent:
                after_failure += 1
            failed = failed or (row.sent and row.success is not True)
    return {
        "sessions": len(sessions),
        "all_turns_sent": complete,
        "all_turns_succeeded": succeeded,
        "turns_sent_after_a_failed_turn": after_failure,
        "note": (
            "A failed turn does not stop its session, as in AIPerf; the next "
            "turn still resends the planned history."
        ),
        "provenance": MetricProvenance.DERIVED.value,
    }


def _target(base: APICollectionConfig, environ: Mapping[str, str]) -> dict[str, Any]:
    plan = build_request_plan(base, environ=environ).to_dict()
    target = {
        key: plan[key]
        for key in (
            "provider",
            "endpoint_origin",
            "endpoint_path",
            "endpoint_query_keys",
            "model_id",
            "model_revision",
            "credential_env_var",
            "request_parameters",
            "provider_extensions",
            "request_timeout_seconds",
            "command",
        )
    }
    target["max_tokens"] = "per request: the trace's output length"
    return target


def build_result(
    schedule: ReplaySchedule,
    replay: DispatchedReplay,
    options: ReplayOptions,
    *,
    base: APICollectionConfig,
    environ: Mapping[str, str],
    schedule_file_sha256: str,
    schedule_document: Mapping[str, Any],
    requests_file_sha256: str,
    stopped_by_operator: bool,
    server_metrics: Mapping[str, Any] | None,
) -> dict[str, Any]:
    report = replay.report
    rows = replay.rows
    sent = [row for row in rows if row.sent]
    succeeded = [row for row in sent if row.success]
    failed = Counter(
        row.failure_category or "unknown" for row in sent if row.success is False
    )
    client_errors = Counter(row.client_error for row in sent if row.client_error)
    by_index = {request.index: request for request in schedule.requests}
    mismatched = sum(
        1
        for row in sent
        if row.prompt_sha256 is not None
        and _bare_sha256(row.prompt_sha256) != by_index[row.index].prompt_sha256
    )
    dispatched = [
        row.dispatched_offset_s for row in sent if row.dispatched_offset_s is not None
    ]
    completed = [
        row.completed_offset_s for row in sent if row.completed_offset_s is not None
    ]
    cached = [
        row.cached_prompt_tokens for row in sent if row.cached_prompt_tokens is not None
    ]
    summary = schedule.summary()
    document: dict[str, Any] = {
        "kind": RESULT_KIND,
        "schema_version": SCHEMA_VERSION,
        "generated_by": tool_record(),
        "schedule": {
            "file": SCHEDULE_FILE,
            "file_sha256": schedule_file_sha256,
            "schedule_sha256": schedule_document["schedule_sha256"],
        },
        "requests_file": {
            "file": REQUESTS_FILE,
            "file_sha256": requests_file_sha256,
            "lines": len(rows),
        },
        "source": schedule.trace.source_dict(),
        "target": _target(base, environ),
        "options": options.to_dict(),
        "dispatch": report.to_dict(),
        "stopped_by_operator": stopped_by_operator,
        "counts": {
            "requests": len(rows),
            "sent": len(sent),
            "not_sent": report.unsent_count,
            "succeeded": len(succeeded),
            "failed_by_category": dict(sorted(failed.items())),
            "client_errors": dict(sorted(client_errors.items())),
            "prompt_sha256_mismatches": mismatched,
        },
        **(
            {}
            if (sessions := session_counts(schedule, rows)) is None
            else {"sessions": sessions}
        ),
        "latency_ms": {
            "basis": "successful requests, nearest-rank percentiles",
            "provenance": MetricProvenance.DERIVED.value,
            "ttft": distribution(
                [row.ttft_ms for row in succeeded if row.ttft_ms is not None]
            ),
            "e2e": distribution(
                [row.e2e_ms for row in succeeded if row.e2e_ms is not None]
            ),
        },
        "timing": {
            "provenance": MetricProvenance.DERIVED.value,
            "planned_duration_s": summary["planned_duration_s"],
            "last_dispatch_offset_s": max(dispatched) if dispatched else None,
            "last_completion_offset_s": max(completed) if completed else None,
            "prompt_build_ms": distribution(list(replay.prompt_build_ms.values())),
        },
        "tokens": {
            "input_vs_trace": _ratio_summary(
                [
                    (row.prompt_tokens, by_index[row.index].target_input_tokens)
                    for row in sent
                ]
            ),
            "input_vs_estimate": _ratio_summary(
                [
                    (row.prompt_tokens, by_index[row.index].estimated_prompt_tokens)
                    for row in sent
                ]
            ),
            "input_minus_tokenizer": (
                None
                if schedule.tokenizer is None
                else _overhead_summary(
                    [
                        (
                            row.prompt_tokens,
                            by_index[row.index].tokenizer_prompt_tokens,
                        )
                        for row in sent
                    ]
                )
            ),
            "output_vs_trace": _ratio_summary(
                [
                    (row.output_tokens, by_index[row.index].target_output_tokens)
                    for row in succeeded
                ]
            ),
            "output_reached_target": sum(
                1
                for row in succeeded
                if row.output_tokens is not None
                and row.output_tokens >= by_index[row.index].target_output_tokens
            ),
            "cached_prompt_tokens": {
                "provenance": MetricProvenance.PROVIDER_REPORTED.value,
                "requests_reporting": len(cached),
                "total": sum(cached) if cached else None,
            },
            "planned_prefix_reuse": summary.get("prefix_reuse"),
        },
        "field_provenance": REPLAY_FIELD_PROVENANCE,
        "fidelity": fidelity(report, options),
        "server_metrics": None if server_metrics is None else dict(server_metrics),
    }
    return seal(document, "result_sha256")


def write_result(path: Path, document: dict[str, Any]) -> str:
    return write_text(path, render(document))


def read_result(path: str | Path) -> dict[str, Any]:
    return read_document(path, RESULT_KIND, "result_sha256")


def exit_code(document: Mapping[str, Any]) -> int:
    counts = document["counts"]
    if counts["sent"] and not counts["succeeded"]:
        return EXIT_ERROR
    if document["fidelity"]["verdict"] != FAITHFUL:
        return EXIT_NOT_FAITHFUL
    return EXIT_FAITHFUL


def summary_lines(document: Mapping[str, Any]) -> list[str]:
    counts = document["counts"]
    lines = [
        f"sent {counts['sent']} of {counts['requests']} requests; "
        f"{counts['succeeded']} succeeded",
    ]
    for category, count in counts["failed_by_category"].items():
        lines.append(f"  failed: {count} {category}")
    for name, count in counts["client_errors"].items():
        lines.append(f"  client error: {count} {name}")
    for name in ("ttft", "e2e"):
        stats = document["latency_ms"][name]
        if stats is not None:
            lines.append(
                f"{name.upper()} ms: p50 {stats['p50']:.1f}, p90 {stats['p90']:.1f}, "
                f"p99 {stats['p99']:.1f} (measured)"
            )
    lag = document["dispatch"]["diagnostics"]["dispatch_lag"]
    if lag is not None:
        lines.append(
            f"dispatch lag ms: p50 {lag['p50_ms']:.2f}, p99 {lag['p99_ms']:.2f}, "
            f"max {lag['max_ms']:.2f}"
        )
    tokens = document["tokens"]["input_vs_trace"]
    if "ratio_median" in tokens:
        lines.append(
            "prompt tokens vs trace: median ratio "
            f"{tokens['ratio_median']:.3f} over {tokens['requests_compared']} requests"
        )
    overhead = document["tokens"].get("input_minus_tokenizer")
    if overhead is not None and "median" in overhead:
        lines.append(
            "prompt tokens minus tokenizer count: median "
            f"{overhead['median']} over {overhead['requests_compared']} requests "
            "(chat template and special tokens if the server uses that tokenizer)"
        )
    cached = document["tokens"]["cached_prompt_tokens"]
    if cached["total"] is not None:
        lines.append(
            f"cached prompt tokens: {cached['total']} "
            f"({cached['requests_reporting']} requests reporting)"
        )
    verdict = document["fidelity"]
    lines.append(f"fidelity: {verdict['verdict']}: {'; '.join(verdict['reasons'])}")
    return lines


def server_metrics_record(
    evidence: Mapping[str, Any], file_sha256: str
) -> dict[str, Any]:
    return {
        "file": SERVER_METRICS_FILE,
        "file_sha256": file_sha256,
        "attribution": evidence.get("attribution"),
    }


def write_server_metrics(path: Path, evidence: Mapping[str, Any]) -> str:
    return write_text(path, json.dumps(evidence, indent=2, allow_nan=False) + "\n")


def failed_server_metrics(message: str) -> dict[str, Any]:
    return {"file": None, "error": message}
