"""Turn a parsed trace into a timed list of requests.

Selection happens in a fixed order:

1. Filters on BurstGPT ``Model`` and ``Log Type`` (exact match).
2. A window ``[start_s, end_s)`` in source seconds measured from the first
   data row of the file.
3. ``max_requests``, the first N requests left.

The first selected request goes out at offset 0. Each later request goes out
at ``(source arrival - first selected arrival) * time_scale``, so a scale of
2.0 replays twice as slowly and 0.5 twice as fast. Requests that share a
source timestamp keep sharing their offset.

A multi-turn trace is selected by whole sessions: the window applies to each
session's first turn and ``max_sessions`` keeps the first N sessions.
``max_requests`` is refused, since it would cut a session part way. Only a
session's first turn has a planned offset. A later turn is sent after the
previous turn of its session completes, plus its scaled ``delay``, or not
before its scaled ``timestamp``. Its prompt resends the earlier turns, with a
synthetic stand-in for each assistant answer.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..schema import MetricProvenance
from .artifact import (
    SCHEDULE_KIND,
    SCHEMA_VERSION,
    TOKENIZER_EXACT,
    TRACE_DECLARED,
    read_document,
    render,
    seal,
    sha256_text,
    tool_record,
    write_text,
)
from .formats import (
    AIPERF_MULTI_TURN,
    AIPERF_UPSTREAM,
    BURSTGPT,
    MOONCAKE,
    OTEL_GENAI,
    SESSION_FORMATS,
    SESSION_UPSTREAMS,
    WEKA,
    ParsedTrace,
    TraceEntry,
)
from .prompts import (
    WORD_LIST_SHA256,
    WORD_VOCABULARY,
    PromptBuilder,
    Vocabulary,
    word_count,
)
from .tokenizer import ReplayTokenizer, TokenUnits, count_tokens, fit_units

History = tuple[tuple[str, str], ...]

_MAX_TIME_SCALE = 1e6
_OFFSET_DECIMALS = 6
_TOKENIZER_CHUNK = 64


class ScheduleError(ValueError):
    pass


@dataclass(frozen=True)
class Selection:
    time_scale: float = 1.0
    start_s: float | None = None
    end_s: float | None = None
    max_requests: int | None = None
    models: tuple[str, ...] = ()
    log_types: tuple[str, ...] = ()
    max_sessions: int | None = None

    def validate(self, trace_format: str) -> None:
        if not math.isfinite(self.time_scale) or not (
            0 < self.time_scale <= _MAX_TIME_SCALE
        ):
            raise ScheduleError(
                f"--time-scale must be greater than 0 and at most {_MAX_TIME_SCALE:g}"
            )
        for name, value in (("--start-s", self.start_s), ("--end-s", self.end_s)):
            if value is not None and (not math.isfinite(value) or value < 0):
                raise ScheduleError(f"{name} must be a finite number of seconds >= 0")
        if (
            self.start_s is not None
            and self.end_s is not None
            and self.end_s <= self.start_s
        ):
            raise ScheduleError("--end-s must be greater than --start-s")
        if self.max_requests is not None and self.max_requests < 1:
            raise ScheduleError("--max-requests must be at least 1")
        if trace_format in SESSION_FORMATS:
            if self.max_requests is not None:
                raise ScheduleError(
                    "--max-requests would cut a session part way; use "
                    "--max-sessions for a multi-turn trace"
                )
            if self.max_sessions is not None and self.max_sessions < 1:
                raise ScheduleError("--max-sessions must be at least 1")
        elif self.max_sessions is not None:
            raise ScheduleError("--max-sessions applies to multi-turn traces only")
        if (self.models or self.log_types) and trace_format != BURSTGPT:
            raise ScheduleError("--model and --log-type filter BurstGPT traces only")

    def to_dict(self, trace_format: str | None = None) -> dict[str, Any]:
        if trace_format in SESSION_FORMATS:
            return {
                "order": ["window", "max_sessions"],
                "unit": "session",
                "window_start_s": self.start_s,
                "window_end_s": self.end_s,
                "window_reference": (
                    "first data row of the trace file"
                    if trace_format == AIPERF_MULTI_TURN
                    else "earliest session start in the trace file"
                ),
                "window_applies_to": "each session's first turn",
                "max_sessions": self.max_sessions,
                "time_scale": self.time_scale,
            }
        return {
            "order": ["filters", "window", "max_requests"],
            "models": list(self.models),
            "log_types": list(self.log_types),
            "window_start_s": self.start_s,
            "window_end_s": self.end_s,
            "window_reference": "first data row of the trace file",
            "max_requests": self.max_requests,
            "time_scale": self.time_scale,
        }


@dataclass(frozen=True)
class ScheduledRequest:
    index: int
    source_line: int
    source_arrival_s: float
    arrival_offset_s: float | None
    """Planned send offset. ``None`` for a later turn of a session, whose
    send time depends on when the previous turn completes."""
    target_output_tokens: int
    target_input_tokens: int | None
    estimated_prompt_tokens: int
    prompt_sha256: str
    hash_blocks: int | None = None
    reused_blocks: int | None = None
    reusable_tokens: int | None = None
    tokenizer_prompt_tokens: int | None = None
    session: int | None = None
    turn: int | None = None
    delay_s: float | None = None
    not_before_offset_s: float | None = None
    history_turns: int | None = None
    history_sha256: str | None = None

    @property
    def request_id(self) -> str:
        return f"r{self.index:07d}"

    def session_dict(self) -> dict[str, Any]:
        """Multi-turn fields, empty for a single-turn trace."""
        if self.session is None:
            return {}
        return {
            "session": self.session,
            "turn": self.turn,
            "delay_s": self.delay_s,
            "not_before_offset_s": self.not_before_offset_s,
            "history_turns": self.history_turns,
            "history_sha256": self.history_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "source_line": self.source_line,
            "source_arrival_s": self.source_arrival_s,
            "arrival_offset_s": self.arrival_offset_s,
            "target_input_tokens": self.target_input_tokens,
            "target_output_tokens": self.target_output_tokens,
            "max_tokens": self.target_output_tokens,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "tokenizer_prompt_tokens": self.tokenizer_prompt_tokens,
            "prompt_sha256": self.prompt_sha256,
            "hash_blocks": self.hash_blocks,
            "reused_blocks": self.reused_blocks,
            "reusable_tokens_upper_bound": self.reusable_tokens,
            **self.session_dict(),
        }


@dataclass(frozen=True)
class ReplaySchedule:
    trace: ParsedTrace
    selection: Selection
    salt: str
    requests: tuple[ScheduledRequest, ...]
    entries: tuple[TraceEntry, ...]
    """The selected trace rows, aligned with ``requests``."""
    tokenizer: ReplayTokenizer | None = None
    units: TokenUnits | None = None

    @property
    def vocabulary(self) -> Vocabulary:
        return WORD_VOCABULARY if self.units is None else self.units.vocabulary

    def prompt_builder(self) -> PromptBuilder:
        return PromptBuilder(
            salt=self.salt,
            trace_sha256=self.trace.sha256,
            block_size=self.trace.block_size,
            vocabulary=self.vocabulary,
        )

    def prompts(self) -> Iterator[str]:
        """Regenerate every prompt in order. Text is never stored."""
        builder = self.prompt_builder()
        for entry in self.entries:
            yield _prompt(builder, entry)

    def prompt_at(self, builder: PromptBuilder, index: int) -> str:
        """Regenerate the prompt for request ``index`` with a shared builder."""
        return _prompt(builder, self.entries[index])

    def history_at(self, builder: PromptBuilder, index: int) -> History:
        """Regenerate the earlier turns sent before request ``index``."""
        if not self.threads_history:
            return ()
        return _history(builder, self.entries, index)

    @property
    def multi_turn(self) -> bool:
        """Requests form sessions whose later turns wait for completions."""
        return self.trace.format in SESSION_FORMATS

    @property
    def threads_history(self) -> bool:
        """Earlier turns are resent as chat messages before each later turn.

        Only an AIPerf conversation holds per-turn deltas. A Weka or OTel
        request already declares its whole context length, so it is sent as
        one prompt of that length.
        """
        return self.trace.format == AIPERF_MULTI_TURN

    @property
    def offsets_s(self) -> tuple[float, ...]:
        if self.multi_turn:
            raise ScheduleError("a multi-turn schedule has no full list of offsets")
        return tuple(_offset(request) for request in self.requests)

    def summary(self) -> dict[str, Any]:
        requests = self.requests
        first = self.entries[0].arrival_s
        span = self.entries[-1].arrival_s - first
        declared = [r.target_input_tokens for r in requests]
        input_total = (
            None if any(v is None for v in declared) else sum(v or 0 for v in declared)
        )
        summary: dict[str, Any] = {
            "requests": len(requests),
            "first_selected_source_s": first,
            "source_span_s": span,
            "planned_duration_s": max(
                r.arrival_offset_s for r in requests if r.arrival_offset_s is not None
            ),
            "planned_duration_provenance": MetricProvenance.DERIVED.value,
            "target_input_tokens_total": input_total,
            "target_output_tokens_total": sum(r.target_output_tokens for r in requests),
            "target_provenance": TRACE_DECLARED,
            "estimated_prompt_tokens_total": sum(
                r.estimated_prompt_tokens for r in requests
            ),
            "estimated_provenance": MetricProvenance.ESTIMATED.value,
        }
        if self.tokenizer is not None:
            summary["tokenizer_prompt_tokens_total"] = sum(
                r.tokenizer_prompt_tokens or 0 for r in requests
            )
            summary["tokenizer_provenance"] = TOKENIZER_EXACT
        if self.multi_turn:
            summary["sessions"] = len({r.session for r in requests})
            summary["planned_duration_basis"] = (
                "offset of the last session's first turn; later turns wait for "
                "completions, so the run takes longer"
            )
        if self.trace.format in _PREFIX_FORMATS:
            blocks = sum(r.hash_blocks or 0 for r in requests)
            reused = sum(r.reused_blocks or 0 for r in requests)
            reusable = sum(r.reusable_tokens or 0 for r in requests)
            summary["prefix_reuse"] = {
                "provenance": MetricProvenance.DERIVED.value,
                "basis": _PREFIX_FORMATS[self.trace.format],
                "blocks_total": blocks,
                "blocks_reused": reused,
                "unique_blocks": blocks - reused,
                "reusable_tokens_upper_bound": reusable,
                "reusable_fraction_upper_bound": (
                    reusable / input_total if input_total else None
                ),
            }
        return summary

    def to_document(self) -> dict[str, Any]:
        document: dict[str, Any] = {
            "kind": SCHEDULE_KIND,
            "schema_version": SCHEMA_VERSION,
            "generated_by": tool_record(),
            "source": self.trace.source_dict(),
            "selection": self.selection.to_dict(self.trace.format),
            "prompts": self._prompts_record(),
            "output_length": {
                "policy": "max_tokens ceiling",
                "provenance": TRACE_DECLARED,
            },
            "summary": self.summary(),
            "requests": [request.to_dict() for request in self.requests],
        }
        if self.threads_history:
            document["prompts"]["history"] = self._history_record()
        if self.multi_turn:
            document["sessions"] = _session_policy(self.trace.format)
        return seal(document, "schedule_sha256")

    def _prompts_record(self) -> dict[str, Any]:
        vocabulary = self.vocabulary
        record: dict[str, Any] = {
            "method": vocabulary.method,
            "unit": "word",
            "word_list_sha256": WORD_LIST_SHA256,
            "salt": self.salt,
            "block_size": self.trace.block_size,
            "structure": _structure(self.trace.format),
            "length_provenance": MetricProvenance.ESTIMATED.value,
            "note": (
                "One word is not exactly one token and the chat template "
                "adds tokens. Compare provider prompt_tokens at run time."
            ),
        }
        if self.tokenizer is None:
            return record
        record.update(
            unit="token",
            unit_words=len(vocabulary.words),
            unit_words_sha256=vocabulary.words_sha256(),
            leading_space=vocabulary.leading_space,
            length_provenance=TOKENIZER_EXACT,
            tokenizer=self.tokenizer.identity.to_dict(),
            counted="the prompt text alone, encoded without special tokens",
            note=(
                "Each word is one token in the recorded tokenizer and every "
                "prompt was encoded and checked token by token, so lengths and "
                "hash block boundaries are exact for the prompt text. The chat "
                "template and special tokens the server adds are not counted; "
                "compare provider prompt_tokens at run time."
            ),
        )
        return record

    def _history_record(self) -> dict[str, Any]:
        exact = self.tokenizer is not None
        record: dict[str, Any] = {
            "messages": (
                "each earlier user turn's trace text, each followed by a "
                "synthetic assistant turn, then this turn's text"
            ),
            "assistant_method": f"{self.vocabulary.method}:reply",
            "assistant_length_basis": (
                f"the earlier turn's output_length, in {'tokens' if exact else 'words'}"
            ),
            "assistant_length_provenance": TRACE_DECLARED,
            "assistant_token_provenance": (
                TOKENIZER_EXACT if exact else MetricProvenance.ESTIMATED.value
            ),
            "note": (
                "No response text is stored, so the server's real answers are "
                "never resent. AIPerf's default (DELTAS_WITHOUT_RESPONSES) "
                "threads the live answers instead, so prompt lengths after the "
                "first turn differ from an AIPerf run by the gap between real "
                "and declared output lengths."
            ),
        }
        if exact:
            record["counted"] = (
                "every message encoded on its own and summed; the chat template "
                "between messages is not counted"
            )
        return record


_SESSION_POLICY: dict[str, Any] = {
    "upstream": dict(AIPERF_UPSTREAM),
    "first_turn": "sent at its scaled timestamp",
    "later_turn": (
        "sent after the previous turn of the session completes, then after "
        "its scaled delay, or not before its scaled timestamp, or at once"
    ),
    "failed_turn": "the session continues, as in AIPerf",
    "delay_cap": None,
    "stop": "turns not yet sent are recorded as not sent",
}


_REQUEST_SESSION_POLICY: dict[str, Any] = {
    "first_turn": "sent at its scaled start",
    "later_turn": (
        "sent after the previous request of the session completes, then after "
        "the scaled gap between the previous request's recorded end and its "
        "own recorded start, or not before its own scaled start when the "
        "requests overlapped or the previous end is not recorded"
    ),
    "context": (
        "each request is sent as one prompt of its declared input length; "
        "earlier turns are not resent as messages"
    ),
    "failed_turn": "the session continues with the next request's planned prompt",
    "delay_cap": None,
    "stop": "turns not yet sent are recorded as not sent",
}

_PREFIX_BASIS = (
    "hash_ids already seen earlier in the selected requests; an "
    "upper bound that assumes an unbounded cache and block-aligned "
    "reuse"
)
_PREFIX_FORMATS = {
    MOONCAKE: _PREFIX_BASIS,
    WEKA: (
        "hash_ids already seen earlier in the same conversation, since Weka "
        "numbers ids per conversation; an upper bound that assumes an "
        "unbounded cache and block-aligned reuse"
    ),
}


def _session_policy(trace_format: str) -> dict[str, Any]:
    if trace_format == AIPERF_MULTI_TURN:
        return _SESSION_POLICY
    return {
        "upstream": json.loads(json.dumps(SESSION_UPSTREAMS[trace_format])),
        **_REQUEST_SESSION_POLICY,
    }


def _offset(request: ScheduledRequest) -> float:
    if request.arrival_offset_s is None:
        raise ScheduleError(f"{request.request_id} has no planned offset")
    return request.arrival_offset_s


def _structure(trace_format: str) -> str:
    if trace_format == MOONCAKE:
        return "hash_ids: each block id maps to fixed text, so shared ids share bytes"
    if trace_format == BURSTGPT:
        return "none: each prompt is seeded by its source line"
    if trace_format == WEKA:
        return (
            "hash_ids: each block id maps to fixed text within its conversation, "
            "so a conversation's shared ids share bytes; tokens past the last "
            "hashed block are seeded by the request"
        )
    if trace_format == OTEL_GENAI:
        return "none: each prompt is seeded by its span; no prefix sharing is inferred"
    return "trace text sent as is"


def _prompt(builder: PromptBuilder, entry: TraceEntry) -> str:
    if entry.text is not None:
        return entry.text
    if entry.input_tokens is None:
        raise ScheduleError(f"line {entry.line} has neither text nor a prompt length")
    if entry.hash_ids is not None:
        if entry.block_scope is None:
            return builder.for_hash_ids(entry.hash_ids, entry.input_tokens)
        return builder.for_hash_ids(
            entry.hash_ids,
            entry.input_tokens,
            scope=entry.block_scope,
            tail_key=f"{entry.line}.{entry.line_turn}",
        )
    if entry.line_turn is not None:
        return builder.for_item(entry.line, entry.line_turn, entry.input_tokens)
    return builder.for_line(entry.line, entry.input_tokens)


def _history(
    builder: PromptBuilder, entries: Sequence[TraceEntry], index: int
) -> History:
    entry = entries[index]
    if not entry.turn:
        return ()
    earlier = entries[index - entry.turn : index]
    if any(e.session != entry.session for e in earlier):
        raise ScheduleError(f"line {entry.line}: earlier turns were not selected")
    history: list[tuple[str, str]] = []
    for prior in earlier:
        history.append(("user", _prompt(builder, prior)))
        history.append(("assistant", _reply(builder, prior)))
    return tuple(history)


def _reply(builder: PromptBuilder, entry: TraceEntry) -> str:
    return builder.for_reply(entry.line, entry.line_turn or 0, entry.output_tokens)


def history_sha256(history: History) -> str:
    """The sha256 recorded for a regenerated history, never the text itself."""
    return sha256_text(
        json.dumps(
            [list(turn) for turn in history], ensure_ascii=False, separators=(",", ":")
        )
    )


def _select(trace: ParsedTrace, selection: Selection) -> list[TraceEntry]:
    entries: Sequence[TraceEntry] = trace.entries
    for values, attribute, flag in (
        (selection.models, "model", "--model"),
        (selection.log_types, "log_type", "--log-type"),
    ):
        if not values:
            continue
        present = {getattr(entry, attribute) for entry in trace.entries}
        absent = sorted(set(values) - present)
        if absent:
            seen = sorted(str(value) for value in present)
            raise ScheduleError(
                f"{flag} {', '.join(absent)} does not appear in the trace "
                f"(values present: {', '.join(seen)})"
            )
        wanted = set(values)
        entries = [entry for entry in entries if getattr(entry, attribute) in wanted]
    origin = trace.origin_s
    start = selection.start_s
    end = selection.end_s
    selected = [
        entry
        for entry in entries
        if (start is None or entry.arrival_s - origin >= start)
        and (end is None or entry.arrival_s - origin < end)
    ]
    if selection.max_requests is not None:
        selected = selected[: selection.max_requests]
    if selection.max_sessions is not None:
        kept: set[int | None] = set()
        for entry in selected:
            if len(kept) == selection.max_sessions and entry.session not in kept:
                break
            kept.add(entry.session)
        selected = [entry for entry in selected if entry.session in kept]
    if not selected:
        raise ScheduleError("the selection is empty; widen the window or filters")
    return selected


def prefix_reuse(
    entries: Sequence[TraceEntry], block_size: int
) -> list[tuple[int, int]]:
    """Leading blocks per entry already seen earlier, and their token count."""
    seen: set[tuple[int | None, int]] = set()
    result: list[tuple[int, int]] = []
    for entry in entries:
        reused = reusable = 0
        scope = entry.block_scope
        keys = [(scope, block) for block in entry.hash_ids or ()]
        input_tokens = entry.input_tokens or 0
        for position, key in enumerate(keys):
            if key not in seen:
                break
            reused += 1
            reusable += min(block_size, input_tokens - position * block_size)
        seen.update(keys)
        result.append((reused, reusable))
    return result


def build_schedule(
    trace: ParsedTrace,
    selection: Selection,
    *,
    salt: str,
    tokenizer: ReplayTokenizer | None = None,
) -> ReplaySchedule:
    selection.validate(trace.format)
    selected = _select(trace, selection)
    units = None if tokenizer is None else fit_units(tokenizer)
    builder = PromptBuilder(
        salt=salt,
        trace_sha256=trace.sha256,
        block_size=trace.block_size,
        vocabulary=WORD_VOCABULARY if units is None else units.vocabulary,
    )
    first = selected[0].arrival_s
    reuse = (
        prefix_reuse(selected, trace.block_size)
        if trace.format in _PREFIX_FORMATS and trace.block_size
        else None
    )
    requests: list[ScheduledRequest] = []
    # Prompts are built a chunk at a time so a tokenizer can encode them in
    # one batch without the whole trace's text being held at once.
    for start in range(0, len(selected), _TOKENIZER_CHUNK):
        chunk = selected[start : start + _TOKENIZER_CHUNK]
        prompts = [_prompt(builder, entry) for entry in chunk]
        counts: Sequence[int | None] = (
            [None] * len(chunk)
            if tokenizer is None or units is None
            else count_tokens(
                tokenizer,
                units,
                prompts,
                [entry.text is None for entry in chunk],
                [entry.line for entry in chunk],
            )
        )
        for index, (entry, prompt, count) in enumerate(
            zip(chunk, prompts, counts, strict=True), start=start
        ):
            requests.append(
                _scheduled(
                    index,
                    entry,
                    prompt,
                    count,
                    first=first,
                    selection=selection,
                    reuse=reuse,
                )
            )
    if trace.format in SESSION_FORMATS:
        requests = _with_sessions(
            builder,
            selected,
            requests,
            first,
            selection,
            tokenizer if trace.format == AIPERF_MULTI_TURN else None,
            units,
            threads_history=trace.format == AIPERF_MULTI_TURN,
        )
    return ReplaySchedule(
        trace=trace,
        selection=selection,
        salt=salt,
        requests=tuple(requests),
        entries=tuple(selected),
        tokenizer=tokenizer,
        units=units,
    )


def _scheduled(
    index: int,
    entry: TraceEntry,
    prompt: str,
    tokenizer_tokens: int | None,
    *,
    first: float,
    selection: Selection,
    reuse: list[tuple[int, int]] | None,
) -> ScheduledRequest:
    hash_blocks = reused = reusable = None
    if reuse is not None and entry.hash_ids is not None:
        hash_blocks = len(entry.hash_ids)
        reused, reusable = reuse[index]
    offset = round((entry.arrival_s - first) * selection.time_scale, _OFFSET_DECIMALS)
    return ScheduledRequest(
        index=index,
        source_line=entry.line,
        source_arrival_s=entry.arrival_s,
        arrival_offset_s=offset,
        target_output_tokens=entry.output_tokens,
        target_input_tokens=entry.input_tokens,
        estimated_prompt_tokens=word_count(prompt),
        prompt_sha256=sha256_text(prompt),
        hash_blocks=hash_blocks,
        reused_blocks=reused,
        reusable_tokens=reusable,
        tokenizer_prompt_tokens=tokenizer_tokens,
    )


def _reply_counts(
    builder: PromptBuilder,
    entries: Sequence[TraceEntry],
    tokenizer: ReplayTokenizer | None,
    units: TokenUnits | None,
) -> list[int | None]:
    """Exact token counts of each turn's synthetic reply, checked token by token."""
    if tokenizer is None or units is None:
        return [None] * len(entries)
    counts: list[int | None] = []
    for start in range(0, len(entries), _TOKENIZER_CHUNK):
        chunk = entries[start : start + _TOKENIZER_CHUNK]
        counts.extend(
            count_tokens(
                tokenizer,
                units,
                [_reply(builder, entry) for entry in chunk],
                [True] * len(chunk),
                [entry.line for entry in chunk],
            )
        )
    return counts


def _with_sessions(
    builder: PromptBuilder,
    entries: Sequence[TraceEntry],
    requests: Sequence[ScheduledRequest],
    first: float,
    selection: Selection,
    tokenizer: ReplayTokenizer | None,
    units: TokenUnits | None,
    *,
    threads_history: bool = True,
) -> list[ScheduledRequest]:
    """Add history, session timing and whole-conversation lengths to each turn."""

    def scaled(seconds: float) -> float:
        return round(seconds * selection.time_scale, _OFFSET_DECIMALS)

    replies = _reply_counts(builder, entries, tokenizer, units)
    result: list[ScheduledRequest] = []
    for index, (entry, request) in enumerate(zip(entries, requests, strict=True)):
        history = _history(builder, entries, index) if threads_history else ()
        earlier = range(index - (entry.turn or 0), index) if history else range(0)
        tokens = request.tokenizer_prompt_tokens
        if tokens is not None:
            # Each message is encoded on its own; the template between them
            # is not counted, as for a single-turn prompt.
            tokens += sum(
                (requests[j].tokenizer_prompt_tokens or 0) + (replies[j] or 0)
                for j in earlier
            )
        result.append(
            replace(
                request,
                arrival_offset_s=None if entry.turn else request.arrival_offset_s,
                estimated_prompt_tokens=request.estimated_prompt_tokens
                + sum(word_count(text) for _, text in history),
                tokenizer_prompt_tokens=tokens,
                session=entry.session,
                turn=entry.turn,
                delay_s=None if entry.delay_s is None else scaled(entry.delay_s),
                not_before_offset_s=(
                    None
                    if entry.not_before_s is None
                    else scaled(entry.not_before_s - first)
                ),
                history_turns=len(history) // 2,
                history_sha256=history_sha256(history) if history else None,
            )
        )
    return result


def write_schedule(path: Path, schedule: ReplaySchedule) -> tuple[str, dict[str, Any]]:
    """Write the schedule artifact; return its file sha256 and the document."""
    document = schedule.to_document()
    return write_text(path, render(document)), document


def read_schedule(path: str | Path) -> dict[str, Any]:
    return read_document(path, SCHEDULE_KIND, "schedule_sha256")
