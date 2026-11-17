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
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schema import MetricProvenance
from .artifact import (
    SCHEDULE_KIND,
    SCHEMA_VERSION,
    TRACE_DECLARED,
    read_document,
    render,
    seal,
    sha256_text,
    tool_record,
    write_text,
)
from .formats import BURSTGPT, MOONCAKE, ParsedTrace, TraceEntry
from .prompts import PROMPT_METHOD, WORD_LIST_SHA256, PromptBuilder, word_count

_MAX_TIME_SCALE = 1e6
_OFFSET_DECIMALS = 6


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
        if (self.models or self.log_types) and trace_format != BURSTGPT:
            raise ScheduleError("--model and --log-type filter BurstGPT traces only")

    def to_dict(self) -> dict[str, Any]:
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
    arrival_offset_s: float
    target_output_tokens: int
    target_input_tokens: int | None
    estimated_prompt_tokens: int
    prompt_sha256: str
    hash_blocks: int | None = None
    reused_blocks: int | None = None
    reusable_tokens: int | None = None

    @property
    def request_id(self) -> str:
        return f"r{self.index:07d}"

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
            "prompt_sha256": self.prompt_sha256,
            "hash_blocks": self.hash_blocks,
            "reused_blocks": self.reused_blocks,
            "reusable_tokens_upper_bound": self.reusable_tokens,
        }


@dataclass(frozen=True)
class ReplaySchedule:
    trace: ParsedTrace
    selection: Selection
    salt: str
    requests: tuple[ScheduledRequest, ...]
    entries: tuple[TraceEntry, ...]
    """The selected trace rows, aligned with ``requests``."""

    def prompt_builder(self) -> PromptBuilder:
        return PromptBuilder(
            salt=self.salt,
            trace_sha256=self.trace.sha256,
            block_size=self.trace.block_size,
        )

    def prompts(self) -> Iterator[str]:
        """Regenerate every prompt in order. Text is never stored."""
        builder = self.prompt_builder()
        for entry in self.entries:
            yield _prompt(builder, entry)

    @property
    def offsets_s(self) -> tuple[float, ...]:
        return tuple(request.arrival_offset_s for request in self.requests)

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
            "planned_duration_s": requests[-1].arrival_offset_s,
            "planned_duration_provenance": MetricProvenance.DERIVED.value,
            "target_input_tokens_total": input_total,
            "target_output_tokens_total": sum(r.target_output_tokens for r in requests),
            "target_provenance": TRACE_DECLARED,
            "estimated_prompt_tokens_total": sum(
                r.estimated_prompt_tokens for r in requests
            ),
            "estimated_provenance": MetricProvenance.ESTIMATED.value,
        }
        if self.trace.format == MOONCAKE:
            blocks = sum(r.hash_blocks or 0 for r in requests)
            reused = sum(r.reused_blocks or 0 for r in requests)
            reusable = sum(r.reusable_tokens or 0 for r in requests)
            summary["prefix_reuse"] = {
                "provenance": MetricProvenance.DERIVED.value,
                "basis": (
                    "hash_ids already seen earlier in the selected requests; an "
                    "upper bound that assumes an unbounded cache and block-aligned "
                    "reuse"
                ),
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
        document = {
            "kind": SCHEDULE_KIND,
            "schema_version": SCHEMA_VERSION,
            "generated_by": tool_record(),
            "source": self.trace.source_dict(),
            "selection": self.selection.to_dict(),
            "prompts": {
                "method": PROMPT_METHOD,
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
            },
            "output_length": {
                "policy": "max_tokens ceiling",
                "provenance": TRACE_DECLARED,
            },
            "summary": self.summary(),
            "requests": [request.to_dict() for request in self.requests],
        }
        return seal(document, "schedule_sha256")


def _structure(trace_format: str) -> str:
    if trace_format == MOONCAKE:
        return "hash_ids: each block id maps to fixed text, so shared ids share bytes"
    if trace_format == BURSTGPT:
        return "none: each prompt is seeded by its source line"
    return "trace text sent as is"


def _prompt(builder: PromptBuilder, entry: TraceEntry) -> str:
    if entry.text is not None:
        return entry.text
    if entry.input_tokens is None:
        raise ScheduleError(f"line {entry.line} has neither text nor a prompt length")
    if entry.hash_ids is not None:
        return builder.for_hash_ids(entry.hash_ids, entry.input_tokens)
    return builder.for_line(entry.line, entry.input_tokens)


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
    if not selected:
        raise ScheduleError("the selection is empty; widen the window or filters")
    return selected


def prefix_reuse(
    entries: Sequence[TraceEntry], block_size: int
) -> list[tuple[int, int]]:
    """Leading blocks per entry already seen earlier, and their token count."""
    seen: set[int] = set()
    result: list[tuple[int, int]] = []
    for entry in entries:
        reused = reusable = 0
        hash_ids = entry.hash_ids or ()
        input_tokens = entry.input_tokens or 0
        for position, block in enumerate(hash_ids):
            if block not in seen:
                break
            reused += 1
            reusable += min(block_size, input_tokens - position * block_size)
        seen.update(hash_ids)
        result.append((reused, reusable))
    return result


def build_schedule(
    trace: ParsedTrace, selection: Selection, *, salt: str
) -> ReplaySchedule:
    selection.validate(trace.format)
    selected = _select(trace, selection)
    builder = PromptBuilder(
        salt=salt, trace_sha256=trace.sha256, block_size=trace.block_size
    )
    first = selected[0].arrival_s
    reuse = (
        prefix_reuse(selected, trace.block_size)
        if trace.format == MOONCAKE and trace.block_size
        else None
    )
    requests: list[ScheduledRequest] = []
    for index, entry in enumerate(selected):
        prompt = _prompt(builder, entry)
        hash_blocks = reused = reusable = None
        if reuse is not None and entry.hash_ids is not None:
            hash_blocks = len(entry.hash_ids)
            reused, reusable = reuse[index]
        offset = round((entry.arrival_s - first) * selection.time_scale, _OFFSET_DECIMALS)
        requests.append(
            ScheduledRequest(
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
            )
        )
    return ReplaySchedule(
        trace=trace,
        selection=selection,
        salt=salt,
        requests=tuple(requests),
        entries=tuple(selected),
    )


def write_schedule(path: Path, schedule: ReplaySchedule) -> tuple[str, dict[str, Any]]:
    """Write the schedule artifact; return its file sha256 and the document."""
    document = schedule.to_document()
    return write_text(path, render(document)), document


def read_schedule(path: str | Path) -> dict[str, Any]:
    return read_document(path, SCHEDULE_KIND, "schedule_sha256")
