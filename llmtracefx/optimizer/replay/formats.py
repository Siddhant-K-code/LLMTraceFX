"""Strict parsers for production arrival traces.

Six formats are read, each checked against its upstream definition:

``mooncake``
    kvcache-ai/Mooncake ``FAST25-release/traces/*.jsonl`` and
    ``arxiv-trace/mooncake_trace.jsonl``. One JSON object per line with exactly
    ``timestamp`` (integer milliseconds), ``input_length``, ``output_length``
    and ``hash_ids`` (one id per 512-token block of the prompt).

``burstgpt``
    HPMLL/BurstGPT CSV. Columns are read by header name. ``Timestamp`` is in
    seconds and ``Request tokens`` / ``Response tokens`` are integers. The v2.0
    ``BurstGPT_3.csv`` adds ``Session ID`` and ``Elapsed time``, which are
    recorded as present but not used.

``aiperf-single-turn``
    The AIPerf ``single_turn`` custom dataset JSONL with ``timestamp``
    (milliseconds), ``text`` and ``output_length``. Fields replay cannot
    honour, such as ``delay``, ``session_id`` and multimodal content, are
    refused rather than dropped.

``aiperf-multi-turn``
    The AIPerf ``multi_turn`` custom dataset JSONL (ai-dynamo/aiperf
    ``src/aiperf/dataset/loader/multi_turn.py`` at commit
    ``fc7d61eafa813d53c8c6056de5b64648cb60d843``). One line is
    ``{"session_id": ..., "turns": [...]}``. Lines that share a
    ``session_id`` form one session, with turns in file order; a line with
    no ``session_id`` is a session of its own. Each turn has ``text`` and
    ``output_length``. The first turn of a session needs ``timestamp``; a
    later turn may set ``timestamp`` (not before) or ``delay`` (after the
    previous turn completes), both in milliseconds, as in AIPerf's fixed
    schedule. Sessions are ordered by their first timestamp. Raw session
    ids are never kept, only the session's position in that order.

``weka``
    Claude Code session traces (callanjfox/agentic-coding-analysis
    ``docs/TRACE_FORMAT.md`` at commit
    ``15e88f72f902c7852c330caf2cc11722e5dbc1aa``), as published in the
    ``semianalysisai/cc-traces-weka-*`` datasets. One line is one
    conversation ``{"id", "block_size", "requests": [...], ...}``; each
    request has ``t`` (seconds from the conversation start), ``in``,
    ``out`` and ``hash_ids``. Each conversation is a session. Hash ids are
    numbered per conversation, so a block's text is scoped to its line.
    ``hash_ids`` may hold full blocks only, as the spec says, or a partial
    last block too, as the published corpus does. ``t`` restarts at each
    conversation, so every conversation starts at its first ``t``, near
    zero, as in AIPerf's ``weka_trace`` loader at
    ``ebcdc98ad69388b86564e25bc1a6976fc50051a7``. A later request waits
    for the previous one, then for the recorded idle gap
    ``t - (t_prev + api_time_prev)``, as in that loader. When
    ``api_time_prev`` is missing or the gap is negative, the request is sent
    not before its own ``t`` instead of at once. Nested ``subagent``
    entries are refused.

``otel-genai``
    OTLP/JSON ``TracesData``, one object per line, as the OpenTelemetry
    file exporter and ``export otlp`` write it, with spans that follow the
    GenAI semantic conventions (open-telemetry/semantic-conventions-genai
    ``model/gen-ai/registry.yaml`` at commit
    ``76e0c19e7fe11dc693fe258642363a438af4d3aa``). Spans whose
    ``gen_ai.operation.name`` is ``chat``, ``text_completion`` or
    ``generate_content`` are requests, sized by ``gen_ai.usage.input_tokens``
    and ``gen_ai.usage.output_tokens``. Other spans are counted and skipped.
    Spans that share a trace id form one session, ordered by start time; a
    later span waits for the previous one, then for the recorded gap from
    the previous span's end, so concurrent spans are sent one at a time.
    Failed spans are refused unless ``--exclude-failed`` drops them. Raw
    trace ids are never kept, and message content attributes are never
    read into a prompt or stored.

Mooncake publishes no multi-turn format: its FAST25 ``conversation_trace``
uses the plain four-field schema with no session field, so it replays as a
``mooncake`` trace.

The format is always named by the caller and the file is never sniffed. The
whole file is validated in one streaming pass, including rows a later
selection would skip, so a trace is either accepted as a whole or refused with
the file, line and field that failed. No value is filled in.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from .._artifact_io import reject_non_finite_json_constant

MOONCAKE = "mooncake"
BURSTGPT = "burstgpt"
AIPERF_SINGLE_TURN = "aiperf-single-turn"
AIPERF_MULTI_TURN = "aiperf-multi-turn"
WEKA = "weka"
OTEL_GENAI = "otel-genai"
TRACE_FORMATS = (
    MOONCAKE,
    BURSTGPT,
    AIPERF_SINGLE_TURN,
    AIPERF_MULTI_TURN,
    WEKA,
    OTEL_GENAI,
)
SESSION_FORMATS = frozenset({AIPERF_MULTI_TURN, WEKA, OTEL_GENAI})
"""Formats replayed as sessions: a later request waits for the previous one."""

AIPERF_UPSTREAM = {
    "repository": "ai-dynamo/aiperf",
    "commit": "fc7d61eafa813d53c8c6056de5b64648cb60d843",
    "loader": "src/aiperf/dataset/loader/multi_turn.py",
    "timing": "src/aiperf/timing/strategies/fixed_schedule.py",
}
"""The upstream definition the multi-turn parser and session timing follow."""

WEKA_UPSTREAM = {
    "repository": "callanjfox/agentic-coding-analysis",
    "commit": "15e88f72f902c7852c330caf2cc11722e5dbc1aa",
    "spec": "docs/TRACE_FORMAT.md",
    "timing": {
        "repository": "ai-dynamo/aiperf",
        "commit": "ebcdc98ad69388b86564e25bc1a6976fc50051a7",
        "loader": "src/aiperf/dataset/loader/weka_trace.py",
    },
}
"""The Weka trace spec, and the AIPerf loader whose idle-gap timing is used."""

OTEL_GENAI_UPSTREAM = {
    "repository": "open-telemetry/semantic-conventions-genai",
    "commit": "76e0c19e7fe11dc693fe258642363a438af4d3aa",
    "registry": "model/gen-ai/registry.yaml",
    "encoding": {
        "repository": "open-telemetry/opentelemetry-proto",
        "commit": "b3f75588eb23c5fca62264edd05d382de49beb1a",
        "message": "opentelemetry/proto/trace/v1/trace.proto TracesData",
        "mapping": "OTLP/JSON, one TracesData per line",
    },
}
"""The GenAI attribute registry and the OTLP span encoding that are read."""

SESSION_UPSTREAMS = {
    AIPERF_MULTI_TURN: AIPERF_UPSTREAM,
    WEKA: WEKA_UPSTREAM,
    OTEL_GENAI: OTEL_GENAI_UPSTREAM,
}

MOONCAKE_BLOCK_SIZE = 512
"""Tokens per hash block in the published Mooncake traces."""

_MAX_LINE_BYTES = 16 * 1024 * 1024
_MAX_SESSION_LINE_BYTES = 64 * 1024 * 1024
"""A Weka conversation or an OTLP batch is one line and can be large."""
_MAX_EXACT_INT = 2**53
_MAX_UINT64 = 2**64 - 1
MAX_BLOCK_SIZE = 1 << 16

_MOONCAKE_KEYS = frozenset({"timestamp", "input_length", "output_length", "hash_ids"})

BURSTGPT_REQUIRED_COLUMNS = ("Timestamp", "Request tokens", "Response tokens")
BURSTGPT_OPTIONAL_COLUMNS = (
    "Model",
    "Total tokens",
    "Log Type",
    "Session ID",
    "Elapsed time",
)

EXCLUDED_ZERO_REQUEST = "zero_request_tokens"
EXCLUDED_ZERO_RESPONSE = "zero_response_tokens"
EXCLUDED_ZERO_BOTH = "zero_request_and_response_tokens"

_AIPERF_SUPPORTED_KEYS = frozenset(
    {"type", "timestamp", "text", "output_length", "role"}
)
_AIPERF_UNSUPPORTED_KEYS = {
    "delay": "replay schedules by timestamp only",
    "session_id": "replay sends independent requests with no session ordering",
    "texts": "replay sends one text prompt per request",
    "image": "replay sends text prompts only",
    "images": "replay sends text prompts only",
    "image_uuids": "replay sends text prompts only",
    "audio": "replay sends text prompts only",
    "audios": "replay sends text prompts only",
    "audio_duration_seconds": "replay sends text prompts only",
    "video": "replay sends text prompts only",
    "videos": "replay sends text prompts only",
    "extra": "per-request body fields are not forwarded by replay",
}

_AIPERF_MULTI_TURN_LINE_KEYS = frozenset({"type", "session_id", "turns"})
_AIPERF_MULTI_TURN_TURN_KEYS = frozenset(
    {"type", "text", "output_length", "timestamp", "delay", "role"}
)
_AIPERF_MULTI_TURN_UNSUPPORTED = {
    **{
        key: reason
        for key, reason in _AIPERF_UNSUPPORTED_KEYS.items()
        if key not in ("delay", "session_id")
    },
    "session_id": "a turn's session is the session_id of its line",
}

_WEKA_LINE_KEYS = frozenset(
    {
        "id",
        "models",
        "block_size",
        "hash_id_scope",
        "tool_tokens",
        "system_tokens",
        "requests",
        "_analysis",
        "totals",
    }
)
_WEKA_LINE_USED = frozenset({"id", "block_size", "hash_id_scope", "requests"})
_WEKA_REQUEST_KEYS = frozenset(
    {
        "t",
        "type",
        "model",
        "in",
        "out",
        "hash_ids",
        "input_types",
        "output_types",
        "stop",
        "api_time",
        "ttft",
        "think_time",
    }
)
_WEKA_REQUEST_USED = frozenset({"t", "type", "model", "in", "out", "hash_ids"})
_WEKA_REQUEST_TYPES = ("s", "n")

_OTLP_TRACES_KEYS = frozenset({"resourceSpans"})
_OTLP_RESOURCE_SPANS_KEYS = frozenset({"resource", "scopeSpans", "schemaUrl"})
_OTLP_SCOPE_SPANS_KEYS = frozenset({"scope", "spans", "schemaUrl"})
_OTLP_SPAN_KEYS = frozenset(
    {
        "traceId",
        "spanId",
        "traceState",
        "parentSpanId",
        "flags",
        "name",
        "kind",
        "startTimeUnixNano",
        "endTimeUnixNano",
        "attributes",
        "droppedAttributesCount",
        "events",
        "droppedEventsCount",
        "links",
        "droppedLinksCount",
        "status",
    }
)
_OTLP_ANY_VALUE_KEYS = frozenset(
    {
        "stringValue",
        "boolValue",
        "intValue",
        "doubleValue",
        "arrayValue",
        "kvlistValue",
        "bytesValue",
    }
)
_OTLP_STATUS_ERROR = 2
_OTEL_INFERENCE_OPERATIONS = frozenset({"chat", "text_completion", "generate_content"})
_OTEL_TOKENS = (
    ("input", "gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens"),
    ("output", "gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens"),
)
_OTEL_CONTENT_ATTRIBUTES = frozenset(
    {
        "gen_ai.input.messages",
        "gen_ai.output.messages",
        "gen_ai.system_instructions",
        "gen_ai.tool.definitions",
        "gen_ai.prompt",
        "gen_ai.completion",
    }
)
EXCLUDED_ERROR_SPAN = "error_spans"
_TRACE_ID = re.compile(r"[0-9a-f]{32}")

_DECIMAL = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_INTEGER = re.compile(r"[0-9]+")


class TraceFormatError(ValueError):
    """The trace is malformed or cannot be replayed faithfully."""


@dataclass(frozen=True, slots=True)
class TraceEntry:
    """One validated request from a trace, with its arrival in seconds."""

    line: int
    """1-based physical line in the source file."""
    arrival_s: float
    """Source arrival time in seconds, before any rebasing or scaling. For a
    multi-turn trace this is the session's arrival, the first turn's
    timestamp, for every turn of the session."""
    output_tokens: int
    input_tokens: int | None = None
    """Declared prompt length. ``None`` for text formats, which declare text."""
    hash_ids: tuple[int, ...] | None = None
    text: str | None = None
    model: str | None = None
    log_type: str | None = None
    session: int | None = None
    """Session formats only: the session's position after sorting by arrival."""
    turn: int | None = None
    """Session formats only: 0-based request number within the session."""
    line_turn: int | None = None
    """Session formats only: 0-based position of the request on its line."""
    delay_s: float | None = None
    """Later turns only: wait after the previous turn completes."""
    not_before_s: float | None = None
    """Later turns only: declared timestamp, in source seconds."""
    block_scope: int | None = None
    """Weka only: the line whose hash id namespace ``hash_ids`` belong to."""


@dataclass(frozen=True)
class ParsedTrace:
    format: str
    file_name: str
    sha256: str
    size_bytes: int
    rows_total: int
    """Data rows read, including excluded rows."""
    origin_s: float
    """Arrival time of the first data row in the file, in seconds."""
    entries: tuple[TraceEntry, ...]
    timestamp_unit: str
    block_size: int | None = None
    excluded: dict[str, int] = field(default_factory=dict)
    columns: tuple[str, ...] = ()
    """BurstGPT header columns, in file order."""
    ignored_columns: tuple[str, ...] = ()
    sessions: int | None = None
    """Session formats only: the number of sessions."""
    details: dict[str, Any] = field(default_factory=dict)
    """Format-specific counts recorded with the source, such as skipped spans."""

    def source_dict(self) -> dict[str, Any]:
        """The source description recorded in every replay artifact."""
        source = {
            "format": self.format,
            "file_name": self.file_name,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "timestamp_unit": self.timestamp_unit,
            "block_size": self.block_size,
            "origin_s": self.origin_s,
            "rows_total": self.rows_total,
            "rows_valid": len(self.entries),
            "excluded_rows": dict(sorted(self.excluded.items())),
            "columns": list(self.columns),
            "ignored_columns": list(self.ignored_columns),
        }
        if self.sessions is not None:
            source["sessions"] = self.sessions
            source["upstream"] = json.loads(json.dumps(SESSION_UPSTREAMS[self.format]))
        if self.details:
            source["details"] = json.loads(json.dumps(self.details))
        return source


class _HashingLines:
    """Yield decoded lines while hashing the exact bytes read."""

    def __init__(
        self, handle: IO[bytes], label: str, max_bytes: int = _MAX_LINE_BYTES
    ) -> None:
        self._handle = handle
        self._label = label
        self._max_bytes = max_bytes
        self.digest = hashlib.sha256()
        self.size = 0
        self.line = 0

    def __iter__(self) -> Iterator[str]:
        while True:
            raw = self._handle.readline(self._max_bytes + 1)
            if not raw:
                return
            self.line += 1
            if len(raw) > self._max_bytes:
                raise TraceFormatError(
                    f"{self._label}:{self.line}: line exceeds {self._max_bytes} bytes"
                )
            self.digest.update(raw)
            self.size += len(raw)
            try:
                yield raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise TraceFormatError(
                    f"{self._label}:{self.line}: not valid UTF-8"
                ) from exc


def _open_regular(path: Path) -> IO[bytes]:
    try:
        handle = path.open("rb")
    except OSError as exc:
        raise TraceFormatError(f"{path}: cannot open: {exc.strerror}") from exc
    if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
        handle.close()
        raise TraceFormatError(f"{path}: not a regular file")
    return handle


def parse_trace(
    path: str | Path,
    fmt: str,
    *,
    block_size: int = MOONCAKE_BLOCK_SIZE,
    exclude_failed: bool = False,
) -> ParsedTrace:
    """Read and validate a whole trace file, or raise ``TraceFormatError``."""
    if fmt not in TRACE_FORMATS:
        raise TraceFormatError(
            f"unknown trace format {fmt!r}; expected one of {', '.join(TRACE_FORMATS)}"
        )
    if fmt != MOONCAKE and block_size != MOONCAKE_BLOCK_SIZE:
        raise TraceFormatError(
            "--block-size applies to the mooncake format only"
            + ("; a Weka trace declares its block_size" if fmt == WEKA else "")
        )
    if exclude_failed and fmt not in (BURSTGPT, OTEL_GENAI):
        raise TraceFormatError(
            "--exclude-failed applies to the burstgpt and otel-genai formats only"
        )
    if not 1 <= block_size <= MAX_BLOCK_SIZE:
        raise TraceFormatError(f"block size must be between 1 and {MAX_BLOCK_SIZE}")
    trace_path = Path(path)
    label = trace_path.name
    with _open_regular(trace_path) as handle:
        lines = _HashingLines(
            handle,
            label,
            _MAX_SESSION_LINE_BYTES if fmt in (WEKA, OTEL_GENAI) else _MAX_LINE_BYTES,
        )
        if fmt == MOONCAKE:
            result = _parse_mooncake(lines, label, block_size)
        elif fmt == BURSTGPT:
            result = _parse_burstgpt(lines, label, exclude_failed)
        elif fmt == AIPERF_MULTI_TURN:
            result = _parse_aiperf_multi_turn(lines, label)
        elif fmt == WEKA:
            result = _parse_weka(lines, label)
        elif fmt == OTEL_GENAI:
            result = _parse_otel_genai(lines, label, exclude_failed)
        else:
            result = _parse_aiperf_single_turn(lines, label)
    entries, rows_total, origin_s, extra = result
    if fmt == MOONCAKE:
        extra["block_size"] = block_size
    return ParsedTrace(
        format=fmt,
        file_name=label,
        sha256=lines.digest.hexdigest(),
        size_bytes=lines.size,
        rows_total=rows_total,
        origin_s=origin_s,
        entries=entries,
        timestamp_unit=_TIMESTAMP_UNITS.get(fmt, "ms"),
        **extra,
    )


_TIMESTAMP_UNITS = {BURSTGPT: "s", WEKA: "s", OTEL_GENAI: "ns"}


_ParseResult = tuple[tuple[TraceEntry, ...], int, float, dict[str, Any]]


def _json_rows(
    lines: _HashingLines, label: str
) -> Iterator[tuple[int, dict[str, Any]]]:
    for text in lines:
        line = lines.line
        if not text.strip():
            raise TraceFormatError(f"{label}:{line}: blank line")
        try:
            row = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=reject_non_finite_json_constant,
            )
        except ValueError as exc:
            raise TraceFormatError(f"{label}:{line}: not valid JSON: {exc}") from exc
        if not isinstance(row, dict):
            raise TraceFormatError(f"{label}:{line}: expected a JSON object")
        yield line, row


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _int_field(row: dict[str, Any], key: str, where: str, *, minimum: int) -> int:
    value = row[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TraceFormatError(f"{where}: {key} must be an integer, got {value!r}")
    if value < minimum or value > _MAX_EXACT_INT:
        raise TraceFormatError(
            f"{where}: {key} must be between {minimum} and {_MAX_EXACT_INT}, "
            f"got {value}"
        )
    return value


def _require_non_decreasing(
    arrival_s: float, previous: float | None, where: str, field_name: str
) -> None:
    if previous is not None and arrival_s < previous:
        raise TraceFormatError(
            f"{where}: {field_name} goes backwards; replay requires a "
            "non-decreasing trace"
        )


class _BlockIndex:
    """Check that each hash id names one block, as a shared prefix needs.

    An id must always sit at the same position after the same id, and a
    partial last block must always have the same length.
    """

    def __init__(self) -> None:
        self._position: dict[int, int] = {}
        self._predecessor: dict[int, int | None] = {}
        self._partial_length: dict[int, int] = {}
        self._full_block: set[int] = set()

    def check(
        self, hash_ids: tuple[int, ...], last_length: int, block_size: int, where: str
    ) -> None:
        position = self._position
        predecessor = self._predecessor
        partial_length = self._partial_length
        last = len(hash_ids) - 1
        for index, block in enumerate(hash_ids):
            before = hash_ids[index - 1] if index else None
            seen_at = position.setdefault(block, index)
            if seen_at != index:
                raise TraceFormatError(
                    f"{where}: hash_id {block} is at block {index} here and at "
                    f"block {seen_at} elsewhere, so it cannot name a shared prefix"
                )
            seen_after = predecessor.setdefault(block, before)
            if seen_after != before:
                raise TraceFormatError(
                    f"{where}: hash_id {block} follows {before} here and "
                    f"{seen_after} elsewhere, so it cannot name a shared prefix"
                )
            if index == last and last_length < block_size:
                known = partial_length.setdefault(block, last_length)
                if known != last_length or block in self._full_block:
                    raise TraceFormatError(
                        f"{where}: hash_id {block} names a {last_length}-token "
                        "block here and a block of a different length elsewhere"
                    )
            else:
                if block in partial_length:
                    raise TraceFormatError(
                        f"{where}: hash_id {block} names a full block here and a "
                        f"{partial_length[block]}-token block elsewhere"
                    )
                self._full_block.add(block)


def _hash_id_list(row: dict[str, Any], where: str) -> tuple[int, ...]:
    raw_ids = row["hash_ids"]
    if not isinstance(raw_ids, list):
        raise TraceFormatError(f"{where}: hash_ids must be a list")
    for value in raw_ids:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= _MAX_EXACT_INT
        ):
            raise TraceFormatError(
                f"{where}: hash_ids must hold non-negative integers, got {value!r}"
            )
    return tuple(raw_ids)


def _parse_mooncake(lines: _HashingLines, label: str, block_size: int) -> _ParseResult:
    entries: list[TraceEntry] = []
    blocks = _BlockIndex()
    previous: float | None = None
    for line, row in _json_rows(lines, label):
        where = f"{label}:{line}"
        keys = set(row)
        if keys != _MOONCAKE_KEYS:
            detail = []
            if missing := sorted(_MOONCAKE_KEYS - keys):
                detail.append(f"missing {', '.join(missing)}")
            if unknown := sorted(keys - _MOONCAKE_KEYS):
                detail.append(f"unknown {', '.join(unknown)}")
            raise TraceFormatError(
                f"{where}: a Mooncake row has exactly timestamp, input_length, "
                f"output_length and hash_ids ({'; '.join(detail)})"
            )
        timestamp = _int_field(row, "timestamp", where, minimum=0)
        input_length = _int_field(row, "input_length", where, minimum=1)
        output_length = _int_field(row, "output_length", where, minimum=1)
        hash_ids = _hash_id_list(row, where)
        expected_blocks = -(-input_length // block_size)
        if len(hash_ids) != expected_blocks:
            raise TraceFormatError(
                f"{where}: hash_ids has {len(hash_ids)} blocks but input_length "
                f"{input_length} needs {expected_blocks} blocks of {block_size} "
                "tokens; check --block-size"
            )
        last_length = input_length - (expected_blocks - 1) * block_size
        blocks.check(hash_ids, last_length, block_size, where)
        arrival_s = timestamp / 1000.0
        _require_non_decreasing(arrival_s, previous, where, "timestamp")
        previous = arrival_s
        entries.append(
            TraceEntry(
                line=line,
                arrival_s=arrival_s,
                input_tokens=input_length,
                output_tokens=output_length,
                hash_ids=hash_ids,
            )
        )
    if not entries:
        raise TraceFormatError(f"{label}: the trace has no rows")
    return tuple(entries), len(entries), entries[0].arrival_s, {}


def _parse_burstgpt(
    lines: _HashingLines, label: str, exclude_failed: bool
) -> _ParseResult:
    reader = csv.reader(lines, strict=True)
    try:
        header = next(reader)
    except StopIteration:
        raise TraceFormatError(f"{label}: the file is empty") from None
    except csv.Error as exc:
        raise TraceFormatError(f"{label}:1: not valid CSV: {exc}") from exc
    if len(set(header)) != len(header):
        raise TraceFormatError(f"{label}:1: the header repeats a column name")
    missing = [name for name in BURSTGPT_REQUIRED_COLUMNS if name not in header]
    if missing:
        raise TraceFormatError(
            f"{label}:1: the BurstGPT header must name {', '.join(missing)}; "
            f"found {', '.join(header)}"
        )
    known = set(BURSTGPT_REQUIRED_COLUMNS) | set(BURSTGPT_OPTIONAL_COLUMNS)
    index = {name: position for position, name in enumerate(header)}
    model_at = index.get("Model")
    log_type_at = index.get("Log Type")
    entries: list[TraceEntry] = []
    excluded: dict[str, int] = {}
    first_excluded_line: int | None = None
    rows_total = 0
    origin_s: float | None = None
    previous: float | None = None
    interned: dict[str, str] = {}
    while True:
        try:
            row = next(reader)
        except StopIteration:
            break
        except csv.Error as exc:
            raise TraceFormatError(
                f"{label}:{lines.line}: not valid CSV: {exc}"
            ) from exc
        line = lines.line
        where = f"{label}:{line}"
        if len(row) != len(header):
            raise TraceFormatError(
                f"{where}: expected {len(header)} columns, found {len(row)}"
            )
        rows_total += 1
        timestamp_text = row[index["Timestamp"]]
        if not _DECIMAL.fullmatch(timestamp_text):
            raise TraceFormatError(
                f"{where}: Timestamp must be a non-negative number of seconds, "
                f"got {timestamp_text!r}"
            )
        arrival_s = float(timestamp_text)
        if not math.isfinite(arrival_s) or arrival_s > _MAX_EXACT_INT:
            raise TraceFormatError(f"{where}: Timestamp is out of range")
        _require_non_decreasing(arrival_s, previous, where, "Timestamp")
        previous = arrival_s
        if origin_s is None:
            origin_s = arrival_s
        request = _csv_int(row, index, "Request tokens", where)
        response = _csv_int(row, index, "Response tokens", where)
        if "Total tokens" in index:
            total = _csv_int(row, index, "Total tokens", where)
            if total != request + response:
                raise TraceFormatError(
                    f"{where}: Total tokens {total} is not Request tokens "
                    f"{request} + Response tokens {response}"
                )
        reason = None
        if request == 0 and response == 0:
            reason = EXCLUDED_ZERO_BOTH
        elif response == 0:
            reason = EXCLUDED_ZERO_RESPONSE
        elif request == 0:
            reason = EXCLUDED_ZERO_REQUEST
        if reason is not None:
            excluded[reason] = excluded.get(reason, 0) + 1
            if first_excluded_line is None:
                first_excluded_line = line
            continue
        model = None if model_at is None else row[model_at]
        log_type = None if log_type_at is None else row[log_type_at]
        entries.append(
            TraceEntry(
                line=line,
                arrival_s=arrival_s,
                input_tokens=request,
                output_tokens=response,
                model=None if model is None else interned.setdefault(model, model),
                log_type=(
                    None
                    if log_type is None
                    else interned.setdefault(log_type, log_type)
                ),
            )
        )
    if rows_total == 0 or origin_s is None:
        raise TraceFormatError(f"{label}: the trace has no rows")
    if excluded and not exclude_failed:
        counts = ", ".join(
            f"{count} {name}" for name, count in sorted(excluded.items())
        )
        raise TraceFormatError(
            f"{label}:{first_excluded_line}: the trace has rows with 0 request or "
            f"0 response tokens ({counts}). BurstGPT marks 0 response tokens as "
            "a failed request. Pass --exclude-failed to drop and count them, or "
            "use a BurstGPT_without_fails file"
        )
    if not entries:
        raise TraceFormatError(f"{label}: every row was excluded")
    return (
        tuple(entries),
        rows_total,
        origin_s,
        {
            "excluded": excluded,
            "columns": tuple(header),
            "ignored_columns": tuple(name for name in header if name not in known),
        },
    )


def _csv_int(row: list[str], index: dict[str, int], name: str, where: str) -> int:
    text = row[index[name]]
    if not _INTEGER.fullmatch(text):
        raise TraceFormatError(
            f"{where}: {name} must be a non-negative integer, got {text!r}"
        )
    value = int(text)
    if value > _MAX_EXACT_INT:
        raise TraceFormatError(f"{where}: {name} is out of range")
    return value


def _parse_aiperf_single_turn(lines: _HashingLines, label: str) -> _ParseResult:
    entries: list[TraceEntry] = []
    previous: float | None = None
    for line, row in _json_rows(lines, label):
        where = f"{label}:{line}"
        for key in row:
            if key in _AIPERF_UNSUPPORTED_KEYS:
                raise TraceFormatError(
                    f"{where}: field {key!r} is not supported: "
                    f"{_AIPERF_UNSUPPORTED_KEYS[key]}"
                )
            if key not in _AIPERF_SUPPORTED_KEYS:
                raise TraceFormatError(
                    f"{where}: unknown field {key!r} for an AIPerf single_turn row"
                )
        if row.get("type", "single_turn") != "single_turn":
            raise TraceFormatError(f"{where}: type must be 'single_turn'")
        if row.get("role") not in ("user", None):
            raise TraceFormatError(f"{where}: role must be 'user' when present")
        for required in ("timestamp", "text", "output_length"):
            if row.get(required) is None:
                raise TraceFormatError(f"{where}: {required} is required for replay")
        arrival_s = _milliseconds(row, "timestamp", where)
        text = _text_field(row, where)
        output_length = _int_field(row, "output_length", where, minimum=1)
        _require_non_decreasing(arrival_s, previous, where, "timestamp")
        previous = arrival_s
        entries.append(
            TraceEntry(
                line=line,
                arrival_s=arrival_s,
                output_tokens=output_length,
                text=text,
            )
        )
    if not entries:
        raise TraceFormatError(f"{label}: the trace has no rows")
    return tuple(entries), len(entries), entries[0].arrival_s, {}


def _milliseconds(row: dict[str, Any], key: str, where: str) -> float:
    """A non-negative millisecond value, returned in seconds."""
    value = row[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= _MAX_EXACT_INT
    ):
        raise TraceFormatError(
            f"{where}: {key} must be a non-negative number of milliseconds, "
            f"got {value!r}"
        )
    return float(value) / 1000.0


def _text_field(row: dict[str, Any], where: str) -> str:
    text = row["text"]
    if not isinstance(text, str) or not text.strip():
        raise TraceFormatError(f"{where}: text must be a non-empty string")
    return text


@dataclass(frozen=True, slots=True)
class _Turn:
    line: int
    line_turn: int
    text: str
    output_tokens: int
    timestamp_s: float | None
    delay_s: float | None


def _check_keys(
    row: dict[str, Any],
    where: str,
    supported: frozenset[str],
    unsupported: dict[str, str],
    what: str,
) -> None:
    for key in row:
        if key in unsupported:
            raise TraceFormatError(
                f"{where}: field {key!r} is not supported: {unsupported[key]}"
            )
        if key not in supported:
            raise TraceFormatError(f"{where}: unknown field {key!r} for {what}")


def _multi_turn_turn(raw: Any, line: int, position: int, label: str) -> _Turn:
    where = f"{label}:{line}: turns[{position}]"
    if not isinstance(raw, dict):
        raise TraceFormatError(f"{where}: expected a JSON object")
    _check_keys(
        raw,
        where,
        _AIPERF_MULTI_TURN_TURN_KEYS,
        _AIPERF_MULTI_TURN_UNSUPPORTED,
        "an AIPerf turn",
    )
    if raw.get("type", "single_turn") != "single_turn":
        raise TraceFormatError(f"{where}: type must be 'single_turn'")
    if raw.get("role") not in ("user", None):
        raise TraceFormatError(
            f"{where}: role must be 'user' when present; system and assistant "
            "turns are not replayed"
        )
    for required in ("text", "output_length"):
        if raw.get(required) is None:
            raise TraceFormatError(f"{where}: {required} is required for replay")
    timestamp = (
        None if raw.get("timestamp") is None else _milliseconds(raw, "timestamp", where)
    )
    delay = None if raw.get("delay") is None else _milliseconds(raw, "delay", where)
    if timestamp is not None and delay is not None:
        raise TraceFormatError(f"{where}: a turn sets timestamp or delay, not both")
    return _Turn(
        line=line,
        line_turn=position,
        text=_text_field(raw, where),
        output_tokens=_int_field(raw, "output_length", where, minimum=1),
        timestamp_s=timestamp,
        delay_s=delay,
    )


def _parse_aiperf_multi_turn(lines: _HashingLines, label: str) -> _ParseResult:
    sessions: dict[str | int, list[_Turn]] = {}
    rows_total = 0
    for line, row in _json_rows(lines, label):
        rows_total += 1
        where = f"{label}:{line}"
        _check_keys(
            row,
            where,
            _AIPERF_MULTI_TURN_LINE_KEYS,
            {},
            "an AIPerf multi_turn row",
        )
        if row.get("type", "multi_turn") != "multi_turn":
            raise TraceFormatError(f"{where}: type must be 'multi_turn'")
        session_id = row.get("session_id")
        if session_id is not None and (
            not isinstance(session_id, str) or not session_id
        ):
            raise TraceFormatError(
                f"{where}: session_id must be a non-empty string when present"
            )
        turns = row.get("turns")
        if not isinstance(turns, list) or not turns:
            raise TraceFormatError(f"{where}: turns must be a non-empty list")
        # A line without a session_id is its own session. Its key is the line
        # number, an int, so it can never collide with a string session_id.
        key: str | int = line if session_id is None else session_id
        sessions.setdefault(key, []).extend(
            _multi_turn_turn(turn, line, position, label)
            for position, turn in enumerate(turns)
        )
    if not sessions:
        raise TraceFormatError(f"{label}: the trace has no rows")
    starts: list[tuple[float, list[_Turn]]] = []
    for turns in sessions.values():
        first = turns[0]
        where = f"{label}:{first.line}: turns[{first.line_turn}]"
        if first.timestamp_s is None or first.delay_s is not None:
            raise TraceFormatError(
                f"{where}: the first turn of a session needs a timestamp and no "
                "delay; it is sent at that time"
            )
        previous = first.timestamp_s
        for turn in turns[1:]:
            if turn.timestamp_s is None:
                continue
            _require_non_decreasing(
                turn.timestamp_s,
                previous,
                f"{label}:{turn.line}: turns[{turn.line_turn}]",
                "timestamp within the session",
            )
            previous = turn.timestamp_s
        starts.append((first.timestamp_s, turns))
    # sorted() is stable, so sessions that start together keep file order.
    starts.sort(key=lambda item: item[0])
    entries = [
        TraceEntry(
            line=turn.line,
            arrival_s=start,
            output_tokens=turn.output_tokens,
            text=turn.text,
            session=session,
            turn=number,
            line_turn=turn.line_turn,
            delay_s=None if number == 0 else turn.delay_s,
            not_before_s=None if number == 0 else turn.timestamp_s,
        )
        for session, (start, turns) in enumerate(starts)
        for number, turn in enumerate(turns)
    ]
    return tuple(entries), rows_total, starts[0][0], {"sessions": len(starts)}


@dataclass(frozen=True, slots=True)
class _Request:
    """One request of a session format, before sessions are ordered."""

    line: int
    position: int
    start: int | float
    end: int | float | None
    """Recorded completion in the same unit as ``start``, when known."""
    input_tokens: int
    output_tokens: int
    hash_ids: tuple[int, ...] | None = None
    model: str | None = None
    block_scope: int | None = None


_LATER_DELAY = "delay_after_recorded_end"
_LATER_OVERLAP = "not_before_start_after_overlap"
_LATER_NO_END = "not_before_start_no_recorded_end"


def _session_entries(
    sessions: list[list[_Request]], per_second: int
) -> tuple[list[TraceEntry], dict[str, int]]:
    """Order sessions by first start and time each later request.

    A later request waits for the previous one to complete, then for the
    recorded idle gap between the previous request's end and its own start.
    When the end is unknown, or the requests overlapped, it is sent not
    before its own recorded start instead.
    """
    timing = {_LATER_DELAY: 0, _LATER_OVERLAP: 0, _LATER_NO_END: 0}
    # sorted() is stable, so sessions that start together keep file order.
    ordered = sorted(sessions, key=lambda requests: requests[0].start)
    entries: list[TraceEntry] = []
    for session, requests in enumerate(ordered):
        arrival_s = requests[0].start / per_second
        previous: _Request | None = None
        for number, request in enumerate(requests):
            delay_s = not_before_s = None
            if previous is not None:
                if previous.end is None:
                    timing[_LATER_NO_END] += 1
                    not_before_s = request.start / per_second
                elif request.start >= previous.end:
                    timing[_LATER_DELAY] += 1
                    delay_s = round((request.start - previous.end) / per_second, 9)
                else:
                    timing[_LATER_OVERLAP] += 1
                    not_before_s = request.start / per_second
            entries.append(
                TraceEntry(
                    line=request.line,
                    arrival_s=arrival_s,
                    output_tokens=request.output_tokens,
                    input_tokens=request.input_tokens,
                    hash_ids=request.hash_ids,
                    model=request.model,
                    session=session,
                    turn=number,
                    line_turn=request.position,
                    delay_s=delay_s,
                    not_before_s=not_before_s,
                    block_scope=request.block_scope,
                )
            )
            previous = request
    return entries, timing


def _seconds(row: dict[str, Any], key: str, where: str) -> float:
    value = row[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= _MAX_EXACT_INT
    ):
        raise TraceFormatError(
            f"{where}: {key} must be a non-negative number of seconds, got {value!r}"
        )
    return float(value)


def _string_list(row: dict[str, Any], key: str, where: str) -> None:
    value = row[key]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise TraceFormatError(f"{where}: {key} must be a list of strings")


def _weka_request(
    raw: Any,
    where: str,
    line: int,
    position: int,
    block_size: int,
    blocks: _BlockIndex,
    counts: dict[str, Any],
    interned: dict[str, str],
) -> _Request:
    if not isinstance(raw, dict):
        raise TraceFormatError(f"{where}: expected a JSON object")
    if raw.get("type") == "subagent":
        raise TraceFormatError(
            f"{where}: subagent entries are not supported: a subagent is a nested "
            "conversation that runs alongside its parent, and a session replays "
            "one request at a time; use a trace built without subagents"
        )
    _check_keys(raw, where, _WEKA_REQUEST_KEYS, {}, "a Weka request")
    for required in ("t", "in", "out", "hash_ids"):
        if raw.get(required) is None:
            raise TraceFormatError(f"{where}: {required} is required for replay")
    kind = raw.get("type")
    if kind is not None and kind not in _WEKA_REQUEST_TYPES:
        raise TraceFormatError(
            f"{where}: type must be 's' or 'n' when present, got {kind!r}"
        )
    start = _seconds(raw, "t", where)
    api_time = None
    for key in ("api_time", "ttft", "think_time"):
        if raw.get(key) is not None:
            value = _seconds(raw, key, where)
            if key == "api_time":
                api_time = value
    for key in ("input_types", "output_types"):
        if key in raw:
            _string_list(raw, key, where)
    if "stop" in raw and not isinstance(raw["stop"], str):
        raise TraceFormatError(f"{where}: stop must be a string")
    model = raw.get("model")
    if model is not None and (not isinstance(model, str) or not model):
        raise TraceFormatError(f"{where}: model must be a non-empty string")
    input_tokens = _int_field(raw, "in", where, minimum=1)
    output_tokens = _int_field(raw, "out", where, minimum=1)
    hash_ids = _hash_id_list(raw, where)
    full, rest = divmod(input_tokens, block_size)
    if len(hash_ids) == full:
        last_length = block_size
        if rest:
            counts["unhashed_tail"] += 1
    elif rest and len(hash_ids) == full + 1:
        last_length = rest
        counts["partial_last_block"] += 1
    else:
        allowed = f"{full}" + (f" or {full + 1}" if rest else "")
        raise TraceFormatError(
            f"{where}: hash_ids has {len(hash_ids)} blocks but in {input_tokens} "
            f"needs {allowed} blocks of {block_size} tokens"
        )
    blocks.check(hash_ids, last_length, block_size, where)
    types = counts["request_types"]
    types[kind or "unset"] = types.get(kind or "unset", 0) + 1
    return _Request(
        line=line,
        position=position,
        start=start,
        end=None if api_time is None else start + api_time,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        hash_ids=hash_ids,
        model=None if model is None else interned.setdefault(model, model),
        block_scope=line,
    )


def _parse_weka(lines: _HashingLines, label: str) -> _ParseResult:
    sessions: list[list[_Request]] = []
    block_size: int | None = None
    counts: dict[str, Any] = {
        "partial_last_block": 0,
        "unhashed_tail": 0,
        "request_types": {},
    }
    ignored: set[str] = set()
    interned: dict[str, str] = {}
    for line, row in _json_rows(lines, label):
        where = f"{label}:{line}"
        _check_keys(row, where, _WEKA_LINE_KEYS, {}, "a Weka trace line")
        for required in ("block_size", "requests"):
            if row.get(required) is None:
                raise TraceFormatError(f"{where}: {required} is required for replay")
        if "id" in row and not isinstance(row["id"], str):
            raise TraceFormatError(f"{where}: id must be a string")
        size = _int_field(row, "block_size", where, minimum=1)
        if size > MAX_BLOCK_SIZE:
            raise TraceFormatError(
                f"{where}: block_size must be at most {MAX_BLOCK_SIZE}"
            )
        if block_size is not None and size != block_size:
            raise TraceFormatError(
                f"{where}: block_size {size} differs from {block_size} on an "
                "earlier line; a trace file must use one block size"
            )
        block_size = size
        scope = row.get("hash_id_scope", "local")
        if scope != "local":
            raise TraceFormatError(
                f"{where}: hash_id_scope must be 'local' when present, got "
                f"{scope!r}; the pinned spec numbers hash ids per conversation "
                "and defines no shared namespace"
            )
        requests = row["requests"]
        if not isinstance(requests, list) or not requests:
            raise TraceFormatError(f"{where}: requests must be a non-empty list")
        ignored.update(key for key in row if key not in _WEKA_LINE_USED)
        blocks = _BlockIndex()
        conversation: list[_Request] = []
        for position, raw in enumerate(requests):
            request_where = f"{where}: requests[{position}]"
            request = _weka_request(
                raw,
                request_where,
                line,
                position,
                size,
                blocks,
                counts,
                interned,
            )
            if conversation:
                _require_non_decreasing(
                    float(request.start),
                    float(conversation[-1].start),
                    request_where,
                    "t within the conversation",
                )
            ignored.update(
                f"requests[].{key}" for key in raw if key not in _WEKA_REQUEST_USED
            )
            conversation.append(request)
        sessions.append(conversation)
    if not sessions:
        raise TraceFormatError(f"{label}: the trace has no rows")
    entries, timing = _session_entries(sessions, 1)
    details = {
        "hash_id_scope": "local: hash ids are numbered per conversation",
        "hash_blocks": {
            "partial_last_block": counts["partial_last_block"],
            "unhashed_tail": counts["unhashed_tail"],
        },
        "request_types": dict(sorted(counts["request_types"].items())),
        "later_requests": timing,
        "ignored_fields": sorted(ignored),
    }
    return (
        tuple(entries),
        len(entries),
        entries[0].arrival_s,
        {"sessions": len(sessions), "block_size": block_size, "details": details},
    )


def _otlp_value(raw: Any, where: str) -> Any:
    if not isinstance(raw, dict) or len(raw) > 1:
        raise TraceFormatError(f"{where}: an AnyValue is an object with one field")
    if not raw:
        return None
    ((kind, value),) = raw.items()
    if kind not in _OTLP_ANY_VALUE_KEYS:
        raise TraceFormatError(f"{where}: unknown AnyValue field {kind!r}")
    if kind in ("stringValue", "bytesValue") and isinstance(value, str):
        return value
    if kind == "boolValue" and isinstance(value, bool):
        return value
    if kind == "intValue" and not isinstance(value, bool):
        if isinstance(value, int):
            return value
        if isinstance(value, str) and re.fullmatch(r"-?[0-9]+", value):
            return int(value)
    if (
        kind == "doubleValue"
        and not isinstance(value, bool)
        and isinstance(value, (int, float))
    ):
        return float(value)
    if kind in ("arrayValue", "kvlistValue") and isinstance(value, dict):
        if set(value) - {"values"}:
            raise TraceFormatError(f"{where}: {kind} holds only values")
        items = value.get("values", [])
        if not isinstance(items, list):
            raise TraceFormatError(f"{where}: {kind}.values must be a list")
        if kind == "kvlistValue":
            return _otlp_attributes(items, where)
        return tuple(_otlp_value(item, where) for item in items)
    raise TraceFormatError(f"{where}: {kind} has an invalid value {value!r}")


def _otlp_attributes(raw: Any, where: str) -> dict[str, Any]:
    if not isinstance(raw, list):
        raise TraceFormatError(f"{where}: attributes must be a list")
    attributes: dict[str, Any] = {}
    for item in raw:
        if not isinstance(item, dict) or not set(item) <= {"key", "value"}:
            raise TraceFormatError(
                f"{where}: an attribute is an object with key and value"
            )
        key = item.get("key")
        if not isinstance(key, str) or not key:
            raise TraceFormatError(f"{where}: attribute key must be a string")
        if key in attributes:
            raise TraceFormatError(f"{where}: attribute {key!r} is repeated")
        attributes[key] = _otlp_value(item.get("value", {}), f"{where}: {key}")
    return attributes


def _otlp_time(span: dict[str, Any], key: str, where: str) -> int:
    value = span.get(key)
    if isinstance(value, str) and _INTEGER.fullmatch(value):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TraceFormatError(
            f"{where}: {key} must be an integer number of nanoseconds, got {value!r}"
        )
    if not 0 < value <= _MAX_UINT64:
        raise TraceFormatError(f"{where}: {key} is out of range")
    return value


def _otel_failed(span: dict[str, Any], attributes: dict[str, Any], where: str) -> bool:
    status = span.get("status")
    code = 0
    if status is not None:
        if not isinstance(status, dict) or not set(status) <= {"code", "message"}:
            raise TraceFormatError(f"{where}: status holds only code and message")
        code = status.get("code", 0)
        if isinstance(code, bool) or code not in (0, 1, 2):
            raise TraceFormatError(
                f"{where}: status.code must be 0, 1 or 2, got {code!r}"
            )
    return code == _OTLP_STATUS_ERROR or attributes.get("error.type") is not None


def _otel_tokens(
    attributes: dict[str, Any], where: str, aliases: dict[str, int]
) -> tuple[int, int]:
    values: list[int] = []
    for _, current, alias in _OTEL_TOKENS:
        value = attributes.get(current)
        old = attributes.get(alias)
        if value is not None and old is not None and value != old:
            raise TraceFormatError(
                f"{where}: {current} {value!r} and the deprecated {alias} "
                f"{old!r} disagree"
            )
        if value is None and old is not None:
            value = old
            aliases[alias] = aliases.get(alias, 0) + 1
        if value is None:
            raise TraceFormatError(
                f"{where}: {current} is required; replay sizes each request by "
                "its declared usage"
            )
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= _MAX_EXACT_INT
        ):
            raise TraceFormatError(
                f"{where}: {current} must be an integer between 1 and "
                f"{_MAX_EXACT_INT}, got {value!r}"
            )
        values.append(value)
    return values[0], values[1]


def _otel_spans(row: dict[str, Any], where: str) -> Iterator[tuple[str, Any]]:
    _check_keys(row, where, _OTLP_TRACES_KEYS, {}, "an OTLP TracesData line")
    resource_spans = row.get("resourceSpans", [])
    if not isinstance(resource_spans, list):
        raise TraceFormatError(f"{where}: resourceSpans must be a list")
    for i, resource in enumerate(resource_spans):
        resource_where = f"{where}: resourceSpans[{i}]"
        if not isinstance(resource, dict):
            raise TraceFormatError(f"{resource_where}: expected a JSON object")
        _check_keys(
            resource, resource_where, _OTLP_RESOURCE_SPANS_KEYS, {}, "ResourceSpans"
        )
        if not isinstance(resource.get("resource", {}), dict):
            raise TraceFormatError(f"{resource_where}: resource must be an object")
        scope_spans = resource.get("scopeSpans", [])
        if not isinstance(scope_spans, list):
            raise TraceFormatError(f"{resource_where}: scopeSpans must be a list")
        for j, scope in enumerate(scope_spans):
            scope_where = f"{resource_where}.scopeSpans[{j}]"
            if not isinstance(scope, dict):
                raise TraceFormatError(f"{scope_where}: expected a JSON object")
            _check_keys(scope, scope_where, _OTLP_SCOPE_SPANS_KEYS, {}, "ScopeSpans")
            spans = scope.get("spans", [])
            if not isinstance(spans, list):
                raise TraceFormatError(f"{scope_where}: spans must be a list")
            for k, span in enumerate(spans):
                yield f"{scope_where}.spans[{k}]", span


def _parse_otel_genai(
    lines: _HashingLines, label: str, exclude_failed: bool
) -> _ParseResult:
    traces: dict[str, list[_Request]] = {}
    skipped: dict[str, int] = {}
    aliases: dict[str, int] = {}
    provenance: dict[str, int] = {}
    content_spans = 0
    failed = 0
    first_failed_line: int | None = None
    rows_total = 0
    interned: dict[str, str] = {}
    for line, row in _json_rows(lines, label):
        position = -1
        for where, span in _otel_spans(row, f"{label}:{line}"):
            position += 1
            rows_total += 1
            if not isinstance(span, dict):
                raise TraceFormatError(f"{where}: expected a JSON object")
            _check_keys(span, where, _OTLP_SPAN_KEYS, {}, "an OTLP span")
            attributes = _otlp_attributes(span.get("attributes", []), where)
            operation = attributes.get("gen_ai.operation.name")
            if operation is None:
                usage = sorted(k for k in attributes if k.startswith("gen_ai.usage."))
                if usage:
                    raise TraceFormatError(
                        f"{where}: the span has {usage[0]} but no "
                        "gen_ai.operation.name, so it cannot be told apart from "
                        "an agent or workflow total; set gen_ai.operation.name"
                    )
                skipped["(no gen_ai.operation.name)"] = (
                    skipped.get("(no gen_ai.operation.name)", 0) + 1
                )
                continue
            if not isinstance(operation, str) or not operation:
                raise TraceFormatError(
                    f"{where}: gen_ai.operation.name must be a non-empty string"
                )
            if operation not in _OTEL_INFERENCE_OPERATIONS:
                skipped[operation] = skipped.get(operation, 0) + 1
                continue
            trace_id = span.get("traceId")
            if (
                not isinstance(trace_id, str)
                or not _TRACE_ID.fullmatch(trace_id.lower())
                or not trace_id.strip("0")
            ):
                raise TraceFormatError(
                    f"{where}: traceId must be 32 hex digits and not all zero"
                )
            start = _otlp_time(span, "startTimeUnixNano", where)
            end = _otlp_time(span, "endTimeUnixNano", where)
            if end < start:
                raise TraceFormatError(
                    f"{where}: endTimeUnixNano is before startTimeUnixNano"
                )
            if _otel_failed(span, attributes, where):
                failed += 1
                if first_failed_line is None:
                    first_failed_line = line
                continue
            input_tokens, output_tokens = _otel_tokens(attributes, where, aliases)
            model = attributes.get("gen_ai.request.model")
            if model is not None and (not isinstance(model, str) or not model):
                raise TraceFormatError(
                    f"{where}: gen_ai.request.model must be a non-empty string"
                )
            if not _OTEL_CONTENT_ATTRIBUTES.isdisjoint(attributes):
                content_spans += 1
            basis = attributes.get("llmtracefx.tokens.provenance")
            if isinstance(basis, str):
                provenance[basis] = provenance.get(basis, 0) + 1
            traces.setdefault(trace_id.lower(), []).append(
                _Request(
                    line=line,
                    position=position,
                    start=start,
                    end=end,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=None if model is None else interned.setdefault(model, model),
                )
            )
    if failed and not exclude_failed:
        raise TraceFormatError(
            f"{label}:{first_failed_line}: the trace has {failed} failed inference "
            "spans (status ERROR or error.type). Pass --exclude-failed to drop "
            "and count them"
        )
    if not traces:
        found = ", ".join(f"{n} {name}" for name, n in sorted(skipped.items()))
        raise TraceFormatError(
            f"{label}: the trace has no successful chat, text_completion or "
            f"generate_content spans"
            + (f" (spans skipped: {found})" if found else "")
            + (f" ({failed} excluded as failed)" if failed else "")
        )
    # sorted() is stable, so spans that start together keep file order.
    sessions = [sorted(spans, key=lambda r: r.start) for spans in traces.values()]
    entries, timing = _session_entries(sessions, 1_000_000_000)
    details: dict[str, Any] = {
        "inference_operations": sorted(_OTEL_INFERENCE_OPERATIONS),
        "session_key": "traceId",
        "skipped_spans": dict(sorted(skipped.items())),
        "deprecated_token_attributes": dict(sorted(aliases.items())),
        "spans_with_content_attributes": content_spans,
        "content_attributes": "never read into a prompt or stored",
        "later_requests": timing,
    }
    if provenance:
        details["llmtracefx_token_provenance"] = dict(sorted(provenance.items()))
    return (
        tuple(entries),
        rows_total,
        entries[0].arrival_s,
        {
            "sessions": len(sessions),
            "excluded": {EXCLUDED_ERROR_SPAN: failed} if failed else {},
            "details": details,
        },
    )
