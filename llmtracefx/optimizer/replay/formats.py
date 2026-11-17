"""Strict parsers for production arrival traces.

Four formats are read, each checked against its upstream definition:

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
TRACE_FORMATS = (MOONCAKE, BURSTGPT, AIPERF_SINGLE_TURN, AIPERF_MULTI_TURN)

AIPERF_UPSTREAM = {
    "repository": "ai-dynamo/aiperf",
    "commit": "fc7d61eafa813d53c8c6056de5b64648cb60d843",
    "loader": "src/aiperf/dataset/loader/multi_turn.py",
    "timing": "src/aiperf/timing/strategies/fixed_schedule.py",
}
"""The upstream definition the multi-turn parser and session timing follow."""

MOONCAKE_BLOCK_SIZE = 512
"""Tokens per hash block in the published Mooncake traces."""

_MAX_LINE_BYTES = 16 * 1024 * 1024
_MAX_EXACT_INT = 2**53
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
    """Multi-turn only: the session's position after sorting by arrival."""
    turn: int | None = None
    """Multi-turn only: 0-based turn number within the session."""
    line_turn: int | None = None
    """Multi-turn only: 0-based position in its line's ``turns`` list."""
    delay_s: float | None = None
    """Later turns only: wait after the previous turn completes."""
    not_before_s: float | None = None
    """Later turns only: declared timestamp, in source seconds."""


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
    """Multi-turn only: the number of sessions."""

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
            source["upstream"] = dict(AIPERF_UPSTREAM)
        return source


class _HashingLines:
    """Yield decoded lines while hashing the exact bytes read."""

    def __init__(self, handle: IO[bytes], label: str) -> None:
        self._handle = handle
        self._label = label
        self.digest = hashlib.sha256()
        self.size = 0
        self.line = 0

    def __iter__(self) -> Iterator[str]:
        while True:
            raw = self._handle.readline(_MAX_LINE_BYTES + 1)
            if not raw:
                return
            self.line += 1
            if len(raw) > _MAX_LINE_BYTES:
                raise TraceFormatError(
                    f"{self._label}:{self.line}: line exceeds {_MAX_LINE_BYTES} bytes"
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
        raise TraceFormatError("--block-size applies to the mooncake format only")
    if exclude_failed and fmt != BURSTGPT:
        raise TraceFormatError("--exclude-failed applies to the burstgpt format only")
    if not 1 <= block_size <= MAX_BLOCK_SIZE:
        raise TraceFormatError(f"block size must be between 1 and {MAX_BLOCK_SIZE}")
    trace_path = Path(path)
    label = trace_path.name
    with _open_regular(trace_path) as handle:
        lines = _HashingLines(handle, label)
        if fmt == MOONCAKE:
            result = _parse_mooncake(lines, label, block_size)
        elif fmt == BURSTGPT:
            result = _parse_burstgpt(lines, label, exclude_failed)
        elif fmt == AIPERF_MULTI_TURN:
            result = _parse_aiperf_multi_turn(lines, label)
        else:
            result = _parse_aiperf_single_turn(lines, label)
    entries, rows_total, origin_s, extra = result
    return ParsedTrace(
        format=fmt,
        file_name=label,
        sha256=lines.digest.hexdigest(),
        size_bytes=lines.size,
        rows_total=rows_total,
        origin_s=origin_s,
        entries=entries,
        timestamp_unit="s" if fmt == BURSTGPT else "ms",
        block_size=block_size if fmt == MOONCAKE else None,
        **extra,
    )


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


def _parse_mooncake(lines: _HashingLines, label: str, block_size: int) -> _ParseResult:
    entries: list[TraceEntry] = []
    position: dict[int, int] = {}
    predecessor: dict[int, int | None] = {}
    partial_length: dict[int, int] = {}
    full_block: set[int] = set()
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
        hash_ids = tuple(raw_ids)
        expected_blocks = -(-input_length // block_size)
        if len(hash_ids) != expected_blocks:
            raise TraceFormatError(
                f"{where}: hash_ids has {len(hash_ids)} blocks but input_length "
                f"{input_length} needs {expected_blocks} blocks of {block_size} "
                "tokens; check --block-size"
            )
        last_length = input_length - (expected_blocks - 1) * block_size
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
            if index == expected_blocks - 1 and last_length < block_size:
                known = partial_length.setdefault(block, last_length)
                if known != last_length or block in full_block:
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
                full_block.add(block)
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
