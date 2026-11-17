"""Strict parsers for production arrival traces.

Three formats are read, each checked against its upstream definition:

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
TRACE_FORMATS = (MOONCAKE, BURSTGPT, AIPERF_SINGLE_TURN)

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
    """Source arrival time in seconds, before any rebasing or scaling."""
    output_tokens: int
    input_tokens: int | None = None
    """Declared prompt length. ``None`` for text formats, which declare text."""
    hash_ids: tuple[int, ...] | None = None
    text: str | None = None
    model: str | None = None
    log_type: str | None = None


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

    def source_dict(self) -> dict[str, Any]:
        """The source description recorded in every replay artifact."""
        return {
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
        timestamp = row["timestamp"]
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, (int, float))
            or not math.isfinite(timestamp)
            or not 0 <= timestamp <= _MAX_EXACT_INT
        ):
            raise TraceFormatError(
                f"{where}: timestamp must be a non-negative number of "
                f"milliseconds, got {timestamp!r}"
            )
        text = row["text"]
        if not isinstance(text, str) or not text.strip():
            raise TraceFormatError(f"{where}: text must be a non-empty string")
        output_length = _int_field(row, "output_length", where, minimum=1)
        arrival_s = float(timestamp) / 1000.0
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
