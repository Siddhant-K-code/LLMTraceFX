import hashlib
import json
from pathlib import Path

import pytest

from llmtracefx.optimizer.replay.formats import (
    EXCLUDED_ZERO_BOTH,
    EXCLUDED_ZERO_RESPONSE,
    TraceFormatError,
    parse_trace,
)

FIXTURES = Path(__file__).parent / "fixtures" / "replay"
MOONCAKE = FIXTURES / "mooncake_small.jsonl"
BURST_V1 = FIXTURES / "burstgpt_v1_small.csv"
BURST_V3 = FIXTURES / "burstgpt_v3_small.csv"
AIPERF = FIXTURES / "aiperf_single_turn.jsonl"


def _write(tmp_path: Path, text: str, name: str = "trace.jsonl") -> Path:
    path = tmp_path / name
    path.write_bytes(text.encode("utf-8"))
    return path


def _mooncake_rows(*rows: dict[str, object]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def test_mooncake_fixture_parses_with_hash_and_size() -> None:
    trace = parse_trace(MOONCAKE, "mooncake")
    data = MOONCAKE.read_bytes()
    assert trace.sha256 == hashlib.sha256(data).hexdigest()
    assert trace.size_bytes == len(data)
    assert trace.timestamp_unit == "ms"
    assert trace.block_size == 512
    assert [e.arrival_s for e in trace.entries] == [0.0, 0.25, 0.25, 1.0]
    assert trace.entries[1].hash_ids == (0, 1, 3)
    assert trace.entries[1].input_tokens == 1300
    assert [e.line for e in trace.entries] == [1, 2, 3, 4]


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {"timestamp": 0, "input_length": 10, "output_length": 1},
            "exactly timestamp",
        ),
        (
            {
                "timestamp": 0,
                "input_length": 10,
                "output_length": 1,
                "hash_ids": [0],
                "extra": 1,
            },
            "exactly timestamp",
        ),
        (
            {"timestamp": 0.5, "input_length": 10, "output_length": 1, "hash_ids": [0]},
            "timestamp must be an integer",
        ),
        (
            {"timestamp": 0, "input_length": True, "output_length": 1, "hash_ids": [0]},
            "input_length must be an integer",
        ),
        (
            {"timestamp": 0, "input_length": 10, "output_length": 0, "hash_ids": [0]},
            "output_length must be between 1",
        ),
        (
            {"timestamp": -1, "input_length": 10, "output_length": 1, "hash_ids": [0]},
            "timestamp must be between 0",
        ),
        (
            {"timestamp": 0, "input_length": 10, "output_length": 1, "hash_ids": [-1]},
            "non-negative integers",
        ),
        (
            {"timestamp": 0, "input_length": 600, "output_length": 1, "hash_ids": [0]},
            "needs 2 blocks of 512",
        ),
    ],
)
def test_mooncake_refuses_malformed_rows(
    tmp_path: Path, row: dict[str, object], message: str
) -> None:
    path = _write(tmp_path, _mooncake_rows(row))
    with pytest.raises(TraceFormatError, match=message) as info:
        parse_trace(path, "mooncake")
    assert str(info.value).startswith("trace.jsonl:1:")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ('{"timestamp": 0, "timestamp": 1}\n', "duplicate key"),
        (
            '{"timestamp": NaN, "input_length": 1, "output_length": 1, '
            '"hash_ids": [0]}\n',
            "not valid JSON",
        ),
        ("\n", "blank line"),
        ("[1, 2]\n", "expected a JSON object"),
        ("", "no rows"),
    ],
)
def test_mooncake_refuses_bad_json(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(TraceFormatError, match=message):
        parse_trace(_write(tmp_path, text), "mooncake")


def test_mooncake_refuses_backwards_time(tmp_path: Path) -> None:
    text = _mooncake_rows(
        {"timestamp": 10, "input_length": 5, "output_length": 1, "hash_ids": [0]},
        {"timestamp": 9, "input_length": 5, "output_length": 1, "hash_ids": [0]},
    )
    with pytest.raises(TraceFormatError, match=r"trace.jsonl:2: timestamp goes back"):
        parse_trace(_write(tmp_path, text), "mooncake")


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        ((8, [0, 1]), (8, [1, 2]), "hash_id 1 is at block 0 here and at block 1"),
        ((8, [0, 1]), (8, [5, 1]), "hash_id 1 follows 5 here and 0 elsewhere"),
        ((8, [0, 1]), (6, [0, 1]), "names a 2-token block here"),
        ((6, [0, 1]), (7, [0, 1]), "names a 3-token block here"),
        ((6, [0, 1]), (8, [0, 1]), "names a full block here and a 2-token block"),
    ],
)
def test_mooncake_refuses_inconsistent_prefix_tree(
    tmp_path: Path,
    first: tuple[int, list[int]],
    second: tuple[int, list[int]],
    message: str,
) -> None:
    rows = [
        {"timestamp": t, "input_length": n, "output_length": 1, "hash_ids": ids}
        for t, (n, ids) in enumerate((first, second))
    ]
    path = _write(tmp_path, _mooncake_rows(*rows))
    with pytest.raises(TraceFormatError, match=message) as info:
        parse_trace(path, "mooncake", block_size=4)
    assert str(info.value).startswith("trace.jsonl:2:")


def test_mooncake_block_size_is_explicit(tmp_path: Path) -> None:
    text = _mooncake_rows(
        {"timestamp": 0, "input_length": 20, "output_length": 1, "hash_ids": [0, 1, 2]}
    )
    path = _write(tmp_path, text)
    with pytest.raises(TraceFormatError, match="check --block-size"):
        parse_trace(path, "mooncake")
    assert parse_trace(path, "mooncake", block_size=8).block_size == 8


def test_format_specific_flags_are_refused_elsewhere() -> None:
    with pytest.raises(TraceFormatError, match="--block-size applies"):
        parse_trace(BURST_V3, "burstgpt", block_size=16)
    with pytest.raises(TraceFormatError, match="--exclude-failed applies"):
        parse_trace(MOONCAKE, "mooncake", exclude_failed=True)
    with pytest.raises(TraceFormatError, match="unknown trace format"):
        parse_trace(MOONCAKE, "sharegpt")


def test_refuses_invalid_utf8_and_non_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_bytes(b'{"timestamp": "\xff"}\n')
    with pytest.raises(TraceFormatError, match="bad.jsonl:1: not valid UTF-8"):
        parse_trace(path, "mooncake")
    with pytest.raises(TraceFormatError, match="cannot open|not a regular file"):
        parse_trace(tmp_path, "mooncake")
    with pytest.raises(TraceFormatError, match="cannot open"):
        parse_trace(tmp_path / "missing.jsonl", "mooncake")


def test_burstgpt_refuses_failed_rows_without_flag() -> None:
    with pytest.raises(TraceFormatError) as info:
        parse_trace(BURST_V1, "burstgpt")
    message = str(info.value)
    assert message.startswith("burstgpt_v1_small.csv:3:")
    assert "1 zero_request_and_response_tokens" in message
    assert "1 zero_response_tokens" in message
    assert "--exclude-failed" in message


def test_burstgpt_exclude_failed_counts_reasons() -> None:
    trace = parse_trace(BURST_V1, "burstgpt", exclude_failed=True)
    assert trace.rows_total == 5
    assert trace.excluded == {EXCLUDED_ZERO_BOTH: 1, EXCLUDED_ZERO_RESPONSE: 1}
    assert [e.line for e in trace.entries] == [2, 4, 6]
    assert [e.model for e in trace.entries] == ["ChatGPT", "GPT-4", "ChatGPT"]
    assert trace.entries[1].log_type == "API log"
    assert trace.entries[1].input_tokens == 350
    assert trace.entries[1].output_tokens == 250
    assert trace.timestamp_unit == "s"
    assert trace.origin_s == 3.0
    assert trace.ignored_columns == ()


def test_burstgpt_v3_columns_and_float_timestamps() -> None:
    trace = parse_trace(BURST_V3, "burstgpt")
    assert trace.columns[:3] == ("Timestamp", "Session ID", "Elapsed time")
    assert [e.arrival_s for e in trace.entries] == [10.0, 10.5, 12.25]
    assert trace.ignored_columns == ()
    assert trace.source_dict()["rows_valid"] == 3


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("Timestamp,Model\n1,ChatGPT\n", "must name Request tokens, Response tokens"),
        (
            "Timestamp,Request tokens,Response tokens,Total tokens\n1,2,3,9\n",
            "Total tokens 9 is not",
        ),
        ("Timestamp,Request tokens,Response tokens\n-1,2,3\n", "Timestamp must be"),
        ("Timestamp,Request tokens,Response tokens\n1e3,2,3\n", "Timestamp must be"),
        ("Timestamp,Request tokens,Response tokens\n1,2.0,3\n", "Request tokens must"),
        ("Timestamp,Request tokens,Response tokens\n1,2\n", "expected 3 columns"),
        (
            "Timestamp,Request tokens,Response tokens\n5,2,3\n4,2,3\n",
            "Timestamp goes backwards",
        ),
        ("Timestamp,Timestamp,Request tokens,Response tokens\n", "repeats a column"),
        ("Timestamp,Request tokens,Response tokens\n", "no rows"),
        ("", "empty"),
    ],
)
def test_burstgpt_refuses_malformed_csv(
    tmp_path: Path, text: str, message: str
) -> None:
    with pytest.raises(TraceFormatError, match=message):
        parse_trace(_write(tmp_path, text, "trace.csv"), "burstgpt")


def test_burstgpt_records_unknown_columns(tmp_path: Path) -> None:
    text = "Timestamp,Request tokens,Response tokens,Region\n1,2,3,eu\n"
    trace = parse_trace(_write(tmp_path, text, "trace.csv"), "burstgpt")
    assert trace.ignored_columns == ("Region",)
    assert trace.entries[0].model is None


def test_burstgpt_every_row_excluded_is_refused(tmp_path: Path) -> None:
    text = "Timestamp,Request tokens,Response tokens\n1,0,0\n"
    with pytest.raises(TraceFormatError, match="every row was excluded"):
        parse_trace(_write(tmp_path, text, "t.csv"), "burstgpt", exclude_failed=True)


def test_aiperf_single_turn_fixture() -> None:
    trace = parse_trace(AIPERF, "aiperf-single-turn")
    assert [e.arrival_s for e in trace.entries] == [0.0, 1.5005, 4.0]
    assert trace.entries[0].text == "Summarize the release notes."
    assert trace.entries[0].input_tokens is None
    assert [e.output_tokens for e in trace.entries] == [32, 8, 4]
    assert trace.block_size is None


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"timestamp": 0, "text": "a", "output_length": 1, "delay": 5}, "'delay'"),
        (
            {"timestamp": 0, "text": "a", "output_length": 1, "session_id": "s"},
            "'session_id' is not supported",
        ),
        ({"timestamp": 0, "texts": ["a"], "output_length": 1}, "'texts'"),
        ({"timestamp": 0, "text": "a", "output_length": 1, "foo": 1}, "unknown field"),
        (
            {"type": "multi_turn", "timestamp": 0, "text": "a", "output_length": 1},
            "type",
        ),
        (
            {"timestamp": 0, "text": "a", "output_length": 1, "role": "system"},
            "role must be 'user'",
        ),
        ({"timestamp": 0, "text": "a"}, "output_length is required"),
        ({"text": "a", "output_length": 1}, "timestamp is required"),
        ({"timestamp": 0, "text": " ", "output_length": 1}, "non-empty string"),
        ({"timestamp": "0", "text": "a", "output_length": 1}, "milliseconds"),
    ],
)
def test_aiperf_refuses_unsupported_rows(
    tmp_path: Path, row: dict[str, object], message: str
) -> None:
    with pytest.raises(TraceFormatError, match=message):
        parse_trace(_write(tmp_path, json.dumps(row) + "\n"), "aiperf-single-turn")
