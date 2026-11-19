import hashlib
import json
from pathlib import Path
from typing import Any

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


AIPERF_MULTI = FIXTURES / "aiperf_multi_turn.jsonl"


def test_aiperf_multi_turn_groups_and_orders_sessions() -> None:
    trace = parse_trace(AIPERF_MULTI, "aiperf-multi-turn")
    entries = trace.entries
    assert trace.sessions == 3
    assert trace.rows_total == 4
    # Lines sharing a session_id join in file order; sessions sort by start.
    assert [e.line for e in entries] == [1, 1, 3, 4, 2]
    assert [e.session for e in entries] == [0, 0, 0, 1, 2]
    assert [e.turn for e in entries] == [0, 1, 2, 0, 0]
    assert [e.arrival_s for e in entries] == [0.0, 0.0, 0.0, 0.2, 0.5]
    assert entries[1].delay_s == 0.01 and entries[1].not_before_s is None
    assert entries[2].not_before_s == 0.8 and entries[2].delay_s is None
    source = trace.source_dict()
    assert source["sessions"] == 3
    assert source["upstream"]["repository"] == "ai-dynamo/aiperf"
    assert "trip" not in json.dumps(source)


@pytest.mark.parametrize(
    ("turns", "match"),
    [
        ([{"timestamp": 0, "texts": ["a"], "output_length": 1}], "texts"),
        ([{"delay": 5, "text": "a", "output_length": 1}], "first turn"),
        ([{"text": "a", "output_length": 1}], "first turn"),
        (
            [
                {"timestamp": 0, "text": "a", "output_length": 1},
                {"timestamp": 9, "delay": 1, "text": "b", "output_length": 1},
            ],
            "not both",
        ),
        (
            [{"timestamp": 0, "text": "a", "output_length": 1, "role": "system"}],
            "role",
        ),
        (
            [{"timestamp": 0, "text": "a", "output_length": 1, "session_id": "x"}],
            "session_id",
        ),
        (
            [
                {"timestamp": 9, "text": "a", "output_length": 1},
                {"timestamp": 5, "text": "b", "output_length": 1},
            ],
            "backwards",
        ),
        ([{"timestamp": 0, "text": "a"}], "output_length"),
        ([], "non-empty list"),
    ],
)
def test_aiperf_multi_turn_refusals(
    tmp_path: Path, turns: list[dict[str, object]], match: str
) -> None:
    path = _write(tmp_path, json.dumps({"session_id": "s", "turns": turns}) + "\n")
    with pytest.raises(TraceFormatError, match=match):
        parse_trace(path, "aiperf-multi-turn")


def test_aiperf_multi_turn_refuses_a_single_turn_row(tmp_path: Path) -> None:
    path = _write(tmp_path, AIPERF.read_text())
    with pytest.raises(TraceFormatError, match="unknown field"):
        parse_trace(path, "aiperf-multi-turn")


WEKA = FIXTURES / "weka_small.jsonl"
OTEL = FIXTURES / "otel_genai_small.jsonl"


def test_weka_conversations_are_sessions_timed_by_idle_gaps() -> None:
    trace = parse_trace(WEKA, "weka")
    assert (trace.block_size, trace.sessions, trace.timestamp_unit) == (4, 2, "s")
    entries = trace.entries
    assert [(e.session, e.turn, e.line) for e in entries] == [
        (0, 0, 1),
        (0, 1, 1),
        (0, 2, 1),
        (1, 0, 2),
        (1, 1, 2),
    ]
    # Every turn carries its session's start so a window selects whole sessions.
    assert [e.arrival_s for e in entries] == [0.0, 0.0, 0.0, 0.25, 0.25]
    # 1.5 - (0.0 + 1.0); then 1.8 overlaps the previous end 2.0; then no api_time.
    assert [e.delay_s for e in entries] == [None, 0.5, None, None, None]
    assert [e.not_before_s for e in entries] == [None, None, 1.8, None, 0.5]
    assert [e.block_scope for e in entries] == [1, 1, 1, 2, 2]
    assert entries[3].hash_ids == ()
    details = trace.details
    assert details["hash_blocks"] == {"partial_last_block": 2, "unhashed_tail": 2}
    assert details["request_types"] == {"n": 1, "s": 4}
    assert "requests[].api_time" in details["ignored_fields"]
    source = json.dumps(trace.source_dict())
    assert "conv-secret" not in source
    assert '"commit": "15e88f72f902c7852c330caf2cc11722e5dbc1aa"' in source


def _weka_line(**overrides: object) -> dict[str, object]:
    line: dict[str, object] = {
        "id": "c",
        "block_size": 4,
        "requests": [{"t": 0, "type": "s", "in": 8, "out": 1, "hash_ids": [0, 1]}],
    }
    line.update(overrides)
    return line


def _weka_request(**overrides: object) -> dict[str, object]:
    request: dict[str, object] = {
        "t": 0,
        "type": "s",
        "in": 8,
        "out": 1,
        "hash_ids": [0, 1],
    }
    request.update(overrides)
    return request


@pytest.mark.parametrize(
    ("lines", "message"),
    [
        (
            [_weka_line(requests=[{"type": "subagent", "requests": []}])],
            "subagent entries are not supported",
        ),
        ([_weka_line(hash_id_scope="global")], "hash_id_scope must be 'local'"),
        ([_weka_line(session="x")], "field 'session'"),
        ([_weka_line(requests=[_weka_request(prompt="hi")])], "field 'prompt'"),
        ([_weka_line(block_size=None)], "block_size is required"),
        ([_weka_line(requests=[])], "requests must be a non-empty list"),
        ([_weka_line(), _weka_line(block_size=8)], "one block size"),
        (
            [_weka_line(requests=[_weka_request(hash_ids=[0])])],
            "needs 2 blocks of 4",
        ),
        (
            [_weka_line(requests=[_weka_request(hash_ids=[0, 1, 2])])],
            "needs 2 blocks of 4",
        ),
        (
            [
                _weka_line(
                    requests=[_weka_request(t=2), _weka_request(t=1)],
                )
            ],
            "t within the conversation goes backwards",
        ),
        (
            [
                _weka_line(
                    requests=[_weka_request(), _weka_request(hash_ids=[1, 0])],
                )
            ],
            "cannot name a shared prefix",
        ),
        ([_weka_line(requests=[_weka_request(type="x")])], "type must be"),
        ([_weka_line(requests=[_weka_request(api_time=-1)])], "api_time must be"),
    ],
)
def test_weka_refusals(
    tmp_path: Path, lines: list[dict[str, object]], message: str
) -> None:
    path = _write(tmp_path, _mooncake_rows(*lines))
    with pytest.raises(TraceFormatError, match=message):
        parse_trace(path, "weka")


def test_weka_hash_ids_are_scoped_to_their_conversation(tmp_path: Path) -> None:
    # The same id names different text on each line, so a block that would
    # conflict across lines is fine.
    path = _write(
        tmp_path,
        _mooncake_rows(
            _weka_line(),
            _weka_line(requests=[_weka_request(**{"in": 6, "hash_ids": [1, 0]})]),
        ),
    )
    trace = parse_trace(path, "weka")
    assert [e.block_scope for e in trace.entries] == [1, 2]


def test_weka_declares_its_block_size() -> None:
    with pytest.raises(TraceFormatError, match="a Weka trace declares its block_size"):
        parse_trace(WEKA, "weka", block_size=64)


def test_otel_genai_spans_group_by_trace_and_skip_other_operations() -> None:
    trace = parse_trace(OTEL, "otel-genai")
    assert (trace.sessions, trace.timestamp_unit, trace.rows_total) == (2, "ns", 7)
    entries = trace.entries
    assert [(e.session, e.turn, e.line) for e in entries] == [
        (0, 0, 1),
        (0, 1, 1),
        (0, 2, 2),
        (1, 0, 2),
    ]
    assert [(e.input_tokens, e.output_tokens) for e in entries] == [
        (100, 10),
        (120, 12),
        (130, 8),
        (40, 4),
    ]
    assert [e.arrival_s for e in entries] == [1.0, 1.0, 1.0, 1.2]
    # The second chat starts 0.5 s after the first ends; the third overlaps it.
    assert [e.delay_s for e in entries] == [None, 0.5, None, None]
    assert [e.not_before_s for e in entries] == [None, None, 2.8, None]
    assert [e.line_turn for e in entries] == [1, 3, 1, 0]
    details = trace.details
    assert details["skipped_spans"] == {
        "(no gen_ai.operation.name)": 1,
        "execute_tool": 1,
        "invoke_agent": 1,
    }
    assert details["deprecated_token_attributes"] == {
        "gen_ai.usage.completion_tokens": 1,
        "gen_ai.usage.prompt_tokens": 1,
    }
    source = json.dumps(trace.source_dict())
    assert "0af7651916cd43dd8448eb211c80319c" not in source
    assert "SECRET" not in source


def _otel_span(*attributes: tuple[str, object], **fields: object) -> dict[str, Any]:
    values = {
        "gen_ai.operation.name": "chat",
        "gen_ai.usage.input_tokens": 5,
        "gen_ai.usage.output_tokens": 2,
    }
    values.update(dict(attributes))
    span: dict[str, Any] = {
        "traceId": "0af7651916cd43dd8448eb211c80319c",
        "spanId": "b7ad6b7169203331",
        "startTimeUnixNano": "1000",
        "endTimeUnixNano": "2000",
        "attributes": [
            {
                "key": key,
                "value": (
                    {"stringValue": value}
                    if isinstance(value, str)
                    else (
                        {"doubleValue": value}
                        if isinstance(value, float)
                        else {"intValue": str(value)}
                    )
                ),
            }
            for key, value in values.items()
            if value is not None
        ],
    }
    span.update(fields)
    return span


def _otel_rows(*spans: dict[str, Any]) -> str:
    batch = {"resourceSpans": [{"scopeSpans": [{"spans": list(spans)}]}]}
    return json.dumps(batch) + "\n"


@pytest.mark.parametrize(
    ("span", "message"),
    [
        (
            _otel_span(("gen_ai.operation.name", None)),
            "gen_ai.usage.input_tokens but no gen_ai.operation.name",
        ),
        (_otel_span(("gen_ai.usage.output_tokens", None)), "output_tokens is required"),
        (_otel_span(("gen_ai.usage.input_tokens", 5.0)), "must be an integer"),
        (_otel_span(("gen_ai.usage.input_tokens", 0)), "must be an integer"),
        (
            _otel_span(("gen_ai.usage.prompt_tokens", 6)),
            "deprecated gen_ai.usage.prompt_tokens 6 disagree",
        ),
        (_otel_span(traceId="0" * 32), "traceId must be 32 hex digits"),
        (_otel_span(traceId="xyz"), "traceId must be 32 hex digits"),
        (_otel_span(endTimeUnixNano="999"), "endTimeUnixNano is before"),
        (_otel_span(startTimeUnixNano=1.5), "must be an integer number"),
        (_otel_span(body="x"), "field 'body'"),
        (_otel_span(status={"code": 7}), "status.code must be 0, 1 or 2"),
        (_otel_span(("gen_ai.operation.name", "embeddings")), "no successful chat"),
    ],
)
def test_otel_genai_refusals(
    tmp_path: Path, span: dict[str, Any], message: str
) -> None:
    path = _write(tmp_path, _otel_rows(span))
    with pytest.raises(TraceFormatError, match=message):
        parse_trace(path, "otel-genai")


def test_otel_genai_refuses_repeated_attributes(tmp_path: Path) -> None:
    span = _otel_span()
    span["attributes"].append(span["attributes"][0])
    path = _write(tmp_path, _otel_rows(span))
    with pytest.raises(TraceFormatError, match="is repeated"):
        parse_trace(path, "otel-genai")


def test_otel_genai_failed_spans_need_exclude_failed(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        _otel_rows(
            _otel_span(),
            _otel_span(status={"code": 2, "message": "boom"}),
            _otel_span(("error.type", "timeout")),
        ),
    )
    with pytest.raises(TraceFormatError, match="2 failed inference spans"):
        parse_trace(path, "otel-genai")
    trace = parse_trace(path, "otel-genai", exclude_failed=True)
    assert len(trace.entries) == 1
    assert trace.excluded == {"error_spans": 2}
