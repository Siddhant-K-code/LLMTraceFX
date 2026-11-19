import json
from pathlib import Path

import pytest

from llmtracefx.optimizer.replay.artifact import ReplayArtifactError, sha256_text
from llmtracefx.optimizer.replay.formats import parse_trace
from llmtracefx.optimizer.replay.prompts import word_count
from llmtracefx.optimizer.replay.schedule import (
    ScheduleError,
    Selection,
    build_schedule,
    history_sha256,
    read_schedule,
    write_schedule,
)

FIXTURES = Path(__file__).parent / "fixtures" / "replay"
MOONCAKE = parse_trace(FIXTURES / "mooncake_small.jsonl", "mooncake")
BURST = parse_trace(FIXTURES / "burstgpt_v1_small.csv", "burstgpt", exclude_failed=True)
AIPERF = parse_trace(FIXTURES / "aiperf_single_turn.jsonl", "aiperf-single-turn")
MULTI = parse_trace(FIXTURES / "aiperf_multi_turn.jsonl", "aiperf-multi-turn")


def test_offsets_start_at_zero_and_scale() -> None:
    schedule = build_schedule(MOONCAKE, Selection(), salt="s")
    assert schedule.offsets_s == (0.0, 0.25, 0.25, 1.0)
    slow = build_schedule(MOONCAKE, Selection(time_scale=2.0), salt="s")
    assert slow.offsets_s == (0.0, 0.5, 0.5, 2.0)
    burst = build_schedule(BURST, Selection(time_scale=0.1), salt="s")
    assert burst.offsets_s == (0.0, 10.7, 19.7)
    assert [r.source_line for r in burst.requests] == [2, 4, 6]


def test_window_is_relative_to_the_first_row_and_half_open() -> None:
    schedule = build_schedule(BURST, Selection(start_s=107.0, end_s=197.0), salt="s")
    assert [r.source_line for r in schedule.requests] == [4]
    assert schedule.offsets_s == (0.0,)
    edge = build_schedule(BURST, Selection(start_s=107.0), salt="s")
    assert [r.source_line for r in edge.requests] == [4, 6]


def test_max_requests_applies_after_filters_and_window() -> None:
    selection = Selection(models=("ChatGPT",), start_s=1.0, max_requests=1)
    schedule = build_schedule(BURST, selection, salt="s")
    assert [r.source_line for r in schedule.requests] == [6]


def test_filters_refuse_values_absent_from_the_trace() -> None:
    with pytest.raises(ScheduleError, match=r"--model gpt-4 does not appear.*GPT-4"):
        build_schedule(BURST, Selection(models=("gpt-4",)), salt="s")
    api = build_schedule(BURST, Selection(log_types=("API log",)), salt="s")
    assert [r.source_line for r in api.requests] == [4, 6]


@pytest.mark.parametrize(
    ("selection", "message"),
    [
        (Selection(time_scale=0.0), "--time-scale"),
        (Selection(time_scale=float("inf")), "--time-scale"),
        (Selection(start_s=-1.0), "--start-s"),
        (Selection(start_s=5.0, end_s=5.0), "--end-s must be greater"),
        (Selection(max_requests=0), "--max-requests"),
        (Selection(models=("x",)), "BurstGPT traces only"),
        (Selection(start_s=100.0), "selection is empty"),
    ],
)
def test_bad_selections_are_refused(selection: Selection, message: str) -> None:
    with pytest.raises(ScheduleError, match=message):
        build_schedule(MOONCAKE, selection, salt="s")


def test_mooncake_reuse_and_targets() -> None:
    schedule = build_schedule(MOONCAKE, Selection(), salt="s")
    requests = schedule.requests
    assert [r.reused_blocks for r in requests] == [0, 2, 0, 2]
    assert [r.reusable_tokens for r in requests] == [0, 1024, 0, 1024]
    assert requests[0].target_input_tokens == 1100
    assert requests[0].estimated_prompt_tokens == 1100
    assert requests[3].request_id == "r0000003"
    reuse = schedule.summary()["prefix_reuse"]
    assert reuse["blocks_total"] == 10
    assert reuse["unique_blocks"] == 6
    assert reuse["reusable_fraction_upper_bound"] == pytest.approx(2048 / 4024)
    prompts = list(schedule.prompts())
    assert prompts[0][: 1024 * 5] == prompts[1][: 1024 * 5] == prompts[3]
    assert [sha256_text(p) for p in prompts] == [r.prompt_sha256 for r in requests]


def test_window_does_not_count_reuse_from_unselected_rows() -> None:
    schedule = build_schedule(MOONCAKE, Selection(start_s=0.1), salt="s")
    assert [r.reused_blocks for r in schedule.requests] == [0, 0, 2]


def test_aiperf_text_is_sent_as_is() -> None:
    schedule = build_schedule(AIPERF, Selection(), salt="s")
    assert next(schedule.prompts()) == "Summarize the release notes."
    assert schedule.requests[0].target_input_tokens is None
    assert schedule.requests[0].estimated_prompt_tokens == 4
    assert schedule.summary()["target_input_tokens_total"] is None
    assert "prefix_reuse" not in schedule.summary()


def test_schedule_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    path = tmp_path / "replay_schedule.json"
    schedule = build_schedule(BURST, Selection(time_scale=0.5), salt="fixed")
    _, document = write_schedule(path, schedule)
    loaded = read_schedule(path)
    assert loaded == document
    assert loaded["source"]["sha256"] == BURST.sha256
    assert loaded["source"]["excluded_rows"] == {
        "zero_request_and_response_tokens": 1,
        "zero_response_tokens": 1,
    }
    assert loaded["selection"]["time_scale"] == 0.5
    assert loaded["prompts"]["salt"] == "fixed"
    assert "text" not in json.dumps(loaded["requests"])

    document["requests"][0]["max_tokens"] = 9999
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ReplayArtifactError, match="was edited"):
        read_schedule(path)


def test_same_salt_gives_the_same_schedule() -> None:
    first = build_schedule(MOONCAKE, Selection(), salt="a").to_document()
    second = build_schedule(MOONCAKE, Selection(), salt="a").to_document()
    other = build_schedule(MOONCAKE, Selection(), salt="b").to_document()
    assert first["schedule_sha256"] == second["schedule_sha256"]
    assert first["schedule_sha256"] != other["schedule_sha256"]


def test_multi_turn_history_resends_earlier_turns_with_synthetic_replies() -> None:
    schedule = build_schedule(MULTI, Selection(time_scale=2.0), salt="s")
    builder = schedule.prompt_builder()
    requests = schedule.requests
    assert [r.arrival_offset_s for r in requests] == [0.0, None, None, 0.4, 1.0]
    assert requests[1].delay_s == 0.02 and requests[2].not_before_offset_s == 1.6
    assert schedule.history_at(builder, 0) == ()
    history = schedule.history_at(builder, 2)
    assert [role for role, _ in history] == ["user", "assistant"] * 2
    assert [history[0][1], history[2][1]] == ["Plan a short trip.", "Add a budget."]
    # Each reply has the earlier turn's declared output_length in words.
    assert [word_count(history[1][1]), word_count(history[3][1])] == [6, 4]
    # The same earlier turn has the same bytes in every later request.
    assert schedule.history_at(builder, 1) == history[:2]
    assert requests[2].history_turns == 2
    assert requests[2].history_sha256 == history_sha256(history)
    assert requests[2].prompt_sha256 == sha256_text("Make it shorter.")
    assert requests[2].estimated_prompt_tokens == 3 + sum(
        word_count(text) for _, text in history
    )
    document = schedule.to_document()
    assert document["prompts"]["history"]["assistant_length_provenance"] == (
        "trace_declared"
    )
    assert document["sessions"]["upstream"]["commit"].startswith("fc7d61e")
    assert document["summary"]["sessions"] == 3
    assert document["selection"]["unit"] == "session"
    assert "Plan a short trip" not in json.dumps(document)
    with pytest.raises(ScheduleError, match="no full list"):
        _ = schedule.offsets_s


def test_single_turn_documents_have_no_session_fields() -> None:
    document = build_schedule(AIPERF, Selection(), salt="s").to_document()
    assert "sessions" not in document
    assert "history" not in document["prompts"]
    assert "session" not in document["requests"][0]
    assert "max_sessions" not in document["selection"]


def test_multi_turn_selection_keeps_whole_sessions() -> None:
    first_two = build_schedule(MULTI, Selection(max_sessions=2), salt="s")
    assert [r.source_line for r in first_two.requests] == [1, 1, 3, 4]
    late = build_schedule(MULTI, Selection(start_s=0.1), salt="s")
    assert [r.source_line for r in late.requests] == [4, 2]
    assert late.requests[0].arrival_offset_s == 0.0
    with pytest.raises(ScheduleError, match="--max-sessions"):
        build_schedule(MULTI, Selection(max_requests=2), salt="s")
    with pytest.raises(ScheduleError, match="multi-turn traces only"):
        build_schedule(AIPERF, Selection(max_sessions=1), salt="s")
    with pytest.raises(ScheduleError, match="at least 1"):
        build_schedule(MULTI, Selection(max_sessions=0), salt="s")


WEKA = parse_trace(FIXTURES / "weka_small.jsonl", "weka")
OTEL = parse_trace(FIXTURES / "otel_genai_small.jsonl", "otel-genai")


def test_weka_blocks_share_bytes_within_a_conversation_only() -> None:
    schedule = build_schedule(WEKA, Selection(time_scale=2.0), salt="s")
    prompts = list(schedule.prompts())
    assert [word_count(p) for p in prompts] == [10, 14, 8, 3, 6]
    # Line 1's requests share hash ids 0 and 1: two 4-word blocks.
    assert prompts[0].startswith(prompts[2]) and prompts[1].startswith(prompts[2])
    # Request 1 hashes 3 full blocks; its last 2 words are its own tail.
    assert prompts[1][:60] != prompts[0][:60]
    # Line 2's id 0 is a different block from line 1's id 0.
    assert prompts[4][:20] != prompts[0][:20]
    requests = schedule.requests
    assert [r.reused_blocks for r in requests] == [0, 2, 2, 0, 0]
    assert [r.arrival_offset_s for r in requests] == [0.0, None, None, 0.5, None]
    assert [r.delay_s for r in requests] == [None, 1.0, None, None, None]
    assert [r.not_before_offset_s for r in requests] == [None, None, 3.6, None, 1.0]
    assert {r.history_turns for r in requests} == {0}
    assert {r.history_sha256 for r in requests} == {None}
    builder = schedule.prompt_builder()
    assert schedule.history_at(builder, 1) == ()
    document = schedule.to_document()
    assert "history" not in document["prompts"]
    assert document["sessions"]["upstream"]["commit"].startswith("15e88f7")
    assert "earlier turns are not resent" in document["sessions"]["context"]
    assert document["summary"]["prefix_reuse"]["blocks_reused"] == 4
    assert document["selection"]["window_reference"] == (
        "earliest session start in the trace file"
    )
    assert "conv-secret" not in json.dumps(document)


def test_otel_genai_prompts_are_seeded_per_span() -> None:
    schedule = build_schedule(OTEL, Selection(), salt="s")
    prompts = list(schedule.prompts())
    assert [word_count(p) for p in prompts] == [100, 120, 130, 40]
    assert len({p[:20] for p in prompts}) == 4
    requests = schedule.requests
    assert [r.arrival_offset_s for r in requests] == [0.0, None, None, 0.2]
    assert [r.delay_s for r in requests] == [None, 0.5, None, None]
    assert [r.not_before_offset_s for r in requests] == [None, None, 1.8, None]
    document = schedule.to_document()
    assert "history" not in document["prompts"]
    assert "prefix_reuse" not in document["summary"]
    assert document["sessions"]["upstream"]["commit"].startswith("76e0c19")
    assert [
        r.source_line
        for r in build_schedule(OTEL, Selection(max_sessions=1), salt="s").requests
    ] == [1, 1, 2]
