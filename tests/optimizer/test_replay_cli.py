import json
import sys
from pathlib import Path

import pytest
from _tokenizer_fakes import ByteLevel, fake_tokenizer

from llmtracefx.optimizer import cli
from llmtracefx.optimizer.replay import cli as replay_cli
from llmtracefx.optimizer.replay.schedule import read_schedule

FIXTURES = Path(__file__).parent / "fixtures" / "replay"
MOONCAKE = str(FIXTURES / "mooncake_small.jsonl")
BURST = str(FIXTURES / "burstgpt_v1_small.csv")


def _run(argv: list[str]) -> object:
    with pytest.raises(SystemExit) as info:
        cli.main(argv)
    return info.value.code


def test_validate_prints_shape_and_reuse(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["replay", "validate", MOONCAKE, "--format", "mooncake"]) == 0
    out = capsys.readouterr().out
    assert "valid mooncake trace" in out
    assert "4 read, 4 replayable" in out
    assert "6 unique of 10" in out


def test_validate_json_counts_exclusions(capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["replay", "validate", BURST, "--format", "burstgpt", "--exclude-failed"]
    assert _run([*argv, "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["source"]["excluded_rows"] == {
        "zero_request_and_response_tokens": 1,
        "zero_response_tokens": 1,
    }
    assert stats["model_counts"] == {"ChatGPT": 2, "GPT-4": 1}
    assert stats["span_s"] == 197.0


def test_validate_refuses_with_status_1(capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["replay", "validate", BURST, "--format", "burstgpt"]) == 1
    assert "replay: burstgpt_v1_small.csv:3:" in capsys.readouterr().err


def test_format_is_required() -> None:
    assert _run(["replay", "validate", MOONCAKE]) == 2


def test_plan_writes_a_sealed_schedule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out_dir = tmp_path / "out"
    argv = [
        "replay",
        "plan",
        MOONCAKE,
        "--format",
        "mooncake",
        "--time-scale",
        "2",
        "--max-requests",
        "3",
        "--prompt-salt",
        "fixed",
        "--output-dir",
        str(out_dir),
    ]
    assert _run(argv) == 0
    assert "prompt salt: fixed" in capsys.readouterr().out
    schedule = read_schedule(out_dir / "replay_schedule.json")
    assert [r["arrival_offset_s"] for r in schedule["requests"]] == [0.0, 0.5, 0.5]
    assert schedule["selection"]["max_requests"] == 3

    assert _run(argv) == 1
    assert "already holds replay_schedule.json" in capsys.readouterr().err


def test_plan_records_a_random_salt(tmp_path: Path) -> None:
    for name in ("a", "b"):
        argv = ["replay", "plan", MOONCAKE, "--format", "mooncake"]
        assert _run([*argv, "--output-dir", str(tmp_path / name)]) == 0
    salts = {
        read_schedule(tmp_path / name / "replay_schedule.json")["prompts"]["salt"]
        for name in ("a", "b")
    }
    assert len(salts) == 2


def test_plan_refuses_bad_salt_and_non_finite_scale(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["replay", "plan", MOONCAKE, "--format", "mooncake"]
    out = ["--output-dir", str(tmp_path / "x")]
    assert _run([*base, "--prompt-salt", "a b", *out]) == 1
    assert "--prompt-salt" in capsys.readouterr().err
    assert _run([*base, "--time-scale", "nan", *out]) == 2
    assert not (tmp_path / "x").exists()


def test_plan_with_a_tokenizer_records_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None]] = []

    def load(spec: str, revision: str | None = None) -> object:
        calls.append((spec, revision))
        return fake_tokenizer(ByteLevel(), name=spec)

    monkeypatch.setattr(replay_cli, "load_tokenizer", load)
    out_dir = tmp_path / "out"
    argv = ["replay", "plan", MOONCAKE, "--format", "mooncake"]
    tokenizer = ["--tokenizer", "org/model", "--tokenizer-revision", "v1"]
    assert _run([*argv, *tokenizer, "--output-dir", str(out_dir)]) == 0
    assert calls == [("org/model", "v1")]
    assert "tokenizer: org/model" in capsys.readouterr().out
    schedule = read_schedule(out_dir / "replay_schedule.json")
    assert schedule["prompts"]["tokenizer"]["name"] == "org/model"
    assert [r["tokenizer_prompt_tokens"] for r in schedule["requests"]] == [
        r["target_input_tokens"] for r in schedule["requests"]
    ]


def test_plan_refuses_tokenizer_problems(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    base = ["replay", "plan", MOONCAKE, "--format", "mooncake"]
    out = ["--output-dir", str(tmp_path / "x")]
    assert _run([*base, "--tokenizer-revision", "v1", *out]) == 1
    assert "--tokenizer-revision needs --tokenizer" in capsys.readouterr().err
    monkeypatch.setitem(sys.modules, "tokenizers", None)
    assert _run([*base, "--tokenizer", "org/model", *out]) == 1
    assert "--extra tokenizer" in capsys.readouterr().err
    assert not (tmp_path / "x").exists()


MULTI = str(FIXTURES / "aiperf_multi_turn.jsonl")


def test_validate_and_plan_a_multi_turn_trace(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(["replay", "validate", MULTI, "--format", "aiperf-multi-turn"]) == 0
    out = capsys.readouterr().out
    assert "sessions: 3, turns per session median 1, max 3" in out
    argv = ["replay", "plan", MULTI, "--format", "aiperf-multi-turn"]
    output = tmp_path / "plan"
    assert _run([*argv, "--max-sessions", "1", "--output-dir", str(output)]) == 0
    assert "1 sessions" in capsys.readouterr().out
    assert len(read_schedule(output / "replay_schedule.json")["requests"]) == 3
    refused = [*argv, "--max-requests", "1", "--output-dir", str(tmp_path / "x")]
    assert _run(refused) == 1
    assert "--max-sessions" in capsys.readouterr().err
