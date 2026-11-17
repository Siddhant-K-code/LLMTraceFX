import json
from pathlib import Path

import pytest

from llmtracefx.optimizer import cli
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
