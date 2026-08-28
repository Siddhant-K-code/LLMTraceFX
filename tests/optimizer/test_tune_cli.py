"""Tests for the `llmtracefx-optimizer tune` CLI subcommand."""

from __future__ import annotations

import json

from _tune_fixtures import write_run

from llmtracefx.optimizer.cli import build_parser
from llmtracefx.optimizer.workloads.verify import RowStatus


def _run_tune(argv):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _write_policy(tmp_path, payload):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(payload), encoding="utf-8")
    return policy_path


def test_tune_cli_exits_0_and_writes_report_when_recommended(tmp_path, capsys):
    results_dir = tmp_path / "results"
    write_run(results_dir, "r1", total_ms=1000.0)
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})
    output_path = tmp_path / "report.json"

    exit_code = _run_tune(
        [
            "tune",
            "--results",
            str(results_dir),
            "--policy",
            str(policy_path),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "RECOMMENDED" in captured.out
    assert output_path.exists()
    payload = json.loads(output_path.read_text())
    assert payload["groups"][0]["outcome"] == "recommended"


def test_tune_cli_exits_2_when_inconclusive(tmp_path, capsys):
    results_dir = tmp_path / "results"
    write_run(results_dir, "r1", status=RowStatus.FAILED, success=False)
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})

    exit_code = _run_tune(
        ["tune", "--results", str(results_dir), "--policy", str(policy_path)]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "INCONCLUSIVE" in captured.out


def test_tune_cli_exits_1_when_no_groups_found(tmp_path, capsys):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})

    exit_code = _run_tune(
        ["tune", "--results", str(results_dir), "--policy", str(policy_path)]
    )

    assert exit_code == 1


def test_tune_cli_exits_1_on_invalid_policy(tmp_path):
    results_dir = tmp_path / "results"
    write_run(results_dir, "r1")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text("not json", encoding="utf-8")

    exit_code = _run_tune(
        ["tune", "--results", str(results_dir), "--policy", str(policy_path)]
    )

    assert exit_code == 1


def test_tune_cli_exits_1_on_duplicate_conflict(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    write_run(dir_a, "r1", total_ms=1000.0)
    write_run(dir_b, "r1", total_ms=9999.0)
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})

    exit_code = _run_tune(
        [
            "tune",
            "--results",
            str(dir_a),
            str(dir_b),
            "--policy",
            str(policy_path),
        ]
    )

    assert exit_code == 1


def test_tune_cli_explain_flag_shows_every_rejection_reason(tmp_path, capsys):
    results_dir = tmp_path / "results"
    write_run(
        results_dir,
        "r1",
        status=RowStatus.FAILED,
        success=False,
        total_ms=10_000.0,
        peak_bytes=25 * 1024**3,
    )
    policy_path = _write_policy(
        tmp_path,
        {
            "objective": "min_mean_total_latency_ms",
            "constraints": {
                "max_total_latency_ms": 1000,
                "max_peak_memory_bytes": 20 * 1024**3,
            },
        },
    )

    exit_code = _run_tune(
        [
            "tune",
            "--results",
            str(results_dir),
            "--policy",
            str(policy_path),
            "--explain",
        ]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "status" in captured.out
    assert "total latency" in captured.out
    assert "peak memory" in captured.out
    assert "more reason(s)" not in captured.out


def test_tune_cli_default_mode_truncates_rejection_reasons(tmp_path, capsys):
    results_dir = tmp_path / "results"
    write_run(
        results_dir,
        "r1",
        status=RowStatus.FAILED,
        success=False,
        total_ms=10_000.0,
        peak_bytes=25 * 1024**3,
    )
    policy_path = _write_policy(
        tmp_path,
        {
            "objective": "min_mean_total_latency_ms",
            "constraints": {
                "max_total_latency_ms": 1000,
                "max_peak_memory_bytes": 20 * 1024**3,
            },
        },
    )

    exit_code = _run_tune(
        ["tune", "--results", str(results_dir), "--policy", str(policy_path)]
    )

    assert exit_code == 2
    captured = capsys.readouterr()
    assert "more reason(s)" in captured.out


def test_tune_cli_accepts_multiple_results_dirs(tmp_path, capsys):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    write_run(dir_a, "r1", total_ms=1000.0)
    write_run(dir_b, "r2", total_ms=1000.0)
    policy_path = _write_policy(tmp_path, {"objective": "min_mean_total_latency_ms"})

    exit_code = _run_tune(
        [
            "tune",
            "--results",
            str(dir_a),
            str(dir_b),
            "--policy",
            str(policy_path),
        ]
    )

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "evidence=2" in captured.out
