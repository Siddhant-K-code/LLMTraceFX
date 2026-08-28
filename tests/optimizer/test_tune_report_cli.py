"""Tests for the `llmtracefx-optimizer tune-report` CLI subcommand.

Covers file errors, exit codes, atomic output, the full synthetic example
render, and that no network/CDN references leak into the generated HTML.
"""

from __future__ import annotations

import json
from pathlib import Path

from _tune_fixtures import write_run

from llmtracefx.optimizer.cli import build_parser
from llmtracefx.optimizer.tune.policy import TuneObjective, TunePolicy
from llmtracefx.optimizer.tune.tuner import tune

EXAMPLE_REPORT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "optimizer"
    / "tune-report-example.json"
)


def _run_cli(argv):
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def _write_report(tmp_path: Path) -> Path:
    write_run(tmp_path / "results", "r1", total_ms=1000.0)
    policy = TunePolicy(objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS)
    report = tune(results_dirs=(tmp_path / "results",), policy=policy)
    report_path = tmp_path / "report.json"
    report_path.write_text(report.to_json(), encoding="utf-8")
    return report_path


def test_tune_report_cli_writes_html_and_exits_0(tmp_path):
    report_path = _write_report(tmp_path)
    output_path = tmp_path / "report.html"

    exit_code = _run_cli(
        ["tune-report", "--input", str(report_path), "--output", str(output_path)]
    )

    assert exit_code == 0
    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert content.startswith("<!DOCTYPE html>")
    assert "RECOMMENDED" in content


def test_tune_report_cli_exits_1_on_missing_input_file(tmp_path):
    exit_code = _run_cli(
        [
            "tune-report",
            "--input",
            str(tmp_path / "does-not-exist.json"),
            "--output",
            str(tmp_path / "out.html"),
        ]
    )

    assert exit_code == 1


def test_tune_report_cli_exits_1_on_malformed_json(tmp_path):
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("not json", encoding="utf-8")

    exit_code = _run_cli(
        [
            "tune-report",
            "--input",
            str(bad_path),
            "--output",
            str(tmp_path / "out.html"),
        ]
    )

    assert exit_code == 1


def test_tune_report_cli_exits_1_on_invalid_report_schema(tmp_path):
    bad_path = tmp_path / "bad_report.json"
    bad_path.write_text(json.dumps({"not": "a tune report"}), encoding="utf-8")

    exit_code = _run_cli(
        [
            "tune-report",
            "--input",
            str(bad_path),
            "--output",
            str(tmp_path / "out.html"),
        ]
    )

    assert exit_code == 1


def test_tune_report_cli_redacts_paths_by_default(tmp_path):
    report_path = _write_report(tmp_path)
    output_path = tmp_path / "report.html"

    _run_cli(["tune-report", "--input", str(report_path), "--output", str(output_path)])

    content = output_path.read_text(encoding="utf-8")
    assert str(tmp_path) not in content


def test_tune_report_cli_include_paths_flag_includes_full_paths(tmp_path):
    report_path = _write_report(tmp_path)
    output_path = tmp_path / "report.html"

    _run_cli(
        [
            "tune-report",
            "--input",
            str(report_path),
            "--output",
            str(output_path),
            "--include-paths",
        ]
    )

    content = output_path.read_text(encoding="utf-8")
    assert str(tmp_path / "results") in content


def test_tune_report_cli_writes_output_atomically(tmp_path):
    report_path = _write_report(tmp_path)
    output_path = tmp_path / "report.html"
    output_path.write_text("stale content", encoding="utf-8")

    exit_code = _run_cli(
        ["tune-report", "--input", str(report_path), "--output", str(output_path)]
    )

    assert exit_code == 0
    content = output_path.read_text(encoding="utf-8")
    assert "stale content" not in content
    # No leftover temp file from the atomic-write helper.
    leftovers = [p for p in output_path.parent.iterdir() if p.name.startswith(".")]
    assert leftovers == []


def test_tune_report_cli_renders_full_synthetic_example(tmp_path):
    output_path = tmp_path / "example.html"

    exit_code = _run_cli(
        [
            "tune-report",
            "--input",
            str(EXAMPLE_REPORT_PATH),
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    content = output_path.read_text(encoding="utf-8")
    assert "RECOMMENDED" in content
    assert "Rejected candidates" in content
    assert "Speculative baseline comparison" in content
    assert "Excluded runs" in content
    assert "SYNTHETIC" in content


def test_tune_report_cli_output_has_no_network_or_cdn_references(tmp_path):
    output_path = tmp_path / "example.html"

    _run_cli(
        [
            "tune-report",
            "--input",
            str(EXAMPLE_REPORT_PATH),
            "--output",
            str(output_path),
        ]
    )

    content = output_path.read_text(encoding="utf-8")
    for token in ("http://", "https://", "<script", "cdn."):
        assert token not in content


def test_tune_report_cli_output_is_deterministic(tmp_path):
    output_path_a = tmp_path / "a.html"
    output_path_b = tmp_path / "b.html"

    _run_cli(
        [
            "tune-report",
            "--input",
            str(EXAMPLE_REPORT_PATH),
            "--output",
            str(output_path_a),
        ]
    )
    _run_cli(
        [
            "tune-report",
            "--input",
            str(EXAMPLE_REPORT_PATH),
            "--output",
            str(output_path_b),
        ]
    )

    assert output_path_a.read_text() == output_path_b.read_text()
