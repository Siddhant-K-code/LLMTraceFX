"""CLI tests for the workload matrix generator and evaluators."""

from __future__ import annotations

import json

from llmtracefx.optimizer import cli


def test_workloads_list_cli_prints_catalog(capsys):
    parser = cli.build_parser()
    args = parser.parse_args(["workloads", "list"])
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "code-completion-palindrome-check" in out
    assert "structured-json-profile-extraction" in out
    assert "prose-reasoning-two-train-gap" in out


def test_workloads_generate_matrix_cli_is_dry_run(tmp_path, capsys):
    output_dir = tmp_path / "matrix"
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "workloads",
            "generate-matrix",
            "--model-id",
            "Qwen/Qwen3.8-27B",
            "--model-family",
            "qwen3_next",
            "--output-dir",
            str(output_dir),
        ]
    )
    assert args.func(args) == 0
    out = capsys.readouterr().out
    assert "No model was loaded or downloaded" in out
    assert (output_dir / "manifest.json").exists()


def test_workloads_generate_matrix_cli_respects_context_tier_filter(tmp_path):
    output_dir = tmp_path / "matrix"
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "workloads",
            "generate-matrix",
            "--model-id",
            "m",
            "--model-family",
            "qwen3_next",
            "--output-dir",
            str(output_dir),
            "--context-tiers",
            "2k",
        ]
    )
    assert args.func(args) == 0
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert all(entry["context_tier"] == "2k" for entry in manifest["entries"])


def test_workloads_evaluate_cli_passes_correct_response(tmp_path):
    response_path = tmp_path / "response.txt"
    response_path.write_text(
        '{"name": "Priya", "age": 34, "is_active": true}', encoding="utf-8"
    )

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "workloads",
            "evaluate",
            "--workload-id",
            "structured-json-profile-extraction",
            "--response-file",
            str(response_path),
        ]
    )
    assert args.func(args) == 0


def test_workloads_evaluate_cli_fails_wrong_response(tmp_path, capsys):
    response_path = tmp_path / "response.txt"
    response_path.write_text("not json", encoding="utf-8")

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "workloads",
            "evaluate",
            "--workload-id",
            "structured-json-profile-extraction",
            "--response-file",
            str(response_path),
        ]
    )
    assert args.func(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["success"] is False


def test_workloads_evaluate_cli_reports_unknown_workload(tmp_path, capsys):
    response_path = tmp_path / "response.txt"
    response_path.write_text("x", encoding="utf-8")

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "workloads",
            "evaluate",
            "--workload-id",
            "does-not-exist",
            "--response-file",
            str(response_path),
        ]
    )
    assert args.func(args) == 1
    assert "Unknown workload" in capsys.readouterr().err
