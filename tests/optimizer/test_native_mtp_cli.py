"""CLI tests for native-MTP capability report/collection subcommands."""

from __future__ import annotations

import json
from pathlib import Path

from llmtracefx.optimizer import cli


def _write_config(path: Path, **fields) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {"model_type": "qwen3_next", "hidden_size": 4096, "vocab_size": 151936}
    payload.update(fields)
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def test_capability_report_cli_prints_unsupported_and_exits_3(tmp_path, capsys):
    target = tmp_path / "target"
    _write_config(target)

    parser = cli.build_parser()
    args = parser.parse_args(
        ["native-mtp", "capability-report", "--target-model-path", str(target)]
    )
    assert args.func(args) == 3
    payload = json.loads(capsys.readouterr().out)
    assert payload["supported"] is False
    assert payload["model_family"] == "qwen3_next"


def test_capability_report_cli_writes_to_output_file(tmp_path):
    target = tmp_path / "target"
    _write_config(target)
    output = tmp_path / "capability.json"

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "native-mtp",
            "capability-report",
            "--target-model-path",
            str(target),
            "--output",
            str(output),
        ]
    )
    assert args.func(args) == 3
    assert json.loads(output.read_text())["supported"] is False


def test_capability_report_cli_reports_missing_checkpoint(tmp_path, capsys):
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "native-mtp",
            "capability-report",
            "--target-model-path",
            str(tmp_path / "missing"),
        ]
    )
    assert args.func(args) == 1
    assert "Failed to determine" in capsys.readouterr().err


def test_native_mtp_collect_cli_writes_unsupported_record(tmp_path):
    target = tmp_path / "target"
    sidecar = tmp_path / "sidecar"
    _write_config(target)
    _write_config(sidecar)
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("hello", encoding="utf-8")
    output_dir = tmp_path / "artifacts"

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "native-mtp",
            "collect",
            "--run-id",
            "cli-native-mtp-1",
            "--target-model-path",
            str(target),
            "--mtp-sidecar-path",
            str(sidecar),
            "--model-id",
            "local/qwen3.8-27b",
            "--prompt-file",
            str(prompt_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert args.func(args) == 1
    record = json.loads((output_dir / "record.json").read_text())
    assert record["outcome"]["success"] is False
    assert record["error"]["category"] == "NativeMTPUnsupported"
    assert (output_dir / "capability_report.json").exists()
    assert not (output_dir / "response.txt").exists()


def test_native_mtp_collect_cli_reports_checkpoint_mismatch(tmp_path, capsys):
    target = tmp_path / "target"
    sidecar = tmp_path / "sidecar"
    _write_config(target, hidden_size=4096)
    _write_config(sidecar, hidden_size=2048)
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("hello", encoding="utf-8")

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "native-mtp",
            "collect",
            "--run-id",
            "cli-native-mtp-2",
            "--target-model-path",
            str(target),
            "--mtp-sidecar-path",
            str(sidecar),
            "--model-id",
            "local/qwen3.8-27b",
            "--prompt-file",
            str(prompt_path),
            "--output-dir",
            str(tmp_path / "artifacts"),
        ]
    )
    assert args.func(args) == 1
    assert "Failed to collect native-MTP evidence" in capsys.readouterr().err
