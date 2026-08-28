"""CLI tests for the workload matrix generator and evaluators."""

from __future__ import annotations

import json
from dataclasses import dataclass

from llmtracefx.optimizer import cli
from llmtracefx.optimizer.collectors.mlx import MLXMemorySnapshot
from llmtracefx.optimizer.workloads.matrix import generate_matrix, write_matrix
from llmtracefx.optimizer.workloads.schema import ContextTier


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


@dataclass
class _FakeResponse:
    text: str = '{"name": "Priya", "age": 34, "is_active": true}'
    from_draft: bool = False
    prompt_tokens: int = 3
    generation_tokens: int = 1
    finish_reason: str | None = None


class _FakeTokenizer:
    bos_token = None


class _FakeMLXRuntime:
    mlx_version = "0.32.0"
    mlx_lm_version = "0.31.3"

    def __init__(self, response_text=None):
        self.response_text = response_text or _FakeResponse.text
        self.load_calls = []

    def load_model(self, path):
        self.load_calls.append(path)
        return object(), _FakeTokenizer()

    def encode(self, tokenizer, prompt):
        return [1, 2, 3]

    def seed(self, seed):
        pass

    def synchronize(self):
        pass

    def reset_peak_memory(self):
        pass

    def memory_snapshot(self):
        return MLXMemorySnapshot(active_bytes=1024, cache_bytes=256, peak_bytes=2048)

    def accelerator_name(self):
        return "Apple M5 Pro (test)"

    def stream_generate(
        self,
        model,
        tokenizer,
        prompt_tokens,
        *,
        max_tokens,
        draft_model,
        num_draft_tokens,
    ):
        yield _FakeResponse(self.response_text)


def _build_small_matrix(tmp_path):
    from llmtracefx.optimizer.workloads.catalog import (
        STRUCTURED_JSON_PROFILE_EXTRACTION,
    )

    output_dir = tmp_path / "matrix"
    manifest = generate_matrix(
        model_id="local/test-model",
        model_family="qwen3_next",
        output_dir=str(output_dir),
        workloads=(STRUCTURED_JSON_PROFILE_EXTRACTION,),
        context_tiers=(ContextTier.TIER_2K,),
        mtp_depths=(2,),
    )
    write_matrix(manifest)
    return output_dir / "manifest.json"


def test_workloads_run_cli_executes_autoregressive_rows(tmp_path, monkeypatch):
    matrix_path = _build_small_matrix(tmp_path)
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: _FakeMLXRuntime())

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "workloads",
            "run",
            "--matrix",
            str(matrix_path),
            "--model-path",
            str(model_path),
            "--output-dir",
            str(results_dir),
            "--mode",
            "autoregressive",
        ]
    )
    assert args.func(args) == 0
    run_dirs = list((results_dir / "runs").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "final_record.json").exists()
    assert (run_dirs[0] / "verification.json").exists()


def test_workloads_run_cli_rejects_native_mtp_rows_as_unsupported(
    tmp_path, monkeypatch
):
    matrix_path = _build_small_matrix(tmp_path)
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: _FakeMLXRuntime())

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "workloads",
            "run",
            "--matrix",
            str(matrix_path),
            "--model-path",
            str(model_path),
            "--output-dir",
            str(results_dir),
            "--mode",
            "native-mtp",
        ]
    )
    assert args.func(args) == 0
    verification = json.loads(
        (next((results_dir / "runs").iterdir()) / "verification.json").read_text()
    )
    assert verification["status"] == "unsupported"


def test_workloads_run_cli_requires_model_path_without_dry_run(tmp_path):
    matrix_path = _build_small_matrix(tmp_path)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "workloads",
            "run",
            "--matrix",
            str(matrix_path),
            "--output-dir",
            str(tmp_path / "results"),
        ]
    )
    assert args.func(args) == 1


def test_workloads_run_cli_dry_run_reports_blockers_without_loading_model(
    tmp_path, capsys
):
    matrix_path = _build_small_matrix(tmp_path)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "workloads",
            "run",
            "--matrix",
            str(matrix_path),
            "--output-dir",
            str(tmp_path / "results"),
            "--mode",
            "autoregressive",
            "--dry-run",
        ]
    )
    assert args.func(args) == 2  # blocked: no --model-path given
    out = capsys.readouterr().out
    assert "no model was loaded or downloaded" in out
    assert "blocker" in out


def test_workloads_run_cli_dry_run_ready_with_valid_model_path(tmp_path, capsys):
    matrix_path = _build_small_matrix(tmp_path)
    model_path = tmp_path / "model"
    model_path.mkdir()

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "workloads",
            "run",
            "--matrix",
            str(matrix_path),
            "--model-path",
            str(model_path),
            "--output-dir",
            str(tmp_path / "results"),
            "--mode",
            "autoregressive",
            "--dry-run",
        ]
    )
    assert args.func(args) == 0
    assert "READY" in capsys.readouterr().out


def test_workloads_run_cli_reports_runtime_failure_with_exit_1(tmp_path, monkeypatch):
    matrix_path = _build_small_matrix(tmp_path)
    model_path = tmp_path / "model"
    model_path.mkdir()

    class FailingRuntime(_FakeMLXRuntime):
        def load_model(self, path):
            raise RuntimeError("boom")

    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: FailingRuntime())

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "workloads",
            "run",
            "--matrix",
            str(matrix_path),
            "--model-path",
            str(model_path),
            "--output-dir",
            str(tmp_path / "results"),
            "--mode",
            "autoregressive",
        ]
    )
    assert args.func(args) == 1


def test_workloads_run_cli_unknown_matrix_manifest_fails(tmp_path, capsys):
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "workloads",
            "run",
            "--matrix",
            str(tmp_path / "missing-manifest.json"),
            "--output-dir",
            str(tmp_path / "results"),
        ]
    )
    assert args.func(args) == 1
    assert "Failed to load matrix manifest" in capsys.readouterr().err


def test_workloads_summarize_cli_reports_pass_rate(tmp_path, monkeypatch):
    matrix_path = _build_small_matrix(tmp_path)
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: _FakeMLXRuntime())

    run_parser = cli.build_parser()
    run_args = run_parser.parse_args(
        [
            "workloads",
            "run",
            "--matrix",
            str(matrix_path),
            "--model-path",
            str(model_path),
            "--output-dir",
            str(results_dir),
            "--mode",
            "autoregressive",
        ]
    )
    assert run_args.func(run_args) == 0

    summarize_parser = cli.build_parser()
    summarize_args = summarize_parser.parse_args(
        ["workloads", "summarize", "--results", str(results_dir)]
    )
    assert summarize_args.func(summarize_args) == 0


def test_workloads_summarize_cli_writes_to_output_file(tmp_path, monkeypatch):
    matrix_path = _build_small_matrix(tmp_path)
    model_path = tmp_path / "model"
    model_path.mkdir()
    results_dir = tmp_path / "results"
    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: _FakeMLXRuntime())

    run_parser = cli.build_parser()
    run_args = run_parser.parse_args(
        [
            "workloads",
            "run",
            "--matrix",
            str(matrix_path),
            "--model-path",
            str(model_path),
            "--output-dir",
            str(results_dir),
        ]
    )
    run_args.func(run_args)

    summary_path = tmp_path / "summary.json"
    summarize_parser = cli.build_parser()
    summarize_args = summarize_parser.parse_args(
        [
            "workloads",
            "summarize",
            "--results",
            str(results_dir),
            "--output",
            str(summary_path),
        ]
    )
    assert summarize_args.func(summarize_args) == 0
    payload = json.loads(summary_path.read_text())
    assert payload["overall"]["total"] > 0
