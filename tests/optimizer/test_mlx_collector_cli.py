"""CLI tests for local MLX evidence collection."""

from dataclasses import dataclass
from pathlib import Path

from llmtracefx.optimizer import cli
from llmtracefx.optimizer.collectors.mlx import MLXMemorySnapshot
from llmtracefx.optimizer.schema import ExperimentRecord


@dataclass
class FakeResponse:
    text: str = "ok"
    from_draft: bool = False
    prompt_tokens: int = 2
    generation_tokens: int = 1
    finish_reason: str | None = None


class FakeTokenizer:
    bos_token = None


class FakeRuntime:
    mlx_version = "0.32.0"
    mlx_lm_version = "0.31.3"

    def __init__(self):
        self.snapshot = MLXMemorySnapshot()

    def load_model(self, path):
        return object(), FakeTokenizer()

    def encode(self, tokenizer, prompt):
        return [1, 2]

    def seed(self, seed):
        pass

    def synchronize(self):
        pass

    def reset_peak_memory(self):
        pass

    def memory_snapshot(self):
        return self.snapshot

    def accelerator_name(self):
        return "Apple M5 Pro"

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
        yield FakeResponse()


def test_collect_mlx_cli_uses_local_paths_and_writes_evidence(tmp_path, monkeypatch):
    model_path = tmp_path / "model"
    model_path.mkdir()
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("test prompt", encoding="utf-8")
    output_dir = tmp_path / "artifacts"
    runtime = FakeRuntime()
    runtime.snapshot = MLXMemorySnapshot(active_bytes=42)
    monkeypatch.setattr(cli, "MLXLMRuntime", lambda: runtime)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "collect-mlx",
            "--run-id",
            "cli-mlx-1",
            "--model-path",
            str(model_path),
            "--model-id",
            "local/test-model",
            "--prompt-file",
            str(prompt_path),
            "--output-dir",
            str(output_dir),
            "--quantization",
            "4bit",
        ]
    )

    assert args.func(args) == 0
    record = ExperimentRecord.read_json(output_dir / "record.json")
    assert record.run_id == "cli-mlx-1"
    assert record.model.quantization == "4bit"
    assert record.memory.active.value == 42
    assert Path(record.command.argv[0]).name == "llmtracefx-optimizer"


def test_collect_mlx_cli_fallback_records_all_reproducibility_flags(
    tmp_path, monkeypatch
):
    model_path = tmp_path / "model"
    model_path.mkdir()
    draft_path = tmp_path / "draft"
    draft_path.mkdir()
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("test prompt", encoding="utf-8")
    output_dir = tmp_path / "artifacts"
    monkeypatch.setattr(cli, "MLXLMRuntime", FakeRuntime)

    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "collect-mlx",
            "--run-id",
            "cli-mlx-2",
            "--model-path",
            str(model_path),
            "--model-id",
            "local/test-model",
            "--model-revision",
            "model-sha",
            "--tokenizer-revision",
            "tokenizer-sha",
            "--quantization",
            "4bit",
            "--accelerator",
            "Apple M5 Pro",
            "--draft-model-path",
            str(draft_path),
            "--num-draft-tokens",
            "4",
            "--prompt-file",
            str(prompt_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert args.func(args) == 0
    argv = ExperimentRecord.read_json(output_dir / "record.json").command.argv
    for expected in (
        "--model-revision",
        "model-sha",
        "--tokenizer-revision",
        "tokenizer-sha",
        "--quantization",
        "4bit",
        "--accelerator",
        "Apple M5 Pro",
        "--draft-model-path",
        str(draft_path),
        "--num-draft-tokens",
        "4",
    ):
        assert expected in argv


def test_collect_mlx_cli_reports_missing_prompt_without_traceback(tmp_path, capsys):
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "collect-mlx",
            "--run-id",
            "cli-mlx-1",
            "--model-path",
            str(tmp_path / "model"),
            "--model-id",
            "local/test-model",
            "--prompt-file",
            str(tmp_path / "missing.txt"),
            "--output-dir",
            str(tmp_path / "artifacts"),
        ]
    )

    assert args.func(args) == 1
    assert "Failed to collect MLX evidence" in capsys.readouterr().err
