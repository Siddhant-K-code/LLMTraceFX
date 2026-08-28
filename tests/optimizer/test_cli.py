"""Tests for the `llmtracefx-optimizer parse-llama-cpp` CLI subcommand.

Focuses on the accelerator precedence contract: an explicit --accelerator
flag must win over any device name llama.cpp reports in its own output,
and the parsed device hint must otherwise be used to fill
`platform.accelerator` so downstream comparability checks (e.g. the
speculative-decoding doctor rule) can tell different accelerators apart.
"""

from pathlib import Path

from llmtracefx.optimizer.cli import build_parser
from llmtracefx.optimizer.schema import ExperimentRecord

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "llama_cpp"


def _run_parse_llama_cpp(tmp_path, *, accelerator=None, output_name="record.json"):
    output_path = tmp_path / output_name
    argv = [
        "parse-llama-cpp",
        "--run-id",
        "cli-run-1",
        "--model-id",
        "Qwen/Qwen3.8-27B",
        "--quantization",
        "Q4_K_M",
        "--stdout-file",
        str(FIXTURES_DIR / "qwen3_8b_baseline_run1.log"),
        "--output",
        str(output_path),
    ]
    if accelerator is not None:
        argv += ["--accelerator", accelerator]
    argv += ["--", "llama-cli", "-m", "qwen3.8-27b-q4.gguf"]

    parser = build_parser()
    args = parser.parse_args(argv)
    exit_code = args.func(args)
    assert exit_code == 0
    return ExperimentRecord.read_json(output_path)


def test_parse_llama_cpp_fills_accelerator_from_device_hint_by_default(tmp_path):
    record = _run_parse_llama_cpp(tmp_path)
    assert record.platform.accelerator == "Apple M5 Pro"


def test_parse_llama_cpp_explicit_accelerator_overrides_device_hint(tmp_path):
    record = _run_parse_llama_cpp(tmp_path, accelerator="NVIDIA RTX 4090")
    assert record.platform.accelerator == "NVIDIA RTX 4090"
