"""Tests for the llama.cpp text-output parser/collector."""

from pathlib import Path

import pytest

from llmtracefx.optimizer.parsers.llama_cpp import (
    LlamaCppParseError,
    build_experiment_record,
    parse_llama_cpp_output,
)
from llmtracefx.optimizer.schema import (
    CommandInfo,
    ModelInfo,
    PlatformInfo,
    RepetitionInfo,
    utc_now_iso,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "llama_cpp"


def _read_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def _platform() -> PlatformInfo:
    return PlatformInfo(os_name="Darwin", os_version="24.0", architecture="arm64")


def _model() -> ModelInfo:
    return ModelInfo(model_id="Qwen/Qwen3.8-27B", quantization="Q4_K_M")


def _repetition(index: int = 0) -> RepetitionInfo:
    return RepetitionInfo(
        warmup_repetitions=1, measured_repetitions=2, repetition_index=index
    )


def test_parses_baseline_timings():
    text = _read_fixture("qwen3_8b_baseline_run1.log")
    parsed = parse_llama_cpp_output(text)

    assert parsed.load_ms == pytest.approx(345.20)
    assert parsed.prompt_eval_ms == pytest.approx(118.40)
    assert parsed.prompt_eval_tokens == 50
    assert parsed.eval_ms == pytest.approx(4790.10)
    assert parsed.eval_tokens == 200
    assert parsed.total_ms == pytest.approx(5160.20)
    assert parsed.total_tokens == 250
    assert parsed.device_hint == "Apple M5 Pro"
    assert parsed.backend_hint == "Metal"
    assert not parsed.speculative_reported


def test_parses_speculative_counters():
    text = _read_fixture("qwen3_8b_mtp_regression_run1.log")
    parsed = parse_llama_cpp_output(text)

    assert parsed.n_draft == 16
    assert parsed.n_predict == 200
    assert parsed.n_drafted == 320
    assert parsed.n_accepted == 96
    assert parsed.speculative_reported


def test_tolerates_missing_optional_lines():
    parsed = parse_llama_cpp_output(
        "llama_perf_context_print:        load time =     345.20 ms\n"
    )

    assert parsed.load_ms == pytest.approx(345.20)
    assert parsed.prompt_eval_ms is None
    assert parsed.eval_ms is None
    assert parsed.total_ms is None
    assert parsed.n_draft is None
    assert not parsed.speculative_reported


def test_empty_text_yields_all_none():
    parsed = parse_llama_cpp_output("")
    assert parsed.load_ms is None
    assert parsed.total_ms is None


def test_malformed_load_time_raises_explicitly():
    text = _read_fixture("malformed_load_time.log")
    with pytest.raises(LlamaCppParseError, match="load time"):
        parse_llama_cpp_output(text)


def test_build_experiment_record_from_baseline_fixture():
    record = build_experiment_record(
        run_id="baseline-run-1",
        started_at=utc_now_iso(),
        platform=_platform(),
        model=_model(),
        command=CommandInfo(
            argv=("llama-cli", "-m", "qwen3.8-27b-q4.gguf", "-p", "prompt", "-n", "200")
        ),
        repetition=_repetition(),
        stdout_text=_read_fixture("qwen3_8b_baseline_run1.log"),
        runtime_version="b4500",
    )

    assert record.runtime.name == "llama.cpp"
    assert record.runtime.backend == "Metal"
    assert record.timing.total is not None
    assert record.timing.total.value == pytest.approx(5160.20)
    assert record.tokens.input_tokens == 50
    assert record.tokens.generated_tokens == 200
    assert record.speculative.enabled is False
    assert record.speculative.method is None


def test_build_experiment_record_from_speculative_fixture_sets_method():
    record = build_experiment_record(
        run_id="mtp-run-1",
        started_at=utc_now_iso(),
        platform=_platform(),
        model=_model(),
        command=CommandInfo(
            argv=("llama-cli", "-m", "qwen3.8-27b-q4.gguf", "--draft-max", "16")
        ),
        repetition=_repetition(),
        stdout_text=_read_fixture("qwen3_8b_mtp_regression_run1.log"),
        speculative_method="mtp",
    )

    assert record.speculative.enabled is True
    assert record.speculative.method == "mtp"
    assert record.speculative.configured_depth == 16
    assert record.speculative.proposed_tokens == 320
    assert record.speculative.accepted_tokens == 96
    assert record.speculative.acceptance_rate == pytest.approx(96 / 320)


def test_build_experiment_record_fills_accelerator_from_device_hint():
    record = build_experiment_record(
        run_id="baseline-run-1",
        started_at=utc_now_iso(),
        platform=_platform(),
        model=_model(),
        command=CommandInfo(argv=("llama-cli", "-m", "qwen3.8-27b-q4.gguf")),
        repetition=_repetition(),
        stdout_text=_read_fixture("qwen3_8b_baseline_run1.log"),
    )

    assert record.platform.accelerator == "Apple M5 Pro"


def test_build_experiment_record_caller_supplied_accelerator_wins():
    platform = PlatformInfo(
        os_name="Darwin",
        os_version="24.0",
        architecture="arm64",
        accelerator="NVIDIA RTX 4090 24GB",
    )
    record = build_experiment_record(
        run_id="baseline-run-1",
        started_at=utc_now_iso(),
        platform=platform,
        model=_model(),
        command=CommandInfo(argv=("llama-cli", "-m", "qwen3.8-27b-q4.gguf")),
        repetition=_repetition(),
        stdout_text=_read_fixture("qwen3_8b_baseline_run1.log"),
    )

    # The stdout text reports "Apple M5 Pro", but the caller's explicit
    # accelerator identity must not be overwritten by the parsed hint.
    assert record.platform.accelerator == "NVIDIA RTX 4090 24GB"


def test_build_experiment_record_propagates_parse_errors():
    with pytest.raises(LlamaCppParseError):
        build_experiment_record(
            run_id="broken-run",
            started_at=utc_now_iso(),
            platform=_platform(),
            model=_model(),
            command=CommandInfo(argv=("llama-cli",)),
            repetition=_repetition(),
            stdout_text=_read_fixture("malformed_load_time.log"),
        )
