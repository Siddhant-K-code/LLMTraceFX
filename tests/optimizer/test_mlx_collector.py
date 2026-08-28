"""Tests for normalized MLX-LM experiment collection."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from llmtracefx.optimizer.collectors.mlx import (
    MLXCollectionConfig,
    MLXCollectorError,
    MLXLMRuntime,
    MLXMemorySnapshot,
    collect_mlx,
)
from llmtracefx.optimizer.schema import (
    ExperimentRecord,
    MetricProvenance,
)


class StepClock:
    def __init__(self, step: float = 0.01):
        self.value = 0.0
        self.step = step

    def __call__(self):
        current = self.value
        self.value += self.step
        return current


@dataclass
class FakeResponse:
    text: str
    from_draft: bool
    prompt_tokens: int
    generation_tokens: int
    finish_reason: str | None = None


class FakeRuntime:
    mlx_version = "0.32.0"
    mlx_lm_version = "0.31.3"

    def __init__(self):
        self.load_calls = []
        self.synchronize_calls = 0
        self.reset_peak_calls = 0
        self.seed_calls = []
        self.generate_calls = []
        self.responses = [
            FakeResponse("hello", False, 3, 1),
            FakeResponse(" world", True, 3, 2),
        ]
        self.snapshot = MLXMemorySnapshot(
            active_bytes=1024,
            cache_bytes=256,
            peak_bytes=2048,
        )

    def load_model(self, path):
        self.load_calls.append(path)
        return object(), FakeTokenizer()

    def encode(self, tokenizer, prompt):
        assert isinstance(tokenizer, FakeTokenizer)
        assert prompt == "test prompt"
        return [1, 2, 3]

    def seed(self, seed):
        self.seed_calls.append(seed)

    def synchronize(self):
        self.synchronize_calls += 1

    def reset_peak_memory(self):
        self.reset_peak_calls += 1

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
        self.generate_calls.append(
            {
                "prompt_tokens": prompt_tokens,
                "max_tokens": max_tokens,
                "draft_model": draft_model,
                "num_draft_tokens": num_draft_tokens,
            }
        )
        yield from self.responses


class FakeTokenizer:
    bos_token = None


class FailingRuntime(FakeRuntime):
    def load_model(self, path):
        raise RuntimeError("model load failed")


class MalformedModelRuntime(FakeRuntime):
    def load_model(self, path):
        raise KeyError("model_type")


def make_config(tmp_path, **overrides):
    model_path = tmp_path / "model"
    model_path.mkdir(exist_ok=True)
    values = {
        "run_id": "mlx-run-1",
        "model_path": model_path,
        "model_id": "local/test-model",
        "prompt": "test prompt",
        "output_dir": tmp_path / "artifacts",
        "command_argv": (
            "llmtracefx-optimizer",
            "collect-mlx",
            "--model-path",
            str(model_path),
        ),
        "max_tokens": 16,
        "seed": 7,
        "quantization": "4bit",
    }
    values.update(overrides)
    return MLXCollectionConfig(**values)


def test_collect_mlx_writes_normalized_record_and_response(tmp_path):
    runtime = FakeRuntime()
    result = collect_mlx(
        make_config(tmp_path),
        runtime=runtime,
        clock=StepClock(),
    )

    record = result.record
    assert record.outcome.success is True
    assert record.platform.accelerator == "Apple M5 Pro"
    assert record.runtime.name == "mlx-lm"
    assert record.runtime.version == "0.31.3"
    assert record.runtime.backend == "Metal"
    assert record.tokens.input_tokens == 3
    assert record.tokens.context_tokens == 3
    assert record.tokens.generated_tokens == 2
    assert record.command.workload_hash.startswith("sha256:")
    assert record.command.config_hash.startswith("sha256:")
    assert record.timing.prefill.provenance == MetricProvenance.MEASURED_WALL_CLOCK
    assert record.timing.decode.provenance == MetricProvenance.MEASURED_WALL_CLOCK
    assert record.memory.active.value == 1024
    assert record.memory.active.provenance == MetricProvenance.MEASURED_NATIVE
    assert record.memory.cache.value == 256
    assert record.memory.peak.value == 2048
    assert record.memory.wired is None
    assert record.speculative.enabled is False
    assert result.response_text == "hello world"

    persisted = ExperimentRecord.read_json(tmp_path / "artifacts" / "record.json")
    assert persisted == record
    assert (tmp_path / "artifacts" / "response.txt").read_text() == "hello world"
    assert (tmp_path / "artifacts" / "environment.json").exists()
    assert runtime.seed_calls == [7]
    assert runtime.reset_peak_calls == 1


def test_normal_collection_synchronizes_only_at_phase_boundaries(tmp_path):
    runtime = FakeRuntime()
    runtime.responses = [
        FakeResponse(str(index), False, 3, index + 1) for index in range(20)
    ]

    collect_mlx(make_config(tmp_path), runtime=runtime, clock=StepClock())

    assert runtime.synchronize_calls == 3


def test_collect_mlx_records_generic_draft_acceptance_without_inventing_proposals(
    tmp_path,
):
    draft_path = tmp_path / "draft"
    draft_path.mkdir()
    runtime = FakeRuntime()

    record = collect_mlx(
        make_config(
            tmp_path,
            draft_model_path=draft_path,
            num_draft_tokens=4,
        ),
        runtime=runtime,
        clock=StepClock(),
    ).record

    assert runtime.load_calls == [tmp_path / "model", draft_path]
    assert record.speculative.enabled is True
    assert record.speculative.method == "draft-model"
    assert record.speculative.configured_depth == 4
    assert record.speculative.accepted_tokens == 1
    assert record.speculative.proposed_tokens is None
    assert record.speculative.verification_time is None


def test_eos_summary_does_not_inflate_generated_or_accepted_tokens(tmp_path):
    draft_path = tmp_path / "draft"
    draft_path.mkdir()
    runtime = FakeRuntime()
    runtime.responses = [
        FakeResponse("hello", False, 3, 1),
        FakeResponse(" world", True, 3, 2),
        FakeResponse("", True, 3, 3, finish_reason="stop"),
    ]

    result = collect_mlx(
        make_config(tmp_path, draft_model_path=draft_path),
        runtime=runtime,
        clock=StepClock(),
    )

    assert result.response_text == "hello world"
    assert result.record.tokens.generated_tokens == 2
    assert result.record.speculative.accepted_tokens == 1


def test_length_summary_counts_the_final_generated_token(tmp_path):
    runtime = FakeRuntime()
    runtime.responses = [
        FakeResponse("hello", False, 3, 1),
        FakeResponse(" world", False, 3, 2, finish_reason="length"),
    ]

    result = collect_mlx(make_config(tmp_path), runtime=runtime, clock=StepClock())

    assert result.response_text == "hello world"
    assert result.record.tokens.generated_tokens == 2


def test_collect_mlx_leaves_unavailable_allocator_metrics_absent(tmp_path):
    runtime = FakeRuntime()
    runtime.snapshot = MLXMemorySnapshot()

    record = collect_mlx(
        make_config(tmp_path), runtime=runtime, clock=StepClock()
    ).record

    assert record.memory.active is None
    assert record.memory.cache is None
    assert record.memory.peak is None


def test_explicit_accelerator_overrides_runtime_detection(tmp_path):
    record = collect_mlx(
        make_config(tmp_path, accelerator="Test accelerator"),
        runtime=FakeRuntime(),
        clock=StepClock(),
    ).record

    assert record.platform.accelerator == "Test accelerator"


def test_runtime_failure_is_persisted_without_success_fallback(tmp_path):
    result = collect_mlx(
        make_config(tmp_path),
        runtime=FailingRuntime(),
        clock=StepClock(),
    )

    assert result.record.outcome.success is False
    assert result.record.error.category == "RuntimeError"
    assert result.record.error.message == "model load failed"
    assert result.record.timing.total is not None
    assert result.record.timing.model_load is None
    persisted = ExperimentRecord.read_json(tmp_path / "artifacts" / "record.json")
    assert persisted.outcome.success is False


def test_malformed_local_model_is_persisted_without_traceback(tmp_path):
    result = collect_mlx(
        make_config(tmp_path),
        runtime=MalformedModelRuntime(),
        clock=StepClock(),
    )

    assert result.record.outcome.success is False
    assert result.record.error.category == "KeyError"
    assert "model_type" in result.record.error.message
    persisted = ExperimentRecord.read_json(tmp_path / "artifacts" / "record.json")
    assert persisted.error.category == "KeyError"


def test_missing_model_path_is_rejected_without_downloading(tmp_path):
    with pytest.raises(MLXCollectorError, match="Download or convert"):
        make_config(tmp_path, model_path=tmp_path / "missing")


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_tokens": True},
        {"max_tokens": 1.5},
        {"max_tokens": 0},
        {"seed": False},
        {"seed": "0"},
        {"num_draft_tokens": True},
        {"num_draft_tokens": 0},
    ],
)
def test_collection_config_rejects_malformed_numeric_values(tmp_path, overrides):
    with pytest.raises(MLXCollectorError):
        make_config(tmp_path, **overrides)


def test_runtime_rejects_non_apple_platform_before_import(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")

    with pytest.raises(MLXCollectorError, match="Apple Silicon"):
        MLXLMRuntime()
