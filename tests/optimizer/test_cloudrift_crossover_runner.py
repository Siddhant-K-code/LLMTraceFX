"""Offline tests for the CloudRift crossover measured-cell runner."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from llmtracefx.optimizer.lab.qwen3_8b import cloudrift_crossover_runner as runner
from llmtracefx.optimizer.lab.qwen3_8b import cloudrift_runner as base_runner
from llmtracefx.optimizer.lab.qwen3_8b import vllm_compile as contract


class FakeNumpyModule:
    def __init__(self) -> None:
        self.seed_values: list[int] = []
        self.random = SimpleNamespace(seed=self.seed_values.append)


class FakeTorchModule:
    def __init__(self) -> None:
        self.manual_seed_values: list[int] = []
        self.cuda_seed_values: list[int] = []
        self.deterministic_algorithms_enabled = False
        self.matmul_precision: str | None = None
        self.backends = SimpleNamespace(
            cudnn=SimpleNamespace(
                deterministic=False,
                benchmark=True,
                allow_tf32=True,
            ),
            cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=True)),
        )
        self.cuda = SimpleNamespace(manual_seed_all=self.cuda_seed_values.append)

    def manual_seed(self, value: int) -> None:
        self.manual_seed_values.append(value)

    def use_deterministic_algorithms(self, enabled: bool, *, warn_only: bool) -> None:
        assert enabled is True
        assert warn_only is False
        self.deterministic_algorithms_enabled = True

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self.deterministic_algorithms_enabled

    def set_float32_matmul_precision(self, value: str) -> None:
        self.matmul_precision = value


class FakeSamplingParams:
    created: list[dict[str, Any]] = []

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        type(self).created.append(kwargs)


class FakeTokensPrompt:
    def __init__(self, *, prompt_token_ids: list[int]) -> None:
        self.prompt_token_ids = prompt_token_ids


class FakeCompilationConfig:
    def __init__(self, *, mode: Any, cudagraph_mode: Any) -> None:
        self.mode = mode
        self.cudagraph_mode = cudagraph_mode
        self.backend = "inductor"
        self.compile_sizes = [1]
        self.inductor_compile_config = {"max_autotune": False}
        self.pass_config = None
        self.splitting_ops = None


class FakeLLM:
    init_kwargs: list[dict[str, Any]] = []
    generate_calls: list[dict[str, Any]] = []
    fail_at: int | None = None
    natural_finish_reason = "stop"
    natural_tokens = [7, 8, 9]
    compilation_time: float | None = None
    encoder_compilation_time: float | None = None
    metrics_factory: Any = None

    def __init__(self, **kwargs: Any) -> None:
        type(self).init_kwargs.append(kwargs)
        compilation_config = kwargs["compilation_config"]
        self.llm_engine = SimpleNamespace(
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(enforce_eager=kwargs["enforce_eager"]),
                compilation_config=SimpleNamespace(
                    mode=SimpleNamespace(name=compilation_config.mode.name),
                    cudagraph_mode=SimpleNamespace(
                        name=compilation_config.cudagraph_mode.name
                    ),
                    backend=compilation_config.backend,
                    compile_sizes=compilation_config.compile_sizes,
                    inductor_compile_config=(
                        compilation_config.inductor_compile_config
                    ),
                    compilation_time=type(self).compilation_time,
                    pass_config=compilation_config.pass_config,
                    splitting_ops=compilation_config.splitting_ops,
                ),
                encoder_compilation_config=(
                    None
                    if type(self).encoder_compilation_time is None
                    else SimpleNamespace(
                        compilation_time=type(self).encoder_compilation_time
                    )
                ),
            )
        )

    def generate(
        self,
        prompts: list[FakeTokensPrompt],
        sampling: FakeSamplingParams,
        *,
        use_tqdm: bool,
    ) -> list[Any]:
        index = len(type(self).generate_calls) + 1
        type(self).generate_calls.append(
            {"prompts": prompts, "sampling": sampling, "use_tqdm": use_tqdm}
        )
        if type(self).fail_at == index:
            return [
                SimpleNamespace(
                    finished=False,
                    outputs=[],
                    metrics=SimpleNamespace(),
                )
            ]
        if sampling.kwargs["detokenize"]:
            output_ids = list(type(self).natural_tokens)
            finish_reason = type(self).natural_finish_reason
            decoded = f"decoded-{index}"
        else:
            output_ids = list(range(1, 97))
            finish_reason = "length"
            decoded = ""
        metrics = type(self).metrics_factory(index)
        return [
            SimpleNamespace(
                finished=True,
                outputs=[
                    SimpleNamespace(
                        token_ids=output_ids,
                        finish_reason=finish_reason,
                        text=decoded,
                    )
                ],
                metrics=metrics,
            )
        ]


class FakeMemorySampler:
    peak_value: int | None = None

    def __init__(self) -> None:
        self.peak_mib = type(self).peak_value
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def receipt(self) -> dict[str, Any]:
        samples = (
            [{"offset_ns": 1, "memory_used_mib": self.peak_mib}]
            if self.peak_mib is not None
            else None
        )
        return {
            "value": samples,
            "unit": "MiB",
            "clock_domain": "same_process_perf_counter_offset_ns",
            "provenance": "sampled_nvidia_smi",
            "observability_state": "observed" if samples else "unobservable",
            "null_reason": None if samples else "nvidia_smi_memory_series_unavailable",
            "target_interval_ms": 200,
            "sampling_error_count": 0,
            "sampling_error_types": [],
        }


class RuntimeBundle(SimpleNamespace):
    pass


@pytest.fixture(autouse=True)
def clear_fake_state() -> None:
    FakeSamplingParams.created = []
    FakeLLM.init_kwargs = []
    FakeLLM.generate_calls = []
    FakeLLM.fail_at = None
    FakeLLM.natural_finish_reason = "stop"
    FakeLLM.natural_tokens = [7, 8, 9]
    FakeLLM.compilation_time = None
    FakeLLM.encoder_compilation_time = None
    FakeLLM.metrics_factory = lambda index: SimpleNamespace()
    FakeMemorySampler.peak_value = None


def _staged_state(tmp_path: Path) -> tuple[Path, Path]:
    model_path = tmp_path / "model"
    model_path.mkdir()
    state_path = tmp_path / "state"
    state_path.mkdir()
    prompt_map = {
        f"{descriptor.context_tier}/{descriptor.workload_id}": [
            descriptor.ordinal,
            descriptor.ordinal + 1,
            descriptor.ordinal + 2,
        ]
        for descriptor in contract.workload_descriptors()
        if descriptor.repetition == 1
    }
    prompt_payload = base_runner._seal(
        {"schema_version": "1", "prompts": prompt_map},
        "prompt_ids_sha256",
    )
    base_runner._atomic_json(state_path / base_runner.PROMPT_FILE, prompt_payload)
    stage_payload = base_runner._seal(
        {
            "schema_version": "1",
            "provider": "cloudrift",
            "model_id": contract.MODEL_ID,
            "model_revision": contract.MODEL_REVISION,
            "model_file_count": contract.EXPECTED_MODEL_FILE_COUNT,
            "model_bytes": contract.EXPECTED_MODEL_BYTES,
            "inventory": [
                {
                    "path": "weights.safetensors",
                    "size_bytes": contract.EXPECTED_MODEL_BYTES,
                    "sha256": "f" * 64,
                }
            ],
            "prompts": [],
            "prompt_ids_sha256": prompt_payload["prompt_ids_sha256"],
            "runtime": dict(contract.RUNTIME_PINS),
        },
        "receipt_sha256",
    )
    base_runner._atomic_json(state_path / base_runner.STAGING_FILE, stage_payload)
    return model_path, state_path


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metrics_factory: Any | None = None,
    peak_mib: int | None = None,
) -> RuntimeBundle:
    verify_seal = base_runner._verify_seal
    numpy_module = FakeNumpyModule()
    torch_module = FakeTorchModule()
    metrics_factory = metrics_factory or (lambda index: SimpleNamespace())
    FakeLLM.metrics_factory = metrics_factory
    FakeMemorySampler.peak_value = peak_mib
    imported: list[str] = []
    runtime_import_order: list[list[str]] = []
    verification_calls = {"runtime": 0, "hardware": 0, "binding": 0, "seal": 0}

    fake_vllm = SimpleNamespace(LLM=FakeLLM, SamplingParams=FakeSamplingParams)
    fake_config = SimpleNamespace(CompilationConfig=FakeCompilationConfig)
    fake_compilation = SimpleNamespace(
        CompilationMode=SimpleNamespace(
            VLLM_COMPILE=SimpleNamespace(name="VLLM_COMPILE"),
            NONE=SimpleNamespace(name="NONE"),
        ),
        CUDAGraphMode=SimpleNamespace(
            FULL_AND_PIECEWISE=SimpleNamespace(name="FULL_AND_PIECEWISE"),
            NONE=SimpleNamespace(name="NONE"),
        ),
    )
    fake_inputs = SimpleNamespace(TokensPrompt=FakeTokensPrompt)
    modules = {
        "numpy": numpy_module,
        "torch": torch_module,
        "vllm": fake_vllm,
        "vllm.config": fake_config,
        "vllm.config.compilation": fake_compilation,
        "vllm.inputs": fake_inputs,
    }

    def fake_import_module(name: str) -> Any:
        imported.append(name)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert os.environ["PYTHONHASHSEED"] == str(contract.SAMPLING_SEED)
        assert os.environ["VLLM_DISABLE_COMPILE_CACHE"] == "1"
        assert os.environ["VLLM_BATCH_INVARIANT"] == "0"
        return modules[name]

    def fake_verify_runtime() -> dict[str, str]:
        verification_calls["runtime"] += 1
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        runtime_import_order.append(list(imported))
        return dict(contract.RUNTIME_PINS)

    def fake_hardware() -> dict[str, Any]:
        verification_calls["hardware"] += 1
        return {
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_count": 1,
            "driver_version": "580.159.03",
            "memory_total_mib": 24564,
            "memory_used_mib": 1024,
            "gpu_uuid_sha256": "sha256:" + ("a" * 64),
        }

    def fake_verify_staging_binding(
        staging: Any,
        prompts: Any,
        model_path: Path,
    ) -> None:
        verification_calls["binding"] += 1
        assert model_path.exists()
        assert staging["model_revision"] == contract.MODEL_REVISION
        assert staging["prompt_ids_sha256"] == prompts["prompt_ids_sha256"]

    def wrapped_verify_seal(value: Any, field: str) -> None:
        verification_calls["seal"] += 1
        verify_seal(value, field)

    monkeypatch.setattr(runner, "_import_module", fake_import_module)
    monkeypatch.setattr(base_runner, "_verify_runtime", fake_verify_runtime)
    monkeypatch.setattr(base_runner, "_hardware", fake_hardware)
    monkeypatch.setattr(
        base_runner,
        "_verify_staging_binding",
        fake_verify_staging_binding,
    )
    monkeypatch.setattr(base_runner, "_verify_seal", wrapped_verify_seal)
    monkeypatch.setattr(runner, "_MemorySeriesSampler", FakeMemorySampler)
    return RuntimeBundle(
        numpy=numpy_module,
        torch=torch_module,
        imported=imported,
        runtime_import_order=runtime_import_order,
        verification_calls=verification_calls,
    )


def _run(
    cell: contract.ScheduleCell,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    metrics_factory: Any | None = None,
    peak_mib: int | None = None,
    fail_at: int | None = None,
) -> tuple[dict[str, Any], RuntimeBundle, Path]:
    model_path, state_path = _staged_state(tmp_path)
    cache_root = tmp_path / "cache"
    output = tmp_path / "output.json"
    bundle = _install_fakes(
        monkeypatch,
        metrics_factory=metrics_factory,
        peak_mib=peak_mib,
    )
    FakeLLM.fail_at = fail_at
    runner.run_cell(
        cell.cell_id,
        model_path=model_path,
        state_path=state_path,
        cache_root=cache_root,
        output=output,
        experiment_nonce="public-nonce-001",
    )
    return json.loads(output.read_text(encoding="utf-8")), bundle, output


def test_module_import_is_safe_without_torch_or_vllm() -> None:
    sys.modules.pop("vllm", None)
    sys.modules.pop("torch", None)
    module = importlib.reload(runner)
    assert module.PROTOCOL_ID == contract.PROTOCOL_ID


def test_parser_only_exposes_run_cell_inputs() -> None:
    parser = runner.build_parser()
    help_text = parser.format_help()
    for forbidden in (
        "--host",
        "--port",
        "--user",
        "--token",
        "--credential",
        "tokenizer-canary",
        "stage",
    ):
        assert forbidden not in help_text
    args = parser.parse_args(
        [
            "run-cell",
            "--cell-id",
            contract.CROSSOVER_SCHEDULE[0].cell_id,
            "--model-path",
            "model",
            "--state-path",
            "state",
            "--cache-root",
            "cache",
            "--output",
            "output.json",
            "--experiment-nonce",
            "nonce",
        ]
    )
    assert args.command == "run-cell"
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run-cell",
                "--cell-id",
                contract.CROSSOVER_SCHEDULE[0].cell_id,
                "--model-path",
                "model",
                "--state-path",
                "state",
                "--cache-root",
                "cache",
                "--output",
                "output.json",
                "--experiment-nonce",
                "nonce",
                "--host",
                "127.0.0.1",
            ]
        )


def test_memory_series_sampler_records_samples_and_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampler = runner._MemorySeriesSampler()
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="2048\n"),
    )
    sampler._observe()
    observed = sampler.receipt()
    assert observed["observability_state"] == "observed"
    assert observed["value"][0]["memory_used_mib"] == 2048
    assert observed["sampling_error_count"] == 0

    def fail(*args: Any, **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired("nvidia-smi", 5)

    monkeypatch.setattr(runner.subprocess, "run", fail)
    sampler._observe_or_record_error()
    with_gap = sampler.receipt()
    assert with_gap["sampling_error_count"] == 1
    assert with_gap["sampling_error_types"] == ["TimeoutExpired"]


def test_runner_refuses_a_nonempty_cache_root(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    (cache_root / "stale").write_text("stale", encoding="utf-8")
    with pytest.raises(contract.VLLMCompileContractError, match="must be empty"):
        runner.prepare_deterministic_environment(
            contract.CROSSOVER_SCHEDULE[0],
            cache_root,
        )


def test_controlled_cell_uses_exact_sampling_and_records_each_generate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = next(
        item
        for item in contract.CROSSOVER_SCHEDULE
        if item.lane == "controlled" and item.mode == "compiled"
    )

    payload, bundle, _ = _run(
        cell,
        tmp_path,
        monkeypatch,
        metrics_factory=lambda index: SimpleNamespace(
            first_token_latency=0.25,
            time_in_queue=0.05,
            prefill_time=0.15,
            inference_time=0.4,
            decode_time=0.1,
            mean_time_per_output_token=0.01,
            e2e_request_latency=0.5,
        ),
        peak_mib=2048,
    )

    assert payload["terminal"] is True
    assert payload["cell_sha256"].startswith("sha256:")
    assert payload["process_tree"]["clock_domain"] == "runner_process_snapshot"
    assert '"pid"' not in json.dumps(payload["process_tree"])
    assert payload["request_count_observed"] == 144
    assert len(payload["requests"]) == 144
    assert len(FakeLLM.generate_calls) == 144
    assert len(FakeSamplingParams.created) == 1
    assert FakeSamplingParams.created[0] == contract.CONTROLLED_SAMPLING.to_dict()
    assert FakeLLM.init_kwargs[0]["tensor_parallel_size"] == 1
    assert FakeLLM.init_kwargs[0]["max_num_seqs"] == 1
    assert FakeLLM.init_kwargs[0]["enable_prefix_caching"] is False
    assert FakeLLM.init_kwargs[0]["disable_custom_all_reduce"] is True
    assert FakeLLM.init_kwargs[0]["speculative_config"] is None
    assert payload["runtime"]["resolved_execution_config"] == {
        "enforce_eager": False,
        "compilation_mode": "VLLM_COMPILE",
        "cuda_graph_mode": "FULL_AND_PIECEWISE",
    }
    assert (
        payload["hardware_commitment"]["public_experiment_nonce"] == "public-nonce-001"
    )
    assert "gpu_uuid_sha256" not in json.dumps(payload, sort_keys=True)
    assert bundle.imported == [
        "numpy",
        "torch",
        "vllm",
        "vllm.config",
        "vllm.config.compilation",
        "vllm.inputs",
    ]
    assert bundle.verification_calls == {
        "runtime": 1,
        "hardware": 1,
        "binding": 1,
        "seal": 2,
    }
    assert bundle.runtime_import_order == [[]]
    assert bundle.numpy.seed_values == [contract.SAMPLING_SEED]
    assert bundle.torch.manual_seed_values == [contract.SAMPLING_SEED]
    assert bundle.torch.cuda_seed_values == [contract.SAMPLING_SEED]
    assert bundle.torch.backends.cudnn.deterministic is True
    assert bundle.torch.backends.cudnn.benchmark is False
    assert bundle.torch.backends.cudnn.allow_tf32 is False
    assert bundle.torch.backends.cuda.matmul.allow_tf32 is False
    assert bundle.torch.matmul_precision == "highest"

    first = payload["requests"][0]
    last = payload["requests"][-1]
    assert first["cycle_index"] == 1
    assert first["base_ordinal"] == 1
    assert first["request_sequence_index"] == 1
    assert last["cycle_index"] == 12
    assert last["base_ordinal"] == 12
    assert last["request_sequence_index"] == 144
    assert first["finish_reason"] == "length"
    assert first["output_token_count"] == 96
    assert len(first["output_token_ids"]) == 96
    assert "decoded_output" not in first
    assert first["metrics"]["ttft_seconds"] == {
        "value": 0.25,
        "unit": "seconds",
        "clock_domain": "request_output_metrics",
        "provenance": "version_pinned_vllm_0_28_request_state_stats",
        "observability_state": "observed",
        "null_reason": None,
    }
    for key in (
        "queue_seconds",
        "prefill_seconds",
        "inference_seconds",
        "decode_seconds",
        "mean_time_per_output_token_seconds",
        "e2e_seconds",
    ):
        assert first["metrics"][key]["value"] is None
        assert (
            first["metrics"][key]["provenance"]
            == "version_pinned_vllm_0_28_request_state_stats"
        )
    assert first["timing"]["output_token_rate_tokens_per_second"]["value"] is not None
    assert (
        first["timing"]["output_token_rate_tokens_per_second"]["unit"]
        == "tokens_per_second"
    )
    assert payload["measurements"]["peak_gpu_memory_mib"] == {
        "value": 2048.0,
        "unit": "MiB",
        "clock_domain": "sampled_nvidia_smi",
        "provenance": "sampled_nvidia_smi",
        "observability_state": "observed",
        "null_reason": None,
    }
    assert payload["measurements"]["gpu_memory_series"] == {
        "value": [{"offset_ns": 1, "memory_used_mib": 2048}],
        "unit": "MiB",
        "clock_domain": "same_process_perf_counter_offset_ns",
        "provenance": "sampled_nvidia_smi",
        "observability_state": "observed",
        "null_reason": None,
        "target_interval_ms": 200,
        "sampling_error_count": 0,
        "sampling_error_types": [],
    }
    env = payload["deterministic_environment"]
    assert env["variables"] == {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "HF_HUB_OFFLINE": "1",
        "PYTHONHASHSEED": str(contract.SAMPLING_SEED),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        "VLLM_BATCH_INVARIANT": "0",
        "VLLM_NO_USAGE_STATS": "1",
    }
    assert env["cache_root_role"]["relative_identity"] == cell.cell_id
    assert env["cache_roles"]["vllm"]["relative_path"] == "vllm"
    output_text = json.dumps(payload, sort_keys=True)
    assert str(tmp_path) not in output_text


def test_compilation_time_fields_are_version_pinned_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = next(
        item
        for item in contract.CROSSOVER_SCHEDULE
        if item.lane == "controlled" and item.mode == "compiled"
    )
    FakeLLM.compilation_time = 1.25
    FakeLLM.encoder_compilation_time = 0.5

    payload, _, _ = _run(
        cell,
        tmp_path,
        monkeypatch,
        metrics_factory=lambda index: SimpleNamespace(
            first_token_latency=0.25,
            time_in_queue=9.0,
            prefill_time=8.0,
            inference_time=7.0,
            decode_time=6.0,
            mean_time_per_output_token=5.0,
            e2e_request_latency=4.0,
        ),
    )

    optional = payload["runtime"]["optional_version_pinned_fields"]
    assert optional["compilation_time_seconds"] == {
        "value": 1.25,
        "unit": "seconds",
        "clock_domain": "vllm_internal_runtime",
        "provenance": "version_pinned_vllm_0_28_internal",
        "observability_state": "observed",
        "null_reason": None,
    }
    assert optional["encoder_compilation_time_seconds"] == {
        "value": 0.5,
        "unit": "seconds",
        "clock_domain": "vllm_internal_runtime",
        "provenance": "version_pinned_vllm_0_28_internal",
        "observability_state": "observed",
        "null_reason": None,
    }
    assert optional["cuda_graph_capture_duration_seconds"]["value"] is None
    assert (
        optional["cuda_graph_capture_duration_seconds"]["provenance"]
        == "version_pinned_vllm_0_28_internal"
    )


def test_natural_cell_keeps_decoded_text_and_null_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = next(
        item
        for item in contract.CROSSOVER_SCHEDULE
        if item.lane == "natural" and item.mode == "eager"
    )

    payload, _, _ = _run(
        cell,
        tmp_path,
        monkeypatch,
        metrics_factory=lambda index: SimpleNamespace(
            first_token_latency=0.0,
            time_in_queue=None,
            prefill_time=None,
            inference_time=None,
            decode_time=None,
            mean_time_per_output_token=None,
            e2e_request_latency=None,
        ),
        peak_mib=None,
    )

    assert payload["request_count_observed"] == 12
    assert len(FakeLLM.generate_calls) == 12
    assert FakeSamplingParams.created[0] == contract.NATURAL_SAMPLING.to_dict()
    assert payload["runtime"]["resolved_execution_config"] == {
        "enforce_eager": True,
        "compilation_mode": "NONE",
        "cuda_graph_mode": "NONE",
    }
    first = payload["requests"][0]
    assert first["decoded_output"] == "decoded-1"
    assert first["finish_reason"] == "stop"
    for value in first["metrics"].values():
        if value["provenance"] == "version_pinned_vllm_0_28_request_state_stats":
            assert value["value"] is None or value["value"] > 0
    assert first["metrics"]["ttft_seconds"]["value"] is None
    assert (
        first["metrics"]["ttft_seconds"]["null_reason"]
        == "request_state_stats_first_token_latency_unavailable"
    )
    for key in (
        "queue_seconds",
        "prefill_seconds",
        "inference_seconds",
        "decode_seconds",
        "mean_time_per_output_token_seconds",
        "e2e_seconds",
    ):
        assert first["metrics"][key]["value"] is None
        assert first["metrics"][key]["observability_state"] == "unobservable"
        assert first["metrics"][key]["null_reason"] is not None
    assert payload["measurements"]["peak_gpu_memory_mib"]["value"] is None
    assert (
        payload["measurements"]["peak_gpu_memory_mib"]["null_reason"]
        == "nvidia_smi_peak_memory_unavailable"
    )
    assert payload["measurements"]["gpu_memory_series"]["value"] is None
    assert (
        payload["measurements"]["gpu_memory_series"]["null_reason"]
        == "nvidia_smi_memory_series_unavailable"
    )
    optional = payload["runtime"]["optional_version_pinned_fields"]
    assert (
        optional["compilation_time_seconds"]["observability_state"] == "not_applicable"
    )
    assert (
        optional["compilation_time_seconds"]["null_reason"]
        == "not_applicable_eager_mode"
    )
    assert (
        optional["encoder_compilation_time_seconds"]["observability_state"]
        == "not_applicable"
    )
    assert (
        optional["cuda_graph_capture_duration_seconds"]["observability_state"]
        == "not_applicable"
    )


def test_runner_writes_only_progress_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = next(
        item
        for item in contract.CROSSOVER_SCHEDULE
        if item.lane == "natural" and item.mode == "compiled"
    )
    model_path, state_path = _staged_state(tmp_path)
    cache_root = tmp_path / "cache"
    output = tmp_path / "output.json"
    _install_fakes(monkeypatch)
    FakeLLM.fail_at = 3

    with pytest.raises(contract.VLLMCompileContractError, match="did not complete"):
        runner.run_cell(
            cell.cell_id,
            model_path=model_path,
            state_path=state_path,
            cache_root=cache_root,
            output=output,
            experiment_nonce="public-nonce-001",
        )

    assert not output.exists()
    progress = json.loads(
        (tmp_path / ".output-progress.json").read_text(encoding="utf-8")
    )
    assert progress["progress_sha256"].startswith("sha256:")
    assert progress["terminal"] is False
    assert progress["request_count_completed"] == 2
    assert len(progress["requests"]) == 2
    assert str(tmp_path) not in json.dumps(progress, sort_keys=True)
