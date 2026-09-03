"""Offline tests for the internal Qwen3 Modal harness."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from _fakes import build_fake_modal

from llmtracefx.optimizer.lab.qwen3_8b.vllm_compile import CURRENT_RATES, build_plan

MODULE = "llmtracefx.deploy.modal_qwen3_compile_app"
DIGEST = "sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635"


@pytest.fixture
def artifact_dir(request: pytest.FixtureRequest) -> Iterator[Path]:
    path = Path(".cache/llmtracefx-tests/modal-qwen3") / request.node.name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _plan_json() -> str:
    return build_plan(
        prices={key: str(value) for key, value in CURRENT_RATES.items()},
        effective_date="2026-09-01",
        price_source="https://modal.com/pricing/2026-09-01",
        price_source_sha256="sha256:" + "9" * 64,
        image_digest=DIGEST,
        runtime_pins={
            "python_version": "3.12",
            "vllm_version": "0.28.0",
            "torch_version": "2.13.0+cu130",
            "cuda_version": "13.0",
            "typing_extensions_version": "4.15.0",
        },
        as_of_date="2026-09-03",
    ).to_json()


def _hashes() -> tuple[str, str]:
    source = Path("llmtracefx/deploy/modal_qwen3_compile_app.py").read_text()
    # Import once with deliberately wrong hashes; the module has computed its
    # constants before refusing registration, without touching Modal.
    assert "WORKLOAD_CONTRACT_SHA256" in source
    import hashlib

    from llmtracefx.optimizer.lab.qwen3_8b.vllm_compile import (
        canonical_json,
        workload_descriptors,
    )

    sampling = {
        "max_tokens": 96,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 20260831,
    }
    tokenizer = {
        "tokenize": True,
        "add_generation_prompt": True,
        "enable_thinking": False,
        "messages": "single_user_message",
    }
    domains = sorted(
        (
            "client_observed",
            "vllm",
            "cuda",
            "modal_provider",
            "model_reported",
            "derived",
        )
    )

    def digest(value: Any) -> str:
        return "sha256:" + hashlib.sha256(canonical_json(value).encode()).hexdigest()

    return (
        digest(
            {
                "schema_version": "1",
                "descriptors": [item.to_dict() for item in workload_descriptors()],
                "sampling": sampling,
                "tokenizer": tokenizer,
            }
        ),
        digest(
            {
                "schema_version": "1",
                "request_terminal_required": True,
                "finish_reason_required": True,
                "input_count_source": "persisted_prompt_token_ids",
                "output_count_source": "request_output_token_ids",
                "decoded_output_max_utf8_bytes": 65536,
                "remote_correctness_evaluation": False,
                "resolved_execution_config_required": True,
                "missing_timing_reason_required": True,
                "provenance_domains": domains,
            }
        ),
    )


def _environment() -> dict[str, str]:
    workload, output = _hashes()
    return {
        "LLMTRACEFX_QWEN3_COMPILE_PLAN_JSON": _plan_json(),
        "LLMTRACEFX_QWEN3_COMPILE_APP_NAME": "qwen3-compile",
        "LLMTRACEFX_QWEN3_COMPILE_VOLUME_NAME": "qwen3-compile-volume-a1",
        "LLMTRACEFX_QWEN3_COMPILE_EXPERIMENT_TAG": "Approved Run 01",
        "LLMTRACEFX_QWEN3_COMPILE_WORKLOAD_SHA256": workload,
        "LLMTRACEFX_QWEN3_COMPILE_OUTPUT_SHA256": output,
    }


@contextmanager
def _imported(environ: dict[str, str]) -> Iterator[tuple[Any, ModuleType]]:
    fake = build_fake_modal()
    saved_modal = sys.modules.get("modal")
    saved_app = sys.modules.pop(MODULE, None)
    saved_environ = dict(os.environ)
    sys.modules["modal"] = fake
    os.environ.clear()
    os.environ.update(environ)
    try:
        yield importlib.import_module(MODULE), fake
    finally:
        sys.modules.pop(MODULE, None)
        if saved_app is not None:
            sys.modules[MODULE] = saved_app
        if saved_modal is not None:
            sys.modules["modal"] = saved_modal
        else:
            sys.modules.pop("modal", None)
        os.environ.clear()
        os.environ.update(saved_environ)


def test_missing_pins_fail_before_modal_registration() -> None:
    fake = build_fake_modal()
    saved = sys.modules.get("modal")
    sys.modules["modal"] = fake
    sys.modules.pop(MODULE, None)
    with pytest.raises(ValueError, match="supply exactly one"):
        importlib.import_module(MODULE)
    assert fake._fake_apps == []
    if saved is not None:
        sys.modules["modal"] = saved
    else:
        sys.modules.pop("modal", None)


def test_registration_is_exact_and_secretless() -> None:
    with _imported(_environment()) as (module, fake):
        app = fake._fake_apps[0]
        assert app.name == "qwen3-compile-approved-run-01"
        assert app.kwargs["tags"] == {
            "experiment": "approved-run-01",
            "project": "llmtracefx-vllm-compile",
        }
        assert fake._fake_volumes[0].kwargs == {"create_if_missing": False}
        assert fake._fake_secrets == []
        assert fake._fake_images[0].kwargs["tag"] == module.IMAGE_REFERENCE
        assert fake._fake_images[0].kwargs["setup_dockerfile_commands"] == [
            "RUN ln -sf /usr/bin/python3 /usr/local/bin/python"
        ]
        assert fake._fake_images[0].entrypoint_commands == []
        assert fake._fake_images[0].pip_packages == ["typing_extensions==4.15.0"]
        assert fake._fake_images[0].local_dir_options == [{"copy": True}]
        assert list(app.registrations) == [
            "stage_qwen3",
            "l40s_eager",
            "l40s_compiled",
            "h100_eager",
            "h100_compiled",
        ]
        stage = app.registrations["stage_qwen3"]
        assert "gpu" not in stage.function_kwargs
        assert "block_network" not in stage.function_kwargs
        assert stage.concurrent_kwargs == {"max_inputs": 1}
        assert (
            stage.function_kwargs["cpu"],
            stage.function_kwargs["memory"],
            stage.function_kwargs["timeout"],
            stage.function_kwargs["retries"],
            stage.function_kwargs["max_containers"],
            stage.function_kwargs["min_containers"],
        ) == (4, 32768, 2700, 0, 1, 0)
        assert stage.function_kwargs["single_use_containers"] is True
        assert "restrict_modal_access" not in stage.function_kwargs
        for name, gpu in (
            ("l40s_eager", "L40S"),
            ("l40s_compiled", "L40S"),
            ("h100_eager", "H100!"),
            ("h100_compiled", "H100!"),
        ):
            registration = app.registrations[name]
            assert registration.function_kwargs["gpu"] == gpu
            assert registration.function_kwargs["block_network"] is True
            assert registration.function_kwargs["enable_memory_snapshot"] is False
            assert registration.function_kwargs["single_use_containers"] is True
            assert registration.function_kwargs["restrict_modal_access"] is True
            assert registration.concurrent_kwargs == {"max_inputs": 1}
            assert (
                registration.function_kwargs["cpu"],
                registration.function_kwargs["memory"],
                registration.function_kwargs["timeout"],
                registration.function_kwargs["max_containers"],
                registration.function_kwargs["min_containers"],
            ) == (4, 32768, 2700, 1, 0)
            assert "retries" not in registration.function_kwargs


def test_real_modal_sdk_accepts_registration_without_provider_access() -> None:
    pytest.importorskip("modal")
    environment = {
        key: value
        for key, value in {**os.environ, **_environment()}.items()
        if not key.startswith("MODAL_TOKEN")
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import llmtracefx.deploy.modal_qwen3_compile_app",
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_hardware_substitution_refuses_before_model_construction() -> None:
    with _imported(_environment()) as (module, _):
        constructed = False

        def factory(*_: Any) -> Any:
            nonlocal constructed
            constructed = True
            raise AssertionError

        events = module._run_cell(
            module.CELLS[2],
            command_runner=lambda _: "NVIDIA H200, 570.1, 141000, 0\n",
            llm_factory=factory,
        )
        assert next(events)["event"] == "container_started"
        with pytest.raises(ValueError, match="NVIDIA H100"):
            next(events)
        assert constructed is False


def test_incomplete_hardware_identity_refuses_before_model_construction() -> None:
    with _imported(_environment()) as (module, _):
        events = module._run_cell(
            module.CELLS[0],
            command_runner=lambda _: "NVIDIA L40S, [N/A], 46068, 0\n",
            llm_factory=lambda *_: (_ for _ in ()).throw(AssertionError()),
        )
        assert next(events)["event"] == "container_started"
        with pytest.raises(module.VLLMCompileContractError, match="incomplete"):
            next(events)


def test_memory_sampler_requires_an_initial_observation() -> None:
    with _imported(_environment()) as (module, _):
        sampler = module._MemorySampler(lambda _: "[N/A]\n")

        with pytest.raises(module.VLLMCompileContractError, match="memory sample"):
            sampler.start()


def test_terminal_request_uses_token_ids_and_requires_finish_reason() -> None:
    with _imported(_environment()) as (module, _):
        descriptor = module.workload_descriptors()[0]
        response = SimpleNamespace(
            finished=True,
            outputs=[
                SimpleNamespace(
                    finish_reason="stop",
                    token_ids=[7, 8, 9],
                    text="answer",
                )
            ],
            metrics=SimpleNamespace(),
        )
        record = module._request_record(
            descriptor=descriptor,
            ids=[1, 2],
            response=response,
            started_at="2026-09-03T00:00:00+00:00",
            ended_at="2026-09-03T00:00:01+00:00",
            elapsed=1.0,
        )
        assert record["input_token_count"] == 2
        assert record["output_token_count"] == 3
        assert record["correctness"] is None
        assert record["ttft_seconds"] is None
        response.outputs[0].finish_reason = None
        with pytest.raises(ValueError, match="finish reason"):
            module._request_record(
                descriptor=descriptor,
                ids=[1],
                response=response,
                started_at="a",
                ended_at="b",
                elapsed=1.0,
            )


def test_staging_uses_exact_public_revision_and_persists_receipt(
    artifact_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with _imported(_environment()) as (module, fake):
        calls: list[dict[str, Any]] = []
        model_path = artifact_dir / module.MODEL_DIRECTORY

        def download(**kwargs: Any) -> str:
            calls.append(kwargs)
            model_path.mkdir()
            return str(model_path)

        inventory = [
            {
                "path": f"file-{index}",
                "size_bytes": module.EXPECTED_MODEL_BYTES if index == 0 else 0,
                "sha256": "0" * 64,
            }
            for index in range(15)
        ]
        prompt_ids = {
            f"{tier}/{workload}": [index, index + 1]
            for index, (tier, workload) in enumerate(
                (tier, workload)
                for tier in ("2k", "8k", "16k")
                for workload in (
                    "structured-json-profile-extraction",
                    "prose-reasoning-two-train-gap",
                )
            )
        }
        monkeypatch.setattr(module, "_verify_model_inventory", lambda _: inventory)
        monkeypatch.setattr(
            module,
            "_materialize_token_ids",
            lambda *_: ([], prompt_ids),
        )

        receipt = module._stage_impl(
            snapshot_download=download,
            tokenizer_factory=lambda *_args, **_kwargs: None,
            mount_path=artifact_dir,
        )

        assert calls == [
            {
                "repo_id": "Qwen/Qwen3-8B",
                "revision": "b968826d9c46dd6066d109eabc6255188de91218",
                "local_dir": str(model_path),
                "token": False,
            }
        ]
        assert receipt["model_file_count"] == 15
        module._verify_seal(receipt, "receipt_sha256")
        assert (
            artifact_dir / module.STAGING_RECEIPT
        ).read_text() == module.canonical_json(receipt)
        assert fake._fake_volumes[0].commits == 1


def test_exact_eager_and_compiled_constructor_configuration() -> None:
    with _imported(_environment()) as (module, _):
        llm_calls: list[dict[str, Any]] = []
        fake_vllm = SimpleNamespace(LLM=lambda **kwargs: llm_calls.append(kwargs))

        config_module = ModuleType("vllm.config")
        compilation_module = ModuleType("vllm.config.compilation")

        @dataclass
        class CompilationConfig:
            mode: object
            cudagraph_mode: object

        config_module.CompilationConfig = CompilationConfig  # type: ignore[attr-defined]
        compilation_module.CompilationMode = SimpleNamespace(  # type: ignore[attr-defined]
            NONE="NONE", VLLM_COMPILE="VLLM_COMPILE"
        )
        compilation_module.CUDAGraphMode = SimpleNamespace(  # type: ignore[attr-defined]
            NONE="NONE", FULL_AND_PIECEWISE="FULL_AND_PIECEWISE"
        )
        saved_config = sys.modules.get("vllm.config")
        saved_compilation = sys.modules.get("vllm.config.compilation")
        sys.modules["vllm.config"] = config_module
        sys.modules["vllm.config.compilation"] = compilation_module
        try:
            module._construct_llm(fake_vllm, module.CELLS[0], Path("/model"))
            module._construct_llm(fake_vllm, module.CELLS[1], Path("/model"))
        finally:
            if saved_config is None:
                sys.modules.pop("vllm.config", None)
            else:
                sys.modules["vllm.config"] = saved_config
            if saved_compilation is None:
                sys.modules.pop("vllm.config.compilation", None)
            else:
                sys.modules["vllm.config.compilation"] = saved_compilation
        eager, compiled = llm_calls
        assert eager["enforce_eager"] is True
        assert eager["compilation_config"] == CompilationConfig(
            mode="NONE", cudagraph_mode="NONE"
        )
        assert compiled["enforce_eager"] is False
        assert compiled["compilation_config"] == CompilationConfig(
            mode="VLLM_COMPILE", cudagraph_mode="FULL_AND_PIECEWISE"
        )


def test_cell_uses_exact_order_and_persists_only_complete_terminal(
    artifact_dir: Path,
) -> None:
    with _imported(_environment()) as (module, fake):
        prompt_ids = {
            f"{tier}/{workload}": [index + 1, index + 2]
            for index, (tier, workload) in enumerate(
                (tier, workload)
                for tier in ("2k", "8k", "16k")
                for workload in (
                    "structured-json-profile-extraction",
                    "prose-reasoning-two-train-gap",
                )
            )
        }
        prompt_payload = module._seal(
            {
                "schema_version": "1",
                "workload_sha256": module.WORKLOAD_CONTRACT_SHA256,
                "prompts": prompt_ids,
            },
            "prompt_ids_sha256",
        )
        receipt = module._seal(
            {
                "schema_version": "1",
                "plan_sha256": module.PLAN.content_sha256,
                "workload_sha256": module.WORKLOAD_CONTRACT_SHA256,
                "output_contract_sha256": module.OUTPUT_CONTRACT_SHA256,
                "runtime_sha256": module._sha256_json(
                    module.PLAN.runtime_pins.to_dict()
                ),
                "image_sha256": module.IMAGE_CONTRACT_SHA256,
                "image_digest": module.IMAGE_DIGEST,
                "model_id": module.MODEL_ID,
                "model_revision": module.MODEL_REVISION,
                "model_file_count": 15,
                "model_bytes": module.EXPECTED_MODEL_BYTES,
                "prompt_ids_sha256": prompt_payload["prompt_ids_sha256"],
            },
            "receipt_sha256",
        )
        module._atomic_json(artifact_dir / module.PROMPT_IDS_FILE, prompt_payload)
        module._atomic_json(artifact_dir / module.STAGING_RECEIPT, receipt)

        generated_prompts: list[list[int]] = []

        class FakeLLM:
            llm_engine = SimpleNamespace(
                vllm_config=SimpleNamespace(
                    model_config=SimpleNamespace(enforce_eager=True),
                    compilation_config=SimpleNamespace(
                        compilation_time=0.0,
                        mode="NONE",
                        cudagraph_mode="NONE",
                    ),
                )
            )

            def generate(
                self, prompts: list[Any], _sampling: Any, *, use_tqdm: bool
            ) -> list[Any]:
                assert use_tqdm is False
                generated_prompts.append(prompts[0]["prompt_token_ids"])
                return [
                    SimpleNamespace(
                        finished=True,
                        metrics=SimpleNamespace(),
                        outputs=[
                            SimpleNamespace(
                                finish_reason="length",
                                token_ids=[41, 42],
                                text="local evaluator input",
                            )
                        ],
                    )
                ]

        events = list(
            module._run_cell(
                module.CELLS[0],
                mount_path=artifact_dir,
                command_runner=lambda argv: (
                    "NVIDIA L40S, 570.1, 46068, 0\n"
                    if "name,driver_version" in argv[1]
                    else "1\n"
                ),
                runtime_observer=lambda: module.PLAN.runtime_pins.to_dict(),
                vllm_module=SimpleNamespace(),
                llm_factory=lambda *_: FakeLLM(),
                sampling_factory=lambda **kwargs: kwargs,
                tokens_prompt_factory=lambda **kwargs: kwargs,
            )
        )
        descriptors = module.workload_descriptors()
        assert generated_prompts == [
            prompt_ids[f"{item.context_tier}/{item.workload_id}"]
            for item in descriptors
        ]
        assert len(generated_prompts) == 12
        assert all(not item.warmup for item in descriptors)
        assert events[0]["event"] == "container_started"
        assert all(
            event["provenance"] in module._PROVENANCE_DOMAINS for event in events
        )
        terminal = events[-1]["record"]
        assert terminal["correctness_evaluated_remotely"] is False
        assert terminal["compilation_seconds"] is None
        assert terminal["cuda_graph_seconds"] is None
        assert terminal["resolved_execution_config"] == {
            "enforce_eager": True,
            "compilation_mode": "NONE",
            "cuda_graph_mode": "NONE",
        }
        assert (
            terminal["compilation_seconds_unobservable_reason"]
            == "not_applicable_eager_mode"
        )
        assert all(request["correctness"] is None for request in terminal["requests"])
        module._verify_seal(terminal, "cell_sha256")
        assert all(
            request["output_tokens_per_second"] is not None
            for request in terminal["requests"]
        )
        assert not (artifact_dir / "cells" / "l40s-eager.json").exists()
        assert fake._fake_volumes[0].commits == 0
