from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import llmtracefx.cache_audit.real_mlx as real_mlx_module
import llmtracefx.cache_audit.runner as cache_runner
from llmtracefx.cache_audit.adapters.base import (
    AdapterAuditIdentity,
    CacheAuditCapability,
)
from llmtracefx.cache_audit.adapters.mlx import (
    REQUIRED_MLX_LM_VERSION,
    REQUIRED_MLX_VERSION,
    MLXStageObservation,
)
from llmtracefx.cache_audit.adapters.reference import ReferenceCacheAdapter
from llmtracefx.cache_audit.real_mlx import (
    EXPECTED_MODEL_ARTIFACT_DIGEST,
    MODEL_ID,
    REPLICATE_IDS,
    ROTATION_OFFSETS,
    TOKENIZER_ID,
    FrozenMLXWorkload,
    LifecycleMLXAdapter,
    RealMLXExperimentError,
    StageRecorder,
    _digest_bytes,
    _experiment_cache_config,
    _json_bytes,
    _safe_environment,
    _write_json,
    _write_recursive_checksums,
    assemble_aggregate,
    block_schedule,
    calibrate_seed,
    compile_workload,
    requests_for_replicate,
    sanitize_aggregate,
    verify_aggregate,
)
from llmtracefx.cache_audit.runner import run_audit
from llmtracefx.cache_audit.schema import (
    CacheConfig,
    PublicationMode,
    RequestSpec,
)


class FakeTokenizer:
    def _id(self, word: str) -> int:
        return 10 + sum((index + 1) * ord(char) for index, char in enumerate(word))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> list[int]:
        assert tokenize and add_generation_prompt and not enable_thinking
        tokens: list[int] = []
        for message in messages:
            tokens.append(1 if message["role"] == "user" else 3)
            tokens.extend(self._id(word) for word in message["content"].split())
            tokens.append(2)
        tokens.append(3)
        return tokens

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [self._id(word) for word in text.split()]


class MLXIdentityReference:
    backend = "mlx_lm_local"

    def capabilities(self) -> CacheAuditCapability:
        return CacheAuditCapability(backend=self.backend, supported=True)

    def audit_identity(self) -> AdapterAuditIdentity:
        return AdapterAuditIdentity(
            backend_version=REQUIRED_MLX_LM_VERSION,
            runtime_identity={
                "mlx": REQUIRED_MLX_VERSION,
                "mlx_lm": REQUIRED_MLX_LM_VERSION,
                "platform_machine": "arm64",
                "platform_system": "Darwin",
            },
            model_artifact_digest=EXPECTED_MODEL_ARTIFACT_DIGEST,
            cache_type="mlx_lru_prompt_cache",
            max_entries=2,
            max_bytes=1 << 63,
        )

    def run(self, requests: tuple[RequestSpec, ...]) -> list[Any]:
        return ReferenceCacheAdapter(max_entries=2, max_bytes=1 << 63).run(requests)


def test_workload_invariants_and_counterbalanced_schedule() -> None:
    workload = compile_workload(FakeTokenizer())
    assert len(workload.base) == 1025
    assert len(workload.shorter_seed) == 769
    assert len(workload.extension_32) == 32
    assert workload.mutation_137[137] != workload.base[137]
    assert workload.mutation_256[256] != workload.base[256]
    assert workload.suffix_change[:-16] == workload.base[:-16]
    assert workload.different_ids[0] != workload.base[0]
    assert all(
        left != right
        for left, right in zip(
            workload.suffix_change[-16:], workload.base[-16:], strict=True
        )
    )
    payload = workload.to_dict()
    payload["arrays"]["seed_output"] = [7]
    payload["workload_digest"] = FrozenMLXWorkload(
        **{**workload.__dict__, "seed_output": (7,)}
    ).to_dict()["workload_digest"]
    calibrated = FrozenMLXWorkload.from_dict(payload)
    schedules = [block_schedule(item) for item in REPLICATE_IDS]
    assert [schedule[0] for schedule in schedules] == [
        schedule[offset % len(schedule)]
        for schedule, offset in zip([schedules[0]] * 6, ROTATION_OFFSETS, strict=True)
    ]
    for replicate_id, schedule in zip(REPLICATE_IDS, schedules, strict=True):
        requests = requests_for_replicate(calibrated, replicate_id)
        observed = tuple(
            dict.fromkeys(request.request_id.split(":", 1)[0] for request in requests)
        )
        assert observed == schedule
        assert max(request.output_tokens for request in requests) == 8


def test_stage_records_have_explicit_nonconflated_scopes(tmp_path: Path) -> None:
    recorder = StageRecorder(
        "replicate-0",
        rss_reader=lambda: 123,
        swap_reader=lambda: 456,
    )
    recorder(
        MLXStageObservation(
            request_id="block:request",
            stage="request_after_lookup",
            active_bytes=1,
            peak_bytes=2,
            allocator_cache_bytes=3,
            logical_cache_bytes=4,
        )
    )
    output = tmp_path / "stages.jsonl"
    recorder.write(output)
    row = json.loads(output.read_text())
    assert row["allocator"]["scope"] == "mlx_process_global_allocator"
    assert row["logical_cache"]["scope"] == "current_lru_entry_when_observable"
    assert row["process_rss"] == {
        "bytes": 123,
        "scope": "current_replicate_child_process_only",
    }
    assert row["system_swap"] == {"used_bytes": 456, "scope": "system_wide"}
    assert row["thermal_power"] == {
        "thermal_state": None,
        "power_watts": None,
        "scope": "unavailable_without_safe_collector",
    }
    assert "pid" not in output.read_text()


def test_calibration_repeats_seed_with_fresh_adapters() -> None:
    workload = compile_workload(FakeTokenizer())
    creations = 0

    def factory() -> Any:
        nonlocal creations
        creations += 1
        return ReferenceCacheAdapter()

    calibrated = calibrate_seed(workload, factory)
    assert creations == 2
    assert calibrated.seed_output is not None
    assert 1 <= len(calibrated.seed_output) <= 8


def test_lifecycle_adapter_batches_each_block_before_baselines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grouped: list[tuple[str, ...]] = []

    class Probe:
        mlx_version = REQUIRED_MLX_VERSION
        mlx_lm_version = REQUIRED_MLX_LM_VERSION
        platform_machine = "arm64"
        platform_system = "Darwin"

    class FakeAdapter:
        def __init__(self, **_: Any) -> None:
            pass

        def run(self, requests: tuple[RequestSpec, ...]) -> list[Any]:
            grouped.append(tuple(request.request_id for request in requests))
            return []

    monkeypatch.setattr(real_mlx_module, "MLXLocalCacheAdapter", FakeAdapter)
    monkeypatch.setattr(
        real_mlx_module,
        "check_mlx_capabilities",
        lambda *_args, **_kwargs: CacheAuditCapability(
            backend="mlx_lm_local", supported=True
        ),
    )
    adapter = LifecycleMLXAdapter(
        runtime_factory=Probe,
        model="model",
        tokenizer="tokenizer",
        model_key="key",
        model_artifact_digest=EXPECTED_MODEL_ARTIFACT_DIGEST,
        correctness_evaluator=lambda _: True,
        stage_observer=lambda _: None,
    )
    adapter.run(
        (
            _request_spec("first:control", 0),
            _request_spec("first:treatment", 1),
            _request_spec("second:control", 2),
        )
    )

    assert grouped == [
        ("first:control", "first:treatment"),
        ("second:control",),
    ]


def _request_spec(request_id: str, order: int) -> RequestSpec:
    return RequestSpec(
        request_id=request_id,
        scenario=real_mlx_module.ScenarioKind.COLD,
        order=order,
        input_token_ids=(order + 1,),
        input_token_count=1,
    )


def _make_attempts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(cache_runner, "source_commit", lambda: (None, None))
    attempts = tmp_path / "attempts"
    attempts.mkdir()
    environment = _safe_environment()
    workload = replace(compile_workload(FakeTokenizer()), seed_output=(7, 8, 9))
    teardown = {
        "schema_version": "1",
        "cache_lifecycles_released": True,
        "garbage_collection_completed": True,
        "mlx_allocator_cleanup": "completed",
        "allocator_cache_bytes_after": 0,
        "scope": "current_replicate_child_process",
    }
    monkeypatch.setattr(
        real_mlx_module,
        "EXPECTED_CALIBRATED_WORKLOAD_DIGEST",
        workload.to_dict()["workload_digest"],
    )
    for index, replicate_id in enumerate(REPLICATE_IDS):
        replicate = attempts / replicate_id
        replicate.mkdir()
        if index == 5:
            _write_json(
                replicate / "attempt.json",
                {
                    "schema_version": "1",
                    "replicate_id": replicate_id,
                    "status": "failed",
                    "replacement": False,
                },
            )
            _write_json(
                replicate / "teardown.json",
                {
                    **teardown,
                    "cache_lifecycles_released": None,
                    "garbage_collection_completed": None,
                    "mlx_allocator_cleanup": "externally_terminated",
                    "allocator_cache_bytes_after": None,
                },
            )
            continue
        bundle = replicate / "bundle"
        requests = requests_for_replicate(workload, replicate_id)
        run_audit(
            adapter=MLXIdentityReference(),
            requests=requests,
            cache_config=CacheConfig(
                namespace_id="experiment-namespaces",
                cache_type="mlx_lru_prompt_cache",
                max_entries=2,
                max_bytes=1 << 63,
            ),
            output_dir=bundle,
            backend_version=REQUIRED_MLX_LM_VERSION,
            model_id=MODEL_ID,
            tokenizer_id=TOKENIZER_ID,
            model_artifact_digest=EXPECTED_MODEL_ARTIFACT_DIGEST,
            publication_mode=PublicationMode.PRIVATE,
            seed=index,
        )
        _write_json(
            replicate / "attempt.json",
            {
                "schema_version": "1",
                "replicate_id": replicate_id,
                "status": "complete",
                "rotation": list(block_schedule(replicate_id)),
                "request_count": len(requests),
                "independent_unit": True,
                "replacement": False,
                "frozen_workload_digest": workload.to_dict()["workload_digest"],
                "model_artifact_digest": EXPECTED_MODEL_ARTIFACT_DIGEST,
                "model_id": MODEL_ID,
                "tokenizer_id": TOKENIZER_ID,
                "runtime_identity_digest": _digest_bytes(
                    _json_bytes(real_mlx_module._EXPECTED_RUNTIME_IDENTITY)
                ),
                "cache_config_digest": _digest_bytes(
                    _json_bytes(_experiment_cache_config().to_dict())
                ),
            },
        )
        _write_json(replicate / "environment.json", environment)
        _write_json(replicate / "workload.json", workload.to_dict())
        recorder = StageRecorder(
            replicate_id,
            rss_reader=lambda: 100,
            swap_reader=lambda: 200,
        )
        request_index = 0
        for block in block_schedule(replicate_id):
            block_start = request_index
            recorder(
                MLXStageObservation(
                    request_id=None,
                    stage="lifecycle_ready",
                    active_bytes=1,
                    peak_bytes=2,
                    allocator_cache_bytes=3,
                    logical_cache_bytes=4,
                )
            )
            while (
                request_index < len(requests)
                and requests[request_index].request_id.split(":", 1)[0] == block
            ):
                request_id = requests[request_index].request_id
                for stage in (
                    "request_before_lookup",
                    "request_after_lookup",
                    "request_after_generation",
                    "request_after_insertion",
                ):
                    recorder(
                        MLXStageObservation(
                            request_id=request_id,
                            stage=stage,
                            active_bytes=1,
                            peak_bytes=2,
                            allocator_cache_bytes=3,
                            logical_cache_bytes=4,
                        )
                    )
                request_index += 1
            for request in requests[block_start:request_index]:
                recorder(
                    MLXStageObservation(
                        request_id=request.request_id,
                        stage="request_after_baseline",
                        active_bytes=1,
                        peak_bytes=2,
                        allocator_cache_bytes=3,
                        logical_cache_bytes=4,
                    )
                )
        recorder.write(replicate / "stages.jsonl")
        _write_json(replicate / "teardown.json", teardown)
    return attempts


def test_aggregate_regeneration_checksums_and_public_redaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = _make_attempts(tmp_path, monkeypatch)
    private = tmp_path / "private"
    assert assemble_aggregate(attempts, private)["complete_replicates"] == 5
    assert verify_aggregate(private)["verified"] is True
    result = subprocess.run(
        [sys.executable, str(private / "aggregate_verifier.py")],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    summary = private / "summary.json"
    value = json.loads(summary.read_text())
    value["request_count"] += 1
    _write_json(summary, value)
    _write_recursive_checksums(private)
    with pytest.raises(RealMLXExperimentError, match="derived aggregate"):
        verify_aggregate(private)

    shutil.rmtree(private)
    assemble_aggregate(attempts, private)
    public = tmp_path / "public"
    assert sanitize_aggregate(private, public)["publication_mode"] == "public_redacted"
    result = subprocess.run(
        [sys.executable, str(public / "aggregate_verifier.py")],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    text = "\n".join(
        path.read_text(encoding="utf-8") for path in public.rglob("*") if path.is_file()
    )
    assert '"input_token_ids":[' not in text
    assert 'request-0"' not in text

    stage_tamper = tmp_path / "stage-tamper"
    shutil.copytree(public, stage_tamper)
    stages = stage_tamper / "replicates" / "replicate-0" / "stages.jsonl"
    rows = stages.read_text().splitlines()
    rows[0], rows[1] = rows[1], rows[0]
    stages.write_text("\n".join(rows) + "\n", encoding="ascii")
    _write_recursive_checksums(stage_tamper)
    result = subprocess.run(
        [sys.executable, str(stage_tamper / "aggregate_verifier.py")],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode != 0
    assert "raw replicate evidence digest mismatch" in result.stderr

    descriptive = public / "descriptive-summary.json"
    descriptive_value = json.loads(descriptive.read_text())
    descriptive_value["paired_comparisons"] = {}
    _write_json(descriptive, descriptive_value)
    _write_recursive_checksums(public)
    result = subprocess.run(
        [sys.executable, str(public / "aggregate_verifier.py")],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode != 0
    assert "descriptive summary digest mismatch" in result.stderr


def test_aggregate_requires_five_of_six_without_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = _make_attempts(tmp_path, monkeypatch)
    for replicate_id in ("replicate-3", "replicate-4"):
        shutil.rmtree(attempts / replicate_id)
        replicate = attempts / replicate_id
        replicate.mkdir()
        _write_json(
            replicate / "attempt.json",
            {
                "schema_version": "1",
                "replicate_id": replicate_id,
                "status": "failed",
                "replacement": False,
            },
        )
        _write_json(
            replicate / "teardown.json",
            {
                "schema_version": "1",
                "cache_lifecycles_released": None,
                "garbage_collection_completed": None,
                "mlx_allocator_cleanup": "externally_terminated",
                "allocator_cache_bytes_after": None,
                "scope": "current_replicate_child_process",
            },
        )
    with pytest.raises(RealMLXExperimentError, match="at least 5"):
        assemble_aggregate(attempts, tmp_path / "aggregate")

    (attempts / "replicate-5").rename(attempts / "replicate-6")
    with pytest.raises(RealMLXExperimentError, match="replicate-0..5 exactly"):
        assemble_aggregate(attempts, tmp_path / "replacement")
