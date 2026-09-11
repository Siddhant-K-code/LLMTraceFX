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
    DEFAULT_CONVERSION_SUMMARY,
    EXPECTED_CONVERSION_SUMMARY_SHA256,
    EXPECTED_MODEL_ARTIFACT_DIGEST,
    LANE_IDS,
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
    expected_lengths = {
        "1k": (1025, 769, 513),
        "4k": (4097, 3073, 2049),
    }
    for lane in workload.lanes:
        base, shorter, eviction = expected_lengths[lane.lane_id]
        assert len(lane.base) == base
        assert len(lane.shorter_seed) == shorter
        assert lane.shorter_seed == lane.base[:shorter]
        assert len(lane.extension_32) == 32
        assert {len(lane.eviction_a), len(lane.eviction_b), len(lane.eviction_c)} == {
            eviction
        }
        assert lane.mutation_137[137] != lane.base[137]
        assert lane.mutation_256[256] != lane.base[256]
        assert [
            index
            for index, (left, right) in enumerate(
                zip(lane.base, lane.mutation_137, strict=True)
            )
            if left != right
        ] == [137]
        assert [
            index
            for index, (left, right) in enumerate(
                zip(lane.base, lane.mutation_256, strict=True)
            )
            if left != right
        ] == [256]
        assert lane.suffix_change[:-16] == lane.base[:-16]
        assert lane.different_ids[0] != lane.base[0]
        assert all(
            left != right
            for left, right in zip(
                lane.suffix_change[-16:], lane.base[-16:], strict=True
            )
        )
    calibrated_source = replace(
        workload,
        lanes=tuple(replace(lane, seed_output=(7, 8, 9)) for lane in workload.lanes),
    )
    payload = calibrated_source.to_dict()
    calibrated = FrozenMLXWorkload.from_dict(payload)
    tampered_payload = json.loads(json.dumps(payload))
    tampered_payload["lanes"]["1k"]["lane_id"] = "4k"
    with pytest.raises(RealMLXExperimentError, match="lane identity"):
        FrozenMLXWorkload.from_dict(tampered_payload)
    schedules = [block_schedule(item) for item in REPLICATE_IDS]
    assert [schedule[0] for schedule in schedules] == [
        schedule[offset % len(schedule)]
        for schedule, offset in zip([schedules[0]] * 6, ROTATION_OFFSETS, strict=True)
    ]
    for replicate_id, schedule in zip(REPLICATE_IDS, schedules, strict=True):
        assert len(schedule) == 18
        assert set(schedule) == {
            f"{lane_id}:{case}"
            for lane_id in LANE_IDS
            for case in (
                "cold-exact-duplicate",
                "longer-to-shorter",
                "shorter-to-longer",
                "interior-mutation",
                "allocation-step-mutation",
                "same-length-different-ids",
                "suffix-only-change",
                "namespace-isolation",
                "capacity-eviction",
            )
        }
        requests = requests_for_replicate(calibrated, replicate_id)
        observed = tuple(
            dict.fromkeys(request.request_id.rsplit(":", 1)[0] for request in requests)
        )
        assert observed == schedule
        assert len(requests) == 44
        assert {
            lane_id: sum(
                request.request_id.startswith(f"{lane_id}:") for request in requests
            )
            for lane_id in LANE_IDS
        } == {"1k": 22, "4k": 22}
        assert max(request.output_tokens for request in requests) == 8
        for lane_id, seed_length in (("1k", 769), ("4k", 3073)):
            extension = next(
                request
                for request in requests
                if request.request_id == f"{lane_id}:shorter-to-longer:longer"
            )
            assert extension.input_token_count == seed_length + 3 + 32


def test_qwen3_4b_conversion_metadata_is_exact() -> None:
    summary_bytes = DEFAULT_CONVERSION_SUMMARY.read_bytes()
    assert (
        _digest_bytes(summary_bytes) == f"sha256:{EXPECTED_CONVERSION_SUMMARY_SHA256}"
    )
    summary = json.loads(summary_bytes)
    assert summary["source"]["official_id"] == "Qwen/Qwen3-4B"
    assert (
        summary["source"]["official_revision"]
        == "1cfa9a7208912126459214e8b04321603b3df60c"
    )
    assert summary["source"]["license"] == "Apache-2.0"
    assert summary["converter"]["version"] == "0.31.3"
    assert (
        summary["converter"]["git_revision"]
        == "ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd"
    )
    assert summary["parameters"]["q_group_size"] == 64
    assert summary["parameters"]["q_bits"] == 4
    assert summary["parameters"]["q_mode"] == "affine"
    assert summary["output"]["repository_id"] == MODEL_ID
    assert summary["output"]["total_bytes"] == 2_274_515_269
    assert len(summary["output"]["files"]) == 8
    assert EXPECTED_MODEL_ARTIFACT_DIGEST == (
        "sha256:057a37f4ebc76420f8ab2edb17bc8442e050c8d13f7334f829356e2f9cab6802"
    )
    assert TOKENIZER_ID == ("Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c")
    manifest = json.loads(
        (
            DEFAULT_CONVERSION_SUMMARY.parent / "qwen3-4b-conversion-manifest-v1.json"
        ).read_text()
    )
    assert manifest["source"]["official_id"] == "Qwen/Qwen3-4B"
    assert (
        manifest["source"]["official_revision"]
        == "1cfa9a7208912126459214e8b04321603b3df60c"
    )
    assert manifest["source"]["license"] == "Apache-2.0"
    assert manifest["expected_output_bytes"] == 2_300_000_000


def test_stage_records_have_explicit_nonconflated_scopes(tmp_path: Path) -> None:
    recorder = StageRecorder(
        "replicate-0",
        rss_reader=lambda: 123,
        swap_reader=lambda: 456,
        memory_reader=lambda: 78.0,
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
    assert row["system_memory"] == {
        "free_percent": 78.0,
        "scope": "system_wide_memory_pressure",
    }
    assert row["thermal_power"] == {
        "thermal_state": None,
        "power_watts": None,
        "scope": "unavailable_without_safe_collector",
    }
    assert "pid" not in output.read_text()


def test_calibration_repeats_seed_with_fresh_adapters() -> None:
    workload = compile_workload(FakeTokenizer())
    creations = 0
    input_lengths: list[int] = []

    def factory() -> Any:
        nonlocal creations
        creations += 1
        adapter = ReferenceCacheAdapter()
        original = adapter.run

        def run(requests: tuple[RequestSpec, ...]) -> list[Any]:
            input_lengths.extend(request.input_token_count for request in requests)
            return original(requests)

        adapter.run = run  # type: ignore[method-assign]
        return adapter

    calibrated = calibrate_seed(workload, factory)
    assert creations == 4
    assert input_lengths == [769, 769, 3073, 3073]
    for lane in calibrated.lanes:
        assert lane.seed_output is not None
        assert 1 <= len(lane.seed_output) <= 8


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
            _request_spec("1k:cold-exact-duplicate:control", 0),
            _request_spec("1k:cold-exact-duplicate:treatment", 1),
            _request_spec("4k:cold-exact-duplicate:control", 2),
        )
    )

    assert grouped == [
        (
            "1k:cold-exact-duplicate:control",
            "1k:cold-exact-duplicate:treatment",
        ),
        ("4k:cold-exact-duplicate:control",),
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
    environment = {
        **_safe_environment(),
        "mlx": REQUIRED_MLX_VERSION,
        "mlx_lm": REQUIRED_MLX_LM_VERSION,
    }
    workload = compile_workload(FakeTokenizer())
    workload = replace(
        workload,
        lanes=tuple(replace(lane, seed_output=(7, 8, 9)) for lane in workload.lanes),
    )
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
    monkeypatch.setattr(
        real_mlx_module,
        "EXPECTED_CALIBRATED_LANE_DIGESTS",
        {lane.lane_id: lane.to_dict()["lane_digest"] for lane in workload.lanes},
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
                "lane_request_counts": {"1k": 22, "4k": 22},
                "independent_unit": True,
                "replacement": False,
                "frozen_workload_digest": workload.to_dict()["workload_digest"],
                "frozen_lane_digests": {
                    lane.lane_id: lane.to_dict()["lane_digest"]
                    for lane in workload.lanes
                },
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
            memory_reader=lambda: 80.0,
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
                and requests[request_index].request_id.rsplit(":", 1)[0] == block
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
    private_index = json.loads((private / "replicate-index.json").read_text())
    assert private_index["lane_request_counts"] == {"1k": 110, "4k": 110}
    assert set(private_index["lane_scenario_counts"]) == set(LANE_IDS)
    private_claims = json.loads((private / "claim-matrix.json").read_text())
    assert private_claims["lane_request_observations"] == {
        "1k": 110,
        "4k": 110,
    }
    comparisons = json.loads((private / "descriptive-summary.json").read_text())[
        "comparisons"
    ]
    assert len(comparisons) == 18
    assert all(
        sample["lane_id"] == name.split(":", 1)[0]
        for name, comparison in comparisons.items()
        for sample in comparison["samples"]
    )
    contract = json.loads((private / "experiment-contract.json").read_text())
    assert set(contract["lanes"]) == set(LANE_IDS)
    assert len(contract["combined_blocks"]) == 18
    result = subprocess.run(
        [sys.executable, str(private / "aggregate_verifier.py")],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    lane_tamper = tmp_path / "lane-tamper"
    shutil.copytree(private, lane_tamper)
    attempt = lane_tamper / "replicates" / "replicate-0" / "attempt.json"
    attempt_value = json.loads(attempt.read_text())
    attempt_value["lane_request_counts"] = {"1k": 21, "4k": 23}
    _write_json(attempt, attempt_value)
    _write_recursive_checksums(lane_tamper)
    with pytest.raises(RealMLXExperimentError, match="binding is invalid"):
        verify_aggregate(lane_tamper)
    result = subprocess.run(
        [sys.executable, str(lane_tamper / "aggregate_verifier.py")],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode != 0
    assert "raw replicate evidence digest mismatch" in result.stderr

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
    public_index = json.loads((public / "replicate-index.json").read_text())
    assert public_index["lane_request_counts"] == {"1k": 110, "4k": 110}
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
