from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import fields, replace
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
    SCHEDULE_AFFINE_PERMUTATIONS,
    TOKENIZER_ID,
    FrozenMLXWorkload,
    LifecycleMLXAdapter,
    RealMLXExperimentError,
    StageRecorder,
    _digest_bytes,
    _experiment_cache_config,
    _json_bytes,
    _request_stage_memory,
    _require_private_output_gates,
    _safe_environment,
    _write_json,
    _write_recursive_checksums,
    assemble_aggregate,
    block_schedule,
    calibrate_outputs,
    compile_workload,
    load_workload,
    requests_for_replicate,
    sanitize_aggregate,
    verify_aggregate,
)
from llmtracefx.cache_audit.runner import run_audit
from llmtracefx.cache_audit.schema import (
    CacheConfig,
    PublicationMode,
    RequestSpec,
    Verdict,
)
from llmtracefx.optimizer.lab.qwen3_8b.conversion import conversion_manifest_hash
from llmtracefx.optimizer.lab.qwen3_8b.conversion_manifest import ConversionManifest


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
        return [
            replace(
                record,
                output=replace(
                    record.output,
                    output_token_ids=record.output.output_token_ids[:3],
                    baseline_token_ids=record.output.baseline_token_ids[:3],
                ),
            )
            for record in ReferenceCacheAdapter(max_entries=2, max_bytes=1 << 63).run(
                requests
            )
        ]


def _reference_output(tokens: tuple[int, ...]) -> tuple[int, ...]:
    payload = b"|".join(str(token).encode("ascii") for token in tokens)
    return tuple(hashlib.sha256(payload).digest()[:3])


def test_workload_invariants_and_counterbalanced_schedule() -> None:
    workload = compile_workload(FakeTokenizer())
    expected_lengths = {
        "1k": (1025, 513),
        "4k": (4097, 2049),
    }
    for lane in workload.lanes:
        base, eviction = expected_lengths[lane.lane_id]
        assert len(lane.base) == base
        assert {field.name for field in fields(lane)} == {
            "lane_id",
            "base",
            "different_ids",
            "mutation_137",
            "mutation_256",
            "suffix_change",
            "eviction_a",
            "eviction_b",
            "eviction_c",
            "calibration_output",
        }
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
        assert lane.suffix_change[: base - 32] == lane.base[: base - 32]
        assert lane.suffix_change[-16:] == lane.base[-16:]
        assert lane.different_ids[0] == lane.base[0]
        assert lane.different_ids != lane.base
        assert all(
            left != right
            for left, right in zip(
                lane.suffix_change[-32:-16], lane.base[-32:-16], strict=True
            )
        )
        assert lane.base[-1] == lane.different_ids[-1] == lane.suffix_change[-1] == 3
        assert lane.eviction_a[-1] == lane.eviction_b[-1] == lane.eviction_c[-1] == 3
    calibrated_source = replace(
        workload,
        lanes=tuple(
            replace(
                lane,
                calibration_output=_reference_output(lane.base),
            )
            for lane in workload.lanes
        ),
    )
    payload = calibrated_source.to_dict()
    assert all(
        set(payload["lanes"][lane_id]["arrays"])
        == {
            "base",
            "different_ids",
            "mutation_137",
            "mutation_256",
            "suffix_change",
            "eviction_a",
            "eviction_b",
            "eviction_c",
            "calibration_output",
        }
        for lane_id in LANE_IDS
    )
    calibrated = FrozenMLXWorkload.from_dict(payload)
    tampered_payload = json.loads(json.dumps(payload))
    tampered_payload["lanes"]["1k"]["lane_id"] = "4k"
    with pytest.raises(RealMLXExperimentError, match="lane identity"):
        FrozenMLXWorkload.from_dict(tampered_payload)
    schedules = [block_schedule(item) for item in REPLICATE_IDS]
    assert len(set(schedules)) == len(REPLICATE_IDS)
    assert SCHEDULE_AFFINE_PERMUTATIONS == (
        (0, 1),
        (1, 3),
        (2, 5),
        (3, 9),
        (4, 11),
        (5, 13),
    )
    assert all(
        block.split(":", 1)[0] != schedules[0][index + 1].split(":", 1)[0]
        for index, block in enumerate(schedules[0][:-1])
    )
    adjacency_sets = {
        frozenset(zip(schedule[:-1], schedule[1:], strict=True))
        for schedule in schedules
    }
    assert len(adjacency_sets) == len(REPLICATE_IDS)
    for replicate_id, schedule in zip(REPLICATE_IDS, schedules, strict=True):
        assert len(schedule) == 14
        assert set(schedule) == {
            f"{lane_id}:{case}"
            for lane_id in LANE_IDS
            for case in (
                "cold-exact-duplicate",
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
        assert len(requests) == 36
        assert {
            lane_id: sum(
                request.request_id.startswith(f"{lane_id}:") for request in requests
            )
            for lane_id in LANE_IDS
        } == {"1k": 18, "4k": 18}
        assert max(request.output_tokens for request in requests) == 8
        for lane_id in LANE_IDS:
            lane_requests = [
                request
                for request in requests
                if request.request_id.startswith(f"{lane_id}:")
            ]
            assert {request.input_token_count for request in lane_requests} == {
                *expected_lengths[lane_id],
            }
            eviction_requests = [
                request
                for request in requests
                if request.request_id.startswith(f"{lane_id}:capacity-eviction:")
            ]
            namespaces = {
                request.request_id.rsplit(":", 1)[-1]: request.namespace_id
                for request in eviction_requests
            }
            assert namespaces["a-seed"] == namespaces["a-hit"] == namespaces["a-miss"]
            assert len({namespaces["a-seed"], namespaces["b"], namespaces["c"]}) == 3


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
    assert (
        conversion_manifest_hash(ConversionManifest.from_dict(manifest))
        == summary["conversion_manifest_hash"]
    )
    assert manifest["source"]["official_id"] == "Qwen/Qwen3-4B"
    assert (
        manifest["source"]["official_revision"]
        == "1cfa9a7208912126459214e8b04321603b3df60c"
    )
    assert manifest["source"]["license"] == "Apache-2.0"
    assert manifest["expected_output_bytes"] == 2_300_000_000


def test_calibrated_workload_refuses_unpinned_digests(tmp_path: Path) -> None:
    workload = compile_workload(FakeTokenizer())
    calibrated = replace(
        workload,
        lanes=tuple(
            replace(
                lane,
                calibration_output=_reference_output(lane.base),
            )
            for lane in workload.lanes
        ),
    )
    path = tmp_path / "calibrated.json"
    _write_json(path, calibrated.to_dict())

    with pytest.raises(RealMLXExperimentError, match="contract mismatch"):
        load_workload(path, calibrated=True)


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


def test_request_stage_memory_excludes_fresh_baseline_observation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "stages.jsonl"
    rows = []
    for stage, rss, swap, free in (
        ("request_before_lookup", 100, 200, 80.0),
        ("request_after_generation", 150, 250, 75.0),
        ("request_after_baseline", 9_999, 8_888, 1.0),
    ):
        rows.append(
            {
                "request_id": "request",
                "stage": stage,
                "process_rss": {"bytes": rss},
                "system_swap": {"used_bytes": swap},
                "system_memory": {"free_percent": free},
            }
        )
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="ascii",
    )

    assert _request_stage_memory(path) == {
        "request": {
            "process_rss_bytes": 150,
            "system_swap_used_bytes": 250,
            "system_memory_free_percent": 75.0,
        }
    }


def test_calibration_repeats_base_with_fresh_adapters() -> None:
    workload = compile_workload(FakeTokenizer())
    creations = 0
    input_lengths: list[int] = []
    outputs = {len(lane.base): _reference_output(lane.base) for lane in workload.lanes}

    def factory() -> Any:
        nonlocal creations
        creations += 1
        adapter = ReferenceCacheAdapter()
        original = adapter.run

        def run(requests: tuple[RequestSpec, ...]) -> list[Any]:
            input_lengths.extend(request.input_token_count for request in requests)
            records = original(requests)
            return [
                replace(
                    record,
                    output=replace(
                        record.output,
                        output_token_ids=outputs[record.spec.input_token_count],
                    ),
                )
                for record in records
            ]

        adapter.run = run  # type: ignore[method-assign]
        return adapter

    calibrated = calibrate_outputs(workload, factory)
    assert creations == 4
    assert input_lengths == [1025, 1025, 4097, 4097]
    for lane in calibrated.lanes:
        assert lane.calibration_output == outputs[len(lane.base)]
        assert len(lane.calibration_output) == 3


def test_cold_controls_must_match_lane_calibration_outputs() -> None:
    workload = compile_workload(FakeTokenizer())
    calibrated = replace(
        workload,
        lanes=tuple(
            replace(lane, calibration_output=_reference_output(lane.base))
            for lane in workload.lanes
        ),
    )
    requests = tuple(
        RequestSpec(
            request_id=f"{lane.lane_id}:cold-exact-duplicate:cold",
            scenario=real_mlx_module.ScenarioKind.COLD,
            order=index,
            input_token_ids=lane.base,
            input_token_count=len(lane.base),
            output_tokens=8,
        )
        for index, lane in enumerate(calibrated.lanes)
    )
    records = MLXIdentityReference().run(requests)
    _require_private_output_gates(records, calibrated)

    mismatched = replace(
        calibrated,
        lanes=(
            replace(calibrated.lanes[0], calibration_output=(0, 0, 0)),
            calibrated.lanes[1],
        ),
    )
    with pytest.raises(RealMLXExperimentError, match="differs from calibration"):
        _require_private_output_gates(records, mismatched)


def test_failed_replicate_finishes_with_one_valid_terminal_marker(
    tmp_path: Path,
) -> None:
    workload = tmp_path / "missing-workload.json"
    output = tmp_path / "replicate"

    with pytest.raises(RealMLXExperimentError):
        real_mlx_module.run_replicate(
            workload,
            output,
            replicate_id="replicate-0",
            model_dir=tmp_path / "model",
        )

    assert real_mlx_module.verify_replicate(
        output,
        replicate_id="replicate-0",
        public=False,
    ) == {"replicate_id": "replicate-0", "status": "failed"}
    attempt = json.loads((output / "attempt.json").read_text())
    assert attempt["reason"] == "replicate_execution_failed"


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
    workload = compile_workload(FakeTokenizer())
    workload = replace(
        workload,
        lanes=tuple(
            replace(
                lane,
                calibration_output=_reference_output(lane.base),
            )
            for lane in workload.lanes
        ),
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
                    "reason": "synthetic_failure",
                    "failed_at": "2026-01-01T00:00:00.000000Z",
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
                "lane_request_counts": {"1k": 18, "4k": 18},
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
        _write_json(
            replicate / "environment.json",
            {
                **_safe_environment(instance_id=f"{index + 1:032x}"),
                "mlx": REQUIRED_MLX_VERSION,
                "mlx_lm": REQUIRED_MLX_LM_VERSION,
            },
        )
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
    assert private_index["lane_request_counts"] == {"1k": 90, "4k": 90}
    assert set(private_index["lane_scenario_counts"]) == set(LANE_IDS)
    private_claims = json.loads((private / "claim-matrix.json").read_text())
    assert private_claims["lane_request_observations"] == {
        "1k": 90,
        "4k": 90,
    }
    comparisons = json.loads((private / "descriptive-summary.json").read_text())[
        "comparisons"
    ]
    assert len(comparisons) == 14
    assert all(
        sample["lane_id"] == name.split(":", 1)[0]
        for name, comparison in comparisons.items()
        for sample in comparison["samples"]
    )
    assert all(
        sample["paired_latency_comparable"] is False
        and sample["client_ttft_ratio"] is None
        and sample["client_ttft_difference_seconds"] is None
        and sample["process_rss_difference_bytes"] is None
        for comparison in comparisons.values()
        for sample in comparison["samples"]
    )
    contract = json.loads((private / "experiment-contract.json").read_text())
    assert set(contract["lanes"]) == set(LANE_IDS)
    assert len(contract["combined_blocks"]) == 14
    assert set(contract["exact_block_schedules"]) == set(REPLICATE_IDS)
    assert not (private / "aggregate_verifier.py").exists()

    lane_tamper = tmp_path / "lane-tamper"
    shutil.copytree(private, lane_tamper)
    attempt = lane_tamper / "replicates" / "replicate-0" / "attempt.json"
    attempt_value = json.loads(attempt.read_text())
    attempt_value["lane_request_counts"] = {"1k": 17, "4k": 19}
    _write_json(attempt, attempt_value)
    _write_recursive_checksums(lane_tamper)
    with pytest.raises(RealMLXExperimentError, match="binding is invalid"):
        verify_aggregate(lane_tamper)

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
    assert public_index["lane_request_counts"] == {"1k": 90, "4k": 90}
    assert not (public / "aggregate_verifier.py").exists()
    assert "run_instance_id" not in json.loads(
        (public / "environment.json").read_text()
    )
    assert all(
        "run_instance_id"
        not in json.loads(
            (public / "replicates" / replicate_id / "environment.json").read_text()
        )
        for replicate_id in REPLICATE_IDS[:5]
    )
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
    with pytest.raises(RealMLXExperimentError, match="stage boundary"):
        verify_aggregate(stage_tamper)

    descriptive = public / "descriptive-summary.json"
    descriptive_value = json.loads(descriptive.read_text())
    descriptive_value["paired_comparisons"] = {}
    _write_json(descriptive, descriptive_value)
    _write_recursive_checksums(public)
    with pytest.raises(RealMLXExperimentError, match="derived aggregate"):
        verify_aggregate(public)


def test_capacity_eviction_gate_requires_both_lane_treatments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempts = _make_attempts(tmp_path, monkeypatch)
    _, records = real_mlx_module.read_bundle(attempts / "replicate-0" / "bundle")
    real_mlx_module._require_capacity_eviction_verdicts(records)
    tampered = [
        (
            replace(record, verdict=Verdict.VERIFIED_MISS)
            if record.spec.request_id == "4k:capacity-eviction:a-miss"
            else record
        )
        for record in records
    ]

    with pytest.raises(RealMLXExperimentError, match="4k capacity-eviction"):
        real_mlx_module._require_capacity_eviction_verdicts(tampered)


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
                "reason": "synthetic_failure",
                "failed_at": "2026-01-01T00:00:00.000000Z",
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


def test_run_all_uses_fresh_offline_sandboxed_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launches: list[dict[str, Any]] = []
    observed_instances: set[str] = set()

    class FakeProcess:
        def __init__(self, command: list[str], **kwargs: Any) -> None:
            self.pid = 10_000 + len(launches)
            self.returncode = 0
            output = Path(command[-1])
            output.mkdir()
            instance = kwargs["env"][real_mlx_module.RUN_INSTANCE_ENV]
            assert instance not in observed_instances
            observed_instances.add(instance)
            launches.append({"command": command, **kwargs})

        def poll(self) -> int:
            return 0

        def wait(self) -> int:
            return 0

    safe_observation = {
        "chip": "Apple M5 Pro",
        "total_memory_bytes": 24 * 1024**3,
        "vm_stat_available_ratio": 0.5,
        "system_swap_used_bytes": 0,
        "disk_free_bytes": 100 * 1024**3,
        "heavy_process_count": 0,
        "heavy_process_categories": [],
        "scopes": {},
    }
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-child")
    monkeypatch.setattr(real_mlx_module, "_validate_supervisor_source", lambda _: None)
    monkeypatch.setattr(
        real_mlx_module,
        "_replicate_child_command",
        lambda **kwargs: [
            "child",
            "--replicate-id",
            str(kwargs["replicate_id"]),
            "--output",
            str(kwargs["output_dir"]),
        ],
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_machine_observation",
        lambda *_args, **_kwargs: safe_observation,
    )
    monkeypatch.setattr(real_mlx_module.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(real_mlx_module, "_group_exists", lambda _: False)
    monkeypatch.setattr(real_mlx_module, "verify_replicate", lambda *_a, **_k: {})

    workspace = tmp_path / "run"
    result = real_mlx_module.run_all_replicates(
        workload=tmp_path / "workload.json",
        model_dir=tmp_path / "model",
        conversion_summary=tmp_path / "summary.json",
        output_workspace=workspace,
        expected_commit="a" * 40,
    )

    assert result == {
        "run_all_complete": True,
        "attempted_replicates": 6,
        "complete_replicates": 6,
        "failed_replicates": 0,
        "aggregate_created": False,
        "sanitized": False,
    }
    assert len(launches) == 6
    assert len(observed_instances) == 6
    assert all(launch["start_new_session"] is True for launch in launches)
    assert all(launch["shell"] is False for launch in launches)
    assert all("OPENAI_API_KEY" not in launch["env"] for launch in launches)
    assert all(launch["env"]["HF_HUB_OFFLINE"] == "1" for launch in launches)
    assert all(
        launch["env"][real_mlx_module.RUN_INSTANCE_ENV] in observed_instances
        for launch in launches
    )
    assert [
        launch["command"][launch["command"].index("--replicate-id") + 1]
        for launch in launches
    ] == list(REPLICATE_IDS)
    assert {path.name for path in (workspace / "attempts").iterdir()} == set(
        REPLICATE_IDS
    )
    ledger = [
        json.loads(line)
        for line in (workspace / "run-ledger.jsonl").read_text().splitlines()
    ]
    assert [row["replicate_id"] for row in ledger[:6]] == list(REPLICATE_IDS)
    assert all(row["event"] == "planned" for row in ledger[:6])
    assert [
        row["replicate_id"] for row in ledger if row["event"] == "finalized"
    ] == list(REPLICATE_IDS)
    assert not any("pid" in row or "path" in row for row in ledger)


def test_run_all_marks_each_preflight_failure_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocked = {
        "chip": "Apple M5 Pro",
        "total_memory_bytes": 24 * 1024**3,
        "vm_stat_available_ratio": 0.24,
        "system_swap_used_bytes": 0,
        "disk_free_bytes": 100 * 1024**3,
        "heavy_process_count": 0,
        "heavy_process_categories": [],
        "scopes": {},
    }
    monkeypatch.setattr(real_mlx_module, "_validate_supervisor_source", lambda _: None)
    monkeypatch.setattr(
        real_mlx_module,
        "_replicate_child_command",
        lambda **_kwargs: ["child"],
    )
    monkeypatch.setattr(
        real_mlx_module,
        "_machine_observation",
        lambda *_args, **_kwargs: blocked,
    )

    workspace = tmp_path / "blocked"
    result = real_mlx_module.run_all_replicates(
        workload=tmp_path / "workload.json",
        model_dir=tmp_path / "model",
        conversion_summary=tmp_path / "summary.json",
        output_workspace=workspace,
        expected_commit="a" * 40,
    )

    assert result["complete_replicates"] == 0
    for replicate_id in REPLICATE_IDS:
        attempt = json.loads(
            (workspace / "attempts" / replicate_id / "attempt.json").read_text()
        )
        assert attempt["status"] == "failed"
        assert attempt["reason"] == "preflight_memory_below_floor"
    ledger = [
        json.loads(line)
        for line in (workspace / "run-ledger.jsonl").read_text().splitlines()
    ]
    assert sum(row["event"] == "planned" for row in ledger) == 6
    assert sum(row["event"] == "finalized" for row in ledger) == 6
