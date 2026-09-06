from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from vllm_kv_truth.vllm_live import (
    SourceFileDigest,
    canonical_json,
    compute_sha256_cbor_block_hashes,
    parse_identity_receipt,
    parse_live_kv_event_batch,
    parse_runtime_attestation,
    required_source_file_digests,
    required_source_file_paths,
)
from vllm_kv_truth import runner
from vllm_kv_truth.workload import (
    BLOCK_SIZE,
    EVICTION_LANE_REQUESTS,
    NESTED_PROBES,
    PREFIX_MATCH_UNIT,
)


def _hex(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


#: A well-formed placeholder derived-image id ("sha256:" + 64 hex chars)
#: used as the fixed test-suite stand-in for what the deploy orchestrator's
#: ``EXPECTED_IMAGE_ID`` environment variable carries in production: the
#: real, freshly built-and-inspected Docker image id of the derived overlay
#: image, injected only *after* that build -- never a preregistered
#: historical constant, since the derived image is built from whichever
#: repository HEAD was actually staged for this run.
_TEST_EXPECTED_IMAGE_ID = "sha256:" + "7" * 64


@pytest.fixture(autouse=True)
def _expected_image_id_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every real deploy-orchestrated run always has ``EXPECTED_IMAGE_ID``
    already set in the process environment before this module's CLI ever
    runs (see ``vllm_kv_truth_lifecycle.build_docker_run_argv``), so tests
    that exercise the default ``environ=None`` -> ``os.environ`` fallback
    path (``protocol_reasons_for_attestation``/``build_engine_kwargs``/
    ``assert_protocol_attestation``) need a matching value present too,
    exactly like production always has one. Unlike
    ``EXPECTED_REPOSITORY_COMMIT``, there is no historical constant this
    could silently fall back to instead.
    """

    monkeypatch.setenv("EXPECTED_IMAGE_ID", _TEST_EXPECTED_IMAGE_ID)


def _source_file_digests() -> list[dict[str, Any]]:
    expected = required_source_file_digests()
    return [
        {"path": path, "sha256": digest, "matches_manifest": True}
        for path, digest in sorted(expected.items())
    ]


def _identity_payload(**overrides: Any) -> dict[str, Any]:
    runtime_pins = runner.RUNTIME_PINS
    payload: dict[str, Any] = {
        "schema_version": "1",
        "protocol_id": runner.PROTOCOL_ID,
        "generated_at": "2026-09-06T08:10:00Z",
        "repository_commit": runner.PLAN_APPROVED_REPOSITORY_COMMIT,
        "image_repository_digest": runner.BASE_IMAGE_REFERENCE,
        "image_id": _TEST_EXPECTED_IMAGE_ID,
        "vllm_version": "0.28.0",
        "vllm_commit": runner.VLLM_SOURCE_COMMIT,
        "python_version": runtime_pins["python_version"],
        "torch_version": runtime_pins["torch_version"],
        "cuda_runtime_version": runtime_pins["cuda_version"],
        "transformers_version": runtime_pins["transformers_version"],
        "typing_extensions_version": runtime_pins["typing_extensions_version"],
        "cuda_driver_version": runner.EXPECTED_DRIVER,
        "gpu_name": runner.EXPECTED_GPU_NAME,
        "gpu_memory_mib": runner.EXPECTED_MEMORY_MIB,
        "gpu_compute_capability": "8.9",
        "gpu_uuid_commitment": _hex("gpu-uuid-salted"),
        "experiment_nonce": "nonce-abc",
        "installed_distributions_digest": "sha256:" + _hex("dist"),
        "wheel_record_digest": "sha256:" + _hex("wheel"),
        "package_tree_digest": "sha256:" + _hex("tree"),
        "source_file_digests": _source_file_digests(),
        "model_id": runner.MODEL_ID,
        "model_revision": runner.MODEL_REVISION,
        "tokenizer_artifact_digest": "sha256:" + _hex("tok"),
        "model_inventory_digest": "sha256:" + _hex("inv"),
        "model_path_commitment": _hex("modelpath"),
        "runner_source_digest": "sha256:" + _hex("runner"),
    }
    payload.update(overrides)
    sealed = dict(payload)
    payload["seal"] = hashlib.sha256(canonical_json(sealed).encode()).hexdigest()
    return payload


def _resolved_config(**overrides: Any) -> dict[str, Any]:
    base = {
        "max_model_len": runner.MAX_MODEL_LEN,
        "max_num_seqs": runner.MAX_NUM_SEQS,
        "tensor_parallel_size": runner.TENSOR_PARALLEL_SIZE,
        "data_parallel_size": runner.DATA_PARALLEL_SIZE,
        "block_size": BLOCK_SIZE,
        "prefix_match_unit": PREFIX_MATCH_UNIT,
        "num_gpu_blocks_override": 96,
        "gpu_memory_utilization": runner.GPU_MEMORY_UTILIZATION,
        "cache_dtype": runner.CACHE_DTYPE,
        "prefix_caching_hash_algo": runner.PREFIX_CACHING_HASH_ALGO,
        "kv_events_use_int_block_hashes": runner.KV_EVENTS_USE_INT_BLOCK_HASHES,
        "pythonhashseed": "0",
        "enable_prefix_caching": True,
        "enable_kv_cache_events": True,
        "enforce_eager": runner.ENFORCE_EAGER,
        "speculative_config_enabled": runner.SPECULATIVE_CONFIG_ENABLED,
        "lora_enabled": runner.LORA_ENABLED,
        "multimodal_enabled": runner.MULTIMODAL_ENABLED,
        "cache_salt_present": False,
        "cache_salt": None,
    }
    base.update(overrides)
    return base


def _kv_events_config(**overrides: Any) -> dict[str, Any]:
    base = {
        "topic": runner.KV_EVENTS_TOPIC,
        "endpoint_role": "loopback_pub",
        "replay_endpoint_role": "loopback_replay",
        "buffer_steps": 10_000,
        "hwm": 100_000,
        "max_queue_size": 100_000,
        "data_parallel_rank": 0,
        "first_sequence": 0,
        "last_sequence": 1,
        "capture_start_monotonic": 1.0,
        "capture_end_monotonic": 2.0,
    }
    base.update(overrides)
    return base


def _protocol_attestation_payload(
    *,
    identity: dict[str, Any] | None = None,
    resolved_config: dict[str, Any] | None = None,
    kv_events_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity_payload = identity if identity is not None else _identity_payload()
    resolved_config_payload = _resolved_config(**(resolved_config or {}))
    kv_events_config_payload = _kv_events_config(**(kv_events_config or {}))
    payload: dict[str, Any] = {
        "identity": identity_payload,
        "resolved_config": resolved_config_payload,
        "kv_events_config": kv_events_config_payload,
        "attested_at": "2026-09-06T08:30:00Z",
    }
    unsealed = {
        "identity_seal": identity_payload["seal"],
        "resolved_config": resolved_config_payload,
        "kv_events_config": kv_events_config_payload,
        "attested_at": payload["attested_at"],
    }
    payload["seal"] = hashlib.sha256(canonical_json(unsealed).encode()).hexdigest()
    return payload


def _valid_attestation() -> Any:
    return parse_runtime_attestation(_protocol_attestation_payload())


def test_valid_protocol_attestation_has_no_refusal_reasons() -> None:
    attestation = _valid_attestation()
    assert runner.protocol_reasons_for_attestation(attestation) == ()
    runner.assert_protocol_attestation(attestation)  # must not raise


def test_build_engine_kwargs_uses_local_verified_path_not_remote_id() -> None:
    attestation = _valid_attestation()
    kwargs = runner.build_engine_kwargs(
        attestation, model_path="/verified/models/qwen3-8b"
    )
    assert kwargs["model"] == "/verified/models/qwen3-8b"
    assert kwargs["tokenizer"] == "/verified/models/qwen3-8b"
    assert "revision" not in kwargs
    assert kwargs["trust_remote_code"] is False
    assert kwargs["tensor_parallel_size"] == 1
    assert kwargs["data_parallel_size"] == 1
    assert kwargs["max_model_len"] == 1024
    assert kwargs["max_num_seqs"] == 1
    assert kwargs["block_size"] == 16
    assert kwargs["prefix_match_unit"] == 16
    assert kwargs["num_gpu_blocks_override"] == 96
    assert kwargs["gpu_memory_utilization"] == 0.90
    assert kwargs["enforce_eager"] is True
    assert kwargs["enable_prefix_caching"] is True
    assert kwargs["prefix_caching_hash_algo"] == "sha256_cbor"
    assert kwargs["seed"] == runner.SAMPLING_SEED
    assert kwargs["kv_events_config"] == runner.kv_events_config_kwargs()


def test_build_engine_kwargs_refuses_empty_model_path() -> None:
    attestation = _valid_attestation()
    with pytest.raises(runner.KVTruthProtocolError, match="model_path"):
        runner.build_engine_kwargs(attestation, model_path="  ")


def test_kv_events_config_kwargs_uses_fixed_loopback_ports() -> None:
    config = runner.kv_events_config_kwargs()
    assert config["endpoint"] == "tcp://127.0.0.1:57003"
    assert config["replay_endpoint"] == "tcp://127.0.0.1:57004"
    assert config["publisher"] == "zmq"
    assert config["enable_kv_cache_events"] is True


@pytest.mark.parametrize(
    ("overrides", "expected_reason"),
    [
        ({"protocol_id": "some-other-protocol"}, "protocol_id_mismatch"),
        ({"model_id": "Qwen/Qwen3-14B"}, "model_id_mismatch"),
        ({"gpu_name": "NVIDIA GeForce RTX 3090"}, "gpu_name_mismatch"),
        ({"gpu_memory_mib": 12288}, "gpu_memory_mismatch"),
        ({"repository_commit": "0" * 40}, "repository_commit_mismatch"),
    ],
)
def test_top_level_protocol_mismatch_is_reported(
    overrides: dict[str, Any], expected_reason: str
) -> None:
    attestation = parse_runtime_attestation(
        _protocol_attestation_payload(identity=_identity_payload(**overrides))
    )
    reasons = runner.protocol_reasons_for_attestation(attestation)
    assert expected_reason in reasons
    with pytest.raises(runner.KVTruthProtocolError):
        runner.build_engine_kwargs(attestation, model_path="/verified/models/qwen3-8b")


@pytest.mark.parametrize(
    ("resolved_overrides", "expected_reason"),
    [
        ({"max_model_len": 2048}, "max_model_len_mismatch"),
        ({"max_num_seqs": 2}, "max_num_seqs_mismatch"),
        ({"num_gpu_blocks_override": 64}, "num_gpu_blocks_override_mismatch"),
        ({"lora_enabled": True}, "lora_enabled_mismatch"),
        ({"multimodal_enabled": True}, "multimodal_enabled_mismatch"),
        ({"speculative_config_enabled": True}, "speculative_config_enabled_mismatch"),
    ],
)
def test_resolved_config_protocol_mismatch_is_reported(
    resolved_overrides: dict[str, Any], expected_reason: str
) -> None:
    attestation = parse_runtime_attestation(
        _protocol_attestation_payload(resolved_config=resolved_overrides)
    )
    reasons = runner.protocol_reasons_for_attestation(attestation)
    assert expected_reason in reasons


def test_kv_events_topic_mismatch_is_reported() -> None:
    attestation = parse_runtime_attestation(
        _protocol_attestation_payload(kv_events_config={"topic": "wrong-topic"})
    )
    assert "kv_events_topic_mismatch" in runner.protocol_reasons_for_attestation(
        attestation
    )


# ---------------------------------------------------------------------------
# Offline/determinism environment contract
# ---------------------------------------------------------------------------


def test_environment_reasons_empty_for_exact_required_environment() -> None:
    assert runner.environment_reasons(runner.REQUIRED_ENVIRONMENT_VARIABLES) == ()
    runner.assert_offline_environment(runner.REQUIRED_ENVIRONMENT_VARIABLES)  # no raise


def test_environment_reasons_flags_every_missing_or_wrong_variable() -> None:
    reasons = runner.environment_reasons({})
    assert "environment_variable_pythonhashseed_mismatch" in reasons
    assert "environment_variable_hf_hub_offline_mismatch" in reasons
    assert "environment_variable_transformers_offline_mismatch" in reasons
    assert (
        "environment_variable_vllm_kv_events_use_int_block_hashes_mismatch" in reasons
    )


def test_assert_offline_environment_refuses_partial_environment() -> None:
    partial = dict(runner.REQUIRED_ENVIRONMENT_VARIABLES)
    del partial["HF_HUB_OFFLINE"]
    with pytest.raises(runner.KVTruthProtocolError, match="hf_hub_offline"):
        runner.assert_offline_environment(partial)


def test_build_llm_refuses_non_attestation_input() -> None:
    with pytest.raises(runner.KVTruthProtocolError, match="RuntimeAttestation"):
        runner.build_llm(
            None,  # type: ignore[arg-type]
            model_path="/verified/models/qwen3-8b",
            environ=runner.REQUIRED_ENVIRONMENT_VARIABLES,
        )


def test_build_llm_refuses_without_offline_environment() -> None:
    attestation = _valid_attestation()
    with pytest.raises(runner.KVTruthProtocolError, match="offline"):
        runner.build_llm(
            attestation, model_path="/verified/models/qwen3-8b", environ={}
        )


def test_build_llm_refuses_cleanly_without_vllm_installed() -> None:
    attestation = _valid_attestation()
    with pytest.raises(runner.KVTruthProtocolError, match="vllm is not importable"):
        runner.build_llm(
            attestation,
            model_path="/verified/models/qwen3-8b",
            environ=runner.REQUIRED_ENVIRONMENT_VARIABLES,
        )


def test_workload_manifest_is_deterministic_and_matches_workload() -> None:
    first = runner.build_workload_manifest()
    second = runner.build_workload_manifest()
    assert first.digest == second.digest
    assert first.protocol_id == runner.PROTOCOL_ID
    assert first.block_size == BLOCK_SIZE
    assert len(first.probes) == len(NESTED_PROBES)
    assert first.eviction_lane_request_lengths == (257,) * 7


def test_request_ids_are_unique_and_sequential() -> None:
    ids = runner.request_ids()
    assert len(ids) == len(NESTED_PROBES)
    assert len(set(ids)) == len(ids)
    assert ids[0].endswith("cold_seed")


def test_eviction_lane_request_ids_are_unique() -> None:
    ids = runner.eviction_lane_request_ids()
    assert len(ids) == 7
    assert len(set(ids)) == 7


def test_no_warmup_canary_reasons_is_empty_for_the_fixed_protocol() -> None:
    assert runner.no_warmup_canary_reasons() == ()


# ---------------------------------------------------------------------------
# Fake-driven end-to-end lane execution
# ---------------------------------------------------------------------------


class _FakeOutput:
    request_id: str
    prompt_token_ids: Sequence[int]
    output_token_ids: Sequence[int]
    num_cached_tokens: int | None
    num_cache_creation_tokens: int | None
    finished: bool
    finish_reason: str | None
    timing: runner.RequestTiming

    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        *,
        num_cached_tokens: int | None = 0,
        finish_reason: str | None = "length",
    ) -> None:
        self.request_id = request_id
        self.prompt_token_ids = prompt_token_ids
        self.output_token_ids = [1, 2]
        self.num_cached_tokens = num_cached_tokens
        self.num_cache_creation_tokens = (
            max(len(prompt_token_ids) - num_cached_tokens, 0)
            if num_cached_tokens is not None
            else None
        )
        self.finished = True
        self.finish_reason = finish_reason
        self.timing = runner.RequestTiming(
            queued_ts=1.0,
            scheduled_ts=1.1,
            first_token_ts=1.2,
            last_token_ts=1.3,
            first_token_latency=0.2,
            finished_request_stats=None,
            null_reasons=("finished_request_stats_unavailable",),
        )


class _FakeEngine:
    def __init__(self, *, reset_ok: bool = True, cache_disabled: bool = False) -> None:
        self.reset_calls = 0
        self.generate_calls: list[str] = []
        self._reset_ok = reset_ok
        self._cache_disabled = cache_disabled

    def reset_prefix_cache(self) -> bool:
        self.reset_calls += 1
        return self._reset_ok

    def generate_one(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        sampling_params: Mapping[str, Any],
    ) -> _FakeOutput:
        self.generate_calls.append(request_id)
        num_cached_tokens = None if self._cache_disabled else 0
        return _FakeOutput(
            request_id, list(prompt_token_ids), num_cached_tokens=num_cached_tokens
        )


def _batch(sequence: int, request_token_ids: Sequence[int] = tuple(range(16))) -> Any:
    token_ids = list(request_token_ids[:16])
    block_hashes = compute_sha256_cbor_block_hashes(
        token_ids=token_ids,
        block_size=16,
        parent_block_hash=None,
        extra_keys=None,
    )
    return parse_live_kv_event_batch(
        runner.KV_EVENTS_TOPIC.encode(),
        sequence.to_bytes(8, "big"),
        [
            float(sequence),
            [
                {
                    "type": "BlockStored",
                    "block_hashes": [bytes.fromhex(block_hashes[0])],
                    "parent_block_hash": None,
                    "token_ids": token_ids,
                    "block_size": 16,
                    "lora_id": None,
                    "medium": "GPU",
                    "lora_name": None,
                    "extra_keys": None,
                    "group_idx": 0,
                    "kv_cache_spec_kind": "full_attention",
                    "kv_cache_spec_sliding_window": None,
                    "locality": "LOCAL",
                }
            ],
            0,
        ],
    )


def _reset_batch(sequence: int = 0) -> Any:
    return parse_live_kv_event_batch(
        runner.KV_EVENTS_TOPIC.encode(),
        sequence.to_bytes(8, "big"),
        [float(sequence), [{"type": "AllBlocksCleared"}], 0],
    )


# ---------------------------------------------------------------------------
# _view_of: exact RequestOutput adaptation (outputs[0].token_ids/
# finish_reason, nullable cached/created counts, per-request timing).
# ---------------------------------------------------------------------------


def _real_shaped_output(
    *,
    num_cached_tokens: int | None,
    num_cache_creation_tokens: int | None = None,
    prompt_token_ids: list[int] | None = None,
    metrics: Any = "unset",
) -> Any:
    """Build a minimal object shaped exactly like the real vLLM 0.28.0
    ``RequestOutput``: ``outputs[0].token_ids``/``.finish_reason`` (never a
    top-level ``output_token_ids``/``finish_reason``), and a top-level,
    possibly-``None``, ``num_cached_tokens``."""

    completion = type(
        "Completion",
        (),
        {"token_ids": [11, 12, 13], "finished": True, "finish_reason": "stop"},
    )()
    fields = {
        "request_id": "req-1",
        "prompt_token_ids": prompt_token_ids or [1, 2, 3, 4, 5],
        "num_cached_tokens": num_cached_tokens,
        "num_cache_creation_tokens": num_cache_creation_tokens,
        "outputs": [completion],
        "finished": True,
    }
    if metrics != "unset":
        fields["metrics"] = metrics
    return type("RequestOutput", (), fields)()


def test_view_of_reads_nested_completion_token_ids_and_finish_reason() -> None:
    output = _real_shaped_output(num_cached_tokens=2)
    view = runner._view_of(output)
    assert view.output_token_ids == (11, 12, 13)
    assert view.finish_reason == "stop"


def test_view_of_reads_engine_reported_creation_tokens() -> None:
    output = _real_shaped_output(
        num_cached_tokens=2,
        num_cache_creation_tokens=7,
        prompt_token_ids=[1, 2, 3, 4, 5],
    )
    view = runner._view_of(output)
    assert view.num_cached_tokens == 2
    assert view.num_cache_creation_tokens == 7


def test_view_of_treats_none_cached_tokens_as_a_legitimate_non_fatal_outcome() -> None:
    """A cache-disabled engine (this protocol's A lane) genuinely reports
    ``num_cached_tokens=None``; this must be captured as-is, never raised
    on."""

    output = _real_shaped_output(num_cached_tokens=None)
    view = runner._view_of(output)
    assert view.num_cached_tokens is None
    assert view.num_cache_creation_tokens is None


def test_view_of_refuses_more_than_one_completion() -> None:
    output = _real_shaped_output(num_cached_tokens=0)
    output.outputs = [output.outputs[0], output.outputs[0]]
    with pytest.raises(runner.KVTruthProtocolError, match="exactly one"):
        runner._view_of(output)


def test_view_of_refuses_zero_completions() -> None:
    output = _real_shaped_output(num_cached_tokens=0)
    output.outputs = []
    with pytest.raises(runner.KVTruthProtocolError, match="exactly one"):
        runner._view_of(output)


def test_view_of_extracts_full_timing_when_metrics_present() -> None:
    metrics = type(
        "Metrics",
        (),
        {
            "queued_ts": 10.0,
            "scheduled_ts": 10.5,
            "first_token_ts": 11.0,
            "last_token_ts": 12.0,
            "first_token_latency": 1.0,
        },
    )()
    output = _real_shaped_output(num_cached_tokens=0, metrics=metrics)
    view = runner._view_of(output)
    assert view.timing.queued_ts == 10.0
    assert view.timing.scheduled_ts == 10.5
    assert view.timing.first_token_ts == 11.0
    assert view.timing.last_token_ts == 12.0
    assert view.timing.first_token_latency == 1.0
    assert "finished_request_stats_unavailable" in view.timing.null_reasons


def test_view_of_records_explicit_null_reason_when_metrics_is_absent() -> None:
    output = _real_shaped_output(num_cached_tokens=0, metrics=None)
    view = runner._view_of(output)
    assert view.timing.queued_ts is None
    assert view.timing.first_token_latency is None
    assert "metrics_unavailable" in view.timing.null_reasons


def test_view_of_records_per_field_null_reason_for_a_missing_metrics_field() -> None:
    metrics = type("Metrics", (), {"queued_ts": 1.0})()
    output = _real_shaped_output(num_cached_tokens=0, metrics=metrics)
    view = runner._view_of(output)
    assert view.timing.queued_ts == 1.0
    assert view.timing.scheduled_ts is None
    assert "metrics_scheduled_ts_unavailable" in view.timing.null_reasons


def test_view_of_records_malformed_reason_for_a_non_numeric_metrics_field() -> None:
    metrics = type("Metrics", (), {"queued_ts": "not-a-number"})()
    output = _real_shaped_output(num_cached_tokens=0, metrics=metrics)
    view = runner._view_of(output)
    assert view.timing.queued_ts is None
    assert "metrics_queued_ts_malformed" in view.timing.null_reasons


def test_extract_finished_request_stats_reads_duration_and_time_fields() -> None:
    stats = type(
        "FinishedRequestStats",
        (),
        {"e2e_latency_time": 3.5, "inference_duration": 2.0, "_private": 99.0},
    )()
    metrics = type("Metrics", (), {"finished_request_stats": stats})()
    output = _real_shaped_output(num_cached_tokens=0, metrics=metrics)
    view = runner._view_of(output)
    assert view.timing.finished_request_stats == {
        "e2e_latency_time": 3.5,
        "inference_duration": 2.0,
    }
    assert "finished_request_stats_unavailable" not in view.timing.null_reasons


def test_request_record_to_dict_carries_private_token_arrays_and_timing() -> None:
    engine = _FakeEngine(cache_disabled=True)
    record = runner.run_request(
        engine,
        None,
        request_id="req-1",
        scenario="cold",
        prompt_token_ids=[1, 2, 3],
        expect_cache_counts=False,
    )
    payload = record.to_dict()
    assert payload["prompt_token_ids"] == [1, 2, 3]
    assert payload["output_token_ids"] == [1, 2]
    assert payload["timing"]["queued_ts"] == 1.0
    assert payload["num_cached_tokens"] is None


def test_run_request_flags_unexpectedly_present_cached_tokens_on_a_disabled_engine() -> (
    None
):
    """If a supposedly cache-disabled A-lane engine nonetheless reports a
    real integer ``num_cached_tokens``, that is itself a protocol
    violation worth surfacing, not silently accepted."""

    engine = _FakeEngine(cache_disabled=False)
    record = runner.run_request(
        engine,
        None,
        request_id="req-1",
        scenario="cold",
        prompt_token_ids=[1, 2, 3],
        expect_cache_counts=False,
    )
    assert record.boundary_valid is False
    assert "cached_tokens_unexpectedly_present" in record.boundary_reasons


def test_run_request_flags_missing_cached_tokens_on_a_cache_enabled_engine() -> None:
    engine = _FakeEngine(cache_disabled=True)
    record = runner.run_request(
        engine,
        None,
        request_id="req-1",
        scenario="cold",
        prompt_token_ids=[1, 2, 3],
        expect_cache_counts=True,
    )
    assert record.boundary_valid is False
    assert "cached_tokens_unavailable" in record.boundary_reasons


class _FakeCapture:
    """Delivers pre-programmed batches once per request, then goes quiet."""

    def __init__(
        self,
        batches_per_request: list[list[Any]] | None = None,
        *,
        reset_batches: tuple[Any, ...] | None = None,
        late_batches: tuple[Any, ...] = (),
    ) -> None:
        self.started = False
        self.stopped = False
        self._queue = list(batches_per_request or [])
        self._quiet_remaining = 0
        self._reset_batches = (
            (_reset_batch(),) if reset_batches is None else reset_batches
        )
        self._late_batches = late_batches

    def start(self) -> None:
        self.started = True

    def replay_from_start(self) -> None:
        if not self.started:
            raise AssertionError("replay requested before capture start")

    def drain_before_dispatch(self) -> tuple[Any, ...]:
        drained = self._late_batches
        self._late_batches = ()
        return drained

    def drain_reset(self) -> tuple[Any, ...]:
        return self._reset_batches

    def drain(self) -> tuple[Any, ...]:
        if self._quiet_remaining:
            self._quiet_remaining -= 1
            return ()
        if self._queue:
            self._quiet_remaining = 2
            return tuple(self._queue.pop(0))
        return ()

    def stop(self) -> tuple[int | None, int | None]:
        self.stopped = True
        return (0, 0)


def test_run_request_binds_events_and_reports_boundary_valid() -> None:
    engine = _FakeEngine()
    capture = _FakeCapture(batches_per_request=[[_batch(0)]])
    record = runner.run_request(
        engine,
        capture,
        request_id="r-0",
        scenario="cold",
        prompt_token_ids=[1, 2, 3],
        sleep=lambda _seconds: None,
    )
    assert record.boundary_valid is True
    assert record.event_batches == (_batch(0),)
    assert engine.generate_calls == ["r-0"]


def test_run_request_flags_missing_events_as_a_boundary_defect() -> None:
    engine = _FakeEngine()
    capture = _FakeCapture(batches_per_request=[])
    record = runner.run_request(
        engine,
        capture,
        request_id="r-0",
        scenario="cold",
        prompt_token_ids=[1, 2, 3],
        sleep=lambda _seconds: None,
    )
    assert record.boundary_valid is False
    assert "no_kv_events_observed_for_request" in record.boundary_reasons


def test_run_request_flags_late_events_before_dispatch() -> None:
    engine = _FakeEngine()
    capture = _FakeCapture(
        batches_per_request=[[_batch(1)]],
        late_batches=(_batch(0),),
    )
    record = runner.run_request(
        engine,
        capture,
        request_id="r-0",
        scenario="cold",
        prompt_token_ids=[1, 2, 3],
        sleep=lambda _seconds: None,
    )
    assert record.boundary_valid is False
    assert "late_kv_events_before_request_dispatch" in record.boundary_reasons
    assert [batch.sequence for batch in record.event_batches] == [0, 1]


def test_run_request_without_capture_has_no_events_expected() -> None:
    engine = _FakeEngine()
    record = runner.run_request(
        engine,
        None,
        request_id="r-0",
        scenario="cold",
        prompt_token_ids=[1, 2, 3],
    )
    assert record.boundary_valid is True
    assert record.event_batches == ()


def test_run_request_rejects_ambiguous_request_binding() -> None:
    class _WrongIdEngine(_FakeEngine):
        def generate_one(
            self,
            request_id: str,
            prompt_token_ids: Sequence[int],
            sampling_params: Mapping[str, Any],
        ) -> _FakeOutput:
            return _FakeOutput("not-the-request-id", list(prompt_token_ids))

    with pytest.raises(runner.KVTruthProtocolError, match="ambiguous"):
        runner.run_request(
            _WrongIdEngine(),
            None,
            request_id="r-0",
            scenario="cold",
            prompt_token_ids=[1, 2, 3],
        )


def test_run_b_lane_resets_cache_and_runs_every_probe_in_order() -> None:
    engine = _FakeEngine()
    capture = _FakeCapture(
        batches_per_request=[
            [_batch(i, probe.request_tokens)] for i, probe in enumerate(NESTED_PROBES)
        ]
    )
    result = runner.run_b_lane(engine, capture)
    assert engine.reset_calls == 1
    assert result.reset_boundary_reasons == ()
    assert result.reset_event_batches == (_reset_batch(),)
    assert len(result.records) == len(NESTED_PROBES)
    assert [r.request_id for r in result.records] == list(runner.request_ids())


def test_run_b_lane_refuses_when_reset_prefix_cache_fails() -> None:
    engine = _FakeEngine(reset_ok=False)
    capture = _FakeCapture()
    with pytest.raises(runner.KVTruthProtocolError, match="reset_prefix_cache"):
        runner.run_b_lane(engine, capture)


def test_run_b_lane_marks_missing_reset_event_invalid() -> None:
    engine = _FakeEngine()
    capture = _FakeCapture(
        batches_per_request=[[_batch(i)] for i in range(len(NESTED_PROBES))],
        reset_batches=(),
    )
    result = runner.run_b_lane(engine, capture)
    assert result.all_boundaries_valid is False
    assert result.reset_boundary_reasons == ("reset_all_blocks_cleared_event_missing",)


def test_run_a_lane_runs_every_probe_without_event_capture() -> None:
    engine = _FakeEngine(cache_disabled=True)
    result = runner.run_a_lane(engine)
    assert engine.reset_calls == 0
    assert len(result.records) == len(NESTED_PROBES)
    assert all(record.event_batches == () for record in result.records)
    assert result.all_boundaries_valid is True


def test_run_eviction_lane_resets_and_runs_seed_five_fillers_and_final_probe() -> None:
    engine = _FakeEngine()
    capture = _FakeCapture(batches_per_request=[[_batch(i)] for i in range(20)])
    result = runner.run_eviction_lane(engine, capture)
    assert engine.reset_calls == 1
    assert len(result.records) == len(EVICTION_LANE_REQUESTS)
    assert [r.request_id for r in result.records] == list(
        runner.eviction_lane_request_ids()
    )


def test_protocol_receipt_round_trips_through_disk(tmp_path: Path) -> None:
    engine = _FakeEngine(cache_disabled=True)
    result = runner.run_a_lane(engine)
    receipt = runner.build_protocol_receipt(result)
    output = tmp_path / "receipt.json"
    runner.write_protocol_receipt(receipt, output)
    reloaded = runner.verify_protocol_receipt(output)
    assert reloaded.digest == receipt.digest
    assert reloaded.lane == "A"


def test_write_protocol_receipt_refuses_to_overwrite(tmp_path: Path) -> None:
    result = runner.run_a_lane(_FakeEngine(cache_disabled=True))
    receipt = runner.build_protocol_receipt(result)
    output = tmp_path / "receipt.json"
    runner.write_protocol_receipt(receipt, output)
    with pytest.raises(runner.KVTruthProtocolError, match="overwrite"):
        runner.write_protocol_receipt(receipt, output)


def test_verify_protocol_receipt_rejects_tampered_contents(tmp_path: Path) -> None:
    result = runner.run_a_lane(_FakeEngine(cache_disabled=True))
    receipt = runner.build_protocol_receipt(result)
    output = tmp_path / "receipt.json"
    runner.write_protocol_receipt(receipt, output)
    tampered = output.read_text(encoding="utf-8").replace('"lane":"A"', '"lane":"B"')
    output.write_text(tampered, encoding="utf-8")
    with pytest.raises(runner.KVTruthProtocolError, match="digest"):
        runner.verify_protocol_receipt(output)


def test_verify_protocol_receipt_rejects_resealed_unknown_key(tmp_path: Path) -> None:
    result = runner.run_a_lane(_FakeEngine(cache_disabled=True))
    output = tmp_path / "receipt.json"
    raw = runner.build_protocol_receipt(result).to_dict()
    raw["unexpected"] = True
    raw["digest"] = runner.sha256_digest(
        {key: value for key, value in raw.items() if key != "digest"}
    )
    output.write_text(runner.canonical_json(raw), encoding="utf-8")
    with pytest.raises(runner.KVTruthProtocolError, match="exact schema"):
        runner.verify_protocol_receipt(output)


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------


def test_build_parser_parses_required_arguments() -> None:
    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "--lane",
            "B",
            "--model-path",
            "/verified/models/qwen3-8b",
            "--attestation",
            "attestation.json",
            "--output",
            "receipt.json",
        ]
    )
    assert args.lane == "B"
    assert args.model_path == "/verified/models/qwen3-8b"


def test_run_dispatches_a_lane_without_requiring_capture(tmp_path: Path) -> None:
    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "--lane",
            "A",
            "--model-path",
            "/verified/models/qwen3-8b",
            "--attestation",
            "attestation.json",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    receipt = runner.run(
        args,
        build_attestation=lambda _path, _model_path: _valid_attestation(),
        make_engine=lambda _attestation, _model_path, _cache_enabled: _FakeEngine(
            cache_disabled=True
        ),
    )
    assert receipt.lane == "A"
    assert (tmp_path / "receipt.json").exists()


def test_run_b_lane_without_capture_factory_refuses() -> None:
    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "--lane",
            "B",
            "--model-path",
            "/verified/models/qwen3-8b",
            "--attestation",
            "attestation.json",
            "--output",
            "receipt.json",
        ]
    )
    with pytest.raises(runner.KVTruthProtocolError, match="event capture"):
        runner.run(
            args,
            build_attestation=lambda _path, _model_path: _valid_attestation(),
            make_engine=lambda _attestation, _model_path, _cache_enabled: _FakeEngine(),
        )


def test_run_dispatches_b_lane_with_capture(tmp_path: Path) -> None:
    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "--lane",
            "B",
            "--model-path",
            "/verified/models/qwen3-8b",
            "--attestation",
            "attestation.json",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    capture = _FakeCapture(batches_per_request=[[_batch(i)] for i in range(20)])
    receipt = runner.run(
        args,
        build_attestation=lambda _path, _model_path: _valid_attestation(),
        make_engine=lambda _attestation, _model_path, _cache_enabled: _FakeEngine(),
        make_capture=lambda: capture,
    )
    assert receipt.lane == "B"
    assert capture.started is True
    assert capture.stopped is True


@pytest.mark.parametrize("lane", ["B", "eviction"])
def test_run_binds_capture_before_constructing_the_engine(
    tmp_path: Path, lane: str
) -> None:
    """The pinned vLLM 0.28.0 KV-events publisher *connects out* to its
    configured loopback endpoint rather than binding it, so the real
    subscriber's SUB socket must already be bound (via ``capture.start()``)
    before the engine -- and therefore the publisher -- is constructed, or
    every event the publisher emits during/just after construction is lost
    with nothing listening to receive it. This proves ``run()`` genuinely
    calls ``capture.start()`` strictly before ``make_engine`` for both
    cache-enabled lanes, not merely that both happen to run somewhere.
    """

    parser = runner.build_parser()
    args = parser.parse_args(
        [
            "--lane",
            lane,
            "--model-path",
            "/verified/models/qwen3-8b",
            "--attestation",
            "attestation.json",
            "--output",
            str(tmp_path / "receipt.json"),
        ]
    )
    call_order: list[str] = []
    capture = _FakeCapture(batches_per_request=[[_batch(i)] for i in range(20)])
    original_start = capture.start

    def recording_start() -> None:
        call_order.append("capture.start")
        original_start()

    capture.start = recording_start  # type: ignore[method-assign]

    def fake_make_engine(
        _attestation: Any, _model_path: str, _cache_enabled: bool
    ) -> _FakeEngine:
        call_order.append("make_engine")
        return _FakeEngine(cache_disabled=not _cache_enabled)

    runner.run(
        args,
        build_attestation=lambda _path, _model_path: _valid_attestation(),
        make_engine=fake_make_engine,
        make_capture=lambda: capture,
    )
    assert call_order == ["capture.start", "make_engine"]
    assert capture.started is True
    assert capture.stopped is True


def test_main_refuses_cleanly_when_vllm_is_not_installed(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``main()`` fails only on a genuine, real validation error -- a missing
    ``vllm`` package in this test/dev environment -- never by construction:
    it actually attempts real identity-receipt generation first, and that
    generation's own real ``vllm`` import is what fails closed here.
    """

    exit_code = runner.main(
        [
            "--lane",
            "A",
            "--model-path",
            "/verified/models/qwen3-8b",
            "--attestation",
            "attestation.json",
            "--output",
            "receipt.json",
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "refused" in captured.err
    assert "vllm is not importable" in captured.err


def test_main_reaches_real_execution_for_the_a_lane_via_monkeypatched_factories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Proves ``main()`` genuinely reaches and completes real dispatch end to
    end -- it is not an always-refuse placeholder. Only the actual
    system-boundary factories this module's own real ``main()`` code calls
    (identity/attestation generation, the installed-vllm-package
    discovery, and engine construction) are monkeypatched; CLI argument
    parsing, ``run()`` dispatch, and lane execution all run for real,
    unmodified, and the "A" lane genuinely receives ``cache_enabled=False``.
    """

    attestation = _valid_attestation()
    build_llm_calls: list[dict[str, Any]] = []

    def fake_discover_vllm_package_root() -> Path:
        return tmp_path / "fake-vllm-package"

    def fake_generate_identity_receipt(**kwargs: Any) -> Any:
        assert kwargs["model_path"] == "/verified/models/qwen3-8b"
        assert kwargs["vllm_package_root"] == tmp_path / "fake-vllm-package"
        return attestation.identity

    def fake_generate_runtime_attestation(identity: Any, **_kwargs: Any) -> Any:
        assert identity is attestation.identity
        return attestation

    def fake_build_llm(
        given_attestation: Any, *, model_path: str, cache_enabled: bool
    ) -> Any:
        assert given_attestation is attestation
        build_llm_calls.append(
            {"model_path": model_path, "cache_enabled": cache_enabled}
        )
        return _FakeEngine(cache_disabled=not cache_enabled)

    monkeypatch.setattr(
        runner, "_discover_vllm_package_root", fake_discover_vllm_package_root
    )
    monkeypatch.setattr(
        runner, "generate_identity_receipt", fake_generate_identity_receipt
    )
    monkeypatch.setattr(
        runner, "generate_runtime_attestation", fake_generate_runtime_attestation
    )
    monkeypatch.setattr(runner, "build_llm", fake_build_llm)

    output = tmp_path / "receipt.json"
    attestation_path = tmp_path / "attestation.json"
    exit_code = runner.main(
        [
            "--lane",
            "A",
            "--model-path",
            "/verified/models/qwen3-8b",
            "--attestation",
            str(attestation_path),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    assert build_llm_calls == [
        {"model_path": "/verified/models/qwen3-8b", "cache_enabled": False}
    ]
    assert attestation_path.exists()
    assert json.loads(attestation_path.read_text(encoding="utf-8")) == (
        attestation.to_dict()
    )
    reloaded = runner.verify_protocol_receipt(output)
    assert reloaded.lane == "A"
    assert reloaded.runtime_attestation == attestation.to_dict()


def test_main_reaches_real_execution_for_the_b_lane_via_monkeypatched_factories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same as the A-lane proof above, but for the "B" lane: proves
    ``main()`` also genuinely wires and reaches a real KV-event capture for
    the cache-enabled lanes, receiving ``cache_enabled=True``.
    """

    attestation = _valid_attestation()
    capture = _FakeCapture(
        batches_per_request=[
            [_batch(i, probe.request_tokens)] for i, probe in enumerate(NESTED_PROBES)
        ]
    )
    build_llm_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        runner, "_discover_vllm_package_root", lambda: tmp_path / "fake-vllm"
    )
    monkeypatch.setattr(
        runner, "generate_identity_receipt", lambda **_kwargs: attestation.identity
    )
    monkeypatch.setattr(
        runner,
        "generate_runtime_attestation",
        lambda _identity, **_kwargs: attestation,
    )

    def fake_build_llm(
        _attestation: Any, *, model_path: str, cache_enabled: bool
    ) -> Any:
        build_llm_calls.append(
            {"model_path": model_path, "cache_enabled": cache_enabled}
        )
        return _FakeEngine()

    monkeypatch.setattr(runner, "build_llm", fake_build_llm)
    monkeypatch.setattr(runner, "LiveKVEventSubscriber", lambda **_kwargs: capture)

    output = tmp_path / "receipt.json"
    exit_code = runner.main(
        [
            "--lane",
            "B",
            "--model-path",
            "/verified/models/qwen3-8b",
            "--attestation",
            str(tmp_path / "attestation.json"),
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    assert build_llm_calls == [
        {"model_path": "/verified/models/qwen3-8b", "cache_enabled": True}
    ]
    assert capture.started is True
    assert capture.stopped is True
    reloaded = runner.verify_protocol_receipt(output)
    assert reloaded.lane == "B"


# ---------------------------------------------------------------------------
# Real system-fact collection, identity/attestation generation, real-engine
# construction, and live KV-event subscription -- exercising the genuine
# (non-placeholder) implementations added to fix the always-refuse defect.
# ---------------------------------------------------------------------------


def test_expected_repository_commit_prefers_env_var_over_hardcoded_fallback() -> None:
    assert (
        runner.expected_repository_commit({}) == runner.PLAN_APPROVED_REPOSITORY_COMMIT
    )
    assert (
        runner.expected_repository_commit({"EXPECTED_REPOSITORY_COMMIT": "deadbeef"})
        == "deadbeef"
    )


def test_expected_image_id_returns_none_when_unset_with_no_historical_fallback() -> (
    None
):
    """Unlike ``expected_repository_commit``, a missing ``EXPECTED_IMAGE_ID``
    must never silently fall back to any preregistered/historical image id:
    the derived overlay image is built from whichever repository HEAD was
    actually staged for this run, so there is no safe constant to fall back
    to."""

    assert runner.expected_image_id({}) is None


@pytest.mark.parametrize(
    "malformed",
    [
        "not-a-digest",
        "sha256:" + "a" * 63,  # too short
        "sha256:" + "a" * 65,  # too long
        "sha256:" + "G" * 64,  # not lowercase hex
        "SHA256:" + "a" * 64,  # wrong case prefix
        "a" * 64,  # missing the "sha256:" prefix
    ],
)
def test_expected_image_id_returns_none_for_every_malformed_shape(
    malformed: str,
) -> None:
    assert runner.expected_image_id({"EXPECTED_IMAGE_ID": malformed}) is None


def test_expected_image_id_returns_the_real_env_value_when_well_formed() -> None:
    real_looking = "sha256:" + "9" * 64
    assert runner.expected_image_id({"EXPECTED_IMAGE_ID": real_looking}) == real_looking


def test_protocol_reasons_flags_missing_expected_image_id_env() -> None:
    attestation = _valid_attestation()
    reasons = runner.protocol_reasons_for_attestation(attestation, environ={})
    assert "expected_image_id_env_missing_or_malformed" in reasons


def test_protocol_reasons_flags_a_real_derived_image_id_mismatch() -> None:
    attestation = _valid_attestation()
    reasons = runner.protocol_reasons_for_attestation(
        attestation, environ={"EXPECTED_IMAGE_ID": "sha256:" + "8" * 64}
    )
    assert "derived_image_id_mismatch" in reasons


def test_protocol_reasons_accepts_a_matching_expected_image_id() -> None:
    attestation = _valid_attestation()
    reasons = runner.protocol_reasons_for_attestation(
        attestation, environ={"EXPECTED_IMAGE_ID": _TEST_EXPECTED_IMAGE_ID}
    )
    assert "derived_image_id_mismatch" not in reasons
    assert "expected_image_id_env_missing_or_malformed" not in reasons


def _nvidia_smi_line(
    *,
    name: str = runner.EXPECTED_GPU_NAME,
    driver: str = runner.EXPECTED_DRIVER,
    memory: str = str(runner.EXPECTED_MEMORY_MIB),
    uuid: str = "GPU-real-uuid",
    compute_cap: str = "8.9",
) -> str:
    return f"{name}, {driver}, {memory}, {uuid}, {compute_cap}\n"


def test_collect_gpu_facts_succeeds_with_exactly_one_gpu() -> None:
    facts = runner.collect_gpu_facts(run_command=lambda _argv: _nvidia_smi_line())
    assert facts["gpu_name"] == runner.EXPECTED_GPU_NAME
    assert facts["gpu_memory_mib"] == runner.EXPECTED_MEMORY_MIB
    assert facts["gpu_uuid_commitment"] == hashlib.sha256(b"GPU-real-uuid").hexdigest()


def test_collect_gpu_facts_refuses_when_not_exactly_one_gpu() -> None:
    with pytest.raises(runner.KVTruthProtocolError, match="exactly one visible GPU"):
        runner.collect_gpu_facts(run_command=lambda _argv: "")
    with pytest.raises(runner.KVTruthProtocolError, match="exactly one visible GPU"):
        runner.collect_gpu_facts(
            run_command=lambda _argv: _nvidia_smi_line() + _nvidia_smi_line()
        )


def test_collect_gpu_facts_refuses_incomplete_line() -> None:
    with pytest.raises(runner.KVTruthProtocolError, match="complete GPU identity"):
        runner.collect_gpu_facts(run_command=lambda _argv: "only, three, fields\n")


def test_collect_gpu_facts_refuses_empty_uuid() -> None:
    with pytest.raises(runner.KVTruthProtocolError, match="empty GPU UUID"):
        runner.collect_gpu_facts(run_command=lambda _argv: _nvidia_smi_line(uuid=""))


def test_collect_source_file_digests_hashes_every_required_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the actually-hashed bytes agree with the (here, fixture-supplied)
    expected digests, ``matches_manifest`` reflects a genuine byte-content
    comparison, not a hardcoded ``True``."""

    expected_digests: dict[str, str] = {}
    for relative in required_source_file_paths():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        content = relative.encode("utf-8")
        target.write_bytes(content)
        expected_digests[relative] = hashlib.sha256(content).hexdigest()
    monkeypatch.setattr(
        runner, "required_source_file_digests", lambda: expected_digests
    )
    digests = runner.collect_source_file_digests(tmp_path)
    assert {digest.path for digest in digests} == set(required_source_file_paths())
    assert all(digest.matches_manifest for digest in digests)


def test_collect_source_file_digests_flags_a_real_mismatch_against_the_manifest(
    tmp_path: Path,
) -> None:
    """Without a matching expected digest, real installed bytes that differ
    from the committed manifest's genuine expected SHA-256 must be reported
    as non-matching -- this is never hardcoded to ``True``."""

    for relative in required_source_file_paths():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(relative.encode("utf-8"))
    digests = runner.collect_source_file_digests(tmp_path)
    assert {digest.path for digest in digests} == set(required_source_file_paths())
    assert not any(digest.matches_manifest for digest in digests)


def test_collect_source_file_digests_refuses_when_a_required_file_is_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(runner.KVTruthProtocolError, match="missing from the installed"):
        runner.collect_source_file_digests(tmp_path)


def test_collect_source_file_digests_refuses_a_symlinked_source_file(
    tmp_path: Path,
) -> None:
    real_file = tmp_path / "_real_target"
    real_file.write_bytes(b"payload")
    for relative in required_source_file_paths():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            target.unlink()
        target.symlink_to(real_file)
    with pytest.raises(runner.KVTruthProtocolError, match="missing from the installed"):
        runner.collect_source_file_digests(tmp_path)


def _tiny_model_manifest_and_files(
    tmp_path: Path, *, extra_untracked_file: bool = False
) -> tuple[Path, Path, int]:
    model_root = tmp_path / "model"
    model_root.mkdir()
    contents = {"config.json": b"{}", "tokenizer.json": b"tok-bytes"}
    files: list[dict[str, Any]] = []
    total = 0
    for name, content in contents.items():
        path = model_root / name
        path.write_bytes(content)
        files.append(
            {
                "path": name,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
        total += len(content)
    if extra_untracked_file:
        (model_root / "unexpected.bin").write_bytes(b"surprise")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "source": {
                    "official_id": runner.MODEL_ID,
                    "official_revision": runner.MODEL_REVISION,
                    "expected_source_bytes": total,
                    "files": files,
                }
            }
        ),
        encoding="utf-8",
    )
    return model_root, manifest_path, total


def _patch_tiny_model_manifest(
    monkeypatch: pytest.MonkeyPatch, manifest_path: Path, total_bytes: int
) -> None:
    monkeypatch.setattr(runner, "_QWEN3_8B_MODEL_MANIFEST_PATH", manifest_path)
    monkeypatch.setattr(runner, "EXPECTED_MODEL_BYTES", total_bytes)
    monkeypatch.setattr(runner, "EXPECTED_MODEL_FILE_COUNT", 2)


def test_verify_model_inventory_succeeds_against_a_small_monkeypatched_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root, manifest_path, total_bytes = _tiny_model_manifest_and_files(tmp_path)
    _patch_tiny_model_manifest(monkeypatch, manifest_path, total_bytes)
    verified, digest = runner.verify_model_inventory(model_root)
    assert {item["path"] for item in verified} == {"config.json", "tokenizer.json"}
    assert isinstance(digest, str) and digest.startswith("sha256:")


def test_verify_model_inventory_refuses_tampered_file_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root, manifest_path, total_bytes = _tiny_model_manifest_and_files(tmp_path)
    _patch_tiny_model_manifest(monkeypatch, manifest_path, total_bytes)
    # Same byte length as the original "{}" so this exercises the hash
    # check specifically, not the (separately tested) size-mismatch check.
    (model_root / "config.json").write_bytes(b"[]")
    with pytest.raises(runner.KVTruthProtocolError, match="hash mismatch"):
        runner.verify_model_inventory(model_root)


def test_verify_model_inventory_refuses_a_missing_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root, manifest_path, total_bytes = _tiny_model_manifest_and_files(tmp_path)
    _patch_tiny_model_manifest(monkeypatch, manifest_path, total_bytes)
    (model_root / "config.json").unlink()
    with pytest.raises(
        runner.KVTruthProtocolError, match="does not match the committed manifest"
    ):
        runner.verify_model_inventory(model_root)


def test_verify_model_inventory_refuses_an_extra_untracked_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root, manifest_path, total_bytes = _tiny_model_manifest_and_files(
        tmp_path, extra_untracked_file=True
    )
    _patch_tiny_model_manifest(monkeypatch, manifest_path, total_bytes)
    with pytest.raises(
        runner.KVTruthProtocolError, match="does not match the committed manifest"
    ):
        runner.verify_model_inventory(model_root)


def _patch_identity_collection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    model_root, manifest_path, total_bytes = _tiny_model_manifest_and_files(tmp_path)
    _patch_tiny_model_manifest(monkeypatch, manifest_path, total_bytes)
    commit_marker = tmp_path / "RUNNER_COMMIT"
    commit_marker.write_text(
        "1234567890abcdef1234567890abcdef12345678\n", encoding="ascii"
    )
    monkeypatch.setattr(runner, "RUNNER_COMMIT_MARKER_PATH", commit_marker)
    monkeypatch.setattr(
        runner,
        "collect_runtime_versions",
        lambda: {
            "python_version": runner.RUNTIME_PINS["python_version"],
            "vllm_version": "0.28.0",
            "torch_version": runner.RUNTIME_PINS["torch_version"],
            "cuda_runtime_version": runner.RUNTIME_PINS["cuda_version"],
            "transformers_version": runner.RUNTIME_PINS["transformers_version"],
            "typing_extensions_version": runner.RUNTIME_PINS[
                "typing_extensions_version"
            ],
        },
    )
    monkeypatch.setattr(
        runner,
        "collect_gpu_facts",
        lambda **_kwargs: {
            "cuda_driver_version": runner.EXPECTED_DRIVER,
            "gpu_name": runner.EXPECTED_GPU_NAME,
            "gpu_memory_mib": runner.EXPECTED_MEMORY_MIB,
            "gpu_compute_capability": "8.9",
            "gpu_uuid_commitment": _hex("gpu-uuid"),
        },
    )
    monkeypatch.setattr(
        runner,
        "collect_source_file_digests",
        lambda _root: tuple(
            SourceFileDigest(path=path, sha256=digest, matches_manifest=True)
            for path, digest in sorted(required_source_file_digests().items())
        ),
    )
    monkeypatch.setattr(
        runner,
        "collect_installed_distributions_digest",
        lambda: "sha256:" + _hex("dist"),
    )
    monkeypatch.setattr(
        runner,
        "collect_wheel_record_digest",
        lambda **_kwargs: "sha256:" + _hex("wheel"),
    )
    monkeypatch.setattr(
        runner, "collect_package_tree_digest", lambda _root: "sha256:" + _hex("tree")
    )
    return model_root


def test_generate_identity_receipt_requires_expected_repository_commit_env(
    tmp_path: Path,
) -> None:
    with pytest.raises(runner.KVTruthProtocolError, match="EXPECTED_REPOSITORY_COMMIT"):
        runner.generate_identity_receipt(
            model_path=str(tmp_path), vllm_package_root=tmp_path, environ={}
        )


def test_generate_identity_receipt_requires_expected_experiment_nonce_env(
    tmp_path: Path,
) -> None:
    with pytest.raises(runner.KVTruthProtocolError, match="EXPECTED_EXPERIMENT_NONCE"):
        runner.generate_identity_receipt(
            model_path=str(tmp_path),
            vllm_package_root=tmp_path,
            environ={"EXPECTED_REPOSITORY_COMMIT": "abc123"},
        )


def _generate_identity_environ(**overrides: str) -> dict[str, str]:
    environ = {
        "EXPECTED_REPOSITORY_COMMIT": "1234567890abcdef1234567890abcdef12345678",
        "EXPECTED_EXPERIMENT_NONCE": "nonce-under-test",
        "EXPECTED_IMAGE_ID": _TEST_EXPECTED_IMAGE_ID,
    }
    environ.update(overrides)
    return environ


def test_generate_identity_receipt_requires_expected_image_id_env(
    tmp_path: Path,
) -> None:
    with pytest.raises(runner.KVTruthProtocolError, match="EXPECTED_IMAGE_ID"):
        runner.generate_identity_receipt(
            model_path=str(tmp_path),
            vllm_package_root=tmp_path,
            environ={
                "EXPECTED_REPOSITORY_COMMIT": "abc123",
                "EXPECTED_EXPERIMENT_NONCE": "nonce-under-test",
            },
        )


def test_generate_identity_receipt_refuses_a_malformed_expected_image_id_env(
    tmp_path: Path,
) -> None:
    with pytest.raises(runner.KVTruthProtocolError, match="EXPECTED_IMAGE_ID"):
        runner.generate_identity_receipt(
            model_path=str(tmp_path),
            vllm_package_root=tmp_path,
            environ=_generate_identity_environ(EXPECTED_IMAGE_ID="not-a-real-digest"),
        )


def test_generate_identity_receipt_produces_a_self_verifying_real_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = _patch_identity_collection(monkeypatch, tmp_path)
    identity = runner.generate_identity_receipt(
        model_path=str(model_root),
        vllm_package_root=tmp_path,
        environ=_generate_identity_environ(),
    )
    assert identity.repository_commit == "1234567890abcdef1234567890abcdef12345678"
    assert identity.experiment_nonce == "nonce-under-test"
    assert identity.image_id == _TEST_EXPECTED_IMAGE_ID
    reparsed = parse_identity_receipt(identity.to_dict())
    assert reparsed.seal == identity.seal


def test_generate_identity_receipt_refuses_when_model_inventory_is_tampered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = _patch_identity_collection(monkeypatch, tmp_path)
    (model_root / "config.json").write_bytes(b"tampered-bytes")
    with pytest.raises(runner.KVTruthProtocolError):
        runner.generate_identity_receipt(
            model_path=str(model_root),
            vllm_package_root=tmp_path,
            environ=_generate_identity_environ(),
        )


def test_generate_runtime_attestation_is_valid_and_cache_enabled_shaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = _patch_identity_collection(monkeypatch, tmp_path)
    identity = runner.generate_identity_receipt(
        model_path=str(model_root),
        vllm_package_root=tmp_path,
        environ=_generate_identity_environ(),
    )
    attestation = runner.generate_runtime_attestation(identity)
    assert (
        runner.protocol_reasons_for_attestation(
            attestation,
            environ={
                "EXPECTED_REPOSITORY_COMMIT": "1234567890abcdef1234567890abcdef12345678",
                "EXPECTED_IMAGE_ID": _TEST_EXPECTED_IMAGE_ID,
            },
        )
        == ()
    )
    assert attestation.resolved_config.enable_prefix_caching is True
    assert attestation.kv_events_config.first_sequence is None
    assert attestation.kv_events_config.last_sequence is None


def test_assert_engine_matches_protocol_accepts_a_conforming_fake_engine() -> None:
    class _Config:
        def __init__(self, **fields: Any) -> None:
            self.__dict__.update(fields)

    class _FakeVllmConfig:
        model_config = _Config(
            max_model_len=runner.MAX_MODEL_LEN,
            enforce_eager=runner.ENFORCE_EAGER,
            multimodal_config=None,
        )
        cache_config = _Config(
            block_size=BLOCK_SIZE,
            prefix_match_unit=runner.PREFIX_MATCH_UNIT,
            num_gpu_blocks_override=runner.NUM_GPU_BLOCKS_OVERRIDE,
            gpu_memory_utilization=runner.GPU_MEMORY_UTILIZATION,
            cache_dtype=runner.CACHE_DTYPE,
            prefix_caching_hash_algo=runner.PREFIX_CACHING_HASH_ALGO,
            enable_prefix_caching=True,
        )
        scheduler_config = _Config(max_num_seqs=runner.MAX_NUM_SEQS)
        parallel_config = _Config(
            tensor_parallel_size=runner.TENSOR_PARALLEL_SIZE,
            data_parallel_size=runner.DATA_PARALLEL_SIZE,
        )
        kv_events_config = _Config(**runner.kv_events_config_kwargs())
        speculative_config = None
        lora_config = None

    class _FakeEngineForAssertion:
        vllm_config = _FakeVllmConfig()

    runner._assert_engine_matches_protocol(
        _FakeEngineForAssertion(), cache_enabled=True
    )  # must not raise


def test_assert_engine_matches_protocol_refuses_a_cache_flag_mismatch() -> None:
    class _Config:
        def __init__(self, **fields: Any) -> None:
            self.__dict__.update(fields)

    class _FakeVllmConfig:
        model_config = _Config(
            max_model_len=runner.MAX_MODEL_LEN,
            enforce_eager=runner.ENFORCE_EAGER,
            multimodal_config=None,
        )
        cache_config = _Config(
            block_size=BLOCK_SIZE,
            prefix_match_unit=runner.PREFIX_MATCH_UNIT,
            num_gpu_blocks_override=runner.NUM_GPU_BLOCKS_OVERRIDE,
            gpu_memory_utilization=runner.GPU_MEMORY_UTILIZATION,
            cache_dtype=runner.CACHE_DTYPE,
            prefix_caching_hash_algo=runner.PREFIX_CACHING_HASH_ALGO,
            enable_prefix_caching=True,
        )
        scheduler_config = _Config(max_num_seqs=runner.MAX_NUM_SEQS)
        parallel_config = _Config(
            tensor_parallel_size=runner.TENSOR_PARALLEL_SIZE,
            data_parallel_size=runner.DATA_PARALLEL_SIZE,
        )
        kv_events_config = _Config(**runner.kv_events_config_kwargs())
        speculative_config = None
        lora_config = None

    class _FakeEngineForAssertion:
        vllm_config = _FakeVllmConfig()

    with pytest.raises(
        runner.KVTruthProtocolError, match="engine_enable_prefix_caching_mismatch"
    ):
        runner._assert_engine_matches_protocol(
            _FakeEngineForAssertion(), cache_enabled=False
        )


def test_assert_engine_matches_protocol_refuses_a_missing_config_surface() -> None:
    class _EmptyEngine:
        pass

    with pytest.raises(
        runner.KVTruthProtocolError, match="does not expose the expected"
    ):
        runner._assert_engine_matches_protocol(_EmptyEngine(), cache_enabled=True)


class _FakeLLMEngine:
    """Stands in for ``vllm.LLMEngine`` for offline, no-GPU tests."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self._pending: list[tuple[str, list[int]]] = []
        self._finished_ids: set[str] = set()

    @classmethod
    def from_engine_args(cls, engine_args: Any) -> _FakeLLMEngine:
        return cls(**engine_args.kwargs)

    @property
    def vllm_config(self) -> Any:
        class _Config:
            def __init__(self, **fields: Any) -> None:
                self.__dict__.update(fields)

        cache_enabled = bool(self.kwargs.get("enable_prefix_caching"))
        kv_events_config = self.kwargs.get("kv_events_config")
        return _Config(
            model_config=_Config(
                max_model_len=self.kwargs["max_model_len"],
                enforce_eager=self.kwargs["enforce_eager"],
                multimodal_config=None,
            ),
            cache_config=_Config(
                block_size=self.kwargs["block_size"],
                prefix_match_unit=self.kwargs["prefix_match_unit"],
                num_gpu_blocks_override=self.kwargs["num_gpu_blocks_override"],
                gpu_memory_utilization=self.kwargs["gpu_memory_utilization"],
                cache_dtype=self.kwargs["kv_cache_dtype"],
                prefix_caching_hash_algo=self.kwargs["prefix_caching_hash_algo"],
                enable_prefix_caching=cache_enabled,
            ),
            scheduler_config=_Config(max_num_seqs=self.kwargs["max_num_seqs"]),
            parallel_config=_Config(
                tensor_parallel_size=self.kwargs["tensor_parallel_size"],
                data_parallel_size=self.kwargs["data_parallel_size"],
            ),
            kv_events_config=kv_events_config,
            speculative_config=None,
            lora_config=None,
        )

    def reset_prefix_cache(self) -> bool:
        return True

    def add_request(
        self, request_id: str, prompt: dict[str, Any], _params: Any
    ) -> None:
        self._pending.append((request_id, list(prompt["prompt_token_ids"])))

    def has_unfinished_requests(self) -> bool:
        return bool(self._pending)

    def step(self) -> list[Any]:
        request_id, prompt_token_ids = self._pending.pop(0)
        self._finished_ids.add(request_id)
        completion = type(
            "Completion",
            (),
            {"token_ids": [7, 8, 9], "finished": True, "finish_reason": "stop"},
        )()
        output = type(
            "RequestOutput",
            (),
            {
                "request_id": request_id,
                "prompt_token_ids": prompt_token_ids,
                "num_cached_tokens": min(1, len(prompt_token_ids)),
                "outputs": [completion],
                "finished": True,
            },
        )()
        return [output]


class _FakeEngineArgs:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeKVEventsConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.__dict__.update(kwargs)


class _FakeSamplingParams:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _install_fake_vllm_module(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    vllm_module = types.ModuleType("vllm")
    vllm_module.LLMEngine = _FakeLLMEngine  # type: ignore[attr-defined]
    vllm_module.SamplingParams = _FakeSamplingParams  # type: ignore[attr-defined]

    vllm_config_module = types.ModuleType("vllm.config")
    vllm_config_kv_events_module = types.ModuleType("vllm.config.kv_events")
    vllm_config_kv_events_module.KVEventsConfig = _FakeKVEventsConfig  # type: ignore[attr-defined]

    vllm_engine_module = types.ModuleType("vllm.engine")
    vllm_engine_arg_utils_module = types.ModuleType("vllm.engine.arg_utils")
    vllm_engine_arg_utils_module.EngineArgs = _FakeEngineArgs  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.config", vllm_config_module)
    monkeypatch.setitem(
        sys.modules, "vllm.config.kv_events", vllm_config_kv_events_module
    )
    monkeypatch.setitem(sys.modules, "vllm.engine", vllm_engine_module)
    monkeypatch.setitem(
        sys.modules, "vllm.engine.arg_utils", vllm_engine_arg_utils_module
    )


def _offline_environ() -> dict[str, str]:
    return dict(runner.REQUIRED_ENVIRONMENT_VARIABLES)


def test_build_llm_constructs_a_real_llm_engine_end_to_end_with_a_fake_vllm_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vllm_module(monkeypatch)
    attestation = _valid_attestation()

    handle = runner.build_llm(
        attestation,
        model_path="/verified/models/qwen3-8b",
        environ=_offline_environ(),
        cache_enabled=True,
    )
    output = handle.generate_one("req-0", [1, 2, 3], {"temperature": 0.0})
    assert output.request_id == "req-0"
    assert output.output_token_ids == (7, 8, 9)
    assert handle.reset_prefix_cache() is True


def test_build_llm_a_lane_constructs_a_cache_disabled_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vllm_module(monkeypatch)
    attestation = _valid_attestation()

    handle = runner.build_llm(
        attestation,
        model_path="/verified/models/qwen3-8b",
        environ=_offline_environ(),
        cache_enabled=False,
    )
    assert isinstance(handle, runner._LLMEngineHandle)
    # cache_enabled=False must have actually been threaded into the engine's
    # own real resolved kwargs, not merely accepted and ignored.
    inner_engine = handle._engine  # noqa: SLF001 -- white-box test of real wiring
    assert inner_engine.kwargs["enable_prefix_caching"] is False
    assert inner_engine.kwargs["kv_events_config"] is None


def test_build_llm_refuses_when_the_real_engine_resolves_a_mismatched_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vllm_module(monkeypatch)

    class _WrongLLMEngine(_FakeLLMEngine):
        @property
        def vllm_config(self) -> Any:
            config = super().vllm_config
            config.model_config.max_model_len = runner.MAX_MODEL_LEN + 1
            return config

    import sys

    monkeypatch.setattr(sys.modules["vllm"], "LLMEngine", _WrongLLMEngine)
    attestation = _valid_attestation()

    with pytest.raises(
        runner.KVTruthProtocolError, match="engine_max_model_len_mismatch"
    ):
        runner.build_llm(
            attestation,
            model_path="/verified/models/qwen3-8b",
            environ=_offline_environ(),
            cache_enabled=True,
        )


# ---------------------------------------------------------------------------
# LiveKVEventSubscriber: real loopback ZMQ SUB/REQ framing, decoded with a
# fake in-memory zmq/msgspec module so this stays a genuine no-GPU/no-network
# test of the actual wire-decoding control flow.
# ---------------------------------------------------------------------------


class _FakeZmqSocket:
    def __init__(self, kind: int) -> None:
        self.kind = kind
        self.opts: dict[int, Any] = {}
        self.connected_to: str | None = None
        self.bound_to: str | None = None
        self.sent: list[bytes] = []
        self.inbox: list[list[bytes]] = []
        self.closed = False

    def setsockopt(self, option: int, value: Any) -> None:
        self.opts[option] = value

    def connect(self, endpoint: str) -> None:
        self.connected_to = endpoint

    def bind(self, endpoint: str) -> None:
        self.bound_to = endpoint

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)

    def recv_multipart(self) -> list[bytes]:
        if not self.inbox:
            raise _FakeZmqAgain()
        return self.inbox.pop(0)

    def close(self, linger: int = 0) -> None:
        self.closed = True


class _FakeZmqAgain(Exception):
    pass


class _FakeZmqContext:
    _instance: _FakeZmqContext | None = None

    def __init__(self) -> None:
        self.sockets: list[_FakeZmqSocket] = []

    @classmethod
    def instance(cls) -> _FakeZmqContext:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def socket(self, kind: int) -> _FakeZmqSocket:
        sock = _FakeZmqSocket(kind)
        self.sockets.append(sock)
        return sock


def _install_fake_zmq_module(
    monkeypatch: pytest.MonkeyPatch, *, replay_frames: list[list[bytes]] | None = None
) -> tuple[_FakeZmqContext, list[_FakeZmqSocket]]:
    import sys
    import types

    import cbor2

    _FakeZmqContext._instance = None
    context = _FakeZmqContext()

    zmq_module = types.ModuleType("zmq")
    zmq_module.SUB = 1  # type: ignore[attr-defined]
    zmq_module.REQ = 2  # type: ignore[attr-defined]
    zmq_module.RCVTIMEO = 3  # type: ignore[attr-defined]
    zmq_module.SUBSCRIBE = 4  # type: ignore[attr-defined]
    zmq_module.Again = _FakeZmqAgain  # type: ignore[attr-defined]
    zmq_module.Context = _FakeZmqContext  # type: ignore[attr-defined]

    msgspec_module = types.ModuleType("msgspec")
    msgpack_module = types.ModuleType("msgspec.msgpack")
    msgpack_module.decode = lambda data: cbor2.loads(data)  # type: ignore[attr-defined]
    msgspec_module.msgpack = msgpack_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "zmq", zmq_module)
    monkeypatch.setitem(sys.modules, "msgspec", msgspec_module)
    monkeypatch.setitem(sys.modules, "msgspec.msgpack", msgpack_module)

    original_socket = context.socket

    created: list[_FakeZmqSocket] = []

    def socket(kind: int) -> _FakeZmqSocket:
        sock = original_socket(kind)
        created.append(sock)
        if kind == zmq_module.REQ and replay_frames is not None:
            sock.inbox = list(replay_frames)
        return sock

    context.socket = socket  # type: ignore[method-assign]
    monkeypatch.setattr(_FakeZmqContext, "instance", classmethod(lambda cls: context))
    return context, created


def _encode_batch_payload(
    sequence: int, *, data_parallel_rank: int | None = 0
) -> bytes:
    """Encode a stand-in for the real array-like ``EventBatch`` msgpack
    payload (``[ts, events]`` or ``[ts, events, data_parallel_rank]``).
    Pass ``data_parallel_rank=None`` to omit the trailing element entirely,
    exercising the default-rank case a real batch produces when published
    from the default data-parallel rank."""

    import cbor2

    if data_parallel_rank is None:
        return cbor2.dumps([float(sequence), []])
    return cbor2.dumps([float(sequence), [], data_parallel_rank])


def test_live_kv_event_subscriber_refuses_non_loopback_endpoints() -> None:
    with pytest.raises(runner.KVTruthProtocolError, match="loopback-only"):
        runner.LiveKVEventSubscriber(
            endpoint="tcp://0.0.0.0:5557",
            replay_endpoint="tcp://127.0.0.1:5558",
            topic=runner.KV_EVENTS_TOPIC,
        )
    with pytest.raises(runner.KVTruthProtocolError, match="loopback-only"):
        runner.LiveKVEventSubscriber(
            endpoint="tcp://127.0.0.1:5557",
            replay_endpoint="tcp://0.0.0.0:5558",
            topic=runner.KV_EVENTS_TOPIC,
        )


def test_live_kv_event_subscriber_binds_sub_and_connects_req(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pinned vLLM 0.28.0 KV-events publisher itself *connects out* to
    its configured endpoint (it does not bind one), so this SUB socket must
    bind rather than connect -- the opposite of the more common PUB-binds/
    SUB-connects pattern. The replay endpoint is the mirror image: the
    engine's replay ROUTER is publisher-bound, so this REQ connects."""

    _context, created = _install_fake_zmq_module(monkeypatch, replay_frames=[])
    subscriber = runner.LiveKVEventSubscriber(
        endpoint="tcp://127.0.0.1:5557",
        replay_endpoint="tcp://127.0.0.1:5558",
        topic=runner.KV_EVENTS_TOPIC,
    )
    subscriber.start()
    sub_socket, replay_socket = created[0], created[1]
    assert sub_socket.bound_to == "tcp://127.0.0.1:5557"
    assert sub_socket.connected_to is None
    assert replay_socket.connected_to == "tcp://127.0.0.1:5558"
    assert replay_socket.bound_to is None


def test_live_kv_event_subscriber_decodes_replay_then_live_frames_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_frames = [
        [
            runner.KV_EVENTS_TOPIC.encode(),
            (0).to_bytes(8, "big", signed=True),
            _encode_batch_payload(0),
        ],
        [
            runner.KV_EVENTS_TOPIC.encode(),
            (1).to_bytes(8, "big", signed=True),
            _encode_batch_payload(1),
        ],
        [b"", runner.END_OF_REPLAY_SEQUENCE.to_bytes(8, "big", signed=True), b""],
    ]
    _context, created = _install_fake_zmq_module(
        monkeypatch, replay_frames=replay_frames
    )
    subscriber = runner.LiveKVEventSubscriber(
        endpoint="tcp://127.0.0.1:5557",
        replay_endpoint="tcp://127.0.0.1:5558",
        topic=runner.KV_EVENTS_TOPIC,
    )
    subscriber.start()
    subscriber.replay_from_start()
    sub_socket = created[0]
    sub_socket.inbox = [
        [
            runner.KV_EVENTS_TOPIC.encode(),
            (2).to_bytes(8, "big", signed=True),
            _encode_batch_payload(2),
        ]
    ]
    drained = subscriber.drain()
    assert [batch.sequence for batch in drained] == [0, 1, 2]
    first, last = subscriber.stop()
    assert (first, last) == (0, 2)


def test_live_kv_event_subscriber_treats_a_duplicate_sequence_as_a_no_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_frames = [
        [
            runner.KV_EVENTS_TOPIC.encode(),
            (0).to_bytes(8, "big", signed=True),
            _encode_batch_payload(0),
        ],
        [b"", runner.END_OF_REPLAY_SEQUENCE.to_bytes(8, "big", signed=True), b""],
    ]
    _context, created = _install_fake_zmq_module(
        monkeypatch, replay_frames=replay_frames
    )
    subscriber = runner.LiveKVEventSubscriber(
        endpoint="tcp://127.0.0.1:5557",
        replay_endpoint="tcp://127.0.0.1:5558",
        topic=runner.KV_EVENTS_TOPIC,
    )
    subscriber.start()
    subscriber.replay_from_start()
    sub_socket = created[0]
    # Consume the replay handshake's own buffered batch first.
    initial = subscriber.drain()
    assert [batch.sequence for batch in initial] == [0]
    # A duplicate delivery of sequence 0 (e.g. the live SUB socket redelivers
    # something the replay handshake already ingested) must not raise and
    # must not be double-counted.
    sub_socket.inbox = [
        [
            runner.KV_EVENTS_TOPIC.encode(),
            (0).to_bytes(8, "big", signed=True),
            _encode_batch_payload(0),
        ]
    ]
    drained = subscriber.drain()
    assert drained == ()


def test_live_kv_event_subscriber_refuses_a_malformed_sequence_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _context, created = _install_fake_zmq_module(monkeypatch, replay_frames=[])
    subscriber = runner.LiveKVEventSubscriber(
        endpoint="tcp://127.0.0.1:5557",
        replay_endpoint="tcp://127.0.0.1:5558",
        topic=runner.KV_EVENTS_TOPIC,
    )
    subscriber.start()
    sub_socket = created[0]
    sub_socket.inbox = [
        [runner.KV_EVENTS_TOPIC.encode(), b"short", _encode_batch_payload(0)]
    ]
    with pytest.raises(runner.KVTruthProtocolError, match="malformed KV-event"):
        subscriber.drain()


def test_live_kv_event_subscriber_refuses_an_unexpected_multipart_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _context, created = _install_fake_zmq_module(monkeypatch, replay_frames=[])
    subscriber = runner.LiveKVEventSubscriber(
        endpoint="tcp://127.0.0.1:5557",
        replay_endpoint="tcp://127.0.0.1:5558",
        topic=runner.KV_EVENTS_TOPIC,
    )
    subscriber.start()
    sub_socket = created[0]
    sub_socket.inbox = [[(0).to_bytes(8, "big", signed=True)]]  # only one frame
    with pytest.raises(runner.KVTruthProtocolError, match="unexpected multipart shape"):
        subscriber.drain()


def test_live_kv_event_subscriber_refuses_the_end_sentinel_on_the_live_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _context, created = _install_fake_zmq_module(monkeypatch, replay_frames=[])
    subscriber = runner.LiveKVEventSubscriber(
        endpoint="tcp://127.0.0.1:5557",
        replay_endpoint="tcp://127.0.0.1:5558",
        topic=runner.KV_EVENTS_TOPIC,
    )
    subscriber.start()
    sub_socket = created[0]
    sub_socket.inbox = [
        [
            b"",
            runner.END_OF_REPLAY_SEQUENCE.to_bytes(8, "big", signed=True),
            b"",
        ]
    ]
    with pytest.raises(runner.KVTruthProtocolError, match="end-of-replay sentinel"):
        subscriber.drain()
