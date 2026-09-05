from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from llmtracefx.cache_audit.adapters.vllm import (
    REQUIRED_KV_EVENTS_USE_INT_BLOCK_HASHES,
    REQUIRED_PREFIX_CACHING_HASH_ALGO,
    REQUIRED_VLLM_COMMIT,
    REQUIRED_VLLM_VERSION,
    KVEventStreamReport,
    KVEventType,
    VLLMCapabilityConfig,
    VLLMRequestObservation,
    assess_vllm_capabilities,
    derive_reuse_expectation,
    parse_kv_event,
    parse_kv_event_stream,
    parse_vllm_request_observation,
    to_reuse_config,
)
from llmtracefx.optimizer.schema import SchemaValidationError


def _valid_config(**overrides: Any) -> VLLMCapabilityConfig:
    base = VLLMCapabilityConfig(
        version=REQUIRED_VLLM_VERSION,
        commit=REQUIRED_VLLM_COMMIT,
        prefix_caching_hash_algo=REQUIRED_PREFIX_CACHING_HASH_ALGO,
        pythonhashseed="0",
        enable_prefix_caching=True,
        enable_kv_cache_events=True,
        kv_events_use_int_block_hashes=REQUIRED_KV_EVENTS_USE_INT_BLOCK_HASHES,
        hash_block_size=4,
        physical_block_sizes=(8, 16),
        fine_grained_hits=False,
        runtime_attestation_digest="sha256:" + "a" * 64,
        runtime_attestation_exported=True,
        hash_representation="sha256_bytes",
        hash_width_bits=256,
    )
    return replace(base, **overrides)


def _block_stored(
    *,
    sequence: int = 0,
    block_hashes: list[str] | None = None,
    parent_block_hash: str | None = None,
    token_ids: list[int] | None = None,
    block_size: int = 4,
    medium: str | None = "GPU",
    lora_name: str | None = None,
    group_idx: int | None = 0,
    extra_keys: list[str] | None = None,
    cache_salt: str | None = None,
) -> dict[str, object]:
    return {
        "sequence": sequence,
        "type": "BlockStored",
        "block_hashes": block_hashes if block_hashes is not None else ["h0"],
        "parent_block_hash": parent_block_hash,
        "token_ids": token_ids if token_ids is not None else [1, 2, 3, 4],
        "block_size": block_size,
        "medium": medium,
        "lora_name": lora_name,
        "group_idx": group_idx,
        "extra_keys": extra_keys,
        "cache_salt": cache_salt,
    }


def _block_removed(
    *,
    sequence: int = 1,
    block_hashes: list[str] | None = None,
    medium: str | None = "GPU",
    group_idx: int | None = 0,
) -> dict[str, object]:
    return {
        "sequence": sequence,
        "type": "BlockRemoved",
        "block_hashes": block_hashes if block_hashes is not None else ["h0"],
        "medium": medium,
        "group_idx": group_idx,
    }


def _all_blocks_cleared(*, sequence: int = 2) -> dict[str, object]:
    return {"sequence": sequence, "type": "AllBlocksCleared"}


# --- capability: valid configuration -----------------------------------


def test_valid_configuration_is_supported_with_no_refusal_reasons() -> None:
    result = assess_vllm_capabilities(_valid_config())
    assert result.backend == "vllm"
    assert result.supported is True
    assert result.reasons == ("read_only_offline_configuration_verified",)
    assert "engine_cached_blocks" in result.observable_facts
    assert "client_ttft" in result.unavailable_facts


def test_environment_claims_never_satisfy_runtime_attestation() -> None:
    config = VLLMCapabilityConfig.from_environment(
        environ={
            "LLMTRACEFX_VLLM_COMMIT": REQUIRED_VLLM_COMMIT,
            "LLMTRACEFX_VLLM_PREFIX_CACHING_HASH_ALGO": (
                REQUIRED_PREFIX_CACHING_HASH_ALGO
            ),
            "LLMTRACEFX_VLLM_ENABLE_PREFIX_CACHING": "1",
            "LLMTRACEFX_VLLM_ENABLE_KV_CACHE_EVENTS": "1",
            "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES": "0",
            "LLMTRACEFX_VLLM_HASH_BLOCK_SIZE": "4",
            "LLMTRACEFX_VLLM_PHYSICAL_BLOCK_SIZES": "8",
            "PYTHONHASHSEED": "0",
        }
    )
    result = assess_vllm_capabilities(config)
    assert result.supported is False
    assert "runtime_exported_attestation_missing" in result.reasons


def test_event_stream_missing_gapped_duplicate_and_ambiguous_are_ineligible() -> None:
    assert parse_kv_event_stream([]).ineligibility_reasons == ("kv_events_missing",)

    gapped = parse_kv_event_stream(
        [_block_stored(sequence=0), _block_removed(sequence=2)]
    )
    assert "kv_event_sequence_gaps" in gapped.ineligibility_reasons
    assert gapped.eligible is False

    duplicate = parse_kv_event_stream(
        [_block_stored(sequence=0), _block_removed(sequence=0)]
    )
    assert "kv_event_duplicate_sequences" in duplicate.ineligibility_reasons

    ambiguous = parse_kv_event_stream(
        [
            _block_stored(sequence=0, block_hashes=["same"]),
            _block_stored(
                sequence=1,
                block_hashes=["same"],
                token_ids=[9, 8, 7, 6],
            ),
        ]
    )
    assert "kv_event_hash_identity_ambiguous" in ambiguous.ineligibility_reasons


def test_event_stream_inconsistent_block_metadata_is_ineligible() -> None:
    report = parse_kv_event_stream(
        [
            _block_stored(sequence=0, block_size=2, token_ids=[1, 2]),
            _block_stored(
                sequence=1,
                block_size=4,
                token_ids=[1, 2, 3, 4],
            ),
        ]
    )
    assert report.inconsistent_block_metadata is True
    assert "kv_event_block_metadata_inconsistent" in report.ineligibility_reasons


def test_capability_to_dict_matches_expected_shape() -> None:
    result = assess_vllm_capabilities(_valid_config())
    assert result.to_dict() == {
        "backend": "vllm",
        "supported": True,
        "reasons": ["read_only_offline_configuration_verified"],
        "observable_facts": list(result.observable_facts),
        "unavailable_facts": list(result.unavailable_facts),
    }


# --- capability: each refusal reason, in isolation ----------------------


def test_missing_vllm_installation_is_refused() -> None:
    result = assess_vllm_capabilities(_valid_config(version=None))
    assert result.supported is False
    assert "vllm_not_installed" in result.reasons


def test_wrong_vllm_version_is_refused() -> None:
    result = assess_vllm_capabilities(_valid_config(version="0.27.0"))
    assert result.supported is False
    assert "vllm_version_mismatch" in result.reasons


def test_missing_commit_is_refused() -> None:
    result = assess_vllm_capabilities(_valid_config(commit=None))
    assert result.supported is False
    assert "vllm_commit_unavailable" in result.reasons


def test_wrong_commit_is_refused() -> None:
    result = assess_vllm_capabilities(_valid_config(commit="0" * 40))
    assert result.supported is False
    assert "vllm_commit_mismatch" in result.reasons


def test_wrong_hash_algorithm_is_refused() -> None:
    result = assess_vllm_capabilities(_valid_config(prefix_caching_hash_algo="sha256"))
    assert result.supported is False
    assert "prefix_caching_hash_algo_not_sha256_cbor" in result.reasons


@pytest.mark.parametrize("pythonhashseed", [None, "", "random", "RANDOM", "abc"])
def test_unfixed_pythonhashseed_is_refused(pythonhashseed: str | None) -> None:
    result = assess_vllm_capabilities(_valid_config(pythonhashseed=pythonhashseed))
    assert result.supported is False
    assert "pythonhashseed_not_fixed" in result.reasons


def test_disabled_kv_events_is_refused() -> None:
    result = assess_vllm_capabilities(_valid_config(enable_kv_cache_events=False))
    assert result.supported is False
    assert "kv_cache_events_disabled" in result.reasons


def test_disabled_prefix_caching_is_refused() -> None:
    result = assess_vllm_capabilities(_valid_config(enable_prefix_caching=False))
    assert result.supported is False
    assert "prefix_caching_disabled" in result.reasons


@pytest.mark.parametrize("value", [None, "1", "true"])
def test_int_block_hashes_enabled_is_refused(value: str | None) -> None:
    result = assess_vllm_capabilities(
        _valid_config(kv_events_use_int_block_hashes=value)
    )
    assert result.supported is False
    assert "kv_events_int_block_hashes_enabled" in result.reasons


@pytest.mark.parametrize("hash_block_size", [None, 0, -1])
def test_non_positive_hash_block_size_is_refused(hash_block_size: int | None) -> None:
    result = assess_vllm_capabilities(_valid_config(hash_block_size=hash_block_size))
    assert result.supported is False
    assert "hash_block_size_not_positive" in result.reasons


def test_missing_physical_block_sizes_is_refused() -> None:
    result = assess_vllm_capabilities(_valid_config(physical_block_sizes=()))
    assert result.supported is False
    assert "physical_block_sizes_missing" in result.reasons


def test_non_positive_physical_block_size_is_refused() -> None:
    result = assess_vllm_capabilities(_valid_config(physical_block_sizes=(0, 16)))
    assert result.supported is False
    assert "physical_block_size_not_positive" in result.reasons


def test_misaligned_physical_block_size_is_refused() -> None:
    result = assess_vllm_capabilities(
        _valid_config(hash_block_size=4, physical_block_sizes=(8, 15))
    )
    assert result.supported is False
    assert "physical_block_size_misaligned" in result.reasons


def test_multiple_refusal_reasons_are_all_reported_in_order() -> None:
    result = assess_vllm_capabilities(
        _valid_config(version="0.27.0", enable_kv_cache_events=False)
    )
    assert result.supported is False
    assert result.reasons.index("vllm_version_mismatch") < result.reasons.index(
        "kv_cache_events_disabled"
    )


# --- from_environment -----------------------------------------------------


def test_from_environment_is_non_raising_and_defaults_closed() -> None:
    config = VLLMCapabilityConfig.from_environment(environ={})
    assert config.commit is None
    assert config.enable_prefix_caching is False
    assert config.enable_kv_cache_events is False
    assert config.hash_block_size is None
    assert config.physical_block_sizes == ()
    result = assess_vllm_capabilities(config)
    assert result.supported is False


def test_from_environment_reads_declared_attestation_variables() -> None:
    environ = {
        "LLMTRACEFX_VLLM_COMMIT": REQUIRED_VLLM_COMMIT,
        "LLMTRACEFX_VLLM_PREFIX_CACHING_HASH_ALGO": REQUIRED_PREFIX_CACHING_HASH_ALGO,
        "LLMTRACEFX_VLLM_ENABLE_PREFIX_CACHING": "true",
        "LLMTRACEFX_VLLM_ENABLE_KV_CACHE_EVENTS": "true",
        "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES": "0",
        "PYTHONHASHSEED": "0",
        "LLMTRACEFX_VLLM_HASH_BLOCK_SIZE": "4",
        "LLMTRACEFX_VLLM_PHYSICAL_BLOCK_SIZES": "8, 16",
        "LLMTRACEFX_VLLM_FINE_GRAINED_HITS": "false",
    }
    config = VLLMCapabilityConfig.from_environment(environ=environ)
    assert config.commit == REQUIRED_VLLM_COMMIT
    assert config.prefix_caching_hash_algo == REQUIRED_PREFIX_CACHING_HASH_ALGO
    assert config.enable_prefix_caching is True
    assert config.enable_kv_cache_events is True
    assert config.kv_events_use_int_block_hashes == "0"
    assert config.pythonhashseed == "0"
    assert config.hash_block_size == 4
    assert config.physical_block_sizes == (8, 16)
    assert config.fine_grained_hits is False


def test_from_environment_ignores_malformed_integers() -> None:
    config = VLLMCapabilityConfig.from_environment(
        environ={
            "LLMTRACEFX_VLLM_HASH_BLOCK_SIZE": "not-an-int",
            "LLMTRACEFX_VLLM_PHYSICAL_BLOCK_SIZES": "8,not-an-int",
        }
    )
    assert config.hash_block_size is None
    assert config.physical_block_sizes == ()


# --- reuse expectation helper --------------------------------------------


def test_derive_reuse_expectation_matches_expected_vllm_reuse_alignment() -> None:
    config = _valid_config(hash_block_size=4, physical_block_sizes=(8, 16))
    request = tuple(range(30))
    cached = request[:23] + (999,)
    result = derive_reuse_expectation(config, cached, request)
    assert result.semantic_prefix_tokens == 23
    assert result.policy_reusable_tokens == 16
    assert result.reusable_blocks == 4
    assert result.partial_block_tokens == 3


def test_derive_reuse_expectation_refuses_unsupported_configuration() -> None:
    config = _valid_config(enable_kv_cache_events=False)
    with pytest.raises(ValueError, match="kv_cache_events_disabled"):
        derive_reuse_expectation(config, (1, 2, 3), (1, 2, 3))


def test_to_reuse_config_rejects_missing_hash_block_size() -> None:
    with pytest.raises(ValueError, match="hash_block_size"):
        to_reuse_config(_valid_config(hash_block_size=None))


# --- synthetic event parsing ----------------------------------------------


def test_parse_block_stored_preserves_token_ids_privately() -> None:
    event = parse_kv_event(_block_stored(token_ids=[1, 2, 3, 4], cache_salt="secret"))
    assert event.event_type is KVEventType.BLOCK_STORED
    assert event.token_ids == (1, 2, 3, 4)
    assert event.cache_salt == "secret"


def test_parse_block_removed() -> None:
    event = parse_kv_event(_block_removed(block_hashes=["a", "b"]))
    assert event.event_type is KVEventType.BLOCK_REMOVED
    assert event.block_hashes == ("a", "b")


def test_parse_all_blocks_cleared() -> None:
    event = parse_kv_event(_all_blocks_cleared())
    assert event.event_type is KVEventType.ALL_BLOCKS_CLEARED
    assert event.block_hashes == ()
    assert event.token_ids == ()


def test_parse_rejects_unknown_event_type() -> None:
    with pytest.raises(SchemaValidationError, match="invalid"):
        parse_kv_event({"sequence": 0, "type": "NotARealEvent"})


def test_parse_rejects_missing_required_key() -> None:
    payload = _block_stored()
    del payload["block_size"]
    with pytest.raises(SchemaValidationError, match="missing"):
        parse_kv_event(payload)


def test_parse_rejects_unexpected_extra_key() -> None:
    payload = _all_blocks_cleared()
    payload["surprise"] = True
    with pytest.raises(SchemaValidationError, match="extra"):
        parse_kv_event(payload)


def test_parse_rejects_non_mapping() -> None:
    with pytest.raises(SchemaValidationError, match="object"):
        parse_kv_event([("sequence", 0)])  # type: ignore[arg-type]


def test_parse_rejects_negative_sequence() -> None:
    with pytest.raises(SchemaValidationError, match="sequence"):
        parse_kv_event(_all_blocks_cleared(sequence=-1))


def test_parse_rejects_block_hash_token_cardinality_mismatch() -> None:
    with pytest.raises(SchemaValidationError, match="one complete block"):
        parse_kv_event(
            _block_stored(
                block_hashes=["h0", "h1"],
                token_ids=[1, 2, 3, 4],
                block_size=4,
            )
        )


# --- redaction --------------------------------------------------------


def test_redact_removes_raw_hashes_token_ids_extra_keys_and_cache_salt() -> None:
    event = parse_kv_event(
        _block_stored(
            block_hashes=["raw-hash-0"],
            parent_block_hash="raw-parent-hash",
            token_ids=[10, 20, 30, 40],
            extra_keys=["mm-id-0"],
            cache_salt="tenant-secret",
        )
    )
    redacted = event.redact()
    assert "block_hashes" not in redacted
    assert "parent_block_hash" not in redacted
    assert "token_ids" not in redacted
    assert "extra_keys" not in redacted
    assert "cache_salt" not in redacted
    for value in redacted.values():
        assert value != "raw-hash-0"
        assert value != "raw-parent-hash"
        assert value != "tenant-secret"
    assert redacted["sequence"] == event.sequence
    assert redacted["event_type"] == "BlockStored"
    assert redacted["block_size"] == 4


def test_redact_removes_lora_name_as_cache_identity_metadata() -> None:
    event = parse_kv_event(_block_stored(lora_name="tenant-lora-adapter"))
    redacted = event.redact()
    assert "lora_name" not in redacted
    assert "tenant-lora-adapter" not in redacted.values()


def test_redact_all_blocks_cleared_is_minimal() -> None:
    event = parse_kv_event(_all_blocks_cleared(sequence=9))
    assert event.redact() == {
        "sequence": 9,
        "event_type": "AllBlocksCleared",
        "block_size": None,
        "medium": None,
        "group_idx": None,
    }


# --- event-stream gap and duplicate detection -----------------------------


def test_stream_with_no_defects_reports_none() -> None:
    report = parse_kv_event_stream(
        [
            _block_stored(sequence=0),
            _block_removed(sequence=1),
            _all_blocks_cleared(sequence=2),
        ]
    )
    assert isinstance(report, KVEventStreamReport)
    assert len(report.events) == 3
    assert report.has_gaps is False
    assert report.has_duplicate_sequences is False
    assert report.sequence_gaps == ()
    assert report.duplicate_sequences == ()


def test_stream_detects_sequence_gap() -> None:
    report = parse_kv_event_stream(
        [
            _block_stored(sequence=0),
            _block_stored(sequence=1),
            _block_stored(sequence=4),
            _block_stored(sequence=5),
        ]
    )
    assert report.has_gaps is True
    assert report.sequence_gaps == ((2, 3),)
    assert report.duplicate_sequences == ()


def test_stream_detects_duplicate_sequence_numbers() -> None:
    report = parse_kv_event_stream(
        [
            _block_stored(sequence=0),
            _block_stored(sequence=1),
            _block_stored(sequence=1),
            _block_stored(sequence=2),
        ]
    )
    assert report.has_duplicate_sequences is True
    assert report.duplicate_sequences == (1,)
    assert report.sequence_gaps == ()


def test_stream_detects_gaps_and_duplicates_independent_of_arrival_order() -> None:
    report = parse_kv_event_stream(
        [
            _block_stored(sequence=5),
            _block_stored(sequence=0),
            _block_stored(sequence=0),
            _block_stored(sequence=2),
        ]
    )
    assert report.duplicate_sequences == (0,)
    assert report.sequence_gaps == ((1, 1), (3, 4))


# --- ambiguous raw block hash detection -----------------------------------


def test_same_hash_with_same_token_ids_is_not_ambiguous() -> None:
    report = parse_kv_event_stream(
        [
            _block_stored(sequence=0, block_hashes=["h0"], token_ids=[1, 2, 3, 4]),
            _block_stored(sequence=1, block_hashes=["h0"], token_ids=[1, 2, 3, 4]),
        ]
    )
    assert report.has_ambiguous_block_hashes is False
    assert report.ambiguous_block_hash_count == 0
    assert report.ambiguous_block_hashes == ()


def test_same_hash_with_different_token_ids_is_ambiguous() -> None:
    report = parse_kv_event_stream(
        [
            _block_stored(sequence=0, block_hashes=["h0"], token_ids=[1, 2, 3, 4]),
            _block_stored(sequence=1, block_hashes=["h0"], token_ids=[9, 9, 9, 9]),
        ]
    )
    assert report.has_ambiguous_block_hashes is True
    assert report.ambiguous_block_hash_count == 1
    assert report.ambiguous_block_hashes == (((1, 2, 3, 4), (9, 9, 9, 9)),)
    # The raw hash "h0" is never stored: ambiguous_block_hashes is typed as
    # nested tuples of token IDs only, so no hash string can appear in it.


def test_ambiguity_ignores_block_removed_and_all_blocks_cleared_events() -> None:
    report = parse_kv_event_stream(
        [
            _block_stored(sequence=0, block_hashes=["h0"], token_ids=[1, 2, 3, 4]),
            _block_removed(sequence=1, block_hashes=["h0"]),
            _all_blocks_cleared(sequence=2),
        ]
    )
    assert report.has_ambiguous_block_hashes is False


def test_different_hashes_are_independently_not_ambiguous() -> None:
    report = parse_kv_event_stream(
        [
            _block_stored(sequence=0, block_hashes=["h0"], token_ids=[1, 2, 3, 4]),
            _block_stored(sequence=1, block_hashes=["h1"], token_ids=[9, 9, 9, 9]),
        ]
    )
    assert report.has_ambiguous_block_hashes is False
    assert report.ambiguous_block_hash_count == 0


def test_batched_hashes_are_matched_to_their_own_token_blocks() -> None:
    report = parse_kv_event_stream(
        [
            _block_stored(
                sequence=0,
                block_hashes=["h0", "h1"],
                token_ids=[1, 2, 3, 4, 5, 6, 7, 8],
            ),
            _block_stored(
                sequence=1,
                block_hashes=["h0"],
                token_ids=[1, 2, 3, 4],
            ),
        ]
    )
    assert report.has_ambiguous_block_hashes is False


# --- request observation parsing -----------------------------------------


def _request_observation(
    *,
    request_id: str = "req-0",
    prompt_token_ids: list[int] | None = None,
    output_token_ids: list[int] | None = None,
    num_cached_tokens: int = 4,
    num_cache_creation_tokens: int = 4,
    finished: bool = True,
    finish_reason: str | None = "stop",
    arrival_time: float | None = 0.0,
    first_token_time: float | None = 0.5,
    finished_time: float | None = 1.5,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "prompt_token_ids": (
            prompt_token_ids
            if prompt_token_ids is not None
            else [1, 2, 3, 4, 5, 6, 7, 8]
        ),
        "output_token_ids": output_token_ids
        if output_token_ids is not None
        else [9, 10],
        "num_cached_tokens": num_cached_tokens,
        "num_cache_creation_tokens": num_cache_creation_tokens,
        "finished": finished,
        "finish_reason": finish_reason,
        "arrival_time": arrival_time,
        "first_token_time": first_token_time,
        "finished_time": finished_time,
    }


def test_parse_valid_request_observation_and_durations() -> None:
    observation = parse_vllm_request_observation(_request_observation())
    assert isinstance(observation, VLLMRequestObservation)
    assert observation.request_id == "req-0"
    assert observation.prompt_token_ids == (1, 2, 3, 4, 5, 6, 7, 8)
    assert observation.output_token_ids == (9, 10)
    assert observation.num_cached_tokens == 4
    assert observation.num_cache_creation_tokens == 4
    assert observation.finished is True
    assert observation.finish_reason == "stop"
    assert observation.arrival_to_first_token_duration == pytest.approx(0.5)
    assert observation.ttft_duration == pytest.approx(0.5)
    assert observation.complete_duration == pytest.approx(1.5)


def test_parse_rejects_non_mapping_request_observation() -> None:
    with pytest.raises(SchemaValidationError, match="object"):
        parse_vllm_request_observation([("request_id", "req-0")])  # type: ignore[arg-type]


def test_parse_rejects_missing_or_extra_request_observation_keys() -> None:
    payload = _request_observation()
    del payload["finish_reason"]
    with pytest.raises(SchemaValidationError, match="missing"):
        parse_vllm_request_observation(payload)

    payload = _request_observation()
    payload["unexpected"] = True
    with pytest.raises(SchemaValidationError, match="extra"):
        parse_vllm_request_observation(payload)


def test_parse_rejects_non_bool_finished_field() -> None:
    with pytest.raises(SchemaValidationError, match="boolean"):
        parse_vllm_request_observation(_request_observation(finished=1))  # type: ignore[arg-type]


def test_parse_rejects_non_integer_token_arrays() -> None:
    with pytest.raises(SchemaValidationError, match="array"):
        parse_vllm_request_observation(
            _request_observation(prompt_token_ids="not-a-list")  # type: ignore[arg-type]
        )


def test_parse_rejects_negative_counters() -> None:
    with pytest.raises(SchemaValidationError, match="num_cached_tokens"):
        parse_vllm_request_observation(_request_observation(num_cached_tokens=-1))


@pytest.mark.parametrize("field", ["arrival_time", "first_token_time", "finished_time"])
def test_parse_rejects_non_finite_timing(field: str) -> None:
    overrides: dict[str, Any] = {field: float("nan")}
    with pytest.raises(SchemaValidationError, match="finite"):
        parse_vllm_request_observation(_request_observation(**overrides))


def test_parse_allows_null_timing_and_yields_null_durations() -> None:
    observation = parse_vllm_request_observation(
        _request_observation(
            arrival_time=None, first_token_time=None, finished_time=None
        )
    )
    assert observation.arrival_time is None
    assert observation.arrival_to_first_token_duration is None
    assert observation.ttft_duration is None
    assert observation.complete_duration is None


def test_parse_allows_partial_null_timing() -> None:
    observation = parse_vllm_request_observation(
        _request_observation(
            arrival_time=0.0, first_token_time=None, finished_time=None
        )
    )
    assert observation.arrival_to_first_token_duration is None
    assert observation.complete_duration is None


# --- request observation contradictions ----------------------------------


def test_parse_rejects_cached_tokens_exceeding_prompt_length() -> None:
    with pytest.raises(SchemaValidationError, match="num_cached_tokens exceeds"):
        parse_vllm_request_observation(
            _request_observation(prompt_token_ids=[1, 2], num_cached_tokens=3)
        )


def test_parse_rejects_creation_tokens_exceeding_prompt_length() -> None:
    with pytest.raises(
        SchemaValidationError, match="num_cache_creation_tokens exceeds"
    ):
        parse_vllm_request_observation(
            _request_observation(
                prompt_token_ids=[1, 2],
                num_cached_tokens=0,
                num_cache_creation_tokens=3,
            )
        )


def test_parse_rejects_jointly_impossible_cache_counters() -> None:
    with pytest.raises(SchemaValidationError, match="cache-creation tokens exceed"):
        parse_vllm_request_observation(
            _request_observation(
                prompt_token_ids=[1, 2, 3, 4],
                num_cached_tokens=3,
                num_cache_creation_tokens=2,
            )
        )


def test_parse_rejects_finished_without_finish_reason() -> None:
    with pytest.raises(SchemaValidationError, match="finish_reason is null"):
        parse_vllm_request_observation(
            _request_observation(finished=True, finish_reason=None)
        )


def test_parse_allows_unfinished_without_finish_reason() -> None:
    observation = parse_vllm_request_observation(
        _request_observation(finished=False, finish_reason=None, finished_time=None)
    )
    assert observation.finished is False
    assert observation.finish_reason is None


def test_parse_rejects_arrival_after_first_token() -> None:
    with pytest.raises(SchemaValidationError, match="not monotonic"):
        parse_vllm_request_observation(
            _request_observation(arrival_time=1.0, first_token_time=0.5)
        )


def test_parse_rejects_first_token_after_finished() -> None:
    with pytest.raises(SchemaValidationError, match="not monotonic"):
        parse_vllm_request_observation(
            _request_observation(first_token_time=2.0, finished_time=1.0)
        )


def test_parse_rejects_arrival_after_finished_when_first_token_missing() -> None:
    with pytest.raises(SchemaValidationError, match="not monotonic"):
        parse_vllm_request_observation(
            _request_observation(
                arrival_time=2.0, first_token_time=None, finished_time=1.0
            )
        )


# --- request observation redaction ---------------------------------------


def test_request_observation_redact_removes_token_arrays_but_keeps_counts() -> None:
    observation = parse_vllm_request_observation(
        _request_observation(
            prompt_token_ids=[1, 2, 3, 4, 5, 6, 7, 8], output_token_ids=[9, 10, 11]
        )
    )
    redacted = observation.redact()
    assert "prompt_token_ids" not in redacted
    assert "output_token_ids" not in redacted
    assert redacted["num_prompt_tokens"] == 8
    assert redacted["num_output_tokens"] == 3
    assert redacted["request_id"] == "req-0"
    assert redacted["num_cached_tokens"] == 4
    assert redacted["num_cache_creation_tokens"] == 4
    assert redacted["finished"] is True
    assert redacted["finish_reason"] == "stop"
    assert redacted["arrival_to_first_token_duration"] == pytest.approx(0.5)
    assert "queue_duration" not in redacted
    assert redacted["ttft_duration"] == pytest.approx(0.5)
    assert redacted["complete_duration"] == pytest.approx(1.5)
