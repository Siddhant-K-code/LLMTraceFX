from __future__ import annotations

import hashlib
from typing import Any

import pytest

from vllm_kv_truth.vllm_live import (
    DEFAULT_DATA_PARALLEL_RANK,
    END_OF_REPLAY_SEQUENCE,
    REQUIRED_KV_EVENTS_USE_INT_BLOCK_HASHES,
    REQUIRED_PREFIX_CACHING_HASH_ALGO,
    REQUIRED_VLLM_COMMIT,
    REQUIRED_VLLM_VERSION,
    LiveAllBlocksCleared,
    LiveBlockRemoved,
    LiveBlockStored,
    RuntimeAttestationError,
    assess_live_vllm_capabilities,
    canonical_json,
    composite_sequence,
    compute_sha256_cbor_block_hashes,
    decode_sequence_frame,
    decompose_composite_sequence,
    parse_identity_receipt,
    parse_live_kv_event,
    parse_live_kv_event_batch,
    parse_live_kv_event_stream,
    parse_runtime_attestation,
    required_source_file_digests,
    required_source_file_paths,
    sha256_digest,
)
from llmtracefx.cache_audit.schema import EvidenceBasis
from llmtracefx.optimizer.schema import SchemaValidationError


def _hex(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _source_file_digests(**per_path_overrides: bool) -> list[dict[str, Any]]:
    """Every required source file, reported with its real committed-manifest
    expected digest and ``matches_manifest=True`` by default. Pass
    ``{path: False}`` to instead report a deliberately wrong (tampered)
    digest for that path, consistently flagged as not matching -- the
    validator now independently recomputes this comparison itself, so a
    fixture can no longer merely assert a mismatched boolean against an
    unrelated hash value.
    """

    expected = required_source_file_digests()
    digests = []
    for path in sorted(required_source_file_paths()):
        matches = per_path_overrides.get(path, True)
        sha256 = expected[path] if matches else _hex(path)
        digests.append(
            {
                "path": path,
                "sha256": sha256,
                "matches_manifest": matches,
            }
        )
    return digests


def _identity_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1",
        "protocol_id": "qwen3-8b-vllm-kv-truth-v1",
        "generated_at": "2026-09-06T08:10:00Z",
        "repository_commit": "596d1a97ea2fab4dfc20e9262134925986257c09",
        "image_repository_digest": (
            "vllm/vllm-openai:v0.28.0@sha256:" + _hex("base-image")
        ),
        "image_id": "sha256:" + _hex("derived-image"),
        "vllm_version": REQUIRED_VLLM_VERSION,
        "vllm_commit": REQUIRED_VLLM_COMMIT,
        "python_version": "3.12.3",
        "torch_version": "2.13.0+cu130",
        "cuda_runtime_version": "13.0",
        "transformers_version": "5.15.1",
        "typing_extensions_version": "4.15.0",
        "cuda_driver_version": "580.159.03",
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "gpu_memory_mib": 24564,
        "gpu_compute_capability": "8.9",
        "gpu_uuid_commitment": _hex("gpu-uuid-salted"),
        "experiment_nonce": "nonce-abc",
        "installed_distributions_digest": "sha256:" + _hex("dist"),
        "wheel_record_digest": "sha256:" + _hex("wheel"),
        "package_tree_digest": "sha256:" + _hex("tree"),
        "source_file_digests": _source_file_digests(),
        "model_id": "Qwen/Qwen3-8B",
        "model_revision": "b968826d9c46dd6066d109eabc6255188de91218",
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
        "max_model_len": 1024,
        "max_num_seqs": 1,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "block_size": 16,
        "prefix_match_unit": 16,
        "num_gpu_blocks_override": 96,
        "gpu_memory_utilization": 0.90,
        "cache_dtype": "bfloat16",
        "prefix_caching_hash_algo": REQUIRED_PREFIX_CACHING_HASH_ALGO,
        "kv_events_use_int_block_hashes": REQUIRED_KV_EVENTS_USE_INT_BLOCK_HASHES,
        "pythonhashseed": "0",
        "enable_prefix_caching": True,
        "enable_kv_cache_events": True,
        "enforce_eager": True,
        "speculative_config_enabled": False,
        "lora_enabled": False,
        "multimodal_enabled": False,
        "cache_salt_present": False,
        "cache_salt": None,
    }
    base.update(overrides)
    return base


def _kv_events_config(**overrides: Any) -> dict[str, Any]:
    base = {
        "topic": "kv-events",
        "endpoint_role": "loopback_pub",
        "replay_endpoint_role": "loopback_replay",
        "buffer_steps": 10_000,
        "hwm": 100_000,
        "max_queue_size": 100_000,
        "data_parallel_rank": 0,
        "first_sequence": 0,
        "last_sequence": 41,
        "capture_start_monotonic": 1.0,
        "capture_end_monotonic": 5.0,
    }
    base.update(overrides)
    return base


def _attestation_payload(
    *,
    identity: dict[str, Any] | None = None,
    resolved_config: dict[str, Any] | None = None,
    kv_events_config: dict[str, Any] | None = None,
    attested_at: str = "2026-09-06T08:30:00Z",
) -> dict[str, Any]:
    identity_payload = identity if identity is not None else _identity_payload()
    resolved_config_payload = _resolved_config(**(resolved_config or {}))
    kv_events_config_payload = _kv_events_config(**(kv_events_config or {}))
    payload: dict[str, Any] = {
        "identity": identity_payload,
        "resolved_config": resolved_config_payload,
        "kv_events_config": kv_events_config_payload,
        "attested_at": attested_at,
    }
    unsealed = {
        "identity_seal": identity_payload["seal"],
        "resolved_config": resolved_config_payload,
        "kv_events_config": kv_events_config_payload,
        "attested_at": attested_at,
    }
    payload["seal"] = hashlib.sha256(canonical_json(unsealed).encode()).hexdigest()
    return payload


def test_valid_identity_receipt_parses() -> None:
    receipt = parse_identity_receipt(_identity_payload())
    assert receipt.vllm_commit == REQUIRED_VLLM_COMMIT
    assert receipt.source_manifest_verified is True


def test_identity_receipt_seal_mismatch_is_refused() -> None:
    payload = _identity_payload()
    payload["seal"] = "0" * 64
    with pytest.raises(RuntimeAttestationError, match="seal"):
        parse_identity_receipt(payload)


def test_identity_receipt_vllm_identity_mismatch_is_refused() -> None:
    with pytest.raises(RuntimeAttestationError, match="vllm_version"):
        parse_identity_receipt(_identity_payload(vllm_version="0.27.0"))
    with pytest.raises(RuntimeAttestationError, match="vllm_commit"):
        parse_identity_receipt(_identity_payload(vllm_commit="0" * 40))


def test_identity_receipt_incomplete_source_manifest_is_refused() -> None:
    tampered_path = sorted(required_source_file_paths())[0]
    payload = _identity_payload(
        source_file_digests=_source_file_digests(**{tampered_path: False})
    )
    with pytest.raises(RuntimeAttestationError, match="source_file_digests"):
        parse_identity_receipt(payload)


def test_identity_receipt_source_digest_is_independently_recomputed_not_trusted() -> (
    None
):
    """A caller cannot claim a tampered file "matches" by simply setting the
    reported ``matches_manifest`` boolean to ``True``: this module
    independently recomputes the comparison against the committed
    manifest's real expected digest and refuses the receipt outright when
    the two disagree.
    """

    tampered_path = sorted(required_source_file_paths())[0]
    digests = _source_file_digests()
    for entry in digests:
        if entry["path"] == tampered_path:
            entry["sha256"] = _hex("tampered-bytes")
            entry["matches_manifest"] = True
    payload = _identity_payload(source_file_digests=digests)
    with pytest.raises(SchemaValidationError, match="matches_manifest"):
        parse_identity_receipt(payload)


def test_identity_receipt_source_digests_match_the_real_committed_manifest() -> None:
    """Sanity check that the fixture's default digests are the genuine
    upstream-fetched expected values, not merely internally self-consistent
    placeholders -- so :func:`test_valid_identity_receipt_parses` proves a
    real digest-value comparison, not a vacuous one."""

    expected = required_source_file_digests()
    for entry in _source_file_digests():
        assert entry["sha256"] == expected[entry["path"]]
        assert entry["matches_manifest"] is True


def test_identity_receipt_missing_required_source_file_is_refused() -> None:
    payload = _identity_payload()
    del payload["source_file_digests"][0]
    sealed = dict(payload)
    del sealed["seal"]
    payload["seal"] = hashlib.sha256(canonical_json(sealed).encode()).hexdigest()
    with pytest.raises(SchemaValidationError, match="missing="):
        parse_identity_receipt(payload)


def test_identity_receipt_extra_unlisted_source_file_is_refused() -> None:
    payload = _identity_payload()
    payload["source_file_digests"].append(
        {
            "path": "vllm/not/a/pinned/critical/file.py",
            "sha256": _hex("unexpected"),
            "matches_manifest": True,
        }
    )
    sealed = dict(payload)
    del sealed["seal"]
    payload["seal"] = hashlib.sha256(canonical_json(sealed).encode()).hexdigest()
    with pytest.raises(SchemaValidationError, match="extra="):
        parse_identity_receipt(payload)


def test_required_source_file_paths_is_a_fixed_nonempty_set() -> None:
    paths = required_source_file_paths()
    assert isinstance(paths, frozenset)
    assert paths
    assert "vllm/distributed/kv_events.py" in paths
    assert paths == required_source_file_paths()


def test_valid_attestation_parses_and_reports_supported() -> None:
    payload = _attestation_payload()
    attestation = parse_runtime_attestation(payload)
    assert attestation.vllm_commit == REQUIRED_VLLM_COMMIT
    assert attestation.resolved_config.block_size == 16
    assert attestation.source_manifest_verified is True
    # ergonomic passthroughs read through to the embedded identity receipt
    assert attestation.model_id == "Qwen/Qwen3-8B"
    assert attestation.gpu_name == "NVIDIA GeForce RTX 4090"

    capability = assess_live_vllm_capabilities(payload)
    assert capability.supported is True
    assert capability.reasons == ()


def test_attestation_seal_mismatch_is_refused() -> None:
    payload = _attestation_payload()
    payload["seal"] = "0" * 64
    with pytest.raises(RuntimeAttestationError, match="seal"):
        parse_runtime_attestation(payload)


def test_attestation_cannot_predate_its_identity_receipt() -> None:
    """Coherence: an attestation sealed before engine init is a contradiction."""

    payload = _attestation_payload(attested_at="2026-09-06T00:00:00Z")
    with pytest.raises(RuntimeAttestationError, match="attested_at"):
        parse_runtime_attestation(payload)


def test_attestation_rejects_an_invalid_embedded_identity() -> None:
    tampered_identity = _identity_payload()
    tampered_identity["gpu_name"] = "a different tampered GPU"
    payload = _attestation_payload(identity=tampered_identity)
    # the embedded identity's own seal no longer matches its tampered payload
    with pytest.raises(RuntimeAttestationError, match="seal"):
        parse_runtime_attestation(payload)


@pytest.mark.parametrize(
    ("resolved_config_overrides", "match"),
    [
        ({"prefix_caching_hash_algo": "sha256"}, "prefix_caching_hash_algo"),
        ({"kv_events_use_int_block_hashes": "1"}, "kv_events_use_int_block_hashes"),
        ({"enable_prefix_caching": False}, "enable_prefix_caching"),
        ({"enable_kv_cache_events": False}, "enable_kv_cache_events"),
        ({"pythonhashseed": "random"}, "pythonhashseed"),
        ({"pythonhashseed": ""}, "pythonhashseed"),
        ({"gpu_memory_utilization": 1.5}, "gpu_memory_utilization"),
        ({"gpu_memory_utilization": 0.0}, "gpu_memory_utilization"),
        ({"prefix_match_unit": 17}, "prefix_match_unit"),
        ({"cache_salt_present": True, "cache_salt": None}, "cache_salt"),
        ({"cache_salt_present": False, "cache_salt": "s"}, "cache_salt"),
    ],
)
def test_resolved_config_defect_is_refused(
    resolved_config_overrides: dict[str, Any], match: str
) -> None:
    payload = _attestation_payload(resolved_config=resolved_config_overrides)
    with pytest.raises(SchemaValidationError, match=match):
        parse_runtime_attestation(payload)


@pytest.mark.parametrize(
    ("kv_events_overrides", "match"),
    [
        ({"endpoint_role": "tcp://0.0.0.0:5557"}, "endpoint_role"),
        ({"replay_endpoint_role": None}, "replay_endpoint_role"),
        ({"first_sequence": 10, "last_sequence": 5}, "sequence"),
        ({"capture_start_monotonic": 5.0, "capture_end_monotonic": 1.0}, "monotonic"),
    ],
)
def test_kv_events_config_defect_is_refused(
    kv_events_overrides: dict[str, Any], match: str
) -> None:
    payload = _attestation_payload(kv_events_config=kv_events_overrides)
    with pytest.raises(SchemaValidationError, match=match):
        parse_runtime_attestation(payload)


def test_missing_and_extra_top_level_fields_are_refused() -> None:
    payload = _attestation_payload()
    del payload["attested_at"]
    with pytest.raises(SchemaValidationError, match="runtime_attestation"):
        parse_runtime_attestation(payload)

    payload = _attestation_payload()
    payload["unexpected_field"] = "nope"
    with pytest.raises(SchemaValidationError, match="runtime_attestation"):
        parse_runtime_attestation(payload)

    identity_payload = _identity_payload()
    del identity_payload["experiment_nonce"]
    with pytest.raises(SchemaValidationError, match="identity_receipt"):
        parse_identity_receipt(identity_payload)


def test_assess_live_vllm_capabilities_never_raises_on_bad_input() -> None:
    capability = assess_live_vllm_capabilities({"not": "an attestation"})
    assert capability.supported is False
    assert capability.reasons
    assert capability.backend == "vllm_live"


def test_redact_never_exposes_raw_cache_salt() -> None:
    payload = _attestation_payload(
        resolved_config={"cache_salt_present": True, "cache_salt": "top-secret-salt"}
    )
    attestation = parse_runtime_attestation(payload)
    assert attestation.resolved_config.cache_salt == "top-secret-salt"
    redacted = attestation.redact()
    assert redacted["resolved_config"]["cache_salt"] is None
    assert "top-secret-salt" not in canonical_json(redacted)


def test_sha256_digest_is_stable_and_prefixed() -> None:
    digest = sha256_digest({"a": 1, "b": 2})
    assert digest == sha256_digest({"b": 2, "a": 1})
    assert digest.startswith("sha256:")


# ---------------------------------------------------------------------------
# Live KV event parsing (exact vLLM 0.28.0 schema at the pinned commit)
# ---------------------------------------------------------------------------


def _block_stored(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "type": "BlockStored",
        "block_hashes": list(
            compute_sha256_cbor_block_hashes(
                token_ids=list(range(16)),
                block_size=16,
                parent_block_hash=None,
                extra_keys=None,
            )
        ),
        "parent_block_hash": None,
        "token_ids": list(range(16)),
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
    base.update(overrides)
    return base


def test_parse_live_block_stored_event() -> None:
    event = parse_live_kv_event(_block_stored())
    assert isinstance(event, LiveBlockStored)
    assert event.block_size == 16
    assert not hasattr(event, "ownership")
    assert not hasattr(event, "session_id")
    assert event.redact() == {
        "type": "BlockStored",
        "block_count": 1,
        "token_count": 16,
        "block_size": 16,
        "medium": "GPU",
        "group_idx": 0,
        "kv_cache_spec_kind": "full_attention",
        "kv_cache_spec_sliding_window": None,
        "locality": "LOCAL",
    }


def test_parse_live_block_stored_accepts_exact_msgpack_byte_hashes() -> None:
    data = _block_stored()
    data["block_hashes"] = [bytes.fromhex(data["block_hashes"][0])]
    event = parse_live_kv_event(data)
    assert isinstance(event, LiveBlockStored)
    assert event.block_hashes == tuple(_block_stored()["block_hashes"])


def test_block_stored_rejects_tampered_canonical_cbor_hash_chain() -> None:
    with pytest.raises(SchemaValidationError, match="canonical sha256_cbor"):
        parse_live_kv_event(_block_stored(block_hashes=[_hex("tampered")]))


def test_block_stored_rejects_wrong_token_count() -> None:
    with pytest.raises(SchemaValidationError, match="one complete block"):
        parse_live_kv_event(_block_stored(token_ids=list(range(10))))


def test_block_stored_rejects_bad_hash_format() -> None:
    with pytest.raises(SchemaValidationError, match="256-bit SHA-256"):
        parse_live_kv_event(_block_stored(block_hashes=["not-a-hash"]))


def test_block_stored_rejects_extra_keys_length_mismatch() -> None:
    with pytest.raises(SchemaValidationError, match="extra_keys"):
        parse_live_kv_event(_block_stored(extra_keys=[["lora-a"], ["lora-b"]]))


def test_block_stored_rejects_missing_field() -> None:
    data = _block_stored()
    del data["locality"]
    with pytest.raises(SchemaValidationError, match="BlockStored"):
        parse_live_kv_event(data)


def test_block_stored_rejects_unknown_field() -> None:
    """The exact pinned commit has no ``ownership``/``session_id`` fields."""

    data = _block_stored()
    data["ownership"] = "LOCAL"
    with pytest.raises(SchemaValidationError, match="BlockStored"):
        parse_live_kv_event(data)
    data = _block_stored()
    data["session_id"] = "s-0"
    with pytest.raises(SchemaValidationError, match="BlockStored"):
        parse_live_kv_event(data)


def test_parse_live_block_removed_event() -> None:
    event = parse_live_kv_event(
        {
            "type": "BlockRemoved",
            "block_hashes": [_hex("removed-0")],
            "medium": "GPU",
            "group_idx": 0,
            "locality": "LOCAL",
        }
    )
    assert isinstance(event, LiveBlockRemoved)
    assert not hasattr(event, "ownership")
    assert event.redact()["block_count"] == 1


def test_block_removed_rejects_unknown_field() -> None:
    with pytest.raises(SchemaValidationError, match="BlockRemoved"):
        parse_live_kv_event(
            {
                "type": "BlockRemoved",
                "block_hashes": [_hex("removed-0")],
                "medium": "GPU",
                "group_idx": 0,
                "locality": "LOCAL",
                "ownership": "LOCAL",
            }
        )


def test_parse_live_all_blocks_cleared_event() -> None:
    event = parse_live_kv_event({"type": "AllBlocksCleared"})
    assert isinstance(event, LiveAllBlocksCleared)
    assert event.redact() == {"type": "AllBlocksCleared"}


def test_unknown_event_type_is_rejected() -> None:
    with pytest.raises(SchemaValidationError, match="invalid"):
        parse_live_kv_event({"type": "SomethingElse"})


def test_decode_sequence_frame() -> None:
    assert decode_sequence_frame((0).to_bytes(8, "big")) == 0
    assert decode_sequence_frame((41).to_bytes(8, "big")) == 41
    assert (
        decode_sequence_frame((-1).to_bytes(8, "big", signed=True))
        == END_OF_REPLAY_SEQUENCE
    )
    with pytest.raises(SchemaValidationError, match="8 bytes"):
        decode_sequence_frame(b"\x00\x01")
    with pytest.raises(SchemaValidationError, match="non-negative"):
        decode_sequence_frame((-2).to_bytes(8, "big", signed=True))


def test_parse_live_kv_event_batch() -> None:
    batch = parse_live_kv_event_batch(
        b"kv-events",
        (7).to_bytes(8, "big"),
        [1.5, [{"type": "AllBlocksCleared"}], 0],
    )
    assert batch.sequence == 7
    assert batch.topic == "kv-events"
    records = batch.to_cache_event_records()
    assert len(records) == 1
    assert records[0].basis is EvidenceBasis.ENGINE_ATTESTED
    assert records[0].sequence == composite_sequence(7, 0)


def test_parse_live_kv_event_batch_defaults_omitted_data_parallel_rank() -> None:
    """vLLM 0.28.0's ``EventBatch`` is ``msgspec.Struct(array_like=True)``,
    so a batch published from the default data-parallel rank omits the
    trailing array element entirely; this must default to
    :data:`DEFAULT_DATA_PARALLEL_RANK`, not raise or silently become
    ``None``.
    """

    batch = parse_live_kv_event_batch(
        b"kv-events",
        (7).to_bytes(8, "big"),
        [1.5, [{"type": "AllBlocksCleared"}]],
    )
    assert batch.data_parallel_rank == DEFAULT_DATA_PARALLEL_RANK == 0


def test_parse_live_kv_event_batch_accepts_explicit_nonzero_data_parallel_rank() -> (
    None
):
    batch = parse_live_kv_event_batch(
        b"kv-events",
        (7).to_bytes(8, "big"),
        [1.5, [], 3],
    )
    assert batch.data_parallel_rank == 3


def test_parse_live_kv_event_batch_rejects_a_mapping_payload() -> None:
    """The real wire payload is array-like, never a mapping; accepting a
    mapping would silently fabricate support for an encoding vLLM 0.28.0
    does not actually produce."""

    with pytest.raises(SchemaValidationError, match="array-like"):
        parse_live_kv_event_batch(
            b"kv-events",
            (7).to_bytes(8, "big"),
            {"ts": 1.5, "events": [], "data_parallel_rank": 0},  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad_length", [0, 1, 4])
def test_parse_live_kv_event_batch_rejects_wrong_array_length(bad_length: int) -> None:
    payload = list(range(bad_length))
    with pytest.raises(SchemaValidationError, match="2 or 3 elements"):
        parse_live_kv_event_batch(b"kv-events", (7).to_bytes(8, "big"), payload)


def test_parse_live_kv_event_batch_rejects_non_array_events_element() -> None:
    with pytest.raises(SchemaValidationError, match="events.*must be an array"):
        parse_live_kv_event_batch(
            b"kv-events", (7).to_bytes(8, "big"), [1.5, "not-a-list"]
        )


def test_composite_sequence_avoids_duplicate_sequences_within_one_batch() -> None:
    """A batch with several events must not collapse them onto one sequence."""

    batch = parse_live_kv_event_batch(
        b"kv-events",
        (3).to_bytes(8, "big"),
        [
            9.0,
            [
                {"type": "AllBlocksCleared"},
                {"type": "AllBlocksCleared"},
                {"type": "AllBlocksCleared"},
            ],
            0,
        ],
    )
    records = batch.to_cache_event_records()
    sequences = [record.sequence for record in records]
    assert len(set(sequences)) == len(sequences)
    assert sequences == sorted(sequences)
    for ordinal, sequence in enumerate(sequences):
        assert decompose_composite_sequence(sequence) == (3, ordinal)


def test_composite_sequence_round_trips() -> None:
    assert decompose_composite_sequence(composite_sequence(41, 5)) == (41, 5)
    assert decompose_composite_sequence(composite_sequence(0, 0)) == (0, 0)
    with pytest.raises(SchemaValidationError, match="non-negative"):
        composite_sequence(-1, 0)
    with pytest.raises(SchemaValidationError, match="event_ordinal"):
        composite_sequence(0, -1)
    with pytest.raises(SchemaValidationError, match="non-negative"):
        decompose_composite_sequence(-1)


def test_parse_live_kv_event_batch_rejects_end_of_replay_marker() -> None:
    with pytest.raises(SchemaValidationError, match="replay marker"):
        parse_live_kv_event_batch(
            b"kv-events",
            (-1).to_bytes(8, "big", signed=True),
            [1.0, [], 0],
        )


def test_stream_report_flags_gaps_and_duplicates() -> None:
    def batch(seq: int) -> Any:
        return parse_live_kv_event_batch(
            b"kv-events",
            seq.to_bytes(8, "big"),
            [float(seq), [], 0],
        )

    batches = [batch(0), batch(1), batch(1), batch(4)]
    report = parse_live_kv_event_stream(
        batches, capture_start_sequence=0, capture_end_sequence=4
    )
    assert report.sequence_gaps == ((2, 3),)
    assert report.duplicate_sequences == (1,)
    assert report.eligible is False
    assert "kv_event_sequence_gaps" in report.ineligibility_reasons
    assert "kv_event_duplicate_sequences" in report.ineligibility_reasons


def test_stream_report_structurally_eligible_when_clean() -> None:
    def batch(seq: int) -> Any:
        return parse_live_kv_event_batch(
            b"kv-events",
            seq.to_bytes(8, "big"),
            [float(seq), [], 0],
        )

    batches = [batch(0), batch(1), batch(2)]
    report = parse_live_kv_event_stream(
        batches, capture_start_sequence=0, capture_end_sequence=2
    )
    assert report.eligible is False
    assert report.structurally_eligible is True
    assert report.ineligibility_reasons == (
        "runtime_event_attestation_request_binding_unavailable",
    )


def test_stream_report_flags_topic_and_dp_rank_inconsistency() -> None:
    a = parse_live_kv_event_batch(
        b"topic-a",
        (0).to_bytes(8, "big"),
        [0.0, [], 0],
    )
    b = parse_live_kv_event_batch(
        b"topic-b",
        (1).to_bytes(8, "big"),
        [1.0, [], 1],
    )
    report = parse_live_kv_event_stream([a, b])
    assert report.topic_consistent is False
    assert report.data_parallel_rank_consistent is False
    assert "kv_event_topic_inconsistent" in report.ineligibility_reasons
    assert "kv_event_data_parallel_rank_inconsistent" in report.ineligibility_reasons
