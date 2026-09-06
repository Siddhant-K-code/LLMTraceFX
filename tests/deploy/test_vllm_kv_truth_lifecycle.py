"""Hermetic, no-GPU tests for the vLLM KV-truth host orchestrator.

Every test here drives :mod:`vllm_kv_truth.lifecycle`
through a fake :class:`~vllm_kv_truth.lifecycle.CommandRunner`
that never spawns a process, opens a socket, or touches the network/GPU. The
fake still has to answer every real ``ssh``/``scp``/``docker`` argv the
orchestrator builds with a plausible, stage-appropriate transcript, which is
what proves the orchestrator issues genuine, complete stage commands rather
than a placeholder that always refuses.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from vllm_kv_truth import evidence, lifecycle

VALID_HEAD = "a" * 40
VALID_NONCE = "c" * 40
BILLING_STARTED_AT = datetime(2030, 1, 1, tzinfo=timezone.utc)
DERIVED_IMAGE_ID = "sha256:" + "2" * 64


def _source_archive_bytes(commit: str = VALID_HEAD) -> bytes:
    """Build a minimal, valid checked-source archive: a tar containing
    exactly one ``COMMIT_HEAD`` marker file with the given commit hex."""

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        data = (commit + "\n").encode("ascii")
        info = tarfile.TarInfo(name="COMMIT_HEAD")
        info.size = len(data)
        tar.addfile(info, fileobj=io.BytesIO(data))
        runner_data = b"# exact fixture runner source\n"
        runner_info = tarfile.TarInfo(name="vllm_kv_truth/runner.py")
        runner_info.size = len(runner_data)
        tar.addfile(runner_info, fileobj=io.BytesIO(runner_data))
    return buffer.getvalue()


VALID_DERIVED_SOURCE_DIGEST = (
    "sha256:" + hashlib.sha256(_source_archive_bytes()).hexdigest()
)


def _write_source_archive(path: Path, commit: str = VALID_HEAD) -> None:
    path.write_bytes(_source_archive_bytes(commit))


def _fixture_event_batch(sequence: int, request_token_ids: Sequence[int]) -> Any:
    from vllm_kv_truth import runner as runner_mod
    from vllm_kv_truth.vllm_live import (
        compute_sha256_cbor_block_hashes,
        parse_live_kv_event_batch,
    )

    token_ids = list(request_token_ids[:16])
    block_hash = compute_sha256_cbor_block_hashes(
        token_ids=token_ids,
        block_size=16,
        parent_block_hash=None,
        extra_keys=None,
    )[0]
    return parse_live_kv_event_batch(
        runner_mod.KV_EVENTS_TOPIC.encode(),
        sequence.to_bytes(8, "big"),
        [
            float(sequence),
            [
                {
                    "type": "BlockStored",
                    "block_hashes": [bytes.fromhex(block_hash)],
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


def _fixture_reset_batch() -> Any:
    from vllm_kv_truth import runner as runner_mod
    from vllm_kv_truth.vllm_live import parse_live_kv_event_batch

    return parse_live_kv_event_batch(
        runner_mod.KV_EVENTS_TOPIC.encode(),
        (0).to_bytes(8, "big"),
        [0.0, [{"type": "AllBlocksCleared"}], 0],
    )


def _fixture_b_lane_records(*, cache_enabled: bool = True) -> tuple[Any, ...]:
    """One exact, event-bearing ``RequestRecord`` per fixed nested probe."""

    from vllm_kv_truth import runner as runner_mod
    from vllm_kv_truth.workload import NESTED_PROBES

    ids = runner_mod.request_ids()
    return tuple(
        runner_mod.RequestRecord(
            request_id=request_id,
            scenario=probe.scenario,
            prompt_token_count=len(probe.request_tokens),
            output_token_count=2,
            num_cached_tokens=(
                probe.expected_reusable_tokens if cache_enabled else None
            ),
            num_cache_creation_tokens=(
                len(probe.request_tokens) - probe.expected_reusable_tokens
                if cache_enabled
                else None
            ),
            finished=True,
            finish_reason="stop",
            event_batches=(
                (_fixture_event_batch(index + 1, probe.request_tokens),)
                if cache_enabled
                else ()
            ),
            prompt_token_ids=probe.request_tokens,
            output_token_ids=(7, 8),
        )
        for index, (request_id, probe) in enumerate(
            zip(ids, NESTED_PROBES, strict=True)
        )
    )


def _fixture_eviction_records() -> tuple[Any, ...]:
    """A minimal eviction-lane fixture whose final record is eligible for
    the plan's ``evicted`` verdict: zero cached tokens and full prompt work
    with a valid boundary."""

    from vllm_kv_truth import runner as runner_mod

    ids = runner_mod.eviction_lane_request_ids()
    from vllm_kv_truth.workload import (
        EVICTION_LANE_REQUESTS,
    )

    records = [
        runner_mod.RequestRecord(
            request_id=request_id,
            scenario=f"eviction_step_{index}",
            prompt_token_count=len(EVICTION_LANE_REQUESTS[index]),
            output_token_count=2,
            num_cached_tokens=0,
            num_cache_creation_tokens=len(EVICTION_LANE_REQUESTS[index]),
            finished=True,
            finish_reason="stop",
            event_batches=(
                _fixture_event_batch(index + 1, EVICTION_LANE_REQUESTS[index]),
            ),
            prompt_token_ids=EVICTION_LANE_REQUESTS[index],
            output_token_ids=(7, 8),
        )
        for index, request_id in enumerate(ids)
    ]
    return tuple(records)


def _fixture_runtime_attestation() -> dict[str, Any]:
    from vllm_kv_truth import runner as runner_mod
    from vllm_kv_truth.vllm_live import (
        canonical_json,
        required_source_file_digests,
    )

    identity: dict[str, Any] = {
        "schema_version": "1",
        "protocol_id": lifecycle.PROTOCOL_ID,
        "generated_at": "2030-01-01T00:00:00Z",
        "repository_commit": VALID_HEAD,
        "image_repository_digest": lifecycle.BASE_IMAGE_REFERENCE,
        "image_id": DERIVED_IMAGE_ID,
        "vllm_version": lifecycle.REQUIRED_VLLM_VERSION,
        "vllm_commit": lifecycle.REQUIRED_VLLM_COMMIT,
        "python_version": runner_mod.RUNTIME_PINS["python_version"],
        "torch_version": runner_mod.RUNTIME_PINS["torch_version"],
        "cuda_runtime_version": runner_mod.RUNTIME_PINS["cuda_version"],
        "transformers_version": runner_mod.RUNTIME_PINS["transformers_version"],
        "typing_extensions_version": runner_mod.RUNTIME_PINS[
            "typing_extensions_version"
        ],
        "cuda_driver_version": lifecycle.EXPECTED_DRIVER,
        "gpu_name": lifecycle.EXPECTED_GPU_NAME,
        "gpu_memory_mib": lifecycle.EXPECTED_MEMORY_MIB,
        "gpu_compute_capability": lifecycle.EXPECTED_GPU_COMPUTE_CAPABILITY,
        "gpu_uuid_commitment": hashlib.sha256(b"gpu").hexdigest(),
        "experiment_nonce": VALID_NONCE,
        "installed_distributions_digest": "sha256:" + hashlib.sha256(b"d").hexdigest(),
        "wheel_record_digest": "sha256:" + hashlib.sha256(b"w").hexdigest(),
        "package_tree_digest": "sha256:" + hashlib.sha256(b"p").hexdigest(),
        "source_file_digests": [
            {"path": path, "sha256": digest, "matches_manifest": True}
            for path, digest in sorted(required_source_file_digests().items())
        ],
        "model_id": lifecycle.MODEL_ID,
        "model_revision": lifecycle.MODEL_REVISION,
        "tokenizer_artifact_digest": lifecycle._expected_runtime_model_digests()[1],
        "model_inventory_digest": lifecycle._expected_runtime_model_digests()[0],
        "model_path_commitment": hashlib.sha256(b"/model").hexdigest(),
        "runner_source_digest": lifecycle.runner_source_digest_from_archive(
            _source_archive_bytes()
        ),
    }
    identity["seal"] = hashlib.sha256(
        canonical_json(identity).encode("utf-8")
    ).hexdigest()
    resolved = {
        "max_model_len": 1024,
        "max_num_seqs": 1,
        "tensor_parallel_size": 1,
        "data_parallel_size": 1,
        "block_size": 16,
        "prefix_match_unit": 16,
        "num_gpu_blocks_override": 96,
        "gpu_memory_utilization": 0.9,
        "cache_dtype": "bfloat16",
        "prefix_caching_hash_algo": "sha256_cbor",
        "kv_events_use_int_block_hashes": "0",
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
    events = {
        "topic": runner_mod.KV_EVENTS_TOPIC,
        "endpoint_role": "loopback_pub",
        "replay_endpoint_role": "loopback_replay",
        "buffer_steps": 10_000,
        "hwm": 100_000,
        "max_queue_size": 100_000,
        "data_parallel_rank": 0,
        "first_sequence": 0,
        "last_sequence": 10,
        "capture_start_monotonic": 1.0,
        "capture_end_monotonic": 2.0,
    }
    attested_at = "2030-01-01T00:00:01Z"
    payload = {
        "identity": identity,
        "resolved_config": resolved,
        "kv_events_config": events,
        "attested_at": attested_at,
    }
    payload["seal"] = hashlib.sha256(
        canonical_json(
            {
                "identity_seal": identity["seal"],
                "resolved_config": resolved,
                "kv_events_config": events,
                "attested_at": attested_at,
            }
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _fixture_lane_receipt_bytes(*, lane: str, records: tuple[Any, ...]) -> bytes:
    from vllm_kv_truth import runner as runner_mod

    lane_result = runner_mod.LaneResult(
        lane=lane,
        records=records,
        reset_event_batches=(
            (_fixture_reset_batch(),) if lane in {"B", "eviction"} else ()
        ),
    )
    receipt = runner_mod.build_protocol_receipt(
        lane_result, runtime_attestation=_fixture_runtime_attestation()
    )
    return (json.dumps(receipt.to_dict()) + "\n").encode("utf-8")


def _build_fixture_evidence_tar_bytes() -> bytes:
    """A real, valid tar archive of every fixed lane's output receipt, shaped
    exactly like what :mod:`kv_truth_runner`'s real CLI would have written
    to ``/evidence/{tag}.json`` inside each container -- this is what makes
    the full-run test exercise the real claim-matrix/evidence-bundle wiring
    end to end instead of a placeholder."""

    b_records = _fixture_b_lane_records()
    a_records = _fixture_b_lane_records(cache_enabled=False)
    eviction_records = _fixture_eviction_records()

    files: dict[str, bytes] = {
        "canary.json": _fixture_lane_receipt_bytes(lane="B", records=b_records),
        "eviction.json": _fixture_lane_receipt_bytes(
            lane="eviction", records=eviction_records
        ),
    }
    for tag in lifecycle.ab_pair_lane_tags():
        lane = "B" if tag.endswith("-b") else "A"
        records = b_records if lane == "B" else a_records
        files[f"{tag}.json"] = _fixture_lane_receipt_bytes(lane=lane, records=records)

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for name, data in files.items():
            info = tarfile.TarInfo(name=f"evidence/{name}")
            info.size = len(data)
            tar.addfile(info, fileobj=io.BytesIO(data))
        receipts_dir = tarfile.TarInfo(name="receipts")
        receipts_dir.type = tarfile.DIRTYPE
        tar.addfile(receipts_dir)
    return buffer.getvalue()


def _model_inventory_sha256() -> str:
    raw = lifecycle.MODEL_CONVERSION_MANIFEST_PATH.read_bytes()
    return hashlib.sha256(raw).hexdigest()


def _authorization_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": lifecycle.AUTHORIZATION_SCHEMA_VERSION,
        "protocol_id": lifecycle.PROTOCOL_ID,
        "repository_head": VALID_HEAD,
        "base_image_reference": lifecycle.BASE_IMAGE_REFERENCE,
        "derived_image_source_digest": VALID_DERIVED_SOURCE_DIGEST,
        "vllm_version": lifecycle.REQUIRED_VLLM_VERSION,
        "vllm_commit": lifecycle.REQUIRED_VLLM_COMMIT,
        "model_id": lifecycle.MODEL_ID,
        "model_revision": lifecycle.MODEL_REVISION,
        "model_inventory_sha256": _model_inventory_sha256(),
        "gpu_expected_count": 1,
        "gpu_expected_name": lifecycle.EXPECTED_GPU_NAME,
        "gpu_expected_driver": lifecycle.EXPECTED_DRIVER,
        "gpu_expected_memory_mib": lifecycle.EXPECTED_MEMORY_MIB,
        "rate_usd_per_hour": "0.500000",
        "total_cap_usd": "10.000000",
        "billing_started_at": lifecycle._canonical_timestamp(BILLING_STARTED_AT),
        "operational_cutoff": lifecycle._canonical_timestamp(
            BILLING_STARTED_AT
            + timedelta(minutes=lifecycle.STOP_STARTING_NEW_WORK_MINUTES)
        ),
        "cleanup_reserve_minutes": lifecycle.CLEANUP_RESERVE_MINUTES,
        "authorized_at": lifecycle._canonical_timestamp(
            BILLING_STARTED_AT - timedelta(minutes=5)
        ),
        "authorization_expiry": lifecycle._canonical_timestamp(
            BILLING_STARTED_AT + timedelta(hours=6)
        ),
        "automatic_retries": 0,
        "replacement_allowed": False,
        "nonce": VALID_NONCE,
    }
    payload.update(overrides)
    seal_input = {k: v for k, v in payload.items() if k != "authorization_sha256"}
    payload["authorization_sha256"] = lifecycle.build_authorization_seal(seal_input)
    return payload


def _authorization(**overrides: Any) -> lifecycle.RunAuthorization:
    return lifecycle.RunAuthorization.from_dict(_authorization_payload(**overrides))


class TestRunAuthorization:
    def test_valid_authorization_round_trips(self) -> None:
        auth = _authorization()
        assert auth.repository_head == VALID_HEAD
        assert auth.to_dict()["protocol_id"] == lifecycle.PROTOCOL_ID
        # Re-parsing its own to_dict() must succeed and reproduce the seal.
        again = lifecycle.RunAuthorization.from_dict(auth.to_dict())
        assert again.authorization_sha256 == auth.authorization_sha256

    def test_rejects_wrong_protocol_id(self) -> None:
        payload = _authorization_payload()
        payload["protocol_id"] = "wrong-protocol"
        payload["authorization_sha256"] = lifecycle.build_authorization_seal(
            {k: v for k, v in payload.items() if k != "authorization_sha256"}
        )
        with pytest.raises(lifecycle.HostOrchestrationError, match="protocol_id"):
            lifecycle.RunAuthorization.from_dict(payload)

    def test_rejects_wrong_model_inventory_digest(self) -> None:
        payload = _authorization_payload(model_inventory_sha256="0" * 64)
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="committed conversion manifest"
        ):
            lifecycle.RunAuthorization.from_dict(payload)

    def test_rejects_tampered_seal(self) -> None:
        payload = _authorization_payload()
        payload["total_cap_usd"] = "999999.000000"
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="authorization_sha256"
        ):
            lifecycle.RunAuthorization.from_dict(payload)

    def test_rejects_nonzero_retries(self) -> None:
        payload = _authorization_payload(automatic_retries=1)
        with pytest.raises(lifecycle.HostOrchestrationError, match="automatic_retries"):
            lifecycle.RunAuthorization.from_dict(payload)

    def test_rejects_replacement_allowed(self) -> None:
        payload = _authorization_payload(replacement_allowed=True)
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="replacement_allowed"
        ):
            lifecycle.RunAuthorization.from_dict(payload)

    def test_rejects_gpu_count_other_than_one(self) -> None:
        payload = _authorization_payload(gpu_expected_count=2)
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="gpu_expected_count"
        ):
            lifecycle.RunAuthorization.from_dict(payload)

    def test_accepts_explicit_operational_cutoff(self) -> None:
        cutoff = BILLING_STARTED_AT + timedelta(minutes=300)
        payload = _authorization_payload(
            operational_cutoff=lifecycle._canonical_timestamp(cutoff)
        )
        auth = lifecycle.RunAuthorization.from_dict(payload)
        assert auth.operational_cutoff == cutoff

    def test_rejects_operational_cutoff_beyond_absolute_cap(self) -> None:
        payload = _authorization_payload(
            operational_cutoff=lifecycle._canonical_timestamp(
                BILLING_STARTED_AT + timedelta(hours=20)
            ),
            authorization_expiry=lifecycle._canonical_timestamp(
                BILLING_STARTED_AT + timedelta(hours=21)
            ),
        )
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="absolute list-rate cap"
        ):
            lifecycle.RunAuthorization.from_dict(payload)

    def test_rejects_cap_below_mandatory_cost(self) -> None:
        payload = _authorization_payload(
            rate_usd_per_hour="100.000000", total_cap_usd="1.000000"
        )
        with pytest.raises(lifecycle.HostOrchestrationError, match="total_cap_usd"):
            lifecycle.RunAuthorization.from_dict(payload)

    def test_rejects_missing_key(self) -> None:
        payload = _authorization_payload()
        del payload["nonce"]
        with pytest.raises(lifecycle.HostOrchestrationError, match="required set"):
            lifecycle.RunAuthorization.from_dict(payload)

    def test_rejects_extra_key(self) -> None:
        payload = _authorization_payload()
        payload["unexpected_field"] = "value"
        with pytest.raises(lifecycle.HostOrchestrationError, match="required set"):
            lifecycle.RunAuthorization.from_dict(payload)

    def test_accepts_authorized_at_after_billing_started(self) -> None:
        payload = _authorization_payload(
            authorized_at=lifecycle._canonical_timestamp(
                BILLING_STARTED_AT + timedelta(minutes=1)
            )
        )
        auth = lifecycle.RunAuthorization.from_dict(payload)
        assert auth.authorized_at == BILLING_STARTED_AT + timedelta(minutes=1)

    def test_rejects_expiry_before_authorized_at(self) -> None:
        payload = _authorization_payload(
            authorization_expiry=lifecycle._canonical_timestamp(
                BILLING_STARTED_AT - timedelta(minutes=10)
            )
        )
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="authorization_expiry"
        ):
            lifecycle.RunAuthorization.from_dict(payload)

    def test_signature_fields_must_come_in_pairs(self) -> None:
        payload = _authorization_payload()
        payload["signature_path"] = "/tmp/nonexistent.sig"
        with pytest.raises(lifecycle.HostOrchestrationError, match="required set"):
            lifecycle.RunAuthorization.from_dict(payload)

    def test_signature_paths_are_preserved_but_not_public(self) -> None:
        payload = _authorization_payload()
        payload["signature_path"] = "/protected/authorization.sig"
        payload["authorized_signers_path"] = "/protected/authorized_signers"
        auth = lifecycle.RunAuthorization.from_dict(payload)
        assert auth.signature_path == Path("/protected/authorization.sig")
        assert auth.authorized_signers_path == Path("/protected/authorized_signers")
        assert "signature_path" not in auth.to_dict()
        assert "authorized_signers_path" not in auth.to_dict()

    def test_read_from_file(self, tmp_path: Path) -> None:
        payload = _authorization_payload()
        path = tmp_path / "authorization.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        auth = lifecycle.RunAuthorization.read(path)
        assert auth.nonce == VALID_NONCE

    def test_read_rejects_symlink(self, tmp_path: Path) -> None:
        real = tmp_path / "real.json"
        real.write_text(json.dumps(_authorization_payload()), encoding="utf-8")
        link = tmp_path / "link.json"
        link.symlink_to(real)
        with pytest.raises(lifecycle.HostOrchestrationError):
            lifecycle.RunAuthorization.read(link)


class TestBudgetLedger:
    def test_stage_duration_totals_are_fixed(self) -> None:
        assert lifecycle.STOP_STARTING_NEW_WORK_MINUTES == 175
        assert lifecycle.TEARDOWN_COMPLETE_MINUTES == 210
        assert lifecycle.CLEANUP_RESERVE_MINUTES == 35

    def test_cutoffs_derive_from_supplied_boot_time(self) -> None:
        operational = BILLING_STARTED_AT + timedelta(minutes=175)
        cutoffs = lifecycle.compute_cutoffs(
            BILLING_STARTED_AT,
            operational,
            35,
            Decimal("0.5"),
            Decimal("10"),
        )
        assert cutoffs.operational_cutoff == BILLING_STARTED_AT + timedelta(minutes=175)
        assert cutoffs.teardown_complete_cutoff == BILLING_STARTED_AT + timedelta(
            minutes=210
        )

    def test_stage_budget_ok_true_when_ample_reserve_remains(self) -> None:
        cutoffs = _authorization().cutoffs
        now = BILLING_STARTED_AT
        assert cutoffs.stage_budget_ok(now, 0) is True

    def test_stage_budget_ok_false_when_reserve_exhausted(self) -> None:
        cutoffs = _authorization().cutoffs
        now = BILLING_STARTED_AT + timedelta(minutes=209)
        assert cutoffs.stage_budget_ok(now, 0) is False

    def test_fresh_vm_absolute_cap_uses_decimal_and_rounds_down(self) -> None:
        boot = datetime(2026, 9, 6, 9, 43, 25, tzinfo=timezone.utc)
        cutoff = lifecycle.absolute_cap_cutoff(boot, Decimal("0.39"), Decimal("5.00"))
        assert lifecycle._canonical_timestamp(cutoff) == ("2026-09-06T22:32:38.846153Z")

    def test_no_module_level_absolute_datetime_constant_exists(self) -> None:
        """The historical prototype hardcoded an already-terminated VM's
        absolute boot/cutoff timestamps as module constants; this module
        must never do that again."""

        for name in dir(lifecycle):
            if name.startswith("_"):
                continue
            value = getattr(lifecycle, name)
            assert not isinstance(value, datetime), (
                f"{name} is an absolute datetime module constant; cutoffs "
                "must always be derived from a caller-supplied "
                "billing_started_at instead"
            )


class TestProtectedExecutionConfig:
    def _write_key(self, path: Path, mode: int = 0o600) -> None:
        path.write_text("fake-key", encoding="utf-8")
        os.chmod(path, mode)

    def _write_known_hosts(self, path: Path, mode: int = 0o600) -> None:
        path.write_text("example.com ssh-ed25519 AAAA", encoding="utf-8")
        os.chmod(path, mode)

    def _payload(self, tmp_path: Path) -> dict[str, Any]:
        key_path = tmp_path / "id_ed25519"
        known_hosts = tmp_path / "known_hosts"
        self._write_key(key_path)
        self._write_known_hosts(known_hosts)
        archive = tmp_path / "runner.tar.gz"
        archive.write_bytes(b"fake archive")
        return {
            "host": "198.51.100.10",
            "port": 57003,
            "user": "deploy",
            "private_key_path": str(key_path),
            "known_hosts_path": str(known_hosts),
            "remote_workspace": "/home/deploy/kv-truth-run",
            "authorized_key_marker": "kv-truth-authorized-key",
            "local_evidence_dir": str(tmp_path / "evidence"),
            "local_runner_archive": str(archive),
        }

    def test_loads_valid_config(self, tmp_path: Path) -> None:
        payload = self._payload(tmp_path)
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        os.chmod(config_path, 0o600)
        config = lifecycle.ProtectedExecutionConfig.load(config_path)
        assert config.host == "198.51.100.10"
        assert config.public_record() == {
            "schema_version": "1",
            "source": "protected_execution_config",
        }

    def test_public_record_never_leaks_sensitive_fields(self, tmp_path: Path) -> None:
        payload = self._payload(tmp_path)
        config = lifecycle.ProtectedExecutionConfig.from_dict(payload)
        record_text = json.dumps(config.public_record())
        for sensitive in (
            config.host,
            config.user,
            str(config.private_key_path),
            str(config.known_hosts_path),
            config.remote_workspace,
        ):
            assert sensitive not in record_text

    def test_rejects_world_readable_config_file(self, tmp_path: Path) -> None:
        payload = self._payload(tmp_path)
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(payload), encoding="utf-8")
        os.chmod(config_path, 0o644)
        with pytest.raises(lifecycle.HostOrchestrationError, match="group- or world"):
            lifecycle.ProtectedExecutionConfig.load(config_path)

    def test_rejects_insecure_private_key_permissions(self, tmp_path: Path) -> None:
        payload = self._payload(tmp_path)
        os.chmod(Path(payload["private_key_path"]), 0o644)
        with pytest.raises(lifecycle.HostOrchestrationError, match="0600"):
            lifecycle.ProtectedExecutionConfig.from_dict(payload)

    def test_rejects_symlinked_private_key(self, tmp_path: Path) -> None:
        payload = self._payload(tmp_path)
        real_key = Path(payload["private_key_path"])
        link_key = tmp_path / "id_ed25519_link"
        link_key.symlink_to(real_key)
        os.chmod(real_key, 0o600)
        payload["private_key_path"] = str(link_key)
        with pytest.raises(lifecycle.HostOrchestrationError, match="symlink"):
            lifecycle.ProtectedExecutionConfig.from_dict(payload)

    def test_rejects_world_readable_known_hosts(self, tmp_path: Path) -> None:
        payload = self._payload(tmp_path)
        os.chmod(Path(payload["known_hosts_path"]), 0o644)
        with pytest.raises(lifecycle.HostOrchestrationError, match="group- or world"):
            lifecycle.ProtectedExecutionConfig.from_dict(payload)

    def test_rejects_missing_key(self, tmp_path: Path) -> None:
        payload = self._payload(tmp_path)
        del payload["host"]
        with pytest.raises(lifecycle.HostOrchestrationError, match="required set"):
            lifecycle.ProtectedExecutionConfig.from_dict(payload)

    def test_rejects_unsafe_remote_workspace(self, tmp_path: Path) -> None:
        payload = self._payload(tmp_path)
        payload["remote_workspace"] = "/home/deploy/../etc"
        with pytest.raises(lifecycle.HostOrchestrationError, match=r"remote_workspace"):
            lifecycle.ProtectedExecutionConfig.from_dict(payload)

    def test_rejects_shell_metacharacters_in_remote_workspace(
        self, tmp_path: Path
    ) -> None:
        payload = self._payload(tmp_path)
        payload["remote_workspace"] = "/home/deploy/$( rm -rf / )"
        with pytest.raises(lifecycle.HostOrchestrationError, match=r"remote_workspace"):
            lifecycle.ProtectedExecutionConfig.from_dict(payload)


class TestStrictSSHOptions:
    def _config(self, tmp_path: Path) -> lifecycle.ProtectedExecutionConfig:
        key = tmp_path / "key"
        key.write_text("x", encoding="utf-8")
        os.chmod(key, 0o600)
        known_hosts = tmp_path / "known_hosts"
        known_hosts.write_text("x", encoding="utf-8")
        os.chmod(known_hosts, 0o600)
        return lifecycle.ProtectedExecutionConfig.from_dict(
            {
                "host": "203.0.113.5",
                "port": 57003,
                "user": "runner",
                "private_key_path": str(key),
                "known_hosts_path": str(known_hosts),
                "remote_workspace": "/srv/kv-truth",
                "authorized_key_marker": "marker",
                "local_evidence_dir": str(tmp_path / "evidence"),
                "local_runner_archive": str(tmp_path / "archive.tar"),
            }
        )

    def test_shared_options_forbid_forwarding_and_control_sockets(
        self, tmp_path: Path
    ) -> None:
        options = lifecycle.StrictSSHOptions(self._config(tmp_path))
        opts = options.shared_options()
        joined = " ".join(opts)
        assert "BatchMode=yes" in joined
        assert "IdentitiesOnly=yes" in joined
        assert "PasswordAuthentication=no" in joined
        assert "KbdInteractiveAuthentication=no" in joined
        assert "StrictHostKeyChecking=yes" in joined
        assert "ForwardAgent=no" in joined
        assert "ForwardX11=no" in joined
        assert "ControlMaster=no" in joined
        assert "ControlPath=none" in joined

    def test_ssh_command_shape(self, tmp_path: Path) -> None:
        options = lifecycle.StrictSSHOptions(self._config(tmp_path))
        argv = options.ssh_command("echo hi")
        assert argv[0] == "ssh"
        assert argv[1:3] == ("-p", "57003")
        assert argv[-2] == "runner@203.0.113.5"
        assert argv[-1] == "echo hi"

    def test_scp_uses_the_configured_nondefault_port(self, tmp_path: Path) -> None:
        options = lifecycle.StrictSSHOptions(self._config(tmp_path))
        upload = options.scp_upload_command(Path("source.tar"), "/srv/source.tar")
        download = options.scp_download_command("/srv/evidence.tar", Path("evidence"))
        assert upload[1:3] == ("-P", "57003")
        assert download[1:3] == ("-P", "57003")

    def test_public_record_omits_host_user_and_paths(self, tmp_path: Path) -> None:
        config = self._config(tmp_path)
        options = lifecycle.StrictSSHOptions(config)
        record_text = json.dumps(options.public_record())
        assert config.host not in record_text
        assert config.user not in record_text
        assert str(config.private_key_path) not in record_text
        assert str(config.known_hosts_path) not in record_text


@dataclass
class RecordedCall:
    argv: tuple[str, ...]
    description: str
    timeout: float
    input_text: str | None


def _preflight_stdout(**overrides: str) -> str:
    values = {
        "NOW_EPOCH": str(int(BILLING_STARTED_AT.timestamp())),
        "BOOT_EPOCH": str(int(BILLING_STARTED_AT.timestamp())),
        "GPU_COUNT": "1",
        "GPU": (
            f"{lifecycle.EXPECTED_GPU_NAME}, {lifecycle.EXPECTED_DRIVER}, "
            f"{lifecycle.EXPECTED_MEMORY_MIB}, "
            f"{lifecycle.EXPECTED_GPU_COMPUTE_CAPABILITY}"
        ),
        "GPU_PROCESS_COUNT": "0",
        "CONTAINER_COUNT": "0",
        "SUDO_NONINTERACTIVE": "1",
        "DISK_FREE_BYTES": str(lifecycle.MINIMUM_DISK_FREE_BYTES),
        "RAM_BYTES": str(lifecycle.MINIMUM_HOST_RAM_BYTES),
        "SWAP_USED_BYTES": "0",
        "PYTHON_VERSION": "3.12.3",
        "DOCKER_VERSION": "29.6.1",
        "HF_CLI": "/usr/local/bin/huggingface-cli",
    }
    values.update(overrides)
    return "\n".join(f"{key}={value}" for key, value in values.items())


@dataclass
class FakeCommandRunner:
    """Answers every argv the orchestrator can build with a plausible,
    stage-appropriate transcript -- never spawning a process or touching a
    network/GPU. ``responses`` lets a test override any stage's transcript
    to simulate a tampered/failed remote result.
    """

    calls: list[RecordedCall] = field(default_factory=list)
    responses: dict[str, lifecycle.CommandResult] = field(default_factory=dict)
    scp_download_target_bytes: bytes = field(
        default_factory=_build_fixture_evidence_tar_bytes
    )
    fail_stages: frozenset[str] = frozenset()

    def run(
        self,
        argv: Sequence[str],
        *,
        description: str,
        timeout: float,
        input_text: str | None = None,
    ) -> lifecycle.CommandResult:
        self.calls.append(
            RecordedCall(
                argv=tuple(argv),
                description=description,
                timeout=timeout,
                input_text=input_text,
            )
        )
        if description in self.fail_stages:
            return lifecycle.CommandResult(returncode=1, stdout="", stderr="denied")
        if description in self.responses:
            return self.responses[description]
        return self._default_response(argv, description)

    def _default_response(
        self, argv: Sequence[str], description: str
    ) -> lifecycle.CommandResult:
        if description == "stage_preflight":
            stdout = _preflight_stdout()
            return lifecycle.CommandResult(returncode=0, stdout=stdout, stderr="")
        if description == "stage_image_preparation":
            expected_digest = lifecycle.BASE_IMAGE_REFERENCE.split("@", 1)[-1]
            stdout = "\n".join(
                [
                    f'BASE_REPODIGESTS=["vllm/vllm-openai@{expected_digest}"]',
                    "BASE_IMAGE_ID=sha256:" + "1" * 64,
                    f"EXPECTED_HEAD={VALID_HEAD}",
                    f"DERIVED_IMAGE_ID={DERIVED_IMAGE_ID}",
                ]
            )
            return lifecycle.CommandResult(returncode=0, stdout=stdout, stderr="")
        if description == "stage_transfer_evidence_digest":
            digest = hashlib.sha256(self.scp_download_target_bytes).hexdigest()
            return lifecycle.CommandResult(
                returncode=0, stdout=f"{digest}  evidence.tar\n", stderr=""
            )
        if description == "stage_transfer_evidence_download":
            # Emulates ``scp`` by writing the fixture bytes to the local
            # destination path (the last argv element for a download).
            local_path = Path(argv[-1])
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(self.scp_download_target_bytes)
            return lifecycle.CommandResult(returncode=0, stdout="", stderr="")
        if description == "stage_teardown_cleanup":
            stdout = "RESIDUAL_CONTAINERS=0\nRESIDUAL_GPU_PROCESSES=0\n"
            return lifecycle.CommandResult(returncode=0, stdout=stdout, stderr="")
        return lifecycle.CommandResult(returncode=0, stdout="", stderr="")


def _config(tmp_path: Path) -> lifecycle.ProtectedExecutionConfig:
    key = tmp_path / "key"
    key.write_text("x", encoding="utf-8")
    os.chmod(key, 0o600)
    known_hosts = tmp_path / "known_hosts"
    known_hosts.write_text("x", encoding="utf-8")
    os.chmod(known_hosts, 0o600)
    archive = tmp_path / "archive.tar"
    if not archive.exists():
        _write_source_archive(archive)
    return lifecycle.ProtectedExecutionConfig.from_dict(
        {
            "host": "203.0.113.5",
            "port": 57003,
            "user": "runner",
            "private_key_path": str(key),
            "known_hosts_path": str(known_hosts),
            "remote_workspace": "/srv/kv-truth",
            "authorized_key_marker": "marker",
            "local_evidence_dir": str(tmp_path / "evidence"),
            "local_runner_archive": str(archive),
        }
    )


class TestRemoteOrchestratorFullRun:
    def test_full_run_never_touches_network_or_gpu_and_succeeds(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        outcomes = orchestrator.run(local_evidence_bundle_dir=tmp_path / "bundle")
        assert all(outcome.ok for outcome in outcomes)
        assert orchestrator.state == lifecycle.OrchestratorState.COMPLETE
        # Real, complete commands were issued for every mandatory stage --
        # not an always-refusing placeholder.
        descriptions = {call.description for call in runner.calls}
        assert "stage_preflight" in descriptions
        assert "stage_identity_gate" in descriptions
        assert "stage_model_acquisition_download" in descriptions
        assert "stage_model_acquisition_verify" in descriptions
        assert "stage_image_preparation" in descriptions
        assert "stage_canary" in descriptions
        assert any(d.startswith("stage_four_ab_pairs[pair1") for d in descriptions)
        assert any(d.startswith("stage_four_ab_pairs[pair4") for d in descriptions)
        assert "stage_eviction_lane" in descriptions
        assert "stage_transfer_evidence_archive" in descriptions
        assert "stage_teardown_cleanup" in descriptions
        assert "stage_teardown_shutdown" in descriptions

        # The private/public evidence bundle wiring actually ran: a private
        # bundle with a real teardown receipt and a portably-verifiable
        # public-redacted bundle directory were both written to disk.
        bundle_dir = tmp_path / "bundle"
        private = evidence.PrivateEvidenceBundle.read(
            bundle_dir / "private_bundle.json"
        )
        assert private.run_mode == evidence.RUN_MODE_REAL_RUN
        assert private.teardown_receipt is not None
        assert private.teardown_receipt.residual_containers == 0
        assert private.teardown_receipt.safe_to_terminate_message_emitted is True
        # Four B lanes' ten nested probes plus the one eviction claim.
        assert len(private.claim_matrix) == 4 * 10 + 1
        salt_entries = [
            entry
            for entry in private.claim_matrix
            if entry.scenario == "namespace_isolation"
        ]
        assert len(salt_entries) == 4
        assert all(
            entry.verdict == evidence.VERDICT_UNSUPPORTED for entry in salt_entries
        )
        public = evidence.verify_public_bundle_directory(bundle_dir / "public")
        assert public.to_dict()["run_mode"] == evidence.RUN_MODE_REAL_RUN
        evidence.assert_publication_safe(public.to_dict())

    def test_ab_pairs_use_the_fixed_ab_ba_ba_ab_order(self, tmp_path: Path) -> None:
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        orchestrator.stage_preflight()
        orchestrator.stage_identity_gate()
        orchestrator.stage_model_acquisition()
        orchestrator.stage_image_preparation()
        orchestrator.stage_canary()
        orchestrator.stage_four_ab_pairs()
        tags = [
            call.description.split("[", 1)[1].rstrip("]")
            for call in runner.calls
            if call.description.startswith("stage_four_ab_pairs")
        ]
        lanes = [tag.rsplit("-", 1)[-1] for tag in tags]
        assert lanes == ["a", "b", "b", "a", "b", "a", "a", "b"]

    def test_all_lane_containers_run_with_network_none_and_run_label(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner()
        auth = _authorization()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=auth,
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        orchestrator.stage_preflight()
        orchestrator.stage_identity_gate()
        orchestrator.stage_model_acquisition()
        orchestrator.stage_image_preparation()
        orchestrator.stage_canary()
        orchestrator.stage_four_ab_pairs()
        orchestrator.stage_eviction_lane()
        lane_calls = [
            call
            for call in runner.calls
            if call.description in {"stage_canary", "stage_eviction_lane"}
            or call.description.startswith("stage_four_ab_pairs")
        ]
        assert len(lane_calls) == 1 + 8 + 1
        for call in lane_calls:
            command_string = call.argv[-1]
            assert "--network none" in command_string
            assert lifecycle.docker_run_label(auth.nonce) in command_string
            assert "HF_HUB_OFFLINE=1" in command_string
            assert (
                f"EXPECTED_REPOSITORY_COMMIT={auth.repository_head}" in command_string
            )
            from vllm_kv_truth.runner import (
                VLLM_SOURCE_COMMIT,
            )

            assert f"EXPECTED_VLLM_SOURCE_COMMIT={VLLM_SOURCE_COMMIT}" in command_string
            assert f"EXPECTED_IMAGE_ID={DERIVED_IMAGE_ID}" in command_string

        canary = next(call for call in lane_calls if call.description == "stage_canary")
        assert "--lane B" in canary.argv[-1]

    def test_image_build_uses_checked_containerfile_and_source_context(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        orchestrator.stage_image_preparation()
        call = next(
            call
            for call in runner.calls
            if call.description == "stage_image_preparation"
        )
        script = call.input_text or ""
        assert "-f containers/vllm-kv-truth/Containerfile" in script
        assert f"--build-arg RUNNER_COMMIT={VALID_HEAD}" in script
        assert "COPY" not in script
        assert orchestrator.derived_image_id == DERIVED_IMAGE_ID

    def test_teardown_only_targets_run_scoped_label(self, tmp_path: Path) -> None:
        runner = FakeCommandRunner()
        auth = _authorization()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=auth,
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        orchestrator.stage_teardown(tmp_path / "bundle")
        cleanup_call = next(
            call
            for call in runner.calls
            if call.description == "stage_teardown_cleanup"
        )
        script = cleanup_call.input_text or ""
        assert (
            "docker ps -q" not in script.replace("docker ps -q --filter", "PLACEHOLDER")
            or "--filter" in script
        )
        # Never a blanket, unfiltered container listing/removal.
        for line in script.splitlines():
            if line.strip().startswith("docker ps") or line.strip().startswith(
                "docker images"
            ):
                assert "--filter" in line
                assert lifecycle.docker_run_label(auth.nonce) in line

    def test_teardown_emits_safe_to_terminate_message(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        orchestrator.stage_teardown(tmp_path / "bundle")
        captured = capsys.readouterr()
        assert "SAFE TO TERMINATE INSTANCE NOW" in captured.out

    def test_teardown_never_claims_provider_deletion(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        orchestrator.stage_teardown(tmp_path / "bundle")
        captured = capsys.readouterr()
        lowered = captured.out.lower()
        assert "instance deleted" not in lowered
        assert "instance terminated" not in lowered

    def test_teardown_still_runs_when_an_earlier_stage_fails(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner(fail_stages=frozenset({"stage_canary"}))
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(lifecycle.HostOrchestrationError):
            orchestrator.run(local_evidence_bundle_dir=tmp_path / "bundle")
        descriptions = {call.description for call in runner.calls}
        assert "stage_teardown_cleanup" in descriptions
        assert orchestrator.state == lifecycle.OrchestratorState.COMPLETE

    def test_teardown_still_runs_on_keyboard_interrupt(self, tmp_path: Path) -> None:
        class _InterruptingRunner(FakeCommandRunner):
            def run(
                self,
                argv: Sequence[str],
                *,
                description: str,
                timeout: float,
                input_text: str | None = None,
            ) -> lifecycle.CommandResult:
                if description == "stage_canary":
                    raise KeyboardInterrupt
                return super().run(
                    argv,
                    description=description,
                    timeout=timeout,
                    input_text=input_text,
                )

        runner = _InterruptingRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(KeyboardInterrupt):
            orchestrator.run(local_evidence_bundle_dir=tmp_path / "bundle")
        descriptions = {call.description for call in runner.calls}
        assert "stage_teardown_cleanup" in descriptions

    def test_refuses_to_start_when_budget_reserve_is_exhausted(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT + timedelta(minutes=209),
        )
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="insufficient reserve"
        ):
            orchestrator.stage_preflight()
        assert runner.calls == []

    def test_refuses_when_authorization_expired(self, tmp_path: Path) -> None:
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT + timedelta(hours=100),
        )
        with pytest.raises(lifecycle.HostOrchestrationError, match="validity window"):
            orchestrator.stage_preflight()

    def test_preflight_rejects_wrong_gpu_name(self, tmp_path: Path) -> None:
        runner = FakeCommandRunner(
            responses={
                "stage_preflight": lifecycle.CommandResult(
                    returncode=0,
                    stdout=_preflight_stdout(
                        GPU=(
                            "Wrong GPU, 580.159.03, 24564, "
                            f"{lifecycle.EXPECTED_GPU_COMPUTE_CAPABILITY}"
                        )
                    ),
                    stderr="",
                )
            }
        )
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(lifecycle.HostOrchestrationError, match="GPU name"):
            orchestrator.stage_preflight()

    def test_preflight_rejects_wrong_gpu_count(self, tmp_path: Path) -> None:
        runner = FakeCommandRunner(
            responses={
                "stage_preflight": lifecycle.CommandResult(
                    returncode=0,
                    stdout=_preflight_stdout(GPU_COUNT="2"),
                    stderr="",
                )
            }
        )
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(lifecycle.HostOrchestrationError, match="GPU count"):
            orchestrator.stage_preflight()

    @pytest.mark.parametrize(
        ("marker", "value", "message"),
        [
            ("GPU_PROCESS_COUNT", "1", "GPU process"),
            ("CONTAINER_COUNT", "1", "existing container"),
            ("SWAP_USED_BYTES", "1", "swap"),
            ("RAM_BYTES", "1", "RAM"),
            ("DISK_FREE_BYTES", "1", "disk"),
        ],
    )
    def test_preflight_rejects_nonquiescent_or_undersized_host(
        self, tmp_path: Path, marker: str, value: str, message: str
    ) -> None:
        runner = FakeCommandRunner(
            responses={
                "stage_preflight": lifecycle.CommandResult(
                    returncode=0,
                    stdout=_preflight_stdout(**{marker: value}),
                    stderr="",
                )
            }
        )
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(lifecycle.HostOrchestrationError, match=message):
            orchestrator.stage_preflight()

    def test_image_preparation_rejects_wrong_base_digest(self, tmp_path: Path) -> None:
        runner = FakeCommandRunner(
            responses={
                "stage_image_preparation": lifecycle.CommandResult(
                    returncode=0,
                    stdout="\n".join(
                        [
                            'BASE_REPODIGESTS=["vllm/vllm-openai@sha256:'
                            + "9" * 64
                            + '"]',
                            "BASE_IMAGE_ID=sha256:" + "1" * 64,
                            f"EXPECTED_HEAD={VALID_HEAD}",
                            f"DERIVED_IMAGE_ID={DERIVED_IMAGE_ID}",
                        ]
                    ),
                    stderr="",
                )
            }
        )
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(lifecycle.HostOrchestrationError, match="RepoDigests"):
            orchestrator.stage_image_preparation()

    def test_image_preparation_rejects_wrong_derived_image_id(
        self, tmp_path: Path
    ) -> None:
        expected_digest = lifecycle.BASE_IMAGE_REFERENCE.split("@", 1)[-1]
        runner = FakeCommandRunner(
            responses={
                "stage_image_preparation": lifecycle.CommandResult(
                    returncode=0,
                    stdout="\n".join(
                        [
                            f'BASE_REPODIGESTS=["vllm/vllm-openai@{expected_digest}"]',
                            "BASE_IMAGE_ID=sha256:" + "1" * 64,
                            f"EXPECTED_HEAD={VALID_HEAD}",
                            "DERIVED_IMAGE_ID=sha256:" + "1" * 64,
                        ]
                    ),
                    stderr="",
                )
            }
        )
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(lifecycle.HostOrchestrationError, match="derived image id"):
            orchestrator.stage_image_preparation()

    def test_image_preparation_rejects_archive_with_wrong_embedded_commit(
        self, tmp_path: Path
    ) -> None:
        """The checked source archive's COMMIT_HEAD marker must match the
        authorization's repository_head -- verified purely locally, before
        anything is staged onto the remote host."""

        archive = tmp_path / "archive.tar"
        _write_source_archive(archive, commit="f" * 40)
        config = _config(tmp_path)
        object.__setattr__(config, "local_runner_archive", archive)
        orchestrator = lifecycle.RemoteOrchestrator(
            config=config,
            authorization=_authorization(),
            runner=FakeCommandRunner(),
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(lifecycle.HostOrchestrationError, match="repository_head"):
            orchestrator.stage_image_preparation()

    def test_image_preparation_rejects_archive_digest_not_authorized(
        self, tmp_path: Path
    ) -> None:
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(
                derived_image_source_digest="sha256:" + "0" * 64
            ),
            runner=FakeCommandRunner(),
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="derived_image_source_digest"
        ):
            orchestrator.stage_image_preparation()

    def test_image_preparation_rejects_remote_confirmed_wrong_head(
        self, tmp_path: Path
    ) -> None:
        """Even if the local archive check somehow passed, a remote-echoed
        commit that disagrees with the authorization must still fail."""

        expected_digest = lifecycle.BASE_IMAGE_REFERENCE.split("@", 1)[-1]
        runner = FakeCommandRunner(
            responses={
                "stage_image_preparation": lifecycle.CommandResult(
                    returncode=0,
                    stdout="\n".join(
                        [
                            f'BASE_REPODIGESTS=["vllm/vllm-openai@{expected_digest}"]',
                            "BASE_IMAGE_ID=sha256:" + "1" * 64,
                            f"EXPECTED_HEAD={'f' * 40}",
                            f"DERIVED_IMAGE_ID={DERIVED_IMAGE_ID}",
                        ]
                    ),
                    stderr="",
                )
            }
        )
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(lifecycle.HostOrchestrationError, match="repository_head"):
            orchestrator.stage_image_preparation()

    def test_image_preparation_rejects_archive_with_unsafe_member_path(
        self, tmp_path: Path
    ) -> None:
        archive = tmp_path / "archive.tar"
        with tarfile.open(archive, "w") as tar:
            data = (VALID_HEAD + "\n").encode("ascii")
            head_info = tarfile.TarInfo(name="COMMIT_HEAD")
            head_info.size = len(data)
            tar.addfile(head_info, fileobj=io.BytesIO(data))
            evil_info = tarfile.TarInfo(name="../../etc/passwd")
            evil_info.size = 0
            tar.addfile(evil_info, fileobj=io.BytesIO(b""))
        config = _config(tmp_path)
        object.__setattr__(config, "local_runner_archive", archive)
        orchestrator = lifecycle.RemoteOrchestrator(
            config=config,
            authorization=_authorization(),
            runner=FakeCommandRunner(),
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(lifecycle.HostOrchestrationError, match="unsafe member"):
            orchestrator.stage_image_preparation()

    def test_image_preparation_rejects_archive_missing_commit_marker(
        self, tmp_path: Path
    ) -> None:
        archive = tmp_path / "archive.tar"
        with tarfile.open(archive, "w") as tar:
            data = b"nothing"
            info = tarfile.TarInfo(name="README")
            info.size = len(data)
            tar.addfile(info, fileobj=io.BytesIO(data))
        config = _config(tmp_path)
        object.__setattr__(config, "local_runner_archive", archive)
        orchestrator = lifecycle.RemoteOrchestrator(
            config=config,
            authorization=_authorization(),
            runner=FakeCommandRunner(),
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="COMMIT_HEAD marker"
        ):
            orchestrator.stage_image_preparation()

    def test_image_preparation_uploads_the_checked_source_archive(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        orchestrator.stage_image_preparation()
        upload_calls = [
            call
            for call in runner.calls
            if call.description == "stage_image_preparation_upload_source"
        ]
        assert len(upload_calls) == 1
        assert upload_calls[0].argv[0] == "scp"

    def test_transfer_evidence_rejects_tampered_download(self, tmp_path: Path) -> None:
        runner = FakeCommandRunner()
        # Digest response commits to different bytes than what "scp" later
        # writes locally -- simulating a tampered-in-transit archive.
        runner.responses["stage_transfer_evidence_digest"] = lifecycle.CommandResult(
            returncode=0, stdout=("0" * 64) + "  evidence.tar\n", stderr=""
        )
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(lifecycle.HostOrchestrationError, match="does not match"):
            orchestrator.stage_transfer_evidence(tmp_path / "bundle")

    def test_transfer_evidence_rejects_tampered_lane_receipt(
        self, tmp_path: Path
    ) -> None:
        # The outer archive digest matches (no in-transit tampering of the
        # tar itself), but one lane receipt inside it was replaced with a
        # forged one whose own canonical seal does not verify -- this must
        # be caught by the local receipt-level re-verification, not just
        # the archive-level digest check.
        tampered = json.loads(
            _fixture_lane_receipt_bytes(
                lane="B", records=_fixture_b_lane_records()
            ).decode("utf-8")
        )
        tampered["lane_result"]["records"][0]["num_cached_tokens"] = 999999
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            data = (json.dumps(tampered) + "\n").encode("utf-8")
            info = tarfile.TarInfo(name="evidence/pair1-slot2-b.json")
            info.size = len(data)
            tar.addfile(info, fileobj=io.BytesIO(data))
        runner = FakeCommandRunner(scp_download_target_bytes=buffer.getvalue())
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="failed local verification"
        ):
            orchestrator.stage_transfer_evidence(tmp_path / "bundle")

    def test_transfer_evidence_rejects_missing_lane_receipt(
        self, tmp_path: Path
    ) -> None:
        # A well-formed but incomplete evidence archive (e.g. one lane's
        # container never produced its output file) must fail cleanly, not
        # silently score an incomplete claim matrix.
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            data = _fixture_lane_receipt_bytes(
                lane="A", records=_fixture_b_lane_records()
            )
            info = tarfile.TarInfo(name="evidence/canary.json")
            info.size = len(data)
            tar.addfile(info, fileobj=io.BytesIO(data))
        runner = FakeCommandRunner(scp_download_target_bytes=buffer.getvalue())
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="failed local verification"
        ):
            orchestrator.stage_transfer_evidence(tmp_path / "bundle")

    def test_transfer_evidence_rejects_unsafe_archive_member_path(
        self, tmp_path: Path
    ) -> None:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            data = b"malicious"
            info = tarfile.TarInfo(name="../../etc/passwd")
            info.size = len(data)
            tar.addfile(info, fileobj=io.BytesIO(data))
        runner = FakeCommandRunner(scp_download_target_bytes=buffer.getvalue())
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(lifecycle.HostOrchestrationError, match="unsafe member"):
            orchestrator.stage_transfer_evidence(tmp_path / "bundle")

    def test_teardown_skips_finalization_when_transfer_never_ran(
        self, tmp_path: Path
    ) -> None:
        # If an earlier stage fails before stage_transfer_evidence ever
        # writes a private bundle, teardown must still clean up and shut
        # down without crashing on a missing bundle file.
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        bundle_dir = tmp_path / "bundle"
        outcome = orchestrator.stage_teardown(bundle_dir)
        assert outcome.ok
        assert not (bundle_dir / "private_bundle.json").exists()
        assert not (bundle_dir / "public").exists()

    def test_teardown_rejects_residual_containers(self, tmp_path: Path) -> None:
        runner = FakeCommandRunner(
            responses={
                "stage_teardown_cleanup": lifecycle.CommandResult(
                    returncode=0,
                    stdout="RESIDUAL_CONTAINERS=1\nRESIDUAL_GPU_PROCESSES=0\n",
                    stderr="",
                )
            }
        )
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(lifecycle.HostOrchestrationError, match="residual"):
            orchestrator.stage_teardown(tmp_path / "bundle")

    def test_a_failed_remote_stage_raises_without_leaking_stderr(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner(fail_stages=frozenset({"stage_preflight"}))
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(lifecycle.HostOrchestrationError) as excinfo:
            orchestrator.stage_preflight()
        assert "denied" not in str(excinfo.value)


class TestNoNetworkOrProcessSpawned:
    def test_full_run_never_calls_subprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess as subprocess_module

        def _forbidden(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("subprocess must never be invoked by tests")

        monkeypatch.setattr(subprocess_module, "run", _forbidden)
        monkeypatch.setattr(subprocess_module, "Popen", _forbidden)
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        orchestrator.run(local_evidence_bundle_dir=tmp_path / "bundle")


class TestModelInventoryVerificationScript:
    def test_script_embeds_every_manifest_file_and_hash(self) -> None:
        paths = lifecycle.RunPaths("/srv/kv-truth")
        script = lifecycle.build_model_inventory_verification_script(paths)
        manifest = json.loads(
            lifecycle.MODEL_CONVERSION_MANIFEST_PATH.read_text(encoding="utf-8")
        )
        files = manifest["source"]["files"]
        assert len(files) == lifecycle.EXPECTED_MODEL_FILE_COUNT
        for entry in files:
            assert entry["sha256"] in script
        assert f"TOTAL_BYTES={lifecycle.EXPECTED_MODEL_BYTES}" in script
        assert f"TOTAL_FILES={lifecycle.EXPECTED_MODEL_FILE_COUNT}" in script

    def test_script_uses_sha256sum_dash_c(self) -> None:
        paths = lifecycle.RunPaths("/srv/kv-truth")
        script = lifecycle.build_model_inventory_verification_script(paths)
        assert "sha256sum -c -" in script


class TestSubprocessCommandRunnerCredentialGuard:
    def test_rejects_credential_shaped_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SOME_API_KEY", "leaked")
        with pytest.raises(lifecycle.HostOrchestrationError, match="credential-shaped"):
            lifecycle.SubprocessCommandRunner()

    def test_rejects_forbidden_routing_variable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("DOCKER_HOST", "tcp://evil:2375")
        with pytest.raises(lifecycle.HostOrchestrationError, match="command-routing"):
            lifecycle.SubprocessCommandRunner()
