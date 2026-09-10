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
import shutil
import stat
import subprocess
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
            timing=runner_mod.RequestTiming(
                queued_ts=1.0,
                scheduled_ts=1.1,
                first_token_ts=1.2,
                last_token_ts=1.3,
                first_token_latency=0.2,
                finished_request_stats=None,
                null_reasons=("finished_request_stats_unavailable",),
            ),
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
            timing=runner_mod.RequestTiming(
                queued_ts=1.0,
                scheduled_ts=1.1,
                first_token_ts=1.2,
                last_token_ts=1.3,
                first_token_latency=0.2,
                finished_request_stats=None,
                null_reasons=("finished_request_stats_unavailable",),
            ),
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
        "model_download_package": lifecycle.DOWNLOADER_PACKAGE,
        "model_download_version": lifecycle.DOWNLOADER_VERSION,
        "model_download_interface": lifecycle.DOWNLOADER_INTERFACE,
        "model_download_source": lifecycle.DOWNLOADER_SOURCE,
        "gpu_expected_count": 1,
        "gpu_expected_name": lifecycle.EXPECTED_GPU_NAME,
        "gpu_expected_driver": lifecycle.EXPECTED_DRIVER,
        "gpu_expected_memory_mib": lifecycle.EXPECTED_MEMORY_MIB,
        "docker_execution_mode": lifecycle.DockerExecutionMode.DIRECT.value,
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
    if "docker_execution_config_sha256" not in payload:
        try:
            mode = lifecycle.DockerExecutionMode.parse(payload["docker_execution_mode"])
        except lifecycle.HostOrchestrationError:
            payload["docker_execution_config_sha256"] = "0" * 64
        else:
            payload["docker_execution_config_sha256"] = (
                lifecycle.docker_execution_config_sha256(mode)
            )
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

    def test_rejects_tampered_docker_execution_mode(self) -> None:
        payload = _authorization_payload()
        payload["docker_execution_mode"] = (
            lifecycle.DockerExecutionMode.SUDO_NONINTERACTIVE.value
        )
        with pytest.raises(
            lifecycle.HostOrchestrationError,
            match="docker_execution_config_sha256",
        ):
            lifecycle.RunAuthorization.from_dict(payload)

    def test_rejects_unknown_docker_execution_mode_with_valid_seal(self) -> None:
        payload = _authorization_payload(docker_execution_mode="auto")
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="docker_execution_mode"
        ):
            lifecycle.RunAuthorization.from_dict(payload)

    def test_rejects_tampered_docker_execution_config_hash(self) -> None:
        payload = _authorization_payload(docker_execution_config_sha256="0" * 64)
        with pytest.raises(
            lifecycle.HostOrchestrationError,
            match="docker_execution_config_sha256",
        ):
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

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("signature_path", "relative.sig"),
            ("authorized_signers_path", "relative-signers"),
            ("signature_path", "/protected/../authorization.sig"),
            ("authorized_signers_path", "/protected/../authorized_signers"),
            ("signature_path", "/protected//authorization.sig"),
            ("authorized_signers_path", "/protected/authorized_signers/"),
        ],
    )
    def test_signature_paths_must_be_unambiguous_absolute_paths(
        self, field: str, value: str
    ) -> None:
        payload = _authorization_payload()
        payload["signature_path"] = "/protected/authorization.sig"
        payload["authorized_signers_path"] = "/protected/authorized_signers"
        payload[field] = value
        with pytest.raises(
            lifecycle.HostOrchestrationError,
            match=f"{field} must be an unambiguous absolute path",
        ):
            lifecycle.RunAuthorization.from_dict(payload)

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

    def _payload(
        self,
        tmp_path: Path,
        *,
        docker_execution_mode: str = lifecycle.DockerExecutionMode.DIRECT.value,
    ) -> dict[str, Any]:
        key_path = tmp_path / "id_ed25519"
        known_hosts = tmp_path / "known_hosts"
        self._write_key(key_path)
        self._write_known_hosts(known_hosts)
        archive = tmp_path / "runner.tar.gz"
        archive.write_bytes(b"fake archive")
        return {
            "schema_version": lifecycle.PROTECTED_CONFIG_SCHEMA_VERSION,
            "host": "198.51.100.10",
            "port": 57003,
            "user": "deploy",
            "docker_execution_mode": docker_execution_mode,
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
            "schema_version": lifecycle.PROTECTED_CONFIG_SCHEMA_VERSION,
            "source": "protected_execution_config",
            "docker_execution_mode": lifecycle.DockerExecutionMode.DIRECT.value,
            "docker_execution_config_sha256": (
                lifecycle.docker_execution_config_sha256(
                    lifecycle.DockerExecutionMode.DIRECT
                )
            ),
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
        config_path.chmod(  # codeql[py/overly-permissive-file]
            config_path.stat().st_mode | stat.S_IRGRP
        )
        with pytest.raises(lifecycle.HostOrchestrationError, match="group- or world"):
            lifecycle.ProtectedExecutionConfig.load(config_path)

    def test_rejects_insecure_private_key_permissions(self, tmp_path: Path) -> None:
        payload = self._payload(tmp_path)
        private_key_path = Path(payload["private_key_path"])
        private_key_path.chmod(  # codeql[py/overly-permissive-file]
            private_key_path.stat().st_mode | stat.S_IRGRP
        )
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
        known_hosts_path = Path(payload["known_hosts_path"])
        known_hosts_path.chmod(  # codeql[py/overly-permissive-file]
            known_hosts_path.stat().st_mode | stat.S_IRGRP
        )
        with pytest.raises(lifecycle.HostOrchestrationError, match="group- or world"):
            lifecycle.ProtectedExecutionConfig.from_dict(payload)

    @pytest.mark.parametrize(
        ("field", "token"),
        [
            ("private_key_path", "%h"),
            ("private_key_path", "$HOME"),
            ("private_key_path", "second file"),
            ("private_key_path", '"quoted"'),
            ("private_key_path", r"back\slash"),
            ("private_key_path", "~user"),
            ("known_hosts_path", "%h"),
            ("known_hosts_path", "$HOME"),
            ("known_hosts_path", "second file"),
            ("known_hosts_path", '"quoted"'),
            ("known_hosts_path", r"back\slash"),
            ("known_hosts_path", "~user"),
        ],
    )
    def test_rejects_paths_reinterpreted_by_openssh(
        self, tmp_path: Path, field: str, token: str
    ) -> None:
        payload = self._payload(tmp_path)
        payload[field] = f"{tmp_path}/{token}"
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="interpreted by OpenSSH"
        ):
            lifecycle.ProtectedExecutionConfig.from_dict(payload)

    @pytest.mark.parametrize(
        "field",
        [
            "private_key_path",
            "known_hosts_path",
            "local_evidence_dir",
            "local_runner_archive",
        ],
    )
    def test_rejects_normalized_local_path_spellings(
        self, tmp_path: Path, field: str
    ) -> None:
        payload = self._payload(tmp_path)
        payload[field] = str(payload[field]).replace("/", "//", 1)
        with pytest.raises(
            lifecycle.HostOrchestrationError,
            match=f"{field} must be an unambiguous absolute path",
        ):
            lifecycle.ProtectedExecutionConfig.from_dict(payload)

    def test_rejects_missing_key(self, tmp_path: Path) -> None:
        payload = self._payload(tmp_path)
        del payload["host"]
        with pytest.raises(lifecycle.HostOrchestrationError, match="required set"):
            lifecycle.ProtectedExecutionConfig.from_dict(payload)

    @pytest.mark.parametrize("mode", ["", "sudo", "auto", 1, None])
    def test_rejects_unexpected_docker_execution_mode(
        self, tmp_path: Path, mode: Any
    ) -> None:
        payload = self._payload(tmp_path)
        payload["docker_execution_mode"] = mode
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="docker_execution_mode"
        ):
            lifecycle.ProtectedExecutionConfig.from_dict(payload)

    def test_rejects_legacy_config_without_explicit_mode(self, tmp_path: Path) -> None:
        payload = self._payload(tmp_path)
        del payload["docker_execution_mode"]
        with pytest.raises(lifecycle.HostOrchestrationError, match="required set"):
            lifecycle.ProtectedExecutionConfig.from_dict(payload)

    def test_rejects_legacy_config_schema(self, tmp_path: Path) -> None:
        payload = self._payload(tmp_path)
        payload["schema_version"] = "1"
        with pytest.raises(lifecycle.HostOrchestrationError, match="schema_version"):
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
                "schema_version": lifecycle.PROTECTED_CONFIG_SCHEMA_VERSION,
                "host": "203.0.113.5",
                "port": 57003,
                "user": "runner",
                "docker_execution_mode": lifecycle.DockerExecutionMode.DIRECT.value,
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
        assert opts[:2] == ("-F", "/dev/null")
        assert "BatchMode=yes" in joined
        assert "IdentitiesOnly=yes" in joined
        assert "PasswordAuthentication=no" in joined
        assert "KbdInteractiveAuthentication=no" in joined
        assert "StrictHostKeyChecking=yes" in joined
        assert "GlobalKnownHostsFile=/dev/null" in joined
        assert "ForwardAgent=no" in joined
        assert "ForwardX11=no" in joined
        assert "ProxyCommand=none" in joined
        assert "ProxyJump=none" in joined
        assert "PermitLocalCommand=no" in joined
        assert "KnownHostsCommand=none" in joined
        assert "IdentityAgent=none" in joined
        assert "PKCS11Provider=none" in joined
        assert "ControlMaster=no" in joined
        assert "ControlPath=none" in joined

    def test_ssh_command_shape(self, tmp_path: Path) -> None:
        options = lifecycle.StrictSSHOptions(self._config(tmp_path))
        argv = options.ssh_command("echo hi")
        assert argv[0] == "ssh"
        assert argv[1:3] == ("-p", "57003")
        assert argv[-2] == "runner@203.0.113.5"
        assert argv[-1] == "echo hi"

    def test_openssh_effective_config_disables_global_known_hosts(
        self, tmp_path: Path
    ) -> None:
        ssh = shutil.which("ssh")
        if ssh is None:
            pytest.skip("OpenSSH client is unavailable")
        options = lifecycle.StrictSSHOptions(self._config(tmp_path))
        completed = subprocess.run(
            [ssh, "-G", *options.shared_options(), "runner@203.0.113.5"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        effective = {
            line.split(maxsplit=1)[0]: line.split(maxsplit=1)[1]
            for line in completed.stdout.lower().splitlines()
            if len(line.split(maxsplit=1)) == 2
        }
        assert effective["globalknownhostsfile"] == "/dev/null"

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


class TestDockerCommand:
    def test_direct_mode_builds_exact_argv(self) -> None:
        docker = lifecycle.DockerCommand(lifecycle.DockerExecutionMode.DIRECT)
        assert docker.argv("ps", "-q") == ("docker", "ps", "-q")
        assert docker.shell("ps", "-q") == "docker ps -q"

    def test_sudo_mode_builds_exact_noninteractive_argv(self) -> None:
        docker = lifecycle.DockerCommand(
            lifecycle.DockerExecutionMode.SUDO_NONINTERACTIVE
        )
        assert docker.argv("ps", "-q") == (
            "sudo",
            "-n",
            "--",
            "docker",
            "ps",
            "-q",
        )
        assert docker.shell("ps", "-q") == "sudo -n -- docker ps -q"
        assert docker.xargs_shell("rm", "-f") == ("xargs -r sudo -n -- docker rm -f")


@dataclass
class RecordedCall:
    argv: tuple[str, ...]
    description: str
    timeout: float
    input_text: str | None


def _preflight_stdout(**overrides: str) -> str:
    values = {
        "OS_NAME": "Linux",
        "NOW_EPOCH": str(int(BILLING_STARTED_AT.timestamp())),
        "BOOT_EPOCH": str(int(BILLING_STARTED_AT.timestamp())),
        "GPU_COUNT": "1",
        "GPU": (
            f"{lifecycle.EXPECTED_GPU_NAME}, {lifecycle.EXPECTED_DRIVER}, "
            f"{lifecycle.EXPECTED_MEMORY_MIB}, "
            f"{lifecycle.EXPECTED_GPU_COMPUTE_CAPABILITY}"
        ),
        "GPU_PROCESS_COUNT": "0",
        "DOCKER_EXECUTION_MODE": lifecycle.DockerExecutionMode.DIRECT.value,
        "DOCKER_EXECUTION_CONFIG_SHA256": (
            lifecycle.docker_execution_config_sha256(
                lifecycle.DockerExecutionMode.DIRECT
            )
        ),
        "CONTAINER_COUNT": "0",
        "SUDO_NONINTERACTIVE": "1",
        "DISK_FREE_BYTES": str(lifecycle.MINIMUM_DISK_FREE_BYTES),
        "RAM_BYTES": str(lifecycle.MINIMUM_HOST_RAM_BYTES),
        "SWAP_USED_BYTES": "0",
        "PYTHON_VERSION": "3.12.3",
        "DOCKER_VERSION": "29.6.1",
    }
    values.update(overrides)
    return "\n".join(f"{key}={value}" for key, value in values.items())


def _image_preparation_stdout(**overrides: str) -> str:
    expected_digest = lifecycle.BASE_IMAGE_REFERENCE.split("@", 1)[-1]
    markers = {
        "BASE_REPODIGESTS": f'["vllm/vllm-openai@{expected_digest}"]',
        "BASE_IMAGE_ID": "sha256:" + "1" * 64,
        "EXPECTED_HEAD": VALID_HEAD,
        "DERIVED_IMAGE_ID": DERIVED_IMAGE_ID,
        "DOWNLOADER_PACKAGE": lifecycle.DOWNLOADER_PACKAGE,
        "DOWNLOADER_VERSION": lifecycle.DOWNLOADER_VERSION,
        "DOWNLOADER_INTERFACE": lifecycle.DOWNLOADER_INTERFACE,
        "DOWNLOADER_SOURCE": lifecycle.DOWNLOADER_SOURCE,
    }
    markers.update(overrides)
    downloader = {
        "schema_version": "1",
        "package": markers["DOWNLOADER_PACKAGE"],
        "version": markers["DOWNLOADER_VERSION"],
        "interface": markers["DOWNLOADER_INTERFACE"],
        "source": markers["DOWNLOADER_SOURCE"],
    }
    return "\n".join(
        [
            *(f"{key}={value}" for key, value in markers.items()),
            json.dumps(downloader, sort_keys=True, separators=(",", ":")),
        ]
    )


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
            stdout = _image_preparation_stdout()
            return lifecycle.CommandResult(returncode=0, stdout=stdout, stderr="")
        if description == "stage_model_acquisition_download":
            payload = {
                "schema_version": "1",
                "package": lifecycle.DOWNLOADER_PACKAGE,
                "version": lifecycle.DOWNLOADER_VERSION,
                "interface": lifecycle.DOWNLOADER_INTERFACE,
                "source": lifecycle.DOWNLOADER_SOURCE,
                "model_id": lifecycle.MODEL_ID,
                "model_revision": lifecycle.MODEL_REVISION,
            }
            return lifecycle.CommandResult(
                returncode=0,
                stdout=json.dumps(payload, sort_keys=True, separators=(",", ":")),
                stderr="",
            )
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
            stdout = (
                "RESIDUAL_CONTAINERS=0\n"
                "RESIDUAL_GPU_PROCESSES=0\n"
                "SHUTDOWN_ISSUED=1\n"
            )
            return lifecycle.CommandResult(returncode=0, stdout=stdout, stderr="")
        return lifecycle.CommandResult(returncode=0, stdout="", stderr="")


def _config(
    tmp_path: Path,
    *,
    docker_execution_mode: lifecycle.DockerExecutionMode = (
        lifecycle.DockerExecutionMode.DIRECT
    ),
) -> lifecycle.ProtectedExecutionConfig:
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
            "schema_version": lifecycle.PROTECTED_CONFIG_SCHEMA_VERSION,
            "host": "203.0.113.5",
            "port": 57003,
            "user": "runner",
            "docker_execution_mode": docker_execution_mode.value,
            "private_key_path": str(key),
            "known_hosts_path": str(known_hosts),
            "remote_workspace": "/srv/kv-truth",
            "authorized_key_marker": "marker",
            "local_evidence_dir": str(tmp_path / "evidence"),
            "local_runner_archive": str(archive),
        }
    )


class TestRemoteOrchestratorFullRun:
    @staticmethod
    def _remote_docker_lines(runner: FakeCommandRunner) -> list[str]:
        lines: list[str] = []
        for call in runner.calls:
            texts = [call.input_text or ""]
            if call.argv and call.argv[0] == "ssh":
                texts.append(call.argv[-1])
            for text in texts:
                lines.extend(
                    line.strip()
                    for line in text.splitlines()
                    if "docker" in line and "command -v docker" not in line
                )
        return lines

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
        assert "stage_identity_source_upload" in descriptions
        assert "stage_image_preparation" in descriptions
        assert "stage_model_acquisition_download" in descriptions
        assert "stage_model_acquisition_verify" in descriptions
        assert "stage_canary" in descriptions
        assert any(d.startswith("stage_four_ab_pairs[pair1") for d in descriptions)
        assert any(d.startswith("stage_four_ab_pairs[pair4") for d in descriptions)
        assert "stage_eviction_lane" in descriptions
        assert "stage_transfer_evidence_archive" in descriptions
        assert "stage_teardown_cleanup" in descriptions

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

    def test_full_run_uses_the_reordered_stage_and_reserve_sequence(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        orchestrator.run(local_evidence_bundle_dir=tmp_path / "bundle")
        descriptions = [call.description for call in runner.calls]
        assert descriptions.index("stage_identity_source_upload") < descriptions.index(
            "stage_image_preparation"
        )
        assert descriptions.index("stage_image_preparation") < descriptions.index(
            "stage_model_acquisition_download"
        )
        assert descriptions.index(
            "stage_model_acquisition_verify"
        ) < descriptions.index("stage_canary")

        receipts = [
            json.loads(line)
            for line in orchestrator.operation_receipt_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        first_by_stage = {
            receipt["stage"]: receipt["reserved_minutes"] for receipt in receipts
        }
        assert first_by_stage["preflight"] == 210
        assert first_by_stage["identity_source_transfer"] == 195
        assert first_by_stage["image_preparation"] == 190
        assert first_by_stage["model_acquisition"] == 170
        assert first_by_stage["canary"] == 130
        assert first_by_stage["four_ab_pairs"] == 115
        assert first_by_stage["eviction_lane"] == 55
        assert first_by_stage["evidence_transfer"] == 45
        assert first_by_stage["teardown"] == 35

    def test_vanilla_host_preflight_has_no_host_downloader_dependency(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        outcome = orchestrator.stage_preflight()
        script = runner.calls[-1].input_text or ""
        assert outcome.ok
        assert "HF_CLI" not in script
        assert "command -v huggingface-cli" not in script
        assert "DOCKER_EXECUTION_MODE=direct" in script
        assert "sudo -n -- docker" not in script

    @pytest.mark.parametrize(
        ("container_ids", "expected_count"),
        [
            ("", "0"),
            ("container-one", "1"),
            ("container-one\ncontainer-two", "2"),
        ],
    )
    def test_preflight_container_count_preserves_the_last_id(
        self, tmp_path: Path, container_ids: str, expected_count: str
    ) -> None:
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        orchestrator.stage_preflight()
        script = runner.calls[-1].input_text or ""
        count_line = next(
            line
            for line in script.splitlines()
            if line.startswith('echo "CONTAINER_COUNT=')
        )
        result = subprocess.run(
            ["bash", "-c", count_line],
            env={"CONTAINER_IDS": container_ids},
            check=True,
            capture_output=True,
            text=True,
        )
        marker, value = result.stdout.split("=", 1)
        assert marker == "CONTAINER_COUNT"
        assert value.strip() == expected_count
        assert "CONTAINER_IDS=$(docker ps -aq)" in script

    def test_preflight_gpu_process_query_failure_cannot_report_zero(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        orchestrator.stage_preflight()
        script = runner.calls[-1].input_text or ""
        lines = script.splitlines()
        start = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("GPU_PROCESS_IDS=")
        )
        fragment = "\n".join(lines[start : start + 2])
        result = subprocess.run(
            ["bash", "-c", f"nvidia-smi() {{ return 42; }}\n{fragment}"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "LLMTRACEFX_REASON=preflight_probe_failed" in result.stderr
        assert "GPU_PROCESS_COUNT=" not in result.stdout

    def test_sudo_noninteractive_mode_prefixes_every_docker_call(
        self, tmp_path: Path
    ) -> None:
        mode = lifecycle.DockerExecutionMode.SUDO_NONINTERACTIVE
        config = _config(tmp_path, docker_execution_mode=mode)
        authorization = _authorization(docker_execution_mode=mode.value)
        runner = FakeCommandRunner(
            responses={
                "stage_preflight": lifecycle.CommandResult(
                    returncode=0,
                    stdout=_preflight_stdout(
                        DOCKER_EXECUTION_MODE=mode.value,
                        DOCKER_EXECUTION_CONFIG_SHA256=(
                            lifecycle.docker_execution_config_sha256(mode)
                        ),
                    ),
                    stderr="",
                )
            }
        )
        orchestrator = lifecycle.RemoteOrchestrator(
            config=config,
            authorization=authorization,
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        outcomes = orchestrator.run(local_evidence_bundle_dir=tmp_path / "bundle")
        assert all(outcome.ok for outcome in outcomes)
        docker_lines = self._remote_docker_lines(runner)
        assert docker_lines
        assert all("sudo -n -- docker" in line for line in docker_lines)
        assert all(
            line.replace("preflight_docker_execution_denied", "").count("docker")
            == line.count("sudo -n -- docker")
            for line in docker_lines
        )
        receipts = [
            json.loads(line)
            for line in orchestrator.operation_receipt_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert receipts
        assert {receipt["docker_execution_mode"] for receipt in receipts} == {
            mode.value
        }
        assert {receipt["docker_execution_config_sha256"] for receipt in receipts} == {
            lifecycle.docker_execution_config_sha256(mode)
        }

    def test_direct_mode_never_mixes_in_sudo_docker_calls(self, tmp_path: Path) -> None:
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        outcomes = orchestrator.run(local_evidence_bundle_dir=tmp_path / "bundle")
        assert all(outcome.ok for outcome in outcomes)
        docker_lines = self._remote_docker_lines(runner)
        assert docker_lines
        assert all("sudo -n -- docker" not in line for line in docker_lines)
        assert all("docker" in line for line in docker_lines)

    def test_config_authorization_mode_mismatch_refuses_before_ssh(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner()
        with pytest.raises(
            lifecycle.HostOrchestrationError,
            match="does not match authorization",
        ):
            lifecycle.RemoteOrchestrator(
                config=_config(
                    tmp_path,
                    docker_execution_mode=(
                        lifecycle.DockerExecutionMode.SUDO_NONINTERACTIVE
                    ),
                ),
                authorization=_authorization(),
                runner=runner,
                now_fn=lambda: BILLING_STARTED_AT,
            )
        assert runner.calls == []

    def test_docker_execution_policy_cannot_be_reassigned_after_validation(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(AttributeError):
            orchestrator.docker = lifecycle.DockerCommand(  # type: ignore[misc]
                lifecycle.DockerExecutionMode.SUDO_NONINTERACTIVE
            )
        orchestrator.stage_preflight()
        script = runner.calls[-1].input_text or ""
        assert "CONTAINER_IDS=$(docker ps -aq)" in script
        assert "sudo -n -- docker" not in script

    def test_denied_sudo_docker_refuses_before_image_or_gpu_work(
        self, tmp_path: Path
    ) -> None:
        mode = lifecycle.DockerExecutionMode.SUDO_NONINTERACTIVE
        runner = FakeCommandRunner(
            responses={
                "stage_preflight": lifecycle.CommandResult(
                    returncode=1,
                    stdout="",
                    stderr=(
                        "sudo: a password is required\n"
                        "LLMTRACEFX_REASON=preflight_docker_execution_denied\n"
                    ),
                )
            }
        )
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path, docker_execution_mode=mode),
            authorization=_authorization(docker_execution_mode=mode.value),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(
            lifecycle.HostOrchestrationError,
            match="preflight_docker_execution_denied",
        ):
            orchestrator.run(local_evidence_bundle_dir=tmp_path / "bundle")
        descriptions = [call.description for call in runner.calls]
        assert descriptions == ["stage_preflight", "stage_teardown_cleanup"]
        assert not any(
            description
            in {
                "stage_image_preparation",
                "stage_model_acquisition_download",
                "stage_canary",
                "stage_eviction_lane",
            }
            or description.startswith("stage_four_ab_pairs")
            for description in descriptions
        )
        teardown = runner.calls[-1].input_text or ""
        assert "sudo -n -- docker" in teardown
        assert "xargs -r sudo -n -- docker" in teardown

    def test_model_acquisition_uses_only_labeled_digest_bound_container_mounts(
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
        orchestrator.stage_identity_gate()
        orchestrator.stage_image_preparation()
        orchestrator.stage_model_acquisition()
        call = next(
            item
            for item in runner.calls
            if item.description == "stage_model_acquisition_download"
        )
        script = call.input_text or ""
        assert "huggingface-cli" not in script
        assert "--network bridge" in script
        assert lifecycle.docker_run_label(auth.nonce) in script
        assert DERIVED_IMAGE_ID in script
        assert script.count(" -v ") == 2
        assert f"{orchestrator.paths.model_dir}:/model" in script
        assert f"{orchestrator.paths.hf_scratch_dir}:/hf" in script
        assert "cleanup_download_scratch" in script
        receipt = json.loads(
            (
                _config(tmp_path).local_evidence_dir
                / "private-model-acquisition-receipt.json"
            ).read_text(encoding="utf-8")
        )
        assert receipt["schema_version"] == "2"
        assert receipt["downloader"]["version"] == lifecycle.DOWNLOADER_VERSION
        assert receipt["downloader"]["source"] == lifecycle.DOWNLOADER_SOURCE
        assert receipt["docker_execution_mode"] == "direct"
        assert receipt["docker_execution_config_sha256"] == (
            lifecycle.docker_execution_config_sha256(
                lifecycle.DockerExecutionMode.DIRECT
            )
        )
        assert receipt["verified_file_count"] == 15
        assert receipt["verified_total_bytes"] == 16_397_461_266

    @pytest.mark.parametrize(
        ("marker", "reason"),
        [
            ("DOWNLOADER_PACKAGE", "model_download_interface_missing"),
            ("DOWNLOADER_VERSION", "model_download_version_mismatch"),
        ],
    )
    def test_image_attestation_refuses_missing_or_wrong_downloader(
        self, tmp_path: Path, marker: str, reason: str
    ) -> None:
        value = "" if marker == "DOWNLOADER_PACKAGE" else "0.0.0"
        runner = FakeCommandRunner(
            responses={
                "stage_image_preparation": lifecycle.CommandResult(
                    returncode=0,
                    stdout=_image_preparation_stdout(**{marker: value}),
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
        orchestrator.stage_identity_gate()
        with pytest.raises(lifecycle.HostOrchestrationError, match=reason):
            orchestrator.stage_image_preparation()
        assert not any(
            call.description == "stage_model_acquisition_download"
            for call in runner.calls
        )

    def test_model_inventory_mismatch_refuses_before_any_gpu_container(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner(
            fail_stages=frozenset({"stage_model_acquisition_verify"})
        )
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="model_inventory_mismatch"
        ):
            orchestrator.run(local_evidence_bundle_dir=tmp_path / "bundle")
        assert not any(
            call.description == "stage_canary"
            or call.description.startswith("stage_four_ab_pairs")
            or call.description == "stage_eviction_lane"
            for call in runner.calls
        )

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
        orchestrator.stage_image_preparation()
        orchestrator.stage_model_acquisition()
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
        orchestrator.stage_image_preparation()
        orchestrator.stage_model_acquisition()
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
        orchestrator.stage_identity_gate()
        orchestrator.stage_image_preparation()
        call = next(
            call
            for call in runner.calls
            if call.description == "stage_image_preparation"
        )
        script = call.input_text or ""
        assert "-f containers/vllm-kv-truth/Containerfile" in script
        assert f"--build-arg RUNNER_COMMIT={VALID_HEAD}" in script
        assert "--network none" in script
        authorized_archive_digest = (
            orchestrator.authorization.derived_image_source_digest.removeprefix(
                "sha256:"
            )
        )
        assert authorized_archive_digest in script
        assert "COPY" not in script
        assert f"{orchestrator.paths.model_dir}:/model" not in script
        assert orchestrator.derived_image_id == DERIVED_IMAGE_ID

    def test_image_build_rechecks_remote_archive_against_authorization_digest(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner()
        config = _config(tmp_path)
        orchestrator = lifecycle.RemoteOrchestrator(
            config=config,
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        orchestrator.stage_identity_gate()
        config.local_runner_archive.write_bytes(
            config.local_runner_archive.read_bytes() + b"replaced-after-upload"
        )
        replaced_digest = hashlib.sha256(
            config.local_runner_archive.read_bytes()
        ).hexdigest()

        orchestrator.stage_image_preparation()

        image_call = next(
            call
            for call in runner.calls
            if call.description == "stage_image_preparation"
        )
        script = image_call.input_text or ""
        authorized_digest = (
            orchestrator.authorization.derived_image_source_digest.removeprefix(
                "sha256:"
            )
        )
        assert authorized_digest in script
        assert replaced_digest not in script

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
            f"if [ -d {orchestrator.config.remote_workspace} ]; then "
            f"rmdir {orchestrator.config.remote_workspace}; fi"
        ) in script
        assert f"rm -rf {orchestrator.paths.model_dir}" in script
        assert f"rm -rf {orchestrator.paths.hf_scratch_dir}" in script
        assert f"rm -rf {orchestrator.paths.repo_dir}" in script
        assert f"rm -f {orchestrator.paths.source_archive_remote_path}" in script
        assert "sudo -n shutdown -h +1" in script
        assert script.index("authorized_keys") < script.index(
            "sudo -n shutdown -h +1", script.index("authorized_keys")
        )
        # Mutating discovery is label-scoped; the sole unfiltered query is the
        # read-only final residual check.
        label_filter = f"--filter label={lifecycle.docker_run_label(auth.nonce)}"
        docker_listing_lines = [
            line for line in script.splitlines() if "docker ps" in line
        ]
        assert len(docker_listing_lines) == 3
        assert all(label_filter in line for line in docker_listing_lines[:2])
        assert docker_listing_lines[2].startswith(
            "RESIDUAL_CONTAINER_IDS=$(docker ps -aq)"
        )
        image_listing_lines = [
            line for line in script.splitlines() if "docker images" in line
        ]
        assert len(image_listing_lines) == 1
        assert all(label_filter in line for line in image_listing_lines)
        assert (
            f"RUNNING_CONTAINERS=$(docker ps -q --filter "
            f"label={lifecycle.docker_run_label(auth.nonce)})"
        ) in script
        assert (
            f"ALL_RUN_CONTAINERS=$(docker ps -aq --filter "
            f"label={lifecycle.docker_run_label(auth.nonce)})"
        ) in script
        assert (
            f"LABELED_IMAGES=$(docker images -q --filter "
            f"label={lifecycle.docker_run_label(auth.nonce)})"
        ) in script
        assert "RESIDUAL_CONTAINER_IDS=$(docker ps -aq) ||" in script
        assert "docker rmi -f; fi" in script
        assert "docker rmi -f || true; fi" not in script
        assert script.count("LLMTRACEFX_REASON=teardown_cleanup_failed") >= 4

    @pytest.mark.parametrize(
        ("container_ids", "expected_count"),
        [
            ("", "0"),
            ("container-one", "1"),
            ("container-one\ncontainer-two", "2"),
        ],
    )
    def test_teardown_residual_container_query_counts_exactly(
        self, tmp_path: Path, container_ids: str, expected_count: str
    ) -> None:
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        orchestrator.stage_teardown(tmp_path / "bundle")
        script = runner.calls[-1].input_text or ""
        lines = script.splitlines()
        start = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("RESIDUAL_CONTAINER_IDS=")
        )
        fragment = "\n".join(lines[start : start + 2])
        result = subprocess.run(
            ["bash", "-c", f'docker() {{ printf %s "$DOCKER_OUTPUT"; }}\n{fragment}'],
            env={"DOCKER_OUTPUT": container_ids},
            check=True,
            capture_output=True,
            text=True,
        )
        marker, value = result.stdout.split("=", 1)
        assert marker == "RESIDUAL_CONTAINERS"
        assert value.strip() == expected_count

    def test_teardown_residual_container_query_failure_cannot_report_zero(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        orchestrator.stage_teardown(tmp_path / "bundle")
        script = runner.calls[-1].input_text or ""
        lines = script.splitlines()
        start = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("RESIDUAL_CONTAINER_IDS=")
        )
        fragment = "\n".join(lines[start : start + 2])
        result = subprocess.run(
            ["bash", "-c", f"docker() {{ return 42; }}\n{fragment}"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "LLMTRACEFX_REASON=teardown_cleanup_failed" in result.stderr
        assert "RESIDUAL_CONTAINERS=" not in result.stdout

    def test_teardown_gpu_process_query_failure_cannot_report_zero(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        orchestrator.stage_teardown(tmp_path / "bundle")
        script = runner.calls[-1].input_text or ""
        lines = script.splitlines()
        start = next(
            index
            for index, line in enumerate(lines)
            if line.startswith("RESIDUAL_GPU_PIDS=")
        )
        fragment = "\n".join(lines[start : start + 2])
        result = subprocess.run(
            ["bash", "-c", f"nvidia-smi() {{ return 42; }}\n{fragment}"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "LLMTRACEFX_REASON=teardown_cleanup_failed" in result.stderr
        assert "RESIDUAL_GPU_PROCESSES=" not in result.stdout

    def test_sudo_mode_teardown_uses_only_selected_docker_prefix(
        self, tmp_path: Path
    ) -> None:
        mode = lifecycle.DockerExecutionMode.SUDO_NONINTERACTIVE
        runner = FakeCommandRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path, docker_execution_mode=mode),
            authorization=_authorization(docker_execution_mode=mode.value),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        orchestrator.stage_teardown(tmp_path / "bundle")
        script = runner.calls[-1].input_text or ""
        docker_lines = [
            line.strip() for line in script.splitlines() if "docker" in line
        ]
        assert docker_lines
        assert all("sudo -n -- docker" in line for line in docker_lines)

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

    @pytest.mark.parametrize(
        "failed_description",
        [
            "stage_preflight",
            "stage_identity_gate",
            "stage_identity_source_upload",
            "stage_image_preparation",
            "stage_model_acquisition_download",
            "stage_model_acquisition_verify",
            "stage_canary",
            "stage_four_ab_pairs[pair1-slot1-a]",
            "stage_four_ab_pairs[pair1-slot2-b]",
            "stage_four_ab_pairs[pair2-slot1-b]",
            "stage_four_ab_pairs[pair2-slot2-a]",
            "stage_four_ab_pairs[pair3-slot1-b]",
            "stage_four_ab_pairs[pair3-slot2-a]",
            "stage_four_ab_pairs[pair4-slot1-a]",
            "stage_four_ab_pairs[pair4-slot2-b]",
            "stage_eviction_lane",
            "stage_transfer_evidence_archive",
            "stage_transfer_evidence_digest",
            "stage_transfer_evidence_download",
        ],
    )
    def test_every_substage_failure_is_receipted_and_tears_down_safely(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
        failed_description: str,
    ) -> None:
        runner = FakeCommandRunner(fail_stages=frozenset({failed_description}))
        config = _config(tmp_path)
        orchestrator = lifecycle.RemoteOrchestrator(
            config=config,
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(lifecycle.HostOrchestrationError):
            orchestrator.run(local_evidence_bundle_dir=tmp_path / "bundle")
        receipts = [
            json.loads(line)
            for line in orchestrator.operation_receipt_path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        failed_receipt = next(
            receipt
            for receipt in receipts
            if receipt["command_description"] == failed_description
        )
        assert failed_receipt["return_code"] == 1
        assert failed_receipt["timed_out"] is False
        assert failed_receipt["reason_code"] in lifecycle._SAFE_REASON_MESSAGES
        assert failed_receipt["stderr_message"] != "denied"
        assert len(failed_receipt["stderr_message"]) <= 160
        assert failed_receipt["reserved_minutes"] >= 35
        descriptions = [call.description for call in runner.calls]
        assert "stage_teardown_cleanup" in descriptions
        cleanup = next(
            call
            for call in runner.calls
            if call.description == "stage_teardown_cleanup"
        )
        assert "authorized_keys" in (cleanup.input_text or "")
        assert config.authorized_key_marker in (cleanup.input_text or "")
        assert "SAFE TO TERMINATE INSTANCE NOW" in capsys.readouterr().out

    def test_teardown_reaches_shutdown_when_preflight_fails(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner(fail_stages=frozenset({"stage_preflight"}))
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

    def test_teardown_attempts_shutdown_in_same_session_when_cleanup_fails(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner(fail_stages=frozenset({"stage_teardown_cleanup"}))
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(lifecycle.HostOrchestrationError):
            orchestrator.stage_teardown(tmp_path / "bundle")
        descriptions = {call.description for call in runner.calls}
        assert "stage_teardown_cleanup" in descriptions
        cleanup = next(
            call
            for call in runner.calls
            if call.description == "stage_teardown_cleanup"
        )
        assert "finish_teardown" in (cleanup.input_text or "")
        assert "sudo -n shutdown -h +1" in (cleanup.input_text or "")
        assert orchestrator.state == lifecycle.OrchestratorState.TEARDOWN

    def test_teardown_reports_shutdown_failure_from_same_session(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner(
            responses={
                "stage_teardown_cleanup": lifecycle.CommandResult(
                    returncode=1,
                    stdout="",
                    stderr="LLMTRACEFX_REASON=teardown_shutdown_failed\n",
                )
            }
        )
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(
            lifecycle.HostOrchestrationError,
            match="teardown_shutdown_failed",
        ):
            orchestrator.stage_teardown(tmp_path / "bundle")

    def test_teardown_shutdown_failure_takes_priority_over_cleanup_failure(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner(
            responses={
                "stage_teardown_cleanup": lifecycle.CommandResult(
                    returncode=1,
                    stdout="",
                    stderr=(
                        "LLMTRACEFX_REASON=teardown_cleanup_failed\n"
                        "LLMTRACEFX_REASON=teardown_shutdown_failed\n"
                    ),
                )
            }
        )
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(
            lifecycle.HostOrchestrationError,
            match="teardown_shutdown_failed",
        ):
            orchestrator.stage_teardown(tmp_path / "bundle")
        receipt = json.loads(
            orchestrator.operation_receipt_path.read_text(
                encoding="utf-8"
            ).splitlines()[-1]
        )
        assert receipt["reason_code"] == "teardown_shutdown_failed"

    def test_teardown_refuses_missing_shutdown_marker_with_receipt(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner(
            responses={
                "stage_teardown_cleanup": lifecycle.CommandResult(
                    returncode=0,
                    stdout=("RESIDUAL_CONTAINERS=0\n" "RESIDUAL_GPU_PROCESSES=0\n"),
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
        with pytest.raises(
            lifecycle.HostOrchestrationError,
            match="teardown_shutdown_failed",
        ):
            orchestrator.stage_teardown(tmp_path / "bundle")
        receipt = json.loads(
            orchestrator.operation_receipt_path.read_text(
                encoding="utf-8"
            ).splitlines()[-1]
        )
        assert receipt["substage"] == "verify_cleanup_shutdown"
        assert receipt["reason_code"] == "teardown_shutdown_failed"

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

    def test_teardown_still_runs_on_sigterm(self, tmp_path: Path) -> None:
        installed_handlers: dict[int, object] = {}

        def fake_signal(signum: int, handler: object) -> object:
            installed_handlers[signum] = handler
            return lifecycle.signal.SIG_DFL

        class _TerminatingRunner(FakeCommandRunner):
            def run(
                self,
                argv: Sequence[str],
                *,
                description: str,
                timeout: float,
                input_text: str | None = None,
            ) -> lifecycle.CommandResult:
                if description == "stage_canary":
                    handler = installed_handlers[lifecycle.signal.SIGTERM]
                    assert callable(handler)
                    handler(lifecycle.signal.SIGTERM, None)
                return super().run(
                    argv,
                    description=description,
                    timeout=timeout,
                    input_text=input_text,
                )

        runner = _TerminatingRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(lifecycle.signal, "getsignal", lambda _signum: None)
            monkeypatch.setattr(lifecycle.signal, "signal", fake_signal)
            with pytest.raises(
                lifecycle.HostOrchestrationError,
                match="run/signal: run_interrupted",
            ) as caught:
                orchestrator.run(local_evidence_bundle_dir=tmp_path / "bundle")
        assert caught.value.signal_number == lifecycle.signal.SIGTERM
        descriptions = {call.description for call in runner.calls}
        assert "stage_teardown_cleanup" in descriptions

    def test_sigterm_status_survives_teardown_failure(self, tmp_path: Path) -> None:
        installed_handlers: dict[int, object] = {}

        def fake_signal(signum: int, handler: object) -> object:
            previous = installed_handlers.get(signum, lifecycle.signal.SIG_DFL)
            installed_handlers[signum] = handler
            return previous

        class _TerminatingRunner(FakeCommandRunner):
            def run(
                self,
                argv: Sequence[str],
                *,
                description: str,
                timeout: float,
                input_text: str | None = None,
            ) -> lifecycle.CommandResult:
                if description == "stage_canary":
                    handler = installed_handlers[lifecycle.signal.SIGTERM]
                    assert callable(handler)
                    handler(lifecycle.signal.SIGTERM, None)
                return super().run(
                    argv,
                    description=description,
                    timeout=timeout,
                    input_text=input_text,
                )

        runner = _TerminatingRunner(fail_stages=frozenset({"stage_teardown_cleanup"}))
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(lifecycle.signal, "getsignal", lambda _signum: None)
            monkeypatch.setattr(lifecycle.signal, "signal", fake_signal)
            with pytest.raises(lifecycle.HostOrchestrationError) as caught:
                orchestrator.run(local_evidence_bundle_dir=tmp_path / "bundle")
        assert caught.value.signal_number == lifecycle.signal.SIGTERM
        assert isinstance(caught.value.__cause__, lifecycle.HostOrchestrationError)

    def test_sigterm_during_teardown_is_deferred_until_cleanup_finishes(
        self, tmp_path: Path
    ) -> None:
        installed_handlers: dict[int, object] = {}

        def fake_signal(signum: int, handler: object) -> object:
            previous = installed_handlers.get(signum, lifecycle.signal.SIG_DFL)
            installed_handlers[signum] = handler
            return previous

        class _TeardownTerminatingRunner(FakeCommandRunner):
            def run(
                self,
                argv: Sequence[str],
                *,
                description: str,
                timeout: float,
                input_text: str | None = None,
            ) -> lifecycle.CommandResult:
                if description == "stage_teardown_cleanup":
                    handler = installed_handlers[lifecycle.signal.SIGTERM]
                    assert callable(handler)
                    handler(lifecycle.signal.SIGTERM, None)
                return super().run(
                    argv,
                    description=description,
                    timeout=timeout,
                    input_text=input_text,
                )

        runner = _TeardownTerminatingRunner()
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(lifecycle.signal, "getsignal", lambda _signum: None)
            monkeypatch.setattr(lifecycle.signal, "signal", fake_signal)
            with pytest.raises(lifecycle.HostOrchestrationError) as caught:
                orchestrator.run(local_evidence_bundle_dir=tmp_path / "bundle")
        assert caught.value.signal_number == lifecycle.signal.SIGTERM
        assert orchestrator.state == lifecycle.OrchestratorState.COMPLETE

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
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="preflight_probe_failed"
        ):
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
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="preflight_probe_failed"
        ):
            orchestrator.stage_preflight()

    def test_preflight_surfaces_allowlisted_missing_docker_reason(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner(
            responses={
                "stage_preflight": lifecycle.CommandResult(
                    returncode=1,
                    stdout="",
                    stderr=(
                        "private host detail\n"
                        "LLMTRACEFX_REASON=preflight_missing_docker\n"
                    ),
                )
            }
        )
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="preflight_missing_docker"
        ) as excinfo:
            orchestrator.stage_preflight()
        assert "private host detail" not in str(excinfo.value)

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
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="preflight_probe_failed"
        ):
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
        orchestrator.stage_identity_gate()
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="image_preparation_failed"
        ):
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
        orchestrator.stage_identity_gate()
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="image_preparation_failed"
        ):
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
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="source_transfer_failed"
        ):
            orchestrator.stage_identity_gate()

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
            lifecycle.HostOrchestrationError, match="source_transfer_failed"
        ):
            orchestrator.stage_identity_gate()

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
        orchestrator.stage_identity_gate()
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="image_preparation_failed"
        ):
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
        with pytest.raises(
            lifecycle.HostOrchestrationError, match="source_transfer_failed"
        ):
            orchestrator.stage_identity_gate()

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
            lifecycle.HostOrchestrationError, match="source_transfer_failed"
        ):
            orchestrator.stage_identity_gate()

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
        orchestrator.stage_identity_gate()
        upload_calls = [
            call
            for call in runner.calls
            if call.description == "stage_identity_source_upload"
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
        with pytest.raises(
            lifecycle.HostOrchestrationError,
            match="evidence_download_failed",
        ):
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
            lifecycle.HostOrchestrationError,
            match="evidence_download_failed",
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
            lifecycle.HostOrchestrationError,
            match="evidence_download_failed",
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
        with pytest.raises(
            lifecycle.HostOrchestrationError,
            match="evidence_download_failed",
        ):
            orchestrator.stage_transfer_evidence(tmp_path / "bundle")

    def test_transfer_evidence_rejects_empty_digest_with_receipt(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner(
            responses={
                "stage_transfer_evidence_digest": lifecycle.CommandResult(
                    returncode=0,
                    stdout="",
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
        with pytest.raises(
            lifecycle.HostOrchestrationError,
            match="evidence_digest_failed",
        ):
            orchestrator.stage_transfer_evidence(tmp_path / "bundle")
        receipt = json.loads(
            orchestrator.operation_receipt_path.read_text(
                encoding="utf-8"
            ).splitlines()[-1]
        )
        assert receipt["substage"] == "verify_remote_digest"
        assert receipt["reason_code"] == "evidence_digest_failed"

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
                    stdout=(
                        "RESIDUAL_CONTAINERS=1\n"
                        "RESIDUAL_GPU_PROCESSES=0\n"
                        "SHUTDOWN_ISSUED=1\n"
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
        with pytest.raises(
            lifecycle.HostOrchestrationError,
            match="teardown_cleanup_failed",
        ):
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

    def test_timeout_receipt_is_bounded_and_contains_no_raw_stderr(
        self, tmp_path: Path
    ) -> None:
        runner = FakeCommandRunner(
            responses={
                "stage_preflight": lifecycle.CommandResult(
                    returncode=-1,
                    stdout="",
                    stderr="private-host.example /secret/key TOKEN=value",
                    timed_out=True,
                )
            }
        )
        orchestrator = lifecycle.RemoteOrchestrator(
            config=_config(tmp_path),
            authorization=_authorization(),
            runner=runner,
            now_fn=lambda: BILLING_STARTED_AT,
        )
        with pytest.raises(lifecycle.HostOrchestrationError, match="operation_timeout"):
            orchestrator.stage_preflight()
        receipt = json.loads(
            orchestrator.operation_receipt_path.read_text(
                encoding="utf-8"
            ).splitlines()[-1]
        )
        serialized = json.dumps(receipt)
        assert receipt["timed_out"] is True
        assert receipt["return_code"] == -1
        assert receipt["stderr_category"] == "timeout"
        assert receipt["stderr_message"] == "the operation timed out"
        assert "private-host" not in serialized
        assert "/secret/key" not in serialized
        assert "TOKEN=value" not in serialized


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

    @pytest.mark.parametrize(
        "name",
        [
            "PYTHONPATH",
            "PYTHONHOME",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "SSH_AUTH_SOCK",
            "DOCKER_HOST",
            "DOCKER_CONFIG",
            "GIT_ASKPASS",
            "GIT_CONFIG_GLOBAL",
            "GIT_SSH_COMMAND",
            "BASH_ENV",
            "ENV",
            "SHELLOPTS",
        ],
    )
    def test_rejects_forbidden_routing_variable(
        self, monkeypatch: pytest.MonkeyPatch, name: str
    ) -> None:
        monkeypatch.setenv(name, "host-controlled")
        with pytest.raises(lifecycle.HostOrchestrationError, match="command-routing"):
            lifecycle.SubprocessCommandRunner()
