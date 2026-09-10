"""Explicit, evidence-labeled SSH trust-on-first-use enrollment tests."""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from vllm_kv_truth import lifecycle

OBSERVED_AT = datetime(2030, 1, 1, tzinfo=timezone.utc)


def _key_base64(fill: int = 7) -> str:
    algorithm = b"ssh-ed25519"
    raw = (
        len(algorithm).to_bytes(4, "big")
        + algorithm
        + (32).to_bytes(4, "big")
        + bytes([fill]) * 32
    )
    return base64.b64encode(raw).decode("ascii")


@dataclass
class FakeRunner:
    result: lifecycle.CommandResult
    calls: list[tuple[tuple[str, ...], str, float, str | None]] = field(
        default_factory=list
    )

    def run(
        self,
        argv: Sequence[str],
        *,
        description: str,
        timeout: float,
        input_text: str | None = None,
    ) -> lifecycle.CommandResult:
        self.calls.append((tuple(argv), description, timeout, input_text))
        return self.result


def _protected_dir(tmp_path: Path) -> Path:
    protected = tmp_path / "protected"
    protected.mkdir(mode=0o700)
    return protected


def _request(
    protected: Path,
    *,
    host: str = "203.0.113.8",
    port: int = 57003,
) -> tuple[Path, Path, Path]:
    known_hosts = protected / "known_hosts"
    known_hosts.write_bytes(b"")
    known_hosts.chmod(0o600)
    receipt = protected / "tofu-enrollment-receipt.json"
    request = protected / "tofu-enrollment-request.json"
    request.write_text(
        json.dumps(
            {
                "schema_version": lifecycle.TOFU_ENROLLMENT_REQUEST_SCHEMA_VERSION,
                "host_key_trust_policy": (
                    lifecycle.HostKeyTrustPolicy.TOFU_UNVERIFIED.value
                ),
                "tofu_unverified_acknowledgement": (
                    lifecycle.TOFU_UNVERIFIED_ACKNOWLEDGEMENT
                ),
                "host": host,
                "port": port,
                "known_hosts_path": str(known_hosts),
                "receipt_path": str(receipt),
            }
        ),
        encoding="utf-8",
    )
    request.chmod(0o600)
    return request, known_hosts, receipt


def _successful_runner(
    *, host: str = "203.0.113.8", port: int = 57003, key: str | None = None
) -> FakeRunner:
    key = key or _key_base64()
    return FakeRunner(
        lifecycle.CommandResult(
            returncode=0,
            stdout=(
                f"# {host}:{port} SSH-2.0-test-server\n"
                f"{lifecycle.openssh_known_hosts_host_token(host, port)} "
                f"ssh-ed25519 {key}\n"
            ),
            stderr="",
        )
    )


def _enroll(
    tmp_path: Path,
) -> tuple[
    Path,
    Path,
    lifecycle.TofuEnrollmentReceipt,
    FakeRunner,
]:
    protected = _protected_dir(tmp_path)
    request, known_hosts, receipt_path = _request(protected)
    runner = _successful_runner()
    receipt = lifecycle.enroll_tofu_host_key(
        request, runner, now_fn=lambda: OBSERVED_AT
    )
    return known_hosts, receipt_path, receipt, runner


def _config_payload(
    tmp_path: Path,
    known_hosts: Path,
    receipt_path: Path,
    receipt: lifecycle.TofuEnrollmentReceipt,
) -> dict[str, Any]:
    private_key = known_hosts.parent / "id_ed25519"
    private_key.write_text("private", encoding="utf-8")
    private_key.chmod(0o600)
    archive = known_hosts.parent / "runner.tar"
    archive.write_bytes(b"archive")
    return {
        "schema_version": lifecycle.TOFU_CONFIG_SCHEMA_VERSION,
        "host": receipt.host,
        "port": receipt.port,
        "user": "runner",
        "host_key_trust_policy": (lifecycle.HostKeyTrustPolicy.TOFU_UNVERIFIED.value),
        "host_key_ed25519_base64": receipt.host_key_ed25519_base64,
        "host_key_fingerprint_sha256": receipt.host_key_fingerprint_sha256,
        "known_hosts_sha256": receipt.known_hosts_sha256,
        "tofu_enrollment_receipt_path": str(receipt_path),
        "tofu_enrollment_receipt_sha256": receipt.receipt_sha256,
        "tofu_unverified_acknowledgement": (lifecycle.TOFU_UNVERIFIED_ACKNOWLEDGEMENT),
        "docker_execution_mode": lifecycle.DockerExecutionMode.DIRECT.value,
        "private_key_path": str(private_key),
        "known_hosts_path": str(known_hosts),
        "remote_workspace": "/srv/kv-truth",
        "authorized_key_marker": "marker",
        "local_evidence_dir": str(tmp_path / "evidence"),
        "local_runner_archive": str(archive),
    }


def _authorization_payload(
    config: lifecycle.ProtectedExecutionConfig,
    **overrides: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": lifecycle.TOFU_AUTHORIZATION_SCHEMA_VERSION,
        "protocol_id": lifecycle.PROTOCOL_ID,
        "repository_head": "a" * 40,
        "base_image_reference": lifecycle.BASE_IMAGE_REFERENCE,
        "derived_image_source_digest": "sha256:" + "b" * 64,
        "vllm_version": lifecycle.REQUIRED_VLLM_VERSION,
        "vllm_commit": lifecycle.REQUIRED_VLLM_COMMIT,
        "model_id": lifecycle.MODEL_ID,
        "model_revision": lifecycle.MODEL_REVISION,
        "model_inventory_sha256": lifecycle._model_inventory_sha256(),
        "model_download_package": lifecycle.DOWNLOADER_PACKAGE,
        "model_download_version": lifecycle.DOWNLOADER_VERSION,
        "model_download_interface": lifecycle.DOWNLOADER_INTERFACE,
        "model_download_source": lifecycle.DOWNLOADER_SOURCE,
        "gpu_expected_count": 1,
        "gpu_expected_name": lifecycle.EXPECTED_GPU_NAME,
        "gpu_expected_driver": lifecycle.EXPECTED_DRIVER,
        "gpu_expected_memory_mib": lifecycle.EXPECTED_MEMORY_MIB,
        "docker_execution_mode": lifecycle.DockerExecutionMode.DIRECT.value,
        "docker_execution_config_sha256": (
            lifecycle.docker_execution_config_sha256(
                lifecycle.DockerExecutionMode.DIRECT
            )
        ),
        "rate_usd_per_hour": "0.500000",
        "total_cap_usd": "10.000000",
        "billing_started_at": lifecycle._canonical_timestamp(OBSERVED_AT),
        "operational_cutoff": lifecycle._canonical_timestamp(
            OBSERVED_AT + timedelta(minutes=175)
        ),
        "cleanup_reserve_minutes": lifecycle.CLEANUP_RESERVE_MINUTES,
        "authorized_at": lifecycle._canonical_timestamp(
            OBSERVED_AT - timedelta(minutes=5)
        ),
        "authorization_expiry": lifecycle._canonical_timestamp(
            OBSERVED_AT + timedelta(hours=6)
        ),
        "automatic_retries": 0,
        "replacement_allowed": False,
        "nonce": "c" * 40,
        "host_key_trust_policy": (lifecycle.HostKeyTrustPolicy.TOFU_UNVERIFIED.value),
        "host_key_ed25519_sha256": hashlib.sha256(
            base64.b64decode(config.host_key_ed25519_base64 or "")
        ).hexdigest(),
        "host_key_fingerprint_sha256": config.host_key_fingerprint_sha256,
        "known_hosts_sha256": config.known_hosts_sha256,
        "tofu_enrollment_receipt_sha256": (config.tofu_enrollment_receipt_sha256),
        "host_key_trust_binding_sha256": (config.host_key_trust_binding_sha256()),
        "tofu_unverified_acknowledgement": (lifecycle.TOFU_UNVERIFIED_ACKNOWLEDGEMENT),
    }
    payload.update(overrides)
    payload["authorization_sha256"] = lifecycle.build_authorization_seal(payload)
    return payload


def test_explicit_tofu_happy_path_has_exact_native_argv_and_receipt(
    tmp_path: Path,
) -> None:
    known_hosts, receipt_path, receipt, runner = _enroll(tmp_path)
    assert runner.calls == [
        (
            (
                "/usr/bin/ssh-keyscan",
                "-T",
                "5",
                "-t",
                "ed25519",
                "-p",
                "57003",
                "203.0.113.8",
            ),
            "enroll_tofu_host_key",
            10,
            None,
        )
    ]
    assert "known_hosts" not in " ".join(runner.calls[0][0])
    assert receipt.host_key_fingerprint_sha256.startswith("SHA256:")
    assert receipt.observed_at == OBSERVED_AT
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["policy"] == "tofu_unverified"
    assert payload["provider_identity_independently_authenticated"] is False
    assert payload["identity_statement"] == lifecycle.TOFU_IDENTITY_STATEMENT
    assert payload["tool"]["path"] == "/usr/bin/ssh-keyscan"
    assert payload["tool"]["environment"] == lifecycle.SSH_KEYSCAN_ENVIRONMENT
    assert hashlib.sha256(known_hosts.read_bytes()).hexdigest() == (
        payload["known_hosts"]["content_sha256"]
    )


def test_tofu_subprocess_uses_only_fixed_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    protected = _protected_dir(tmp_path)
    request, _known_hosts, _receipt = _request(protected)
    observed: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed["argv"] = argv
        observed.update(kwargs)
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout=(
                "# 203.0.113.8:57003 SSH-2.0-test-server\n"
                f"[203.0.113.8]:57003 ssh-ed25519 {_key_base64()}\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(lifecycle.subprocess, "run", fake_run)
    monkeypatch.setattr(lifecycle, "reject_credential_environment", lambda _env: None)
    lifecycle.enroll_tofu_host_key(
        request,
        lifecycle.SubprocessCommandRunner(),
        now_fn=lambda: OBSERVED_AT,
    )
    assert observed["argv"] == [
        "/usr/bin/ssh-keyscan",
        "-T",
        "5",
        "-t",
        "ed25519",
        "-p",
        "57003",
        "203.0.113.8",
    ]
    assert observed["env"] == lifecycle.SSH_KEYSCAN_ENVIRONMENT
    assert observed.get("shell", False) is False
    assert observed["timeout"] == 10
    assert observed["check"] is False


@pytest.mark.parametrize(
    ("stdout", "stderr", "returncode", "timed_out", "expected"),
    [
        ("", "", 0, False, "truncated"),
        ("\n", "", 0, False, "malformed"),
        (
            "[203.0.113.8]:57003 ssh-rsa AAAA\n",
            "",
            0,
            False,
            "algorithm",
        ),
        (
            f"[203.0.113.8]:57003 ssh-ed25519 {_key_base64()}\n"
            f"[203.0.113.8]:57003 ssh-ed25519 {_key_base64()}\n",
            "",
            0,
            False,
            "exactly one",
        ),
        (
            f"[203.0.113.8]:57003 ssh-ed25519 {_key_base64()}\n"
            f"[203.0.113.8]:57003 ssh-ed25519 {_key_base64(8)}\n",
            "",
            0,
            False,
            "exactly one",
        ),
        (
            f"[203.0.113.9]:57003 ssh-ed25519 {_key_base64()}\n",
            "",
            0,
            False,
            "endpoint",
        ),
        (
            f"[203.0.113.8]:57003 ssh-ed25519 {_key_base64()}",
            "",
            0,
            False,
            "truncated",
        ),
        ("malformed output\n", "", 0, False, "malformed"),
        ("[203.0.113.8]:57003 ssh-ed25519 AAAA\n", "", 0, False, "truncated"),
        (
            f"[203.0.113.8]:57003 ssh-ed25519 {_key_base64()}\n",
            "unexpected warning\n",
            0,
            False,
            "stderr",
        ),
        ("", "", 1, False, "nonzero"),
        ("", "", -1, True, "timed out"),
    ],
)
def test_tofu_refuses_ambiguous_or_invalid_keyscan_results(
    tmp_path: Path,
    stdout: str,
    stderr: str,
    returncode: int,
    timed_out: bool,
    expected: str,
) -> None:
    protected = _protected_dir(tmp_path)
    request, known_hosts, receipt = _request(protected)
    runner = FakeRunner(
        lifecycle.CommandResult(
            returncode=returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
        )
    )
    with pytest.raises(lifecycle.HostOrchestrationError, match=expected):
        lifecycle.enroll_tofu_host_key(request, runner)
    assert known_hosts.read_bytes() == b""
    assert not receipt.exists()


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        ("nonempty", "fresh empty"),
        ("symlink", "symlink"),
        ("writable", "0600"),
        ("unsafe-parent", "mode 0700"),
    ],
)
def test_tofu_requires_fresh_protected_known_hosts(
    tmp_path: Path, mutation: str, expected: str
) -> None:
    protected = _protected_dir(tmp_path)
    request, known_hosts, _receipt = _request(protected)
    if mutation == "nonempty":
        known_hosts.write_text("existing", encoding="utf-8")
    elif mutation == "symlink":
        known_hosts.unlink()
        target = protected / "target"
        target.write_bytes(b"")
        target.chmod(0o600)
        known_hosts.symlink_to(target)
    elif mutation == "writable":
        known_hosts.chmod(0o666)
    else:
        protected.chmod(0o755)
    with pytest.raises(lifecycle.HostOrchestrationError, match=expected):
        lifecycle.enroll_tofu_host_key(request, _successful_runner())


@pytest.mark.parametrize("mutation", ["symlink-ancestor", "writable-ancestor"])
def test_tofu_rejects_unsafe_higher_ancestry(tmp_path: Path, mutation: str) -> None:
    protected = _protected_dir(tmp_path)
    request, _unused_known_hosts, _unused_receipt = _request(protected)
    if mutation == "symlink-ancestor":
        actual = tmp_path / "actual"
        actual.mkdir(mode=0o700)
        leaf = actual / "leaf"
        leaf.mkdir(mode=0o700)
        ancestor = tmp_path / "ancestor-link"
        ancestor.symlink_to(actual, target_is_directory=True)
    else:
        ancestor = tmp_path / "writable-ancestor"
        ancestor.mkdir(mode=0o700)
        ancestor.chmod(0o777)
        leaf = ancestor / "leaf"
        leaf.mkdir(mode=0o700)
    known_hosts = ancestor / "leaf" / "known_hosts"
    known_hosts.write_bytes(b"")
    known_hosts.chmod(0o600)
    raw = json.loads(request.read_text(encoding="utf-8"))
    raw["known_hosts_path"] = str(known_hosts)
    raw["receipt_path"] = str(ancestor / "leaf" / "receipt.json")
    request.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(
        lifecycle.HostOrchestrationError,
        match="ancestry.*symlink|ancestry.*writable",
    ):
        lifecycle.enroll_tofu_host_key(request, _successful_runner())


@pytest.mark.parametrize("host", ["example.com", "203.0.113.008", "2001:0db8::1"])
def test_tofu_requires_canonical_direct_ip(tmp_path: Path, host: str) -> None:
    protected = _protected_dir(tmp_path)
    request, _known_hosts, _receipt = _request(protected, host=host)
    with pytest.raises(lifecycle.HostOrchestrationError, match="IP literal|canonical"):
        lifecycle.enroll_tofu_host_key(request, _successful_runner(host=host))


@pytest.mark.parametrize("port", [0, 65536, True, "22"])
def test_tofu_requires_valid_literal_port(tmp_path: Path, port: Any) -> None:
    protected = _protected_dir(tmp_path)
    request, _known_hosts, _receipt = _request(protected, port=port)
    with pytest.raises(lifecycle.HostOrchestrationError, match="port"):
        lifecycle.enroll_tofu_host_key(request, _successful_runner(port=22))


def test_tofu_uses_native_default_port_known_hosts_token(tmp_path: Path) -> None:
    protected = _protected_dir(tmp_path)
    request, known_hosts, _receipt = _request(protected, port=22)
    lifecycle.enroll_tofu_host_key(
        request,
        _successful_runner(port=22),
        now_fn=lambda: OBSERVED_AT,
    )
    assert known_hosts.read_text(encoding="ascii") == (
        f"203.0.113.8 ssh-ed25519 {_key_base64()}\n"
    )


def test_tofu_config_and_authorization_bind_every_trust_artifact(
    tmp_path: Path,
) -> None:
    known_hosts, receipt_path, receipt, _runner = _enroll(tmp_path)
    config = lifecycle.ProtectedExecutionConfig.from_dict(
        _config_payload(tmp_path, known_hosts, receipt_path, receipt)
    )
    authorization = lifecycle.RunAuthorization.from_dict(_authorization_payload(config))
    runner = FakeRunner(lifecycle.CommandResult(0, "", ""))
    orchestrator = lifecycle.RemoteOrchestrator(
        config=config,
        authorization=authorization,
        runner=runner,
        now_fn=lambda: OBSERVED_AT,
    )
    ssh = orchestrator.ssh_options.shared_options()
    assert "StrictHostKeyChecking=yes" in ssh
    assert "GlobalKnownHostsFile=/dev/null" in ssh
    assert f"UserKnownHostsFile={known_hosts}" in ssh
    assert (
        orchestrator.ssh_options.public_record()["host_key_trust_policy"]
        == "tofu_unverified"
    )
    assert (
        orchestrator.ssh_options.public_record()[
            "provider_identity_independently_authenticated"
        ]
        is False
    )
    assert runner.calls == []


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("host_key_ed25519_base64", _key_base64(8), "fingerprint"),
        (
            "host_key_fingerprint_sha256",
            "SHA256:" + "A" * 43,
            "fingerprint",
        ),
        ("known_hosts_sha256", "0" * 64, "known_hosts_sha256"),
        ("tofu_enrollment_receipt_sha256", "0" * 64, "receipt digest"),
        ("host_key_trust_policy", "provider_pinned", "schema 3"),
        ("tofu_unverified_acknowledgement", "yes", "acknowledgement"),
    ],
)
def test_tofu_config_tampering_refuses_before_ssh(
    tmp_path: Path, field: str, value: str, expected: str
) -> None:
    known_hosts, receipt_path, receipt, _runner = _enroll(tmp_path)
    payload = _config_payload(tmp_path, known_hosts, receipt_path, receipt)
    payload[field] = value
    with pytest.raises(lifecycle.HostOrchestrationError, match=expected):
        lifecycle.ProtectedExecutionConfig.from_dict(payload)


def test_changed_known_hosts_or_receipt_refuses_before_ssh(tmp_path: Path) -> None:
    known_hosts, receipt_path, receipt, _runner = _enroll(tmp_path)
    payload = _config_payload(tmp_path, known_hosts, receipt_path, receipt)
    known_hosts.write_text(
        f"[203.0.113.8]:57003 ssh-ed25519 {_key_base64(9)}\n",
        encoding="ascii",
    )
    with pytest.raises(lifecycle.HostOrchestrationError, match="known_hosts"):
        lifecycle.ProtectedExecutionConfig.from_dict(payload)
    known_hosts.write_text(
        f"[203.0.113.8]:57003 ssh-ed25519 {_key_base64()}\n",
        encoding="ascii",
    )
    raw_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    raw_receipt["observed_at"] = "2031-01-01T00:00:00.000000Z"
    receipt_path.write_text(json.dumps(raw_receipt), encoding="utf-8")
    with pytest.raises(lifecycle.HostOrchestrationError, match="own content"):
        lifecycle.ProtectedExecutionConfig.from_dict(payload)


def test_tofu_config_rejects_private_key_with_symlinked_ancestor(
    tmp_path: Path,
) -> None:
    known_hosts, receipt_path, receipt, _runner = _enroll(tmp_path)
    payload = _config_payload(tmp_path, known_hosts, receipt_path, receipt)
    actual = tmp_path / "actual-key-parent"
    actual.mkdir(mode=0o700)
    key = actual / "key"
    key.write_text("private", encoding="utf-8")
    key.chmod(0o600)
    alias = tmp_path / "key-parent-alias"
    alias.symlink_to(actual, target_is_directory=True)
    payload["private_key_path"] = str(alias / "key")
    with pytest.raises(lifecycle.HostOrchestrationError, match="ancestry.*symlink"):
        lifecycle.ProtectedExecutionConfig.from_dict(payload)


def test_policy_switch_and_authorization_mismatch_refuse_before_ssh(
    tmp_path: Path,
) -> None:
    known_hosts, receipt_path, receipt, _runner = _enroll(tmp_path)
    config = lifecycle.ProtectedExecutionConfig.from_dict(
        _config_payload(tmp_path, known_hosts, receipt_path, receipt)
    )
    payload = _authorization_payload(
        config,
        host_key_fingerprint_sha256="SHA256:" + "A" * 43,
    )
    authorization = lifecycle.RunAuthorization.from_dict(payload)
    runner = FakeRunner(lifecycle.CommandResult(0, "", ""))
    with pytest.raises(lifecycle.HostOrchestrationError, match="fingerprint"):
        lifecycle.RemoteOrchestrator(
            config=config, authorization=authorization, runner=runner
        )
    assert runner.calls == []


def test_provider_pinned_schema_cannot_consume_tofu_receipt(tmp_path: Path) -> None:
    known_hosts, receipt_path, receipt, _runner = _enroll(tmp_path)
    payload = _config_payload(tmp_path, known_hosts, receipt_path, receipt)
    payload["schema_version"] = lifecycle.PROVIDER_PINNED_CONFIG_SCHEMA_VERSION
    with pytest.raises(lifecycle.HostOrchestrationError, match="required set"):
        lifecycle.ProtectedExecutionConfig.from_dict(payload)


def test_provider_authorization_cannot_consume_tofu_fields(tmp_path: Path) -> None:
    known_hosts, receipt_path, receipt, _runner = _enroll(tmp_path)
    config = lifecycle.ProtectedExecutionConfig.from_dict(
        _config_payload(tmp_path, known_hosts, receipt_path, receipt)
    )
    payload = _authorization_payload(
        config, schema_version=lifecycle.AUTHORIZATION_SCHEMA_VERSION
    )
    payload["authorization_sha256"] = lifecycle.build_authorization_seal(payload)
    with pytest.raises(lifecycle.HostOrchestrationError, match="required set"):
        lifecycle.RunAuthorization.from_dict(payload)


def test_tofu_authorization_cannot_omit_receipt_binding(tmp_path: Path) -> None:
    known_hosts, receipt_path, receipt, _runner = _enroll(tmp_path)
    config = lifecycle.ProtectedExecutionConfig.from_dict(
        _config_payload(tmp_path, known_hosts, receipt_path, receipt)
    )
    payload = _authorization_payload(config)
    del payload["tofu_enrollment_receipt_sha256"]
    payload["authorization_sha256"] = lifecycle.build_authorization_seal(payload)
    with pytest.raises(lifecycle.HostOrchestrationError, match="required set"):
        lifecycle.RunAuthorization.from_dict(payload)
