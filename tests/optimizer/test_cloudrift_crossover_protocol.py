"""Tests for offline crossover orchestration and protocol evidence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from llmtracefx.optimizer.lab.qwen3_8b import cloudrift_crossover as orchestrator
from llmtracefx.optimizer.lab.qwen3_8b import cloudrift_crossover_evidence as evidence
from llmtracefx.optimizer.lab.qwen3_8b import vllm_compile


def _authorization(
    plan: vllm_compile.VLLMCompilePlan,
    *,
    head: str = "a" * 40,
    image_id: str = vllm_compile.DERIVED_IMAGE_ID,
    workspace: Path = Path("."),
) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    value = {
        "schema_version": "1",
        "protocol_id": vllm_compile.PROTOCOL_ID,
        "provider": "CloudRift",
        "approved": True,
        "plan_sha256": plan.content_sha256,
        "source_head": head,
        "runtime_image_id": image_id,
        "experiment_nonce": "c" * 32,
        "workspace_sha256": orchestrator._workspace_sha256(workspace),
        "authorized_at": now,
        "billing_started_at": now,
        "scheduled_shutdown_at": (
            datetime.fromisoformat(now)
            + timedelta(seconds=vllm_compile.ACTIVE_PLANNED_SECONDS)
        ).isoformat(),
        "rate_usd_per_hour": "0.39",
        "hard_cap_usd": "3",
        "automatic_retries": 0,
        "provider_access_managed_externally": True,
    }
    value["authorization_sha256"] = orchestrator._sha256_json(value)
    return value


def _signature_arguments(root: Path) -> dict[str, Path]:
    signature = root / "authorization.sig"
    authorized_signers = root / "allowed_signers"
    signature.write_text("test detached signature\n", encoding="utf-8")
    authorized_signers.write_text(
        f"{orchestrator.AUTHORIZATION_SIGNER_IDENTITY} ssh-ed25519 TEST\n",
        encoding="utf-8",
    )
    return {
        "authorization_signature": signature,
        "authorized_signers": authorized_signers,
    }


class FakeRunner:
    def __init__(
        self,
        *,
        head: str,
        image_id: str,
        output_dir: Path | None = None,
        fail_cell: bool = False,
        fail_signature: bool = False,
    ) -> None:
        self.head = head
        self.image_id = image_id
        self.output_dir = output_dir
        self.fail_cell = fail_cell
        self.fail_signature = fail_signature
        self.commands: list[tuple[str, ...]] = []
        self.inputs: list[str | None] = []
        self.timeouts: list[int] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        input_text: str | None = None,
    ) -> orchestrator.CommandResult:
        command = tuple(argv)
        self.commands.append(command)
        self.inputs.append(input_text)
        self.timeouts.append(timeout_seconds)
        if (
            len(command) >= 4
            and command[:2] == ("git", "-C")
            and command[3] == "rev-parse"
        ):
            return orchestrator.CommandResult(0, self.head + "\n", "")
        if (
            len(command) >= 4
            and command[:2] == ("git", "-C")
            and command[3] == "status"
        ):
            return orchestrator.CommandResult(0, "", "")
        if command[:3] == ("ssh-keygen", "-Y", "verify"):
            if self.fail_signature:
                return orchestrator.CommandResult(255, "", "bad signature")
            return orchestrator.CommandResult(0, "Good signature\n", "")
        docker_prefix = ("docker", "--host", orchestrator.LOCAL_DOCKER_SOCKET)
        if command[:3] == docker_prefix and command[3:5] == ("info", "--format"):
            return orchestrator.CommandResult(0, "local-daemon\n", "")
        if command[:3] == docker_prefix and command[3:5] == ("image", "inspect"):
            return orchestrator.CommandResult(0, self.image_id + "\n", "")
        if command[:3] == docker_prefix and command[3:4] == ("ps",):
            return orchestrator.CommandResult(0, "", "")
        if (
            command
            and command[0] == "nvidia-smi"
            and command[1].startswith("--query-gpu=")
        ):
            return orchestrator.CommandResult(
                0,
                (
                    "NVIDIA GeForce RTX 4090, 580.159.03, 24564, 1024, "
                    "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, 45, 0, "
                    "450.00, 210, 8.9\n"
                ),
                "",
            )
        if command and command[0] == "nvidia-smi":
            return orchestrator.CommandResult(0, "", "")
        if command[:3] == docker_prefix and command[3:4] == ("run",):
            if self.fail_cell:
                self.fail_cell = False
                return orchestrator.CommandResult(1, "", "failed")
            output_mount = next(
                value
                for value in command
                if value.startswith("type=bind,src=") and value.endswith(",dst=/output")
            )
            output_dir = Path(
                output_mount.removeprefix("type=bind,src=").removesuffix(",dst=/output")
            )
            cell_id = command[command.index("--cell-id") + 1]
            (output_dir / f"{cell_id}.json").write_text(
                '{"terminal":true}\n',
                encoding="utf-8",
            )
            (output_dir / f".{cell_id}-progress.json").write_text(
                '{"terminal":false}\n',
                encoding="utf-8",
            )
        return orchestrator.CommandResult(0, "", "")


def test_offline_plan_is_refusal_and_parser_defaults_to_it() -> None:
    document = orchestrator.offline_plan_document()
    assert document["execution_authorized"] is False
    assert document["offline_only"] is True
    assert document["network_request_performed"] is False
    assert document["provider_authentication_used"] is False
    assert document["gpu_used"] is False
    assert document["spend_usd"] == "0"
    assert document["blockers"]
    parser = orchestrator.build_parser()
    assert parser.parse_args([]).action is None
    run_options = parser.parse_args(
        [
            "run",
            "--plan",
            "plan.json",
            "--authorization",
            "authorization.json",
            "--authorization-signature",
            "authorization.sig",
            "--authorized-signers",
            "allowed_signers",
            "--repository",
            "repo",
            "--workspace",
            "workspace",
            "--model-path",
            "model",
            "--state-path",
            "state",
            "--image-reference",
            "image",
        ]
    )
    assert not hasattr(run_options, "host")
    assert not hasattr(run_options, "username")
    assert not hasattr(run_options, "password")
    assert not hasattr(run_options, "token")


def test_authorization_is_exact_and_rejects_scope_drift() -> None:
    plan = vllm_compile.build_default_plan()
    value = _authorization(plan)
    parsed = orchestrator.ExecutionAuthorization.from_dict(value)
    assert parsed.plan_sha256 == plan.content_sha256
    assert parsed.to_dict() == value
    for field, replacement in (
        ("hard_cap_usd", "4"),
        ("approved", False),
        ("automatic_retries", 1),
        ("provider_access_managed_externally", False),
    ):
        mutated = dict(value)
        mutated[field] = replacement
        with pytest.raises(
            orchestrator.CrossoverOrchestratorError,
            match="approved execution envelope",
        ):
            orchestrator.ExecutionAuthorization.from_dict(mutated)
    extra = {**value, "host": "private"}
    with pytest.raises(orchestrator.CrossoverOrchestratorError, match="keys"):
        orchestrator.ExecutionAuthorization.from_dict(extra)
    tampered = {**value, "plan_sha256": "sha256:" + ("d" * 64)}
    with pytest.raises(orchestrator.CrossoverOrchestratorError, match="content hash"):
        orchestrator.ExecutionAuthorization.from_dict(tampered)
    reversed_time = dict(value)
    reversed_time["authorized_at"] = (
        datetime.fromisoformat(value["billing_started_at"]) + timedelta(seconds=1)
    ).isoformat()
    reversed_time["authorization_sha256"] = orchestrator._sha256_json(
        {
            key: item
            for key, item in reversed_time.items()
            if key != "authorization_sha256"
        }
    )
    with pytest.raises(orchestrator.CrossoverOrchestratorError, match="precede"):
        orchestrator.ExecutionAuthorization.from_dict(reversed_time)


def test_credential_environment_is_rejected() -> None:
    with pytest.raises(
        orchestrator.CrossoverOrchestratorError,
        match="credential-shaped",
    ):
        orchestrator._reject_credential_environment({"CLOUD_TOKEN": "secret"})


@pytest.mark.parametrize(
    "name",
    ["DOCKER_HOST", "DOCKER_CONTEXT", "DOCKER_CONFIG", "SSH_AUTH_SOCK"],
)
def test_command_routing_environment_is_rejected(name: str) -> None:
    with pytest.raises(
        orchestrator.CrossoverOrchestratorError,
        match="command-routing",
    ):
        orchestrator._reject_credential_environment({name: "/hostile"})


def test_subprocess_runner_uses_fixed_minimal_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict = {}

    def fake_run(argv: list[str], **kwargs: object) -> object:
        observed["argv"] = argv
        observed.update(kwargs)
        return type(
            "Completed",
            (),
            {"returncode": 0, "stdout": "", "stderr": ""},
        )()

    monkeypatch.setattr(orchestrator.subprocess, "run", fake_run)
    result = orchestrator.SubprocessCommandRunner().run(
        ("docker", "version"),
        timeout_seconds=3,
    )
    assert result.returncode == 0
    assert observed["env"] == {
        "PATH": orchestrator._SAFE_EXECUTION_PATH,
        "LANG": "C",
        "LC_ALL": "C",
    }
    assert observed["shell"] is False


def test_authorization_signature_and_workspace_binding_are_required(
    tmp_path: Path,
) -> None:
    plan = vllm_compile.build_default_plan()
    repository = tmp_path / "repository"
    model = tmp_path / "model"
    state = tmp_path / "state"
    authorized_workspace = tmp_path / "authorized-workspace"
    replay_workspace = tmp_path / "replay-workspace"
    for path in (repository, model, state):
        path.mkdir()
    authorization = orchestrator.ExecutionAuthorization.from_dict(
        _authorization(plan, workspace=authorized_workspace)
    )
    signature_arguments = _signature_arguments(tmp_path)
    with pytest.raises(orchestrator.CrossoverOrchestratorError, match="host command"):
        orchestrator.HostOrchestrator(
            runner=FakeRunner(
                head="a" * 40,
                image_id=vllm_compile.DERIVED_IMAGE_ID,
                fail_signature=True,
            ),
            plan=plan,
            authorization=authorization,
            repository=repository,
            workspace=authorized_workspace,
            model_path=model,
            state_path=state,
            image_reference="local-image",
            **signature_arguments,
            environ={},
        )
    with pytest.raises(
        orchestrator.CrossoverOrchestratorError, match="exact workspace"
    ):
        orchestrator.HostOrchestrator(
            runner=FakeRunner(
                head="a" * 40,
                image_id=vllm_compile.DERIVED_IMAGE_ID,
            ),
            plan=plan,
            authorization=authorization,
            repository=repository,
            workspace=replay_workspace,
            model_path=model,
            state_path=state,
            image_reference="local-image",
            **signature_arguments,
            environ={},
        )


def test_host_orchestrator_rejects_nonempty_workspace(tmp_path: Path) -> None:
    plan = vllm_compile.build_default_plan()
    repository = tmp_path / "repository"
    model = tmp_path / "model"
    state = tmp_path / "state"
    workspace = tmp_path / "workspace"
    for path in (repository, model, state, workspace):
        path.mkdir()
    (workspace / "prior-attempt.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(orchestrator.CrossoverOrchestratorError, match="must be empty"):
        orchestrator.HostOrchestrator(
            runner=FakeRunner(
                head="a" * 40,
                image_id=vllm_compile.DERIVED_IMAGE_ID,
            ),
            plan=plan,
            authorization=orchestrator.ExecutionAuthorization.from_dict(
                _authorization(plan, workspace=workspace)
            ),
            repository=repository,
            workspace=workspace,
            model_path=model,
            state_path=state,
            image_reference="local-image",
            **_signature_arguments(tmp_path),
            environ={},
        )


def test_host_orchestrator_runs_exact_schedule_and_tears_down(
    tmp_path: Path,
) -> None:
    plan = vllm_compile.build_default_plan()
    repository = tmp_path / "repository"
    model = tmp_path / "model"
    state = tmp_path / "state"
    workspace = tmp_path / "workspace"
    for path in (repository, model, state):
        path.mkdir()
    output_dir = workspace / "raw"
    runner = FakeRunner(
        head="a" * 40,
        image_id=vllm_compile.DERIVED_IMAGE_ID,
        output_dir=output_dir,
    )
    host = orchestrator.HostOrchestrator(
        runner=runner,
        plan=plan,
        authorization=orchestrator.ExecutionAuthorization.from_dict(
            _authorization(plan, workspace=workspace)
        ),
        repository=repository,
        workspace=workspace,
        model_path=model,
        state_path=state,
        image_reference="local-image",
        **_signature_arguments(tmp_path),
        environ={},
    )
    receipt = host.execute()
    docker_runs = [command for command in runner.commands if "run" in command[:5]]
    docker_timeouts = [
        timeout
        for command, timeout in zip(runner.commands, runner.timeouts, strict=True)
        if "run" in command[:5]
    ]
    assert len(docker_runs) == len(plan.schedule) == 32
    assert docker_timeouts == [
        (
            vllm_compile.CONTROLLED_CELL_ALLOWANCE_SECONDS
            if cell.lane == "controlled"
            else vllm_compile.NATURAL_CELL_ALLOWANCE_SECONDS
        )
        - orchestrator.CELL_HOST_CHECK_RESERVE_SECONDS
        for cell in plan.schedule
    ]
    first_command = docker_runs[0]
    assert first_command[:3] == (
        "docker",
        "--host",
        orchestrator.LOCAL_DOCKER_SOCKET,
    )
    assert any(
        command[:5]
        == (
            "docker",
            "--host",
            orchestrator.LOCAL_DOCKER_SOCKET,
            "info",
            "--format",
        )
        for command in runner.commands
    )
    assert f"PYTHONHASHSEED={vllm_compile.SAMPLING_SEED}" in first_command
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in first_command
    assert "VLLM_DISABLE_COMPILE_CACHE=1" in first_command
    assert "VLLM_BATCH_INVARIANT=0" in first_command
    assert "VLLM_NO_USAGE_STATS=1" in first_command
    assert "HF_HUB_OFFLINE=1" in first_command
    assert "TRANSFORMERS_OFFLINE=1" in first_command
    assert "--read-only" in first_command
    assert ("--cap-drop", "ALL") == first_command[
        first_command.index("--cap-drop") : first_command.index("--cap-drop") + 2
    ]
    assert "no-new-privileges:true" in first_command
    assert "/tmp:rw,nosuid,nodev,size=1g" in first_command
    assert f"HOME=/cache/{plan.schedule[0].cell_id}/home" in first_command
    assert receipt["completed_cell_ids"] == [cell.cell_id for cell in plan.schedule]
    assert receipt["status"] == "complete"
    assert receipt["teardown_status"] == "local_cleanup_complete"
    assert receipt["provider_teardown"] is None
    assert receipt["host_shutdown_observed_at"] is None
    assert receipt["external_provider_console_confirmation"] is None
    assert receipt["independently_verified_provider_termination"] is None
    assert runner.inputs.count("3\n") == 32
    assert list((workspace / "cell-caches").iterdir()) == []
    assert list((workspace / "cell-outputs").iterdir()) == []
    assert len(receipt["hardware_observations"]) == 65
    assert (
        len(
            {
                observation["gpu_identity_commitment"]
                for observation in receipt["hardware_observations"]
            }
        )
        == 1
    )
    assert "GPU-aaaaaaaa" not in json.dumps(receipt)
    assert receipt["ledger_abort_failures"] == []
    assert receipt["orchestration_sha256"] == orchestrator._sha256_json(
        {key: value for key, value in receipt.items() if key != "orchestration_sha256"}
    )
    assert (
        json.loads((workspace / "authorization.json").read_text(encoding="utf-8"))
        == host.authorization.to_dict()
    )
    ledger = json.loads((workspace / "budget-ledger.json").read_text(encoding="utf-8"))
    statuses = {
        entry["line_id"]: entry["status"]
        for entry in ledger["entries"]
        if entry["line_id"] in {"preflight", "export", "teardown"}
    }
    assert statuses == {
        "preflight": "completed",
        "export": "completed",
        "teardown": "completed",
    }


def test_late_start_refuses_measurement_but_still_runs_teardown(
    tmp_path: Path,
) -> None:
    plan = vllm_compile.build_default_plan()
    repository = tmp_path / "repository"
    model = tmp_path / "model"
    state = tmp_path / "state"
    workspace = tmp_path / "workspace"
    for path in (repository, model, state):
        path.mkdir()
    billing = datetime.now(timezone.utc) - timedelta(
        seconds=vllm_compile.ACTIVE_PLANNED_SECONDS - 100
    )
    authorization = _authorization(plan, workspace=workspace)
    authorization["authorized_at"] = billing.isoformat()
    authorization["billing_started_at"] = billing.isoformat()
    authorization["scheduled_shutdown_at"] = (
        billing + timedelta(seconds=vllm_compile.ACTIVE_PLANNED_SECONDS)
    ).isoformat()
    authorization["authorization_sha256"] = orchestrator._sha256_json(
        {
            key: value
            for key, value in authorization.items()
            if key != "authorization_sha256"
        }
    )
    runner = FakeRunner(
        head="a" * 40,
        image_id=vllm_compile.DERIVED_IMAGE_ID,
    )
    host = orchestrator.HostOrchestrator(
        runner=runner,
        plan=plan,
        authorization=orchestrator.ExecutionAuthorization.from_dict(authorization),
        repository=repository,
        workspace=workspace,
        model_path=model,
        state_path=state,
        image_reference="local-image",
        **_signature_arguments(tmp_path),
        environ={},
    )
    with pytest.raises(orchestrator.CrossoverOrchestratorError, match="incomplete"):
        host.execute()
    receipt = json.loads(
        (workspace / "orchestration-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["completed_cell_ids"] == []
    assert receipt["teardown_status"] == "local_cleanup_complete"
    assert receipt["status"] == "incomplete"
    assert receipt["orchestration_sha256"] == orchestrator._sha256_json(
        {key: value for key, value in receipt.items() if key != "orchestration_sha256"}
    )
    assert (
        json.loads((workspace / "authorization.json").read_text(encoding="utf-8"))
        == host.authorization.to_dict()
    )
    ledger = json.loads((workspace / "budget-ledger.json").read_text(encoding="utf-8"))
    statuses = {
        entry["line_id"]: entry["status"]
        for entry in ledger["entries"]
        if entry["line_id"] in {"preflight", "export", "teardown"}
    }
    assert statuses == {
        "preflight": "planned",
        "export": "planned",
        "teardown": "completed",
    }


def test_host_orchestrator_failure_is_sanitized_and_teardown_runs(
    tmp_path: Path,
) -> None:
    plan = vllm_compile.build_default_plan()
    repository = tmp_path / "repository"
    model = tmp_path / "model"
    state = tmp_path / "state"
    workspace = tmp_path / "workspace"
    for path in (repository, model, state):
        path.mkdir()
    runner = FakeRunner(
        head="a" * 40,
        image_id=vllm_compile.DERIVED_IMAGE_ID,
        output_dir=workspace / "raw",
        fail_cell=True,
    )
    host = orchestrator.HostOrchestrator(
        runner=runner,
        plan=plan,
        authorization=orchestrator.ExecutionAuthorization.from_dict(
            _authorization(plan, workspace=workspace)
        ),
        repository=repository,
        workspace=workspace,
        model_path=model,
        state_path=state,
        image_reference="local-image",
        **_signature_arguments(tmp_path),
        environ={},
    )
    with pytest.raises(orchestrator.CrossoverOrchestratorError, match="incomplete"):
        host.execute()
    receipt = json.loads(
        (workspace / "orchestration-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "incomplete"
    assert receipt["failure"] == {
        "type": "CrossoverOrchestratorError",
        "reason": "execution_failed",
    }
    assert receipt["teardown_status"] == "local_cleanup_complete"
    assert runner.inputs.count("3\n") == 1
    assert "failed" not in json.dumps(receipt).lower().replace("execution_failed", "")


def test_preflight_failure_still_writes_receipt_and_runs_teardown(
    tmp_path: Path,
) -> None:
    plan = vllm_compile.build_default_plan()
    repository = tmp_path / "repository"
    model = tmp_path / "model"
    state = tmp_path / "state"
    workspace = tmp_path / "workspace"
    for path in (repository, model, state):
        path.mkdir()
    runner = FakeRunner(
        head="c" * 40,
        image_id=vllm_compile.DERIVED_IMAGE_ID,
        output_dir=workspace / "raw",
    )
    host = orchestrator.HostOrchestrator(
        runner=runner,
        plan=plan,
        authorization=orchestrator.ExecutionAuthorization.from_dict(
            _authorization(plan, workspace=workspace)
        ),
        repository=repository,
        workspace=workspace,
        model_path=model,
        state_path=state,
        image_reference="local-image",
        **_signature_arguments(tmp_path),
        environ={},
    )

    with pytest.raises(orchestrator.CrossoverOrchestratorError, match="incomplete"):
        host.execute()

    receipt = json.loads(
        (workspace / "orchestration-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["completed_cell_ids"] == []
    assert receipt["status"] == "incomplete"
    assert receipt["teardown_status"] == "local_cleanup_complete"


def test_offline_bundle_is_deterministic_and_tamper_evident(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    evidence.build_offline_bundle(first, repo_root=Path(__file__).parents[2])
    evidence.build_offline_bundle(second, repo_root=Path(__file__).parents[2])
    evidence.verify_offline_bundle(first, repo_root=Path(__file__).parents[2])
    assert {path.name: path.read_bytes() for path in first.iterdir()} == {
        path.name: path.read_bytes() for path in second.iterdir()
    }
    claims = json.loads((first / "claim-matrix.json").read_text(encoding="utf-8"))
    assert claims["execution_state"] == "not_run"
    assert all(
        item["state"] != "supported"
        for item in claims["claims"]
        if item["claim_id"]
        not in {"offline-protocol-defined", "zero-spend-offline-generation"}
    )
    (first / "README.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(evidence.CrossoverEvidenceError, match="differs"):
        evidence.verify_offline_bundle(first, repo_root=Path(__file__).parents[2])


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("/Users/private/run.json", "private home path"),
        ("/home/private/run.json", "private home path"),
        ("203.0.113.7", "IP address"),
        ("private@example.com", "email address"),
        ("-----BEGIN OPENSSH PRIVATE KEY-----", "private key"),
        ("GPU-00000000-0000-0000-0000-000000000000", "GPU UUID"),
    ],
)
def test_privacy_scan_rejects_private_values(value: str, message: str) -> None:
    with pytest.raises(evidence.CrossoverEvidenceError, match=message):
        evidence._scan_privacy("artifact.txt", value)
