"""Local-only host orchestration for the controlled vLLM crossover protocol.

This module never authenticates to a provider or opens an SSH connection. The
only executable path controls Docker on the current, operator-provisioned host
after validating a separate authorization receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol

from ..._artifact_io import (
    MAX_METADATA_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from ...collectors._shared import atomic_write_text
from .vllm_compile import (
    ABSOLUTE_CEILING_SECONDS,
    ACTIVE_PLANNED_SECONDS,
    ANTICIPATED_RATE_USD_PER_HOUR,
    DERIVED_IMAGE_ID,
    EXPECTED_DRIVER,
    EXPECTED_GPU_NAME,
    EXPECTED_MEMORY_MIB,
    HARD_CAP_USD,
    PROTOCOL_ID,
    SAMPLING_SEED,
    TEARDOWN_ALLOWANCE_SECONDS,
    LifecycleBudgetLedger,
    ScheduleCell,
    VLLMCompileContractError,
    VLLMCompilePlan,
    build_default_plan,
    canonical_decimal,
)

AUTHORIZATION_SCHEMA_VERSION = "1"
ORCHESTRATOR_SCHEMA_VERSION = "1"
RUNNER_MODULE = "llmtracefx.optimizer.lab.qwen3_8b.cloudrift_crossover_runner"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_NONCE = re.compile(r"^[0-9a-f]{32,64}$")
_CREDENTIAL_ENV = re.compile(
    r"(?:TOKEN|PASSWORD|SECRET|API_KEY|PRIVATE_KEY|CREDENTIAL|COOKIE)",
    re.IGNORECASE,
)
_GPU_OBSERVATION_FIELDS = (
    "name",
    "driver_version",
    "memory.total",
    "memory.used",
    "uuid",
    "temperature.gpu",
    "utilization.gpu",
    "power.limit",
    "clocks.sm",
    "compute_cap",
)
MAX_BASELINE_GPU_MEMORY_MIB = 2_048
MAX_IDLE_GPU_TEMPERATURE_C = 80
MAX_IDLE_GPU_UTILIZATION_PERCENT = 5
QUIESCENCE_COMMAND_TIMEOUT_SECONDS = 10
HARDWARE_COMMAND_TIMEOUT_SECONDS = 10
RESET_SYNC_TIMEOUT_SECONDS = 10
RESET_DROP_TIMEOUT_SECONDS = 20
CELL_HOST_CHECK_RESERVE_SECONDS = (
    2 * HARDWARE_COMMAND_TIMEOUT_SECONDS + 2 * QUIESCENCE_COMMAND_TIMEOUT_SECONDS + 10
)


class CrossoverOrchestratorError(RuntimeError):
    """Raised when host execution cannot satisfy the sealed protocol."""


@dataclass(frozen=True)
class ExecutionAuthorization:
    """Explicit authority for one future local-host execution."""

    plan_sha256: str
    source_head: str
    runtime_image_id: str
    experiment_nonce: str
    authorized_at: str
    billing_started_at: str
    scheduled_shutdown_at: str
    authorization_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AUTHORIZATION_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "provider": "CloudRift",
            "approved": True,
            "plan_sha256": self.plan_sha256,
            "source_head": self.source_head,
            "runtime_image_id": self.runtime_image_id,
            "experiment_nonce": self.experiment_nonce,
            "authorized_at": self.authorized_at,
            "billing_started_at": self.billing_started_at,
            "scheduled_shutdown_at": self.scheduled_shutdown_at,
            "rate_usd_per_hour": canonical_decimal(ANTICIPATED_RATE_USD_PER_HOUR),
            "hard_cap_usd": canonical_decimal(HARD_CAP_USD),
            "automatic_retries": 0,
            "provider_access_managed_externally": True,
            "authorization_sha256": self.authorization_sha256,
        }

    @classmethod
    def from_dict(cls, data: Any) -> ExecutionAuthorization:
        if not isinstance(data, dict):
            raise CrossoverOrchestratorError("authorization must be an object")
        expected_keys = {
            "schema_version",
            "protocol_id",
            "provider",
            "approved",
            "plan_sha256",
            "source_head",
            "runtime_image_id",
            "experiment_nonce",
            "authorized_at",
            "billing_started_at",
            "scheduled_shutdown_at",
            "rate_usd_per_hour",
            "hard_cap_usd",
            "automatic_retries",
            "provider_access_managed_externally",
            "authorization_sha256",
        }
        if set(data) != expected_keys:
            raise CrossoverOrchestratorError("authorization keys differ")
        fixed = {
            "schema_version": AUTHORIZATION_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "provider": "CloudRift",
            "approved": True,
            "rate_usd_per_hour": canonical_decimal(ANTICIPATED_RATE_USD_PER_HOUR),
            "hard_cap_usd": canonical_decimal(HARD_CAP_USD),
            "automatic_retries": 0,
            "provider_access_managed_externally": True,
        }
        if any(data[key] != value for key, value in fixed.items()):
            raise CrossoverOrchestratorError(
                "authorization does not match the approved execution envelope"
            )
        if not isinstance(data["plan_sha256"], str) or not _SHA256.fullmatch(
            data["plan_sha256"]
        ):
            raise CrossoverOrchestratorError("authorization plan hash is invalid")
        if not isinstance(data["source_head"], str) or not _COMMIT.fullmatch(
            data["source_head"]
        ):
            raise CrossoverOrchestratorError("authorization source head is invalid")
        if not isinstance(data["runtime_image_id"], str) or not _SHA256.fullmatch(
            data["runtime_image_id"]
        ):
            raise CrossoverOrchestratorError("authorization image ID is invalid")
        if data["runtime_image_id"] != DERIVED_IMAGE_ID:
            raise CrossoverOrchestratorError(
                "authorization image ID differs from the pinned runtime"
            )
        if not isinstance(data["experiment_nonce"], str) or not _NONCE.fullmatch(
            data["experiment_nonce"]
        ):
            raise CrossoverOrchestratorError("authorization nonce is invalid")
        authorized_at = _parse_timestamp(
            data["authorized_at"], "authorization authorized_at"
        )
        billing_started_at = _parse_timestamp(
            data["billing_started_at"], "authorization billing_started_at"
        )
        scheduled_shutdown_at = _parse_timestamp(
            data["scheduled_shutdown_at"], "authorization scheduled_shutdown_at"
        )
        if scheduled_shutdown_at != billing_started_at + timedelta(
            seconds=ACTIVE_PLANNED_SECONDS
        ):
            raise CrossoverOrchestratorError(
                "scheduled shutdown must equal billing start plus active envelope"
            )
        if authorized_at > billing_started_at:
            raise CrossoverOrchestratorError(
                "authorization must precede or equal the billing start"
            )
        expected_authorization_sha256 = _sha256_json(
            {key: value for key, value in data.items() if key != "authorization_sha256"}
        )
        if data["authorization_sha256"] != expected_authorization_sha256:
            raise CrossoverOrchestratorError(
                "authorization content hash does not verify"
            )
        return cls(
            plan_sha256=data["plan_sha256"],
            source_head=data["source_head"],
            runtime_image_id=data["runtime_image_id"],
            experiment_nonce=data["experiment_nonce"],
            authorized_at=data["authorized_at"],
            billing_started_at=data["billing_started_at"],
            scheduled_shutdown_at=data["scheduled_shutdown_at"],
            authorization_sha256=data["authorization_sha256"],
        )

    @classmethod
    def read(cls, path: Path) -> ExecutionAuthorization:
        try:
            payload = json.loads(
                read_bounded_regular_text(path, MAX_METADATA_ARTIFACT_BYTES),
                parse_constant=reject_non_finite_json_constant,
            )
        except (OSError, ArtifactReadError, ValueError, RecursionError) as exc:
            raise CrossoverOrchestratorError(
                f"authorization could not be read safely: {exc}"
            ) from exc
        return cls.from_dict(payload)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        input_text: str | None = None,
    ) -> CommandResult: ...


class SubprocessCommandRunner:
    """Run fixed argv commands without an interpolation shell."""

    def run(
        self,
        argv: Sequence[str],
        *,
        timeout_seconds: int,
        input_text: str | None = None,
    ) -> CommandResult:
        completed = subprocess.run(
            list(argv),
            input=input_text,
            capture_output=True,
            text=True,
            check=False,
            shell=False,
            timeout=timeout_seconds,
        )
        return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise CrossoverOrchestratorError(f"{field} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CrossoverOrchestratorError(
            f"{field} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CrossoverOrchestratorError(f"{field} must include a timezone")
    return parsed


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_text(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    )


def _write_json(path: Path, value: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
    )


def _checked(
    runner: CommandRunner,
    argv: Sequence[str],
    *,
    timeout_seconds: int,
    input_text: str | None = None,
) -> CommandResult:
    result = runner.run(
        argv,
        timeout_seconds=timeout_seconds,
        input_text=input_text,
    )
    if result.returncode != 0:
        raise CrossoverOrchestratorError(f"host command failed: {Path(argv[0]).name}")
    return result


def _reject_credential_environment(environ: Mapping[str, str]) -> None:
    names = sorted(
        name
        for name, value in environ.items()
        if value and _CREDENTIAL_ENV.search(name)
    )
    if names:
        raise CrossoverOrchestratorError(
            "credential-shaped environment variables are forbidden: " + ", ".join(names)
        )


def _require_safe_directory(path: Path, *, label: str, create: bool = False) -> Path:
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise CrossoverOrchestratorError(f"{label} must be a non-symlink directory")
    return path.resolve()


def offline_plan_document() -> dict[str, Any]:
    """Return the deterministic no-spend plan/refusal document."""

    plan = build_default_plan()
    return {
        "schema_version": ORCHESTRATOR_SCHEMA_VERSION,
        "kind": "llmtracefx.vllm_crossover.offline_plan",
        "protocol_id": PROTOCOL_ID,
        "plan": plan.to_dict(),
        "execution_authorized": False,
        "offline_only": True,
        "network_request_performed": False,
        "provider_authentication_used": False,
        "instance_created": False,
        "model_downloaded": False,
        "gpu_used": False,
        "spend_usd": "0",
        "blockers": [
            "No explicit execution authorization receipt is present.",
            "No fresh provider price and billing-start receipt is present.",
            "No operator-provisioned host or runtime-image inspection receipt is present.",
            "Provider teardown must be externally confirmed after any future run.",
        ],
        "unsupported_claims": [
            "compilation crossover",
            "performance improvement",
            "output identity",
            "correctness preservation",
            "runtime component timing",
            "provider-reported spend",
        ],
    }


def _cell_budget_lifecycle(plan: VLLMCompilePlan, cell: ScheduleCell) -> Any:
    matches = [
        lifecycle
        for lifecycle in plan.budget_lifecycles
        if lifecycle.cell_id == cell.cell_id
    ]
    if len(matches) != 1:
        raise CrossoverOrchestratorError("cell budget lifecycle is not unique")
    return matches[0]


def _docker_command(
    *,
    cell: ScheduleCell,
    image_reference: str,
    repository: Path,
    model_path: Path,
    state_path: Path,
    output_dir: Path,
    cache_dir: Path,
    experiment_nonce: str,
) -> tuple[str, ...]:
    return (
        "docker",
        "run",
        "--rm",
        "--name",
        f"llmtracefx-{cell.cell_id}",
        "--label",
        f"llmtracefx.protocol={PROTOCOL_ID}",
        "--gpus",
        "device=0",
        "--cpus",
        "4",
        "--memory",
        "32g",
        "--shm-size",
        "8g",
        "--pids-limit",
        "4096",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges:true",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=1g",
        "--network",
        "none",
        "--env",
        "PYTHONPATH=/workspace",
        "--env",
        f"PYTHONHASHSEED={SAMPLING_SEED}",
        "--env",
        "CUBLAS_WORKSPACE_CONFIG=:4096:8",
        "--env",
        "VLLM_DISABLE_COMPILE_CACHE=1",
        "--env",
        "VLLM_BATCH_INVARIANT=0",
        "--env",
        "VLLM_NO_USAGE_STATS=1",
        "--env",
        "HF_HUB_OFFLINE=1",
        "--env",
        "TRANSFORMERS_OFFLINE=1",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        f"HOME=/cache/{cell.cell_id}/home",
        "--env",
        f"HF_HOME=/cache/{cell.cell_id}/huggingface",
        "--env",
        f"XDG_CACHE_HOME=/cache/{cell.cell_id}/xdg",
        "--mount",
        f"type=bind,src={repository},dst=/workspace,readonly",
        "--mount",
        f"type=bind,src={model_path},dst=/model,readonly",
        "--mount",
        f"type=bind,src={state_path},dst=/state,readonly",
        "--mount",
        f"type=bind,src={output_dir},dst=/output",
        "--mount",
        f"type=bind,src={cache_dir},dst=/cache",
        "--entrypoint",
        "/usr/bin/python3",
        image_reference,
        "-m",
        RUNNER_MODULE,
        "run-cell",
        "--cell-id",
        cell.cell_id,
        "--model-path",
        "/model",
        "--state-path",
        "/state",
        "--cache-root",
        "/cache",
        "--output",
        f"/output/{cell.cell_id}.json",
        "--experiment-nonce",
        experiment_nonce,
    )


class HostOrchestrator:
    """Execute the sealed schedule on one already-provisioned local host."""

    def __init__(
        self,
        *,
        runner: CommandRunner,
        plan: VLLMCompilePlan,
        authorization: ExecutionAuthorization,
        repository: Path,
        workspace: Path,
        model_path: Path,
        state_path: Path,
        image_reference: str,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.runner = runner
        self.plan = plan
        self.authorization = authorization
        self.billing_started_at = _parse_timestamp(
            authorization.billing_started_at,
            "authorization billing_started_at",
        )
        self.repository = _require_safe_directory(repository, label="repository")
        self.workspace = _require_safe_directory(
            workspace, label="workspace", create=True
        )
        if any(self.workspace.iterdir()):
            raise CrossoverOrchestratorError(
                "workspace must be empty; retries and resumed execution are forbidden"
            )
        self.model_path = _require_safe_directory(model_path, label="model path")
        self.state_path = _require_safe_directory(state_path, label="state path")
        if not isinstance(image_reference, str) or not image_reference:
            raise CrossoverOrchestratorError("image reference must be non-empty")
        self.image_reference = image_reference
        _reject_credential_environment(os.environ if environ is None else environ)
        if authorization.plan_sha256 != plan.content_sha256:
            raise CrossoverOrchestratorError("authorization plan hash differs")
        self.output_dir = self.workspace / "raw"
        self.cell_output_root = self.workspace / "cell-outputs"
        self.cache_root = self.workspace / "cell-caches"
        self.output_dir.mkdir(exist_ok=True)
        self.cell_output_root.mkdir(exist_ok=True)
        self.cache_root.mkdir(exist_ok=True)
        self.operation_receipts: list[dict[str, Any]] = []
        self.hardware_observations: list[dict[str, Any]] = []
        self.ledger_abort_failures: list[str] = []
        self._gpu_identity_commitment: str | None = None
        self._power_limit_watts: float | None = None
        _write_json(self.workspace / "authorization.json", authorization.to_dict())
        self.ledger = LifecycleBudgetLedger.initialize(
            self.workspace / "budget-ledger.json",
            plan=plan,
            git_head=authorization.source_head,
            workspace_path=self.workspace,
        )

    def _billing_elapsed_seconds(self) -> int:
        elapsed = (datetime.now(timezone.utc) - self.billing_started_at).total_seconds()
        if elapsed < 0:
            raise CrossoverOrchestratorError("billing start is in the future")
        if elapsed >= ABSOLUTE_CEILING_SECONDS:
            raise CrossoverOrchestratorError("absolute billed-time ceiling reached")
        return math.ceil(elapsed)

    def _require_operation_budget(self, lifecycle: Any) -> None:
        if lifecycle.line_id == "teardown":
            return
        elapsed = self._billing_elapsed_seconds()
        if (
            elapsed + lifecycle.planned_seconds + TEARDOWN_ALLOWANCE_SECONDS
            > ACTIVE_PLANNED_SECONDS
        ):
            raise CrossoverOrchestratorError(
                "operation would consume the mandatory teardown window"
            )

    def _preflight(self) -> None:
        head = _checked(
            self.runner,
            ("git", "-C", str(self.repository), "rev-parse", "HEAD"),
            timeout_seconds=30,
        ).stdout.strip()
        if head != self.authorization.source_head:
            raise CrossoverOrchestratorError("checked-out source head differs")
        dirty = _checked(
            self.runner,
            ("git", "-C", str(self.repository), "status", "--porcelain"),
            timeout_seconds=30,
        ).stdout.strip()
        if dirty:
            raise CrossoverOrchestratorError("execution checkout must be clean")
        image_id = _checked(
            self.runner,
            ("docker", "image", "inspect", "--format", "{{.Id}}", self.image_reference),
            timeout_seconds=30,
        ).stdout.strip()
        if image_id != self.authorization.runtime_image_id:
            raise CrossoverOrchestratorError("runtime image identity differs")
        self._reset()
        self._observe_hardware(observation_id="preflight-after-reset")

    def _require_quiescent(self) -> None:
        containers = _checked(
            self.runner,
            (
                "docker",
                "ps",
                "--filter",
                f"label=llmtracefx.protocol={PROTOCOL_ID}",
                "--format",
                "{{.ID}}",
            ),
            timeout_seconds=QUIESCENCE_COMMAND_TIMEOUT_SECONDS,
        ).stdout.split()
        if containers:
            raise CrossoverOrchestratorError("an experiment container is still live")
        processes = _checked(
            self.runner,
            (
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ),
            timeout_seconds=QUIESCENCE_COMMAND_TIMEOUT_SECONDS,
        ).stdout.split()
        if processes:
            raise CrossoverOrchestratorError("a GPU compute process is still live")

    def _reset(self) -> None:
        self._require_quiescent()
        _checked(
            self.runner,
            ("sync",),
            timeout_seconds=RESET_SYNC_TIMEOUT_SECONDS,
        )
        _checked(
            self.runner,
            ("sudo", "tee", "/proc/sys/vm/drop_caches"),
            timeout_seconds=RESET_DROP_TIMEOUT_SECONDS,
            input_text="3\n",
        )

    def _observe_hardware(self, *, observation_id: str) -> None:
        result = _checked(
            self.runner,
            (
                "nvidia-smi",
                "--query-gpu=" + ",".join(_GPU_OBSERVATION_FIELDS),
                "--format=csv,noheader,nounits",
            ),
            timeout_seconds=HARDWARE_COMMAND_TIMEOUT_SECONDS,
        ).stdout.strip()
        rows = [row for row in result.splitlines() if row.strip()]
        if len(rows) != 1:
            raise CrossoverOrchestratorError(
                "GPU resource observation must contain exactly one device"
            )
        values = [value.strip() for value in rows[0].split(",")]
        if len(values) != len(_GPU_OBSERVATION_FIELDS):
            raise CrossoverOrchestratorError("GPU resource observation is incomplete")
        (
            name,
            driver,
            memory_total,
            memory_used,
            gpu_uuid,
            temperature,
            utilization,
            power_limit,
            sm_clock,
            compute_capability,
        ) = values
        try:
            memory_total_mib = int(memory_total)
            memory_used_mib = int(memory_used)
            temperature_c = int(temperature)
            utilization_percent = int(utilization)
            power_limit_watts = float(power_limit)
            sm_clock_mhz = int(sm_clock)
        except ValueError as exc:
            raise CrossoverOrchestratorError(
                "GPU resource observation is not numeric"
            ) from exc
        if (
            name != EXPECTED_GPU_NAME
            or driver != EXPECTED_DRIVER
            or memory_total_mib != EXPECTED_MEMORY_MIB
            or compute_capability != "8.9"
            or not gpu_uuid.startswith("GPU-")
        ):
            raise CrossoverOrchestratorError(
                "GPU resource identity differs from the approved VM"
            )
        if (
            memory_used_mib > MAX_BASELINE_GPU_MEMORY_MIB
            or temperature_c > MAX_IDLE_GPU_TEMPERATURE_C
            or utilization_percent > MAX_IDLE_GPU_UTILIZATION_PERCENT
            or power_limit_watts <= 0
            or sm_clock_mhz <= 0
        ):
            raise CrossoverOrchestratorError("GPU resource or thermal guard failed")
        commitment = _sha256_json(
            {
                "public_experiment_nonce": self.authorization.experiment_nonce,
                "private_gpu_uuid_sha256": _sha256_text(gpu_uuid),
            }
        )
        if self._gpu_identity_commitment not in (None, commitment):
            raise CrossoverOrchestratorError(
                "GPU identity commitment changed during execution"
            )
        if self._power_limit_watts not in (None, power_limit_watts):
            raise CrossoverOrchestratorError("GPU power limit changed during execution")
        self._gpu_identity_commitment = commitment
        self._power_limit_watts = power_limit_watts
        self.hardware_observations.append(
            {
                "observation_id": observation_id,
                "clock_domain": "host_perf_counter",
                "host_perf_counter_ns": time.perf_counter_ns(),
                "gpu_identity_commitment": commitment,
                "gpu_name": name,
                "driver_version": driver,
                "memory_total_mib": memory_total_mib,
                "memory_used_mib": memory_used_mib,
                "temperature_c": temperature_c,
                "utilization_percent": utilization_percent,
                "power_limit_watts": power_limit_watts,
                "sm_clock_mhz": sm_clock_mhz,
                "compute_capability": compute_capability,
            }
        )

    def _reserve_and_run(
        self,
        *,
        command_id: str,
        lifecycle: Any,
        argv: Sequence[str],
        action: Any,
    ) -> None:
        self._require_operation_budget(lifecycle)
        self.ledger.reserve(
            command_id,
            line_id=lifecycle.line_id,
            lifecycle_id=lifecycle.lifecycle_id,
            ceiling_usd=lifecycle.ceiling_usd,
            argv=argv,
            reserved_at=_now(),
        )
        before_ns = time.perf_counter_ns()
        try:
            action()
            duration_ns = time.perf_counter_ns() - before_ns
            actual = math.ceil(duration_ns / 1_000_000_000)
            self.ledger.complete(
                command_id,
                completed_at=_now(),
                actual_seconds=actual,
            )
        except Exception:
            ended_ns = time.perf_counter_ns()
            duration_ns = ended_ns - before_ns
            self.operation_receipts.append(
                {
                    "command_id": command_id,
                    "lifecycle_id": lifecycle.lifecycle_id,
                    "line_id": lifecycle.line_id,
                    "clock_domain": "host_perf_counter",
                    "started_ns": before_ns,
                    "ended_ns": ended_ns,
                    "duration_ns": duration_ns,
                    "status": "aborted",
                }
            )
            try:
                self.ledger.abort(
                    command_id,
                    aborted_at=_now(),
                    reason="operation_failed",
                )
            except (OSError, ValueError, VLLMCompileContractError) as ledger_exc:
                self.ledger_abort_failures.append(type(ledger_exc).__name__)
            raise
        ended_ns = before_ns + duration_ns
        self.operation_receipts.append(
            {
                "command_id": command_id,
                "lifecycle_id": lifecycle.lifecycle_id,
                "line_id": lifecycle.line_id,
                "clock_domain": "host_perf_counter",
                "started_ns": before_ns,
                "ended_ns": ended_ns,
                "duration_ns": duration_ns,
                "status": "completed",
            }
        )

    def _cleanup_known_caches(self) -> None:
        root = self.cache_root.resolve()
        for child in list(root.iterdir()):
            resolved = child.resolve()
            if resolved.parent != root or child.is_symlink():
                raise CrossoverOrchestratorError("cell cache path is unsafe")
            shutil.rmtree(resolved)

    def _execute_cell(
        self,
        *,
        cell: ScheduleCell,
        command: Sequence[str],
        timeout_seconds: int,
        cache_dir: Path,
        cell_output_dir: Path,
    ) -> None:
        self._observe_hardware(observation_id=f"{cell.cell_id}-before-container")
        _checked(
            self.runner,
            command,
            timeout_seconds=timeout_seconds,
        )
        self._require_quiescent()
        self._observe_hardware(observation_id=f"{cell.cell_id}-after-container")
        expected_outputs = {
            f"{cell.cell_id}.json",
            f".{cell.cell_id}-progress.json",
        }
        staged = list(cell_output_dir.iterdir())
        if (
            cell_output_dir.is_symlink()
            or cell_output_dir.resolve().parent != self.cell_output_root.resolve()
            or {path.name for path in staged} != expected_outputs
            or any(path.is_symlink() or not path.is_file() for path in staged)
        ):
            raise CrossoverOrchestratorError("cell output staging inventory differs")
        for path in staged:
            destination = self.output_dir / path.name
            if destination.exists():
                raise CrossoverOrchestratorError(
                    "cell output destination already exists"
                )
            path.replace(destination)
        cell_output_dir.rmdir()
        resolved_cache = cache_dir.resolve()
        if (
            cache_dir.is_symlink()
            or resolved_cache.parent != self.cache_root.resolve()
            or resolved_cache.name != cell.cell_id
        ):
            raise CrossoverOrchestratorError("completed cell cache path is unsafe")
        shutil.rmtree(resolved_cache)

    def _teardown_local(self) -> None:
        self._require_quiescent()
        self._cleanup_known_caches()
        for child in list(self.cell_output_root.iterdir()):
            resolved = child.resolve()
            if resolved.parent != self.cell_output_root.resolve() or child.is_symlink():
                raise CrossoverOrchestratorError("cell output staging path is unsafe")
            shutil.rmtree(resolved)

    def execute(self) -> dict[str, Any]:
        """Run all cells once; any failure produces an incomplete receipt."""

        preflight = self.plan.budget_lifecycles[0]
        completed_cells: list[str] = []
        resets = iter(
            item for item in self.plan.budget_lifecycles if item.line_id == "reset"
        )
        try:
            self._reserve_and_run(
                command_id="preflight",
                lifecycle=preflight,
                argv=(
                    "host-preflight",
                    "initial-cache-reset",
                    self.authorization.plan_sha256,
                ),
                action=self._preflight,
            )
            for schedule_index, cell in enumerate(self.plan.schedule):
                lifecycle = _cell_budget_lifecycle(self.plan, cell)
                cache_dir = self.cache_root / cell.cell_id
                cell_output_dir = self.cell_output_root / cell.cell_id
                cache_dir.mkdir()
                cell_output_dir.mkdir()
                command = _docker_command(
                    cell=cell,
                    image_reference=self.image_reference,
                    repository=self.repository,
                    model_path=self.model_path,
                    state_path=self.state_path,
                    output_dir=cell_output_dir,
                    cache_dir=cache_dir,
                    experiment_nonce=self.authorization.experiment_nonce,
                )
                self._reserve_and_run(
                    command_id=f"cell-{schedule_index + 1:02d}",
                    lifecycle=lifecycle,
                    argv=command,
                    action=lambda command=command, lifecycle=lifecycle, cell=cell, cache_dir=cache_dir, cell_output_dir=cell_output_dir: (
                        self._execute_cell(
                            cell=cell,
                            command=command,
                            timeout_seconds=(
                                lifecycle.planned_seconds
                                - CELL_HOST_CHECK_RESERVE_SECONDS
                            ),
                            cache_dir=cache_dir,
                            cell_output_dir=cell_output_dir,
                        )
                    ),
                )
                completed_cells.append(cell.cell_id)
                if schedule_index + 1 < len(self.plan.schedule):
                    reset = next(resets)
                    self._reserve_and_run(
                        command_id=f"reset-{schedule_index + 1:02d}",
                        lifecycle=reset,
                        argv=("sync", "drop-host-page-cache", "require-quiescent"),
                        action=self._reset,
                    )

            export = next(
                item for item in self.plan.budget_lifecycles if item.line_id == "export"
            )
            self._reserve_and_run(
                command_id="export",
                lifecycle=export,
                argv=("validate-raw-receipts",),
                action=self._require_all_cell_outputs,
            )
            status = "complete"
            failure = None
        except Exception as exc:
            status = "incomplete"
            failure = {"type": type(exc).__name__, "reason": "execution_failed"}
        finally:
            teardown = next(
                item
                for item in self.plan.budget_lifecycles
                if item.line_id == "teardown"
            )
            try:
                self._reserve_and_run(
                    command_id="teardown",
                    lifecycle=teardown,
                    argv=("require-quiescent", "remove-cell-caches"),
                    action=self._teardown_local,
                )
                teardown_status = "local_cleanup_complete"
            except Exception:
                teardown_status = "local_cleanup_incomplete"
                status = "incomplete"

        receipt = {
            "schema_version": ORCHESTRATOR_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "plan_sha256": self.plan.content_sha256,
            "source_head": self.authorization.source_head,
            "runtime_image_id": self.authorization.runtime_image_id,
            "authorization_sha256": self.authorization.authorization_sha256,
            "scheduled_shutdown_at": self.authorization.scheduled_shutdown_at,
            "repository_path_sha256": _sha256_text(str(self.repository)),
            "workspace_path_sha256": _sha256_text(str(self.workspace)),
            "completed_cell_ids": completed_cells,
            "operation_receipts": self.operation_receipts,
            "hardware_observations": self.hardware_observations,
            "ledger_abort_failures": self.ledger_abort_failures,
            "status": status,
            "failure": failure,
            "teardown_status": teardown_status,
            "host_shutdown_observed_at": None,
            "host_shutdown_observed_null_reason": (
                "the local process cannot observe its own later host shutdown"
            ),
            "external_provider_console_confirmation": None,
            "external_provider_console_confirmation_null_reason": (
                "external operator confirmation was not supplied to the local runner"
            ),
            "independently_verified_provider_termination": None,
            "independently_verified_provider_termination_null_reason": (
                "no provider API receipt is available"
            ),
            "provider_teardown": None,
            "provider_teardown_null_reason": (
                "provider teardown remains externally user-confirmed"
            ),
            "ledger_sha256": _sha256_text(
                (self.workspace / "budget-ledger.json").read_text(encoding="utf-8")
            ),
        }
        receipt["orchestration_sha256"] = _sha256_json(receipt)
        _write_json(self.workspace / "orchestration-receipt.json", receipt)
        if status != "complete":
            raise CrossoverOrchestratorError(
                "execution incomplete; inspect sanitized orchestration receipt"
            )
        return receipt

    def _require_all_cell_outputs(self) -> None:
        expected = {
            name
            for cell in self.plan.schedule
            for name in (
                f"{cell.cell_id}.json",
                f".{cell.cell_id}-progress.json",
            )
        }
        children = list(self.output_dir.iterdir())
        if any(path.is_symlink() or not path.is_file() for path in children):
            raise CrossoverOrchestratorError(
                "raw cell output contains a non-regular artifact"
            )
        actual = {path.name for path in children}
        if actual != expected:
            raise CrossoverOrchestratorError("raw cell output inventory differs")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmtracefx-vllm-crossover",
        description=(
            "Plan the controlled vLLM crossover offline by default. The run "
            "subcommand controls Docker only on an already-provisioned local host."
        ),
        allow_abbrev=False,
    )
    subparsers = parser.add_subparsers(dest="action")
    plan = subparsers.add_parser("plan", allow_abbrev=False)
    plan.add_argument("--output", type=Path)
    verify = subparsers.add_parser("verify-plan", allow_abbrev=False)
    verify.add_argument("--plan", required=True, type=Path)
    run = subparsers.add_parser("run", allow_abbrev=False)
    run.add_argument("--plan", required=True, type=Path)
    run.add_argument("--authorization", required=True, type=Path)
    run.add_argument("--repository", required=True, type=Path)
    run.add_argument("--workspace", required=True, type=Path)
    run.add_argument("--model-path", required=True, type=Path)
    run.add_argument("--state-path", required=True, type=Path)
    run.add_argument("--image-reference", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    action = args.action or "plan"
    try:
        if action == "plan":
            document = offline_plan_document()
            rendered = (
                json.dumps(
                    document,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )
            if getattr(args, "output", None) is not None:
                atomic_write_text(args.output, rendered)
            print(rendered, end="")
            return 0
        plan = VLLMCompilePlan.read_json(args.plan)
        if action == "verify-plan":
            print(plan.content_sha256)
            return 0
        authorization = ExecutionAuthorization.read(args.authorization)
        orchestrator = HostOrchestrator(
            runner=SubprocessCommandRunner(),
            plan=plan,
            authorization=authorization,
            repository=args.repository,
            workspace=args.workspace,
            model_path=args.model_path,
            state_path=args.state_path,
            image_reference=args.image_reference,
        )
        orchestrator.execute()
        return 0
    except (
        OSError,
        ValueError,
        subprocess.SubprocessError,
        VLLMCompileContractError,
        CrossoverOrchestratorError,
    ) as exc:
        print(f"llmtracefx-vllm-crossover: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
