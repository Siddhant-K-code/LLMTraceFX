"""Provider-neutral host orchestration for ``qwen3-8b-vllm-kv-truth-v1``.

This module is the one place that is allowed to build a real, executable
``ssh``/``scp``/``docker`` argv for the CloudRift vLLM KV-cache truth audit
plan's remote protocol -- and the one place that enforces every hard
execution gate before it may run. It never itself opens a socket, imports
``vllm``/``torch``, or reads a private key's *contents*; every stage method
builds argv and hands it to an injectable :class:`CommandRunner`, so this
module's own tests drive the complete stage/state machine with a hermetic
fake and never perform SSH, network, provider, model, image, or GPU I/O.

Design constraints this module exists to satisfy (see the approved plan and
the delegated task's hard requirements):

1. **No individual host/user/key/known-hosts/remote-path CLI args.** The
   orchestrator is constructed from a single :class:`ProtectedExecutionConfig`
   loaded from one protected file path, plus a single
   :class:`RunAuthorization` loaded from one explicit authorization JSON
   path. Sensitive fields are ``repr=False`` and are never written to any
   evidence artifact, receipt, or log line this module produces --
   :meth:`StrictSSHOptions.public_record` is the only representation that
   ever feeds anything persisted.
2. **A signed, self-sealed run authorization** binds the protocol ID, the
   exact repository HEAD for the *future* merged commit (never the current
   feature commit -- this module has no code path that reads or defaults to
   ``git rev-parse HEAD``), the base image digest, the expected derived
   image source digest, the exact model/revision/inventory digest, GPU
   expectations, rate/cap, billing/boot timestamp, the derived operational
   cutoff and cleanup reserve, authorization time/expiry, zero
   retries/replacements, and a nonce -- see :class:`RunAuthorization`. No
   absolute timestamp from the plan's already-terminated VM is hardcoded
   anywhere in this module; every cutoff is computed from the
   caller-supplied boot time, rate, cap, and explicit operational cutoff.
3. Model acquisition is verified file-by-file against the already-committed
   15-file/16,397,461,266-byte SHA-256 inventory
   (``qwen3-8b-conversion-manifest-v1.json``), not just a count/byte total.
4. Every GPU-lane container runs with ``--network none``, the fixed offline
   environment, and a run-scoped Docker label so teardown can find and stop
   *only* this run's containers -- never every container on the host.
5. Teardown always runs, in a fixed order, transferring and locally
   verifying evidence *before* any remote deletion, and never issues an
   unscoped ``rm -rf`` or ``docker rm``/``docker stop`` against unvalidated
   input.

The in-container runner generates its own identity receipt and runtime
attestation from installed package/source/model/GPU state. This host module
supplies only independently verified expectations (checked source HEAD,
inspected derived image ID, immutable model identity, and GPU pins); caller
strings alone never establish runtime support.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shlex
import stat
import subprocess
import tarfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from llmtracefx.cache_audit.adapters.vllm import (
    REQUIRED_VLLM_COMMIT,
    REQUIRED_VLLM_VERSION,
)
from llmtracefx.deploy.errors import DeploymentPlanError
from llmtracefx.optimizer._artifact_io import (
    ArtifactReadError,
    read_bounded_regular_bytes,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from vllm_kv_truth.runner import (
    BASE_IMAGE_REFERENCE,
    EXPECTED_DRIVER,
    EXPECTED_GPU_NAME,
    EXPECTED_MEMORY_MIB,
    EXPECTED_MODEL_BYTES,
    EXPECTED_MODEL_FILE_COUNT,
    MODEL_ID,
    MODEL_REVISION,
    PROTOCOL_ID,
    REQUIRED_ENVIRONMENT_VARIABLES,
    VLLM_SOURCE_COMMIT,
    KVTruthProtocolError,
    verify_protocol_receipt,
)

from . import evidence

MAX_CONFIG_ARTIFACT_BYTES = 64 * 1024
MAX_AUTHORIZATION_ARTIFACT_BYTES = 64 * 1024
MAX_MANIFEST_ARTIFACT_BYTES = 4 * 1024 * 1024
MAX_SOURCE_ARCHIVE_BYTES = 256 * 1024 * 1024

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_SHA256_REF = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_HEX = re.compile(r"^[0-9a-f]{40}$")
_NONCE_HEX = re.compile(r"^[0-9a-f]{32,64}$")
_SAFE_REMOTE_PATH = re.compile(r"^/[A-Za-z0-9._/-]{0,4096}$")
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_HOSTNAME_OR_IP = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9.:-]{0,253}[A-Za-z0-9])?$")
_SAFE_USER = re.compile(r"^[A-Za-z_][A-Za-z0-9._-]{0,31}$")

EXPECTED_GPU_COMPUTE_CAPABILITY = "8.9"
MINIMUM_HOST_RAM_BYTES = 48_000_000_000
MINIMUM_DISK_FREE_BYTES = 250_000_000_000

MODEL_CONVERSION_MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "llmtracefx"
    / "optimizer"
    / "lab"
    / "qwen3_8b"
    / "data"
    / "qwen3-8b-conversion-manifest-v1.json"
)


class HostOrchestrationError(DeploymentPlanError):
    """Raised whenever a config/authorization/stage invariant is violated."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def _parse_utc(value: Any, *, field_name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise HostOrchestrationError(f"{field_name} must be an ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise HostOrchestrationError(
            f"{field_name} must be an ISO-8601 timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise HostOrchestrationError(f"{field_name} must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def _canonical_timestamp(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _canonical_decimal(value: Any, *, field_name: str, minimum: str = "0") -> Decimal:
    if not isinstance(value, str) or not value:
        raise HostOrchestrationError(f"{field_name} must be a decimal string")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise HostOrchestrationError(f"{field_name} is not a valid decimal") from exc
    if not parsed.is_finite() or parsed <= Decimal(minimum):
        raise HostOrchestrationError(f"{field_name} must be a finite positive decimal")
    quantized = parsed.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
    if str(parsed) != _money_str(quantized) and parsed != quantized:
        raise HostOrchestrationError(f"{field_name} must already be canonical")
    return quantized


def _money_str(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN), "f")


def _require_safe_regular_file(path: Path, *, label: str) -> Path:
    """Reject a symlink or non-regular file without following it.

    Uses ``os.lstat`` (never ``stat``/``os.stat``) so a symlink is detected
    *before* any attempt to resolve or open it, closing the classic
    TOCTOU/symlink-substitution gap a naive ``path.is_file()`` check leaves
    open.
    """

    try:
        info = path.lstat()
    except OSError as exc:
        raise HostOrchestrationError(f"{label} could not be inspected: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise HostOrchestrationError(f"{label} must not be a symlink")
    if not stat.S_ISREG(info.st_mode):
        raise HostOrchestrationError(f"{label} must be a regular file")
    return path


def _require_private_key_permissions(path: Path, *, label: str) -> Path:
    """Require exactly ``0600`` (owner read/write only, no group/other bits)."""

    _require_safe_regular_file(path, label=label)
    mode = stat.S_IMODE(path.lstat().st_mode)
    if mode != 0o600:
        raise HostOrchestrationError(
            f"{label} must be mode 0600 (owner read/write only); found " f"{oct(mode)}"
        )
    return path


def _require_not_group_or_world_readable(path: Path, *, label: str) -> Path:
    """Require no group/other permission bits at all (``mode & 0o077 == 0``)."""

    _require_safe_regular_file(path, label=label)
    mode = stat.S_IMODE(path.lstat().st_mode)
    if mode & 0o077:
        raise HostOrchestrationError(
            f"{label} must not be group- or world-readable/writable; found "
            f"{oct(mode)}"
        )
    return path


def _require_safe_remote_path(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or _SAFE_REMOTE_PATH.fullmatch(value) is None:
        raise HostOrchestrationError(
            f"{field_name} must be an absolute path of safe characters"
        )
    if any(part == ".." for part in value.split("/")):
        raise HostOrchestrationError(f"{field_name} must not contain '..'")
    return value


def _require_label(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or _SAFE_LABEL.fullmatch(value) is None:
        raise HostOrchestrationError(
            f"{field_name} must be a short label matching "
            "[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
        )
    return value


def _require_pattern(value: Any, pattern: re.Pattern[str], *, field_name: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise HostOrchestrationError(f"{field_name} does not match the required shape")
    return value


# ---------------------------------------------------------------------------
# Protected execution config: the only place host/user/key/known-hosts and
# remote-path facts may be read from, and never as individual CLI arguments.
# ---------------------------------------------------------------------------

_CONFIG_REQUIRED_KEYS = frozenset(
    {
        "host",
        "port",
        "user",
        "private_key_path",
        "known_hosts_path",
        "remote_workspace",
        "authorized_key_marker",
        "local_evidence_dir",
        "local_runner_archive",
    }
)


@dataclass(frozen=True)
class ProtectedExecutionConfig:
    """Private execution inputs, loaded only from one protected file path.

    Every sensitive field is ``repr=False`` so it can never appear in a
    default repr, an exception's default ``str()``, or an accidental log
    line; :meth:`public_record` is the only representation meant to feed
    anything persisted, and it omits every field below entirely.
    """

    host: str = field(repr=False)
    port: int = field(repr=False)
    user: str = field(repr=False)
    private_key_path: Path = field(repr=False)
    known_hosts_path: Path = field(repr=False)
    remote_workspace: str = field(repr=False)
    authorized_key_marker: str = field(repr=False)
    local_evidence_dir: Path = field(repr=False)
    local_runner_archive: Path = field(repr=False)

    @classmethod
    def load(cls, path: Path) -> ProtectedExecutionConfig:
        """Load and fully validate the protected execution config file.

        Requires: the config file itself is a bounded, non-symlink, regular
        file that is not group/world-readable; every referenced path is
        independently validated (private key mode exactly ``0600``, known
        hosts not group/world-readable, both non-symlink regular files); the
        JSON object has *exactly* the required key set (no more, no less);
        and every value is independently shape-checked.
        """

        _require_not_group_or_world_readable(path, label="protected execution config")
        try:
            text = read_bounded_regular_text(path, MAX_CONFIG_ARTIFACT_BYTES)
            payload = json.loads(text, parse_constant=reject_non_finite_json_constant)
        except (OSError, ArtifactReadError, ValueError, RecursionError) as exc:
            raise HostOrchestrationError(
                f"protected execution config could not be read safely: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise HostOrchestrationError("protected execution config must be an object")
        if set(payload) != _CONFIG_REQUIRED_KEYS:
            missing = sorted(_CONFIG_REQUIRED_KEYS - set(payload))
            extra = sorted(set(payload) - _CONFIG_REQUIRED_KEYS)
            raise HostOrchestrationError(
                "protected execution config keys differ from the required set "
                f"(missing={missing}, extra={extra})"
            )
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ProtectedExecutionConfig:
        if set(payload) != _CONFIG_REQUIRED_KEYS:
            raise HostOrchestrationError(
                "protected execution config keys differ from the required set"
            )
        host = payload["host"]
        user = payload["user"]
        port = payload["port"]
        if not isinstance(host, str) or _SAFE_HOSTNAME_OR_IP.fullmatch(host) is None:
            raise HostOrchestrationError("host must be a safe hostname or IP literal")
        if (
            not isinstance(port, int)
            or isinstance(port, bool)
            or not 1 <= port <= 65535
        ):
            raise HostOrchestrationError("port must be an integer from 1 through 65535")
        if not isinstance(user, str) or _SAFE_USER.fullmatch(user) is None:
            raise HostOrchestrationError("user must be a safe POSIX user name")
        private_key_path = Path(
            _require_nonempty_str(payload["private_key_path"], "private_key_path")
        )
        known_hosts_path = Path(
            _require_nonempty_str(payload["known_hosts_path"], "known_hosts_path")
        )
        _require_private_key_permissions(private_key_path, label="private_key_path")
        _require_not_group_or_world_readable(known_hosts_path, label="known_hosts_path")
        remote_workspace = _require_safe_remote_path(
            payload["remote_workspace"], field_name="remote_workspace"
        )
        authorized_key_marker = _require_label(
            payload["authorized_key_marker"], field_name="authorized_key_marker"
        )
        local_evidence_dir = Path(
            _require_nonempty_str(payload["local_evidence_dir"], "local_evidence_dir")
        )
        local_runner_archive = Path(
            _require_nonempty_str(
                payload["local_runner_archive"], "local_runner_archive"
            )
        )
        if local_runner_archive.exists():
            _require_safe_regular_file(
                local_runner_archive, label="local_runner_archive"
            )
        if local_evidence_dir.exists() and local_evidence_dir.is_symlink():
            raise HostOrchestrationError("local_evidence_dir must not be a symlink")
        return cls(
            host=host,
            port=port,
            user=user,
            private_key_path=private_key_path,
            known_hosts_path=known_hosts_path,
            remote_workspace=remote_workspace,
            authorized_key_marker=authorized_key_marker,
            local_evidence_dir=local_evidence_dir,
            local_runner_archive=local_runner_archive,
        )

    def public_record(self) -> dict[str, Any]:
        """Publication/evidence-safe view: no host/user/key/known-hosts/paths."""

        return {"schema_version": "1", "source": "protected_execution_config"}


def _require_nonempty_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HostOrchestrationError(f"{field_name} must be a non-empty string")
    return value


# ---------------------------------------------------------------------------
# Fixed, protocol-invariant stage-duration ledger (durations only -- no
# absolute clock, and in particular none of the plan's already-terminated
# VM's specific timestamps, is ever hardcoded as a default in this module).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetStage:
    """One row of the fixed stage-duration ledger, in execution order."""

    name: str
    allowance_minutes: int


#: The plan's "One-attempt staging and runtime schedule" table, expressed as
#: durations. The authorization supplies its operational cutoff explicitly;
#: boot time, rate, and cap independently impose an absolute ceiling.
BUDGET_STAGES: tuple[BudgetStage, ...] = (
    BudgetStage("existing_preflight_planning_reserve", 15),
    BudgetStage("approval_handoff_ssh_identity_gate", 5),
    BudgetStage("model_acquisition_verification", 40),
    BudgetStage("image_pull_rebuild_inspect_attestation", 20),
    BudgetStage("no_warmup_canary_event_feasibility_gate", 15),
    BudgetStage("four_fresh_ab_lifecycle_pairs", 60),
    BudgetStage("fixed_eviction_lane", 10),
    BudgetStage("private_verify_redact_report_export", 10),
    BudgetStage("transfer_verify_cleanup_shutdown_handoff", 35),
)

#: Cumulative minutes from boot through (and including) the stage whose name
#: this maps to.
_CUMULATIVE_MINUTES: dict[str, int] = {}
_running_total = 0
for _entry in BUDGET_STAGES:
    _running_total += _entry.allowance_minutes
    _CUMULATIVE_MINUTES[_entry.name] = _running_total
STOP_STARTING_NEW_WORK_MINUTES = _CUMULATIVE_MINUTES[
    "private_verify_redact_report_export"
]
TEARDOWN_COMPLETE_MINUTES = _CUMULATIVE_MINUTES[
    "transfer_verify_cleanup_shutdown_handoff"
]
CLEANUP_RESERVE_MINUTES = TEARDOWN_COMPLETE_MINUTES - STOP_STARTING_NEW_WORK_MINUTES
assert STOP_STARTING_NEW_WORK_MINUTES == 175
assert TEARDOWN_COMPLETE_MINUTES == 210
assert CLEANUP_RESERVE_MINUTES == 35
del _running_total, _entry


def stage_index(name: str) -> int:
    for index, entry in enumerate(BUDGET_STAGES):
        if entry.name == name:
            return index
    raise HostOrchestrationError(f"unknown budget stage: {name!r}")


def remaining_reserve_minutes(from_stage_index: int) -> int:
    """Sum of every stage's allowance from ``from_stage_index`` onward."""

    if not 0 <= from_stage_index < len(BUDGET_STAGES):
        raise HostOrchestrationError(f"stage index {from_stage_index} is out of range")
    return sum(entry.allowance_minutes for entry in BUDGET_STAGES[from_stage_index:])


@dataclass(frozen=True)
class RunCutoffs:
    """Explicit operational cutoff bounded by the boot-derived cost cap."""

    billing_started_at: datetime
    operational_cutoff: datetime
    teardown_complete_cutoff: datetime
    absolute_cap_cutoff: datetime

    def stage_budget_ok(self, now: datetime, from_stage_index: int) -> bool:
        """The plan's gate: ``now + every remaining mandatory reserve <= cutoff``."""

        projected = now + timedelta(minutes=remaining_reserve_minutes(from_stage_index))
        return projected <= min(self.teardown_complete_cutoff, self.absolute_cap_cutoff)

    def may_start_new_experiment_work(self, now: datetime) -> bool:
        return now <= self.operational_cutoff


def absolute_cap_cutoff(
    billing_started_at: datetime,
    rate_usd_per_hour: Decimal,
    total_cap_usd: Decimal,
) -> datetime:
    """Return the latest instant below the list-rate cap, rounded down to µs."""

    if rate_usd_per_hour <= 0 or total_cap_usd <= 0:
        raise HostOrchestrationError("rate and total cap must both be positive")
    elapsed_microseconds = (
        total_cap_usd / rate_usd_per_hour * Decimal(3600) * Decimal(1_000_000)
    ).to_integral_value(rounding=ROUND_FLOOR)
    return billing_started_at + timedelta(microseconds=int(elapsed_microseconds))


def compute_cutoffs(
    billing_started_at: datetime,
    operational_cutoff: datetime,
    cleanup_reserve_minutes: int,
    rate_usd_per_hour: Decimal,
    total_cap_usd: Decimal,
) -> RunCutoffs:
    """Combine explicit run authority with the boot-derived absolute cap."""

    teardown_complete_cutoff = operational_cutoff + timedelta(
        minutes=cleanup_reserve_minutes
    )
    return RunCutoffs(
        billing_started_at=billing_started_at,
        operational_cutoff=operational_cutoff,
        teardown_complete_cutoff=teardown_complete_cutoff,
        absolute_cap_cutoff=absolute_cap_cutoff(
            billing_started_at, rate_usd_per_hour, total_cap_usd
        ),
    )


def list_rate_cost_usd(elapsed_minutes: Decimal, rate_usd_per_hour: Decimal) -> Decimal:
    if elapsed_minutes < 0:
        raise HostOrchestrationError("elapsed_minutes must not be negative")
    return (elapsed_minutes / Decimal(60) * rate_usd_per_hour).quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_EVEN
    )


# ---------------------------------------------------------------------------
# Explicit, self-sealed run authorization.
# ---------------------------------------------------------------------------

AUTHORIZATION_SCHEMA_VERSION = "1"
AUTHORIZATION_SIGNER_IDENTITY = "vllm-kv-truth-coordinator"
AUTHORIZATION_SIGNATURE_NAMESPACE = "llmtracefx-vllm-kv-truth-authorization-v1"

_AUTHORIZATION_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "repository_head",
        "base_image_reference",
        "derived_image_source_digest",
        "vllm_version",
        "vllm_commit",
        "model_id",
        "model_revision",
        "model_inventory_sha256",
        "gpu_expected_count",
        "gpu_expected_name",
        "gpu_expected_driver",
        "gpu_expected_memory_mib",
        "rate_usd_per_hour",
        "total_cap_usd",
        "billing_started_at",
        "operational_cutoff",
        "cleanup_reserve_minutes",
        "authorized_at",
        "authorization_expiry",
        "automatic_retries",
        "replacement_allowed",
        "nonce",
        "authorization_sha256",
    }
)
_AUTHORIZATION_OPTIONAL_SIGNATURE_KEYS = frozenset(
    {"signature_path", "authorized_signers_path"}
)


def _model_inventory_sha256() -> str:
    """The real, current SHA-256 of the committed model conversion manifest.

    Read fresh every time (never cached across a process's lifetime) so a
    change to the committed manifest is always reflected; bounded and
    symlink-safe via :func:`read_bounded_regular_bytes`.
    """

    raw = read_bounded_regular_bytes(
        MODEL_CONVERSION_MANIFEST_PATH, MAX_MANIFEST_ARTIFACT_BYTES
    )
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class RunAuthorization:
    """Explicit, signed authority for one future one-attempt remote run.

    Every field the plan requires this authorization to bind is present and
    independently validated in :meth:`from_dict`; there is no path that
    fills in a default for ``repository_head``, ``billing_started_at``, or
    any cutoff from this module's own constants -- those are exactly the
    "no old terminated-VM times may be defaults" facts this dataclass exists
    to keep caller-supplied and explicit.
    """

    repository_head: str
    derived_image_source_digest: str
    model_inventory_sha256: str
    gpu_expected_count: int
    rate_usd_per_hour: Decimal
    total_cap_usd: Decimal
    billing_started_at: datetime
    operational_cutoff: datetime
    cleanup_reserve_minutes: int
    authorized_at: datetime
    authorization_expiry: datetime
    nonce: str
    authorization_sha256: str
    signature_path: Path | None = field(default=None, repr=False)
    authorized_signers_path: Path | None = field(default=None, repr=False)

    @property
    def cutoffs(self) -> RunCutoffs:
        return compute_cutoffs(
            self.billing_started_at,
            self.operational_cutoff,
            self.cleanup_reserve_minutes,
            self.rate_usd_per_hour,
            self.total_cap_usd,
        )

    def is_within_validity_window(self, now: datetime) -> bool:
        return self.authorized_at <= now <= self.authorization_expiry

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": AUTHORIZATION_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "repository_head": self.repository_head,
            "base_image_reference": BASE_IMAGE_REFERENCE,
            "derived_image_source_digest": self.derived_image_source_digest,
            "vllm_version": REQUIRED_VLLM_VERSION,
            "vllm_commit": REQUIRED_VLLM_COMMIT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_inventory_sha256": self.model_inventory_sha256,
            "gpu_expected_count": self.gpu_expected_count,
            "gpu_expected_name": EXPECTED_GPU_NAME,
            "gpu_expected_driver": EXPECTED_DRIVER,
            "gpu_expected_memory_mib": EXPECTED_MEMORY_MIB,
            "rate_usd_per_hour": _money_str(self.rate_usd_per_hour),
            "total_cap_usd": _money_str(self.total_cap_usd),
            "billing_started_at": _canonical_timestamp(self.billing_started_at),
            "operational_cutoff": _canonical_timestamp(self.operational_cutoff),
            "cleanup_reserve_minutes": self.cleanup_reserve_minutes,
            "authorized_at": _canonical_timestamp(self.authorized_at),
            "authorization_expiry": _canonical_timestamp(self.authorization_expiry),
            "automatic_retries": 0,
            "replacement_allowed": False,
            "nonce": self.nonce,
            "authorization_sha256": self.authorization_sha256,
        }

    @classmethod
    def from_dict(
        cls, data: Any, *, verify_model_inventory: bool = True
    ) -> RunAuthorization:
        if not isinstance(data, dict):
            raise HostOrchestrationError("authorization must be a JSON object")
        observed_keys = set(data)
        if observed_keys != _AUTHORIZATION_REQUIRED_KEYS:
            signed_extra = observed_keys - _AUTHORIZATION_REQUIRED_KEYS
            if signed_extra and (
                signed_extra != _AUTHORIZATION_OPTIONAL_SIGNATURE_KEYS
                or observed_keys - signed_extra != _AUTHORIZATION_REQUIRED_KEYS
            ):
                raise HostOrchestrationError(
                    "authorization keys differ from the required set "
                    "(optional signature fields must both be present together)"
                )
            if not signed_extra:
                raise HostOrchestrationError(
                    "authorization keys differ from the required set"
                )

        fixed = {
            "schema_version": AUTHORIZATION_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "base_image_reference": BASE_IMAGE_REFERENCE,
            "vllm_version": REQUIRED_VLLM_VERSION,
            "vllm_commit": REQUIRED_VLLM_COMMIT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "gpu_expected_name": EXPECTED_GPU_NAME,
            "gpu_expected_driver": EXPECTED_DRIVER,
            "gpu_expected_memory_mib": EXPECTED_MEMORY_MIB,
            "automatic_retries": 0,
            "replacement_allowed": False,
        }
        for key, expected in fixed.items():
            if data.get(key) != expected:
                raise HostOrchestrationError(
                    f"authorization {key!r} does not match the approved protocol "
                    "envelope"
                )

        repository_head = _require_pattern(
            data["repository_head"], _COMMIT_HEX, field_name="repository_head"
        )
        derived_image_source_digest = _require_pattern(
            data["derived_image_source_digest"],
            _SHA256_REF,
            field_name="derived_image_source_digest",
        )
        model_inventory_sha256 = _require_pattern(
            data["model_inventory_sha256"],
            _SHA256_HEX,
            field_name="model_inventory_sha256",
        )
        if verify_model_inventory:
            actual = _model_inventory_sha256()
            if model_inventory_sha256 != actual:
                raise HostOrchestrationError(
                    "authorization model_inventory_sha256 does not match the "
                    "committed conversion manifest currently on disk"
                )
        gpu_expected_count = data["gpu_expected_count"]
        if not isinstance(gpu_expected_count, int) or isinstance(
            gpu_expected_count, bool
        ):
            raise HostOrchestrationError("gpu_expected_count must be an integer")
        if gpu_expected_count != 1:
            raise HostOrchestrationError(
                "gpu_expected_count must be exactly 1 for this fixed protocol "
                "(tensor_parallel_size=1, data_parallel_size=1)"
            )
        rate_usd_per_hour = _canonical_decimal(
            data["rate_usd_per_hour"], field_name="rate_usd_per_hour"
        )
        total_cap_usd = _canonical_decimal(
            data["total_cap_usd"], field_name="total_cap_usd"
        )
        minimum_cap = list_rate_cost_usd(
            Decimal(TEARDOWN_COMPLETE_MINUTES), rate_usd_per_hour
        )
        if total_cap_usd < minimum_cap:
            raise HostOrchestrationError(
                "total_cap_usd is below the cost of completing every mandatory "
                "stage at the authorized rate"
            )
        billing_started_at = _parse_utc(
            data["billing_started_at"], field_name="billing_started_at"
        )
        operational_cutoff = _parse_utc(
            data["operational_cutoff"], field_name="operational_cutoff"
        )
        cleanup_reserve_minutes = data["cleanup_reserve_minutes"]
        if (
            not isinstance(cleanup_reserve_minutes, int)
            or isinstance(cleanup_reserve_minutes, bool)
            or cleanup_reserve_minutes < CLEANUP_RESERVE_MINUTES
        ):
            raise HostOrchestrationError(
                "cleanup_reserve_minutes must be an integer at least equal to "
                f"the fixed {CLEANUP_RESERVE_MINUTES}-minute cleanup allowance"
            )
        authorized_at = _parse_utc(data["authorized_at"], field_name="authorized_at")
        authorization_expiry = _parse_utc(
            data["authorization_expiry"], field_name="authorization_expiry"
        )
        if authorized_at >= operational_cutoff:
            raise HostOrchestrationError(
                "authorized_at must be strictly before operational_cutoff"
            )
        if authorization_expiry <= authorized_at:
            raise HostOrchestrationError(
                "authorization_expiry must be strictly after authorized_at"
            )
        cutoffs = compute_cutoffs(
            billing_started_at,
            operational_cutoff,
            cleanup_reserve_minutes,
            rate_usd_per_hour,
            total_cap_usd,
        )
        if cutoffs.teardown_complete_cutoff > cutoffs.absolute_cap_cutoff:
            raise HostOrchestrationError(
                "operational_cutoff plus cleanup reserve exceeds the boot-derived "
                "absolute list-rate cap"
            )
        if authorization_expiry < cutoffs.teardown_complete_cutoff:
            raise HostOrchestrationError(
                "authorization_expiry must cover the complete cleanup window"
            )
        nonce = _require_pattern(data["nonce"], _NONCE_HEX, field_name="nonce")

        expected_seal = sha256_json(
            {
                k: v
                for k, v in data.items()
                if k != "authorization_sha256"
                and k not in _AUTHORIZATION_OPTIONAL_SIGNATURE_KEYS
            }
        )
        if data.get("authorization_sha256") != expected_seal:
            raise HostOrchestrationError(
                "authorization_sha256 does not verify against its own content"
            )

        signature_path: Path | None = None
        authorized_signers_path: Path | None = None
        if _AUTHORIZATION_OPTIONAL_SIGNATURE_KEYS <= observed_keys:
            signature_path = Path(
                _require_nonempty_str(data["signature_path"], "signature_path")
            )
            authorized_signers_path = Path(
                _require_nonempty_str(
                    data["authorized_signers_path"], "authorized_signers_path"
                )
            )

        return cls(
            repository_head=repository_head,
            derived_image_source_digest=derived_image_source_digest,
            model_inventory_sha256=model_inventory_sha256,
            gpu_expected_count=gpu_expected_count,
            rate_usd_per_hour=rate_usd_per_hour,
            total_cap_usd=total_cap_usd,
            billing_started_at=billing_started_at,
            operational_cutoff=operational_cutoff,
            cleanup_reserve_minutes=cleanup_reserve_minutes,
            authorized_at=authorized_at,
            authorization_expiry=authorization_expiry,
            nonce=nonce,
            authorization_sha256=data["authorization_sha256"],
            signature_path=signature_path,
            authorized_signers_path=authorized_signers_path,
        )

    @classmethod
    def read(cls, path: Path) -> RunAuthorization:
        try:
            text = read_bounded_regular_text(path, MAX_AUTHORIZATION_ARTIFACT_BYTES)
            payload = json.loads(text, parse_constant=reject_non_finite_json_constant)
        except (OSError, ArtifactReadError, ValueError, RecursionError) as exc:
            raise HostOrchestrationError(
                f"authorization could not be read safely: {exc}"
            ) from exc
        return cls.from_dict(payload)

    def redact(self) -> dict[str, Any]:
        """Publication-safe view: every field here is already a digest,
        commitment, timestamp, or non-sensitive identity value."""

        return self.to_dict()


def build_authorization_seal(payload_without_seal: Mapping[str, Any]) -> str:
    """Compute the canonical self-seal a caller must place under
    ``authorization_sha256`` before writing an authorization file."""

    return sha256_json(
        {
            key: value
            for key, value in payload_without_seal.items()
            if key not in _AUTHORIZATION_OPTIONAL_SIGNATURE_KEYS
        }
    )


def verify_authorization_signature(
    runner: CommandRunner,
    authorization: RunAuthorization,
    *,
    signature_path: Path,
    authorized_signers_path: Path,
) -> None:
    """Verify a detached OpenSSH signature over the authorization payload.

    Reuses this repository's existing safe pattern (``ssh-keygen -Y
    verify``, fed the canonical authorization JSON on stdin, against a
    caller-provided ``authorized_signers`` file) rather than inventing a new
    signature mechanism; both paths are independently required to be
    bounded, non-symlink, regular files first.
    """

    _require_safe_regular_file(signature_path, label="authorization signature")
    _require_safe_regular_file(authorized_signers_path, label="authorized signers")
    message = canonical_json(authorization.to_dict())
    result = runner.run(
        (
            "ssh-keygen",
            "-Y",
            "verify",
            "-f",
            str(authorized_signers_path),
            "-I",
            AUTHORIZATION_SIGNER_IDENTITY,
            "-n",
            AUTHORIZATION_SIGNATURE_NAMESPACE,
            "-s",
            str(signature_path),
        ),
        description="verify_authorization_signature",
        timeout=10,
        input_text=message,
    )
    if not result.ok:
        raise HostOrchestrationError(
            "authorization signature did not verify against authorized_signers"
        )


# ---------------------------------------------------------------------------
# Injectable command execution: this module never calls ``subprocess`` or
# opens a socket directly outside of ``SubprocessCommandRunner`` below, so
# every stage can be driven end to end by a hermetic fake in tests.
# ---------------------------------------------------------------------------

_SAFE_EXECUTION_PATH = "/usr/bin:/bin:/usr/local/bin"
_CREDENTIAL_NAME_FRAGMENTS = (
    "TOKEN",
    "PASSWORD",
    "SECRET",
    "API_KEY",
    "APIKEY",
    "PRIVATE_KEY",
    "CREDENTIAL",
    "COOKIE",
    "AUTH",
)
_FORBIDDEN_ROUTING_VARS = (
    "DOCKER_HOST",
    "SSH_AUTH_SOCK",
    "SSH_ASKPASS",
    "GIT_SSH_COMMAND",
    "GIT_SSH",
)


def reject_credential_environment(env: Mapping[str, str]) -> None:
    """Refuse to run if the ambient environment carries credential-shaped or
    command-routing variables that could silently change execution
    semantics or exfiltrate secrets (mirrors the safe pattern already used
    by this repository's local Docker orchestrator)."""

    offending: list[str] = []
    for name, value in env.items():
        if not value:
            continue
        upper = name.upper()
        if any(fragment in upper for fragment in _CREDENTIAL_NAME_FRAGMENTS):
            offending.append(name)
        elif upper in _FORBIDDEN_ROUTING_VARS:
            offending.append(name)
    if offending:
        raise HostOrchestrationError(
            "refusing to run with credential-shaped or command-routing "
            f"environment variables set: {sorted(set(offending))}"
        )


@dataclass(frozen=True)
class CommandResult:
    """The only view of a completed command this module ever inspects."""

    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class CommandRunner(Protocol):
    """Executes one argv and returns its result; the sole I/O seam."""

    def run(
        self,
        argv: Sequence[str],
        *,
        description: str,
        timeout: float,
        input_text: str | None = None,
    ) -> CommandResult: ...


class SubprocessCommandRunner:
    """The only concrete :class:`CommandRunner` that actually spawns a
    process. Never uses ``shell=True``; always runs with a fixed, minimal,
    credential-free environment; always enforces the caller's timeout."""

    def __init__(self) -> None:
        reject_credential_environment(os.environ)

    def run(
        self,
        argv: Sequence[str],
        *,
        description: str,
        timeout: float,
        input_text: str | None = None,
    ) -> CommandResult:
        try:
            completed = subprocess.run(  # noqa: S603 - argv is built by this
                # module from validated fields; never shell-interpreted.
                list(argv),
                input=input_text,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={"PATH": _SAFE_EXECUTION_PATH, "LANG": "C", "LC_ALL": "C"},
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HostOrchestrationError(f"{description} timed out") from exc
        except OSError as exc:
            raise HostOrchestrationError(
                f"{description} could not start: {exc}"
            ) from exc
        return CommandResult(
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def checked(
    runner: CommandRunner,
    argv: Sequence[str],
    *,
    description: str,
    timeout: float,
    input_text: str | None = None,
) -> CommandResult:
    """Run ``argv`` and raise a non-leaking error on failure.

    The raised message names only the failed stage's ``description`` --
    never ``argv`` content or ``stderr`` -- so a host path, key fingerprint,
    or any other operator-identifying detail can never reach a log line or
    exception message by accident.
    """

    result = runner.run(
        argv, description=description, timeout=timeout, input_text=input_text
    )
    if not result.ok:
        raise HostOrchestrationError(f"stage failed: {description}")
    return result


# ---------------------------------------------------------------------------
# Strict, key-only SSH/SCP option set shared by every remote command this
# module builds. No forwarding, no control sockets, no agent, no password or
# keyboard-interactive fallback, ever.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StrictSSHOptions:
    config: ProtectedExecutionConfig

    def shared_options(self) -> tuple[str, ...]:
        return (
            "-o",
            "BatchMode=yes",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "PasswordAuthentication=no",
            "-o",
            "KbdInteractiveAuthentication=no",
            "-o",
            "StrictHostKeyChecking=yes",
            "-o",
            f"UserKnownHostsFile={self.config.known_hosts_path}",
            "-o",
            "ForwardAgent=no",
            "-o",
            "ForwardX11=no",
            "-o",
            "ClearAllForwardings=yes",
            "-o",
            "ControlMaster=no",
            "-o",
            "ControlPath=none",
            "-i",
            str(self.config.private_key_path),
        )

    def ssh_command(self, remote_command: str) -> tuple[str, ...]:
        return (
            "ssh",
            "-p",
            str(self.config.port),
            *self.shared_options(),
            f"{self.config.user}@{self.config.host}",
            remote_command,
        )

    def scp_upload_command(self, local_path: Path, remote_path: str) -> tuple[str, ...]:
        return (
            "scp",
            "-P",
            str(self.config.port),
            *self.shared_options(),
            str(local_path),
            f"{self.config.user}@{self.config.host}:{remote_path}",
        )

    def scp_download_command(
        self, remote_path: str, local_path: Path
    ) -> tuple[str, ...]:
        return (
            "scp",
            "-P",
            str(self.config.port),
            *self.shared_options(),
            f"{self.config.user}@{self.config.host}:{remote_path}",
            str(local_path),
        )

    def public_record(self) -> dict[str, Any]:
        """Publication-safe: only the fixed option *names*, never values
        that could identify the host/user/key/known-hosts path."""

        return {
            "batch_mode": True,
            "identities_only": True,
            "password_authentication": False,
            "keyboard_interactive_authentication": False,
            "strict_host_key_checking": True,
            "forward_agent": False,
            "forward_x11": False,
            "control_master": False,
            "control_path": "none",
        }


# ---------------------------------------------------------------------------
# Fixed AB-pair ordering and Docker labeling.
# ---------------------------------------------------------------------------

#: The plan's fixed four fresh pairs, in order. Each entry is the lane order
#: within that pair's two fresh container invocations.
AB_PAIR_ORDER: tuple[tuple[str, str], ...] = (
    ("A", "B"),
    ("B", "A"),
    ("B", "A"),
    ("A", "B"),
)

RUN_LABEL_PREFIX = "llmtracefx.kv_truth.run"

#: Cache-salt support remains false until the exact pinned runtime surface is
#: positively exercised. Per the plan ("salt only if exact surface supports
#: it"), the conservative result is ``unsupported``, never a fabricated pass.
CACHE_SALT_SUPPORT_CONFIRMED = False


def ab_pair_lane_tags() -> tuple[str, ...]:
    """The eight fixed ``pair{N}-slot{M}-{lane}`` output tags, in the exact
    order :meth:`RemoteOrchestrator.stage_four_ab_pairs` invokes them."""

    tags = []
    for pair_index, order in enumerate(AB_PAIR_ORDER, start=1):
        for slot, lane in enumerate(order, start=1):
            tags.append(f"pair{pair_index}-slot{slot}-{lane.lower()}")
    return tuple(tags)


def b_lane_output_tags() -> tuple[str, ...]:
    """The four fixed cache-enabled ("B") lane output tags among the eight
    AB-pair invocations -- the only ones eligible for cache-reuse claims."""

    return tuple(tag for tag in ab_pair_lane_tags() if tag.endswith("-b"))


def docker_run_label(nonce: str) -> str:
    return f"{RUN_LABEL_PREFIX}={nonce}"


def _quote(value: str) -> str:
    """Defensive remote-shell quoting; used even for already-validated
    values so a future refactor cannot silently reintroduce injection."""

    return shlex.quote(value)


@dataclass(frozen=True)
class RunPaths:
    """The one fixed remote directory layout this protocol ever uses."""

    remote_workspace: str

    @property
    def repo_dir(self) -> str:
        return f"{self.remote_workspace}/repo"

    @property
    def model_dir(self) -> str:
        return f"{self.remote_workspace}/model"

    @property
    def evidence_dir(self) -> str:
        return f"{self.remote_workspace}/evidence"

    @property
    def receipts_dir(self) -> str:
        return f"{self.remote_workspace}/receipts"

    @property
    def source_archive_remote_path(self) -> str:
        return f"{self.remote_workspace}/source.tar"


_COMMIT_HEAD_MEMBER = "COMMIT_HEAD"
_RUNNER_SOURCE_MEMBER = "vllm_kv_truth/runner.py"


def _validate_safe_tar_member(member: tarfile.TarInfo) -> None:
    """Reject any tar member that is unsafe to trust: absolute paths, ``..``
    traversal, or anything other than a plain regular file or directory
    (no symlinks, hardlinks, devices, or fifos)."""

    name = member.name
    if name.startswith("/") or name.startswith("..") or ".." in name.split("/"):
        raise HostOrchestrationError(
            f"archive contains an unsafe member path: {name!r}"
        )
    if not (member.isfile() or member.isdir()):
        raise HostOrchestrationError(
            f"archive contains a non-regular member (symlink/device/etc.): {name!r}"
        )


def read_runner_archive_commit_marker(archive_path: Path) -> tuple[bytes, str]:
    """Read a checked source archive's bytes and its embedded exact commit.

    The archive is expected to be a deterministic tarball of this
    repository's tree at the *future* merged commit, containing exactly one
    top-level plain-text member named ``COMMIT_HEAD`` whose stripped
    contents are the 40-hex commit the archive was produced from. This lets
    the orchestrator verify "archive hash/HEAD" purely locally, with no
    network access and no live ``git fetch`` against any remote -- the
    archive is staged once, out of band, by whoever prepares the run.

    Every tar member is validated as a safe, bounded, regular file (no
    absolute paths, no ``..`` traversal, no symlinks/hardlinks/devices)
    before any bytes are trusted.
    """

    archive_bytes = read_bounded_regular_bytes(archive_path, MAX_SOURCE_ARCHIVE_BYTES)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as tar:
            marker_member = None
            for member in tar.getmembers():
                _validate_safe_tar_member(member)
                name = member.name
                if name in (_COMMIT_HEAD_MEMBER, f"./{_COMMIT_HEAD_MEMBER}"):
                    marker_member = member
            if marker_member is None:
                raise HostOrchestrationError(
                    "checked source archive does not contain a COMMIT_HEAD marker"
                )
            extracted = tar.extractfile(marker_member)
            if extracted is None:
                raise HostOrchestrationError(
                    "checked source archive's COMMIT_HEAD marker is not a "
                    "readable regular file"
                )
            commit = extracted.read().decode("ascii", errors="strict").strip()
    except tarfile.TarError as exc:
        raise HostOrchestrationError(
            f"checked source archive is not a valid tar archive: {exc}"
        ) from exc
    commit = _require_pattern(commit, _COMMIT_HEX, field_name="archive COMMIT_HEAD")
    return archive_bytes, commit


def runner_source_digest_from_archive(archive_bytes: bytes) -> str:
    """Hash the exact runner source member carried by the authorized archive."""

    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as tar:
            matches = [
                member
                for member in tar.getmembers()
                if member.name in {_RUNNER_SOURCE_MEMBER, f"./{_RUNNER_SOURCE_MEMBER}"}
            ]
            if len(matches) != 1 or not matches[0].isfile():
                raise HostOrchestrationError(
                    "checked source archive must contain exactly one regular "
                    "kv_truth_runner.py member"
                )
            extracted = tar.extractfile(matches[0])
            if extracted is None:
                raise HostOrchestrationError(
                    "checked source archive runner member is unreadable"
                )
            return "sha256:" + hashlib.sha256(extracted.read()).hexdigest()
    except tarfile.TarError as exc:
        raise HostOrchestrationError(
            f"checked source archive is not a valid tar archive: {exc}"
        ) from exc


def _expected_runtime_model_digests() -> tuple[str, str]:
    manifest = json.loads(
        read_bounded_regular_text(
            MODEL_CONVERSION_MANIFEST_PATH, MAX_MANIFEST_ARTIFACT_BYTES
        )
    )
    files = sorted(manifest["source"]["files"], key=lambda item: item["path"])
    tokenizer_files = [
        item for item in files if "tokenizer" in str(item["path"]).lower()
    ]
    return (
        "sha256:" + sha256_json(files),
        "sha256:" + sha256_json(tokenizer_files),
    )


def extract_safe_tar(archive_bytes: bytes, destination: Path) -> None:
    """Safely extract a bounded tar archive of evidence into ``destination``.

    Every member is validated the same way as
    :func:`read_runner_archive_commit_marker` (no absolute paths, no ``..``
    traversal, no symlinks/hardlinks/devices) *before* any bytes are
    written, and files are written individually (never via
    ``TarFile.extractall``) so this behaves identically across Python
    versions regardless of their default extraction-filter policy.
    """

    if len(archive_bytes) > MAX_SOURCE_ARCHIVE_BYTES:
        raise HostOrchestrationError(
            "evidence archive exceeds the maximum trusted archive size"
        )
    destination.mkdir(parents=True, exist_ok=True)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive_bytes)) as tar:
            members = tar.getmembers()
            for member in members:
                _validate_safe_tar_member(member)
            for member in members:
                target = destination / member.name
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(extracted.read())
    except tarfile.TarError as exc:
        raise HostOrchestrationError(
            f"evidence archive is not a valid tar archive: {exc}"
        ) from exc


def build_model_inventory_verification_script(paths: RunPaths) -> str:
    """A remote POSIX-shell script verifying the staged model tree against
    the committed 15-file/16,397,461,266-byte SHA-256 inventory.

    Reads the manifest from *this* (local, already-committed) file so the
    remote host is verified against the exact same inventory the offline
    live adapter/runner already trust -- never a separate, driftable copy.
    """

    manifest_text = read_bounded_regular_text(
        MODEL_CONVERSION_MANIFEST_PATH, MAX_MANIFEST_ARTIFACT_BYTES
    )
    manifest = json.loads(manifest_text)
    files = manifest["source"]["files"]
    if len(files) != EXPECTED_MODEL_FILE_COUNT:
        raise HostOrchestrationError(
            "committed model manifest file count no longer matches the "
            "pinned protocol expectation"
        )
    total_bytes = sum(int(entry["size_bytes"]) for entry in files)
    if total_bytes != EXPECTED_MODEL_BYTES:
        raise HostOrchestrationError(
            "committed model manifest total bytes no longer match the "
            "pinned protocol expectation"
        )
    lines = ["set -eu", f"cd {_quote(paths.model_dir)}"]
    expected_paths = "\n".join(sorted(str(entry["path"]) for entry in files))
    lines.extend(
        [
            "ACTUAL_FILES=$(find . -type f -printf '%P\\n' | LC_ALL=C sort)",
            "EXPECTED_FILES=$(cat <<'LLMTRACEFX_EXPECTED_MODEL_FILES'",
            expected_paths,
            "LLMTRACEFX_EXPECTED_MODEL_FILES",
            ")",
            'test "$ACTUAL_FILES" = "$EXPECTED_FILES"',
        ]
    )
    for entry in files:
        rel_path = str(entry["path"])
        if rel_path.startswith("/") or ".." in rel_path.split("/"):
            raise HostOrchestrationError(
                f"committed model manifest path is unsafe: {rel_path!r}"
            )
        expected_hash = _require_pattern(
            entry["sha256"], _SHA256_HEX, field_name="manifest file sha256"
        )
        expected_size = int(entry["size_bytes"])
        quoted_path = _quote(rel_path)
        lines.append(f'test "$(wc -c < {quoted_path})" = "{expected_size}"')
        lines.append(f'echo "{expected_hash}  {rel_path}" | sha256sum -c -')
    lines.append(f"echo TOTAL_BYTES={total_bytes}")
    lines.append(f"echo TOTAL_FILES={len(files)}")
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class LaneInvocation:
    """One fully-formed lane container invocation, ready to execute."""

    lane: str
    output_relative_path: str
    container_name: str


def build_docker_run_argv(
    *,
    authorization: RunAuthorization,
    paths: RunPaths,
    invocation: LaneInvocation,
    derived_image_id: str,
) -> tuple[str, ...]:
    """The one real ``docker run`` command every lane invocation uses.

    Always ``--network none``, always the fixed offline environment, always
    labeled with this run's nonce so teardown can find (and only find)
    this run's containers.
    """

    env_pairs: list[str] = []
    for key, value in sorted(REQUIRED_ENVIRONMENT_VARIABLES.items()):
        env_pairs.extend(("-e", f"{key}={value}"))
    expected_identity_env = {
        "EXPECTED_REPOSITORY_COMMIT": authorization.repository_head,
        "EXPECTED_VLLM_SOURCE_COMMIT": VLLM_SOURCE_COMMIT,
        "EXPECTED_IMAGE_ID": _require_pattern(
            derived_image_id, _SHA256_REF, field_name="derived_image_id"
        ),
        "EXPECTED_EXPERIMENT_NONCE": authorization.nonce,
        "EXPECTED_MODEL_REVISION": MODEL_REVISION,
        "EXPECTED_GPU_NAME": EXPECTED_GPU_NAME,
        "EXPECTED_GPU_DRIVER": EXPECTED_DRIVER,
        "EXPECTED_GPU_MEMORY_MIB": str(EXPECTED_MEMORY_MIB),
    }
    for key, value in sorted(expected_identity_env.items()):
        env_pairs.extend(("-e", f"{key}={value}"))

    return (
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--gpus",
        f"count={authorization.gpu_expected_count}",
        "--label",
        docker_run_label(authorization.nonce),
        "--name",
        invocation.container_name,
        *env_pairs,
        "-v",
        f"{paths.model_dir}:/model:ro",
        "-v",
        f"{paths.evidence_dir}:/evidence",
        derived_image_id,
        "python3",
        "-m",
        "vllm_kv_truth.runner",
        "--lane",
        invocation.lane,
        "--model-path",
        "/model",
        "--attestation",
        "/evidence/identity-receipt.json",
        "--output",
        f"/evidence/{invocation.output_relative_path}",
    )


def container_name(nonce: str, tag: str) -> str:
    return _require_label(f"kv-truth-{nonce}-{tag}"[:128], field_name="container_name")


class OrchestratorState(str, Enum):
    """The one-attempt run's coarse-grained state machine states."""

    PENDING = "pending"
    PREFLIGHT = "preflight"
    IDENTITY_GATE = "identity_gate"
    MODEL_ACQUISITION = "model_acquisition"
    IMAGE_PREPARATION = "image_preparation"
    CANARY = "canary"
    AB_PAIRS = "ab_pairs"
    EVICTION_LANE = "eviction_lane"
    EVIDENCE_EXPORT = "evidence_export"
    TEARDOWN = "teardown"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class StageOutcome:
    stage: str
    ok: bool
    detail: str = ""


NowFn = Callable[[], datetime]


class RemoteOrchestrator:
    """Drives the fixed one-attempt remote protocol for one authorized run.

    Every method below builds a real, complete argv (ssh/scp/docker) and
    executes it only through the injected :class:`CommandRunner` -- nothing
    here opens a socket or spawns ``ssh``/``docker`` directly, which is what
    lets this module's own tests exercise the *entire* stage/state machine
    with a hermetic fake and assert zero SSH/network/GPU activity ever
    actually occurs during ``pytest``.
    """

    def __init__(
        self,
        *,
        config: ProtectedExecutionConfig,
        authorization: RunAuthorization,
        runner: CommandRunner,
        now_fn: NowFn = lambda: datetime.now(timezone.utc),
    ) -> None:
        self.config = config
        self.authorization = authorization
        self.runner = runner
        self.now_fn = now_fn
        self.ssh_options = StrictSSHOptions(config)
        self.paths = RunPaths(config.remote_workspace)
        self.state = OrchestratorState.PENDING
        self.outcomes: list[StageOutcome] = []
        self.derived_image_id: str | None = None
        self.expected_runner_source_digest: str | None = None

    def _require_derived_image_id(self) -> str:
        if self.derived_image_id is None:
            raise HostOrchestrationError(
                "derived image has not passed preparation and inspection"
            )
        return self.derived_image_id

    # -- budget gate ------------------------------------------------------

    def _require_budget(self, stage_name: str) -> None:
        now = self.now_fn()
        if not self.authorization.is_within_validity_window(now):
            raise HostOrchestrationError(
                "authorization is not within its validity window"
            )
        index = stage_index(stage_name)
        if not self.authorization.cutoffs.stage_budget_ok(now, index):
            raise HostOrchestrationError(
                f"refusing to start {stage_name!r}: insufficient reserve for "
                "every remaining mandatory stage before the teardown cutoff"
            )

    def _record(self, stage: str, ok: bool, detail: str = "") -> None:
        self.outcomes.append(StageOutcome(stage=stage, ok=ok, detail=detail))

    def _ledger_entry(self, note: str) -> evidence.ListRateLedgerEntry:
        elapsed_seconds = Decimal(
            str(
                max(
                    (
                        self.now_fn() - self.authorization.billing_started_at
                    ).total_seconds(),
                    0,
                )
            )
        )
        elapsed_minutes = elapsed_seconds / Decimal(60)
        return evidence.ListRateLedgerEntry(
            elapsed_minutes=format(elapsed_minutes.quantize(Decimal("0.000001")), "f"),
            cost_usd=_money_str(
                list_rate_cost_usd(
                    elapsed_minutes, self.authorization.rate_usd_per_hour
                )
            ),
            note=note,
        )

    # -- stage 1: read-only preflight --------------------------------------

    def stage_preflight(self) -> StageOutcome:
        self._require_budget("existing_preflight_planning_reserve")
        self.state = OrchestratorState.PREFLIGHT
        script = "\n".join(
            [
                "set -eu",
                'echo "NOW_EPOCH=$(date -u +%s)"',
                'echo "BOOT_EPOCH=$(date -u -d "$(uptime -s)" +%s)"',
                'echo "GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)"',
                "nvidia-smi --query-gpu=name,driver_version,memory.total,"
                "compute_cap --format=csv,noheader,nounits | "
                "sed 's/^/GPU=/'",
                'echo "GPU_PROCESS_COUNT=$(nvidia-smi '
                "--query-compute-apps=pid --format=csv,noheader 2>/dev/null | "
                'sed "/^[[:space:]]*$/d" | wc -l)"',
                'echo "CONTAINER_COUNT=$(docker ps -q | wc -l)"',
                "docker info >/dev/null",
                "sudo -n true",
                'echo "SUDO_NONINTERACTIVE=1"',
                'echo "DISK_FREE_BYTES=$(df -PB1 "$HOME" | awk \'NR==2 {print $4}\')"',
                "echo \"RAM_BYTES=$(free -b | awk '/^Mem:/ {print $2}')\"",
                "echo \"SWAP_USED_BYTES=$(free -b | awk '/^Swap:/ {print $3}')\"",
                'echo "PYTHON_VERSION=$(python3 -c '
                "'import platform; print(platform.python_version())'"
                ')"',
                'echo "DOCKER_VERSION=$(docker version '
                "--format '{{.Server.Version}}')\"",
                'echo "HF_CLI=$(command -v huggingface-cli)"',
            ]
        )
        result = checked(
            self.runner,
            self.ssh_options.ssh_command("bash -s"),
            description="stage_preflight",
            timeout=60,
            input_text=script,
        )
        self._verify_preflight_output(result.stdout)
        self._record("preflight", True)
        return self.outcomes[-1]

    def _verify_preflight_output(self, stdout: str) -> None:
        markers: dict[str, str] = {}
        for line in stdout.splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                markers[key.strip()] = value.strip()
        required = {
            "NOW_EPOCH",
            "BOOT_EPOCH",
            "GPU_COUNT",
            "GPU",
            "GPU_PROCESS_COUNT",
            "CONTAINER_COUNT",
            "SUDO_NONINTERACTIVE",
            "DISK_FREE_BYTES",
            "RAM_BYTES",
            "SWAP_USED_BYTES",
            "PYTHON_VERSION",
            "DOCKER_VERSION",
            "HF_CLI",
        }
        if set(markers) != required:
            raise HostOrchestrationError(
                "preflight output markers differ from the exact required set"
            )
        parts = [part.strip() for part in markers["GPU"].split(",")]
        if len(parts) != 4:
            raise HostOrchestrationError("preflight GPU inventory line is malformed")
        name, driver, memory_str, compute_capability = parts
        if int(markers["GPU_COUNT"]) != self.authorization.gpu_expected_count:
            raise HostOrchestrationError(
                "preflight GPU count does not match authorization"
            )
        if name != EXPECTED_GPU_NAME:
            raise HostOrchestrationError(
                "preflight GPU name does not match protocol pin"
            )
        if driver != EXPECTED_DRIVER:
            raise HostOrchestrationError(
                "preflight GPU driver does not match protocol pin"
            )
        memory_mib = int(memory_str)
        if memory_mib != EXPECTED_MEMORY_MIB:
            raise HostOrchestrationError(
                "preflight GPU memory does not match protocol pin"
            )
        if compute_capability != EXPECTED_GPU_COMPUTE_CAPABILITY:
            raise HostOrchestrationError(
                "preflight GPU compute capability does not match protocol pin"
            )
        if int(markers["GPU_PROCESS_COUNT"]) != 0:
            raise HostOrchestrationError("preflight found an existing GPU process")
        if int(markers["CONTAINER_COUNT"]) != 0:
            raise HostOrchestrationError("preflight found an existing container")
        if int(markers["SWAP_USED_BYTES"]) != 0:
            raise HostOrchestrationError("preflight found swap in use")
        if int(markers["RAM_BYTES"]) < MINIMUM_HOST_RAM_BYTES:
            raise HostOrchestrationError("preflight host RAM is below the minimum")
        if int(markers["DISK_FREE_BYTES"]) < MINIMUM_DISK_FREE_BYTES:
            raise HostOrchestrationError(
                "preflight disk free space is below the minimum"
            )
        if markers["SUDO_NONINTERACTIVE"] != "1":
            raise HostOrchestrationError("preflight sudo is not noninteractive")
        if not markers["PYTHON_VERSION"].startswith("3.12."):
            raise HostOrchestrationError("preflight Python version is not 3.12.x")
        if not markers["DOCKER_VERSION"]:
            raise HostOrchestrationError("preflight Docker version is empty")
        if not markers["HF_CLI"].startswith("/"):
            raise HostOrchestrationError(
                "preflight did not find an absolute huggingface-cli executable"
            )
        now = datetime.fromtimestamp(int(markers["NOW_EPOCH"]), tz=timezone.utc)
        boot = datetime.fromtimestamp(int(markers["BOOT_EPOCH"]), tz=timezone.utc)
        if abs((boot - self.authorization.billing_started_at).total_seconds()) > 2:
            raise HostOrchestrationError(
                "preflight boot time does not match authorization"
            )
        if abs((now - self.now_fn()).total_seconds()) > 120:
            raise HostOrchestrationError(
                "preflight remote clock differs from the coordinator clock"
            )

    # -- stage 2: approval handoff / ssh identity gate ---------------------

    def stage_identity_gate(self) -> StageOutcome:
        self._require_budget("approval_handoff_ssh_identity_gate")
        self.state = OrchestratorState.IDENTITY_GATE
        script = "\n".join(
            [
                "set -eu",
                f"mkdir -p {_quote(self.paths.repo_dir)}",
                f"mkdir -p {_quote(self.paths.model_dir)}",
                f"mkdir -p {_quote(self.paths.evidence_dir)}",
                f"mkdir -p {_quote(self.paths.receipts_dir)}",
                'AUTHORIZED_KEYS="$HOME/.ssh/authorized_keys"',
                f"KEY_MARKER={_quote(self.config.authorized_key_marker)}",
                'test -f "$AUTHORIZED_KEYS"',
                'test "$(awk -v marker="$KEY_MARKER" '
                "'$NF == marker { count++ } END { print count + 0 }' "
                '"$AUTHORIZED_KEYS")" = "1"',
            ]
        )
        checked(
            self.runner,
            self.ssh_options.ssh_command("bash -s"),
            description="stage_identity_gate",
            timeout=30,
            input_text=script,
        )
        self._record("identity_gate", True)
        return self.outcomes[-1]

    # -- stage 3: model acquisition and per-file verification ---------------

    def stage_model_acquisition(self) -> StageOutcome:
        self._require_budget("model_acquisition_verification")
        self.state = OrchestratorState.MODEL_ACQUISITION
        download_script = "\n".join(
            [
                "set -eu",
                f"rm -rf {_quote(self.paths.model_dir)}",
                f"mkdir -p {_quote(self.paths.model_dir)}",
                f"rm -rf {_quote(self.paths.remote_workspace + '/hf-home')}",
                "env -i PATH=/usr/local/bin:/usr/bin:/bin "
                f"HOME={_quote(self.paths.remote_workspace)} "
                f"HF_HOME={_quote(self.paths.remote_workspace + '/hf-home')} "
                "HF_HUB_DISABLE_TELEMETRY=1 "
                "huggingface-cli download "
                f"{_quote(MODEL_ID)} --revision {_quote(MODEL_REVISION)} "
                f"--local-dir {_quote(self.paths.model_dir)}",
                f"rm -rf {_quote(self.paths.remote_workspace + '/hf-home')}",
                f"rm -rf {_quote(self.paths.model_dir + '/.cache')}",
            ]
        )
        checked(
            self.runner,
            self.ssh_options.ssh_command("bash -s"),
            description="stage_model_acquisition_download",
            timeout=1800,
            input_text=download_script,
        )
        verification_script = build_model_inventory_verification_script(self.paths)
        checked(
            self.runner,
            self.ssh_options.ssh_command("bash -s"),
            description="stage_model_acquisition_verify",
            timeout=600,
            input_text=verification_script,
        )
        self._record("model_acquisition", True)
        return self.outcomes[-1]

    # -- stage 4: checked source staging, image pull/build, inspection ------

    def stage_image_preparation(self) -> StageOutcome:
        self._require_budget("image_pull_rebuild_inspect_attestation")
        self.state = OrchestratorState.IMAGE_PREPARATION

        # Verify the checked source archive's embedded commit purely
        # locally (no network, no live `git fetch` against any remote)
        # before it is ever staged onto the host.
        archive_bytes, archive_commit = read_runner_archive_commit_marker(
            self.config.local_runner_archive
        )
        if archive_commit != self.authorization.repository_head:
            raise HostOrchestrationError(
                "checked source archive's embedded commit does not match the "
                "authorization's repository_head "
                f"({archive_commit} != {self.authorization.repository_head})"
            )
        local_archive_digest = hashlib.sha256(archive_bytes).hexdigest()
        if (
            f"sha256:{local_archive_digest}"
            != self.authorization.derived_image_source_digest
        ):
            raise HostOrchestrationError(
                "checked source archive digest does not match the authorization's "
                "derived_image_source_digest"
            )
        self.expected_runner_source_digest = runner_source_digest_from_archive(
            archive_bytes
        )

        checked(
            self.runner,
            self.ssh_options.scp_upload_command(
                self.config.local_runner_archive,
                self.paths.source_archive_remote_path,
            ),
            description="stage_image_preparation_upload_source",
            timeout=600,
            input_text=None,
        )

        script = "\n".join(
            [
                "set -eu",
                f"docker pull {_quote(BASE_IMAGE_REFERENCE)}",
                f"echo BASE_REPODIGESTS=$(docker image inspect "
                f"{_quote(BASE_IMAGE_REFERENCE)} "
                '--format "{{json .RepoDigests}}")',
                f"echo BASE_IMAGE_ID=$(docker image inspect "
                f"{_quote(BASE_IMAGE_REFERENCE)} "
                '--format "{{.Id}}")',
                f"echo {_quote(local_archive_digest)}  "
                f"{_quote(self.paths.source_archive_remote_path)} | sha256sum -c -",
                f"rm -rf {_quote(self.paths.repo_dir)}",
                f"mkdir -p {_quote(self.paths.repo_dir)}",
                f"tar -xf {_quote(self.paths.source_archive_remote_path)} "
                f"-C {_quote(self.paths.repo_dir)}",
                f"test \"$(cat {_quote(self.paths.repo_dir + '/' + _COMMIT_HEAD_MEMBER)})\" "
                f"= {_quote(self.authorization.repository_head)}",
                f"echo EXPECTED_HEAD={_quote(self.authorization.repository_head)}",
                f"cd {_quote(self.paths.repo_dir)}",
                f"test \"$(sed -n '1p' containers/vllm-kv-truth/Containerfile)\" "
                f"= {_quote('FROM ' + BASE_IMAGE_REFERENCE)}",
                "docker build -q "
                f"--label {_quote(docker_run_label(self.authorization.nonce))} "
                f"--build-arg RUNNER_COMMIT={_quote(self.authorization.repository_head)} "
                "-f containers/vllm-kv-truth/Containerfile "
                f"-t kv-truth-derived-{_quote(self.authorization.nonce)} .",
                "echo DERIVED_IMAGE_ID=$(docker image inspect "
                f"kv-truth-derived-{_quote(self.authorization.nonce)} "
                '--format "{{.Id}}")',
            ]
        )
        result = checked(
            self.runner,
            self.ssh_options.ssh_command("bash -s"),
            description="stage_image_preparation",
            timeout=1800,
            input_text=script,
        )
        self.derived_image_id = self._verify_image_preparation_output(result.stdout)
        self._record("image_preparation", True)
        return self.outcomes[-1]

    def _verify_image_preparation_output(self, stdout: str) -> str:
        markers = dict(
            line.strip().split("=", 1)
            for line in stdout.splitlines()
            if "=" in line
            and line.strip().split("=", 1)[0]
            in {
                "BASE_REPODIGESTS",
                "BASE_IMAGE_ID",
                "EXPECTED_HEAD",
                "DERIVED_IMAGE_ID",
            }
        )
        if "BASE_REPODIGESTS" not in markers:
            raise HostOrchestrationError(
                "image preparation output did not include base RepoDigests"
            )
        try:
            digests = json.loads(markers["BASE_REPODIGESTS"])
        except ValueError as exc:
            raise HostOrchestrationError(
                "image preparation RepoDigests output was not valid JSON"
            ) from exc
        expected_digest = BASE_IMAGE_REFERENCE.split("@", 1)[-1]
        if not any(entry.endswith(expected_digest) for entry in digests):
            raise HostOrchestrationError(
                "pulled base image RepoDigests do not include the pinned digest"
            )
        if "EXPECTED_HEAD" not in markers:
            raise HostOrchestrationError(
                "image preparation output did not confirm the extracted source "
                "commit"
            )
        remote_head = markers["EXPECTED_HEAD"].strip()
        if remote_head != self.authorization.repository_head:
            raise HostOrchestrationError(
                "remote-confirmed source commit does not match the "
                "authorization's repository_head "
                f"({remote_head} != {self.authorization.repository_head})"
            )
        if "BASE_IMAGE_ID" not in markers or "DERIVED_IMAGE_ID" not in markers:
            raise HostOrchestrationError(
                "image preparation output did not include both base and derived "
                "image IDs"
            )
        base_id = markers["BASE_IMAGE_ID"].strip()
        derived_id = markers["DERIVED_IMAGE_ID"].strip()
        if _SHA256_REF.fullmatch(base_id) is None:
            raise HostOrchestrationError("base image id is malformed")
        if _SHA256_REF.fullmatch(derived_id) is None or derived_id == base_id:
            raise HostOrchestrationError(
                "derived image id is malformed or identical to the base image id"
            )
        return derived_id

    # -- stage 5: no-warmup canary -----------------------------------------

    def stage_canary(self) -> StageOutcome:
        self._require_budget("no_warmup_canary_event_feasibility_gate")
        self.state = OrchestratorState.CANARY
        invocation = LaneInvocation(
            lane="B",
            output_relative_path="canary.json",
            container_name=container_name(self.authorization.nonce, "canary"),
        )
        argv = build_docker_run_argv(
            authorization=self.authorization,
            paths=self.paths,
            invocation=invocation,
            derived_image_id=self._require_derived_image_id(),
        )
        checked(
            self.runner,
            self.ssh_options.ssh_command(" ".join(_quote(part) for part in argv)),
            description="stage_canary",
            timeout=900,
            input_text=None,
        )
        self._record("canary", True)
        return self.outcomes[-1]

    # -- stage 6: four fresh AB/BA/BA/AB pairs ------------------------------

    def stage_four_ab_pairs(self) -> tuple[StageOutcome, ...]:
        self._require_budget("four_fresh_ab_lifecycle_pairs")
        self.state = OrchestratorState.AB_PAIRS
        outcomes: list[StageOutcome] = []
        for pair_index, order in enumerate(AB_PAIR_ORDER, start=1):
            for slot, lane in enumerate(order, start=1):
                tag = f"pair{pair_index}-slot{slot}-{lane.lower()}"
                invocation = LaneInvocation(
                    lane=lane,
                    output_relative_path=f"{tag}.json",
                    container_name=container_name(self.authorization.nonce, tag),
                )
                argv = build_docker_run_argv(
                    authorization=self.authorization,
                    paths=self.paths,
                    invocation=invocation,
                    derived_image_id=self._require_derived_image_id(),
                )
                checked(
                    self.runner,
                    self.ssh_options.ssh_command(
                        " ".join(_quote(part) for part in argv)
                    ),
                    description=f"stage_four_ab_pairs[{tag}]",
                    timeout=900,
                    input_text=None,
                )
                self._record(tag, True)
                outcomes.append(self.outcomes[-1])
        return tuple(outcomes)

    # -- stage 7: fixed eviction lane ---------------------------------------

    def stage_eviction_lane(self) -> StageOutcome:
        self._require_budget("fixed_eviction_lane")
        self.state = OrchestratorState.EVICTION_LANE
        invocation = LaneInvocation(
            lane="eviction",
            output_relative_path="eviction.json",
            container_name=container_name(self.authorization.nonce, "eviction"),
        )
        argv = build_docker_run_argv(
            authorization=self.authorization,
            paths=self.paths,
            invocation=invocation,
            derived_image_id=self._require_derived_image_id(),
        )
        checked(
            self.runner,
            self.ssh_options.ssh_command(" ".join(_quote(part) for part in argv)),
            description="stage_eviction_lane",
            timeout=900,
            input_text=None,
        )
        self._record("eviction_lane", True)
        return self.outcomes[-1]

    # -- stage 8: private verify / redact / report / export -----------------

    def _load_verified_lane_receipt(
        self, evidence_dir: Path, tag: str
    ) -> dict[str, Any]:
        """Load and locally verify one lane's persisted receipt.

        :func:`kv_truth_runner.verify_protocol_receipt` re-derives the
        receipt's own canonical hash from its contents and refuses any
        mismatch -- this is what actually detects tampering/replacement of
        a transferred receipt, not just the outer archive digest check.
        """

        receipt_path = evidence_dir / f"{tag}.json"
        try:
            receipt = verify_protocol_receipt(receipt_path)
        except (KVTruthProtocolError, OSError, ValueError) as exc:
            raise HostOrchestrationError(
                f"lane receipt {tag!r} failed local verification: {exc}"
            ) from exc
        if receipt.protocol_id != PROTOCOL_ID:
            raise HostOrchestrationError(
                f"lane receipt {tag!r} has an unexpected protocol_id"
            )
        expected_lane = (
            "B"
            if tag == "canary"
            else "eviction" if tag == "eviction" else tag.rsplit("-", 1)[-1].upper()
        )
        if receipt.lane != expected_lane:
            raise HostOrchestrationError(f"lane receipt {tag!r} has an unexpected lane")
        if not receipt.lane_result.get("all_boundaries_valid", False):
            raise HostOrchestrationError(
                f"lane receipt {tag!r} contains an invalid capture boundary"
            )
        if receipt.runtime_attestation is None:
            raise HostOrchestrationError(
                f"lane receipt {tag!r} has no runtime attestation"
            )
        attestation = receipt.runtime_attestation
        identity = attestation["identity"]
        expected_identity = {
            "repository_commit": self.authorization.repository_head,
            "image_repository_digest": BASE_IMAGE_REFERENCE,
            "image_id": self._require_derived_image_id(),
            "vllm_version": REQUIRED_VLLM_VERSION,
            "vllm_commit": REQUIRED_VLLM_COMMIT,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "experiment_nonce": self.authorization.nonce,
        }
        model_inventory_digest, tokenizer_artifact_digest = (
            _expected_runtime_model_digests()
        )
        if self.expected_runner_source_digest is None:
            raise HostOrchestrationError(
                "authorized runner source digest was not established"
            )
        expected_identity.update(
            {
                "model_inventory_digest": model_inventory_digest,
                "tokenizer_artifact_digest": tokenizer_artifact_digest,
                "runner_source_digest": self.expected_runner_source_digest,
            }
        )
        mismatched = [
            key
            for key, expected in expected_identity.items()
            if identity.get(key) != expected
        ]
        if mismatched:
            raise HostOrchestrationError(
                f"lane receipt {tag!r} runtime identity does not match "
                f"authorization/image pins: {mismatched}"
            )
        events = attestation["kv_events_config"]
        if expected_lane in {"B", "eviction"} and (
            events.get("first_sequence") is None
            or events.get("last_sequence") is None
            or events.get("capture_end_monotonic", 0)
            <= events.get("capture_start_monotonic", 0)
        ):
            raise HostOrchestrationError(
                f"lane receipt {tag!r} has no valid measured KV-event boundary"
            )
        return receipt.to_dict()

    def _build_claim_matrix_and_lane_receipts(
        self, evidence_dir: Path
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Verify every executed lane's receipt and build the claim matrix.

        Only the four fresh "B" lanes and the eviction lane feed cache-reuse
        claims (the plan's claim matrix is about cache-enabled behaviour);
        the canary and the four "A" lanes are cache-disabled output-identity
        controls and are recorded in ``lane_receipts`` for completeness but
        never scored against ``expected_reusable_tokens``.
        """

        lane_receipts: dict[str, Any] = {}
        for tag in ("canary", *ab_pair_lane_tags(), "eviction"):
            lane_receipts[tag] = self._load_verified_lane_receipt(evidence_dir, tag)

        claims: list[Any] = []
        for tag in b_lane_output_tags():
            lane_result = lane_receipts[tag]["lane_result"]
            salt_entry = evidence.build_salt_isolation_claim(
                lane_result, salt_supported=CACHE_SALT_SUPPORT_CONFIRMED
            )
            for entry in evidence.build_claim_matrix(lane_result):
                claims.append(
                    salt_entry if entry.scenario == "namespace_isolation" else entry
                )
        claims.append(
            evidence.build_eviction_claim(lane_receipts["eviction"]["lane_result"])
        )
        return tuple(claims), lane_receipts

    def stage_transfer_evidence(self, local_bundle_dir: Path) -> StageOutcome:
        self._require_budget("private_verify_redact_report_export")
        self.state = OrchestratorState.EVIDENCE_EXPORT
        remote_tar = f"{self.config.remote_workspace}/evidence.tar"
        checked(
            self.runner,
            self.ssh_options.ssh_command(
                f"tar -cf {_quote(remote_tar)} -C "
                f"{_quote(self.config.remote_workspace)} evidence receipts"
            ),
            description="stage_transfer_evidence_archive",
            timeout=300,
            input_text=None,
        )
        digest_result = checked(
            self.runner,
            self.ssh_options.ssh_command(f"sha256sum {_quote(remote_tar)}"),
            description="stage_transfer_evidence_digest",
            timeout=60,
            input_text=None,
        )
        remote_digest = digest_result.stdout.split()[0].strip()
        if _SHA256_HEX.fullmatch(remote_digest) is None:
            raise HostOrchestrationError(
                "remote evidence archive digest output was malformed"
            )
        local_bundle_dir.mkdir(parents=True, exist_ok=True)
        local_tar = local_bundle_dir / "evidence.tar"
        checked(
            self.runner,
            self.ssh_options.scp_download_command(remote_tar, local_tar),
            description="stage_transfer_evidence_download",
            timeout=300,
            input_text=None,
        )
        local_bytes = local_tar.read_bytes()
        local_digest = hashlib.sha256(local_bytes).hexdigest()
        if local_digest != remote_digest:
            raise HostOrchestrationError(
                "locally verified evidence archive digest does not match the "
                "digest computed on the remote host before transfer"
            )
        raw_evidence_dir = local_bundle_dir / "raw_evidence"
        extract_safe_tar(local_bytes, raw_evidence_dir)
        claim_matrix, lane_receipts = self._build_claim_matrix_and_lane_receipts(
            raw_evidence_dir / "evidence"
        )
        private_bundle = evidence.PrivateEvidenceBundle(
            run_mode=evidence.RUN_MODE_REAL_RUN,
            experiment_nonce=self.authorization.nonce,
            authorization=self.authorization.to_dict(),
            ssh_options_public_record=self.ssh_options.public_record(),
            lane_receipts=lane_receipts,
            claim_matrix=claim_matrix,
            list_rate_ledger=evidence.ListRateLedger(
                entries=(self._ledger_entry("evidence_transferred_and_verified"),)
            ),
            teardown_receipt=None,
        )
        private_bundle.write(local_bundle_dir / "private_bundle.json")
        self._record("transfer_evidence", True, detail=local_digest)
        return self.outcomes[-1]

    # -- stage 9: teardown ----------------------------------------------------

    def _finalize_evidence_bundle(
        self,
        local_bundle_dir: Path,
        *,
        residual_containers: int,
        residual_gpu_processes: int,
    ) -> None:
        """Patch the teardown receipt into the private bundle and publish
        the deterministic public-redacted bundle -- only if the private
        bundle actually exists (i.e. ``stage_transfer_evidence`` completed
        on this run; an earlier failed stage means there is nothing to
        finalize, and teardown must still proceed regardless)."""

        private_path = local_bundle_dir / "private_bundle.json"
        if not private_path.exists():
            return
        private_bundle = evidence.PrivateEvidenceBundle.read(private_path)
        teardown_receipt = evidence.TeardownReceipt(
            residual_containers=residual_containers,
            residual_gpu_processes=residual_gpu_processes,
            evidence_transferred_and_verified=True,
            shutdown_issued=True,
            safe_to_terminate_message_emitted=True,
        )
        finalized = replace(
            private_bundle,
            teardown_receipt=teardown_receipt,
            list_rate_ledger=evidence.ListRateLedger(
                entries=(
                    *private_bundle.list_rate_ledger.entries,
                    self._ledger_entry("shutdown_issued"),
                )
            ),
        )
        finalized.write(private_path)
        public_bundle = evidence.PublicRedactedBundle.from_private(finalized)
        evidence.write_public_bundle_directory(
            public_bundle, local_bundle_dir / "public"
        )

    def stage_teardown(self, local_bundle_dir: Path) -> StageOutcome:
        self.state = OrchestratorState.TEARDOWN
        nonce = self.authorization.nonce
        label_filter = f"label={docker_run_label(nonce)}"
        script = "\n".join(
            [
                "set -eu",
                f"docker ps -q --filter {_quote(label_filter)} | "
                "xargs -r docker stop",
                f"docker ps -aq --filter {_quote(label_filter)} | "
                "xargs -r docker rm -f",
                f"docker images -q --filter {_quote(label_filter)} | "
                "xargs -r docker rmi -f || true",
                f"docker rmi -f {_quote(BASE_IMAGE_REFERENCE)} || true",
                f"rm -rf {_quote(self.paths.model_dir)}",
                f"rm -rf {_quote(self.paths.repo_dir)}",
                f"rm -rf {_quote(self.paths.evidence_dir)}",
                f"rm -rf {_quote(self.paths.receipts_dir)}",
                f"rm -rf {_quote(self.config.remote_workspace + '/hf-home')}",
                f"rm -f {_quote(self.paths.source_archive_remote_path)}",
                f"rm -f {_quote(self.config.remote_workspace + '/evidence.tar')}",
                'AUTHORIZED_KEYS="$HOME/.ssh/authorized_keys"',
                f"KEY_MARKER={_quote(self.config.authorized_key_marker)}",
                'test -f "$AUTHORIZED_KEYS"',
                'KEY_TMP=$(mktemp "$HOME/.ssh/authorized_keys.llmtracefx.XXXXXX")',
                "awk -v marker=\"$KEY_MARKER\" '$NF != marker' "
                '"$AUTHORIZED_KEYS" > "$KEY_TMP"',
                'chmod --reference="$AUTHORIZED_KEYS" "$KEY_TMP"',
                'mv "$KEY_TMP" "$AUTHORIZED_KEYS"',
                'test "$(awk -v marker="$KEY_MARKER" '
                "'$NF == marker { count++ } END { print count + 0 }' "
                '"$AUTHORIZED_KEYS")" = "0"',
                'echo "RESIDUAL_CONTAINERS=$(' 'docker ps -q | wc -l)"',
                'echo "RESIDUAL_GPU_PROCESSES=$('
                "nvidia-smi --query-compute-apps=pid --format=csv,noheader "
                '2>/dev/null | sed "/^[[:space:]]*$/d" | wc -l)"',
                f"rmdir {_quote(self.config.remote_workspace)}",
            ]
        )
        result = checked(
            self.runner,
            self.ssh_options.ssh_command("bash -s"),
            description="stage_teardown_cleanup",
            timeout=300,
            input_text=script,
        )
        residual_containers, residual_gpu_processes = self._verify_teardown_output(
            result.stdout
        )
        checked(
            self.runner,
            self.ssh_options.ssh_command("sudo -n shutdown -h now"),
            description="stage_teardown_shutdown",
            timeout=30,
            input_text=None,
        )
        self._finalize_evidence_bundle(
            local_bundle_dir,
            residual_containers=residual_containers,
            residual_gpu_processes=residual_gpu_processes,
        )
        self._record("teardown", True)
        self.state = OrchestratorState.COMPLETE
        print("SAFE TO TERMINATE INSTANCE NOW")
        return self.outcomes[-1]

    def _verify_teardown_output(self, stdout: str) -> tuple[int, int]:
        markers = dict(
            line.split("=", 1)
            for line in stdout.splitlines()
            if "=" in line
            and line.split("=", 1)[0]
            in {"RESIDUAL_CONTAINERS", "RESIDUAL_GPU_PROCESSES"}
        )
        if (
            "RESIDUAL_CONTAINERS" not in markers
            or "RESIDUAL_GPU_PROCESSES" not in markers
        ):
            raise HostOrchestrationError(
                "teardown residual-state check produced no parsable markers"
            )
        residual_containers = int(markers["RESIDUAL_CONTAINERS"].strip())
        residual_gpu_processes = int(markers["RESIDUAL_GPU_PROCESSES"].strip())
        if residual_containers != 0:
            raise HostOrchestrationError(
                "residual experiment-scoped containers remain after teardown"
            )
        if residual_gpu_processes != 0:
            raise HostOrchestrationError(
                "residual GPU compute processes remain after teardown"
            )
        return residual_containers, residual_gpu_processes

    # -- top-level, interruption-safe run ------------------------------------

    def run(self, *, local_evidence_bundle_dir: Path) -> tuple[StageOutcome, ...]:
        """Execute every stage in order; teardown always runs in ``finally``,
        even if an earlier stage raised or the process is interrupted."""

        try:
            self.stage_preflight()
            self.stage_identity_gate()
            self.stage_model_acquisition()
            self.stage_image_preparation()
            self.stage_canary()
            self.stage_four_ab_pairs()
            self.stage_eviction_lane()
            self.stage_transfer_evidence(local_evidence_bundle_dir)
        except BaseException as exc:  # noqa: BLE001 - deliberate: teardown
            # must run on every possible exit path, including
            # KeyboardInterrupt, before the exception propagates.
            self.state = OrchestratorState.FAILED
            self._record("run", False, detail=type(exc).__name__)
            raise
        finally:
            self.stage_teardown(local_evidence_bundle_dir)
        return tuple(self.outcomes)
