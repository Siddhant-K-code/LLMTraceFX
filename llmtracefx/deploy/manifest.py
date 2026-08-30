"""Manifests: what was staged, and what actually served the request.

Two records, written at two different moments, for two different
questions.

The staging manifest answers "what is on the volume". It is written by
the CPU staging step and records the repository, the revision and every
file that landed, with sizes and with hashes where the source provided
them. Sizes always exist because they are observed locally; hashes are
optional because they depend on what the upstream metadata carried, and
inventing one would be worse than admitting its absence.

The server manifest answers "what served this". It is deliberately a
description of configuration and observation, never of quality: it
records that four H200s were requested and how long start up took, and it
records nothing about tokens per second. Performance evidence belongs to
the collector, which measures from outside.

Neither record has a field capable of holding a credential. Environment
variables appear as names, never as values.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import DeploymentPlanError

MANIFEST_SCHEMA_VERSION = "1"

CONFIGURED = "configured"
OBSERVED = "observed"


@dataclass(frozen=True)
class StagedFile:
    """One file present on the volume after staging."""

    path: str
    size_bytes: int
    sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise DeploymentPlanError("staged file path must be a non-empty string")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise DeploymentPlanError("staged file size_bytes must be an integer")
        if self.size_bytes < 0:
            raise DeploymentPlanError("staged file size_bytes must not be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WeightStagingManifest:
    """The inventory written next to the weights on the volume."""

    schema_version: str
    completed_at: str
    repo_id: str
    revision: str
    mount_path: str
    files: tuple[StagedFile, ...] = field(default_factory=tuple)
    generator: str = "llmtracefx.deploy.manifest"

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.files)

    @property
    def hashed_file_count(self) -> int:
        return sum(1 for entry in self.files if entry.sha256 is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "llmtracefx.deploy.weight_staging_manifest",
            "completed_at": self.completed_at,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "mount_path": self.mount_path,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "total_gib": round(self.total_bytes / (1024**3), 3),
            "hashed_file_count": self.hashed_file_count,
            "files": [entry.to_dict() for entry in self.files],
            "generator": self.generator,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WeightStagingManifest:
        try:
            files = tuple(
                StagedFile(
                    path=str(entry["path"]),
                    size_bytes=int(entry["size_bytes"]),
                    sha256=entry.get("sha256"),
                )
                for entry in data.get("files", ())
            )
            return cls(
                schema_version=str(data.get("schema_version", MANIFEST_SCHEMA_VERSION)),
                completed_at=str(data["completed_at"]),
                repo_id=str(data["repo_id"]),
                revision=str(data["revision"]),
                mount_path=str(data["mount_path"]),
                files=files,
                generator=str(data.get("generator", "llmtracefx.deploy.manifest")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DeploymentPlanError(
                f"malformed weight staging manifest: {exc}"
            ) from exc

    def matches(self, *, repo_id: str, revision: str) -> bool:
        """Whether this manifest describes exactly the requested weights.

        Used to make re-staging idempotent. A manifest for a different
        revision is not a partial match to be topped up; it describes
        different weights, so the answer is simply no.
        """
        return self.repo_id == repo_id and self.revision == revision


@dataclass(frozen=True)
class ServerManifest:
    """What was configured, and what was observed, for one served process.

    ``provenance`` marks every substantive field as either configured (an
    intent this harness expressed) or observed (something the container
    reported at run time). Without that distinction a reader cannot tell
    "we asked for four H200s" from "four H200s were present", and those
    are different claims.
    """

    schema_version: str
    collected_at: str
    app_name: str
    gpu_type: str
    gpu_count: int
    framework: str
    framework_version: str
    image_reference: str
    image_digest: str | None
    model_repo_id: str
    model_revision: str
    quantization: str
    quantization_format: str
    activation_scheme: str
    tensor_parallel_size: int
    context_length: int
    expert_parallel_size: int | None = None
    observed_gpus: tuple[str, ...] = field(default_factory=tuple)
    observed_cuda_version: str | None = None
    startup_seconds: float | None = None
    credential_env_var_names_present: tuple[str, ...] = field(default_factory=tuple)
    generator: str = "llmtracefx.deploy.manifest"

    def provenance(self) -> dict[str, str]:
        return {
            "gpu_type": CONFIGURED,
            "gpu_count": CONFIGURED,
            "framework": CONFIGURED,
            "framework_version": CONFIGURED,
            "image_reference": CONFIGURED,
            "image_digest": CONFIGURED,
            "model_repo_id": CONFIGURED,
            "model_revision": CONFIGURED,
            "quantization": CONFIGURED,
            "quantization_format": CONFIGURED,
            "activation_scheme": CONFIGURED,
            "tensor_parallel_size": CONFIGURED,
            "expert_parallel_size": CONFIGURED,
            "context_length": CONFIGURED,
            "observed_gpus": OBSERVED,
            "observed_cuda_version": OBSERVED,
            "startup_seconds": OBSERVED,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "llmtracefx.deploy.server_manifest",
            "collected_at": self.collected_at,
            "app_name": self.app_name,
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "image_reference": self.image_reference,
            "image_digest": self.image_digest,
            "image_digest_pinned": self.image_digest is not None,
            "model_repo_id": self.model_repo_id,
            "model_revision": self.model_revision,
            "quantization": self.quantization,
            "quantization_format": self.quantization_format,
            "activation_scheme": self.activation_scheme,
            "tensor_parallel_size": self.tensor_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "context_length": self.context_length,
            "observed_gpus": list(self.observed_gpus),
            "observed_cuda_version": self.observed_cuda_version,
            "startup_seconds": (
                None if self.startup_seconds is None else round(self.startup_seconds, 3)
            ),
            "credential_env_var_names_present": list(
                self.credential_env_var_names_present
            ),
            "provenance": self.provenance(),
            "performance_claims": (
                "None. Throughput and latency evidence is produced by the "
                "LLMTraceFX API collector against this endpoint, not by the "
                "server describing itself."
            ),
            "generator": self.generator,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


def present_env_var_names(
    environ: Mapping[str, str], names: Sequence[str]
) -> tuple[str, ...]:
    """Which of ``names`` are set and non-empty. Names only, never values.

    This is the only function in the package that looks at an
    environment mapping that may contain a credential, and it returns
    keys. There is no code path by which a value reaches a manifest.
    """
    return tuple(name for name in names if (environ.get(name) or "").strip())


VERIFICATION_FILENAME = "verification.json"

MISSING = "missing"
SIZE_MISMATCH = "size_mismatch"
HASH_MISMATCH = "hash_mismatch"
UNREADABLE = "unreadable"


@dataclass(frozen=True)
class VerificationIssue:
    """One file that does not match what staging recorded."""

    path: str
    problem: str
    expected: str | None = None
    observed: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WeightVerification:
    """Evidence that the staged bytes are still the bytes staging recorded.

    Written by a CPU step and required by the serving container. A
    manifest naming the right revision proves only that a download was
    started for it: an interruption near the end leaves the manifest and
    a short file behind. Finding that on CPU costs container seconds;
    finding it after the serving container has allocated four
    accelerators costs the whole start up.
    """

    schema_version: str
    verified_at: str
    repo_id: str
    revision: str
    mount_path: str
    files_checked: int
    bytes_checked: int
    hashes_checked: int
    hashes_available: int
    issues: tuple[VerificationIssue, ...] = field(default_factory=tuple)
    generator: str = "llmtracefx.deploy.manifest"

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": "llmtracefx.deploy.weight_verification",
            "verified_at": self.verified_at,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "mount_path": self.mount_path,
            "ok": self.ok,
            "files_checked": self.files_checked,
            "bytes_checked": self.bytes_checked,
            "hashes_checked": self.hashes_checked,
            "hashes_available": self.hashes_available,
            "issues": [issue.to_dict() for issue in self.issues],
            "generator": self.generator,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WeightVerification:
        try:
            issues = tuple(
                VerificationIssue(
                    path=str(entry["path"]),
                    problem=str(entry["problem"]),
                    expected=entry.get("expected"),
                    observed=entry.get("observed"),
                )
                for entry in data.get("issues", ())
            )
            return cls(
                schema_version=str(data.get("schema_version", MANIFEST_SCHEMA_VERSION)),
                verified_at=str(data["verified_at"]),
                repo_id=str(data["repo_id"]),
                revision=str(data["revision"]),
                mount_path=str(data["mount_path"]),
                files_checked=int(data["files_checked"]),
                bytes_checked=int(data["bytes_checked"]),
                hashes_checked=int(data["hashes_checked"]),
                hashes_available=int(data.get("hashes_available", 0)),
                issues=issues,
                generator=str(data.get("generator", "llmtracefx.deploy.manifest")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DeploymentPlanError(f"malformed weight verification: {exc}") from exc

    def covers(self, *, repo_id: str, revision: str) -> bool:
        return self.ok and self.repo_id == repo_id and self.revision == revision


def _sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def verify_staged_weights(
    manifest: WeightStagingManifest,
    root: Path,
    *,
    now: Callable[[], str],
    check_hashes: bool = True,
) -> WeightVerification:
    """Re-check a staged tree against its manifest. CPU and disk only.

    Sizes are always checked because they are nearly free and catch the
    common failure, a truncated or absent shard. Hashes are checked when
    the repository published one and the caller asked for it; that reads
    every byte, so it is the expensive half and the record says which
    half ran rather than implying both did.
    """
    issues: list[VerificationIssue] = []
    bytes_checked = 0
    hashes_checked = 0
    hashes_available = 0

    for entry in manifest.files:
        target = root / entry.path
        if entry.sha256 is not None:
            hashes_available += 1
        try:
            if not target.is_file():
                issues.append(VerificationIssue(path=entry.path, problem=MISSING))
                continue
            observed_size = target.stat().st_size
        except OSError as exc:
            issues.append(
                VerificationIssue(
                    path=entry.path, problem=UNREADABLE, observed=type(exc).__name__
                )
            )
            continue

        if observed_size != entry.size_bytes:
            issues.append(
                VerificationIssue(
                    path=entry.path,
                    problem=SIZE_MISMATCH,
                    expected=str(entry.size_bytes),
                    observed=str(observed_size),
                )
            )
            continue
        bytes_checked += observed_size

        if check_hashes and entry.sha256 is not None:
            try:
                observed_hash = _sha256_file(target)
            except OSError as exc:
                issues.append(
                    VerificationIssue(
                        path=entry.path,
                        problem=UNREADABLE,
                        observed=type(exc).__name__,
                    )
                )
                continue
            hashes_checked += 1
            if observed_hash != entry.sha256:
                # The digests themselves are not recorded in the issue.
                # They are long, they are not actionable, and the answer
                # is always the same: re-stage that file.
                issues.append(VerificationIssue(path=entry.path, problem=HASH_MISMATCH))

    return WeightVerification(
        schema_version=MANIFEST_SCHEMA_VERSION,
        verified_at=now(),
        repo_id=manifest.repo_id,
        revision=manifest.revision,
        mount_path=manifest.mount_path,
        files_checked=manifest.file_count,
        bytes_checked=bytes_checked,
        hashes_checked=hashes_checked,
        hashes_available=hashes_available,
        issues=tuple(issues),
    )
