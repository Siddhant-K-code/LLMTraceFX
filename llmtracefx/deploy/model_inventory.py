"""Shared validation for published model-file inventories."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import DeploymentPlanError
from .manifest import MANIFEST_SCHEMA_VERSION, StagedFile, WeightStagingManifest

GLM_53_FLASH_REVISION = "03eb5366286afd40d2221b1d9c63a6dd1ba4832e"
GLM_53_FLASH_FILE_COUNT = 72
GLM_53_FLASH_TOTAL_BYTES = 328_366_172_318
GLM_53_FLASH_SAFETENSORS_SHARDS = 62
GLM_53_FLASH_PUBLISHED_HASHES = 63
GLM_53_FLASH_INVENTORY_SHA256 = (
    "298d7174291301065e2e62ee87b7fa62763d11ec4724184293a249a52861e613"
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PublishedInventory:
    """An immutable upstream inventory, suitable for remote verification."""

    source: str
    repo_id: str
    revision: str
    files: tuple[StagedFile, ...]

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def total_bytes(self) -> int:
        return sum(entry.size_bytes for entry in self.files)

    @property
    def published_hash_count(self) -> int:
        return sum(entry.sha256 is not None for entry in self.files)

    @property
    def safetensors_shard_count(self) -> int:
        return sum(entry.path.endswith(".safetensors") for entry in self.files)

    @property
    def canonical_sha256(self) -> str:
        payload = {
            "source": self.source,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "files": [entry.to_dict() for entry in self.files],
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def assert_glm_53_flash(self) -> None:
        expected = {
            "repo_id": "zai-org/GLM-5.3-Flash",
            "revision": GLM_53_FLASH_REVISION,
            "file_count": GLM_53_FLASH_FILE_COUNT,
            "total_bytes": GLM_53_FLASH_TOTAL_BYTES,
            "safetensors_shard_count": GLM_53_FLASH_SAFETENSORS_SHARDS,
            "published_hash_count": GLM_53_FLASH_PUBLISHED_HASHES,
            "canonical_sha256": GLM_53_FLASH_INVENTORY_SHA256,
        }
        observed = {
            "repo_id": self.repo_id,
            "revision": self.revision,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "safetensors_shard_count": self.safetensors_shard_count,
            "published_hash_count": self.published_hash_count,
            "canonical_sha256": self.canonical_sha256,
        }
        mismatches = [
            f"{name}={observed[name]!r}, expected {value!r}"
            for name, value in expected.items()
            if observed[name] != value
        ]
        if mismatches:
            raise DeploymentPlanError(
                "published GLM-5.3-Flash inventory mismatch: " + "; ".join(mismatches)
            )

    def expected_staging_manifest(
        self, *, mount_path: str, completed_at: str
    ) -> WeightStagingManifest:
        return WeightStagingManifest(
            schema_version=MANIFEST_SCHEMA_VERSION,
            completed_at=completed_at,
            repo_id=self.repo_id,
            revision=self.revision,
            mount_path=mount_path,
            files=self.files,
            generator="llmtracefx.deploy.model_inventory",
        )

    def summary(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "safetensors_shard_count": self.safetensors_shard_count,
            "files_with_published_sha256": self.published_hash_count,
            "canonical_inventory_sha256": self.canonical_sha256,
        }


def inventory_from_dict(data: Mapping[str, Any]) -> PublishedInventory:
    """Parse and validate a published inventory without fetching anything."""
    try:
        raw_files = data["files"]
        if not isinstance(raw_files, list):
            raise TypeError("files must be a list")
        files = tuple(
            StagedFile(
                path=str(entry["path"]),
                size_bytes=int(entry["size_bytes"]),
                sha256=entry.get("sha256"),
            )
            for entry in raw_files
        )
        inventory = PublishedInventory(
            source=str(data["source"]),
            repo_id=str(data["repo_id"]),
            revision=str(data["revision"]),
            files=files,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DeploymentPlanError(f"malformed published inventory: {exc}") from exc

    paths = [entry.path for entry in files]
    if paths != sorted(paths):
        raise DeploymentPlanError("published inventory files must be path sorted")
    if len(paths) != len(set(paths)):
        raise DeploymentPlanError("published inventory contains duplicate paths")
    for entry in files:
        if entry.sha256 is not None and (
            not isinstance(entry.sha256, str) or not _SHA256.fullmatch(entry.sha256)
        ):
            raise DeploymentPlanError(
                f"published inventory has malformed SHA-256 for {entry.path}"
            )

    declared = {
        "file_count": inventory.file_count,
        "total_bytes": inventory.total_bytes,
        "safetensors_shard_count": inventory.safetensors_shard_count,
        "files_with_published_sha256": inventory.published_hash_count,
    }
    for field, observed in declared.items():
        if field in data and data[field] != observed:
            raise DeploymentPlanError(
                f"published inventory {field}={data[field]!r}, observed {observed!r}"
            )
    return inventory


def load_inventory(path: Path) -> PublishedInventory:
    """Load a committed metadata inventory; model bytes are never opened."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeploymentPlanError(f"cannot load published inventory: {exc}") from exc
    if not isinstance(data, dict):
        raise DeploymentPlanError("published inventory root must be an object")
    return inventory_from_dict(data)
