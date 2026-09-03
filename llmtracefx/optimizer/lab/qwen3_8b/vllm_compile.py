"""Shared immutable contract for the Qwen3-8B vLLM compilation experiment."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPECTED_MODEL_FILE_COUNT = 15
EXPECTED_MODEL_BYTES = 16_397_461_266
REQUESTS_PER_CELL = 12
WORKLOAD_IDS = (
    "structured-json-profile-extraction",
    "prose-reasoning-two-train-gap",
)
CONTEXT_TIERS = ("2k", "8k", "16k")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024
_PROMPT_MANIFEST = (
    Path(__file__).parent / "data" / "qwen3-8b-control-manifest-template-v1.json"
)


class VLLMCompileContractError(ValueError):
    """Raised when the immutable experiment contract is violated."""


def canonical_json(value: Any) -> str:
    """Return finite, stable JSON used for experiment hashes."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VLLMCompileContractError(f"value is not canonical JSON: {exc}") from exc


@dataclass(frozen=True)
class WorkloadDescriptor:
    """One request in the fixed two-workload, three-tier experiment."""

    ordinal: int
    workload_id: str
    workload_version: str
    context_tier: str
    repetition: int
    prompt_sha256: str
    warmup: bool = False

    @property
    def request_id(self) -> str:
        return f"{self.context_tier}-{self.workload_id}-rep-{self.repetition:02d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "request_id": self.request_id,
            "workload_id": self.workload_id,
            "workload_version": self.workload_version,
            "context_tier": self.context_tier,
            "repetition": self.repetition,
            "prompt_sha256": self.prompt_sha256,
            "warmup": self.warmup,
        }


def _prompt_hashes() -> dict[str, dict[str, str]]:
    path = _PROMPT_MANIFEST
    if path.is_symlink() or not path.is_file():
        raise VLLMCompileContractError("packaged Qwen manifest is unavailable")
    if path.stat().st_size > _MAX_MANIFEST_BYTES:
        raise VLLMCompileContractError("packaged Qwen manifest exceeds its bound")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda item: (_ for _ in ()).throw(
                VLLMCompileContractError(f"non-finite manifest value: {item}")
            ),
        )
        workloads = value["workloads"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise VLLMCompileContractError(
            f"packaged Qwen manifest is invalid: {exc}"
        ) from exc
    if not isinstance(workloads, list) or len(workloads) != len(WORKLOAD_IDS):
        raise VLLMCompileContractError("packaged workloads differ from the contract")
    result: dict[str, dict[str, str]] = {}
    for workload_id, item in zip(WORKLOAD_IDS, workloads, strict=True):
        if not isinstance(item, dict) or item.get("workload_id") != workload_id:
            raise VLLMCompileContractError("packaged workload order drifted")
        hashes = item.get("prompt_hashes")
        if not isinstance(hashes, dict) or set(hashes) != set(CONTEXT_TIERS):
            raise VLLMCompileContractError("packaged prompt hashes are incomplete")
        if any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in hashes.values()
        ):
            raise VLLMCompileContractError("packaged prompt hash is invalid")
        result[workload_id] = dict(hashes)
    return result


def workload_descriptors() -> tuple[WorkloadDescriptor, ...]:
    """Return the exact 12-request order with no warmup requests."""
    hashes = _prompt_hashes()
    descriptors: list[WorkloadDescriptor] = []
    for tier in CONTEXT_TIERS:
        for workload_id in WORKLOAD_IDS:
            for repetition in (1, 2):
                descriptors.append(
                    WorkloadDescriptor(
                        ordinal=len(descriptors) + 1,
                        workload_id=workload_id,
                        workload_version="1",
                        context_tier=tier,
                        repetition=repetition,
                        prompt_sha256=hashes[workload_id][tier],
                    )
                )
    if len(descriptors) != REQUESTS_PER_CELL:
        raise VLLMCompileContractError("request count drifted")
    return tuple(descriptors)


def token_ids_sha256(token_ids: list[int]) -> str:
    """Return the canonical identity of one exact token array."""
    if (
        not token_ids
        or any(
            isinstance(item, bool) or not isinstance(item, int) for item in token_ids
        )
        or any(item < 0 for item in token_ids)
    ):
        raise VLLMCompileContractError("token IDs must be non-negative integers")
    return "sha256:" + hashlib.sha256(canonical_json(token_ids).encode()).hexdigest()


__all__ = [
    "EXPECTED_MODEL_BYTES",
    "EXPECTED_MODEL_FILE_COUNT",
    "MODEL_ID",
    "MODEL_REVISION",
    "REQUESTS_PER_CELL",
    "VLLMCompileContractError",
    "WorkloadDescriptor",
    "canonical_json",
    "token_ids_sha256",
    "workload_descriptors",
]
