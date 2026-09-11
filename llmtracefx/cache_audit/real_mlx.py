"""Private, fail-closed runner for the real Apple Silicon MLX cache experiment.

The normal cache-audit CLI remains the small cross-backend interface.  This
module owns the deliberately narrower, private workflow: compile exact local
tokens, calibrate deterministic output, execute one isolated replicate, then
assemble and redact the six-attempt evidence envelope.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import re
import secrets
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from importlib import metadata
from itertools import groupby
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from llmtracefx.evidence.core import (
    PRIVACY_PATTERNS,
    PRIVATE_JSON_KEYS,
    canonical_json,
)
from llmtracefx.optimizer.collectors._shared import atomic_write_text

from .adapters.base import AdapterAuditIdentity, CacheAuditCapability
from .adapters.mlx import (
    REQUIRED_MLX_LM_VERSION,
    REQUIRED_MLX_VERSION,
    MLXCacheRuntime,
    MLXLocalCacheAdapter,
    MLXStageObservation,
    ProductionMLXRuntime,
    check_mlx_capabilities,
)
from .bundle import (
    CacheAuditBundleError,
    _git_package_digest,
    copy_data_only_bundle,
    package_source_digest,
    read_bundle,
    sanitize_bundle_records,
    verify_bundle,
    write_bundle,
)
from .runner import run_audit, source_commit
from .schema import (
    CacheConfig,
    EligibilityStatus,
    PairRole,
    PublicationMode,
    RequestEvidence,
    RequestSpec,
    ScenarioKind,
    TerminalState,
    Verdict,
)

WORKLOAD_SCHEMA_VERSION = "real-mlx-workload-v2"
AGGREGATE_SCHEMA_VERSION = "real-mlx-aggregate-v2"
REPLICATE_IDS = tuple(f"replicate-{index}" for index in range(6))
LANE_IDS = ("1k", "4k")
SCHEDULE_AFFINE_PERMUTATIONS = (
    (0, 1),
    (1, 3),
    (2, 5),
    (3, 9),
    (4, 11),
    (5, 13),
)
CALIBRATION_ARRAY_NAMES = (
    "base",
    "different_ids",
    "mutation_137",
    "mutation_256",
    "suffix_change",
    "eviction_a",
    "eviction_b",
    "eviction_c",
)
MAX_CACHE_ENTRIES = 2
MAX_CACHE_BYTES = 1 << 63
MAX_OUTPUT_TOKENS = 8
MAX_ALLOCATOR_PEAK_BYTES = 8 * 1024**3
MAX_PROCESS_RSS_BYTES = 12 * 1024**3
MAX_SWAP_BYTES = 14 * 1024**3
PREFLIGHT_MAX_SWAP_BYTES = 12 * 1024**3
MAX_SWAP_GROWTH_BYTES = 2 * 1024**3
MIN_RUNTIME_MEMORY_FREE_PERCENT = 15.0
PREFLIGHT_MIN_AVAILABLE_RATIO = 0.25
RUNTIME_MIN_AVAILABLE_RATIO = 0.15
PREFLIGHT_MIN_DISK_BYTES = 20 * 1024**3
RUNTIME_MIN_DISK_BYTES = 12 * 1024**3
HEAVY_PROCESS_RSS_BYTES = 1 * 1024**3
REQUIRED_HOST_CHIP = "Apple M5 Pro"
REQUIRED_HOST_MEMORY_BYTES = 24 * 1024**3
REQUIRED_LLMTRACEFX_VERSION = "1.0.0"
REQUIRED_TRANSFORMERS_VERSION = "5.16.1"
REQUIRED_SAFETENSORS_VERSION = "0.8.0"
REQUIRED_NUMPY_VERSION = "2.2.6"
REQUIRED_TOKENIZERS_VERSION = "0.23.1"
EXPECTED_RUNTIME_PACKAGE_IDENTITIES_SHA256 = (
    "sha256:391e14ce1b09b5de11b96ab24f98dd73c4c8334625599e1cda5dd5f4906c16a5"
)
CHILD_TIMEOUT_SECONDS = 12 * 60
TOTAL_TIMEOUT_SECONDS = 90 * 60
MONITOR_INTERVAL_SECONDS = 2.0
PROCESS_GROUP_GRACE_SECONDS = 5.0
MAX_CHILD_LOG_BYTES = 64 * 1024
RUN_INSTANCE_ENV = "LLMTRACEFX_CACHE_AUDIT_INSTANCE_ID"
EXPECTED_MODEL_FILE_COUNT = 8
EXPECTED_CONVERSION_SUMMARY_SHA256 = (
    "9c87cad2a7de7bbc42bfd6a1d7f502c32422ba00b29df27a2363c07aa2a45c25"
)
EXPECTED_MODEL_ARTIFACT_DIGEST = (
    "sha256:057a37f4ebc76420f8ab2edb17bc8442e050c8d13f7334f829356e2f9cab6802"
)
EXPECTED_CALIBRATED_WORKLOAD_DIGEST = (
    "sha256:99469aa8b3c35b361abe3cc27c3b6fa7fee063ae39318b52968bd1ce3f74d56a"
)
EXPECTED_CALIBRATED_LANE_DIGESTS = {
    "1k": "sha256:ccb34da854656dd955987ba470c7896063c54f3cdebb990b6961205fbf882ff1",
    "4k": "sha256:44accae13537893a4ff1ffeac2c224045e11a228741c1f0c3c75ee891477f674",
}
EXPECTED_CALIBRATION_OUTPUT_TOKENS = {"1k": 3, "4k": 3}
MODEL_ID = "local-self-converted/qwen3-4b-mlx-q4g64"
TOKENIZER_ID = "Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c"
_INSTALLED_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path(
    os.environ.get("LLMTRACEFX_TRUSTED_REPO_ROOT", str(_INSTALLED_PROJECT_ROOT))
).resolve()
_TRUSTED_BOOTSTRAP = _PROJECT_ROOT / "scripts" / "run-real-mlx-cache-audit-trusted.py"
DEFAULT_CONVERSION_SUMMARY = (
    _PROJECT_ROOT / "llmtracefx/cache_audit/data/qwen3-4b-conversion-summary.json"
)
EXPECTED_RUNTIME_PACKAGE_IDENTITIES = (
    _PROJECT_ROOT
    / "llmtracefx/cache_audit/data/apple-silicon-python313-mlx-lm-runtime-v1.json"
)
_CASES = (
    "cold-exact-duplicate",
    "interior-mutation",
    "allocation-step-mutation",
    "same-length-different-ids",
    "suffix-only-change",
    "namespace-isolation",
    "capacity-eviction",
)
_CASE_REQUEST_COUNTS = {
    "cold-exact-duplicate": 3,
    "interior-mutation": 2,
    "allocation-step-mutation": 2,
    "same-length-different-ids": 2,
    "suffix-only-change": 2,
    "namespace-isolation": 2,
    "capacity-eviction": 5,
}
_BLOCKS = tuple(f"{lane_id}:{case}" for case in _CASES for lane_id in LANE_IDS)
_BLOCK_REQUEST_COUNTS = {
    f"{lane_id}:{case}": count
    for lane_id in LANE_IDS
    for case, count in _CASE_REQUEST_COUNTS.items()
}
_REQUESTS_PER_LANE = sum(_CASE_REQUEST_COUNTS.values())
_LANE_CONTRACTS = {
    "1k": {"base": 1025, "eviction": 513},
    "4k": {"base": 4097, "eviction": 2049},
}
_EXPECTED_RUNTIME_IDENTITY = {
    "mlx": REQUIRED_MLX_VERSION,
    "mlx_lm": REQUIRED_MLX_LM_VERSION,
    "platform_machine": "arm64",
    "platform_system": "Darwin",
}
SANDBOX_POLICY = "(version 1) (allow default) (deny network*)"
SANDBOX_POLICY_DIGEST = (
    "sha256:" + hashlib.sha256(SANDBOX_POLICY.encode("ascii")).hexdigest()
)
_RUNTIME_DISTRIBUTION_VERSIONS = {
    "anyio": "4.9.0",
    "certifi": "2025.7.9",
    "click": "8.1.8",
    "filelock": "3.32.4",
    "fsspec": "2026.7.0",
    "h11": "0.16.0",
    "hf-xet": "1.6.0",
    "httpcore": "1.0.9",
    "httpx": "0.28.1",
    "huggingface-hub": "1.13.0",
    "idna": "3.15",
    "jinja2": "3.1.6",
    "markupsafe": "3.0.2",
    "markdown-it-py": "3.0.0",
    "mdurl": "0.1.2",
    "mlx": REQUIRED_MLX_VERSION,
    "mlx-lm": REQUIRED_MLX_LM_VERSION,
    "mlx-metal": "0.32.2",
    "numpy": REQUIRED_NUMPY_VERSION,
    "packaging": "26.3",
    "protobuf": "6.33.5",
    "pyyaml": "6.0.2",
    "pygments": "2.20.0",
    "regex": "2026.7.19",
    "rich": "14.0.0",
    "safetensors": REQUIRED_SAFETENSORS_VERSION,
    "sentencepiece": "0.2.2",
    "shellingham": "1.5.4",
    "sniffio": "1.3.1",
    "tokenizers": REQUIRED_TOKENIZERS_VERSION,
    "tqdm": "4.70.0",
    "transformers": REQUIRED_TRANSFORMERS_VERSION,
    "typer": "0.16.0",
    "typing-extensions": "4.14.1",
}
_RUNTIME_IMPORT_PACKAGES = (
    "anyio",
    "certifi",
    "click",
    "filelock",
    "fsspec",
    "h11",
    "hf_xet",
    "httpcore",
    "httpx",
    "huggingface_hub",
    "idna",
    "jinja2",
    "markupsafe",
    "markdown_it",
    "mdurl",
    "mlx",
    "mlx_lm",
    "mlx_metal",
    "numpy",
    "packaging",
    "google.protobuf",
    "yaml",
    "pygments",
    "regex",
    "rich",
    "safetensors",
    "sentencepiece",
    "shellingham",
    "sniffio",
    "tokenizers",
    "tqdm",
    "transformers",
    "typer",
    "typing_extensions",
)
_IMPORT_SHADOW_CANDIDATES = tuple(
    candidate
    for package in _RUNTIME_IMPORT_PACKAGES
    for candidate in (f"{package.split('.', 1)[0]}.py", package.split(".", 1)[0])
)
_STARTED_TERMINAL_REASONS = {
    "child_exit_nonzero",
    "child_timeout",
    "invalid_complete_artifact",
    "orphaned_child_process",
    "process_cleanup_failed",
    "source_validation_failed",
    "supervisor_aborted",
    "total_timeout",
}
_NOT_STARTED_TERMINAL_REASONS = {
    "child_launch_failed",
    "source_validation_failed",
    "supervisor_aborted_before_start",
    "total_timeout_before_start",
}
_POLICY_SUPERSEDING_REASONS = {
    "process_cleanup_failed",
    "source_validation_failed",
}
_ABORT_LATER_REASONS = {
    "orphaned_child_process",
    "process_cleanup_failed",
    "source_validation_failed",
    "total_timeout",
    "total_timeout_before_start",
}
_PRIVATE_REPLICATE_FILES = {
    "attempt.json",
    "environment.json",
    "stages.jsonl",
    "teardown.json",
    "workload.json",
    "bundle",
}
_PUBLIC_REPLICATE_FILES = {
    "attempt.json",
    "environment.json",
    "stages.jsonl",
    "teardown.json",
    "workload-binding.json",
    "bundle",
}
_FAILED_REPLICATE_FILES = {"attempt.json", "teardown.json"}
_AGGREGATE_FILES = {
    "experiment-contract.json",
    "environment.json",
    "replicate-index.json",
    "claim-matrix.json",
    "descriptive-summary.json",
    "results.json",
    "run-ledger.jsonl",
    "summary.json",
    "report.html",
    "reuse-alignment.svg",
    "timing-memory.svg",
    "teardown.json",
    "replicates",
    "SHA256SUMS",
}


class RealMLXExperimentError(RuntimeError):
    """Raised when any experiment gate cannot be established exactly."""


def _json_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("ascii")


def _digest_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _experiment_cache_config() -> CacheConfig:
    return CacheConfig(
        namespace_id="experiment-namespaces",
        cache_type="mlx_lru_prompt_cache",
        max_entries=MAX_CACHE_ENTRIES,
        max_bytes=MAX_CACHE_BYTES,
    )


def _write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, canonical_json(value))


def _json_line(value: Any) -> str:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    )


def _safe_object(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RealMLXExperimentError(f"{path.name} must be a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RealMLXExperimentError(f"invalid JSON in {path.name}") from exc
    if not isinstance(value, dict):
        raise RealMLXExperimentError(f"{path.name} must contain an object")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], context: str) -> None:
    if set(value) != keys:
        raise RealMLXExperimentError(f"{context} field allowlist mismatch")


def _integer_array(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise RealMLXExperimentError(f"{context} must be a non-empty integer array")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value
    ):
        raise RealMLXExperimentError(f"{context} contains an invalid token ID")
    return tuple(value)


@dataclass(frozen=True)
class FrozenMLXLane:
    """Exact private arrays for one explicitly identified workload lane."""

    lane_id: str
    base: tuple[int, ...]
    different_ids: tuple[int, ...]
    mutation_137: tuple[int, ...]
    mutation_256: tuple[int, ...]
    suffix_change: tuple[int, ...]
    eviction_a: tuple[int, ...]
    eviction_b: tuple[int, ...]
    eviction_c: tuple[int, ...]
    calibration_outputs: Mapping[str, tuple[int, ...]] | None = None

    def __post_init__(self) -> None:
        contract = _LANE_CONTRACTS.get(self.lane_id)
        if contract is None:
            raise RealMLXExperimentError("lane ID must be one of 1k or 4k")
        base_tokens = contract["base"]
        eviction_tokens = contract["eviction"]
        if len(self.base) != base_tokens:
            raise RealMLXExperimentError(
                f"{self.lane_id} base must contain exactly {base_tokens} tokens"
            )
        for name in (
            "different_ids",
            "mutation_137",
            "mutation_256",
            "suffix_change",
        ):
            if len(getattr(self, name)) != base_tokens:
                raise RealMLXExperimentError(
                    f"{self.lane_id} {name} must contain {base_tokens} tokens"
                )
        for name in ("eviction_a", "eviction_b", "eviction_c"):
            if len(getattr(self, name)) != eviction_tokens:
                raise RealMLXExperimentError(
                    f"{self.lane_id} {name} must contain {eviction_tokens} tokens"
                )
        if self.different_ids == self.base:
            raise RealMLXExperimentError(
                "same-length different-ID control is identical"
            )
        for position, name in ((137, "mutation_137"), (256, "mutation_256")):
            candidate = getattr(self, name)
            differences = [
                index
                for index, pair in enumerate(zip(self.base, candidate, strict=True))
                if pair[0] != pair[1]
            ]
            if differences != [position]:
                raise RealMLXExperimentError(
                    f"{name} must differ only at token index {position}"
                )
        differences = [
            index
            for index, pair in enumerate(
                zip(self.base, self.suffix_change, strict=True)
            )
            if pair[0] != pair[1]
        ]
        if differences != list(range(len(self.base) - 32, len(self.base) - 16)):
            raise RealMLXExperimentError(
                "suffix change must replace the 16 tokens immediately before "
                "the final 16 generation-template tokens"
            )
        if self.suffix_change[-16:] != self.base[-16:]:
            raise RealMLXExperimentError(
                "suffix change must preserve the final generation-template tokens"
            )
        if len({self.eviction_a, self.eviction_b, self.eviction_c}) != 3:
            raise RealMLXExperimentError("eviction A/B/C arrays must be distinct")
        if len({getattr(self, name) for name in CALIBRATION_ARRAY_NAMES}) != len(
            CALIBRATION_ARRAY_NAMES
        ):
            raise RealMLXExperimentError(
                "all calibrated prompt arrays must be distinct"
            )
        if self.calibration_outputs is not None:
            if set(self.calibration_outputs) != set(CALIBRATION_ARRAY_NAMES):
                raise RealMLXExperimentError(
                    "calibration outputs must cover every valid prompt array"
                )
            if any(
                len(output) != EXPECTED_CALIBRATION_OUTPUT_TOKENS[self.lane_id]
                for output in self.calibration_outputs.values()
            ):
                raise RealMLXExperimentError(
                    "every calibration output must contain exactly 3 tokens"
                )
            object.__setattr__(
                self,
                "calibration_outputs",
                MappingProxyType(dict(self.calibration_outputs)),
            )

    def to_dict(self) -> dict[str, Any]:
        arrays: dict[str, Any] = {
            name: list(getattr(self, name))
            for name in (
                "base",
                "different_ids",
                "mutation_137",
                "mutation_256",
                "suffix_change",
                "eviction_a",
                "eviction_b",
                "eviction_c",
            )
        }
        arrays["calibration_outputs"] = (
            None
            if self.calibration_outputs is None
            else {
                name: list(self.calibration_outputs[name])
                for name in CALIBRATION_ARRAY_NAMES
            }
        )
        return {
            "lane_id": self.lane_id,
            "arrays": arrays,
            "lane_digest": _digest_bytes(_json_bytes(arrays)),
        }

    @classmethod
    def from_dict(cls, value: Any, *, lane_id: str) -> FrozenMLXLane:
        if not isinstance(value, dict):
            raise RealMLXExperimentError(f"{lane_id} lane must be an object")
        _exact_keys(value, {"lane_id", "arrays", "lane_digest"}, f"{lane_id} lane")
        if value["lane_id"] != lane_id:
            raise RealMLXExperimentError("workload lane identity mismatch")
        arrays = value["arrays"]
        if not isinstance(arrays, dict):
            raise RealMLXExperimentError("workload arrays must be an object")
        names = {
            "base",
            "different_ids",
            "mutation_137",
            "mutation_256",
            "suffix_change",
            "eviction_a",
            "eviction_b",
            "eviction_c",
            "calibration_outputs",
        }
        _exact_keys(arrays, names, "workload arrays")
        calibration_raw = arrays["calibration_outputs"]
        if calibration_raw is not None and (
            not isinstance(calibration_raw, dict)
            or set(calibration_raw) != set(CALIBRATION_ARRAY_NAMES)
        ):
            raise RealMLXExperimentError(
                "calibration output mapping must cover every valid prompt array"
            )
        lane = cls(
            lane_id=lane_id,
            base=_integer_array(arrays["base"], "arrays.base"),
            different_ids=_integer_array(
                arrays["different_ids"], "arrays.different_ids"
            ),
            mutation_137=_integer_array(arrays["mutation_137"], "arrays.mutation_137"),
            mutation_256=_integer_array(arrays["mutation_256"], "arrays.mutation_256"),
            suffix_change=_integer_array(
                arrays["suffix_change"], "arrays.suffix_change"
            ),
            eviction_a=_integer_array(arrays["eviction_a"], "arrays.eviction_a"),
            eviction_b=_integer_array(arrays["eviction_b"], "arrays.eviction_b"),
            eviction_c=_integer_array(arrays["eviction_c"], "arrays.eviction_c"),
            calibration_outputs=(
                None
                if calibration_raw is None
                else {
                    name: _integer_array(
                        calibration_raw[name],
                        f"arrays.calibration_outputs.{name}",
                    )
                    for name in CALIBRATION_ARRAY_NAMES
                }
            ),
        )
        if value["lane_digest"] != lane.to_dict()["lane_digest"]:
            raise RealMLXExperimentError(f"{lane_id} lane digest mismatch")
        return lane


@dataclass(frozen=True)
class FrozenMLXWorkload:
    """Both exact private lanes used by every replicate."""

    lanes: tuple[FrozenMLXLane, ...]
    model_artifact_digest: str = EXPECTED_MODEL_ARTIFACT_DIGEST

    def __post_init__(self) -> None:
        if tuple(lane.lane_id for lane in self.lanes) != LANE_IDS:
            raise RealMLXExperimentError(
                "workload must contain ordered 1k and 4k lanes"
            )
        if self.model_artifact_digest != EXPECTED_MODEL_ARTIFACT_DIGEST:
            raise RealMLXExperimentError("workload model/tokenizer artifact mismatch")

    def lane(self, lane_id: str) -> FrozenMLXLane:
        try:
            return next(lane for lane in self.lanes if lane.lane_id == lane_id)
        except StopIteration as exc:
            raise RealMLXExperimentError(f"unknown workload lane: {lane_id}") from exc

    def to_dict(self) -> dict[str, Any]:
        lanes = {lane.lane_id: lane.to_dict() for lane in self.lanes}
        return {
            "schema_version": WORKLOAD_SCHEMA_VERSION,
            "tokenizer_contract": {
                "chat_template": True,
                "enable_thinking": False,
                "content_encoding": "project-authored-ascii",
                "final_instruction": "CACHE_OK",
                "model_artifact_digest": self.model_artifact_digest,
            },
            "lanes": lanes,
            "workload_digest": _digest_bytes(_json_bytes(lanes)),
        }

    @classmethod
    def from_dict(cls, value: Any) -> FrozenMLXWorkload:
        if not isinstance(value, dict):
            raise RealMLXExperimentError("workload must be an object")
        _exact_keys(
            value,
            {"schema_version", "tokenizer_contract", "lanes", "workload_digest"},
            "workload",
        )
        if value["schema_version"] != WORKLOAD_SCHEMA_VERSION:
            raise RealMLXExperimentError("unsupported real MLX workload schema")
        if value["tokenizer_contract"] != {
            "chat_template": True,
            "enable_thinking": False,
            "content_encoding": "project-authored-ascii",
            "final_instruction": "CACHE_OK",
            "model_artifact_digest": EXPECTED_MODEL_ARTIFACT_DIGEST,
        }:
            raise RealMLXExperimentError("workload tokenizer contract mismatch")
        lanes = value["lanes"]
        if not isinstance(lanes, dict) or tuple(lanes) != LANE_IDS:
            raise RealMLXExperimentError("workload lane allowlist mismatch")
        workload = cls(
            lanes=tuple(
                FrozenMLXLane.from_dict(lanes[lane_id], lane_id=lane_id)
                for lane_id in LANE_IDS
            )
        )
        if value["workload_digest"] != workload.to_dict()["workload_digest"]:
            raise RealMLXExperimentError("workload digest mismatch")
        return workload


def load_workload(path: Path, *, calibrated: bool | None = None) -> FrozenMLXWorkload:
    workload = FrozenMLXWorkload.from_dict(_safe_object(path))
    if calibrated is True and any(
        lane.calibration_outputs is None for lane in workload.lanes
    ):
        raise RealMLXExperimentError("workload has not been calibrated")
    if calibrated is True and (
        EXPECTED_CALIBRATED_WORKLOAD_DIGEST == "RECALIBRATION_REQUIRED"
        or any(
            digest == "RECALIBRATION_REQUIRED"
            for digest in EXPECTED_CALIBRATED_LANE_DIGESTS.values()
        )
    ):
        raise RealMLXExperimentError(
            "calibrated workload digest placeholders require recalibration"
        )
    if calibrated is True and (
        workload.to_dict()["workload_digest"] != EXPECTED_CALIBRATED_WORKLOAD_DIGEST
        or any(
            lane.to_dict()["lane_digest"]
            != EXPECTED_CALIBRATED_LANE_DIGESTS[lane.lane_id]
            or any(
                len(output) != EXPECTED_CALIBRATION_OUTPUT_TOKENS[lane.lane_id]
                for output in (lane.calibration_outputs or {}).values()
            )
            for lane in workload.lanes
        )
    ):
        raise RealMLXExperimentError("calibrated workload contract mismatch")
    if calibrated is False and any(
        lane.calibration_outputs is not None for lane in workload.lanes
    ):
        raise RealMLXExperimentError("compile output is already calibrated")
    return workload


def _template_messages(
    tokenizer: Any, messages: Sequence[Mapping[str, str]]
) -> tuple[int, ...]:
    if any(not value.isascii() for message in messages for value in message.values()):
        raise RealMLXExperimentError("synthetic workload text must be ASCII")
    try:
        value = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception as exc:
        raise RealMLXExperimentError("local tokenizer chat-template failed") from exc
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if (
        not isinstance(value, list)
        or not value
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise RealMLXExperimentError("chat template returned invalid token IDs")
    return tuple(value)


def _template_tokens(tokenizer: Any, body: str) -> tuple[int, ...]:
    return _template_messages(tokenizer, ({"role": "user", "content": body},))


def _exact_prompt_with_body(
    tokenizer: Any,
    *,
    target: int,
    label: str,
    final_instruction: str = "Answer exactly CACHE_OK.",
) -> tuple[tuple[int, ...], str]:
    prefix = (
        "LLMTraceFX public synthetic cache experiment. "
        f"Variant {label}. Ignore padding and follow the final instruction."
    )
    for count in range(target * 2):
        body = (
            prefix
            + " "
            + final_instruction
            + " The remaining cache words are inert padding."
            + (" cache" * count)
        )
        tokens = _template_tokens(tokenizer, body)
        if len(tokens) == target:
            return tokens, body
        if len(tokens) > target + 8 and count > target:
            break
    raise RealMLXExperimentError(
        f"local tokenizer could not compile exact {target}-token {label} prompt"
    )


def _exact_prompt(
    tokenizer: Any,
    *,
    target: int,
    label: str,
    final_instruction: str = "Answer exactly CACHE_OK.",
) -> tuple[int, ...]:
    return _exact_prompt_with_body(
        tokenizer,
        target=target,
        label=label,
        final_instruction=final_instruction,
    )[0]


def _replacement_token(tokenizer: Any, original: int) -> int:
    for text in (" X", " Y", " zero", " one", " cache", " audit"):
        try:
            value = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            value = tokenizer.encode(text)
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list):
            for token in value:
                if (
                    isinstance(token, int)
                    and not isinstance(token, bool)
                    and token != original
                ):
                    return token
    raise RealMLXExperimentError("tokenizer supplied no safe mutation token")


def compile_workload(tokenizer: Any) -> FrozenMLXWorkload:
    """Compile the exact private workload without loading model weights."""

    lanes: list[FrozenMLXLane] = []
    for lane_id in LANE_IDS:
        contract = _LANE_CONTRACTS[lane_id]
        base = _exact_prompt(
            tokenizer, target=contract["base"], label=f"{lane_id}-BASE"
        )
        different = _exact_prompt(
            tokenizer, target=contract["base"], label=f"{lane_id}-DIFFERENT"
        )
        if different == base:
            raise RealMLXExperimentError("different prompt tokenized identically")
        mutations: dict[int, tuple[int, ...]] = {}
        for position in (137, 256):
            changed = list(base)
            changed[position] = _replacement_token(tokenizer, changed[position])
            mutations[position] = tuple(changed)
        suffix_start = len(base) - 32
        suffix_end = len(base) - 16
        suffix = (
            base[:suffix_start]
            + tuple(
                _replacement_token(tokenizer, token)
                for token in base[suffix_start:suffix_end]
            )
            + base[suffix_end:]
        )
        eviction = tuple(
            _exact_prompt(
                tokenizer,
                target=contract["eviction"],
                label=f"{lane_id}-EVICTION-{label}",
            )
            for label in ("A", "B", "C")
        )
        lanes.append(
            FrozenMLXLane(
                lane_id=lane_id,
                base=base,
                different_ids=different,
                mutation_137=mutations[137],
                mutation_256=mutations[256],
                suffix_change=suffix,
                eviction_a=eviction[0],
                eviction_b=eviction[1],
                eviction_c=eviction[2],
            )
        )
    return FrozenMLXWorkload(
        lanes=tuple(lanes),
    )


def write_compiled_workload(path: Path, tokenizer: Any) -> FrozenMLXWorkload:
    if path.exists():
        raise RealMLXExperimentError("workload output already exists")
    workload = compile_workload(tokenizer)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, workload.to_dict())
    return workload


def output_is_cache_ok(tokenizer: Any, token_ids: Sequence[int]) -> bool:
    """Accept only normalized exact ``CACHE_OK`` output."""

    try:
        text = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    except TypeError:
        text = tokenizer.decode(list(token_ids))
    return isinstance(text, str) and text.strip() == "CACHE_OK"


def calibrate_outputs(
    workload: FrozenMLXWorkload,
    adapter_factory: Callable[[], MLXLocalCacheAdapter],
) -> FrozenMLXWorkload:
    """Calibrate every valid prompt array and repeat each base in a fresh cache."""

    if any(lane.calibration_outputs is not None for lane in workload.lanes):
        raise RealMLXExperimentError("workload is already calibrated")
    calibrated: list[FrozenMLXLane] = []
    for lane in workload.lanes:
        outputs: dict[str, tuple[int, ...]] = {}
        base_repeat: tuple[int, ...] | None = None
        calibration_items = tuple(
            (name, getattr(lane, name)) for name in CALIBRATION_ARRAY_NAMES
        ) + (("base-repeat", lane.base),)
        for index, (name, tokens) in enumerate(calibration_items):
            request = RequestSpec(
                request_id=f"{lane.lane_id}:calibration:{name}",
                scenario=ScenarioKind.COLD,
                order=0,
                input_token_ids=tokens,
                input_token_count=len(tokens),
                output_tokens=MAX_OUTPUT_TOKENS,
                replicate_id="calibration",
            )
            adapter = adapter_factory()
            records = adapter.run((request,))
            del adapter
            if len(records) != 1:
                raise RealMLXExperimentError(
                    f"{lane.lane_id} calibration returned invalid record count"
                )
            record = records[0]
            output = record.output.output_token_ids
            if (
                output is None
                or len(output) != EXPECTED_CALIBRATION_OUTPUT_TOKENS[lane.lane_id]
                or record.output.baseline_token_ids != output
                or record.output.token_identity.value is not True
                or record.output.correctness.value is not True
                or record.terminal_state.value != "completed"
            ):
                raise RealMLXExperimentError(
                    f"{lane.lane_id} calibration failed exact "
                    "CACHE_OK correctness gate"
                )
            if index == len(calibration_items) - 1:
                base_repeat = output
            else:
                outputs[name] = output
        if outputs["base"] != base_repeat:
            raise RealMLXExperimentError(
                f"{lane.lane_id} calibration is not exactly repeatable"
            )
        calibrated.append(replace(lane, calibration_outputs=outputs))
    return replace(workload, lanes=tuple(calibrated))


def _request(
    items: list[RequestSpec],
    *,
    block: str,
    name: str,
    scenario: ScenarioKind,
    tokens: tuple[int, ...],
    replicate_id: str,
    namespace: str = "experiment",
    predecessors: tuple[str, ...] = (),
    mutation_position: int | None = None,
    pair_id: str | None = None,
    pair_role: PairRole = PairRole.SINGLE,
) -> None:
    items.append(
        RequestSpec(
            request_id=f"{block}:{name}",
            scenario=scenario,
            order=len(items),
            input_token_ids=tokens,
            input_token_count=len(tokens),
            output_tokens=MAX_OUTPUT_TOKENS,
            pair_id=pair_id,
            pair_role=pair_role,
            mutation_position=mutation_position,
            expected_predecessors=predecessors,
            namespace_id=namespace,
            replicate_id=replicate_id,
        )
    )


def block_schedule(replicate_id: str) -> tuple[str, ...]:
    if replicate_id not in REPLICATE_IDS:
        raise RealMLXExperimentError("replicate ID must be one of replicate-0..5")
    offset, step = SCHEDULE_AFFINE_PERMUTATIONS[REPLICATE_IDS.index(replicate_id)]
    schedule = tuple(
        _BLOCKS[(offset + step * index) % len(_BLOCKS)] for index in range(len(_BLOCKS))
    )
    if len(schedule) != len(_BLOCKS) or set(schedule) != set(_BLOCKS):
        raise RealMLXExperimentError("replicate schedule is not a full permutation")
    return schedule


def _request_block_id(request_id: str) -> str:
    block, separator, _ = request_id.rpartition(":")
    if not separator or block not in _BLOCK_REQUEST_COUNTS:
        raise RealMLXExperimentError("request ID has no valid lane/block identity")
    return block


def _lane_request_counts(
    records: Sequence[RequestEvidence | RequestSpec],
) -> dict[str, int]:
    counts = dict.fromkeys(LANE_IDS, 0)
    for record in records:
        spec = record.spec if isinstance(record, RequestEvidence) else record
        lane_id = _request_block_id(spec.request_id).split(":", 1)[0]
        counts[lane_id] += 1
    return counts


def _scheduled_record_lanes(replicate_id: str) -> tuple[str, ...]:
    return tuple(
        block.split(":", 1)[0]
        for block in block_schedule(replicate_id)
        for _ in range(_BLOCK_REQUEST_COUNTS[block])
    )


def requests_for_replicate(
    workload: FrozenMLXWorkload, replicate_id: str
) -> tuple[RequestSpec, ...]:
    """Build exact requests in the fixed counterbalanced block rotation."""

    if any(lane.calibration_outputs is None for lane in workload.lanes):
        raise RealMLXExperimentError("measurement requires calibration output")
    by_block: dict[str, list[RequestSpec]] = {}
    for block in _BLOCKS:
        lane_id, case = block.split(":", 1)
        lane = workload.lane(lane_id)
        items: list[RequestSpec] = []
        pair = f"{block}:pair"
        if case == "cold-exact-duplicate":
            _request(
                items,
                block=block,
                name="cold",
                scenario=ScenarioKind.COLD,
                tokens=lane.base,
                replicate_id=replicate_id,
                pair_id=pair,
                pair_role=PairRole.CONTROL,
            )
            _request(
                items,
                block=block,
                name="exact",
                scenario=ScenarioKind.IDENTICAL_PREFIX,
                tokens=lane.base,
                replicate_id=replicate_id,
                predecessors=(f"{block}:cold",),
                pair_id=pair,
                pair_role=PairRole.TREATMENT,
            )
            _request(
                items,
                block=block,
                name="duplicate",
                scenario=ScenarioKind.DUPLICATE,
                tokens=lane.base,
                replicate_id=replicate_id,
                predecessors=(f"{block}:exact",),
            )
        elif case in {
            "interior-mutation",
            "allocation-step-mutation",
            "same-length-different-ids",
            "suffix-only-change",
        }:
            variants = {
                "interior-mutation": (
                    ScenarioKind.WITHIN_BLOCK_MUTATION,
                    lane.mutation_137,
                    137,
                ),
                "allocation-step-mutation": (
                    ScenarioKind.BLOCK_BOUNDARY_MUTATION,
                    lane.mutation_256,
                    256,
                ),
                "same-length-different-ids": (
                    ScenarioKind.SAME_LENGTH_DIFFERENT_IDS,
                    lane.different_ids,
                    None,
                ),
                "suffix-only-change": (
                    ScenarioKind.SUFFIX_CHANGE,
                    lane.suffix_change,
                    len(lane.base) - 32,
                ),
            }
            scenario, variant, position = variants[case]
            _request(
                items,
                block=block,
                name="seed",
                scenario=ScenarioKind.COLD,
                tokens=lane.base,
                replicate_id=replicate_id,
                pair_id=pair,
                pair_role=PairRole.CONTROL,
            )
            _request(
                items,
                block=block,
                name="variant",
                scenario=scenario,
                tokens=variant,
                replicate_id=replicate_id,
                predecessors=(f"{block}:seed",),
                mutation_position=position,
                pair_id=pair,
                pair_role=PairRole.TREATMENT,
            )
        elif case == "namespace-isolation":
            _request(
                items,
                block=block,
                name="tenant-a",
                scenario=ScenarioKind.COLD,
                tokens=lane.base,
                replicate_id=replicate_id,
                namespace="tenant-a",
                pair_id=pair,
                pair_role=PairRole.CONTROL,
            )
            _request(
                items,
                block=block,
                name="tenant-b",
                scenario=ScenarioKind.NAMESPACE_ISOLATION,
                tokens=lane.base,
                replicate_id=replicate_id,
                namespace="tenant-b",
                predecessors=(f"{block}:tenant-a",),
                pair_id=pair,
                pair_role=PairRole.TREATMENT,
            )
        else:
            for name, scenario, tokens, namespace, predecessors, role in (
                (
                    "a-seed",
                    ScenarioKind.COLD,
                    lane.eviction_a,
                    f"{lane_id}-eviction-a",
                    (),
                    PairRole.CONTROL,
                ),
                (
                    "a-hit",
                    ScenarioKind.IDENTICAL_PREFIX,
                    lane.eviction_a,
                    f"{lane_id}-eviction-a",
                    (f"{block}:a-seed",),
                    PairRole.SINGLE,
                ),
                (
                    "b",
                    ScenarioKind.COLD,
                    lane.eviction_b,
                    f"{lane_id}-eviction-b",
                    (),
                    PairRole.SINGLE,
                ),
                (
                    "c",
                    ScenarioKind.COLD,
                    lane.eviction_c,
                    f"{lane_id}-eviction-c",
                    (),
                    PairRole.SINGLE,
                ),
                (
                    "a-miss",
                    ScenarioKind.EVICTION_COUNT,
                    lane.eviction_a,
                    f"{lane_id}-eviction-a",
                    (f"{block}:a-hit",),
                    PairRole.TREATMENT,
                ),
            ):
                _request(
                    items,
                    block=block,
                    name=name,
                    scenario=scenario,
                    tokens=tokens,
                    replicate_id=replicate_id,
                    namespace=namespace,
                    predecessors=predecessors,
                    pair_id=pair if role is not PairRole.SINGLE else None,
                    pair_role=role,
                )
        by_block[block] = items

    ordered: list[RequestSpec] = []
    for block in block_schedule(replicate_id):
        for request in by_block[block]:
            ordered.append(replace(request, order=len(ordered)))
    requests = tuple(ordered)
    _verify_request_schedule(requests, workload, replicate_id)
    return requests


def _verify_request_schedule(
    requests: Sequence[RequestSpec],
    workload: FrozenMLXWorkload,
    replicate_id: str,
) -> None:
    schedule = block_schedule(replicate_id)
    if (
        len(requests) != sum(_BLOCK_REQUEST_COUNTS.values())
        or tuple(request.order for request in requests) != tuple(range(len(requests)))
        or tuple(
            dict.fromkeys(_request_block_id(request.request_id) for request in requests)
        )
        != schedule
        or any(request.replicate_id != replicate_id for request in requests)
        or _lane_request_counts(requests) != dict.fromkeys(LANE_IDS, _REQUESTS_PER_LANE)
    ):
        raise RealMLXExperimentError("replicate request schedule invariant failed")
    for lane_id in LANE_IDS:
        lane = workload.lane(lane_id)
        by_id = {request.request_id: request for request in requests}
        suffix = by_id[f"{lane_id}:suffix-only-change:variant"]
        if (
            suffix.input_token_ids != lane.suffix_change
            or suffix.mutation_position != len(lane.base) - 32
        ):
            raise RealMLXExperimentError("suffix mutation schedule invariant failed")
        eviction = {
            name: by_id[f"{lane_id}:capacity-eviction:{name}"]
            for name in ("a-seed", "a-hit", "b", "c", "a-miss")
        }
        if (
            len(
                {
                    eviction["a-seed"].namespace_id,
                    eviction["b"].namespace_id,
                    eviction["c"].namespace_id,
                }
            )
            != 3
            or eviction["a-seed"].namespace_id != eviction["a-hit"].namespace_id
            or eviction["a-seed"].namespace_id != eviction["a-miss"].namespace_id
        ):
            raise RealMLXExperimentError("eviction namespace schedule invariant failed")


def _schedule_shape_specs(replicate_id: str, *, public: bool) -> list[dict[str, Any]]:
    lanes: list[FrozenMLXLane] = []
    for lane_id in LANE_IDS:
        contract = _LANE_CONTRACTS[lane_id]
        base = (1,) * contract["base"]
        different = (2,) + base[1:]
        mutation_137 = base[:137] + (2,) + base[138:]
        mutation_256 = base[:256] + (2,) + base[257:]
        suffix = base[:-32] + (2,) * 16 + base[-16:]
        lanes.append(
            FrozenMLXLane(
                lane_id=lane_id,
                base=base,
                different_ids=different,
                mutation_137=mutation_137,
                mutation_256=mutation_256,
                suffix_change=suffix,
                eviction_a=(3,) * contract["eviction"],
                eviction_b=(4,) * contract["eviction"],
                eviction_c=(5,) * contract["eviction"],
                calibration_outputs=dict.fromkeys(
                    CALIBRATION_ARRAY_NAMES,
                    (6,) * EXPECTED_CALIBRATION_OUTPUT_TOKENS[lane_id],
                ),
            )
        )
    workload = FrozenMLXWorkload(
        lanes=tuple(lanes),
    )
    specs = requests_for_replicate(workload, replicate_id)
    if not public:
        return [spec.to_dict(include_tokens=False) for spec in specs]
    request_ids = {
        spec.request_id: f"request-{index:04d}" for index, spec in enumerate(specs)
    }
    pair_ids = {
        pair_id: f"pair-{index:04d}"
        for index, pair_id in enumerate(
            dict.fromkeys(spec.pair_id for spec in specs if spec.pair_id is not None)
        )
    }
    namespace_ids = {
        namespace_id: f"namespace-{index:04d}"
        for index, namespace_id in enumerate(
            dict.fromkeys(spec.namespace_id for spec in specs)
        )
    }
    return [
        replace(
            spec,
            request_id=request_ids[spec.request_id],
            input_token_ids=None,
            pair_id=None if spec.pair_id is None else pair_ids[spec.pair_id],
            expected_predecessors=tuple(
                request_ids[predecessor] for predecessor in spec.expected_predecessors
            ),
            namespace_id=namespace_ids[spec.namespace_id],
            replicate_id=f"replicate-{index:04d}",
        ).to_dict(include_tokens=False)
        for index, spec in enumerate(specs)
    ]


class LifecycleMLXAdapter:
    """Composite adapter with one fresh LRU cache per experimental block."""

    backend = "mlx_lm_local"

    def __init__(
        self,
        *,
        runtime_factory: Callable[[], MLXCacheRuntime],
        model: Any,
        tokenizer: Any,
        model_key: Any,
        model_artifact_digest: str,
        correctness_evaluator: Callable[[tuple[int, ...]], bool],
        stage_observer: Callable[[MLXStageObservation], None],
    ) -> None:
        self._runtime_factory = runtime_factory
        self._model = model
        self._tokenizer = tokenizer
        self._model_key = model_key
        self._digest = model_artifact_digest
        self._correctness = correctness_evaluator
        self._stage_observer = stage_observer
        probe = runtime_factory()
        self._capability = check_mlx_capabilities(probe, backend=self.backend)
        self._identity = AdapterAuditIdentity(
            backend_version=REQUIRED_MLX_LM_VERSION,
            runtime_identity={
                "mlx": probe.mlx_version or "unavailable",
                "mlx_lm": probe.mlx_lm_version or "unavailable",
                "platform_machine": probe.platform_machine,
                "platform_system": probe.platform_system,
            },
            model_artifact_digest=model_artifact_digest,
            cache_type="mlx_lru_prompt_cache",
            max_entries=MAX_CACHE_ENTRIES,
            max_bytes=MAX_CACHE_BYTES,
        )

    def capabilities(self) -> CacheAuditCapability:
        return self._capability

    def audit_identity(self) -> AdapterAuditIdentity:
        return self._identity

    def run(self, requests: Sequence[RequestSpec]) -> list[RequestEvidence]:
        records: list[RequestEvidence] = []
        for _, grouped in groupby(
            requests, key=lambda request: _request_block_id(request.request_id)
        ):
            adapter = MLXLocalCacheAdapter(
                runtime=self._runtime_factory(),
                model=self._model,
                tokenizer=self._tokenizer,
                model_key=self._model_key,
                model_artifact_digest=self._digest,
                max_cache_entries=MAX_CACHE_ENTRIES,
                max_cache_bytes=MAX_CACHE_BYTES,
                correctness_evaluator=self._correctness,
                stage_observer=self._stage_observer,
            )
            records.extend(adapter.run(tuple(grouped)))
        return records


def _current_rss_bytes() -> int | None:
    if platform.system() != "Darwin":
        return None

    class _TimeValue(ctypes.Structure):
        _pack_ = 4
        _fields_ = [
            ("seconds", ctypes.c_int32),
            ("microseconds", ctypes.c_int32),
        ]

    class _MachTaskBasicInfo(ctypes.Structure):
        _fields_ = [
            ("virtual_size", ctypes.c_uint64),
            ("resident_size", ctypes.c_uint64),
            ("resident_size_max", ctypes.c_uint64),
            ("user_time", _TimeValue),
            ("system_time", _TimeValue),
            ("policy", ctypes.c_int32),
            ("suspend_count", ctypes.c_int32),
        ]

    try:
        libsystem = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        mach_task_self = libsystem.mach_task_self
        mach_task_self.argtypes = []
        mach_task_self.restype = ctypes.c_uint32
        task_info = libsystem.task_info
        task_info.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        task_info.restype = ctypes.c_int
        info = _MachTaskBasicInfo()
        count = ctypes.c_uint32(
            ctypes.sizeof(_MachTaskBasicInfo) // ctypes.sizeof(ctypes.c_uint32)
        )
        status = task_info(
            mach_task_self(),
            20,  # MACH_TASK_BASIC_INFO.
            ctypes.cast(ctypes.byref(info), ctypes.POINTER(ctypes.c_int32)),
            ctypes.byref(count),
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    expected_count = ctypes.sizeof(_MachTaskBasicInfo) // ctypes.sizeof(ctypes.c_uint32)
    if status != 0 or count.value != expected_count or info.resident_size <= 0:
        return None
    return int(info.resident_size)


def _system_swap_used_bytes() -> int | None:
    result = subprocess.run(
        ["/usr/sbin/sysctl", "-n", "vm.swapusage"],
        capture_output=True,
        check=False,
        text=True,
    )
    match = re.search(r"\bused\s*=\s*([0-9.]+)([MG])", result.stdout)
    if result.returncode != 0 or match is None:
        return None
    scale = 1024**2 if match.group(2) == "M" else 1024**3
    return int(float(match.group(1)) * scale)


def _system_memory_free_percent() -> float | None:
    result = subprocess.run(
        ["/usr/bin/memory_pressure"],
        capture_output=True,
        check=False,
        text=True,
    )
    match = re.search(
        r"System-wide memory free percentage:\s*([0-9.]+)%",
        result.stdout,
    )
    if result.returncode != 0 or match is None:
        return None
    return float(match.group(1))


class StageRecorder:
    """Write privacy-safe, canonical stage records."""

    def __init__(
        self,
        replicate_id: str,
        *,
        rss_reader: Callable[[], int | None] = _current_rss_bytes,
        swap_reader: Callable[[], int | None] = _system_swap_used_bytes,
        memory_reader: Callable[[], float | None] = _system_memory_free_percent,
    ) -> None:
        if replicate_id not in REPLICATE_IDS and replicate_id != "calibration":
            raise RealMLXExperimentError("invalid stage replicate ID")
        self._replicate_id = replicate_id
        self._rss_reader = rss_reader
        self._swap_reader = swap_reader
        self._memory_reader = memory_reader
        self._swap_baseline = swap_reader()
        self.rows: list[dict[str, Any]] = []

    def __call__(self, observation: MLXStageObservation) -> None:
        rss = self._rss_reader()
        swap = self._swap_reader()
        memory_free = self._memory_reader()
        if observation.peak_bytes > MAX_ALLOCATOR_PEAK_BYTES:
            raise RealMLXExperimentError("MLX allocator peak safety gate exceeded")
        if rss is None:
            raise RealMLXExperimentError("replicate RSS could not be measured")
        if rss > MAX_PROCESS_RSS_BYTES:
            raise RealMLXExperimentError("replicate RSS safety gate exceeded")
        if swap is None:
            raise RealMLXExperimentError("system swap could not be measured")
        if swap > MAX_SWAP_BYTES:
            raise RealMLXExperimentError("system swap safety gate exceeded")
        if memory_free is None:
            raise RealMLXExperimentError("system memory pressure could not be measured")
        if memory_free < MIN_RUNTIME_MEMORY_FREE_PERCENT:
            raise RealMLXExperimentError("system memory free safety gate exceeded")
        if (
            swap is not None
            and self._swap_baseline is not None
            and swap - self._swap_baseline > MAX_SWAP_GROWTH_BYTES
        ):
            raise RealMLXExperimentError("system swap growth safety gate exceeded")
        self.rows.append(
            {
                "schema_version": "1",
                "replicate_id": self._replicate_id,
                "request_id": observation.request_id,
                "stage": observation.stage,
                "allocator": {
                    "active_bytes": observation.active_bytes,
                    "cache_bytes": observation.allocator_cache_bytes,
                    "peak_bytes": observation.peak_bytes,
                    "scope": "mlx_process_global_allocator",
                },
                "logical_cache": {
                    "bytes": observation.logical_cache_bytes,
                    "scope": "current_lru_entry_when_observable",
                },
                "process_rss": {
                    "bytes": rss,
                    "scope": "current_replicate_child_process_only",
                },
                "system_swap": {
                    "used_bytes": swap,
                    "scope": "system_wide",
                },
                "system_memory": {
                    "free_percent": memory_free,
                    "scope": "system_wide_memory_pressure",
                },
                "thermal_power": {
                    "thermal_state": None,
                    "power_watts": None,
                    "scope": "unavailable_without_safe_collector",
                },
            }
        )

    def write(self, path: Path) -> None:
        text = "".join(_json_line(row) for row in self.rows)
        atomic_write_text(path, text)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _run_instance_id() -> str:
    instance_id = os.environ.get(RUN_INSTANCE_ENV) or secrets.token_hex(16)
    if re.fullmatch(r"[0-9a-f]{32}", instance_id) is None:
        raise RealMLXExperimentError("run instance ID is invalid")
    return instance_id


def _safe_environment(*, instance_id: str | None = None) -> dict[str, Any]:
    environment = {
        "schema_version": "1",
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "os_release": platform.release(),
        "python": platform.python_version(),
        "mlx": _distribution_version("mlx"),
        "mlx_lm": _distribution_version("mlx-lm"),
        "process_scope": "one_fresh_replicate_child",
    }
    if instance_id is not None:
        environment["run_instance_id"] = instance_id
    return environment


def _common_environment_value(environment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in environment.items() if key != "run_instance_id"
    }


def _distribution_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _runtime_distribution(name: str) -> metadata.Distribution:
    try:
        return metadata.distribution(name)
    except metadata.PackageNotFoundError as exc:
        raise RealMLXExperimentError(
            f"{name} runtime package distribution is unavailable"
        ) from exc


def _regular_file_identity(path: Path) -> tuple[int, str]:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path.absolute()
    ):
        raise RealMLXExperimentError("runtime package tree contains an unsafe file")
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RealMLXExperimentError("runtime package tree contains an unsafe file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.lstat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise RealMLXExperimentError("runtime package file changed while hashing")
    return before.st_size, "sha256:" + digest.hexdigest()


def _trusted_site_roots() -> tuple[tuple[str, Path, Path], ...]:
    executable = Path(os.path.abspath(sys.executable))
    if executable.parent.name != "bin":
        raise RealMLXExperimentError("trusted runtime venv is unavailable")
    venv_root = executable.parent.parent
    site_root = (
        venv_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    try:
        resolved_venv = venv_root.resolve(strict=True)
        resolved_site = site_root.resolve(strict=True)
        scripts_root = (resolved_venv / "bin").resolve(strict=True)
    except OSError as exc:
        raise RealMLXExperimentError("trusted site-packages is unavailable") from exc
    if (
        venv_root.is_symlink()
        or site_root.is_symlink()
        or scripts_root.is_symlink()
        or resolved_venv != venv_root.absolute()
        or resolved_site != site_root.absolute()
        or not resolved_site.is_dir()
        or not scripts_root.is_dir()
        or not _is_relative_to(resolved_site, resolved_venv)
    ):
        raise RealMLXExperimentError("trusted site-packages is unavailable")
    return (("purelib", resolved_site, scripts_root),)


def _normalized_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _reject_site_startup_artifacts(trusted_root: Path) -> None:
    for path in trusted_root.rglob("*"):
        if path.is_symlink():
            raise RealMLXExperimentError(
                "runtime site-packages contains an unsafe symlink"
            )
        if (
            path.name == "__pycache__"
            or path.name in {"sitecustomize.py", "usercustomize.py"}
            or path.suffix.lower() in {".pth", ".pyc", ".pyo"}
        ):
            raise RealMLXExperimentError(
                "runtime site-packages contains a startup hook or bytecode"
            )


def _validate_expected_distribution_uniqueness(trusted_root: Path) -> None:
    counts: Counter[str] = Counter()
    for distribution in metadata.distributions(path=[str(trusted_root)]):
        installed_name = distribution.metadata["Name"]
        if isinstance(installed_name, str):
            normalized = _normalized_distribution_name(installed_name)
            if normalized in _RUNTIME_DISTRIBUTION_VERSIONS:
                counts[normalized] += 1
    if any(counts[name] != 1 for name in _RUNTIME_DISTRIBUTION_VERSIONS):
        raise RealMLXExperimentError(
            "runtime package closure has missing or duplicate distributions"
        )


def _distribution_tree_identity(
    distribution: metadata.Distribution,
    *,
    distribution_name: str,
    trusted_root: Path,
    trusted_scripts_root: Path,
    output_workspace: Path,
) -> tuple[int, int, str]:
    declared = distribution.files
    if declared is None or not declared:
        raise RealMLXExperimentError(
            f"{distribution_name} runtime distribution files are unavailable"
        )
    files: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for item in declared:
        declared_path = Path(str(item))
        if (
            declared_path.is_absolute()
            or not declared_path.parts
            or declared_path.suffix.lower() in {".pyc", ".pyo"}
            or "__pycache__" in declared_path.parts
        ):
            raise RealMLXExperimentError(
                f"{distribution_name} runtime distribution declares unsafe bytecode or path"
            )
        located = Path(os.path.abspath(str(distribution.locate_file(item))))
        if not _is_relative_to(located, trusted_root):
            if _is_relative_to(located, trusted_scripts_root):
                continue
            raise RealMLXExperimentError(
                f"{distribution_name} runtime distribution file is out of root"
            )
        relative = located.relative_to(trusted_root)
        if relative.name == "RECORD" and relative.parent.name.endswith(".dist-info"):
            continue
        try:
            resolved = located.resolve(strict=True)
        except OSError as exc:
            raise RealMLXExperimentError(
                f"{distribution_name} runtime distribution contains a missing file"
            ) from exc
        if resolved != located:
            raise RealMLXExperimentError(
                f"{distribution_name} runtime distribution contains a symlink"
            )
        if not _is_relative_to(resolved, trusted_root):
            raise RealMLXExperimentError(
                f"{distribution_name} runtime distribution file is out of root"
            )
        logical_path = "site-packages/" + resolved.relative_to(trusted_root).as_posix()
        if logical_path in seen or _is_relative_to(resolved, output_workspace.parent):
            raise RealMLXExperimentError(
                f"{distribution_name} runtime distribution contains an unsafe file"
            )
        seen.add(logical_path)
        files.append((logical_path, resolved))

    tree_digest = hashlib.sha256()
    total_bytes = 0
    for logical_path, path in sorted(files):
        size, digest = _regular_file_identity(path)
        total_bytes += size
        tree_digest.update(
            _json_bytes(
                {
                    "path": logical_path,
                    "size": size,
                    "sha256": digest,
                }
            )
        )
        tree_digest.update(b"\n")
    return len(files), total_bytes, "sha256:" + tree_digest.hexdigest()


def _compute_runtime_package_identity(
    output_workspace: Path,
) -> dict[str, dict[str, str | int]]:
    site_roots = _trusted_site_roots()
    for _, root, _ in site_roots:
        _reject_site_startup_artifacts(root)
        _validate_expected_distribution_uniqueness(root)
    identities: dict[str, dict[str, str | int]] = {}
    for distribution_name, required_version in _RUNTIME_DISTRIBUTION_VERSIONS.items():
        distribution = _runtime_distribution(distribution_name)
        installed_name = distribution.metadata["Name"]
        if (
            not isinstance(installed_name, str)
            or _normalized_distribution_name(installed_name) != distribution_name
            or distribution.version != required_version
        ):
            raise RealMLXExperimentError(
                f"{distribution_name} runtime distribution identity mismatch"
            )
        try:
            distribution_root = Path(str(distribution.locate_file(""))).resolve(
                strict=True
            )
        except OSError as exc:
            raise RealMLXExperimentError(
                f"{distribution_name} runtime distribution root is unavailable"
            ) from exc
        trusted = next(
            (
                (label, root, scripts_root)
                for label, root, scripts_root in site_roots
                if distribution_root == root
            ),
            None,
        )
        if trusted is None:
            raise RealMLXExperimentError(
                f"{distribution_name} runtime distribution root is untrusted"
            )
        trusted_label, trusted_root, trusted_scripts_root = trusted
        file_count, total_bytes, tree_sha256 = _distribution_tree_identity(
            distribution,
            distribution_name=distribution_name,
            trusted_root=trusted_root,
            trusted_scripts_root=trusted_scripts_root,
            output_workspace=output_workspace,
        )
        identities[distribution_name] = {
            "distribution": distribution_name,
            "version": required_version,
            "trusted_root": trusted_label,
            "file_count": file_count,
            "total_bytes": total_bytes,
            "tree_sha256": tree_sha256,
        }
    return identities


def _validate_runtime_package_identity_schema(value: Any) -> None:
    if not isinstance(value, dict) or set(value) != set(_RUNTIME_DISTRIBUTION_VERSIONS):
        raise RealMLXExperimentError("runtime package identity is invalid")
    for name, required_version in _RUNTIME_DISTRIBUTION_VERSIONS.items():
        entry = value[name]
        if not isinstance(entry, dict):
            raise RealMLXExperimentError("runtime package identity is invalid")
        _exact_keys(
            entry,
            {
                "distribution",
                "version",
                "trusted_root",
                "file_count",
                "total_bytes",
                "tree_sha256",
            },
            "runtime package identity",
        )
        if (
            entry["distribution"] != name
            or entry["version"] != required_version
            or entry["trusted_root"] != "purelib"
            or isinstance(entry["file_count"], bool)
            or not isinstance(entry["file_count"], int)
            or entry["file_count"] <= 0
            or isinstance(entry["total_bytes"], bool)
            or not isinstance(entry["total_bytes"], int)
            or entry["total_bytes"] <= 0
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(entry["tree_sha256"])) is None
        ):
            raise RealMLXExperimentError("runtime package identity is invalid")


def _expected_runtime_package_identity() -> dict[str, dict[str, str | int]]:
    path = EXPECTED_RUNTIME_PACKAGE_IDENTITIES
    if (
        path.is_symlink()
        or not path.is_file()
        or path.resolve(strict=True) != path.absolute()
    ):
        raise RealMLXExperimentError("expected runtime package identity is unavailable")
    raw = path.read_bytes()
    if (
        "sha256:" + hashlib.sha256(raw).hexdigest()
        != EXPECTED_RUNTIME_PACKAGE_IDENTITIES_SHA256
    ):
        raise RealMLXExperimentError(
            "expected runtime package identity digest mismatch"
        )

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            if key in result:
                raise RealMLXExperimentError(
                    "expected runtime package identity contains duplicate keys"
                )
            result[key] = item
        return result

    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RealMLXExperimentError(
            "expected runtime package identity is invalid"
        ) from exc
    _validate_runtime_package_identity_schema(value)
    return cast(dict[str, dict[str, str | int]], value)


def _verify_runtime_package_identity(value: Any) -> None:
    _validate_runtime_package_identity_schema(value)
    if value != _expected_runtime_package_identity():
        raise RealMLXExperimentError(
            "runtime package identity does not match canonical expected tree"
        )


def _runtime_package_identity(
    output_workspace: Path,
) -> dict[str, dict[str, str | int]]:
    expected = _expected_runtime_package_identity()
    actual = _compute_runtime_package_identity(output_workspace)
    if actual != expected:
        raise RealMLXExperimentError(
            "installed runtime package identity does not match canonical expected tree"
        )
    return actual


def _hash_regular_file(path: Path, expected_size: int, expected_digest: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise RealMLXExperimentError(
            f"model artifact is not a regular file: {path.name}"
        )
    before = path.stat()
    if before.st_size != expected_size:
        raise RealMLXExperimentError(f"model artifact size mismatch: {path.name}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()

    def signature(item: os.stat_result) -> tuple[int, int, int, int]:
        return (
            item.st_dev,
            item.st_ino,
            item.st_size,
            item.st_mtime_ns,
        )

    if signature(before) != signature(after):
        raise RealMLXExperimentError(
            f"model artifact changed while hashing: {path.name}"
        )
    if digest.hexdigest() != expected_digest:
        raise RealMLXExperimentError(f"model artifact digest mismatch: {path.name}")


def verify_model_contract(model_dir: Path, summary_path: Path) -> str:
    """Verify the committed conversion summary's exact eight-file output contract."""

    if model_dir.is_symlink() or not model_dir.is_dir():
        raise RealMLXExperimentError("model directory must be local and non-symlinked")
    if (
        summary_path.is_symlink()
        or not summary_path.is_file()
        or hashlib.sha256(summary_path.read_bytes()).hexdigest()
        != EXPECTED_CONVERSION_SUMMARY_SHA256
    ):
        raise RealMLXExperimentError("conversion summary identity mismatch")
    summary = _safe_object(summary_path)
    if (
        summary.get("conversion_id") != "qwen3-4b-mlx-q4g64-self-convert-v1"
        or summary.get("source", {}).get("official_id") != "Qwen/Qwen3-4B"
        or summary.get("source", {}).get("official_revision")
        != "1cfa9a7208912126459214e8b04321603b3df60c"
        or summary.get("source", {}).get("license") != "Apache-2.0"
        or summary.get("converter", {}).get("package") != "mlx-lm"
        or summary.get("converter", {}).get("version") != REQUIRED_MLX_LM_VERSION
        or summary.get("converter", {}).get("git_revision")
        != "ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd"
        or summary.get("parameters", {}).get("q_group_size") != 64
        or summary.get("parameters", {}).get("q_bits") != 4
        or summary.get("parameters", {}).get("q_mode") != "affine"
        or summary.get("output", {}).get("repository_id") != MODEL_ID
        or summary.get("output", {}).get("total_bytes") != 2_274_515_269
    ):
        raise RealMLXExperimentError("conversion summary provenance contract mismatch")
    output = summary.get("output")
    if not isinstance(output, dict) or not isinstance(output.get("files"), list):
        raise RealMLXExperimentError("conversion summary has no output file contract")
    files = output["files"]
    if len(files) != EXPECTED_MODEL_FILE_COUNT:
        raise RealMLXExperimentError("conversion summary must contain exactly 8 files")
    contract: list[dict[str, Any]] = []
    expected_names: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise RealMLXExperimentError("invalid conversion-summary file entry")
        name, size, digest = item["path"], item["size_bytes"], item["sha256"]
        if (
            not isinstance(name, str)
            or Path(name).parts != (name,)
            or name in expected_names
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 1
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise RealMLXExperimentError("unsafe conversion-summary file contract")
        expected_names.add(name)
        _hash_regular_file(model_dir / name, size, digest)
        contract.append({"path": name, "size_bytes": size, "sha256": digest})
    actual_names = {item.name for item in model_dir.iterdir()}
    if actual_names != expected_names:
        raise RealMLXExperimentError(
            "model directory differs from exact 8-file contract"
        )
    digest = _digest_bytes(_json_bytes(contract))
    if digest != EXPECTED_MODEL_ARTIFACT_DIGEST:
        raise RealMLXExperimentError("model artifact contract digest mismatch")
    return digest


def _load_local_tokenizer(model_dir: Path) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RealMLXExperimentError(
            "transformers is required for local compilation"
        ) from exc
    try:
        return AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception as exc:
        raise RealMLXExperimentError(
            "local tokenizer load failed; downloads are forbidden"
        ) from exc


def _adapter_factory(
    *,
    model: Any,
    tokenizer: Any,
    model_key: Any,
    digest: str,
    observer: Callable[[MLXStageObservation], None] | None = None,
) -> Callable[[], MLXLocalCacheAdapter]:
    def create() -> MLXLocalCacheAdapter:
        return MLXLocalCacheAdapter(
            runtime=ProductionMLXRuntime(
                max_cache_entries=MAX_CACHE_ENTRIES,
                max_cache_bytes=MAX_CACHE_BYTES,
            ),
            model=model,
            tokenizer=tokenizer,
            model_key=model_key,
            model_artifact_digest=digest,
            max_cache_entries=MAX_CACHE_ENTRIES,
            max_cache_bytes=MAX_CACHE_BYTES,
            correctness_evaluator=lambda tokens: output_is_cache_ok(tokenizer, tokens),
            stage_observer=observer,
        )

    return create


def _run_warmup_lanes(
    workload: FrozenMLXWorkload,
    *,
    model: Any,
    tokenizer: Any,
) -> tuple[str, ...]:
    warmed: list[str] = []
    for lane in workload.lanes:
        calibration_outputs = lane.calibration_outputs
        if calibration_outputs is None:
            raise RealMLXExperimentError("warm-up requires calibrated lane outputs")
        runtime = ProductionMLXRuntime(
            max_cache_entries=MAX_CACHE_ENTRIES,
            max_cache_bytes=MAX_CACHE_BYTES,
        )
        cache = runtime.make_cache(model)
        try:
            generated = tuple(
                step.token
                for step in runtime.generate(
                    model,
                    tokenizer,
                    cache,
                    lane.base,
                    max_tokens=MAX_OUTPUT_TOKENS,
                    prompt_progress_callback=lambda _processed, _total: None,
                )
            )
            runtime.synchronize()
            if generated != calibration_outputs["base"] or not output_is_cache_ok(
                tokenizer, generated
            ):
                raise RealMLXExperimentError(
                    f"{lane.lane_id} warm-up failed frozen output gates"
                )
        finally:
            del cache
            del runtime
            _teardown()
        warmed.append(lane.lane_id)
    return tuple(warmed)


def _load_verified_model(
    model_dir: Path, conversion_summary: Path
) -> tuple[Any, Any, Any, str, tempfile.TemporaryDirectory[str]]:
    owner, snapshot, digest = _verified_model_snapshot(model_dir, conversion_summary)
    runtime = ProductionMLXRuntime(
        max_cache_entries=MAX_CACHE_ENTRIES,
        max_cache_bytes=MAX_CACHE_BYTES,
    )
    capability = check_mlx_capabilities(runtime)
    if not capability.supported:
        raise RealMLXExperimentError(
            "MLX preflight failed: " + "; ".join(capability.reasons)
        )
    try:
        model, tokenizer, model_key = runtime.load_model(snapshot)
    except Exception:
        owner.cleanup()
        raise
    return model, tokenizer, model_key, digest, owner


def _verified_model_snapshot(
    model_dir: Path, conversion_summary: Path
) -> tuple[tempfile.TemporaryDirectory[str], Path, str]:
    digest = verify_model_contract(model_dir, conversion_summary)
    summary = _safe_object(conversion_summary)
    files = summary["output"]["files"]
    owner = tempfile.TemporaryDirectory(prefix="llmtracefx-real-mlx-model-")
    snapshot = Path(owner.name)
    try:
        for item in files:
            source = model_dir / item["path"]
            if source.is_symlink() or not source.is_file():
                raise RealMLXExperimentError("model artifact changed before snapshot")
            shutil.copyfile(source, snapshot / item["path"])
        snapshot_digest = verify_model_contract(snapshot, conversion_summary)
        if snapshot_digest != digest:
            raise RealMLXExperimentError("verified model snapshot digest mismatch")
    except Exception:
        owner.cleanup()
        raise
    return owner, snapshot, digest


def calibrate_workload_file(
    source: Path,
    destination: Path,
    *,
    model_dir: Path,
    conversion_summary: Path = DEFAULT_CONVERSION_SUMMARY,
) -> FrozenMLXWorkload:
    if destination.exists():
        raise RealMLXExperimentError("calibrated workload output already exists")
    workload = load_workload(source, calibrated=False)
    model, tokenizer, model_key, digest, snapshot_owner = _load_verified_model(
        model_dir, conversion_summary
    )
    try:
        calibrated = calibrate_outputs(
            workload,
            _adapter_factory(
                model=model,
                tokenizer=tokenizer,
                model_key=model_key,
                digest=digest,
            ),
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_json(destination, calibrated.to_dict())
        return calibrated
    finally:
        del model
        del tokenizer
        _teardown()
        snapshot_owner.cleanup()


def _teardown(runtime_cleanup: bool = True) -> dict[str, Any]:
    gc.collect()
    allocator_cache = None
    cleanup = "not_requested"
    if runtime_cleanup:
        try:
            import mlx.core as mx

            mx.synchronize()
            mx.clear_cache()
            mx.synchronize()
            allocator_cache = int(mx.get_cache_memory())
            cleanup = "completed"
        except Exception:
            cleanup = "unavailable"
    return {
        "schema_version": "1",
        "cache_lifecycles_released": True,
        "garbage_collection_completed": True,
        "mlx_allocator_cleanup": cleanup,
        "allocator_cache_bytes_after": allocator_cache,
        "scope": "current_replicate_child_process",
    }


def run_replicate(
    workload_path: Path,
    output_dir: Path,
    *,
    replicate_id: str,
    model_dir: Path,
    expected_commit: str,
    conversion_summary: Path = DEFAULT_CONVERSION_SUMMARY,
    expected_runtime_packages_digest: str | None = None,
) -> dict[str, Any]:
    """Run one replicate. Parent-level timeout/process isolation stays external."""

    _reject_import_shadows(output_dir)
    if output_dir.exists():
        raise RealMLXExperimentError("replicate output already exists")
    output_dir.mkdir(parents=True)
    snapshot_owner: tempfile.TemporaryDirectory[str] | None = None
    observer: StageRecorder | None = None
    instance_id = _run_instance_id()
    try:
        if os.environ.get(RUN_INSTANCE_ENV) is not None and sys.flags.isolated != 1:
            raise RealMLXExperimentError("supervised replicate requires isolated mode")
        _validate_supervisor_source(expected_commit)
        runtime_packages_digest = _digest_bytes(
            _json_bytes(_runtime_package_identity(output_dir))
        )
        if (
            expected_runtime_packages_digest is not None
            and runtime_packages_digest != expected_runtime_packages_digest
        ):
            raise RealMLXExperimentError("runtime package identity changed")
        workload = load_workload(workload_path, calibrated=True)
        model, tokenizer, model_key, digest, snapshot_owner = _load_verified_model(
            model_dir, conversion_summary
        )
        _write_json(
            output_dir / "environment.json",
            _safe_environment(instance_id=instance_id),
        )
        _write_json(output_dir / "workload.json", workload.to_dict())
        warmup_lanes = _run_warmup_lanes(
            workload,
            model=model,
            tokenizer=tokenizer,
        )
        observer = StageRecorder(replicate_id)
        adapter = LifecycleMLXAdapter(
            runtime_factory=lambda: ProductionMLXRuntime(
                max_cache_entries=MAX_CACHE_ENTRIES,
                max_cache_bytes=MAX_CACHE_BYTES,
            ),
            model=model,
            tokenizer=tokenizer,
            model_key=model_key,
            model_artifact_digest=digest,
            correctness_evaluator=lambda tokens: output_is_cache_ok(tokenizer, tokens),
            stage_observer=observer,
        )
        requests = requests_for_replicate(workload, replicate_id)
        run_audit(
            adapter=adapter,
            requests=requests,
            cache_config=_experiment_cache_config(),
            output_dir=output_dir / "bundle",
            backend_version=REQUIRED_MLX_LM_VERSION,
            model_id=MODEL_ID,
            tokenizer_id=TOKENIZER_ID,
            model_artifact_digest=digest,
            publication_mode=PublicationMode.PRIVATE,
            seed=REPLICATE_IDS.index(replicate_id),
        )
        verify_bundle(output_dir / "bundle")
        observer.write(output_dir / "stages.jsonl")
        del adapter
        del model
        teardown = _teardown()
        snapshot_owner.cleanup()
        _validate_supervisor_source(expected_commit)
        if (
            _digest_bytes(_json_bytes(_runtime_package_identity(output_dir)))
            != runtime_packages_digest
        ):
            raise RealMLXExperimentError("runtime package identity changed")
        _write_json(
            output_dir / "attempt.json",
            {
                "schema_version": "2",
                "replicate_id": replicate_id,
                "status": "complete",
                "rotation": list(block_schedule(replicate_id)),
                "request_count": len(requests),
                "lane_request_counts": _lane_request_counts(requests),
                "independent_unit": True,
                "replacement": False,
                "expected_commit": expected_commit,
                "warmup_lanes": list(warmup_lanes),
                "warmup_excluded": True,
                "frozen_workload_digest": workload.to_dict()["workload_digest"],
                "frozen_lane_digests": {
                    lane.lane_id: lane.to_dict()["lane_digest"]
                    for lane in workload.lanes
                },
                "model_artifact_digest": digest,
                "model_id": MODEL_ID,
                "tokenizer_id": TOKENIZER_ID,
                "runtime_identity_digest": _digest_bytes(
                    _json_bytes(_EXPECTED_RUNTIME_IDENTITY)
                ),
                "runtime_packages_digest": runtime_packages_digest,
                "cache_config_digest": _digest_bytes(
                    _json_bytes(_experiment_cache_config().to_dict())
                ),
            },
        )
        _write_json(output_dir / "teardown.json", teardown)
        verify_replicate(output_dir, replicate_id=replicate_id, public=False)
        return {"replicate_id": replicate_id, "status": "complete"}
    except Exception:
        if snapshot_owner is not None:
            snapshot_owner.cleanup()
        if observer is not None and not (output_dir / "stages.jsonl").exists():
            observer.write(output_dir / "stages.jsonl")
        attempt_path = output_dir / "attempt.json"
        if attempt_path.exists() or attempt_path.is_symlink():
            attempt_path.unlink()
        _write_json(
            attempt_path,
            {
                "schema_version": "1",
                "replicate_id": replicate_id,
                "status": "failed",
                "replacement": False,
                "reason": "replicate_execution_failed",
                "failed_at": _utc_now(),
            },
        )
        _write_json(output_dir / "teardown.json", _teardown())
        raise


def record_failed_attempt(
    output_dir: Path,
    replicate_id: str,
    *,
    reason: str = "externally_terminated",
    failed_at: str | None = None,
) -> None:
    """Create a path-free failed-attempt marker for an externally timed-out child."""

    if output_dir.exists():
        raise RealMLXExperimentError("attempt output already exists")
    if replicate_id not in REPLICATE_IDS:
        raise RealMLXExperimentError("invalid replicate ID")
    if re.fullmatch(r"[a-z][a-z0-9_]{2,63}", reason) is None:
        raise RealMLXExperimentError("failed attempt reason must be a safe reason code")
    output_dir.mkdir(parents=True)
    _write_json(
        output_dir / "attempt.json",
        {
            "schema_version": "1",
            "replicate_id": replicate_id,
            "status": "failed",
            "replacement": False,
            "reason": reason,
            "failed_at": failed_at or _utc_now(),
        },
    )
    _write_json(
        output_dir / "teardown.json",
        {
            "schema_version": "1",
            "cache_lifecycles_released": None,
            "garbage_collection_completed": None,
            "mlx_allocator_cleanup": "externally_terminated",
            "allocator_cache_bytes_after": None,
            "scope": "current_replicate_child_process",
        },
    )


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RealMLXExperimentError(f"invalid JSONL in {path.name}") from exc
        if not isinstance(row, dict):
            raise RealMLXExperimentError(f"invalid row in {path.name}")
        rows.append(row)
    return rows


def _scan_private_keys(value: Any, context: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key.casefold() in PRIVATE_JSON_KEYS:
                raise RealMLXExperimentError(f"{context}.{key} is private")
            _scan_private_keys(item, f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _scan_private_keys(item, f"{context}[{index}]")


def _verify_stage_rows(
    rows: Sequence[dict[str, Any]],
    *,
    public: bool,
    replicate_id: str,
    records: Sequence[RequestEvidence],
    rotation: Sequence[str],
) -> None:
    expected = {
        "schema_version",
        "replicate_id",
        "request_id",
        "stage",
        "allocator",
        "logical_cache",
        "process_rss",
        "system_swap",
        "system_memory",
        "thermal_power",
    }
    expected_sequence: list[tuple[str | None, str]] = []
    record_index = 0
    for block in rotation:
        expected_sequence.append((None, "lifecycle_ready"))
        count = _BLOCK_REQUEST_COUNTS.get(block)
        if count is None:
            raise RealMLXExperimentError("attempt rotation contains an unknown block")
        block_records = records[record_index : record_index + count]
        if not public and any(
            _request_block_id(record.spec.request_id) != block
            for record in block_records
        ):
            raise RealMLXExperimentError("stage sequence lane/block identity mismatch")
        for record in block_records:
            expected_sequence.extend(
                (record.spec.request_id, stage)
                for stage in (
                    "request_before_lookup",
                    "request_after_lookup",
                    "request_after_generation",
                    "request_after_insertion",
                )
            )
        expected_sequence.extend(
            (record.spec.request_id, "request_after_baseline")
            for record in block_records
        )
        record_index += count
    if record_index != len(records) or len(rows) != len(expected_sequence):
        raise RealMLXExperimentError("stage sequence does not cover every request")

    for row, expected_boundary in zip(rows, expected_sequence, strict=True):
        _exact_keys(row, expected, "stage")
        _exact_keys(
            row["allocator"],
            {"active_bytes", "cache_bytes", "peak_bytes", "scope"},
            "stage allocator",
        )
        _exact_keys(row["logical_cache"], {"bytes", "scope"}, "logical cache")
        _exact_keys(row["process_rss"], {"bytes", "scope"}, "process RSS")
        _exact_keys(row["system_swap"], {"used_bytes", "scope"}, "system swap")
        _exact_keys(
            row["system_memory"],
            {"free_percent", "scope"},
            "system memory",
        )
        _exact_keys(
            row["thermal_power"],
            {"thermal_state", "power_watts", "scope"},
            "thermal and power",
        )
        if row["allocator"]["scope"] != "mlx_process_global_allocator":
            raise RealMLXExperimentError("allocator scope mismatch")
        if row["logical_cache"]["scope"] != "current_lru_entry_when_observable":
            raise RealMLXExperimentError("logical cache scope mismatch")
        if (
            row["schema_version"] != "1"
            or row["replicate_id"] != replicate_id
            or (row["request_id"], row["stage"]) != expected_boundary
        ):
            raise RealMLXExperimentError("stage boundary sequence mismatch")
        for field in ("active_bytes", "cache_bytes", "peak_bytes"):
            value = row["allocator"][field]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RealMLXExperimentError("allocator stage value is invalid")
        logical = row["logical_cache"]["bytes"]
        if logical is not None and (
            isinstance(logical, bool) or not isinstance(logical, int) or logical < 0
        ):
            raise RealMLXExperimentError("logical cache stage value is invalid")
        rss = row["process_rss"]["bytes"]
        swap = row["system_swap"]["used_bytes"]
        memory_free = row["system_memory"]["free_percent"]
        if (
            isinstance(rss, bool)
            or not isinstance(rss, int)
            or rss < 0
            or rss > MAX_PROCESS_RSS_BYTES
            or isinstance(swap, bool)
            or not isinstance(swap, int)
            or swap < 0
            or swap > MAX_SWAP_BYTES
            or isinstance(memory_free, bool)
            or not isinstance(memory_free, (int, float))
            or memory_free < MIN_RUNTIME_MEMORY_FREE_PERCENT
            or memory_free > 100
            or row["allocator"]["peak_bytes"] > MAX_ALLOCATOR_PEAK_BYTES
        ):
            raise RealMLXExperimentError("stage safety measurement is invalid")
        if row["process_rss"]["scope"] != "current_replicate_child_process_only":
            raise RealMLXExperimentError("RSS scope mismatch")
        if row["system_swap"]["scope"] != "system_wide":
            raise RealMLXExperimentError("swap scope mismatch")
        if row["system_memory"]["scope"] != "system_wide_memory_pressure":
            raise RealMLXExperimentError("system memory scope mismatch")
        if row["thermal_power"] != {
            "thermal_state": None,
            "power_watts": None,
            "scope": "unavailable_without_safe_collector",
        }:
            raise RealMLXExperimentError("thermal/power unavailable contract mismatch")
        if (
            public
            and row["request_id"] is not None
            and re.fullmatch(r"request-[0-9]{4}", row["request_id"]) is None
        ):
            raise RealMLXExperimentError("public stage request ID is not redacted")


def _require_capacity_eviction_verdicts(
    records: Sequence[RequestEvidence],
) -> None:
    records_by_id = {record.spec.request_id: record for record in records}
    for lane_id in LANE_IDS:
        treatment = records_by_id.get(f"{lane_id}:capacity-eviction:a-miss")
        if treatment is None or treatment.verdict is not Verdict.EVICTED:
            raise RealMLXExperimentError(
                f"{lane_id} capacity-eviction treatment is not evicted"
            )


def _require_private_output_gates(
    records: Sequence[RequestEvidence],
    workload: FrozenMLXWorkload,
) -> None:
    if any(
        record.terminal_state is not TerminalState.COMPLETED
        or record.output.token_identity.value is not True
        or record.output.correctness.value is not True
        or len(record.output.output_token_ids or ())
        != EXPECTED_CALIBRATION_OUTPUT_TOKENS[
            _request_block_id(record.spec.request_id).split(":", 1)[0]
        ]
        or record.output.baseline_token_ids != record.output.output_token_ids
        for record in records
    ):
        raise RealMLXExperimentError(
            "complete replicate output identity/correctness gate failed"
        )
    for record in records:
        lane_id = _request_block_id(record.spec.request_id).split(":", 1)[0]
        lane = workload.lane(lane_id)
        calibration_outputs = lane.calibration_outputs
        if calibration_outputs is None:
            raise RealMLXExperimentError("replicate workload is not calibrated")
        array_name = _request_source_array(record.spec.request_id)
        if record.output.output_token_ids != calibration_outputs[array_name]:
            raise RealMLXExperimentError(
                f"{record.spec.request_id} output differs from calibration"
            )


def _request_source_array(request_id: str) -> str:
    block = _request_block_id(request_id)
    _, case = block.split(":", 1)
    name = request_id.rsplit(":", 1)[-1]
    if case in {"cold-exact-duplicate", "namespace-isolation"} or name == "seed":
        return "base"
    if case == "interior-mutation":
        return "mutation_137"
    if case == "allocation-step-mutation":
        return "mutation_256"
    if case == "same-length-different-ids":
        return "different_ids"
    if case == "suffix-only-change":
        return "suffix_change"
    if case == "capacity-eviction":
        return {
            "a-seed": "eviction_a",
            "a-hit": "eviction_a",
            "a-miss": "eviction_a",
            "b": "eviction_b",
            "c": "eviction_c",
        }[name]
    raise RealMLXExperimentError("request has no calibrated source array")


def verify_replicate(
    directory: Path,
    *,
    replicate_id: str,
    public: bool,
    data_only_bundle: bool = False,
) -> dict[str, Any]:
    if directory.is_symlink() or not directory.is_dir():
        raise RealMLXExperimentError("replicate must be a regular directory")
    attempt = _safe_object(directory / "attempt.json")
    status = attempt.get("status")
    expected = (
        (_PUBLIC_REPLICATE_FILES if public else _PRIVATE_REPLICATE_FILES)
        if status == "complete"
        else _FAILED_REPLICATE_FILES
    )
    if {item.name for item in directory.iterdir()} != expected:
        raise RealMLXExperimentError("replicate file allowlist mismatch")
    if (
        replicate_id not in REPLICATE_IDS
        or attempt.get("replicate_id") != replicate_id
        or attempt.get("replacement") is not False
        or status not in {"complete", "failed"}
    ):
        raise RealMLXExperimentError("replicate attempt contract mismatch")
    if status == "failed":
        _exact_keys(
            attempt,
            {
                "schema_version",
                "replicate_id",
                "status",
                "replacement",
                "reason",
                "failed_at",
            },
            "failed attempt",
        )
        if (
            attempt["schema_version"] != "1"
            or re.fullmatch(r"[a-z][a-z0-9_]{2,63}", str(attempt["reason"])) is None
            or not _valid_utc_timestamp(attempt["failed_at"])
        ):
            raise RealMLXExperimentError("failed attempt reason/timestamp is invalid")
        _verify_teardown(_safe_object(directory / "teardown.json"), complete=False)
        return {
            "replicate_id": replicate_id,
            "status": status,
            "reason": attempt["reason"],
        }

    _exact_keys(
        attempt,
        {
            "schema_version",
            "replicate_id",
            "status",
            "rotation",
            "request_count",
            "lane_request_counts",
            "independent_unit",
            "replacement",
            "expected_commit",
            "warmup_lanes",
            "warmup_excluded",
            "frozen_workload_digest",
            "frozen_lane_digests",
            "model_artifact_digest",
            "model_id",
            "tokenizer_id",
            "runtime_identity_digest",
            "runtime_packages_digest",
            "cache_config_digest",
        },
        "complete attempt",
    )
    rotation = tuple(attempt["rotation"])
    if (
        attempt["schema_version"] != "2"
        or rotation != block_schedule(replicate_id)
        or attempt["request_count"] != sum(_BLOCK_REQUEST_COUNTS.values())
        or attempt["lane_request_counts"] != dict.fromkeys(LANE_IDS, _REQUESTS_PER_LANE)
        or attempt["independent_unit"] is not True
        or re.fullmatch(r"[0-9a-f]{40}", str(attempt["expected_commit"])) is None
        or attempt["warmup_lanes"] != list(LANE_IDS)
        or attempt["warmup_excluded"] is not True
        or attempt["model_id"] != MODEL_ID
        or attempt["tokenizer_id"] != TOKENIZER_ID
        or attempt["frozen_workload_digest"] != EXPECTED_CALIBRATED_WORKLOAD_DIGEST
        or attempt["frozen_lane_digests"] != EXPECTED_CALIBRATED_LANE_DIGESTS
        or attempt["model_artifact_digest"] != EXPECTED_MODEL_ARTIFACT_DIGEST
        or attempt["runtime_identity_digest"]
        != _digest_bytes(_json_bytes(_EXPECTED_RUNTIME_IDENTITY))
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(attempt["runtime_packages_digest"]),
        )
        is None
        or attempt["cache_config_digest"]
        != _digest_bytes(_json_bytes(_experiment_cache_config().to_dict()))
    ):
        raise RealMLXExperimentError("complete attempt binding is invalid")

    result = verify_bundle(directory / "bundle", data_only=data_only_bundle)
    manifest, records = read_bundle(directory / "bundle", data_only=data_only_bundle)
    expected_mode = (
        PublicationMode.PUBLIC_REDACTED if public else PublicationMode.PRIVATE
    )
    expected_cache = _experiment_cache_config()
    if public:
        expected_cache = replace(
            expected_cache,
            namespace_id="redacted-namespace",
            cache_type="redacted-cache",
        )
    if (
        manifest.publication_mode is not expected_mode
        or manifest.backend != "mlx_lm_local"
        or manifest.backend_version
        != ("redacted" if public else REQUIRED_MLX_LM_VERSION)
        or manifest.model_id != ("redacted-model" if public else MODEL_ID)
        or manifest.tokenizer_id != ("redacted-tokenizer" if public else TOKENIZER_ID)
        or manifest.model_artifact_digest
        != (None if public else attempt["model_artifact_digest"])
        or manifest.runtime_identity
        != ({"redaction": "public"} if public else _EXPECTED_RUNTIME_IDENTITY)
        or manifest.generator_commit != (None if public else attempt["expected_commit"])
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(manifest.generator_package_digest),
        )
        is None
        or manifest.cache_config != expected_cache
        or manifest.seed != REPLICATE_IDS.index(replicate_id)
        or result["request_count"] != attempt["request_count"]
        or tuple(manifest.request_order)
        != tuple(record.spec.request_id for record in records)
    ):
        raise RealMLXExperimentError("replicate standard bundle contract mismatch")
    if not public:
        _require_capacity_eviction_verdicts(records)

    if public:
        binding = _safe_object(directory / "workload-binding.json")
        expected_specs = _schedule_shape_specs(replicate_id, public=True)
        actual_specs = [record.spec.to_dict(include_tokens=False) for record in records]
        if actual_specs != expected_specs:
            raise RealMLXExperimentError("public replicate request schedule drifted")
        request_specs_digest = _digest_bytes(_json_bytes(expected_specs))
        expected_binding = {
            "schema_version": "1",
            "frozen_workload_digest": attempt["frozen_workload_digest"],
            "frozen_lane_digests": attempt["frozen_lane_digests"],
            "lane_request_counts": attempt["lane_request_counts"],
            "request_specs_digest": request_specs_digest,
        }
        if binding != expected_binding:
            raise RealMLXExperimentError("public workload binding mismatch")
    else:
        workload = load_workload(directory / "workload.json", calibrated=True)
        if workload.to_dict()["workload_digest"] != attempt["frozen_workload_digest"]:
            raise RealMLXExperimentError("replicate workload digest mismatch")
        if {
            lane.lane_id: lane.to_dict()["lane_digest"] for lane in workload.lanes
        } != attempt["frozen_lane_digests"]:
            raise RealMLXExperimentError("replicate lane digest mismatch")
        private_expected_specs = requests_for_replicate(workload, replicate_id)
        if [record.spec.to_dict() for record in records] != [
            spec.to_dict() for spec in private_expected_specs
        ]:
            raise RealMLXExperimentError("replicate request schedule drifted")
        if (
            _lane_request_counts(private_expected_specs)
            != attempt["lane_request_counts"]
        ):
            raise RealMLXExperimentError("replicate lane request count drifted")
        _require_private_output_gates(records, workload)

    environment = _safe_object(directory / "environment.json")
    _exact_keys(
        environment,
        {
            "schema_version",
            "platform_system",
            "platform_machine",
            "os_release",
            "python",
            "mlx",
            "mlx_lm",
            "process_scope",
            *(() if public else ("run_instance_id",)),
        },
        "environment",
    )
    if (
        environment["schema_version"] != "1"
        or environment["platform_system"] != "Darwin"
        or environment["platform_machine"] != "arm64"
        or environment["mlx"] != REQUIRED_MLX_VERSION
        or environment["mlx_lm"] != REQUIRED_MLX_LM_VERSION
        or environment["process_scope"] != "one_fresh_replicate_child"
    ):
        raise RealMLXExperimentError("replicate environment binding is invalid")
    if (
        not public
        and re.fullmatch(r"[0-9a-f]{32}", str(environment["run_instance_id"])) is None
    ):
        raise RealMLXExperimentError("replicate run instance ID is invalid")
    _verify_stage_rows(
        _parse_jsonl(directory / "stages.jsonl"),
        public=public,
        replicate_id=replicate_id,
        records=records,
        rotation=rotation,
    )
    _verify_teardown(_safe_object(directory / "teardown.json"), complete=True)
    return {"replicate_id": replicate_id, "status": status}


def _verify_teardown(value: Mapping[str, Any], *, complete: bool) -> None:
    _exact_keys(
        value,
        {
            "schema_version",
            "cache_lifecycles_released",
            "garbage_collection_completed",
            "mlx_allocator_cleanup",
            "allocator_cache_bytes_after",
            "scope",
        },
        "replicate teardown",
    )
    cache_bytes = value["allocator_cache_bytes_after"]
    if (
        value["schema_version"] != "1"
        or value["scope"] != "current_replicate_child_process"
        or value["mlx_allocator_cleanup"]
        not in {"completed", "unavailable", "externally_terminated"}
        or (
            cache_bytes is not None
            and (
                isinstance(cache_bytes, bool)
                or not isinstance(cache_bytes, int)
                or cache_bytes < 0
            )
        )
    ):
        raise RealMLXExperimentError("replicate teardown contract mismatch")
    for field in ("cache_lifecycles_released", "garbage_collection_completed"):
        if value[field] is not None and not isinstance(value[field], bool):
            raise RealMLXExperimentError("replicate teardown state is invalid")
    if complete and (
        value["cache_lifecycles_released"] is not True
        or value["garbage_collection_completed"] is not True
        or value["mlx_allocator_cleanup"] == "externally_terminated"
    ):
        raise RealMLXExperimentError("complete replicate teardown is invalid")


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        raise RealMLXExperimentError("aggregate destination is not empty")
    shutil.copytree(source, destination, symlinks=True)
    for path in destination.rglob("*"):
        if path.is_symlink():
            raise RealMLXExperimentError("copied evidence contains a symlink")


def _copy_replicate_data_only(source: Path, destination: Path) -> None:
    """Copy one verified replicate while rebuilding its bundle as data-only."""

    if destination.exists():
        raise RealMLXExperimentError("aggregate destination is not empty")
    attempt = _safe_object(source / "attempt.json")
    if attempt.get("status") == "failed":
        _copy_tree(source, destination)
        return
    destination.mkdir()
    for item in source.iterdir():
        if item.name == "bundle":
            continue
        if item.is_symlink() or not item.is_file():
            raise RealMLXExperimentError("replicate contains an unsafe entry")
        shutil.copyfile(item, destination / item.name)
    copy_data_only_bundle(source / "bundle", destination / "bundle")


def _replicate_index(
    root: Path,
    *,
    public: bool,
    run_binding: Mapping[str, Any],
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    verdicts: Counter[str] = Counter()
    scenarios: Counter[str] = Counter()
    lane_requests: Counter[str] = Counter()
    lane_scenarios: dict[str, Counter[str]] = {
        lane_id: Counter() for lane_id in LANE_IDS
    }
    total_requests = 0
    complete = 0
    compatible_bindings: set[tuple[str, ...]] = set()
    run_instance_ids: set[str] = set()
    for replicate_id in REPLICATE_IDS:
        directory = root / "replicates" / replicate_id
        state = verify_replicate(
            directory,
            replicate_id=replicate_id,
            public=public,
            data_only_bundle=True,
        )
        entry: dict[str, Any] = dict(state)
        if state["status"] == "complete":
            complete += 1
            manifest, records = read_bundle(directory / "bundle", data_only=True)
            attempt = _safe_object(directory / "attempt.json")
            environment = _safe_object(directory / "environment.json")
            if (
                attempt["expected_commit"] != run_binding["expected_commit"]
                or manifest.generator_package_digest
                != run_binding["generator_package_digest"]
                or attempt["runtime_packages_digest"]
                != _digest_bytes(_json_bytes(run_binding["runtime_packages"]))
            ):
                raise RealMLXExperimentError(
                    "replicate does not match aggregate ledger binding"
                )
            if not public:
                instance_id = str(environment["run_instance_id"])
                if instance_id in run_instance_ids:
                    raise RealMLXExperimentError(
                        "complete replicates must have distinct run instance IDs"
                    )
                run_instance_ids.add(instance_id)
            compatible_bindings.add(
                (
                    str(attempt["frozen_workload_digest"]),
                    canonical_json(attempt["frozen_lane_digests"]),
                    str(attempt["model_artifact_digest"]),
                    str(attempt["model_id"]),
                    str(attempt["tokenizer_id"]),
                    str(attempt["runtime_identity_digest"]),
                    str(attempt["cache_config_digest"]),
                    str(attempt["expected_commit"]),
                    str(manifest.generator_package_digest),
                    _digest_bytes(_json_bytes(_common_environment_value(environment))),
                )
            )
            counts = Counter(
                "unclassified" if row.verdict is None else row.verdict.value
                for row in records
            )
            for key, count in counts.items():
                verdicts[key] += count
            record_lanes = _scheduled_record_lanes(replicate_id)
            if len(record_lanes) != len(records):
                raise RealMLXExperimentError("lane schedule request count mismatch")
            for row, lane_id in zip(records, record_lanes, strict=True):
                scenarios[row.spec.scenario.value] += 1
                lane_requests[lane_id] += 1
                lane_scenarios[lane_id][row.spec.scenario.value] += 1
            total_requests += len(records)
            entry.update(
                {
                    "request_count": len(records),
                    "run_id": manifest.run_id,
                    "verdict_counts": dict(sorted(counts.items())),
                }
            )
        entries.append(entry)
    if complete != len(REPLICATE_IDS):
        raise RealMLXExperimentError("aggregate requires all 6 attempts complete")
    if not _aggregate_eligibility_from_rows(_parse_jsonl(root / "run-ledger.jsonl")):
        raise RealMLXExperimentError("aggregate requires all 6 attempts complete")
    if len(compatible_bindings) != 1:
        raise RealMLXExperimentError("complete replicates have incompatible bindings")
    binding = next(iter(compatible_bindings))
    return {
        "schema_version": "1",
        "replicates": entries,
        "complete_replicates": complete,
        "attempted_replicates": 6,
        "request_count": total_requests,
        "lane_request_counts": dict(sorted(lane_requests.items())),
        "verdict_counts": dict(sorted(verdicts.items())),
        "scenario_counts": dict(sorted(scenarios.items())),
        "lane_scenario_counts": {
            lane_id: dict(sorted(lane_scenarios[lane_id].items()))
            for lane_id in LANE_IDS
        },
        "evidence_binding": {
            **dict(run_binding),
            "frozen_workload_digest": binding[0],
            "frozen_lane_digests": json.loads(binding[1]),
            "model_artifact_digest": binding[2],
            "model_id": binding[3],
            "tokenizer_id": binding[4],
            "runtime_identity_digest": binding[5],
            "cache_config_digest": binding[6],
            "generator_commit": binding[7],
            "generator_package_digest": binding[8],
            "environment_digest": binding[9],
        },
    }


def _claim_matrix(index: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "claim_rule": (
            "A hit alone does not prove saved work or latency; article claims require "
            "the compatible verified aggregate cell and its raw paired samples."
        ),
        "complete_replicates": index["complete_replicates"],
        "scenario_observations": index["scenario_counts"],
        "lane_scenario_observations": index["lane_scenario_counts"],
        "lane_request_observations": index["lane_request_counts"],
        "verdict_observations": index["verdict_counts"],
        "allocation_step_boundary_256_is_block_cache_claim": False,
        "same_length_different_ids_interpretation": (
            "early_divergence_same_length_case_without_zero_reuse_guarantee"
        ),
        "namespace_isolation_scope": "harness_enforced_cache_key_separation",
        "native_mlx_tenancy_claim": False,
        "process_and_system_memory_rule": (
            "Allocator active/cache, process RSS, system swap, and system memory "
            "pressure are scoped level observations only and are not causal deltas; "
            "only the per-request-reset allocator peak supports a paired difference."
        ),
        "mlx_fetch_refreshes_lru": False,
        "mlx_insertion_refreshes_lru": True,
        "exact_repeat_policy": "N-1",
    }


def _comparison_name(
    lane_id: str, control: RequestEvidence, treatment: RequestEvidence
) -> str:
    scenario: ScenarioKind = treatment.spec.scenario
    if scenario is ScenarioKind.IDENTICAL_PREFIX:
        return f"{lane_id}:cold-exact"
    if scenario is ScenarioKind.SUFFIX_CHANGE:
        return f"{lane_id}:suffix-only-change"
    names: dict[ScenarioKind, str] = {
        ScenarioKind.WITHIN_BLOCK_MUTATION: "interior-mutation",
        ScenarioKind.BLOCK_BOUNDARY_MUTATION: "allocation-step-mutation",
        ScenarioKind.SAME_LENGTH_DIFFERENT_IDS: "same-length-different-ids",
        ScenarioKind.NAMESPACE_ISOLATION: "namespace-isolation",
        ScenarioKind.EVICTION_COUNT: "capacity-eviction",
    }
    return f"{lane_id}:{names.get(scenario, str(scenario.value))}"


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _fact_number(value: Any) -> int | float | None:
    return _number(value.value)


def _measurement_number(value: Any) -> int | float | None:
    return None if value is None else _number(value.value)


def _delta(left: int | float | None, right: int | float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(right) - float(left)


def _ratio(left: int | float | None, right: int | float | None) -> float | None:
    if left is None or right is None or float(left) <= 0:
        return None
    return float(right) / float(left)


def _request_stage_memory(path: Path) -> dict[str, dict[str, int | float]]:
    observations: dict[str, dict[str, list[int | float]]] = {}
    for row in _parse_jsonl(path):
        request_id = row["request_id"]
        if request_id is None or row["stage"] == "request_after_baseline":
            continue
        values = observations.setdefault(
            request_id,
            {
                "process_rss_bytes": [],
                "system_swap_used_bytes": [],
                "system_memory_free_percent": [],
            },
        )
        values["process_rss_bytes"].append(row["process_rss"]["bytes"])
        values["system_swap_used_bytes"].append(row["system_swap"]["used_bytes"])
        values["system_memory_free_percent"].append(
            row["system_memory"]["free_percent"]
        )
    return {
        request_id: {
            "process_rss_bytes": max(values["process_rss_bytes"]),
            "system_swap_used_bytes": max(values["system_swap_used_bytes"]),
            "system_memory_free_percent": min(values["system_memory_free_percent"]),
        }
        for request_id, values in observations.items()
    }


def _statistics(values: Sequence[int | float | None]) -> dict[str, Any]:
    available = sorted(float(value) for value in values if value is not None)
    if not available:
        return {
            "count": 0,
            "values": [],
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    middle = len(available) // 2
    median = (
        available[middle]
        if len(available) % 2
        else (available[middle - 1] + available[middle]) / 2
    )
    return {
        "count": len(available),
        "values": available,
        "median": median,
        "minimum": available[0],
        "maximum": available[-1],
    }


def _paired_sample(
    replicate_id: str,
    lane_id: str,
    control: RequestEvidence,
    treatment: RequestEvidence,
    stage_memory: Mapping[str, Mapping[str, int | float]],
) -> dict[str, Any]:
    control_ttft = _measurement_number(control.timing.client_ttft)
    treatment_ttft = _measurement_number(treatment.timing.client_ttft)
    control_total = _measurement_number(control.timing.total)
    treatment_total = _measurement_number(treatment.timing.total)
    control_peak = _fact_number(control.memory.runtime_peak_bytes)
    treatment_peak = _fact_number(treatment.memory.runtime_peak_bytes)
    control_active = _fact_number(control.memory.runtime_active_bytes)
    treatment_active = _fact_number(treatment.memory.runtime_active_bytes)
    control_allocator_cache = _fact_number(control.memory.allocator_cache_bytes)
    treatment_allocator_cache = _fact_number(treatment.memory.allocator_cache_bytes)
    control_stage = stage_memory[control.spec.request_id]
    treatment_stage = stage_memory[treatment.spec.request_id]
    control_rss = control_stage["process_rss_bytes"]
    treatment_rss = treatment_stage["process_rss_bytes"]
    control_swap = control_stage["system_swap_used_bytes"]
    treatment_swap = treatment_stage["system_swap_used_bytes"]
    control_memory_free = control_stage["system_memory_free_percent"]
    treatment_memory_free = treatment_stage["system_memory_free_percent"]
    control_recomputed = _fact_number(control.reuse.unexpected_recomputed_tokens)
    treatment_recomputed = _fact_number(treatment.reuse.unexpected_recomputed_tokens)
    control_output_tokens = len(control.output.output_token_ids or ())
    treatment_output_tokens = len(treatment.output.output_token_ids or ())
    supported_treatment_verdicts = {
        Verdict.VERIFIED_HIT,
        Verdict.PARTIAL_REUSE,
        Verdict.VERIFIED_MISS,
        Verdict.EVICTED,
    }
    paired_latency_comparable = (
        treatment.spec.scenario
        not in {ScenarioKind.EVICTION_COUNT, ScenarioKind.EVICTION_BYTES}
        and treatment.spec.order == control.spec.order + 1
        and control.terminal_state is TerminalState.COMPLETED
        and treatment.terminal_state is TerminalState.COMPLETED
        and control.output.token_identity.value is True
        and treatment.output.token_identity.value is True
        and control.output.correctness.value is True
        and treatment.output.correctness.value is True
        and control_output_tokens == treatment_output_tokens
        and control_ttft is not None
        and treatment_ttft is not None
        and control_total is not None
        and treatment_total is not None
        and treatment.verdict in supported_treatment_verdicts
        and control_recomputed == 0
        and treatment_recomputed == 0
    )

    def comparable_delta(
        left: int | float | None, right: int | float | None
    ) -> float | None:
        return _delta(left, right) if paired_latency_comparable else None

    def comparable_ratio(
        left: int | float | None, right: int | float | None
    ) -> float | None:
        return _ratio(left, right) if paired_latency_comparable else None

    return {
        "replicate_id": replicate_id,
        "lane_id": lane_id,
        "control_input_tokens": control.spec.input_token_count,
        "treatment_input_tokens": treatment.spec.input_token_count,
        "control_generated_output_tokens": control_output_tokens,
        "treatment_generated_output_tokens": treatment_output_tokens,
        "semantic_prefix_tokens": _fact_number(treatment.reuse.semantic_prefix_tokens),
        "policy_reusable_tokens": _fact_number(treatment.reuse.policy_reusable_tokens),
        "policy_reusable_blocks": _fact_number(treatment.reuse.reusable_blocks),
        "engine_cached_tokens": _fact_number(treatment.reuse.engine_cached_tokens),
        "engine_created_tokens": _fact_number(treatment.reuse.engine_created_tokens),
        "observed_prompt_tokens": _fact_number(treatment.reuse.observed_prompt_tokens),
        "unexpected_recomputed_tokens": treatment_recomputed,
        "paired_latency_comparable": paired_latency_comparable,
        "control_client_ttft_seconds": control_ttft,
        "treatment_client_ttft_seconds": treatment_ttft,
        "client_ttft_difference_seconds": comparable_delta(
            control_ttft, treatment_ttft
        ),
        "client_ttft_ratio": comparable_ratio(control_ttft, treatment_ttft),
        "control_total_seconds": control_total,
        "treatment_total_seconds": treatment_total,
        "total_difference_seconds": comparable_delta(control_total, treatment_total),
        "total_ratio": comparable_ratio(control_total, treatment_total),
        "control_allocator": {
            "active_bytes": control_active,
            "peak_bytes": control_peak,
            "cache_bytes": control_allocator_cache,
            "scope": "mlx_process_global_allocator",
        },
        "treatment_allocator": {
            "active_bytes": treatment_active,
            "peak_bytes": treatment_peak,
            "cache_bytes": treatment_allocator_cache,
            "scope": "mlx_process_global_allocator",
        },
        "allocator_peak_difference_bytes": comparable_delta(
            control_peak, treatment_peak
        ),
        "control_process_rss": {
            "bytes": control_rss,
            "scope": "current_replicate_child_process_only",
        },
        "treatment_process_rss": {
            "bytes": treatment_rss,
            "scope": "current_replicate_child_process_only",
        },
        "control_system_swap": {
            "used_bytes": control_swap,
            "scope": "system_wide",
        },
        "treatment_system_swap": {
            "used_bytes": treatment_swap,
            "scope": "system_wide",
        },
        "control_system_memory": {
            "free_percent": control_memory_free,
            "scope": "system_wide_memory_pressure",
        },
        "treatment_system_memory": {
            "free_percent": treatment_memory_free,
            "scope": "system_wide_memory_pressure",
        },
        "control_output_token_identity": control.output.token_identity.value,
        "treatment_output_token_identity": treatment.output.token_identity.value,
        "control_deterministic_correctness": control.output.correctness.value,
        "treatment_deterministic_correctness": treatment.output.correctness.value,
        "verdict": (None if treatment.verdict is None else treatment.verdict.value),
        "performance_eligibility": treatment.eligibility.performance.value,
        "output_eligibility": treatment.eligibility.output_equivalence.value,
        "quality_eligibility": treatment.eligibility.quality.value,
    }


def _descriptive_summary(
    replicates: Path, *, public: bool, data_only_bundle: bool = True
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for replicate_id in REPLICATE_IDS:
        directory = replicates / replicate_id
        attempt = _safe_object(directory / "attempt.json")
        if attempt.get("status") != "complete":
            continue
        _, records = read_bundle(directory / "bundle", data_only=data_only_bundle)
        stage_memory = _request_stage_memory(directory / "stages.jsonl")
        scheduled_lanes = _scheduled_record_lanes(replicate_id)
        if len(scheduled_lanes) != len(records):
            raise RealMLXExperimentError("descriptive lane schedule is incomplete")
        request_lanes = {
            record.spec.request_id: lane_id
            for record, lane_id in zip(records, scheduled_lanes, strict=True)
        }
        pairs: dict[str, dict[PairRole, RequestEvidence]] = {}
        for record in records:
            if record.spec.pair_id is None:
                continue
            pairs.setdefault(record.spec.pair_id, {})[record.spec.pair_role] = record
        for roles in pairs.values():
            if set(roles) != {PairRole.CONTROL, PairRole.TREATMENT}:
                raise RealMLXExperimentError(
                    "paired descriptive sample lacks one control and treatment"
                )
            control = roles[PairRole.CONTROL]
            treatment = roles[PairRole.TREATMENT]
            lane_id = request_lanes[control.spec.request_id]
            if request_lanes[treatment.spec.request_id] != lane_id:
                raise RealMLXExperimentError("descriptive pair crosses lanes")
            name = _comparison_name(lane_id, control, treatment)
            grouped.setdefault(name, []).append(
                _paired_sample(
                    replicate_id,
                    lane_id,
                    control,
                    treatment,
                    stage_memory,
                )
            )
    expected = {
        f"{lane_id}:{case}"
        for lane_id in LANE_IDS
        for case in (
            "cold-exact",
            "interior-mutation",
            "allocation-step-mutation",
            "same-length-different-ids",
            "suffix-only-change",
            "namespace-isolation",
            "capacity-eviction",
        )
    }
    if set(grouped) != expected:
        raise RealMLXExperimentError("descriptive comparison matrix is incomplete")
    metrics = (
        "semantic_prefix_tokens",
        "policy_reusable_tokens",
        "policy_reusable_blocks",
        "engine_cached_tokens",
        "engine_created_tokens",
        "observed_prompt_tokens",
        "unexpected_recomputed_tokens",
        "client_ttft_difference_seconds",
        "client_ttft_ratio",
        "total_difference_seconds",
        "total_ratio",
        "allocator_peak_difference_bytes",
    )
    comparisons: dict[str, Any] = {}
    for name in sorted(grouped):
        samples = sorted(grouped[name], key=lambda item: item["replicate_id"])
        comparisons[name] = {
            "independent_units": len(samples),
            "samples": samples,
            "statistics": {
                metric: _statistics([sample[metric] for sample in samples])
                for metric in metrics
            },
            "output_identity_true_pair_count": sum(
                sample["control_output_token_identity"] is True
                and sample["treatment_output_token_identity"] is True
                for sample in samples
            ),
            "correctness_true_pair_count": sum(
                sample["control_deterministic_correctness"] is True
                and sample["treatment_deterministic_correctness"] is True
                for sample in samples
            ),
            "identity_and_correctness_unavailable": public,
        }
    return {
        "schema_version": "1",
        "publication_mode": "public_redacted" if public else "private",
        "independent_unit": "one_fresh_replicate_child_process",
        "no_replacement": True,
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }


def _public_results_from_private(
    replicates: Path, *, data_only_bundle: bool = False
) -> dict[str, Any]:
    descriptive = _descriptive_summary(
        replicates,
        public=False,
        data_only_bundle=data_only_bundle,
    )
    comparisons = json.loads(json.dumps(descriptive["comparisons"]))
    for comparison in comparisons.values():
        comparison.pop("identity_and_correctness_unavailable", None)
    return {
        "schema_version": "real-mlx-public-results-v1",
        "source": "verified_private_records_before_redaction",
        "independent_unit": "one_fresh_replicate_child_process",
        "raw_sample_scope": "one_control_treatment_pair_per_cell_per_replicate",
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
    }


def _verify_public_results(
    value: Any,
    *,
    complete_replicate_ids: set[str] | None = None,
) -> None:
    if not isinstance(value, dict):
        raise RealMLXExperimentError("results must be an object")
    text = canonical_json(value)
    for pattern, label in PRIVACY_PATTERNS:
        if pattern.search(text):
            raise RealMLXExperimentError(f"results contain {label}")
    _scan_private_keys(value, "results")
    if '"input_token_ids":[' in text or '"output_token_ids":[' in text:
        raise RealMLXExperimentError("results contain exact token arrays")
    _exact_keys(
        value,
        {
            "schema_version",
            "source",
            "independent_unit",
            "raw_sample_scope",
            "comparison_count",
            "comparisons",
        },
        "public results",
    )
    expected_names = {
        f"{lane_id}:{case}"
        for lane_id in LANE_IDS
        for case in (
            "cold-exact",
            "interior-mutation",
            "allocation-step-mutation",
            "same-length-different-ids",
            "suffix-only-change",
            "namespace-isolation",
            "capacity-eviction",
        )
    }
    comparisons = value["comparisons"]
    if (
        value["schema_version"] != "real-mlx-public-results-v1"
        or value["source"] != "verified_private_records_before_redaction"
        or value["independent_unit"] != "one_fresh_replicate_child_process"
        or value["raw_sample_scope"]
        != "one_control_treatment_pair_per_cell_per_replicate"
        or value["comparison_count"] != len(expected_names)
        or not isinstance(comparisons, dict)
        or set(comparisons) != expected_names
    ):
        raise RealMLXExperimentError("public results contract mismatch")
    metric_names = {
        "semantic_prefix_tokens",
        "policy_reusable_tokens",
        "policy_reusable_blocks",
        "engine_cached_tokens",
        "engine_created_tokens",
        "observed_prompt_tokens",
        "unexpected_recomputed_tokens",
        "client_ttft_difference_seconds",
        "client_ttft_ratio",
        "total_difference_seconds",
        "total_ratio",
        "allocator_peak_difference_bytes",
    }
    sample_keys = {
        "replicate_id",
        "lane_id",
        "control_input_tokens",
        "treatment_input_tokens",
        "control_generated_output_tokens",
        "treatment_generated_output_tokens",
        "semantic_prefix_tokens",
        "policy_reusable_tokens",
        "policy_reusable_blocks",
        "engine_cached_tokens",
        "engine_created_tokens",
        "observed_prompt_tokens",
        "unexpected_recomputed_tokens",
        "paired_latency_comparable",
        "control_client_ttft_seconds",
        "treatment_client_ttft_seconds",
        "client_ttft_difference_seconds",
        "client_ttft_ratio",
        "control_total_seconds",
        "treatment_total_seconds",
        "total_difference_seconds",
        "total_ratio",
        "control_allocator",
        "treatment_allocator",
        "allocator_peak_difference_bytes",
        "control_process_rss",
        "treatment_process_rss",
        "control_system_swap",
        "treatment_system_swap",
        "control_system_memory",
        "treatment_system_memory",
        "control_output_token_identity",
        "treatment_output_token_identity",
        "control_deterministic_correctness",
        "treatment_deterministic_correctness",
        "verdict",
        "performance_eligibility",
        "output_eligibility",
        "quality_eligibility",
    }
    expected_replicates: set[str] | None = None
    for name, comparison in comparisons.items():
        if not isinstance(comparison, dict):
            raise RealMLXExperimentError("public result comparison is invalid")
        _exact_keys(
            comparison,
            {
                "independent_units",
                "samples",
                "statistics",
                "output_identity_true_pair_count",
                "correctness_true_pair_count",
            },
            "public result comparison",
        )
        samples = comparison["samples"]
        if (
            not isinstance(samples, list)
            or not 5 <= len(samples) <= 6
            or comparison["independent_units"] != len(samples)
            or comparison["output_identity_true_pair_count"] != len(samples)
            or comparison["correctness_true_pair_count"] != len(samples)
        ):
            raise RealMLXExperimentError("public result sample count is invalid")
        lane_id = name.split(":", 1)[0]
        case = name.split(":", 1)[1]
        expected_input_tokens = _LANE_CONTRACTS[lane_id][
            "eviction" if case == "capacity-eviction" else "base"
        ]
        seen_replicates: set[str] = set()
        for sample in samples:
            if not isinstance(sample, dict):
                raise RealMLXExperimentError("public result sample is invalid")
            _exact_keys(sample, sample_keys, "public result sample")
            replicate_id = sample["replicate_id"]
            if (
                replicate_id not in REPLICATE_IDS
                or replicate_id in seen_replicates
                or sample["lane_id"] != lane_id
                or sample["control_input_tokens"] != expected_input_tokens
                or sample["treatment_input_tokens"] != expected_input_tokens
                or sample["control_output_token_identity"] is not True
                or sample["treatment_output_token_identity"] is not True
                or sample["control_deterministic_correctness"] is not True
                or sample["treatment_deterministic_correctness"] is not True
                or not isinstance(sample["paired_latency_comparable"], bool)
            ):
                raise RealMLXExperimentError("public result sample binding is invalid")
            seen_replicates.add(replicate_id)
            for key in (
                "control_generated_output_tokens",
                "treatment_generated_output_tokens",
            ):
                count = sample[key]
                if (
                    isinstance(count, bool)
                    or not isinstance(count, int)
                    or count != EXPECTED_CALIBRATION_OUTPUT_TOKENS[lane_id]
                    or count > MAX_OUTPUT_TOKENS
                    or count > expected_input_tokens
                ):
                    raise RealMLXExperimentError(
                        "public result output-token count is invalid"
                    )
            for key in (
                "semantic_prefix_tokens",
                "policy_reusable_tokens",
                "policy_reusable_blocks",
                "engine_cached_tokens",
                "engine_created_tokens",
                "observed_prompt_tokens",
                "unexpected_recomputed_tokens",
                "control_client_ttft_seconds",
                "treatment_client_ttft_seconds",
                "control_total_seconds",
                "treatment_total_seconds",
            ):
                number = sample[key]
                if number is not None and (
                    isinstance(number, bool)
                    or not isinstance(number, (int, float))
                    or not math.isfinite(number)
                    or number < 0
                ):
                    raise RealMLXExperimentError("public result value is out of bounds")
            semantic = sample["semantic_prefix_tokens"]
            policy = sample["policy_reusable_tokens"]
            reusable_blocks = sample["policy_reusable_blocks"]
            engine_cached = sample["engine_cached_tokens"]
            engine_created = sample["engine_created_tokens"]
            observed = sample["observed_prompt_tokens"]
            recomputed = sample["unexpected_recomputed_tokens"]
            if (
                any(
                    isinstance(number, bool) or not isinstance(number, int)
                    for number in (
                        semantic,
                        policy,
                        engine_cached,
                        engine_created,
                        observed,
                        recomputed,
                    )
                )
                or reusable_blocks is not None
                or not 0 <= semantic <= expected_input_tokens
                or not 0 <= policy <= semantic
                or not 0 <= engine_cached <= expected_input_tokens
                or not 0 <= engine_created <= expected_input_tokens
                or not 0 <= observed <= expected_input_tokens
                or not 0 <= recomputed <= observed
                or engine_cached + engine_created != expected_input_tokens
                or observed != expected_input_tokens - policy
                or recomputed != 0
                or engine_cached != policy
            ):
                raise RealMLXExperimentError(
                    "public result reuse counters are inconsistent"
                )
            verdict = sample["verdict"]
            if verdict == Verdict.VERIFIED_HIT.value:
                verdict_valid = policy > 0 and semantic == expected_input_tokens
            elif verdict == Verdict.PARTIAL_REUSE.value:
                verdict_valid = policy > 0 and semantic < expected_input_tokens
            elif verdict == Verdict.VERIFIED_MISS.value:
                verdict_valid = (
                    policy == 0
                    and engine_cached == 0
                    and observed == expected_input_tokens
                    and case != "capacity-eviction"
                )
            elif verdict == Verdict.EVICTED.value:
                verdict_valid = (
                    case == "capacity-eviction"
                    and policy == 0
                    and engine_cached == 0
                    and observed == expected_input_tokens
                )
            else:
                verdict_valid = False
            if not verdict_valid or (
                sample["performance_eligibility"] != EligibilityStatus.INELIGIBLE.value
                or sample["output_eligibility"] != EligibilityStatus.ELIGIBLE.value
                or sample["quality_eligibility"]
                not in {
                    EligibilityStatus.UNAVAILABLE.value,
                    EligibilityStatus.NOT_APPLICABLE.value,
                }
            ):
                raise RealMLXExperimentError(
                    f"public result verdict relationship is invalid: {name}/"
                    f"{replicate_id}"
                )
            for prefix in ("control", "treatment"):
                ttft = sample[f"{prefix}_client_ttft_seconds"]
                total = sample[f"{prefix}_total_seconds"]
                if ttft is None or total is None:
                    raise RealMLXExperimentError("public result timing is unavailable")
                if ttft > total:
                    raise RealMLXExperimentError(
                        "public result TTFT exceeds total time"
                    )
            if case == "capacity-eviction" and sample["paired_latency_comparable"]:
                raise RealMLXExperimentError(
                    "eviction latency samples are not adjacent"
                )
            if sample["paired_latency_comparable"]:
                if any(
                    sample[key] is None
                    for key in (
                        "client_ttft_difference_seconds",
                        "client_ttft_ratio",
                        "total_difference_seconds",
                        "total_ratio",
                    )
                ):
                    raise RealMLXExperimentError(
                        "public comparable timing result is incomplete"
                    )
            elif any(
                sample[key] is not None
                for key in (
                    "client_ttft_difference_seconds",
                    "client_ttft_ratio",
                    "total_difference_seconds",
                    "total_ratio",
                    "allocator_peak_difference_bytes",
                )
            ):
                raise RealMLXExperimentError(
                    "public incomparable result contains causal deltas"
                )
            for key in (
                "client_ttft_difference_seconds",
                "client_ttft_ratio",
                "total_difference_seconds",
                "total_ratio",
                "allocator_peak_difference_bytes",
            ):
                number = sample[key]
                if number is not None and (
                    isinstance(number, bool)
                    or not isinstance(number, (int, float))
                    or not math.isfinite(number)
                    or key.endswith("_ratio")
                    and number < 0
                ):
                    raise RealMLXExperimentError("public result delta is invalid")
            for key in ("control_allocator", "treatment_allocator"):
                allocator = sample[key]
                if not isinstance(allocator, dict):
                    raise RealMLXExperimentError("public allocator level is invalid")
                _exact_keys(
                    allocator,
                    {"active_bytes", "peak_bytes", "cache_bytes", "scope"},
                    "public allocator level",
                )
                if allocator["scope"] != "mlx_process_global_allocator":
                    raise RealMLXExperimentError("public allocator scope is invalid")
                for field in ("active_bytes", "peak_bytes", "cache_bytes"):
                    number = allocator[field]
                    if (
                        isinstance(number, bool)
                        or not isinstance(number, int)
                        or number < 0
                    ):
                        raise RealMLXExperimentError(
                            "public allocator value is out of bounds"
                        )
                peak = allocator["peak_bytes"]
                if peak is not None and peak > MAX_ALLOCATOR_PEAK_BYTES:
                    raise RealMLXExperimentError("public allocator peak exceeds policy")
            for key in ("control_process_rss", "treatment_process_rss"):
                level = sample[key]
                if (
                    not isinstance(level, dict)
                    or level
                    != {
                        "bytes": level.get("bytes"),
                        "scope": "current_replicate_child_process_only",
                    }
                    or (
                        isinstance(level["bytes"], bool)
                        or not isinstance(level["bytes"], int)
                        or level["bytes"] < 0
                        or level["bytes"] > MAX_PROCESS_RSS_BYTES
                    )
                ):
                    raise RealMLXExperimentError("public RSS level is invalid")
            for key in ("control_system_swap", "treatment_system_swap"):
                level = sample[key]
                if (
                    not isinstance(level, dict)
                    or level
                    != {
                        "used_bytes": level.get("used_bytes"),
                        "scope": "system_wide",
                    }
                    or (
                        isinstance(level["used_bytes"], bool)
                        or not isinstance(level["used_bytes"], int)
                        or level["used_bytes"] < 0
                        or level["used_bytes"] > MAX_SWAP_BYTES
                    )
                ):
                    raise RealMLXExperimentError("public swap level is invalid")
            for key in ("control_system_memory", "treatment_system_memory"):
                level = sample[key]
                if not isinstance(level, dict):
                    raise RealMLXExperimentError(
                        "public system-memory level is invalid"
                    )
                free = level.get("free_percent")
                if (
                    level
                    != {
                        "free_percent": free,
                        "scope": "system_wide_memory_pressure",
                    }
                    or isinstance(free, bool)
                    or not isinstance(free, (int, float))
                    or not math.isfinite(free)
                    or not MIN_RUNTIME_MEMORY_FREE_PERCENT <= free <= 100
                ):
                    raise RealMLXExperimentError(
                        "public system-memory level is invalid"
                    )
            expected_deltas = {
                "client_ttft_difference_seconds": _delta(
                    sample["control_client_ttft_seconds"],
                    sample["treatment_client_ttft_seconds"],
                ),
                "client_ttft_ratio": _ratio(
                    sample["control_client_ttft_seconds"],
                    sample["treatment_client_ttft_seconds"],
                ),
                "total_difference_seconds": _delta(
                    sample["control_total_seconds"],
                    sample["treatment_total_seconds"],
                ),
                "total_ratio": _ratio(
                    sample["control_total_seconds"],
                    sample["treatment_total_seconds"],
                ),
                "allocator_peak_difference_bytes": _delta(
                    sample["control_allocator"]["peak_bytes"],
                    sample["treatment_allocator"]["peak_bytes"],
                ),
            }
            for key, expected_delta in expected_deltas.items():
                expected = (
                    expected_delta if sample["paired_latency_comparable"] else None
                )
                if sample[key] != expected:
                    raise RealMLXExperimentError(
                        "public result delta does not match raw levels"
                    )
        if expected_replicates is None:
            expected_replicates = seen_replicates
        elif seen_replicates != expected_replicates:
            raise RealMLXExperimentError(
                "public result replicate coverage is inconsistent"
            )
        statistics = comparison["statistics"]
        if not isinstance(statistics, dict) or set(statistics) != metric_names:
            raise RealMLXExperimentError("public result statistics are invalid")
        for metric in metric_names:
            if statistics[metric] != _statistics(
                [sample[metric] for sample in samples]
            ):
                raise RealMLXExperimentError(
                    "public result statistic does not match raw samples"
                )
    if (
        complete_replicate_ids is not None
        and expected_replicates != complete_replicate_ids
    ):
        raise RealMLXExperimentError(
            "public result replicate IDs do not match complete ledger attempts"
        )


def _complete_attempt_ids(attempts_dir: Path) -> set[str]:
    return {
        replicate_id
        for replicate_id in REPLICATE_IDS
        if _safe_object(attempts_dir / replicate_id / "attempt.json").get("status")
        == "complete"
    }


def _verify_public_result_record_bindings(
    results: Mapping[str, Any],
    replicates: Path,
    *,
    public: bool,
) -> None:
    expected = _descriptive_summary(
        replicates,
        public=public,
        data_only_bundle=True,
    )
    retained_fields = {
        "replicate_id",
        "lane_id",
        "control_input_tokens",
        "treatment_input_tokens",
        "policy_reusable_blocks",
        "engine_cached_tokens",
        "engine_created_tokens",
        "observed_prompt_tokens",
        "control_client_ttft_seconds",
        "treatment_client_ttft_seconds",
        "control_total_seconds",
        "treatment_total_seconds",
        "control_allocator",
        "treatment_allocator",
        "control_process_rss",
        "treatment_process_rss",
        "control_system_swap",
        "treatment_system_swap",
        "control_system_memory",
        "treatment_system_memory",
    }
    for name, comparison in results["comparisons"].items():
        observed_samples = comparison["samples"]
        expected_samples = expected["comparisons"][name]["samples"]
        if len(observed_samples) != len(expected_samples):
            raise RealMLXExperimentError("public result sample binding is incomplete")
        for observed, retained in zip(observed_samples, expected_samples, strict=True):
            if any(observed[field] != retained[field] for field in retained_fields):
                raise RealMLXExperimentError(
                    "public result sample does not match retained nested evidence"
                )


def _summary(index: Mapping[str, Any], *, public: bool) -> dict[str, Any]:
    return {
        "schema_version": "1",
        "publication_mode": "public_redacted" if public else "private",
        "attempted_replicates": 6,
        "complete_replicates": index["complete_replicates"],
        "failed_replicates": 6 - int(index["complete_replicates"]),
        "request_count": index["request_count"],
        "lane_request_counts": index["lane_request_counts"],
        "no_replacement": True,
        "sequential_boundary_minutes": 90,
        "missing_facts": "null",
    }


def _aggregate_html(summary: Mapping[str, Any]) -> str:
    return (
        '<!doctype html><meta charset="utf-8"><title>Real MLX cache audit</title>'
        "<h1>Real Apple Silicon MLX KV-cache experiment</h1>"
        f"<p>Complete replicates: {summary['complete_replicates']}/6; "
        f"requests: {summary['request_count']}.</p>"
        "<p>A cache hit alone does not prove saved work or latency.</p>\n"
    )


def _reuse_svg(index: Mapping[str, Any]) -> str:
    complete = int(index["complete_replicates"])
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="120" '
        'role="img" aria-label="complete replicate count">'
        '<rect width="640" height="120" fill="#fff"/>'
        f'<rect x="20" y="50" width="{complete * 90}" height="30" fill="#2867b2"/>'
        f'<text x="20" y="30">Complete replicates: {complete}/6</text></svg>\n'
    )


def _timing_svg(index: Mapping[str, Any]) -> str:
    requests = int(index["request_count"])
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="640" height="120" '
        'role="img" aria-label="timing and memory evidence scope">'
        '<rect width="640" height="120" fill="#fff"/>'
        f'<text x="20" y="35">Requests with preserved observations: {requests}</text>'
        '<text x="20" y="70">Timing and memory remain separate claim dimensions.</text>'
        "</svg>\n"
    )


def _experiment_contract(
    public: bool,
    *,
    binding: Mapping[str, Any],
    results_digest: str,
) -> dict[str, Any]:
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "publication_mode": "public_redacted" if public else "private",
        "replicate_ids": list(REPLICATE_IDS),
        "minimum_complete": 6,
        "replacement_allowed": False,
        "replicate_eligibility": {
            "attempted_replicates": 6,
            "required_complete_replicates": 6,
            "maximum_failed_replicates": 0,
        },
        "evidence_binding": dict(binding),
        "results_digest": results_digest,
        "integrity_and_authenticity": {
            "sha256sums": "integrity_only_unkeyed",
            "authenticity_anchor": "git_commit_containing_final_public_evidence",
        },
        "execution": "six independent fresh child processes; sequential",
        "child_temporary_directory": "private_output_workspace_filesystem",
        "performance_dry_run_used": False,
        "parent_timeout_minutes": 90,
        "child_timeout_minutes": 12,
        "max_output_tokens_per_request": MAX_OUTPUT_TOKENS,
        "max_cache_entries_per_lifecycle": MAX_CACHE_ENTRIES,
        "lanes": {
            lane_id: {
                "base_tokens": _LANE_CONTRACTS[lane_id]["base"],
                "eviction_prompt_tokens": _LANE_CONTRACTS[lane_id]["eviction"],
                "cases": list(_CASES),
                "requests_per_replicate": _REQUESTS_PER_LANE,
            }
            for lane_id in LANE_IDS
        },
        "combined_blocks": list(_BLOCKS),
        "schedule_affine_permutations": [
            {"offset": offset, "step": step}
            for offset, step in SCHEDULE_AFFINE_PERMUTATIONS
        ],
        "exact_block_schedules": {
            replicate_id: list(block_schedule(replicate_id))
            for replicate_id in REPLICATE_IDS
        },
        "model_contract_file_count": EXPECTED_MODEL_FILE_COUNT,
        "network_allowed": False,
        "limitations": [
            "one observation per cell per replicate",
            "descriptive medians and ranges only",
            "schedule, order, and thermal effects remain possible",
            "allocator active/cache, process RSS, system swap, and system-memory "
            "pressure are non-causal scoped levels",
            "2-second monitoring and stage-observation subprocesses can perturb host "
            "scheduling but are excluded from client clocks",
            "allocation step 256 is not block-cache behavior",
            "MLX cache reuse is token-granular",
            "constant-target CACHE_OK identity/correctness is a low-power guard "
            "that cannot rule out all KV corruption",
            "evidence is scoped to one host, model, and conversion",
            "all six preregistered replicates must complete; failures are preserved "
            "but invalidate the run and no replacements are allowed",
            "within each pair the cold control always precedes the warm treatment, "
            "so monotone drift can inflate apparent latency benefit",
            "paired timing deltas are descriptive and are not causal speedups",
            "no power, energy, kernel, or utilization claims",
            "namespace isolation is harness-enforced key separation, not native MLX tenancy",
        ],
    }


def _write_recursive_checksums(root: Path) -> None:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path != root / "SHA256SUMS"
    )
    text = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  "
        f"{path.relative_to(root).as_posix()}\n"
        for path in paths
    )
    atomic_write_text(root / "SHA256SUMS", text)


def _common_environment(replicates: Path) -> dict[str, Any]:
    values = [
        _common_environment_value(
            _safe_object(replicates / replicate_id / "environment.json")
        )
        for replicate_id in REPLICATE_IDS
        if _safe_object(replicates / replicate_id / "attempt.json").get("status")
        == "complete"
    ]
    if not values or any(value != values[0] for value in values[1:]):
        raise RealMLXExperimentError(
            "complete replicates must share one privacy-safe environment"
        )
    return values[0]


def _write_derived(root: Path, *, public: bool) -> None:
    results = _safe_object(root / "results.json")
    _verify_public_results(results)
    if not public and results != _public_results_from_private(
        root / "replicates", data_only_bundle=True
    ):
        raise RealMLXExperimentError(
            "public results do not reproduce verified private records"
        )
    results_digest = _digest_bytes(_json_bytes(results))
    run_binding = _verify_run_ledger(
        root / "run-ledger.jsonl",
        attempts_dir=root / "replicates",
        expected_results_digest=results_digest,
    )
    complete_ids = _complete_attempt_ids(root / "replicates")
    _verify_public_results(results, complete_replicate_ids=complete_ids)
    _verify_public_result_record_bindings(
        results,
        root / "replicates",
        public=public,
    )
    index = _replicate_index(root, public=public, run_binding=run_binding)
    summary = _summary(index, public=public)
    descriptive_summary = _descriptive_summary(root / "replicates", public=public)
    _write_json(
        root / "experiment-contract.json",
        _experiment_contract(
            public,
            binding=run_binding,
            results_digest=results_digest,
        ),
    )
    _write_json(root / "environment.json", _common_environment(root / "replicates"))
    _write_json(root / "replicate-index.json", index)
    _write_json(root / "claim-matrix.json", _claim_matrix(index))
    _write_json(root / "descriptive-summary.json", descriptive_summary)
    _write_json(root / "summary.json", summary)
    atomic_write_text(root / "report.html", _aggregate_html(summary))
    atomic_write_text(root / "reuse-alignment.svg", _reuse_svg(index))
    atomic_write_text(root / "timing-memory.svg", _timing_svg(index))
    _write_json(
        root / "teardown.json",
        {
            "schema_version": "1",
            "replicate_records_verified": 6,
            "complete_teardown_records": index["complete_replicates"],
            "scope": "aggregate_of_child_teardown_records",
        },
    )
    _write_recursive_checksums(root)


def assemble_aggregate(run_workspace: Path, output_dir: Path) -> dict[str, Any]:
    """Assemble one exact run-all workspace without publishing private artifacts."""

    if output_dir.exists():
        raise RealMLXExperimentError("aggregate output already exists")
    if run_workspace.is_symlink() or not run_workspace.is_dir():
        raise RealMLXExperimentError("run workspace must be a regular directory")
    if {item.name for item in run_workspace.iterdir()} != {
        "attempts",
        "private-artifacts",
        "run-ledger.jsonl",
    }:
        raise RealMLXExperimentError("run workspace allowlist mismatch")
    attempts_dir = run_workspace / "attempts"
    artifacts_dir = run_workspace / "private-artifacts"
    ledger = run_workspace / "run-ledger.jsonl"
    if (
        attempts_dir.is_symlink()
        or not attempts_dir.is_dir()
        or artifacts_dir.is_symlink()
        or not artifacts_dir.is_dir()
        or ledger.is_symlink()
        or not ledger.is_file()
    ):
        raise RealMLXExperimentError("run workspace contains an unsafe entry")
    actual = {item.name for item in attempts_dir.iterdir()}
    if actual != set(REPLICATE_IDS):
        raise RealMLXExperimentError(
            "attempt directory must contain replicate-0..5 exactly"
        )
    states = [
        verify_replicate(
            attempts_dir / replicate_id,
            replicate_id=replicate_id,
            public=False,
        )
        for replicate_id in REPLICATE_IDS
    ]
    if sum(state["status"] == "complete" for state in states) != len(REPLICATE_IDS):
        raise RealMLXExperimentError("aggregate requires all 6 attempts complete")
    _verify_run_ledger(ledger, attempts_dir=attempts_dir)
    if not _aggregate_eligibility_from_rows(_parse_jsonl(ledger)):
        raise RealMLXExperimentError("aggregate requires all 6 attempts complete")
    output_dir.mkdir(parents=True)
    (output_dir / "replicates").mkdir()
    try:
        for replicate_id in REPLICATE_IDS:
            _copy_replicate_data_only(
                attempts_dir / replicate_id,
                output_dir / "replicates" / replicate_id,
            )
        results = _public_results_from_private(
            output_dir / "replicates", data_only_bundle=True
        )
        _verify_public_results(
            results,
            complete_replicate_ids=_complete_attempt_ids(attempts_dir),
        )
        results_digest = _digest_bytes(_json_bytes(results))
        _verify_run_ledger(
            ledger,
            attempts_dir=attempts_dir,
            expected_results_digest=results_digest,
        )
        _write_json(output_dir / "results.json", results)
        _write_sanitized_run_ledger(
            ledger,
            output_dir / "run-ledger.jsonl",
            attempts_dir=output_dir / "replicates",
        )
        _write_derived(output_dir, public=False)
        return verify_aggregate(output_dir, public=False)
    except Exception:
        shutil.rmtree(output_dir)
        raise


def _sanitize_stages(
    source: Path, destination: Path, records: Sequence[RequestEvidence]
) -> None:
    request_ids = {
        record.spec.request_id: f"request-{index:04d}"
        for index, record in enumerate(records)
    }
    rows = _parse_jsonl(source)
    redacted: list[dict[str, Any]] = []
    for row in rows:
        row = json.loads(json.dumps(row))
        request_id = row.get("request_id")
        row["request_id"] = None if request_id is None else request_ids.get(request_id)
        redacted.append(row)
    atomic_write_text(destination, "".join(_json_line(row) for row in redacted))


def sanitize_aggregate(source: Path, destination: Path) -> dict[str, Any]:
    """Create the public aggregate by applying standard and supplemental redaction."""

    verify_aggregate(source, public=False)
    if destination.exists():
        raise RealMLXExperimentError("public aggregate output already exists")
    destination.mkdir(parents=True)
    (destination / "replicates").mkdir()
    try:
        for replicate_id in REPLICATE_IDS:
            source_rep = source / "replicates" / replicate_id
            target_rep = destination / "replicates" / replicate_id
            target_rep.mkdir()
            attempt = _safe_object(source_rep / "attempt.json")
            if attempt["status"] == "failed":
                _write_json(target_rep / "attempt.json", attempt)
                shutil.copyfile(
                    source_rep / "teardown.json", target_rep / "teardown.json"
                )
                continue
            manifest, records = read_bundle(source_rep / "bundle", data_only=True)
            redacted_manifest, redacted_records = sanitize_bundle_records(
                manifest, records
            )
            write_bundle(
                target_rep / "bundle",
                redacted_manifest,
                redacted_records,
                data_only=True,
            )
            verify_bundle(target_rep / "bundle", data_only=True)
            _write_json(target_rep / "attempt.json", attempt)
            _write_json(
                target_rep / "workload-binding.json",
                {
                    "schema_version": "1",
                    "frozen_workload_digest": attempt["frozen_workload_digest"],
                    "frozen_lane_digests": attempt["frozen_lane_digests"],
                    "lane_request_counts": attempt["lane_request_counts"],
                    "request_specs_digest": _digest_bytes(
                        _json_bytes(
                            [
                                record.spec.to_dict(include_tokens=False)
                                for record in redacted_records
                            ]
                        )
                    ),
                },
            )
            _write_json(
                target_rep / "environment.json",
                _common_environment_value(
                    _safe_object(source_rep / "environment.json")
                ),
            )
            shutil.copyfile(source_rep / "teardown.json", target_rep / "teardown.json")
            _sanitize_stages(
                source_rep / "stages.jsonl",
                target_rep / "stages.jsonl",
                records,
            )
        shutil.copyfile(source / "results.json", destination / "results.json")
        _write_sanitized_run_ledger(
            source / "run-ledger.jsonl",
            destination / "run-ledger.jsonl",
            attempts_dir=destination / "replicates",
        )
        _write_derived(destination, public=True)
        return verify_aggregate(destination, public=True)
    except Exception:
        shutil.rmtree(destination)
        raise


def _verify_recursive_checksums(root: Path) -> None:
    lines = (root / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    found: dict[str, str] = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._/-]+)", line)
        if match is None:
            raise RealMLXExperimentError("invalid recursive checksum")
        digest, name = match.groups()
        if name in found or name == "SHA256SUMS" or ".." in Path(name).parts:
            raise RealMLXExperimentError("invalid recursive checksum path")
        found[name] = digest
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path != root / "SHA256SUMS"
    }
    if set(found) != actual:
        raise RealMLXExperimentError("recursive checksum allowlist mismatch")
    for name, digest in found.items():
        path = root / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RealMLXExperimentError(f"recursive checksum mismatch: {name}")


_SCRIPT_SUFFIXES = {
    ".bash",
    ".bat",
    ".cjs",
    ".cmd",
    ".command",
    ".fish",
    ".js",
    ".jsx",
    ".mjs",
    ".pl",
    ".ps1",
    ".py",
    ".pyc",
    ".pyo",
    ".rb",
    ".sh",
    ".ts",
    ".tsx",
    ".zsh",
}


def _verify_data_only_aggregate(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RealMLXExperimentError("aggregate contains a symlink")
        if not path.is_file():
            continue
        if path.suffix.casefold() in _SCRIPT_SUFFIXES:
            raise RealMLXExperimentError(
                f"aggregate contains a script-like file: {path.name}"
            )
        if path.stat().st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH):
            raise RealMLXExperimentError(
                f"aggregate contains an executable file: {path.name}"
            )
        if path.read_bytes()[:2] == b"#!":
            raise RealMLXExperimentError(
                f"aggregate contains a script-like file: {path.name}"
            )


def _verify_public_aggregate_privacy(root: Path) -> None:
    _verify_data_only_aggregate(root)
    for path in root.rglob("*"):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        text = path.read_text(encoding="utf-8")
        for pattern, label in PRIVACY_PATTERNS:
            if pattern.search(text):
                raise RealMLXExperimentError(
                    f"public aggregate contains {label}: {path.name}"
                )
        if '"input_token_ids":[' in text or '"output_token_ids":[' in text:
            raise RealMLXExperimentError("public aggregate contains exact token arrays")
        if path.suffix in {".json", ".jsonl"}:
            values = (
                [json.loads(line) for line in text.splitlines()]
                if path.suffix == ".jsonl"
                else [json.loads(text)]
            )
            for value in values:
                _scan_private_keys(value)


def verify_aggregate(root: Path, *, public: bool | None = None) -> dict[str, Any]:
    """Verify allowlists, nested bundles, derivations, checksums, and privacy."""

    if root.is_symlink() or not root.is_dir():
        raise RealMLXExperimentError("aggregate must be a regular directory")
    if {item.name for item in root.iterdir()} != _AGGREGATE_FILES:
        raise RealMLXExperimentError("aggregate root allowlist mismatch")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise RealMLXExperimentError("aggregate contains a symlink")
    _verify_data_only_aggregate(root)
    _verify_recursive_checksums(root)
    results = _safe_object(root / "results.json")
    _verify_public_results(results)
    results_digest = _digest_bytes(_json_bytes(results))
    run_binding = _verify_run_ledger(
        root / "run-ledger.jsonl",
        attempts_dir=root / "replicates",
        expected_results_digest=results_digest,
    )
    complete_ids = _complete_attempt_ids(root / "replicates")
    _verify_public_results(results, complete_replicate_ids=complete_ids)
    contract = _safe_object(root / "experiment-contract.json")
    mode = contract.get("publication_mode")
    inferred_public = mode == "public_redacted"
    if public is not None and public != inferred_public:
        raise RealMLXExperimentError("aggregate publication mode mismatch")
    expected_contract = _experiment_contract(
        inferred_public,
        binding=run_binding,
        results_digest=results_digest,
    )
    if contract != expected_contract:
        raise RealMLXExperimentError("experiment contract mismatch")
    if {item.name for item in (root / "replicates").iterdir()} != set(REPLICATE_IDS):
        raise RealMLXExperimentError("aggregate replicate allowlist mismatch")
    index = _replicate_index(
        root,
        public=inferred_public,
        run_binding=run_binding,
    )
    if not inferred_public and results != _public_results_from_private(
        root / "replicates", data_only_bundle=True
    ):
        raise RealMLXExperimentError(
            "public results do not reproduce verified private records"
        )
    _verify_public_result_record_bindings(
        results,
        root / "replicates",
        public=inferred_public,
    )
    descriptive_summary = _descriptive_summary(
        root / "replicates",
        public=inferred_public,
    )
    expected_files: dict[str, str] = {
        "experiment-contract.json": canonical_json(expected_contract),
        "environment.json": canonical_json(_common_environment(root / "replicates")),
        "replicate-index.json": canonical_json(index),
        "claim-matrix.json": canonical_json(_claim_matrix(index)),
        "descriptive-summary.json": canonical_json(descriptive_summary),
        "results.json": canonical_json(results),
        "summary.json": canonical_json(_summary(index, public=inferred_public)),
        "report.html": _aggregate_html(_summary(index, public=inferred_public)),
        "reuse-alignment.svg": _reuse_svg(index),
        "timing-memory.svg": _timing_svg(index),
        "teardown.json": canonical_json(
            {
                "schema_version": "1",
                "replicate_records_verified": 6,
                "complete_teardown_records": index["complete_replicates"],
                "scope": "aggregate_of_child_teardown_records",
            }
        ),
    }
    for name, expected in expected_files.items():
        if (root / name).read_text(encoding="utf-8") != expected:
            raise RealMLXExperimentError(f"derived aggregate artifact mismatch: {name}")
    if inferred_public:
        _verify_public_aggregate_privacy(root)
    return {
        "verified": True,
        "publication_mode": mode,
        "attempted_replicates": 6,
        "complete_replicates": index["complete_replicates"],
    }


def _command_text(argv: Sequence[str]) -> str | None:
    try:
        result = subprocess.run(
            list(argv),
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _vm_stat_available_ratio() -> float | None:
    output = _command_text(("/usr/bin/vm_stat",))
    total_text = _command_text(("/usr/sbin/sysctl", "-n", "hw.memsize"))
    if output is None or total_text is None or not total_text.isdigit():
        return None
    page_match = re.search(r"page size of ([0-9]+) bytes", output)
    if page_match is None:
        return None
    pages: dict[str, int] = {}
    for label, raw in re.findall(
        r"^Pages ([^:]+):\s*([0-9]+)\.$", output, re.MULTILINE
    ):
        pages[label] = int(raw)
    required = ("free", "inactive", "speculative")
    if any(label not in pages for label in required):
        return None
    available_bytes = sum(pages[label] for label in required) * int(page_match.group(1))
    total_bytes = int(total_text)
    if total_bytes <= 0:
        return None
    return min(1.0, available_bytes / total_bytes)


def _host_identity() -> tuple[str | None, int | None]:
    chip = _command_text(("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"))
    memory = _command_text(("/usr/sbin/sysctl", "-n", "hw.memsize"))
    return chip, int(memory) if memory is not None and memory.isdigit() else None


def _process_inventory() -> list[tuple[int, int, int]] | None:
    output = _command_text(("/bin/ps", "-axo", "pid=,ppid=,rss="))
    if output is None:
        return None
    rows: list[tuple[int, int, int]] = []
    for line in output.splitlines():
        parts = line.strip().split()
        if len(parts) != 3:
            return None
        try:
            pid, parent, rss_kib = map(int, parts)
        except ValueError:
            return None
        rows.append((pid, parent, rss_kib * 1024))
    return rows


def _excluded_process_tree(
    rows: Sequence[tuple[int, int, int]], roots: set[int]
) -> set[int]:
    excluded = set(roots)
    changed = True
    while changed:
        changed = False
        for pid, parent, _ in rows:
            if parent in excluded and pid not in excluded:
                excluded.add(pid)
                changed = True
    return excluded


def _heavy_process_summary(
    excluded_roots: set[int],
) -> tuple[int, tuple[str, ...]] | None:
    rows = _process_inventory()
    if rows is None:
        return None
    excluded = _excluded_process_tree(rows, excluded_roots)
    count = 0
    for pid, _, rss_bytes in rows:
        if pid in excluded or rss_bytes < HEAVY_PROCESS_RSS_BYTES:
            continue
        count += 1
    return count, (() if count == 0 else ("other_large_process",))


def _machine_observation(
    workspace: Path,
    *,
    excluded_roots: set[int],
) -> dict[str, Any]:
    chip, total_memory = _host_identity()
    process_summary = _heavy_process_summary(excluded_roots)
    try:
        disk_free = shutil.disk_usage(workspace).free
    except OSError:
        disk_free = None
    return {
        "chip": chip,
        "total_memory_bytes": total_memory,
        "vm_stat_available_ratio": _vm_stat_available_ratio(),
        "system_swap_used_bytes": _system_swap_used_bytes(),
        "disk_free_bytes": disk_free,
        "heavy_process_count": (
            None if process_summary is None else process_summary[0]
        ),
        "heavy_process_categories": (
            None if process_summary is None else list(process_summary[1])
        ),
        "scopes": {
            "memory": "system_wide_vm_stat",
            "swap": "system_wide",
            "disk": "output_workspace_filesystem",
            "processes": "host_process_table_excluding_supervisor_and_child_tree",
        },
    }


def _machine_policy_reason(
    observation: Mapping[str, Any], *, preflight: bool
) -> str | None:
    chip = observation["chip"]
    total_memory = observation["total_memory_bytes"]
    available = observation["vm_stat_available_ratio"]
    swap = observation["system_swap_used_bytes"]
    disk = observation["disk_free_bytes"]
    heavy = observation["heavy_process_count"]
    if preflight and chip != REQUIRED_HOST_CHIP:
        return "preflight_host_chip_mismatch"
    if preflight and total_memory != REQUIRED_HOST_MEMORY_BYTES:
        return "preflight_host_memory_mismatch"
    if not isinstance(available, (int, float)) or isinstance(available, bool):
        return (
            "preflight_memory_unavailable"
            if preflight
            else "runtime_memory_unavailable"
        )
    minimum_available = (
        PREFLIGHT_MIN_AVAILABLE_RATIO if preflight else RUNTIME_MIN_AVAILABLE_RATIO
    )
    if available < minimum_available:
        return (
            "preflight_memory_below_floor"
            if preflight
            else "runtime_memory_below_floor"
        )
    if isinstance(swap, bool) or not isinstance(swap, int):
        return "preflight_swap_unavailable" if preflight else "runtime_swap_unavailable"
    maximum_swap = PREFLIGHT_MAX_SWAP_BYTES if preflight else MAX_SWAP_BYTES
    if swap > maximum_swap:
        return (
            "preflight_swap_above_ceiling"
            if preflight
            else "runtime_swap_above_ceiling"
        )
    if isinstance(disk, bool) or not isinstance(disk, int):
        return "preflight_disk_unavailable" if preflight else "runtime_disk_unavailable"
    minimum_disk = PREFLIGHT_MIN_DISK_BYTES if preflight else RUNTIME_MIN_DISK_BYTES
    if disk < minimum_disk:
        return "preflight_disk_below_floor" if preflight else "runtime_disk_below_floor"
    if isinstance(heavy, bool) or not isinstance(heavy, int):
        return (
            "preflight_process_inventory_unavailable"
            if preflight
            else "runtime_process_inventory_unavailable"
        )
    if heavy:
        return (
            "preflight_heavy_process_present"
            if preflight
            else "runtime_heavy_process_present"
        )
    return None


def _offline_child_environment(
    instance_id: str, *, private_temp_root: Path | None = None
) -> dict[str, str]:
    if re.fullmatch(r"[0-9a-f]{32}", instance_id) is None:
        raise RealMLXExperimentError("run instance ID is invalid")
    environment = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "WANDB_MODE": "offline",
        "WANDB_DISABLED": "true",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONSAFEPATH": "1",
        RUN_INSTANCE_ENV: instance_id,
    }
    if private_temp_root is not None:
        resolved = private_temp_root.resolve(strict=True)
        if (
            private_temp_root.is_symlink()
            or not resolved.is_dir()
            or resolved != private_temp_root.absolute()
        ):
            raise RealMLXExperimentError("private child temp root is unsafe")
        environment["TMPDIR"] = str(resolved)
    return environment


def _trusted_bootstrap_command(*args: str) -> list[str]:
    bootstrap = _TRUSTED_BOOTSTRAP
    if (
        bootstrap.is_symlink()
        or not bootstrap.is_file()
        or bootstrap.resolve(strict=True) != bootstrap.absolute()
    ):
        raise RealMLXExperimentError("trusted bootstrap script is unavailable")
    return [sys.executable, "-I", "-S", str(bootstrap), *args]


def _replicate_child_command(
    *,
    workload: Path,
    model_dir: Path,
    conversion_summary: Path,
    replicate_id: str,
    output_dir: Path,
    expected_commit: str,
    runtime_packages_digest: str,
) -> list[str]:
    sandbox = Path("/usr/bin/sandbox-exec")
    if not sandbox.is_file() or not os.access(sandbox, os.X_OK):
        raise RealMLXExperimentError("macOS sandbox-exec is unavailable")
    return [
        str(sandbox),
        "-p",
        SANDBOX_POLICY,
        *_trusted_bootstrap_command("replicate"),
        "--workload",
        str(workload),
        "--model-dir",
        str(model_dir),
        "--replicate-id",
        replicate_id,
        "--output-dir",
        str(output_dir),
        "--conversion-summary",
        str(conversion_summary),
        "--expected-commit",
        expected_commit,
        "--runtime-packages-digest",
        runtime_packages_digest,
    ]


def _git_path_is_tracked(path: Path) -> bool:
    try:
        relative = path.relative_to(_PROJECT_ROOT).as_posix()
    except ValueError:
        return False
    result = subprocess.run(
        [
            "git",
            "-C",
            str(_PROJECT_ROOT),
            "ls-files",
            "--error-unmatch",
            "--",
            relative,
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.returncode == 0


def _reject_import_shadows(output_workspace: Path) -> None:
    roots = (_PROJECT_ROOT, output_workspace.parent)
    checked: set[Path] = set()
    for root in roots:
        for name in _IMPORT_SHADOW_CANDIDATES:
            candidate = root / name
            if candidate in checked:
                continue
            checked.add(candidate)
            if not (candidate.exists() or candidate.is_symlink()):
                continue
            if root == _PROJECT_ROOT and _git_path_is_tracked(candidate):
                continue
            raise RealMLXExperimentError("unsafe top-level import shadow candidate")


def _validate_probe_module_origin(output_workspace: Path) -> None:
    spec = importlib.util.find_spec("llmtracefx.cache_audit.real_mlx")
    if spec is None or spec.origin is None:
        raise RealMLXExperimentError("real_mlx module origin is unavailable")
    try:
        origin = Path(spec.origin).resolve(strict=True)
        expected = Path(__file__).resolve(strict=True)
    except OSError as exc:
        raise RealMLXExperimentError("real_mlx module origin is unavailable") from exc
    if (
        origin != expected
        or not _is_relative_to(origin, _INSTALLED_PROJECT_ROOT)
        or _is_relative_to(origin, output_workspace)
        or _distribution_version("llmtracefx") != REQUIRED_LLMTRACEFX_VERSION
    ):
        raise RealMLXExperimentError("real_mlx module origin is untrusted")


def _ensure_non_repository_cwd(path: Path) -> None:
    if not path.is_dir() or _is_relative_to(path, _PROJECT_ROOT):
        raise RealMLXExperimentError(
            "isolated child cwd must be outside the repository"
        )
    if any((candidate / ".git").exists() for candidate in (path, *path.parents)):
        raise RealMLXExperimentError("isolated child cwd must be non-repository")


def _sandbox_probe_command(
    *,
    output_workspace: Path,
    expected_commit: str,
    expected_package_digest: str,
) -> list[str]:
    sandbox = Path("/usr/bin/sandbox-exec")
    if not sandbox.is_file() or not os.access(sandbox, os.X_OK):
        raise RealMLXExperimentError("macOS sandbox-exec is unavailable")
    return [
        str(sandbox),
        "-p",
        SANDBOX_POLICY,
        *_trusted_bootstrap_command("sandbox-probe"),
        "--output-workspace",
        str(output_workspace),
        "--expected-commit",
        expected_commit,
        "--expected-package-digest",
        expected_package_digest,
    ]


def _sandbox_probe(
    *,
    output_workspace: Path,
    expected_commit: str,
    expected_package_digest: str,
) -> dict[str, Any]:
    if sys.flags.isolated != 1:
        raise RealMLXExperimentError("sandbox probe requires isolated Python mode")
    _reject_import_shadows(output_workspace)
    _validate_probe_module_origin(output_workspace)
    runtime_packages = _runtime_package_identity(output_workspace)
    package_digest = _validate_supervisor_source(expected_commit)
    if package_digest != expected_package_digest:
        raise RealMLXExperimentError("sandbox probe package source digest mismatch")
    if _current_rss_bytes() is None:
        raise RealMLXExperimentError("sandbox probe current RSS query failed")
    if _system_swap_used_bytes() is None:
        raise RealMLXExperimentError("sandbox probe sysctl query failed")
    if _system_memory_free_percent() is None:
        raise RealMLXExperimentError("sandbox probe memory-pressure query failed")
    return {
        "sandbox_probe_passed": True,
        "runtime_packages": runtime_packages,
    }


def _run_exact_sandbox_probe(
    *,
    output_workspace: Path,
    expected_commit: str,
    expected_package_digest: str,
) -> dict[str, dict[str, str | int]]:
    cwd = output_workspace.parent
    _ensure_non_repository_cwd(cwd)
    command = _sandbox_probe_command(
        output_workspace=output_workspace,
        expected_commit=expected_commit,
        expected_package_digest=expected_package_digest,
    )
    private_temp_root = (
        output_workspace.parent / f".{output_workspace.name}.preflight-tmp"
    )
    private_temp_root.mkdir(mode=0o700)
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            check=False,
            text=True,
            timeout=60,
            env=_offline_child_environment(
                secrets.token_hex(16), private_temp_root=private_temp_root
            ),
            cwd=cwd,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RealMLXExperimentError("exact sandbox probe failed") from exc
    finally:
        shutil.rmtree(private_temp_root, ignore_errors=False)
    if result.returncode != 0 or len(result.stdout) > MAX_CHILD_LOG_BYTES:
        raise RealMLXExperimentError("exact sandbox probe failed")
    try:
        payload = json.loads(result.stdout)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RealMLXExperimentError(
            "exact sandbox probe returned invalid output"
        ) from exc
    if not isinstance(payload, dict) or set(payload) != {
        "runtime_packages",
        "sandbox_probe_passed",
    }:
        raise RealMLXExperimentError("exact sandbox probe returned invalid output")
    if payload["sandbox_probe_passed"] is not True:
        raise RealMLXExperimentError("exact sandbox probe failed")
    _verify_runtime_package_identity(payload["runtime_packages"])
    return {
        str(name): {str(key): value for key, value in entry.items()}
        for name, entry in payload["runtime_packages"].items()
    }


def _validate_supervisor_source(expected_commit: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise RealMLXExperimentError("expected commit must be a full lowercase SHA")
    _reject_import_shadows(_PROJECT_ROOT / ".source-validation-output")
    if _PROJECT_ROOT == _INSTALLED_PROJECT_ROOT:
        commit, commit_at = source_commit()
    else:
        commit_result = subprocess.run(
            [
                "git",
                "-C",
                str(_PROJECT_ROOT),
                "show",
                "-s",
                "--format=%H%n%cI",
                "HEAD",
            ],
            capture_output=True,
            check=False,
            text=True,
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
        )
        lines = commit_result.stdout.strip().splitlines()
        commit, commit_at = (
            (lines[0], lines[1])
            if commit_result.returncode == 0 and len(lines) == 2
            else (None, None)
        )
    if commit != expected_commit or commit_at is None:
        raise RealMLXExperimentError("current Git HEAD does not match expected commit")
    status = subprocess.run(
        [
            "git",
            "-C",
            str(_PROJECT_ROOT),
            "status",
            "--porcelain=v1",
            "--untracked-files=no",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    if status.returncode != 0 or status.stdout:
        raise RealMLXExperimentError("tracked worktree must be clean")
    package_digest = package_source_digest()
    if _git_package_digest(_PROJECT_ROOT, expected_commit) != package_digest:
        raise RealMLXExperimentError(
            "installed package source digest does not match expected commit tree"
        )
    try:
        commit_time = datetime.fromisoformat(commit_at.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RealMLXExperimentError("expected commit timestamp is invalid") from exc
    if commit_time.tzinfo is None or commit_time > datetime.now(timezone.utc):
        raise RealMLXExperimentError("expected commit chronology is invalid")
    return package_digest


def _resolve_existing_file(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise RealMLXExperimentError(f"{label} must not be a symlink")
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise RealMLXExperimentError(f"{label} must exist") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise RealMLXExperimentError(f"{label} must be a regular file")
    return resolved


def _resolve_existing_directory(path: Path, label: str) -> Path:
    if path.is_symlink():
        raise RealMLXExperimentError(f"{label} must not be a symlink")
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise RealMLXExperimentError(f"{label} must exist") from exc
    if not resolved.is_dir() or resolved.is_symlink():
        raise RealMLXExperimentError(f"{label} must be a regular directory")
    return resolved


def _resolve_new_directory(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.exists() or expanded.is_symlink():
        raise RealMLXExperimentError(f"{label} already exists")
    try:
        parent = expanded.parent.resolve(strict=True)
    except OSError as exc:
        raise RealMLXExperimentError(f"{label} parent must exist") from exc
    if not parent.is_dir():
        raise RealMLXExperimentError(f"{label} parent must be a directory")
    return parent / expanded.name


def _resolve_new_file(path: Path, label: str) -> Path:
    expanded = path.expanduser()
    if expanded.exists() or expanded.is_symlink():
        raise RealMLXExperimentError(f"{label} already exists")
    try:
        parent = expanded.parent.resolve(strict=True)
    except OSError as exc:
        raise RealMLXExperimentError(f"{label} parent must exist") from exc
    if not parent.is_dir():
        raise RealMLXExperimentError(f"{label} parent must be a directory")
    return parent / expanded.name


def _resolve_run_all_paths(
    *,
    workload: Path,
    model_dir: Path,
    conversion_summary: Path,
    output_workspace: Path,
) -> tuple[Path, Path, Path, Path]:
    return (
        _resolve_existing_file(workload, "workload"),
        _resolve_existing_directory(model_dir, "model directory"),
        _resolve_existing_file(conversion_summary, "conversion summary"),
        _resolve_new_directory(output_workspace, "run-all output workspace"),
    )


@dataclass(frozen=True)
class _RunPreflight:
    workload_path: Path
    model_dir: Path
    conversion_summary: Path
    output_workspace: Path
    package_digest: str
    workload: FrozenMLXWorkload
    runtime_packages: Mapping[str, Mapping[str, str | int]]
    machine_observation: Mapping[str, Any]


def _run_global_preflight(
    *,
    workload: Path,
    model_dir: Path,
    conversion_summary: Path,
    output_workspace: Path,
    expected_commit: str,
) -> _RunPreflight:
    (
        workload_path,
        resolved_model_dir,
        resolved_conversion_summary,
        resolved_output_workspace,
    ) = _resolve_run_all_paths(
        workload=workload,
        model_dir=model_dir,
        conversion_summary=conversion_summary,
        output_workspace=output_workspace,
    )
    _ensure_non_repository_cwd(resolved_output_workspace.parent)
    _reject_import_shadows(resolved_output_workspace)
    package_digest = _validate_supervisor_source(expected_commit)
    calibrated_workload = load_workload(workload_path, calibrated=True)
    model_digest = verify_model_contract(
        resolved_model_dir,
        resolved_conversion_summary,
    )
    if model_digest != EXPECTED_MODEL_ARTIFACT_DIGEST:
        raise RealMLXExperimentError("run-all model artifact digest mismatch")
    runtime_packages = _runtime_package_identity(resolved_output_workspace)
    sandbox_runtime_packages = _run_exact_sandbox_probe(
        output_workspace=resolved_output_workspace,
        expected_commit=expected_commit,
        expected_package_digest=package_digest,
    )
    if sandbox_runtime_packages != runtime_packages:
        raise RealMLXExperimentError(
            "sandbox runtime package identity does not match parent"
        )
    observation = _machine_observation(
        resolved_output_workspace.parent,
        excluded_roots={os.getpid()},
    )
    reason = _machine_policy_reason(observation, preflight=True)
    if reason is not None:
        raise RealMLXExperimentError(f"NEEDS_CLEAN_BOOT:{reason}")
    return _RunPreflight(
        workload_path=workload_path,
        model_dir=resolved_model_dir,
        conversion_summary=resolved_conversion_summary,
        output_workspace=resolved_output_workspace,
        package_digest=package_digest,
        workload=calibrated_workload,
        runtime_packages=runtime_packages,
        machine_observation=observation,
    )


def run_preflight(
    *,
    workload: Path,
    model_dir: Path,
    conversion_summary: Path,
    output_workspace: Path,
    expected_commit: str,
    output: Path,
) -> dict[str, Any]:
    """Perform canonical pre-creation gates and write one noncanonical receipt."""

    receipt_path = _resolve_new_file(output, "preflight output")
    try:
        state = _run_global_preflight(
            workload=workload,
            model_dir=model_dir,
            conversion_summary=conversion_summary,
            output_workspace=output_workspace,
            expected_commit=expected_commit,
        )
    except (
        CacheAuditBundleError,
        OSError,
        RealMLXExperimentError,
        RuntimeError,
        ValueError,
    ) as exc:
        message = str(exc)
        reason = (
            message
            if message.startswith("NEEDS_CLEAN_BOOT:")
            else "PREFLIGHT_VALIDATION_FAILED"
        )
        receipt = {
            "schema_version": "1",
            "canonical_execution": False,
            "status": "refused",
            "reason": reason,
        }
    else:
        receipt = {
            "schema_version": "1",
            "canonical_execution": False,
            "status": "passed",
            "reason": None,
            "checks": {
                "source": True,
                "model": True,
                "workload": True,
                "sandbox": True,
                "global_machine": True,
            },
            "machine_observation": dict(state.machine_observation),
        }
    _write_json(receipt_path, receipt)
    return {
        "preflight_passed": receipt["status"] == "passed",
        "reason": receipt["reason"],
    }


def _run_binding(
    *,
    expected_commit: str,
    package_digest: str,
    workload: FrozenMLXWorkload,
    runtime_packages: Mapping[str, Any],
) -> dict[str, Any]:
    if re.fullmatch(r"sha256:[0-9a-f]{64}", package_digest) is None:
        raise RealMLXExperimentError("package source digest is invalid")
    _verify_runtime_package_identity(runtime_packages)
    return {
        "expected_commit": expected_commit,
        "generator_package_digest": package_digest,
        "runtime_packages": {
            name: dict(entry) for name, entry in runtime_packages.items()
        },
        "frozen_workload_digest": workload.to_dict()["workload_digest"],
        "frozen_lane_digests": {
            lane.lane_id: lane.to_dict()["lane_digest"] for lane in workload.lanes
        },
        "model_artifact_digest": EXPECTED_MODEL_ARTIFACT_DIGEST,
        "conversion_summary_sha256": ("sha256:" + EXPECTED_CONVERSION_SUMMARY_SHA256),
        "sandbox_policy_digest": SANDBOX_POLICY_DIGEST,
        "planned_replicate_ids": list(REPLICATE_IDS),
    }


def _create_run_ledger(path: Path, binding: Mapping[str, Any]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        run_start = {
            "schema_version": "2",
            "sequence": 0,
            "timestamp": _utc_now(),
            "event": "run-start",
            "status": "running",
            "reason": None,
            **dict(binding),
        }
        os.write(descriptor, _json_line(run_start).encode("ascii"))
        for sequence, replicate_id in enumerate(REPLICATE_IDS, start=1):
            row = {
                "schema_version": "2",
                "sequence": sequence,
                "timestamp": _utc_now(),
                "replicate_id": replicate_id,
                "event": "planned",
                "status": "planned",
                "reason": None,
            }
            os.write(descriptor, _json_line(row).encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_run_ledger(path: Path, row: Mapping[str, Any]) -> None:
    if path.is_symlink() or not path.is_file():
        raise RealMLXExperimentError("run ledger must be a regular file")
    flags = os.O_WRONLY | os.O_APPEND
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        os.write(descriptor, _json_line(dict(row)).encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _valid_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value.endswith("Z"):
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _directory_digest(directory: Path) -> str:
    if directory.is_symlink() or not directory.is_dir():
        raise RealMLXExperimentError("attempt digest source must be a directory")
    digest = hashlib.sha256()
    paths = sorted(path for path in directory.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in directory.rglob("*")):
        raise RealMLXExperimentError("attempt digest source contains a symlink")
    for path in paths:
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return "sha256:" + digest.hexdigest()


def _verify_machine_observation(
    value: Any,
    *,
    preflight: bool,
    status: str,
    reason: str | None,
) -> None:
    if not isinstance(value, dict):
        raise RealMLXExperimentError("run ledger machine observation is invalid")
    _exact_keys(
        value,
        {
            "chip",
            "total_memory_bytes",
            "vm_stat_available_ratio",
            "system_swap_used_bytes",
            "disk_free_bytes",
            "heavy_process_count",
            "heavy_process_categories",
            "scopes",
        },
        "run ledger machine observation",
    )
    if value["chip"] is not None and not isinstance(value["chip"], str):
        raise RealMLXExperimentError("run ledger chip observation is invalid")
    for key in (
        "total_memory_bytes",
        "system_swap_used_bytes",
        "disk_free_bytes",
        "heavy_process_count",
    ):
        observed = value[key]
        if observed is not None and (
            isinstance(observed, bool) or not isinstance(observed, int) or observed < 0
        ):
            raise RealMLXExperimentError("run ledger integer observation is invalid")
    available = value["vm_stat_available_ratio"]
    if available is not None and (
        isinstance(available, bool)
        or not isinstance(available, (int, float))
        or not math.isfinite(available)
        or not 0 <= available <= 1
    ):
        raise RealMLXExperimentError("run ledger memory-ratio observation is invalid")
    categories = value["heavy_process_categories"]
    count = value["heavy_process_count"]
    if categories is not None and (
        not isinstance(categories, list)
        or any(not isinstance(item, str) or not item for item in categories)
        or categories != sorted(set(categories))
        or (count == 0 and categories)
        or (
            isinstance(count, int)
            and count > 0
            and categories != ["other_large_process"]
        )
    ):
        raise RealMLXExperimentError("run ledger process observation is invalid")
    if (categories is None) != (count is None):
        raise RealMLXExperimentError("run ledger process observation is incomplete")
    if value["scopes"] != {
        "memory": "system_wide_vm_stat",
        "swap": "system_wide",
        "disk": "output_workspace_filesystem",
        "processes": "host_process_table_excluding_supervisor_and_child_tree",
    }:
        raise RealMLXExperimentError("run ledger observation scopes are invalid")
    policy_reason = _machine_policy_reason(value, preflight=preflight)
    if status == "passed":
        if reason is not None or policy_reason is not None:
            raise RealMLXExperimentError(
                "passed run ledger observation violates policy"
            )
    elif status == "failed":
        if reason is None or reason != policy_reason:
            raise RealMLXExperimentError(
                "failed run ledger observation does not match policy"
            )
    elif status == "not_started":
        if not isinstance(reason, str) or not reason:
            raise RealMLXExperimentError(
                "not-started run ledger observation lacks a reason"
            )
    else:
        raise RealMLXExperimentError("run ledger observation status is invalid")


def _finite_elapsed(value: Any, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise RealMLXExperimentError(f"{context} elapsed time is invalid")
    return float(value)


def _aggregate_eligibility_from_rows(rows: Sequence[Mapping[str, Any]]) -> bool:
    finalized = {
        str(row.get("replicate_id")): row
        for row in rows
        if row.get("event") == "finalized" and row.get("replicate_id") in REPLICATE_IDS
    }
    if set(finalized) != set(REPLICATE_IDS):
        return False
    return all(row.get("status") == "complete" for row in finalized.values())


def _verify_run_ledger(
    path: Path,
    *,
    attempts_dir: Path | None = None,
    expected_binding: Mapping[str, Any] | None = None,
    expected_results_digest: str | None = None,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RealMLXExperimentError("run ledger must be a regular file")
    text = path.read_text(encoding="ascii")
    for pattern, label in PRIVACY_PATTERNS:
        if pattern.search(text):
            raise RealMLXExperimentError(f"run ledger contains {label}")
    rows = _parse_jsonl(path)
    if len(rows) < len(REPLICATE_IDS) * 4 + 2:
        raise RealMLXExperimentError("run ledger is incomplete")
    sequences = [row.get("sequence") for row in rows]
    if any(isinstance(item, bool) or not isinstance(item, int) for item in sequences):
        raise RealMLXExperimentError("run ledger sequence is invalid")
    if sequences != list(range(len(rows))):
        raise RealMLXExperimentError("run ledger sequence is not append-only")
    if (
        sum(row.get("event") == "run-start" for row in rows) != 1
        or sum(row.get("event") == "run-finalized" for row in rows) != 1
        or rows[0].get("event") != "run-start"
        or rows[-1].get("event") != "run-finalized"
    ):
        raise RealMLXExperimentError("run ledger boundary events are invalid")
    timestamps: list[datetime] = []
    for row in rows:
        timestamp = row.get("timestamp")
        if not _valid_utc_timestamp(timestamp):
            raise RealMLXExperimentError("run ledger timestamp is invalid")
        assert isinstance(timestamp, str)
        timestamps.append(datetime.fromisoformat(timestamp.replace("Z", "+00:00")))
    if any(
        right < left for left, right in zip(timestamps, timestamps[1:], strict=False)
    ):
        raise RealMLXExperimentError("run ledger timestamps are out of order")
    binding_keys = {
        "expected_commit",
        "generator_package_digest",
        "runtime_packages",
        "frozen_workload_digest",
        "frozen_lane_digests",
        "model_artifact_digest",
        "conversion_summary_sha256",
        "sandbox_policy_digest",
        "planned_replicate_ids",
    }
    run_start = rows[0]
    _exact_keys(
        run_start,
        {
            "schema_version",
            "sequence",
            "timestamp",
            "event",
            "status",
            "reason",
            *binding_keys,
        },
        "run-start ledger row",
    )
    binding = {key: run_start[key] for key in binding_keys}
    if (
        run_start["schema_version"] != "2"
        or run_start["event"] != "run-start"
        or run_start["status"] != "running"
        or run_start["reason"] is not None
        or re.fullmatch(r"[0-9a-f]{40}", str(binding["expected_commit"])) is None
        or re.fullmatch(
            r"sha256:[0-9a-f]{64}",
            str(binding["generator_package_digest"]),
        )
        is None
        or binding["frozen_workload_digest"] != EXPECTED_CALIBRATED_WORKLOAD_DIGEST
        or binding["frozen_lane_digests"] != EXPECTED_CALIBRATED_LANE_DIGESTS
        or binding["model_artifact_digest"] != EXPECTED_MODEL_ARTIFACT_DIGEST
        or binding["conversion_summary_sha256"]
        != "sha256:" + EXPECTED_CONVERSION_SUMMARY_SHA256
        or binding["sandbox_policy_digest"] != SANDBOX_POLICY_DIGEST
        or binding["planned_replicate_ids"] != list(REPLICATE_IDS)
    ):
        raise RealMLXExperimentError("run ledger binding is invalid")
    _verify_runtime_package_identity(binding["runtime_packages"])
    if expected_binding is not None and binding != dict(expected_binding):
        raise RealMLXExperimentError("run ledger binding does not match run inputs")
    planned = rows[1 : len(REPLICATE_IDS) + 1]
    if Counter(
        row.get("replicate_id") for row in rows if row.get("event") == "planned"
    ) != Counter(dict.fromkeys(REPLICATE_IDS, 1)):
        raise RealMLXExperimentError("run ledger must plan every ID exactly once")
    if [
        (row.get("replicate_id"), row.get("event"), row.get("status"))
        for row in planned
    ] != [(replicate_id, "planned", "planned") for replicate_id in REPLICATE_IDS]:
        raise RealMLXExperimentError("run ledger planned ID set is invalid")
    started_instance_ids: set[str] = set()
    started_instances_by_replicate: dict[str, str] = {}
    final_rows: dict[str, dict[str, Any]] = {}
    for replicate_id in REPLICATE_IDS:
        lifecycle = [row for row in rows if row.get("replicate_id") == replicate_id]
        if not lifecycle or lifecycle[0].get("event") != "planned":
            raise RealMLXExperimentError(
                "run ledger lifecycle does not begin as planned"
            )
        events = [str(row.get("event")) for row in lifecycle]
        if events.count("planned") != 1 or events.count("preflight") != 1:
            raise RealMLXExperimentError(
                "run ledger lifecycle planning/preflight count is invalid"
            )
        if events.count("postflight") != 1 or events.count("finalized") != 1:
            raise RealMLXExperimentError(
                "run ledger lifecycle terminal count is invalid"
            )
        if events[-1] != "finalized":
            raise RealMLXExperimentError(
                "run ledger contains an event after replicate finalization"
            )
        preflight = lifecycle[1]
        _exact_keys(
            preflight,
            {
                "schema_version",
                "sequence",
                "timestamp",
                "replicate_id",
                "event",
                "status",
                "reason",
                "observation",
            },
            "preflight ledger row",
        )
        if preflight["event"] != "preflight":
            raise RealMLXExperimentError("run ledger preflight order is invalid")
        preflight_reason = preflight["reason"]
        external_preflight_failure = preflight_reason in {
            "supervisor_aborted_before_start",
            "total_timeout_before_start",
        }
        if external_preflight_failure:
            if preflight["status"] != "failed":
                raise RealMLXExperimentError(
                    "run ledger external preflight state is invalid"
                )
            _verify_machine_observation(
                preflight["observation"],
                preflight=True,
                status="not_started",
                reason=preflight_reason,
            )
        else:
            _verify_machine_observation(
                preflight["observation"],
                preflight=True,
                status=preflight["status"],
                reason=preflight_reason,
            )

        started_rows = [row for row in lifecycle if row.get("event") == "started"]
        monitor_rows = [row for row in lifecycle if row.get("event") == "monitor"]
        postflight = lifecycle[-2]
        finalized = lifecycle[-1]
        if postflight.get("event") != "postflight":
            raise RealMLXExperimentError("run ledger postflight order is invalid")
        _exact_keys(
            finalized,
            {
                "schema_version",
                "sequence",
                "timestamp",
                "replicate_id",
                "event",
                "status",
                "reason",
                "elapsed_seconds",
                "attempt_digest",
            },
            "finalized ledger row",
        )
        finalized_elapsed = _finite_elapsed(finalized["elapsed_seconds"], "finalized")
        if (
            finalized["schema_version"] != "2"
            or finalized["event"] != "finalized"
            or finalized["status"] not in {"complete", "failed"}
            or (finalized["status"] == "complete") != (finalized["reason"] is None)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(finalized["attempt_digest"]))
            is None
        ):
            raise RealMLXExperimentError("run ledger finalization is invalid")

        if started_rows:
            if len(started_rows) != 1 or preflight["status"] != "passed":
                raise RealMLXExperimentError("run ledger started lifecycle is invalid")
            started = started_rows[0]
            _exact_keys(
                started,
                {
                    "schema_version",
                    "sequence",
                    "timestamp",
                    "replicate_id",
                    "event",
                    "status",
                    "reason",
                    "child_instance_id_digest",
                },
                "started ledger row",
            )
            child_instance_id = started["child_instance_id_digest"]
            if (
                started["status"] != "running"
                or started["reason"] is not None
                or re.fullmatch(r"sha256:[0-9a-f]{64}", str(child_instance_id)) is None
                or child_instance_id in started_instance_ids
            ):
                raise RealMLXExperimentError(
                    "run ledger child instance binding is invalid"
                )
            started_instance_ids.add(str(child_instance_id))
            started_instances_by_replicate[replicate_id] = str(child_instance_id)
            expected_events = [
                "planned",
                "preflight",
                "started",
                *(["monitor"] * len(monitor_rows)),
                "postflight",
                "finalized",
            ]
            if events != expected_events:
                raise RealMLXExperimentError(
                    "run ledger started lifecycle order is invalid"
                )
            previous_elapsed = 0.0
            failed_monitors = 0
            failed_monitor_reason: str | None = None
            for index, monitor in enumerate(monitor_rows):
                _exact_keys(
                    monitor,
                    {
                        "schema_version",
                        "sequence",
                        "timestamp",
                        "replicate_id",
                        "event",
                        "status",
                        "reason",
                        "observation",
                        "elapsed_seconds",
                    },
                    "monitor ledger row",
                )
                elapsed = _finite_elapsed(monitor["elapsed_seconds"], "monitor")
                cadence = elapsed - previous_elapsed
                if not (
                    MONITOR_INTERVAL_SECONDS - 0.1
                    <= cadence
                    <= MONITOR_INTERVAL_SECONDS * 2 + 0.1
                    and elapsed < CHILD_TIMEOUT_SECONDS
                ):
                    raise RealMLXExperimentError(
                        "run ledger monitor cadence is invalid"
                    )
                previous_elapsed = elapsed
                _verify_machine_observation(
                    monitor["observation"],
                    preflight=False,
                    status=monitor["status"],
                    reason=monitor["reason"],
                )
                if monitor["status"] == "failed":
                    failed_monitors += 1
                    failed_monitor_reason = str(monitor["reason"])
                    if index != len(monitor_rows) - 1:
                        raise RealMLXExperimentError(
                            "run ledger has events after a failed monitor"
                        )
            if failed_monitors > 1:
                raise RealMLXExperimentError(
                    "run ledger contains multiple failed monitors"
                )
            _exact_keys(
                postflight,
                {
                    "schema_version",
                    "sequence",
                    "timestamp",
                    "replicate_id",
                    "event",
                    "status",
                    "reason",
                    "observation",
                    "elapsed_seconds",
                },
                "postflight ledger row",
            )
            postflight_elapsed = _finite_elapsed(
                postflight["elapsed_seconds"], "postflight"
            )
            if (
                postflight_elapsed < previous_elapsed
                or postflight_elapsed
                > TOTAL_TIMEOUT_SECONDS + 2 * PROCESS_GROUP_GRACE_SECONDS
            ):
                raise RealMLXExperimentError(
                    "run ledger postflight elapsed time is invalid"
                )
            _verify_machine_observation(
                postflight["observation"],
                preflight=False,
                status=postflight["status"],
                reason=postflight["reason"],
            )
            if finalized_elapsed < postflight_elapsed:
                raise RealMLXExperimentError(
                    "run ledger finalization elapsed time regressed"
                )
            if finalized["status"] == "complete" and (
                any(row["status"] != "passed" for row in monitor_rows)
                or postflight["status"] != "passed"
            ):
                raise RealMLXExperimentError(
                    "complete run ledger lifecycle contains a failed gate"
                )
            if finalized["status"] == "failed":
                finalized_reason = finalized["reason"]
                if failed_monitor_reason is not None:
                    if finalized_reason not in {
                        failed_monitor_reason,
                        *_POLICY_SUPERSEDING_REASONS,
                    }:
                        raise RealMLXExperimentError(
                            "run ledger final reason contradicts failed policy gate"
                        )
                elif postflight["status"] == "failed":
                    if finalized_reason not in {
                        postflight["reason"],
                        *_STARTED_TERMINAL_REASONS,
                        *_POLICY_SUPERSEDING_REASONS,
                    }:
                        raise RealMLXExperimentError(
                            "run ledger final reason contradicts failed policy gate"
                        )
                elif finalized_reason not in _STARTED_TERMINAL_REASONS:
                    raise RealMLXExperimentError(
                        "run ledger started terminal reason is invalid"
                    )
        else:
            if monitor_rows or events != [
                "planned",
                "preflight",
                "postflight",
                "finalized",
            ]:
                raise RealMLXExperimentError(
                    "run ledger not-started lifecycle order is invalid"
                )
            _exact_keys(
                postflight,
                {
                    "schema_version",
                    "sequence",
                    "timestamp",
                    "replicate_id",
                    "event",
                    "status",
                    "reason",
                    "observation",
                },
                "not-started postflight ledger row",
            )
            if finalized["status"] != "failed":
                raise RealMLXExperimentError(
                    "run ledger not-started terminal state is invalid"
                )
            if preflight["status"] == "passed":
                if finalized["reason"] not in _NOT_STARTED_TERMINAL_REASONS:
                    raise RealMLXExperimentError(
                        "run ledger not-started terminal state is invalid"
                    )
            elif finalized["reason"] != preflight["reason"]:
                raise RealMLXExperimentError(
                    "run ledger not-started terminal state is invalid"
                )
            if postflight["status"] == "not_started":
                if (
                    not isinstance(postflight["reason"], str)
                    or postflight["reason"] != finalized["reason"]
                ):
                    raise RealMLXExperimentError(
                        "run ledger not-started terminal state is invalid"
                    )
                _verify_machine_observation(
                    postflight["observation"],
                    preflight=False,
                    status="not_started",
                    reason=postflight["reason"],
                )
            elif postflight["status"] == "failed":
                _verify_machine_observation(
                    postflight["observation"],
                    preflight=False,
                    status="failed",
                    reason=postflight["reason"],
                )
            else:
                raise RealMLXExperimentError(
                    "run ledger not-started terminal state is invalid"
                )
        if finalized["status"] == "complete" and len(started_rows) != 1:
            raise RealMLXExperimentError(
                "complete replicate must have exactly one started event"
            )
        final_rows[replicate_id] = finalized
        row = finalized
        _exact_keys(
            row,
            {
                "schema_version",
                "sequence",
                "timestamp",
                "replicate_id",
                "event",
                "status",
                "reason",
                "elapsed_seconds",
                "attempt_digest",
            },
            "finalized ledger row",
        )
        if (
            row["schema_version"] != "2"
            or row["status"] not in {"complete", "failed"}
            or (row["status"] == "complete") != (row["reason"] is None)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", str(row["attempt_digest"])) is None
        ):
            raise RealMLXExperimentError("run ledger finalization is invalid")
        if attempts_dir is not None:
            attempt_dir = attempts_dir / replicate_id
            attempt = _safe_object(attempt_dir / "attempt.json")
            reason = (
                None if attempt.get("status") == "complete" else attempt.get("reason")
            )
            if (
                row["status"] != attempt.get("status")
                or row["reason"] != reason
                or row["attempt_digest"] != _directory_digest(attempt_dir)
            ):
                raise RealMLXExperimentError(
                    "run ledger finalization does not match attempt"
                )
            environment_path = attempt_dir / "environment.json"
            if row["status"] == "complete" and environment_path.is_file():
                environment = _safe_object(environment_path)
                run_instance_id = environment.get("run_instance_id")
                if run_instance_id is not None and (
                    not isinstance(run_instance_id, str)
                    or re.fullmatch(r"[0-9a-f]{32}", run_instance_id) is None
                    or _digest_bytes(run_instance_id.encode("ascii"))
                    != started_instances_by_replicate.get(replicate_id)
                ):
                    raise RealMLXExperimentError(
                        "run ledger child instance does not match attempt"
                    )
    for index, replicate_id in enumerate(REPLICATE_IDS):
        finalized = final_rows[replicate_id]
        if finalized["reason"] not in _ABORT_LATER_REASONS:
            continue
        for later_id in REPLICATE_IDS[index + 1 :]:
            later_lifecycle = [
                row for row in rows if row.get("replicate_id") == later_id
            ]
            later_finalized = final_rows[later_id]
            if (
                any(row.get("event") == "started" for row in later_lifecycle)
                or later_finalized["status"] != "failed"
                or later_finalized["reason"] != "supervisor_aborted_before_start"
            ):
                raise RealMLXExperimentError(
                    "run ledger continued after an aborting terminal reason"
                )
    aborting_finalization_seen = False
    for replicate_id in REPLICATE_IDS:
        reason = final_rows[replicate_id]["reason"]
        if (
            reason == "supervisor_aborted_before_start"
            and not aborting_finalization_seen
        ):
            raise RealMLXExperimentError(
                "run ledger supervisor abort lacks an earlier aborting finalization"
            )
        if reason in _ABORT_LATER_REASONS:
            aborting_finalization_seen = True
    finalized_elapsed_values = [
        _finite_elapsed(row["elapsed_seconds"], "finalized")
        for row in rows
        if row.get("event") == "finalized"
    ]
    if any(
        right < left
        for left, right in zip(
            finalized_elapsed_values,
            finalized_elapsed_values[1:],
            strict=False,
        )
    ) or (
        finalized_elapsed_values
        and finalized_elapsed_values[-1] > TOTAL_TIMEOUT_SECONDS + 300
    ):
        raise RealMLXExperimentError("run ledger finalized elapsed times are invalid")
    run_finalized = rows[-1]
    _exact_keys(
        run_finalized,
        {
            "schema_version",
            "sequence",
            "timestamp",
            "event",
            "status",
            "reason",
            "complete_replicates",
            "results_digest",
        },
        "run-finalized ledger row",
    )
    complete = sum(row["status"] == "complete" for row in final_rows.values())
    eligible = _aggregate_eligibility_from_rows(rows)
    results_derivation_failed = (
        run_finalized["status"] == "results_derivation_failed"
        and run_finalized["reason"] == "results_derivation_failed"
        and run_finalized["results_digest"] is None
    )
    if (
        run_finalized["schema_version"] != "2"
        or run_finalized["event"] != "run-finalized"
        or run_finalized["complete_replicates"] != complete
        or (results_derivation_failed and not eligible)
        or (
            not results_derivation_failed
            and (
                run_finalized["status"]
                != (
                    "aggregate_eligible"
                    if eligible
                    else (
                        "insufficient_complete_replicates"
                        if complete < len(REPLICATE_IDS)
                        else "aggregate_ineligible"
                    )
                )
                or run_finalized["reason"]
                != (
                    None
                    if eligible
                    else (
                        "insufficient_complete_replicates"
                        if complete < len(REPLICATE_IDS)
                        else "ineligible_replicate_failure"
                    )
                )
                or (
                    eligible
                    and re.fullmatch(
                        r"sha256:[0-9a-f]{64}",
                        str(run_finalized["results_digest"]),
                    )
                    is None
                )
                or (not eligible and run_finalized["results_digest"] is not None)
            )
        )
    ):
        raise RealMLXExperimentError("run ledger completion is invalid")
    if (
        expected_results_digest is not None
        and run_finalized["results_digest"] != expected_results_digest
    ):
        raise RealMLXExperimentError("run ledger results digest mismatch")
    for row in rows:
        _scan_private_keys(row, "run_ledger")
        if row.get("schema_version") != "2":
            raise RealMLXExperimentError("run ledger schema is invalid")
        event = row.get("event")
        if event not in {
            "run-start",
            "planned",
            "preflight",
            "started",
            "monitor",
            "postflight",
            "finalized",
            "run-finalized",
        }:
            raise RealMLXExperimentError("run ledger event is invalid")
        if (
            event not in {"run-start", "run-finalized"}
            and row.get("replicate_id") not in REPLICATE_IDS
        ):
            raise RealMLXExperimentError("run ledger replicate ID is invalid")
        if event == "planned":
            _exact_keys(
                row,
                {
                    "schema_version",
                    "sequence",
                    "timestamp",
                    "replicate_id",
                    "event",
                    "status",
                    "reason",
                },
                "planned ledger row",
            )
            if row["status"] != "planned" or row["reason"] is not None:
                raise RealMLXExperimentError("planned ledger row is invalid")
        reason = row.get("reason")
        if reason is not None and (
            not isinstance(reason, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{2,63}", reason) is None
        ):
            raise RealMLXExperimentError("run ledger reason is invalid")
    return binding


def _ledger_event(
    ledger: Path,
    *,
    sequence: int,
    replicate_id: str,
    event: str,
    status: str,
    reason: str | None,
    observation: Mapping[str, Any] | None = None,
    elapsed_seconds: float | None = None,
    attempt_digest: str | None = None,
    child_instance_id_digest: str | None = None,
) -> None:
    row: dict[str, Any] = {
        "schema_version": "2",
        "sequence": sequence,
        "timestamp": _utc_now(),
        "replicate_id": replicate_id,
        "event": event,
        "status": status,
        "reason": reason,
    }
    if observation is not None:
        row["observation"] = dict(observation)
    if elapsed_seconds is not None:
        row["elapsed_seconds"] = round(elapsed_seconds, 6)
    if attempt_digest is not None:
        row["attempt_digest"] = attempt_digest
    if child_instance_id_digest is not None:
        row["child_instance_id_digest"] = child_instance_id_digest
    _append_run_ledger(ledger, row)


def _finalize_run_ledger(
    ledger: Path,
    *,
    sequence: int,
    complete_replicates: int,
    results_digest: str | None,
    results_derivation_failed: bool = False,
) -> None:
    rows = _parse_jsonl(ledger)
    complete = sum(
        row.get("event") == "finalized" and row.get("status") == "complete"
        for row in rows
    )
    if complete != complete_replicates:
        raise RealMLXExperimentError(
            "run ledger complete replicate count does not match finalizations"
        )
    eligible = _aggregate_eligibility_from_rows(rows)
    if results_derivation_failed and not eligible:
        raise RealMLXExperimentError(
            "ineligible run cannot have a results derivation failure"
        )
    if not eligible and results_digest is not None:
        raise RealMLXExperimentError("ineligible run cannot bind results")
    status = (
        "results_derivation_failed"
        if results_derivation_failed
        else (
            "aggregate_eligible"
            if eligible
            else (
                "insufficient_complete_replicates"
                if complete_replicates < len(REPLICATE_IDS)
                else "aggregate_ineligible"
            )
        )
    )
    _append_run_ledger(
        ledger,
        {
            "schema_version": "2",
            "sequence": sequence,
            "timestamp": _utc_now(),
            "event": "run-finalized",
            "status": status,
            "reason": (
                "results_derivation_failed"
                if results_derivation_failed
                else (
                    None
                    if eligible
                    else (
                        "insufficient_complete_replicates"
                        if complete_replicates < len(REPLICATE_IDS)
                        else "ineligible_replicate_failure"
                    )
                )
            ),
            "complete_replicates": complete_replicates,
            "results_digest": results_digest,
        },
    )


def _write_sanitized_run_ledger(
    source: Path,
    destination: Path,
    *,
    attempts_dir: Path,
) -> None:
    rows = _parse_jsonl(source)
    retained = [dict(row) for row in rows]
    for row in retained:
        if row["event"] == "finalized":
            row["attempt_digest"] = _directory_digest(
                attempts_dir / str(row["replicate_id"])
            )
    atomic_write_text(destination, "".join(_json_line(row) for row in retained))


def _group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(
    process: subprocess.Popen[bytes],
    *,
    grace_seconds: float = PROCESS_GROUP_GRACE_SECONDS,
) -> bool:
    process_group = process.pid
    if not _group_exists(process_group):
        return True
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        process.poll()
        if not _group_exists(process_group):
            return True
        time.sleep(0.05)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        process.poll()
        if not _group_exists(process_group):
            return True
        time.sleep(0.05)
    return not _group_exists(process_group)


def _write_bounded_child_log(path: Path, stream: Any) -> None:
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(max(0, size - MAX_CHILD_LOG_BYTES))
    content = stream.read(MAX_CHILD_LOG_BYTES)
    if not isinstance(content, bytes):
        content = bytes(content)
    prefix = b"[earlier output truncated]\n" if size > MAX_CHILD_LOG_BYTES else b""
    atomic_write_text(path, (prefix + content).decode("utf-8", errors="replace"))


def _preserve_child_artifacts(
    staging: Path,
    artifact_dir: Path,
    stdout: Any,
    stderr: Any,
) -> None:
    artifact_dir.mkdir(parents=True)
    _write_bounded_child_log(artifact_dir / "stdout.log", stdout)
    _write_bounded_child_log(artifact_dir / "stderr.log", stderr)
    if staging.exists():
        attempt = staging / "attempt.json"
        if attempt.is_file() and not attempt.is_symlink():
            try:
                status = _safe_object(attempt).get("status")
            except (OSError, RealMLXExperimentError):
                status = None
            if status == "complete":
                os.replace(attempt, staging / "stale-complete-attempt.json")
        os.replace(staging, artifact_dir / "partial-output")


def run_all_replicates(
    *,
    workload: Path,
    model_dir: Path,
    conversion_summary: Path,
    output_workspace: Path,
    expected_commit: str,
) -> dict[str, Any]:
    """Run the six fixed replicate IDs once under a fail-closed supervisor."""

    preflight_state = _run_global_preflight(
        workload=workload,
        model_dir=model_dir,
        conversion_summary=conversion_summary,
        output_workspace=output_workspace,
        expected_commit=expected_commit,
    )
    workload = preflight_state.workload_path
    model_dir = preflight_state.model_dir
    conversion_summary = preflight_state.conversion_summary
    output_workspace = preflight_state.output_workspace
    package_digest = preflight_state.package_digest
    binding = _run_binding(
        expected_commit=expected_commit,
        package_digest=package_digest,
        workload=preflight_state.workload,
        runtime_packages=preflight_state.runtime_packages,
    )
    output_workspace.mkdir(parents=True)
    attempts = output_workspace / "attempts"
    artifacts = output_workspace / "private-artifacts"
    attempts.mkdir()
    artifacts.mkdir()
    ledger = output_workspace / "run-ledger.jsonl"
    _create_run_ledger(ledger, binding)
    ledger_sequence = len(REPLICATE_IDS) + 1
    completed = 0
    supervisor_started = time.monotonic()
    abort_remaining = False
    instance_ids: set[str] = set()

    for replicate_id in REPLICATE_IDS:
        attempt_dir = attempts / replicate_id
        staging = output_workspace / f".{replicate_id}.partial"
        artifact_dir = artifacts / replicate_id
        if attempt_dir.exists() or staging.exists() or artifact_dir.exists():
            raise RealMLXExperimentError("replicate ID was already attempted")
        reason: str | None = None
        preflight = _machine_observation(
            output_workspace,
            excluded_roots={os.getpid()},
        )
        if abort_remaining:
            reason = "supervisor_aborted_before_start"
        elif time.monotonic() - supervisor_started >= TOTAL_TIMEOUT_SECONDS:
            reason = "total_timeout_before_start"
            abort_remaining = True
        else:
            reason = _machine_policy_reason(preflight, preflight=True)
        _ledger_event(
            ledger,
            sequence=ledger_sequence,
            replicate_id=replicate_id,
            event="preflight",
            status="passed" if reason is None else "failed",
            reason=reason,
            observation=preflight,
        )
        ledger_sequence += 1

        if reason is None:
            try:
                _reject_import_shadows(output_workspace)
                if _validate_supervisor_source(expected_commit) != package_digest:
                    raise RealMLXExperimentError("package source digest changed")
            except (CacheAuditBundleError, OSError, RealMLXExperimentError):
                reason = "source_validation_failed"
                abort_remaining = True

        if reason is None:
            instance_id = secrets.token_hex(16)
            while instance_id in instance_ids:
                instance_id = secrets.token_hex(16)
            instance_ids.add(instance_id)
            command = _replicate_child_command(
                workload=workload,
                model_dir=model_dir,
                conversion_summary=conversion_summary,
                replicate_id=replicate_id,
                output_dir=staging,
                expected_commit=expected_commit,
                runtime_packages_digest=_digest_bytes(
                    _json_bytes(binding["runtime_packages"])
                ),
            )
            private_temp_root = output_workspace / f".{replicate_id}.tmp"
            private_temp_root.mkdir(mode=0o700)
            child_started = time.monotonic()
            with (
                tempfile.TemporaryFile(mode="w+b") as stdout,
                tempfile.TemporaryFile(mode="w+b") as stderr,
            ):
                process: subprocess.Popen[bytes] | None = None
                started_recorded = False
                try:
                    process = subprocess.Popen(
                        command,
                        stdin=subprocess.DEVNULL,
                        stdout=stdout,
                        stderr=stderr,
                        env=_offline_child_environment(
                            instance_id, private_temp_root=private_temp_root
                        ),
                        cwd=output_workspace.parent,
                        shell=False,
                        start_new_session=True,
                    )
                    _ledger_event(
                        ledger,
                        sequence=ledger_sequence,
                        replicate_id=replicate_id,
                        event="started",
                        status="running",
                        reason=None,
                        child_instance_id_digest=_digest_bytes(
                            instance_id.encode("ascii")
                        ),
                    )
                    ledger_sequence += 1
                    started_recorded = True
                    next_monitor = time.monotonic() + MONITOR_INTERVAL_SECONDS
                    while process.poll() is None:
                        now = time.monotonic()
                        if now - child_started >= CHILD_TIMEOUT_SECONDS:
                            reason = "child_timeout"
                            break
                        if now - supervisor_started >= TOTAL_TIMEOUT_SECONDS:
                            reason = "total_timeout"
                            abort_remaining = True
                            break
                        if now >= next_monitor:
                            observation = _machine_observation(
                                output_workspace,
                                excluded_roots={os.getpid(), process.pid},
                            )
                            monitor_reason = _machine_policy_reason(
                                observation, preflight=False
                            )
                            _ledger_event(
                                ledger,
                                sequence=ledger_sequence,
                                replicate_id=replicate_id,
                                event="monitor",
                                status=(
                                    "passed" if monitor_reason is None else "failed"
                                ),
                                reason=monitor_reason,
                                observation=observation,
                                elapsed_seconds=now - child_started,
                            )
                            ledger_sequence += 1
                            if monitor_reason is not None:
                                reason = monitor_reason
                                break
                            next_monitor = now + MONITOR_INTERVAL_SECONDS
                        time.sleep(0.1)
                    if reason is not None:
                        if not _terminate_process_group(process):
                            reason = "process_cleanup_failed"
                            abort_remaining = True
                    else:
                        returncode = process.wait()
                        if returncode != 0:
                            reason = "child_exit_nonzero"
                            if _group_exists(process.pid):
                                reason = "orphaned_child_process"
                                abort_remaining = True
                            if not _terminate_process_group(process):
                                reason = "process_cleanup_failed"
                                abort_remaining = True
                        elif _group_exists(process.pid):
                            reason = "orphaned_child_process"
                            abort_remaining = True
                            if not _terminate_process_group(process):
                                reason = "process_cleanup_failed"
                                abort_remaining = True
                except KeyboardInterrupt:
                    abort_remaining = True
                    reason = (
                        "supervisor_aborted"
                        if started_recorded
                        else "supervisor_aborted_before_start"
                    )
                    if process is not None and not _terminate_process_group(process):
                        reason = "process_cleanup_failed"
                        abort_remaining = True
                except (OSError, subprocess.SubprocessError):
                    reason = "child_launch_failed"
                    if process is not None and not _terminate_process_group(process):
                        reason = "process_cleanup_failed"
                        abort_remaining = True

                try:
                    _reject_import_shadows(output_workspace)
                    if _validate_supervisor_source(expected_commit) != package_digest:
                        raise RealMLXExperimentError("package source digest changed")
                except (CacheAuditBundleError, OSError, RealMLXExperimentError):
                    reason = "source_validation_failed"
                    abort_remaining = True

                postflight = _machine_observation(
                    output_workspace,
                    excluded_roots=(
                        {os.getpid()} if process is None else {os.getpid(), process.pid}
                    ),
                )
                postflight_reason = _machine_policy_reason(postflight, preflight=False)
                if postflight_reason is not None and reason is None:
                    reason = postflight_reason
                postflight_status = (
                    "failed"
                    if postflight_reason is not None
                    else "not_started" if not started_recorded else "passed"
                )
                _ledger_event(
                    ledger,
                    sequence=ledger_sequence,
                    replicate_id=replicate_id,
                    event="postflight",
                    status=postflight_status,
                    reason=(
                        postflight_reason
                        if postflight_reason is not None
                        else reason if not started_recorded else None
                    ),
                    observation=postflight,
                    elapsed_seconds=(
                        None
                        if not started_recorded
                        else time.monotonic() - child_started
                    ),
                )
                ledger_sequence += 1
                temp_cleanup_failed = False
                try:
                    shutil.rmtree(private_temp_root, ignore_errors=False)
                except OSError:
                    reason = "process_cleanup_failed"
                    abort_remaining = True
                    temp_cleanup_failed = True
                if reason is None:
                    try:
                        verify_replicate(
                            staging,
                            replicate_id=replicate_id,
                            public=False,
                        )
                    except (
                        CacheAuditBundleError,
                        IndexError,
                        KeyError,
                        OSError,
                        RealMLXExperimentError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ):
                        reason = "invalid_complete_artifact"
                if reason is None:
                    os.replace(staging, attempt_dir)
                    artifact_dir.mkdir()
                    _write_bounded_child_log(artifact_dir / "stdout.log", stdout)
                    _write_bounded_child_log(artifact_dir / "stderr.log", stderr)
                    completed += 1
                else:
                    _preserve_child_artifacts(
                        staging,
                        artifact_dir,
                        stdout,
                        stderr,
                    )
                    if temp_cleanup_failed and private_temp_root.exists():
                        os.replace(private_temp_root, artifact_dir / "private-temp")
        else:
            postflight = _machine_observation(
                output_workspace,
                excluded_roots={os.getpid()},
            )
            postflight_reason = _machine_policy_reason(postflight, preflight=False)
            _ledger_event(
                ledger,
                sequence=ledger_sequence,
                replicate_id=replicate_id,
                event="postflight",
                status="failed" if postflight_reason is not None else "not_started",
                reason=postflight_reason if postflight_reason is not None else reason,
                observation=postflight,
            )
            ledger_sequence += 1

        if reason is not None:
            record_failed_attempt(
                attempt_dir,
                replicate_id,
                reason=reason,
                failed_at=_utc_now(),
            )
        attempt_digest = _directory_digest(attempt_dir)
        _ledger_event(
            ledger,
            sequence=ledger_sequence,
            replicate_id=replicate_id,
            event="finalized",
            status="complete" if reason is None else "failed",
            reason=reason,
            elapsed_seconds=time.monotonic() - supervisor_started,
            attempt_digest=attempt_digest,
        )
        ledger_sequence += 1

    eligible = _aggregate_eligibility_from_rows(_parse_jsonl(ledger))
    results_digest = None
    results_derivation_failed = False
    if eligible:
        try:
            results = _public_results_from_private(attempts)
            _verify_public_results(
                results,
                complete_replicate_ids=_complete_attempt_ids(attempts),
            )
            results_digest = _digest_bytes(_json_bytes(results))
        except (
            CacheAuditBundleError,
            OSError,
            RealMLXExperimentError,
            RuntimeError,
            ValueError,
        ):
            results_derivation_failed = True
    _finalize_run_ledger(
        ledger,
        sequence=ledger_sequence,
        complete_replicates=completed,
        results_digest=results_digest,
        results_derivation_failed=results_derivation_failed,
    )
    _verify_run_ledger(
        ledger,
        attempts_dir=attempts,
        expected_binding=binding,
        expected_results_digest=results_digest,
    )
    return {
        "run_all_complete": eligible and not results_derivation_failed,
        "attempted_replicates": len(REPLICATE_IDS),
        "complete_replicates": completed,
        "failed_replicates": len(REPLICATE_IDS) - completed,
        "aggregate_created": False,
        "sanitized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmtracefx-real-mlx-cache-audit")
    commands = parser.add_subparsers(dest="command", required=True)
    compile_parser = commands.add_parser("compile")
    compile_parser.add_argument("--model-dir", type=Path, required=True)
    compile_parser.add_argument("--output", type=Path, required=True)
    compile_parser.add_argument(
        "--conversion-summary", type=Path, default=DEFAULT_CONVERSION_SUMMARY
    )
    calibrate_parser = commands.add_parser("calibrate")
    calibrate_parser.add_argument("--workload", type=Path, required=True)
    calibrate_parser.add_argument("--model-dir", type=Path, required=True)
    calibrate_parser.add_argument("--output", type=Path, required=True)
    calibrate_parser.add_argument(
        "--conversion-summary", type=Path, default=DEFAULT_CONVERSION_SUMMARY
    )
    replicate_parser = commands.add_parser("replicate")
    replicate_parser.add_argument("--workload", type=Path, required=True)
    replicate_parser.add_argument("--model-dir", type=Path, required=True)
    replicate_parser.add_argument(
        "--replicate-id", choices=REPLICATE_IDS, required=True
    )
    replicate_parser.add_argument("--output-dir", type=Path, required=True)
    replicate_parser.add_argument("--expected-commit", required=True)
    replicate_parser.add_argument("--runtime-packages-digest")
    replicate_parser.add_argument(
        "--conversion-summary", type=Path, default=DEFAULT_CONVERSION_SUMMARY
    )
    run_all_parser = commands.add_parser("run-all")
    run_all_parser.add_argument("--workload", type=Path, required=True)
    run_all_parser.add_argument("--model-dir", type=Path, required=True)
    run_all_parser.add_argument("--output-workspace", type=Path, required=True)
    run_all_parser.add_argument("--expected-commit", required=True)
    run_all_parser.add_argument(
        "--conversion-summary", type=Path, default=DEFAULT_CONVERSION_SUMMARY
    )
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--workload", type=Path, required=True)
    preflight_parser.add_argument("--model-dir", type=Path, required=True)
    preflight_parser.add_argument("--output-workspace", type=Path, required=True)
    preflight_parser.add_argument("--expected-commit", required=True)
    preflight_parser.add_argument("--output", type=Path, required=True)
    preflight_parser.add_argument(
        "--conversion-summary", type=Path, default=DEFAULT_CONVERSION_SUMMARY
    )
    probe_parser = commands.add_parser("sandbox-probe", help=argparse.SUPPRESS)
    probe_parser.add_argument("--output-workspace", type=Path, required=True)
    probe_parser.add_argument("--expected-commit", required=True)
    probe_parser.add_argument("--expected-package-digest", required=True)
    failed_parser = commands.add_parser("record-failure")
    failed_parser.add_argument("--replicate-id", choices=REPLICATE_IDS, required=True)
    failed_parser.add_argument("--output-dir", type=Path, required=True)
    failed_parser.add_argument("--reason", default="externally_terminated")
    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--run-workspace", type=Path, required=True)
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)
    sanitize_parser = commands.add_parser("sanitize")
    sanitize_parser.add_argument("aggregate", type=Path)
    sanitize_parser.add_argument("--output-dir", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("aggregate", type=Path)
    return parser


def _run_cli(args: argparse.Namespace) -> dict[str, Any]:
    if args.command in {"run-all", "preflight", "replicate", "sandbox-probe"} and (
        os.environ.get("LLMTRACEFX_TRUSTED_BOOTSTRAP") != "1"
    ):
        raise RealMLXExperimentError(
            "canonical execution requires the trusted Python -I -S bootstrap"
        )
    if args.command == "compile":
        snapshot_owner, snapshot, _ = _verified_model_snapshot(
            args.model_dir, args.conversion_summary
        )
        try:
            tokenizer = _load_local_tokenizer(snapshot)
            workload = write_compiled_workload(args.output, tokenizer)
        finally:
            snapshot_owner.cleanup()
        return {
            "compiled": True,
            "base_tokens": {lane.lane_id: len(lane.base) for lane in workload.lanes},
            "model_loaded": False,
            "network_used": False,
        }
    if args.command == "calibrate":
        workload = calibrate_workload_file(
            args.workload,
            args.output,
            model_dir=args.model_dir,
            conversion_summary=args.conversion_summary,
        )
        return {
            "calibrated": True,
            "calibration_output_tokens": {
                lane.lane_id: {
                    name: len((lane.calibration_outputs or {}).get(name, ()))
                    for name in CALIBRATION_ARRAY_NAMES
                }
                for lane in workload.lanes
            },
        }
    if args.command == "replicate":
        return run_replicate(
            args.workload,
            args.output_dir,
            replicate_id=args.replicate_id,
            model_dir=args.model_dir,
            expected_commit=args.expected_commit,
            conversion_summary=args.conversion_summary,
            expected_runtime_packages_digest=args.runtime_packages_digest,
        )
    if args.command == "run-all":
        return run_all_replicates(
            workload=args.workload,
            model_dir=args.model_dir,
            conversion_summary=args.conversion_summary,
            output_workspace=args.output_workspace,
            expected_commit=args.expected_commit,
        )
    if args.command == "preflight":
        return run_preflight(
            workload=args.workload,
            model_dir=args.model_dir,
            conversion_summary=args.conversion_summary,
            output_workspace=args.output_workspace,
            expected_commit=args.expected_commit,
            output=args.output,
        )
    if args.command == "sandbox-probe":
        return _sandbox_probe(
            output_workspace=args.output_workspace,
            expected_commit=args.expected_commit,
            expected_package_digest=args.expected_package_digest,
        )
    if args.command == "record-failure":
        record_failed_attempt(
            args.output_dir,
            args.replicate_id,
            reason=args.reason,
        )
        return {"recorded": True, "status": "failed"}
    if args.command == "aggregate":
        return assemble_aggregate(args.run_workspace, args.output_dir)
    if args.command == "sanitize":
        return sanitize_aggregate(args.aggregate, args.output_dir)
    return verify_aggregate(args.aggregate)


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        result = _run_cli(args)
    except (
        CacheAuditBundleError,
        OSError,
        RealMLXExperimentError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    exit_code = (
        1
        if (
            args.command == "run-all"
            and not result["run_all_complete"]
            or args.command == "preflight"
            and not result["preflight_passed"]
        )
        else 0
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main(sys.argv[1:])
