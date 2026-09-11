"""Private, fail-closed runner for the real Apple Silicon MLX cache experiment.

The normal cache-audit CLI remains the small cross-backend interface.  This
module owns the deliberately narrower, private workflow: compile exact local
tokens, calibrate deterministic output, execute one isolated replicate, then
assemble and redact the six-attempt evidence envelope.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import groupby
from pathlib import Path
from typing import Any

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
    BUNDLE_FILES,
    CacheAuditBundleError,
    read_bundle,
    sanitize_bundle_records,
    verify_bundle,
    write_bundle,
)
from .runner import run_audit
from .schema import (
    CacheConfig,
    PairRole,
    PublicationMode,
    RequestEvidence,
    RequestSpec,
    ScenarioKind,
)

WORKLOAD_SCHEMA_VERSION = "real-mlx-workload-v1"
AGGREGATE_SCHEMA_VERSION = "real-mlx-aggregate-v1"
REPLICATE_IDS = tuple(f"replicate-{index}" for index in range(6))
LANE_IDS = ("1k", "4k")
ROTATION_OFFSETS = (0, 3, 6, 9, 12, 15)
MAX_CACHE_ENTRIES = 2
MAX_CACHE_BYTES = 1 << 63
MAX_OUTPUT_TOKENS = 8
MAX_ALLOCATOR_PEAK_BYTES = 8 * 1024**3
MAX_PROCESS_RSS_BYTES = 12 * 1024**3
MAX_SWAP_BYTES = 14 * 1024**3
MAX_SWAP_GROWTH_BYTES = 2 * 1024**3
MIN_RUNTIME_MEMORY_FREE_PERCENT = 15.0
EXPECTED_MODEL_FILE_COUNT = 8
EXPECTED_CONVERSION_SUMMARY_SHA256 = (
    "9c87cad2a7de7bbc42bfd6a1d7f502c32422ba00b29df27a2363c07aa2a45c25"
)
EXPECTED_MODEL_ARTIFACT_DIGEST = (
    "sha256:057a37f4ebc76420f8ab2edb17bc8442e050c8d13f7334f829356e2f9cab6802"
)
EXPECTED_CALIBRATED_WORKLOAD_DIGEST = (
    "sha256:c8b3dc8939c65af48987c3f0abd3aa89e5ed2d5113499a809b9cadb904f2923b"
)
EXPECTED_CALIBRATED_LANE_DIGESTS = {
    "1k": "sha256:692d85f271130c48f19d5157c043f94950439cdea8d7259d055e30a4fe8e74d6",
    "4k": "sha256:ff5d9dca4b3edbf677df6e3093dbe60cf48ae3edaa95e74a3e80cc08357cc476",
}
EXPECTED_SEED_OUTPUT_TOKENS = {"1k": 3, "4k": 3}
MODEL_ID = "local-self-converted/qwen3-4b-mlx-q4g64"
TOKENIZER_ID = "Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONVERSION_SUMMARY = (
    _PROJECT_ROOT / "llmtracefx/cache_audit/data/qwen3-4b-conversion-summary.json"
)
_CASES = (
    "cold-exact-duplicate",
    "longer-to-shorter",
    "shorter-to-longer",
    "interior-mutation",
    "allocation-step-mutation",
    "same-length-different-ids",
    "suffix-only-change",
    "namespace-isolation",
    "capacity-eviction",
)
_CASE_REQUEST_COUNTS = {
    "cold-exact-duplicate": 3,
    "longer-to-shorter": 2,
    "shorter-to-longer": 2,
    "interior-mutation": 2,
    "allocation-step-mutation": 2,
    "same-length-different-ids": 2,
    "suffix-only-change": 2,
    "namespace-isolation": 2,
    "capacity-eviction": 5,
}
_BLOCKS = tuple(f"{lane_id}:{case}" for lane_id in LANE_IDS for case in _CASES)
_BLOCK_REQUEST_COUNTS = {
    f"{lane_id}:{case}": count
    for lane_id in LANE_IDS
    for case, count in _CASE_REQUEST_COUNTS.items()
}
_REQUESTS_PER_LANE = sum(_CASE_REQUEST_COUNTS.values())
_LANE_CONTRACTS = {
    "1k": {"base": 1025, "shorter_seed": 769, "eviction": 513},
    "4k": {"base": 4097, "shorter_seed": 3073, "eviction": 2049},
}
_EXPECTED_RUNTIME_IDENTITY = {
    "mlx": REQUIRED_MLX_VERSION,
    "mlx_lm": REQUIRED_MLX_LM_VERSION,
    "platform_machine": "arm64",
    "platform_system": "Darwin",
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
    "summary.json",
    "report.html",
    "reuse-alignment.svg",
    "timing-memory.svg",
    "teardown.json",
    "aggregate_verifier.py",
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
    extension_32: tuple[int, ...]
    eviction_a: tuple[int, ...]
    eviction_b: tuple[int, ...]
    eviction_c: tuple[int, ...]
    shorter_seed: tuple[int, ...]
    seed_output: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        contract = _LANE_CONTRACTS.get(self.lane_id)
        if contract is None:
            raise RealMLXExperimentError("lane ID must be one of 1k or 4k")
        base_tokens = contract["base"]
        seed_tokens = contract["shorter_seed"]
        eviction_tokens = contract["eviction"]
        if len(self.base) != base_tokens:
            raise RealMLXExperimentError(
                f"{self.lane_id} base must contain exactly {base_tokens} tokens"
            )
        if (
            len(self.shorter_seed) != seed_tokens
            or self.shorter_seed != self.base[:seed_tokens]
        ):
            raise RealMLXExperimentError(
                f"{self.lane_id} shorter seed must be the "
                f"{seed_tokens}-token base prefix"
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
        if len(self.extension_32) != 32:
            raise RealMLXExperimentError("extension must contain exactly 32 tokens")
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
        if differences != list(range(len(self.base) - 16, len(self.base))):
            raise RealMLXExperimentError(
                "suffix change must replace exactly the final 16 tokens"
            )
        if len({self.eviction_a, self.eviction_b, self.eviction_c}) != 3:
            raise RealMLXExperimentError("eviction A/B/C arrays must be distinct")
        if self.seed_output is not None and not 1 <= len(self.seed_output) <= 8:
            raise RealMLXExperimentError(
                "calibrated seed output must contain 1-8 tokens"
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
                "extension_32",
                "eviction_a",
                "eviction_b",
                "eviction_c",
                "shorter_seed",
            )
        }
        arrays["seed_output"] = (
            None if self.seed_output is None else list(self.seed_output)
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
            "extension_32",
            "eviction_a",
            "eviction_b",
            "eviction_c",
            "shorter_seed",
            "seed_output",
        }
        _exact_keys(arrays, names, "workload arrays")
        seed_raw = arrays["seed_output"]
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
            extension_32=_integer_array(arrays["extension_32"], "arrays.extension_32"),
            eviction_a=_integer_array(arrays["eviction_a"], "arrays.eviction_a"),
            eviction_b=_integer_array(arrays["eviction_b"], "arrays.eviction_b"),
            eviction_c=_integer_array(arrays["eviction_c"], "arrays.eviction_c"),
            shorter_seed=_integer_array(arrays["shorter_seed"], "arrays.shorter_seed"),
            seed_output=(
                None
                if seed_raw is None
                else _integer_array(seed_raw, "arrays.seed_output")
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
    if calibrated is True and any(lane.seed_output is None for lane in workload.lanes):
        raise RealMLXExperimentError("workload has not been calibrated")
    if calibrated is True and (
        workload.to_dict()["workload_digest"] != EXPECTED_CALIBRATED_WORKLOAD_DIGEST
        or any(
            lane.to_dict()["lane_digest"]
            != EXPECTED_CALIBRATED_LANE_DIGESTS[lane.lane_id]
            or len(lane.seed_output or ()) != EXPECTED_SEED_OUTPUT_TOKENS[lane.lane_id]
            for lane in workload.lanes
        )
    ):
        raise RealMLXExperimentError("calibrated workload contract mismatch")
    if calibrated is False and any(
        lane.seed_output is not None for lane in workload.lanes
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


def _continued_base_prompt(
    tokenizer: Any,
    shorter: tuple[int, ...],
    first_body: str,
    *,
    target: int,
    lane_id: str,
) -> tuple[int, ...]:
    for count in range(target * 2):
        second_body = (
            "LLMTraceFX public synthetic continuation. "
            + (" audit" * count)
            + " Answer exactly CACHE_OK."
        )
        tokens = _template_messages(
            tokenizer,
            (
                {"role": "user", "content": first_body},
                {"role": "assistant", "content": "CACHE_OK"},
                {"role": "user", "content": second_body},
            ),
        )
        divergence = next(
            (
                index
                for index, pair in enumerate(zip(shorter, tokens, strict=False))
                if pair[0] != pair[1]
            ),
            len(shorter),
        )
        continued = shorter + tokens[divergence:]
        if len(continued) == target:
            return continued
        if len(continued) > target + 8 and count > target:
            break
    raise RealMLXExperimentError(
        f"local tokenizer could not compile the {target}-token "
        f"{lane_id} continued base prompt"
    )


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


def _extension_tokens(tokenizer: Any) -> tuple[int, ...]:
    for count in range(1, 100):
        text = (" extension" * count) + " continue"
        try:
            value = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            value = tokenizer.encode(text)
        if hasattr(value, "tolist"):
            value = value.tolist()
        if isinstance(value, list) and len(value) >= 32:
            return _integer_array(value[:32], "extension")
    raise RealMLXExperimentError("tokenizer could not compile a 32-token extension")


def compile_workload(tokenizer: Any) -> FrozenMLXWorkload:
    """Compile the exact private workload without loading model weights."""

    lanes: list[FrozenMLXLane] = []
    extension = _extension_tokens(tokenizer)
    for lane_id in LANE_IDS:
        contract = _LANE_CONTRACTS[lane_id]
        shorter, first_body = _exact_prompt_with_body(
            tokenizer,
            target=contract["shorter_seed"],
            label=f"{lane_id}-BASE-FIRST",
        )
        base = _continued_base_prompt(
            tokenizer,
            shorter,
            first_body,
            target=contract["base"],
            lane_id=lane_id,
        )
        different = _exact_prompt(
            tokenizer, target=contract["base"], label=f"{lane_id}-DIFFERENT"
        )
        if different == base:
            raise RealMLXExperimentError("different prompt tokenized identically")
        if different[0] == base[0]:
            changed = list(different)
            changed[0] = _replacement_token(tokenizer, changed[0])
            different = tuple(changed)
        mutations: dict[int, tuple[int, ...]] = {}
        for position in (137, 256):
            changed = list(base)
            changed[position] = _replacement_token(tokenizer, changed[position])
            mutations[position] = tuple(changed)
        suffix = base[:-16] + tuple(
            _replacement_token(tokenizer, token) for token in base[-16:]
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
                extension_32=extension,
                eviction_a=eviction[0],
                eviction_b=eviction[1],
                eviction_c=eviction[2],
                shorter_seed=shorter,
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


def calibrate_seed(
    workload: FrozenMLXWorkload,
    adapter_factory: Callable[[], MLXLocalCacheAdapter],
) -> FrozenMLXWorkload:
    """Run each lane's seed twice in fresh caches and freeze exact output IDs."""

    if any(lane.seed_output is not None for lane in workload.lanes):
        raise RealMLXExperimentError("workload is already calibrated")
    calibrated: list[FrozenMLXLane] = []
    for lane in workload.lanes:
        outputs: list[tuple[int, ...]] = []
        for index in range(2):
            request = RequestSpec(
                request_id=f"{lane.lane_id}:calibration-{index}",
                scenario=ScenarioKind.COLD,
                order=0,
                input_token_ids=lane.shorter_seed,
                input_token_count=len(lane.shorter_seed),
                output_tokens=MAX_OUTPUT_TOKENS,
                replicate_id="calibration",
            )
            records = adapter_factory().run((request,))
            if len(records) != 1:
                raise RealMLXExperimentError(
                    f"{lane.lane_id} seed calibration returned invalid record count"
                )
            record = records[0]
            output = record.output.output_token_ids
            if (
                output is None
                or not output
                or record.output.correctness.value is not True
                or record.terminal_state.value != "completed"
            ):
                raise RealMLXExperimentError(
                    f"{lane.lane_id} seed calibration failed exact "
                    "CACHE_OK correctness gate"
                )
            outputs.append(output)
        if outputs[0] != outputs[1]:
            raise RealMLXExperimentError(
                f"{lane.lane_id} seed calibration is not exactly repeatable"
            )
        calibrated.append(replace(lane, seed_output=outputs[0]))
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
    offset = ROTATION_OFFSETS[REPLICATE_IDS.index(replicate_id)]
    return _BLOCKS[offset:] + _BLOCKS[:offset]


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

    if any(lane.seed_output is None for lane in workload.lanes):
        raise RealMLXExperimentError("measurement requires calibrated seed output")
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
        elif case == "longer-to-shorter":
            _request(
                items,
                block=block,
                name="longer",
                scenario=ScenarioKind.COLD,
                tokens=lane.base,
                replicate_id=replicate_id,
                pair_id=pair,
                pair_role=PairRole.CONTROL,
            )
            _request(
                items,
                block=block,
                name="shorter",
                scenario=ScenarioKind.IDENTICAL_PREFIX,
                tokens=lane.shorter_seed,
                replicate_id=replicate_id,
                predecessors=(f"{block}:longer",),
                pair_id=pair,
                pair_role=PairRole.TREATMENT,
            )
        elif case == "shorter-to-longer":
            extension = lane.shorter_seed + (lane.seed_output or ()) + lane.extension_32
            _request(
                items,
                block=block,
                name="shorter",
                scenario=ScenarioKind.COLD,
                tokens=lane.shorter_seed,
                replicate_id=replicate_id,
                pair_id=pair,
                pair_role=PairRole.CONTROL,
            )
            _request(
                items,
                block=block,
                name="longer",
                scenario=ScenarioKind.SUFFIX_CHANGE,
                tokens=extension,
                replicate_id=replicate_id,
                predecessors=(f"{block}:shorter",),
                pair_id=pair,
                pair_role=PairRole.TREATMENT,
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
                    len(lane.base) - 16,
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
            for name, scenario, tokens, predecessors, role in (
                (
                    "a-seed",
                    ScenarioKind.COLD,
                    lane.eviction_a,
                    (),
                    PairRole.CONTROL,
                ),
                (
                    "a-hit",
                    ScenarioKind.IDENTICAL_PREFIX,
                    lane.eviction_a,
                    (f"{block}:a-seed",),
                    PairRole.SINGLE,
                ),
                ("b", ScenarioKind.COLD, lane.eviction_b, (), PairRole.SINGLE),
                ("c", ScenarioKind.COLD, lane.eviction_c, (), PairRole.SINGLE),
                (
                    "a-miss",
                    ScenarioKind.EVICTION_COUNT,
                    lane.eviction_a,
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
                    predecessors=predecessors,
                    pair_id=pair if role is not PairRole.SINGLE else None,
                    pair_role=role,
                )
        by_block[block] = items

    ordered: list[RequestSpec] = []
    for block in block_schedule(replicate_id):
        for request in by_block[block]:
            ordered.append(replace(request, order=len(ordered)))
    return tuple(ordered)


def _schedule_shape_specs(replicate_id: str, *, public: bool) -> list[dict[str, Any]]:
    lanes: list[FrozenMLXLane] = []
    for lane_id in LANE_IDS:
        contract = _LANE_CONTRACTS[lane_id]
        base = (1,) * contract["base"]
        different = (2,) + base[1:]
        mutation_137 = base[:137] + (2,) + base[138:]
        mutation_256 = base[:256] + (2,) + base[257:]
        suffix = base[:-16] + (2,) * 16
        lanes.append(
            FrozenMLXLane(
                lane_id=lane_id,
                base=base,
                different_ids=different,
                mutation_137=mutation_137,
                mutation_256=mutation_256,
                suffix_change=suffix,
                extension_32=(6,) * 32,
                eviction_a=(3,) * contract["eviction"],
                eviction_b=(4,) * contract["eviction"],
                eviction_c=(5,) * contract["eviction"],
                shorter_seed=base[: contract["shorter_seed"]],
                seed_output=(7,) * EXPECTED_SEED_OUTPUT_TOKENS[lane_id],
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
    result = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())],
        capture_output=True,
        check=False,
        text=True,
    )
    try:
        return int(result.stdout.strip()) * 1024 if result.returncode == 0 else None
    except ValueError:
        return None


def _system_swap_used_bytes() -> int | None:
    result = subprocess.run(
        ["sysctl", "-n", "vm.swapusage"],
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


def _safe_environment() -> dict[str, Any]:
    return {
        "schema_version": "1",
        "platform_system": platform.system(),
        "platform_machine": platform.machine(),
        "os_release": platform.release(),
        "python": platform.python_version(),
        "mlx": _distribution_version("mlx"),
        "mlx_lm": _distribution_version("mlx-lm"),
        "process_scope": "one_fresh_replicate_child",
    }


def _distribution_version(name: str) -> str | None:
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


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
        calibrated = calibrate_seed(
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
    conversion_summary: Path = DEFAULT_CONVERSION_SUMMARY,
) -> dict[str, Any]:
    """Run one replicate. Parent-level timeout/process isolation stays external."""

    if output_dir.exists():
        raise RealMLXExperimentError("replicate output already exists")
    output_dir.mkdir(parents=True)
    snapshot_owner: tempfile.TemporaryDirectory[str] | None = None
    try:
        workload = load_workload(workload_path, calibrated=True)
        model, tokenizer, model_key, digest, snapshot_owner = _load_verified_model(
            model_dir, conversion_summary
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
        _write_json(output_dir / "environment.json", _safe_environment())
        _write_json(output_dir / "workload.json", workload.to_dict())
        _write_json(
            output_dir / "attempt.json",
            {
                "schema_version": "1",
                "replicate_id": replicate_id,
                "status": "complete",
                "rotation": list(block_schedule(replicate_id)),
                "request_count": len(requests),
                "lane_request_counts": _lane_request_counts(requests),
                "independent_unit": True,
                "replacement": False,
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
                "cache_config_digest": _digest_bytes(
                    _json_bytes(_experiment_cache_config().to_dict())
                ),
            },
        )
        del adapter
        del model
        teardown = _teardown()
        snapshot_owner.cleanup()
        _write_json(output_dir / "teardown.json", teardown)
        verify_replicate(output_dir, replicate_id=replicate_id, public=False)
        return {"replicate_id": replicate_id, "status": "complete"}
    except Exception:
        if snapshot_owner is not None:
            snapshot_owner.cleanup()
        for child in tuple(output_dir.iterdir()):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        _write_json(
            output_dir / "attempt.json",
            {
                "schema_version": "1",
                "replicate_id": replicate_id,
                "status": "failed",
                "replacement": False,
            },
        )
        _write_json(output_dir / "teardown.json", _teardown())
        raise


def record_failed_attempt(output_dir: Path, replicate_id: str) -> None:
    """Create a path-free failed-attempt marker for an externally timed-out child."""

    if output_dir.exists():
        raise RealMLXExperimentError("attempt output already exists")
    if replicate_id not in REPLICATE_IDS:
        raise RealMLXExperimentError("invalid replicate ID")
    output_dir.mkdir(parents=True)
    _write_json(
        output_dir / "attempt.json",
        {
            "schema_version": "1",
            "replicate_id": replicate_id,
            "status": "failed",
            "replacement": False,
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


def verify_replicate(
    directory: Path, *, replicate_id: str, public: bool
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
            {"schema_version", "replicate_id", "status", "replacement"},
            "failed attempt",
        )
        _verify_teardown(_safe_object(directory / "teardown.json"), complete=False)
        return {"replicate_id": replicate_id, "status": status}

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
            "frozen_workload_digest",
            "frozen_lane_digests",
            "model_artifact_digest",
            "model_id",
            "tokenizer_id",
            "runtime_identity_digest",
            "cache_config_digest",
        },
        "complete attempt",
    )
    rotation = tuple(attempt["rotation"])
    if (
        attempt["schema_version"] != "1"
        or rotation != block_schedule(replicate_id)
        or attempt["request_count"] != sum(_BLOCK_REQUEST_COUNTS.values())
        or attempt["lane_request_counts"] != dict.fromkeys(LANE_IDS, _REQUESTS_PER_LANE)
        or attempt["independent_unit"] is not True
        or attempt["model_id"] != MODEL_ID
        or attempt["tokenizer_id"] != TOKENIZER_ID
        or attempt["frozen_workload_digest"] != EXPECTED_CALIBRATED_WORKLOAD_DIGEST
        or attempt["frozen_lane_digests"] != EXPECTED_CALIBRATED_LANE_DIGESTS
        or attempt["model_artifact_digest"] != EXPECTED_MODEL_ARTIFACT_DIGEST
        or attempt["runtime_identity_digest"]
        != _digest_bytes(_json_bytes(_EXPECTED_RUNTIME_IDENTITY))
        or attempt["cache_config_digest"]
        != _digest_bytes(_json_bytes(_experiment_cache_config().to_dict()))
    ):
        raise RealMLXExperimentError("complete attempt binding is invalid")

    result = verify_bundle(directory / "bundle")
    manifest, records = read_bundle(directory / "bundle")
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
        or manifest.cache_config != expected_cache
        or manifest.seed != REPLICATE_IDS.index(replicate_id)
        or result["request_count"] != attempt["request_count"]
        or tuple(manifest.request_order)
        != tuple(record.spec.request_id for record in records)
    ):
        raise RealMLXExperimentError("replicate standard bundle contract mismatch")

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


def _replicate_index(root: Path, *, public: bool) -> dict[str, Any]:
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
    for replicate_id in REPLICATE_IDS:
        directory = root / "replicates" / replicate_id
        state = verify_replicate(directory, replicate_id=replicate_id, public=public)
        entry: dict[str, Any] = dict(state)
        if state["status"] == "complete":
            complete += 1
            manifest, records = read_bundle(directory / "bundle")
            attempt = _safe_object(directory / "attempt.json")
            environment = _safe_object(directory / "environment.json")
            compatible_bindings.add(
                (
                    str(attempt["frozen_workload_digest"]),
                    canonical_json(attempt["frozen_lane_digests"]),
                    str(attempt["model_artifact_digest"]),
                    str(attempt["model_id"]),
                    str(attempt["tokenizer_id"]),
                    str(attempt["runtime_identity_digest"]),
                    str(attempt["cache_config_digest"]),
                    str(manifest.generator_commit),
                    str(manifest.generator_package_digest),
                    _digest_bytes(_json_bytes(environment)),
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
    if complete < 5:
        raise RealMLXExperimentError(
            "aggregate requires at least 5 complete of 6 attempts"
        )
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
        "mlx_fetch_refreshes_lru": False,
        "mlx_insertion_refreshes_lru": True,
        "exact_repeat_policy": "N-1",
    }


def _comparison_name(
    lane_id: str, control: RequestEvidence, treatment: RequestEvidence
) -> str:
    scenario: ScenarioKind = treatment.spec.scenario
    if scenario is ScenarioKind.IDENTICAL_PREFIX:
        if treatment.spec.input_token_count < control.spec.input_token_count:
            case = "longer-to-shorter"
        else:
            case = "cold-exact"
        return f"{lane_id}:{case}"
    if scenario is ScenarioKind.SUFFIX_CHANGE:
        if treatment.spec.input_token_count > control.spec.input_token_count:
            case = "shorter-to-longer"
        else:
            case = "suffix-only-change"
        return f"{lane_id}:{case}"
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
        if request_id is None:
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
    return {
        "replicate_id": replicate_id,
        "lane_id": lane_id,
        "control_input_tokens": control.spec.input_token_count,
        "treatment_input_tokens": treatment.spec.input_token_count,
        "semantic_prefix_tokens": _fact_number(treatment.reuse.semantic_prefix_tokens),
        "policy_reusable_tokens": _fact_number(treatment.reuse.policy_reusable_tokens),
        "policy_reusable_blocks": _fact_number(treatment.reuse.reusable_blocks),
        "engine_cached_tokens": _fact_number(treatment.reuse.engine_cached_tokens),
        "engine_created_tokens": _fact_number(treatment.reuse.engine_created_tokens),
        "observed_prompt_tokens": _fact_number(treatment.reuse.observed_prompt_tokens),
        "unexpected_recomputed_tokens": _fact_number(
            treatment.reuse.unexpected_recomputed_tokens
        ),
        "control_client_ttft_seconds": control_ttft,
        "treatment_client_ttft_seconds": treatment_ttft,
        "client_ttft_difference_seconds": _delta(control_ttft, treatment_ttft),
        "client_ttft_ratio": _ratio(control_ttft, treatment_ttft),
        "control_total_seconds": control_total,
        "treatment_total_seconds": treatment_total,
        "total_difference_seconds": _delta(control_total, treatment_total),
        "total_ratio": _ratio(control_total, treatment_total),
        "control_allocator_peak_bytes": control_peak,
        "treatment_allocator_peak_bytes": treatment_peak,
        "allocator_peak_difference_bytes": _delta(control_peak, treatment_peak),
        "allocator_active_difference_bytes": _delta(control_active, treatment_active),
        "allocator_cache_difference_bytes": _delta(
            control_allocator_cache, treatment_allocator_cache
        ),
        "control_process_rss_bytes": control_rss,
        "treatment_process_rss_bytes": treatment_rss,
        "process_rss_difference_bytes": _delta(control_rss, treatment_rss),
        "control_system_swap_used_bytes": control_swap,
        "treatment_system_swap_used_bytes": treatment_swap,
        "system_swap_difference_bytes": _delta(control_swap, treatment_swap),
        "control_system_memory_free_percent": control_memory_free,
        "treatment_system_memory_free_percent": treatment_memory_free,
        "system_memory_free_difference_percentage_points": _delta(
            control_memory_free,
            treatment_memory_free,
        ),
        "output_token_identity": treatment.output.token_identity.value,
        "deterministic_correctness": treatment.output.correctness.value,
        "verdict": (None if treatment.verdict is None else treatment.verdict.value),
        "performance_eligibility": treatment.eligibility.performance.value,
        "output_eligibility": treatment.eligibility.output_equivalence.value,
        "quality_eligibility": treatment.eligibility.quality.value,
    }


def _descriptive_summary(root: Path, *, public: bool) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for replicate_id in REPLICATE_IDS:
        directory = root / "replicates" / replicate_id
        attempt = _safe_object(directory / "attempt.json")
        if attempt.get("status") != "complete":
            continue
        _, records = read_bundle(directory / "bundle")
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
            "longer-to-shorter",
            "shorter-to-longer",
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
        "allocator_active_difference_bytes",
        "allocator_cache_difference_bytes",
        "process_rss_difference_bytes",
        "system_swap_difference_bytes",
        "system_memory_free_difference_percentage_points",
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
            "identity_true_count": sum(
                sample["output_token_identity"] is True for sample in samples
            ),
            "correctness_true_count": sum(
                sample["deterministic_correctness"] is True for sample in samples
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


def _experiment_contract(public: bool) -> dict[str, Any]:
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "publication_mode": "public_redacted" if public else "private",
        "replicate_ids": list(REPLICATE_IDS),
        "minimum_complete": 5,
        "replacement_allowed": False,
        "execution": "six independent fresh child processes; sequential",
        "parent_timeout_minutes": 90,
        "max_output_tokens_per_request": MAX_OUTPUT_TOKENS,
        "max_cache_entries_per_lifecycle": MAX_CACHE_ENTRIES,
        "lanes": {
            lane_id: {
                "base_tokens": _LANE_CONTRACTS[lane_id]["base"],
                "shorter_seed_tokens": _LANE_CONTRACTS[lane_id]["shorter_seed"],
                "eviction_prompt_tokens": _LANE_CONTRACTS[lane_id]["eviction"],
                "cases": list(_CASES),
                "requests_per_replicate": _REQUESTS_PER_LANE,
            }
            for lane_id in LANE_IDS
        },
        "combined_blocks": list(_BLOCKS),
        "rotation_offsets": list(ROTATION_OFFSETS),
        "model_contract_file_count": EXPECTED_MODEL_FILE_COUNT,
        "network_allowed": False,
    }


def _replicate_tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "replicates").rglob("*")):
        if path.is_symlink():
            raise RealMLXExperimentError("replicate evidence contains a symlink")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("ascii")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _aggregate_verifier_source(
    public: bool,
    descriptive_summary_digest: str,
    replicate_tree_digest: str,
) -> str:
    expected_specs = {
        replicate_id: _schedule_shape_specs(replicate_id, public=public)
        for replicate_id in REPLICATE_IDS
    }
    expected_record_lanes = {
        replicate_id: _scheduled_record_lanes(replicate_id)
        for replicate_id in REPLICATE_IDS
    }
    return f'''"""Portable standard-library verifier for one real MLX aggregate."""
import hashlib
import json
import re
import sys
from pathlib import Path

PUBLIC = {public!r}
ROOT_FILES = {sorted(_AGGREGATE_FILES)!r}
REPLICATES = {list(REPLICATE_IDS)!r}
LANES = {list(LANE_IDS)!r}
BUNDLE_FILES = {list(BUNDLE_FILES)!r}
DESCRIPTIVE_SUMMARY_SHA256 = {descriptive_summary_digest!r}
REPLICATE_TREE_SHA256 = {replicate_tree_digest!r}
EXPECTED_SPECS = {expected_specs!r}
EXPECTED_ROTATIONS = {{replicate_id: rotation for replicate_id, rotation in {dict(zip(REPLICATE_IDS, (block_schedule(item) for item in REPLICATE_IDS), strict=True))!r}.items()}}
EXPECTED_RECORD_LANES = {expected_record_lanes!r}
BLOCK_COUNTS = {_BLOCK_REQUEST_COUNTS!r}
EXPECTED_LANE_REQUEST_COUNTS = {{lane_id: {_REQUESTS_PER_LANE} for lane_id in LANES}}
EXPECTED_WORKLOAD_DIGEST = {EXPECTED_CALIBRATED_WORKLOAD_DIGEST!r}
EXPECTED_LANE_DIGESTS = {EXPECTED_CALIBRATED_LANE_DIGESTS!r}
EXPECTED_MODEL_ARTIFACT_DIGEST = {EXPECTED_MODEL_ARTIFACT_DIGEST!r}
EXPECTED_MODEL_ID = {MODEL_ID!r}
EXPECTED_TOKENIZER_ID = {TOKENIZER_ID!r}
EXPECTED_RUNTIME_DIGEST = {_digest_bytes(_json_bytes(_EXPECTED_RUNTIME_IDENTITY))!r}
EXPECTED_CACHE_CONFIG_DIGEST = {_digest_bytes(_json_bytes(_experiment_cache_config().to_dict()))!r}
PRIVATE_RUNTIME = {_EXPECTED_RUNTIME_IDENTITY!r}
PRIVATE_CACHE_CONFIG = {_experiment_cache_config().to_dict()!r}
MAX_ALLOCATOR_PEAK_BYTES = {MAX_ALLOCATOR_PEAK_BYTES}
MAX_PROCESS_RSS_BYTES = {MAX_PROCESS_RSS_BYTES}
MAX_SWAP_BYTES = {MAX_SWAP_BYTES}
MIN_RUNTIME_MEMORY_FREE_PERCENT = {MIN_RUNTIME_MEMORY_FREE_PERCENT!r}

def fail(message):
    raise SystemExit(message)

def read(path):
    if path.is_symlink() or not path.is_file():
        fail("non-regular artifact")
    return path.read_bytes()

def canonical(value):
    return (json.dumps(value, indent=2, sort_keys=True,
                       ensure_ascii=True, allow_nan=False) + "\\n").encode("ascii")

def replicate_tree_digest(root):
    digest = hashlib.sha256()
    for path in sorted((root / "replicates").rglob("*")):
        if path.is_symlink():
            fail("replicate evidence contains a symlink")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("ascii")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = read(path)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()

def claim(index):
    return {{
        "schema_version": "1",
        "claim_rule": "A hit alone does not prove saved work or latency; article claims require the compatible verified aggregate cell and its raw paired samples.",
        "complete_replicates": index["complete_replicates"],
        "scenario_observations": index["scenario_counts"],
        "lane_scenario_observations": index["lane_scenario_counts"],
        "lane_request_observations": index["lane_request_counts"],
        "verdict_observations": index["verdict_counts"],
        "allocation_step_boundary_256_is_block_cache_claim": False,
        "mlx_fetch_refreshes_lru": False,
        "mlx_insertion_refreshes_lru": True,
        "exact_repeat_policy": "N-1",
    }}

def summary(index):
    return {{
        "schema_version": "1",
        "publication_mode": "public_redacted" if PUBLIC else "private",
        "attempted_replicates": 6,
        "complete_replicates": index["complete_replicates"],
        "failed_replicates": 6 - int(index["complete_replicates"]),
        "request_count": index["request_count"],
        "lane_request_counts": index["lane_request_counts"],
        "no_replacement": True,
        "sequential_boundary_minutes": 90,
        "missing_facts": "null",
    }}

def html(value):
    return ('<!doctype html><meta charset="utf-8"><title>Real MLX cache audit</title>'
            '<h1>Real Apple Silicon MLX KV-cache experiment</h1>'
            f'<p>Complete replicates: {{value["complete_replicates"]}}/6; '
            f'requests: {{value["request_count"]}}.</p>'
            '<p>A cache hit alone does not prove saved work or latency.</p>\\n').encode()

def reuse(index):
    complete = int(index["complete_replicates"])
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="120" '
            'role="img" aria-label="complete replicate count">'
            '<rect width="640" height="120" fill="#fff"/>'
            f'<rect x="20" y="50" width="{{complete * 90}}" height="30" fill="#2867b2"/>'
            f'<text x="20" y="30">Complete replicates: {{complete}}/6</text></svg>\\n').encode()

def timing(index):
    requests = int(index["request_count"])
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="640" height="120" '
            'role="img" aria-label="timing and memory evidence scope">'
            '<rect width="640" height="120" fill="#fff"/>'
            f'<text x="20" y="35">Requests with preserved observations: {{requests}}</text>'
            '<text x="20" y="70">Timing and memory remain separate claim dimensions.</text>'
            '</svg>\\n').encode()

def verify_schedule_and_stages(directory, replicate_id, attempt, records):
    specs = []
    for record in records:
        spec = dict(record["spec"])
        spec["input_token_ids"] = None
        specs.append(spec)
    if specs != EXPECTED_SPECS[replicate_id]:
        fail("replicate request schedule drifted")
    if PUBLIC:
        binding = json.loads(read(directory / "workload-binding.json"))
        expected_digest = "sha256:" + hashlib.sha256(canonical(specs)).hexdigest()
        if binding != {{
            "schema_version": "1",
            "frozen_workload_digest": attempt["frozen_workload_digest"],
            "frozen_lane_digests": attempt["frozen_lane_digests"],
            "lane_request_counts": attempt["lane_request_counts"],
            "request_specs_digest": expected_digest,
        }}:
            fail("public workload binding mismatch")
    rows = [
        json.loads(line)
        for line in read(directory / "stages.jsonl").splitlines()
    ]
    expected = []
    record_index = 0
    for block in attempt["rotation"]:
        expected.append((None, "lifecycle_ready"))
        count = BLOCK_COUNTS.get(block)
        if count is None:
            fail("unknown lifecycle block")
        block_records = records[record_index:record_index + count]
        for record in block_records:
            expected.extend(
                (record["spec"]["request_id"], stage)
                for stage in (
                    "request_before_lookup",
                    "request_after_lookup",
                    "request_after_generation",
                    "request_after_insertion",
                )
            )
        expected.extend(
            (record["spec"]["request_id"], "request_after_baseline")
            for record in block_records
        )
        record_index += count
    if record_index != len(records) or len(rows) != len(expected):
        fail("stage sequence does not cover every request")
    row_keys = {{
        "schema_version", "replicate_id", "request_id", "stage", "allocator",
        "logical_cache", "process_rss", "system_swap", "system_memory",
        "thermal_power",
    }}
    for row, boundary in zip(rows, expected):
        if set(row) != row_keys or (row["request_id"], row["stage"]) != boundary:
            fail("stage boundary sequence mismatch")
        if row["schema_version"] != "1" or row["replicate_id"] != replicate_id:
            fail("stage replicate binding mismatch")
        allocator = row["allocator"]
        if (set(allocator) != {{"active_bytes", "cache_bytes", "peak_bytes", "scope"}}
                or allocator["scope"] != "mlx_process_global_allocator"):
            fail("allocator stage contract mismatch")
        values = (allocator["active_bytes"], allocator["cache_bytes"], allocator["peak_bytes"])
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            fail("allocator stage value is invalid")
        if allocator["peak_bytes"] > MAX_ALLOCATOR_PEAK_BYTES:
            fail("allocator safety limit exceeded")
        logical = row["logical_cache"]
        if (set(logical) != {{"bytes", "scope"}}
                or logical["scope"] != "current_lru_entry_when_observable"
                or (logical["bytes"] is not None and (
                    isinstance(logical["bytes"], bool)
                    or not isinstance(logical["bytes"], int)
                    or logical["bytes"] < 0
                ))):
            fail("logical cache stage contract mismatch")
        rss = row["process_rss"]
        swap = row["system_swap"]
        if (set(rss) != {{"bytes", "scope"}}
                or rss["scope"] != "current_replicate_child_process_only"
                or isinstance(rss["bytes"], bool)
                or not isinstance(rss["bytes"], int)
                or not 0 <= rss["bytes"] <= MAX_PROCESS_RSS_BYTES):
            fail("RSS stage contract mismatch")
        if (set(swap) != {{"used_bytes", "scope"}}
                or swap["scope"] != "system_wide"
                or isinstance(swap["used_bytes"], bool)
                or not isinstance(swap["used_bytes"], int)
                or not 0 <= swap["used_bytes"] <= MAX_SWAP_BYTES):
            fail("swap stage contract mismatch")
        memory = row["system_memory"]
        if (set(memory) != {{"free_percent", "scope"}}
                or memory["scope"] != "system_wide_memory_pressure"
                or isinstance(memory["free_percent"], bool)
                or not isinstance(memory["free_percent"], (int, float))
                or not MIN_RUNTIME_MEMORY_FREE_PERCENT <= memory["free_percent"] <= 100):
            fail("system memory stage contract mismatch")
        if row["thermal_power"] != {{
            "thermal_state": None,
            "power_watts": None,
            "scope": "unavailable_without_safe_collector",
        }}:
            fail("thermal and power contract mismatch")

def rebuild_index(root):
    entries = []
    verdicts = {{}}
    scenarios = {{}}
    lane_requests = {{lane_id: 0 for lane_id in LANES}}
    lane_scenarios = {{lane_id: {{}} for lane_id in LANES}}
    request_count = 0
    complete = 0
    environments = []
    bindings = []
    for replicate_id in REPLICATES:
        directory = root / "replicates" / replicate_id
        attempt = json.loads(read(directory / "attempt.json"))
        entry = {{"replicate_id": replicate_id, "status": attempt["status"]}}
        if attempt["status"] == "complete":
            complete += 1
            environments.append(json.loads(read(directory / "environment.json")))
            bundle = directory / "bundle"
            checksum_lines = read(bundle / "SHA256SUMS").decode("ascii").splitlines()
            nested = {{}}
            for line in checksum_lines:
                digest, name = line.split("  ")
                if name in nested:
                    fail("duplicate standard bundle checksum")
                nested[name] = digest
            if set(nested) != set(BUNDLE_FILES) - {{"SHA256SUMS"}}:
                fail("standard bundle checksum allowlist mismatch")
            for name, digest in nested.items():
                if hashlib.sha256(read(bundle / name)).hexdigest() != digest:
                    fail("standard bundle checksum mismatch")
            manifest = json.loads(read(bundle / "audit-manifest.json"))
            expected_cache = dict(PRIVATE_CACHE_CONFIG)
            if PUBLIC:
                expected_cache["namespace_id"] = "redacted-namespace"
                expected_cache["cache_type"] = "redacted-cache"
            if (manifest["backend"] != "mlx_lm_local"
                    or manifest["backend_version"] != ("redacted" if PUBLIC else "0.31.3")
                    or manifest["model_id"] != ("redacted-model" if PUBLIC else EXPECTED_MODEL_ID)
                    or manifest["tokenizer_id"] != ("redacted-tokenizer" if PUBLIC else EXPECTED_TOKENIZER_ID)
                    or manifest["model_artifact_digest"] != (None if PUBLIC else EXPECTED_MODEL_ARTIFACT_DIGEST)
                    or manifest["runtime_identity"] != ({{"redaction": "public"}} if PUBLIC else PRIVATE_RUNTIME)
                    or manifest["cache_config"] != expected_cache):
                fail("standard bundle identity mismatch")
            bindings.append({{
                "frozen_workload_digest": attempt["frozen_workload_digest"],
                "frozen_lane_digests": attempt["frozen_lane_digests"],
                "model_artifact_digest": attempt["model_artifact_digest"],
                "model_id": attempt["model_id"],
                "tokenizer_id": attempt["tokenizer_id"],
                "runtime_identity_digest": attempt["runtime_identity_digest"],
                "cache_config_digest": attempt["cache_config_digest"],
                "generator_commit": str(manifest["generator_commit"]),
                "generator_package_digest": str(manifest["generator_package_digest"]),
                "environment_digest": "sha256:" + hashlib.sha256(
                    canonical(environments[-1])
                ).hexdigest(),
            }})
            records = [
                json.loads(line)
                for line in read(bundle / "request-evidence.jsonl").splitlines()
            ]
            verify_schedule_and_stages(directory, replicate_id, attempt, records)
            counts = {{}}
            record_lanes = EXPECTED_RECORD_LANES[replicate_id]
            if len(record_lanes) != len(records):
                fail("lane schedule request count mismatch")
            for record, lane_id in zip(records, record_lanes):
                verdict = record["verdict"] or "unclassified"
                counts[verdict] = counts.get(verdict, 0) + 1
                scenario = record["spec"]["scenario"]
                scenarios[scenario] = scenarios.get(scenario, 0) + 1
                lane_requests[lane_id] += 1
                lane_counts = lane_scenarios[lane_id]
                lane_counts[scenario] = lane_counts.get(scenario, 0) + 1
            for verdict, count in counts.items():
                verdicts[verdict] = verdicts.get(verdict, 0) + count
            request_count += len(records)
            entry.update({{
                "request_count": len(records),
                "run_id": manifest["run_id"],
                "verdict_counts": dict(sorted(counts.items())),
            }})
        entries.append(entry)
    if complete < 5:
        fail("minimum replicate gate failed")
    if not environments or any(value != environments[0] for value in environments[1:]):
        fail("environment mismatch")
    if not bindings or any(value != bindings[0] for value in bindings[1:]):
        fail("evidence binding mismatch")
    return ({{
        "schema_version": "1",
        "replicates": entries,
        "complete_replicates": complete,
        "attempted_replicates": 6,
        "request_count": request_count,
        "lane_request_counts": dict(sorted(lane_requests.items())),
        "verdict_counts": dict(sorted(verdicts.items())),
        "scenario_counts": dict(sorted(scenarios.items())),
        "lane_scenario_counts": {{
            lane_id: dict(sorted(lane_scenarios[lane_id].items()))
            for lane_id in LANES
        }},
        "evidence_binding": bindings[0],
    }}, environments[0])

def main():
    root = Path(__file__).resolve().parent
    if {{item.name for item in root.iterdir()}} != set(ROOT_FILES):
        fail("aggregate root allowlist mismatch")
    lines = read(root / "SHA256SUMS").decode("ascii").splitlines()
    found = {{}}
    for line in lines:
        if not re.fullmatch(r"[0-9a-f]{{64}}  [A-Za-z0-9._/-]+", line):
            fail("invalid recursive checksum")
        digest, name = line.split("  ")
        if name in found or name == "SHA256SUMS" or ".." in Path(name).parts:
            fail("invalid checksum path")
        found[name] = digest
    actual = {{
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink() and path != root / "SHA256SUMS"
    }}
    if set(found) != actual:
        fail("recursive checksum allowlist mismatch")
    for name, digest in found.items():
        if hashlib.sha256(read(root / name)).hexdigest() != digest:
            fail("checksum mismatch: " + name)
    if hashlib.sha256(read(root / "descriptive-summary.json")).hexdigest() != DESCRIPTIVE_SUMMARY_SHA256:
        fail("descriptive summary digest mismatch")
    if replicate_tree_digest(root) != REPLICATE_TREE_SHA256:
        fail("raw replicate evidence digest mismatch")
    index, environment = rebuild_index(root)
    if read(root / "replicate-index.json") != canonical(index):
        fail("replicate index is not derived")
    if index["attempted_replicates"] != 6 or index["complete_replicates"] < 5:
        fail("minimum replicate gate failed")
    if [item["replicate_id"] for item in index["replicates"]] != REPLICATES:
        fail("replicate set permits replacement")
    for replicate_id in REPLICATES:
        directory = root / "replicates" / replicate_id
        if directory.is_symlink() or not directory.is_dir():
            fail("replicate directory mismatch")
        attempt = json.loads(read(directory / "attempt.json"))
        if (attempt.get("replicate_id") != replicate_id
                or attempt.get("replacement") is not False
                or attempt.get("status") not in {{"complete", "failed"}}):
            fail("replicate attempt contract mismatch")
        if attempt["status"] == "complete":
            complete_keys = {{
                "schema_version", "replicate_id", "status", "rotation",
                "request_count", "independent_unit", "replacement",
                "lane_request_counts", "frozen_workload_digest",
                "frozen_lane_digests", "model_artifact_digest", "model_id",
                "tokenizer_id", "runtime_identity_digest", "cache_config_digest",
            }}
            if (set(attempt) != complete_keys
                    or tuple(attempt["rotation"]) != tuple(EXPECTED_ROTATIONS[replicate_id])
                    or attempt["request_count"] != sum(BLOCK_COUNTS.values())
                    or attempt["lane_request_counts"] != EXPECTED_LANE_REQUEST_COUNTS
                    or attempt["independent_unit"] is not True
                    or attempt["frozen_workload_digest"] != EXPECTED_WORKLOAD_DIGEST
                    or attempt["frozen_lane_digests"] != EXPECTED_LANE_DIGESTS
                    or attempt["model_artifact_digest"] != EXPECTED_MODEL_ARTIFACT_DIGEST
                    or attempt["model_id"] != EXPECTED_MODEL_ID
                    or attempt["tokenizer_id"] != EXPECTED_TOKENIZER_ID
                    or attempt["runtime_identity_digest"] != EXPECTED_RUNTIME_DIGEST
                    or attempt["cache_config_digest"] != EXPECTED_CACHE_CONFIG_DIGEST):
                fail("complete attempt binding mismatch")
        elif set(attempt) != {{"schema_version", "replicate_id", "status", "replacement"}}:
            fail("failed attempt field mismatch")
        names = {{item.name for item in directory.iterdir()}}
        expected_names = (
            {sorted(_PUBLIC_REPLICATE_FILES if public else _PRIVATE_REPLICATE_FILES)!r}
            if attempt["status"] == "complete"
            else {sorted(_FAILED_REPLICATE_FILES)!r}
        )
        if names != set(expected_names):
            fail("replicate file allowlist mismatch")
        if attempt["status"] == "complete":
            bundle = directory / "bundle"
            if bundle.is_symlink() or {{item.name for item in bundle.iterdir()}} != set(BUNDLE_FILES):
                fail("standard bundle allowlist mismatch")
    expected = {{
        "experiment-contract.json": canonical({_experiment_contract(public)!r}),
        "environment.json": canonical(environment),
        "claim-matrix.json": canonical(claim(index)),
        "summary.json": canonical(summary(index)),
        "report.html": html(summary(index)),
        "reuse-alignment.svg": reuse(index),
        "timing-memory.svg": timing(index),
        "teardown.json": canonical({{
            "schema_version": "1",
            "replicate_records_verified": 6,
            "complete_teardown_records": index["complete_replicates"],
            "scope": "aggregate_of_child_teardown_records",
        }}),
    }}
    for name, content in expected.items():
        if read(root / name) != content:
            fail("derived artifact mismatch: " + name)
    if PUBLIC:
        private_markers = ("/Us" + "ers/", "/ho" + "me/", ":\\\\Us" + "ers\\\\")
        private_keys = {{
            "absolute_path", "account_id", "account_identifier", "api_key",
            "cache_path", "cookie", "email", "home", "host_name", "hostname",
            "local_path", "model_path", "pid", "process_id", "raw_prompt",
            "raw_response", "reasoning_text", "user_id", "user_name", "username",
        }}
        def scan(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key.casefold() in private_keys:
                        fail("public aggregate contains private JSON field")
                    if key in {{"input_token_ids", "output_token_ids", "baseline_token_ids"}} and isinstance(item, list):
                        fail("public aggregate contains exact token arrays")
                    scan(item)
            elif isinstance(value, list):
                for item in value:
                    scan(item)
        for name in found:
            text = read(root / name).decode("utf-8", "ignore")
            if any(marker in text for marker in private_markers):
                fail("public aggregate contains private path")
            if ('"input_token_' + 'ids":[') in text or ('"output_token_' + 'ids":[') in text:
                fail("public aggregate contains exact token arrays")
            if (re.search(r"\\b[\\w.+-]+@[\\w.-]+\\.[A-Za-z]{{2,}}\\b", text)
                    or re.search(r"\\b(?:gh[pousr]_|github_pat_|sk-|hf[_-]|wk-|ws-)[A-Za-z0-9_-]{{8,}}\\b", text)
                    or re.search(r"\\b[0-9a-f]{{8}}-(?:[0-9a-f]{{4}}-){{3}}[0-9a-f]{{12}}\\b", text, re.I)
                    or ("-----BEGIN " + "PRIVATE KEY-----") in text):
                fail("public aggregate contains sensitive text")
            path = root / name
            if path.suffix == ".json":
                scan(json.loads(text))
            elif path.suffix == ".jsonl":
                for line in text.splitlines():
                    scan(json.loads(line))
    print(json.dumps({{"verified": True, "complete_replicates": index["complete_replicates"]}}, sort_keys=True))

if __name__ == "__main__":
    main()
'''


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
        _safe_object(replicates / replicate_id / "environment.json")
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
    index = _replicate_index(root, public=public)
    summary = _summary(index, public=public)
    descriptive_summary = _descriptive_summary(root, public=public)
    _write_json(root / "experiment-contract.json", _experiment_contract(public))
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
    atomic_write_text(
        root / "aggregate_verifier.py",
        _aggregate_verifier_source(
            public,
            hashlib.sha256(
                canonical_json(descriptive_summary).encode("ascii")
            ).hexdigest(),
            _replicate_tree_digest(root),
        ),
    )
    _write_recursive_checksums(root)


def assemble_aggregate(attempts_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Assemble exactly six attempted directories; failed IDs are never replaced."""

    if output_dir.exists():
        raise RealMLXExperimentError("aggregate output already exists")
    actual = {item.name for item in attempts_dir.iterdir()}
    if actual != set(REPLICATE_IDS):
        raise RealMLXExperimentError(
            "attempt directory must contain replicate-0..5 exactly"
        )
    output_dir.mkdir(parents=True)
    (output_dir / "replicates").mkdir()
    try:
        for replicate_id in REPLICATE_IDS:
            verify_replicate(
                attempts_dir / replicate_id,
                replicate_id=replicate_id,
                public=False,
            )
            _copy_tree(
                attempts_dir / replicate_id,
                output_dir / "replicates" / replicate_id,
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
            manifest, records = read_bundle(source_rep / "bundle")
            redacted_manifest, redacted_records = sanitize_bundle_records(
                manifest, records
            )
            write_bundle(target_rep / "bundle", redacted_manifest, redacted_records)
            verify_bundle(target_rep / "bundle")
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
            shutil.copyfile(
                source_rep / "environment.json", target_rep / "environment.json"
            )
            shutil.copyfile(source_rep / "teardown.json", target_rep / "teardown.json")
            _sanitize_stages(
                source_rep / "stages.jsonl",
                target_rep / "stages.jsonl",
                records,
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


def _verify_public_aggregate_privacy(root: Path) -> None:
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
    _verify_recursive_checksums(root)
    contract = _safe_object(root / "experiment-contract.json")
    mode = contract.get("publication_mode")
    inferred_public = mode == "public_redacted"
    if public is not None and public != inferred_public:
        raise RealMLXExperimentError("aggregate publication mode mismatch")
    expected_contract = _experiment_contract(inferred_public)
    if contract != expected_contract:
        raise RealMLXExperimentError("experiment contract mismatch")
    if {item.name for item in (root / "replicates").iterdir()} != set(REPLICATE_IDS):
        raise RealMLXExperimentError("aggregate replicate allowlist mismatch")
    index = _replicate_index(root, public=inferred_public)
    descriptive_summary = _descriptive_summary(root, public=inferred_public)
    expected_files: dict[str, str] = {
        "experiment-contract.json": canonical_json(expected_contract),
        "environment.json": canonical_json(_common_environment(root / "replicates")),
        "replicate-index.json": canonical_json(index),
        "claim-matrix.json": canonical_json(_claim_matrix(index)),
        "descriptive-summary.json": canonical_json(descriptive_summary),
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
        "aggregate_verifier.py": _aggregate_verifier_source(
            inferred_public,
            hashlib.sha256(
                canonical_json(descriptive_summary).encode("ascii")
            ).hexdigest(),
            _replicate_tree_digest(root),
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
    replicate_parser.add_argument(
        "--conversion-summary", type=Path, default=DEFAULT_CONVERSION_SUMMARY
    )
    failed_parser = commands.add_parser("record-failure")
    failed_parser.add_argument("--replicate-id", choices=REPLICATE_IDS, required=True)
    failed_parser.add_argument("--output-dir", type=Path, required=True)
    aggregate_parser = commands.add_parser("aggregate")
    aggregate_parser.add_argument("--attempts-dir", type=Path, required=True)
    aggregate_parser.add_argument("--output-dir", type=Path, required=True)
    sanitize_parser = commands.add_parser("sanitize")
    sanitize_parser.add_argument("aggregate", type=Path)
    sanitize_parser.add_argument("--output-dir", type=Path, required=True)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("aggregate", type=Path)
    return parser


def _run_cli(args: argparse.Namespace) -> dict[str, Any]:
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
            "seed_output_tokens": {
                lane.lane_id: len(lane.seed_output or ()) for lane in workload.lanes
            },
        }
    if args.command == "replicate":
        return run_replicate(
            args.workload,
            args.output_dir,
            replicate_id=args.replicate_id,
            model_dir=args.model_dir,
            conversion_summary=args.conversion_summary,
        )
    if args.command == "record-failure":
        record_failed_attempt(args.output_dir, args.replicate_id)
        return {"recorded": True, "status": "failed"}
    if args.command == "aggregate":
        return assemble_aggregate(args.attempts_dir, args.output_dir)
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
    raise SystemExit(0)


if __name__ == "__main__":
    main(sys.argv[1:])
