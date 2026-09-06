"""Source-bound live vLLM 0.28.0 runtime-attestation validator and event parser.

This module is the *only* path by which a vLLM audit can be treated as
runtime-backed rather than offline/synthetic, and it is intentionally
separate from :mod:`llmtracefx.cache_audit.adapters.vllm`, whose
``assess_vllm_capabilities`` remains unconditionally unsupported regardless of
input. Nothing here weakens that offline refusal: this module adds a new,
narrower path with two sealed artifacts, generated in strict order:

1. :class:`IdentityReceipt` -- exported *before* engine construction, sealed
   with SHA-256 over its own canonical JSON. It contains only what is
   knowable pre-init: image/runtime/GPU/model identity and the preregistered
   source-file manifest comparison. It never claims resolved cache layout or
   KV-event capture boundaries, because those do not exist yet.
2. :class:`RuntimeAttestation` -- exported *after* engine construction and
   after the KV-event subscriber has started, binding to the identity
   receipt's own seal (not duplicating its fields) and adding the resolved
   cache layout and event-capture boundaries the engine and its publisher
   actually report, sealed with its own SHA-256 over its own canonical JSON.

A caller cannot construct a supported result by passing plain strings,
environment variables, or CLI flags -- every identity claim must already be
present, self-consistent, and sealed inside the mappings handed to
:func:`parse_identity_receipt` and :func:`parse_runtime_attestation`.

This module never imports ``vllm``, ``torch``, ``zmq``, or ``msgspec``, opens
no socket, and performs no GPU or network work. It only parses and validates
mappings that a separate, gated in-container exporter is expected to produce
(see ``llmtracefx.optimizer.lab.qwen3_8b.kv_truth_runner`` for the pinned
protocol that consumes this validator) and normalizes already msgpack-decoded
KV-cache event batches captured from vLLM's ZMQ publisher.

The event schema below mirrors vLLM 0.28.0's actual
``vllm/distributed/kv_events.py`` structures, at the exact pinned commit
``2cf0a6915ce544dc493a0990f2ea38d81601128a`` -- ``EventBatch`` is an
``array_like=True`` msgspec Struct, so it decodes off the wire as a plain
``[ts, events]``/``[ts, events, data_parallel_rank]`` array, never a
mapping; ``BlockStored`` (including the
deprecated ``lora_id``, plus ``lora_name``, ``extra_keys``, ``group_idx``,
``kv_cache_spec_kind``, ``kv_cache_spec_sliding_window``, ``locality``),
``BlockRemoved`` (``locality`` in addition to
``block_hashes``/``medium``/``group_idx``), and ``AllBlocksCleared`` -- plus
the ZMQ multipart *sequence frame* that ``ZmqEventPublisher`` prepends to
every published payload: an 8-byte big-endian sequence number carried as its
own frame alongside the topic and the msgpack payload. This is a strictly
larger, distinct schema from the existing synthetic ``SyntheticKVEvent`` in
``adapters.vllm``, which is
intentionally not reused as a live schema.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from llmtracefx.optimizer.schema import SchemaValidationError

from ..schema import CacheEventRecord, EvidenceBasis
from .base import CacheAuditCapability
from .vllm import (
    REQUIRED_KV_EVENTS_USE_INT_BLOCK_HASHES,
    REQUIRED_PREFIX_CACHING_HASH_ALGO,
    REQUIRED_VLLM_COMMIT,
    REQUIRED_VLLM_VERSION,
)

BACKEND = "vllm_live"

#: The ZMQ multipart sequence frame vLLM's ``ZmqEventPublisher`` prepends to
#: every published batch is an 8-byte big-endian integer. ``END_SEQ`` is the
#: reserved sentinel the replay ROUTER socket sends to mark "no more buffered
#: batches"; it is never a real batch sequence number.
SEQUENCE_FRAME_BYTES = 8
END_OF_REPLAY_SEQUENCE = -1

#: vLLM 0.28.0 declares its published ``EventBatch`` as a
#: ``msgspec.Struct(array_like=True)``, so msgpack encodes it as a plain
#: array, not a mapping, and msgspec (like this module's ``msgspec``-free
#: stand-in decode) omits any trailing field left at its declared default
#: when encoding an array-like struct. ``data_parallel_rank: int = 0`` is
#: the trailing field, so a batch published from the default data-parallel
#: rank decodes to a two-element ``[ts, events]`` array, never a
#: three-element one; this is the default this module fills in for that
#: omitted case, not an assumption about a "None" rank.
DEFAULT_DATA_PARALLEL_RANK = 0

_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_COMMIT_HEX = re.compile(r"^[0-9a-f]{40}$")

#: The committed, offline-readable manifest fixing exactly which vLLM 0.28.0
#: source files (at the pinned commit) an identity receipt must independently
#: hash and compare. This module never fetches upstream source itself; it
#: only enforces that the *set of paths* a receipt claims to have compared
#: equals this fixed list exactly, in addition to every comparison having
#: succeeded (see :func:`_parse_source_file_digests`).
_SOURCE_MANIFEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "optimizer"
    / "lab"
    / "qwen3_8b"
    / "data"
    / "qwen3-8b-vllm-source-manifest-v1.json"
)


@functools.lru_cache(maxsize=1)
def _load_source_manifest() -> dict[str, Any]:
    """Read and structurally validate the committed vLLM source manifest.

    Raises :class:`SchemaValidationError` (never a raw I/O or JSON error) if
    the manifest is missing, malformed, or does not pin this module's
    required ``vllm_version``/``vllm_commit``, so a corrupted or tampered
    manifest fails closed exactly like every other check in this module.
    """

    try:
        raw = json.loads(_SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SchemaValidationError(
            f"vLLM source manifest is unreadable: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise SchemaValidationError("vLLM source manifest is not a JSON object")
    if raw.get("vllm_version") != REQUIRED_VLLM_VERSION:
        raise SchemaValidationError(
            "vLLM source manifest vllm_version does not match "
            f"{REQUIRED_VLLM_VERSION!r}"
        )
    if raw.get("vllm_commit") != REQUIRED_VLLM_COMMIT:
        raise SchemaValidationError(
            "vLLM source manifest vllm_commit does not match "
            f"{REQUIRED_VLLM_COMMIT!r}"
        )
    paths = raw.get("critical_source_files")
    if (
        not isinstance(paths, list)
        or not paths
        or not all(isinstance(item, str) and item for item in paths)
    ):
        raise SchemaValidationError(
            "vLLM source manifest critical_source_files must be a non-empty "
            "array of non-empty strings"
        )
    if len(set(paths)) != len(paths):
        raise SchemaValidationError(
            "vLLM source manifest critical_source_files contains duplicates"
        )
    digests = raw.get("critical_source_file_sha256")
    if not isinstance(digests, dict):
        raise SchemaValidationError(
            "vLLM source manifest critical_source_file_sha256 must be a JSON object"
        )
    if set(digests) != set(paths):
        raise SchemaValidationError(
            "vLLM source manifest critical_source_file_sha256 keys must "
            "exactly match critical_source_files"
        )
    for path, digest in digests.items():
        if not isinstance(digest, str) or _SHA256_HEX.fullmatch(digest) is None:
            raise SchemaValidationError(
                f"vLLM source manifest expected digest for {path!r} is not "
                "256-bit SHA-256 lowercase hex"
            )
    return raw


def required_source_file_paths() -> frozenset[str]:
    """Return the fixed set of critical vLLM source file paths, from the
    committed manifest at :data:`_SOURCE_MANIFEST_PATH`.
    """

    return frozenset(_load_source_manifest()["critical_source_files"])


def required_source_file_digests() -> Mapping[str, str]:
    """Return the fixed mapping of critical vLLM source file path to its
    real, independently fetched (source fetch only, at the pinned commit)
    expected SHA-256 hex digest, from the committed manifest.

    This is what makes source-file verification a genuine byte-content
    comparison against real expected values rather than a caller-supplied
    ``matches_manifest`` boolean this module would otherwise merely trust:
    see :func:`_parse_source_file_digests`, which recomputes this
    comparison itself from every reported ``sha256`` value.
    """

    return dict(_load_source_manifest()["critical_source_file_sha256"])


def canonical_json(value: Any) -> str:
    """Return finite, key-sorted JSON used for every seal in this module."""

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SchemaValidationError(f"value is not canonical JSON: {exc}") from exc


def sha256_digest(value: Any) -> str:
    """Return a ``sha256:<hex>`` digest over ``value``'s canonical JSON."""

    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{context} must be an object")
    return dict(value)


def _exact_keys(data: Mapping[str, Any], keys: set[str], context: str) -> None:
    actual = set(data)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise SchemaValidationError(
            f"{context} fields differ: missing={missing}, extra={extra}"
        )


def _require_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise SchemaValidationError(f"{context} must be a boolean")
    return value


def _require_int(value: Any, context: str, *, minimum: int | None = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise SchemaValidationError(f"{context} must be >= {minimum}")
    return value


def _optional_int(value: Any, context: str, *, minimum: int | None = 0) -> int | None:
    if value is None:
        return None
    return _require_int(value, context, minimum=minimum)


def _require_finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{context} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise SchemaValidationError(f"{context} must be finite")
    return number


def _require_str(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaValidationError(f"{context} must be a non-empty string")
    return value


def _optional_str(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, context)


def _require_pattern(value: Any, context: str, pattern: re.Pattern[str]) -> str:
    text = _require_str(value, context)
    if pattern.fullmatch(text) is None:
        raise SchemaValidationError(f"{context} does not match the required pattern")
    return text


def _require_str_tuple(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SchemaValidationError(f"{context} must be an array")
    return tuple(
        _require_str(item, f"{context}[{index}]") for index, item in enumerate(value)
    )


def _require_int_tuple(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise SchemaValidationError(f"{context} must be an array")
    return tuple(
        _require_int(item, f"{context}[{index}]", minimum=None)
        for index, item in enumerate(value)
    )


# ---------------------------------------------------------------------------
# Runtime attestation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ResolvedVLLMConfig:
    """Canonical resolved runtime configuration exported from inside the container.

    Every field here is the *resolved* value vLLM actually applied, not a
    caller-supplied request: it is only ever constructed by
    :func:`_parse_resolved_config` from a mapping the in-container exporter
    produced from the live ``VllmConfig``.
    """

    max_model_len: int
    max_num_seqs: int
    tensor_parallel_size: int
    data_parallel_size: int
    block_size: int
    prefix_match_unit: int
    num_gpu_blocks_override: int | None
    gpu_memory_utilization: float
    cache_dtype: str
    prefix_caching_hash_algo: str
    kv_events_use_int_block_hashes: str
    pythonhashseed: str
    enable_prefix_caching: bool
    enable_kv_cache_events: bool
    enforce_eager: bool
    speculative_config_enabled: bool
    lora_enabled: bool
    multimodal_enabled: bool
    cache_salt_present: bool
    cache_salt: str | None

    def redact(self) -> dict[str, Any]:
        """Publication-safe view: never emits the raw ``cache_salt`` value."""

        payload = self.to_dict()
        payload["cache_salt"] = None
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "block_size": self.block_size,
            "prefix_match_unit": self.prefix_match_unit,
            "num_gpu_blocks_override": self.num_gpu_blocks_override,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "cache_dtype": self.cache_dtype,
            "prefix_caching_hash_algo": self.prefix_caching_hash_algo,
            "kv_events_use_int_block_hashes": self.kv_events_use_int_block_hashes,
            "pythonhashseed": self.pythonhashseed,
            "enable_prefix_caching": self.enable_prefix_caching,
            "enable_kv_cache_events": self.enable_kv_cache_events,
            "enforce_eager": self.enforce_eager,
            "speculative_config_enabled": self.speculative_config_enabled,
            "lora_enabled": self.lora_enabled,
            "multimodal_enabled": self.multimodal_enabled,
            "cache_salt_present": self.cache_salt_present,
            "cache_salt": self.cache_salt,
        }


_RESOLVED_CONFIG_KEYS = set(ResolvedVLLMConfig.__dataclass_fields__)


def _parse_resolved_config(value: Any) -> ResolvedVLLMConfig:
    data = _require_mapping(value, "resolved_config")
    _exact_keys(data, _RESOLVED_CONFIG_KEYS, "resolved_config")
    cache_salt = _optional_str(data["cache_salt"], "resolved_config.cache_salt")
    cache_salt_present = _require_bool(
        data["cache_salt_present"], "resolved_config.cache_salt_present"
    )
    if cache_salt_present != (cache_salt is not None):
        raise SchemaValidationError(
            "resolved_config.cache_salt_present is inconsistent with cache_salt"
        )
    config = ResolvedVLLMConfig(
        max_model_len=_require_int(data["max_model_len"], "max_model_len", minimum=1),
        max_num_seqs=_require_int(data["max_num_seqs"], "max_num_seqs", minimum=1),
        tensor_parallel_size=_require_int(
            data["tensor_parallel_size"], "tensor_parallel_size", minimum=1
        ),
        data_parallel_size=_require_int(
            data["data_parallel_size"], "data_parallel_size", minimum=1
        ),
        block_size=_require_int(data["block_size"], "block_size", minimum=1),
        prefix_match_unit=_require_int(
            data["prefix_match_unit"], "prefix_match_unit", minimum=1
        ),
        num_gpu_blocks_override=_optional_int(
            data["num_gpu_blocks_override"], "num_gpu_blocks_override", minimum=1
        ),
        gpu_memory_utilization=_require_finite_float(
            data["gpu_memory_utilization"], "gpu_memory_utilization"
        ),
        cache_dtype=_require_str(data["cache_dtype"], "cache_dtype"),
        prefix_caching_hash_algo=_require_str(
            data["prefix_caching_hash_algo"], "prefix_caching_hash_algo"
        ),
        kv_events_use_int_block_hashes=_require_str(
            data["kv_events_use_int_block_hashes"],
            "kv_events_use_int_block_hashes",
        ),
        pythonhashseed=_require_str(data["pythonhashseed"], "pythonhashseed"),
        enable_prefix_caching=_require_bool(
            data["enable_prefix_caching"], "enable_prefix_caching"
        ),
        enable_kv_cache_events=_require_bool(
            data["enable_kv_cache_events"], "enable_kv_cache_events"
        ),
        enforce_eager=_require_bool(data["enforce_eager"], "enforce_eager"),
        speculative_config_enabled=_require_bool(
            data["speculative_config_enabled"], "speculative_config_enabled"
        ),
        lora_enabled=_require_bool(data["lora_enabled"], "lora_enabled"),
        multimodal_enabled=_require_bool(
            data["multimodal_enabled"], "multimodal_enabled"
        ),
        cache_salt_present=cache_salt_present,
        cache_salt=cache_salt,
    )
    if not 0.0 < config.gpu_memory_utilization <= 1.0 or not math.isfinite(
        config.gpu_memory_utilization
    ):
        raise SchemaValidationError(
            "resolved_config.gpu_memory_utilization must be in (0, 1]"
        )
    if config.prefix_match_unit % config.block_size:
        raise SchemaValidationError(
            "resolved_config.prefix_match_unit must be a multiple of block_size"
        )
    if not _pythonhashseed_is_fixed(config.pythonhashseed):
        raise SchemaValidationError(
            "resolved_config.pythonhashseed must be a fixed non-random integer"
        )
    return config


def _pythonhashseed_is_fixed(value: str) -> bool:
    stripped = value.strip()
    return bool(stripped) and stripped.lower() != "random" and stripped.isdigit()


@dataclass(frozen=True)
class KVEventsPublisherAttestation:
    """Resolved KV-event publisher configuration, with endpoints redacted to roles.

    The plan requires loopback PUB/replay endpoints to be represented "by
    redacted endpoint roles", never literal host/port strings, so this
    dataclass never stores an ``endpoint``/``replay_endpoint`` string at all
    -- only whether each socket role is bound, which is exactly what remains
    true after redaction.
    """

    topic: str
    endpoint_role: str
    replay_endpoint_role: str | None
    buffer_steps: int
    hwm: int
    max_queue_size: int
    data_parallel_rank: int
    first_sequence: int | None
    last_sequence: int | None
    capture_start_monotonic: float
    capture_end_monotonic: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "endpoint_role": self.endpoint_role,
            "replay_endpoint_role": self.replay_endpoint_role,
            "buffer_steps": self.buffer_steps,
            "hwm": self.hwm,
            "max_queue_size": self.max_queue_size,
            "data_parallel_rank": self.data_parallel_rank,
            "first_sequence": self.first_sequence,
            "last_sequence": self.last_sequence,
            "capture_start_monotonic": self.capture_start_monotonic,
            "capture_end_monotonic": self.capture_end_monotonic,
        }


_KV_EVENTS_ATTESTATION_KEYS = set(KVEventsPublisherAttestation.__dataclass_fields__)
REQUIRED_ENDPOINT_ROLE = "loopback_pub"
REQUIRED_REPLAY_ENDPOINT_ROLE = "loopback_replay"


def _parse_kv_events_config(value: Any) -> KVEventsPublisherAttestation:
    data = _require_mapping(value, "kv_events_config")
    _exact_keys(data, _KV_EVENTS_ATTESTATION_KEYS, "kv_events_config")
    first_sequence = _optional_int(
        data["first_sequence"], "kv_events_config.first_sequence"
    )
    last_sequence = _optional_int(
        data["last_sequence"], "kv_events_config.last_sequence"
    )
    if (
        first_sequence is not None
        and last_sequence is not None
        and last_sequence < first_sequence
    ):
        raise SchemaValidationError(
            "kv_events_config.last_sequence precedes first_sequence"
        )
    start = _require_finite_float(
        data["capture_start_monotonic"], "kv_events_config.capture_start_monotonic"
    )
    end = _require_finite_float(
        data["capture_end_monotonic"], "kv_events_config.capture_end_monotonic"
    )
    if end < start:
        raise SchemaValidationError(
            "kv_events_config.capture_end_monotonic precedes capture_start_monotonic"
        )
    return KVEventsPublisherAttestation(
        topic=(
            data["topic"]
            if isinstance(data["topic"], str)
            else _require_str(data["topic"], "kv_events_config.topic")
        ),
        endpoint_role=_require_str(
            data["endpoint_role"], "kv_events_config.endpoint_role"
        ),
        replay_endpoint_role=_optional_str(
            data["replay_endpoint_role"], "kv_events_config.replay_endpoint_role"
        ),
        buffer_steps=_require_int(
            data["buffer_steps"], "kv_events_config.buffer_steps", minimum=1
        ),
        hwm=_require_int(data["hwm"], "kv_events_config.hwm", minimum=1),
        max_queue_size=_require_int(
            data["max_queue_size"], "kv_events_config.max_queue_size", minimum=1
        ),
        data_parallel_rank=_require_int(
            data["data_parallel_rank"], "kv_events_config.data_parallel_rank"
        ),
        first_sequence=first_sequence,
        last_sequence=last_sequence,
        capture_start_monotonic=start,
        capture_end_monotonic=end,
    )


@dataclass(frozen=True)
class SourceFileDigest:
    """One critical vLLM source file, compared against a preregistered
    manifest's real expected SHA-256 digest (:func:`required_source_file_digests`)
    -- ``matches_manifest`` is independently recomputed by
    :func:`_parse_source_file_digests`, never merely trusted from the
    reported value."""

    path: str
    sha256: str
    matches_manifest: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "matches_manifest": self.matches_manifest,
        }


def _parse_source_file_digests(value: Any) -> tuple[SourceFileDigest, ...]:
    """Strictly parse the receipt's reported per-file digests, independently
    recomputing ``matches_manifest`` from the committed manifest's real
    expected SHA-256 value for that path -- never trusting the reported
    ``matches_manifest`` boolean at face value. A receipt whose reported
    boolean disagrees with the recomputed comparison is rejected outright
    (it is either lying or internally inconsistent), not silently
    corrected. Path-coverage (missing/extra/duplicate) is checked first, so
    an unlisted extra path is reported as such rather than through an
    unrelated digest-consistency error.
    """

    if not isinstance(value, list) or not value:
        raise SchemaValidationError("source_file_digests must be a non-empty array")
    parsed: list[tuple[str, str, bool]] = []
    for index, item in enumerate(value):
        context = f"source_file_digests[{index}]"
        data = _require_mapping(item, context)
        _exact_keys(data, {"path", "sha256", "matches_manifest"}, context)
        parsed.append(
            (
                _require_str(data["path"], f"{context}.path"),
                _require_pattern(data["sha256"], f"{context}.sha256", _SHA256_HEX),
                _require_bool(data["matches_manifest"], f"{context}.matches_manifest"),
            )
        )
    paths = [path for path, _sha256, _matches in parsed]
    if len(set(paths)) != len(paths):
        raise SchemaValidationError("source_file_digests contains duplicate paths")
    required = required_source_file_paths()
    actual = set(paths)
    if actual != required:
        missing = sorted(required - actual)
        extra = sorted(actual - required)
        raise SchemaValidationError(
            "source_file_digests does not cover exactly the committed "
            f"vLLM source manifest: missing={missing}, extra={extra}"
        )

    expected_digests = required_source_file_digests()
    digests: list[SourceFileDigest] = []
    for index, (path, sha256, reported_matches) in enumerate(parsed):
        context = f"source_file_digests[{index}]"
        recomputed_matches = sha256 == expected_digests[path]
        if reported_matches != recomputed_matches:
            raise SchemaValidationError(
                f"{context}.matches_manifest={reported_matches!r} disagrees "
                "with the recomputed comparison of the reported sha256 "
                "against the committed manifest's real expected digest for "
                f"{path!r}"
            )
        digests.append(
            SourceFileDigest(
                path=path,
                sha256=sha256,
                matches_manifest=recomputed_matches,
            )
        )
    return tuple(digests)


@dataclass(frozen=True)
class IdentityReceipt:
    """An immutable, SHA-256-sealed identity receipt, exported *before* engine init.

    Every field here is knowable before ``LLM(...)``/``LLMEngine`` construction:
    image/runtime/GPU/model identity and the preregistered source-file
    manifest comparison. It deliberately contains no resolved cache layout,
    block counts, or KV-event capture boundaries -- those only exist once the
    engine has actually initialized, which is exactly why they live on the
    separate, later :class:`RuntimeAttestation` instead of here. This is the
    artifact the pinned runner is expected to emit immediately after
    acquiring/verifying the model and pulling/inspecting the image, strictly
    before constructing the engine.
    """

    schema_version: str
    protocol_id: str
    generated_at: str
    repository_commit: str
    image_repository_digest: str
    image_id: str
    vllm_version: str
    vllm_commit: str
    python_version: str
    torch_version: str
    cuda_runtime_version: str
    transformers_version: str
    typing_extensions_version: str
    cuda_driver_version: str
    gpu_name: str
    gpu_memory_mib: int
    gpu_compute_capability: str
    gpu_uuid_commitment: str
    experiment_nonce: str
    installed_distributions_digest: str
    wheel_record_digest: str
    package_tree_digest: str
    source_file_digests: tuple[SourceFileDigest, ...]
    model_id: str
    model_revision: str
    tokenizer_artifact_digest: str
    model_inventory_digest: str
    model_path_commitment: str
    runner_source_digest: str
    seal: str

    @property
    def source_manifest_verified(self) -> bool:
        return bool(self.source_file_digests) and all(
            item.matches_manifest for item in self.source_file_digests
        )

    def unsealed_payload(self) -> dict[str, Any]:
        payload = self.to_dict()
        del payload["seal"]
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_id": self.protocol_id,
            "generated_at": self.generated_at,
            "repository_commit": self.repository_commit,
            "image_repository_digest": self.image_repository_digest,
            "image_id": self.image_id,
            "vllm_version": self.vllm_version,
            "vllm_commit": self.vllm_commit,
            "python_version": self.python_version,
            "torch_version": self.torch_version,
            "cuda_runtime_version": self.cuda_runtime_version,
            "transformers_version": self.transformers_version,
            "typing_extensions_version": self.typing_extensions_version,
            "cuda_driver_version": self.cuda_driver_version,
            "gpu_name": self.gpu_name,
            "gpu_memory_mib": self.gpu_memory_mib,
            "gpu_compute_capability": self.gpu_compute_capability,
            "gpu_uuid_commitment": self.gpu_uuid_commitment,
            "experiment_nonce": self.experiment_nonce,
            "installed_distributions_digest": self.installed_distributions_digest,
            "wheel_record_digest": self.wheel_record_digest,
            "package_tree_digest": self.package_tree_digest,
            "source_file_digests": [
                item.to_dict() for item in self.source_file_digests
            ],
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "tokenizer_artifact_digest": self.tokenizer_artifact_digest,
            "model_inventory_digest": self.model_inventory_digest,
            "model_path_commitment": self.model_path_commitment,
            "runner_source_digest": self.runner_source_digest,
            "seal": self.seal,
        }

    def redact(self) -> dict[str, Any]:
        """Publication-safe view with host-unique GPU identity omitted."""

        redacted = self.to_dict()
        del redacted["gpu_uuid_commitment"]
        return redacted


_IDENTITY_RECEIPT_KEYS = set(IdentityReceipt.__dataclass_fields__)


class RuntimeAttestationError(SchemaValidationError):
    """Raised when an identity receipt or runtime attestation fails
    structural, seal, or coherence validation."""


def parse_identity_receipt(mapping: Mapping[str, Any]) -> IdentityReceipt:
    """Strictly parse and validate a pre-init identity receipt.

    Every field is required and exactly typed; unknown or missing top-level
    keys are refused. The recomputed SHA-256 seal over the canonical JSON of
    every other field must equal the supplied ``seal`` byte-for-byte, and the
    ``vllm_version``/``vllm_commit`` must equal the exact pinned release this
    adapter was validated against. Any mismatch raises
    :class:`RuntimeAttestationError`; there is no partial-success return.
    """

    data = _require_mapping(mapping, "identity_receipt")
    _exact_keys(data, _IDENTITY_RECEIPT_KEYS, "identity_receipt")

    receipt = IdentityReceipt(
        schema_version=_require_str(data["schema_version"], "schema_version"),
        protocol_id=_require_str(data["protocol_id"], "protocol_id"),
        generated_at=_require_str(data["generated_at"], "generated_at"),
        repository_commit=_require_pattern(
            data["repository_commit"], "repository_commit", _COMMIT_HEX
        ),
        image_repository_digest=_require_str(
            data["image_repository_digest"], "image_repository_digest"
        ),
        image_id=_require_pattern(data["image_id"], "image_id", _SHA256_DIGEST),
        vllm_version=_require_str(data["vllm_version"], "vllm_version"),
        vllm_commit=_require_pattern(data["vllm_commit"], "vllm_commit", _COMMIT_HEX),
        python_version=_require_str(data["python_version"], "python_version"),
        torch_version=_require_str(data["torch_version"], "torch_version"),
        cuda_runtime_version=_require_str(
            data["cuda_runtime_version"], "cuda_runtime_version"
        ),
        transformers_version=_require_str(
            data["transformers_version"], "transformers_version"
        ),
        typing_extensions_version=_require_str(
            data["typing_extensions_version"], "typing_extensions_version"
        ),
        cuda_driver_version=_require_str(
            data["cuda_driver_version"], "cuda_driver_version"
        ),
        gpu_name=_require_str(data["gpu_name"], "gpu_name"),
        gpu_memory_mib=_require_int(
            data["gpu_memory_mib"], "gpu_memory_mib", minimum=1
        ),
        gpu_compute_capability=_require_str(
            data["gpu_compute_capability"], "gpu_compute_capability"
        ),
        gpu_uuid_commitment=_require_pattern(
            data["gpu_uuid_commitment"], "gpu_uuid_commitment", _SHA256_HEX
        ),
        experiment_nonce=_require_str(data["experiment_nonce"], "experiment_nonce"),
        installed_distributions_digest=_require_pattern(
            data["installed_distributions_digest"],
            "installed_distributions_digest",
            _SHA256_DIGEST,
        ),
        wheel_record_digest=_require_pattern(
            data["wheel_record_digest"], "wheel_record_digest", _SHA256_DIGEST
        ),
        package_tree_digest=_require_pattern(
            data["package_tree_digest"], "package_tree_digest", _SHA256_DIGEST
        ),
        source_file_digests=_parse_source_file_digests(data["source_file_digests"]),
        model_id=_require_str(data["model_id"], "model_id"),
        model_revision=_require_pattern(
            data["model_revision"], "model_revision", _COMMIT_HEX
        ),
        tokenizer_artifact_digest=_require_pattern(
            data["tokenizer_artifact_digest"],
            "tokenizer_artifact_digest",
            _SHA256_DIGEST,
        ),
        model_inventory_digest=_require_pattern(
            data["model_inventory_digest"], "model_inventory_digest", _SHA256_DIGEST
        ),
        model_path_commitment=_require_pattern(
            data["model_path_commitment"], "model_path_commitment", _SHA256_HEX
        ),
        runner_source_digest=_require_pattern(
            data["runner_source_digest"], "runner_source_digest", _SHA256_DIGEST
        ),
        seal=_require_pattern(data["seal"], "seal", _SHA256_HEX),
    )

    expected_seal = hashlib.sha256(
        canonical_json(receipt.unsealed_payload()).encode("utf-8")
    ).hexdigest()
    if expected_seal != receipt.seal:
        raise RuntimeAttestationError(
            "identity_receipt seal does not match its canonical payload"
        )
    if receipt.vllm_version != REQUIRED_VLLM_VERSION:
        raise RuntimeAttestationError(
            f"identity_receipt.vllm_version must be {REQUIRED_VLLM_VERSION!r}"
        )
    if receipt.vllm_commit != REQUIRED_VLLM_COMMIT:
        raise RuntimeAttestationError(
            f"identity_receipt.vllm_commit must be {REQUIRED_VLLM_COMMIT!r}"
        )
    if not receipt.source_manifest_verified:
        raise RuntimeAttestationError(
            "identity_receipt.source_file_digests do not all match the "
            "preregistered manifest"
        )
    return receipt


@dataclass(frozen=True)
class RuntimeAttestation:
    """An immutable, post-init, SHA-256-sealed runtime attestation.

    Extends an already-sealed, already-validated :class:`IdentityReceipt`
    (bound by its own seal, not duplicated field-by-field) with the resolved
    cache layout and KV-event capture boundaries that only exist after the
    engine has actually initialized and the event subscriber has actually
    started. This is the only object the live adapter accepts as proof that
    a request's evidence is runtime-backed; there is no constructor path from
    caller strings, environment labels, or CLI booleans, and it cannot be
    built without an already-verified pre-init identity receipt.
    """

    identity: IdentityReceipt
    resolved_config: ResolvedVLLMConfig
    kv_events_config: KVEventsPublisherAttestation
    attested_at: str
    seal: str

    # Ergonomic passthroughs so identity fields read naturally off the
    # composed attestation (``attestation.model_id``, not
    # ``attestation.identity.model_id``) without duplicating them.
    @property
    def schema_version(self) -> str:
        return self.identity.schema_version

    @property
    def protocol_id(self) -> str:
        return self.identity.protocol_id

    @property
    def generated_at(self) -> str:
        return self.identity.generated_at

    @property
    def repository_commit(self) -> str:
        return self.identity.repository_commit

    @property
    def image_repository_digest(self) -> str:
        return self.identity.image_repository_digest

    @property
    def image_id(self) -> str:
        return self.identity.image_id

    @property
    def vllm_version(self) -> str:
        return self.identity.vllm_version

    @property
    def vllm_commit(self) -> str:
        return self.identity.vllm_commit

    @property
    def python_version(self) -> str:
        return self.identity.python_version

    @property
    def torch_version(self) -> str:
        return self.identity.torch_version

    @property
    def cuda_runtime_version(self) -> str:
        return self.identity.cuda_runtime_version

    @property
    def transformers_version(self) -> str:
        return self.identity.transformers_version

    @property
    def typing_extensions_version(self) -> str:
        return self.identity.typing_extensions_version

    @property
    def cuda_driver_version(self) -> str:
        return self.identity.cuda_driver_version

    @property
    def gpu_name(self) -> str:
        return self.identity.gpu_name

    @property
    def gpu_memory_mib(self) -> int:
        return self.identity.gpu_memory_mib

    @property
    def model_id(self) -> str:
        return self.identity.model_id

    @property
    def model_revision(self) -> str:
        return self.identity.model_revision

    @property
    def model_path_commitment(self) -> str:
        return self.identity.model_path_commitment

    @property
    def runner_source_digest(self) -> str:
        return self.identity.runner_source_digest

    @property
    def source_manifest_verified(self) -> bool:
        return self.identity.source_manifest_verified

    def unsealed_payload(self) -> dict[str, Any]:
        return {
            "identity_seal": self.identity.seal,
            "resolved_config": self.resolved_config.to_dict(),
            "kv_events_config": self.kv_events_config.to_dict(),
            "attested_at": self.attested_at,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity": self.identity.to_dict(),
            "resolved_config": self.resolved_config.to_dict(),
            "kv_events_config": self.kv_events_config.to_dict(),
            "attested_at": self.attested_at,
            "seal": self.seal,
        }

    def redact(self) -> dict[str, Any]:
        """Publication-safe view: drops the raw KV cache-salt value only.

        Every other field is already a digest, commitment, role label, or
        non-sensitive identity string, so no further redaction is required
        to satisfy the plan's public-redacted bundle constraints.
        """

        return {
            "identity": self.identity.redact(),
            "resolved_config": self.resolved_config.redact(),
            "kv_events_config": self.kv_events_config.to_dict(),
            "attested_at": self.attested_at,
            "seal": self.seal,
        }


_RUNTIME_ATTESTATION_KEYS = {
    "identity",
    "resolved_config",
    "kv_events_config",
    "attested_at",
    "seal",
}


def parse_runtime_attestation(mapping: Mapping[str, Any]) -> RuntimeAttestation:
    """Strictly parse and validate a post-init runtime attestation.

    ``mapping["identity"]`` must itself be a valid, sealed identity receipt
    (parsed with :func:`parse_identity_receipt`); ``resolved_config`` and
    ``kv_events_config`` must be the exact resolved values the engine and its
    KV-event publisher actually report. The recomputed outer SHA-256 seal
    binds to the identity receipt's own seal (not its raw fields again) plus
    the resolved config/kv-events config/``attested_at`` and must equal the
    supplied ``seal`` byte-for-byte. ``attested_at`` must not precede
    ``identity.generated_at``: an attestation cannot be sealed before the
    identity it extends. Any mismatch raises
    :class:`RuntimeAttestationError`; there is no partial-success return.
    """

    data = _require_mapping(mapping, "runtime_attestation")
    _exact_keys(data, _RUNTIME_ATTESTATION_KEYS, "runtime_attestation")

    identity = parse_identity_receipt(data["identity"])
    resolved_config = _parse_resolved_config(data["resolved_config"])
    kv_events_config = _parse_kv_events_config(data["kv_events_config"])
    attested_at = _require_str(data["attested_at"], "attested_at")
    seal = _require_pattern(data["seal"], "seal", _SHA256_HEX)

    if attested_at < identity.generated_at:
        raise RuntimeAttestationError(
            "runtime_attestation.attested_at precedes identity.generated_at; "
            "an attestation cannot be sealed before the identity it extends"
        )

    attestation = RuntimeAttestation(
        identity=identity,
        resolved_config=resolved_config,
        kv_events_config=kv_events_config,
        attested_at=attested_at,
        seal=seal,
    )

    expected_seal = hashlib.sha256(
        canonical_json(attestation.unsealed_payload()).encode("utf-8")
    ).hexdigest()
    if expected_seal != attestation.seal:
        raise RuntimeAttestationError(
            "runtime_attestation seal does not match its canonical payload"
        )
    if (
        attestation.resolved_config.prefix_caching_hash_algo
        != REQUIRED_PREFIX_CACHING_HASH_ALGO
    ):
        raise RuntimeAttestationError(
            "runtime_attestation.resolved_config.prefix_caching_hash_algo must be "
            f"{REQUIRED_PREFIX_CACHING_HASH_ALGO!r}"
        )
    if (
        attestation.resolved_config.kv_events_use_int_block_hashes
        != REQUIRED_KV_EVENTS_USE_INT_BLOCK_HASHES
    ):
        raise RuntimeAttestationError(
            "runtime_attestation.resolved_config.kv_events_use_int_block_hashes "
            f"must be {REQUIRED_KV_EVENTS_USE_INT_BLOCK_HASHES!r}"
        )
    if not attestation.resolved_config.enable_prefix_caching:
        raise RuntimeAttestationError(
            "runtime_attestation.resolved_config.enable_prefix_caching must be true"
        )
    if not attestation.resolved_config.enable_kv_cache_events:
        raise RuntimeAttestationError(
            "runtime_attestation.resolved_config.enable_kv_cache_events must be true"
        )
    if attestation.kv_events_config.endpoint_role != REQUIRED_ENDPOINT_ROLE:
        raise RuntimeAttestationError(
            "runtime_attestation.kv_events_config.endpoint_role must be "
            f"{REQUIRED_ENDPOINT_ROLE!r}"
        )
    if (
        attestation.kv_events_config.replay_endpoint_role
        != REQUIRED_REPLAY_ENDPOINT_ROLE
    ):
        raise RuntimeAttestationError(
            "runtime_attestation.kv_events_config.replay_endpoint_role must be "
            f"{REQUIRED_REPLAY_ENDPOINT_ROLE!r}"
        )
    return attestation


def assess_live_vllm_capabilities(mapping: Mapping[str, Any]) -> CacheAuditCapability:
    """Return a non-raising verdict for an attempted live runtime attestation.

    Mirrors ``assess_vllm_capabilities``'s never-raising, always-complete
    style, but for the live path: unlike the offline adapter, a fully valid,
    sealed, exact-commit attestation is reported ``supported=True`` here.
    Any parse/validation failure is caught and reported as a single reason
    code; it never propagates.
    """

    try:
        parse_runtime_attestation(mapping)
    except SchemaValidationError as exc:
        return CacheAuditCapability(
            backend=BACKEND,
            supported=False,
            reasons=(f"runtime_attestation_invalid: {exc}",),
        )
    return CacheAuditCapability(
        backend=BACKEND,
        supported=True,
        reasons=(),
        observable_facts=(
            "engine_cached_tokens",
            "engine_created_tokens",
            "kv_event_stream",
        ),
        unavailable_facts=(),
    )


# ---------------------------------------------------------------------------
# Live KV-cache event schema (vLLM 0.28.0 exact fields)
# ---------------------------------------------------------------------------


class LiveKVEventType(str, Enum):
    BLOCK_STORED = "BlockStored"
    BLOCK_REMOVED = "BlockRemoved"
    ALL_BLOCKS_CLEARED = "AllBlocksCleared"


def _optional_extra_keys(
    value: Any, context: str
) -> tuple[tuple[Any, ...] | None, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise SchemaValidationError(f"{context} must be an array or null")
    result: list[tuple[Any, ...] | None] = []
    for index, entry in enumerate(value):
        if entry is None:
            result.append(None)
            continue
        if not isinstance(entry, list):
            raise SchemaValidationError(f"{context}[{index}] must be an array or null")
        result.append(tuple(entry))
    return tuple(result)


def _cbor_head(major_type: int, value: int) -> bytes:
    if value < 0:
        raise SchemaValidationError("canonical CBOR length/value cannot be negative")
    prefix = major_type << 5
    if value < 24:
        return bytes((prefix | value,))
    if value < 2**8:
        return bytes((prefix | 24, value))
    if value < 2**16:
        return bytes((prefix | 25,)) + value.to_bytes(2, "big")
    if value < 2**32:
        return bytes((prefix | 26,)) + value.to_bytes(4, "big")
    if value < 2**64:
        return bytes((prefix | 27,)) + value.to_bytes(8, "big")
    raise SchemaValidationError("canonical CBOR integer exceeds 64 bits")


def _canonical_cbor(value: Any) -> bytes:
    """Encode the vLLM block-hash value domain using RFC 8949 canonical CBOR."""

    if value is None:
        return b"\xf6"
    if value is False:
        return b"\xf4"
    if value is True:
        return b"\xf5"
    if isinstance(value, int):
        return _cbor_head(0, value) if value >= 0 else _cbor_head(1, -1 - value)
    if isinstance(value, bytes):
        return _cbor_head(2, len(value)) + value
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        return _cbor_head(3, len(encoded)) + encoded
    if isinstance(value, (list, tuple)):
        return _cbor_head(4, len(value)) + b"".join(
            _canonical_cbor(item) for item in value
        )
    if isinstance(value, dict):
        encoded_items = [
            (_canonical_cbor(key), _canonical_cbor(item)) for key, item in value.items()
        ]
        encoded_items.sort(key=lambda item: (len(item[0]), item[0]))
        return _cbor_head(5, len(encoded_items)) + b"".join(
            key + item for key, item in encoded_items
        )
    raise SchemaValidationError(
        f"unsupported value in canonical CBOR block-hash input: {type(value).__name__}"
    )


def compute_sha256_cbor_block_hashes(
    *,
    token_ids: Sequence[int],
    block_size: int,
    parent_block_hash: str | None,
    extra_keys: Sequence[Sequence[Any] | None] | None,
    pythonhashseed: str = "0",
) -> tuple[str, ...]:
    """Independently reproduce vLLM 0.28.0's ``sha256_cbor`` hash chain."""

    if block_size <= 0 or len(token_ids) % block_size:
        raise SchemaValidationError(
            "block hash input must contain complete positive-sized blocks"
        )
    block_count = len(token_ids) // block_size
    if extra_keys is not None and len(extra_keys) != block_count:
        raise SchemaValidationError(
            "block hash extra_keys must have one entry per complete block"
        )
    parent = (
        bytes.fromhex(parent_block_hash)
        if parent_block_hash is not None
        else hashlib.sha256(_canonical_cbor(pythonhashseed)).digest()
    )
    hashes: list[str] = []
    for index in range(block_count):
        start = index * block_size
        block_tokens = tuple(token_ids[start : start + block_size])
        extra_entry = extra_keys[index] if extra_keys is not None else None
        block_extra_keys = tuple(extra_entry) if extra_entry is not None else None
        parent = hashlib.sha256(
            _canonical_cbor((parent, block_tokens, block_extra_keys))
        ).digest()
        hashes.append(parent.hex())
    return tuple(hashes)


def _normalize_external_hash(value: Any, context: str) -> str:
    if isinstance(value, bytes):
        if len(value) != hashlib.sha256().digest_size:
            raise SchemaValidationError(f"{context} must contain exactly 32 bytes")
        return value.hex()
    if not isinstance(value, str) or _SHA256_HEX.fullmatch(value) is None:
        raise SchemaValidationError(
            f"{context} must be a 256-bit SHA-256 lowercase hex string or 32-byte value"
        )
    return value


def _require_hash_tuple(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise SchemaValidationError(f"{context} must be a non-empty array")
    return tuple(
        _normalize_external_hash(item, f"{context}[{index}]")
        for index, item in enumerate(value)
    )


@dataclass(frozen=True)
class LiveBlockStored:
    """Exact vLLM 0.28.0 ``BlockStored`` fields, at pinned commit
    ``2cf0a6915ce544dc493a0990f2ea38d81601128a`` of ``kv_events.py``.
    """

    block_hashes: tuple[str, ...]
    parent_block_hash: str | None
    token_ids: tuple[int, ...]
    block_size: int
    lora_id: int | None
    medium: str | None
    lora_name: str | None
    extra_keys: tuple[tuple[Any, ...] | None, ...] | None
    group_idx: int | None
    kv_cache_spec_kind: str | None
    kv_cache_spec_sliding_window: int | None
    locality: str | None

    @property
    def event_type(self) -> LiveKVEventType:
        return LiveKVEventType.BLOCK_STORED

    def redact(self) -> dict[str, Any]:
        return {
            "type": self.event_type.value,
            "block_count": len(self.block_hashes),
            "token_count": len(self.token_ids),
            "block_size": self.block_size,
            "medium": self.medium,
            "group_idx": self.group_idx,
            "kv_cache_spec_kind": self.kv_cache_spec_kind,
            "kv_cache_spec_sliding_window": self.kv_cache_spec_sliding_window,
            "locality": self.locality,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.event_type.value,
            "block_hashes": list(self.block_hashes),
            "parent_block_hash": self.parent_block_hash,
            "token_ids": list(self.token_ids),
            "block_size": self.block_size,
            "lora_id": self.lora_id,
            "medium": self.medium,
            "lora_name": self.lora_name,
            "extra_keys": (
                [
                    list(value) if value is not None else None
                    for value in self.extra_keys
                ]
                if self.extra_keys is not None
                else None
            ),
            "group_idx": self.group_idx,
            "kv_cache_spec_kind": self.kv_cache_spec_kind,
            "kv_cache_spec_sliding_window": self.kv_cache_spec_sliding_window,
            "locality": self.locality,
        }


@dataclass(frozen=True)
class LiveBlockRemoved:
    """Exact vLLM 0.28.0 ``BlockRemoved`` fields, at pinned commit
    ``2cf0a6915ce544dc493a0990f2ea38d81601128a`` of ``kv_events.py``.
    """

    block_hashes: tuple[str, ...]
    medium: str | None
    group_idx: int | None
    locality: str | None

    @property
    def event_type(self) -> LiveKVEventType:
        return LiveKVEventType.BLOCK_REMOVED

    def redact(self) -> dict[str, Any]:
        return {
            "type": self.event_type.value,
            "block_count": len(self.block_hashes),
            "medium": self.medium,
            "group_idx": self.group_idx,
            "locality": self.locality,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.event_type.value,
            "block_hashes": list(self.block_hashes),
            "medium": self.medium,
            "group_idx": self.group_idx,
            "locality": self.locality,
        }


@dataclass(frozen=True)
class LiveAllBlocksCleared:
    """Exact vLLM 0.28.0 ``AllBlocksCleared`` marker (no fields)."""

    @property
    def event_type(self) -> LiveKVEventType:
        return LiveKVEventType.ALL_BLOCKS_CLEARED

    def redact(self) -> dict[str, Any]:
        return {"type": self.event_type.value}

    def to_dict(self) -> dict[str, Any]:
        return self.redact()


LiveKVEvent = LiveBlockStored | LiveBlockRemoved | LiveAllBlocksCleared

_BLOCK_STORED_KEYS = {
    "type",
    "block_hashes",
    "parent_block_hash",
    "token_ids",
    "block_size",
    "lora_id",
    "medium",
    "lora_name",
    "extra_keys",
    "group_idx",
    "kv_cache_spec_kind",
    "kv_cache_spec_sliding_window",
    "locality",
}
_BLOCK_REMOVED_KEYS = {
    "type",
    "block_hashes",
    "medium",
    "group_idx",
    "locality",
}
_ALL_BLOCKS_CLEARED_KEYS = {"type"}


def parse_live_kv_event(mapping: Mapping[str, Any]) -> LiveKVEvent:
    """Strictly parse one already msgpack-decoded vLLM 0.28.0 KV-cache event.

    Rejects any mapping whose keys differ from the exact declared type's
    field set, and rejects malformed field values instead of coercing them.
    """

    data = _require_mapping(mapping, "live_kv_event")
    if "type" not in data:
        raise SchemaValidationError("live_kv_event.type is required")
    try:
        event_type = LiveKVEventType(data["type"])
    except ValueError as exc:
        raise SchemaValidationError(
            f"live_kv_event.type is invalid: {data['type']!r}"
        ) from exc

    if event_type is LiveKVEventType.BLOCK_STORED:
        _exact_keys(data, _BLOCK_STORED_KEYS, "BlockStored event")
        block_hashes = _require_hash_tuple(
            data["block_hashes"], "live_kv_event.block_hashes"
        )
        parent_value = data["parent_block_hash"]
        parent_block_hash = (
            None
            if parent_value is None
            else _normalize_external_hash(
                parent_value, "live_kv_event.parent_block_hash"
            )
        )
        block_size = _require_int(
            data["block_size"], "live_kv_event.block_size", minimum=1
        )
        token_ids = _require_int_tuple(data["token_ids"], "live_kv_event.token_ids")
        if len(token_ids) != len(block_hashes) * block_size:
            raise SchemaValidationError(
                "BlockStored token_ids must contain one complete block per block hash"
            )
        extra_keys = _optional_extra_keys(
            data["extra_keys"], "live_kv_event.extra_keys"
        )
        if extra_keys is not None and len(extra_keys) != len(block_hashes):
            raise SchemaValidationError(
                "BlockStored extra_keys must have one entry per block hash"
            )
        recomputed_hashes = compute_sha256_cbor_block_hashes(
            token_ids=token_ids,
            block_size=block_size,
            parent_block_hash=parent_block_hash,
            extra_keys=extra_keys,
        )
        if block_hashes != recomputed_hashes:
            raise SchemaValidationError(
                "BlockStored hash chain does not match canonical sha256_cbor over "
                "parent hash, token IDs, and extra keys"
            )
        return LiveBlockStored(
            block_hashes=block_hashes,
            parent_block_hash=parent_block_hash,
            token_ids=token_ids,
            block_size=block_size,
            lora_id=_optional_int(
                data["lora_id"], "live_kv_event.lora_id", minimum=None
            ),
            medium=_optional_str(data["medium"], "live_kv_event.medium"),
            lora_name=_optional_str(data["lora_name"], "live_kv_event.lora_name"),
            extra_keys=extra_keys,
            group_idx=_optional_int(data["group_idx"], "live_kv_event.group_idx"),
            kv_cache_spec_kind=_optional_str(
                data["kv_cache_spec_kind"], "live_kv_event.kv_cache_spec_kind"
            ),
            kv_cache_spec_sliding_window=_optional_int(
                data["kv_cache_spec_sliding_window"],
                "live_kv_event.kv_cache_spec_sliding_window",
                minimum=1,
            ),
            locality=_optional_str(data["locality"], "live_kv_event.locality"),
        )

    if event_type is LiveKVEventType.BLOCK_REMOVED:
        _exact_keys(data, _BLOCK_REMOVED_KEYS, "BlockRemoved event")
        block_hashes = _require_hash_tuple(
            data["block_hashes"], "live_kv_event.block_hashes"
        )
        return LiveBlockRemoved(
            block_hashes=block_hashes,
            medium=_optional_str(data["medium"], "live_kv_event.medium"),
            group_idx=_optional_int(data["group_idx"], "live_kv_event.group_idx"),
            locality=_optional_str(data["locality"], "live_kv_event.locality"),
        )

    _exact_keys(data, _ALL_BLOCKS_CLEARED_KEYS, "AllBlocksCleared event")
    return LiveAllBlocksCleared()


EVENT_ORDINAL_BITS = 20
EVENT_ORDINAL_MASK = (1 << EVENT_ORDINAL_BITS) - 1


def composite_sequence(batch_sequence: int, event_ordinal: int) -> int:
    """Combine a batch's ZMQ sequence with its event's in-batch ordinal.

    ``CacheEventRecord.sequence`` is one integer per *event*, but one
    published ``KVEventBatch`` can carry several events under a single ZMQ
    sequence number. Reusing the raw batch sequence for every event in that
    batch would silently alias distinct events onto the same recorded
    sequence -- an unsafe collision, not a formatting detail. Shifting the
    batch sequence left by :data:`EVENT_ORDINAL_BITS` and OR-ing in the
    event's zero-based position within the batch keeps every record's
    sequence unique and strictly increasing with both batch delivery order
    and in-batch event order, while remaining trivially decodable back to
    ``(batch_sequence, event_ordinal)`` via
    :func:`decompose_composite_sequence`. 20 bits allows up to 1,048,576
    events per batch, far beyond any batch this protocol can produce.
    """

    if batch_sequence < 0:
        raise SchemaValidationError("batch_sequence must be non-negative")
    if not 0 <= event_ordinal <= EVENT_ORDINAL_MASK:
        raise SchemaValidationError(
            f"event_ordinal must be in [0, {EVENT_ORDINAL_MASK}]"
        )
    return (batch_sequence << EVENT_ORDINAL_BITS) | event_ordinal


def decompose_composite_sequence(value: int) -> tuple[int, int]:
    """Invert :func:`composite_sequence`, returning ``(batch_sequence, event_ordinal)``."""

    if value < 0:
        raise SchemaValidationError("composite sequence must be non-negative")
    return value >> EVENT_ORDINAL_BITS, value & EVENT_ORDINAL_MASK


@dataclass(frozen=True)
class LiveKVEventBatch:
    """One normalized ``KVEventBatch`` plus its ZMQ multipart sequence frame.

    ``sequence`` and ``topic`` come from the two multipart frames
    ``ZmqEventPublisher`` sends alongside the msgpack payload (``topic_bytes``,
    an 8-byte big-endian ``seq_bytes``, then the payload); ``ts`` and
    ``data_parallel_rank`` come from the decoded, array-like ``EventBatch``
    payload (see :func:`parse_live_kv_event_batch`). ``data_parallel_rank``
    is always a concrete ``int`` here (never ``None``): a real batch either
    reports it explicitly or omits it because it equals
    :data:`DEFAULT_DATA_PARALLEL_RANK`, and this class always normalizes to
    the latter in that case.
    """

    sequence: int
    topic: str
    ts: float
    events: tuple[LiveKVEvent, ...]
    data_parallel_rank: int

    def to_cache_event_records(self) -> tuple[CacheEventRecord, ...]:
        """Project onto the generic, privacy-safe :class:`CacheEventRecord`.

        This is the only bridge between the live schema above and the
        existing backend-agnostic evidence schema
        (``llmtracefx.cache_audit.schema``): raw hashes and token IDs never
        cross it, only counts and structural fields, matching
        :class:`CacheEventRecord`'s own privacy contract. Each event's
        recorded ``sequence`` is :func:`composite_sequence` of this batch's
        ZMQ sequence and the event's position within the batch, not the raw
        batch sequence, so multiple events in one batch never alias onto the
        same recorded sequence.
        """

        records: list[CacheEventRecord] = []
        for ordinal, event in enumerate(self.events):
            sequence = composite_sequence(self.sequence, ordinal)
            if isinstance(event, LiveBlockStored):
                records.append(
                    CacheEventRecord(
                        sequence=sequence,
                        event_type=event.event_type.value,
                        basis=EvidenceBasis.ENGINE_ATTESTED,
                        token_count=len(event.token_ids),
                        block_count=len(event.block_hashes),
                        medium=event.medium,
                        group_index=event.group_idx,
                    )
                )
            elif isinstance(event, LiveBlockRemoved):
                records.append(
                    CacheEventRecord(
                        sequence=sequence,
                        event_type=event.event_type.value,
                        basis=EvidenceBasis.ENGINE_ATTESTED,
                        token_count=None,
                        block_count=len(event.block_hashes),
                        medium=event.medium,
                        group_index=event.group_idx,
                    )
                )
            else:
                records.append(
                    CacheEventRecord(
                        sequence=sequence,
                        event_type=event.event_type.value,
                        basis=EvidenceBasis.ENGINE_ATTESTED,
                    )
                )
        return tuple(records)

    def redact(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "topic": self.topic,
            "ts": self.ts,
            "data_parallel_rank": self.data_parallel_rank,
            "events": [event.redact() for event in self.events],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "topic": self.topic,
            "ts": self.ts,
            "data_parallel_rank": self.data_parallel_rank,
            "events": [event.to_dict() for event in self.events],
        }


def decode_sequence_frame(seq_bytes: bytes) -> int:
    """Decode the 8-byte big-endian ZMQ sequence frame vLLM publishes.

    Accepts the reserved ``END_OF_REPLAY_SEQUENCE`` sentinel (signed ``-1``)
    used by the replay ROUTER socket to mark end-of-buffer, and otherwise
    requires a non-negative sequence number.
    """

    if (
        not isinstance(seq_bytes, (bytes, bytearray))
        or len(seq_bytes) != SEQUENCE_FRAME_BYTES
    ):
        raise SchemaValidationError(
            f"sequence frame must be exactly {SEQUENCE_FRAME_BYTES} bytes"
        )
    value = int.from_bytes(bytes(seq_bytes), "big", signed=True)
    if value != END_OF_REPLAY_SEQUENCE and value < 0:
        raise SchemaValidationError(
            "sequence frame must be non-negative or END_OF_REPLAY_SEQUENCE"
        )
    return value


def parse_live_kv_event_batch(
    topic_bytes: bytes,
    seq_bytes: bytes,
    payload: Sequence[Any],
) -> LiveKVEventBatch:
    """Normalize one ZMQ multipart ``(topic, seq, payload)`` frame triple.

    ``payload`` is the already msgpack-decoded ``EventBatch`` value. vLLM
    0.28.0 declares ``EventBatch`` as ``msgspec.Struct(array_like=True)``,
    so a real decode (via ``msgspec.msgpack.decode`` with no target type,
    or any other conformant msgpack decoder) produces a plain *array*,
    never a mapping: ``[ts, events]`` when ``data_parallel_rank`` equals its
    default (:data:`DEFAULT_DATA_PARALLEL_RANK`, omitted by
    ``array_like=True`` encoding), or ``[ts, events, data_parallel_rank]``
    when it does not. A mapping payload is rejected -- it is not the real
    wire shape -- rather than accepted as an alternate encoding. This
    function never decodes raw msgpack bytes itself since
    ``msgspec``/``msgpack`` are not available offline. The sequence is
    taken strictly from ``seq_bytes``, matching
    ``ZmqEventPublisher._publisher_thread``'s wire framing, not from any
    field inside the payload.
    """

    if not isinstance(topic_bytes, (bytes, bytearray)):
        raise SchemaValidationError("topic frame must be bytes")
    sequence = decode_sequence_frame(seq_bytes)
    if sequence == END_OF_REPLAY_SEQUENCE:
        raise SchemaValidationError(
            "END_OF_REPLAY_SEQUENCE is a replay marker, not a decodable event batch"
        )
    if isinstance(payload, (str, bytes, bytearray, Mapping)) or not isinstance(
        payload, Sequence
    ):
        raise SchemaValidationError(
            "live_kv_event_batch payload must be the array-like EventBatch "
            "encoding ([ts, events] or [ts, events, data_parallel_rank]), "
            f"not {type(payload).__name__}"
        )
    elements = list(payload)
    if len(elements) not in (2, 3):
        raise SchemaValidationError(
            "live_kv_event_batch payload must have 2 or 3 elements (ts, "
            f"events[, data_parallel_rank]), got {len(elements)}"
        )
    ts = _require_finite_float(elements[0], "live_kv_event_batch[0] (ts)")
    events_raw = elements[1]
    if not isinstance(events_raw, list):
        raise SchemaValidationError("live_kv_event_batch[1] (events) must be an array")
    events = tuple(parse_live_kv_event(item) for item in events_raw)
    if len(elements) == 3:
        data_parallel_rank = _require_int(
            elements[2], "live_kv_event_batch[2] (data_parallel_rank)"
        )
    else:
        data_parallel_rank = DEFAULT_DATA_PARALLEL_RANK
    return LiveKVEventBatch(
        sequence=sequence,
        topic=bytes(topic_bytes).decode("utf-8"),
        ts=ts,
        events=events,
        data_parallel_rank=data_parallel_rank,
    )


@dataclass(frozen=True)
class LiveKVEventStreamReport:
    """Sequence-integrity findings across a captured batch stream.

    Mirrors ``KVEventStreamReport``'s fail-closed style at batch granularity:
    ``eligible`` is always ``False`` because no automated parse can attest to
    the request-boundary binding the plan requires; ``structurally_eligible``
    reports whether the stream itself is otherwise gap-free, duplicate-free,
    and topic/DP-rank consistent.
    """

    batches: tuple[LiveKVEventBatch, ...]
    sequence_gaps: tuple[tuple[int, int], ...]
    duplicate_sequences: tuple[int, ...]
    capture_start_sequence: int | None
    capture_end_sequence: int | None
    topic_consistent: bool
    data_parallel_rank_consistent: bool

    @property
    def has_gaps(self) -> bool:
        return bool(self.sequence_gaps)

    @property
    def has_duplicate_sequences(self) -> bool:
        return bool(self.duplicate_sequences)

    @property
    def ineligibility_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.batches:
            reasons.append("kv_event_batches_missing")
        if self.capture_start_sequence is None:
            reasons.append("kv_event_capture_start_missing")
        if self.capture_end_sequence is None:
            reasons.append("kv_event_capture_end_missing")
        if self.batches and self.capture_start_sequence is not None:
            if self.batches[0].sequence != self.capture_start_sequence:
                reasons.append("kv_event_capture_start_mismatch")
        if self.batches and self.capture_end_sequence is not None:
            if self.batches[-1].sequence != self.capture_end_sequence:
                reasons.append("kv_event_capture_end_mismatch")
        if self.has_gaps:
            reasons.append("kv_event_sequence_gaps")
        if self.has_duplicate_sequences:
            reasons.append("kv_event_duplicate_sequences")
        if not self.topic_consistent:
            reasons.append("kv_event_topic_inconsistent")
        if not self.data_parallel_rank_consistent:
            reasons.append("kv_event_data_parallel_rank_inconsistent")
        reasons.append("runtime_event_attestation_request_binding_unavailable")
        return tuple(reasons)

    @property
    def eligible(self) -> bool:
        return False

    @property
    def structurally_eligible(self) -> bool:
        return self.ineligibility_reasons == (
            "runtime_event_attestation_request_binding_unavailable",
        )


def parse_live_kv_event_stream(
    batches: Sequence[LiveKVEventBatch],
    *,
    capture_start_sequence: int | None = None,
    capture_end_sequence: int | None = None,
) -> LiveKVEventStreamReport:
    """Parse an ordered batch sequence and flag sequence/topic/DP-rank defects."""

    ordered = tuple(batches)
    counts = Counter(batch.sequence for batch in ordered)
    duplicate_sequences = tuple(
        sorted(sequence for sequence, count in counts.items() if count > 1)
    )
    unique_sorted = sorted(counts)
    sequence_gaps = tuple(
        (unique_sorted[index] + 1, unique_sorted[index + 1] - 1)
        for index in range(len(unique_sorted) - 1)
        if unique_sorted[index + 1] - unique_sorted[index] > 1
    )
    topics = {batch.topic for batch in ordered}
    dp_ranks = {batch.data_parallel_rank for batch in ordered}
    return LiveKVEventStreamReport(
        batches=ordered,
        sequence_gaps=sequence_gaps,
        duplicate_sequences=duplicate_sequences,
        capture_start_sequence=capture_start_sequence,
        capture_end_sequence=capture_end_sequence,
        topic_consistent=len(topics) <= 1,
        data_parallel_rank_consistent=len(dp_ranks) <= 1,
    )
