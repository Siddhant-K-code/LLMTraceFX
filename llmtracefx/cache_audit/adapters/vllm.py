"""Read-only, offline vLLM 0.28.0 capability and KV-cache event adapter.

This module never imports the ``vllm`` package, never executes GPU work, and
never accesses the network. It only reasons about a caller-supplied
description of a vLLM deployment's runtime identity and KV-event
configuration (:class:`VLLMCapabilityConfig`), and it parses normalized
synthetic ``BlockStored``/``BlockRemoved``/``AllBlocksCleared`` event mappings
and synthetic/offline-captured ``RequestOutput`` observations that were
recorded elsewhere (for example by a separate, gated live-GPU runner). Actual
vLLM request execution is intentionally out of scope here;
:func:`assess_vllm_capabilities` only says whether a described runtime is
trustworthy enough for offline KV-cache accounting.
"""

from __future__ import annotations

import importlib.metadata
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from llmtracefx.optimizer.schema import SchemaValidationError

from ..expected import ReuseExpectation, VLLMReuseConfig, expected_vllm_reuse
from .base import CacheAuditCapability

BACKEND = "vllm"

# The exact vLLM release this adapter was validated against. Both the version
# string and the immutable release commit must match: a version match alone
# does not prove which commit built the wheel actually running.
REQUIRED_VLLM_VERSION = "0.28.0"
REQUIRED_VLLM_COMMIT = "2cf0a6915ce544dc493a0990f2ea38d81601128a"

# vLLM's ``sha256_cbor`` prefix-caching hash algorithm serializes block-hash
# inputs with canonical CBOR before hashing, giving a reproducible,
# cross-language hash. The default ``sha256`` algorithm instead pickles
# inputs, which is not guaranteed reproducible across Python/vLLM versions.
REQUIRED_PREFIX_CACHING_HASH_ALGO = "sha256_cbor"

# vLLM truncates block hashes to 64-bit integers for KV events unless this is
# disabled, which risks hash collisions in the audit trail. The auditor
# requires raw hash bytes instead.
REQUIRED_KV_EVENTS_USE_INT_BLOCK_HASHES = "0"

_ENV_VLLM_COMMIT = "LLMTRACEFX_VLLM_COMMIT"
_ENV_PREFIX_CACHING_HASH_ALGO = "LLMTRACEFX_VLLM_PREFIX_CACHING_HASH_ALGO"
_ENV_ENABLE_PREFIX_CACHING = "LLMTRACEFX_VLLM_ENABLE_PREFIX_CACHING"
_ENV_ENABLE_KV_CACHE_EVENTS = "LLMTRACEFX_VLLM_ENABLE_KV_CACHE_EVENTS"
_ENV_HASH_BLOCK_SIZE = "LLMTRACEFX_VLLM_HASH_BLOCK_SIZE"
_ENV_PHYSICAL_BLOCK_SIZES = "LLMTRACEFX_VLLM_PHYSICAL_BLOCK_SIZES"
_ENV_FINE_GRAINED_HITS = "LLMTRACEFX_VLLM_FINE_GRAINED_HITS"
_ENV_KV_EVENTS_USE_INT_BLOCK_HASHES = "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES"
_ENV_PYTHONHASHSEED = "PYTHONHASHSEED"

_OBSERVABLE_FACTS = (
    "engine_cached_blocks",
    "reusable_blocks",
    "partial_block_tokens",
    "eviction_observed",
)
_UNAVAILABLE_FACTS = (
    "client_ttft",
    "runtime_active_bytes",
    "runtime_peak_bytes",
    "output_token_ids",
    "correctness",
)
_SUPPORTED_REASONS = ("read_only_offline_configuration_verified",)


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_bool(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _parse_int(value: str | None) -> int | None:
    cleaned = _clean(value)
    if cleaned is None:
        return None
    try:
        return int(cleaned)
    except ValueError:
        return None


def _parse_int_tuple(value: str | None) -> tuple[int, ...]:
    cleaned = _clean(value)
    if cleaned is None:
        return ()
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    try:
        return tuple(int(part) for part in parts)
    except ValueError:
        return ()


def _installed_vllm_version() -> str | None:
    """Read the installed ``vllm`` distribution's version without importing it."""

    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        return None


def _pythonhashseed_is_fixed(value: str | None) -> bool:
    """Return whether ``PYTHONHASHSEED`` pins a deterministic hash seed.

    An unset value or the literal ``"random"`` (case-insensitive) leaves hash
    randomization on, which makes any hashing that depends on Python's
    built-in ``hash()`` non-reproducible across process restarts.
    """

    if value is None:
        return False
    stripped = value.strip()
    if not stripped or stripped.lower() == "random":
        return False
    return stripped.isdigit()


@dataclass(frozen=True)
class VLLMCapabilityConfig:
    """A caller-supplied snapshot of one vLLM deployment's audit-relevant identity."""

    version: str | None = None
    commit: str | None = None
    prefix_caching_hash_algo: str | None = None
    pythonhashseed: str | None = None
    enable_prefix_caching: bool = False
    enable_kv_cache_events: bool = False
    kv_events_use_int_block_hashes: str | None = None
    hash_block_size: int | None = None
    physical_block_sizes: tuple[int, ...] = ()
    fine_grained_hits: bool = False

    @classmethod
    def from_environment(
        cls, *, environ: Mapping[str, str] | None = None
    ) -> VLLMCapabilityConfig:
        """Build a best-effort, non-raising snapshot from the process environment.

        This never imports ``vllm``: the installed version is read from
        package metadata, and every other field is read from environment
        variables so it works whether or not ``vllm`` itself is installed.
        """

        env = os.environ if environ is None else environ
        return cls(
            version=_installed_vllm_version(),
            commit=_clean(env.get(_ENV_VLLM_COMMIT)),
            prefix_caching_hash_algo=_clean(env.get(_ENV_PREFIX_CACHING_HASH_ALGO)),
            pythonhashseed=env.get(_ENV_PYTHONHASHSEED),
            enable_prefix_caching=_parse_bool(env.get(_ENV_ENABLE_PREFIX_CACHING)),
            enable_kv_cache_events=_parse_bool(env.get(_ENV_ENABLE_KV_CACHE_EVENTS)),
            kv_events_use_int_block_hashes=_clean(
                env.get(_ENV_KV_EVENTS_USE_INT_BLOCK_HASHES)
            ),
            hash_block_size=_parse_int(env.get(_ENV_HASH_BLOCK_SIZE)),
            physical_block_sizes=_parse_int_tuple(env.get(_ENV_PHYSICAL_BLOCK_SIZES)),
            fine_grained_hits=_parse_bool(env.get(_ENV_FINE_GRAINED_HITS)),
        )


def assess_vllm_capabilities(config: VLLMCapabilityConfig) -> CacheAuditCapability:
    """Validate a described vLLM deployment and return an ordered verdict.

    Every applicable check runs and appends at most one reason code, in a
    fixed order, so ``reasons`` is a deterministic, complete list of every
    way the configuration falls short rather than the first failure only.
    """

    reasons: list[str] = []

    if config.version != REQUIRED_VLLM_VERSION:
        reasons.append(
            "vllm_not_installed" if config.version is None else "vllm_version_mismatch"
        )
    if config.commit != REQUIRED_VLLM_COMMIT:
        reasons.append(
            "vllm_commit_unavailable"
            if config.commit is None
            else "vllm_commit_mismatch"
        )
    if config.prefix_caching_hash_algo != REQUIRED_PREFIX_CACHING_HASH_ALGO:
        reasons.append("prefix_caching_hash_algo_not_sha256_cbor")
    if not _pythonhashseed_is_fixed(config.pythonhashseed):
        reasons.append("pythonhashseed_not_fixed")
    if not config.enable_prefix_caching:
        reasons.append("prefix_caching_disabled")
    if not config.enable_kv_cache_events:
        reasons.append("kv_cache_events_disabled")
    if config.kv_events_use_int_block_hashes != REQUIRED_KV_EVENTS_USE_INT_BLOCK_HASHES:
        reasons.append("kv_events_int_block_hashes_enabled")
    if config.hash_block_size is None or config.hash_block_size <= 0:
        reasons.append("hash_block_size_not_positive")
    if not config.physical_block_sizes:
        reasons.append("physical_block_sizes_missing")
    elif any(size <= 0 for size in config.physical_block_sizes):
        reasons.append("physical_block_size_not_positive")
    elif config.hash_block_size and any(
        size % config.hash_block_size for size in config.physical_block_sizes
    ):
        reasons.append("physical_block_size_misaligned")

    return CacheAuditCapability(
        backend=BACKEND,
        supported=not reasons,
        reasons=tuple(reasons) if reasons else _SUPPORTED_REASONS,
        observable_facts=_OBSERVABLE_FACTS,
        unavailable_facts=_UNAVAILABLE_FACTS,
    )


def to_reuse_config(config: VLLMCapabilityConfig) -> VLLMReuseConfig:
    """Project a compliant :class:`VLLMCapabilityConfig` onto ``VLLMReuseConfig``."""

    if config.hash_block_size is None or config.hash_block_size < 1:
        raise ValueError(
            "hash_block_size must be a positive integer to build a VLLMReuseConfig"
        )
    return VLLMReuseConfig(
        hash_block_size=config.hash_block_size,
        physical_block_sizes=config.physical_block_sizes,
        fine_grained_hits=config.fine_grained_hits,
    )


def derive_reuse_expectation(
    config: VLLMCapabilityConfig,
    cached_tokens: Sequence[int],
    request_tokens: Sequence[int],
    *,
    identity_matches: bool = True,
) -> ReuseExpectation:
    """Derive an independent :class:`ReuseExpectation` for a compliant config.

    Refuses (raising ``ValueError``) unless :func:`assess_vllm_capabilities`
    reports the configuration as supported, so a caller cannot accidentally
    derive an expectation from a runtime this adapter would otherwise refuse.
    """

    capability = assess_vllm_capabilities(config)
    if not capability.supported:
        raise ValueError(
            "cannot derive a vLLM reuse expectation from a refused configuration: "
            + ", ".join(capability.reasons)
        )
    return expected_vllm_reuse(
        cached_tokens,
        request_tokens,
        to_reuse_config(config),
        identity_matches=identity_matches,
    )


class KVEventType(str, Enum):
    """The three normalized vLLM KV-cache event kinds this adapter parses."""

    BLOCK_STORED = "BlockStored"
    BLOCK_REMOVED = "BlockRemoved"
    ALL_BLOCKS_CLEARED = "AllBlocksCleared"


@dataclass(frozen=True)
class SyntheticKVEvent:
    """One normalized, synthetic vLLM KV-cache event.

    ``token_ids`` is preserved privately on the instance (needed to derive
    reuse expectations offline) but is never included in :meth:`redact`.
    """

    sequence: int
    event_type: KVEventType
    block_hashes: tuple[str, ...] = ()
    parent_block_hash: str | None = None
    token_ids: tuple[int, ...] = ()
    block_size: int | None = None
    medium: str | None = None
    lora_name: str | None = None
    group_idx: int | None = None
    extra_keys: tuple[str, ...] = ()
    cache_salt: str | None = None

    def redact(self) -> dict[str, Any]:
        """Return a publication-safe view of this event.

        Removes raw block/parent hashes, token IDs, extra keys, and cache
        salts, as well as ``lora_name`` (cache identity metadata), keeping
        only structural fields that are safe to publish.
        """

        return {
            "sequence": self.sequence,
            "event_type": self.event_type.value,
            "block_size": self.block_size,
            "medium": self.medium,
            "group_idx": self.group_idx,
        }


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SchemaValidationError(f"{context} must be an object")
    return dict(value)


def _exact_keys(data: dict[str, Any], keys: set[str], context: str) -> None:
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


def _optional_finite_number(value: Any, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{context} must be a finite number or null")
    number = float(value)
    if not math.isfinite(number):
        raise SchemaValidationError(f"{context} must be finite")
    return number


def _require_int(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SchemaValidationError(
            f"{context} must be an integer greater than or equal to {minimum}"
        )
    return value


def _optional_int(value: Any, context: str) -> int | None:
    if value is None:
        return None
    return _require_int(value, context)


def _require_str(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise SchemaValidationError(f"{context} must be a non-empty string")
    return value


def _optional_str(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, context)


def _require_str_list(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SchemaValidationError(f"{context} must be an array")
    return tuple(
        _require_str(item, f"{context}[{index}]") for index, item in enumerate(value)
    )


def _optional_str_list(value: Any, context: str) -> tuple[str, ...]:
    if value is None:
        return ()
    return _require_str_list(value, context)


def _require_int_list(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise SchemaValidationError(f"{context} must be an array")
    return tuple(
        _require_int(item, f"{context}[{index}]") for index, item in enumerate(value)
    )


_BLOCK_STORED_KEYS = {
    "sequence",
    "type",
    "block_hashes",
    "parent_block_hash",
    "token_ids",
    "block_size",
    "medium",
    "lora_name",
    "group_idx",
    "extra_keys",
    "cache_salt",
}
_BLOCK_REMOVED_KEYS = {"sequence", "type", "block_hashes", "medium", "group_idx"}
_ALL_BLOCKS_CLEARED_KEYS = {"sequence", "type"}


def parse_kv_event(mapping: Mapping[str, Any]) -> SyntheticKVEvent:
    """Strictly parse one normalized synthetic KV-cache event mapping.

    Rejects mappings with missing or unexpected keys for the declared
    ``type``, and rejects malformed field values, rather than silently
    ignoring or coercing them.
    """

    data = _require_mapping(mapping, "kv_event")
    if "type" not in data:
        raise SchemaValidationError("kv_event.type is required")
    try:
        event_type = KVEventType(data["type"])
    except ValueError as exc:
        raise SchemaValidationError(
            f"kv_event.type is invalid: {data['type']!r}"
        ) from exc

    if event_type is KVEventType.BLOCK_STORED:
        _exact_keys(data, _BLOCK_STORED_KEYS, "BlockStored event")
        event = SyntheticKVEvent(
            sequence=_require_int(data["sequence"], "kv_event.sequence"),
            event_type=event_type,
            block_hashes=_require_str_list(
                data["block_hashes"], "kv_event.block_hashes"
            ),
            parent_block_hash=_optional_str(
                data["parent_block_hash"], "kv_event.parent_block_hash"
            ),
            token_ids=_require_int_list(data["token_ids"], "kv_event.token_ids"),
            block_size=_require_int(
                data["block_size"], "kv_event.block_size", minimum=1
            ),
            medium=_optional_str(data["medium"], "kv_event.medium"),
            lora_name=_optional_str(data["lora_name"], "kv_event.lora_name"),
            group_idx=_optional_int(data["group_idx"], "kv_event.group_idx"),
            extra_keys=_optional_str_list(data["extra_keys"], "kv_event.extra_keys"),
            cache_salt=_optional_str(data["cache_salt"], "kv_event.cache_salt"),
        )
        assert event.block_size is not None
        if len(event.token_ids) != len(event.block_hashes) * event.block_size:
            raise SchemaValidationError(
                "BlockStored token_ids must contain one complete block per block hash"
            )
        return event

    if event_type is KVEventType.BLOCK_REMOVED:
        _exact_keys(data, _BLOCK_REMOVED_KEYS, "BlockRemoved event")
        return SyntheticKVEvent(
            sequence=_require_int(data["sequence"], "kv_event.sequence"),
            event_type=event_type,
            block_hashes=_require_str_list(
                data["block_hashes"], "kv_event.block_hashes"
            ),
            medium=_optional_str(data["medium"], "kv_event.medium"),
            group_idx=_optional_int(data["group_idx"], "kv_event.group_idx"),
        )

    _exact_keys(data, _ALL_BLOCKS_CLEARED_KEYS, "AllBlocksCleared event")
    return SyntheticKVEvent(
        sequence=_require_int(data["sequence"], "kv_event.sequence"),
        event_type=event_type,
    )


@dataclass(frozen=True)
class KVEventStreamReport:
    """Parsed events plus sequence-integrity findings for one event stream.

    ``ambiguous_block_hashes`` flags a raw-hash collision: a block hash
    observed across ``BlockStored`` events with more than one distinct
    ``token_ids`` tuple, meaning the same cache key was reused for different
    content. The raw hash itself is intentionally discarded — each element
    is only the set of distinct ``token_ids`` tuples that collided under it
    — so callers must use :attr:`ambiguous_block_hash_count` or
    :attr:`has_ambiguous_block_hashes` to report ambiguity publicly without
    ever surfacing the hash.
    """

    events: tuple[SyntheticKVEvent, ...]
    sequence_gaps: tuple[tuple[int, int], ...]
    duplicate_sequences: tuple[int, ...]
    ambiguous_block_hashes: tuple[tuple[tuple[int, ...], ...], ...] = ()

    @property
    def has_gaps(self) -> bool:
        return bool(self.sequence_gaps)

    @property
    def has_duplicate_sequences(self) -> bool:
        return bool(self.duplicate_sequences)

    @property
    def ambiguous_block_hash_count(self) -> int:
        """The number of distinct raw block hashes found to be ambiguous."""

        return len(self.ambiguous_block_hashes)

    @property
    def has_ambiguous_block_hashes(self) -> bool:
        return bool(self.ambiguous_block_hashes)


def parse_kv_event_stream(
    mappings: Sequence[Mapping[str, Any]],
) -> KVEventStreamReport:
    """Parse an ordered batch of event mappings and flag sequence defects.

    ``sequence_gaps`` lists inclusive ``(first_missing, last_missing)``
    ranges between the lowest and highest observed sequence numbers.
    ``duplicate_sequences`` lists sequence numbers that appear more than
    once. Both are computed from the full set of observed sequence numbers,
    independent of the order the mappings were supplied in. See
    :class:`KVEventStreamReport` for ``ambiguous_block_hashes`` semantics.
    """

    events = tuple(parse_kv_event(mapping) for mapping in mappings)
    counts = Counter(event.sequence for event in events)
    duplicate_sequences = tuple(
        sorted(sequence for sequence, count in counts.items() if count > 1)
    )
    unique_sorted = sorted(counts)
    sequence_gaps = tuple(
        (unique_sorted[index] + 1, unique_sorted[index + 1] - 1)
        for index in range(len(unique_sorted) - 1)
        if unique_sorted[index + 1] - unique_sorted[index] > 1
    )

    hash_token_ids: dict[str, set[tuple[int, ...]]] = {}
    for event in events:
        if event.event_type is not KVEventType.BLOCK_STORED:
            continue
        assert event.block_size is not None
        for index, block_hash in enumerate(event.block_hashes):
            start = index * event.block_size
            block_tokens = event.token_ids[start : start + event.block_size]
            hash_token_ids.setdefault(block_hash, set()).add(block_tokens)
    ambiguous_block_hashes = tuple(
        tuple(sorted(token_id_tuples))
        for _, token_id_tuples in sorted(hash_token_ids.items())
        if len(token_id_tuples) > 1
    )

    return KVEventStreamReport(
        events=events,
        sequence_gaps=sequence_gaps,
        duplicate_sequences=duplicate_sequences,
        ambiguous_block_hashes=ambiguous_block_hashes,
    )


_REQUEST_OBSERVATION_KEYS = {
    "request_id",
    "prompt_token_ids",
    "output_token_ids",
    "num_cached_tokens",
    "num_cache_creation_tokens",
    "finished",
    "finish_reason",
    "arrival_time",
    "first_token_time",
    "finished_time",
}


@dataclass(frozen=True)
class VLLMRequestObservation:
    """One normalized, synthetic/offline-captured vLLM ``RequestOutput`` observation.

    Token ID arrays are preserved on the instance (needed for offline reuse
    accounting) but are never included in :meth:`redact`, which keeps their
    counts instead. Timing fields are nullable wall-clock timestamps
    (seconds); durations are derived only when both of their endpoints are
    present, and :func:`parse_vllm_request_observation` guarantees any
    present timestamps are already mutually monotonic.
    """

    request_id: str
    prompt_token_ids: tuple[int, ...]
    output_token_ids: tuple[int, ...]
    num_cached_tokens: int
    num_cache_creation_tokens: int
    finished: bool
    finish_reason: str | None = None
    arrival_time: float | None = None
    first_token_time: float | None = None
    finished_time: float | None = None

    @property
    def queue_duration(self) -> float | None:
        """Arrival-to-first-token wait, or ``None`` unless both timestamps exist."""

        if self.arrival_time is None or self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def ttft_duration(self) -> float | None:
        """Time-to-first-token, or ``None`` unless both timestamps exist.

        This synthetic schema exposes only ``arrival_time`` (there is no
        separate engine-scheduled timestamp), so queueing delay and
        time-to-first-token cannot be distinguished from each other; this
        property intentionally mirrors :attr:`queue_duration` so callers can
        key on whichever name matches their terminology.
        """

        return self.queue_duration

    @property
    def complete_duration(self) -> float | None:
        """Arrival-to-finished total duration, or ``None`` unless both exist."""

        if self.arrival_time is None or self.finished_time is None:
            return None
        return self.finished_time - self.arrival_time

    def redact(self) -> dict[str, Any]:
        """Return a publication-safe view of this observation.

        Removes the raw prompt/output token ID arrays, keeping only their
        counts, alongside the counters, status, timestamps, and derived
        durations.
        """

        return {
            "request_id": self.request_id,
            "num_prompt_tokens": len(self.prompt_token_ids),
            "num_output_tokens": len(self.output_token_ids),
            "num_cached_tokens": self.num_cached_tokens,
            "num_cache_creation_tokens": self.num_cache_creation_tokens,
            "finished": self.finished,
            "finish_reason": self.finish_reason,
            "arrival_time": self.arrival_time,
            "first_token_time": self.first_token_time,
            "finished_time": self.finished_time,
            "queue_duration": self.queue_duration,
            "ttft_duration": self.ttft_duration,
            "complete_duration": self.complete_duration,
        }


def parse_vllm_request_observation(
    mapping: Mapping[str, Any],
) -> VLLMRequestObservation:
    """Strictly parse one synthetic/offline-captured vLLM ``RequestOutput`` mapping.

    Rejects mappings with missing or unexpected keys, malformed field types,
    and internally contradictory observations: more cached or newly-created
    tokens than the prompt holds, a finished request without a finish
    reason, or timestamps that are not ``arrival_time <= first_token_time <=
    finished_time`` whenever both endpoints of a pair are present.
    """

    data = _require_mapping(mapping, "request_observation")
    _exact_keys(data, _REQUEST_OBSERVATION_KEYS, "VLLMRequestObservation")

    request_id = _require_str(data["request_id"], "request_observation.request_id")
    prompt_token_ids = _require_int_list(
        data["prompt_token_ids"], "request_observation.prompt_token_ids"
    )
    output_token_ids = _require_int_list(
        data["output_token_ids"], "request_observation.output_token_ids"
    )
    num_cached_tokens = _require_int(
        data["num_cached_tokens"], "request_observation.num_cached_tokens"
    )
    num_cache_creation_tokens = _require_int(
        data["num_cache_creation_tokens"],
        "request_observation.num_cache_creation_tokens",
    )
    finished = _require_bool(data["finished"], "request_observation.finished")
    finish_reason = _optional_str(
        data["finish_reason"], "request_observation.finish_reason"
    )
    arrival_time = _optional_finite_number(
        data["arrival_time"], "request_observation.arrival_time"
    )
    first_token_time = _optional_finite_number(
        data["first_token_time"], "request_observation.first_token_time"
    )
    finished_time = _optional_finite_number(
        data["finished_time"], "request_observation.finished_time"
    )

    if num_cached_tokens > len(prompt_token_ids):
        raise SchemaValidationError(
            "request_observation.num_cached_tokens exceeds prompt length"
        )
    if num_cache_creation_tokens > len(prompt_token_ids):
        raise SchemaValidationError(
            "request_observation.num_cache_creation_tokens exceeds prompt length"
        )
    if num_cached_tokens + num_cache_creation_tokens > len(prompt_token_ids):
        raise SchemaValidationError(
            "request_observation cached and cache-creation tokens exceed prompt length"
        )
    if finished and finish_reason is None:
        raise SchemaValidationError(
            "request_observation.finished is true but finish_reason is null"
        )
    if (
        arrival_time is not None
        and first_token_time is not None
        and arrival_time > first_token_time
    ):
        raise SchemaValidationError(
            "request_observation timestamps are not monotonic: "
            "arrival_time is after first_token_time"
        )
    if (
        first_token_time is not None
        and finished_time is not None
        and first_token_time > finished_time
    ):
        raise SchemaValidationError(
            "request_observation timestamps are not monotonic: "
            "first_token_time is after finished_time"
        )
    if (
        arrival_time is not None
        and finished_time is not None
        and arrival_time > finished_time
    ):
        raise SchemaValidationError(
            "request_observation timestamps are not monotonic: "
            "arrival_time is after finished_time"
        )

    return VLLMRequestObservation(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        output_token_ids=output_token_ids,
        num_cached_tokens=num_cached_tokens,
        num_cache_creation_tokens=num_cache_creation_tokens,
        finished=finished,
        finish_reason=finish_reason,
        arrival_time=arrival_time,
        first_token_time=first_token_time,
        finished_time=finished_time,
    )
