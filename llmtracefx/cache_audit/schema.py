"""Strict schema for cache-audit inputs and per-request evidence."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar, cast

from llmtracefx.optimizer.schema import Measurement, SchemaValidationError

CACHE_AUDIT_SCHEMA_VERSION = "2"
T = TypeVar("T")


class EvidenceBasis(str, Enum):
    """How a cache-audit fact was obtained."""

    OBSERVED = "observed"
    ENGINE_ATTESTED = "engine_attested"
    INDEPENDENTLY_DERIVED = "independently_derived"
    ESTIMATED = "estimated"
    UNAVAILABLE = "unavailable"


class PublicationMode(str, Enum):
    PRIVATE = "private"
    PUBLIC_SYNTHETIC = "public_synthetic"
    PUBLIC_REDACTED = "public_redacted"


class ScenarioKind(str, Enum):
    COLD = "cold"
    IDENTICAL_PREFIX = "identical_prefix"
    FIRST_TOKEN_MUTATION = "first_token_mutation"
    WITHIN_BLOCK_MUTATION = "within_block_mutation"
    BLOCK_BOUNDARY_MUTATION = "block_boundary_mutation"
    SUFFIX_CHANGE = "suffix_change"
    SAME_LENGTH_DIFFERENT_IDS = "same_length_different_ids"
    DUPLICATE = "duplicate"
    EVICTION_COUNT = "eviction_count"
    EVICTION_BYTES = "eviction_bytes"
    MIXED_LENGTH_CONCURRENT = "mixed_length_concurrent"
    SAVED_CACHE_MISMATCH = "saved_cache_mismatch"
    QUANTIZED_CACHE = "quantized_cache"
    ROTATING_CACHE = "rotating_cache"
    NAMESPACE_ISOLATION = "namespace_isolation"
    MULTIMODAL_IDENTITY = "multimodal_identity"
    HASH_COLLISION = "hash_collision"
    PREEMPTION = "preemption"
    SPECULATIVE = "speculative"


class Verdict(str, Enum):
    """Cache-reuse verdict only; output eligibility is tracked separately."""

    INVALID = "invalid"
    UNSUPPORTED = "unsupported"
    EVICTED = "evicted"
    RECOMPUTED = "recomputed"
    VERIFIED_MISS = "verified_miss"
    PARTIAL_REUSE = "partial_reuse"
    VERIFIED_HIT = "verified_hit"
    ATTESTED_ONLY = "attested_only"


class EligibilityStatus(str, Enum):
    ELIGIBLE = "eligible"
    INELIGIBLE = "ineligible"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"


class PairRole(str, Enum):
    SINGLE = "single"
    CONTROL = "control"
    TREATMENT = "treatment"


class TerminalState(str, Enum):
    COMPLETED = "completed"
    REFUSED = "refused"
    FAILED = "failed"


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{context} must be an object")
    return value


def _exact(data: dict[str, Any], keys: set[str], context: str) -> None:
    actual = set(data)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise SchemaValidationError(
            f"{context} fields differ: missing={missing}, extra={extra}"
        )


def _string(value: Any, context: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise SchemaValidationError(f"{context} must be a non-empty string")
    return value


def _optional_string(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _string(value, context)


def _integer(value: Any, context: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SchemaValidationError(
            f"{context} must be an integer greater than or equal to {minimum}"
        )
    return value


def _optional_integer(value: Any, context: str, *, minimum: int = 0) -> int | None:
    if value is None:
        return None
    return _integer(value, context, minimum=minimum)


def _strings(value: Any, context: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise SchemaValidationError(f"{context} must be an array")
    return tuple(
        _string(item, f"{context}[{index}]") for index, item in enumerate(value)
    )


def _integers(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise SchemaValidationError(f"{context} must be an array")
    return tuple(
        _integer(item, f"{context}[{index}]") for index, item in enumerate(value)
    )


@dataclass(frozen=True)
class EvidenceFact(Generic[T]):
    """One claim-bearing value with explicit evidence strength and limitations."""

    value: T | None
    basis: EvidenceBasis
    source: str
    scope: str = "request"
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.basis is EvidenceBasis.UNAVAILABLE and self.value is not None:
            raise SchemaValidationError("unavailable facts must have a null value")
        if self.basis is not EvidenceBasis.UNAVAILABLE and self.value is None:
            raise SchemaValidationError("available facts must have a non-null value")
        _string(self.source, "EvidenceFact.source")
        _string(self.scope, "EvidenceFact.scope")
        for index, limitation in enumerate(self.limitations):
            _string(limitation, f"EvidenceFact.limitations[{index}]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "basis": self.basis.value,
            "source": self.source,
            "scope": self.scope,
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(
        cls, value: Any, *, context: str = "EvidenceFact"
    ) -> EvidenceFact[Any]:
        data = _object(value, context)
        _exact(data, {"value", "basis", "source", "scope", "limitations"}, context)
        try:
            basis = EvidenceBasis(data["basis"])
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(f"{context}.basis is invalid") from exc
        return cls(
            value=data["value"],
            basis=basis,
            source=_string(data["source"], f"{context}.source"),
            scope=_string(data["scope"], f"{context}.scope"),
            limitations=_strings(data["limitations"], f"{context}.limitations"),
        )


def unavailable(source: str, *limitations: str) -> EvidenceFact[Any]:
    """Construct an unavailable fact without inventing a zero value."""

    return EvidenceFact(
        value=None,
        basis=EvidenceBasis.UNAVAILABLE,
        source=source,
        limitations=tuple(limitations),
    )


@dataclass(frozen=True)
class Limitation:
    code: str
    message: str
    blocks_verdict: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "blocks_verdict": self.blocks_verdict,
        }

    @classmethod
    def from_dict(cls, value: Any) -> Limitation:
        data = _object(value, "Limitation")
        _exact(data, {"code", "message", "blocks_verdict"}, "Limitation")
        if not isinstance(data["blocks_verdict"], bool):
            raise SchemaValidationError("Limitation.blocks_verdict must be a boolean")
        return cls(
            code=_string(data["code"], "Limitation.code"),
            message=_string(data["message"], "Limitation.message"),
            blocks_verdict=data["blocks_verdict"],
        )


@dataclass(frozen=True)
class CacheConfig:
    namespace_id: str
    cache_type: str
    max_entries: int | None = None
    max_bytes: int | None = None
    hash_algorithm: str | None = None
    hash_block_size: int | None = None
    physical_block_sizes: tuple[int, ...] = ()
    fine_grained_hits: bool = False
    cache_salt_relationship: str | None = None

    def __post_init__(self) -> None:
        _string(self.namespace_id, "CacheConfig.namespace_id")
        _string(self.cache_type, "CacheConfig.cache_type")
        _optional_integer(self.max_entries, "CacheConfig.max_entries", minimum=1)
        _optional_integer(self.max_bytes, "CacheConfig.max_bytes", minimum=1)
        _optional_integer(
            self.hash_block_size, "CacheConfig.hash_block_size", minimum=1
        )
        for index, size in enumerate(self.physical_block_sizes):
            _integer(size, f"CacheConfig.physical_block_sizes[{index}]", minimum=1)
        if self.hash_block_size is None and self.physical_block_sizes:
            raise SchemaValidationError(
                "physical block sizes require a hash block size"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace_id": self.namespace_id,
            "cache_type": self.cache_type,
            "max_entries": self.max_entries,
            "max_bytes": self.max_bytes,
            "hash_algorithm": self.hash_algorithm,
            "hash_block_size": self.hash_block_size,
            "physical_block_sizes": list(self.physical_block_sizes),
            "fine_grained_hits": self.fine_grained_hits,
            "cache_salt_relationship": self.cache_salt_relationship,
        }

    @classmethod
    def from_dict(cls, value: Any) -> CacheConfig:
        data = _object(value, "CacheConfig")
        keys = {
            "namespace_id",
            "cache_type",
            "max_entries",
            "max_bytes",
            "hash_algorithm",
            "hash_block_size",
            "physical_block_sizes",
            "fine_grained_hits",
            "cache_salt_relationship",
        }
        _exact(data, keys, "CacheConfig")
        if not isinstance(data["fine_grained_hits"], bool):
            raise SchemaValidationError("CacheConfig.fine_grained_hits must be boolean")
        return cls(
            namespace_id=_string(data["namespace_id"], "CacheConfig.namespace_id"),
            cache_type=_string(data["cache_type"], "CacheConfig.cache_type"),
            max_entries=_optional_integer(
                data["max_entries"], "CacheConfig.max_entries", minimum=1
            ),
            max_bytes=_optional_integer(
                data["max_bytes"], "CacheConfig.max_bytes", minimum=1
            ),
            hash_algorithm=_optional_string(
                data["hash_algorithm"], "CacheConfig.hash_algorithm"
            ),
            hash_block_size=_optional_integer(
                data["hash_block_size"], "CacheConfig.hash_block_size", minimum=1
            ),
            physical_block_sizes=_integers(
                data["physical_block_sizes"], "CacheConfig.physical_block_sizes"
            ),
            fine_grained_hits=data["fine_grained_hits"],
            cache_salt_relationship=_optional_string(
                data["cache_salt_relationship"],
                "CacheConfig.cache_salt_relationship",
            ),
        )


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    scenario: ScenarioKind
    order: int
    input_token_ids: tuple[int, ...] | None
    input_token_count: int
    output_tokens: int = 2
    pair_id: str | None = None
    mutation_position: int | None = None
    expected_predecessors: tuple[str, ...] = ()
    namespace_id: str = "default"
    replicate_id: str = "replicate-0"
    pair_role: PairRole = PairRole.SINGLE

    def __post_init__(self) -> None:
        _string(self.request_id, "RequestSpec.request_id")
        _integer(self.order, "RequestSpec.order")
        _integer(self.input_token_count, "RequestSpec.input_token_count", minimum=1)
        if self.input_token_ids is not None:
            if len(self.input_token_ids) != self.input_token_count:
                raise SchemaValidationError(
                    "RequestSpec.input_token_count is inconsistent"
                )
            for index, token in enumerate(self.input_token_ids):
                _integer(token, f"RequestSpec.input_token_ids[{index}]")
        _integer(self.output_tokens, "RequestSpec.output_tokens", minimum=1)
        if self.mutation_position is not None:
            position = _integer(self.mutation_position, "RequestSpec.mutation_position")
            if position >= self.input_token_count:
                raise SchemaValidationError(
                    "RequestSpec.mutation_position is outside the input"
                )
        _string(self.namespace_id, "RequestSpec.namespace_id")
        _string(self.replicate_id, "RequestSpec.replicate_id")
        if self.pair_id is None and self.pair_role is not PairRole.SINGLE:
            raise SchemaValidationError("paired roles require RequestSpec.pair_id")
        if self.pair_id is not None and self.pair_role is PairRole.SINGLE:
            raise SchemaValidationError("RequestSpec.pair_id requires a paired role")

    def to_dict(self, *, include_tokens: bool = True) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "scenario": self.scenario.value,
            "order": self.order,
            "input_token_ids": (
                list(self.input_token_ids)
                if include_tokens and self.input_token_ids is not None
                else None
            ),
            "input_token_count": self.input_token_count,
            "output_tokens": self.output_tokens,
            "pair_id": self.pair_id,
            "mutation_position": self.mutation_position,
            "expected_predecessors": list(self.expected_predecessors),
            "namespace_id": self.namespace_id,
            "replicate_id": self.replicate_id,
            "pair_role": self.pair_role.value,
        }

    @classmethod
    def from_dict(cls, value: Any) -> RequestSpec:
        data = _object(value, "RequestSpec")
        keys = {
            "request_id",
            "scenario",
            "order",
            "input_token_ids",
            "input_token_count",
            "output_tokens",
            "pair_id",
            "mutation_position",
            "expected_predecessors",
            "namespace_id",
            "replicate_id",
            "pair_role",
        }
        _exact(data, keys, "RequestSpec")
        tokens = (
            None
            if data["input_token_ids"] is None
            else _integers(data["input_token_ids"], "RequestSpec.input_token_ids")
        )
        count = _integer(data["input_token_count"], "RequestSpec.input_token_count")
        if tokens is not None and count != len(tokens):
            raise SchemaValidationError("RequestSpec.input_token_count is inconsistent")
        try:
            scenario = ScenarioKind(data["scenario"])
            pair_role = PairRole(data["pair_role"])
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError("RequestSpec enum value is invalid") from exc
        return cls(
            request_id=_string(data["request_id"], "RequestSpec.request_id"),
            scenario=scenario,
            order=_integer(data["order"], "RequestSpec.order"),
            input_token_ids=tokens,
            input_token_count=count,
            output_tokens=_integer(
                data["output_tokens"], "RequestSpec.output_tokens", minimum=1
            ),
            pair_id=_optional_string(data["pair_id"], "RequestSpec.pair_id"),
            mutation_position=_optional_integer(
                data["mutation_position"], "RequestSpec.mutation_position"
            ),
            expected_predecessors=_strings(
                data["expected_predecessors"], "RequestSpec.expected_predecessors"
            ),
            namespace_id=_string(data["namespace_id"], "RequestSpec.namespace_id"),
            replicate_id=_string(data["replicate_id"], "RequestSpec.replicate_id"),
            pair_role=pair_role,
        )


@dataclass(frozen=True)
class RequestCacheIdentity:
    """Exact cache identity needed to relate an eviction probe to its predecessor."""

    backend: str
    model_id: str
    tokenizer_id: str
    model_artifact_digest: str | None
    cache_config_digest: str
    namespace_id: str
    input_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in (
            "backend",
            "model_id",
            "tokenizer_id",
            "cache_config_digest",
            "namespace_id",
        ):
            _string(getattr(self, name), f"RequestCacheIdentity.{name}")
        for name, value in (
            ("model_artifact_digest", self.model_artifact_digest),
            ("cache_config_digest", self.cache_config_digest),
        ):
            if (
                value is not None
                and re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
            ):
                raise SchemaValidationError(
                    f"RequestCacheIdentity.{name} must be a SHA-256 digest"
                )
        if not self.input_token_ids:
            raise SchemaValidationError(
                "RequestCacheIdentity.input_token_ids must not be empty"
            )
        for index, token in enumerate(self.input_token_ids):
            _integer(token, f"RequestCacheIdentity.input_token_ids[{index}]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "model_artifact_digest": self.model_artifact_digest,
            "cache_config_digest": self.cache_config_digest,
            "namespace_id": self.namespace_id,
            "input_token_ids": list(self.input_token_ids),
        }

    @classmethod
    def from_dict(cls, value: Any, *, context: str) -> RequestCacheIdentity:
        data = _object(value, context)
        keys = {
            "backend",
            "model_id",
            "tokenizer_id",
            "model_artifact_digest",
            "cache_config_digest",
            "namespace_id",
            "input_token_ids",
        }
        _exact(data, keys, context)
        return cls(
            backend=_string(data["backend"], f"{context}.backend"),
            model_id=_string(data["model_id"], f"{context}.model_id"),
            tokenizer_id=_string(data["tokenizer_id"], f"{context}.tokenizer_id"),
            model_artifact_digest=_optional_string(
                data["model_artifact_digest"],
                f"{context}.model_artifact_digest",
            ),
            cache_config_digest=_string(
                data["cache_config_digest"], f"{context}.cache_config_digest"
            ),
            namespace_id=_string(data["namespace_id"], f"{context}.namespace_id"),
            input_token_ids=_integers(
                data["input_token_ids"], f"{context}.input_token_ids"
            ),
        )


@dataclass(frozen=True)
class EvictionPredecessorProof:
    """Identity-bound predecessor and current request used by an eviction verdict."""

    predecessor_request_id: str
    predecessor: RequestCacheIdentity
    current: RequestCacheIdentity
    reusable_prefix_tokens: int

    def __post_init__(self) -> None:
        _string(
            self.predecessor_request_id,
            "EvictionPredecessorProof.predecessor_request_id",
        )
        _integer(
            self.reusable_prefix_tokens,
            "EvictionPredecessorProof.reusable_prefix_tokens",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "predecessor_request_id": self.predecessor_request_id,
            "predecessor": self.predecessor.to_dict(),
            "current": self.current.to_dict(),
            "reusable_prefix_tokens": self.reusable_prefix_tokens,
        }

    @classmethod
    def from_dict(cls, value: Any) -> EvictionPredecessorProof:
        data = _object(value, "EvictionPredecessorProof")
        _exact(
            data,
            {
                "predecessor_request_id",
                "predecessor",
                "current",
                "reusable_prefix_tokens",
            },
            "EvictionPredecessorProof",
        )
        return cls(
            predecessor_request_id=_string(
                data["predecessor_request_id"],
                "EvictionPredecessorProof.predecessor_request_id",
            ),
            predecessor=RequestCacheIdentity.from_dict(
                data["predecessor"],
                context="EvictionPredecessorProof.predecessor",
            ),
            current=RequestCacheIdentity.from_dict(
                data["current"],
                context="EvictionPredecessorProof.current",
            ),
            reusable_prefix_tokens=_integer(
                data["reusable_prefix_tokens"],
                "EvictionPredecessorProof.reusable_prefix_tokens",
            ),
        )


@dataclass(frozen=True)
class ReuseEvidence:
    semantic_prefix_tokens: EvidenceFact[int]
    policy_reusable_tokens: EvidenceFact[int]
    reusable_blocks: EvidenceFact[int]
    partial_block_tokens: EvidenceFact[int]
    engine_cached_tokens: EvidenceFact[int]
    engine_cached_blocks: EvidenceFact[int]
    engine_created_tokens: EvidenceFact[int]
    observed_prompt_tokens: EvidenceFact[int]
    policy_required_prompt_tokens: EvidenceFact[int]
    unexpected_recomputed_tokens: EvidenceFact[int]
    eviction_observed: EvidenceFact[bool]
    preemption_observed: EvidenceFact[bool]
    prior_residency_observed: EvidenceFact[bool] = field(
        default_factory=lambda: unavailable("cache", "prior_residency_unavailable")
    )
    residency_absence_observed: EvidenceFact[bool] = field(
        default_factory=lambda: unavailable("cache", "residency_absence_unavailable")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            name: cast(EvidenceFact[Any], getattr(self, name)).to_dict()
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: Any) -> ReuseEvidence:
        data = _object(value, "ReuseEvidence")
        keys = set(cls.__dataclass_fields__)
        _exact(data, keys, "ReuseEvidence")
        return cls(
            **{
                name: EvidenceFact.from_dict(
                    data[name], context=f"ReuseEvidence.{name}"
                )
                for name in keys
            }
        )


@dataclass(frozen=True)
class CacheStateSnapshot:
    """Cache state at a request boundary without exposing cache identities."""

    entry_count: EvidenceFact[int]
    logical_bytes: EvidenceFact[int]
    valid_token_offsets: EvidenceFact[list[int]]
    cache_classes: EvidenceFact[list[str]]
    complete: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_count": self.entry_count.to_dict(),
            "logical_bytes": self.logical_bytes.to_dict(),
            "valid_token_offsets": self.valid_token_offsets.to_dict(),
            "cache_classes": self.cache_classes.to_dict(),
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, value: Any) -> CacheStateSnapshot:
        data = _object(value, "CacheStateSnapshot")
        keys = {
            "entry_count",
            "logical_bytes",
            "valid_token_offsets",
            "cache_classes",
            "complete",
        }
        _exact(data, keys, "CacheStateSnapshot")
        if not isinstance(data["complete"], bool):
            raise SchemaValidationError("CacheStateSnapshot.complete must be boolean")
        return cls(
            entry_count=EvidenceFact.from_dict(
                data["entry_count"], context="CacheStateSnapshot.entry_count"
            ),
            logical_bytes=EvidenceFact.from_dict(
                data["logical_bytes"], context="CacheStateSnapshot.logical_bytes"
            ),
            valid_token_offsets=EvidenceFact.from_dict(
                data["valid_token_offsets"],
                context="CacheStateSnapshot.valid_token_offsets",
            ),
            cache_classes=EvidenceFact.from_dict(
                data["cache_classes"], context="CacheStateSnapshot.cache_classes"
            ),
            complete=data["complete"],
        )


@dataclass(frozen=True)
class CacheEventRecord:
    """Privacy-safe normalized cache event."""

    sequence: int
    event_type: str
    basis: EvidenceBasis
    token_count: int | None = None
    block_count: int | None = None
    medium: str | None = None
    group_index: int | None = None
    limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _integer(self.sequence, "CacheEventRecord.sequence")
        _string(self.event_type, "CacheEventRecord.event_type")
        _optional_integer(self.token_count, "CacheEventRecord.token_count")
        _optional_integer(self.block_count, "CacheEventRecord.block_count")
        _optional_integer(self.group_index, "CacheEventRecord.group_index")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "event_type": self.event_type,
            "basis": self.basis.value,
            "token_count": self.token_count,
            "block_count": self.block_count,
            "medium": self.medium,
            "group_index": self.group_index,
            "limitations": list(self.limitations),
        }

    @classmethod
    def from_dict(cls, value: Any) -> CacheEventRecord:
        data = _object(value, "CacheEventRecord")
        keys = {
            "sequence",
            "event_type",
            "basis",
            "token_count",
            "block_count",
            "medium",
            "group_index",
            "limitations",
        }
        _exact(data, keys, "CacheEventRecord")
        try:
            basis = EvidenceBasis(data["basis"])
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError("CacheEventRecord.basis is invalid") from exc
        return cls(
            sequence=_integer(data["sequence"], "CacheEventRecord.sequence"),
            event_type=_string(data["event_type"], "CacheEventRecord.event_type"),
            basis=basis,
            token_count=_optional_integer(
                data["token_count"], "CacheEventRecord.token_count"
            ),
            block_count=_optional_integer(
                data["block_count"], "CacheEventRecord.block_count"
            ),
            medium=_optional_string(data["medium"], "CacheEventRecord.medium"),
            group_index=_optional_integer(
                data["group_index"], "CacheEventRecord.group_index"
            ),
            limitations=_strings(data["limitations"], "CacheEventRecord.limitations"),
        )


@dataclass(frozen=True)
class TimingEvidence:
    client_ttft: Measurement | None = None
    in_process_first_token: Measurement | None = None
    queue: Measurement | None = None
    scheduling: Measurement | None = None
    prefill: Measurement | None = None
    decode: Measurement | None = None
    total: Measurement | None = None
    scope: str = "unavailable"
    exclusions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _string(self.scope, "TimingEvidence.scope")
        for index, exclusion in enumerate(self.exclusions):
            _string(exclusion, f"TimingEvidence.exclusions[{index}]")

    def to_dict(self) -> dict[str, Any]:
        measurements = {
            name: None if value is None else value.to_dict()
            for name, value in (
                (name, getattr(self, name))
                for name in (
                    "client_ttft",
                    "in_process_first_token",
                    "queue",
                    "scheduling",
                    "prefill",
                    "decode",
                    "total",
                )
            )
        }
        return {
            **measurements,
            "scope": self.scope,
            "exclusions": list(self.exclusions),
        }

    @classmethod
    def from_dict(cls, value: Any) -> TimingEvidence:
        data = _object(value, "TimingEvidence")
        keys = set(cls.__dataclass_fields__)
        _exact(data, keys, "TimingEvidence")
        return cls(
            **{
                name: None if data[name] is None else Measurement.from_dict(data[name])
                for name in (
                    "client_ttft",
                    "in_process_first_token",
                    "queue",
                    "scheduling",
                    "prefill",
                    "decode",
                    "total",
                )
            },
            scope=_string(data["scope"], "TimingEvidence.scope"),
            exclusions=_strings(data["exclusions"], "TimingEvidence.exclusions"),
        )


@dataclass(frozen=True)
class MemoryEvidence:
    runtime_active_bytes: EvidenceFact[int]
    runtime_peak_bytes: EvidenceFact[int]
    allocator_cache_bytes: EvidenceFact[int]
    logical_cache_bytes: EvidenceFact[int]
    physical_cache_blocks: EvidenceFact[int]

    def to_dict(self) -> dict[str, Any]:
        return {
            name: cast(EvidenceFact[Any], getattr(self, name)).to_dict()
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: Any) -> MemoryEvidence:
        data = _object(value, "MemoryEvidence")
        keys = set(cls.__dataclass_fields__)
        _exact(data, keys, "MemoryEvidence")
        return cls(
            **{
                name: EvidenceFact.from_dict(
                    data[name], context=f"MemoryEvidence.{name}"
                )
                for name in keys
            }
        )


@dataclass(frozen=True)
class OutputEvidence:
    output_token_ids: tuple[int, ...] | None
    baseline_token_ids: tuple[int, ...] | None
    token_identity: EvidenceFact[bool]
    correctness: EvidenceFact[bool]
    finish_reason: str | None

    def to_dict(self, *, include_tokens: bool = True) -> dict[str, Any]:
        return {
            "output_token_ids": (
                list(self.output_token_ids)
                if include_tokens and self.output_token_ids is not None
                else None
            ),
            "baseline_token_ids": (
                list(self.baseline_token_ids)
                if include_tokens and self.baseline_token_ids is not None
                else None
            ),
            "token_identity": self.token_identity.to_dict(),
            "correctness": self.correctness.to_dict(),
            "finish_reason": self.finish_reason,
        }

    @classmethod
    def from_dict(cls, value: Any) -> OutputEvidence:
        data = _object(value, "OutputEvidence")
        keys = {
            "output_token_ids",
            "baseline_token_ids",
            "token_identity",
            "correctness",
            "finish_reason",
        }
        _exact(data, keys, "OutputEvidence")
        return cls(
            output_token_ids=None
            if data["output_token_ids"] is None
            else _integers(data["output_token_ids"], "OutputEvidence.output_token_ids"),
            baseline_token_ids=None
            if data["baseline_token_ids"] is None
            else _integers(
                data["baseline_token_ids"], "OutputEvidence.baseline_token_ids"
            ),
            token_identity=EvidenceFact.from_dict(
                data["token_identity"], context="OutputEvidence.token_identity"
            ),
            correctness=EvidenceFact.from_dict(
                data["correctness"], context="OutputEvidence.correctness"
            ),
            finish_reason=_optional_string(
                data["finish_reason"], "OutputEvidence.finish_reason"
            ),
        )


@dataclass(frozen=True)
class CostEvidence:
    """Billed and estimated cost stay separate and nullable."""

    billed: EvidenceFact[float] = field(
        default_factory=lambda: unavailable("cost", "billed_cost_unavailable")
    )
    estimated: EvidenceFact[float] = field(
        default_factory=lambda: unavailable("cost", "estimated_cost_unavailable")
    )
    currency: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "billed": self.billed.to_dict(),
            "estimated": self.estimated.to_dict(),
            "currency": self.currency,
        }

    @classmethod
    def from_dict(cls, value: Any) -> CostEvidence:
        data = _object(value, "CostEvidence")
        _exact(data, {"billed", "estimated", "currency"}, "CostEvidence")
        return cls(
            billed=EvidenceFact.from_dict(
                data["billed"], context="CostEvidence.billed"
            ),
            estimated=EvidenceFact.from_dict(
                data["estimated"], context="CostEvidence.estimated"
            ),
            currency=_optional_string(data["currency"], "CostEvidence.currency"),
        )


@dataclass(frozen=True)
class ClaimEligibility:
    """Eligibility gates distinct from the cache-reuse verdict."""

    output_equivalence: EligibilityStatus
    performance: EligibilityStatus
    quality: EligibilityStatus
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_equivalence": self.output_equivalence.value,
            "performance": self.performance.value,
            "quality": self.quality.value,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, value: Any) -> ClaimEligibility:
        data = _object(value, "ClaimEligibility")
        _exact(
            data,
            {"output_equivalence", "performance", "quality", "reasons"},
            "ClaimEligibility",
        )
        try:
            return cls(
                output_equivalence=EligibilityStatus(data["output_equivalence"]),
                performance=EligibilityStatus(data["performance"]),
                quality=EligibilityStatus(data["quality"]),
                reasons=_strings(data["reasons"], "ClaimEligibility.reasons"),
            )
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(
                "ClaimEligibility enum value is invalid"
            ) from exc


def unavailable_eligibility() -> ClaimEligibility:
    return ClaimEligibility(
        output_equivalence=EligibilityStatus.UNAVAILABLE,
        performance=EligibilityStatus.INELIGIBLE,
        quality=EligibilityStatus.UNAVAILABLE,
        reasons=("not_classified",),
    )


@dataclass(frozen=True)
class RequestEvidence:
    spec: RequestSpec
    reuse: ReuseEvidence
    timing: TimingEvidence
    memory: MemoryEvidence
    output: OutputEvidence
    terminal_state: TerminalState
    verdict: Verdict | None = None
    verdict_reasons: tuple[str, ...] = ()
    limitations: tuple[Limitation, ...] = ()
    cache_before: CacheStateSnapshot | None = None
    cache_after: CacheStateSnapshot | None = None
    events: tuple[CacheEventRecord, ...] = ()
    eviction_predecessor: EvictionPredecessorProof | None = None
    cost: CostEvidence = field(default_factory=CostEvidence)
    eligibility: ClaimEligibility = field(default_factory=unavailable_eligibility)

    def to_dict(self, *, include_tokens: bool = True) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(include_tokens=include_tokens),
            "reuse": self.reuse.to_dict(),
            "timing": self.timing.to_dict(),
            "memory": self.memory.to_dict(),
            "output": self.output.to_dict(include_tokens=include_tokens),
            "terminal_state": self.terminal_state.value,
            "verdict": None if self.verdict is None else self.verdict.value,
            "verdict_reasons": list(self.verdict_reasons),
            "limitations": [item.to_dict() for item in self.limitations],
            "cache_before": (
                None if self.cache_before is None else self.cache_before.to_dict()
            ),
            "cache_after": (
                None if self.cache_after is None else self.cache_after.to_dict()
            ),
            "events": [event.to_dict() for event in self.events],
            "eviction_predecessor": (
                None
                if self.eviction_predecessor is None
                else self.eviction_predecessor.to_dict()
            ),
            "cost": self.cost.to_dict(),
            "eligibility": self.eligibility.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Any) -> RequestEvidence:
        data = _object(value, "RequestEvidence")
        keys = {
            "spec",
            "reuse",
            "timing",
            "memory",
            "output",
            "terminal_state",
            "verdict",
            "verdict_reasons",
            "limitations",
            "cache_before",
            "cache_after",
            "events",
            "eviction_predecessor",
            "cost",
            "eligibility",
        }
        _exact(data, keys, "RequestEvidence")
        try:
            terminal = TerminalState(data["terminal_state"])
            verdict = None if data["verdict"] is None else Verdict(data["verdict"])
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(
                "RequestEvidence enum value is invalid"
            ) from exc
        limitations_value = data["limitations"]
        if not isinstance(limitations_value, list):
            raise SchemaValidationError("RequestEvidence.limitations must be an array")
        events_value = data["events"]
        if not isinstance(events_value, list):
            raise SchemaValidationError("RequestEvidence.events must be an array")
        return cls(
            spec=RequestSpec.from_dict(data["spec"]),
            reuse=ReuseEvidence.from_dict(data["reuse"]),
            timing=TimingEvidence.from_dict(data["timing"]),
            memory=MemoryEvidence.from_dict(data["memory"]),
            output=OutputEvidence.from_dict(data["output"]),
            terminal_state=terminal,
            verdict=verdict,
            verdict_reasons=_strings(
                data["verdict_reasons"], "RequestEvidence.verdict_reasons"
            ),
            limitations=tuple(Limitation.from_dict(item) for item in limitations_value),
            cache_before=(
                None
                if data["cache_before"] is None
                else CacheStateSnapshot.from_dict(data["cache_before"])
            ),
            cache_after=(
                None
                if data["cache_after"] is None
                else CacheStateSnapshot.from_dict(data["cache_after"])
            ),
            events=tuple(CacheEventRecord.from_dict(item) for item in events_value),
            eviction_predecessor=(
                None
                if data["eviction_predecessor"] is None
                else EvictionPredecessorProof.from_dict(data["eviction_predecessor"])
            ),
            cost=CostEvidence.from_dict(data["cost"]),
            eligibility=ClaimEligibility.from_dict(data["eligibility"]),
        )


@dataclass(frozen=True)
class AuditManifest:
    run_id: str
    created_at: str
    generated_at: str
    backend: str
    backend_version: str
    adapter_version: str
    model_id: str
    tokenizer_id: str
    model_artifact_digest: str | None
    runtime_identity: dict[str, str]
    cache_config: CacheConfig
    publication_mode: PublicationMode
    request_order: tuple[str, ...]
    workload_digest: str
    seed: int
    generator_commit: str | None = None
    generator_commit_at: str | None = None
    generator_package_digest: str | None = None
    limitations: tuple[Limitation, ...] = field(default_factory=tuple)
    schema_version: str = CACHE_AUDIT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CACHE_AUDIT_SCHEMA_VERSION:
            raise SchemaValidationError(
                f"unsupported cache-audit schema version: {self.schema_version}"
            )
        for name in (
            "run_id",
            "created_at",
            "generated_at",
            "backend",
            "backend_version",
            "adapter_version",
            "model_id",
            "tokenizer_id",
            "workload_digest",
        ):
            _string(getattr(self, name), f"AuditManifest.{name}")
        if (
            self.model_artifact_digest is not None
            and re.fullmatch(r"sha256:[0-9a-f]{64}", self.model_artifact_digest) is None
        ):
            raise SchemaValidationError(
                "AuditManifest.model_artifact_digest must be a SHA-256 digest"
            )
        if (
            self.generator_commit is not None
            and re.fullmatch(r"[0-9a-f]{40}", self.generator_commit) is None
        ):
            raise SchemaValidationError(
                "AuditManifest.generator_commit must be a full git SHA"
            )
        if (self.generator_commit is None) != (self.generator_commit_at is None):
            raise SchemaValidationError(
                "AuditManifest.generator_commit and generator_commit_at "
                "must be present together"
            )
        if self.generator_commit_at is not None:
            _string(
                self.generator_commit_at,
                "AuditManifest.generator_commit_at",
            )
        if (
            self.generator_package_digest is not None
            and re.fullmatch(r"sha256:[0-9a-f]{64}", self.generator_package_digest)
            is None
        ):
            raise SchemaValidationError(
                "AuditManifest.generator_package_digest must be a SHA-256 digest"
            )
        _integer(self.seed, "AuditManifest.seed")
        if len(set(self.request_order)) != len(self.request_order):
            raise SchemaValidationError(
                "AuditManifest.request_order contains duplicates"
            )
        for key, value in self.runtime_identity.items():
            _string(key, "AuditManifest.runtime_identity key")
            _string(value, f"AuditManifest.runtime_identity[{key}]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "generated_at": self.generated_at,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "adapter_version": self.adapter_version,
            "model_id": self.model_id,
            "tokenizer_id": self.tokenizer_id,
            "model_artifact_digest": self.model_artifact_digest,
            "runtime_identity": dict(sorted(self.runtime_identity.items())),
            "cache_config": self.cache_config.to_dict(),
            "publication_mode": self.publication_mode.value,
            "request_order": list(self.request_order),
            "workload_digest": self.workload_digest,
            "generator_commit": self.generator_commit,
            "generator_commit_at": self.generator_commit_at,
            "generator_package_digest": self.generator_package_digest,
            "seed": self.seed,
            "limitations": [item.to_dict() for item in self.limitations],
        }

    @classmethod
    def from_dict(cls, value: Any) -> AuditManifest:
        data = _object(value, "AuditManifest")
        keys = {
            "schema_version",
            "run_id",
            "created_at",
            "generated_at",
            "backend",
            "backend_version",
            "adapter_version",
            "model_id",
            "tokenizer_id",
            "model_artifact_digest",
            "runtime_identity",
            "cache_config",
            "publication_mode",
            "request_order",
            "workload_digest",
            "generator_commit",
            "generator_commit_at",
            "generator_package_digest",
            "seed",
            "limitations",
        }
        _exact(data, keys, "AuditManifest")
        runtime = _object(data["runtime_identity"], "AuditManifest.runtime_identity")
        limitations_value = data["limitations"]
        if not isinstance(limitations_value, list):
            raise SchemaValidationError("AuditManifest.limitations must be an array")
        try:
            publication_mode = PublicationMode(data["publication_mode"])
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(
                "AuditManifest.publication_mode is invalid"
            ) from exc
        return cls(
            schema_version=_string(
                data["schema_version"], "AuditManifest.schema_version"
            ),
            run_id=_string(data["run_id"], "AuditManifest.run_id"),
            created_at=_string(data["created_at"], "AuditManifest.created_at"),
            generated_at=_string(data["generated_at"], "AuditManifest.generated_at"),
            backend=_string(data["backend"], "AuditManifest.backend"),
            backend_version=_string(
                data["backend_version"], "AuditManifest.backend_version"
            ),
            adapter_version=_string(
                data["adapter_version"], "AuditManifest.adapter_version"
            ),
            model_id=_string(data["model_id"], "AuditManifest.model_id"),
            tokenizer_id=_string(data["tokenizer_id"], "AuditManifest.tokenizer_id"),
            model_artifact_digest=(
                None
                if data["model_artifact_digest"] is None
                else _string(
                    data["model_artifact_digest"],
                    "AuditManifest.model_artifact_digest",
                )
            ),
            runtime_identity={
                _string(key, "AuditManifest.runtime_identity key"): _string(
                    item, f"AuditManifest.runtime_identity[{key}]"
                )
                for key, item in runtime.items()
            },
            cache_config=CacheConfig.from_dict(data["cache_config"]),
            publication_mode=publication_mode,
            request_order=_strings(
                data["request_order"], "AuditManifest.request_order"
            ),
            workload_digest=_string(
                data["workload_digest"], "AuditManifest.workload_digest"
            ),
            generator_commit=(
                None
                if data["generator_commit"] is None
                else _string(data["generator_commit"], "AuditManifest.generator_commit")
            ),
            generator_commit_at=(
                None
                if data["generator_commit_at"] is None
                else _string(
                    data["generator_commit_at"],
                    "AuditManifest.generator_commit_at",
                )
            ),
            generator_package_digest=(
                None
                if data["generator_package_digest"] is None
                else _string(
                    data["generator_package_digest"],
                    "AuditManifest.generator_package_digest",
                )
            ),
            seed=_integer(data["seed"], "AuditManifest.seed"),
            limitations=tuple(Limitation.from_dict(item) for item in limitations_value),
        )
