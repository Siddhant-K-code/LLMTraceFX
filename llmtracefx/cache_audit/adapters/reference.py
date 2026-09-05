"""Independent synthetic engine and oracle positive control."""

from __future__ import annotations

import hashlib
import platform
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from ..expected import MLXCacheOracle
from ..schema import (
    CacheStateSnapshot,
    EvictionPredecessorProof,
    EvidenceBasis,
    EvidenceFact,
    Limitation,
    MemoryEvidence,
    OutputEvidence,
    RequestCacheIdentity,
    RequestEvidence,
    RequestSpec,
    ReuseEvidence,
    ScenarioKind,
    TerminalState,
    TimingEvidence,
    unavailable,
)
from ..verdicts import classify_request
from .base import AdapterAuditIdentity, CacheAuditCapability


def _fact(
    value: int | bool | list[str],
    basis: EvidenceBasis,
    source: str,
    *,
    scope: str = "synthetic_arithmetic_control",
    limitations: tuple[str, ...] = (),
) -> EvidenceFact:
    return EvidenceFact(
        value=value,
        basis=basis,
        source=source,
        scope=scope,
        limitations=limitations,
    )


def _common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


@dataclass(frozen=True)
class _EngineEntry:
    entry_id: str
    namespace_id: str
    tokens: tuple[int, ...]
    nbytes: int


@dataclass(frozen=True)
class SyntheticEngineObservation:
    cached_tokens: int
    created_tokens: int
    prompt_policy_operations: int
    output_token_ids: tuple[int, ...]
    entry_count_before: int
    logical_bytes_before: int
    entry_count_after: int
    logical_bytes_after: int
    prior_residency_observed: bool
    residency_absence_observed: bool
    eviction_observed: bool


class SyntheticCacheEngine:
    """Synthetic cache implementation with state independent from the oracle."""

    def __init__(
        self,
        *,
        max_entries: int,
        max_bytes: int,
        bytes_per_token: int,
        cached_token_offsets: Mapping[str, int] | None = None,
        prompt_operation_offsets: Mapping[str, int] | None = None,
    ) -> None:
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._bytes_per_token = bytes_per_token
        self._entries: OrderedDict[str, _EngineEntry] = OrderedDict()
        self._ever_resident: set[str] = set()
        self._cached_token_offsets = dict(cached_token_offsets or {})
        self._prompt_operation_offsets = dict(prompt_operation_offsets or {})

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def logical_bytes(self) -> int:
        return sum(entry.nbytes for entry in self._entries.values())

    @property
    def entry_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def _lookup(
        self, namespace_id: str, request: tuple[int, ...]
    ) -> tuple[int, str | None]:
        candidates = [
            entry
            for entry in self._entries.values()
            if entry.namespace_id == namespace_id
        ]
        exact = next((entry for entry in candidates if entry.tokens == request), None)
        if exact is not None:
            return len(request), exact.entry_id

        shorter = [
            entry
            for entry in candidates
            if len(entry.tokens) < len(request)
            and tuple(request[: len(entry.tokens)]) == entry.tokens
        ]
        if shorter:
            entry = max(shorter, key=lambda item: (len(item.tokens), item.entry_id))
            return len(entry.tokens), entry.entry_id

        longer = [
            (_common_prefix(entry.tokens, request), entry)
            for entry in candidates
            if len(entry.tokens) > len(request)
        ]
        longer = [item for item in longer if item[0] > 0]
        if longer:
            common, entry = max(
                longer,
                key=lambda item: (item[0], -len(item[1].tokens), item[1].entry_id),
            )
            return min(len(request) - 1, common), entry.entry_id
        return 0, None

    @staticmethod
    def _generate(tokens: tuple[int, ...], count: int) -> tuple[int, ...]:
        payload = b"|".join(str(token).encode("ascii") for token in tokens)
        digest = hashlib.sha256(payload).digest()
        return tuple(digest[index] for index in range(count))

    def _insert(self, entry: _EngineEntry) -> set[str]:
        duplicate = next(
            (
                entry_id
                for entry_id, candidate in self._entries.items()
                if candidate.namespace_id == entry.namespace_id
                and candidate.tokens == entry.tokens
            ),
            None,
        )
        if duplicate is not None:
            del self._entries[duplicate]
        compacted = [
            entry_id
            for entry_id, candidate in self._entries.items()
            if candidate.namespace_id == entry.namespace_id
            and len(candidate.tokens) < len(entry.tokens)
            and entry.tokens[: len(candidate.tokens)] == candidate.tokens
        ]
        for entry_id in compacted:
            del self._entries[entry_id]
        self._entries[entry.entry_id] = entry
        self._ever_resident.add(entry.entry_id)
        evicted: set[str] = set()
        while (
            len(self._entries) > self._max_entries
            or self.logical_bytes > self._max_bytes
        ):
            entry_id, _ = self._entries.popitem(last=False)
            evicted.add(entry_id)
        return evicted

    def execute(self, request: RequestSpec) -> SyntheticEngineObservation:
        if request.input_token_ids is None:
            raise ValueError("synthetic engine requires exact input token IDs")
        tokens = request.input_token_ids
        entries_before = self.entry_count
        bytes_before = self.logical_bytes
        predecessor_resident = any(
            predecessor in self._entries
            for predecessor in request.expected_predecessors
        )
        predecessor_ever_resident = any(
            predecessor in self._ever_resident
            for predecessor in request.expected_predecessors
        )
        predecessor_absent = (
            bool(request.expected_predecessors) and not predecessor_resident
        )
        cached_tokens, matched_entry = self._lookup(request.namespace_id, tokens)
        cached_tokens += self._cached_token_offsets.get(request.request_id, 0)
        cached_tokens = max(0, min(len(tokens), cached_tokens))
        created_tokens = len(tokens) - cached_tokens
        operations = created_tokens + self._prompt_operation_offsets.get(
            request.request_id, 0
        )
        operations = max(0, operations)
        output = self._generate(tokens, request.output_tokens)
        full_sequence = tokens + output
        nbytes = len(full_sequence) * self._bytes_per_token
        self._insert(
            _EngineEntry(
                entry_id=request.request_id,
                namespace_id=request.namespace_id,
                tokens=full_sequence,
                nbytes=nbytes,
            )
        )
        controlled_eviction = (
            request.scenario
            in {ScenarioKind.EVICTION_COUNT, ScenarioKind.EVICTION_BYTES}
            and predecessor_ever_resident
            and predecessor_absent
            and matched_entry is None
            and cached_tokens == 0
        )
        return SyntheticEngineObservation(
            cached_tokens=cached_tokens,
            created_tokens=created_tokens,
            prompt_policy_operations=operations,
            output_token_ids=output,
            entry_count_before=entries_before,
            logical_bytes_before=bytes_before,
            entry_count_after=self.entry_count,
            logical_bytes_after=self.logical_bytes,
            prior_residency_observed=predecessor_resident
            or predecessor_ever_resident
            or matched_entry is not None,
            residency_absence_observed=predecessor_absent,
            eviction_observed=controlled_eviction,
        )


class SyntheticNoCacheBaseline:
    """Independent no-cache execution path for deterministic equivalence."""

    def __init__(self, corrupt_requests: Sequence[str] = ()) -> None:
        self._corrupt_requests = set(corrupt_requests)

    def execute(self, request: RequestSpec) -> tuple[int, ...]:
        if request.input_token_ids is None:
            raise ValueError("synthetic baseline requires exact input token IDs")
        text = "|".join(map(str, request.input_token_ids))
        digest = hashlib.new("sha256", text.encode("ascii")).digest()
        output = tuple(digest[index] for index in range(request.output_tokens))
        if request.request_id in self._corrupt_requests and output:
            return ((output[0] + 1) % 256, *output[1:])
        return output


class ReferenceCacheAdapter:
    """Run an independent synthetic engine, oracle, and no-cache baseline."""

    def __init__(
        self,
        *,
        max_entries: int = 32,
        max_bytes: int = 1 << 30,
        bytes_per_token: int = 64,
        model_key: str = "synthetic-tiny-model",
        cached_token_offsets: Mapping[str, int] | None = None,
        prompt_operation_offsets: Mapping[str, int] | None = None,
        corrupt_baseline_requests: Sequence[str] = (),
    ) -> None:
        self._oracle = MLXCacheOracle(
            max_entries=max_entries,
            max_bytes=max_bytes,
        )
        self._engine = SyntheticCacheEngine(
            max_entries=max_entries,
            max_bytes=max_bytes,
            bytes_per_token=bytes_per_token,
            cached_token_offsets=cached_token_offsets,
            prompt_operation_offsets=prompt_operation_offsets,
        )
        self._baseline = SyntheticNoCacheBaseline(corrupt_baseline_requests)
        self._bytes_per_token = bytes_per_token
        self._model_key = model_key
        self._max_entries = max_entries
        self._max_bytes = max_bytes

    @property
    def backend(self) -> str:
        return "synthetic_reference"

    def capabilities(self) -> CacheAuditCapability:
        return CacheAuditCapability(
            backend=self.backend,
            supported=True,
            reasons=("synthetic_arithmetic_positive_control_only",),
            observable_facts=(
                "synthetic_engine_cached_tokens",
                "synthetic_engine_policy_operations",
                "synthetic_engine_cache_state",
                "deterministic_output_token_ids",
            ),
            unavailable_facts=(
                "runtime_compute_avoidance",
                "client_ttft",
                "engine_queue_time",
                "runtime_allocator_memory",
                "model_quality",
                "billed_cost",
            ),
        )

    def audit_identity(self) -> AdapterAuditIdentity:
        return AdapterAuditIdentity(
            backend_version="1",
            runtime_identity={
                "implementation": platform.python_implementation(),
                "platform": sys.platform,
                "python": platform.python_version(),
                "synthetic_engine": "independent-state-machine-v2",
            },
            model_artifact_digest=None,
            cache_type="token_trie",
            max_entries=self._max_entries,
            max_bytes=self._max_bytes,
        )

    @staticmethod
    def _snapshot(entries: int, logical_bytes: int) -> CacheStateSnapshot:
        return CacheStateSnapshot(
            entry_count=_fact(
                entries, EvidenceBasis.OBSERVED, "synthetic_engine.resident_entries"
            ),
            logical_bytes=_fact(
                logical_bytes,
                EvidenceBasis.OBSERVED,
                "synthetic_engine.logical_bytes",
                limitations=("not_runtime_or_allocator_memory",),
            ),
            valid_token_offsets=unavailable(
                "synthetic_engine", "cache_offsets_not_exposed"
            ),
            cache_classes=_fact(
                ["SyntheticTokenCache"],
                EvidenceBasis.OBSERVED,
                "synthetic_engine.cache_type",
            ),
            complete=True,
        )

    def run(self, requests: Sequence[RequestSpec]) -> list[RequestEvidence]:
        records: list[RequestEvidence] = []
        prior_specs: dict[str, RequestSpec] = {}
        gated_scenarios = {
            ScenarioKind.MIXED_LENGTH_CONCURRENT,
            ScenarioKind.SAVED_CACHE_MISMATCH,
            ScenarioKind.QUANTIZED_CACHE,
            ScenarioKind.ROTATING_CACHE,
            ScenarioKind.MULTIMODAL_IDENTITY,
            ScenarioKind.HASH_COLLISION,
            ScenarioKind.PREEMPTION,
            ScenarioKind.SPECULATIVE,
        }
        for request in requests:
            if request.input_token_ids is None:
                raise ValueError("reference execution requires exact input token IDs")
            expectation = self._oracle.lookup(
                self._model_key,
                request.namespace_id,
                request.input_token_ids,
            )
            observation = self._engine.execute(request)
            baseline = self._baseline.execute(request)
            output_identity = observation.output_token_ids == baseline
            full_sequence = request.input_token_ids + observation.output_token_ids
            self._oracle.insert(
                entry_id=request.request_id,
                model_key=self._model_key,
                namespace_id=request.namespace_id,
                tokens=full_sequence,
                nbytes=len(full_sequence) * self._bytes_per_token,
            )
            recomputed = max(
                0,
                observation.prompt_policy_operations
                - expectation.policy_required_prompt_tokens,
            )
            limitations = (
                (
                    Limitation(
                        code="backend_specific_instrumentation_required",
                        message=(
                            f"{request.scenario.value} requires a real backend "
                            "or an explicitly injected failure fixture"
                        ),
                        blocks_verdict=True,
                    ),
                )
                if request.scenario in gated_scenarios
                else ()
            )
            record = RequestEvidence(
                spec=request,
                reuse=ReuseEvidence(
                    semantic_prefix_tokens=_fact(
                        expectation.semantic_prefix_tokens,
                        EvidenceBasis.INDEPENDENTLY_DERIVED,
                        "independent_oracle.longest_common_prefix",
                    ),
                    policy_reusable_tokens=_fact(
                        expectation.policy_reusable_tokens,
                        EvidenceBasis.INDEPENDENTLY_DERIVED,
                        "independent_oracle.synthetic_policy",
                    ),
                    reusable_blocks=unavailable(
                        "synthetic_reference", "token_granular_cache"
                    ),
                    partial_block_tokens=unavailable(
                        "synthetic_reference", "token_granular_cache"
                    ),
                    engine_cached_tokens=_fact(
                        observation.cached_tokens,
                        EvidenceBasis.ENGINE_ATTESTED,
                        "synthetic_engine.cached_token_counter",
                    ),
                    engine_cached_blocks=unavailable(
                        "synthetic_reference", "token_granular_cache"
                    ),
                    engine_created_tokens=_fact(
                        observation.created_tokens,
                        EvidenceBasis.ENGINE_ATTESTED,
                        "synthetic_engine.created_token_counter",
                    ),
                    observed_prompt_tokens=_fact(
                        observation.prompt_policy_operations,
                        EvidenceBasis.OBSERVED,
                        "synthetic_engine.policy_operation_counter",
                        limitations=("not_observed_runtime_compute",),
                    ),
                    policy_required_prompt_tokens=_fact(
                        expectation.policy_required_prompt_tokens,
                        EvidenceBasis.INDEPENDENTLY_DERIVED,
                        "independent_oracle.synthetic_policy",
                    ),
                    unexpected_recomputed_tokens=_fact(
                        recomputed,
                        EvidenceBasis.INDEPENDENTLY_DERIVED,
                        "independent_oracle.operation_delta",
                        limitations=("synthetic_operations_not_runtime_compute",),
                    ),
                    prior_residency_observed=_fact(
                        observation.prior_residency_observed,
                        EvidenceBasis.OBSERVED,
                        "synthetic_engine.residency_probe",
                    ),
                    residency_absence_observed=_fact(
                        observation.residency_absence_observed,
                        EvidenceBasis.OBSERVED,
                        "synthetic_engine.residency_probe",
                    ),
                    eviction_observed=_fact(
                        observation.eviction_observed,
                        EvidenceBasis.OBSERVED,
                        "synthetic_engine.controlled_residency_probe",
                    ),
                    preemption_observed=unavailable(
                        "synthetic_reference", "preemption_not_implemented"
                    ),
                ),
                timing=TimingEvidence(
                    scope="unavailable",
                    exclusions=("synthetic_control_emits_no_runtime_timing",),
                ),
                memory=MemoryEvidence(
                    runtime_active_bytes=unavailable(
                        "synthetic_reference", "no_runtime_allocator"
                    ),
                    runtime_peak_bytes=unavailable(
                        "synthetic_reference", "no_runtime_allocator"
                    ),
                    allocator_cache_bytes=unavailable(
                        "synthetic_reference", "no_runtime_allocator"
                    ),
                    logical_cache_bytes=_fact(
                        observation.logical_bytes_after,
                        EvidenceBasis.OBSERVED,
                        "synthetic_engine.logical_bytes",
                        limitations=("not_runtime_or_allocator_memory",),
                    ),
                    physical_cache_blocks=unavailable(
                        "synthetic_reference", "token_granular_cache"
                    ),
                ),
                output=OutputEvidence(
                    output_token_ids=observation.output_token_ids,
                    baseline_token_ids=baseline,
                    token_identity=_fact(
                        output_identity,
                        EvidenceBasis.OBSERVED,
                        "synthetic_engine.separate_execution_comparison",
                    ),
                    correctness=_fact(
                        output_identity,
                        EvidenceBasis.OBSERVED,
                        "synthetic_baseline.deterministic_equivalence",
                        limitations=("not_model_quality",),
                    ),
                    finish_reason="length",
                ),
                terminal_state=TerminalState.COMPLETED,
                limitations=limitations,
                cache_before=self._snapshot(
                    observation.entry_count_before,
                    observation.logical_bytes_before,
                ),
                cache_after=self._snapshot(
                    observation.entry_count_after,
                    observation.logical_bytes_after,
                ),
            )
            if (
                observation.eviction_observed
                and len(request.expected_predecessors) == 1
            ):
                predecessor = prior_specs.get(request.expected_predecessors[0])
                if (
                    predecessor is not None
                    and predecessor.input_token_ids is not None
                    and request.input_token_ids is not None
                ):
                    config_material = (
                        f"token_trie:{self._max_entries}:{self._max_bytes}"
                    ).encode("ascii")
                    config_digest = (
                        "sha256:" + hashlib.sha256(config_material).hexdigest()
                    )
                    predecessor_identity = RequestCacheIdentity(
                        backend=self.backend,
                        model_id=self._model_key,
                        tokenizer_id="integer-tokenizer-v1",
                        model_artifact_digest=None,
                        cache_config_digest=config_digest,
                        namespace_id=predecessor.namespace_id,
                        input_token_ids=predecessor.input_token_ids,
                    )
                    current_identity = RequestCacheIdentity(
                        backend=self.backend,
                        model_id=self._model_key,
                        tokenizer_id="integer-tokenizer-v1",
                        model_artifact_digest=None,
                        cache_config_digest=config_digest,
                        namespace_id=request.namespace_id,
                        input_token_ids=request.input_token_ids,
                    )
                    record = replace(
                        record,
                        eviction_predecessor=EvictionPredecessorProof(
                            predecessor_request_id=predecessor.request_id,
                            predecessor=predecessor_identity,
                            current=current_identity,
                            reusable_prefix_tokens=min(
                                max(0, len(request.input_token_ids) - 1),
                                _common_prefix(
                                    predecessor.input_token_ids,
                                    request.input_token_ids,
                                ),
                            ),
                        ),
                    )
            records.append(classify_request(record))
            prior_specs[request.request_id] = request
        return records
