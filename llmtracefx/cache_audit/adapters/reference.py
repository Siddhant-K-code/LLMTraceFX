"""Deterministic synthetic positive-control cache adapter."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from ..expected import MLXCacheOracle
from ..schema import (
    CacheStateSnapshot,
    EvidenceBasis,
    EvidenceFact,
    Limitation,
    MemoryEvidence,
    OutputEvidence,
    RequestEvidence,
    RequestSpec,
    ReuseEvidence,
    ScenarioKind,
    TerminalState,
    TimingEvidence,
    unavailable,
)
from ..verdicts import classify_request
from .base import CacheAuditCapability


def _fact(value: int | bool, basis: EvidenceBasis, source: str) -> EvidenceFact:
    return EvidenceFact(value=value, basis=basis, source=source)


class ReferenceCacheAdapter:
    """A transparent oracle-backed engine used only as synthetic evidence."""

    def __init__(
        self,
        *,
        max_entries: int = 32,
        max_bytes: int = 1 << 30,
        bytes_per_token: int = 64,
        model_key: str = "synthetic-tiny-model",
    ) -> None:
        self._oracle = MLXCacheOracle(
            max_entries=max_entries,
            max_bytes=max_bytes,
        )
        self._bytes_per_token = bytes_per_token
        self._model_key = model_key

    @property
    def backend(self) -> str:
        return "synthetic_reference"

    def capabilities(self) -> CacheAuditCapability:
        return CacheAuditCapability(
            backend=self.backend,
            supported=True,
            reasons=("synthetic_positive_control_only",),
            observable_facts=(
                "engine_cached_tokens",
                "prompt_tokens_processed",
                "logical_cache_bytes",
                "output_token_ids",
            ),
            unavailable_facts=(
                "client_ttft",
                "engine_queue_time",
                "runtime_allocator_memory",
                "billed_cost",
            ),
        )

    @staticmethod
    def _output(tokens: Sequence[int], count: int) -> tuple[int, ...]:
        encoded = ",".join(str(token) for token in tokens).encode("ascii")
        digest = hashlib.sha256(encoded).digest()
        return tuple(digest[index] for index in range(count))

    def run(self, requests: Sequence[RequestSpec]) -> list[RequestEvidence]:
        records: list[RequestEvidence] = []
        evicted: set[str] = set()
        for request in requests:
            if request.input_token_ids is None:
                raise ValueError("reference execution requires exact input token IDs")
            cache_before = CacheStateSnapshot(
                entry_count=_fact(
                    self._oracle.entry_count,
                    EvidenceBasis.OBSERVED,
                    "synthetic_reference.cache_entries",
                ),
                logical_bytes=_fact(
                    self._oracle.nbytes,
                    EvidenceBasis.OBSERVED,
                    "synthetic_reference.cache_bytes",
                ),
                valid_token_offsets=unavailable(
                    "synthetic_reference", "cache_offsets_not_exposed"
                ),
                cache_classes=EvidenceFact(
                    value=["SyntheticTokenCache"],
                    basis=EvidenceBasis.OBSERVED,
                    source="synthetic_reference.cache_type",
                ),
                complete=True,
            )
            expectation = self._oracle.lookup(
                self._model_key,
                request.namespace_id,
                request.input_token_ids,
            )
            output = self._output(request.input_token_ids, request.output_tokens)
            cache_tokens = request.input_token_ids + output
            cache_bytes = len(cache_tokens) * self._bytes_per_token
            newly_evicted = self._oracle.insert(
                entry_id=request.request_id,
                model_key=self._model_key,
                namespace_id=request.namespace_id,
                tokens=cache_tokens,
                nbytes=cache_bytes,
            )
            evicted.update(newly_evicted)
            request_was_evicted = request.scenario in {
                ScenarioKind.EVICTION_COUNT,
                ScenarioKind.EVICTION_BYTES,
            } and bool(set(request.expected_predecessors) & evicted)
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
            observed_prompt = expectation.policy_required_prompt_tokens
            record = RequestEvidence(
                spec=request,
                reuse=ReuseEvidence(
                    semantic_prefix_tokens=_fact(
                        expectation.semantic_prefix_tokens,
                        EvidenceBasis.INDEPENDENTLY_DERIVED,
                        "oracle.longest_common_prefix",
                    ),
                    policy_reusable_tokens=_fact(
                        expectation.policy_reusable_tokens,
                        EvidenceBasis.INDEPENDENTLY_DERIVED,
                        "oracle.mlx_lru_policy",
                    ),
                    reusable_blocks=unavailable(
                        "synthetic_reference", "token_granular_cache"
                    ),
                    partial_block_tokens=unavailable(
                        "synthetic_reference", "token_granular_cache"
                    ),
                    engine_cached_tokens=_fact(
                        expectation.policy_reusable_tokens,
                        EvidenceBasis.ENGINE_ATTESTED,
                        "synthetic_reference.cached_tokens",
                    ),
                    engine_cached_blocks=unavailable(
                        "synthetic_reference", "token_granular_cache"
                    ),
                    engine_created_tokens=_fact(
                        request.input_token_count - expectation.policy_reusable_tokens,
                        EvidenceBasis.ENGINE_ATTESTED,
                        "synthetic_reference.created_tokens",
                    ),
                    observed_prompt_tokens=_fact(
                        observed_prompt,
                        EvidenceBasis.OBSERVED,
                        "synthetic_reference.model_input_counter",
                    ),
                    policy_required_prompt_tokens=_fact(
                        expectation.policy_required_prompt_tokens,
                        EvidenceBasis.INDEPENDENTLY_DERIVED,
                        "oracle.mlx_lru_policy",
                    ),
                    unexpected_recomputed_tokens=_fact(
                        0,
                        EvidenceBasis.INDEPENDENTLY_DERIVED,
                        "oracle.prompt_work_delta",
                    ),
                    eviction_observed=_fact(
                        request_was_evicted,
                        EvidenceBasis.INDEPENDENTLY_DERIVED,
                        "oracle.controlled_eviction",
                    ),
                    preemption_observed=unavailable(
                        "synthetic_reference", "preemption_not_implemented"
                    ),
                ),
                timing=TimingEvidence(),
                memory=MemoryEvidence(
                    runtime_active_bytes=unavailable(
                        "synthetic_reference", "no_allocator"
                    ),
                    runtime_peak_bytes=unavailable(
                        "synthetic_reference", "no_allocator"
                    ),
                    allocator_cache_bytes=unavailable(
                        "synthetic_reference", "no_allocator"
                    ),
                    logical_cache_bytes=_fact(
                        self._oracle.nbytes,
                        EvidenceBasis.OBSERVED,
                        "synthetic_reference.cache_bytes",
                    ),
                    physical_cache_blocks=unavailable(
                        "synthetic_reference", "token_granular_cache"
                    ),
                ),
                output=OutputEvidence(
                    output_token_ids=output,
                    baseline_token_ids=output,
                    token_identity=_fact(
                        True,
                        EvidenceBasis.OBSERVED,
                        "synthetic_reference.output_tokens",
                    ),
                    correctness=_fact(
                        True,
                        EvidenceBasis.OBSERVED,
                        "synthetic_reference.baseline",
                    ),
                    finish_reason="length",
                ),
                terminal_state=TerminalState.COMPLETED,
                limitations=limitations,
                cache_before=cache_before,
                cache_after=CacheStateSnapshot(
                    entry_count=_fact(
                        self._oracle.entry_count,
                        EvidenceBasis.OBSERVED,
                        "synthetic_reference.cache_entries",
                    ),
                    logical_bytes=_fact(
                        self._oracle.nbytes,
                        EvidenceBasis.OBSERVED,
                        "synthetic_reference.cache_bytes",
                    ),
                    valid_token_offsets=unavailable(
                        "synthetic_reference", "cache_offsets_not_exposed"
                    ),
                    cache_classes=EvidenceFact(
                        value=["SyntheticTokenCache"],
                        basis=EvidenceBasis.OBSERVED,
                        source="synthetic_reference.cache_type",
                    ),
                    complete=True,
                ),
            )
            records.append(classify_request(record))
        return records
