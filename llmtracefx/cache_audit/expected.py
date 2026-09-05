"""Independent cache-reuse reference models."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Hashable, Sequence
from dataclasses import dataclass


def longest_common_prefix(left: Sequence[int], right: Sequence[int]) -> int:
    """Return the exact token-ID common-prefix length."""

    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


@dataclass(frozen=True)
class ReuseExpectation:
    semantic_prefix_tokens: int
    policy_reusable_tokens: int
    reusable_blocks: int | None
    partial_block_tokens: int | None
    policy_required_prompt_tokens: int
    matched_entry_id: str | None = None
    match_kind: str = "miss"


@dataclass(frozen=True)
class _MLXEntry:
    entry_id: str
    model_key: Hashable
    namespace_id: str
    tokens: tuple[int, ...]
    nbytes: int
    cache_type: str
    trimmable: bool


class MLXCacheOracle:
    """Independent model of MLX-LM 0.31.3 ``LRUPromptCache`` semantics."""

    _ORDER = ("assistant", "user", "system")

    def __init__(self, *, max_entries: int = 10, max_bytes: int = (1 << 63)) -> None:
        if max_entries < 1 or max_bytes < 1:
            raise ValueError("cache limits must be positive")
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._entries: dict[str, _MLXEntry] = {}
        self._queues: dict[str, deque[str]] = {
            cache_type: deque() for cache_type in self._ORDER
        }
        self._nbytes = 0
        self.evicted_entry_ids: list[str] = []

    @property
    def nbytes(self) -> int:
        return self._nbytes

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    @property
    def entry_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))

    def _matching_entries(
        self, model_key: Hashable, namespace_id: str
    ) -> list[_MLXEntry]:
        return [
            entry
            for entry in self._entries.values()
            if entry.model_key == model_key and entry.namespace_id == namespace_id
        ]

    def lookup(
        self,
        model_key: Hashable,
        namespace_id: str,
        tokens: Sequence[int],
    ) -> ReuseExpectation:
        """Reproduce exact/shorter/longer trie selection without engine counters."""

        request = tuple(tokens)
        entries = self._matching_entries(model_key, namespace_id)
        if not entries:
            return ReuseExpectation(0, 0, None, None, len(request))

        exact = next((entry for entry in entries if entry.tokens == request), None)
        if exact is not None:
            return ReuseExpectation(
                len(request),
                len(request),
                None,
                None,
                0,
                exact.entry_id,
                "exact",
            )

        shorter = [
            entry
            for entry in entries
            if len(entry.tokens) < len(request)
            and tuple(request[: len(entry.tokens)]) == entry.tokens
        ]
        best_shorter = max(shorter, key=lambda item: len(item.tokens), default=None)
        short_length = len(best_shorter.tokens) if best_shorter is not None else 0

        longer = [
            (longest_common_prefix(entry.tokens, request), entry)
            for entry in entries
            if len(entry.tokens) > len(request)
        ]
        longer = [item for item in longer if item[0] > short_length]
        if longer:
            common, entry = min(
                longer,
                key=lambda item: (-item[0], len(item[1].tokens), item[1].entry_id),
            )
            if entry.trimmable:
                reusable = min(len(request) - 1, common)
                return ReuseExpectation(
                    common,
                    reusable,
                    None,
                    None,
                    len(request) - reusable,
                    entry.entry_id,
                    "longer_trimmed",
                )

        if best_shorter is not None:
            reusable = len(best_shorter.tokens)
            return ReuseExpectation(
                reusable,
                reusable,
                None,
                None,
                len(request) - reusable,
                best_shorter.entry_id,
                "shorter",
            )

        max_common, matched_entry = max(
            (
                (longest_common_prefix(entry.tokens, request), entry)
                for entry in entries
            ),
            key=lambda item: (item[0], -len(item[1].tokens), item[1].entry_id),
            default=(0, None),
        )
        return ReuseExpectation(
            max_common,
            0,
            None,
            None,
            len(request),
            None if matched_entry is None else matched_entry.entry_id,
            match_kind="non_trimmable" if max_common else "miss",
        )

    def insert(
        self,
        *,
        entry_id: str,
        model_key: Hashable,
        namespace_id: str,
        tokens: Sequence[int],
        nbytes: int,
        cache_type: str = "assistant",
        trimmable: bool = True,
    ) -> tuple[str, ...]:
        """Insert one entry and return entries evicted by the operation."""

        if cache_type not in self._queues:
            raise ValueError(f"unsupported MLX cache type: {cache_type}")
        if nbytes < 0:
            raise ValueError("cache entry bytes must be non-negative")
        before = len(self.evicted_entry_ids)

        old = self._entries.pop(entry_id, None)
        if old is not None:
            self._nbytes -= old.nbytes
            self._queues[old.cache_type].remove(entry_id)
        duplicate = next(
            (
                candidate
                for candidate in self._matching_entries(model_key, namespace_id)
                if candidate.tokens == tuple(tokens)
            ),
            None,
        )
        if duplicate is not None:
            self._remove(duplicate.entry_id, evicted=False)

        entry = _MLXEntry(
            entry_id=entry_id,
            model_key=model_key,
            namespace_id=namespace_id,
            tokens=tuple(tokens),
            nbytes=nbytes,
            cache_type=cache_type,
            trimmable=trimmable,
        )
        self._entries[entry_id] = entry
        self._queues[cache_type].append(entry_id)
        self._nbytes += nbytes

        if trimmable:
            prefixes = [
                candidate
                for candidate in self._matching_entries(model_key, namespace_id)
                if candidate.entry_id != entry_id
                and len(candidate.tokens) < len(entry.tokens)
                and entry.tokens[: len(candidate.tokens)] == candidate.tokens
            ]
            for candidate in prefixes:
                self._remove(candidate.entry_id, evicted=False)

        while len(self._entries) > self.max_entries or self._nbytes > self.max_bytes:
            self._remove(self._pop_candidate(), evicted=True)
        return tuple(self.evicted_entry_ids[before:])

    def _pop_candidate(self) -> str:
        index = 0
        while index + 1 < len(self._ORDER):
            left = self._queues[self._ORDER[index]]
            right = self._queues[self._ORDER[index + 1]]
            if left and len(left) >= len(right):
                return left[0]
            index += 1
        if not right:
            raise RuntimeError("MLX cache oracle has no eviction candidate")
        return right[0]

    def _remove(self, entry_id: str, *, evicted: bool) -> None:
        entry = self._entries.pop(entry_id)
        self._queues[entry.cache_type].remove(entry_id)
        self._nbytes -= entry.nbytes
        if evicted:
            self.evicted_entry_ids.append(entry_id)


@dataclass(frozen=True)
class VLLMReuseConfig:
    hash_block_size: int
    physical_block_sizes: tuple[int, ...] = ()
    fine_grained_hits: bool = False

    def __post_init__(self) -> None:
        if self.hash_block_size < 1:
            raise ValueError("hash_block_size must be positive")
        if any(
            size < 1 or size % self.hash_block_size
            for size in self.physical_block_sizes
        ):
            raise ValueError(
                "physical block sizes must be positive multiples of hash_block_size"
            )


def expected_vllm_reuse(
    cached_tokens: Sequence[int],
    request_tokens: Sequence[int],
    config: VLLMReuseConfig,
    *,
    identity_matches: bool = True,
) -> ReuseExpectation:
    """Derive vLLM token/block eligibility independently from hit counters."""

    common = (
        longest_common_prefix(cached_tokens, request_tokens) if identity_matches else 0
    )
    eligible_common = min(common, max(0, len(request_tokens) - 1))
    full_hash_units = eligible_common // config.hash_block_size
    reusable = full_hash_units * config.hash_block_size
    if not config.fine_grained_hits and config.physical_block_sizes:
        alignment = math.lcm(*config.physical_block_sizes)
        reusable = (reusable // alignment) * alignment
    return ReuseExpectation(
        semantic_prefix_tokens=common,
        policy_reusable_tokens=reusable,
        reusable_blocks=reusable // config.hash_block_size,
        partial_block_tokens=eligible_common % config.hash_block_size,
        policy_required_prompt_tokens=len(request_tokens) - reusable,
        match_kind="hit" if reusable else "miss",
    )
