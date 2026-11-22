"""Fixed workload and independent block-math oracle for the vLLM KV-truth protocol.

This module is the exact, immutable ``qwen3-8b-vllm-kv-truth-v1`` workload
described by the CloudRift vLLM KV-cache truth audit plan: one synthetic,
tokenizer-valid base array, a fixed nested sequence of ten probes with
preregistered expected reusable-token/block counts, and a fixed eviction
lane. Every array is generated once, deterministically, from a fixed seed and
a fixed "ordinary" (non-special) token-ID sub-range of the public Qwen3
tokenizer vocabulary; nothing here depends on a live tokenizer or model, so it
is fully importable and testable offline.

The independent oracle is exactly the formula the plan specifies:

    reusable = floor(min(LCP(cached, request), len(request) - 1) / 16) * 16

which is already implemented, generically, by
``llmtracefx.cache_audit.expected.expected_vllm_reuse``. This module only
supplies the fixed inputs (arrays, block size, cached references) and
verifies the table's arithmetic follows from that one shared formula, so the
oracle used here can never silently drift from the oracle used by the rest
of the cache-audit suite.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from llmtracefx.cache_audit.expected import (
    ReuseExpectation,
    VLLMReuseConfig,
    expected_vllm_reuse,
)

#: The public Qwen3-8B tokenizer's vocabulary size. Documented for context
#: only; this module never imports a tokenizer.
QWEN3_8B_VOCAB_SIZE = 151_936

#: A fixed, conservative sub-range of ordinary (non-special, non-control)
#: token IDs. Qwen3 tokenizers append special/control tokens near the top of
#: the vocabulary (at and above roughly 151_643); this range stays far below
#: that boundary so every array below is composed only of ordinary token
#: IDs, as the plan requires, without depending on a live tokenizer to prove
#: it offline.
ORDINARY_TOKEN_ID_LOW = 1_000
ORDINARY_TOKEN_ID_HIGH_EXCLUSIVE = 140_000

BLOCK_SIZE = 16
PREFIX_MATCH_UNIT = 16

#: The protocol's fixed override, used only to size the eviction lane's
#: cache-pressure arithmetic below; the KV-truth runner pins the same value
#: for the real engine (see ``kv_truth_runner.NUM_GPU_BLOCKS_OVERRIDE``).
NUM_GPU_BLOCKS_OVERRIDE = 96
#: vLLM's block pool reserves one null block (block id 0) that is never a
#: real cacheable block, so the usable capacity is one less than the raw
#: override.
NULL_BLOCK_RESERVE = 1
USABLE_BLOCK_CAPACITY = NUM_GPU_BLOCKS_OVERRIDE - NULL_BLOCK_RESERVE

_WORKLOAD_SEED = 20260906
_TOKEN_POOL_SIZE = 2_000

_REUSE_CONFIG = VLLMReuseConfig(hash_block_size=BLOCK_SIZE)


def _draw_token_pool() -> tuple[int, ...]:
    """Draw a fixed, distinct pool of ordinary token IDs.

    ``random.Random(seed).sample`` without replacement guarantees every
    element is distinct, so every named slice below is disjoint from every
    other named slice: no probe accidentally shares a token with an array it
    is supposed to be independent from.
    """

    rng = random.Random(_WORKLOAD_SEED)
    return tuple(
        rng.sample(
            range(ORDINARY_TOKEN_ID_LOW, ORDINARY_TOKEN_ID_HIGH_EXCLUSIVE),
            _TOKEN_POOL_SIZE,
        )
    )


_POOL = _draw_token_pool()

# Named, disjoint slices of the fixed pool. Every offset below is a literal
# constant (not computed from list lengths at call time) so the layout is
# stable and auditable by inspection.
BASE_ARRAY: tuple[int, ...] = _POOL[0:256]
_TOKEN_A = _POOL[256]
_EXTENSION: tuple[int, ...] = _POOL[257:321]
_WITHIN_BLOCK_MUTATION_TOKEN = _POOL[321]
_BLOCK_BOUNDARY_MUTATION_TOKEN = _POOL[322]
_SUFFIX_MUTATION_TOKEN = _POOL[323]
_DISJOINT_ARRAY: tuple[int, ...] = _POOL[324:581]
_EVICTION_FILLERS: tuple[tuple[int, ...], ...] = tuple(
    _POOL[581 + index * 257 : 581 + (index + 1) * 257] for index in range(5)
)

#: The cold-seed probe array: the first 256 tokens of ``BASE_ARRAY`` plus one
#: additional distinct token, for 257 tokens total.
SEED_ARRAY: tuple[int, ...] = (*BASE_ARRAY, _TOKEN_A)
#: The seed extended by 64 more distinct tokens, for 321 tokens total.
LONGER_ARRAY: tuple[int, ...] = (*SEED_ARRAY, *_EXTENSION)

assert len(BASE_ARRAY) == 256
assert len(SEED_ARRAY) == 257
assert len(LONGER_ARRAY) == 321
assert len(_DISJOINT_ARRAY) == 257
assert all(len(filler) == 257 for filler in _EVICTION_FILLERS)
assert len(
    {
        *BASE_ARRAY,
        _TOKEN_A,
        *_EXTENSION,
        _WITHIN_BLOCK_MUTATION_TOKEN,
        _BLOCK_BOUNDARY_MUTATION_TOKEN,
        _SUFFIX_MUTATION_TOKEN,
        *_DISJOINT_ARRAY,
        *(t for filler in _EVICTION_FILLERS for t in filler),
    }
) == (
    len(BASE_ARRAY)
    + 1
    + len(_EXTENSION)
    + 3
    + len(_DISJOINT_ARRAY)
    + sum(len(filler) for filler in _EVICTION_FILLERS)
), "workload token pool slices are not mutually disjoint"


def _mutate(tokens: tuple[int, ...], index: int, replacement: int) -> tuple[int, ...]:
    if replacement in tokens:
        raise ValueError("mutation replacement token must not already appear")
    mutated = list(tokens)
    mutated[index] = replacement
    return tuple(mutated)


@dataclass(frozen=True)
class KVTruthProbe:
    """One row of the fixed nested probe sequence, with its independent verdict."""

    name: str
    scenario: str
    cached_reference: tuple[int, ...]
    request_tokens: tuple[int, ...]
    identity_matches: bool
    expectation: ReuseExpectation

    @property
    def expected_reusable_tokens(self) -> int:
        return self.expectation.policy_reusable_tokens

    @property
    def expected_reusable_blocks(self) -> int:
        assert self.expectation.reusable_blocks is not None
        return self.expectation.reusable_blocks


def _probe(
    name: str,
    scenario: str,
    cached_reference: tuple[int, ...],
    request_tokens: tuple[int, ...],
    *,
    identity_matches: bool = True,
) -> KVTruthProbe:
    expectation = expected_vllm_reuse(
        cached_reference,
        request_tokens,
        _REUSE_CONFIG,
        identity_matches=identity_matches,
    )
    return KVTruthProbe(
        name=name,
        scenario=scenario,
        cached_reference=cached_reference,
        request_tokens=request_tokens,
        identity_matches=identity_matches,
        expectation=expectation,
    )


#: The fixed, preregistered, order-significant nested probe sequence.
#: Each probe after "cold_seed" is compared against ``SEED_ARRAY`` (the state
#: the cache holds once the seed has been processed) except "duplicate",
#: which repeats ``LONGER_ARRAY`` verbatim and is therefore compared against
#: itself, matching what vLLM would actually hold cached after the "longer
#: prefix" probe ran.
NESTED_PROBES: tuple[KVTruthProbe, ...] = (
    _probe("cold_seed", "cold", SEED_ARRAY, SEED_ARRAY, identity_matches=False),
    _probe("exact_full_hit", "identical_prefix", SEED_ARRAY, SEED_ARRAY),
    _probe("shorter_prefix", "identical_prefix", SEED_ARRAY, SEED_ARRAY[:129]),
    _probe("longer_prefix", "identical_prefix", SEED_ARRAY, LONGER_ARRAY),
    _probe(
        "within_block_mutation",
        "within_block_mutation",
        SEED_ARRAY,
        _mutate(SEED_ARRAY, 31, _WITHIN_BLOCK_MUTATION_TOKEN),
    ),
    _probe(
        "block_boundary_mutation",
        "block_boundary_mutation",
        SEED_ARRAY,
        _mutate(SEED_ARRAY, 32, _BLOCK_BOUNDARY_MUTATION_TOKEN),
    ),
    _probe(
        "same_length_different_ids",
        "same_length_different_ids",
        SEED_ARRAY,
        _DISJOINT_ARRAY,
    ),
    _probe(
        "suffix_only_change",
        "suffix_change",
        SEED_ARRAY,
        _mutate(SEED_ARRAY, 256, _SUFFIX_MUTATION_TOKEN),
    ),
    _probe("duplicate_request", "duplicate", LONGER_ARRAY, LONGER_ARRAY),
    _probe(
        "salt_isolation",
        "namespace_isolation",
        SEED_ARRAY,
        SEED_ARRAY,
        identity_matches=False,
    ),
)

#: cold_seed has no meaningful "cache miss" comparison basis before the reset
#: it follows, so it is modeled with ``identity_matches=False`` (nothing is
#: actually resident yet); its expectation must be exactly 0/0.
assert NESTED_PROBES[0].expected_reusable_tokens == 0
assert NESTED_PROBES[0].expected_reusable_blocks == 0

#: The fixed, preregistered eviction-lane request sequence: it starts after
#: an ``AllBlocksCleared`` reset, runs the seed once, then five fixed
#: disjoint 257-token fillers, then repeats the seed as the final probe.
EVICTION_LANE_REQUESTS: tuple[tuple[int, ...], ...] = (
    SEED_ARRAY,
    *_EVICTION_FILLERS,
    SEED_ARRAY,
)

#: Per the plan: each 257-token request in the eviction lane caches
#: ``floor((257 - 1) / 16) * 16 == 256`` tokens, i.e. 16 blocks. Six such
#: sequences (the seed plus five fillers) is 96 blocks, which exceeds the
#: usable capacity of the fixed 96-block pool once the null-block reserve is
#: accounted for, so eviction is a structural certainty rather than an
#: incidental outcome of run-time capacity arithmetic.
CACHEABLE_BLOCKS_PER_EVICTION_LANE_REQUEST = (len(SEED_ARRAY) - 1) // BLOCK_SIZE
TOTAL_CACHEABLE_BLOCKS_BEFORE_FINAL_PROBE = (
    CACHEABLE_BLOCKS_PER_EVICTION_LANE_REQUEST * 6
)
assert CACHEABLE_BLOCKS_PER_EVICTION_LANE_REQUEST == 16
assert TOTAL_CACHEABLE_BLOCKS_BEFORE_FINAL_PROBE == 96
assert TOTAL_CACHEABLE_BLOCKS_BEFORE_FINAL_PROBE > USABLE_BLOCK_CAPACITY


def probe_by_name(name: str) -> KVTruthProbe:
    for probe in NESTED_PROBES:
        if probe.name == name:
            return probe
    raise KeyError(f"unknown KV-truth probe: {name!r}")
