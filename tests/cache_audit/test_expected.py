from __future__ import annotations

import random

import pytest

from llmtracefx.cache_audit.expected import (
    MLXCacheOracle,
    VLLMReuseConfig,
    expected_vllm_reuse,
    longest_common_prefix,
)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ((), (), 0),
        ((1, 2), (1, 2), 2),
        ((1, 2), (1, 3), 1),
        ((1, 2), (9, 2), 0),
        ((1, 2), (1, 2, 3), 2),
    ],
)
def test_longest_common_prefix(
    left: tuple[int, ...], right: tuple[int, ...], expected: int
) -> None:
    assert longest_common_prefix(left, right) == expected


def test_mlx_longer_cache_preserves_sampling_token() -> None:
    oracle = MLXCacheOracle()
    oracle.insert(
        entry_id="prior",
        model_key="model",
        namespace_id="a",
        tokens=(1, 2, 3, 4, 5),
        nbytes=100,
    )
    result = oracle.lookup("model", "a", (1, 2, 3, 4))
    assert result.semantic_prefix_tokens == 4
    assert result.policy_reusable_tokens == 3
    assert result.policy_required_prompt_tokens == 1
    assert result.match_kind == "longer_trimmed"


def test_mlx_namespace_isolation() -> None:
    oracle = MLXCacheOracle()
    oracle.insert(
        entry_id="prior",
        model_key="model",
        namespace_id="tenant-a",
        tokens=(1, 2, 3),
        nbytes=10,
    )
    assert oracle.lookup("model", "tenant-b", (1, 2)).policy_reusable_tokens == 0


def test_mlx_count_and_byte_eviction() -> None:
    oracle = MLXCacheOracle(max_entries=2, max_bytes=25)
    oracle.insert(
        entry_id="a",
        model_key="m",
        namespace_id="n",
        tokens=(1,),
        nbytes=10,
    )
    oracle.insert(
        entry_id="b",
        model_key="m",
        namespace_id="n",
        tokens=(2,),
        nbytes=10,
    )
    evicted = oracle.insert(
        entry_id="c",
        model_key="m",
        namespace_id="n",
        tokens=(3,),
        nbytes=10,
    )
    assert evicted == ("a",)
    assert oracle.entry_ids == ("b", "c")
    assert oracle.nbytes == 20


def test_vllm_block_floor_and_scheduler_alignment() -> None:
    request = tuple(range(30))
    cached = request[:23] + (999,)
    result = expected_vllm_reuse(
        cached,
        request,
        VLLMReuseConfig(
            hash_block_size=4,
            physical_block_sizes=(8, 16),
            fine_grained_hits=False,
        ),
    )
    assert result.semantic_prefix_tokens == 23
    assert result.policy_reusable_tokens == 16
    assert result.reusable_blocks == 4
    assert result.partial_block_tokens == 3


def test_vllm_identity_mismatch_forces_miss() -> None:
    result = expected_vllm_reuse(
        (1, 2, 3, 4),
        (1, 2, 3, 4),
        VLLMReuseConfig(hash_block_size=2),
        identity_matches=False,
    )
    assert result.semantic_prefix_tokens == 0
    assert result.policy_reusable_tokens == 0


def test_vllm_identical_prompt_keeps_one_token_for_logits() -> None:
    result = expected_vllm_reuse(
        tuple(range(8)),
        tuple(range(8)),
        VLLMReuseConfig(hash_block_size=4),
    )
    assert result.semantic_prefix_tokens == 8
    assert result.policy_reusable_tokens == 4
    assert result.policy_required_prompt_tokens == 4


def test_generated_mutations_stop_at_the_exact_position() -> None:
    randomizer = random.Random(20260905)
    for length in range(2, 40):
        tokens = tuple(randomizer.randrange(1, 10_000) for _ in range(length))
        for position in range(length):
            replacement = tokens[position]
            while replacement == tokens[position]:
                replacement = randomizer.randrange(1, 10_000)
            mutated = tokens[:position] + (replacement,) + tokens[position + 1 :]
            assert longest_common_prefix(tokens, mutated) == position
