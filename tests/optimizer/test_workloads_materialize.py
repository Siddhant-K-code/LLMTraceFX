"""Tests for deterministic prompt materialization/padding and hashing."""

from __future__ import annotations

from llmtracefx.optimizer.workloads.catalog import PROSE_REASONING_TRAIN_PROBLEM
from llmtracefx.optimizer.workloads.materialize import (
    APPROX_CHARS_PER_TOKEN,
    materialize_prompt,
)
from llmtracefx.optimizer.workloads.schema import (
    CONTEXT_TIER_TARGET_TOKENS,
    ContextTier,
    ProseReasoningSpec,
    Workload,
    WorkloadCategory,
)


def test_materialize_prompt_is_deterministic_across_calls():
    first = materialize_prompt(PROSE_REASONING_TRAIN_PROBLEM, ContextTier.TIER_8K)
    second = materialize_prompt(PROSE_REASONING_TRAIN_PROBLEM, ContextTier.TIER_8K)
    assert first == second
    assert first.prompt_hash == second.prompt_hash


def test_materialize_prompt_preserves_base_prompt_verbatim():
    materialized = materialize_prompt(
        PROSE_REASONING_TRAIN_PROBLEM, ContextTier.TIER_2K
    )
    assert PROSE_REASONING_TRAIN_PROBLEM.base_prompt in materialized.text


def test_materialize_prompt_pads_toward_target_when_base_is_small():
    materialized = materialize_prompt(
        PROSE_REASONING_TRAIN_PROBLEM, ContextTier.TIER_16K
    )
    target_chars = (
        CONTEXT_TIER_TARGET_TOKENS[ContextTier.TIER_16K] * APPROX_CHARS_PER_TOKEN
    )
    assert materialized.filler_segments_used > 0
    assert len(materialized.text) >= target_chars


def test_materialize_prompt_never_pads_when_base_already_meets_target():
    huge_prompt = Workload(
        workload_id="huge",
        version="1",
        category=WorkloadCategory.PROSE_REASONING,
        title="huge",
        base_prompt="x" * 100_000,
        spec=ProseReasoningSpec(expected_answer_pattern="x"),
    )
    materialized = materialize_prompt(huge_prompt, ContextTier.TIER_2K)
    assert materialized.filler_segments_used == 0
    assert materialized.text == huge_prompt.base_prompt


def test_materialize_prompt_larger_tiers_use_more_filler():
    small = materialize_prompt(PROSE_REASONING_TRAIN_PROBLEM, ContextTier.TIER_2K)
    large = materialize_prompt(PROSE_REASONING_TRAIN_PROBLEM, ContextTier.TIER_16K)
    assert large.filler_segments_used > small.filler_segments_used
    assert len(large.text) > len(small.text)


def test_materialize_prompt_hash_changes_with_tier():
    tier_2k = materialize_prompt(PROSE_REASONING_TRAIN_PROBLEM, ContextTier.TIER_2K)
    tier_8k = materialize_prompt(PROSE_REASONING_TRAIN_PROBLEM, ContextTier.TIER_8K)
    assert tier_2k.prompt_hash != tier_8k.prompt_hash


def test_materialize_prompt_hash_matches_sha256_of_text():
    import hashlib

    materialized = materialize_prompt(
        PROSE_REASONING_TRAIN_PROBLEM, ContextTier.TIER_2K
    )
    expected = f"sha256:{hashlib.sha256(materialized.text.encode('utf-8')).hexdigest()}"
    assert materialized.prompt_hash == expected


def test_materialize_prompt_records_planning_metadata_not_measured_claim():
    materialized = materialize_prompt(
        PROSE_REASONING_TRAIN_PROBLEM, ContextTier.TIER_2K
    )
    payload = materialized.to_dict()
    assert payload["approx_chars_per_token"] == APPROX_CHARS_PER_TOKEN
    assert (
        payload["target_context_tokens"]
        == CONTEXT_TIER_TARGET_TOKENS[ContextTier.TIER_2K]
    )
    # No field claims this is a measured token count.
    assert "measured_tokens" not in payload
