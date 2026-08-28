"""Tests for deterministic prompt materialization/padding and hashing."""

from __future__ import annotations

import pytest

from llmtracefx.optimizer.workloads.catalog import (
    CODE_COMPLETION_PALINDROME,
    CONTEXT_FILLER_CORPUS,
    PROSE_REASONING_TRAIN_PROBLEM,
    STRUCTURED_JSON_PROFILE_EXTRACTION,
)
from llmtracefx.optimizer.workloads.evaluators import evaluate_code_completion
from llmtracefx.optimizer.workloads.materialize import (
    APPROX_CHARS_PER_TOKEN,
    materialize_prompt,
)
from llmtracefx.optimizer.workloads.schema import (
    CONTEXT_TIER_TARGET_TOKENS,
    CodeCompletionSpec,
    ContextTier,
    PaddingPlacement,
    ProseReasoningSpec,
    Workload,
    WorkloadCategory,
    WorkloadSchemaError,
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


# --- Code-completion padding placement: filler must go before the stub ----


def test_code_completion_workload_declares_before_stub_placement():
    assert (
        CODE_COMPLETION_PALINDROME.padding_placement
        == PaddingPlacement.BEFORE_CONTINUATION_STUB
    )
    assert CODE_COMPLETION_PALINDROME.continuation_stub


def test_prose_and_json_workloads_declare_append_placement():
    assert (
        PROSE_REASONING_TRAIN_PROBLEM.padding_placement
        == PaddingPlacement.APPEND_AFTER_BASE_PROMPT
    )
    assert (
        STRUCTURED_JSON_PROFILE_EXTRACTION.padding_placement
        == PaddingPlacement.APPEND_AFTER_BASE_PROMPT
    )


@pytest.mark.parametrize("tier", list(ContextTier))
def test_materialized_code_prompt_ends_with_exact_continuation_stub(tier):
    materialized = materialize_prompt(CODE_COMPLETION_PALINDROME, tier)
    assert materialized.text.endswith(CODE_COMPLETION_PALINDROME.continuation_stub)


@pytest.mark.parametrize("tier", [ContextTier.TIER_8K, ContextTier.TIER_16K])
def test_materialized_code_prompt_has_filler_before_the_stub(tier):
    materialized = materialize_prompt(CODE_COMPLETION_PALINDROME, tier)
    stub = CODE_COMPLETION_PALINDROME.continuation_stub
    stub_index = materialized.text.rindex(stub)
    assert materialized.filler_segments_used > 0
    first_filler_segment = CONTEXT_FILLER_CORPUS.format(index=0)
    filler_index = materialized.text.index(first_filler_segment)
    assert filler_index < stub_index


def test_materialized_code_prompt_keeps_base_task_fixed_across_tiers():
    small = materialize_prompt(CODE_COMPLETION_PALINDROME, ContextTier.TIER_2K)
    large = materialize_prompt(CODE_COMPLETION_PALINDROME, ContextTier.TIER_16K)
    stub = CODE_COMPLETION_PALINDROME.continuation_stub
    prefix = CODE_COMPLETION_PALINDROME.base_prompt[: -len(stub)]
    assert small.text.startswith(prefix)
    assert large.text.startswith(prefix)
    assert small.text.endswith(stub)
    assert large.text.endswith(stub)


def test_materialize_code_prompt_is_deterministic_across_calls():
    first = materialize_prompt(CODE_COMPLETION_PALINDROME, ContextTier.TIER_8K)
    second = materialize_prompt(CODE_COMPLETION_PALINDROME, ContextTier.TIER_8K)
    assert first == second
    assert first.prompt_hash == second.prompt_hash


@pytest.mark.parametrize("tier", list(ContextTier))
def test_materialized_code_prompt_completion_still_evaluates_successfully(tier):
    materialized = materialize_prompt(CODE_COMPLETION_PALINDROME, tier)
    # The prompt still ends with the exact stub regardless of tier/filler,
    # so a correct completion appended after it is still gradeable.
    assert materialized.text.endswith(CODE_COMPLETION_PALINDROME.continuation_stub)

    candidate_completion = (
        CODE_COMPLETION_PALINDROME.continuation_stub
        + "    cleaned = text.lower().replace(' ', '')\n"
        + "    return cleaned == cleaned[::-1]\n"
    )
    outcome = evaluate_code_completion(
        CODE_COMPLETION_PALINDROME.spec, candidate_completion
    )
    assert outcome.success is True


def test_workload_rejects_continuation_stub_not_a_suffix_of_base_prompt():
    with pytest.raises(WorkloadSchemaError, match="continuation_stub"):
        Workload(
            workload_id="bad-stub",
            version="1",
            category=WorkloadCategory.CODE_COMPLETION,
            title="bad",
            base_prompt="def f(): pass\n",
            spec=CodeCompletionSpec(
                function_stub="def f(): pass\n",
                test_code="assert True",
                entry_point="f",
            ),
            continuation_stub="not a suffix",
        )
