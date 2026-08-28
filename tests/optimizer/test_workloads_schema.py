"""Tests for the workload schema (categories, tiers, Workload records)."""

from __future__ import annotations

import pytest

from llmtracefx.optimizer.workloads.schema import (
    CodeCompletionSpec,
    ContextTier,
    ProseReasoningSpec,
    StructuredJSONSpec,
    Workload,
    WorkloadCategory,
    WorkloadSchemaError,
)


def test_context_tier_targets_are_ordered_and_distinct():
    from llmtracefx.optimizer.workloads.schema import CONTEXT_TIER_TARGET_TOKENS

    values = [CONTEXT_TIER_TARGET_TOKENS[tier] for tier in ContextTier]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


def test_workload_rejects_empty_workload_id():
    with pytest.raises(WorkloadSchemaError, match="workload_id"):
        Workload(
            workload_id="",
            version="1",
            category=WorkloadCategory.PROSE_REASONING,
            title="t",
            base_prompt="p",
            spec=ProseReasoningSpec(expected_answer_pattern="x"),
        )


def test_workload_rejects_empty_base_prompt():
    with pytest.raises(WorkloadSchemaError, match="base_prompt"):
        Workload(
            workload_id="w",
            version="1",
            category=WorkloadCategory.PROSE_REASONING,
            title="t",
            base_prompt="",
            spec=ProseReasoningSpec(expected_answer_pattern="x"),
        )


def test_workload_rejects_mismatched_spec_type():
    with pytest.raises(WorkloadSchemaError, match="expected ProseReasoningSpec"):
        Workload(
            workload_id="w",
            version="1",
            category=WorkloadCategory.PROSE_REASONING,
            title="t",
            base_prompt="p",
            spec=StructuredJSONSpec(required_fields=(), field_types={}),
        )


def test_workload_round_trips_through_dict_code_completion():
    workload = Workload(
        workload_id="w",
        version="1",
        category=WorkloadCategory.CODE_COMPLETION,
        title="t",
        base_prompt="p",
        spec=CodeCompletionSpec(
            function_stub="def f(): pass",
            test_code="assert True",
            entry_point="f",
        ),
    )
    restored = Workload.from_dict(workload.to_dict())
    assert restored == workload


def test_workload_round_trips_through_dict_structured_json():
    workload = Workload(
        workload_id="w",
        version="1",
        category=WorkloadCategory.STRUCTURED_JSON,
        title="t",
        base_prompt="p",
        spec=StructuredJSONSpec(
            required_fields=("a", "b"), field_types={"a": "str", "b": "int"}
        ),
    )
    restored = Workload.from_dict(workload.to_dict())
    assert restored == workload


def test_workload_round_trips_through_dict_prose_reasoning():
    workload = Workload(
        workload_id="w",
        version="1",
        category=WorkloadCategory.PROSE_REASONING,
        title="t",
        base_prompt="p",
        spec=ProseReasoningSpec(expected_answer_pattern=r"\b3\b"),
    )
    restored = Workload.from_dict(workload.to_dict())
    assert restored == workload


def test_workload_from_dict_rejects_missing_field():
    with pytest.raises(WorkloadSchemaError, match="missing required field"):
        Workload.from_dict({"workload_id": "w"})


def test_code_completion_spec_from_dict_rejects_missing_field():
    with pytest.raises(WorkloadSchemaError, match="missing required field"):
        CodeCompletionSpec.from_dict({"function_stub": "x"})
