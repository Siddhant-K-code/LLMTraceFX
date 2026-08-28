"""Tests for deterministic workload evaluators."""

from __future__ import annotations

from llmtracefx.optimizer.workloads.catalog import (
    CODE_COMPLETION_PALINDROME,
    PROSE_REASONING_TRAIN_PROBLEM,
    STRUCTURED_JSON_PROFILE_EXTRACTION,
)
from llmtracefx.optimizer.workloads.evaluators import (
    evaluate_code_completion,
    evaluate_prose_reasoning,
    evaluate_structured_json,
    evaluate_workload,
)

# --- Code completion -----------------------------------------------------

_CORRECT_PALINDROME_COMPLETION = (
    "def is_palindrome(text: str) -> bool:\n"
    '    """Return True if text is a palindrome, ignoring case and spaces."""\n'
    "    cleaned = text.lower().replace(' ', '')\n"
    "    return cleaned == cleaned[::-1]\n"
)

_WRONG_PALINDROME_COMPLETION = (
    "def is_palindrome(text: str) -> bool:\n"
    '    """Return True if text is a palindrome, ignoring case and spaces."""\n'
    "    return False\n"
)


def test_evaluate_code_completion_passes_correct_completion():
    outcome = evaluate_code_completion(
        CODE_COMPLETION_PALINDROME.spec, _CORRECT_PALINDROME_COMPLETION
    )
    assert outcome.success is True
    assert outcome.quality_score == 1.0
    assert outcome.quality_metric == "unit_test_pass"
    assert outcome.notes is None


def test_evaluate_code_completion_fails_wrong_completion():
    outcome = evaluate_code_completion(
        CODE_COMPLETION_PALINDROME.spec, _WRONG_PALINDROME_COMPLETION
    )
    assert outcome.success is False
    assert outcome.quality_score == 0.0
    assert outcome.notes


def test_evaluate_code_completion_strips_markdown_fence():
    fenced = f"```python\n{_CORRECT_PALINDROME_COMPLETION}```"
    outcome = evaluate_code_completion(CODE_COMPLETION_PALINDROME.spec, fenced)
    assert outcome.success is True


def test_evaluate_code_completion_fails_on_syntax_error():
    outcome = evaluate_code_completion(
        CODE_COMPLETION_PALINDROME.spec, "def is_palindrome(text: str -> bool:\n"
    )
    assert outcome.success is False
    assert outcome.quality_score == 0.0


def test_evaluate_code_completion_times_out_on_infinite_loop():
    infinite = "def is_palindrome(text: str) -> bool:\n    while True:\n        pass\n"
    outcome = evaluate_code_completion(
        CODE_COMPLETION_PALINDROME.spec, infinite, timeout_seconds=0.5
    )
    assert outcome.success is False
    assert "timed out" in outcome.notes


# --- Structured JSON -------------------------------------------------------


def test_evaluate_structured_json_passes_exact_fields():
    response = '{"name": "Priya Nakamura", "age": 34, "is_active": true}'
    outcome = evaluate_structured_json(
        STRUCTURED_JSON_PROFILE_EXTRACTION.spec, response
    )
    assert outcome.success is True
    assert outcome.quality_score == 1.0


def test_evaluate_structured_json_tolerates_surrounding_prose():
    response = 'Here you go:\n{"name": "Priya", "age": 34, "is_active": true}\nDone.'
    outcome = evaluate_structured_json(
        STRUCTURED_JSON_PROFILE_EXTRACTION.spec, response
    )
    assert outcome.success is True


def test_evaluate_structured_json_reports_missing_field():
    response = '{"name": "Priya", "age": 34}'
    outcome = evaluate_structured_json(
        STRUCTURED_JSON_PROFILE_EXTRACTION.spec, response
    )
    assert outcome.success is False
    assert outcome.quality_score == 2 / 3
    assert "is_active" in outcome.notes


def test_evaluate_structured_json_rejects_wrong_type():
    response = '{"name": "Priya", "age": "34", "is_active": true}'
    outcome = evaluate_structured_json(
        STRUCTURED_JSON_PROFILE_EXTRACTION.spec, response
    )
    assert outcome.success is False
    assert "age" in outcome.notes


def test_evaluate_structured_json_rejects_bool_as_int():
    response = '{"name": "Priya", "age": true, "is_active": true}'
    outcome = evaluate_structured_json(
        STRUCTURED_JSON_PROFILE_EXTRACTION.spec, response
    )
    assert outcome.success is False


def test_evaluate_structured_json_handles_unparseable_response():
    outcome = evaluate_structured_json(
        STRUCTURED_JSON_PROFILE_EXTRACTION.spec, "not json at all"
    )
    assert outcome.success is False
    assert outcome.quality_score == 0.0


def test_evaluate_structured_json_rejects_non_object_root():
    outcome = evaluate_structured_json(
        STRUCTURED_JSON_PROFILE_EXTRACTION.spec, "[1, 2]"
    )
    assert outcome.success is False


# --- Prose reasoning --------------------------------------------------------


def test_evaluate_prose_reasoning_passes_expected_answer():
    outcome = evaluate_prose_reasoning(
        PROSE_REASONING_TRAIN_PROBLEM.spec,
        "3 hours, since they close the gap at 70 mph.",
    )
    assert outcome.success is True
    assert outcome.quality_score == 1.0


def test_evaluate_prose_reasoning_fails_wrong_answer():
    outcome = evaluate_prose_reasoning(PROSE_REASONING_TRAIN_PROBLEM.spec, "5 hours.")
    assert outcome.success is False
    assert outcome.quality_score == 0.0
    assert outcome.notes


# --- Dispatch ---------------------------------------------------------------


def test_evaluate_workload_dispatches_by_category():
    outcome = evaluate_workload(
        PROSE_REASONING_TRAIN_PROBLEM, "3 hours until they meet."
    )
    assert outcome.quality_metric == "exact_answer_pattern_match"
