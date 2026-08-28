"""The pinned, redistributable workload catalog.

All prompt text and evaluator fixtures here are authored for this
project; nothing is copied from third-party or copyrighted sources.
Keeping the catalog small keeps it easy to review, hash, and
redistribute alongside experiment evidence.
"""

from __future__ import annotations

from .schema import (
    CodeCompletionSpec,
    ProseReasoningSpec,
    StructuredJSONSpec,
    Workload,
    WorkloadCategory,
)

_PALINDROME_FUNCTION_STUB = (
    "def is_palindrome(text: str) -> bool:\n"
    '    """Return True if text is a palindrome, ignoring case and spaces."""\n'
)

CODE_COMPLETION_PALINDROME = Workload(
    workload_id="code-completion-palindrome-check",
    version="1",
    category=WorkloadCategory.CODE_COMPLETION,
    title="Complete a palindrome-check function",
    base_prompt=(
        "Complete the following Python function so it returns True if "
        "`text` reads the same forwards and backwards after lowercasing "
        "and removing spaces, and False otherwise. Respond with only the "
        "completed function body as valid Python code, no explanation.\n\n"
        + _PALINDROME_FUNCTION_STUB
    ),
    spec=CodeCompletionSpec(
        function_stub=_PALINDROME_FUNCTION_STUB,
        test_code=(
            "assert is_palindrome('Racecar') is True\n"
            "assert is_palindrome('was it a car or a cat i saw') is True\n"
            "assert is_palindrome('hello') is False\n"
            "assert is_palindrome('') is True\n"
        ),
        entry_point="is_palindrome",
    ),
    continuation_stub=_PALINDROME_FUNCTION_STUB,
)

STRUCTURED_JSON_PROFILE_EXTRACTION = Workload(
    workload_id="structured-json-profile-extraction",
    version="1",
    category=WorkloadCategory.STRUCTURED_JSON,
    title="Extract a structured profile as JSON",
    base_prompt=(
        "Read the profile below and respond with a single JSON object "
        "(no surrounding text, no markdown fences) with exactly these "
        'keys: "name" (string), "age" (integer), "is_active" '
        "(boolean).\n\n"
        "Profile: Priya Nakamura is a 34 year old volunteer coordinator "
        "who is currently active with the downtown reading program.\n"
    ),
    spec=StructuredJSONSpec(
        required_fields=("name", "age", "is_active"),
        field_types={"name": "str", "age": "int", "is_active": "bool"},
    ),
)

PROSE_REASONING_TRAIN_PROBLEM = Workload(
    workload_id="prose-reasoning-two-train-gap",
    version="1",
    category=WorkloadCategory.PROSE_REASONING,
    title="Deterministic arithmetic word problem",
    base_prompt=(
        "Two trains start 210 miles apart and travel toward each other on "
        "the same track. One train travels at 40 miles per hour and the "
        "other at 30 miles per hour. They start at the same time. How "
        "many hours until they meet? Respond with the final number of "
        "hours as a single number, followed by a one sentence "
        "explanation.\n"
    ),
    spec=ProseReasoningSpec(expected_answer_pattern=r"\b3(\.0+)?\b"),
)

WORKLOADS: tuple[Workload, ...] = (
    CODE_COMPLETION_PALINDROME,
    STRUCTURED_JSON_PROFILE_EXTRACTION,
    PROSE_REASONING_TRAIN_PROBLEM,
)

#: Deterministic filler corpus used only to pad prompts toward a target
#: context length (see ``materialize.py``). Self-authored, synthetic,
#: content-free technical filler -- never derived from copyrighted text.
CONTEXT_FILLER_CORPUS = (
    "Context filler segment {index:04d}: this sentence exists only to "
    "occupy context length for a reproducible benchmarking workload and "
    "carries no information relevant to the task above."
)


def workload_by_id(workload_id: str) -> Workload:
    for workload in WORKLOADS:
        if workload.workload_id == workload_id:
            return workload
    raise KeyError(f"unknown workload_id: {workload_id!r}")
