"""Deterministic evaluators for the pinned workload catalog.

Every evaluator here is deterministic (no sampling, no LLM-as-judge) and
returns the canonical ``llmtracefx.optimizer.schema.OutcomeInfo``. None
of them overstate quality: ``quality_score``/``success`` reflect only
what was explicitly checked (exact field/type matches, a fixed test
suite passing, or a fixed regex matching), and ``notes`` is bounded and
factual rather than a free-form judgment.

The code-completion evaluator executes model-generated Python in a
subprocess (never with ``shell=True`` and never via string
interpolation into a shell command) to run a fixed, project-authored
test suite against the candidate completion. This is inherent to
executable-test evaluation (the same approach used by benchmarks like
HumanEval) and should only be run in an already-trusted local
environment, the same trust boundary as any other locally executed
code.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..schema import OutcomeInfo
from .schema import (
    CodeCompletionSpec,
    ProseReasoningSpec,
    StructuredJSONSpec,
    Workload,
    WorkloadCategory,
)

_CODE_FENCE_PATTERN = re.compile(
    r"```(?:python)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)

_JSON_TYPE_CHECKS: dict[str, type | tuple[type, ...]] = {
    "str": str,
    "int": int,
    "float": (int, float),
    "bool": bool,
    "list": list,
    "dict": dict,
}


def _extract_code(response_text: str) -> str:
    """Strip a single markdown code fence if present, else return as-is."""
    match = _CODE_FENCE_PATTERN.search(response_text)
    return match.group(1) if match else response_text


def evaluate_code_completion(
    spec: CodeCompletionSpec,
    response_text: str,
    *,
    timeout_seconds: float = 10.0,
) -> OutcomeInfo:
    """Run ``spec.test_code`` against the candidate completion.

    ``success`` is True only if the combined program (candidate code
    plus the fixed test assertions) exits with status 0.
    """
    completion = _extract_code(response_text)
    program = f"{completion}\n\n{spec.test_code}\n"

    with tempfile.TemporaryDirectory(prefix="llmtracefx-eval-") as tmp_dir:
        program_path = Path(tmp_dir) / "candidate.py"
        program_path.write_text(program, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, str(program_path)],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return OutcomeInfo(
                success=False,
                quality_score=0.0,
                quality_metric="unit_test_pass",
                notes=f"candidate completion timed out after {timeout_seconds}s",
            )

    success = completed.returncode == 0
    notes = None if success else completed.stderr.strip()[-500:] or "non-zero exit"
    return OutcomeInfo(
        success=success,
        quality_score=1.0 if success else 0.0,
        quality_metric="unit_test_pass",
        notes=notes,
    )


def _extract_json_object(response_text: str) -> Any:
    """Parse the first JSON object in ``response_text``.

    Tries the whole (stripped) response first, then falls back to the
    substring between the first ``{`` and the last ``}``. Never uses
    ``eval``; a malformed/missing object raises ``json.JSONDecodeError``.
    """
    stripped = response_text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("no JSON object found", stripped, 0)
    return json.loads(stripped[start : end + 1])


def evaluate_structured_json(
    spec: StructuredJSONSpec, response_text: str
) -> OutcomeInfo:
    """Exact required-field-presence and declared-type checks.

    ``quality_score`` is the fraction of ``required_fields`` present
    with the declared type; ``success`` requires all of them to match.
    """
    try:
        payload = _extract_json_object(response_text)
    except json.JSONDecodeError as exc:
        return OutcomeInfo(
            success=False,
            quality_score=0.0,
            quality_metric="structured_json_exact_field_match",
            notes=f"response did not contain a parseable JSON object: {exc}",
        )

    if not isinstance(payload, dict):
        return OutcomeInfo(
            success=False,
            quality_score=0.0,
            quality_metric="structured_json_exact_field_match",
            notes=f"parsed JSON root must be an object, got {type(payload).__name__}",
        )

    problems: list[str] = []
    matched = 0
    for field_name in spec.required_fields:
        if field_name not in payload:
            problems.append(f"missing field '{field_name}'")
            continue
        expected_type_name = spec.field_types.get(field_name)
        expected_type = _JSON_TYPE_CHECKS.get(expected_type_name or "")
        value = payload[field_name]
        if expected_type is None:
            matched += 1
            continue
        # bool is a subclass of int in Python; only accept it for the
        # "bool" type and reject it for "int"/"float" to avoid a
        # boolean silently passing as a numeric field.
        if expected_type_name in ("int", "float") and isinstance(value, bool):
            problems.append(
                f"field '{field_name}' is bool, expected {expected_type_name}"
            )
            continue
        if not isinstance(value, expected_type):
            problems.append(
                f"field '{field_name}' is {type(value).__name__}, "
                f"expected {expected_type_name}"
            )
            continue
        matched += 1

    total = len(spec.required_fields)
    quality_score = matched / total if total else 0.0
    success = matched == total
    return OutcomeInfo(
        success=success,
        quality_score=quality_score,
        quality_metric="structured_json_exact_field_match",
        notes=None if success else "; ".join(problems)[:500],
    )


def evaluate_prose_reasoning(
    spec: ProseReasoningSpec, response_text: str
) -> OutcomeInfo:
    """Deterministic, case-insensitive expected-answer pattern match."""
    match = re.search(spec.expected_answer_pattern, response_text, re.IGNORECASE)
    success = match is not None
    return OutcomeInfo(
        success=success,
        quality_score=1.0 if success else 0.0,
        quality_metric="exact_answer_pattern_match",
        notes=None if success else "expected answer pattern not found in response",
    )


def evaluate_workload(workload: Workload, response_text: str) -> OutcomeInfo:
    """Dispatch to the evaluator matching ``workload.category``."""
    if workload.category is WorkloadCategory.CODE_COMPLETION:
        assert isinstance(workload.spec, CodeCompletionSpec)
        return evaluate_code_completion(workload.spec, response_text)
    if workload.category is WorkloadCategory.STRUCTURED_JSON:
        assert isinstance(workload.spec, StructuredJSONSpec)
        return evaluate_structured_json(workload.spec, response_text)
    assert isinstance(workload.spec, ProseReasoningSpec)
    return evaluate_prose_reasoning(workload.spec, response_text)
