"""Tests for deterministic workload evaluators."""

from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

from llmtracefx.optimizer.workloads import evaluators
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
from llmtracefx.optimizer.workloads.schema import CodeCompletionSpec

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


# --- Code completion sandboxing: cwd, environment, resource limits --------


def test_evaluate_code_completion_isolates_cwd_from_the_caller(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spec = CodeCompletionSpec(
        function_stub="def f() -> None:\n    pass\n",
        test_code="",
        entry_point="f",
    )
    completion = (
        "def f() -> None:\n"
        "    with open('leaked.txt', 'w') as handle:\n"
        "        handle.write('leaked')\n"
        "\n"
        "f()\n"
    )
    outcome = evaluate_code_completion(spec, completion)
    assert outcome.success is True
    # The candidate's relative file write must land in its own ephemeral
    # temp directory, never in the caller's (here, the test's) cwd.
    assert not (tmp_path / "leaked.txt").exists()


def test_evaluate_code_completion_does_not_expose_parent_secrets(monkeypatch):
    monkeypatch.setenv("LLMTRACEFX_TEST_FAKE_SECRET", "super-secret-value")
    spec = CodeCompletionSpec(
        function_stub="def f() -> None:\n    pass\n",
        test_code=(
            "import os\nassert os.environ.get('LLMTRACEFX_TEST_FAKE_SECRET') is None\n"
        ),
        entry_point="f",
    )
    completion = "def f() -> None:\n    pass\n"
    outcome = evaluate_code_completion(spec, completion)
    assert outcome.success is True


def test_evaluate_code_completion_minimal_env_is_a_strict_allowlist():
    env = evaluators._minimal_subprocess_env()
    assert set(env).issubset(set(evaluators._ALLOWED_SUBPROCESS_ENV_VARS))


@pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX-only resource limits; not enforced on Windows (see module docstring)",
)
def test_evaluate_code_completion_posix_cpu_limit_kills_busy_loop(monkeypatch):
    # Lower the CPU ceiling for this test only so it stays fast; the
    # wall-clock timeout passed below is deliberately much larger so a
    # pass here proves the RLIMIT_CPU kill fired, not the ordinary
    # subprocess timeout path already covered above.
    monkeypatch.setattr(evaluators, "_MAX_CPU_SECONDS", 1)
    spec = CodeCompletionSpec(
        function_stub="def f() -> None:\n    pass\n",
        test_code="",
        entry_point="f",
    )
    busy_loop = "x = 0\nwhile True:\n    x += 1\n"

    started = time.perf_counter()
    outcome = evaluate_code_completion(spec, busy_loop, timeout_seconds=15.0)
    elapsed = time.perf_counter() - started

    assert outcome.success is False
    assert elapsed < 15.0


@pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX-only resource limits; not enforced on Windows (see module docstring)",
)
def test_evaluate_code_completion_posix_file_size_limit_kills_large_write(monkeypatch):
    monkeypatch.setattr(evaluators, "_MAX_FILE_SIZE_BYTES", 1024)
    spec = CodeCompletionSpec(
        function_stub="def f() -> None:\n    pass\n",
        test_code="",
        entry_point="f",
    )
    oversized_write = (
        "def f() -> None:\n"
        "    pass\n"
        "\n"
        "with open('big.bin', 'wb') as handle:\n"
        "    handle.write(b'x' * (10 * 1024 * 1024))\n"
    )
    outcome = evaluate_code_completion(spec, oversized_write)
    assert outcome.success is False


@pytest.mark.skipif(
    os.name != "posix",
    reason="POSIX-only process-group containment",
)
def test_evaluate_code_completion_timeout_kills_descendants(tmp_path):
    child_pid_path = tmp_path / "child.pid"
    spec = CodeCompletionSpec(
        function_stub="def f() -> None:\n    pass\n",
        test_code="",
        entry_point="f",
    )
    completion = (
        "import subprocess\n"
        "import sys\n"
        "import time\n"
        f"pid_path = {str(child_pid_path)!r}\n"
        "child = subprocess.Popen([\n"
        "    sys.executable,\n"
        "    '-c',\n"
        "    'import time; time.sleep(60)',\n"
        "])\n"
        "with open(pid_path, 'w') as handle:\n"
        "    handle.write(str(child.pid))\n"
        "time.sleep(60)\n"
    )

    outcome = evaluate_code_completion(spec, completion, timeout_seconds=0.5)

    assert outcome.success is False
    assert "timed out" in outcome.notes
    child_pid = int(child_pid_path.read_text())
    deadline = time.time() + 2
    while time.time() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        subprocess.run(
            [sys.executable, "-c", f"import os; os.kill({child_pid}, 9)"],
            check=False,
        )
        pytest.fail("candidate descendant survived evaluator timeout")


def test_evaluate_code_completion_ordinary_completion_still_evaluates_after_sandboxing():
    outcome = evaluate_code_completion(
        CODE_COMPLETION_PALINDROME.spec, _CORRECT_PALINDROME_COMPLETION
    )
    assert outcome.success is True
    assert outcome.quality_score == 1.0


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
