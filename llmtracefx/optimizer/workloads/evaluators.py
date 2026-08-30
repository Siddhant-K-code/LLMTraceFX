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
HumanEval). To reduce (not eliminate -- this is not a security
sandbox) the blast radius of executing untrusted model output, the
subprocess:

* runs with its working directory set to a fresh, single-use temporary
  directory that is removed afterward, so relative file writes by the
  candidate land there instead of the caller's cwd;
* runs with a minimal, explicit environment allowlist (``PATH`` plus a
  handful of Windows system variables needed for the interpreter to
  start) instead of the full inherited environment, so parent secrets
  (API keys, tokens, credentials) are never visible to candidate code;
* on POSIX, applies conservative ``resource.setrlimit`` bounds (CPU
  time, address space, output file size) before exec via
  ``preexec_fn``, scoped to that one process only, on a best-effort
  basis (any single limit a given POSIX platform rejects -- notably
  macOS/Darwin's ``RLIMIT_AS`` -- is skipped rather than aborting
  subprocess creation). Process-count limits (``RLIMIT_NPROC``) are
  deliberately left untouched because they are per-user, not
  per-process, and could affect unrelated processes sharing this host.
  ``preexec_fn``/``resource`` are POSIX-only (CPython raises
  ``ValueError`` if ``preexec_fn`` is passed on Windows), so on Windows
  only the cwd/environment isolation above applies and no CPU/memory/
  file-size ceiling is enforced.

This should still only be run in an already-trusted local environment;
it is containment for a benchmarking tool, not a hostile-code sandbox.
"""

from __future__ import annotations

import json
import os
import re
import signal
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

#: Explicit allowlist for the candidate-code subprocess environment.
#: Everything else from the caller's environment (API keys, tokens,
#: credentials, unrelated config) is deliberately excluded.
_ALLOWED_SUBPROCESS_ENV_VARS: tuple[str, ...] = (
    "PATH",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "PATHEXT",
    "COMSPEC",
    "TEMP",
    "TMP",
)

_IS_POSIX = os.name == "posix"

#: Conservative POSIX-only resource ceilings for the candidate-code
#: subprocess. Deliberately generous enough not to interfere with any
#: legitimate small workload-catalog completion; see the module
#: docstring for why ``RLIMIT_NPROC`` is not included.
_MAX_CPU_SECONDS = 10
_MAX_ADDRESS_SPACE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GiB
_MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MiB
_POST_TERMINATION_GRACE_SECONDS = 1.0

_PROCESS_TERMINATION_PERMISSION_WARNING = (
    "warning: the operating system denied permission to terminate the candidate "
    "process group; individual process cleanup was attempted"
)


class _CandidateTimeoutError(Exception):
    """Internal timeout that preserves a process-cleanup warning."""

    def __init__(self, cleanup_warning: str | None) -> None:
        self.cleanup_warning = cleanup_warning


def _minimal_subprocess_env() -> dict[str, str]:
    """Build a minimal, explicit environment for the candidate subprocess."""
    return {
        name: os.environ[name]
        for name in _ALLOWED_SUBPROCESS_ENV_VARS
        if name in os.environ
    }


def _posix_preexec_resource_limits() -> None:
    """Best-effort CPU/address-space/file-size limits before exec (POSIX).

    Runs inside the freshly forked child, before the candidate program's
    process image is loaded, so any limit that does apply bounds only
    that one process. Each limit is applied independently and any
    platform quirk that rejects it is swallowed rather than aborting the
    whole subprocess launch: notably, macOS/Darwin's kernel rejects
    ``RLIMIT_AS`` for reasons unrelated to the requested value (its
    reported hard limit does not match what it actually enforces), so
    treating a single unsupported limit as fatal would break code
    evaluation entirely on that platform. Never call this on a
    non-POSIX platform.
    """
    import resource

    for limit, bounds in (
        (resource.RLIMIT_CPU, (_MAX_CPU_SECONDS, _MAX_CPU_SECONDS)),
        (
            resource.RLIMIT_AS,
            (_MAX_ADDRESS_SPACE_BYTES, _MAX_ADDRESS_SPACE_BYTES),
        ),
        (
            resource.RLIMIT_FSIZE,
            (_MAX_FILE_SIZE_BYTES, _MAX_FILE_SIZE_BYTES),
        ),
    ):
        try:
            resource.setrlimit(limit, bounds)
        except (ValueError, OSError):
            # Platform does not support (or rejects) this specific
            # limit; skip it and keep applying the remaining ones.
            continue


def _extract_code(response_text: str) -> str:
    """Strip a single markdown code fence if present, else return as-is."""
    match = _CODE_FENCE_PATTERN.search(response_text)
    return match.group(1) if match else response_text


def _process_ids_in_group(process_group_id: int) -> tuple[int, ...]:
    """Return live process IDs in a POSIX process group.

    Darwin can report ``ESRCH`` for ``killpg`` after the group leader has
    exited even while descendants with the same PGID remain alive. ``ps`` is
    used only as a cleanup fallback for that condition.
    """
    try:
        completed = subprocess.run(
            ["/bin/ps", "-axo", "pid=,pgid="],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
            shell=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ()

    members: list[int] = []
    for line in completed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            pid, pgid = (int(field) for field in fields)
        except ValueError:
            continue
        if pgid == process_group_id:
            members.append(pid)
    return tuple(members)


def _terminate_candidate_process(
    process: subprocess.Popen[str], *, process_group_id: int | None
) -> str | None:
    """Terminate the candidate and return any cleanup limitation."""
    if _IS_POSIX:
        if process_group_id is None:
            return None
        permission_denied = False
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            permission_denied = True
        else:
            return None

        for pid in _process_ids_in_group(process_group_id):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                continue
            except PermissionError:
                permission_denied = True
        return _PROCESS_TERMINATION_PERMISSION_WARNING if permission_denied else None
    elif process.poll() is None:
        process.kill()
    return None


def _run_candidate(
    program_path: Path,
    *,
    cwd: str,
    timeout_seconds: float,
) -> tuple[subprocess.CompletedProcess[str], str | None]:
    """Run candidate code while containing its process tree on POSIX."""
    process = subprocess.Popen(
        [sys.executable, str(program_path)],
        cwd=cwd,
        env=_minimal_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
        shell=False,
        start_new_session=_IS_POSIX,
        preexec_fn=(_posix_preexec_resource_limits if _IS_POSIX else None),
    )
    process_group_id = process.pid if _IS_POSIX else None
    cleanup_warning = None
    timed_out = False
    stdout = ""
    stderr = ""
    try:
        stdout, stderr = process.communicate(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        cleanup_warning = _terminate_candidate_process(
            process, process_group_id=process_group_id
        )
        try:
            process.communicate(timeout=_POST_TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            # A surviving candidate or descendant can keep the pipes open after
            # cleanup. The configured evaluation timeout has already elapsed, so
            # allow only a fixed termination grace and never block on another read.
            pass
    finally:
        # A candidate can let its direct process exit while leaving detached
        # workers in the new process group. Clean up the whole group before
        # returning from the evaluator.
        final_cleanup_warning = _terminate_candidate_process(
            process, process_group_id=process_group_id
        )
        cleanup_warning = cleanup_warning or final_cleanup_warning

    if timed_out:
        raise _CandidateTimeoutError(cleanup_warning)

    return (
        subprocess.CompletedProcess(
            args=process.args,
            returncode=process.returncode,
            stdout=stdout,
            stderr=stderr,
        ),
        cleanup_warning,
    )


def evaluate_code_completion(
    spec: CodeCompletionSpec,
    response_text: str,
    *,
    timeout_seconds: float = 10.0,
) -> OutcomeInfo:
    """Run ``spec.test_code`` against the candidate completion.

    ``success`` is True only if the combined program (candidate code
    plus the fixed test assertions) exits with status 0. See the module
    docstring for the containment measures applied to this subprocess.
    """
    completion = _extract_code(response_text)
    program = f"{completion}\n\n{spec.test_code}\n"

    with tempfile.TemporaryDirectory(prefix="llmtracefx-eval-") as tmp_dir:
        program_path = Path(tmp_dir) / "candidate.py"
        program_path.write_text(program, encoding="utf-8")
        try:
            completed, cleanup_warning = _run_candidate(
                program_path,
                cwd=tmp_dir,
                timeout_seconds=timeout_seconds,
            )
        except _CandidateTimeoutError as exc:
            timeout_notes = f"candidate completion timed out after {timeout_seconds}s"
            if exc.cleanup_warning:
                timeout_notes = f"{timeout_notes}; {exc.cleanup_warning}"
            return OutcomeInfo(
                success=False,
                quality_score=0.0,
                quality_metric="unit_test_pass",
                notes=timeout_notes,
            )

    success = completed.returncode == 0
    notes = None if success else completed.stderr.strip()[-500:] or "non-zero exit"
    if cleanup_warning:
        notes = f"{notes}; {cleanup_warning}" if notes else cleanup_warning
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
