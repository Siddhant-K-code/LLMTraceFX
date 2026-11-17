"""Identity of load sweep artifacts, kept import-light for readers that refuse them.

A sweep measures many requests that overlapped in time, so its numbers are
not interchangeable with single-request records. ``gate``, ``compare`` and
``export`` import this module to recognise a sweep artifact and refuse it
by name instead of misreading it or skipping it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

SWEEP_KIND = "llmtracefx.load_sweep"
SWEEP_PLAN_KIND = "llmtracefx.load_sweep_plan"
SWEEP_SCHEMA_VERSION = 1
SWEEP_FILE_NAME = "sweep.json"
SWEEP_PLAN_FILE_NAME = "sweep_plan.json"
SWEEP_KINDS = frozenset({SWEEP_KIND, SWEEP_PLAN_KIND})

SWEEP_REFUSAL_REASON = (
    "it is a concurrent load sweep. Its requests overlapped in time, so its "
    "latencies describe the server under that load and cannot stand in for "
    "single-request measurements. Mixing the two needs an explicit decision "
    "this command does not make."
)


def is_sweep_payload(payload: Any) -> bool:
    """True when ``payload`` is a parsed sweep or sweep plan document."""
    return isinstance(payload, dict) and payload.get("kind") in SWEEP_KINDS


def sweep_refusal(path: Path | str) -> str:
    return f"{path} was refused: {SWEEP_REFUSAL_REASON}"


def sweep_files_in(directory: Path) -> list[Path]:
    """Sweep artifacts directly inside or anywhere below ``directory``, by file name."""
    found: list[Path] = []
    for name in (SWEEP_FILE_NAME, SWEEP_PLAN_FILE_NAME):
        found.extend(sorted(directory.rglob(name)))
    return found
