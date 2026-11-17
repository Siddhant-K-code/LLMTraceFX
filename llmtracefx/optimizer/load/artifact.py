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

_HEAD_BYTES = 4096

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


def is_sweep_file(path: Path) -> bool:
    """True for a file named like a sweep artifact or declaring a sweep kind.

    Only the head is read: the writer puts ``kind`` first, and a file that is
    not a sweep is left for its own reader to judge.
    """
    if path.name in (SWEEP_FILE_NAME, SWEEP_PLAN_FILE_NAME):
        return True
    try:
        with path.open("rb") as handle:
            head = handle.read(_HEAD_BYTES)
    except OSError:
        return False
    return any(f'"{kind}"'.encode() in head for kind in SWEEP_KINDS)


def find_sweep_artifact(path: Path) -> Path | None:
    """The first sweep artifact at or below ``path``, if any."""
    if path.is_dir():
        found = sweep_files_in(path)
        return found[0] if found else None
    if path.is_file() and is_sweep_file(path):
        return path
    return None
