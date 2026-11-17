"""Reading and writing trace replay artifacts.

Provenance labels scoped to these artifacts:

``trace_declared``
    A value copied from the trace file as it stands, such as a Mooncake
    ``input_length`` or a BurstGPT ``Response tokens``. The trace states it;
    nothing here measured it.

The shared ``MetricProvenance`` labels keep their usual meaning:
``estimated`` for synthetic prompt lengths counted in words,
``derived`` for figures computed from other recorded values,
``measured_wall_clock`` for client timings and ``provider_reported`` for
server usage counts.
"""

from __future__ import annotations

import hashlib
import json
import re
from importlib import metadata
from pathlib import Path
from typing import Any

from .._artifact_io import (
    MAX_EVIDENCE_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from ..collectors._shared import atomic_write_text

TRACE_DECLARED = "trace_declared"

SCHEDULE_KIND = "llmtracefx.trace_replay_schedule"
RESULT_KIND = "llmtracefx.trace_replay"
SCHEMA_VERSION = "1"
SCHEDULE_FILE = "replay_schedule.json"
REQUESTS_FILE = "replay_requests.jsonl"
RESULT_FILE = "replay_result.json"

_SHA256 = re.compile(r"[0-9a-f]{64}")


class ReplayArtifactError(ValueError):
    pass


def tool_record() -> dict[str, Any]:
    try:
        version: str | None = metadata.version("llmtracefx")
    except metadata.PackageNotFoundError:
        version = None
    return {"name": "llmtracefx-optimizer replay", "version": version}


def canonical_json(document: Any) -> str:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def seal(document: dict[str, Any], field: str) -> dict[str, Any]:
    """Add ``field`` holding the sha256 of the document without that field."""
    body = {key: value for key, value in document.items() if key != field}
    return {**body, field: sha256_text(canonical_json(body))}


def render(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, allow_nan=False) + "\n"


def write_text(path: Path, text: str) -> str:
    """Write atomically and return the sha256 of the bytes written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, text)
    return sha256_text(text)


def read_document(path: str | Path, kind: str, seal_field: str) -> dict[str, Any]:
    """Read a replay artifact, checking its kind, version and seal."""
    try:
        text = read_bounded_regular_text(path, MAX_EVIDENCE_ARTIFACT_BYTES)
        data = json.loads(text, parse_constant=reject_non_finite_json_constant)
    except (OSError, ArtifactReadError, ValueError) as exc:
        raise ReplayArtifactError(f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ReplayArtifactError(f"{path}: expected a JSON object")
    if data.get("kind") != kind:
        raise ReplayArtifactError(f"{path}: kind is {data.get('kind')!r}, not {kind}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ReplayArtifactError(
            f"{path}: unsupported schema_version {data.get('schema_version')!r}"
        )
    recorded = data.get(seal_field)
    if not isinstance(recorded, str) or not _SHA256.fullmatch(recorded):
        raise ReplayArtifactError(f"{path}: {seal_field} is missing or malformed")
    if seal(data, seal_field)[seal_field] != recorded:
        raise ReplayArtifactError(
            f"{path}: {seal_field} does not match the content; the file was edited"
        )
    return data
