"""Pure, Gradio-free logic for the LLMTraceFX evidence viewer Space.

Kept separate from ``app.py`` so it can be tested without Gradio.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmtracefx.optimizer.export import (
    ExportError,
    build_trace,
    render_chrome_trace_json,
    render_otlp_json_line,
)
from llmtracefx.optimizer.schema import (
    ExperimentRecord,
    Measurement,
    SchemaValidationError,
)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"


class ViewerError(ValueError):
    """The uploaded evidence could not be read or exported."""


@dataclass(frozen=True)
class ViewerResult:
    summary_markdown: str
    measurements_markdown: str
    spans_markdown: str
    perfetto_path: Path
    otlp_path: Path


def default_allowed_roots() -> tuple[Path, ...]:
    """Directories uploads may come from: Gradio's upload cache and examples."""
    gradio_tmp = os.environ.get("GRADIO_TEMP_DIR") or os.path.join(
        tempfile.gettempdir(), "gradio"
    )
    return (EXAMPLES_DIR, Path(gradio_tmp))


def _read_text(path: str | Path, label: str, roots: Sequence[str | Path]) -> str:
    resolved = os.path.realpath(path)
    allowed = False
    for root in roots:
        base = os.path.realpath(root)
        if resolved.startswith(base + os.sep):
            allowed = True
            break
    if not allowed:
        raise ViewerError(f"{label} is outside the upload directory")
    with open(resolved, "rb") as handle:
        data = handle.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise ViewerError(f"{label} is larger than {MAX_UPLOAD_BYTES} bytes")
    return data.decode("utf-8")


def _read_mapping(
    path: str | Path | None, label: str, roots: Sequence[str | Path]
) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        data = json.loads(_read_text(path, label, roots))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ViewerError(f"{label} is not readable JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ViewerError(f"{label} must be a JSON object")
    return data


def _measurement_rows(record: ExperimentRecord) -> list[tuple[str, Measurement]]:
    rows: list[tuple[str, Measurement]] = []
    for group_name in ("timing", "memory", "power"):
        group = getattr(record, group_name)
        for name, value in vars(group).items():
            if isinstance(value, Measurement):
                rows.append((f"{group_name}.{name}", value))
    if record.speculative.verification_time is not None:
        rows.append(
            ("speculative.verification_time", record.speculative.verification_time)
        )
    return rows


def _escape(text: object) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def render(
    record_path: str | Path,
    api_evidence_path: str | Path | None = None,
    server_metrics_path: str | Path | None = None,
    *,
    output_dir: Path | None = None,
    allowed_roots: Sequence[str | Path] | None = None,
) -> ViewerResult:
    roots = default_allowed_roots() if allowed_roots is None else allowed_roots
    try:
        record = ExperimentRecord.from_json(
            _read_text(record_path, "record.json", roots)
        )
    except (OSError, UnicodeDecodeError, SchemaValidationError) as exc:
        raise ViewerError(
            f"record.json is not a valid ExperimentRecord: {exc}"
        ) from exc
    try:
        model = build_trace(
            record,
            api_evidence=_read_mapping(api_evidence_path, "API evidence", roots),
            server_evidence=_read_mapping(server_metrics_path, "server metrics", roots),
        )
    except ExportError as exc:
        raise ViewerError(str(exc)) from exc

    status = "succeeded" if record.outcome.success else "failed"
    summary = "\n".join(
        [
            f"**Run** `{_escape(record.run_id)}` {status}",
            "",
            f"- Model: `{_escape(record.model.model_id)}`"
            + (
                f" ({_escape(record.model.quantization)})"
                if record.model.quantization
                else ""
            ),
            f"- Runtime: `{_escape(record.runtime.name)}`"
            + (f" {_escape(record.runtime.version)}" if record.runtime.version else "")
            + (
                f" on {_escape(record.runtime.backend)}"
                if record.runtime.backend
                else ""
            ),
            f"- Platform: {_escape(record.platform.os_name)} "
            f"{_escape(record.platform.architecture)}",
            f"- Tokens: input {record.tokens.input_tokens}, generated "
            f"{record.tokens.generated_tokens}",
        ]
    )
    measurement_lines = [
        "| metric | value | unit | provenance |",
        "|---|---:|---|---|",
    ]
    for name, measurement in _measurement_rows(record):
        measurement_lines.append(
            f"| `{name}` | {measurement.value:.6g} | {_escape(measurement.unit)} | "
            f"`{measurement.provenance.value}` |"
        )
    if len(measurement_lines) == 2:
        measurement_lines.append("| (none recorded) | | | |")
    span_lines = ["| span | duration (ms) |", "|---|---:|"]
    for span in model.spans:
        depth = "&nbsp;&nbsp;" if span.parent_key else ""
        duration_ms = (span.end_ns - span.start_ns) / 1e6
        span_lines.append(f"| {depth}{_escape(span.name)} | {duration_ms:.3f} |")

    out = output_dir or Path(tempfile.mkdtemp(prefix="llmtracefx-viewer-"))
    out.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", model.run_id).strip("._") or "run"
    perfetto = out / f"{stem}.perfetto.json"
    otlp = out / f"{stem}.otlp.jsonl"
    perfetto.write_text(render_chrome_trace_json(model), encoding="utf-8")
    otlp.write_text(render_otlp_json_line(model), encoding="utf-8")
    return ViewerResult(
        summary_markdown=summary,
        measurements_markdown="\n".join(measurement_lines),
        spans_markdown="\n".join(span_lines),
        perfetto_path=perfetto,
        otlp_path=otlp,
    )
