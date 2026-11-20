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
    evidence_markdown: str
    perfetto_path: Path
    otlp_path: Path


def default_allowed_roots() -> tuple[Path, ...]:
    """Directories uploads may come from: Gradio's upload cache and examples."""
    gradio_tmp = os.environ.get("GRADIO_TEMP_DIR") or os.path.join(
        tempfile.gettempdir(), "gradio"
    )
    return (EXAMPLES_DIR, Path(gradio_tmp))


def _read_text(path: str | Path, label: str, roots: Sequence[str | Path]) -> str:
    prefixes = tuple(os.path.realpath(root) + os.sep for root in roots)
    resolved = os.path.realpath(path)
    if not resolved.startswith(prefixes):
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


def _number(value: object) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def _format(value: float | int) -> str:
    return f"{value:.6g}"


def _entry_row(name: str, entry: object) -> str | None:
    """One table row for a ``{value, unit, provenance, source}`` entry."""
    if not isinstance(entry, dict):
        return None
    value = _number(entry.get("value"))
    if value is None:
        return None
    return (
        f"| `{_escape(name)}` | {_format(value)} | {_escape(entry.get('unit') or '')} "
        f"| `{_escape(entry.get('provenance') or '')}` "
        f"| {_escape(entry.get('source') or '')} |"
    )


_ENTRY_HEADER = [
    "| metric | value | unit | provenance | source |",
    "|---|---:|---|---|---|",
]


def _request_markdown(api: dict[str, Any]) -> list[str]:
    lines = ["### Request (api_evidence.json)", ""]
    rows = ["| field | value | provenance |", "|---|---|---|"]
    for key, label in (
        ("response_model", "response model"),
        ("finish_reason", "finish reason"),
    ):
        if isinstance(api.get(key), str):
            rows.append(f"| {label} | `{_escape(api[key])}` | `provider_reported` |")
    timeline = api.get("timeline")
    if isinstance(timeline, dict):
        provenance = _escape(timeline.get("provenance") or "")
        for key, label in (
            ("first_content_token_offset_ms", "first content token (ms)"),
            ("completed_offset_ms", "completed (ms)"),
        ):
            value = _number(timeline.get(key))
            if value is not None:
                rows.append(f"| {label} | {_format(value)} | `{provenance}` |")
    usage = api.get("usage")
    if isinstance(usage, dict) and usage.get("reported") is True:
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "cached_prompt_tokens",
            "reasoning_tokens",
        ):
            value = _number(usage.get(key))
            if value is not None:
                rows.append(f"| {key} | {_format(value)} | `provider_reported` |")
    reported = api.get("server_reported")
    if isinstance(reported, dict) and isinstance(reported.get("normalized"), dict):
        provenance = _escape(reported.get("provenance") or "provider_reported")
        for key in sorted(reported["normalized"]):
            value = _number(reported["normalized"][key])
            if value is not None:
                rows.append(
                    f"| server {_escape(key)} | {_format(value)} | `{provenance}` |"
                )
    lines.extend(rows if len(rows) > 2 else ["No request fields recorded."])
    if not (isinstance(usage, dict) and usage.get("reported") is True):
        lines += ["", "The provider sent no token usage for this request."]
    return lines


def _server_markdown(server: dict[str, Any]) -> list[str]:
    attribution = server.get("attribution")
    attribution = attribution if isinstance(attribution, dict) else {}
    lines = [
        "### Server metrics (server_metrics_evidence.json)",
        "",
        f"Runtime `{_escape(server.get('runtime') or 'unknown')}`, window "
        f"`{_escape(attribution.get('status') or 'unknown')}`: "
        f"{_escape(attribution.get('reason') or '')}",
        "",
    ]
    highlights = server.get("highlights")
    if attribution.get("attributable") is True and isinstance(highlights, dict):
        rows = [
            row
            for key in sorted(highlights)
            if (row := _entry_row(key, highlights[key])) is not None
        ]
        lines.extend(_ENTRY_HEADER + rows if rows else ["No server values recorded."])
    else:
        reason = server.get("highlights_withheld_reason")
        lines.append(
            "Server values are withheld because the window cannot be tied to "
            "this run" + (f": {_escape(reason)}" if reason else ".")
        )
    speculative = server.get("speculative")
    if isinstance(speculative, dict):
        lines += [
            "",
            f"Speculative decoding: `{_escape(speculative.get('status') or '')}`. "
            f"{_escape(speculative.get('reason') or '')}",
        ]
        rows = []
        counters = speculative.get("counters")
        if isinstance(counters, dict):
            rows += [
                row
                for key in sorted(counters)
                if (row := _entry_row(key, counters[key])) is not None
            ]
        for key in ("acceptance_rate", "acceptance_length"):
            if (row := _entry_row(key, speculative.get(key))) is not None:
                rows.append(row)
        if rows:
            lines += [""] + _ENTRY_HEADER + rows
    kv_usage = server.get("kv_usage")
    if isinstance(kv_usage, dict) and kv_usage:
        lines += [
            "",
            "| KV usage | before | after | unit |",
            "|---|---:|---:|---|",
        ]
        for key in sorted(kv_usage):
            entry = kv_usage[key]
            if not isinstance(entry, dict):
                continue
            before, after = _number(entry.get("before")), _number(entry.get("after"))
            lines.append(
                f"| `{_escape(key)}` "
                f"| {'' if before is None else _format(before)} "
                f"| {'' if after is None else _format(after)} "
                f"| {_escape(entry.get('unit') or '')} |"
            )
    return lines


def _tokens(value: int | None) -> str:
    return "not recorded" if value is None else str(value)


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
    api_evidence = _read_mapping(api_evidence_path, "API evidence", roots)
    server_evidence = _read_mapping(server_metrics_path, "server metrics", roots)
    try:
        model = build_trace(
            record, api_evidence=api_evidence, server_evidence=server_evidence
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
            )
            + (
                f", provider `{_escape(record.runtime.provider)}`"
                if record.runtime.provider
                else ""
            ),
            f"- Platform: {_escape(record.platform.os_name)} "
            f"{_escape(record.platform.architecture)}",
            f"- Tokens: input {_tokens(record.tokens.input_tokens)}, generated "
            f"{_tokens(record.tokens.generated_tokens)}",
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

    evidence_lines: list[str] = []
    if api_evidence is not None:
        evidence_lines += _request_markdown(api_evidence)
    if server_evidence is not None:
        evidence_lines += ([""] if evidence_lines else []) + _server_markdown(
            server_evidence
        )

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
        evidence_markdown="\n".join(evidence_lines),
        perfetto_path=perfetto,
        otlp_path=otlp,
    )
