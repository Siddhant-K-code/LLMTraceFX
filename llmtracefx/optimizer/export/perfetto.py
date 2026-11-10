"""Render a :class:`TraceModel` as Chrome trace event JSON.

ui.perfetto.dev and ``chrome://tracing`` both open this format. Times are
microseconds relative to the start of the run, so a trace opens at zero;
the absolute start is kept in ``otherData``.
"""

from __future__ import annotations

import json
from typing import Any

from .trace_model import AttributeValue, TraceModel, TraceSpan

_PID = 1
_TID = 1


def _us(span_ns: int, origin_ns: int) -> float:
    return (span_ns - origin_ns) / 1000.0


def _args(attributes: dict[str, AttributeValue]) -> dict[str, Any]:
    return {key: attributes[key] for key in sorted(attributes)}


def _slice(span: TraceSpan, origin_ns: int) -> dict[str, Any]:
    args = _args(span.attributes)
    if span.error_message is not None:
        args["error.message"] = span.error_message
    return {
        "name": span.name,
        "cat": "llmtracefx.root" if span.parent_key is None else "llmtracefx.phase",
        "ph": "X",
        "ts": _us(span.start_ns, origin_ns),
        "dur": max(0.0, _us(span.end_ns, span.start_ns)),
        "pid": _PID,
        "tid": _TID,
        "args": args,
    }


def render_chrome_trace(model: TraceModel) -> dict[str, Any]:
    origin = model.root.start_ns
    events: list[dict[str, Any]] = [
        {
            "name": "process_name",
            "ph": "M",
            "pid": _PID,
            "tid": _TID,
            "args": {"name": f"llmtracefx {model.run_id}"},
        },
        {
            "name": "thread_name",
            "ph": "M",
            "pid": _PID,
            "tid": _TID,
            "args": {"name": model.root.name},
        },
    ]
    events.extend(_slice(span, origin) for span in model.spans)
    cumulative = 0
    for event in model.root.events:
        ts = _us(event.time_ns, origin)
        events.append(
            {
                "name": str(event.attributes.get("llmtracefx.sse.kind", event.name)),
                "cat": "llmtracefx.sse",
                "ph": "i",
                "s": "t",
                "ts": ts,
                "pid": _PID,
                "tid": _TID,
                "args": _args(event.attributes),
            }
        )
        characters = event.attributes.get("llmtracefx.sse.characters")
        kind = event.attributes.get("llmtracefx.sse.kind")
        if isinstance(characters, int) and kind in ("content", "reasoning"):
            cumulative += characters
            events.append(
                {
                    "name": "output characters (cumulative)",
                    "cat": "llmtracefx.sse",
                    "ph": "C",
                    "ts": ts,
                    "pid": _PID,
                    "args": {"characters": cumulative},
                }
            )
    return {
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "otherData": {
            "producer": "llmtracefx",
            "run_id": model.run_id,
            "started_at": model.started_at,
            "time_origin": "run start; ts values are microseconds from it",
        },
    }


def render_chrome_trace_json(model: TraceModel) -> str:
    return json.dumps(render_chrome_trace(model), indent=1, allow_nan=False) + "\n"
