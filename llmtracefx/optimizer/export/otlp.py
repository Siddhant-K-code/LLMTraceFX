"""Render a :class:`TraceModel` as OTLP/JSON.

The output is one ``TracesData`` object per line, the format of the
OpenTelemetry file exporter. The Collector's ``otlpjsonfile`` receiver
reads it directly, and the same object can be POSTed to any OTLP/HTTP
``/v1/traces`` endpoint with ``Content-Type: application/json``.

OTLP/JSON differs from the protobuf JSON mapping in two ways that matter
here: trace and span IDs are lowercase hex, not base64, and 64-bit
integers (timestamps and ``intValue``) are decimal strings.
"""

from __future__ import annotations

import json
from typing import Any

from .trace_model import (
    AttributeValue,
    TraceModel,
    TraceSpan,
    llmtracefx_version,
    span_id,
    trace_id,
)

SPAN_KIND_INTERNAL = 1
SPAN_KIND_CLIENT = 3
STATUS_CODE_ERROR = 2
SCOPE_NAME = "llmtracefx.optimizer.export"


def _any_value(value: AttributeValue) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    return {"arrayValue": {"values": [{"stringValue": item} for item in value]}}


def _attributes(attributes: dict[str, AttributeValue]) -> list[dict[str, Any]]:
    return [
        {"key": key, "value": _any_value(attributes[key])} for key in sorted(attributes)
    ]


def _span(model: TraceModel, span: TraceSpan) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "traceId": trace_id(model.run_id),
        "spanId": span_id(model.run_id, span.key),
    }
    if span.parent_key is not None:
        payload["parentSpanId"] = span_id(model.run_id, span.parent_key)
    payload.update(
        {
            "name": span.name,
            "kind": SPAN_KIND_CLIENT if span.client else SPAN_KIND_INTERNAL,
            "startTimeUnixNano": str(span.start_ns),
            "endTimeUnixNano": str(span.end_ns),
            "attributes": _attributes(span.attributes),
        }
    )
    if span.events:
        payload["events"] = [
            {
                "timeUnixNano": str(event.time_ns),
                "name": event.name,
                "attributes": _attributes(event.attributes),
            }
            for event in span.events
        ]
    if span.error_message is not None:
        payload["status"] = {
            "code": STATUS_CODE_ERROR,
            "message": span.error_message,
        }
    return payload


def render_otlp_traces(model: TraceModel) -> dict[str, Any]:
    scope: dict[str, Any] = {"name": SCOPE_NAME}
    version = llmtracefx_version()
    if version is not None:
        scope["version"] = version
    return {
        "resourceSpans": [
            {
                "resource": {"attributes": _attributes(model.resource)},
                "scopeSpans": [
                    {
                        "scope": scope,
                        "spans": [_span(model, span) for span in model.spans],
                    }
                ],
            }
        ]
    }


def render_otlp_json_line(model: TraceModel) -> str:
    """One JSON Lines record, including the trailing newline."""
    return (
        json.dumps(render_otlp_traces(model), separators=(",", ":"), allow_nan=False)
        + "\n"
    )
