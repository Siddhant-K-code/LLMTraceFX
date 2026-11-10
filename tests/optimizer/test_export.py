"""Tests for OTLP/JSON and Chrome/Perfetto trace export."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any

import pytest
from test_openai_api_collector import glm_stream, make_config, run
from test_schema import make_record

from llmtracefx.optimizer import cli
from llmtracefx.optimizer.export import (
    ExportError,
    build_trace,
    read_json_artifact,
    render_chrome_trace,
    render_otlp_json_line,
    render_otlp_traces,
    span_id,
    trace_id,
)
from llmtracefx.optimizer.schema import (
    ErrorInfo,
    ExperimentRecord,
    Measurement,
    MetricProvenance,
    ModelInfo,
    TimingMetrics,
)
from llmtracefx.optimizer.server_metrics import diff_snapshots, snapshot_from_body

STARTED_AT = "2026-01-01T00:00:00.000000Z"


def ms(value: float) -> Measurement:
    return Measurement(
        value=value, provenance=MetricProvenance.MEASURED_NATIVE, unit="ms"
    )


def api_run(tmp_path: Path, **stream: Any) -> tuple[ExperimentRecord, dict[str, Any]]:
    run(make_config(tmp_path), glm_stream(**stream))
    artifacts = tmp_path / "artifacts"
    record = ExperimentRecord.read_json(artifacts / "record.json")
    return record, read_json_artifact(artifacts / "api_evidence.json")


def spans_of(otlp: dict[str, Any]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = otlp["resourceSpans"][0]["scopeSpans"][0]["spans"]
    return spans


def attrs(span: dict[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for attribute in span["attributes"]:
        ((kind, value),) = attribute["value"].items()
        if kind == "arrayValue":
            value = [item["stringValue"] for item in value["values"]]
        elif kind == "intValue":
            assert isinstance(value, str)
            value = int(value)
        flattened[attribute["key"]] = value
    return flattened


def server_evidence(*, requests_after: int) -> dict[str, Any]:
    def body(requests: int) -> bytes:
        return (
            "# TYPE vllm:time_to_first_token_seconds histogram\n"
            f'vllm:time_to_first_token_seconds_bucket{{le="+Inf"}} {requests}\n'
            f"vllm:time_to_first_token_seconds_count {requests}\n"
            f"vllm:time_to_first_token_seconds_sum {0.02 * requests}\n"
            "# TYPE vllm:e2e_request_latency_seconds histogram\n"
            f'vllm:e2e_request_latency_seconds_bucket{{le="+Inf"}} {requests}\n'
            f"vllm:e2e_request_latency_seconds_count {requests}\n"
            f"vllm:e2e_request_latency_seconds_sum {0.5 * requests}\n"
        ).encode()

    url = "http://127.0.0.1:8000/metrics"
    before = snapshot_from_body(
        url, body(2), None, captured_at="2000-01-01T00:00:00Z", scrape_duration_ms=1
    )
    after = snapshot_from_body(
        url,
        body(requests_after),
        None,
        captured_at="2100-01-01T00:00:00Z",
        scrape_duration_ms=1,
    )
    return diff_snapshots(before, after, expected_requests=1)


# --- OTLP/JSON -----------------------------------------------------------------


def test_api_run_exports_genai_spans_with_otlp_json_encoding(tmp_path: Path) -> None:
    record, evidence = api_run(tmp_path)
    line = render_otlp_json_line(build_trace(record, api_evidence=evidence))

    assert line.endswith("\n") and "\n" not in line[:-1]
    spans = spans_of(json.loads(line))
    root = spans[0]
    assert root["traceId"] == trace_id(record.run_id)
    assert len(root["traceId"]) == 32 and len(root["spanId"]) == 16
    int(root["traceId"], 16)
    assert root["kind"] == 3
    assert "parentSpanId" not in root
    assert isinstance(root["startTimeUnixNano"], str)
    assert int(root["endTimeUnixNano"]) >= int(root["startTimeUnixNano"])
    assert "status" not in root

    root_attrs = attrs(root)
    assert root_attrs["gen_ai.operation.name"] == "chat"
    assert root_attrs["gen_ai.provider.name"] == "z.ai"
    assert root_attrs["gen_ai.request.model"] == "glm-5.3"
    assert root_attrs["gen_ai.usage.input_tokens"] == 17
    assert root_attrs["gen_ai.usage.output_tokens"] == 5
    assert root_attrs["gen_ai.usage.cache_read.input_tokens"] == 4
    assert root_attrs["gen_ai.response.finish_reasons"] == ["stop"]
    assert root_attrs["server.address"] == "api.z.ai"

    names = [span["name"] for span in spans[1:]]
    assert names[0] == "wait response_headers"
    for child in spans[1:]:
        assert child["parentSpanId"] == root["spanId"]
        assert int(child["startTimeUnixNano"]) >= int(root["startTimeUnixNano"])
        assert int(child["endTimeUnixNano"]) <= int(root["endTimeUnixNano"])
        assert attrs(child)["llmtracefx.span.placement"] == "measured_offsets"
    assert {event["name"] for event in root["events"]} == {"llmtracefx.sse_event"}
    assert root["spanId"] == span_id(record.run_id, "root")
    assert len({span["spanId"] for span in spans}) == len(spans)


def test_export_is_deterministic(tmp_path: Path) -> None:
    record, evidence = api_run(tmp_path)
    first = render_otlp_json_line(build_trace(record, api_evidence=evidence))
    second = render_otlp_json_line(build_trace(record, api_evidence=evidence))
    assert first == second


def test_time_to_first_chunk_counts_reasoning_output(tmp_path: Path) -> None:
    with_reasoning, evidence = api_run(tmp_path / "a")
    reasoning_attrs = attrs(
        spans_of(
            render_otlp_traces(build_trace(with_reasoning, api_evidence=evidence))
        )[0]
    )
    first_reasoning = min(
        event["offset_ms"]
        for event in evidence["timeline"]["events"]
        if event["kind"] == "reasoning" and event["characters"] > 0
    )
    assert reasoning_attrs["gen_ai.response.time_to_first_chunk"] == pytest.approx(
        first_reasoning / 1000
    )
    assert first_reasoning < evidence["timeline"]["first_content_token_offset_ms"]

    content_only, evidence = api_run(tmp_path / "b", reasoning_parts=())
    content_attrs = attrs(
        spans_of(render_otlp_traces(build_trace(content_only, api_evidence=evidence)))[
            0
        ]
    )
    assert content_attrs["gen_ai.response.time_to_first_chunk"] == pytest.approx(
        evidence["timeline"]["first_content_token_offset_ms"] / 1000
    )


def test_local_phases_are_laid_out_sequentially_and_paths_redacted() -> None:
    record = make_record(
        started_at=STARTED_AT,
        model=ModelInfo(model_id="/Users/someone/models/qwen.gguf"),
        timing=TimingMetrics(
            model_load=ms(1000.0),
            prefill=ms(200.0),
            decode=ms(3000.0),
            total=ms(5000.0),
        ),
    )
    spans = spans_of(render_otlp_traces(build_trace(record)))

    assert [span["name"] for span in spans] == [
        "text_completion qwen.gguf",
        "model_load",
        "prefill",
        "decode",
    ]
    assert spans[0]["kind"] == 1
    starts = [int(span["startTimeUnixNano"]) for span in spans]
    ends = [int(span["endTimeUnixNano"]) for span in spans]
    assert starts[0] == 1767225600 * 10**9
    assert ends[0] - starts[0] == 5000 * 10**6
    assert starts[1:] == [starts[0], ends[1], ends[2]]
    assert "/Users" not in render_otlp_json_line(build_trace(record))
    assert "/Users" in render_otlp_json_line(build_trace(record, include_paths=True))


def test_phases_that_exceed_the_total_are_omitted_not_stretched() -> None:
    record = make_record(
        started_at=STARTED_AT,
        timing=TimingMetrics(prefill=ms(9000.0), total=ms(5000.0)),
    )
    model = build_trace(record)
    assert len(model.spans) == 1
    assert (
        model.root.attributes["llmtracefx.span.phase_layout"]
        == "omitted_phases_exceed_total"
    )


def test_failed_runs_carry_error_status() -> None:
    record = make_record(error=ErrorInfo(category="runtime", message="out of memory"))
    root = spans_of(render_otlp_traces(build_trace(record)))[0]
    assert root["status"] == {"code": 2, "message": "out of memory"}
    assert attrs(root)["error.type"] == "runtime"


def test_mismatched_inputs_are_refused(tmp_path: Path) -> None:
    record, evidence = api_run(tmp_path)
    with pytest.raises(ExportError, match="run_id"):
        build_trace(record, api_evidence={**evidence, "run_id": "other"})
    with pytest.raises(ExportError, match="not an API run"):
        build_trace(make_record(), api_evidence=evidence)
    with pytest.raises(ExportError, match="server evidence"):
        build_trace(record, server_evidence={"kind": "something-else"})


# --- Server metrics on the trace ------------------------------------------------


def test_attributed_server_metrics_are_attached_with_provenance(
    tmp_path: Path,
) -> None:
    record, evidence = api_run(tmp_path)
    root = build_trace(
        record, api_evidence=evidence, server_evidence=server_evidence(requests_after=3)
    ).root.attributes
    assert root["llmtracefx.server.attribution.status"] == "attributed"
    assert root["llmtracefx.server.time_to_first_token.value"] == pytest.approx(20.0)
    assert root["llmtracefx.server.time_to_first_token.provenance"] == (
        "measured_native"
    )


def test_unattributed_server_metrics_only_carry_the_status(tmp_path: Path) -> None:
    record, evidence = api_run(tmp_path)
    root = build_trace(
        record, api_evidence=evidence, server_evidence=server_evidence(requests_after=5)
    ).root.attributes
    assert root["llmtracefx.server.attribution.status"] == "contaminated"
    assert not any(key.endswith(".value") for key in root if "server" in key)


def test_server_window_must_bracket_the_run(tmp_path: Path) -> None:
    record, evidence = api_run(tmp_path)
    stale = server_evidence(requests_after=3)
    stale["after"] = {**stale["after"], "captured_at": "2001-01-01T00:00:00Z"}
    with pytest.raises(ExportError, match="bracket"):
        build_trace(record, api_evidence=evidence, server_evidence=stale)


# --- Chrome / Perfetto ------------------------------------------------------------


def test_chrome_trace_uses_relative_microseconds_and_nests_phases(
    tmp_path: Path,
) -> None:
    record, evidence = api_run(tmp_path)
    trace = render_chrome_trace(build_trace(record, api_evidence=evidence))
    events = trace["traceEvents"]

    slices = [event for event in events if event["ph"] == "X"]
    root = slices[0]
    assert root["ts"] == 0
    assert root["dur"] == pytest.approx(record.timing.total.value * 1000)
    for child in slices[1:]:
        assert child["ts"] >= 0
        assert child["ts"] + child["dur"] <= root["dur"] + 1e-6
    instants = [event for event in events if event["ph"] == "i"]
    assert instants and all(event["s"] == "t" for event in instants)
    counters = [event["args"]["characters"] for event in events if event["ph"] == "C"]
    assert counters == sorted(counters) and counters[-1] > 0
    assert trace["otherData"]["run_id"] == record.run_id
    json.dumps(trace, allow_nan=False)


# --- CLI --------------------------------------------------------------------------


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            args = cli.build_parser().parse_args(argv)
            code = int(args.func(args))
    except SystemExit as exc:
        code = int(exc.code or 0)
    return code, stdout.getvalue(), stderr.getvalue()


@pytest.mark.parametrize("fmt", ["otlp", "perfetto"])
def test_export_cli_writes_each_format(tmp_path: Path, fmt: str) -> None:
    api_run(tmp_path)
    artifacts = tmp_path / "artifacts"
    output = tmp_path / f"trace.{fmt}.json"

    code, out, err = run_cli(
        [
            "export",
            fmt,
            "--record",
            str(artifacts / "record.json"),
            "--api-evidence",
            str(artifacts / "api_evidence.json"),
            "--output",
            str(output),
        ]
    )

    assert code == 0, err
    assert "spans written" in out
    payload = json.loads(output.read_text())
    assert ("resourceSpans" if fmt == "otlp" else "traceEvents") in payload


def test_export_cli_rejects_an_invalid_record(tmp_path: Path) -> None:
    bad = tmp_path / "record.json"
    bad.write_text("{}", encoding="utf-8")
    code, _, err = run_cli(
        ["export", "otlp", "--record", str(bad), "--output", str(tmp_path / "o")]
    )
    assert code == 1
    assert "Cannot export" in err
    assert not (tmp_path / "o").exists()
