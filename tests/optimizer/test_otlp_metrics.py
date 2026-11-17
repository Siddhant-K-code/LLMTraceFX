"""Tests for OTLP/JSON GenAI metrics export."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from test_export import STARTED_AT, api_run, attrs, run_cli, server_evidence
from test_schema import make_record

from llmtracefx.optimizer.export import (
    ExportError,
    MetricsExport,
    MetricsRun,
    build_otlp_metrics,
    render_otlp_metrics_json_line,
    span_id,
    trace_id,
)
from llmtracefx.optimizer.schema import (
    ErrorInfo,
    ExperimentRecord,
    Measurement,
    MetricProvenance,
    ModelInfo,
    RuntimeInfo,
    TimingMetrics,
    TokenCounts,
)

DURATION = "gen_ai.client.inference.duration"
TTFC = "gen_ai.client.inference.time_to_first_chunk"
PER_CHUNK = "gen_ai.client.inference.time_per_output_chunk"
INPUT_TOKENS = "gen_ai.client.inference.operation.input_tokens"
OUTPUT_TOKENS = "gen_ai.client.inference.operation.output_tokens"
SERVER_TTFT = "gen_ai.server.time_to_first_token"
SERVER_TPOT = "gen_ai.server.time_per_output_token"
SERVER_DURATION = "gen_ai.server.request.duration"
SECONDS_BOUNDS = [
    0.01,
    0.02,
    0.04,
    0.08,
    0.16,
    0.32,
    0.64,
    1.28,
    2.56,
    5.12,
    10.24,
    20.48,
    40.96,
    81.92,
]


def api_record(
    run_id: str,
    total_ms: float | None = 10.0,
    *,
    provenance: MetricProvenance = MetricProvenance.MEASURED_WALL_CLOCK,
    tokens: TokenCounts | None = None,
    **overrides: Any,
) -> ExperimentRecord:
    return make_record(
        run_id=run_id,
        started_at=STARTED_AT,
        runtime=RuntimeInfo(name="openai-compatible-api", provider="vllm"),
        model=ModelInfo(model_id="qwen"),
        tokens=tokens
        or TokenCounts(
            input_tokens=17,
            generated_tokens=5,
            provenance=MetricProvenance.PROVIDER_REPORTED,
        ),
        timing=TimingMetrics(
            total=(
                None
                if total_ms is None
                else Measurement(value=total_ms, provenance=provenance, unit="ms")
            )
        ),
        **overrides,
    )


def metrics_of(export: MetricsExport) -> dict[str, dict[str, Any]]:
    (resource,) = export.payload["resourceMetrics"]
    (scope,) = resource["scopeMetrics"]
    return {metric["name"]: metric for metric in scope["metrics"]}


def points(export: MetricsExport, name: str) -> list[dict[str, Any]]:
    data_points: list[dict[str, Any]] = metrics_of(export)[name]["histogram"][
        "dataPoints"
    ]
    return data_points


def vllm_evidence(**highlights: dict[str, Any]) -> dict[str, Any]:
    evidence = server_evidence(requests_after=3)
    evidence["highlights"].update(highlights)
    return evidence


def tpot_highlight(value: float, source: str) -> dict[str, Any]:
    return {
        "value": value,
        "unit": "ms",
        "provenance": "measured_native",
        "statistic": "single_observation",
        "source": source,
        "observations": 1,
    }


def test_api_run_exports_genai_histograms_with_otlp_json_encoding(
    tmp_path: Path,
) -> None:
    record, evidence = api_run(tmp_path)
    export = build_otlp_metrics(
        [MetricsRun(record, evidence, server_evidence(requests_after=3))]
    )
    line = render_otlp_metrics_json_line(export)

    assert line.endswith("\n") and "\n" not in line[:-1]
    assert json.loads(line) == export.payload
    assert line == render_otlp_metrics_json_line(
        build_otlp_metrics(
            [MetricsRun(record, evidence, server_evidence(requests_after=3))]
        )
    )
    metrics = metrics_of(export)
    assert set(metrics) == {
        DURATION,
        TTFC,
        PER_CHUNK,
        INPUT_TOKENS,
        OUTPUT_TOKENS,
        SERVER_TTFT,
        SERVER_DURATION,
    }
    assert metrics[INPUT_TOKENS]["unit"] == "{token}"
    for metric in metrics.values():
        histogram = metric["histogram"]
        assert histogram["aggregationTemporality"] == 2
        for point in histogram["dataPoints"]:
            assert isinstance(point["count"], str)
            assert isinstance(point["startTimeUnixNano"], str)
            assert len(point["bucketCounts"]) == len(point["explicitBounds"]) + 1
            assert sum(int(c) for c in point["bucketCounts"]) == int(point["count"])
            point_attrs = attrs(point)
            assert point_attrs["gen_ai.operation.name"] == "chat"
            assert point_attrs["gen_ai.provider.name"] == "z.ai"
            assert point_attrs["gen_ai.request.model"] == "glm-5.3"
            assert point_attrs["server.port"] == 443
            assert "llmtracefx.provenance" in point_attrs
            assert "error.type" not in point_attrs

    (duration,) = points(export, DURATION)
    assert duration["explicitBounds"] == SECONDS_BOUNDS
    assert duration["sum"] == pytest.approx(record.timing.total.value / 1000)
    (exemplar,) = duration["exemplars"]
    assert exemplar["traceId"] == trace_id(record.run_id)
    assert exemplar["spanId"] == span_id(record.run_id, "root")

    (first_chunk,) = points(export, TTFC)
    first_reasoning = min(
        event["offset_ms"]
        for event in evidence["timeline"]["events"]
        if event["kind"] == "reasoning" and event["characters"] > 0
    )
    assert first_chunk["sum"] == pytest.approx(first_reasoning / 1000)
    assert attrs(first_chunk)["llmtracefx.provenance"] == "measured_wall_clock"

    (per_chunk,) = points(export, PER_CHUNK)
    assert per_chunk["count"] == "2"  # reasoning, content, content
    assert "exemplars" not in per_chunk

    (output_tokens,) = points(export, OUTPUT_TOKENS)
    assert output_tokens["sum"] == 5
    assert attrs(output_tokens)["llmtracefx.provenance"] == "provider_reported"

    (server_ttft,) = points(export, SERVER_TTFT)
    assert server_ttft["sum"] == pytest.approx(0.02)
    assert server_ttft["explicitBounds"][0] == 0.001
    assert attrs(server_ttft)["llmtracefx.source"] == (
        "vllm:time_to_first_token_seconds"
    )
    assert any("time_per_output_token highlight" in n for n in export.notes)


def test_runs_aggregate_per_attribute_set_with_upper_inclusive_buckets() -> None:
    runs = [
        MetricsRun(api_record("a", 10.0)),
        MetricsRun(api_record("b", 15.0)),
        MetricsRun(api_record("c", 100_000.0)),
        MetricsRun(api_record("d", 20.0, provenance=MetricProvenance.MEASURED_NATIVE)),
    ]
    export = build_otlp_metrics(runs)

    native, wall = sorted(
        points(export, DURATION), key=lambda p: attrs(p)["llmtracefx.provenance"]
    )
    assert attrs(wall)["llmtracefx.provenance"] == "measured_wall_clock"
    assert wall["count"] == "3"
    counts = [int(c) for c in wall["bucketCounts"]]
    assert counts[0] == 1  # 0.01 s sits on the first bound
    assert counts[1] == 1
    assert counts[-1] == 1  # above the last bound
    assert wall["min"] == pytest.approx(0.01)
    assert wall["max"] == pytest.approx(100.0)
    assert wall["sum"] == pytest.approx(100.025)
    assert [e["filteredAttributes"][0]["value"] for e in wall["exemplars"]] == [
        {"stringValue": run} for run in ("a", "b", "c")
    ]
    assert native["count"] == "1"
    assert int(wall["timeUnixNano"]) - int(wall["startTimeUnixNano"]) == 100_000 * (
        10**6
    )
    assert export.exported_run_ids == ("a", "b", "c", "d")


def test_failed_run_carries_error_type_and_has_no_server_metrics() -> None:
    record = api_record(
        "failed", error=ErrorInfo(category="timeout", message="no response")
    )
    export = build_otlp_metrics(
        [MetricsRun(record, server_evidence=server_evidence(requests_after=3))]
    )

    assert attrs(points(export, DURATION)[0])["error.type"] == "timeout"
    assert attrs(points(export, OUTPUT_TOKENS)[0])["error.type"] == "timeout"
    assert SERVER_TTFT not in metrics_of(export)
    assert "failed: server metrics omitted: the run did not succeed" in export.notes


def test_missing_or_estimated_values_are_omitted_not_filled() -> None:
    export = build_otlp_metrics(
        [
            MetricsRun(api_record("no-total", None)),
            MetricsRun(
                api_record(
                    "estimated",
                    tokens=TokenCounts(
                        input_tokens=3,
                        generated_tokens=4,
                        provenance=MetricProvenance.ESTIMATED,
                    ),
                )
            ),
            MetricsRun(api_record("no-provenance", tokens=TokenCounts(input_tokens=3))),
            MetricsRun(make_record(run_id="local", started_at=STARTED_AT)),
        ]
    )

    (duration,) = points(export, DURATION)
    assert duration["count"] == "2"
    (input_tokens,) = points(export, INPUT_TOKENS)
    assert input_tokens["count"] == "1"
    assert input_tokens["sum"] == 17
    assert export.exported_run_ids == ("estimated", "no-provenance", "no-total")
    joined = "\n".join(export.notes)
    assert "no-total: duration omitted" in joined
    assert "estimated: token metrics omitted: token counts are estimated" in joined
    assert "no-provenance: token metrics omitted" in joined
    assert "local: skipped: not an API run" in joined

    only_local = build_otlp_metrics([MetricsRun(make_record(started_at=STARTED_AT))])
    assert only_local.observation_count == 0
    assert only_local.payload == {"resourceMetrics": []}


def test_chunk_metrics_need_streamed_untimed_free_output(tmp_path: Path) -> None:
    record, evidence = api_run(tmp_path)

    truncated = copy.deepcopy(evidence)
    truncated["timeline"]["events_truncated"] = True
    export = build_otlp_metrics([MetricsRun(record, truncated)])
    assert TTFC in metrics_of(export)
    assert PER_CHUNK not in metrics_of(export)

    silent = copy.deepcopy(evidence)
    silent["timeline"]["events"] = [
        e for e in silent["timeline"]["events"] if e["kind"] == "metadata"
    ]
    silent["timeline"]["first_content_token_offset_ms"] = None
    export = build_otlp_metrics([MetricsRun(record, silent)])
    assert TTFC not in metrics_of(export)
    assert PER_CHUNK not in metrics_of(export)
    assert any("no output chunk arrived" in note for note in export.notes)

    unary = copy.deepcopy(evidence)
    unary["plan"]["request_parameters"]["stream"] = False
    tools = {**evidence, "finish_reason": "tool_calls"}
    for variant in (unary, tools):
        export = build_otlp_metrics([MetricsRun(record, variant)])
        assert TTFC not in metrics_of(export)
        assert DURATION in metrics_of(export)


def test_server_metrics_need_one_attributed_observation_and_a_checked_mapping(
    tmp_path: Path,
) -> None:
    record, evidence = api_run(tmp_path)

    contaminated = build_otlp_metrics(
        [MetricsRun(record, evidence, server_evidence(requests_after=5))]
    )
    assert SERVER_TTFT not in metrics_of(contaminated)
    assert any("not attributable (contaminated)" in n for n in contaminated.notes)

    mean_window = server_evidence(requests_after=3)
    ttft = mean_window["highlights"]["time_to_first_token"]
    mean_window["highlights"]["time_to_first_token"] = {**ttft, "statistic": "mean"}
    mean_only = build_otlp_metrics([MetricsRun(record, evidence, mean_window)])
    assert SERVER_TTFT not in metrics_of(mean_only)
    assert any("only a mean is known" in n for n in mean_only.notes)

    family = "vllm:request_time_per_output_token_seconds"
    zero = build_otlp_metrics(
        [
            MetricsRun(
                record,
                evidence,
                vllm_evidence(time_per_output_token=tpot_highlight(0.0, family)),
            )
        ]
    )
    assert SERVER_TPOT not in metrics_of(zero)
    assert any("records 0 when only one token" in n for n in zero.notes)
    measured = build_otlp_metrics(
        [
            MetricsRun(
                record,
                evidence,
                vllm_evidence(time_per_output_token=tpot_highlight(30.0, family)),
            )
        ]
    )
    (tpot,) = points(measured, SERVER_TPOT)
    assert tpot["sum"] == pytest.approx(0.03)

    tgi = server_evidence(requests_after=3)
    tgi["runtime"] = "tgi"
    tgi["highlights"] = {
        "e2e_latency": {
            **tgi["highlights"]["e2e_latency"],
            "source": "tgi_request_duration",
        },
        "time_per_output_token": tpot_highlight(
            30.0, "tgi_request_mean_time_per_token_duration"
        ),
    }
    tgi_export = build_otlp_metrics([MetricsRun(record, evidence, tgi)])
    assert SERVER_DURATION in metrics_of(tgi_export)
    assert SERVER_TPOT not in metrics_of(tgi_export)

    wrong_source = vllm_evidence(
        time_per_output_token=tpot_highlight(30.0, "vllm:inter_token_latency_seconds")
    )
    export = build_otlp_metrics([MetricsRun(record, evidence, wrong_source)])
    assert SERVER_TPOT not in metrics_of(export)


def test_duplicate_runs_and_mismatched_evidence_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ExportError, match="more than once"):
        build_otlp_metrics([MetricsRun(api_record("a")), MetricsRun(api_record("a"))])
    record, evidence = api_run(tmp_path)
    with pytest.raises(ExportError, match="run_id"):
        build_otlp_metrics([MetricsRun(record, {**evidence, "run_id": "other"})])


def test_cli_reads_sibling_evidence_and_refuses_an_empty_export(
    tmp_path: Path,
) -> None:
    api_run(tmp_path / "runs")
    artifacts = tmp_path / "runs" / "artifacts"
    (artifacts / "server_metrics_evidence.json").write_text(
        json.dumps(server_evidence(requests_after=3)), encoding="utf-8"
    )
    output = tmp_path / "metrics.jsonl"

    code, out, err = run_cli(
        ["export", "otlp-metrics", str(tmp_path / "runs"), "--output", str(output)]
    )

    assert code == 0, err
    assert "from 1 runs written" in out
    payload = json.loads(output.read_text())
    names = {
        metric["name"]
        for metric in payload["resourceMetrics"][0]["scopeMetrics"][0]["metrics"]
    }
    assert {TTFC, PER_CHUNK, SERVER_TTFT} <= names

    local = tmp_path / "local"
    local.mkdir()
    (local / "record.json").write_text(
        json.dumps(make_record(started_at=STARTED_AT).to_dict()), encoding="utf-8"
    )
    empty = tmp_path / "empty.jsonl"
    code, out, err = run_cli(
        ["export", "otlp-metrics", str(local), "--output", str(empty)]
    )
    assert code == 1
    assert "not an API run" in out
    assert "Nothing to export" in err
    assert not empty.exists()
