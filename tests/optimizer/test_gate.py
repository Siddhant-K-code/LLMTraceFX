from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_schema import make_record

from llmtracefx.optimizer import cli
from llmtracefx.optimizer.gate import (
    GATE_REPORT_KIND,
    GateInputError,
    GateVerdict,
    evaluate_gate,
    load_gate_inputs,
    load_records,
)
from llmtracefx.optimizer.importers import (
    AIPERF_TOOL,
    GUIDELLM_TOOL,
    VLLM_BENCH_TOOL,
    ImportedBenchmark,
    import_aiperf_records_file,
    import_benchmark_file,
    read_imported_benchmark,
)
from llmtracefx.optimizer.schema import (
    CommandInfo,
    ErrorInfo,
    Measurement,
    MetricProvenance,
    OutcomeInfo,
    RuntimeInfo,
    TimingMetrics,
    TokenCounts,
)

FIXTURES = Path(__file__).parent / "fixtures" / "gate"


def ms(value: float, provenance: MetricProvenance = MetricProvenance.MEASURED_NATIVE):
    return Measurement(value=value, provenance=provenance, unit="ms")


def runs(prefix: str, totals: list[float], **overrides):
    return [
        make_record(
            run_id=f"{prefix}-{index}",
            timing=TimingMetrics(total=ms(total), decode=ms(total * 0.8)),
            **overrides,
        )
        for index, total in enumerate(totals)
    ]


BASE = [1000.0, 1010.0, 990.0]


def test_pass_when_within_allowed_regression():
    report = evaluate_gate(runs("b", BASE), runs("c", [1020.0, 1030.0, 1010.0]))
    assert report.verdict is GateVerdict.PASS
    assert report.exit_code == 0
    assert report.delta_pct == pytest.approx(0.02)
    assert "worse" in report.reason


def test_improvement_passes():
    report = evaluate_gate(runs("b", BASE), runs("c", [800.0, 805.0, 795.0]))
    assert report.verdict is GateVerdict.PASS
    assert report.delta_pct < 0
    assert "better" in report.reason


def test_regression_beyond_threshold_and_noise():
    report = evaluate_gate(runs("b", BASE), runs("c", [1200.0, 1210.0, 1190.0]))
    assert report.verdict is GateVerdict.REGRESSION
    assert report.exit_code == 1
    assert report.delta_pct == pytest.approx(0.2)


def test_regression_within_noise_is_inconclusive():
    report = evaluate_gate(
        runs("b", [600.0, 1000.0, 1400.0]), runs("c", [700.0, 1200.0, 1700.0])
    )
    assert report.verdict is GateVerdict.INCONCLUSIVE
    assert report.exit_code == 2
    assert "noise" in report.reason


def test_noisy_candidate_cannot_pass_by_chance():
    # Mean is 3.3% worse, inside the 5% allowance, but the spread is huge.
    report = evaluate_gate(
        runs("b", [1000.0, 1000.0, 1000.0]), runs("c", [10.0, 3000.0, 90.0])
    )
    assert report.verdict is GateVerdict.INCONCLUSIVE
    assert report.exit_code == 2
    assert report.delta_pct == pytest.approx(1 / 30)
    assert "larger than the allowed regression" in report.reason


def test_too_few_runs_is_inconclusive():
    report = evaluate_gate(runs("b", BASE[:2]), runs("c", BASE))
    assert report.verdict is GateVerdict.INCONCLUSIVE
    assert "baseline has 2 usable" in report.reason


def test_failed_runs_are_excluded_and_counted():
    failed = make_record(
        run_id="c-failed",
        outcome=OutcomeInfo(success=False),
        error=ErrorInfo(category="runtime", message="boom"),
    )
    report = evaluate_gate(runs("b", BASE), runs("c", BASE[:2]) + [failed])
    assert report.verdict is GateVerdict.INCONCLUSIVE
    assert report.excluded == ("candidate c-failed: run did not succeed",)


def test_different_workload_is_not_comparable():
    other = CommandInfo(argv=("x",), workload_hash="other")
    report = evaluate_gate(
        runs("b", BASE, command=CommandInfo(argv=("x",), workload_hash="w1")),
        runs("c", BASE, command=other),
    )
    assert report.verdict is GateVerdict.INCONCLUSIVE
    assert "not comparable" in report.reason


def test_runtime_version_and_config_may_differ():
    report = evaluate_gate(
        runs("b", BASE),
        runs(
            "c",
            BASE,
            runtime=RuntimeInfo(name="llama.cpp", version="b9999", backend="Metal"),
            command=CommandInfo(argv=("y",), config_hash="new"),
        ),
    )
    assert report.verdict is GateVerdict.PASS


def test_mixed_provenance_is_inconclusive():
    candidate = [
        make_record(
            run_id=f"c-{i}",
            timing=TimingMetrics(total=ms(v, MetricProvenance.MEASURED_WALL_CLOCK)),
        )
        for i, v in enumerate(BASE)
    ]
    report = evaluate_gate(runs("b", BASE), candidate)
    assert report.verdict is GateVerdict.INCONCLUSIVE
    assert "provenance" in report.reason


def test_decode_tokens_per_second_higher_is_better():
    # 200 tokens; decode = 0.8 * total. Slower decode means lower tokens/s.
    report = evaluate_gate(
        runs("b", BASE),
        runs("c", [1500.0, 1510.0, 1490.0]),
        metric="decode_tokens_per_second",
    )
    assert report.verdict is GateVerdict.REGRESSION
    assert report.higher_is_better
    assert report.unit == "tokens/s"
    assert report.provenance == "derived_from_measured_native"
    assert report.candidate_mean < report.baseline_mean


def test_rate_tolerates_different_generated_tokens():
    report = evaluate_gate(
        runs("b", BASE),
        runs("c", BASE, tokens=TokenCounts(input_tokens=50, generated_tokens=210)),
        metric="decode_tokens_per_second",
    )
    assert report.verdict is GateVerdict.PASS
    latency = evaluate_gate(
        runs("b", BASE),
        runs("c", BASE, tokens=TokenCounts(input_tokens=50, generated_tokens=210)),
    )
    assert latency.verdict is GateVerdict.INCONCLUSIVE


@pytest.mark.parametrize(
    "kwargs",
    [{"metric": "nope"}, {"max_regression": 1.0}, {"min_repetitions": 1}],
)
def test_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        evaluate_gate(runs("b", BASE), runs("c", BASE), **kwargs)


def test_report_serializes_and_renders_markdown(tmp_path):
    report = evaluate_gate(runs("b", BASE), runs("c", [1200.0, 1210.0, 1190.0]))
    out = tmp_path / "report.json"
    report.write_json(out)
    data = json.loads(out.read_text())
    assert data["kind"] == GATE_REPORT_KIND
    assert data["verdict"] == "regression"
    assert data["baseline"]["run_ids"] == ["b-0", "b-1", "b-2"]
    markdown = report.to_markdown()
    assert "perf gate: regression" in markdown
    assert "| baseline | 3 |" in markdown


def test_load_records_expands_directories(tmp_path):
    for record in runs("b", BASE):
        record.write_json(tmp_path / record.run_id / "record.json")
    assert [r.run_id for r in load_records([tmp_path])] == ["b-0", "b-1", "b-2"]
    with pytest.raises(GateInputError):
        load_records([tmp_path / "missing.json"])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(GateInputError, match="no record.json"):
        load_records([empty])


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [("candidate-ok", 0), ("candidate-regressed", 1)],
)
def test_cli_against_committed_fixtures(tmp_path, capsys, candidate, expected):
    summary = tmp_path / "summary.md"
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "gate",
                "--baseline",
                str(FIXTURES / "baseline"),
                "--candidate",
                str(FIXTURES / candidate),
                "--output",
                str(output),
                "--summary-markdown",
                str(summary),
            ]
        )
    assert exc.value.code == expected
    assert json.loads(output.read_text())["verdict"] in {"pass", "regression"}
    assert "LLMTraceFX perf gate" in summary.read_text()
    assert capsys.readouterr().out.startswith(("pass:", "regression:"))


def test_cli_bad_input_exits_inconclusive(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "gate",
                "--baseline",
                str(tmp_path / "nope.json"),
                "--candidate",
                str(tmp_path / "nope.json"),
            ]
        )
    assert exc.value.code == 2
    assert "gate:" in capsys.readouterr().err


IMPORTED_FIXTURES = FIXTURES / "imported"


def imported(prefix: str, p99s: list[float], edit=None) -> list[ImportedBenchmark]:
    source = Path(__file__).parent / "fixtures" / "importers" / "vllm_default.json"
    template = import_benchmark_file(source, VLLM_BENCH_TOOL).to_dict()
    items = []
    for index, p99 in enumerate(p99s):
        data = json.loads(json.dumps(template))
        data["source"]["filename"] = f"{prefix}-{index}.json"
        data["source"]["sha256"] = f"{prefix}{index}".encode().hex().ljust(64, "0")
        data["metrics"]["ttft_ms"]["stats"]["p99"]["value"] = p99
        if edit is not None:
            edit(index, data)
        items.append(ImportedBenchmark.from_dict(data))
    return items


IMPORTED_BASE = [230.0, 233.0, 236.0]


def test_imported_pass_and_regression():
    ok = evaluate_gate(
        imported("b", IMPORTED_BASE),
        imported("c", [231.0, 234.0, 237.0]),
        metric="ttft_ms.p99",
    )
    assert ok.verdict is GateVerdict.PASS
    assert ok.input_kind == "imported_benchmark"
    assert ok.source_tool == VLLM_BENCH_TOOL
    assert ok.provenance == "imported_derived"
    assert ok.unit == "ms"
    assert [s["filename"] for s in ok.sources[:1]] == ["b-0.json"]
    assert "did not measure them" in ok.to_markdown()
    assert ok.to_dict()["input_kind"] == "imported_benchmark"

    bad = evaluate_gate(
        imported("b", IMPORTED_BASE),
        imported("c", [270.0, 273.0, 276.0]),
        metric="ttft_ms.p99",
    )
    assert bad.verdict is GateVerdict.REGRESSION


def test_imported_throughput_is_higher_is_better():
    def rate(value):
        def edit(index, data):
            stats = data["metrics"]["output_throughput"]["stats"]
            stats["value"]["value"] = value + index

        return edit

    report = evaluate_gate(
        imported("b", IMPORTED_BASE, rate(500.0)),
        imported("c", IMPORTED_BASE, rate(400.0)),
        metric="output_throughput.value",
    )
    assert report.higher_is_better
    assert report.verdict is GateVerdict.REGRESSION


def server_imported(prefix, p99s, edit=None):
    source = FIXTURES.parent / "importers" / "vllm_server_metrics.json"
    template = import_benchmark_file(source, VLLM_BENCH_TOOL).to_dict()
    items = []
    for index, p99 in enumerate(p99s):
        data = json.loads(json.dumps(template))
        data["source"]["filename"] = f"{prefix}-{index}.json"
        data["source"]["sha256"] = f"{prefix}{index}".encode().hex().ljust(64, "0")
        data["metrics"]["server_ttft_ms"]["stats"]["p99"]["value"] = p99
        if edit is not None:
            edit(index, data)
        items.append(ImportedBenchmark.from_dict(data))
    return items


def test_imported_server_ttft_gates_with_its_own_provenance():
    ok = evaluate_gate(
        server_imported("b", [210.0, 212.0, 214.0]),
        server_imported("c", [211.0, 213.0, 215.0]),
        metric="server_ttft_ms.p99",
    )
    assert ok.verdict is GateVerdict.PASS
    assert ok.provenance == "derived"
    assert "timings the server reported" in ok.to_markdown()

    bad = evaluate_gate(
        server_imported("b", [210.0, 212.0, 214.0]),
        server_imported("c", [250.0, 252.0, 254.0]),
        metric="server_ttft_ms.p99",
    )
    assert bad.verdict is GateVerdict.REGRESSION


def test_imported_server_ttft_needs_every_completed_request(tmp_path):
    source = FIXTURES.parent / "importers" / "vllm_server_metrics.json"
    raw = json.loads(source.read_text())
    raw["server_ttfts"][0] = None
    (tmp_path / "partial.json").write_text(json.dumps(raw))
    seven = import_benchmark_file(tmp_path / "partial.json", VLLM_BENCH_TOOL)

    def partial(index, data):
        if index == 0:
            data["metrics"]["server_ttft_ms"] = seven.to_dict()["metrics"][
                "server_ttft_ms"
            ]

    report = evaluate_gate(
        server_imported("b", [210.0, 212.0, 214.0]),
        server_imported("c", [211.0, 213.0, 215.0], partial),
        metric="server_ttft_ms.p99",
    )
    assert report.verdict is GateVerdict.INCONCLUSIVE
    assert any("7 of 8 completed" in reason for reason in report.excluded)

    def failed(index, data):
        if index == 0:
            data["totals"]["failed_requests"]["value"] = 1

    report = evaluate_gate(
        server_imported("b", [210.0, 212.0, 214.0]),
        server_imported("c", [211.0, 213.0, 215.0], failed),
        metric="server_ttft_ms.p99",
    )
    assert any("1 request(s) failed" in reason for reason in report.excluded)
    # Files from vLLM runs without per-request metrics have no server values.
    missing = evaluate_gate(
        imported("b", IMPORTED_BASE),
        imported("c", IMPORTED_BASE),
        metric="server_ttft_ms.p99",
    )
    assert missing.verdict is GateVerdict.INCONCLUSIVE
    assert any("no server_ttft_ms" in reason for reason in missing.excluded)


def test_imported_refuses_mixing_with_records():
    with pytest.raises(GateInputError, match="cannot be mixed"):
        evaluate_gate(runs("b", BASE), imported("c", IMPORTED_BASE))


def test_imported_refuses_mixed_tools_and_duplicate_files():
    aiperf = Path(__file__).parent / "fixtures" / "importers" / "aiperf_legacy.json"
    other = import_benchmark_file(aiperf, AIPERF_TOOL)
    with pytest.raises(GateInputError, match="different tools"):
        evaluate_gate(imported("b", IMPORTED_BASE), [other] * 3, metric="x.p99")

    base = imported("b", IMPORTED_BASE)
    with pytest.raises(GateInputError, match="more than once"):
        evaluate_gate(base, [base[0], *imported("c", [1.0, 2.0])])


@pytest.mark.parametrize("metric", ["total", "ttft_ms", "ttft_ms.p95", "nope.p99"])
def test_imported_metric_must_name_a_known_metric_and_stat(metric):
    report_or_error = None
    try:
        report_or_error = evaluate_gate(
            imported("b", IMPORTED_BASE), imported("c", IMPORTED_BASE), metric=metric
        )
    except ValueError as exc:
        report_or_error = exc
    if metric == "ttft_ms.p95":
        # A percentile the run did not save is excluded, not guessed.
        assert report_or_error.verdict is GateVerdict.INCONCLUSIVE
        assert any("no ttft_ms.p95" in e for e in report_or_error.excluded)
    else:
        assert isinstance(report_or_error, ValueError)
        assert "ttft_ms.p99" in str(report_or_error)


def test_imported_statistic_may_contain_a_dot():
    def p999(index, data):
        stats = data["metrics"]["ttft_ms"]["stats"]
        stats["p99.9"] = dict(stats["p99"], source_key="p99.9_ttft_ms")

    report = evaluate_gate(
        imported("b", IMPORTED_BASE, p999),
        imported("c", IMPORTED_BASE, p999),
        metric="ttft_ms.p99.9",
    )
    assert report.verdict is GateVerdict.PASS


def test_imported_different_load_is_not_comparable():
    def faster(index, data):
        data["load"]["request_rate"] = 4.0

    report = evaluate_gate(
        imported("b", IMPORTED_BASE),
        imported("c", IMPORTED_BASE, faster),
        metric="ttft_ms.p99",
    )
    assert report.verdict is GateVerdict.INCONCLUSIVE
    assert "load settings" in report.reason


def test_imported_runs_with_failures_or_unknowns_are_excluded():
    def flaky(index, data):
        if index == 0:
            data["totals"]["failed_requests"]["value"] = 1
        if index == 1:
            del data["totals"]["failed_requests"]

    report = evaluate_gate(
        imported("b", IMPORTED_BASE),
        imported("c", IMPORTED_BASE, flaky),
        metric="ttft_ms.p99",
    )
    assert report.verdict is GateVerdict.INCONCLUSIVE
    assert len(report.excluded) == 2
    assert any("failed" in reason for reason in report.excluded)


def test_load_gate_inputs_finds_imported_artifacts(tmp_path):
    for item in imported("b", IMPORTED_BASE):
        (tmp_path / item.source.filename).write_text(item.to_json())
    (tmp_path / "notes.json").write_text('{"other": true}')
    raw = Path(__file__).parent / "fixtures" / "importers" / "vllm_default.json"
    (tmp_path / "raw.json").write_text(raw.read_text())
    found = load_gate_inputs([tmp_path])
    assert [item.source.filename for item in found] == [
        "b-0.json",
        "b-1.json",
        "b-2.json",
    ]
    with pytest.raises(GateInputError, match="import vllm-bench"):
        load_gate_inputs([tmp_path / "raw.json"])
    (tmp_path / "deep.json").write_text("[" * 100000 + "]" * 100000)
    with pytest.raises(GateInputError, match="recursion"):
        load_gate_inputs([tmp_path / "deep.json"])
    assert len(load_gate_inputs([FIXTURES / "baseline"])) == 3


def guidellm_runs(prefix: str, p99s: list[float], benchmark: int = 0):
    source = Path(__file__).parent / "fixtures" / "importers" / "guidellm_v2.json"
    template = import_benchmark_file(source, GUIDELLM_TOOL, benchmark=benchmark)
    items = []
    for index, p99 in enumerate(p99s):
        data = template.to_dict()
        data["source"]["filename"] = f"{prefix}-{index}.json"
        data["source"]["sha256"] = f"{prefix}{index}".encode().hex().ljust(64, "0")
        stats = data["metrics"]["time_to_first_token_ms"]["stats"]
        stats["p99"]["value"] = p99
        items.append(ImportedBenchmark.from_dict(data))
    return items


def test_guidellm_imports_gate_and_exclude_errored_runs():
    report = evaluate_gate(
        guidellm_runs("b", [71.0, 72.0, 73.0]),
        guidellm_runs("c", [90.0, 91.0, 92.0]),
        metric="time_to_first_token_ms.p99",
    )
    assert report.verdict is GateVerdict.REGRESSION
    assert report.source_tool == GUIDELLM_TOOL
    assert report.unit == "ms"

    # Benchmark 1 had one errored request, so every run is excluded.
    errored = evaluate_gate(
        guidellm_runs("b", [71.0, 72.0, 73.0], benchmark=1),
        guidellm_runs("c", [71.0, 72.0, 73.0], benchmark=1),
        metric="time_to_first_token_ms.p99",
    )
    assert errored.verdict is GateVerdict.INCONCLUSIVE
    assert any("failed" in reason for reason in errored.excluded)

    with pytest.raises(GateInputError, match="different tools"):
        evaluate_gate(
            guidellm_runs("b", [71.0, 72.0, 73.0]),
            imported("c", IMPORTED_BASE),
            metric="time_to_first_token_ms.p99",
        )

    raw = Path(__file__).parent / "fixtures" / "importers" / "guidellm_v2.json"
    with pytest.raises(GateInputError, match="import guidellm"):
        load_gate_inputs([raw])


def test_committed_imported_fixtures_match_their_sources():
    sources = Path(__file__).parent / "fixtures" / "importers" / "gate"
    artifacts = sorted(IMPORTED_FIXTURES.rglob("*.json"))
    assert len(artifacts) == 9
    for artifact in artifacts:
        source = sources / artifact.relative_to(IMPORTED_FIXTURES)
        fresh = import_benchmark_file(source, VLLM_BENCH_TOOL).to_dict()
        committed = read_imported_benchmark(artifact).to_dict()
        fresh.pop("importer")
        committed.pop("importer")
        assert committed == fresh


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [("candidate-ok", 0), ("candidate-regressed", 1)],
)
def test_cli_gate_on_imported_fixtures(tmp_path, capsys, candidate, expected):
    output = tmp_path / "report.json"
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "gate",
                "--baseline",
                str(IMPORTED_FIXTURES / "baseline"),
                "--candidate",
                str(IMPORTED_FIXTURES / candidate),
                "--metric",
                "ttft_ms.p99",
                "--output",
                str(output),
            ]
        )
    assert exc.value.code == expected
    report = json.loads(output.read_text())
    assert report["input_kind"] == "imported_benchmark"
    assert report["source_tool"] == VLLM_BENCH_TOOL
    assert capsys.readouterr().out.startswith(("pass:", "regression:"))


def records_imported(prefix: str, values: list[float], slo=None, edit=None):
    importers = Path(__file__).parent / "fixtures" / "importers"
    template = import_aiperf_records_file(
        importers / "aiperf_records_summary.json",
        importers / "aiperf_records.jsonl",
        slo=slo or {"ttft_ms": 60.0},
    ).to_dict()
    items = []
    for index, value in enumerate(values):
        data = json.loads(json.dumps(template))
        data["source"]["sha256"] = f"{prefix}{index}".encode().hex().ljust(64, "0")
        data["metrics"]["goodput"]["stats"]["value"]["value"] = value
        data["metrics"]["time_to_first_token"]["stats"]["p99"]["value"] = value
        if edit is not None:
            edit(index, data)
        items.append(ImportedBenchmark.from_dict(data))
    return items


def test_aiperf_records_gate_goodput_and_refusals(tmp_path):
    base = records_imported("b", [3.0, 3.02, 3.04])
    report = evaluate_gate(
        base, records_imported("c", [2.5, 2.52, 2.54]), metric="goodput.value"
    )
    assert report.verdict is GateVerdict.REGRESSION
    assert report.higher_is_better
    assert report.provenance == "derived"
    assert "computed by LLMTraceFX from AIPerf's per-request" in report.to_markdown()

    stricter = records_imported("c", [3.0, 3.02, 3.04], slo={"ttft_ms": 50.0})
    other_slo = evaluate_gate(base, stricter, metric="goodput.value")
    assert other_slo.verdict is GateVerdict.INCONCLUSIVE
    assert "SLO targets" in other_slo.reason

    summary = (
        Path(__file__).parent / "fixtures" / "importers" / "aiperf_records_summary.json"
    )
    with pytest.raises(GateInputError, match="different tools"):
        evaluate_gate(
            base, [import_benchmark_file(summary, AIPERF_TOOL)] * 3, metric="x.p99"
        )

    raw = tmp_path / "profile_export.jsonl"
    raw.write_text((summary.parent / "aiperf_records.jsonl").read_text())
    with pytest.raises(GateInputError, match="--records"):
        load_gate_inputs([raw])


def test_aiperf_records_goodput_gate_counts_real_failures(tmp_path):
    importers = Path(__file__).parent / "fixtures" / "importers"
    rows = [
        json.loads(line)
        for line in (importers / "aiperf_records.jsonl").read_text().splitlines()
    ]
    profiling = [r for r in rows if r["metadata"]["benchmark_phase"] == "profiling"]
    for row in profiling[:4]:
        row["error"] = {"code": 500, "type": "HTTPError", "message": "boom"}
        row["metrics"] = {"error_isl": row["metrics"]["input_sequence_length"]}
    summary = json.loads((importers / "aiperf_records_summary.json").read_text())
    summary["request_count"]["avg"] = 16
    summary["error_request_count"] = {"unit": "requests", "avg": 4}
    summary["error_summary"] = [{"count": 4}]
    summary["total_isl"]["avg"] -= 4 * 512
    summary["total_osl"]["avg"] -= 4 * 128
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "records.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )
    slo = {"ttft_ms": 100.0}
    failed = import_aiperf_records_file(
        tmp_path / "summary.json", tmp_path / "records.jsonl", slo=slo
    ).to_dict()
    assert failed["metrics"]["goodput_fraction"]["stats"]["value"]["value"] == 0.8

    def runs(prefix, data, values):
        items = []
        for index, value in enumerate(values):
            copy = json.loads(json.dumps(data))
            copy["source"]["sha256"] = f"{prefix}{index}".encode().hex().ljust(64, "0")
            copy["metrics"]["goodput_fraction"]["stats"]["value"]["value"] = value
            items.append(ImportedBenchmark.from_dict(copy))
        return items

    healthy = import_aiperf_records_file(
        importers / "aiperf_records_summary.json",
        importers / "aiperf_records.jsonl",
        slo=slo,
    ).to_dict()
    base = runs("b", healthy, [1.0, 0.99, 0.98])
    report = evaluate_gate(
        base, runs("c", failed, [0.8, 0.79, 0.78]), metric="goodput_fraction.value"
    )
    assert report.verdict is GateVerdict.REGRESSION
    assert not report.excluded

    latency = evaluate_gate(
        base, runs("c", failed, [0.8, 0.79, 0.78]), metric="time_to_first_token.p99"
    )
    assert latency.verdict is GateVerdict.INCONCLUSIVE
    assert any("this value covers successful" in e for e in latency.excluded)
