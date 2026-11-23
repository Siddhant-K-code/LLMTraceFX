from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from llmtracefx.optimizer import cli
from llmtracefx.optimizer.importers import (
    AIPERF_RECORDS_TOOL,
    AIPERF_TOOL,
    DERIVED,
    GUIDELLM_TOOL,
    IMPORTED_DERIVED,
    IMPORTED_REPORTED,
    SERVER_REPORTED,
    VLLM_BENCH_TOOL,
    BenchmarkImportError,
    ImportedArtifactError,
    ImportedBenchmark,
    import_aiperf_records_file,
    import_benchmark_file,
    read_imported_benchmark,
)

FIXTURES = Path(__file__).parent / "fixtures" / "importers"


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text())


def write(tmp_path: Path, data: object, name: str = "source.json") -> Path:
    path = tmp_path / name
    path.write_text(data if isinstance(data, str) else json.dumps(data))
    return path


def test_vllm_default_maps_keys_units_and_provenance():
    path = FIXTURES / "vllm_default.json"
    imported = import_benchmark_file(path, VLLM_BENCH_TOOL)

    assert imported.source.tool == VLLM_BENCH_TOOL
    assert imported.source.tool_version is None
    assert imported.source.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert imported.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert dict(imported.load) == {
        "num_prompts": 8,
        "request_rate": "inf",
        "burstiness": 1.0,
        "max_concurrency": None,
    }
    ttft = imported.metrics["ttft_ms"]
    assert ttft.unit == "ms"
    assert ttft.provenance == IMPORTED_DERIVED
    assert sorted(ttft.stats) == ["mean", "median", "p99", "std"]
    assert ttft.stats["p99"].value == 233.517
    assert ttft.stats["p99"].source_key == "p99_ttft_ms"
    assert imported.metrics["request_throughput"].stats["value"].value == 1.948368
    assert "request_goodput" not in imported.metrics  # null without --goodput
    assert "e2el_ms" not in imported.metrics  # not selected, so not invented
    assert imported.totals["failed_requests"].value == 0
    assert imported.totals["duration"].unit == "s"
    assert {t.provenance for t in imported.totals.values()} == {IMPORTED_REPORTED}
    assert "latencies" in imported.unmapped_keys
    assert "max_output_tokens_per_s" in imported.unmapped_keys


def test_vllm_custom_percentiles_and_metadata():
    imported = import_benchmark_file(FIXTURES / "vllm_full.json", VLLM_BENCH_TOOL)

    assert sorted(imported.metrics["e2el_ms"].stats) == [
        "mean",
        "median",
        "p50",
        "p90",
        "p99",
        "std",
    ]
    assert imported.metrics["request_goodput"].stats["value"].value == 1.487505
    assert dict(imported.source.user_metadata) == {
        "commit": "3f2a9c1",
        "gpu": "H100-80GB",
    }
    assert imported.load["max_concurrency"] == 4
    assert imported.run["label"] == "nightly"


def test_aiperf_legacy_layout():
    imported = import_benchmark_file(FIXTURES / "aiperf_legacy.json", AIPERF_TOOL)

    assert imported.source.tool_version is None
    assert imported.source.schema_version is None
    assert imported.model_id == "Qwen/Qwen3-0.6B"
    assert imported.load["concurrency"] == 1
    assert imported.load["request_count"] == 64
    assert imported.load_unresolved == ()
    assert imported.totals["completed_requests"].value == 64
    # error_request_count is null and error_summary is empty: zero is stated.
    assert imported.totals["failed_requests"].source_key == "error_summary"
    ttft = imported.metrics["time_to_first_token"]
    assert ttft.stats["p99"].source_key == "time_to_first_token.p99"
    assert "count" not in ttft.stats and "sum" not in ttft.stats
    assert any("TPOT" in note for note in imported.metrics["inter_token_latency"].notes)


def test_aiperf_v1_7_layout_and_no_secrets():
    path = FIXTURES / "aiperf_v1_7.json"
    imported = import_benchmark_file(path, AIPERF_TOOL)

    assert imported.source.tool_version == "0.9.0"
    assert imported.source.schema_version == "1.7"
    assert dict(imported.load) == {
        "layout": "phases",
        "type": "concurrency",
        "concurrency": 4,
        "requests": 200,
    }
    assert imported.endpoint["type"] == "chat"
    assert "sk-" not in imported.to_json()


@pytest.mark.parametrize(
    ("fixture", "tool"),
    [
        ("vllm_default.json", VLLM_BENCH_TOOL),
        ("vllm_full.json", VLLM_BENCH_TOOL),
        ("vllm_server_metrics.json", VLLM_BENCH_TOOL),
        ("aiperf_legacy.json", AIPERF_TOOL),
        ("aiperf_v1_7.json", AIPERF_TOOL),
        ("guidellm_v2.json", GUIDELLM_TOOL),
    ],
)
def test_artifact_round_trips(tmp_path, fixture, tool):
    benchmark = 0 if tool == GUIDELLM_TOOL else None
    imported = import_benchmark_file(FIXTURES / fixture, tool, benchmark=benchmark)
    out = write(tmp_path, imported.to_json(), "artifact.json")
    assert read_imported_benchmark(out).to_dict() == imported.to_dict()


def test_artifact_reader_refuses_tampered_labels():
    data = import_benchmark_file(FIXTURES / "vllm_default.json", VLLM_BENCH_TOOL)
    good = data.to_dict()
    for path, value in (
        (("metrics", "ttft_ms", "provenance"), "measured_native"),
        (("metrics", "ttft_ms", "direction"), "higher_is_better"),
        (("totals", "completed_requests", "provenance"), "derived"),
        (("source", "tool"), "genai-perf"),
    ):
        bad = json.loads(json.dumps(good))
        target = bad
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ImportedArtifactError):
            ImportedBenchmark.from_dict(bad)


def vllm(**changes):
    data = load("vllm_default.json")
    for key, value in changes.items():
        if value is ...:
            del data[key]
        else:
            data[key] = value
    return data


def server(**changes):
    data = load("vllm_server_metrics.json")
    data.update(changes)
    return data


def aiperf(**changes):
    data = load("aiperf_v1_7.json")
    data.update(changes)
    return data


def guidellm(edit=None):
    data = load("guidellm_v2.json")
    if edit is not None:
        edit(data)
    return data


def one_benchmark(**changes):
    def edit(data):
        del data["benchmarks"][1]
        item = data["benchmarks"][0]
        for key, value in changes.items():
            if key == "model":
                item["config"]["backend"]["model"] = value
            elif key == "request_totals":
                item["metrics"]["request_totals"] = value
            else:
                item[key] = value

    return edit


def guidellm_v1(data):
    """Reshape the v2 fixture the way GuideLLM 0.5 and 0.6 wrote reports."""
    data["metadata"].update(version=1, guidellm_version="0.6.0")
    data["args"] = data.pop("config")
    data["args"]["backend"] = "openai_http"
    data["args"]["backend_kwargs"] = {
        "target": "http://localhost:8000",
        "model": "Qwen/Qwen3-0.6B",
        "request_format": None,
    }
    del data["conclusions"]
    for item in data["benchmarks"]:
        config = item["config"]
        config["sample_requests"] = config.pop("sample_size")
        config["backend"] = {
            "target": "http://localhost:8000",
            "model": "Qwen/Qwen3-0.6B",
            "timeout": 60.0,
            "http2": True,
        }
        config["constraints"] = {
            "max_number": {"type_": "max_number", "max_num": 64, "current_index": 0}
        }
        del item["metrics"]["time_to_first_output_token_ms"]


def two_phase_profiling():
    data = aiperf()
    phase = dict(data["input_config"]["phases"][1], name="second")
    data["input_config"]["phases"].append(phase)
    return data


@pytest.mark.parametrize(
    ("tool", "source", "message"),
    [
        (VLLM_BENCH_TOOL, "[]", "JSON object"),
        (VLLM_BENCH_TOOL, '{"a": NaN}', "not valid JSON"),
        (VLLM_BENCH_TOOL, "[" * 100000 + "]" * 100000, "nested too deeply"),
        (VLLM_BENCH_TOOL, '{"a": 1}\n{"a": 2}', "more than one JSON document"),
        (VLLM_BENCH_TOOL, vllm(completed=0), "no request completed"),
        (VLLM_BENCH_TOOL, vllm(completed="8"), "completed must be"),
        (VLLM_BENCH_TOOL, vllm(num_prompts="8"), "num_prompts must be"),
        (VLLM_BENCH_TOOL, vllm(request_rate="fast"), "request_rate must be"),
        (VLLM_BENCH_TOOL, vllm(p99_ttft_ms="slow"), "p99_ttft_ms must be"),
        (VLLM_BENCH_TOOL, vllm(total_input_sequences=8), "pooling"),
        (VLLM_BENCH_TOOL, '{"p99_ttft_ms": 1' + "0" * 400 + "}", "missing"),
        (VLLM_BENCH_TOOL, vllm(total_input_tokens=10**400), "total_input_tokens"),
        (VLLM_BENCH_TOOL, vllm(p99_ttft_ms=10**400), "p99_ttft_ms must be"),
        (VLLM_BENCH_TOOL, vllm(model_id=...), "missing model_id"),
        (VLLM_BENCH_TOOL, load("aiperf_v1_7.json"), "looks like an AIPerf"),
        (VLLM_BENCH_TOOL, server(server_ttfts=0.2), "server_ttfts must be a list"),
        (VLLM_BENCH_TOOL, server(server_ttfts=[0.2] * 7), "7 entries.*sent \\(8\\)"),
        (
            VLLM_BENCH_TOOL,
            server(server_queue_times=[0.0] * 7 + [-0.1]),
            r"server_queue_times\[7\] must be",
        ),
        (
            VLLM_BENCH_TOOL,
            server(server_ttfts=["0.2"] * 8),
            r"server_ttfts\[0\] must be",
        ),
        (AIPERF_TOOL, load("vllm_default.json"), "looks like a `vllm bench"),
        (AIPERF_TOOL, aiperf(was_cancelled=True), "cancelled"),
        (
            AIPERF_TOOL,
            aiperf(is_complete=False, incomplete_reason="timeout"),
            "incomplete: timeout",
        ),
        (AIPERF_TOOL, aiperf(schema_version="2.0"), "not supported"),
        (AIPERF_TOOL, aiperf(timeslices=[]), "timeslice"),
        (
            AIPERF_TOOL,
            {"schema_version": "1.0", "aiperf_version": "0.9.0", "metrics": {}},
            "aggregate",
        ),
        (
            AIPERF_TOOL,
            aiperf(request_count={"unit": "requests", "avg": 0.0}),
            "no successful requests",
        ),
        (AIPERF_TOOL, two_phase_profiling(), "2 phases"),
        (GUIDELLM_TOOL, guidellm(), r"2 benchmarks.*0: concurrent streams=4; 1: "),
        (
            GUIDELLM_TOOL,
            guidellm(lambda d: d["metadata"].update(version=3)),
            "version 3 is not supported",
        ),
        (
            GUIDELLM_TOOL,
            guidellm(lambda d: d["metadata"].update(version=True)),
            "not supported",
        ),
        (GUIDELLM_TOOL, guidellm(lambda d: d.pop("metadata")), "0.4 and earlier"),
        (GUIDELLM_TOOL, guidellm(lambda d: d.update(benchmarks=[])), "no benchmarks"),
        (GUIDELLM_TOOL, guidellm(one_benchmark(model=None)), "names no model"),
        (
            GUIDELLM_TOOL,
            guidellm(one_benchmark(type_="generative_embeddings")),
            "only generative",
        ),
        (
            GUIDELLM_TOOL,
            guidellm(one_benchmark(request_totals={"successful": 0})),
            "errored must be",
        ),
        (VLLM_BENCH_TOOL, load("guidellm_v2.json"), "use `import guidellm`"),
        (GUIDELLM_TOOL, load("aiperf_v1_7.json"), "looks like an AIPerf"),
    ],
)
def test_refuses_bad_inputs(tmp_path, tool, source, message):
    with pytest.raises(BenchmarkImportError, match=message):
        import_benchmark_file(write(tmp_path, source), tool)


def test_aiperf_refuses_runs_that_mix_models(tmp_path):
    data = aiperf()
    data["input_config"]["models"]["items"].append({"name": "other/model"})
    with pytest.raises(BenchmarkImportError, match="2 models"):
        import_benchmark_file(write(tmp_path, data), AIPERF_TOOL)


def test_unknowns_are_recorded_not_guessed(tmp_path):
    data = aiperf()
    del data["error_summary"]
    del data["input_config"]["phases"][1]["requests"]
    imported = import_benchmark_file(write(tmp_path, data), AIPERF_TOOL)
    assert "failed_requests" not in imported.totals
    assert imported.load_unresolved == ("requests, duration or sessions",)

    no_failed = vllm(failed=..., burstiness=...)
    imported = import_benchmark_file(write(tmp_path, no_failed), VLLM_BENCH_TOOL)
    assert "failed_requests" not in imported.totals
    assert imported.load_unresolved == ("burstiness",)


def test_cli_import_writes_artifact(tmp_path, capsys):
    out = tmp_path / "out" / "artifact.json"
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "import",
                "aiperf",
                str(FIXTURES / "aiperf_legacy.json"),
                "--output",
                str(out),
            ]
        )
    assert exc.value.code == 0
    assert read_imported_benchmark(out).source.tool == AIPERF_TOOL
    assert "time_to_first_token [ms]" in capsys.readouterr().out


def test_cli_import_refusal_exits_2(tmp_path, capsys):
    source = write(tmp_path, vllm(completed=0))
    with pytest.raises(SystemExit) as exc:
        cli.main(["import", "vllm-bench", str(source)])
    assert exc.value.code == 2
    assert "no request completed" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        cli.main(["import", "vllm-bench", str(source), "--output", str(source)])
    assert exc.value.code == 2


SPEC_KEYS = {
    "spec_decode_acceptance_rate": 62.5,
    "spec_decode_acceptance_length": 2.875,
    "spec_decode_num_drafts": 80,
    "spec_decode_draft_tokens": 240,
    "spec_decode_accepted_tokens": 150,
    "spec_decode_verified_tokens": 240,
    "spec_decode_per_position_acceptance_rates": [0.875, 0.625, 0.375],
}


def test_vllm_speculative_keys_map_to_the_speculative_section(tmp_path):
    source = write(tmp_path, vllm(**SPEC_KEYS))
    imported = import_benchmark_file(source, VLLM_BENCH_TOOL)

    spec = imported.speculative
    assert spec is not None and spec.status == "reported"
    rate = spec.values["acceptance_rate"]
    assert (rate.value, rate.unit, rate.provenance) == (
        62.5,
        "percent",
        IMPORTED_DERIVED,
    )
    assert spec.values["acceptance_length"].unit == "tokens/step"
    assert spec.values["drafts"].provenance == IMPORTED_REPORTED
    assert spec.per_position_acceptance_rate == (0.875, 0.625, 0.375)
    assert not set(SPEC_KEYS) & set(imported.unmapped_keys)
    assert not any(name.startswith("spec") for name in imported.metrics)
    out = write(tmp_path, imported.to_json(), "artifact.json")
    assert read_imported_benchmark(out).to_dict() == imported.to_dict()


def test_vllm_placeholder_acceptance_length_is_dropped(tmp_path):
    source = write(
        tmp_path,
        vllm(
            spec_decode_acceptance_rate=0.0,
            spec_decode_acceptance_length=0.0,
            spec_decode_num_drafts=0,
            spec_decode_draft_tokens=4,
            spec_decode_accepted_tokens=0,
            spec_decode_verified_tokens=0,
        ),
    )
    spec = import_benchmark_file(source, VLLM_BENCH_TOOL).speculative

    assert spec is not None
    assert "acceptance_length" not in spec.values
    assert any("placeholder" in note for note in spec.notes)


def test_vllm_without_speculative_keys_says_not_reported():
    spec = import_benchmark_file(
        FIXTURES / "vllm_default.json", VLLM_BENCH_TOOL
    ).speculative

    assert spec is not None and spec.status == "not_reported"
    assert spec.values == {} and spec.reason


def test_artifact_reader_refuses_relabelled_speculative_values(tmp_path):
    good = import_benchmark_file(
        write(tmp_path, vllm(**SPEC_KEYS)), VLLM_BENCH_TOOL
    ).to_dict()
    for path, value in (
        (("values", "acceptance_rate", "unit"), "ratio"),
        (("values", "drafts", "provenance"), IMPORTED_DERIVED),
        (("status",), "not_reported"),
    ):
        bad = json.loads(json.dumps(good))
        target = bad["speculative"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ImportedArtifactError):
            ImportedBenchmark.from_dict(bad)


def records() -> list[dict]:
    text = (FIXTURES / "aiperf_records.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines()]


def records_run(tmp_path, edit=None, *, lines=None, summary_edit=None, **slo):
    """Write a records file and a summary whose counts and totals match it."""
    rows = records()
    if edit is not None:
        edit(rows)
    profiling = [r for r in rows if r["metadata"]["benchmark_phase"] == "profiling"]
    ok = [r for r in profiling if r.get("error") is None]
    summary = load("aiperf_records_summary.json")
    summary["request_count"]["avg"] = max(len(ok), 1)  # keep the summary valid
    if len(ok) != len(profiling):
        summary["error_request_count"] = {
            "unit": "requests",
            "avg": len(profiling) - len(ok),
        }
        summary["error_summary"] = [{"count": len(profiling) - len(ok)}]
    for key, tag in (
        ("total_isl", "input_sequence_length"),
        ("total_osl", "output_sequence_length"),
    ):
        summary[key]["avg"] = sum(r["metrics"][tag]["value"] for r in ok)
    if summary_edit is not None:
        summary_edit(summary)
    summary_path = write(tmp_path, summary, "profile_export_aiperf.json")
    text = lines if lines is not None else "".join(json.dumps(r) + "\n" for r in rows)
    records_path = write(tmp_path, text, "profile_export.jsonl")
    return summary_path, records_path


def fail(row, message="Internal Server Error"):
    row["error"] = {"code": 500, "type": "HTTPError", "message": message}
    row["metrics"] = {"request_latency": row["metrics"]["request_latency"]}


def test_aiperf_records_compute_statistics_with_provenance():
    summary = FIXTURES / "aiperf_records_summary.json"
    source = FIXTURES / "aiperf_records.jsonl"
    imported = import_aiperf_records_file(summary, source)

    assert imported.source.tool == AIPERF_RECORDS_TOOL
    assert imported.source.tool_version == "0.9.0"
    assert imported.source.sha256 == hashlib.sha256(source.read_bytes()).hexdigest()
    assert imported.source.companion_filename == summary.name
    assert (
        imported.source.companion_sha256
        == hashlib.sha256(summary.read_bytes()).hexdigest()
    )
    assert imported.model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert imported.load["requests"] == 20
    ttft = imported.metrics["time_to_first_token"]
    # The two warmup records (95 ms and 96 ms) are left out.
    assert ttft.sample_count == 20
    assert ttft.stats["max"].value == 88.6
    assert ttft.stats["p50"].value == 51.1  # nearest rank: the 10th of 20
    assert ttft.stats["p90"].value == 71.4
    assert ttft.provenance == DERIVED
    assert ttft.sample_provenance == IMPORTED_REPORTED
    assert ttft.statistic == "nearest_rank"
    assert (ttft.intervals["p50"].lower, ttft.intervals["p50"].upper) == (47.3, 60.3)
    assert "needs at least 368 values" in ttft.intervals["p99"].withheld_reason
    assert ttft.stats["p99"].source_key == "metrics.time_to_first_token.value"
    assert imported.totals["failed_requests"].value == 0
    assert imported.totals["total_output_tokens"].value == 2560
    assert {t.provenance for t in imported.totals.values()} == {DERIVED}
    assert imported.metrics["request_throughput"].stats["value"].value == pytest.approx(
        20 / imported.totals["duration"].value
    )
    assert "goodput" not in imported.metrics  # no SLO, so not invented
    assert imported.unmapped_keys == ("metrics.inter_chunk_latency",)


def test_aiperf_records_round_trip_and_reader_refusals(tmp_path):
    imported = import_aiperf_records_file(
        FIXTURES / "aiperf_records_summary.json",
        FIXTURES / "aiperf_records.jsonl",
        slo={"ttft_ms": 60.0},
    )
    good = imported.to_dict()
    out = write(tmp_path, imported.to_json(), "artifact.json")
    assert read_imported_benchmark(out).to_dict() == good
    ttft = ("metrics", "time_to_first_token")
    for path, value in (
        ((*ttft, "provenance"), IMPORTED_DERIVED),
        ((*ttft, "sample_provenance"), "measured_native"),
        ((*ttft, "statistic"), "linear"),
        ((*ttft, "sample_count"), 0),
        ((*ttft, "confidence_intervals", "p50", "upper_rank"), 21),
        ((*ttft, "confidence_intervals", "p75"), {"lower": 1, "upper": 2}),
        (("totals", "completed_requests", "provenance"), IMPORTED_REPORTED),
        (("source", "companion"), None),
        (("slo", "comparison"), "<"),
        (("slo",), None),
    ):
        bad = json.loads(json.dumps(good))
        target = bad
        for key in path[:-1]:
            target = target[key]
        if value is None:
            del target[path[-1]]
        else:
            target[path[-1]] = value
        with pytest.raises(ImportedArtifactError):
            ImportedBenchmark.from_dict(bad)
    summary = import_benchmark_file(
        FIXTURES / "aiperf_v1_7.json", AIPERF_TOOL
    ).to_dict()
    for key, value in (
        ("slo", {"ttft_ms": 60.0, "comparison": "<="}),
        ("source", dict(summary["source"], companion=good["source"]["companion"])),
    ):
        with pytest.raises(ImportedArtifactError):
            ImportedBenchmark.from_dict(dict(summary, **{key: value}))
    sampled = json.loads(json.dumps(summary))
    sampled["metrics"]["time_to_first_token"]["sample_count"] = 200
    with pytest.raises(ImportedArtifactError):
        ImportedBenchmark.from_dict(sampled)


def test_aiperf_records_count_failures_and_goodput(tmp_path):
    def edit(rows):
        fail(rows[2])  # the first profiling record
        rows[3]["metadata"]["was_cancelled"] = True
        # One output token: no inter-token gap, so the TPOT target does not apply.
        rows[4]["metrics"]["output_sequence_length"]["value"] = 1
        del rows[4]["metrics"]["inter_token_latency"]

    summary, source = records_run(tmp_path, edit)
    # A cancelled record without an error counts as failed in both files.
    with pytest.raises(BenchmarkImportError, match="not from the same run"):
        import_aiperf_records_file(summary, source)

    def cancelled(rows):
        edit(rows)
        fail(rows[3], "cancelled")

    summary, source = records_run(tmp_path, cancelled)
    slo = {"ttft_ms": 60.0, "tpot_ms": 8.2}
    imported = import_aiperf_records_file(summary, source, slo=slo)
    assert imported.totals["completed_requests"].value == 18
    assert imported.totals["failed_requests"].value == 2
    assert imported.metrics["time_to_first_token"].sample_count == 18
    assert imported.metrics["inter_token_latency"].sample_count == 17
    rows = records()[2:]
    ok = [r for i, r in enumerate(rows) if i not in (0, 1)]
    met = sum(
        r["metrics"]["time_to_first_token"]["value"] <= 60
        and (r is rows[2] or r["metrics"]["inter_token_latency"]["value"] <= 8.2)
        for r in ok
    )
    fraction = imported.metrics["goodput_fraction"].stats["value"].value
    assert fraction == met / 20  # failures stay in the denominator
    assert dict(imported.slo) == slo


@pytest.mark.parametrize(
    ("edit", "lines", "message"),
    [
        (
            lambda rows: rows[5]["metadata"].pop("request_start_ns"),
            None,
            "request_start_ns is missing",
        ),
        (
            lambda rows: rows[5]["metadata"].update(benchmark_phase="cooldown"),
            None,
            "benchmark_phase",
        ),
        (
            lambda rows: rows[5]["metrics"]["time_to_first_token"].update(unit="s"),
            None,
            "unit 's' is not 'ms'",
        ),
        (
            lambda rows: rows[5]["metrics"]["request_latency"].update(value=-1),
            None,
            "finite number >= 0",
        ),
        (
            lambda rows: rows[5]["metrics"].pop("request_latency"),
            None,
            "no request_latency",
        ),
        (lambda rows: rows[5].update(error="boom"), None, "error must be an object"),
        (
            lambda rows: rows[5]["metadata"].update(was_cancelled="no"),
            None,
            "was_cancelled must be a bool",
        ),
        (
            lambda rows: rows[5]["metadata"].update(request_end_ns=1),
            None,
            "before request_start_ns",
        ),
        (
            lambda rows: rows[6]["metadata"].update(
                x_request_id=rows[5]["metadata"]["x_request_id"]
            ),
            None,
            "repeats x_request_id",
        ),
        (
            lambda rows: rows[6]["metadata"].update(
                phase_name="second", profiling_index=1
            ),
            None,
            "2 profiling phases",
        ),
        (
            lambda rows: [r["metadata"].update(benchmark_phase="warmup") for r in rows],
            None,
            "no profiling records",
        ),
        (lambda rows: [fail(r) for r in rows], None, "no successful profiling request"),
        (
            None,
            '{"metadata": {"request_start_ns": 1, "benchmark_phase": "x"}, "metadata": {}}\n',
            "duplicate key",
        ),
        (None, '{"metadata": {"request_start_ns": NaN}}\n', "line 1 is not valid JSON"),
        (None, "[1, 2]\n", "line 1 is not an AIPerf record"),
    ],
)
def test_aiperf_records_refuse_bad_records(tmp_path, edit, lines, message):
    summary, source = records_run(tmp_path, edit, lines=lines)
    with pytest.raises(BenchmarkImportError, match=message):
        import_aiperf_records_file(summary, source)


def test_aiperf_records_refuse_mismatched_or_swapped_files(tmp_path):
    summary, source = records_run(
        tmp_path, summary_edit=lambda s: s["total_osl"].update(avg=2559)
    )
    with pytest.raises(BenchmarkImportError, match="not from the same run"):
        import_aiperf_records_file(summary, source)
    with pytest.raises(BenchmarkImportError, match="different file"):
        import_aiperf_records_file(summary, summary)
    with pytest.raises(BenchmarkImportError, match="is an AIPerf summary"):
        import_aiperf_records_file(summary, FIXTURES / "aiperf_v1_7.json")
    with pytest.raises(BenchmarkImportError, match="--records"):
        import_aiperf_records_file(source, summary)
    with pytest.raises(BenchmarkImportError, match="per-request records"):
        import_benchmark_file(source, AIPERF_TOOL)


def test_aiperf_records_goodput_needs_complete_evidence(tmp_path):
    summary, source = records_run(
        tmp_path, lambda rows: rows[7]["metrics"].pop("time_to_first_token")
    )
    assert import_aiperf_records_file(summary, source, slo={"e2e_ms": 2000.0})
    with pytest.raises(BenchmarkImportError, match="streaming run"):
        import_aiperf_records_file(summary, source, slo={"ttft_ms": 1000.0})
    for bad in ({"ttft_ms": 0.0}, {"ttft_ms": float("nan")}, {"itl_ms": 5.0}):
        with pytest.raises(BenchmarkImportError, match="SLO target"):
            import_aiperf_records_file(summary, source, slo=bad)


def test_cli_import_aiperf_records(tmp_path, capsys):
    summary = str(FIXTURES / "aiperf_records_summary.json")
    source = str(FIXTURES / "aiperf_records.jsonl")
    out = tmp_path / "artifact.json"
    with pytest.raises(SystemExit) as exc:
        cli.main(
            ["import", "aiperf", summary, "--records", source]
            + ["--slo-ttft-ms", "60", "--output", str(out)]
        )
    assert exc.value.code == 0
    written = read_imported_benchmark(out)
    assert written.source.tool == AIPERF_RECORDS_TOOL
    assert dict(written.slo) == {"ttft_ms": 60.0}
    assert "time_to_first_token [ms]: avg" in capsys.readouterr().out
    for argv, message in (
        ([summary, "--slo-ttft-ms", "60"], "need --records"),
        ([summary, "--records", source, "--output", source], "overwrite"),
        ([source], "--records"),
    ):
        with pytest.raises(SystemExit) as exc:
            cli.main(["import", "aiperf", *argv])
        assert exc.value.code == 2
        assert message in capsys.readouterr().err


def test_guidellm_v2_maps_one_benchmark():
    path = FIXTURES / "guidellm_v2.json"
    imported = import_benchmark_file(path, GUIDELLM_TOOL, benchmark=0)

    assert imported.source.tool == GUIDELLM_TOOL
    assert (imported.source.tool_version, imported.source.schema_version) == (
        "0.8.1",
        "2",
    )
    assert imported.model_id == "Qwen/Qwen3-0.6B"
    assert dict(imported.endpoint) == {
        "backend": "openai_http",
        "request_format": "/v1/chat/completions",
        "stream": True,
    }
    assert imported.load["strategy"] == "concurrent"
    assert imported.load["strategy.streams"] == 4
    assert "strategy.worker_count" not in imported.load
    # current_index is the sweep position, so it is dropped for a plain limit.
    assert imported.load["constraint.max_requests"] == json.dumps(
        {
            "args": {"count": 64, "kind": "max_requests", "stopping_scope": "current"},
            "type_": "max_requests",
        },
        sort_keys=True,
    )
    assert imported.load_unresolved == ()
    assert dict(imported.run)["benchmark_index"] == "0"
    totals = {name: total.value for name, total in imported.totals.items()}
    assert totals == {
        "completed_requests": 64,
        "failed_requests": 0,
        "incomplete_requests": 0,
        "total_input_tokens": 16384,
        "total_output_tokens": 8192,
        "duration": 16.4213,
    }
    ttft = imported.metrics["time_to_first_token_ms"]
    assert (ttft.unit, ttft.direction, ttft.provenance) == (
        "ms",
        "lower_is_better",
        IMPORTED_DERIVED,
    )
    assert ttft.stats["p99"].value == 71.64
    assert ttft.stats["p99"].source_key == (
        "benchmarks[0].metrics.time_to_first_token_ms.successful.percentiles.p99"
    )
    assert ttft.stats["std"].source_key.endswith(".successful.std_dev")
    assert {"p0.1", "p99.9", "min", "max"} <= set(ttft.stats)
    assert imported.metrics["request_latency"].unit == "s"
    rps = imported.metrics["requests_per_second"]
    assert list(rps.stats) == ["mean"]
    assert rps.direction == "higher_is_better"
    assert any(
        "weighted" in n for n in imported.metrics["inter_token_latency_ms"].notes
    )
    assert "metrics.request_concurrency" in imported.unmapped_keys
    assert "metrics.request_goodput" in imported.unmapped_keys


def test_guidellm_reads_errored_counts_and_skips_empty_statuses():
    imported = import_benchmark_file(
        FIXTURES / "guidellm_v2.json", GUIDELLM_TOOL, benchmark=1
    )

    assert imported.totals["failed_requests"].value == 1
    assert imported.load["strategy.rate"] == 8.0
    # Only the successful distribution is read; the errored request's
    # latency never reaches the imported statistics.
    latency = imported.metrics["request_latency"].stats
    assert latency["max"].value == 1.62
    assert all(".successful." in value.source_key for value in latency.values())


def test_guidellm_v1_report(tmp_path):
    source = write(tmp_path, guidellm(guidellm_v1))
    imported = import_benchmark_file(source, GUIDELLM_TOOL, benchmark=0)

    assert (imported.source.tool_version, imported.source.schema_version) == (
        "0.6.0",
        "1",
    )
    # 0.6 records only what the user passed, so GuideLLM's defaults apply.
    assert dict(imported.endpoint) == {
        "backend": "openai_http",
        "request_format": "/v1/chat/completions",
        "stream": True,
    }
    assert imported.load_unresolved == ()
    assert imported.load["constraint.max_number"] == (
        '{"max_num": 64, "type_": "max_number"}'
    )
    assert "time_to_first_output_token_ms" not in imported.metrics
    assert "time_to_first_token_ms" in imported.metrics


@pytest.mark.parametrize(
    ("args", "endpoint", "unresolved"),
    [
        (
            {"backend_kwargs": {"request_format": "text_completions", "stream": False}},
            ("/v1/completions", False),
            (),
        ),
        ({"data_request_formatter": "text_completions"}, ("/v1/completions", True), ()),
        (
            {
                "data_request_formatter": {
                    "request_type": "chat_completions",
                    "stream": False,
                }
            },
            ("/v1/chat/completions", False),
            (),
        ),
        (
            {"data_request_formatter": {"stream": False}},
            (None, False),
            ("endpoint.request_format",),
        ),
    ],
)
def test_guidellm_v1_endpoint_settings(tmp_path, args, endpoint, unresolved):
    def edit(data):
        guidellm_v1(data)
        data["args"].pop("backend_kwargs")
        data["args"].update(args)

    source = write(tmp_path, guidellm(edit))
    imported = import_benchmark_file(source, GUIDELLM_TOOL, benchmark=0)
    assert (imported.endpoint["request_format"], imported.endpoint["stream"]) == (
        endpoint
    )
    assert imported.load_unresolved == unresolved


def test_guidellm_v2_endpoint_without_stream_is_unresolved(tmp_path):
    def edit(data):
        del data["benchmarks"][0]["config"]["backend"]["stream"]

    source = write(tmp_path, guidellm(edit))
    imported = import_benchmark_file(source, GUIDELLM_TOOL, benchmark=0)
    assert imported.load_unresolved == ("endpoint.stream",)


def test_guidellm_sweep_limits_key_like_a_single_run(tmp_path):
    def sweep(data):
        data["benchmarks"][1]["config"]["constraints"]["max_requests"].update(
            args={
                "kind": "max_requests",
                "stopping_scope": "current",
                "count": [32, 64],
            },
            current_index=1,
        )

    def empty(data):
        data["benchmarks"][1]["config"]["constraints"]["max_requests"]["args"][
            "count"
        ] = []

    single = import_benchmark_file(
        write(tmp_path, guidellm()), GUIDELLM_TOOL, benchmark=1
    )
    swept = import_benchmark_file(
        write(tmp_path, guidellm(sweep), "sweep.json"), GUIDELLM_TOOL, benchmark=1
    )
    assert swept.load == single.load
    assert '"count": 64' in swept.load["constraint.max_requests"]
    assert "current_index" not in swept.load["constraint.max_requests"]

    broken = import_benchmark_file(
        write(tmp_path, guidellm(empty), "empty.json"), GUIDELLM_TOOL, benchmark=1
    )
    assert broken.load_unresolved == ("constraint.max_requests",)


def test_guidellm_never_copies_text_targets_or_keys(tmp_path):
    def edit(data):
        item = data["benchmarks"][0]
        item["config"]["backend"].update(
            target="http://inference.internal:8000", api_key="sk-secret"
        )
        item["requests"]["successful"].append(
            {"request_args": "SECRET PROMPT", "output": "SECRET OUTPUT"}
        )

    imported = import_benchmark_file(
        write(tmp_path, guidellm(edit)), GUIDELLM_TOOL, benchmark=0
    )
    text = imported.to_json()
    for secret in ("inference.internal", "sk-secret", "SECRET"):
        assert secret not in text


def test_guidellm_unresolved_load_and_bad_index(tmp_path):
    def edit(data):
        del data["benchmarks"][0]["config"]["strategy"]["streams"]

    imported = import_benchmark_file(
        write(tmp_path, guidellm(edit)), GUIDELLM_TOOL, benchmark=0
    )
    assert imported.load_unresolved == ("strategy.streams",)

    with pytest.raises(BenchmarkImportError, match="out of range"):
        import_benchmark_file(FIXTURES / "guidellm_v2.json", GUIDELLM_TOOL, benchmark=2)
    with pytest.raises(BenchmarkImportError, match="only to GuideLLM"):
        import_benchmark_file(
            FIXTURES / "vllm_default.json", VLLM_BENCH_TOOL, benchmark=0
        )


def test_cli_import_guidellm_benchmark(tmp_path, capsys):
    out = tmp_path / "artifact.json"
    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "import",
                "guidellm",
                str(FIXTURES / "guidellm_v2.json"),
                "--benchmark",
                "1",
                "--output",
                str(out),
            ]
        )
    assert exc.value.code == 0
    assert read_imported_benchmark(out).run["benchmark_index"] == "1"
    assert "time_to_first_token_ms [ms]" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exc:
        cli.main(["import", "guidellm", str(FIXTURES / "guidellm_v2.json")])
    assert exc.value.code == 2
    assert "--benchmark INDEX" in capsys.readouterr().err


SERVER_TTFTS = [0.18342, 0.21197, 0.19026, 0.17951, 0.20088, 0.18873, 0.19745, 0.18419]


def test_vllm_server_timings_keep_server_provenance():
    path = FIXTURES / "vllm_server_metrics.json"
    imported = import_benchmark_file(path, VLLM_BENCH_TOOL)
    default = import_benchmark_file(FIXTURES / "vllm_default.json", VLLM_BENCH_TOOL)

    server_ttft = imported.metrics["server_ttft_ms"]
    assert (server_ttft.unit, server_ttft.direction) == ("ms", "lower_is_better")
    assert server_ttft.provenance == DERIVED
    assert server_ttft.sample_provenance == SERVER_REPORTED
    assert server_ttft.sample_count == 8
    assert server_ttft.stats["mean"].value == pytest.approx(
        sum(SERVER_TTFTS) / 8 * 1000
    )
    assert server_ttft.stats["p50"].value == pytest.approx(188.73)
    assert server_ttft.stats["max"].value == pytest.approx(211.97)
    assert server_ttft.stats["p99"].source_key == "server_ttfts"
    assert any("not comparable to the client" in n for n in server_ttft.notes)
    queue = imported.metrics["server_queue_time_ms"]
    assert queue.stats["max"].value == pytest.approx(0.41)
    assert queue.sample_provenance == SERVER_REPORTED
    # The client TTFT keeps vLLM's own statistics and label.
    assert imported.metrics["ttft_ms"] == default.metrics["ttft_ms"]
    assert {"server_ttfts", "server_queue_times"}.isdisjoint(imported.unmapped_keys)
    assert "ttfts" in imported.unmapped_keys


def test_vllm_server_timings_partial_or_absent(tmp_path):
    partial = [None, *SERVER_TTFTS[1:]]
    source = write(
        tmp_path, server(server_ttfts=partial, server_queue_times=[None] * 8)
    )
    imported = import_benchmark_file(source, VLLM_BENCH_TOOL)

    server_ttft = imported.metrics["server_ttft_ms"]
    assert server_ttft.sample_count == 7
    assert server_ttft.stats["min"].value == pytest.approx(179.51)
    assert any("7 of 8 entries" in note for note in server_ttft.notes)
    # No request has a server queue time, so nothing is summarised.
    assert "server_queue_time_ms" not in imported.metrics
    assert "server_queue_times" in imported.unmapped_keys


def test_artifact_reader_refuses_relabelled_server_metrics():
    good = import_benchmark_file(
        FIXTURES / "vllm_server_metrics.json", VLLM_BENCH_TOOL
    ).to_dict()
    for field, value in (
        ("sample_provenance", IMPORTED_REPORTED),
        ("provenance", IMPORTED_DERIVED),
        ("sample_count", ...),
    ):
        bad = json.loads(json.dumps(good))
        if value is ...:
            del bad["metrics"]["server_ttft_ms"][field]
        else:
            bad["metrics"]["server_ttft_ms"][field] = value
        with pytest.raises(ImportedArtifactError, match="server_ttft_ms"):
            ImportedBenchmark.from_dict(bad)
