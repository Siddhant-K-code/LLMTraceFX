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
    IMPORTED_DERIVED,
    IMPORTED_REPORTED,
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
        ("aiperf_legacy.json", AIPERF_TOOL),
        ("aiperf_v1_7.json", AIPERF_TOOL),
    ],
)
def test_artifact_round_trips(tmp_path, fixture, tool):
    imported = import_benchmark_file(FIXTURES / fixture, tool)
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


def aiperf(**changes):
    data = load("aiperf_v1_7.json")
    data.update(changes)
    return data


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
