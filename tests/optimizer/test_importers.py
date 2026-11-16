from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from llmtracefx.optimizer import cli
from llmtracefx.optimizer.importers import (
    AIPERF_TOOL,
    IMPORTED_DERIVED,
    IMPORTED_REPORTED,
    VLLM_BENCH_TOOL,
    BenchmarkImportError,
    ImportedArtifactError,
    ImportedBenchmark,
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
