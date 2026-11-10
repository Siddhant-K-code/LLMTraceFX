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
    load_records,
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
