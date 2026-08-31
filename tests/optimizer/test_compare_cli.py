"""End-to-end CLI tests for ``compare`` and ``compare-report``.

Exercises the commands the way a user runs them, against synthetic artifact
trees. Nothing here loads a model, calls an API, or executes a benchmark.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from _compare_fixtures import (
    COMPARE_POLICY,
    PRICING_MANIFEST,
    write_api_run,
    write_json,
    write_run,
)

from llmtracefx.optimizer.cli import main
from llmtracefx.optimizer.compare.report import CompareReport


def _run(argv: list[str]) -> int:
    try:
        main(argv)
    except SystemExit as exc:  # argparse/`main` translate a return code
        return int(exc.code or 0)
    return 0


def _three_systems(results: Path) -> None:
    write_run(results, "local-1", total_ms=8000.0)
    write_api_run(results, "frontier-1", model_id="glm-5.3", total_ms=3000.0)
    write_api_run(
        results,
        "flash-1",
        model_id="glm-5.3-flash",
        reasoning_effort="low",
        total_ms=1200.0,
    )


def test_compare_writes_a_loadable_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    _three_systems(results)
    policy = write_json(tmp_path / "policy.json", COMPARE_POLICY)
    output = tmp_path / "compare.json"

    code = _run(
        [
            "compare",
            "--results",
            str(results),
            "--policy",
            str(policy),
            "--output",
            str(output),
        ]
    )
    assert code == 0
    report = CompareReport.read_json(output)
    assert len(report.strata) == 1
    assert report.has_recommendation is True
    stdout = capsys.readouterr().out
    assert "Cross-system comparison" in stdout
    assert "no universal winner" in stdout


def test_compare_with_pricing_records_the_manifest_hash(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _three_systems(results)
    policy = write_json(tmp_path / "policy.json", COMPARE_POLICY)
    pricing = write_json(tmp_path / "rates.json", PRICING_MANIFEST)
    output = tmp_path / "compare.json"

    assert (
        _run(
            [
                "compare",
                "--results",
                str(results),
                "--policy",
                str(policy),
                "--pricing",
                str(pricing),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = CompareReport.read_json(output)
    assert report.pricing is not None
    assert report.pricing.currency == "USD"
    assert report.pricing.rates_are_illustrative is True
    assert len(report.pricing.manifest_sha256) == 64


def test_compare_exits_two_when_inconclusive(tmp_path: Path) -> None:
    results = tmp_path / "results"
    write_api_run(results, "a", model_id="glm-5.3", total_ms=1000.0)
    write_api_run(results, "b", model_id="glm-5.3-flash", total_ms=1000.0)
    policy = write_json(tmp_path / "policy.json", COMPARE_POLICY)
    assert _run(["compare", "--results", str(results), "--policy", str(policy)]) == 2


def test_compare_exits_one_when_nothing_is_comparable(tmp_path: Path) -> None:
    results = tmp_path / "empty"
    results.mkdir()
    policy = write_json(tmp_path / "policy.json", COMPARE_POLICY)
    assert _run(["compare", "--results", str(results), "--policy", str(policy)]) == 1


def test_compare_rejects_an_invalid_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    _three_systems(results)
    policy = write_json(tmp_path / "policy.json", {"objective": "be_the_best"})
    assert _run(["compare", "--results", str(results), "--policy", str(policy)]) == 1
    assert "Invalid compare policy" in capsys.readouterr().err


def test_compare_rejects_an_ambiguous_pricing_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    _three_systems(results)
    policy = write_json(tmp_path / "policy.json", COMPARE_POLICY)
    ambiguous = json.loads(json.dumps(PRICING_MANIFEST))
    duplicate = json.loads(json.dumps(ambiguous["entries"][0]))
    duplicate["entry_id"] = "glm-5.3-duplicate"
    ambiguous["entries"].append(duplicate)
    pricing = write_json(tmp_path / "rates.json", ambiguous)
    code = _run(
        [
            "compare",
            "--results",
            str(results),
            "--policy",
            str(policy),
            "--pricing",
            str(pricing),
        ]
    )
    assert code == 1
    assert "Invalid pricing manifest" in capsys.readouterr().err


def test_compare_refuses_a_cost_objective_without_pricing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    _three_systems(results)
    policy = write_json(
        tmp_path / "policy.json",
        {**COMPARE_POLICY, "objective": "min_cost_per_correct_case"},
    )
    assert _run(["compare", "--results", str(results), "--policy", str(policy)]) == 1
    assert "ranks on money" in capsys.readouterr().err


def test_compare_counts_repeated_results_dirs_as_repetitions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two results trees for one matrix row are two measurements of it.

    This is the documented way to satisfy a policy's
    ``min_measured_repetitions``: run the matrix again into a second output
    directory and pass both. Both trees carry the same run ids, so rejecting
    that as a conflict made the requirement unsatisfiable.
    """
    local_first = tmp_path / "local-a"
    local_second = tmp_path / "local-b"
    hosted_first = tmp_path / "hosted-a"
    hosted_second = tmp_path / "hosted-b"
    write_run(local_first, "shared", total_ms=8000.0)
    write_run(local_second, "shared", total_ms=8400.0)
    write_api_run(hosted_first, "shared", total_ms=1200.0)
    write_api_run(hosted_second, "shared", total_ms=1300.0)
    policy = write_json(tmp_path / "policy.json", COMPARE_POLICY)
    out = tmp_path / "report.json"
    code = _run(
        [
            "compare",
            "--results",
            str(local_first),
            str(local_second),
            str(hosted_first),
            str(hosted_second),
            "--policy",
            str(policy),
            "--output",
            str(out),
        ]
    )
    assert code == 0, capsys.readouterr().err
    payload = json.loads(out.read_text(encoding="utf-8"))
    ranked = payload["strata"][0]["ranked"]
    assert len(ranked) == 2
    # Each system carries both of its measurements rather than one of them
    # being rejected as a conflict or silently deduplicated away.
    assert {entry["evidence_count"] for entry in ranked} == {2}
    assert all(len(entry["verification_paths"]) == 2 for entry in ranked)


def _real_tune_report(path: Path) -> Path:
    """Copy the committed example, so the path names a real tune report."""
    source = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "optimizer"
        / "tune-report-example.json"
    )
    path.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return path


def test_compare_records_corroborating_tune_reports(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _three_systems(results)
    policy = write_json(tmp_path / "policy.json", COMPARE_POLICY)
    output = tmp_path / "compare.json"
    tune_a = _real_tune_report(tmp_path / "tune-a.json")
    tune_b = _real_tune_report(tmp_path / "tune-b.json")
    assert (
        _run(
            [
                "compare",
                "--results",
                str(results),
                "--policy",
                str(policy),
                "--tune-report",
                str(tune_a),
                str(tune_b),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    report = CompareReport.read_json(output)
    assert report.tune_report_paths == (str(tune_a), str(tune_b))


@pytest.mark.parametrize("content", ["{}", "not json", None])
def test_compare_refuses_a_tune_report_that_is_not_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str | None
) -> None:
    """Provenance has to be true.

    These paths are published under "Corroborating tune reports", so a
    missing or malformed one asserts corroboration that does not exist. The
    pricing manifest is content-hashed for the same reason.
    """
    results = tmp_path / "results"
    _three_systems(results)
    policy = write_json(tmp_path / "policy.json", COMPARE_POLICY)
    bogus = tmp_path / "bogus.json"
    if content is None:
        bogus.mkdir()
    else:
        bogus.write_text(content, encoding="utf-8")

    code = _run(
        [
            "compare",
            "--results",
            str(results),
            "--policy",
            str(policy),
            "--tune-report",
            str(bogus),
        ]
    )
    assert code == 1
    assert "tune report" in capsys.readouterr().err


def test_explain_prints_every_rejection_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    write_run(results, "bad", success=False, quality_score=0.0, total_ms=90_000.0)
    write_api_run(results, "good", total_ms=1000.0)
    policy = write_json(
        tmp_path / "policy.json",
        {
            **COMPARE_POLICY,
            "constraints": {"min_pass_rate": 1.0, "max_mean_total_latency_ms": 5000.0},
        },
    )
    _run(
        [
            "compare",
            "--results",
            str(results),
            "--policy",
            str(policy),
            "--explain",
        ]
    )
    stdout = capsys.readouterr().out
    assert "pass rate" in stdout
    assert "exceeds the maximum" in stdout


# --- compare-report -------------------------------------------------------


def _compare_json(tmp_path: Path, *, with_pricing: bool = False) -> Path:
    results = tmp_path / "results"
    _three_systems(results)
    policy = write_json(tmp_path / "policy.json", COMPARE_POLICY)
    output = tmp_path / "compare.json"
    argv = [
        "compare",
        "--results",
        str(results),
        "--policy",
        str(policy),
        "--output",
        str(output),
    ]
    if with_pricing:
        argv.extend(
            ("--pricing", str(write_json(tmp_path / "rates.json", PRICING_MANIFEST)))
        )
    _run(argv)
    return output


def test_compare_report_renders_html(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_json = _compare_json(tmp_path)
    html_path = tmp_path / "compare.html"
    assert (
        _run(
            [
                "compare-report",
                "--input",
                str(report_json),
                "--output",
                str(html_path),
            ]
        )
        == 0
    )
    document = html_path.read_text(encoding="utf-8")
    assert document.startswith("<!DOCTYPE html>")
    assert "<script" not in document.lower()
    stdout = capsys.readouterr().out
    assert "Compare report HTML written to" in stdout
    assert "redacted" in stdout


def test_compare_report_warns_about_illustrative_rates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_json = _compare_json(tmp_path, with_pricing=True)
    assert (
        _run(
            [
                "compare-report",
                "--input",
                str(report_json),
                "--output",
                str(tmp_path / "compare.html"),
            ]
        )
        == 0
    )
    assert "illustrative rates" in capsys.readouterr().out


def test_compare_report_include_paths_is_opt_in(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report_json = _compare_json(tmp_path)
    html_path = tmp_path / "compare.html"
    _run(
        [
            "compare-report",
            "--input",
            str(report_json),
            "--output",
            str(html_path),
            "--include-paths",
        ]
    )
    assert str(tmp_path) in html_path.read_text(encoding="utf-8")
    assert "Full local artifact paths are included" in capsys.readouterr().out


def test_compare_report_rejects_a_malformed_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "compare.json"
    bad.write_text('{"schema_version": "99"}', encoding="utf-8")
    code = _run(
        [
            "compare-report",
            "--input",
            str(bad),
            "--output",
            str(tmp_path / "out.html"),
        ]
    )
    assert code == 1
    assert "Invalid compare report" in capsys.readouterr().err


def test_compare_report_reports_a_missing_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = _run(
        [
            "compare-report",
            "--input",
            str(tmp_path / "nope.json"),
            "--output",
            str(tmp_path / "out.html"),
        ]
    )
    assert code == 1
    assert "Could not read compare report input" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("label", "payload"),
    [
        ("integer past the digit cap", '{"x": ' + "9" * 4400 + "}"),
        ("array nested past the recursion limit", "[" * 200_000 + "]" * 200_000),
    ],
)
def test_compare_refuses_a_tune_report_json_past_the_parser_limits(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    label: str,
    payload: str,
) -> None:
    """``json`` exceeds its own limits with exceptions that are not decode errors.

    An integer literal over the interpreter's digit cap raises a plain
    ``ValueError`` and deep nesting raises ``RecursionError``. Neither is a
    ``JSONDecodeError``, so both escaped as a stack trace from a merely
    malformed provenance file -- on the one input that had not been hardened
    the way the compare report, the policy and the pricing manifest already
    were.
    """
    results = tmp_path / "results"
    _three_systems(results)
    policy = write_json(tmp_path / "policy.json", COMPARE_POLICY)
    bad = tmp_path / "bad.json"
    bad.write_text(payload, encoding="utf-8")

    code = _run(
        [
            "compare",
            "--results",
            str(results),
            "--policy",
            str(policy),
            "--tune-report",
            str(bad),
        ]
    )
    assert code == 1, label
    assert "Invalid tune report" in capsys.readouterr().err
