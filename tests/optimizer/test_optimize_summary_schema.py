"""Unit tests for the `optimize_summary` orchestration-summary schema."""

from __future__ import annotations

import json

import pytest

from llmtracefx.optimizer.optimize_summary import (
    OPTIMIZE_SUMMARY_SCHEMA_VERSION,
    OptimizeSummary,
    OptimizeSummaryValidationError,
    OverallStatus,
    PhaseName,
    PhaseReport,
    PhaseStatus,
    RecommendedCandidate,
    RowStatusCounts,
)


def _make_summary(**overrides):
    defaults = {
        "schema_version": OPTIMIZE_SUMMARY_SCHEMA_VERSION,
        "generated_at": "2026-01-01T00:00:00Z",
        "dry_run": False,
        "matrix_path": "/tmp/matrix/manifest.json",
        "results_dir": "/tmp/results",
        "policy_path": "/tmp/policy.json",
        "report_json_path": "/tmp/report.json",
        "report_html_path": "/tmp/report.html",
        "phases": (
            PhaseReport(name=PhaseName.PLANNED, status=PhaseStatus.OK),
            PhaseReport(name=PhaseName.EXECUTED, status=PhaseStatus.OK),
            PhaseReport(name=PhaseName.VERIFIED, status=PhaseStatus.OK),
            PhaseReport(name=PhaseName.TUNED, status=PhaseStatus.OK),
            PhaseReport(name=PhaseName.RENDERED, status=PhaseStatus.OK),
        ),
        "row_counts": RowStatusCounts(total=1, completed=1),
        "recommendations": (
            RecommendedCandidate(
                group_label="g",
                run_ids=("r1",),
                objective_name="min_mean_total_latency_ms",
                objective_value=1.0,
            ),
        ),
        "overall_status": OverallStatus.SUCCESS,
        "exit_code": 0,
    }
    defaults.update(overrides)
    return OptimizeSummary(**defaults)


def test_round_trips_through_json():
    summary = _make_summary()
    restored = OptimizeSummary.from_json(summary.to_json())
    assert restored == summary


def test_write_and_read_json_file(tmp_path):
    summary = _make_summary()
    path = tmp_path / "summary.json"
    path.write_text(summary.to_json() + "\n", encoding="utf-8")
    restored = OptimizeSummary.read_json(path)
    assert restored == summary


def test_rejects_invalid_json():
    with pytest.raises(OptimizeSummaryValidationError):
        OptimizeSummary.from_json("not json")


def test_rejects_non_object_payload():
    with pytest.raises(OptimizeSummaryValidationError):
        OptimizeSummary.from_json("[1, 2, 3]")


def test_rejects_unsupported_schema_version():
    summary = _make_summary()
    data = summary.to_dict()
    data["schema_version"] = "999"
    with pytest.raises(OptimizeSummaryValidationError):
        OptimizeSummary.from_dict(data)


def test_requires_every_phase_present_in_order():
    summary = _make_summary()
    data = summary.to_dict()
    data["phases"] = data["phases"][:-1]  # drop "rendered"
    with pytest.raises(OptimizeSummaryValidationError):
        OptimizeSummary.from_dict(data)


def test_rejects_out_of_order_phases():
    summary = _make_summary()
    data = summary.to_dict()
    data["phases"] = list(reversed(data["phases"]))
    with pytest.raises(OptimizeSummaryValidationError):
        OptimizeSummary.from_dict(data)


def test_phase_report_rejects_unknown_status():
    with pytest.raises(OptimizeSummaryValidationError):
        PhaseReport.from_dict({"name": "planned", "status": "not-a-real-status"})


def test_phase_report_rejects_unknown_name():
    with pytest.raises(OptimizeSummaryValidationError):
        PhaseReport.from_dict({"name": "not-a-real-phase", "status": "ok"})


def test_row_status_counts_rejects_negative_values():
    with pytest.raises(OptimizeSummaryValidationError):
        RowStatusCounts.from_dict({"total": -1})


def test_row_status_counts_rejects_non_integer_values():
    with pytest.raises(OptimizeSummaryValidationError):
        RowStatusCounts.from_dict({"total": "one"})


def test_recommended_candidate_requires_group_label():
    with pytest.raises(OptimizeSummaryValidationError):
        RecommendedCandidate.from_dict(
            {
                "run_ids": ["r1"],
                "objective_name": "x",
                "objective_value": 1.0,
            }
        )


def test_rejects_invalid_overall_status():
    summary = _make_summary()
    data = summary.to_dict()
    data["overall_status"] = "somewhat_ok"
    with pytest.raises(OptimizeSummaryValidationError):
        OptimizeSummary.from_dict(data)


def test_rejects_non_integer_exit_code():
    summary = _make_summary()
    data = summary.to_dict()
    data["exit_code"] = "0"
    with pytest.raises(OptimizeSummaryValidationError):
        OptimizeSummary.from_dict(data)


def test_phase_lookup_helper_finds_named_phase():
    summary = _make_summary()
    tuned = summary.phase(PhaseName.TUNED)
    assert tuned is not None
    assert tuned.status == PhaseStatus.OK


def test_to_dict_is_json_serializable_directly():
    summary = _make_summary()
    # Never rely on to_json's json.dumps alone: to_dict must already be a
    # plain, directly-serializable structure (no enums/dataclasses leaking
    # through), the same contract every other schema module in this
    # package upholds.
    json.dumps(summary.to_dict())
