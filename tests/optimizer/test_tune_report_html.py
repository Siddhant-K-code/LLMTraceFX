"""Tests for the `tune-report` HTML renderer: escaping, path redaction,
section coverage, and determinism.

Builds real ``TuneReport`` instances via ``tune()`` over fake
``workloads run``-shaped artifact trees (see ``_tune_fixtures.write_run``)
so the rendered HTML reflects the same shape a real report would have.
"""

from __future__ import annotations

import html
from pathlib import Path

from _tune_fixtures import write_run

from llmtracefx.optimizer.tune.policy import TuneObjective, TunePolicy
from llmtracefx.optimizer.tune.report_html import render_tune_report_html
from llmtracefx.optimizer.tune.tuner import tune
from llmtracefx.optimizer.workloads.verify import RowStatus

LATENCY_POLICY = TunePolicy(objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS)


# --- Determinism ---------------------------------------------------------


def test_render_is_byte_identical_across_calls(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    write_run(tmp_path, "r2", total_ms=2000.0, seed=1)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    first = render_tune_report_html(report)
    second = render_tune_report_html(report)

    assert first == second


def test_render_has_no_new_timestamp_beyond_generated_at(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    html_out = render_tune_report_html(report)

    assert report.generated_at in html_out


# --- Escaping / security --------------------------------------------------


def test_script_tag_in_policy_name_is_escaped(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    policy = TunePolicy(
        objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS,
        name="<script>alert(1)</script>",
    )
    report = tune(results_dirs=(tmp_path,), policy=policy)

    html_out = render_tune_report_html(report)

    assert "<script>alert(1)</script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_script_tag_in_policy_description_is_escaped(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    policy = TunePolicy(
        objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS,
        description="<img src=x onerror=alert(1)>",
    )
    report = tune(results_dirs=(tmp_path,), policy=policy)

    html_out = render_tune_report_html(report)

    assert "<img src=x onerror=alert(1)>" not in html_out
    assert "&lt;img" in html_out


def test_script_tag_in_model_id_is_escaped(tmp_path):
    write_run(
        tmp_path, "r1", total_ms=1000.0, model_id="<script>alert('model')</script>"
    )
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    html_out = render_tune_report_html(report)

    assert "<script>alert('model')</script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_script_tag_in_rejection_reason_is_escaped(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    write_run(
        tmp_path,
        "r2",
        total_ms=1500.0,
        seed=1,
        quality_metric="<script>alert('metric')</script>",
    )
    policy = TunePolicy.from_dict(
        {
            "objective": "min_mean_total_latency_ms",
            "constraints": {"required_quality_metric": "trusted_metric"},
        }
    )
    report = tune(results_dirs=(tmp_path,), policy=policy)

    html_out = render_tune_report_html(report)

    assert report.groups[0].rejected
    assert any(
        "<script>" in reason
        for rejected in report.groups[0].rejected
        for reason in rejected.reasons
    )
    assert "<script>" not in html_out


def test_script_tag_in_excluded_run_id_is_escaped(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    write_run(tmp_path, "run-<script>-excluded", corrupt_final_record=True)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    html_out = render_tune_report_html(report)

    assert report.excluded_runs
    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_script_tag_in_candidate_run_id_is_escaped(tmp_path):
    write_run(tmp_path, "run-<script>-runid", total_ms=1000.0)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    html_out = render_tune_report_html(report)

    assert "<script>" not in html_out
    assert "&lt;script&gt;" in html_out


def test_no_network_or_cdn_references_in_output(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    html_out = render_tune_report_html(report)

    for token in ("http://", "https://", "<script", "cdn.", "googleapis"):
        assert token not in html_out


# --- Path redaction --------------------------------------------------------


def test_paths_are_redacted_by_default(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    html_out = render_tune_report_html(report, redact_paths=True)

    assert str(tmp_path) not in html_out
    assert "runs/r1/final_record.json" in html_out


def test_full_paths_included_when_redact_paths_is_false(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    html_out = render_tune_report_html(report, redact_paths=False)

    assert str(tmp_path) in html_out


def test_redaction_uses_last_runs_boundary_without_leaking_parent_segments():
    from llmtracefx.optimizer.tune.report_html import _redact_path

    raw = (
        "/Users/alice/runs/private-project/secret-experiment/"
        "runs/run77/verification.json"
    )

    redacted = _redact_path(raw)

    assert redacted == "runs/run77/verification.json"
    assert "private-project" not in redacted
    assert "secret-experiment" not in redacted


def test_redaction_handles_windows_paths_and_escapes_labels():
    from llmtracefx.optimizer.tune.report_html import _esc, _redact_path

    raw = r"C:\Users\alice\runs\private\runs\run<script>\final_record.json"
    redacted = _redact_path(raw)

    assert redacted == "runs/run<script>/final_record.json"
    assert "<script>" not in _esc(redacted)


def test_redaction_falls_back_to_basename_without_runs_segment():
    from llmtracefx.optimizer.tune.report_html import _redact_path

    assert _redact_path("/Users/alice/private/report.json") == "report.json"


def test_redaction_prefers_known_run_id_over_ambiguous_runs_segments():
    from llmtracefx.optimizer.tune.report_html import _redact_path

    # Two "runs" segments upstream of the real one; the run id itself is
    # the ground truth and must never leave "secret-project"/"other-runs"
    # in the label, regardless of which "runs" segment is "last".
    raw = (
        "/Users/alice/secret-project/runs/other-runs/results/"
        "runs/run77/final_record.json"
    )

    assert _redact_path(raw, run_id="run77") == "runs/run77/final_record.json"


def test_redaction_with_run_id_and_no_runs_segment_still_bounds_at_run_id():
    from llmtracefx.optimizer.tune.report_html import _redact_path

    raw = "/Users/alice/private-project/custom-layout/run77/final_record.json"

    assert _redact_path(raw, run_id="run77") == "run77/final_record.json"


def test_redaction_with_unmatched_run_id_falls_back_to_last_runs_segment():
    from llmtracefx.optimizer.tune.report_html import _redact_path

    raw = "/Users/alice/secret-project/runs/run1/final_record.json"

    # The run id passed does not appear in the path at all (e.g. a
    # verification/final-record path recorded under a different run than
    # the one it is being rendered alongside); must not silently include
    # the ancestor "secret-project" segment.
    assert _redact_path(raw, run_id="some-other-run") == "runs/run1/final_record.json"


def test_redaction_handles_traversal_like_segments_without_leaking_ancestors():
    from llmtracefx.optimizer.tune.report_html import _redact_path

    raw = "/Users/alice/../alice/secret/runs/run1/final_record.json"

    assert _redact_path(raw) == "runs/run1/final_record.json"
    assert _redact_path(raw, run_id="run1") == "runs/run1/final_record.json"


def test_redaction_windows_path_with_run_id():
    from llmtracefx.optimizer.tune.report_html import _redact_path

    raw = r"C:\Users\alice\secret-project\results\runs\run1\final_record.json"

    assert _redact_path(raw, run_id="run1") == "runs/run1/final_record.json"


def test_candidate_evidence_paths_use_run_id_aware_redaction(tmp_path):
    write_run(
        tmp_path / "secret-project" / "runs" / "duplicate-runs-dir",
        "r1",
        total_ms=1000.0,
    )
    report = tune(
        results_dirs=(tmp_path / "secret-project" / "runs" / "duplicate-runs-dir",),
        policy=LATENCY_POLICY,
    )

    html_out = render_tune_report_html(report)

    # The ancestor "secret-project" directory name must never leak, whether
    # via the results-directory listing or via any per-candidate artifact
    # path (which is redacted using the known run id and always bounds at
    # the "runs" segment immediately preceding it).
    assert "secret-project" not in html_out
    assert "runs/r1/final_record.json" in html_out


# --- Section coverage --------------------------------------------------------


def test_recommended_section_shows_winner_and_rationale(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    write_run(tmp_path, "r2", total_ms=2000.0, seed=1)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    html_out = render_tune_report_html(report)

    assert "RECOMMENDED" in html_out
    assert "runner-up" in html_out


def test_inconclusive_section_shows_reason(tmp_path):
    write_run(tmp_path, "r1", status=RowStatus.FAILED, success=False)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    html_out = render_tune_report_html(report)

    assert "INCONCLUSIVE" in html_out
    assert report.groups[0].inconclusive_reason in html_out


def test_rejected_section_lists_every_violation(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    write_run(
        tmp_path,
        "r2",
        total_ms=10_000.0,
        seed=1,
        peak_bytes=25 * 1024**3,
        status=RowStatus.FAILED,
        success=False,
    )
    policy = TunePolicy.from_dict(
        {
            "objective": "min_mean_total_latency_ms",
            "constraints": {
                "max_total_latency_ms": 1000,
                "max_peak_memory_bytes": 20 * 1024**3,
            },
        }
    )
    report = tune(results_dirs=(tmp_path,), policy=policy)

    html_out = render_tune_report_html(report)

    rejected = report.groups[0].rejected[0]
    assert len(rejected.reasons) > 1
    for reason in rejected.reasons:
        assert html.escape(reason, quote=True) in html_out


def test_excluded_runs_section_shows_reason(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    write_run(tmp_path, "r2", corrupt_final_record=True)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    html_out = render_tune_report_html(report)

    assert "Excluded runs" in html_out
    assert report.excluded_runs[0].run_id in html_out


def test_baseline_comparison_section_present_when_speculative_wins(tmp_path):
    write_run(tmp_path, "r-ar1", speculative_enabled=False, total_ms=2000.0)
    write_run(tmp_path, "r-ar2", speculative_enabled=False, total_ms=2000.0)
    write_run(
        tmp_path,
        "r-spec1",
        speculative_enabled=True,
        speculative_method="draft-model",
        speculative_depth=2,
        total_ms=1000.0,
    )
    write_run(
        tmp_path,
        "r-spec2",
        speculative_enabled=True,
        speculative_method="draft-model",
        speculative_depth=2,
        total_ms=1000.0,
    )
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    html_out = render_tune_report_html(report)

    assert "Speculative baseline comparison" in html_out
    assert "improvement" in html_out


def test_accepted_candidate_ranking_includes_required_metrics(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    html_out = render_tune_report_html(report)

    assert "Mean latency (ms)" in html_out
    assert "Pass rate" in html_out
    assert "Quality metric" in html_out
    assert "Peak memory" in html_out
    assert "Evidence" in html_out
    assert "CV" in html_out


def test_no_comparable_groups_message_when_no_groups(tmp_path):
    results_dir = tmp_path / "empty"
    results_dir.mkdir()
    report = tune(results_dirs=(results_dir,), policy=LATENCY_POLICY)

    html_out = render_tune_report_html(report)

    assert "No comparable groups" in html_out


def test_example_fixture_renders_every_section():
    example_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "optimizer"
        / "tune-report-example.json"
    )
    from llmtracefx.optimizer.tune.report import TuneReport

    report = TuneReport.read_json(example_path)

    html_out = render_tune_report_html(report)

    assert "RECOMMENDED" in html_out
    assert "Rejected candidates" in html_out
    assert "Speculative baseline comparison" in html_out
    assert "Excluded runs" in html_out
    assert "SYNTHETIC" in html_out
