"""Tests for the tune engine: loading, identity, grouping, constraints, ranking.

Builds full fake `workloads run --output-dir`-shaped artifact trees (see
``_tune_fixtures.write_run``) rather than only unit-testing helper
functions, so these tests exercise the same on-disk shape the real verify
pipeline produces.
"""

from __future__ import annotations

import pytest
from _tune_fixtures import write_run

from llmtracefx.optimizer.schema import MetricProvenance
from llmtracefx.optimizer.tune.loader import TuneInputError, load_evidence
from llmtracefx.optimizer.tune.policy import (
    TuneConstraints,
    TuneObjective,
    TunePolicy,
    TunePolicyError,
)
from llmtracefx.optimizer.tune.report import GroupOutcome
from llmtracefx.optimizer.tune.tuner import tune
from llmtracefx.optimizer.workloads.verify import RowStatus

LATENCY_POLICY = TunePolicy(objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS)


# --- Loading / identity checks -----------------------------------------------


def test_load_evidence_reads_completed_run(tmp_path):
    write_run(tmp_path, "r1")
    loaded = load_evidence((tmp_path,))
    assert len(loaded.usable) == 1
    assert loaded.usable[0].run_id == "r1"
    assert not loaded.excluded


def test_unsupported_row_is_excluded_not_a_rejected_candidate(tmp_path):
    write_run(
        tmp_path,
        "mtp-unsupported",
        status=RowStatus.UNSUPPORTED,
        write_final_record=False,
        reason="native-mtp is not implemented",
        decode_mode="native-mtp",
    )
    loaded = load_evidence((tmp_path,))
    assert not loaded.usable
    assert len(loaded.excluded) == 1
    assert "native-mtp" in loaded.excluded[0].reason


def test_corrupt_final_record_is_excluded(tmp_path):
    write_run(tmp_path, "r1", corrupt_final_record=True)
    loaded = load_evidence((tmp_path,))
    assert not loaded.usable
    assert len(loaded.excluded) == 1
    assert "schema validation" in loaded.excluded[0].reason


def test_missing_final_record_file_is_excluded(tmp_path):
    run_dir = write_run(tmp_path, "r1")
    (run_dir / "final_record.json").unlink()
    loaded = load_evidence((tmp_path,))
    assert not loaded.usable
    assert len(loaded.excluded) == 1
    assert "could not read final_record.json" in loaded.excluded[0].reason


def test_run_id_mismatch_between_verification_and_record_is_excluded(tmp_path):
    run_dir = write_run(tmp_path, "r1")
    # Corrupt just the run_id inside an otherwise-valid final_record.json.
    import json

    payload = json.loads((run_dir / "final_record.json").read_text())
    payload["run_id"] = "different-run"
    (run_dir / "final_record.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_evidence((tmp_path,))
    assert not loaded.usable
    assert "run_id" in loaded.excluded[0].reason


def test_prompt_hash_mismatch_between_verification_and_record_is_excluded(tmp_path):
    write_run(tmp_path, "r1", workload_hash_override="sha256:different")
    loaded = load_evidence((tmp_path,))
    assert not loaded.usable
    assert "workload_hash" in loaded.excluded[0].reason


def test_recorded_and_verified_prompt_hash_drift_is_excluded(tmp_path):
    write_run(
        tmp_path,
        "r1",
        prompt_hash="sha256:promptabc",
        recorded_prompt_hash="sha256:stale",
    )
    loaded = load_evidence((tmp_path,))
    assert not loaded.usable
    assert "drift" in loaded.excluded[0].reason


def test_identical_duplicate_run_id_across_dirs_is_deduped(tmp_path):
    import shutil

    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    write_run(dir_a, "r1", total_ms=1000.0)
    shutil.copytree(dir_a, dir_b)
    loaded = load_evidence((dir_a, dir_b))
    assert len(loaded.usable) == 1


def test_conflicting_duplicate_run_id_raises(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    write_run(dir_a, "r1", total_ms=1000.0)
    write_run(dir_b, "r1", total_ms=2000.0)
    with pytest.raises(TuneInputError, match="duplicate run_id"):
        load_evidence((dir_a, dir_b))


# --- Grouping -----------------------------------------------------------


def test_different_accelerators_are_separate_groups(tmp_path):
    write_run(tmp_path, "r-m5", accelerator="Apple M5 Pro")
    write_run(tmp_path, "r-cuda", accelerator="NVIDIA RTX 4090")
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    assert len(report.groups) == 2


def test_different_context_tiers_are_separate_groups(tmp_path):
    write_run(tmp_path, "r-2k", context_tier="2k")
    write_run(tmp_path, "r-8k", context_tier="8k")
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    assert len(report.groups) == 2


def test_different_runtime_backend_are_separate_groups(tmp_path):
    write_run(tmp_path, "r-metal", runtime_backend="Metal")
    write_run(tmp_path, "r-cuda", runtime_backend="CUDA")
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    assert len(report.groups) == 2


def test_same_group_key_different_quantization_are_separate_candidates(tmp_path):
    write_run(tmp_path, "r-q4", quantization="Q4", total_ms=1000.0)
    write_run(tmp_path, "r-q8", quantization="Q8", total_ms=2000.0)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    assert len(report.groups) == 1
    group = report.groups[0]
    assert len(group.accepted) == 2


def test_same_group_key_different_seed_are_separate_candidates(tmp_path):
    write_run(tmp_path, "r-seed0", seed=0)
    write_run(tmp_path, "r-seed1", seed=1)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    group = report.groups[0]
    assert len(group.accepted) == 2


def test_same_candidate_config_multiple_runs_are_averaged(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    write_run(tmp_path, "r2", total_ms=1200.0)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    group = report.groups[0]
    assert len(group.accepted) == 1
    winner = group.recommended
    assert winner.evidence_count == 2
    assert winner.mean_total_latency_ms == pytest.approx(1100.0)


# --- Recommendation / winner selection -----------------------------------


def test_recommends_fastest_candidate_by_latency(tmp_path):
    write_run(tmp_path, "r-fast", quantization="Q4", total_ms=1000.0)
    write_run(tmp_path, "r-slow", quantization="Q8", total_ms=2000.0)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    group = report.groups[0]
    assert group.outcome == GroupOutcome.RECOMMENDED
    assert group.recommended is not None
    assert group.recommended.candidate_key.quantization == "Q4"
    assert len(group.accepted) == 2
    assert not group.rejected


def test_recommends_highest_correct_cases_per_minute(tmp_path):
    # r-a: 1 correct case per 30s => 2/min. r-b: 1 correct case per 10s => 6/min.
    write_run(tmp_path, "r-a", quantization="Q4", total_ms=30_000.0)
    write_run(tmp_path, "r-b", quantization="Q8", total_ms=10_000.0)
    policy = TunePolicy(objective=TuneObjective.MAX_CORRECT_CASES_PER_MINUTE)
    report = tune(results_dirs=(tmp_path,), policy=policy)
    group = report.groups[0]
    assert group.recommended.candidate_key.quantization == "Q8"
    assert group.recommended.correct_cases_per_minute == pytest.approx(6.0)


def test_failing_quality_excludes_case_from_correct_cases_per_minute(tmp_path):
    write_run(
        tmp_path,
        "r-fail",
        quantization="Q4",
        total_ms=1000.0,
        success=False,
        quality_score=0.0,
    )
    policy = TunePolicy(objective=TuneObjective.MAX_CORRECT_CASES_PER_MINUTE)
    report = tune(results_dirs=(tmp_path,), policy=policy)
    group = report.groups[0]
    assert group.outcome == GroupOutcome.INCONCLUSIVE
    assert group.rejected
    assert "no passing run" in group.rejected[0].reasons[-1]


# --- Constraint rejections -------------------------------------------------


def test_rejects_status_not_in_required_statuses(tmp_path):
    write_run(tmp_path, "r-failed", status=RowStatus.FAILED, success=False)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    group = report.groups[0]
    assert group.outcome == GroupOutcome.INCONCLUSIVE
    assert len(group.rejected) == 1
    assert any("status" in reason for reason in group.rejected[0].reasons)


def test_rejects_below_min_pass_rate(tmp_path):
    write_run(tmp_path, "r1", success=True, quality_score=1.0)
    write_run(tmp_path, "r2", success=False, quality_score=0.0)
    policy = TunePolicy(
        objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS,
        constraints=TuneConstraints(min_pass_rate=0.9),
    )
    report = tune(results_dirs=(tmp_path,), policy=policy)
    group = report.groups[0]
    assert group.outcome == GroupOutcome.INCONCLUSIVE
    assert any("pass rate" in reason for reason in group.rejected[0].reasons)


def test_rejects_below_min_quality_score(tmp_path):
    write_run(tmp_path, "r1", quality_score=0.5, quality_metric="m1")
    policy = TunePolicy(
        objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS,
        constraints=TuneConstraints(
            min_quality_score=0.9, required_quality_metric="m1"
        ),
    )
    report = tune(results_dirs=(tmp_path,), policy=policy)
    group = report.groups[0]
    assert group.outcome == GroupOutcome.INCONCLUSIVE
    assert any("quality_score" in reason for reason in group.rejected[0].reasons)


def test_rejects_mismatched_quality_metric(tmp_path):
    write_run(tmp_path, "r1", quality_score=0.99, quality_metric="other_metric")
    policy = TunePolicy(
        objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS,
        constraints=TuneConstraints(
            min_quality_score=0.5, required_quality_metric="m1"
        ),
    )
    report = tune(results_dirs=(tmp_path,), policy=policy)
    group = report.groups[0]
    assert any("quality_metric" in reason for reason in group.rejected[0].reasons)


def test_missing_quality_score_never_treated_as_zero(tmp_path):
    write_run(tmp_path, "r1", quality_score=None, quality_metric=None)
    policy = TunePolicy(
        objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS,
        constraints=TuneConstraints(
            min_quality_score=0.1, required_quality_metric="m1"
        ),
    )
    report = tune(results_dirs=(tmp_path,), policy=policy)
    group = report.groups[0]
    assert any(
        "missing outcome.quality_score" in reason
        for reason in group.rejected[0].reasons
    )


def test_missing_timing_total_never_treated_as_zero(tmp_path):
    write_run(tmp_path, "r1", total_ms=None)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    group = report.groups[0]
    assert group.outcome == GroupOutcome.INCONCLUSIVE
    reasons = " ".join(group.rejected[0].reasons)
    assert "missing timing.total" in reasons
    assert "cannot be computed" in reasons


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_timing_never_enters_objective_or_wins(tmp_path, value):
    write_run(tmp_path, "r-invalid", quantization="Q4", total_ms=value)
    write_run(tmp_path, "r-valid", quantization="Q8", total_ms=1000.0)

    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    group = report.groups[0]

    assert group.outcome == GroupOutcome.RECOMMENDED
    assert group.recommended.candidate_key.quantization == "Q8"
    rejected_reasons = " ".join(group.rejected[0].reasons)
    assert "timing.total is non-finite" in rejected_reasons


def test_all_non_finite_timing_is_inconclusive(tmp_path):
    write_run(tmp_path, "r-nan", total_ms=float("nan"))

    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    assert report.groups[0].outcome == GroupOutcome.INCONCLUSIVE
    assert report.groups[0].recommended is None


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_peak_memory_is_rejected(tmp_path, value):
    write_run(tmp_path, "r1", peak_bytes=value)
    policy = TunePolicy(
        objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS,
        constraints=TuneConstraints(max_peak_memory_bytes=20 * 1024**3),
    )

    report = tune(results_dirs=(tmp_path,), policy=policy)

    assert "memory.peak is non-finite" in " ".join(report.groups[0].rejected[0].reasons)


def test_non_finite_quality_score_is_rejected(tmp_path):
    write_run(
        tmp_path,
        "r1",
        quality_score=float("nan"),
        quality_metric="metric",
    )

    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    assert "quality_score is non-finite" in " ".join(
        report.groups[0].rejected[0].reasons
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_non_finite_peak_memory_without_memory_constraint_does_not_reject(
    tmp_path, value
):
    # No max_peak_memory_bytes configured: a non-finite peak measurement is
    # excluded from memory reporting but must not, by itself, block an
    # otherwise-valid candidate from being recommended (mirrors how a
    # *missing* peak measurement is only a rejection reason when memory is
    # actually a constrained axis).
    write_run(tmp_path, "r1", peak_bytes=value, total_ms=1000.0)

    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    group = report.groups[0]
    assert group.outcome == GroupOutcome.RECOMMENDED
    assert group.recommended.mean_peak_memory_bytes is None


def test_all_non_finite_evidence_across_axes_is_fully_inconclusive(tmp_path):
    # Every numeric signal a candidate could offer is NaN/Infinity: no
    # candidate may survive, and nothing may be recommended.
    write_run(
        tmp_path,
        "r1",
        total_ms=float("nan"),
        peak_bytes=float("inf"),
        quality_score=float("nan"),
        quality_metric="m1",
    )
    policy = TunePolicy(
        objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS,
        constraints=TuneConstraints(
            max_peak_memory_bytes=20 * 1024**3,
            min_quality_score=0.5,
            required_quality_metric="m1",
        ),
    )

    report = tune(results_dirs=(tmp_path,), policy=policy)

    group = report.groups[0]
    assert group.outcome == GroupOutcome.INCONCLUSIVE
    assert group.recommended is None
    assert not group.accepted
    reasons = " ".join(group.rejected[0].reasons)
    assert "timing.total is non-finite" in reasons
    assert "memory.peak is non-finite" in reasons
    assert "quality_score is non-finite" in reasons


def test_non_finite_evidence_never_produces_a_tie(tmp_path):
    # A NaN total must never be treated as "equal" to a finite total by the
    # tie-detection path; the finite candidate should win outright.
    write_run(tmp_path, "r-nan", quantization="Q4", total_ms=float("nan"))
    write_run(tmp_path, "r-inf", quantization="Q8", total_ms=float("inf"))
    write_run(tmp_path, "r-finite", quantization="Q4K", total_ms=1000.0)

    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    group = report.groups[0]
    assert group.outcome == GroupOutcome.RECOMMENDED
    assert group.recommended.candidate_key.quantization == "Q4K"
    assert len(group.accepted) == 1
    assert len(group.rejected) == 2


def test_direct_construction_rejects_non_finite_constraint_values():
    for field_name, kwargs in (
        ("min_pass_rate", {"min_pass_rate": float("nan")}),
        ("max_peak_memory_bytes", {"max_peak_memory_bytes": float("inf")}),
        ("max_total_latency_ms", {"max_total_latency_ms": float("-inf")}),
        (
            "max_coefficient_of_variation",
            {"max_coefficient_of_variation": float("nan")},
        ),
        (
            "min_quality_score",
            {"min_quality_score": float("inf"), "required_quality_metric": "m1"},
        ),
    ):
        with pytest.raises(TunePolicyError, match=field_name):
            TuneConstraints(**kwargs)


def test_missing_peak_memory_rejected_only_when_constrained(tmp_path):
    write_run(tmp_path, "r1", peak_bytes=None)

    # No memory constraint configured: candidate is accepted regardless.
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    assert report.groups[0].outcome == GroupOutcome.RECOMMENDED

    # With a memory constraint configured: missing evidence is a rejection.
    policy = TunePolicy(
        objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS,
        constraints=TuneConstraints(max_peak_memory_bytes=1024.0),
    )
    report = tune(results_dirs=(tmp_path,), policy=policy)
    group = report.groups[0]
    assert group.outcome == GroupOutcome.INCONCLUSIVE
    assert any("memory.peak" in reason for reason in group.rejected[0].reasons)


def test_rejects_over_max_peak_memory(tmp_path):
    write_run(tmp_path, "r1", peak_bytes=25 * 1024**3)
    policy = TunePolicy(
        objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS,
        constraints=TuneConstraints(max_peak_memory_bytes=20 * 1024**3),
    )
    report = tune(results_dirs=(tmp_path,), policy=policy)
    group = report.groups[0]
    assert any("peak memory" in reason for reason in group.rejected[0].reasons)


def test_rejects_over_max_total_latency(tmp_path):
    write_run(tmp_path, "r1", total_ms=5000.0)
    policy = TunePolicy(
        objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS,
        constraints=TuneConstraints(max_total_latency_ms=1000.0),
    )
    report = tune(results_dirs=(tmp_path,), policy=policy)
    group = report.groups[0]
    assert any("total latency" in reason for reason in group.rejected[0].reasons)


def test_rejects_disallowed_timing_provenance(tmp_path):
    write_run(tmp_path, "r1", total_provenance=MetricProvenance.ESTIMATED)
    policy = TunePolicy(
        objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS,
        constraints=TuneConstraints(
            allowed_provenances=frozenset({MetricProvenance.MEASURED_WALL_CLOCK})
        ),
    )
    report = tune(results_dirs=(tmp_path,), policy=policy)
    group = report.groups[0]
    assert group.outcome == GroupOutcome.INCONCLUSIVE
    assert any("provenance" in reason for reason in group.rejected[0].reasons)


def test_rejects_disallowed_memory_provenance(tmp_path):
    write_run(
        tmp_path,
        "r1",
        peak_bytes=1024.0,
        peak_provenance=MetricProvenance.ESTIMATED,
    )
    policy = TunePolicy(
        objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS,
        constraints=TuneConstraints(
            max_peak_memory_bytes=2048.0,
            allowed_provenances=frozenset({MetricProvenance.MEASURED_NATIVE}),
        ),
    )
    report = tune(results_dirs=(tmp_path,), policy=policy)
    group = report.groups[0]
    assert any(
        "memory.peak provenance" in reason for reason in group.rejected[0].reasons
    )


def test_rejects_below_min_measured_repetitions(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    policy = TunePolicy(
        objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS,
        constraints=TuneConstraints(min_measured_repetitions=2),
    )
    report = tune(results_dirs=(tmp_path,), policy=policy)
    group = report.groups[0]
    assert group.outcome == GroupOutcome.INCONCLUSIVE
    assert any("requires at least 2" in reason for reason in group.rejected[0].reasons)


def test_rejects_over_max_coefficient_of_variation(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    write_run(tmp_path, "r2", total_ms=5000.0)
    policy = TunePolicy(
        objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS,
        constraints=TuneConstraints(max_coefficient_of_variation=0.05),
    )
    report = tune(results_dirs=(tmp_path,), policy=policy)
    group = report.groups[0]
    assert group.outcome == GroupOutcome.INCONCLUSIVE
    assert any(
        "coefficient of variation" in reason for reason in group.rejected[0].reasons
    )


def test_rejects_inconsistent_quality_metric_across_repetitions(tmp_path):
    write_run(tmp_path, "r1", quality_score=1.0, quality_metric="m1")
    write_run(tmp_path, "r2", quality_score=1.0, quality_metric="m2")
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    group = report.groups[0]
    assert any("inconsistent" in reason for reason in group.rejected[0].reasons)


def test_multiple_violations_all_reported(tmp_path):
    write_run(
        tmp_path,
        "r1",
        status=RowStatus.FAILED,
        success=False,
        total_ms=10_000.0,
        peak_bytes=25 * 1024**3,
    )
    policy = TunePolicy(
        objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS,
        constraints=TuneConstraints(
            max_total_latency_ms=1000.0, max_peak_memory_bytes=20 * 1024**3
        ),
    )
    report = tune(results_dirs=(tmp_path,), policy=policy)
    reasons = report.groups[0].rejected[0].reasons
    assert any("status" in r for r in reasons)
    assert any("total latency" in r for r in reasons)
    assert any("peak memory" in r for r in reasons)


# --- Ties / inconclusive --------------------------------------------------


def test_exact_tie_is_inconclusive(tmp_path):
    write_run(tmp_path, "r-a", quantization="Q4", total_ms=1000.0)
    write_run(tmp_path, "r-b", quantization="Q8", total_ms=1000.0)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    group = report.groups[0]
    assert group.outcome == GroupOutcome.INCONCLUSIVE
    assert group.recommended is None
    assert len(group.accepted) == 2
    assert "tied" in group.inconclusive_reason


def test_tie_within_noise_is_inconclusive(tmp_path):
    # Candidate A: two reps averaging 1000ms with high variance (stdev ~500).
    write_run(tmp_path, "r-a1", quantization="Q4", total_ms=500.0)
    write_run(tmp_path, "r-a2", quantization="Q4", total_ms=1500.0)
    # Candidate B: single rep at 1050ms -- well within A's noise band.
    write_run(tmp_path, "r-b1", quantization="Q8", total_ms=1050.0)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    group = report.groups[0]
    assert group.outcome == GroupOutcome.INCONCLUSIVE
    assert "noise" in group.inconclusive_reason


def test_clearly_separated_candidates_are_not_a_tie(tmp_path):
    write_run(tmp_path, "r-a1", quantization="Q4", total_ms=990.0)
    write_run(tmp_path, "r-a2", quantization="Q4", total_ms=1010.0)
    write_run(tmp_path, "r-b1", quantization="Q8", total_ms=5000.0)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    group = report.groups[0]
    assert group.outcome == GroupOutcome.RECOMMENDED
    assert group.recommended.candidate_key.quantization == "Q4"


def test_no_candidate_survives_is_inconclusive(tmp_path):
    write_run(tmp_path, "r1", status=RowStatus.FAILED, success=False)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    group = report.groups[0]
    assert group.outcome == GroupOutcome.INCONCLUSIVE
    assert (
        group.inconclusive_reason
        == "no candidate satisfied every constraint in this policy"
    )


# --- Determinism -----------------------------------------------------------


def test_ranking_and_group_ordering_is_deterministic(tmp_path):
    write_run(tmp_path, "r-z", quantization="Q4", total_ms=1000.0, context_tier="8k")
    write_run(tmp_path, "r-a", quantization="Q4", total_ms=1000.0, context_tier="2k")
    report1 = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    report2 = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    labels1 = [g.group_key.label() for g in report1.groups]
    labels2 = [g.group_key.label() for g in report2.groups]
    assert labels1 == labels2 == sorted(labels1)


# --- Speculative candidates & baseline comparison --------------------------


def test_speculative_candidate_can_win_over_autoregressive_baseline(tmp_path):
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
    group = report.groups[0]
    assert group.outcome == GroupOutcome.RECOMMENDED
    assert group.recommended.candidate_key.speculative_enabled is True
    assert group.baseline_comparison is not None
    assert group.baseline_comparison.report.verdict.value == "improvement"


def test_unsupported_native_mtp_evidence_never_wins(tmp_path):
    write_run(tmp_path, "r-ar", speculative_enabled=False, total_ms=2000.0)
    write_run(
        tmp_path,
        "r-mtp",
        status=RowStatus.UNSUPPORTED,
        write_final_record=False,
        decode_mode="native-mtp",
        reason="native-mtp execution is not implemented by this pipeline",
    )
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    assert len(report.groups) == 1
    group = report.groups[0]
    assert group.outcome == GroupOutcome.RECOMMENDED
    assert group.recommended.candidate_key.decode_mode == "autoregressive"
    assert not any(c.candidate_key.decode_mode == "native-mtp" for c in group.accepted)
    assert not any(c.candidate_key.decode_mode == "native-mtp" for c in group.rejected)
    assert any("native-mtp" in run.reason for run in report.excluded_runs)


def test_failed_generic_draft_candidate_is_rejected_not_winning(tmp_path):
    write_run(tmp_path, "r-ar", speculative_enabled=False, total_ms=2000.0)
    write_run(
        tmp_path,
        "r-draft-failed",
        status=RowStatus.FAILED,
        success=False,
        speculative_enabled=True,
        speculative_method="draft-model",
        speculative_depth=2,
        total_ms=1.0,
    )
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    group = report.groups[0]
    assert group.outcome == GroupOutcome.RECOMMENDED
    assert group.recommended.candidate_key.speculative_enabled is False
    assert len(group.rejected) == 1
    assert group.rejected[0].candidate_key.speculative_enabled is True
