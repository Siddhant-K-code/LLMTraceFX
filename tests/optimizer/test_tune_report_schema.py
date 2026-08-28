"""Tests for typed ``TuneReport`` loading: ``from_dict``/``from_json``/
``read_json`` round-tripping, malformed-input rejection, and non-finite
numeric rejection.

Uses the real ``tune()`` engine over fake ``workloads run``-shaped
artifact trees (see ``_tune_fixtures.write_run``) to build realistic
reports to round-trip, plus hand-built minimal payloads to exercise
specific validation failures in isolation.
"""

from __future__ import annotations

import json
import math

import pytest
from _tune_fixtures import write_run

from llmtracefx.optimizer.tune.identity import (
    CandidateKey,
    GroupKey,
    IdentityValidationError,
)
from llmtracefx.optimizer.tune.loader import ExcludedRun, TuneInputError
from llmtracefx.optimizer.tune.policy import TuneObjective, TunePolicy
from llmtracefx.optimizer.tune.report import TuneReport, TuneReportValidationError
from llmtracefx.optimizer.tune.tuner import tune
from llmtracefx.optimizer.workloads.verify import RowStatus

LATENCY_POLICY = TunePolicy(objective=TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS)


def _minimal_report_dict() -> dict:
    return {
        "schema_version": "1",
        "generated_at": "2025-01-01T00:00:00Z",
        "results_dirs": ["results"],
        "policy": {"objective": "min_mean_total_latency_ms"},
        "groups": [],
        "excluded_runs": [],
    }


# --- Round-tripping ----------------------------------------------------------


def test_recommended_report_round_trips_through_json(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    write_run(tmp_path, "r2", total_ms=2000.0, seed=1)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    payload = report.to_json()
    roundtripped = TuneReport.from_json(payload)

    assert roundtripped.to_json() == payload
    assert roundtripped.groups[0].recommended is not None
    assert roundtripped.groups[0].recommended.candidate_key.label() == (
        report.groups[0].recommended.candidate_key.label()
    )


def test_inconclusive_report_round_trips_through_json(tmp_path):
    write_run(tmp_path, "r1", status=RowStatus.FAILED, success=False)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)

    payload = report.to_json()
    roundtripped = TuneReport.from_json(payload)

    assert roundtripped.to_json() == payload
    assert roundtripped.groups[0].recommended is None
    assert roundtripped.groups[0].inconclusive_reason is not None


def test_report_with_excluded_runs_round_trips(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    write_run(tmp_path, "r2", corrupt_final_record=True)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    assert report.excluded_runs

    payload = report.to_json()
    roundtripped = TuneReport.from_json(payload)

    assert roundtripped.to_json() == payload
    assert len(roundtripped.excluded_runs) == len(report.excluded_runs)


def test_report_with_baseline_comparison_round_trips(tmp_path):
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
    assert report.groups[0].baseline_comparison is not None

    payload = report.to_json()
    roundtripped = TuneReport.from_json(payload)

    assert roundtripped.to_json() == payload
    assert roundtripped.groups[0].baseline_comparison is not None
    assert (
        roundtripped.groups[0].baseline_comparison.report.verdict
        == report.groups[0].baseline_comparison.report.verdict
    )


def test_report_with_rejected_candidates_round_trips(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    write_run(
        tmp_path,
        "r2",
        total_ms=1500.0,
        seed=1,
        peak_bytes=25 * 1024**3,
    )
    policy = TunePolicy.from_dict(
        {
            "objective": "min_mean_total_latency_ms",
            "constraints": {"max_peak_memory_bytes": 20 * 1024**3},
        }
    )
    report = tune(results_dirs=(tmp_path,), policy=policy)
    assert report.groups[0].rejected

    payload = report.to_json()
    roundtripped = TuneReport.from_json(payload)

    assert roundtripped.to_json() == payload
    assert roundtripped.groups[0].rejected[0].reasons


def test_read_json_reads_from_file(tmp_path):
    write_run(tmp_path, "r1", total_ms=1000.0)
    report = tune(results_dirs=(tmp_path,), policy=LATENCY_POLICY)
    report_path = tmp_path / "report.json"
    report_path.write_text(report.to_json(), encoding="utf-8")

    loaded = TuneReport.read_json(report_path)

    assert loaded.to_json() == report.to_json()


# --- Malformed input rejection ------------------------------------------------


def test_from_dict_rejects_non_dict():
    with pytest.raises(TuneReportValidationError, match="JSON object"):
        TuneReport.from_dict(["not", "a", "dict"])


def test_from_dict_rejects_missing_required_field():
    data = _minimal_report_dict()
    del data["policy"]
    with pytest.raises(TuneReportValidationError, match="policy"):
        TuneReport.from_dict(data)


def test_from_dict_rejects_invalid_policy():
    data = _minimal_report_dict()
    data["policy"] = {"objective": "not-a-real-objective"}
    with pytest.raises(TuneReportValidationError, match="policy"):
        TuneReport.from_dict(data)


def test_from_dict_rejects_unsupported_schema_version():
    data = _minimal_report_dict()
    data["schema_version"] = "999"
    with pytest.raises(TuneReportValidationError, match="schema_version"):
        TuneReport.from_dict(data)


def test_from_json_rejects_invalid_json():
    with pytest.raises(TuneReportValidationError, match="invalid JSON"):
        TuneReport.from_json("not json")


def test_from_dict_rejects_group_missing_group_key():
    data = _minimal_report_dict()
    data["groups"] = [{"outcome": "inconclusive", "inconclusive_reason": "x"}]
    with pytest.raises(TuneReportValidationError, match="group_key"):
        TuneReport.from_dict(data)


def test_from_dict_rejects_recommended_outcome_without_recommended_candidate():
    data = _minimal_report_dict()
    data["groups"] = [
        {
            "group_key": {
                "workload_id": "w",
                "workload_version": "1",
                "context_tier": "2k",
                "model_id": "m",
                "model_family": None,
                "accelerator": None,
                "runtime_name": "r",
                "runtime_backend": None,
                "workload_prompt_hash": "h",
            },
            "outcome": "recommended",
            "recommended": None,
            "accepted": [],
            "rejected": [],
            "inconclusive_reason": None,
            "baseline_comparison": None,
        }
    ]
    with pytest.raises(TuneReportValidationError, match="recommended"):
        TuneReport.from_dict(data)


def test_from_dict_rejects_inconclusive_outcome_without_reason():
    data = _minimal_report_dict()
    data["groups"] = [
        {
            "group_key": {
                "workload_id": "w",
                "workload_version": "1",
                "context_tier": "2k",
                "model_id": "m",
                "model_family": None,
                "accelerator": None,
                "runtime_name": "r",
                "runtime_backend": None,
                "workload_prompt_hash": "h",
            },
            "outcome": "inconclusive",
            "recommended": None,
            "accepted": [],
            "rejected": [],
            "inconclusive_reason": None,
            "baseline_comparison": None,
        }
    ]
    with pytest.raises(TuneReportValidationError, match="inconclusive_reason"):
        TuneReport.from_dict(data)


def test_from_dict_rejects_excluded_run_missing_field():
    data = _minimal_report_dict()
    data["excluded_runs"] = [{"run_id": "r1", "reason": "x"}]
    with pytest.raises(Exception, match="source_results_dir"):
        TuneReport.from_dict(data)


# --- Non-finite numeric rejection ---------------------------------------------


def test_from_json_rejects_non_finite_objective_value():
    data = _minimal_report_dict()
    data["groups"] = [
        {
            "group_key": {
                "workload_id": "w",
                "workload_version": "1",
                "context_tier": "2k",
                "model_id": "m",
                "model_family": None,
                "accelerator": None,
                "runtime_name": "r",
                "runtime_backend": None,
                "workload_prompt_hash": "h",
            },
            "outcome": "recommended",
            "recommended": {
                "candidate_key": {
                    "decode_mode": "autoregressive",
                    "runtime_version": None,
                    "quantization": None,
                    "model_revision": None,
                    "tokenizer_revision": None,
                    "speculative_enabled": False,
                    "speculative_method": None,
                    "speculative_configured_depth": None,
                    "seed": None,
                    "config_hash": None,
                },
                "rank": 1,
                "run_ids": ["r1"],
                "verification_paths": [],
                "final_record_paths": [],
                "evidence_count": 1,
                "objective_name": "min_mean_total_latency_ms",
                "objective_value": math.nan,
                "mean_total_latency_ms": None,
                "stdev_total_latency_ms": None,
                "coefficient_of_variation": None,
                "correct_cases_per_minute": None,
                "pass_rate": None,
                "mean_quality_score": None,
                "quality_metric": None,
                "mean_peak_memory_bytes": None,
                "max_peak_memory_bytes": None,
            },
            "accepted": [],
            "rejected": [],
            "inconclusive_reason": None,
            "baseline_comparison": None,
        }
    ]
    payload = json.dumps(data, allow_nan=True)
    assert "NaN" in payload

    with pytest.raises(TuneReportValidationError, match="finite"):
        TuneReport.from_json(payload)


def test_from_dict_rejects_infinite_peak_memory():
    data = _minimal_report_dict()
    data["groups"] = [
        {
            "group_key": {
                "workload_id": "w",
                "workload_version": "1",
                "context_tier": "2k",
                "model_id": "m",
                "model_family": None,
                "accelerator": None,
                "runtime_name": "r",
                "runtime_backend": None,
                "workload_prompt_hash": "h",
            },
            "outcome": "recommended",
            "recommended": {
                "candidate_key": {
                    "decode_mode": "autoregressive",
                    "runtime_version": None,
                    "quantization": None,
                    "model_revision": None,
                    "tokenizer_revision": None,
                    "speculative_enabled": False,
                    "speculative_method": None,
                    "speculative_configured_depth": None,
                    "seed": None,
                    "config_hash": None,
                },
                "rank": 1,
                "run_ids": ["r1"],
                "verification_paths": [],
                "final_record_paths": [],
                "evidence_count": 1,
                "objective_name": "min_mean_total_latency_ms",
                "objective_value": 1.0,
                "mean_total_latency_ms": None,
                "stdev_total_latency_ms": None,
                "coefficient_of_variation": None,
                "correct_cases_per_minute": None,
                "pass_rate": None,
                "mean_quality_score": None,
                "quality_metric": None,
                "mean_peak_memory_bytes": float("inf"),
                "max_peak_memory_bytes": None,
            },
            "accepted": [],
            "rejected": [],
            "inconclusive_reason": None,
            "baseline_comparison": None,
        }
    ]
    with pytest.raises(TuneReportValidationError, match="finite"):
        TuneReport.from_dict(data)


# --- GroupKey / CandidateKey / ExcludedRun from_dict --------------------------


def test_group_key_round_trips_through_dict():
    key = GroupKey(
        workload_id="w",
        workload_version="1",
        context_tier="2k",
        model_id="m",
        model_family="fam",
        accelerator="acc",
        runtime_name="rt",
        runtime_backend="Metal",
        workload_prompt_hash="sha256:abc",
    )
    assert GroupKey.from_dict(key.to_dict()) == key


def test_group_key_from_dict_rejects_non_dict():
    with pytest.raises(IdentityValidationError, match="JSON object"):
        GroupKey.from_dict("not a dict")


def test_group_key_from_dict_rejects_missing_field():
    data = {
        "workload_id": "w",
        "workload_version": "1",
        "context_tier": "2k",
        "model_id": "m",
        "model_family": None,
        "accelerator": None,
        "runtime_name": "rt",
        "runtime_backend": None,
        # workload_prompt_hash omitted
    }
    with pytest.raises(IdentityValidationError, match="workload_prompt_hash"):
        GroupKey.from_dict(data)


def test_candidate_key_round_trips_through_dict():
    key = CandidateKey(
        decode_mode="autoregressive",
        runtime_version="1.0",
        quantization="Q4",
        model_revision=None,
        tokenizer_revision=None,
        speculative_enabled=True,
        speculative_method="draft-model",
        speculative_configured_depth=2,
        seed=0,
        config_hash="hash",
    )
    assert CandidateKey.from_dict(key.to_dict()) == key


def test_candidate_key_from_dict_rejects_non_boolean_speculative_enabled():
    data = CandidateKey(
        decode_mode="autoregressive",
        runtime_version=None,
        quantization=None,
        model_revision=None,
        tokenizer_revision=None,
        speculative_enabled=False,
        speculative_method=None,
        speculative_configured_depth=None,
        seed=None,
        config_hash=None,
    ).to_dict()
    data["speculative_enabled"] = "yes"
    with pytest.raises(IdentityValidationError, match="speculative_enabled"):
        CandidateKey.from_dict(data)


def test_excluded_run_round_trips_through_dict():
    excluded = ExcludedRun(run_id="r1", source_results_dir="results", reason="unusable")
    assert ExcludedRun.from_dict(excluded.to_dict()) == excluded


def test_excluded_run_from_dict_rejects_missing_field():
    with pytest.raises(TuneInputError, match="reason"):
        ExcludedRun.from_dict({"run_id": "r1", "source_results_dir": "results"})


def test_excluded_run_from_dict_rejects_non_string_field():
    with pytest.raises(TuneInputError, match="run_id"):
        ExcludedRun.from_dict(
            {"run_id": 123, "source_results_dir": "results", "reason": "x"}
        )
