"""Tests for the tune policy schema: validation, JSON/YAML loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmtracefx.optimizer.schema import MetricProvenance
from llmtracefx.optimizer.tune.policy import (
    TuneConstraints,
    TuneObjective,
    TunePolicy,
    TunePolicyError,
)
from llmtracefx.optimizer.workloads.verify import RowStatus


def test_default_constraints_require_completed_or_skipped_only():
    constraints = TuneConstraints()
    assert constraints.required_statuses == {RowStatus.COMPLETED, RowStatus.SKIPPED}
    assert constraints.min_measured_repetitions == 1


def test_required_statuses_rejects_disallowed_status():
    with pytest.raises(TunePolicyError, match="required_statuses"):
        TuneConstraints(required_statuses=frozenset({RowStatus.FAILED}))


def test_required_statuses_rejects_empty():
    with pytest.raises(TunePolicyError, match="non-empty"):
        TuneConstraints(required_statuses=frozenset())


def test_min_quality_score_requires_required_quality_metric():
    with pytest.raises(TunePolicyError, match="required_quality_metric"):
        TuneConstraints(min_quality_score=0.9)


def test_min_quality_score_with_metric_is_valid():
    constraints = TuneConstraints(
        min_quality_score=0.9,
        required_quality_metric="structured_json_exact_field_match",
    )
    assert constraints.min_quality_score == 0.9


def test_min_measured_repetitions_must_be_positive():
    with pytest.raises(TunePolicyError, match="min_measured_repetitions"):
        TuneConstraints(min_measured_repetitions=0)


def test_policy_round_trips_through_json():
    policy = TunePolicy(
        objective=TuneObjective.MAX_CORRECT_CASES_PER_MINUTE,
        constraints=TuneConstraints(
            min_pass_rate=0.5,
            max_peak_memory_bytes=1024.0,
            allowed_provenances=frozenset({MetricProvenance.MEASURED_WALL_CLOCK}),
        ),
        name="example",
    )
    reloaded = TunePolicy.from_json(policy.to_json())
    assert reloaded.objective == TuneObjective.MAX_CORRECT_CASES_PER_MINUTE
    assert reloaded.constraints.min_pass_rate == 0.5
    assert reloaded.constraints.max_peak_memory_bytes == 1024.0
    assert reloaded.constraints.allowed_provenances == frozenset(
        {MetricProvenance.MEASURED_WALL_CLOCK}
    )
    assert reloaded.name == "example"


def test_from_dict_missing_objective_raises():
    with pytest.raises(TunePolicyError, match="objective"):
        TunePolicy.from_dict({})


def test_from_dict_invalid_objective_raises():
    with pytest.raises(TunePolicyError, match="objective"):
        TunePolicy.from_dict({"objective": "max_vibes"})


def test_from_json_invalid_json_raises():
    with pytest.raises(TunePolicyError, match="invalid JSON"):
        TunePolicy.from_json("not json")


def test_from_file_loads_json(tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps({"objective": "min_mean_total_latency_ms"}), encoding="utf-8"
    )
    policy = TunePolicy.from_file(policy_path)
    assert policy.objective == TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS


def test_from_file_loads_yaml(tmp_path):
    yaml = pytest.importorskip("yaml")
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        yaml.safe_dump(
            {
                "objective": "max_correct_cases_per_minute",
                "constraints": {
                    "required_quality_metric": "structured_json_exact_field_match",
                    "min_quality_score": 0.8,
                },
            }
        ),
        encoding="utf-8",
    )
    policy = TunePolicy.from_file(policy_path)
    assert policy.objective == TuneObjective.MAX_CORRECT_CASES_PER_MINUTE
    assert policy.constraints.min_quality_score == 0.8


def test_from_file_unsupported_extension_raises(tmp_path):
    policy_path = tmp_path / "policy.txt"
    policy_path.write_text("{}", encoding="utf-8")
    with pytest.raises(TunePolicyError, match="unsupported"):
        TunePolicy.from_file(policy_path)


def test_from_dict_rejects_non_object_constraints():
    with pytest.raises(TunePolicyError, match="constraints"):
        TunePolicy.from_dict(
            {"objective": "min_mean_total_latency_ms", "constraints": "nope"}
        )


def test_allowed_provenances_rejects_unknown_value():
    with pytest.raises(TunePolicyError, match="provenance"):
        TuneConstraints.from_dict({"allowed_provenances": ["telepathy"]})


def test_max_peak_memory_bytes_must_be_positive():
    with pytest.raises(TunePolicyError, match="max_peak_memory_bytes"):
        TuneConstraints.from_dict({"max_peak_memory_bytes": -1})


def test_min_pass_rate_must_be_within_unit_interval():
    with pytest.raises(TunePolicyError, match="min_pass_rate"):
        TuneConstraints.from_dict({"min_pass_rate": 1.5})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_peak_memory_bytes", float("nan")),
        ("max_peak_memory_bytes", float("inf")),
        ("max_total_latency_ms", float("-inf")),
        ("max_coefficient_of_variation", float("nan")),
        ("min_pass_rate", float("nan")),
        ("min_quality_score", float("inf")),
    ],
)
def test_numeric_constraints_reject_non_finite_values(field, value):
    data = {field: value}
    if field == "min_quality_score":
        data["required_quality_metric"] = "metric"
    with pytest.raises(TunePolicyError, match=field):
        TuneConstraints.from_dict(data)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_policy_json_rejects_non_standard_non_finite_numbers(token):
    payload = (
        '{"objective":"min_mean_total_latency_ms",'
        f'"constraints":{{"max_total_latency_ms":{token}}}}}'
    )
    with pytest.raises(TunePolicyError, match="max_total_latency_ms"):
        TunePolicy.from_json(payload)


EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples" / "optimizer"


@pytest.mark.parametrize(
    "filename",
    [
        "tune-policy-fastest-under-20gb-m5-pro.json",
        "tune-policy-structured-json-throughput.json",
    ],
)
def test_example_policy_files_load_and_validate(filename):
    policy = TunePolicy.from_file(EXAMPLES_DIR / filename)
    assert policy.objective in (
        TuneObjective.MIN_MEAN_TOTAL_LATENCY_MS,
        TuneObjective.MAX_CORRECT_CASES_PER_MINUTE,
    )
    assert policy.constraints.required_statuses
