"""Tests for the speculative-decoding regression doctor rule."""

import dataclasses
import json
from pathlib import Path

import pytest

from llmtracefx.optimizer.doctor.speculative import (
    DoctorVerdict,
    comparability_key,
    diagnose_speculative_regression,
)
from llmtracefx.optimizer.parsers.llama_cpp import build_experiment_record
from llmtracefx.optimizer.schema import (
    CommandInfo,
    ExperimentRecord,
    ModelInfo,
    OutcomeInfo,
    PlatformInfo,
    RepetitionInfo,
    SchemaValidationError,
    utc_now_iso,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "llama_cpp"


def _read_fixture(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def _platform(
    architecture: str = "arm64", accelerator: str | None = None
) -> PlatformInfo:
    return PlatformInfo(
        os_name="Darwin",
        os_version="24.0",
        architecture=architecture,
        accelerator=accelerator,
    )


def _model() -> ModelInfo:
    return ModelInfo(model_id="Qwen/Qwen3.8-27B", quantization="Q4_K_M")


def _record_from_fixture(
    fixture_name, run_id, *, speculative_method=None, platform=None, success=True
):
    return build_experiment_record(
        run_id=run_id,
        started_at=utc_now_iso(),
        platform=platform or _platform(),
        model=_model(),
        command=CommandInfo(argv=("llama-cli", "-m", "qwen3.8-27b-q4.gguf")),
        repetition=RepetitionInfo(
            warmup_repetitions=1, measured_repetitions=2, repetition_index=0
        ),
        stdout_text=_read_fixture(fixture_name),
        runtime_version="b4500",
        speculative_method=speculative_method,
        outcome=OutcomeInfo(success=success),
    )


def _baseline_records():
    return [
        _record_from_fixture("qwen3_8b_baseline_run1.log", "baseline-1"),
        _record_from_fixture("qwen3_8b_baseline_run2.log", "baseline-2"),
    ]


def test_detects_clear_regression():
    baseline = _baseline_records()
    speculative = [
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run1.log", "mtp-1", speculative_method="mtp"
        ),
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run2.log", "mtp-2", speculative_method="mtp"
        ),
    ]

    report = diagnose_speculative_regression(baseline, speculative)

    assert report.verdict == DoctorVerdict.REGRESSION
    assert report.baseline_run_ids == ("baseline-1", "baseline-2")
    assert report.speculative_run_ids == ("mtp-1", "mtp-2")
    assert report.delta_ms is not None and report.delta_ms > 0
    assert (
        "regression" not in report.reason
    )  # reason should read naturally, not repeat the enum
    assert "increased" in report.reason


def test_detects_clear_improvement():
    baseline = _baseline_records()
    speculative = [
        _record_from_fixture(
            "qwen3_8b_mtp_improvement_run1.log", "mtp-1", speculative_method="mtp"
        ),
        _record_from_fixture(
            "qwen3_8b_mtp_improvement_run2.log", "mtp-2", speculative_method="mtp"
        ),
    ]

    report = diagnose_speculative_regression(baseline, speculative)

    assert report.verdict == DoctorVerdict.IMPROVEMENT
    assert report.delta_ms is not None and report.delta_ms < 0


def test_inconclusive_when_no_speculative_runs_supplied():
    report = diagnose_speculative_regression(_baseline_records(), [])
    assert report.verdict == DoctorVerdict.INCONCLUSIVE
    assert "speculative-decoding" in report.reason


def test_inconclusive_when_no_baseline_runs_supplied():
    speculative = [
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run1.log", "mtp-1", speculative_method="mtp"
        ),
    ]
    report = diagnose_speculative_regression([], speculative)
    assert report.verdict == DoctorVerdict.INCONCLUSIVE
    assert "baseline" in report.reason


def test_inconclusive_when_hardware_differs():
    baseline = _baseline_records()
    speculative = [
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run1.log",
            "mtp-1",
            speculative_method="mtp",
            platform=_platform("x86_64"),
        ),
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run2.log",
            "mtp-2",
            speculative_method="mtp",
            platform=_platform("x86_64"),
        ),
    ]

    report = diagnose_speculative_regression(baseline, speculative)

    assert report.verdict == DoctorVerdict.INCONCLUSIVE
    assert "not comparable" in report.reason


def test_inconclusive_when_accelerator_differs():
    # Same OS/architecture, same model/runtime/workload -- only the
    # accelerator identity differs. Runs must not be treated as
    # comparable just because everything else matches.
    baseline = _baseline_records()
    speculative = [
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run1.log",
            "mtp-1",
            speculative_method="mtp",
            platform=_platform(accelerator="NVIDIA RTX 4090 24GB"),
        ),
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run2.log",
            "mtp-2",
            speculative_method="mtp",
            platform=_platform(accelerator="NVIDIA RTX 4090 24GB"),
        ),
    ]

    report = diagnose_speculative_regression(baseline, speculative)

    assert report.verdict == DoctorVerdict.INCONCLUSIVE
    assert "not comparable" in report.reason


def test_comparability_key_differs_by_accelerator():
    baseline = _record_from_fixture("qwen3_8b_baseline_run1.log", "baseline-1")
    other_gpu = _record_from_fixture(
        "qwen3_8b_baseline_run1.log",
        "baseline-1-other-gpu",
        platform=_platform(accelerator="NVIDIA RTX 4090 24GB"),
    )

    assert baseline.platform.accelerator == "Apple M5 Pro"
    assert comparability_key(baseline) != comparability_key(other_gpu)


def test_inconclusive_when_too_few_repetitions():
    baseline = [_baseline_records()[0]]
    speculative = [
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run1.log", "mtp-1", speculative_method="mtp"
        ),
    ]

    report = diagnose_speculative_regression(baseline, speculative, min_repetitions=2)

    assert report.verdict == DoctorVerdict.INCONCLUSIVE
    assert "repetitions" in report.reason


def test_inconclusive_when_delta_smaller_than_noise():
    # Two baseline runs that are identical to two "speculative" runs
    # (other than the enabled flag) so the tiny delta is pure noise.
    baseline = _baseline_records()
    speculative = [
        _record_from_fixture(
            "qwen3_8b_baseline_run1.log", "mtp-1", speculative_method="mtp"
        ),
        _record_from_fixture(
            "qwen3_8b_baseline_run2.log", "mtp-2", speculative_method="mtp"
        ),
    ]
    # Force speculative.enabled True even though the fixture has no
    # speculative counters, by re-parsing with a synthetic addition.
    import dataclasses

    speculative = [
        dataclasses.replace(
            record,
            speculative=dataclasses.replace(
                record.speculative, enabled=True, method="mtp"
            ),
        )
        for record in speculative
    ]

    report = diagnose_speculative_regression(baseline, speculative)

    assert report.verdict == DoctorVerdict.INCONCLUSIVE
    assert "noise" in report.reason


def test_no_significant_difference_within_threshold():
    import dataclasses

    baseline = _baseline_records()
    # Build "speculative" runs with total times ~1% higher than baseline,
    # which is below the default 3% significance threshold, but pad the
    # apparent noise down by keeping identical repeated values so the
    # noise-vs-delta check does not itself trigger inconclusive.
    speculative = []
    for index, record in enumerate(baseline):
        bumped_total = dataclasses.replace(
            record.timing.total, value=record.timing.total.value * 1.01
        )
        bumped_timing = dataclasses.replace(record.timing, total=bumped_total)
        speculative.append(
            dataclasses.replace(
                record,
                run_id=f"mtp-{index + 1}",
                timing=bumped_timing,
                speculative=dataclasses.replace(
                    record.speculative, enabled=True, method="mtp"
                ),
            )
        )

    report = diagnose_speculative_regression(baseline, speculative)

    assert report.verdict in (
        DoctorVerdict.NO_SIGNIFICANT_DIFFERENCE,
        DoctorVerdict.INCONCLUSIVE,
    )


def test_comparability_key_matches_for_same_workload():
    a, b = _baseline_records()
    assert comparability_key(a) == comparability_key(b)


def test_ineligible_records_without_total_timing_are_excluded():
    import dataclasses

    baseline = _baseline_records()
    speculative = [
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run1.log", "mtp-1", speculative_method="mtp"
        ),
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run2.log", "mtp-2", speculative_method="mtp"
        ),
    ]
    # Drop total timing from one speculative record -- it must be
    # excluded rather than crash the comparison.
    speculative[1] = dataclasses.replace(
        speculative[1], timing=dataclasses.replace(speculative[1].timing, total=None)
    )

    report = diagnose_speculative_regression(baseline, speculative, min_repetitions=1)
    assert report.speculative_run_ids == ("mtp-1",)


def _with_total_ms(record: ExperimentRecord, value: float) -> ExperimentRecord:
    bumped_total = dataclasses.replace(record.timing.total, value=value)
    return dataclasses.replace(
        record, timing=dataclasses.replace(record.timing, total=bumped_total)
    )


def test_inconclusive_when_baseline_mean_is_zero():
    # A generic Measurement only requires value >= 0, so an all-zero
    # baseline total time is schema-valid but is not usable evidence:
    # the old code computed delta_pct=None here and then crashed trying
    # to format it as a percentage.
    baseline = [_with_total_ms(record, 0.0) for record in _baseline_records()]
    speculative = [
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run1.log", "mtp-1", speculative_method="mtp"
        ),
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run2.log", "mtp-2", speculative_method="mtp"
        ),
    ]

    report = diagnose_speculative_regression(baseline, speculative)

    assert report.verdict == DoctorVerdict.INCONCLUSIVE
    assert "baseline" in report.reason
    assert "finite" in report.reason or "positive" in report.reason
    assert report.delta_pct is None


def test_inconclusive_when_baseline_mean_is_negative():
    # Not reachable through ExperimentRecord.validate() (which rejects
    # negative timing values), but a defensive guard nonetheless, since
    # this function accepts any ExperimentRecord instances, validated or
    # not (e.g. built via dataclasses.replace in tests/tooling).
    baseline = [_with_total_ms(record, -10.0) for record in _baseline_records()]
    speculative = [
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run1.log", "mtp-1", speculative_method="mtp"
        ),
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run2.log", "mtp-2", speculative_method="mtp"
        ),
    ]

    report = diagnose_speculative_regression(baseline, speculative)

    assert report.verdict == DoctorVerdict.INCONCLUSIVE
    assert "baseline" in report.reason


def test_inconclusive_when_baseline_mean_is_nan():
    baseline = [_with_total_ms(record, float("nan")) for record in _baseline_records()]
    speculative = [
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run1.log", "mtp-1", speculative_method="mtp"
        ),
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run2.log", "mtp-2", speculative_method="mtp"
        ),
    ]

    report = diagnose_speculative_regression(baseline, speculative)

    assert report.verdict == DoctorVerdict.INCONCLUSIVE
    assert "finite" in report.reason


def test_inconclusive_when_baseline_mean_is_infinite():
    baseline = [_with_total_ms(record, float("inf")) for record in _baseline_records()]
    speculative = [
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run1.log", "mtp-1", speculative_method="mtp"
        ),
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run2.log", "mtp-2", speculative_method="mtp"
        ),
    ]

    report = diagnose_speculative_regression(baseline, speculative)

    assert report.verdict == DoctorVerdict.INCONCLUSIVE
    assert "finite" in report.reason


def test_inconclusive_when_speculative_mean_is_non_finite():
    baseline = _baseline_records()
    speculative = [
        _with_total_ms(
            _record_from_fixture(
                "qwen3_8b_mtp_regression_run1.log", "mtp-1", speculative_method="mtp"
            ),
            float("nan"),
        ),
        _record_from_fixture(
            "qwen3_8b_mtp_regression_run2.log", "mtp-2", speculative_method="mtp"
        ),
    ]

    report = diagnose_speculative_regression(baseline, speculative)

    assert report.verdict == DoctorVerdict.INCONCLUSIVE
    assert "speculative-decoding" in report.reason
    assert "finite" in report.reason


def test_experiment_record_from_dict_rejects_malformed_success_before_reaching_doctor():
    # A malformed persisted outcome.success ("false" is truthy as a
    # Python string) must fail at deserialization time, so it can never
    # reach _eligible_totals()/diagnose_speculative_regression() and
    # silently poison eligibility bucketing by looking "successful".
    baseline = _baseline_records()
    payload = baseline[0].to_dict()
    payload["outcome"]["success"] = "false"
    with pytest.raises(SchemaValidationError, match="OutcomeInfo.success"):
        ExperimentRecord.from_dict(payload)


def test_experiment_record_from_dict_rejects_malformed_speculative_enabled_before_reaching_doctor():
    # Same guarantee for speculative.enabled, which drives the
    # baseline-vs-speculative bucketing in _eligible_totals(): a
    # malformed value must fail before it can be mis-bucketed.
    baseline = _baseline_records()
    payload = baseline[0].to_dict()
    payload["speculative"]["enabled"] = "false"
    with pytest.raises(SchemaValidationError, match="SpeculativeDecodingInfo.enabled"):
        ExperimentRecord.from_dict(payload)


def _improvement_records():
    return [
        _record_from_fixture(
            f"qwen3_8b_mtp_improvement_run{index}.log",
            f"mtp-{index}",
            speculative_method="mtp",
        )
        for index in (1, 2)
    ]


def _acceptance(status="measured", *, steps=100, drafted=300, accepted=150):
    block = {"status": status, "reason": "test"}
    if status == "measured":
        block["counters"] = {
            "verify_steps": {"value": steps},
            "draft_tokens": {"value": drafted},
            "accepted_tokens": {"value": accepted},
        }
    return {"speculative": block}


def test_server_acceptance_consistent_with_latency_keeps_latency_basis():
    report = diagnose_speculative_regression(
        _baseline_records(),
        _improvement_records(),
        server_evidence={"mtp-1": _acceptance(), "mtp-2": _acceptance()},
    )

    assert report.verdict == DoctorVerdict.IMPROVEMENT
    assert report.basis == "client_latency"
    assert report.acceptance is not None
    assert report.acceptance["acceptance_length"] == 2.5
    assert report.acceptance["acceptance_rate"] == 0.5
    assert report.acceptance["measured_runs"] == 2
    assert "consistent" in report.reason


def test_speedup_beyond_the_acceptance_length_is_inconclusive():
    # 1.24x faster, but the server only emitted 1.1 tokens per step.
    evidence = _acceptance(steps=100, drafted=300, accepted=10)
    report = diagnose_speculative_regression(
        _baseline_records(),
        _improvement_records(),
        server_evidence={"mtp-1": evidence, "mtp-2": evidence},
    )

    assert report.verdict == DoctorVerdict.INCONCLUSIVE
    assert report.basis == "server_acceptance"
    assert "cannot explain" in report.reason


def test_zero_accepted_drafts_is_a_regression_on_server_evidence():
    baseline = _baseline_records()
    speculative = [
        dataclasses.replace(
            record,
            run_id=f"mtp-{index}",
            speculative=dataclasses.replace(
                record.speculative, enabled=True, method="mtp"
            ),
        )
        for index, record in enumerate(baseline, start=1)
    ]
    evidence = _acceptance(steps=100, drafted=300, accepted=0)

    latency_only = diagnose_speculative_regression(baseline, speculative)
    report = diagnose_speculative_regression(
        baseline,
        speculative,
        server_evidence={"mtp-1": evidence, "mtp-2": evidence},
    )

    assert latency_only.verdict != DoctorVerdict.REGRESSION
    assert report.verdict == DoctorVerdict.REGRESSION
    assert report.basis == "server_acceptance"
    assert "none of 600 draft tokens" in report.reason


def test_a_window_without_verification_steps_is_inconclusive():
    report = diagnose_speculative_regression(
        _baseline_records(),
        _improvement_records(),
        server_evidence={"mtp-1": _acceptance(), "mtp-2": _acceptance("no_drafts")},
    )

    assert report.verdict == DoctorVerdict.INCONCLUSIVE
    assert report.basis == "server_acceptance"
    assert "mtp-2" in report.reason


@pytest.mark.parametrize(
    ("evidence", "covered"),
    [
        (None, None),
        ({"mtp-1": _acceptance()}, "covers 1 of 2"),
        ({"mtp-1": _acceptance("not_exported")}, "covers 0 of 2"),
    ],
    ids=["none", "partial", "not-exported"],
)
def test_latency_is_the_fallback_without_full_server_acceptance(evidence, covered):
    report = diagnose_speculative_regression(
        _baseline_records(), _improvement_records(), server_evidence=evidence
    )

    assert report.verdict == DoctorVerdict.IMPROVEMENT
    assert report.basis == "client_latency"
    if covered is None:
        assert report.acceptance is None
    else:
        assert covered in report.reason


def test_doctor_cli_reads_server_evidence_next_to_each_record(tmp_path, capsys):
    from llmtracefx.optimizer import cli

    def write(record, name, evidence=None):
        folder = tmp_path / name
        folder.mkdir()
        record.write_json(folder / "record.json")
        if evidence is not None:
            evidence = {"kind": "llmtracefx.server_metrics.evidence", **evidence}
            (folder / "server_metrics_evidence.json").write_text(json.dumps(evidence))
        return str(folder / "record.json")

    baseline = [write(r, r.run_id) for r in _baseline_records()]
    short = _acceptance(steps=100, drafted=300, accepted=10)
    speculative = [write(r, r.run_id, short) for r in _improvement_records()]

    with pytest.raises(SystemExit) as exc:
        cli.main(
            [
                "doctor",
                "speculative",
                "--baseline",
                *baseline,
                "--speculative",
                *speculative,
            ]
        )
    payload = json.loads(capsys.readouterr().out)

    assert exc.value.code == 2
    assert payload["basis"] == "server_acceptance"
    assert payload["acceptance"]["run_status"] == {
        "mtp-1": "measured",
        "mtp-2": "measured",
    }
