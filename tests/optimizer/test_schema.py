"""Tests for the canonical experiment/evidence schema."""

import json

import pytest

from llmtracefx.optimizer.schema import (
    SCHEMA_VERSION,
    CommandInfo,
    ExperimentRecord,
    Measurement,
    MetricProvenance,
    ModelInfo,
    PlatformInfo,
    RepetitionInfo,
    RuntimeInfo,
    SchemaValidationError,
    SpeculativeDecodingInfo,
    TimingMetrics,
    TokenCounts,
    utc_now_iso,
)


def make_record(**overrides):
    defaults = {
        "run_id": "run-1",
        "started_at": utc_now_iso(),
        "platform": PlatformInfo(
            os_name="Darwin", os_version="24.0", architecture="arm64"
        ),
        "model": ModelInfo(model_id="Qwen/Qwen3.8-27B", quantization="Q4_K_M"),
        "runtime": RuntimeInfo(name="llama.cpp", version="b1234", backend="Metal"),
        "command": CommandInfo(argv=("llama-cli", "-m", "model.gguf")),
        "repetition": RepetitionInfo(
            warmup_repetitions=1, measured_repetitions=3, repetition_index=0
        ),
        "tokens": TokenCounts(input_tokens=50, generated_tokens=200),
        "timing": TimingMetrics(
            total=Measurement(
                value=5160.2, provenance=MetricProvenance.MEASURED_NATIVE, unit="ms"
            )
        ),
    }
    defaults.update(overrides)
    return ExperimentRecord(**defaults)


def test_utc_now_iso_ends_with_z():
    assert utc_now_iso().endswith("Z")


def test_measurement_round_trip():
    measurement = Measurement(value=1.5, provenance=MetricProvenance.DERIVED, unit="ms")
    restored = Measurement.from_dict(measurement.to_dict())
    assert restored == measurement


def test_measurement_from_dict_rejects_missing_field():
    with pytest.raises(SchemaValidationError):
        Measurement.from_dict({"value": 1.0})


def test_measurement_from_dict_rejects_bad_provenance():
    with pytest.raises(SchemaValidationError):
        Measurement.from_dict({"value": 1.0, "provenance": "not-a-real-provenance"})


def test_experiment_record_round_trips_through_json():
    record = make_record()
    restored = ExperimentRecord.from_json(record.to_json())
    assert restored == record


def test_experiment_record_json_is_plain_json_serializable():
    record = make_record()
    # to_dict() must not leak Enum objects or other non-JSON-native types.
    payload = json.loads(record.to_json())
    assert payload["timing"]["total"]["provenance"] == "measured_native"
    assert payload["schema_version"] == SCHEMA_VERSION


def test_write_and_read_json_round_trip(tmp_path):
    record = make_record()
    path = tmp_path / "run.json"
    record.write_json(path)

    restored = ExperimentRecord.read_json(path)
    assert restored == record
    # No leftover temp file from the atomic write.
    assert list(tmp_path.iterdir()) == [path]


def test_optional_fields_default_to_none_not_zero():
    record = make_record()
    assert record.memory.active is None
    assert record.power.energy is None
    assert record.tokens.context_tokens is None


def test_validate_rejects_wrong_schema_version():
    record = make_record(schema_version="999")
    with pytest.raises(SchemaValidationError, match="Unsupported schema_version"):
        record.validate()


def test_validate_rejects_empty_run_id():
    record = make_record(run_id="")
    with pytest.raises(SchemaValidationError, match="run_id"):
        record.validate()


def test_validate_rejects_empty_command():
    record = make_record(command=CommandInfo(argv=()))
    with pytest.raises(SchemaValidationError, match="command.argv"):
        record.validate()


def test_validate_rejects_negative_token_counts():
    record = make_record(tokens=TokenCounts(input_tokens=-1))
    with pytest.raises(SchemaValidationError, match="tokens.input_tokens"):
        record.validate()


def test_validate_rejects_negative_measurement():
    record = make_record(
        timing=TimingMetrics(
            total=Measurement(
                value=-5.0, provenance=MetricProvenance.MEASURED_NATIVE, unit="ms"
            )
        )
    )
    with pytest.raises(SchemaValidationError, match="timing.total"):
        record.validate()


def test_validate_rejects_accepted_exceeding_proposed_tokens():
    record = make_record(
        speculative=SpeculativeDecodingInfo(
            enabled=True, proposed_tokens=10, accepted_tokens=11
        )
    )
    with pytest.raises(SchemaValidationError, match="accepted_tokens cannot exceed"):
        record.validate()


def test_validate_rejects_success_true_with_error_set():
    from llmtracefx.optimizer.schema import ErrorInfo, OutcomeInfo

    record = make_record(
        error=ErrorInfo(category="timeout", message="boom"),
        outcome=OutcomeInfo(success=True),
    )
    with pytest.raises(SchemaValidationError, match="error is set"):
        record.validate()


def test_speculative_acceptance_rate():
    spec = SpeculativeDecodingInfo(
        enabled=True, proposed_tokens=320, accepted_tokens=96
    )
    assert spec.acceptance_rate == pytest.approx(0.3)


def test_speculative_acceptance_rate_none_when_not_proposed():
    spec = SpeculativeDecodingInfo(enabled=False)
    assert spec.acceptance_rate is None


def test_from_json_rejects_malformed_json():
    with pytest.raises(SchemaValidationError, match="Invalid JSON"):
        ExperimentRecord.from_json("{not json")


def test_from_dict_rejects_missing_required_field():
    record = make_record()
    payload = record.to_dict()
    del payload["run_id"]
    with pytest.raises(SchemaValidationError, match="missing required field"):
        ExperimentRecord.from_dict(payload)
