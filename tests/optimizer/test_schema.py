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
    OutcomeInfo,
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


@pytest.mark.parametrize(
    "malformed_value", ["not-a-number", True, False, None, 1.5, [1]]
)
@pytest.mark.parametrize(
    "field_name",
    ["warmup_repetitions", "measured_repetitions", "repetition_index"],
)
def test_repetition_info_from_dict_rejects_malformed_required_ints(
    field_name, malformed_value
):
    payload = {
        "warmup_repetitions": 1,
        "measured_repetitions": 2,
        "repetition_index": 0,
        field_name: malformed_value,
    }
    with pytest.raises(SchemaValidationError, match=f"RepetitionInfo.{field_name}"):
        RepetitionInfo.from_dict(payload)


@pytest.mark.parametrize("malformed_value", ["not-a-number", True, False, 1.5, [1]])
def test_repetition_info_from_dict_rejects_malformed_seed(malformed_value):
    payload = {
        "warmup_repetitions": 1,
        "measured_repetitions": 2,
        "repetition_index": 0,
        "seed": malformed_value,
    }
    with pytest.raises(SchemaValidationError, match="RepetitionInfo.seed"):
        RepetitionInfo.from_dict(payload)


def test_repetition_info_from_dict_accepts_valid_ints_and_none_seed():
    info = RepetitionInfo.from_dict(
        {"warmup_repetitions": 1, "measured_repetitions": 2, "repetition_index": 0}
    )
    assert info.seed is None
    assert info.warmup_repetitions == 1


@pytest.mark.parametrize("malformed_value", ["not-a-number", True, False, 1.5, [1]])
@pytest.mark.parametrize(
    "field_name", ["input_tokens", "context_tokens", "generated_tokens"]
)
def test_token_counts_from_dict_rejects_malformed_optional_ints(
    field_name, malformed_value
):
    with pytest.raises(SchemaValidationError, match=f"TokenCounts.{field_name}"):
        TokenCounts.from_dict({field_name: malformed_value})


def test_token_counts_from_dict_accepts_missing_and_none():
    counts = TokenCounts.from_dict({"input_tokens": None})
    assert counts.input_tokens is None
    assert counts.context_tokens is None
    assert counts.generated_tokens is None


def test_token_counts_from_dict_allows_negative_int_value():
    # from_dict accepts a well-typed negative int; range checks are the
    # job of ExperimentRecord.validate(), not deserialization.
    counts = TokenCounts.from_dict({"input_tokens": -1})
    assert counts.input_tokens == -1


@pytest.mark.parametrize("malformed_value", ["not-a-number", True, False, 1.5, [1]])
@pytest.mark.parametrize(
    "field_name", ["configured_depth", "proposed_tokens", "accepted_tokens"]
)
def test_speculative_decoding_info_from_dict_rejects_malformed_optional_ints(
    field_name, malformed_value
):
    with pytest.raises(
        SchemaValidationError, match=f"SpeculativeDecodingInfo.{field_name}"
    ):
        SpeculativeDecodingInfo.from_dict({field_name: malformed_value})


def test_command_info_from_dict_rejects_scalar_string_argv():
    # A scalar string is iterable, so a naive tuple(argv) would silently
    # explode "llama-cli" into its individual characters.
    with pytest.raises(SchemaValidationError, match="argv"):
        CommandInfo.from_dict({"argv": "llama-cli -m model.gguf"})


def test_command_info_from_dict_rejects_missing_argv():
    with pytest.raises(SchemaValidationError, match="missing required field"):
        CommandInfo.from_dict({})


def test_command_info_from_dict_rejects_empty_argv():
    with pytest.raises(SchemaValidationError, match="argv"):
        CommandInfo.from_dict({"argv": []})


@pytest.mark.parametrize(
    "malformed_argv", [["llama-cli", 123], ["llama-cli", ""], [None]]
)
def test_command_info_from_dict_rejects_malformed_argv_entries(malformed_argv):
    with pytest.raises(SchemaValidationError, match="argv"):
        CommandInfo.from_dict({"argv": malformed_argv})


def test_command_info_from_dict_accepts_list_of_strings():
    info = CommandInfo.from_dict({"argv": ["llama-cli", "-m", "model.gguf"]})
    assert info.argv == ("llama-cli", "-m", "model.gguf")


# --- Adjacent audit: PlatformInfo / OutcomeInfo malformed float fields ------


@pytest.mark.parametrize("malformed_value", ["8", True, [8], {}])
def test_platform_info_from_dict_rejects_malformed_cpu_cores(malformed_value):
    with pytest.raises(SchemaValidationError, match="cpu_cores"):
        PlatformInfo.from_dict(
            {
                "os_name": "Darwin",
                "os_version": "24.0",
                "architecture": "arm64",
                "cpu_cores": malformed_value,
            }
        )


@pytest.mark.parametrize("malformed_value", ["24.0", True, [24]])
def test_platform_info_from_dict_rejects_malformed_total_memory_gb(malformed_value):
    with pytest.raises(SchemaValidationError, match="total_memory_gb"):
        PlatformInfo.from_dict(
            {
                "os_name": "Darwin",
                "os_version": "24.0",
                "architecture": "arm64",
                "total_memory_gb": malformed_value,
            }
        )


def test_platform_info_from_dict_accepts_valid_numeric_fields():
    platform = PlatformInfo.from_dict(
        {
            "os_name": "Darwin",
            "os_version": "24.0",
            "architecture": "arm64",
            "cpu_cores": 18,
            "total_memory_gb": 24,  # int is a valid float-compatible input
        }
    )
    assert platform.cpu_cores == 18
    assert platform.total_memory_gb == 24.0


def test_platform_info_from_dict_allows_missing_and_none_numeric_fields():
    platform = PlatformInfo.from_dict(
        {
            "os_name": "Darwin",
            "os_version": "24.0",
            "architecture": "arm64",
            "cpu_cores": None,
            "total_memory_gb": None,
        }
    )
    assert platform.cpu_cores is None
    assert platform.total_memory_gb is None


@pytest.mark.parametrize("malformed_value", ["0.9", True, [0.9]])
def test_outcome_info_from_dict_rejects_malformed_quality_score(malformed_value):
    with pytest.raises(SchemaValidationError, match="quality_score"):
        OutcomeInfo.from_dict({"quality_score": malformed_value})


def test_outcome_info_from_dict_accepts_valid_quality_score():
    outcome = OutcomeInfo.from_dict({"quality_score": 0.87})
    assert outcome.quality_score == pytest.approx(0.87)


def test_outcome_info_from_dict_allows_missing_quality_score():
    outcome = OutcomeInfo.from_dict({})
    assert outcome.quality_score is None
