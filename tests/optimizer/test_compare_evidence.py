"""Tests for comparable-unit/system identity and the offline evidence loader.

Everything here is synthetic. Nothing loads a model, calls an API, or runs a
benchmark; the fixtures write artifact trees directly.
"""

from __future__ import annotations

import json
from itertools import permutations
from pathlib import Path

import pytest
from _compare_fixtures import (
    api_evidence_payload,
    collection_dir_for,
    edit_sidecar,
    refresh_artifact_marker,
    reseal_run,
    write_api_run,
    write_run,
)

from llmtracefx.optimizer.compare.evidence import (
    MAX_EXACT_TOKEN_COUNT,
    ApiEvidence,
    ApiEvidenceError,
    CompareEvidenceError,
    decode_settings_from_argv,
    load_comparison_evidence,
)
from llmtracefx.optimizer.compare.identity import (
    ComparableUnitKey,
    CompareIdentityError,
    SystemKey,
)
from llmtracefx.optimizer.workloads.verify import RowStatus


def _unit(**overrides: object) -> ComparableUnitKey:
    payload: dict[str, object] = {
        "workload_id": "w",
        "workload_version": "1",
        "workload_prompt_hash": "sha256:abc",
        "context_tier": "2k",
        "quality_metric": "exact",
        "max_output_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    payload.update(overrides)
    return ComparableUnitKey.from_dict(payload)


def _system(**overrides: object) -> SystemKey:
    payload: dict[str, object] = {
        "model_id": "glm-5.3",
        "model_revision": None,
        "provider": "z-ai",
        "runtime_name": "openai-compatible-stream",
        "runtime_backend": None,
        "accelerator": None,
        "quantization": None,
        "reasoning_effort": "high",
        "decode_mode": "autoregressive",
    }
    payload.update(overrides)
    return SystemKey.from_dict(payload)


# --- Identity keys --------------------------------------------------------


def test_unit_key_requires_its_mandatory_fields() -> None:
    for missing in (
        "workload_id",
        "workload_version",
        "workload_prompt_hash",
        "context_tier",
    ):
        payload = _unit().to_dict()
        del payload[missing]
        with pytest.raises(CompareIdentityError, match=missing):
            ComparableUnitKey.from_dict(payload)


def test_unit_key_rejects_a_non_object() -> None:
    with pytest.raises(CompareIdentityError, match="must be a JSON object"):
        ComparableUnitKey.from_dict(["not", "an", "object"])


def test_unrecorded_max_output_is_not_a_wildcard() -> None:
    recorded = _unit(max_output_tokens=512)
    unrecorded = _unit(max_output_tokens=None)
    assert recorded != unrecorded
    assert "unrecorded" in unrecorded.label()


def test_unit_key_rejects_a_zero_or_negative_output_cap() -> None:
    with pytest.raises(CompareIdentityError, match=">= 1"):
        _unit(max_output_tokens=0)


def test_unit_key_rejects_non_finite_sampling_values() -> None:
    payload = _unit().to_dict()
    payload["temperature"] = float("nan")
    with pytest.raises(CompareIdentityError, match="finite"):
        ComparableUnitKey.from_dict(payload)


def test_unit_keys_differing_on_any_axis_are_distinct() -> None:
    base = _unit()
    for field, value in (
        ("workload_id", "other"),
        ("workload_version", "2"),
        ("workload_prompt_hash", "sha256:zzz"),
        ("context_tier", "32k"),
        ("quality_metric", "other"),
        ("max_output_tokens", 128),
        ("temperature", 0.7),
        ("top_p", 0.9),
    ):
        assert _unit(**{field: value}) != base


def test_an_unrecorded_setting_never_shares_a_sort_key_with_a_recorded_one() -> None:
    """A sentinel-based sort key used to collide these two distinct units."""
    unrecorded = _unit(temperature=None)
    recorded_negative = _unit(temperature=-1.0)
    assert unrecorded != recorded_negative
    assert unrecorded.sort_key() != recorded_negative.sort_key()

    unrecorded_cap = _unit(max_output_tokens=None)
    assert unrecorded_cap.sort_key() != _unit(max_output_tokens=1).sort_key()


def test_sort_keys_order_units_deterministically() -> None:
    units = [
        _unit(context_tier="32k"),
        _unit(context_tier="2k"),
        _unit(context_tier="8k"),
    ]
    expected = [
        unit.to_dict() for unit in sorted(units, key=lambda unit: unit.sort_key())
    ]
    for arrangement in permutations(units):
        ordered = sorted(arrangement, key=lambda unit: unit.sort_key())
        assert [unit.to_dict() for unit in ordered] == expected


def test_system_locality_follows_from_the_provider_field() -> None:
    assert _system(provider=None).is_local is True
    assert _system(provider="z-ai").is_local is False


def test_system_key_rejects_a_contradictory_locality_claim() -> None:
    payload = _system(provider="z-ai").to_dict()
    payload["is_local"] = True
    with pytest.raises(CompareIdentityError, match="contradicts the provider"):
        SystemKey.from_dict(payload)


def test_system_keys_differing_on_any_label_axis_are_distinct() -> None:
    base = _system()
    for field, value in (
        ("model_id", "glm-5.3-flash"),
        ("model_revision", "2026-06"),
        ("provider", "other"),
        ("runtime_name", "mlx-lm"),
        ("runtime_backend", "Metal"),
        ("accelerator", "Apple M5 Pro"),
        ("quantization", "Q4"),
        ("reasoning_effort", "low"),
        ("decode_mode", "native-mtp"),
    ):
        assert _system(**{field: value}) != base


def test_system_label_names_every_axis_a_reader_needs() -> None:
    label = _system(quantization="Q4", model_revision="2026-06").label()
    assert "glm-5.3@2026-06" in label
    assert "z-ai" in label
    assert "quant=Q4" in label
    assert "reasoning=high" in label
    assert "decode=autoregressive" in label


# --- API evidence sidecar --------------------------------------------------


def test_api_evidence_reads_usage_settings_and_ttft() -> None:
    evidence = ApiEvidence.from_dict(
        api_evidence_payload(
            run_id="r1",
            cached_prompt_tokens=250,
            reasoning_tokens=90,
        )
    )
    assert evidence.provider == "z-ai"
    assert evidence.reasoning_effort == "high"
    assert evidence.usage.prompt_tokens == 1000
    assert evidence.usage.cached_prompt_tokens == 250
    assert evidence.usage.reasoning_tokens == 90
    assert evidence.decode_settings.max_output_tokens == 512
    assert evidence.decode_settings.source == "api_request_plan"
    assert evidence.client_ttft_ms == pytest.approx(220.0)


def test_api_evidence_requires_a_plan() -> None:
    with pytest.raises(ApiEvidenceError, match="'plan'"):
        ApiEvidence.from_dict({"usage": {}})


def test_api_evidence_rejects_a_non_integer_token_count() -> None:
    payload = api_evidence_payload(run_id="r1")
    payload["usage"]["prompt_tokens"] = "1000"
    with pytest.raises(ApiEvidenceError, match="prompt_tokens"):
        ApiEvidence.from_dict(payload)


def test_api_evidence_rejects_a_negative_token_count() -> None:
    payload = api_evidence_payload(run_id="r1")
    payload["usage"]["completion_tokens"] = -1
    with pytest.raises(ApiEvidenceError, match=">= 0"):
        ApiEvidence.from_dict(payload)


@pytest.mark.parametrize(
    "field",
    ["prompt_tokens", "completion_tokens", "cached_prompt_tokens", "reasoning_tokens"],
)
def test_api_evidence_rejects_a_token_count_too_large_to_bill(field: str) -> None:
    """A count above 2**53 cannot round trip through a float.

    The collector caps counts when it writes them, but this layer reads
    artifacts written by earlier builds, and the counts are
    provider-controlled either way. Without this bound the count reaches
    ``estimate_run_cost`` and raises ``OverflowError``, which is an
    ``ArithmeticError`` and so is caught by nothing downstream.
    """
    payload = api_evidence_payload(run_id="r1")
    payload["usage"][field] = MAX_EXACT_TOKEN_COUNT + 1
    with pytest.raises(ApiEvidenceError, match="exact calculation"):
        ApiEvidence.from_dict(payload)


def test_api_evidence_accepts_a_token_count_at_the_exact_bound() -> None:
    payload = api_evidence_payload(run_id="r1")
    payload["usage"]["prompt_tokens"] = MAX_EXACT_TOKEN_COUNT
    assert ApiEvidence.from_dict(payload).usage.prompt_tokens == MAX_EXACT_TOKEN_COUNT


def test_an_oversized_count_excludes_the_run_rather_than_crashing(
    tmp_path: Path,
) -> None:
    write_api_run(tmp_path, "api-1")

    def mutate(payload: dict) -> None:
        payload["usage"]["prompt_tokens"] = 10**400

    edit_sidecar(tmp_path, "api-1", mutate)

    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "exact calculation" in loaded.excluded[0].reason


def test_a_sidecar_past_the_json_integer_cap_excludes_the_run(
    tmp_path: Path,
) -> None:
    """``json`` raises a plain ``ValueError``, not ``JSONDecodeError``, here."""
    write_api_run(tmp_path, "api-1")
    sidecar = tmp_path / "runs" / "api-1" / "collection" / "api_evidence.json"
    huge = "1" * 5000
    sidecar.write_text(
        '{"plan": {"provider": "z-ai"}, "usage": {"prompt_tokens": ' + huge + "}}",
        encoding="utf-8",
    )
    refresh_artifact_marker(collection_dir_for(tmp_path, "api-1"))
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "could not read" in loaded.excluded[0].reason


def test_a_deeply_nested_sidecar_excludes_the_run(tmp_path: Path) -> None:
    """Deep nesting raises ``RecursionError``, which is not a ``ValueError``."""
    write_api_run(tmp_path, "api-1")
    sidecar = tmp_path / "runs" / "api-1" / "collection" / "api_evidence.json"
    depth = 200_000
    sidecar.write_text("[" * depth + "]" * depth, encoding="utf-8")
    refresh_artifact_marker(collection_dir_for(tmp_path, "api-1"))
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "could not read" in loaded.excluded[0].reason


def test_api_evidence_rejects_a_non_finite_ttft() -> None:
    payload = api_evidence_payload(run_id="r1")
    payload["timeline"]["first_content_token_offset_ms"] = float("inf")
    with pytest.raises(ApiEvidenceError, match="finite"):
        ApiEvidence.from_dict(payload)


def test_api_evidence_rejects_a_zero_output_cap() -> None:
    payload = api_evidence_payload(run_id="r1")
    payload["plan"]["request_parameters"]["max_tokens"] = 0
    with pytest.raises(ApiEvidenceError, match=">= 1"):
        ApiEvidence.from_dict(payload)


def test_api_evidence_rejects_a_non_boolean_reported_flag() -> None:
    payload = api_evidence_payload(run_id="r1")
    payload["usage"]["reported"] = "yes"
    with pytest.raises(ApiEvidenceError, match="usage.reported"):
        ApiEvidence.from_dict(payload)


# --- Decode settings from argv --------------------------------------------


def test_decode_settings_read_the_projects_own_long_options() -> None:
    settings = decode_settings_from_argv(
        ("prog", "--max-tokens", "512", "--temperature", "0.2", "--top-p", "0.95")
    )
    assert settings.max_output_tokens == 512
    assert settings.temperature == pytest.approx(0.2)
    assert settings.top_p == pytest.approx(0.95)
    assert settings.source == "command_argv"


def test_decode_settings_are_unrecorded_when_absent() -> None:
    settings = decode_settings_from_argv(("prog", "--run-id", "r1"))
    assert settings.max_output_tokens is None
    assert settings.source is None


def test_conflicting_repeated_options_yield_nothing_rather_than_a_guess() -> None:
    settings = decode_settings_from_argv(
        ("prog", "--max-tokens", "512", "--max-tokens", "128")
    )
    assert settings.max_output_tokens is None


def test_repeated_identical_options_are_accepted() -> None:
    settings = decode_settings_from_argv(
        ("prog", "--max-tokens", "512", "--max-tokens", "512")
    )
    assert settings.max_output_tokens == 512


def test_unparsable_option_values_are_ignored() -> None:
    settings = decode_settings_from_argv(("prog", "--max-tokens", "many"))
    assert settings.max_output_tokens is None


# --- Loading --------------------------------------------------------------


def test_loader_requires_at_least_one_results_directory() -> None:
    with pytest.raises(CompareEvidenceError, match="at least one"):
        load_comparison_evidence(())


def test_local_run_is_labeled_local_with_argv_decode_settings(tmp_path: Path) -> None:
    write_run(tmp_path, "local-1")
    loaded = load_comparison_evidence((tmp_path,))
    assert len(loaded.runs) == 1
    run = loaded.runs[0]
    assert run.system_key.is_local is True
    assert run.system_key.reasoning_effort is None
    assert run.unit_key.max_output_tokens == 512
    assert run.decode_settings.source == "command_argv"
    assert run.local_prefill_ms == pytest.approx(310.0)


def test_api_run_is_labeled_by_provider_and_reasoning_effort(tmp_path: Path) -> None:
    write_api_run(tmp_path, "api-1", reasoning_effort="low")
    run = load_comparison_evidence((tmp_path,)).runs[0]
    assert run.system_key.provider == "z-ai"
    assert run.system_key.is_local is False
    assert run.system_key.reasoning_effort == "low"
    assert run.decode_settings.source == "api_request_plan"
    assert run.api_evidence is not None
    assert run.api_evidence.usage.prompt_tokens == 1000


def test_local_and_api_runs_share_a_unit_when_settings_match(tmp_path: Path) -> None:
    write_run(tmp_path, "local-1", max_tokens_argv=512)
    write_api_run(tmp_path, "api-1", max_output_tokens=512)
    runs = load_comparison_evidence((tmp_path,)).runs
    assert len({run.unit_key for run in runs}) == 1
    assert len({run.system_key for run in runs}) == 2


def test_differing_output_caps_split_the_unit(tmp_path: Path) -> None:
    write_run(tmp_path, "local-1", max_tokens_argv=512)
    write_api_run(tmp_path, "api-1", max_output_tokens=128)
    runs = load_comparison_evidence((tmp_path,)).runs
    assert len({run.unit_key for run in runs}) == 2


def test_a_malformed_sidecar_excludes_the_run(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "api-1",
        provider="z-ai",
        api_evidence_text="{not json",
    )
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert len(loaded.excluded) == 1
    assert "api_evidence.json" in loaded.excluded[0].reason


def test_a_sidecar_naming_another_model_excludes_the_run(tmp_path: Path) -> None:
    write_api_run(tmp_path, "api-1", model_id="glm-5.3")

    def mutate(payload: dict) -> None:
        payload["plan"]["model_id"] = "something-else"

    edit_sidecar(tmp_path, "api-1", mutate)
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "plan.model_id" in loaded.excluded[0].reason
    assert "does not match the final record" in loaded.excluded[0].reason


def test_a_sidecar_naming_another_provider_excludes_the_run(tmp_path: Path) -> None:
    write_api_run(tmp_path, "api-1")

    def mutate(payload: dict) -> None:
        payload["plan"]["provider"] = "someone-else"

    edit_sidecar(tmp_path, "api-1", mutate)
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "plan.provider" in loaded.excluded[0].reason


def test_a_hosted_sidecar_over_a_provider_less_record_excludes_the_run(
    tmp_path: Path,
) -> None:
    """A hosted run whose record omits the provider must not pass as local.

    ``SystemKey`` reads locality from the record alone, so accepting this
    pair would label a hosted system local, publish local-only peak memory
    for it, and exempt it from every provider-keyed pricing lookup.
    """
    write_api_run(tmp_path, "api-1")
    record_path = tmp_path / "runs" / "api-1" / "final_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["runtime"]["provider"] = None
    record["memory"]["peak"] = {
        "value": 9 * 1024**3,
        "provenance": "measured_native",
        "unit": "bytes",
    }
    record_path.write_text(json.dumps(record), encoding="utf-8")
    reseal_run(tmp_path, "api-1")

    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert len(loaded.excluded) == 1
    assert "claims this run was local" in loaded.excluded[0].reason


def test_a_sidecar_that_names_no_provider_still_requires_one_on_the_record(
    tmp_path: Path,
) -> None:
    """The guard keys on the sidecar existing, not on it naming a provider.

    Only a hosted-API collector writes an ``api_evidence.json`` at all, so a
    sidecar that happens to omit ``plan.provider`` is still proof the run was
    not local. Keying the check on ``plan.provider`` would leave this shape
    wide open.
    """
    write_api_run(tmp_path, "api-1")

    def mutate(payload: dict) -> None:
        payload["plan"]["provider"] = None

    edit_sidecar(tmp_path, "api-1", mutate)

    record_path = tmp_path / "runs" / "api-1" / "final_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["runtime"]["provider"] = None
    record_path.write_text(json.dumps(record), encoding="utf-8")
    reseal_run(tmp_path, "api-1")

    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "claims this run was local" in loaded.excluded[0].reason


def test_a_sidecar_that_names_no_provider_is_fine_when_the_record_does(
    tmp_path: Path,
) -> None:
    write_api_run(tmp_path, "api-1")

    def mutate(payload: dict) -> None:
        payload["plan"]["provider"] = None

    edit_sidecar(tmp_path, "api-1", mutate)

    run = load_comparison_evidence((tmp_path,)).runs[0]
    assert run.system_key.provider == "z-ai"
    assert run.system_key.is_local is False


def test_a_local_run_with_no_sidecar_is_still_accepted_as_local(
    tmp_path: Path,
) -> None:
    """The hosted-provider check must not catch genuinely local runs."""
    write_run(tmp_path, "local-1")
    run = load_comparison_evidence((tmp_path,)).runs[0]
    assert run.system_key.is_local is True


def test_a_corrupt_final_record_excludes_the_run(tmp_path: Path) -> None:
    write_run(tmp_path, "local-1", corrupt_final_record=True)
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert len(loaded.excluded) == 1


def test_an_unsupported_row_is_excluded(tmp_path: Path) -> None:
    write_run(
        tmp_path,
        "local-1",
        status=RowStatus.UNSUPPORTED,
        write_final_record=False,
        reason="native MTP is unavailable here",
    )
    loaded = load_comparison_evidence((tmp_path,))
    assert loaded.runs == ()
    assert "native MTP" in loaded.excluded[0].reason


def test_the_same_run_measured_twice_is_kept_as_two_repetitions(
    tmp_path: Path,
) -> None:
    """Two results trees holding one matrix row are two repetitions.

    A matrix ``run_id`` names the task, so every repetition of a matrix
    carries the same ids and the same system identity as the first. Calling
    that a conflict rejected the only way repetitions are ever collected.
    """
    first = tmp_path / "a"
    second = tmp_path / "b"
    write_run(first, "shared", total_ms=1000.0)
    write_run(second, "shared", total_ms=2000.0)
    loaded = load_comparison_evidence((first, second))
    assert len(loaded.runs) == 2
    assert {run.record.timing.total.value for run in loaded.runs} == {1000.0, 2000.0}


def test_an_exactly_duplicated_directory_is_deduplicated(tmp_path: Path) -> None:
    write_run(tmp_path, "local-1")
    loaded = load_comparison_evidence((tmp_path, tmp_path))
    assert len(loaded.runs) == 1


def test_a_relative_collection_dir_is_resolved_against_the_results_dir(
    tmp_path: Path,
) -> None:
    write_api_run(tmp_path, "api-1")
    verification_path = tmp_path / "runs" / "api-1" / "verification.json"
    payload = json.loads(verification_path.read_text(encoding="utf-8"))
    payload["collection_dir"] = "runs/api-1/collection"
    verification_path.write_text(json.dumps(payload), encoding="utf-8")
    reseal_run(tmp_path, "api-1")
    run = load_comparison_evidence((tmp_path,)).runs[0]
    assert run.api_evidence is not None
