"""Adversarial coverage for persisted optimizer artifact boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import llmtracefx.optimizer.schema as schema_module
import llmtracefx.optimizer.workloads.matrix as matrix_module
import llmtracefx.optimizer.workloads.verify as verify_module
from llmtracefx.optimizer.schema import (
    CommandInfo,
    ExperimentRecord,
    ModelInfo,
    PlatformInfo,
    RepetitionInfo,
    RuntimeInfo,
    SchemaValidationError,
    utc_now_iso,
)
from llmtracefx.optimizer.tune.loader import load_evidence
from llmtracefx.optimizer.workloads.aggregate import summarize_results
from llmtracefx.optimizer.workloads.matrix import MatrixManifest, MatrixSchemaError
from llmtracefx.optimizer.workloads.verify import (
    BACKEND_OPENAI_API,
    RowVerification,
    VerifyError,
)


def _verification_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "2",
        "run_id": "run-1",
        "workload_id": "workload-1",
        "workload_version": "1",
        "category": "structured_json",
        "context_tier": "2k",
        "decode_mode": "autoregressive",
        "status": "completed",
        "reason": None,
        "recorded_prompt_hash": "sha256:abc",
        "verified_prompt_hash": "sha256:abc",
        "run_binding_hash": "sha256:def",
        "resumed": False,
        "outcome_success": True,
        "quality_score": 1.0,
        "total_ms": 100.0,
        "started_at": "2026-01-01T00:00:00Z",
        "ended_at": "2026-01-01T00:00:01Z",
        "final_record_path": "final_record.json",
        "collection_dir": "collection",
        "backend": BACKEND_OPENAI_API,
        "provider": "openrouter",
        "api_model_id": "model-1",
        "artifacts_verified": True,
    }
    payload.update(overrides)
    return payload


def _record_payload() -> dict[str, object]:
    return ExperimentRecord(
        run_id="run-1",
        started_at=utc_now_iso(),
        platform=PlatformInfo(
            os_name="Darwin", os_version="24.0", architecture="arm64"
        ),
        model=ModelInfo(model_id="model-1"),
        runtime=RuntimeInfo(name="runtime-1"),
        command=CommandInfo(argv=("runner",)),
        repetition=RepetitionInfo(
            warmup_repetitions=0,
            measured_repetitions=1,
            repetition_index=0,
        ),
    ).to_dict()


@pytest.mark.parametrize("root", [None, [], "verification", 1])
def test_row_verification_rejects_non_object_roots(root: object) -> None:
    with pytest.raises(VerifyError, match="JSON object"):
        RowVerification.from_dict(root)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", []),
        ("context_tier", {}),
        ("decode_mode", []),
        ("backend", {}),
        ("provider", []),
        ("provider", {}),
    ],
)
def test_row_verification_rejects_container_identity_fields(
    field: str, value: object
) -> None:
    with pytest.raises(VerifyError, match=field):
        RowVerification.from_dict(_verification_payload(**{field: value}))


@pytest.mark.parametrize(
    "field",
    [
        "platform",
        "model",
        "runtime",
        "command",
        "repetition",
        "tokens",
        "timing",
        "speculative",
        "memory",
        "power",
        "instruments",
        "outcome",
        "error",
    ],
)
def test_experiment_record_rejects_non_object_nested_values(field: str) -> None:
    payload = _record_payload()
    payload[field] = []
    with pytest.raises(SchemaValidationError, match="object"):
        ExperimentRecord.from_dict(payload)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model", "model_id", []),
        ("model", "model_family", {}),
        ("platform", "accelerator", []),
        ("runtime", "name", {}),
        ("runtime", "backend", []),
        ("command", "config_hash", {}),
        ("speculative", "method", []),
        ("instruments", "tool", []),
        ("instruments", "tool_version", {}),
        ("instruments", "capability", []),
        ("instruments", "template", {}),
        ("instruments", "trace_bundle_name", []),
        ("instruments", "notes", {}),
    ],
)
def test_experiment_record_rejects_container_identity_fields(
    section: str, field: str, value: object
) -> None:
    payload = _record_payload()
    if section == "instruments":
        payload[section] = {}
    payload[section][field] = value  # type: ignore[index]
    with pytest.raises(SchemaValidationError, match=field):
        ExperimentRecord.from_dict(payload)


@pytest.mark.parametrize("root", [None, [], "record", 1])
def test_experiment_record_rejects_non_object_roots(root: object) -> None:
    with pytest.raises(SchemaValidationError, match="object"):
        ExperimentRecord.from_dict(root)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{", "Invalid JSON"),
        ("[" * 10_000 + "0" + "]" * 10_000, "Invalid JSON"),
    ],
)
def test_experiment_record_wraps_truncation_and_recursion(
    payload: str, message: str
) -> None:
    with pytest.raises(SchemaValidationError, match=message):
        ExperimentRecord.from_json(payload)


def test_experiment_record_wraps_overflowing_numbers() -> None:
    payload = _record_payload()
    payload["timing"] = {
        "total": {
            "value": 10**4_000,
            "provenance": "measured_wall_clock",
            "unit": "ms",
        }
    }
    with pytest.raises(SchemaValidationError, match="finite number"):
        ExperimentRecord.from_dict(payload)


@pytest.mark.parametrize("token", ["1e400", "NaN", "Infinity", "-Infinity"])
def test_experiment_record_rejects_non_finite_json_numbers(token: str) -> None:
    payload = _record_payload()
    payload["timing"] = {
        "total": {
            "value": "NON_FINITE",
            "provenance": "measured_wall_clock",
            "unit": "ms",
        }
    }
    serialized = json.dumps(payload).replace('"NON_FINITE"', token)

    with pytest.raises(SchemaValidationError, match="finite|non-finite"):
        ExperimentRecord.from_json(serialized)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{", "Invalid JSON"),
        ("[" * 10_000 + "0" + "]" * 10_000, "Invalid JSON"),
    ],
)
def test_matrix_manifest_wraps_truncation_and_recursion(
    payload: str, message: str
) -> None:
    with pytest.raises(MatrixSchemaError, match=message):
        MatrixManifest.from_json(payload)


@pytest.mark.parametrize("root", [None, [], "manifest", 1])
def test_matrix_manifest_rejects_non_object_roots(root: object) -> None:
    with pytest.raises(MatrixSchemaError, match="JSON must be an object"):
        MatrixManifest.from_json(json.dumps(root))


def test_all_json_parsers_wrap_oversized_integer_errors(tmp_path: Path) -> None:
    huge_integer = "9" * 5_000
    verification_path = tmp_path / "verification.json"
    verification_path.write_text(f'{{"total_ms": {huge_integer}}}', encoding="utf-8")
    with pytest.raises(VerifyError):
        RowVerification.read_json(verification_path)
    with pytest.raises(MatrixSchemaError):
        MatrixManifest.from_json(f'{{"value": {huge_integer}}}')
    with pytest.raises(SchemaValidationError):
        ExperimentRecord.from_json(f'{{"value": {huge_integer}}}')


def test_row_verification_wraps_overflowing_numbers() -> None:
    with pytest.raises(VerifyError, match="finite number"):
        RowVerification.from_dict(_verification_payload(total_ms=10**4_000))


def test_matrix_manifest_rejects_malformed_nested_entry_and_prompt() -> None:
    base = {
        "schema_version": "1",
        "model_id": "model-1",
        "model_family": "family-1",
        "output_dir": "matrix",
    }
    with pytest.raises(MatrixSchemaError, match="MatrixEntry must be an object"):
        MatrixManifest.from_dict({**base, "entries": [[]]})

    entry = {
        "run_id": "run-1",
        "workload_id": "workload-1",
        "workload_version": "1",
        "category": "structured_json",
        "context_tier": "2k",
        "decode_mode": "autoregressive",
        "configured_depth": None,
        "prompt": [],
        "prompt_path": "prompt.txt",
        "runner_results_dir": "runner",
        "collector_output_dir": "collector",
        "runnable": True,
        "unsupported_reason": None,
        "command_argv": ["runner"],
        "max_tokens": 1,
    }
    with pytest.raises(MatrixSchemaError, match="MaterializedPrompt"):
        MatrixManifest.from_dict({**base, "entries": [entry]})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("workload_id", []),
        ("workload_version", {}),
        ("context_tier", []),
        ("target_context_tokens", 1.5),
        ("approx_chars_per_token", "4"),
        ("filler_segments_used", True),
        ("prompt_hash", []),
    ],
)
def test_matrix_manifest_rejects_malformed_prompt_fields(
    field: str, value: object
) -> None:
    prompt = {
        "workload_id": "workload-1",
        "workload_version": "1",
        "context_tier": "2k",
        "target_context_tokens": 2_048,
        "approx_chars_per_token": 4,
        "filler_segments_used": 1,
        "prompt_hash": "sha256:abc",
    }
    prompt[field] = value
    entry = {
        "run_id": "run-1",
        "workload_id": "workload-1",
        "workload_version": "1",
        "category": "structured_json",
        "context_tier": "2k",
        "decode_mode": "autoregressive",
        "configured_depth": None,
        "prompt": prompt,
        "prompt_path": "prompt.txt",
        "runner_results_dir": "runner",
        "collector_output_dir": "collector",
        "runnable": True,
        "unsupported_reason": None,
        "command_argv": ["runner"],
        "max_tokens": 1,
    }
    manifest = {
        "schema_version": "1",
        "model_id": "model-1",
        "model_family": "family-1",
        "output_dir": "matrix",
        "entries": [entry],
    }

    with pytest.raises(MatrixSchemaError, match=field):
        MatrixManifest.from_dict(manifest)


def test_matrix_manifest_wraps_exponent_overflow_in_prompt_integer() -> None:
    prompt = {
        "workload_id": "workload-1",
        "workload_version": "1",
        "context_tier": "2k",
        "target_context_tokens": "NON_FINITE",
        "approx_chars_per_token": 4,
        "filler_segments_used": 1,
        "prompt_hash": "sha256:abc",
    }
    entry = {
        "run_id": "run-1",
        "workload_id": "workload-1",
        "workload_version": "1",
        "category": "structured_json",
        "context_tier": "2k",
        "decode_mode": "autoregressive",
        "configured_depth": None,
        "prompt": prompt,
        "prompt_path": "prompt.txt",
        "runner_results_dir": "runner",
        "collector_output_dir": "collector",
        "runnable": True,
        "unsupported_reason": None,
        "command_argv": ["runner"],
        "max_tokens": 1,
    }
    manifest = {
        "schema_version": "1",
        "model_id": "model-1",
        "model_family": "family-1",
        "output_dir": "matrix",
        "entries": [entry],
    }
    payload = json.dumps(manifest).replace('"NON_FINITE"', "1e400")

    with pytest.raises(MatrixSchemaError, match="target_context_tokens"):
        MatrixManifest.from_json(payload)


@pytest.mark.parametrize(
    ("reader", "error_type", "limit_owner", "limit_name"),
    [
        (
            RowVerification.read_json,
            VerifyError,
            verify_module,
            "MAX_METADATA_ARTIFACT_BYTES",
        ),
        (
            MatrixManifest.read_json,
            MatrixSchemaError,
            matrix_module,
            "MAX_EVIDENCE_ARTIFACT_BYTES",
        ),
        (
            ExperimentRecord.read_json,
            SchemaValidationError,
            schema_module,
            "MAX_EVIDENCE_ARTIFACT_BYTES",
        ),
    ],
)
def test_artifact_readers_reject_invalid_utf8_oversize_and_symlinks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reader: object,
    error_type: type[ValueError],
    limit_owner: object,
    limit_name: str,
) -> None:
    path = tmp_path / "artifact.json"
    path.write_bytes(b"\xff")
    with pytest.raises(error_type, match="valid UTF-8"):
        reader(path)  # type: ignore[operator]

    monkeypatch.setattr(limit_owner, limit_name, 16)
    path.write_bytes(b"x" * 17)
    with pytest.raises(error_type, match="size limit"):
        reader(path)  # type: ignore[operator]

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(error_type, match="symlink"):
        reader(path)  # type: ignore[operator]


def test_summary_excludes_invalid_grouping_identity_without_crashing(
    tmp_path: Path,
) -> None:
    runs_dir = tmp_path / "runs"
    good_dir = runs_dir / "good"
    bad_provider_dir = runs_dir / "bad-provider"
    bad_backend_dir = runs_dir / "bad-backend"
    for run_dir in (good_dir, bad_provider_dir, bad_backend_dir):
        run_dir.mkdir(parents=True)

    (good_dir / "verification.json").write_text(
        json.dumps(_verification_payload()), encoding="utf-8"
    )
    (bad_provider_dir / "verification.json").write_text(
        json.dumps(_verification_payload(run_id="bad-provider", provider=[])),
        encoding="utf-8",
    )
    (bad_backend_dir / "verification.json").write_text(
        json.dumps(_verification_payload(run_id="bad-backend", backend={})),
        encoding="utf-8",
    )

    summary = summarize_results(tmp_path)

    assert summary.overall.total == 1
    assert [group.key for group in summary.by_provider] == ["openrouter"]
    assert [group.key for group in summary.by_backend] == [BACKEND_OPENAI_API]
    assert summary.overall.completed == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_summary_excludes_non_finite_numeric_evidence(
    tmp_path: Path, value: float
) -> None:
    run_dir = tmp_path / "runs" / "non-finite"
    run_dir.mkdir(parents=True)
    (run_dir / "verification.json").write_text(
        json.dumps(_verification_payload(total_ms=value)), encoding="utf-8"
    )

    summary = summarize_results(tmp_path)

    assert summary.overall.total == 0
    assert summary.overall.correct_cases_per_minute is None


def test_tune_loader_excludes_unhashable_record_identity(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    run_dir.mkdir(parents=True)
    final_record_path = run_dir / "final_record.json"
    verification = _verification_payload(
        backend="mlx",
        provider=None,
        final_record_path=str(final_record_path),
    )
    (run_dir / "verification.json").write_text(
        json.dumps(verification), encoding="utf-8"
    )
    record = _record_payload()
    record["model"]["model_id"] = []  # type: ignore[index]
    final_record_path.write_text(json.dumps(record), encoding="utf-8")

    loaded = load_evidence((tmp_path,))

    assert loaded.usable == ()
    assert len(loaded.excluded) == 1
    assert "model_id" in loaded.excluded[0].reason
