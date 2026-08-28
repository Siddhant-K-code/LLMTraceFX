"""Tests for native-MTP capability detection, checkpoint validation, and
honest unsupported-result evidence collection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from llmtracefx.optimizer.collectors.native_mtp import (
    MLX_LM_STRIPS_MTP_WEIGHTS_FAMILIES,
    MLX_VLM_EXPERIMENTAL_MTP_FAMILIES,
    NativeMTPCapabilityReport,
    NativeMTPCollectionConfig,
    NativeMTPCollectorError,
    collect_native_mtp,
    detect_native_mtp_capability,
    validate_checkpoint_compatibility,
)
from llmtracefx.optimizer.schema import ExperimentRecord


def _write_config(path: Path, **fields) -> None:
    path.mkdir(parents=True, exist_ok=True)
    payload = {"model_type": "qwen3_next", "hidden_size": 4096, "vocab_size": 151936}
    payload.update(fields)
    (path / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def make_config(tmp_path: Path, **overrides) -> NativeMTPCollectionConfig:
    target_path = tmp_path / "target"
    sidecar_path = tmp_path / "sidecar"
    if "target_model_path" not in overrides:
        _write_config(target_path)
    if "mtp_sidecar_path" not in overrides:
        _write_config(sidecar_path)
    values = {
        "run_id": "native-mtp-1",
        "target_model_path": target_path,
        "mtp_sidecar_path": sidecar_path,
        "model_id": "local/qwen3.8-27b",
        "prompt": "test prompt",
        "output_dir": tmp_path / "artifacts",
        "command_argv": (
            "llmtracefx-optimizer",
            "native-mtp",
            "collect",
            "--target-model-path",
            str(target_path),
        ),
    }
    values.update(overrides)
    return NativeMTPCollectionConfig(**values)


# --- Capability detection ----------------------------------------------


@pytest.mark.parametrize("family", sorted(MLX_LM_STRIPS_MTP_WEIGHTS_FAMILIES))
def test_known_mlx_lm_families_report_unsupported(family):
    report = detect_native_mtp_capability(
        family, mlx_lm_version="0.31.3", mlx_vlm_version=None
    )
    assert report.supported is False
    assert "strips" in report.reason
    assert family in report.reason


@pytest.mark.parametrize("family", sorted(MLX_VLM_EXPERIMENTAL_MTP_FAMILIES))
def test_known_mlx_vlm_experimental_families_report_unsupported(family):
    report = detect_native_mtp_capability(
        family, mlx_lm_version="0.31.3", mlx_vlm_version="0.5.0"
    )
    assert report.supported is False
    assert "experimental" in report.reason
    assert "draft_model request path" in report.reason


def test_unknown_family_reports_unsupported_conservatively():
    report = detect_native_mtp_capability(
        "totally-unknown-family", mlx_lm_version=None, mlx_vlm_version=None
    )
    assert report.supported is False
    assert "not in this module's verified list" in report.reason


def test_capability_report_round_trips_through_json():
    report = detect_native_mtp_capability(
        "qwen3_next", mlx_lm_version="0.31.3", mlx_vlm_version=None
    )
    restored = NativeMTPCapabilityReport.from_dict(json.loads(report.to_json()))
    assert restored == report


def test_capability_report_from_dict_rejects_missing_field():
    with pytest.raises(NativeMTPCollectorError):
        NativeMTPCapabilityReport.from_dict({"model_family": "qwen3_next"})


# --- Checkpoint compatibility validation --------------------------------


def test_validate_checkpoint_compatibility_accepts_matching_arch():
    target = {"hidden_size": 4096, "vocab_size": 151936}
    sidecar = {"hidden_size": 4096, "vocab_size": 151936}
    validate_checkpoint_compatibility(target, sidecar)  # must not raise


def test_validate_checkpoint_compatibility_rejects_hidden_size_mismatch():
    target = {"hidden_size": 4096, "vocab_size": 151936}
    sidecar = {"hidden_size": 2048, "vocab_size": 151936}
    with pytest.raises(NativeMTPCollectorError, match="hidden_size"):
        validate_checkpoint_compatibility(target, sidecar)


def test_validate_checkpoint_compatibility_rejects_vocab_size_mismatch():
    target = {"hidden_size": 4096, "vocab_size": 151936}
    sidecar = {"hidden_size": 4096, "vocab_size": 32000}
    with pytest.raises(NativeMTPCollectorError, match="vocab_size"):
        validate_checkpoint_compatibility(target, sidecar)


def test_validate_checkpoint_compatibility_rejects_missing_target_signature():
    with pytest.raises(NativeMTPCollectorError, match="target checkpoint"):
        validate_checkpoint_compatibility({}, {"hidden_size": 4096})


def test_validate_checkpoint_compatibility_unwraps_vlm_text_config():
    target = {"text_config": {"hidden_size": 4096, "vocab_size": 151936}}
    sidecar = {"text_config": {"hidden_size": 4096, "vocab_size": 151936}}
    validate_checkpoint_compatibility(target, sidecar)  # must not raise


def test_validate_checkpoint_compatibility_accepts_differing_layer_counts():
    # A native-MTP sidecar/drafter legitimately has far fewer layers than
    # its target model -- that must never be treated as an incompatibility.
    target = {"hidden_size": 4096, "vocab_size": 151936, "num_hidden_layers": 48}
    sidecar = {"hidden_size": 4096, "vocab_size": 151936, "num_hidden_layers": 1}
    validate_checkpoint_compatibility(target, sidecar)  # must not raise


def test_validate_checkpoint_compatibility_rejects_layer_only_sidecar_signature():
    target = {"hidden_size": 4096, "vocab_size": 151936, "num_hidden_layers": 48}
    sidecar = {"num_hidden_layers": 1}
    with pytest.raises(NativeMTPCollectorError, match="sidecar checkpoint"):
        validate_checkpoint_compatibility(target, sidecar)


def test_validate_checkpoint_compatibility_still_rejects_hidden_size_mismatch_with_differing_layers():
    target = {"hidden_size": 4096, "vocab_size": 151936, "num_hidden_layers": 48}
    sidecar = {"hidden_size": 2048, "vocab_size": 151936, "num_hidden_layers": 1}
    with pytest.raises(NativeMTPCollectorError, match="hidden_size") as exc_info:
        validate_checkpoint_compatibility(target, sidecar)
    # Layer counts are surfaced as informational context, not as their own
    # enforced requirement.
    assert "informational, not enforced" in str(exc_info.value)
    assert "num_hidden_layers=48" in str(exc_info.value)
    assert "num_hidden_layers=1" in str(exc_info.value)


# --- Config validation ---------------------------------------------------


def test_missing_target_model_path_is_rejected_without_downloading(tmp_path):
    with pytest.raises(NativeMTPCollectorError, match="target_model_path"):
        make_config(tmp_path, target_model_path=tmp_path / "missing-target")


def test_missing_sidecar_path_is_rejected_without_downloading(tmp_path):
    with pytest.raises(NativeMTPCollectorError, match="mtp_sidecar_path"):
        make_config(tmp_path, mtp_sidecar_path=tmp_path / "missing-sidecar")


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_tokens": True},
        {"max_tokens": 0},
        {"seed": False},
        {"configured_depth": 0},
        {"configured_depth": True},
    ],
)
def test_collection_config_rejects_malformed_numeric_values(tmp_path, overrides):
    with pytest.raises(NativeMTPCollectorError):
        make_config(tmp_path, **overrides)


# --- collect_native_mtp: honest unsupported path ------------------------


def test_collect_native_mtp_produces_explicit_unsupported_record(tmp_path):
    config = make_config(tmp_path)
    result = collect_native_mtp(config)

    assert result.capability.supported is False
    assert result.record.outcome.success is False
    assert result.record.error.category == "NativeMTPUnsupported"
    assert result.record.speculative.enabled is False
    assert result.record.speculative.method is None
    assert result.record.model.model_family == "qwen3_next"
    assert result.response_text == ""

    persisted = ExperimentRecord.read_json(tmp_path / "artifacts" / "record.json")
    assert persisted.outcome.success is False
    capability_payload = json.loads(
        (tmp_path / "artifacts" / "capability_report.json").read_text()
    )
    assert capability_payload["supported"] is False
    assert capability_payload["model_family"] == "qwen3_next"


def test_collect_native_mtp_never_writes_a_response_file_when_unsupported(tmp_path):
    config = make_config(tmp_path)
    collect_native_mtp(config)
    assert not (tmp_path / "artifacts" / "response.txt").exists()


def test_collect_native_mtp_rejects_mismatched_checkpoints(tmp_path):
    target_path = tmp_path / "target"
    sidecar_path = tmp_path / "sidecar"
    _write_config(target_path, hidden_size=4096)
    _write_config(sidecar_path, hidden_size=2048)
    config = make_config(
        tmp_path, target_model_path=target_path, mtp_sidecar_path=sidecar_path
    )

    with pytest.raises(NativeMTPCollectorError, match="incompatible"):
        collect_native_mtp(config)


def test_collect_native_mtp_rejects_missing_config_json(tmp_path):
    target_path = tmp_path / "target"
    target_path.mkdir()
    sidecar_path = tmp_path / "sidecar"
    _write_config(sidecar_path)
    config = make_config(
        tmp_path, target_model_path=target_path, mtp_sidecar_path=sidecar_path
    )

    with pytest.raises(NativeMTPCollectorError, match="config.json"):
        collect_native_mtp(config)


def test_unsupported_family_uses_experimental_reason_for_mlx_vlm_family(tmp_path):
    target_path = tmp_path / "target"
    sidecar_path = tmp_path / "sidecar"
    _write_config(target_path, model_type="qwen4_exp")
    _write_config(sidecar_path, model_type="qwen4_exp_mtp")
    config = make_config(
        tmp_path, target_model_path=target_path, mtp_sidecar_path=sidecar_path
    )

    result = collect_native_mtp(config)
    assert "experimental" in result.record.error.message


# --- collect_native_mtp: the capable extension point (fake runtime only) --


@dataclass
class FakeNativeMTPResponse:
    text: str
    generation_tokens: int
    finish_reason: str | None = None
    accepted_block_tokens: int | None = None
    proposed_block_tokens: int | None = None


class FakeTokenizer:
    pass


class FakeNativeMTPRuntime:
    mlx_version = "0.32.0"
    mlx_lm_version = "99.0.0"

    def __init__(self):
        self.responses = [
            FakeNativeMTPResponse(
                "hello",
                1,
                accepted_block_tokens=1,
                proposed_block_tokens=2,
            ),
            FakeNativeMTPResponse(
                " world",
                2,
                accepted_block_tokens=1,
                proposed_block_tokens=2,
            ),
        ]

    def load_target(self, path):
        return object(), FakeTokenizer()

    def load_sidecar(self, path, target_model):
        return object()

    def encode(self, tokenizer, prompt):
        return [1, 2, 3]

    def seed(self, seed):
        pass

    def synchronize(self):
        pass

    def reset_peak_memory(self):
        pass

    def memory_snapshot(self):
        return None

    def accelerator_name(self):
        return "Apple M5 Pro"

    def generate_with_native_mtp(
        self,
        target_model,
        sidecar,
        tokenizer,
        prompt_tokens,
        *,
        max_tokens,
        configured_depth,
    ):
        yield from self.responses


def test_capable_path_labels_native_mtp_and_records_only_exposed_fields(
    tmp_path, monkeypatch
):
    config = make_config(tmp_path)
    monkeypatch.setattr(
        "llmtracefx.optimizer.collectors.native_mtp.detect_native_mtp_capability",
        lambda *a, **k: NativeMTPCapabilityReport(
            schema_version="1",
            model_family="qwen3_next",
            mlx_lm_version="99.0.0",
            mlx_vlm_version=None,
            supported=True,
            reason="test override: assume a hypothetical stable native API",
            checked_signals=(),
        ),
    )

    result = collect_native_mtp(config, runtime=FakeNativeMTPRuntime())

    assert result.record.outcome.success is True
    assert result.record.speculative.enabled is True
    assert result.record.speculative.method == "native-mtp"
    assert result.record.speculative.proposed_tokens == 4
    assert result.record.speculative.accepted_tokens == 2
    assert result.record.speculative.verification_time is None
    assert result.response_text == "hello world"


def test_capable_path_without_runtime_raises_clear_error(tmp_path, monkeypatch):
    config = make_config(tmp_path)
    monkeypatch.setattr(
        "llmtracefx.optimizer.collectors.native_mtp.detect_native_mtp_capability",
        lambda *a, **k: NativeMTPCapabilityReport(
            schema_version="1",
            model_family="qwen3_next",
            mlx_lm_version="99.0.0",
            mlx_vlm_version=None,
            supported=True,
            reason="test override",
            checked_signals=(),
        ),
    )

    with pytest.raises(NativeMTPCollectorError, match="no NativeMTPRuntime adapter"):
        collect_native_mtp(config, runtime=None)
