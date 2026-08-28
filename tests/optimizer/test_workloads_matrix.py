"""Tests for deterministic workload matrix generation."""

from __future__ import annotations

import json

from llmtracefx.optimizer.workloads.matrix import (
    DECODE_MODE_AUTOREGRESSIVE,
    DECODE_MODE_NATIVE_MTP,
    generate_matrix,
    write_matrix,
)
from llmtracefx.optimizer.workloads.schema import ContextTier


def test_generate_matrix_is_fully_deterministic(tmp_path):
    first = generate_matrix(
        model_id="Qwen/Qwen3.8-27B",
        model_family="qwen3_next",
        output_dir=str(tmp_path / "out"),
    )
    second = generate_matrix(
        model_id="Qwen/Qwen3.8-27B",
        model_family="qwen3_next",
        output_dir=str(tmp_path / "out"),
    )
    assert first == second


def test_generate_matrix_never_touches_the_filesystem(tmp_path):
    output_dir = tmp_path / "not-created-yet"
    generate_matrix(
        model_id="Qwen/Qwen3.8-27B",
        model_family="qwen3_next",
        output_dir=str(output_dir),
    )
    assert not output_dir.exists()


def test_generate_matrix_includes_ar_and_mtp_rows_per_workload_per_tier():
    manifest = generate_matrix(
        model_id="m",
        model_family="qwen3_next",
        output_dir="/tmp/unused",
        mtp_depths=(2, 4),
    )
    from llmtracefx.optimizer.workloads.catalog import WORKLOADS

    expected_count = len(WORKLOADS) * len(tuple(ContextTier)) * (1 + 2)
    assert len(manifest.entries) == expected_count

    ar_entries = [
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    ]
    mtp_entries = [
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_NATIVE_MTP
    ]
    assert len(ar_entries) == len(WORKLOADS) * len(tuple(ContextTier))
    assert len(mtp_entries) == len(WORKLOADS) * len(tuple(ContextTier)) * 2


def test_generate_matrix_marks_unsupported_mtp_family_as_not_runnable():
    manifest = generate_matrix(
        model_id="m", model_family="qwen3_next", output_dir="/tmp/unused"
    )
    ar_entries = [
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    ]
    mtp_entries = [
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_NATIVE_MTP
    ]
    assert all(entry.runnable for entry in ar_entries)
    assert all(not entry.runnable for entry in mtp_entries)
    assert all(entry.unsupported_reason for entry in mtp_entries)


def test_generate_matrix_uses_placeholder_paths_when_omitted():
    manifest = generate_matrix(
        model_id="m", model_family="qwen3_next", output_dir="/tmp/unused"
    )
    ar_entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    assert "<TARGET_MODEL_PATH>" in ar_entry.command_argv


def test_generate_matrix_substitutes_real_paths_when_given():
    manifest = generate_matrix(
        model_id="m",
        model_family="qwen3_next",
        output_dir="/tmp/unused",
        target_model_path="/models/target",
        mtp_sidecar_path="/models/sidecar",
    )
    ar_entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )
    assert "/models/target" in ar_entry.command_argv
    mtp_entry = next(
        e for e in manifest.entries if e.decode_mode == DECODE_MODE_NATIVE_MTP
    )
    assert "/models/target" in mtp_entry.command_argv
    assert "/models/sidecar" in mtp_entry.command_argv


def test_write_matrix_writes_manifest_prompts_and_configs(tmp_path):
    output_dir = tmp_path / "matrix"
    manifest = generate_matrix(
        model_id="Qwen/Qwen3.8-27B",
        model_family="qwen3_next",
        output_dir=str(output_dir),
        mtp_depths=(2,),
    )
    write_matrix(manifest)

    assert (output_dir / "manifest.json").exists()
    manifest_payload = json.loads((output_dir / "manifest.json").read_text())
    assert len(manifest_payload["entries"]) == len(manifest.entries)

    for entry in manifest.entries:
        config_path = output_dir / "configs" / f"{entry.run_id}.json"
        assert config_path.exists()
        config_payload = json.loads(config_path.read_text())
        assert config_payload["command"] == list(entry.command_argv)
        assert config_payload["run_id"] == entry.run_id

    # Prompts are deduplicated per (workload_id, tier), not per decode mode.
    prompt_files = list((output_dir / "prompts").iterdir())
    from llmtracefx.optimizer.workloads.catalog import WORKLOADS

    assert len(prompt_files) == len(WORKLOADS) * len(tuple(ContextTier))


def test_write_matrix_prompt_file_matches_recorded_hash(tmp_path):
    import hashlib

    output_dir = tmp_path / "matrix"
    manifest = generate_matrix(
        model_id="m", model_family="qwen3_next", output_dir=str(output_dir)
    )
    write_matrix(manifest)

    entry = manifest.entries[0]
    expected_name = f"{entry.workload_id}-{entry.context_tier}.txt"
    matching = output_dir / "prompts" / expected_name
    digest = hashlib.sha256(matching.read_text().encode("utf-8")).hexdigest()
    assert entry.prompt.prompt_hash == f"sha256:{digest}"


def test_generate_matrix_respects_context_tier_subset():
    manifest = generate_matrix(
        model_id="m",
        model_family="qwen3_next",
        output_dir="/tmp/unused",
        context_tiers=(ContextTier.TIER_2K,),
    )
    assert all(entry.context_tier == "2k" for entry in manifest.entries)
