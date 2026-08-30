"""Staging and server manifests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from _fakes import OTHER_REVISION, VALID_REVISION

from llmtracefx.deploy.errors import DeploymentPlanError
from llmtracefx.deploy.manifest import (
    MANIFEST_SCHEMA_VERSION,
    OBSERVED,
    ServerManifest,
    StagedFile,
    WeightStagingManifest,
    WeightVerification,
    present_env_var_names,
    verify_staged_weights,
)


def staging(**overrides: object) -> WeightStagingManifest:
    kwargs: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "completed_at": "2026-08-30T00:00:00+00:00",
        "repo_id": "zai-org/GLM-5.3-Flash",
        "revision": VALID_REVISION,
        "mount_path": "/weights",
        "files": (
            StagedFile(path="config.json", size_bytes=4096),
            StagedFile(
                path="model-00001-of-00002.safetensors",
                size_bytes=8 * 1024**3,
                sha256="b" * 64,
            ),
        ),
    }
    kwargs.update(overrides)
    return WeightStagingManifest(**kwargs)  # type: ignore[arg-type]


def test_staging_manifest_totals_sizes_and_counts_published_hashes() -> None:
    payload = staging().to_dict()
    assert payload["file_count"] == 2
    assert payload["total_bytes"] == 4096 + 8 * 1024**3
    assert payload["hashed_file_count"] == 1
    assert payload["total_gib"] == pytest.approx(8.0, abs=0.01)


def test_a_file_without_a_published_hash_records_none_rather_than_a_guess() -> None:
    entry = staging().files[0]
    assert entry.sha256 is None


def test_staging_manifest_round_trips() -> None:
    original = staging()
    restored = WeightStagingManifest.from_dict(json.loads(original.to_json()))
    assert restored.revision == original.revision
    assert restored.total_bytes == original.total_bytes
    assert restored.files[1].sha256 == "b" * 64


def test_a_manifest_for_another_revision_is_not_a_partial_match() -> None:
    manifest = staging()
    assert manifest.matches(repo_id="zai-org/GLM-5.3-Flash", revision=VALID_REVISION)
    assert not manifest.matches(
        repo_id="zai-org/GLM-5.3-Flash", revision=OTHER_REVISION
    )
    assert not manifest.matches(
        repo_id="zai-org/GLM-5.3-Flash-BF16", revision=VALID_REVISION
    )


def test_malformed_manifests_are_refused() -> None:
    with pytest.raises(DeploymentPlanError, match="malformed"):
        WeightStagingManifest.from_dict({"revision": VALID_REVISION})


@pytest.mark.parametrize("bad", [-1, True, "4096"])
def test_file_sizes_must_be_non_negative_integers(bad: object) -> None:
    with pytest.raises(DeploymentPlanError, match="size_bytes"):
        StagedFile(path="x", size_bytes=bad)  # type: ignore[arg-type]


def server(**overrides: object) -> ServerManifest:
    kwargs: dict[str, object] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "collected_at": "2026-08-30T00:00:00+00:00",
        "app_name": "llmtracefx-glm53flash",
        "gpu_type": "H200",
        "gpu_count": 4,
        "framework": "sglang",
        "framework_version": "0.5.6",
        "image_reference": "lmsysorg/sglang:v0.5.6@sha256:" + "a" * 64,
        "image_digest": "sha256:" + "a" * 64,
        "model_repo_id": "zai-org/GLM-5.3-Flash",
        "model_revision": VALID_REVISION,
        "quantization": "fp8",
        "quantization_format": "e4m3",
        "activation_scheme": "dynamic",
        "tensor_parallel_size": 4,
        "context_length": 131072,
    }
    kwargs.update(overrides)
    return ServerManifest(**kwargs)  # type: ignore[arg-type]


def test_server_manifest_captures_the_required_environment_facts() -> None:
    payload = server(
        observed_gpus=("NVIDIA H200", "NVIDIA H200"),
        observed_cuda_version="12.9",
        startup_seconds=612.5,
    ).to_dict()
    assert payload["gpu_type"] == "H200"
    assert payload["gpu_count"] == 4
    assert payload["framework_version"] == "0.5.6"
    assert payload["model_revision"] == VALID_REVISION
    assert payload["quantization"] == "fp8"
    assert payload["tensor_parallel_size"] == 4
    assert payload["expert_parallel_size"] is None
    assert payload["context_length"] == 131072
    assert payload["startup_seconds"] == pytest.approx(612.5)
    assert payload["image_digest_pinned"] is True


def test_observed_fields_are_distinguished_from_configured_ones() -> None:
    provenance = server().to_dict()["provenance"]
    assert provenance["gpu_count"] == "configured"
    assert provenance["observed_gpus"] == OBSERVED
    assert provenance["startup_seconds"] == OBSERVED


def test_unobservable_values_stay_none_rather_than_being_invented() -> None:
    payload = server().to_dict()
    assert payload["observed_gpus"] == []
    assert payload["observed_cuda_version"] is None
    assert payload["startup_seconds"] is None


def test_the_server_makes_no_performance_claim_about_itself() -> None:
    claims = server().to_dict()["performance_claims"]
    assert "None." in claims
    assert "collector" in claims


def test_environment_lookup_returns_names_and_never_values() -> None:
    environ = {"GLM_SELFHOST_API_KEY": "sk-live-secret", "EMPTY": "  "}
    assert present_env_var_names(environ, ["GLM_SELFHOST_API_KEY"]) == (
        "GLM_SELFHOST_API_KEY",
    )
    assert present_env_var_names(environ, ["EMPTY"]) == ()
    assert present_env_var_names(environ, ["ABSENT"]) == ()


def test_no_credential_value_can_reach_a_server_manifest() -> None:
    manifest = server(credential_env_var_names_present=("GLM_SELFHOST_API_KEY",))
    document = manifest.to_json()
    assert "GLM_SELFHOST_API_KEY" in document
    assert "sk-" not in document


def _write_tree(root: Path, files: dict[str, bytes]) -> None:
    for name, payload in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)


def _manifest_for(
    root: Path, files: dict[str, bytes], *, hashed: set[str]
) -> WeightStagingManifest:
    return WeightStagingManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        completed_at="2026-08-30T00:00:00+00:00",
        repo_id="zai-org/GLM-5.3-Flash",
        revision=VALID_REVISION,
        mount_path=str(root),
        files=tuple(
            StagedFile(
                path=name,
                size_bytes=len(payload),
                sha256=(
                    hashlib.sha256(payload).hexdigest() if name in hashed else None
                ),
            )
            for name, payload in files.items()
        ),
    )


def _now() -> str:
    return "2026-08-30T00:00:00+00:00"


def test_a_complete_tree_verifies(tmp_path: Path) -> None:
    files = {"config.json": b"{}", "shard-0.safetensors": b"weights"}
    _write_tree(tmp_path, files)
    manifest = _manifest_for(tmp_path, files, hashed={"shard-0.safetensors"})
    result = verify_staged_weights(manifest, tmp_path, now=_now)
    assert result.ok is True
    assert result.files_checked == 2
    assert result.hashes_checked == 1
    assert result.hashes_available == 1
    assert result.covers(repo_id="zai-org/GLM-5.3-Flash", revision=VALID_REVISION)


def test_a_truncated_shard_is_caught_by_size_alone(tmp_path: Path) -> None:
    """The common failure: an interrupted download leaves a short file.

    A manifest naming the right revision does not catch this, which is
    why the serving container is not allowed to rely on the manifest.
    """
    files = {"shard-0.safetensors": b"weights"}
    manifest = _manifest_for(tmp_path, files, hashed=set())
    _write_tree(tmp_path, {"shard-0.safetensors": b"wei"})
    result = verify_staged_weights(manifest, tmp_path, now=_now, check_hashes=False)
    assert result.ok is False
    assert result.issues[0].problem == "size_mismatch"
    assert result.issues[0].expected == "7"
    assert result.issues[0].observed == "3"
    assert not result.covers(repo_id="zai-org/GLM-5.3-Flash", revision=VALID_REVISION)


def test_a_missing_file_is_caught(tmp_path: Path) -> None:
    files = {"present.json": b"{}", "absent.safetensors": b"gone"}
    manifest = _manifest_for(tmp_path, files, hashed=set())
    _write_tree(tmp_path, {"present.json": b"{}"})
    result = verify_staged_weights(manifest, tmp_path, now=_now)
    assert [issue.problem for issue in result.issues] == ["missing"]


def test_corruption_that_preserves_size_needs_the_hash(tmp_path: Path) -> None:
    """Same length, different bytes. Only the published hash sees it."""
    files = {"shard-0.safetensors": b"weights"}
    manifest = _manifest_for(tmp_path, files, hashed={"shard-0.safetensors"})
    _write_tree(tmp_path, {"shard-0.safetensors": b"corrupt"})

    without = verify_staged_weights(manifest, tmp_path, now=_now, check_hashes=False)
    assert without.ok is True

    with_hashes = verify_staged_weights(manifest, tmp_path, now=_now)
    assert with_hashes.ok is False
    assert with_hashes.issues[0].problem == "hash_mismatch"
    # The digests are not repeated into the issue; they are long and the
    # remedy is the same either way.
    assert with_hashes.issues[0].observed is None


def test_a_verification_round_trips(tmp_path: Path) -> None:
    files = {"config.json": b"{}"}
    _write_tree(tmp_path, files)
    manifest = _manifest_for(tmp_path, files, hashed=set())
    original = verify_staged_weights(manifest, tmp_path, now=_now)
    restored = WeightVerification.from_dict(json.loads(original.to_json()))
    assert restored.ok is True
    assert restored.revision == VALID_REVISION


def test_a_malformed_verification_is_refused() -> None:
    with pytest.raises(DeploymentPlanError, match="malformed"):
        WeightVerification.from_dict({"revision": VALID_REVISION})
