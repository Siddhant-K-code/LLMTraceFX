"""Offline tests for the Qwen3-8B self-conversion runner.

No test in this module downloads model weights or executes a real
model conversion: every filesystem/subprocess boundary is monkeypatched
so the exact conversion-parameter argv, subprocess isolation, timeout
escalation, and output-inventory logic can be verified deterministically
and offline.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from llmtracefx.optimizer.lab.core import HostSnapshot
from llmtracefx.optimizer.lab.qwen3_8b import conversion as conv
from llmtracefx.optimizer.lab.qwen3_8b.conversion_manifest import (
    ConversionManifest,
    ConversionManifestError,
)

CONVERSION_MANIFEST_PATH = Path(
    "llmtracefx/optimizer/lab/qwen3_8b/data/qwen3-8b-conversion-manifest-v1.json"
)


def _manifest() -> ConversionManifest:
    return ConversionManifest.read_json(CONVERSION_MANIFEST_PATH)


def _snapshot(**overrides) -> HostSnapshot:
    manifest = _manifest()
    values = {
        "collected_at": "2026-09-01T00:00:00.000000Z",
        "os_name": "Darwin",
        "os_release": "25.6.0",
        "architecture": "arm64",
        "python_implementation": "CPython",
        "python_version": "3.13.15",
        "cpu_count": 18,
        "chip": manifest.safety.required_chip,
        "total_memory_bytes": manifest.safety.required_total_memory_bytes,
        "memory_free_percent": 50.0,
        "swap_used_bytes": 0,
        "disk_free_bytes": 500 * 1024**3,
        "package_versions": {"mlx-lm": manifest.converter.version},
    }
    values.update(overrides)
    return HostSnapshot(**values)


def _populate_sparse_source(manifest: ConversionManifest, source_path: Path) -> None:
    """Create exact-size (sparse) files for every pinned source file.

    Never writes real content; the real bytes are irrelevant here
    because tests monkeypatch ``conv._sha256_file`` to return the
    pinned digest for anything under ``source_path``.
    """
    for pin in manifest.source.files:
        path = source_path / pin.path
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as handle:
            if pin.size_bytes:
                handle.truncate(pin.size_bytes)


def _patch_source_hash(
    monkeypatch, manifest: ConversionManifest, source_path: Path
) -> None:
    real_sha256_file = conv._sha256_file

    def fake(path: Path) -> str:
        path = Path(path)
        try:
            relative = path.relative_to(source_path)
        except ValueError:
            return real_sha256_file(path)
        for pin in manifest.source.files:
            if pin.path == str(relative):
                return pin.sha256
        raise AssertionError(f"unexpected source path in test fixture: {path}")

    monkeypatch.setattr(conv, "_sha256_file", fake)


def _forbid_network_download(monkeypatch) -> None:
    """Any test that reaches this guard while expecting an offline path
    has a bug in its fixture, not a legitimate reason to download."""
    import huggingface_hub

    def fail(*args, **kwargs):
        raise AssertionError(
            "snapshot_download must never be called in an offline test"
        )

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)


def test_packaged_conversion_manifest_pins_exact_provenance() -> None:
    manifest = _manifest()
    assert manifest.source.official_id == "Qwen/Qwen3-8B"
    assert manifest.source.official_revision == (
        "b968826d9c46dd6066d109eabc6255188de91218"
    )
    assert manifest.source.license == "Apache-2.0"
    assert manifest.source.expected_source_bytes == 16397461266
    assert manifest.source.fully_pinned is True
    assert sum(pin.size_bytes for pin in manifest.source.files) == (
        manifest.source.expected_source_bytes
    )
    assert manifest.converter.package == "mlx-lm"
    assert manifest.converter.version == "0.31.3"
    assert manifest.converter.git_revision == "ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd"
    assert manifest.parameters.quantize is True
    assert manifest.parameters.q_group_size == 64
    assert manifest.parameters.q_bits == 4
    assert manifest.parameters.dequantize is False
    assert manifest.parameters.upload_repo is None
    assert manifest.parameters.trust_remote_code is False


def test_conversion_manifest_rejects_dequantize() -> None:
    payload = json.loads(CONVERSION_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["parameters"]["dequantize"] = True
    with pytest.raises(ConversionManifestError, match="dequantize"):
        ConversionManifest.from_dict(payload)


def test_conversion_manifest_rejects_upload_repo() -> None:
    payload = json.loads(CONVERSION_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["parameters"]["upload_repo"] = "someone/somewhere"
    with pytest.raises(ConversionManifestError, match="upload_repo"):
        ConversionManifest.from_dict(payload)


def test_conversion_manifest_rejects_unpinned_official_revision_format() -> None:
    payload = json.loads(CONVERSION_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["source"]["official_revision"] = "main"
    with pytest.raises(ConversionManifestError, match="git revision"):
        ConversionManifest.from_dict(payload)


def test_conversion_manifest_rejects_source_file_without_sha256() -> None:
    payload = json.loads(CONVERSION_MANIFEST_PATH.read_text(encoding="utf-8"))
    payload["source"]["files"][0]["sha256"] = None
    with pytest.raises(ConversionManifestError, match="sha256"):
        ConversionManifest.from_dict(payload)


def test_conversion_manifest_hash_binds_every_source_file_digest() -> None:
    manifest = _manifest()
    first = manifest.source.files[0]
    changed = replace(
        manifest,
        source=replace(
            manifest.source,
            files=(replace(first, sha256="f" * 64), *manifest.source.files[1:]),
        ),
    )
    assert conv.conversion_manifest_hash(changed) != conv.conversion_manifest_hash(
        manifest
    )


def test_argv_is_the_exact_deterministic_mlx_lm_convert_invocation() -> None:
    manifest = _manifest()
    argv = manifest.parameters.argv(hf_path="/src", mlx_path="/out")
    assert argv == (
        "convert",
        "--hf-path",
        "/src",
        "--mlx-path",
        "/out",
        "--quantize",
        "--q-group-size",
        "64",
        "--q-bits",
        "4",
        "--q-mode",
        "affine",
    )


def test_source_verification_rejects_unpinned_root_entry(tmp_path, monkeypatch) -> None:
    manifest = _manifest()
    source_path = tmp_path / "source"
    _populate_sparse_source(manifest, source_path)
    _patch_source_hash(monkeypatch, manifest, source_path)
    (source_path / "stale-extra-file.bin").write_bytes(b"x")
    with pytest.raises(conv.ConversionError, match="unpinned source-root entry"):
        conv.verify_source(manifest, source_path)


def test_source_verification_rejects_symlinked_file(tmp_path, monkeypatch) -> None:
    manifest = _manifest()
    source_path = tmp_path / "source"
    _populate_sparse_source(manifest, source_path)
    _patch_source_hash(monkeypatch, manifest, source_path)
    real_file = source_path / manifest.source.files[0].path
    real_file.unlink()
    (source_path / "elsewhere.bin").write_bytes(b"x")
    real_file.symlink_to(source_path / "elsewhere.bin")
    with pytest.raises(conv.ConversionError, match="missing regular source file"):
        conv.verify_source(manifest, source_path)


def test_acquire_source_reuses_verified_cache_without_any_network_call(
    tmp_path, monkeypatch
) -> None:
    manifest = _manifest()
    source_path = tmp_path / "source"
    workspace = tmp_path / "workspace"
    _populate_sparse_source(manifest, source_path)
    _patch_source_hash(monkeypatch, manifest, source_path)
    _forbid_network_download(monkeypatch)

    result = conv.acquire_source(manifest, source_path=source_path, workspace=workspace)
    assert result["verified"] is True
    assert result["fully_pinned"] is True


def test_acquire_source_refuses_existing_invalid_cache_without_downloading(
    tmp_path, monkeypatch
) -> None:
    """Per item 4: if source_path already has content that fails
    verification, this refuses rather than falling through to a network
    write into a possibly corrupt or stale cache."""
    manifest = _manifest()
    source_path = tmp_path / "source"
    workspace = tmp_path / "workspace"
    _populate_sparse_source(manifest, source_path)
    # Corrupt exactly one file's size so verification fails.
    corrupted = source_path / manifest.source.files[0].path
    corrupted.write_bytes(b"short")
    _patch_source_hash(monkeypatch, manifest, source_path)
    _forbid_network_download(monkeypatch)

    with pytest.raises(conv.ConversionError, match="size"):
        conv.acquire_source(manifest, source_path=source_path, workspace=workspace)
    # The corrupt file must remain exactly as it was; never overwritten.
    assert corrupted.read_bytes() == b"short"


def test_acquire_source_downloads_only_when_cache_is_absent(
    tmp_path, monkeypatch
) -> None:
    manifest = _manifest()
    source_path = tmp_path / "source"
    workspace = tmp_path / "workspace"
    assert not source_path.exists()

    calls = []

    def fake_download(*, repo_id, revision, local_dir, allow_patterns):
        calls.append((repo_id, revision))
        _populate_sparse_source(manifest, Path(local_dir))
        return str(local_dir)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_download)
    _patch_source_hash(monkeypatch, manifest, source_path)

    result = conv.acquire_source(manifest, source_path=source_path, workspace=workspace)
    assert calls == [(manifest.source.repository_id, manifest.source.official_revision)]
    assert result["verified"] is True


def test_conversion_safety_blocks_on_mismatched_installed_converter_version(
    tmp_path, monkeypatch
) -> None:
    manifest = _manifest()
    monkeypatch.setattr(
        conv,
        "collect_host_snapshot",
        lambda path: _snapshot(package_versions={"mlx-lm": "0.0.0"}),
    )
    decision = conv.assess_conversion_safety(
        manifest, tmp_path, include_source_download=False
    )
    assert decision.safe is False
    assert any("installed mlx-lm is 0.0.0" in blocker for blocker in decision.blockers)


@pytest.mark.parametrize(
    ("overrides", "expected_substring"),
    [
        ({"memory_free_percent": 39.0}, "memory free 39% is below 40%"),
        ({"swap_used_bytes": 12884901889}, "swap used 12884901889 bytes exceeds"),
        ({"disk_free_bytes": 111997967370}, "disk free 111997967370 bytes is below"),
    ],
)
def test_conversion_safety_blocks_on_each_conservative_threshold(
    overrides, expected_substring
) -> None:
    manifest = _manifest()
    decision = conv.ConversionSafetyDecision(
        safe=True,
        blockers=(),
        snapshot=_snapshot(**overrides),
        required_free_disk_bytes=0,
    )
    # Re-run the real assessment against the overridden snapshot so this
    # exercises the exact production code path, not a hand-built decision.
    import llmtracefx.optimizer.lab.qwen3_8b.conversion as conv_module

    original = conv_module.collect_host_snapshot
    try:
        conv_module.collect_host_snapshot = lambda path: decision.snapshot
        result = conv.assess_conversion_safety(
            manifest, Path("."), include_source_download=True
        )
    finally:
        conv_module.collect_host_snapshot = original
    assert result.safe is False
    assert any(expected_substring in blocker for blocker in result.blockers)


def test_required_free_disk_bytes_matches_exact_conservative_policy() -> None:
    manifest = _manifest()
    assert manifest.source.expected_source_bytes == 16397461266
    assert manifest.expected_output_bytes == 4623784971
    assert manifest.safety.minimum_residual_free_disk_bytes == 107374182400
    assert manifest.safety.minimum_memory_free_percent == 40.0
    assert manifest.safety.maximum_swap_used_bytes == 12 * 1024**3

    decision_with_download = conv.assess_conversion_safety(
        manifest, Path("."), include_source_download=True
    )
    assert decision_with_download.required_free_disk_bytes == 128395428637

    decision_without_download = conv.assess_conversion_safety(
        manifest, Path("."), include_source_download=False
    )
    assert decision_without_download.required_free_disk_bytes == 111997967371


def test_run_conversion_refuses_before_any_download_or_subprocess_on_low_memory(
    tmp_path, monkeypatch
) -> None:
    """The exact scenario this hardening pass fixes: a live safety-gate
    refusal must precede any network/download/subprocess activity, write
    a bounded pre_conversion_safety failure record, and never touch the
    source/output paths."""
    manifest = _manifest()
    source_path = tmp_path / "source"
    output_path = tmp_path / "output"
    workspace = tmp_path / "workspace"

    def refused_snapshot(path):
        return _snapshot(
            memory_free_percent=39.0,
            swap_used_bytes=10148705730,
            disk_free_bytes=721039908864,
        )

    monkeypatch.setattr(conv, "collect_host_snapshot", refused_snapshot)
    _forbid_network_download(monkeypatch)

    def fail_popen(*args, **kwargs):
        raise AssertionError("must never launch a subprocess after a safety refusal")

    monkeypatch.setattr(conv.subprocess, "Popen", fail_popen)

    journal = conv.run_conversion(
        manifest, source_path=source_path, output_path=output_path, workspace=workspace
    )

    assert journal["status"] == "safety_blocked"
    assert journal["phase"] == "pre_conversion_safety"
    assert journal["downloads_performed"] is False
    assert journal["conversion_process_started"] is False
    assert journal["retried"] is False
    assert any("memory free 39%" in blocker for blocker in journal["blockers"])
    assert journal["required_free_disk_bytes"] == 128395428637
    assert not source_path.exists()
    assert not output_path.exists()
    failure_path = workspace / "conversion-failure.json"
    assert failure_path.is_file()
    assert json.loads(failure_path.read_text(encoding="utf-8")) == journal
    # Bounded, privacy-safe: no repository-relative or user paths leaked.
    assert str(tmp_path) not in json.dumps(journal)


def test_cli_convert_surfaces_status_2_not_1_for_safety_refusal(
    tmp_path, monkeypatch
) -> None:
    from llmtracefx.optimizer.lab.qwen3_8b import cli as qcli

    def refused_snapshot(path):
        return _snapshot(memory_free_percent=39.0)

    monkeypatch.setattr(conv, "collect_host_snapshot", refused_snapshot)
    exit_code = qcli.main(
        [
            "convert",
            "--source-path",
            str(tmp_path / "source"),
            "--output-path",
            str(tmp_path / "output"),
            "--conversion-workspace",
            str(tmp_path / "ws"),
        ]
    )
    assert exit_code == 2


def _prepare_offline_conversion(tmp_path, monkeypatch):
    manifest = _manifest()
    source_path = tmp_path / "source"
    workspace = tmp_path / "workspace"
    _populate_sparse_source(manifest, source_path)
    _patch_source_hash(monkeypatch, manifest, source_path)
    _forbid_network_download(monkeypatch)
    monkeypatch.setattr(conv, "collect_host_snapshot", lambda path: _snapshot())
    return manifest, source_path, workspace


def test_run_conversion_launches_fresh_process_group_no_shell(
    tmp_path, monkeypatch
) -> None:
    manifest, source_path, workspace = _prepare_offline_conversion(
        tmp_path, monkeypatch
    )
    output_path = tmp_path / "output"
    captured = {}

    class FakeProcess:
        pid = 4242
        returncode = 0

        def __init__(self, argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs

        def wait(self, timeout=None):
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "model.safetensors").write_bytes(b"abc")
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(conv.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(conv, "_clean_process_group", lambda *args: True)

    journal = conv.run_conversion(
        manifest, source_path=source_path, output_path=output_path, workspace=workspace
    )

    assert journal["status"] == "completed"
    assert captured["kwargs"]["start_new_session"] is True
    assert captured["kwargs"]["shell"] is False
    assert captured["argv"][0] == sys.executable
    assert captured["argv"][1:4] == ["-m", "mlx_lm", "convert"]
    assert "--quantize" in captured["argv"]
    assert (workspace / "conversion-receipt.json").is_file()
    receipt = conv.ConversionReceipt.read_json(workspace / "conversion-receipt.json")
    assert receipt.output_total_bytes == 3
    assert receipt.converter["git_revision"] == manifest.converter.git_revision


def test_run_conversion_timeout_escalates_to_kill_and_writes_failure(
    tmp_path, monkeypatch
) -> None:
    manifest, source_path, workspace = _prepare_offline_conversion(
        tmp_path, monkeypatch
    )
    output_path = tmp_path / "output"

    class FakeProcess:
        pid = 4242
        returncode = -9

        def __init__(self, argv, **kwargs):
            self._waits = 0

        def wait(self, timeout=None):
            self._waits += 1
            if self._waits == 1:
                raise conv.subprocess.TimeoutExpired("child", timeout)
            return self.returncode

        def poll(self):
            return self.returncode

    monkeypatch.setattr(conv.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(conv, "_clean_process_group", lambda *args: True)

    journal = conv.run_conversion(
        manifest, source_path=source_path, output_path=output_path, workspace=workspace
    )
    assert journal["status"] == "timeout"
    assert journal["timed_out"] is True
    assert (workspace / "conversion-failure.json").is_file()
    assert not (workspace / "conversion-receipt.json").exists()


def test_run_conversion_cleanup_failure_is_a_distinct_status(
    tmp_path, monkeypatch
) -> None:
    manifest, source_path, workspace = _prepare_offline_conversion(
        tmp_path, monkeypatch
    )
    output_path = tmp_path / "output"

    class FakeProcess:
        pid = 1
        returncode = None

        def __init__(self, argv, **kwargs):
            pass

        def wait(self, timeout=None):
            return None

        def poll(self):
            return None

    monkeypatch.setattr(conv.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(conv, "_clean_process_group", lambda *args: False)

    journal = conv.run_conversion(
        manifest, source_path=source_path, output_path=output_path, workspace=workspace
    )
    assert journal["status"] == "cleanup_failed"
    assert not (workspace / "conversion-receipt.json").exists()


def test_run_conversion_nonzero_exit_writes_failure_not_receipt(
    tmp_path, monkeypatch
) -> None:
    manifest, source_path, workspace = _prepare_offline_conversion(
        tmp_path, monkeypatch
    )
    output_path = tmp_path / "output"

    class FakeProcess:
        pid = 2
        returncode = 1

        def __init__(self, argv, **kwargs):
            pass

        def wait(self, timeout=None):
            return 1

        def poll(self):
            return 1

    monkeypatch.setattr(conv.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(conv, "_clean_process_group", lambda *args: True)

    journal = conv.run_conversion(
        manifest, source_path=source_path, output_path=output_path, workspace=workspace
    )
    assert journal["status"] == "failed"
    assert journal["exit_code"] == 1
    assert (workspace / "conversion-failure.json").is_file()
    assert not (workspace / "conversion-receipt.json").exists()


def test_run_conversion_rejects_symlinked_output_as_failure_record(
    tmp_path, monkeypatch
) -> None:
    manifest, source_path, workspace = _prepare_offline_conversion(
        tmp_path, monkeypatch
    )
    output_path = tmp_path / "output"

    class FakeProcess:
        pid = 3
        returncode = 0

        def __init__(self, argv, **kwargs):
            pass

        def wait(self, timeout=None):
            output_path.mkdir(parents=True, exist_ok=True)
            (output_path / "real.bin").write_bytes(b"data")
            (output_path / "link.bin").symlink_to(output_path / "real.bin")
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(conv.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(conv, "_clean_process_group", lambda *args: True)

    journal = conv.run_conversion(
        manifest, source_path=source_path, output_path=output_path, workspace=workspace
    )
    assert journal["status"] == "unsafe_output"
    assert not (workspace / "conversion-receipt.json").exists()
    assert (workspace / "conversion-failure.json").is_file()


def test_run_conversion_refuses_preexisting_output_without_deleting_it(
    tmp_path, monkeypatch
) -> None:
    manifest, source_path, workspace = _prepare_offline_conversion(
        tmp_path, monkeypatch
    )
    output_path = tmp_path / "output"
    output_path.mkdir()
    (output_path / "keep-me.bin").write_bytes(b"user data")

    def fail_popen(*args, **kwargs):
        raise AssertionError(
            "must never launch a subprocess over an existing output path"
        )

    monkeypatch.setattr(conv.subprocess, "Popen", fail_popen)

    with pytest.raises(conv.ConversionError, match="already exists"):
        conv.run_conversion(
            manifest,
            source_path=source_path,
            output_path=output_path,
            workspace=workspace,
        )
    assert (output_path / "keep-me.bin").read_bytes() == b"user data"


def test_run_conversion_never_retries_automatically(tmp_path, monkeypatch) -> None:
    manifest, source_path, workspace = _prepare_offline_conversion(
        tmp_path, monkeypatch
    )
    output_path = tmp_path / "output"
    popen_calls = []

    class FakeProcess:
        pid = 5
        returncode = 1

        def __init__(self, argv, **kwargs):
            popen_calls.append(argv)

        def wait(self, timeout=None):
            return 1

        def poll(self):
            return 1

    monkeypatch.setattr(conv.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(conv, "_clean_process_group", lambda *args: True)

    conv.run_conversion(
        manifest, source_path=source_path, output_path=output_path, workspace=workspace
    )
    assert len(popen_calls) == 1  # exactly one attempt; no internal retry loop


def test_bounded_journal_tail_truncates_to_max_journal_bytes() -> None:
    text = "x" * 5000
    truncated = conv._bounded_tail(text, 100)
    assert len(truncated.encode("utf-8")) == 100
    assert conv._bounded_tail("short", 100) == "short"


def test_conversion_receipt_rejects_incomplete_status() -> None:
    with pytest.raises(conv.ConversionError, match="completed"):
        conv.ConversionReceipt.from_dict(
            {
                "schema_version": "1",
                "conversion_id": "x",
                "status": "failed",
                "started_at": "a",
                "ended_at": "b",
                "source": {},
                "converter": {},
                "parameters": {},
                "argv": [],
                "output_files": [{"path": "a", "size_bytes": 1, "sha256": "0" * 64}],
                "output_total_bytes": 1,
                "host": {},
            }
        )


def _valid_receipt_payload() -> dict:
    return {
        "schema_version": "1",
        "conversion_id": "qwen3-8b-mlx-q4g64-self-convert-v1",
        "conversion_manifest_hash": "sha256:" + "a" * 64,
        "status": "completed",
        "started_at": "2026-01-01T00:00:00Z",
        "ended_at": "2026-01-01T01:00:00Z",
        "source": {
            "official_id": "Qwen/Qwen3-8B",
            "official_revision": "b968826d9c46dd6066d109eabc6255188de91218",
            "license": "Apache-2.0",
        },
        "converter": {
            "package": "mlx-lm",
            "version": "0.31.3",
            "git_revision": "ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd",
        },
        "parameters": {
            "quantize": True,
            "q_group_size": 64,
            "q_bits": 4,
            "q_mode": "affine",
            "dtype": None,
            "quant_predicate": None,
            "dequantize": False,
            "trust_remote_code": False,
            "upload_repo": None,
        },
        "argv": ["mlx_lm", "convert"],
        "output_files": [
            {"path": "config.json", "size_bytes": 10, "sha256": "b" * 64},
            {"path": "model.safetensors", "size_bytes": 20, "sha256": "c" * 64},
        ],
        "output_total_bytes": 30,
        "host": {"os_name": "Darwin"},
    }


def test_conversion_receipt_parses_a_fully_valid_payload() -> None:
    receipt = conv.ConversionReceipt.from_dict(_valid_receipt_payload())
    assert receipt.output_total_bytes == 30
    assert len(receipt.output_files) == 2
    assert receipt.conversion_manifest_hash == "sha256:" + "a" * 64


def test_conversion_receipt_rejects_duplicate_output_paths() -> None:
    payload = _valid_receipt_payload()
    payload["output_files"] = [
        {"path": "config.json", "size_bytes": 10, "sha256": "b" * 64},
        {"path": "config.json", "size_bytes": 20, "sha256": "c" * 64},
    ]
    with pytest.raises(conv.ConversionError, match="unique"):
        conv.ConversionReceipt.from_dict(payload)


def test_conversion_receipt_rejects_unsafe_output_path() -> None:
    payload = _valid_receipt_payload()
    payload["output_files"][0]["path"] = "../escape.bin"
    with pytest.raises(conv.ConversionError, match="safe relative path"):
        conv.ConversionReceipt.from_dict(payload)


def test_conversion_receipt_rejects_absolute_output_path() -> None:
    payload = _valid_receipt_payload()
    payload["output_files"][0]["path"] = "/etc/passwd"
    with pytest.raises(conv.ConversionError, match="safe relative path"):
        conv.ConversionReceipt.from_dict(payload)


def test_conversion_receipt_rejects_non_positive_size() -> None:
    payload = _valid_receipt_payload()
    payload["output_files"][0]["size_bytes"] = 0
    with pytest.raises(conv.ConversionError, match="positive integer"):
        conv.ConversionReceipt.from_dict(payload)


def test_conversion_receipt_rejects_uppercase_sha256() -> None:
    payload = _valid_receipt_payload()
    payload["output_files"][0]["sha256"] = "B" * 64
    with pytest.raises(conv.ConversionError, match="lowercase SHA-256"):
        conv.ConversionReceipt.from_dict(payload)


def test_conversion_receipt_rejects_total_bytes_mismatch() -> None:
    payload = _valid_receipt_payload()
    payload["output_total_bytes"] = 999
    with pytest.raises(conv.ConversionError, match="does not equal the sum"):
        conv.ConversionReceipt.from_dict(payload)


@pytest.mark.parametrize(
    "value",
    [None, "", "a" * 64, "sha256:" + "A" * 64, "sha256:short"],
)
def test_conversion_receipt_requires_manifest_hash(value) -> None:
    payload = _valid_receipt_payload()
    payload["conversion_manifest_hash"] = value
    with pytest.raises(conv.ConversionError, match="conversion_manifest_hash"):
        conv.ConversionReceipt.from_dict(payload)


@pytest.mark.parametrize(
    ("section", "key"),
    [
        ("source", "official_id"),
        ("source", "official_revision"),
        ("source", "license"),
        ("converter", "package"),
        ("converter", "version"),
        ("converter", "git_revision"),
    ],
)
def test_conversion_receipt_rejects_missing_identity_key(section, key) -> None:
    payload = _valid_receipt_payload()
    del payload[section][key]
    with pytest.raises(conv.ConversionError):
        conv.ConversionReceipt.from_dict(payload)


def test_conversion_receipt_rejects_missing_parameter_key() -> None:
    payload = _valid_receipt_payload()
    del payload["parameters"]["q_group_size"]
    with pytest.raises(conv.ConversionError, match="missing"):
        conv.ConversionReceipt.from_dict(payload)


EXAMPLE_REFUSAL_PATH = Path(
    "examples/optimizer/qwen3-8b-m5-control/conversion-preflight-refusal-example.json"
)


def test_committed_refusal_example_is_named_and_shaped_as_a_refusal_not_evidence() -> (
    None
):
    example = json.loads(EXAMPLE_REFUSAL_PATH.read_text(encoding="utf-8"))
    assert example["artifact_type"] == "conversion_preflight_refusal"
    assert example["phase"] == "pre_conversion_safety"
    assert example["status"] == "safety_blocked"
    assert example["downloads_performed"] is False
    assert example["conversion_process_started"] is False
    assert example["retried"] is False
    assert example["cache_modified"] is False
    assert any("not benchmark evidence" in note for note in example["limitations"])
    assert any("memory_pressure" in note for note in example["limitations"])


def test_committed_refusal_example_contains_no_private_data() -> None:
    from llmtracefx.optimizer.lab.core import assert_shareable

    example = json.loads(EXAMPLE_REFUSAL_PATH.read_text(encoding="utf-8"))
    assert_shareable(example)  # raises LabError if any private pattern is found
    serialized = json.dumps(example)
    for forbidden in ("/Users/", "/home/", "generated_at", "collected_at"):
        assert forbidden not in serialized


def test_committed_refusal_example_matches_the_real_safety_gate_output(
    monkeypatch,
) -> None:
    """Hash/integrity check: replaying the exact observed host state
    recorded in the committed example through the real
    ``assess_conversion_safety`` code path must reproduce the same single
    blocker the example claims -- no drift between the committed example
    and the code that actually produces refusals."""
    example = json.loads(EXAMPLE_REFUSAL_PATH.read_text(encoding="utf-8"))
    manifest = _manifest()
    assert example["safety_thresholds"]["required_free_disk_bytes"] == (
        manifest.source.expected_source_bytes
        + manifest.expected_output_bytes
        + manifest.safety.minimum_residual_free_disk_bytes
    )
    assert example["safety_thresholds"]["minimum_memory_free_percent"] == (
        manifest.safety.minimum_memory_free_percent
    )
    assert example["safety_thresholds"]["maximum_swap_used_bytes"] == (
        manifest.safety.maximum_swap_used_bytes
    )
    assert example["safety_thresholds"]["required_total_memory_bytes"] == (
        manifest.safety.required_total_memory_bytes
    )

    observed = example["observed"]
    monkeypatch.setattr(
        conv,
        "collect_host_snapshot",
        lambda path: _snapshot(
            chip=observed["chip"],
            os_name=observed["os_name"],
            architecture=observed["architecture"],
            total_memory_bytes=observed["total_memory_bytes"],
            memory_free_percent=observed["memory_free_percent"],
            swap_used_bytes=observed["swap_used_bytes"],
            disk_free_bytes=observed["disk_free_bytes"],
            package_versions={"mlx-lm": manifest.converter.version},
        ),
    )
    decision = conv.assess_conversion_safety(
        manifest, Path("."), include_source_download=True
    )
    assert decision.safe is False
    assert list(decision.blockers) == example["blockers"]
    assert decision.required_free_disk_bytes == (
        example["safety_thresholds"]["required_free_disk_bytes"]
    )


def test_committed_refusal_example_reproduces_via_run_conversion(
    tmp_path, monkeypatch
) -> None:
    """End-to-end: feeding the example's exact observed host state
    through ``run_conversion`` must refuse before any download or
    subprocess and produce a journal whose blockers match the committed
    example exactly."""
    example = json.loads(EXAMPLE_REFUSAL_PATH.read_text(encoding="utf-8"))
    manifest = _manifest()
    observed = example["observed"]

    monkeypatch.setattr(
        conv,
        "collect_host_snapshot",
        lambda path: _snapshot(
            chip=observed["chip"],
            os_name=observed["os_name"],
            architecture=observed["architecture"],
            total_memory_bytes=observed["total_memory_bytes"],
            memory_free_percent=observed["memory_free_percent"],
            swap_used_bytes=observed["swap_used_bytes"],
            disk_free_bytes=observed["disk_free_bytes"],
            package_versions={"mlx-lm": manifest.converter.version},
        ),
    )
    _forbid_network_download(monkeypatch)
    monkeypatch.setattr(
        conv.subprocess,
        "Popen",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("must never launch a subprocess after a safety refusal")
        ),
    )

    journal = conv.run_conversion(
        manifest,
        source_path=tmp_path / "source",
        output_path=tmp_path / "output",
        workspace=tmp_path / "ws",
    )
    assert journal["status"] == "safety_blocked"
    assert journal["phase"] == "pre_conversion_safety"
    assert journal["blockers"] == example["blockers"]
    assert not (tmp_path / "source").exists()
    assert not (tmp_path / "output").exists()
