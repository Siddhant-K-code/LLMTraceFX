"""Load-bearing tests for the public Metal evidence example."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT / "examples" / "metal_evidence"
SPEC = importlib.util.spec_from_file_location(
    "metal_evidence_demo", EXAMPLE / "evidence_demo.py"
)
assert SPEC is not None and SPEC.loader is not None
DEMO = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DEMO
SPEC.loader.exec_module(DEMO)


def _fixture(name: str) -> str:
    return (ROOT / "tests" / "optimizer" / "fixtures" / "instruments" / name).read_text(
        encoding="utf-8"
    )


def test_summary_is_pid_scoped_and_counts_reference_cells():
    summary = DEMO.summarize_exports(
        _fixture("toc_metal_system_trace.xml"),
        _fixture("table_metal_gpu_intervals.xml"),
        expected_dispatch_count=3,
    )

    assert summary.attributed_interval_count == 3
    assert summary.all_process_interval_count == 5
    assert summary.known_unrelated_interval_count == 2
    assert summary.known_unrelated_interval_share_percent == 40.0
    assert summary.unattributed_interval_count == 0
    assert summary.unattributed_interval_share_percent == 0.0
    assert summary.window_server_interval_count == 2
    assert summary.window_server_interval_share_percent == 40.0
    assert summary.exported_rows == 5
    assert summary.exported_cells == 20
    assert summary.reference_cells == 7
    assert summary.dispatch_count_matches is True


def test_summary_marks_a_dispatch_mismatch_without_relabelling_it():
    summary = DEMO.summarize_exports(
        _fixture("toc_metal_system_trace.xml"),
        _fixture("table_metal_gpu_intervals.xml"),
        expected_dispatch_count=4,
    )
    assert summary.attributed_interval_count == 3
    assert summary.dispatch_count_matches is False


def test_summary_keeps_unattributed_rows_out_of_known_unrelated():
    table = _fixture("table_metal_gpu_intervals.xml").replace(
        '<process id="12" fmt="WindowServer (77)">'
        '<pid id="13" fmt="77">77</pid><device-session ref="6"/></process>',
        '<process id="12" fmt="unattributed"/>',
    )
    summary = DEMO.summarize_exports(
        _fixture("toc_metal_system_trace.xml"),
        table,
        expected_dispatch_count=3,
    )
    assert summary.known_unrelated_interval_count == 0
    assert summary.unattributed_interval_count == 2


def test_output_directory_must_be_empty(tmp_path):
    output = tmp_path / "capture"
    output.mkdir()
    (output / "stale.json").write_text("{}")

    with pytest.raises(ValueError, match="stale artifact mixing"):
        DEMO._prepare_output(output)


def test_private_artifact_cleanup_fails_closed(monkeypatch, tmp_path):
    private = tmp_path / "private"
    private.mkdir()
    (private / "capture.trace").write_text("private")
    monkeypatch.setattr(DEMO.shutil, "rmtree", lambda path: None)

    with pytest.raises(RuntimeError, match="private capture artifacts remain"):
        DEMO._remove_private_artifacts(private)


def test_public_bundle_privacy_scan_rejects_private_data(tmp_path):
    for name in DEMO.PUBLIC_CONTENT_FILES:
        (tmp_path / name).write_text("safe", encoding="utf-8")
    (tmp_path / "capture-summary.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "attributed_interval_count": "measured_native",
                    "unattributed_interval_count": "measured_native",
                    "known_unrelated_interval_count": "derived: subtraction",
                    "known_unrelated_interval_share_percent": "derived: ratio",
                    "unattributed_interval_share_percent": "derived: ratio",
                    "window_server_interval_share_percent": "derived: ratio",
                },
                "captures": [
                    {
                        "dispatch_count_matches": True,
                        "attributed_interval_count": 1,
                        "known_unrelated_interval_count": 0,
                        "unattributed_interval_count": 0,
                        "all_process_interval_count": 1,
                        "reference_cells": 0,
                        "exported_cells": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "experiment-manifest.json").write_text(
        json.dumps(
            {
                "path": "/Users/private/example",
                "workload": {
                    "source": "metal_workload.swift",
                    "source_sha256": DEMO._sha256(EXAMPLE / "metal_workload.swift"),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="absolute macOS home path"):
        DEMO.verify_public_bundle(tmp_path)


def test_public_bundle_requires_checksum_manifest(tmp_path):
    shutil.copytree(EXAMPLE / "public", tmp_path / "public")
    (tmp_path / "public" / "SHA256SUMS").unlink()

    with pytest.raises(ValueError, match="missing files: SHA256SUMS"):
        DEMO.verify_public_bundle(tmp_path / "public")


def test_public_bundle_rejects_unexpected_entries(tmp_path):
    shutil.copytree(EXAMPLE / "public", tmp_path / "public")
    (tmp_path / "public" / "raw.trace").mkdir()

    with pytest.raises(ValueError, match="unexpected entries: raw.trace"):
        DEMO.verify_public_bundle(tmp_path / "public")


def test_public_bundle_rejects_symlinks(tmp_path):
    shutil.copytree(EXAMPLE / "public", tmp_path / "public")
    summary = tmp_path / "public" / "capture-summary.csv"
    target = tmp_path / "summary.csv"
    summary.replace(target)
    summary.symlink_to(target)

    with pytest.raises(ValueError, match="contains symlinks: capture-summary.csv"):
        DEMO.verify_public_bundle(tmp_path / "public")


def test_manifest_capture_command_records_custom_parameters(monkeypatch, tmp_path):
    monkeypatch.setattr(
        DEMO,
        "_safe_host_metadata",
        lambda: {"hardware": "Apple Silicon", "architecture": "arm64"},
    )
    captures = [
        DEMO.CaptureSummary(
            expected_dispatch_count=17,
            attributed_interval_count=17,
            all_process_interval_count=17,
            known_unrelated_interval_count=0,
            known_unrelated_interval_share_percent=0.0,
            unattributed_interval_count=0,
            unattributed_interval_share_percent=0.0,
            window_server_interval_count=0,
            window_server_interval_share_percent=0.0,
            schema_count=82,
            exported_rows=17,
            exported_columns=18,
            exported_cells=306,
            reference_cells=0,
            dispatch_count_matches=True,
        )
    ]
    public = tmp_path / "public"
    public.mkdir()

    DEMO._write_public_bundle(
        public,
        captures,
        source=EXAMPLE / "metal_workload.swift",
        time_limit="3s",
    )

    manifest = json.loads((public / "experiment-manifest.json").read_text())
    assert (
        manifest["commands"][1]
        == "uv run python examples/metal_evidence/evidence_demo.py capture "
        "--output-dir '<OUTPUT_DIR>' --dispatches 17 --time-limit 3s"
    )


def test_committed_bundle_is_private_data_free_and_self_consistent():
    public = EXAMPLE / "public"
    if not public.exists():
        pytest.skip("generated bundle not present until the live capture is packaged")
    DEMO.verify_public_bundle(public)

    summary = json.loads((public / "capture-summary.json").read_text())
    assert summary["provenance"]["attributed_interval_count"].startswith(
        "measured_native"
    )
    assert summary["provenance"]["known_unrelated_interval_count"].startswith(
        "derived:"
    )
    assert summary["provenance"]["known_unrelated_interval_share_percent"].startswith(
        "derived:"
    )
