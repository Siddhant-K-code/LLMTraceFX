"""Load-bearing tests for the public Metal evidence example."""

from __future__ import annotations

import importlib.util
import json
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
    assert summary.unrelated_interval_count == 2
    assert summary.unrelated_interval_share_percent == 40.0
    assert summary.window_server_interval_count == 2
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


def test_output_directory_must_be_empty(tmp_path):
    output = tmp_path / "capture"
    output.mkdir()
    (output / "stale.json").write_text("{}")

    with pytest.raises(ValueError, match="stale artifact mixing"):
        DEMO._prepare_output(output)


def test_public_bundle_privacy_scan_rejects_private_data(tmp_path):
    for name in DEMO.PUBLIC_FILES:
        (tmp_path / name).write_text("safe", encoding="utf-8")
    (tmp_path / "capture-summary.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "attributed_interval_count": "measured_native",
                    "unrelated_interval_count": "derived: subtraction",
                    "unrelated_interval_share_percent": "derived: ratio",
                },
                "captures": [
                    {
                        "dispatch_count_matches": True,
                        "attributed_interval_count": 1,
                        "unrelated_interval_count": 0,
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


def test_committed_bundle_is_private_data_free_and_self_consistent():
    public = EXAMPLE / "public"
    if not public.exists():
        pytest.skip("generated bundle not present until the live capture is packaged")
    DEMO.verify_public_bundle(public)

    summary = json.loads((public / "capture-summary.json").read_text())
    assert summary["provenance"]["attributed_interval_count"].startswith(
        "measured_native"
    )
    assert summary["provenance"]["unrelated_interval_count"].startswith("derived:")
    assert summary["provenance"]["unrelated_interval_share_percent"].startswith(
        "derived:"
    )
