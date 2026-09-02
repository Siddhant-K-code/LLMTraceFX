"""Load-bearing checks for the committed exploratory Qwen3-8B control."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from llmtracefx.optimizer.lab.core import (
    assert_shareable,
    render_lab_report_html,
)
from llmtracefx.optimizer.lab.qwen3_8b.conversion import (
    conversion_manifest_hash,
)
from llmtracefx.optimizer.lab.qwen3_8b.conversion_manifest import (
    ConversionManifest,
)

ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT / "examples" / "optimizer" / "qwen3-8b-m5-control"
CONVERSION_MANIFEST = (
    ROOT
    / "llmtracefx"
    / "optimizer"
    / "lab"
    / "qwen3_8b"
    / "data"
    / "qwen3-8b-conversion-manifest-v1.json"
)
PUBLIC_FILES = (
    "control-manifest.json",
    "conversion-preflight-refusal-example.json",
    "conversion-summary.json",
    "evidence-manifest.json",
    "evidence-summary.json",
    "report.html",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str) -> dict:
    return json.loads((EXAMPLE / name).read_text(encoding="utf-8"))


def test_checksums_bind_every_public_evidence_file() -> None:
    expected = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in (EXAMPLE / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    }
    assert tuple(expected) == PUBLIC_FILES
    assert {name: _sha256(EXAMPLE / name) for name in PUBLIC_FILES} == expected

    evidence = _load("evidence-manifest.json")
    assert evidence["source_artifacts"] == {
        name: expected[name]
        for name in (
            "conversion-preflight-refusal-example.json",
            "conversion-summary.json",
            "control-manifest.json",
            "evidence-summary.json",
            "report.html",
        )
    }


def test_conversion_summary_binds_exact_source_converter_and_output() -> None:
    packaged = ConversionManifest.read_json(CONVERSION_MANIFEST)
    conversion = _load("conversion-summary.json")
    control = _load("control-manifest.json")

    assert conversion["conversion_manifest_hash"] == conversion_manifest_hash(packaged)
    assert conversion["attempt"] == {
        "status": "completed",
        "started_at": "2026-09-02T08:05:15.114822Z",
        "ended_at": "2026-09-02T08:05:21.947054Z",
        "timestamp_scope": (
            "conversion subprocess only; source download completed before started_at"
        ),
        "downloads_performed": True,
        "conversion_process_started": True,
        "automatic_retry_performed": False,
        "exit_code": 0,
        "timed_out": False,
        "descendants_cleaned": True,
    }
    assert conversion["source"]["official_id"] == "Qwen/Qwen3-8B"
    assert conversion["source"]["official_revision"] == (
        "b968826d9c46dd6066d109eabc6255188de91218"
    )
    assert conversion["source"]["license"] == "Apache-2.0"
    assert conversion["source"]["total_bytes"] == 16397461266
    assert len(conversion["source"]["files"]) == 15
    assert sum(item["size_bytes"] for item in conversion["source"]["files"]) == (
        conversion["source"]["total_bytes"]
    )
    assert conversion["converter"] == {
        "package": "mlx-lm",
        "version": "0.31.3",
        "git_repository": "https://github.com/ml-explore/mlx-lm",
        "git_revision": "ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd",
    }
    assert conversion["parameters"] == {
        "quantize": True,
        "q_group_size": 64,
        "q_bits": 4,
        "q_mode": "affine",
        "dtype": None,
        "quant_predicate": None,
        "dequantize": False,
        "trust_remote_code": False,
        "upload_repo": None,
    }
    assert conversion["output"]["binding_fingerprint"] == (
        "df71c0372db25213fc9ee4efe23b3502ba6fc6d0"
    )
    assert conversion["output"]["total_bytes"] == 4619328159
    assert len(conversion["output"]["files"]) == 8
    assert conversion["output"]["files"] == control["model"]["files"]
    assert control["model"]["expected_download_bytes"] == 4619328159


def test_manifest_records_exploratory_success_without_causal_overclaim() -> None:
    evidence = _load("evidence-manifest.json")
    assert evidence["run"] == {
        "completed_at_utc": "2026-09-02T08:08:36.185127Z",
        "code_checkout_commit": "6b82cf276ee1e1cef03a0c92847082f872c8feba",
        "mode": "exploratory",
        "clean_boot_operator_assertion": False,
        "status": "completed",
        "tiers_requested": [2048, 8192, 16384],
        "measured_runs_per_tier": 4,
        "measured_runs_total": 12,
    }
    assert [
        (
            tier["requested_tokens"],
            tier["mean_actual_input_tokens"],
            tier["passing_runs"],
            tier["pass_rate"],
        )
        for tier in evidence["tier_outcomes"]
    ] == [
        (2048, 1613.0, 4, 1.0),
        (8192, 6373.0, 4, 1.0),
        (16384, 12697.0, 4, 1.0),
    ]
    comparison = evidence["comparison_scope"]
    assert comparison["different_model_and_system_identity"] is True
    assert "does not replace" in comparison["statement"]
    assert "or prove that the 27B OOM" in comparison["statement"]
    assert any("No clean-boot assertion" in item for item in evidence["limitations"])


def test_supported_metrics_stay_scoped_and_report_is_deterministic() -> None:
    report = _load("evidence-summary.json")
    assert_shareable(report)
    for tier in report["tiers"]:
        assert tier["timing_provenance"] == "measured_wall_clock"
        assert tier["memory_provenance"] == "measured_native"
        assert tier["token_rate_provenance"] == "derived"
        assert tier["max_active_memory_bytes"] is not None
        assert tier["max_cache_memory_bytes"] is not None
        assert tier["max_peak_memory_bytes"] is not None
    html = (EXAMPLE / "report.html").read_text(encoding="utf-8")
    assert render_lab_report_html(report) == html
    assert "<h1>M5 Pro × Qwen3-8B</h1>" in html
    assert "Qwen3.8-27B</h1>" not in html
    assert "Observed MLX active memory" in html
    assert "Observed MLX cache memory" in html


def test_public_bundle_contains_no_private_or_raw_artifacts() -> None:
    forbidden = (
        re.compile(r"/Users/"),
        re.compile(r"/home/"),
        re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
        re.compile(r"\b(?:hf_|sk-)[A-Za-z0-9_-]{12,}\b"),
    )
    for path in EXAMPLE.iterdir():
        assert path.is_file()
        assert not path.is_symlink()
        text = path.read_text(encoding="utf-8")
        for pattern in forbidden:
            assert pattern.search(text) is None
    for name in ("conversion-summary.json", "evidence-manifest.json"):
        assert_shareable(_load(name))
    html = (EXAMPLE / "report.html").read_text(encoding="utf-8")
    assert "<script" not in html
    assert "http://" not in html
    assert "https://" not in html
    names = {path.name for path in EXAMPLE.iterdir()}
    assert not names.intersection(
        {"response.txt", "prompt.txt", "journal.json", "conversion-receipt.json"}
    )


def test_readme_links_bundle_and_preserves_comparison_boundary() -> None:
    readme = (EXAMPLE / "README.md").read_text(encoding="utf-8")
    for name in (*PUBLIC_FILES, "SHA256SUMS"):
        assert f"]({name})" in readme
    assert "different model and system identity" in readme
    assert "does not prove that the 27B OOM was caused only by parameter count" in (
        " ".join(readme.split())
    )
    assert "no clean-boot operator assertion was made" in readme.lower()
