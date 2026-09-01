"""Load-bearing tests for the committed clean-boot OOM autopsy evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT / "examples" / "optimizer" / "m5-pro-qwen3.8-27b-oom-autopsy"
PUBLIC = EXAMPLE / "publication"
SPEC = importlib.util.spec_from_file_location(
    "m5_autopsy_evidence_bundle", EXAMPLE / "evidence_bundle.py"
)
assert SPEC is not None and SPEC.loader is not None
BUNDLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUNDLE
SPEC.loader.exec_module(BUNDLE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str) -> dict:
    return json.loads((PUBLIC / name).read_text(encoding="utf-8"))


def test_committed_bundle_verifies_and_binds_verified_source_hashes() -> None:
    BUNDLE.verify_bundle(PUBLIC)
    assert {
        name: _sha256(PUBLIC / name) for name in BUNDLE.SOURCE_SHA256
    } == BUNDLE.SOURCE_SHA256
    assert _sha256(PUBLIC / "autopsy-plan.json") == BUNDLE.PLAN_SHA256


def test_manifest_binds_exact_code_model_and_t256_oom_outcome() -> None:
    manifest = _load("evidence-manifest.json")
    assert manifest["run"] == {
        "completed_at_utc": "2026-09-01T17:45:36.921331Z",
        "code_checkout_commit": "2519bc8da309656d2e2ce2a7063f19b0dfb4c9ed",
        "mode": "publication",
        "clean_boot_operator_assertion": True,
        "terminal_outcome": "oom",
        "reason": "MLX/Metal reported insufficient memory",
        "child_exit_code": 2,
        "timed_out": False,
        "descendants_cleaned": True,
        "journal_complete": True,
        "journal_terminal": "oom",
    }
    assert manifest["model"] == {
        "repository_id": "mlx-community/Qwen3.8-27B-4bit",
        "revision": "3e6447f082e89cc7f0bc6e5441afd38dfce760ff",
        "checkpoint_bytes": 16081490933,
        "checkpoint_file_count": 15,
    }
    assert manifest["workload"] == {
        "tier": "t256",
        "requested_prompt_tokens": 256,
        "actual_prompt_tokens": 256,
        "first_token_observed": False,
        "generation_completion_observed": False,
        "evaluator_result_available": False,
        "quality_metrics_available": False,
        "throughput_metrics_available": False,
    }


def test_summary_has_exact_stage_order_scopes_and_prefill_boundary_delta() -> None:
    summary = _load("oom-autopsy-summary.json")
    assert tuple(item["stage"] for item in summary["checkpoints"]) == BUNDLE.STAGES
    assert tuple(item["sequence"] for item in summary["checkpoints"]) == tuple(range(7))
    assert summary["provenance"]["mlx_active_bytes"]["scope"] == "mlx_allocator"
    assert summary["provenance"]["mlx_cache_bytes"]["scope"] == "mlx_allocator"
    assert summary["provenance"]["mlx_peak_bytes"]["scope"] == "mlx_allocator"
    assert summary["provenance"]["host_rss_bytes"]["scope"] == "host_process"
    assert summary["provenance"]["host_max_rss_bytes"]["scope"] == "host_process"
    assert summary["provenance"]["swap_bytes"]["scope"] == "host_system_swap"

    boundary = next(
        item
        for item in summary["observed_deltas"]
        if item["from_stage"] == "immediately_before_prefill_generation"
        and item["to_stage"] == "caught_oom"
    )
    assert boundary["mlx_active_bytes_delta"] == 2672187942
    assert boundary["mlx_cache_bytes_delta"] == 76420156
    assert boundary["mlx_peak_bytes_delta"] == 2839022214


def test_no_first_token_completion_quality_or_throughput_is_claimed() -> None:
    summary = _load("oom-autopsy-summary.json")
    stages = {item["stage"] for item in summary["checkpoints"]}
    assert "after_first_token" not in stages
    assert "completion" not in stages
    assert summary["terminal_outcome"] == "oom"
    assert summary["journal_complete"] is True
    assert summary["journal_terminal"] == "oom"

    manifest = _load("evidence-manifest.json")
    workload = manifest["workload"]
    assert workload["first_token_observed"] is False
    assert workload["generation_completion_observed"] is False
    assert workload["evaluator_result_available"] is False
    assert workload["quality_metrics_available"] is False
    assert workload["throughput_metrics_available"] is False


def test_public_bundle_privacy_and_embedded_resources_are_fail_closed() -> None:
    for path in PUBLIC.iterdir():
        assert path.is_file()
        assert not path.is_symlink()
        text = path.read_text(encoding="utf-8")
        BUNDLE.scan_privacy(path.name, text)
    report = (PUBLIC / "oom-autopsy-report.html").read_text(encoding="utf-8")
    chart = (PUBLIC / "mlx-memory-by-stage.svg").read_text(encoding="utf-8")
    for pattern in BUNDLE.EXTERNAL_HTML_PATTERNS:
        assert pattern.search(report) is None
        assert pattern.search(chart) is None


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("/Users/private/cache/model", "absolute home path"),
        ("private@example.com", "email address"),
        ("hf_1234567890abcdef", "credential-like token"),
        (".cache/models/private", "model cache path"),
        ("logs/private-run.txt", "private log path"),
    ],
)
def test_privacy_scan_rejects_private_values(payload: str, message: str) -> None:
    with pytest.raises(BUNDLE.EvidenceError, match=message):
        BUNDLE.scan_privacy("unsafe.txt", payload)


def test_json_loader_rejects_private_keys_and_non_finite_numbers(
    tmp_path: Path,
) -> None:
    private = tmp_path / "private.json"
    private.write_text('{"hostname": "private-host"}', encoding="utf-8")
    with pytest.raises(BUNDLE.EvidenceError, match="private identity/path field"):
        BUNDLE._load_json(private)

    non_finite = tmp_path / "non-finite.json"
    non_finite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(BUNDLE.EvidenceError, match="non-finite JSON number"):
        BUNDLE._load_json(non_finite)


def test_verifier_rejects_unexpected_files_and_symlinks(tmp_path: Path) -> None:
    unexpected = tmp_path / "unexpected"
    shutil.copytree(PUBLIC, unexpected)
    (unexpected / "journal.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BUNDLE.EvidenceError, match="unexpected=.*journal.json"):
        BUNDLE.verify_bundle(unexpected)

    linked = tmp_path / "linked"
    shutil.copytree(PUBLIC, linked)
    summary = linked / "oom-autopsy-summary.json"
    target = tmp_path / "summary.json"
    summary.replace(target)
    summary.symlink_to(target)
    with pytest.raises(BUNDLE.EvidenceError, match="regular non-symlink"):
        BUNDLE.verify_bundle(linked)


def test_manifest_binding_fails_even_if_checksum_is_regenerated(tmp_path: Path) -> None:
    public = tmp_path / "publication"
    shutil.copytree(PUBLIC, public)
    manifest_path = public / "evidence-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run"]["code_checkout_commit"] = "0" * 40
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    BUNDLE.write_generated_files(public)
    with pytest.raises(BUNDLE.EvidenceError, match="code_checkout_commit drifted"):
        BUNDLE.verify_bundle(public)


def test_chart_is_deterministic_scope_separated_and_keeps_missing_values_missing() -> (
    None
):
    summary = _load("oom-autopsy-summary.json")
    committed = (PUBLIC / "mlx-memory-by-stage.svg").read_text(encoding="utf-8")
    assert BUNDLE.render_chart(summary) == committed
    assert "scope: mlx_allocator; independent axis" in committed
    assert "scope: host_process; independent axis" in committed
    assert "scope: host_system_swap; independent axis" in committed
    assert committed.count("OOM boundary") == 3

    missing = copy.deepcopy(summary)
    missing["checkpoints"][2]["mlx_active_bytes"] = None
    rendered = BUNDLE.render_chart(missing)
    assert 'data-series="MLX active" data-stage="after_model_load"' not in rendered
    assert "Missing values are omitted, never converted to zero." in rendered


def test_readme_uses_bytes_as_authority_and_links_every_public_file() -> None:
    readme = (EXAMPLE / "README.md").read_text(encoding="utf-8")
    normalized = " ".join(readme.split())
    BUNDLE.scan_privacy("README.md", readme)
    for name in BUNDLE.PUBLIC_FILES:
        assert f"(publication/{name})" in readme
    assert "Bytes are authoritative" in normalized
    assert "not additive" in normalized
    assert "not causal" in normalized
    assert "20,358,144,983 bytes (18.960000 GiB)" in normalized
    assert "18,894,739,574 (17.597098 GiB)" in normalized


def test_prior_pr51_pr52_evidence_bytes_are_immutable() -> None:
    expected = {
        "examples/optimizer/m5-pro-qwen3.8-27b-fit-frontier/README.md": (
            "02d9718bb32f5508dc825ac131bad2ca379a82dfa64debd37e8aada19a1386a6"
        ),
        "examples/optimizer/m5-pro-qwen3.8-27b-fit-frontier/exploratory/"
        "fit-frontier-report.html": (
            "5830a7ca5dc2d3c2e49c9f2477ad44b9adeef694c2e8a5149acf5d52c4b7fc32"
        ),
        "examples/optimizer/m5-pro-qwen3.8-27b-fit-frontier/exploratory/"
        "fit-frontier-summary.json": (
            "7e8116b3fd4a4d639ae0693fc74eaf8f08bc93bb02bb33fdd6556c650ffc4c8a"
        ),
        "examples/optimizer/m5-pro-qwen3.8-27b/README.md": (
            "abb2fa916c61e76b3a681c3d691314c3e02ad6613b4abdc64b3af01f12592beb"
        ),
        "examples/optimizer/m5-pro-qwen3.8-27b/evidence-summary.json": (
            "e09ee3e130e91824b0d7e9c29336e7742f862954b53237edd943a6ffa67da5d4"
        ),
        "examples/optimizer/m5-pro-qwen3.8-27b/report.html": (
            "42523fca6a40a739c3d36cf105a174b8ee180b2c990ce0fa98e58d4d609607f3"
        ),
    }
    assert {name: _sha256(ROOT / name) for name in expected} == expected
