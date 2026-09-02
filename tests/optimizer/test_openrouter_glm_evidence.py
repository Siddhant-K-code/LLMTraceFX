"""Load-bearing tests for the committed OpenRouter GLM evidence bundle."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
PUBLIC = ROOT / "examples" / "optimizer" / "openrouter-glm-2k"
SPEC = importlib.util.spec_from_file_location(
    "openrouter_glm_evidence_bundle", PUBLIC / "evidence_bundle.py"
)
assert SPEC is not None and SPEC.loader is not None
BUNDLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUNDLE
SPEC.loader.exec_module(BUNDLE)


def load(name: str) -> dict:
    return json.loads((PUBLIC / name).read_text(encoding="utf-8"))


def test_committed_public_bundle_verifies() -> None:
    BUNDLE.verify()
    assert "evidence_bundle.py" in (PUBLIC / "SHA256SUMS").read_text(encoding="utf-8")


def test_manifest_records_exact_cap_request_count_spend_and_local_exclusion() -> None:
    manifest = load("experiment-manifest.json")

    assert manifest["run"]["paid_inference_requests"] == 8
    assert manifest["run"]["automatic_retries"] == 0
    assert manifest["budget"]["authorized_total_usd"] == "5.000000000000"
    assert manifest["budget"]["planned_ceiling_usd"] == "0.074532600000"
    assert manifest["budget"]["provider_reported_request_total_usd"] == "0.006152620000"
    assert manifest["budget"]["remaining_authorization_usd"] == "4.993847380000"
    assert manifest["local_qwen3_8b_context"]["direct_ranking_excluded"] is True
    assert manifest["claims"]["universal_winner_claimed"] is False


def test_all_rows_are_exact_pinned_passing_measurements() -> None:
    rows = load("measurements.json")["rows"]

    assert len(rows) == 8
    assert {row["system"]["requested_model_id"] for row in rows} == set(
        BUNDLE.MODEL_BUILDS
    )
    assert all(row["system"]["route_slug"] == "z-ai/fp8" for row in rows)
    assert all(row["verification"]["status"] == "completed" for row in rows)
    assert all(row["verification"]["quality_score"] == 1.0 for row in rows)
    assert all(row["api_evidence"]["reasoning_text_persisted"] is False for row in rows)
    assert all(
        row["request_plan"]["request_parameters"]["max_tokens"] == 96 for row in rows
    )


def test_documented_direct_verifier_command_works_from_checkout() -> None:
    completed = subprocess.run(
        [sys.executable, str(PUBLIC / "evidence_bundle.py"), "verify"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "public evidence verified" in completed.stdout


def test_comparison_is_hosted_only_and_constraints_first() -> None:
    report = load("comparison.json")

    assert report["results_dirs"] == ["measurements.json"]
    assert report["policy"]["objective"] == "max_correct_cases_per_minute"
    assert report["policy"]["constraints"]["min_pass_rate"] == 1.0
    assert len(report["strata"]) == 2
    for stratum in report["strata"]:
        assert stratum["outcome"] == "recommended"
        assert stratum["recommended"]["system_key"]["model_id"] == "z-ai/glm-5.3"
        assert len(stratum["ranked"]) == 2


def test_comparison_html_is_the_sanitized_json_rendering() -> None:
    expected = BUNDLE._render_comparison(PUBLIC / "comparison.json")

    assert (PUBLIC / "comparison.html").read_text(encoding="utf-8") == expected
    assert ".cache/" not in expected
    assert "/Users/" not in expected


def test_generation_metadata_discloses_missing_public_correlation() -> None:
    generation = load("generation-metadata.json")
    assert generation["completion_correlation_status"].startswith(
        "not_publicly_verifiable"
    )
    observations = load("experiment-manifest.json")["systems"][
        "generation_metadata_observations"
    ]
    assert observations["row_level_public_correlation_available"] is False


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("/Users/private/run.json", "private home path"),
        (".cache/private/run.json", "private cache path"),
        ("sk-or-v1-secretvalue", "OpenRouter credential"),
        ("gen-abcdefgh12345678", "provider identifier"),
        ("private@example.com", "email address"),
    ],
)
def test_privacy_scan_refuses_private_values(value: str, message: str) -> None:
    with pytest.raises(BUNDLE.EvidenceError, match=message):
        BUNDLE._scan_privacy("test", value)
