"""Load-bearing checks for the committed CloudRift refusal evidence."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
PUBLIC = ROOT / "examples" / "optimizer" / "cloudrift-glm53flash-preflight"
SPEC = importlib.util.spec_from_file_location(
    "cloudrift_glm_preflight_evidence_bundle", PUBLIC / "evidence_bundle.py"
)
assert SPEC is not None and SPEC.loader is not None
BUNDLE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BUNDLE
_previous_dont_write_bytecode = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    SPEC.loader.exec_module(BUNDLE)
finally:
    sys.dont_write_bytecode = _previous_dont_write_bytecode


def load(name: str) -> dict:
    return json.loads((PUBLIC / name).read_text(encoding="utf-8"))


def test_committed_public_bundle_verifies() -> None:
    BUNDLE.verify()


def test_refusal_preserves_cap_reserve_and_zero_spend() -> None:
    plan = load("budget-plan.json")
    manifest = load("experiment-manifest.json")

    assert plan["approved"] is False
    authorization = plan["authorization"]
    assert authorization["hard_cap_usd"] == "80.000000"
    assert authorization["conditional_h200_cost_usd"] == "60.000000"
    assert authorization["minimum_reserve_usd"] == "20.000000"
    assert manifest["authorization"]["provider_reported_spend_usd"] is None
    assert (
        manifest["authorization"]["experiment_attributable_spend_usd_inferred"]
        == "0.000000"
    )
    assert manifest["provider_policy"]["cloudrift_paid_commands_executed"] == 0
    assert manifest["provider_policy"]["automatic_retries"] == 0


def test_execution_is_withheld_for_access_ttl_inventory_and_recipe() -> None:
    plan = load("budget-plan.json")
    blockers = " ".join(plan["blockers"])

    assert "V100 SXM2" in blockers
    assert "200,366,172,318 bytes below" in blockers
    assert "on-demand H200 pricing" in blockers
    assert "billing rounding" in blockers
    assert "scheduled termination/TTL" in blockers
    assert "8x H200 is not available" in blockers
    assert "CLI/profile or SSH access" in blockers
    assert "tested CloudRift GLM serving recipe" in blockers
    assert plan["executable_steps"] == []


def test_available_shape_arithmetic_is_explicit_and_non_substitutable() -> None:
    plan = load("budget-plan.json")
    available = plan["available_configuration"]
    arithmetic = plan["capability_arithmetic"]

    assert available["gpu_type"] == "V100 SXM2"
    assert available["gpu_count"] == 8
    assert available["gpu_memory_gb_each"] == 16
    assert available["host_memory_gb"] == 52
    assert available["disk_gb"] == 400
    assert available["usd_per_gpu_hour"] == "0.250000"
    assert arithmetic["available_gpu_memory_bytes"] == 128_000_000_000
    assert arithmetic["gpu_memory_shortfall_bytes"] == 200_366_172_318
    assert arithmetic["offload_or_substitution_allowed"] is False


def test_shared_inventory_is_exact_without_copying_the_file_list() -> None:
    reference = load("model-inventory-reference.json")

    assert reference["revision"] == "03eb5366286afd40d2221b1d9c63a6dd1ba4832e"
    assert reference["file_count"] == 72
    assert reference["total_bytes"] == 328_366_172_318
    assert reference["safetensors_shard_count"] == 62
    assert reference["files_with_published_sha256"] == 63
    assert reference["weights_downloaded_locally"] is False
    assert "files" not in reference


def test_no_runtime_observation_or_benchmark_claim_is_invented() -> None:
    manifest = load("experiment-manifest.json")

    assert manifest["execution"]["provisioning_attempted"] is False
    assert manifest["execution"]["weight_staging_attempted"] is False
    assert manifest["execution"]["model_load_attempted"] is False
    assert manifest["execution"]["readiness_attempted"] is False
    assert manifest["execution"]["smoke_requests_attempted"] == 0
    assert manifest["claims"]["benchmark_claimed"] is False
    assert manifest["claims"]["production_readiness_claimed"] is False
    assert manifest["claims"]["memory_fit_claimed"] is False


def test_documented_offline_verifier_command_works() -> None:
    completed = subprocess.run(
        [sys.executable, str(PUBLIC / "evidence_bundle.py"), "verify"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "public CloudRift preflight evidence verified" in completed.stdout


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("/Users/private/run.json", "private home path"),
        ("/home/private/run.json", "private home path"),
        (r"C:\Users\private\run.json", "private home path"),
        ("-----BEGIN OPENSSH PRIVATE KEY-----", "private SSH key"),
        ("sk-secretvalue", "API credential"),
        ("203.0.113.7", "IP address"),
        ("https://user:secret@example.com", "URL credential"),
        ("private@example.com", "email address"),
    ],
)
def test_privacy_scan_refuses_private_values(value: str, message: str) -> None:
    with pytest.raises(BUNDLE.EvidenceError, match=message):
        BUNDLE._scan_privacy("test", value)
