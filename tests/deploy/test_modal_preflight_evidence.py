"""Load-bearing tests for the committed Modal GLM preflight refusal."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
PUBLIC = ROOT / "examples" / "optimizer" / "modal-glm53flash-preflight"
SPEC = importlib.util.spec_from_file_location(
    "modal_glm_preflight_evidence_bundle", PUBLIC / "evidence_bundle.py"
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


def test_refusal_preserves_zero_spend_and_exact_login_step() -> None:
    manifest = load("experiment-manifest.json")

    assert manifest["decision"]["status"] == "refused"
    assert manifest["decision"]["paid_execution_allowed"] is False
    assert manifest["authorization"]["hard_cap_usd"] == "10.000000"
    assert manifest["authorization"]["provider_reported_credit_use_usd"] is None
    assert (
        manifest["authorization"]["experiment_attributable_spend_usd_inferred"]
        == "0.000000"
    )
    assert manifest["authorization"]["modal_cli_authenticated"] is False
    assert manifest["authorization"]["exact_login_step"] == "uv run modal setup"
    assert manifest["provider_policy"]["modal_paid_commands_executed"] == 0
    assert manifest["provider_policy"]["automatic_retries"] == 0


def test_refused_plan_withholds_every_paid_step() -> None:
    plan = load("budget-plan.json")

    assert plan["approved"] is False
    assert plan["cost_envelope"]["worst_case_usd"] == pytest.approx(48.218112)
    assert plan["cost_envelope"]["budget_usd"] == pytest.approx(10.0)
    paid = {step["name"] for step in plan["steps"] if step["spends_money"]}
    assert not paid.intersection(plan["executable_steps"])


def test_model_inventory_and_provenance_limits_are_exact() -> None:
    manifest = load("experiment-manifest.json")
    inventory = load("inventory-summary.json")

    assert manifest["model"]["revision"] == ("03eb5366286afd40d2221b1d9c63a6dd1ba4832e")
    assert inventory["file_count"] == 72
    assert inventory["total_bytes"] == 328366172318
    assert inventory["safetensors_shard_count"] == 62
    assert inventory["safetensors_shards_with_published_sha256"] == 62
    assert len(inventory["files"]) == 72
    assert sum(entry["size_bytes"] for entry in inventory["files"]) == 328366172318
    assert sum(entry["sha256"] is not None for entry in inventory["files"]) == 63
    assert (
        manifest["serving_configuration"]["exact_framework_source_revision_verified"]
        is False
    )


def test_no_runtime_observation_is_invented() -> None:
    manifest = load("experiment-manifest.json")

    assert manifest["execution"]["weight_staging_attempted"] is False
    assert manifest["execution"]["weight_verification_attempted"] is False
    assert manifest["execution"]["deployment_attempted"] is False
    assert manifest["execution"]["readiness_attempted"] is False
    assert manifest["execution"]["smoke_requests_attempted"] == 0
    assert manifest["execution"]["gpu_containers_created"] == 0
    assert manifest["claims"]["hardware_fit_proven"] is False
    assert manifest["claims"]["benchmark_claimed"] is False


def test_documented_offline_verifier_command_works() -> None:
    completed = subprocess.run(
        [sys.executable, str(PUBLIC / "evidence_bundle.py"), "verify"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "public Modal preflight evidence verified" in completed.stdout


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("/Users/private/run.json", "private home path"),
        ("wk-1234abcd", "Modal credential"),
        ("ws-1234abcd", "Modal credential"),
        ("sk-secretvalue", "API credential"),
        ("https://private-name.modal.run", "private Modal endpoint"),
        ("private@example.com", "email address"),
    ],
)
def test_privacy_scan_refuses_private_values(value: str, message: str) -> None:
    with pytest.raises(BUNDLE.EvidenceError, match=message):
        BUNDLE._scan_privacy("test", value)
