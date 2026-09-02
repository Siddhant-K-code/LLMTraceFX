"""Load-bearing checks for the concise CloudRift capability refusal."""

from __future__ import annotations

import json
import socket
from copy import deepcopy
from pathlib import Path

import pytest

from llmtracefx.deploy.cloudrift import (
    HARD_CAP_USD,
    MINIMUM_RESERVE_USD,
    PLANNED_CAP_USD,
    CloudRiftSnapshot,
    build_cloudrift_plan,
)
from llmtracefx.deploy.cloudrift_cli import run
from llmtracefx.deploy.errors import DeploymentPlanError
from llmtracefx.deploy.model_inventory import inventory_from_dict, load_inventory

ROOT = Path(__file__).parents[2]
INVENTORY = (
    ROOT
    / "examples"
    / "optimizer"
    / "modal-glm53flash-preflight"
    / "inventory-summary.json"
)
SNAPSHOT = (
    ROOT
    / "examples"
    / "optimizer"
    / "cloudrift-glm53flash-preflight"
    / "provider-snapshot.json"
)


def plan():  # type: ignore[no-untyped-def]
    snapshot = CloudRiftSnapshot.from_dict(json.loads(SNAPSHOT.read_text()))
    return build_cloudrift_plan(snapshot, load_inventory(INVENTORY))


def test_refusal_preserves_cap_and_conditional_reserve() -> None:
    result = plan()
    payload = result.to_dict()

    assert result.approved is False
    assert result.conditional_h200_cost_usd == PLANNED_CAP_USD
    assert HARD_CAP_USD - result.conditional_h200_cost_usd == MINIMUM_RESERVE_USD
    assert payload["authorization"]["hard_cap_usd"] == "80.000000"
    assert payload["authorization"]["conditional_cost_is_spending_authority"] is False
    assert payload["executable_steps"] == []


def test_actual_v100_inventory_cannot_hold_the_exact_model() -> None:
    payload = plan().to_dict()
    available = payload["available_configuration"]
    arithmetic = payload["capability_arithmetic"]

    assert available["gpu_type"] == "V100 SXM2"
    assert available["gpu_count"] == 8
    assert available["gpu_memory_gb_each"] == 16
    assert available["aggregate_gpu_memory_bytes"] == 128_000_000_000
    assert arithmetic["published_model_bytes"] == 328_366_172_318
    assert arithmetic["gpu_memory_shortfall_bytes"] == 200_366_172_318
    assert arithmetic["host_memory_plus_gpu_memory_bytes"] == 180_000_000_000
    assert arithmetic["offload_or_substitution_allowed"] is False


def test_exact_model_inventory_is_reused_from_modal_preflight() -> None:
    inventory = load_inventory(INVENTORY)
    inventory.assert_glm_53_flash()

    assert inventory.revision == "03eb5366286afd40d2221b1d9c63a6dd1ba4832e"
    assert inventory.file_count == 72
    assert inventory.total_bytes == 328_366_172_318
    assert inventory.safetensors_shard_count == 62
    assert inventory.published_hash_count == 63


def test_exact_inventory_rejects_altered_file_metadata() -> None:
    raw = json.loads(INVENTORY.read_text())
    raw["files"][6]["sha256"] = "0" * 64
    altered = inventory_from_dict(raw)

    with pytest.raises(DeploymentPlanError, match="canonical_sha256"):
        altered.assert_glm_53_flash()


def test_every_lifecycle_unknown_remains_a_blocker() -> None:
    blockers = " ".join(plan().blockers)

    assert "on-demand H200 pricing" in blockers
    assert "billing rounding" in blockers
    assert "Credit and tax" in blockers
    assert "Stop/terminate" in blockers
    assert "Persistent-disk" in blockers
    assert "scheduled termination/TTL" in blockers
    assert "CLI/profile or SSH access" in blockers
    assert "tested CloudRift GLM serving recipe" in blockers


def test_string_false_cannot_bypass_a_boolean_gate() -> None:
    raw = json.loads(SNAPSHOT.read_text())
    raw["h200_available_in_account"] = "false"

    with pytest.raises(DeploymentPlanError, match="JSON boolean"):
        CloudRiftSnapshot.from_dict(raw)


def test_unaccounted_storage_network_or_transfer_fees_are_blockers() -> None:
    raw = deepcopy(json.loads(SNAPSHOT.read_text()))
    raw["attached_storage_and_network_included"] = False
    raw["no_ingress_egress_or_api_fees"] = False
    snapshot = CloudRiftSnapshot.from_dict(raw)
    blockers = " ".join(
        build_cloudrift_plan(snapshot, load_inventory(INVENTORY)).blockers
    )

    assert "storage and networking charges are unbounded" in blockers
    assert "Ingress, egress, or API-call charges are unbounded" in blockers


def test_offline_cli_never_opens_a_socket(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("offline planning must not open a socket")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    output = tmp_path / "plan.json"
    args = type(
        "Args",
        (),
        {"snapshot": SNAPSHOT, "inventory": INVENTORY, "output": output},
    )()

    assert run(args) == 1
    payload = json.loads(output.read_text())
    assert payload["approved"] is False
    assert payload["network_request_performed"] is False
    assert payload["instance_created"] is False
    assert payload["executable_steps"] == []
