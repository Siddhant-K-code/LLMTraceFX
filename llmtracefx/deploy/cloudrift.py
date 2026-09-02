"""Offline capability gate for the CloudRift GLM-5.3-Flash validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from decimal import Decimal
from typing import Any

from .errors import DeploymentPlanError
from .model_inventory import GLM_53_FLASH_REVISION, PublishedInventory
from .recipe import parse_image_reference

HARD_CAP_USD = Decimal("80.00")
PLANNED_CAP_USD = Decimal("60.00")
MINIMUM_RESERVE_USD = Decimal("20.00")
PLANNED_WINDOW_HOURS = Decimal("3")
REQUESTED_GPU_TYPE = "H200"
REQUESTED_GPU_COUNT = 8
REQUESTED_TP = 8


def _money(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.000001')):.6f}"


def _required_bool(data: Mapping[str, Any], field: str) -> bool:
    value = data[field]
    if not isinstance(value, bool):
        raise DeploymentPlanError(f"{field} must be a JSON boolean")
    return value


@dataclass(frozen=True)
class CloudRiftSnapshot:
    """Sanitized provider facts; unknown execution gates stay ``None`` or false."""

    as_of: str
    evidence_kind: str
    pricing_source: str
    advertised_h200_usd_per_gpu_hour: Decimal
    advertised_h200_rate_term: str
    h200_on_demand_rate_verified: bool
    advertised_h200_memory_gb: int
    advertised_h200_local_storage_gb: int
    advertised_h200_max_gpus: int
    attached_storage_and_network_included: bool
    no_ingress_egress_or_api_fees: bool
    billing_increment_seconds: int | None
    billing_rounding_rule_verified: bool
    credits_and_tax_treatment_verified: bool
    stop_terminate_semantics_verified: bool
    persistent_disk_billing_verified: bool
    renter_scheduled_termination_supported: bool
    h200_available_in_account: bool
    access_configured_locally: bool
    official_recipe_verified: bool
    official_recipe_source: str | None
    framework_version: str | None
    image_reference: str | None
    available_gpu_type: str
    available_gpu_memory_gb_each: int
    available_gpu_count: int
    available_host_memory_gb: int
    available_disk_gb: int
    available_usd_per_gpu_hour: Decimal
    notes: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CloudRiftSnapshot:
        try:
            return cls(
                as_of=str(data["as_of"]),
                evidence_kind=str(data["evidence_kind"]),
                pricing_source=str(data["pricing_source"]),
                advertised_h200_usd_per_gpu_hour=Decimal(
                    str(data["advertised_h200_usd_per_gpu_hour"])
                ),
                advertised_h200_rate_term=str(data["advertised_h200_rate_term"]),
                h200_on_demand_rate_verified=_required_bool(
                    data, "h200_on_demand_rate_verified"
                ),
                advertised_h200_memory_gb=int(data["advertised_h200_memory_gb"]),
                advertised_h200_local_storage_gb=int(
                    data["advertised_h200_local_storage_gb"]
                ),
                advertised_h200_max_gpus=int(data["advertised_h200_max_gpus"]),
                attached_storage_and_network_included=_required_bool(
                    data, "attached_storage_and_network_included"
                ),
                no_ingress_egress_or_api_fees=_required_bool(
                    data, "no_ingress_egress_or_api_fees"
                ),
                billing_increment_seconds=(
                    None
                    if data.get("billing_increment_seconds") is None
                    else int(data["billing_increment_seconds"])
                ),
                billing_rounding_rule_verified=_required_bool(
                    data, "billing_rounding_rule_verified"
                ),
                credits_and_tax_treatment_verified=_required_bool(
                    data, "credits_and_tax_treatment_verified"
                ),
                stop_terminate_semantics_verified=_required_bool(
                    data, "stop_terminate_semantics_verified"
                ),
                persistent_disk_billing_verified=_required_bool(
                    data, "persistent_disk_billing_verified"
                ),
                renter_scheduled_termination_supported=_required_bool(
                    data, "renter_scheduled_termination_supported"
                ),
                h200_available_in_account=_required_bool(
                    data, "h200_available_in_account"
                ),
                access_configured_locally=_required_bool(
                    data, "access_configured_locally"
                ),
                official_recipe_verified=_required_bool(
                    data, "official_recipe_verified"
                ),
                official_recipe_source=data.get("official_recipe_source"),
                framework_version=data.get("framework_version"),
                image_reference=data.get("image_reference"),
                available_gpu_type=str(data["available_gpu_type"]),
                available_gpu_memory_gb_each=int(data["available_gpu_memory_gb_each"]),
                available_gpu_count=int(data["available_gpu_count"]),
                available_host_memory_gb=int(data["available_host_memory_gb"]),
                available_disk_gb=int(data["available_disk_gb"]),
                available_usd_per_gpu_hour=Decimal(
                    str(data["available_usd_per_gpu_hour"])
                ),
                notes=tuple(str(note) for note in data.get("notes", ())),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DeploymentPlanError(f"malformed CloudRift snapshot: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["advertised_h200_usd_per_gpu_hour"] = _money(
            self.advertised_h200_usd_per_gpu_hour
        )
        payload["available_usd_per_gpu_hour"] = _money(self.available_usd_per_gpu_hour)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True)
class CloudRiftPlan:
    snapshot: CloudRiftSnapshot
    inventory: PublishedInventory
    blockers: tuple[str, ...]
    conditional_h200_cost_usd: Decimal

    @property
    def approved(self) -> bool:
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        available_gpu_memory = (
            self.snapshot.available_gpu_memory_gb_each
            * self.snapshot.available_gpu_count
        )
        available_gpu_bytes = available_gpu_memory * 1_000_000_000
        return {
            "schema_version": "1",
            "kind": "llmtracefx.cloudrift_glm53flash.capability_plan",
            "approved": self.approved,
            "offline_only": True,
            "network_request_performed": False,
            "provider_authentication_used": False,
            "instance_created": False,
            "executable_steps": [],
            "authorization": {
                "hard_cap_usd": _money(HARD_CAP_USD),
                "planned_instance_cap_usd": _money(PLANNED_CAP_USD),
                "minimum_reserve_usd": _money(MINIMUM_RESERVE_USD),
                "conditional_h200_cost_usd": _money(self.conditional_h200_cost_usd),
                "conditional_cost_is_spending_authority": False,
            },
            "required_configuration": {
                "gpu_type": REQUESTED_GPU_TYPE,
                "gpu_count": REQUESTED_GPU_COUNT,
                "tensor_parallel_size": REQUESTED_TP,
                "model_revision": GLM_53_FLASH_REVISION,
                "quantization": "native FP8",
            },
            "available_configuration": {
                "evidence_kind": self.snapshot.evidence_kind,
                "gpu_type": self.snapshot.available_gpu_type,
                "gpu_count": self.snapshot.available_gpu_count,
                "gpu_memory_gb_each": self.snapshot.available_gpu_memory_gb_each,
                "aggregate_gpu_memory_gb": available_gpu_memory,
                "aggregate_gpu_memory_bytes": available_gpu_bytes,
                "host_memory_gb": self.snapshot.available_host_memory_gb,
                "disk_gb": self.snapshot.available_disk_gb,
                "usd_per_gpu_hour": _money(self.snapshot.available_usd_per_gpu_hour),
                "eight_gpu_usd_per_hour": _money(
                    self.snapshot.available_usd_per_gpu_hour
                    * self.snapshot.available_gpu_count
                ),
            },
            "model_inventory": self.inventory.summary(),
            "capability_arithmetic": {
                "published_model_bytes": self.inventory.total_bytes,
                "available_gpu_memory_bytes": available_gpu_bytes,
                "gpu_memory_shortfall_bytes": max(
                    0, self.inventory.total_bytes - available_gpu_bytes
                ),
                "host_memory_plus_gpu_memory_bytes": (
                    available_gpu_bytes
                    + self.snapshot.available_host_memory_gb * 1_000_000_000
                ),
                "offload_or_substitution_allowed": False,
            },
            "lifecycle_requirements_if_replanned": {
                "provider_ttl_from_instance_creation": True,
                "independent_local_watchdog": True,
                "automatic_retries": 0,
                "persistent_disk": False,
                "authenticated_loopback_server_via_ssh_tunnel": True,
                "smoke_requests": 1,
            },
            "provider_snapshot": self.snapshot.to_dict(),
            "blockers": list(self.blockers),
        }


def build_cloudrift_plan(
    snapshot: CloudRiftSnapshot, inventory: PublishedInventory
) -> CloudRiftPlan:
    """Return an evidence-only decision; this module has no execution path."""
    inventory.assert_glm_53_flash()
    conditional_cost = (
        snapshot.advertised_h200_usd_per_gpu_hour
        * REQUESTED_GPU_COUNT
        * PLANNED_WINDOW_HOURS
    )
    blockers: list[str] = []

    available_gpu_bytes = (
        snapshot.available_gpu_memory_gb_each
        * snapshot.available_gpu_count
        * 1_000_000_000
    )
    if snapshot.available_gpu_type != REQUESTED_GPU_TYPE:
        blockers.append(
            f"Authenticated console inventory offers {snapshot.available_gpu_type}, "
            f"not the required {REQUESTED_GPU_TYPE}."
        )
    if snapshot.available_gpu_count != REQUESTED_GPU_COUNT:
        blockers.append("Available GPU count does not match the required 8 GPUs.")
    if available_gpu_bytes < inventory.total_bytes:
        blockers.append(
            f"Available aggregate GPU memory is {available_gpu_bytes:,} bytes, "
            f"{inventory.total_bytes - available_gpu_bytes:,} bytes below the "
            "published model inventory."
        )
    if not snapshot.h200_available_in_account:
        blockers.append("8x H200 is not available in the authenticated account.")
    if not snapshot.h200_on_demand_rate_verified:
        blockers.append("Short on-demand H200 pricing is unverified.")
    if not snapshot.attached_storage_and_network_included:
        blockers.append("Attached storage and networking charges are unbounded.")
    if not snapshot.no_ingress_egress_or_api_fees:
        blockers.append("Ingress, egress, or API-call charges are unbounded.")
    if conditional_cost > PLANNED_CAP_USD:
        blockers.append("Conditional H200 cost exceeds the $60 planned window.")
    if HARD_CAP_USD - conditional_cost < MINIMUM_RESERVE_USD:
        blockers.append("Conditional H200 cost leaves less than the $20 reserve.")
    if snapshot.billing_increment_seconds is None:
        blockers.append("Billing granularity is unknown.")
    if not snapshot.billing_rounding_rule_verified:
        blockers.append("Monetary billing rounding is unverified.")
    if not snapshot.credits_and_tax_treatment_verified:
        blockers.append("Credit and tax treatment is unverified.")
    if not snapshot.stop_terminate_semantics_verified:
        blockers.append("Stop/terminate billing semantics are unverified.")
    if not snapshot.persistent_disk_billing_verified:
        blockers.append("Persistent-disk pricing is unverified.")
    if not snapshot.renter_scheduled_termination_supported:
        blockers.append("No reliable renter scheduled termination/TTL is published.")
    if not snapshot.access_configured_locally:
        blockers.append("No CloudRift CLI/profile or SSH access is configured locally.")
    if not snapshot.official_recipe_verified:
        blockers.append("The exact tested CloudRift GLM serving recipe is unverified.")
    if snapshot.framework_version is None:
        blockers.append("The exact vLLM version is not pinned.")
    if snapshot.image_reference is None:
        blockers.append("The serving image is not digest pinned.")
    else:
        parse_image_reference(snapshot.image_reference)

    return CloudRiftPlan(
        snapshot=snapshot,
        inventory=inventory,
        blockers=tuple(blockers),
        conditional_h200_cost_usd=conditional_cost,
    )
