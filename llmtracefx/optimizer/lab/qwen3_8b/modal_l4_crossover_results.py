"""Provider-specific result analysis for a completed Modal L4 crossover run.

The public CloudRift results builder is bound to the CloudRift authorization,
lifecycle ledger, and host page-cache receipts, none of which a Modal run can
produce, so delegating a whole Modal workspace to it can never succeed. This
module is the real Modal path instead. It consumes what a Modal run actually
returns -- the orchestration receipt and the thirty-two sealed inner cell
receipts -- validates the Modal envelope (schedule, lane, mode and pair
coverage, one attempt per lifecycle, L4 hardware and runtime-pin and
nonce-bound commitment continuity from both canaries through every cell,
per-cell cache scope, terminal shape, teardown, and budget), and then reuses
the *provider-neutral* statistical primitives from the CloudRift results core
over the inner receipts. It never imports the provider SDK, never fabricates a
CloudRift authorization or host-cache receipt, and never claims host-cache or
causal-compilation control: those claims stay unsupported by construction,
while the fixed-token-count, provider-conditioned paired result is eligible
only when every gate passes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from . import cloudrift_crossover_results as stats
from . import cloudrift_runner as base_runner
from . import modal_l4_rates as rates_module
from . import vllm_compile as core
from .modal_l4_crossover import (
    BLOCKED_CLAIM_IDS,
    EXPECTED_GPU_NAME,
    GPU_COUNT,
    HARD_CAP_USD,
    LIFECYCLE_BY_ID,
    PROTOCOL_ID,
    RUNTIME_IMAGE_SPEC_COMMITMENT,
    TOTAL_PLANNED_USD,
    UNCONTROLLED_CACHE_LIMITATIONS,
    build_default_plan,
    call_sequence,
    crossover_schedule,
    evaluate_attempt_receipts,
    evaluate_memory_gate,
    evaluate_teardown_receipt,
    runtime_image_identity,
    verify_ledger_document,
    verify_official_rate_receipt,
    verify_profile_authentication,
)
from .vllm_compile import PROTOCOL_ID as BASE_PROTOCOL_ID
from .vllm_compile import RUNTIME_PINS, canonical_decimal

RESULT_SCHEMA_VERSION = "1"
ORCHESTRATION_SCHEMA_VERSION = "1"
REUSED_STATISTICAL_PRIMITIVES = (
    "cloudrift_crossover_results._validate_request",
    "cloudrift_crossover_results._compute_pair_effects",
    "cloudrift_crossover_results._identity_summary",
    "cloudrift_crossover_results._analysis_document",
    "cloudrift_crossover_results._natural_evaluation",
    "cloudrift_crossover_results._quality_preservation",
    "vllm_compile.PairCurve",
    "vllm_compile.analyze_pair_curves",
)
# Claims a fresh-container, provider-conditioned crossover cannot make on Modal:
# the host page cache and container placement are chosen by the provider and are
# never observable, so no causal, cache-controlled, or hardware-matched claim is
# available. These stay unsupported by construction and must never be marked
# supported.
_UNSUPPORTED_BY_CONSTRUCTION = {
    "pure-causal-compilation-effect": "host_page_cache_and_placement_uncontrolled",
    "natural-end-to-end-causal-speedup": "host_page_cache_and_placement_uncontrolled",
    "cache-state-controlled-comparison": "host_page_cache_not_observable",
    "hardware-matched-comparison": "container_placement_uncontrolled_across_cells",
    "compile-cuda-graph-component-timing": "no_stable_offline_snapshot_hook",
}


class ModalL4ResultsError(ValueError):
    """Raised when a completed Modal L4 run cannot be validated or analyzed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ModalL4ResultsError(message)


def _require_exact_keys(
    obj: Mapping[str, Any], allowed: Sequence[str], *, label: str
) -> None:
    """Refuse any key outside the closed allowlist, and any missing key.

    A sealed object's hash already covers every field, but a producer can
    re-seal, so an exact closed key set is the real defense against an extra
    field smuggled past the seal after it is recomputed.
    """

    allowed_set = set(allowed)
    keys = set(obj)
    extra = sorted(keys - allowed_set)
    _require(
        not extra,
        f"{label} carries fields outside its closed schema: " + ", ".join(extra),
    )
    missing = sorted(allowed_set - keys)
    _require(not missing, f"{label} is missing required fields: " + ", ".join(missing))


# Exact closed key sets. A run whose sealed objects carry any field outside
# these -- even a re-sealed one -- is refused before any statistic is read.
ORCHESTRATION_KEYS = (
    "schema_version",
    "protocol_id",
    "kind",
    "published",
    "status",
    "failure",
    "plan_sha256",
    "source_head",
    "experiment_nonce",
    "authorization_sha256",
    "run_names",
    "base_image_reference",
    "runtime_image",
    "provider_sdk",
    "profile_authentication",
    "credential_exposure",
    "rate_receipt",
    "rate_refresh",
    "source_checkout",
    "call_sequence_executed",
    "attempt_receipts",
    "attempt_adjudication",
    "memory_gate",
    "completed_cell_ids",
    "ledger",
    "teardown",
    "statistical_publication",
    "uncontrolled_limitations",
    "provider_reported_spend_usd",
    "provider_reported_spend_null_reason",
    "observed_at",
    "orchestration_sha256",
)
CELL_WRAPPER_KEYS = (
    "schema_version",
    "protocol_id",
    "kind",
    "status",
    "cell_id",
    "container_identity_sha256",
    "provider_hardware",
    "runtime_image",
    "cell_receipt",
    "started_at",
    "ended_at",
    "terminal",
    "receipt_sha256",
)
CANARY_RECEIPT_KEYS = (
    "schema_version",
    "protocol_id",
    "kind",
    "status",
    "mode",
    "container_identity_sha256",
    "expected_runtime_pins",
    "hardware_commitment",
    "runtime_image",
    "observation",
    "terminal",
    "receipt_sha256",
)
RATE_REFRESH_KEYS = ("capture", "verification")
RATE_CAPTURE_KEYS = (
    "kind",
    "observed_at",
    "documents",
    "capture_sha256",
    "parsed_from_html",
    "parsing_limitation",
)
SOURCE_CHECKOUT_KEYS = (
    "verified",
    "source_head",
    "tracked_workspace_clean",
    "ignored_untracked_prefix",
)


_GIT_HEAD = re.compile(r"^[0-9a-f]{40}$")
_NONCE = re.compile(r"^[0-9a-f]{32,64}$")


def _verify_orchestration_seal(orchestration: Mapping[str, Any]) -> None:
    """Verify the orchestration receipt's own seal before trusting any field."""

    seal = orchestration.get("orchestration_sha256")
    _require(
        isinstance(seal, str) and seal.startswith("sha256:"),
        "orchestration seal is missing",
    )
    material = {
        key: value
        for key, value in orchestration.items()
        if key != "orchestration_sha256"
    }
    _require(
        base_runner._sha256_json(material) == seal,
        "orchestration seal does not verify",
    )


def _validate_orchestration_envelope(
    orchestration: Mapping[str, Any], *, plan: Any
) -> tuple[str, str]:
    """Validate the immutable orchestration header and return (head, nonce).

    Every published fact is re-derived here rather than trusting the producer's
    ``published`` flag: the seal, the exact schema/protocol identity, the plan
    binding, the source head, and the runtime image commitment. A run whose
    header does not reconcile is refused before any statistic is read.
    """

    _require(
        orchestration.get("schema_version") == ORCHESTRATION_SCHEMA_VERSION,
        "orchestration schema version differs",
    )
    # Exact closed key set: even a re-sealed orchestration cannot carry a field
    # outside the frozen schema.
    _require_exact_keys(orchestration, ORCHESTRATION_KEYS, label="orchestration")
    _require(
        orchestration.get("protocol_id") == PROTOCOL_ID,
        "orchestration protocol identity differs",
    )
    _require(
        str(orchestration.get("kind", "")).endswith(".result"),
        "orchestration is not a result receipt",
    )
    _require(
        orchestration.get("status") == "complete",
        "orchestration status is not complete",
    )
    _require(
        orchestration.get("published") is True,
        "orchestration is not a published result",
    )
    _require(
        orchestration.get("plan_sha256") == plan.content_sha256,
        "orchestration is bound to a different plan",
    )
    source_head = orchestration.get("source_head")
    _require(
        isinstance(source_head, str) and _GIT_HEAD.fullmatch(source_head) is not None,
        "orchestration source head is invalid",
    )
    assert isinstance(source_head, str)
    experiment_nonce = orchestration.get("experiment_nonce")
    _require(
        isinstance(experiment_nonce, str)
        and _NONCE.fullmatch(experiment_nonce) is not None,
        "orchestration experiment nonce is invalid",
    )
    assert isinstance(experiment_nonce, str)
    image = orchestration.get("runtime_image")
    _require(
        isinstance(image, Mapping)
        and image.get("derived_image_spec_commitment") == RUNTIME_IMAGE_SPEC_COMMITMENT
        and image.get("derived_provider_image_digest") is None
        and dict(image) == runtime_image_identity(source_head=source_head),
        "orchestration runtime image commitment differs or claims a digest",
    )
    return source_head, experiment_nonce


def _validate_profile_authentication(
    orchestration: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the embedded local-profile authentication verdict's schema.

    The sanitized boolean-only verdict must match the closed schema exactly:
    the pinned version, the same-interpreter module mechanism, a probed CLI
    version equal to the loaded SDK version, no retained profile identity, and a
    timestamp. It is bound into the analysis so a published result carries the
    proof that a standard local profile was confirmed before any spend.
    """

    profile = orchestration.get("profile_authentication")
    try:
        return verify_profile_authentication(profile)
    except Exception as exc:  # noqa: BLE001 - normalized into a terminal error
        raise ModalL4ResultsError(
            f"orchestration profile authentication is invalid: {exc}"
        ) from exc


def _validate_source_checkout(
    orchestration: Mapping[str, Any], *, source_head: str
) -> dict[str, Any]:
    """Validate the embedded source-checkout receipt and its head binding."""

    receipt = orchestration.get("source_checkout")
    _require(isinstance(receipt, Mapping), "orchestration source checkout is missing")
    assert isinstance(receipt, Mapping)
    _require_exact_keys(receipt, SOURCE_CHECKOUT_KEYS, label="source checkout")
    _require(
        receipt.get("verified") is True
        and receipt.get("tracked_workspace_clean") is True,
        "orchestration source checkout did not verify a clean tracked workspace",
    )
    _require(
        receipt.get("source_head") == source_head,
        "orchestration source checkout head differs from the orchestration head",
    )
    _require(
        isinstance(receipt.get("ignored_untracked_prefix"), str),
        "orchestration source checkout is missing its tolerated untracked prefix",
    )
    return dict(receipt)


def _validate_rate_provenance(orchestration: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the structured rate receipt and its fresh capture provenance.

    The structured receipt is re-adjudicated against the committed rates, then
    the embedded fresh capture+verification is checked to be internally
    consistent (the capture hash recomputes over the hashed documents) and
    bound to that receipt and its document hash -- entirely offline, with no
    network. The capture never claims HTML was parsed; it is provenance for the
    manual structured receipt, and that limitation is preserved.
    """

    receipt = orchestration.get("rate_receipt")
    _require(isinstance(receipt, Mapping), "orchestration rate receipt is missing")
    assert isinstance(receipt, Mapping)
    try:
        verify_official_rate_receipt(receipt)
    except Exception as exc:  # noqa: BLE001 - normalized into a terminal error
        raise ModalL4ResultsError(
            f"orchestration rate receipt does not verify: {exc}"
        ) from exc

    refresh = orchestration.get("rate_refresh")
    _require(isinstance(refresh, Mapping), "orchestration rate refresh is missing")
    assert isinstance(refresh, Mapping)
    _require_exact_keys(refresh, RATE_REFRESH_KEYS, label="rate refresh")
    capture = refresh.get("capture")
    verification = refresh.get("verification")
    _require(
        isinstance(capture, Mapping) and isinstance(verification, Mapping),
        "rate refresh must carry a capture and a verification object",
    )
    assert isinstance(capture, Mapping)
    assert isinstance(verification, Mapping)
    _require_exact_keys(capture, RATE_CAPTURE_KEYS, label="rate capture")
    _require(
        capture.get("parsed_from_html") is False,
        "rate capture must never claim the pricing page markup was parsed",
    )
    documents = capture.get("documents")
    _require(
        isinstance(documents, Sequence)
        and not isinstance(documents, (str, bytes))
        and bool(documents),
        "rate capture is missing its hashed official documents",
    )
    assert isinstance(documents, Sequence)
    # Recompute the capture hash over the hashed documents so a tampered
    # capture (a swapped document hash or URL) cannot pass as fresh provenance.
    # The serialization mirrors ``capture_rate_documents`` exactly so the hash
    # is byte-identical to the one a fresh capture produced.
    expected_capture_sha = rates_module._sha256_uri(
        json.dumps(
            list(documents),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )
    _require(
        capture.get("capture_sha256") == expected_capture_sha,
        "rate capture hash does not recompute over its hashed documents",
    )
    # Re-derive the verification from the receipt and the capture and require
    # byte-equality, binding the receipt to the freshly captured document hash.
    try:
        recomputed = rates_module.verify_rate_refresh(receipt, capture=capture)
    except Exception as exc:  # noqa: BLE001 - normalized into a terminal error
        raise ModalL4ResultsError(
            f"rate refresh verification does not bind to the receipt: {exc}"
        ) from exc
    _require(
        dict(verification) == recomputed,
        "rate refresh verification does not recompute from the receipt and capture",
    )
    _require(
        verification.get("document_sha256") == receipt.get("document_sha256"),
        "rate refresh verification is not bound to the structured receipt document",
    )
    return {"rate_receipt": dict(receipt), "rate_refresh": dict(refresh)}


def _validate_call_sequence(orchestration: Mapping[str, Any]) -> None:
    """Prove the exact sealed call sequence with one terminal attempt each."""

    executed = orchestration.get("call_sequence_executed")
    steps = list(call_sequence())
    _require(
        isinstance(executed, Sequence)
        and not isinstance(executed, (str, bytes))
        and len(executed) == len(steps),
        "orchestration call sequence length differs from the sealed plan",
    )
    assert isinstance(executed, Sequence)
    for item, step in zip(executed, steps, strict=True):
        _require(
            isinstance(item, Mapping)
            and item.get("lifecycle_id") == step["lifecycle_id"]
            and item.get("attempt") == 1
            and item.get("terminal_receipt") is True,
            "orchestration call sequence differs from the sealed order or shows "
            "a non-first or non-terminal attempt",
        )
    receipts = orchestration.get("attempt_receipts")
    _require(
        isinstance(receipts, Sequence) and not isinstance(receipts, (str, bytes)),
        "orchestration attempt receipts are missing",
    )
    assert isinstance(receipts, Sequence)
    verdict = evaluate_attempt_receipts(list(receipts))
    _require(
        verdict["valid"] is True,
        "orchestration attempt receipts invalidate the run",
    )
    _require(
        orchestration.get("attempt_adjudication") == verdict,
        "orchestration attempt adjudication does not recompute",
    )
    lifecycles = {str(item["lifecycle_id"]) for item in receipts}
    _require(
        lifecycles == set(LIFECYCLE_BY_ID),
        "orchestration did not record exactly one attempt for every lifecycle",
    )


def _validate_memory_canaries(
    orchestration: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Validate both canary receipts: seal, mode, verdict, and no tuning.

    Returns the two sealed canary receipts for the placement check. Every
    canary is re-adjudicated from its own observation rather than trusting the
    recorded verdict, and a memory gate that reports any tuning is refused.
    """

    gate = orchestration.get("memory_gate")
    _require(isinstance(gate, Mapping), "orchestration memory gate is missing")
    assert isinstance(gate, Mapping)
    _require(
        gate.get("tuning_applied") is False,
        "orchestration memory gate reports tuning; results are refused",
    )
    canaries = gate.get("canaries")
    _require(
        isinstance(canaries, Sequence)
        and not isinstance(canaries, (str, bytes))
        and len(canaries) == 2,
        "orchestration must record both memory-gate canaries",
    )
    assert isinstance(canaries, Sequence)
    receipts: list[dict[str, Any]] = []
    modes: set[str] = set()
    for entry in canaries:
        _require(isinstance(entry, Mapping), "memory-gate entry must be an object")
        assert isinstance(entry, Mapping)
        receipt = entry.get("receipt")
        _require(
            isinstance(receipt, Mapping),
            "memory-gate entry must carry its sealed canary receipt",
        )
        assert isinstance(receipt, Mapping)
        receipt = dict(receipt)
        _require_exact_keys(receipt, CANARY_RECEIPT_KEYS, label="memory-gate canary")
        try:
            base_runner._verify_seal(receipt, "receipt_sha256")
        except Exception as exc:  # noqa: BLE001 - normalized into a terminal error
            raise ModalL4ResultsError(
                "memory-gate canary receipt seal does not verify"
            ) from exc
        for field, expected in (
            ("protocol_id", PROTOCOL_ID),
            ("kind", "modal_canary"),
            ("status", "completed"),
            ("terminal", True),
        ):
            _require(
                receipt.get(field) == expected,
                f"memory-gate canary {field} differs from the sealed contract",
            )
        observation = receipt.get("observation")
        _require(
            isinstance(observation, Mapping),
            "memory-gate canary observation is missing",
        )
        verdict = evaluate_memory_gate(observation)
        _require(
            verdict["passed"] is True,
            f"memory-gate {verdict['mode']} canary did not pass",
        )
        _require(
            receipt.get("mode") == verdict["mode"],
            "memory-gate canary mode differs from its observation",
        )
        recorded = {key: value for key, value in entry.items() if key != "receipt"}
        _require(
            recorded == verdict,
            "memory-gate canary recorded verdict does not recompute",
        )
        modes.add(verdict["mode"])
        receipts.append(receipt)
    _require(modes == {"eager", "compiled"}, "memory gate is missing a canary mode")
    return receipts


def _validate_ledger(
    orchestration: Mapping[str, Any],
    *,
    plan: Any,
    source_head: str,
    experiment_nonce: str,
) -> None:
    """Comprehensively validate the embedded application ledger.

    The trusted ledger validator (seal, event hash-chain, per-lifecycle state
    machine, and reconciled totals) is reused rather than a shallow reserved
    check. Every planned lifecycle must be completed, reservations must equal
    the exact planned envelope, and the total must remain within the hard cap.
    """

    ledger = orchestration.get("ledger")
    _require(isinstance(ledger, Mapping), "orchestration ledger is missing")
    assert isinstance(ledger, Mapping)
    try:
        summary = verify_ledger_document(
            ledger,
            plan=plan,
            source_head=source_head,
            experiment_nonce=experiment_nonce,
        )
    except Exception as exc:  # noqa: BLE001 - normalized into a terminal error
        raise ModalL4ResultsError(
            f"orchestration ledger does not verify: {exc}"
        ) from exc
    entries = summary["entries"]
    _require(
        len(entries) == len(plan.lifecycles)
        and all(entry["status"] == "completed" for entry in entries),
        "orchestration ledger did not complete every planned lifecycle",
    )
    reserved = Decimal(summary["reserved_usd"])
    _require(
        reserved <= HARD_CAP_USD,
        "orchestration ledger reservations exceed the hard cap",
    )
    _require(
        summary["reserved_usd"] == canonical_decimal(TOTAL_PLANNED_USD),
        "orchestration ledger total differs from the exact planned envelope",
    )


def _validate_teardown(orchestration: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the teardown receipt and its self-consistent adjudication."""

    teardown = orchestration.get("teardown")
    _require(isinstance(teardown, Mapping), "orchestration teardown is missing")
    assert isinstance(teardown, Mapping)
    adjudication = teardown.get("adjudication")
    _require(
        isinstance(adjudication, Mapping),
        "orchestration teardown adjudication is missing",
    )
    assert isinstance(adjudication, Mapping)
    receipt = {key: value for key, value in teardown.items() if key != "adjudication"}
    recomputed = evaluate_teardown_receipt(receipt)
    _require(
        recomputed["complete"] is True,
        "orchestration teardown is incomplete: " + ", ".join(recomputed["failures"]),
    )
    _require(
        dict(adjudication) == recomputed,
        "orchestration teardown adjudication does not recompute",
    )
    return dict(teardown)


def _cell_wrapper(cells: Mapping[str, Any], cell_id: str) -> dict[str, Any]:
    wrapper = cells.get(cell_id)
    _require(
        isinstance(wrapper, Mapping),
        f"cell receipt for {cell_id!r} is missing",
    )
    assert isinstance(wrapper, Mapping)
    wrapper = dict(wrapper)
    _require_exact_keys(wrapper, CELL_WRAPPER_KEYS, label=f"cell {cell_id!r} wrapper")
    try:
        base_runner._verify_seal(wrapper, "receipt_sha256")
    except Exception as exc:  # noqa: BLE001 - normalized into a terminal error
        raise ModalL4ResultsError(
            f"cell {cell_id!r} wrapper seal does not verify"
        ) from exc
    for field, expected in (
        ("protocol_id", PROTOCOL_ID),
        ("kind", "modal_cell"),
        ("status", "completed"),
        ("cell_id", cell_id),
        ("terminal", True),
    ):
        _require(
            wrapper.get(field) == expected,
            f"cell {cell_id!r} wrapper {field} differs from the sealed contract",
        )
    return wrapper


def _validate_wrapper_hardware(cell_id: str, wrapper: Mapping[str, Any]) -> None:
    hardware = wrapper.get("provider_hardware")
    _require(
        isinstance(hardware, Mapping),
        f"cell {cell_id!r} is missing provider hardware",
    )
    assert isinstance(hardware, Mapping)
    _require(
        hardware.get("gpu_name") == EXPECTED_GPU_NAME
        and hardware.get("gpu_count") == GPU_COUNT,
        f"cell {cell_id!r} did not run on the approved L4",
    )
    _require(
        isinstance(hardware.get("driver_version"), str)
        and hardware.get("driver_pinned") is False,
        f"cell {cell_id!r} driver must be recorded and never pinned",
    )
    _require(
        "gpu_uuid_sha256" not in hardware,
        f"cell {cell_id!r} leaked a raw GPU UUID derivative",
    )
    image = wrapper.get("runtime_image")
    _require(
        isinstance(image, Mapping)
        and image.get("derived_image_spec_commitment") == RUNTIME_IMAGE_SPEC_COMMITMENT
        and image.get("derived_provider_image_digest") is None,
        f"cell {cell_id!r} runtime image commitment differs or claims a digest",
    )


def _validate_hardware_placement(
    *,
    experiment_nonce: str,
    canary_receipts: Sequence[Mapping[str, Any]],
    inner_by_cell: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Report nonce-bound GPU commitments and provider-chosen placement.

    This is deliberately not called "continuity": Modal chooses container
    placement, so cells may run on the same or different physical L4s, and that
    is observed, not asserted away. Each commitment's shape and nonce binding
    are verified; differing GPU identities do not invalidate the
    provider-conditioned paired result, but they do block any hardware-matched
    or causal claim.

    Only aggregate placement facts are *published*: the distinct-commitment
    count, whether a single placement was shared, and the driver versions. The
    per-cell placement-group index is computed here but returned separately for
    internal per-pair same/different derivation only -- it is never published,
    because a stable per-cell group label across every published cell could act
    as a derived provider/hardware identifier pattern. The raw GPU identity
    commitment and any hash of it are never emitted.
    """

    def _commitment(source: str, receipt: Mapping[str, Any]) -> dict[str, Any]:
        commitment = receipt.get("hardware_commitment")
        _require(
            isinstance(commitment, Mapping),
            f"{source} is missing its nonce-bound hardware commitment",
        )
        assert isinstance(commitment, Mapping)
        _require(
            commitment.get("public_experiment_nonce") == experiment_nonce,
            f"{source} hardware commitment is bound to a different experiment",
        )
        _require(
            commitment.get("gpu_name") == EXPECTED_GPU_NAME
            and commitment.get("gpu_count") == GPU_COUNT,
            f"{source} hardware commitment is not the approved L4",
        )
        identity = commitment.get("gpu_identity_commitment")
        _require(
            isinstance(commitment.get("driver_version"), str)
            and isinstance(identity, str)
            and identity.startswith("sha256:"),
            f"{source} hardware commitment is incomplete",
        )
        _require(
            "gpu_uuid_sha256" not in commitment,
            f"{source} hardware commitment leaked a raw GPU UUID derivative",
        )
        return dict(commitment)

    drivers: set[str] = set()
    # Anonymised placement grouping: each distinct nonce-bound identity
    # commitment is assigned a small integer in first-seen order so per-pair
    # same/different placement can be derived without persisting the commitment
    # value itself. This mapping is internal only and never published.
    group_index: dict[str, int] = {}

    def _placement(identity: str) -> int:
        return group_index.setdefault(identity, len(group_index))

    for index, canary in enumerate(canary_receipts, start=1):
        commitment = _commitment(f"canary {index}", canary)
        drivers.add(str(commitment["driver_version"]))
        _placement(str(commitment["gpu_identity_commitment"]))
        pins = canary.get("expected_runtime_pins")
        _require(
            isinstance(pins, Mapping) and dict(pins) == dict(RUNTIME_PINS),
            f"canary {index} did not attest the required runtime pins",
        )
    placement_group_by_cell: dict[str, int] = {}
    for cell_id, inner in inner_by_cell.items():
        commitment = _commitment(f"cell {cell_id}", inner)
        drivers.add(str(commitment["driver_version"]))
        placement_group_by_cell[cell_id] = _placement(
            str(commitment["gpu_identity_commitment"])
        )
        runtime = inner.get("runtime")
        _require(
            isinstance(runtime, Mapping)
            and dict(runtime.get("pins", {})) == dict(RUNTIME_PINS)
            and dict(runtime.get("expected_pins", {})) == dict(RUNTIME_PINS),
            f"cell {cell_id} did not attest the required runtime pins",
        )
    published = {
        "experiment_nonce": experiment_nonce,
        "canaries": len(canary_receipts),
        "cells": len(inner_by_cell),
        "observed_driver_versions": sorted(drivers),
        "distinct_gpu_identity_commitments": len(group_index),
        "single_shared_placement": len(group_index) == 1,
        "placement_controlled": False,
        "raw_gpu_identity_exposed": False,
        "per_cell_placement_group_published": False,
        "hardware_matched_or_causal_claims_supported": False,
        "placement_note": (
            "modal chooses container placement; provider-conditioned paired "
            "results hold across placements, hardware-matched and causal claims "
            "do not. only aggregate placement and per-pair same/different "
            "booleans are published, never a stable per-cell group label"
        ),
    }
    return published, placement_group_by_cell


def _validate_cache_scope(cell_id: str, inner: Mapping[str, Any]) -> str:
    environment = inner.get("deterministic_environment")
    _require(
        isinstance(environment, Mapping),
        f"cell {cell_id} is missing its deterministic environment",
    )
    assert isinstance(environment, Mapping)
    roles = environment.get("cache_roles")
    _require(
        isinstance(roles, Mapping)
        and set(roles)
        == {"vllm", "torchinductor", "triton", "cuda", "home", "huggingface", "xdg"},
        f"cell {cell_id} cache roles differ from the sealed per-cell scope",
    )
    cache_root = environment.get("cache_root_role")
    _require(
        isinstance(cache_root, Mapping)
        and cache_root.get("relative_identity") == cell_id,
        f"cell {cell_id} cache root is not scoped to the cell",
    )
    assert isinstance(cache_root, Mapping)
    return str(cache_root.get("path_sha256"))


def _validate_inner_cell(cell: Any, inner: Mapping[str, Any], plan: Any) -> None:
    """Validate the sealed inner CloudRift cell payload's fixed identity.

    The heavy per-request validation is reused from the CloudRift results core;
    this only checks the seal and the sealed-identity fields so the inner
    payload cannot be swapped for another cell, plan, or protocol.
    """

    stats._verify_seal(inner, "cell_sha256")
    _require(
        inner.get("protocol_id") == BASE_PROTOCOL_ID,
        f"cell {cell.cell_id} inner receipt protocol identity differs",
    )
    _require(
        inner.get("plan_sha256") == plan.content_sha256,
        f"cell {cell.cell_id} inner receipt is bound to a different plan",
    )
    _require(
        inner.get("cell") == cell.to_dict(),
        f"cell {cell.cell_id} inner receipt schedule cell differs",
    )
    _require(
        inner.get("terminal") is True
        and inner.get("request_count_expected") == cell.requests_per_cell
        and inner.get("request_count_observed") == cell.requests_per_cell,
        f"cell {cell.cell_id} inner receipt request accounting differs",
    )


def _extract_public_cell(
    cell: Any, inner: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return the public cell and requests, reusing the neutral request validator.

    ``cloudrift_crossover_results._validate_cell`` cannot be reused directly
    because it hard-codes the CloudRift RTX 4090 hardware commitment, which an
    L4 run can never satisfy. Its genuinely provider-neutral pieces are reused
    instead: ``_validate_request`` for every request (the bulk of the work) and,
    downstream, ``_compute_pair_effects`` and ``_pair_effect_distributions``.
    The only cell-level work here is lifting the already-sealed measurement
    dictionaries, not recomputing any statistic.
    """

    descriptors = core.lane_request_descriptors(cell.lane)
    requests = inner.get("requests")
    _require(
        isinstance(requests, Sequence)
        and not isinstance(requests, (str, bytes))
        and len(requests) == len(descriptors),
        f"cell {cell.cell_id} request count differs from the sealed lane",
    )
    assert isinstance(requests, Sequence)
    public_requests = [
        stats._validate_request(request, cell=cell, descriptor=descriptor, index=index)
        for index, (descriptor, request) in enumerate(
            zip(descriptors, requests, strict=True), start=1
        )
    ]
    measurements = inner.get("measurements")
    _require(
        isinstance(measurements, Mapping)
        and isinstance(measurements.get("initialization_seconds"), Mapping)
        and isinstance(measurements.get("peak_gpu_memory_mib"), Mapping),
        f"cell {cell.cell_id} measurements are missing",
    )
    assert isinstance(measurements, Mapping)
    initialization = dict(measurements["initialization_seconds"])
    _require(
        isinstance(initialization.get("value"), (int, float))
        and not isinstance(initialization.get("value"), bool),
        f"cell {cell.cell_id} initialization measurement is not observed",
    )
    curve = [
        request["cumulative_from_initialization_seconds"] for request in public_requests
    ]
    _require(
        bool(curve) and all(isinstance(value, (int, float)) for value in curve),
        f"cell {cell.cell_id} cumulative curve is not observed",
    )
    host_lifecycle_ns = int(
        requests[-1]["timing"]["cumulative_from_initialization_perf_counter_ns"]
    )
    public = {
        "cell_id": cell.cell_id,
        "mode": cell.mode,
        "cumulative_seconds": curve,
        "measurements": {
            "initialization_seconds": initialization,
            "host_lifecycle_seconds": {
                "value": host_lifecycle_ns / 1_000_000_000,
                "unit": "seconds",
                "clock_domain": "container_perf_counter",
                "provenance": "inner_cell_cumulative_perf_counter_ns",
                "observability_state": "observed",
                "null_reason": None,
            },
            "peak_gpu_memory_mib": dict(measurements["peak_gpu_memory_mib"]),
        },
    }
    return public, public_requests


def analyze_modal_run(
    *,
    orchestration: Mapping[str, Any],
    cells: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a completed Modal L4 run and analyze it with reused primitives.

    ``orchestration`` is the sealed orchestration receipt and ``cells`` maps
    each sealed cell id to its Modal cell wrapper receipt. Every failure is
    terminal and nothing is taken on trust: the orchestration seal, the header
    bindings, the exact sealed call sequence, both memory canaries, the
    application ledger, the teardown, the cell inventory, and every wrapper and
    inner seal are re-derived here. The producer's ``published`` flag is never
    relied upon. No CloudRift authorization or host-cache receipt is required or
    produced.
    """

    _require(isinstance(orchestration, Mapping), "orchestration must be an object")
    _verify_orchestration_seal(orchestration)
    modal_plan = build_default_plan()
    core_plan = core.build_default_plan()
    source_head, experiment_nonce = _validate_orchestration_envelope(
        orchestration, plan=modal_plan
    )
    profile_authentication = _validate_profile_authentication(orchestration)
    source_checkout = _validate_source_checkout(orchestration, source_head=source_head)
    rate_provenance = _validate_rate_provenance(orchestration)
    _validate_call_sequence(orchestration)
    canary_receipts = _validate_memory_canaries(orchestration)
    _validate_ledger(
        orchestration,
        plan=modal_plan,
        source_head=source_head,
        experiment_nonce=experiment_nonce,
    )
    teardown = _validate_teardown(orchestration)

    schedule = list(crossover_schedule())
    _require(len(schedule) == 32, "sealed schedule must contain 32 cells")
    _require(
        isinstance(cells, Mapping)
        and set(cells) == {cell.cell_id for cell in schedule},
        "cells mapping differs from the sealed 32-cell schedule with no extras",
    )
    completed = orchestration.get("completed_cell_ids")
    _require(
        isinstance(completed, Sequence)
        and not isinstance(completed, (str, bytes))
        and sorted(completed) == sorted(cell.cell_id for cell in schedule),
        "orchestration completed cells differ from the sealed schedule",
    )

    # Validate every wrapper (seal, identity), extract the inner CloudRift cell
    # receipt, and run the reused per-cell statistical validation over it.
    inner_by_cell: dict[str, dict[str, Any]] = {}
    public_by_cell: dict[str, dict[str, Any]] = {}
    requests_by_cell: dict[str, list[dict[str, Any]]] = {}
    container_identities: dict[str, str] = {}
    cache_scopes: dict[str, str] = {}
    for cell in schedule:
        wrapper = _cell_wrapper(cells, cell.cell_id)
        _validate_wrapper_hardware(cell.cell_id, wrapper)
        identity = wrapper.get("container_identity_sha256")
        _require(
            bool(isinstance(identity, str) and identity),
            f"cell {cell.cell_id} is missing a container identity",
        )
        assert isinstance(identity, str)
        _require(
            identity not in container_identities.values(),
            f"cell {cell.cell_id} reused a container from another cell",
        )
        container_identities[cell.cell_id] = identity
        inner = wrapper.get("cell_receipt")
        _require(
            isinstance(inner, Mapping),
            f"cell {cell.cell_id} is missing its inner cell receipt",
        )
        assert isinstance(inner, Mapping)
        inner = dict(inner)
        _validate_inner_cell(cell, inner, core_plan)
        inner_by_cell[cell.cell_id] = inner
        cache_scopes[cell.cell_id] = _validate_cache_scope(cell.cell_id, inner)
        public, public_requests = _extract_public_cell(cell, inner)
        public_by_cell[cell.cell_id] = public
        requests_by_cell[cell.cell_id] = public_requests

    # Per-cell cache directories must be unique: a fresh single-use container is
    # only meaningful if it did not share a cell's cache root.
    _require(
        len(set(cache_scopes.values())) == len(cache_scopes),
        "two cells shared a cache directory scope",
    )

    placement, placement_group_by_cell = _validate_hardware_placement(
        experiment_nonce=experiment_nonce,
        canary_receipts=canary_receipts,
        inner_by_cell=inner_by_cell,
    )

    inference = _crossover_inference(
        schedule, public_by_cell, requests_by_cell, placement_group_by_cell
    )
    identities = {
        lane: stats._identity_summary(schedule, requests_by_cell, lane)
        for lane in core.LANES
    }
    correctness = _correctness(core_plan, schedule, requests_by_cell)
    claim_matrix = _claim_matrix(
        schedule=schedule,
        requests_by_cell=requests_by_cell,
        inference=inference,
        identities=identities,
        correctness=correctness,
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "kind": "llmtracefx.modal_l4_crossover.analysis",
        "experiment_nonce": experiment_nonce,
        "source_head": source_head,
        "plan_sha256": core_plan.content_sha256,
        "modal_plan_sha256": modal_plan.content_sha256,
        "runtime_image": runtime_image_identity(source_head=source_head),
        "reused_statistical_primitives": list(REUSED_STATISTICAL_PRIMITIVES),
        "profile_authentication": profile_authentication,
        "source_checkout": source_checkout,
        "rate_receipt": rate_provenance["rate_receipt"],
        "rate_refresh": rate_provenance["rate_refresh"],
        "hardware_placement": placement,
        "cell_count": len(schedule),
        "pair_count": inference["pair_count"],
        "crossover_inference": inference["analysis"],
        "pair_records": inference["pair_records"],
        "output_identity": identities,
        "correctness": correctness,
        "teardown_provider_reported_spend_usd": teardown.get(
            "provider_reported_spend_usd"
        ),
        "claim_matrix": claim_matrix,
        "uncontrolled_limitations": list(UNCONTROLLED_CACHE_LIMITATIONS),
        "claims_cloudrift_or_host_cache_proof": False,
    }


def _crossover_inference(
    schedule: Sequence[Any],
    public_by_cell: Mapping[str, Mapping[str, Any]],
    requests_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    placement_group_by_cell: Mapping[str, int],
) -> dict[str, Any]:
    """Build the 16 pair records and the reused crossover inference.

    Mirrors the CloudRift results core exactly: eight controlled ``PairCurve``
    objects feed ``analyze_pair_curves`` (observed first and sustained pair
    crossings, right-censoring, the 20,000 whole-pair bootstrap, the
    simultaneous confidence band crossing, and the terminal-effect sign-flip
    p-value), while the natural lane contributes whole-pair terminal effects for
    the natural-timing bootstrap. No request-level resampling and no headline
    extrapolation are performed. ``placement_group_by_cell`` is used only to
    derive the per-pair same/different-placement boolean; the group indices
    themselves are never published.
    """

    placement_group = placement_group_by_cell
    natural_identity = stats._identity_summary(schedule, requests_by_cell, "natural")
    pair_records: list[dict[str, Any]] = []
    pair_curves: list[core.PairCurve] = []
    natural_terminal_effects: list[float] = []
    for lane in core.LANES:
        for pair_index in range(1, core.PAIRS_PER_LANE + 1):
            pair_cells = [
                cell
                for cell in schedule
                if cell.lane == lane and cell.pair_index == pair_index
            ]
            _require(
                len(pair_cells) == 2
                and abs(schedule.index(pair_cells[0]) - schedule.index(pair_cells[1]))
                == 1
                and {cell.mode for cell in pair_cells} == {"eager", "compiled"},
                f"pair {lane}/{pair_index} is not one adjacent eager/compiled pair",
            )
            by_mode = {cell.mode: cell for cell in pair_cells}
            eager_id = by_mode["eager"].cell_id
            compiled_id = by_mode["compiled"].cell_id
            eager = public_by_cell[eager_id]
            compiled = public_by_cell[compiled_id]
            difference = [
                compiled_value - eager_value
                for eager_value, compiled_value in zip(
                    eager["cumulative_seconds"],
                    compiled["cumulative_seconds"],
                    strict=True,
                )
            ]
            full_record = {
                "pair_id": pair_cells[0].pair_id,
                "pair_index": pair_index,
                "lane": lane,
                "order": pair_cells[0].order,
                "period_indices": [cell.period_index for cell in pair_cells],
                "cell_ids_in_execution_order": [cell.cell_id for cell in pair_cells],
                "eager": eager,
                "compiled": compiled,
                "compiled_minus_eager_seconds": difference,
                "pair_effects": stats._compute_pair_effects(
                    eager,
                    compiled,
                    requests_by_cell[eager_id],
                    requests_by_cell[compiled_id],
                ),
            }
            pair_records.append(full_record)
            if lane == "controlled":
                pair_curves.append(
                    core.PairCurve(
                        pair_id=pair_cells[0].pair_id,
                        order=pair_cells[0].order,
                        eager_cumulative=tuple(eager["cumulative_seconds"]),
                        compiled_cumulative=tuple(compiled["cumulative_seconds"]),
                    )
                )
            else:
                natural_terminal_effects.append(difference[-1])
    _require(len(pair_records) == 16, "the schedule must yield exactly 16 pairs")
    _require(
        len(pair_curves) == core.PAIRS_PER_LANE,
        "the controlled lane must yield exactly eight pair curves",
    )
    analysis = stats._analysis_document(
        pair_curves,
        natural_identity=natural_identity["all_corresponding_outputs_identical"],
        natural_terminal_effects=natural_terminal_effects,
        pair_records=pair_records,
    )
    # Emit lean pair records (no duplicated per-request curves), keeping the
    # order/block/period/lane structure and the whole-pair effects.
    lean_records = [
        {
            "pair_id": record["pair_id"],
            "pair_index": record["pair_index"],
            "block_index": record["pair_index"],
            "lane": record["lane"],
            "order": record["order"],
            "period_indices": record["period_indices"],
            "cell_ids_in_execution_order": record["cell_ids_in_execution_order"],
            "same_placement": (
                placement_group[record["cell_ids_in_execution_order"][0]]
                == placement_group[record["cell_ids_in_execution_order"][1]]
            ),
            "pair_effects": record["pair_effects"],
        }
        for record in pair_records
    ]
    return {
        "pair_count": len(pair_records),
        "pair_records": lean_records,
        "analysis": analysis,
        "natural_identity": natural_identity,
    }


def _correctness(
    core_plan: Any,
    schedule: Sequence[Any],
    requests_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Evaluate natural-lane correctness with the pinned natural evaluator.

    Reuses ``_natural_evaluation`` (the version-pinned ``evaluate_workload``)
    per natural request and the whole-pair ``_quality_preservation`` bootstrap,
    so a correctness or quality claim rests on the pinned evaluator and
    lifecycle-pair inference rather than on an unpinned assertion.
    """

    natural_evaluations: list[dict[str, Any]] = []
    for cell in schedule:
        if cell.lane == "natural":
            for request in requests_by_cell[cell.cell_id]:
                natural_evaluations.append(stats._natural_evaluation(request))
    all_natural_correct = bool(natural_evaluations) and all(
        item["success"] for item in natural_evaluations
    )
    quality = stats._quality_preservation(core_plan, natural_evaluations)
    return {
        "evaluator": "evaluate_workload",
        "natural_request_count": len(natural_evaluations),
        "all_natural_requests_correct": all_natural_correct,
        "quality_preservation": quality,
    }


def _controlled_terminals_complete(
    schedule: Sequence[Any],
    requests_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
) -> bool:
    """Return whether every controlled cell produced 144 x 96-token terminals."""

    for cell in schedule:
        if cell.lane != "controlled":
            continue
        requests = requests_by_cell[cell.cell_id]
        if len(requests) != cell.requests_per_cell or any(
            request.get("output_token_count") != 96
            or request.get("finish_reason") != "length"
            for request in requests
        ):
            return False
    return True


def _claim_matrix(
    *,
    schedule: Sequence[Any],
    requests_by_cell: Mapping[str, Sequence[Mapping[str, Any]]],
    inference: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]],
    correctness: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the Modal claim matrix with the corrected claim semantics.

    * A fixed-token-count, provider-conditioned crossover does NOT require
      equal output tokens: it requires complete controlled 144x96 terminals and
      a statistically supported crossing. It does not need output identity, and
      it is independent of output identity in both directions.
    * Output-identical *crossover* requires BOTH cross-mode output identity AND
      the statistically supported fixed-token crossing: identical tokens alone
      is not a crossover, so identity without a crossing does not support it.
    * Numeric reproducibility is a standalone reproducibility claim (within-mode
      lifecycle identity). It does NOT imply a crossover and is deliberately not
      named as one; it is only a crossover claim when a crossing also exists,
      which the output-identical-generation-crossover claim captures.
    * Natural output quality preservation rests on the pinned natural evaluator
      and the whole-pair non-inferiority bootstrap.
    * Pure causal, hardware-matched, host-cache-controlled, and natural causal
      speedup claims stay unsupported by construction: Modal placement and page
      cache are uncontrolled, so natural timing can never support a causal
      speedup even when outputs are identical.
    """

    controlled = inference["analysis"]["controlled"]
    controlled_identity = identities.get("controlled", {})
    terminals_complete = _controlled_terminals_complete(schedule, requests_by_cell)
    supported_crossing = bool(controlled.get("supported_crossing_gate_satisfied"))

    fixed_blockers: list[str] = []
    if not terminals_complete:
        fixed_blockers.append("controlled_144x96_terminals_incomplete")
    if not supported_crossing:
        fixed_blockers.append("no_statistically_supported_controlled_crossing")

    output_identical = controlled_identity.get("cross_mode_pair_outputs_identical") is (
        True
    )
    numeric_reproducible = (
        controlled_identity.get("within_mode_lifecycles_identical") is True
    )
    # Output-identical *crossover* requires both token identity AND a supported
    # crossing: identity alone is not a timing crossover.
    output_identical_crossover_blockers: list[str] = []
    if not output_identical:
        output_identical_crossover_blockers.append("cross_mode_outputs_not_identical")
    if not supported_crossing:
        output_identical_crossover_blockers.append(
            "no_statistically_supported_controlled_crossing"
        )
    quality = correctness["quality_preservation"]
    quality_supported = bool(
        correctness["all_natural_requests_correct"]
        and quality.get("noninferiority_supported") is True
    )

    claims: list[dict[str, Any]] = [
        {
            "claim_id": "fixed-token-count-provider-conditioned-crossover",
            "state": "supported" if not fixed_blockers else "unsupported",
            "blockers": fixed_blockers,
        },
        {
            "claim_id": "output-identical-generation-crossover",
            "state": (
                "supported"
                if not output_identical_crossover_blockers
                else "unsupported"
            ),
            "blockers": output_identical_crossover_blockers,
        },
        {
            "claim_id": "numerically-reproducible-generation",
            "state": "supported" if numeric_reproducible else "unsupported",
            "blockers": (
                [] if numeric_reproducible else ["within_mode_lifecycles_not_identical"]
            ),
        },
        {
            "claim_id": "natural-output-quality-preserved",
            "state": "supported" if quality_supported else "unsupported",
            "blockers": (
                []
                if quality_supported
                else ["pinned_evaluator_noninferiority_not_established"]
            ),
        },
        {
            "claim_id": "budget-reservations-within-hard-cap",
            "state": "supported",
            "blockers": [],
        },
        {
            "claim_id": "provider-reported-spend-within-hard-cap",
            "state": "unsupported",
            "blockers": ["external_provider_end_receipt_absent"],
        },
        {
            "claim_id": "provider-teardown-complete",
            "state": "unsupported",
            "blockers": ["external_provider_fact_not_observed"],
        },
    ]
    claims.extend(
        {
            "claim_id": claim_id,
            "state": "unsupported",
            "blockers": [blocker],
        }
        for claim_id, blocker in sorted(_UNSUPPORTED_BY_CONSTRUCTION.items())
    )
    supported_blocked = sorted(
        claim["claim_id"]
        for claim in claims
        if claim["claim_id"] in BLOCKED_CLAIM_IDS and claim["state"] == "supported"
    )
    _require(
        not supported_blocked,
        "claims unsupported by construction were marked supported: "
        + ", ".join(supported_blocked),
    )
    return {"schema_version": RESULT_SCHEMA_VERSION, "claims": claims}
