"""Builders for a structurally valid completed Modal L4 crossover run.

These construct the exact artifacts a real run returns -- an orchestration
receipt, thirty-two sealed inner cell receipts wrapped in Modal cell receipts,
and the memory-gate canary receipts -- so the provider result path can be
exercised offline. The inner receipts carry only the fields the Modal analyzer
reads (it reuses the provider-neutral ``_validate_request`` per request rather
than the RTX-4090-bound ``_validate_cell``), and they carry an L4 hardware
commitment because that is what a real L4 run produces.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from llmtracefx.optimizer.lab.qwen3_8b import cloudrift_crossover_results as stats
from llmtracefx.optimizer.lab.qwen3_8b import cloudrift_runner as base_runner
from llmtracefx.optimizer.lab.qwen3_8b import modal_l4_crossover as modal
from llmtracefx.optimizer.lab.qwen3_8b import modal_l4_execute as execute
from llmtracefx.optimizer.lab.qwen3_8b import modal_l4_rates as rates_module
from llmtracefx.optimizer.lab.qwen3_8b import vllm_compile as core

NONCE = "d" * 32
SOURCE_HEAD = "e" * 40
# The sealed design is infeasible on a real L4: decode-only weight streaming for
# one controlled cell needs about 755.6s against the sealed 480s timeout, so
# production preflight refuses it offline. A *completed* run therefore only
# exists on a hypothetical device, and these fixtures say so explicitly rather
# than pretending the sealed constants passed. One terabyte per second is far
# above any real L4 and is obviously not a claim about hardware; it exists only
# so the post-feasibility machinery stays covered.
HYPOTHETICAL_PEAK_BANDWIDTH_BYTES_PER_SECOND = 1_000_000_000_000
DRIVER = "570.86"
RESERVED_AT = "2026-09-04T20:00:00+00:00"
COMPLETED_AT = "2026-09-04T20:04:00+00:00"
OBSERVED_AT = "2026-09-04T20:04:00+00:00"
LEDGER_PATH_SHA256 = (
    "sha256:" + hashlib.sha256(b"/run/workspace/application-ledger.json").hexdigest()
)


def _typed(
    value: float | None,
    *,
    unit: str = "seconds",
    clock_domain: str = "same_process_perf_counter",
    provenance: str = "measured_perf_counter_ns",
    state: str | None = None,
    reason: str = "not_exposed",
) -> dict[str, Any]:
    observed = value is not None
    return {
        "value": value,
        "unit": unit,
        "clock_domain": clock_domain,
        "provenance": provenance,
        "observability_state": state or ("observed" if observed else "unobservable"),
        "null_reason": None if observed else reason,
    }


def _l4_commitment(nonce: str = NONCE) -> dict[str, Any]:
    return {
        "gpu_name": modal.EXPECTED_GPU_NAME,
        "gpu_count": 1,
        "driver_version": DRIVER,
        "memory_total_mib": 23_034,
        "memory_used_mib": 1_024,
        "public_experiment_nonce": nonce,
        "gpu_identity_commitment": "sha256:"
        + hashlib.sha256(nonce.encode()).hexdigest(),
    }


def _request(
    cell: core.ScheduleCell,
    descriptor: core.WorkloadDescriptor,
    index: int,
    *,
    cumulative_ns: int,
    latency_ns: int,
) -> dict[str, Any]:
    if cell.lane == "controlled":
        output_ids = [1000 + descriptor.ordinal] * 96
        decoded = None
        finish_reason = "length"
    else:
        output_ids = [2000 + descriptor.ordinal, 3]
        decoded = (
            '{"name":"Priya Nakamura","age":34,"is_active":true}'
            if descriptor.workload_id == "structured-json-profile-extraction"
            else "3 hours because the combined closing speed is 70 mph."
        )
        finish_reason = "stop"
    latency = latency_ns / 1_000_000_000
    base = (index - 1) % 12 + 1
    request: dict[str, Any] = {
        **descriptor.to_dict(),
        "cycle_index": (index - 1) // 12 + 1,
        "base_ordinal": base,
        "request_sequence_index": index,
        "cell_id": cell.cell_id,
        "pair_id": cell.pair_id,
        "lane": cell.lane,
        "mode": cell.mode,
        "input_token_count": 20 + base,
        "input_token_ids_sha256": "sha256:" + f"{base:064x}",
        "output_token_count": len(output_ids),
        "output_token_ids": output_ids,
        "output_token_ids_sha256": core.token_ids_sha256(output_ids),
        "finish_reason": finish_reason,
        "timing": {
            "latency_seconds": _typed(latency),
            "cumulative_from_initialization_seconds": _typed(
                cumulative_ns / 1_000_000_000
            ),
            "latency_perf_counter_ns": latency_ns,
            "cumulative_from_initialization_perf_counter_ns": cumulative_ns,
            "output_token_rate_tokens_per_second": _typed(
                len(output_ids) / latency,
                unit="tokens_per_second",
                provenance="derived_exact_token_count_over_perf_counter_latency",
            ),
        },
        "metrics": {
            name: _typed(
                None,
                clock_domain="request_output_metrics",
                provenance="version_pinned_vllm_0_28_request_state_stats",
                reason=reason,
            )
            for name, reason in {
                "ttft_seconds": "request_state_stats_first_token_latency_unavailable",
                "queue_seconds": "request_state_stats_has_no_queue_duration_field",
                "prefill_seconds": "request_state_stats_has_no_prefill_duration_field",
                "inference_seconds": "request_state_stats_has_no_inference_duration_field",
                "decode_seconds": "request_state_stats_has_no_decode_duration_field",
                "mean_time_per_output_token_seconds": (
                    "request_state_stats_has_no_mean_output_token_duration_field"
                ),
                "e2e_seconds": "request_state_stats_has_no_e2e_duration_field",
            }.items()
        },
        "terminal": True,
    }
    if decoded is not None:
        request["decoded_output"] = decoded
    return request


def _cache_roles() -> dict[str, Any]:
    return {
        role: {
            "env_var": env_var,
            "relative_path": role,
            "path_sha256": "sha256:" + character * 64,
        }
        for role, env_var, character in (
            ("vllm", "VLLM_CACHE_ROOT", "2"),
            ("torchinductor", "TORCHINDUCTOR_CACHE_DIR", "3"),
            ("triton", "TRITON_CACHE_DIR", "4"),
            ("cuda", "CUDA_CACHE_PATH", "5"),
            ("home", "HOME", "6"),
            ("huggingface", "HF_HOME", "7"),
            ("xdg", "XDG_CACHE_HOME", "8"),
        )
    }


def inner_cell_receipt(
    cell: core.ScheduleCell,
    plan: core.VLLMCompilePlan,
    *,
    nonce: str = NONCE,
    flat: bool = False,
) -> dict[str, Any]:
    descriptors = core.lane_request_descriptors(cell.lane)
    # Model a genuine crossover: the compiled lifecycle pays a one-time warmup
    # (higher initialization and a slow first cumulative point) but then decodes
    # faster per request, so the compiled-minus-eager cumulative difference
    # starts positive and crosses to a sustained negative effect. Eager decodes
    # at a steady rate with no warmup. With ``flat`` set, both modes share the
    # eager timing so there is no crossing (still output-identical).
    if cell.mode == "compiled" and not flat:
        warmup_ns = 1_000_000_000
        per_request_ns = 60_000_000
        initialization = 2.0
    else:
        warmup_ns = 0
        per_request_ns = 100_000_000
        initialization = 1.0
    requests = [
        _request(
            cell,
            descriptor,
            index,
            cumulative_ns=warmup_ns + index * per_request_ns,
            latency_ns=per_request_ns,
        )
        for index, descriptor in enumerate(descriptors, start=1)
    ]
    payload: dict[str, Any] = {
        "schema_version": "2",
        "protocol_id": core.PROTOCOL_ID,
        "cell": cell.to_dict(),
        "plan_sha256": plan.content_sha256,
        "runtime": {
            "pins": dict(core.RUNTIME_PINS),
            "expected_pins": dict(core.RUNTIME_PINS),
            "runtime_image": {
                "base_reference": core.BASE_IMAGE_REFERENCE,
                "derived_image_id": core.DERIVED_IMAGE_ID,
            },
        },
        "deterministic_environment": {
            "cache_root_role": {
                "relative_identity": cell.cell_id,
                "path_sha256": "sha256:"
                + hashlib.sha256(cell.cell_id.encode()).hexdigest(),
            },
            "cache_roles": _cache_roles(),
        },
        "hardware_commitment": _l4_commitment(nonce),
        "measurements": {
            "initialization_seconds": _typed(initialization),
            "peak_gpu_memory_mib": _typed(
                2048.0,
                unit="MiB",
                clock_domain="sampled_nvidia_smi",
                provenance="sampled_nvidia_smi",
            ),
        },
        "request_count_expected": cell.requests_per_cell,
        "request_count_observed": cell.requests_per_cell,
        "requests": requests,
        "terminal": True,
    }
    payload["cell_sha256"] = stats._sha256_json(payload)
    return payload


def cell_wrapper(
    cell: core.ScheduleCell,
    plan: core.VLLMCompilePlan,
    *,
    index: int,
    nonce: str = NONCE,
    flat: bool = False,
) -> dict[str, Any]:
    return base_runner._seal(
        {
            "schema_version": "1",
            "protocol_id": modal.PROTOCOL_ID,
            "kind": "modal_cell",
            "status": "completed",
            "cell_id": cell.cell_id,
            "container_identity_sha256": "sha256:" + f"{index:064d}",
            "provider_hardware": {
                "gpu_name": "NVIDIA L4",
                "gpu_count": 1,
                "driver_version": DRIVER,
                "driver_pinned": False,
                "memory_total_mib": 23_034,
                "memory_used_mib": 1_024,
            },
            "runtime_image": modal.runtime_image_identity(),
            "cell_receipt": inner_cell_receipt(cell, plan, nonce=nonce, flat=flat),
            "started_at": "2026-09-04T20:00:00+00:00",
            "ended_at": "2026-09-04T20:04:00+00:00",
            "terminal": True,
        },
        "receipt_sha256",
    )


def _passing_observation(mode: str) -> dict[str, Any]:
    return {
        "mode": mode,
        "gpu_name": modal.EXPECTED_GPU_NAME,
        "gpu_count": 1,
        "runtime_pins": dict(modal.RUNTIME_PINS),
        "total_vram_mib": 23_034,
        "peak_vram_mib": 21_000,
        "kv_cache_blocks": 640,
        "kv_cache_tokens": 40_960,
        "max_model_len": 16_480,
        "out_of_memory": False,
        "generated_tokens": modal.DECODE_STEPS,
        "terminal": True,
        "used_longest_controlled_prompt": True,
        "runner_kwargs": {
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "max_num_seqs": 1,
            "gpu_memory_utilization": "0.94",
            "enable_prefix_caching": False,
            "speculative_config": None,
            "enforce_eager": mode == "eager",
            "max_model_len": 16_480,
        },
    }


def canary_receipt(mode: str, *, index: int, nonce: str = NONCE) -> dict[str, Any]:
    return base_runner._seal(
        {
            "schema_version": "1",
            "protocol_id": modal.PROTOCOL_ID,
            "kind": "modal_canary",
            "status": "completed",
            "mode": mode,
            "container_identity_sha256": "sha256:" + f"c{index:063d}",
            "expected_runtime_pins": dict(modal.RUNTIME_PINS),
            "hardware_commitment": _l4_commitment(nonce),
            "runtime_image": modal.runtime_image_identity(),
            "observation": _passing_observation(mode),
            "terminal": True,
        },
        "receipt_sha256",
    )


def _memory_gate_entry(mode: str, *, index: int, nonce: str = NONCE) -> dict[str, Any]:
    receipt = canary_receipt(mode, index=index, nonce=nonce)
    verdict = modal.evaluate_memory_gate(receipt["observation"])
    return {**verdict, "receipt": receipt}


def hypothetical_feasibility() -> dict[str, Any]:
    """A feasible decode-bandwidth verdict for a hypothetical fast device."""

    return modal.evaluate_decode_bandwidth_feasibility(
        peak_bandwidth_bytes_per_second=HYPOTHETICAL_PEAK_BANDWIDTH_BYTES_PER_SECOND
    )


def hypothetical_feasibility_probe() -> dict[str, Any]:
    """The same verdict, shaped as an injectable offline feasibility probe."""

    return hypothetical_feasibility()


def _headroom() -> dict[str, Any]:
    return {
        "supported": True,
        "headroom_usd": "25",
        "provenance": "signed_operator_receipt",
        "signature_namespace": rates_module.HEADROOM_SIGNATURE_NAMESPACE,
        "is_provider_spend_proof": False,
        "null_reason": None,
        "authorization_binding": {
            "verified": True,
            "bound_to_authorization": True,
            "protocol_id": modal.PROTOCOL_ID,
            "plan_sha256": modal.build_default_plan().content_sha256,
            "source_head": SOURCE_HEAD,
            "experiment_nonce": NONCE,
            "confirmed_at": "2026-09-04T13:00:00+00:00",
            "expires_at": "2026-09-05T00:00:00+00:00",
            "covers_execution_window": True,
            "records_account_identity": False,
        },
    }


def _teardown() -> dict[str, Any]:
    receipt = {
        "outstanding_calls_cancelled": True,
        "app_context_exited": True,
        "app_stop_mechanism": "ephemeral_app_run_context_exit",
        "app_deletion_provider_verified": None,
        "app_deletion_null_reason": modal.UNSUPPORTED_PROVIDER_CONTROLS[
            "explicit_app_stop_method"
        ],
        "functions_scaled_to_zero": True,
        "function_inventory_observability": "control_plane_scale_to_zero_only",
        "scale_zero_verified_via_control_plane": True,
        "scale_zero_settling": {
            "mechanism": modal.SCALE_ZERO_SETTLING_MECHANISM,
            "is_scientific_retry": False,
            "provider_scaledown_window_seconds": modal.SCALEDOWN_WINDOW_SECONDS,
            "poll_interval_seconds": modal.SCALE_ZERO_POLL_INTERVAL_SECONDS,
            "poll_attempts_max": modal.SCALE_ZERO_POLL_ATTEMPTS,
            "poll_timeout_seconds": modal.SCALE_ZERO_POLL_TIMEOUT_SECONDS,
            "samples_taken": 2,
            "functions_observed": 7,
            "functions_settled": 7,
            "settled_after_samples": {},
            "unsettled_functions": 0,
        },
        "container_inventory_observable": False,
        "container_inventory_null_reason": modal.UNSUPPORTED_PROVIDER_CONTROLS[
            "individual_container_deletion"
        ],
        "individual_container_deletion": None,
        "individual_container_deletion_null_reason": (
            modal.UNSUPPORTED_PROVIDER_CONTROLS["individual_container_deletion"]
        ),
        "volume_deleted": True,
        "named_resource_listing_scope": "volumes_only",
        "named_volume_listing_available": True,
        "live_named_volumes": [],
        "run_created_noncredential_secrets_deleted": True,
        "run_created_secret_count": 0,
        "credential_secret_created": False,
        "sanitized_receipts_retained": True,
        "provider_reported_spend_usd": None,
        "teardown_failures": [],
        "observed_at": OBSERVED_AT,
    }
    return {**receipt, "adjudication": modal.evaluate_teardown_receipt(receipt)}


def _provider_sdk_capabilities() -> dict[str, Any]:
    return {
        "verified": True,
        "version": modal.TESTED_MODAL_VERSION,
        "tested_version": modal.TESTED_MODAL_VERSION,
        "unsupported_controls": dict(modal.UNSUPPORTED_PROVIDER_CONTROLS),
    }


def _credential_exposure() -> dict[str, Any]:
    return {
        "gate": "credential_exposure",
        "cleared": True,
        "exposed_profile_credential_never_used_by_experiment": True,
        "exposed_profile_credential_revocation_confirmed": True,
        "fresh_local_profile_created_without_sharing": True,
        "fresh_profile_shared_anywhere": False,
        "confirmed_by": "coordinator",
        "confirmed_at": "2026-09-04T21:02:06.080+05:30",
        "records_credential_values": False,
        "action": "proceed",
    }


def _profile_authentication() -> dict[str, Any]:
    return {
        "schema_version": modal.PROFILE_AUTHENTICATION_SCHEMA_VERSION,
        "gate": modal.PROFILE_AUTHENTICATION_GATE,
        "authenticated": True,
        "mechanism": modal.PROFILE_AUTHENTICATION_MECHANISM,
        "cli_version": modal.TESTED_MODAL_VERSION,
        "sdk_version": modal.TESTED_MODAL_VERSION,
        "records_profile_identity": False,
        "checked_at": OBSERVED_AT,
    }


def _source_checkout() -> dict[str, Any]:
    return {
        "verified": True,
        "source_head": SOURCE_HEAD,
        "tracked_workspace_clean": True,
        "ignored_untracked_prefix": execute.IGNORED_UNTRACKED_PREFIX,
    }


def rate_receipt() -> dict[str, Any]:
    return {
        "source_url": modal.OFFICIAL_RATE_URL,
        "document_sha256": "sha256:" + "0" * 64,
        "fetched_at": "2026-09-04T19:52:50.511+05:30",
        "rates": {
            "l4_gpu_second": "0.000222",
            "cpu_core_second": "0.0000131",
            "memory_gib_second": "0.00000222",
            "volume_gib_month": "0.09",
        },
        "additional_charges": [],
    }


def rate_refresh() -> dict[str, Any]:
    """A fresh capture+verification bound to the structured rate receipt.

    The capture hashes the two official documents; the receipt's source
    document hash matches the captured one, exactly as a real refresh produces,
    so the whole envelope re-verifies offline with no network.
    """

    receipt = rate_receipt()
    documents = [
        {
            "url": receipt["source_url"],
            "bytes": 20_480,
            "sha256": receipt["document_sha256"],
        },
        {
            "url": rates_module.OFFICIAL_VOLUME_RATE_URL,
            "bytes": 10_240,
            "sha256": "sha256:" + "1" * 64,
        },
    ]
    capture = {
        "kind": "modal_rate_document_capture",
        "observed_at": OBSERVED_AT,
        "documents": documents,
        "capture_sha256": rates_module._sha256_uri(
            json.dumps(
                documents, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        ),
        "parsed_from_html": False,
        "parsing_limitation": (
            "official rates are never derived from page markup; the capture is "
            "provenance for an exact structured receipt"
        ),
    }
    return {
        "capture": capture,
        "verification": rates_module.verify_rate_refresh(receipt, capture=capture),
    }


def _attempt_receipts() -> list[dict[str, Any]]:
    return [
        {
            "lifecycle_id": step["lifecycle_id"],
            "attempt": 1,
            "crashed": False,
            "preempted": False,
            "timed_out": False,
            "terminal_receipt": True,
        }
        for step in modal.call_sequence()
    ]


def build_cells(*, nonce: str = NONCE, flat: bool = False) -> dict[str, dict[str, Any]]:
    plan = core.build_default_plan()
    return {
        cell.cell_id: cell_wrapper(cell, plan, index=index, nonce=nonce, flat=flat)
        for index, cell in enumerate(modal.crossover_schedule(), start=1)
    }


def build_orchestration(*, nonce: str = NONCE) -> dict[str, Any]:
    plan = modal.build_default_plan()
    attempts = _attempt_receipts()
    document: dict[str, Any] = {
        "schema_version": execute.ORCHESTRATION_SCHEMA_VERSION,
        "protocol_id": modal.PROTOCOL_ID,
        "kind": "llmtracefx.modal_l4_crossover.result",
        "published": True,
        "status": "complete",
        "failure": None,
        "plan_sha256": plan.content_sha256,
        "source_head": SOURCE_HEAD,
        "experiment_nonce": nonce,
        "authorization_sha256": "sha256:" + "a" * 64,
        "run_names": modal.run_scoped_names(nonce),
        "base_image_reference": core.BASE_IMAGE_REFERENCE,
        "runtime_image": modal.runtime_image_identity(source_head=SOURCE_HEAD),
        "provider_sdk": _provider_sdk_capabilities(),
        "profile_authentication": _profile_authentication(),
        "credential_exposure": _credential_exposure(),
        "rate_receipt": rate_receipt(),
        "rate_refresh": rate_refresh(),
        "source_checkout": _source_checkout(),
        "decode_feasibility": hypothetical_feasibility(),
        "headroom": _headroom(),
        "call_sequence_executed": [
            {
                "lifecycle_id": item["lifecycle_id"],
                "attempt": item["attempt"],
                "terminal_receipt": item["terminal_receipt"],
            }
            for item in attempts
        ],
        "attempt_receipts": attempts,
        "attempt_adjudication": modal.evaluate_attempt_receipts(attempts),
        "memory_gate": {
            "tuning_applied": False,
            "canaries": [
                _memory_gate_entry("eager", index=1, nonce=nonce),
                _memory_gate_entry("compiled", index=2, nonce=nonce),
            ],
        },
        "completed_cell_ids": sorted(
            cell.cell_id for cell in modal.crossover_schedule()
        ),
        "ledger": modal.build_completed_ledger_document(
            plan=plan,
            source_head=SOURCE_HEAD,
            experiment_nonce=nonce,
            ledger_path_sha256=LEDGER_PATH_SHA256,
            reserved_at=RESERVED_AT,
            completed_at=COMPLETED_AT,
        ),
        "teardown": _teardown(),
        "statistical_publication": dict(modal.STATISTICAL_PUBLICATION),
        "uncontrolled_limitations": list(modal.UNCONTROLLED_CACHE_LIMITATIONS),
        "provider_reported_spend_usd": None,
        "provider_reported_spend_null_reason": (
            "provider spend is external, sanitized, and never inferred"
        ),
        "observed_at": OBSERVED_AT,
    }
    document["orchestration_sha256"] = execute._sha256_json(document)
    return document
