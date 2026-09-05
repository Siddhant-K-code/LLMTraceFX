"""Tests for the offline Modal L4 crossover protocol delta."""

from __future__ import annotations

import json
import sys
from decimal import Decimal
from inspect import signature
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer.lab.qwen3_8b import modal_l4_crossover as modal
from llmtracefx.optimizer.lab.qwen3_8b import modal_l4_rates as rates_module
from llmtracefx.optimizer.lab.qwen3_8b import vllm_compile
from llmtracefx.optimizer.lab.qwen3_8b.modal_l4_rates import RateRefreshError

NONCE = "b" * 32
HEAD = "a" * 40
NOW = "2026-09-04T19:52:50.511+05:30"
LATER = "2026-09-04T20:52:50.511+05:30"

OFFLINE_MODULES = (
    "llmtracefx/optimizer/lab/qwen3_8b/modal_l4_crossover.py",
    "llmtracefx/optimizer/lab/qwen3_8b/modal_l4_crossover_evidence.py",
    "llmtracefx/evidence/modal_l4_crossover_verifier.py",
)


def _rate_receipt(**overrides: Any) -> dict[str, Any]:
    receipt = {
        "source_url": "https://modal.com/pricing",
        "document_sha256": "sha256:" + "0" * 64,
        "fetched_at": NOW,
        "rates": {
            "l4_gpu_second": "0.000222",
            "cpu_core_second": "0.0000131",
            "memory_gib_second": "0.00000222",
            "volume_gib_month": "0.09",
        },
        "additional_charges": [],
    }
    receipt.update(overrides)
    return receipt


def _memory_observation(mode: str = "eager", **overrides: Any) -> dict[str, Any]:
    max_model_len = 16_480
    observation = {
        "mode": mode,
        "gpu_name": modal.EXPECTED_GPU_NAME,
        "gpu_count": 1,
        "runtime_pins": dict(vllm_compile.RUNTIME_PINS),
        "total_vram_mib": 23_034,
        "peak_vram_mib": 21_500,
        "kv_cache_blocks": 640,
        "kv_cache_tokens": 40_960,
        "max_model_len": max_model_len,
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
            "max_model_len": max_model_len,
        },
    }
    observation.update(overrides)
    return observation


def _attempt_receipts(**overrides: Any) -> list[dict[str, Any]]:
    receipts = [
        {
            "lifecycle_id": lifecycle.lifecycle_id,
            "attempt": 1,
            "crashed": False,
            "preempted": False,
            "timed_out": False,
            "terminal_receipt": True,
        }
        for lifecycle in modal.LIFECYCLES
    ]
    if overrides:
        receipts[0].update(overrides)
    return receipts


def _teardown(**overrides: Any) -> dict[str, Any]:
    receipt = {
        "outstanding_calls_cancelled": True,
        "app_context_exited": True,
        "app_deletion_provider_verified": None,
        "functions_scaled_to_zero": True,
        "scale_zero_verified_via_control_plane": True,
        "scale_zero_settling": {
            "mechanism": modal.SCALE_ZERO_SETTLING_MECHANISM,
            "is_scientific_retry": False,
            "poll_timeout_seconds": modal.SCALE_ZERO_POLL_TIMEOUT_SECONDS,
            "samples_taken": 1,
        },
        "container_inventory_observable": False,
        "volume_deleted": True,
        "named_resource_listing_scope": "volumes_only",
        "run_created_noncredential_secrets_deleted": True,
        "sanitized_receipts_retained": True,
        "credential_secret_created": False,
        "live_named_volumes": [],
        "teardown_failures": [],
        "provider_reported_spend_usd": None,
    }
    receipt.update(overrides)
    return receipt


class TestIdentityAndPreservedCore:
    def test_protocol_identity_is_new_and_does_not_touch_the_base(self) -> None:
        assert modal.PROTOCOL_ID == "qwen3-8b-vllm-crossover-modal-l4-v1"
        assert modal.PROTOCOL_ID != vllm_compile.PROTOCOL_ID
        assert modal.BASE_PROTOCOL_ID == "qwen3-8b-vllm-crossover-v2"

    def test_sealed_schedule_and_workload_core_are_preserved_exactly(self) -> None:
        core = modal.build_default_plan().to_dict()["preserved_core"]
        assert core["model"]["id"] == "Qwen/Qwen3-8B"
        assert core["model"]["revision"] == ("b968826d9c46dd6066d109eabc6255188de91218")
        assert core["runtime"]["runtime_pins"] == dict(vllm_compile.RUNTIME_PINS)
        assert core["schedule"] == [
            cell.to_dict() for cell in vllm_compile.crossover_schedule()
        ]
        assert len(core["schedule"]) == 32
        assert core["pairs_per_lane"] == 8
        assert core["controlled_requests_per_cell"] == 144
        assert core["natural_requests_per_cell"] == 12
        assert sorted({cell["order"] for cell in core["schedule"]}) == [
            "compiled-eager",
            "eager-compiled",
        ]

    def test_statistics_are_delegated_not_redefined(self) -> None:
        statistics = modal.build_default_plan().to_dict()["preserved_core"][
            "statistics"
        ]
        assert statistics["bootstrap_unit"] == "whole_pair"
        assert statistics["request_level_resampling"] is False
        assert statistics["headline_extrapolation"] is False
        assert statistics["replacement_cells"] is False
        assert statistics["adaptive_stopping"] is False
        assert statistics["implementation"].endswith("cloudrift_crossover_results")


class TestPricingAndBudget:
    def test_component_rates_derive_the_committed_function_rates(self) -> None:
        assert modal.gpu_function_rate_usd_per_second() == Decimal("0.00034544")
        assert modal.cpu_function_rate_usd_per_second() == Decimal("0.00012344")

    @pytest.mark.parametrize(
        ("stage_id", "seconds", "amount"),
        (
            ("cpu-stage", 1800, "0.222192"),
            ("cpu-verify", 300, "0.037032"),
            ("eager-canary", 300, "0.103632"),
            ("compiled-canary", 420, "0.1450848"),
            ("natural-cell", 3840, "1.3264896"),
            ("controlled-cell", 7680, "2.6529792"),
            ("cpu-analysis", 900, "0.111096"),
        ),
    )
    def test_every_stage_matches_the_approved_envelope(
        self, stage_id: str, seconds: int, amount: str
    ) -> None:
        stage = modal.STAGE_BY_ID[stage_id]
        assert stage.total_seconds == seconds
        assert stage.total_usd == Decimal(amount)

    def test_totals_storage_and_contingency_are_exact(self) -> None:
        assert modal.COMPUTE_PLANNED_SECONDS == 15_240
        assert modal.COMPUTE_PLANNED_USD == Decimal("4.5985056")
        assert modal.storage_reservation_usd() == Decimal("0.48")
        assert modal.TOTAL_PLANNED_USD == Decimal("5.0785056")
        assert modal.UNTOUCHED_MARGIN_USD == Decimal("0.9214944")
        assert modal.HARD_CAP_USD == Decimal("6")
        assert modal.TOTAL_PLANNED_USD + modal.UNTOUCHED_MARGIN_USD == Decimal("6")

    def test_cell_counts_follow_the_sealed_schedule(self) -> None:
        assert modal.STAGE_BY_ID["controlled-cell"].occurrences == 16
        assert modal.STAGE_BY_ID["natural-cell"].occurrences == 16
        assert len(modal.LIFECYCLES) == 37
        assert (
            sum((lifecycle.ceiling_usd for lifecycle in modal.LIFECYCLES), Decimal())
            == modal.COMPUTE_PLANNED_USD
        )

    def test_every_cell_maps_to_exactly_one_lifecycle(self) -> None:
        identities = {
            modal.cell_lifecycle_id(cell) for cell in vllm_compile.crossover_schedule()
        }
        assert len(identities) == 32


class TestResourceAndLifecycleControls:
    def test_runtime_image_binds_a_spec_commitment_not_a_provider_digest(self) -> None:
        image = modal.build_default_plan().to_dict()["runtime_image"]
        assert image["derived_provider_image_digest"] is None
        assert image["derived_provider_image_digest_null_reason"]
        assert image["derived_image_spec_commitment"].startswith("sha256:")
        assert image["provider_sdk_version"] == modal.TESTED_MODAL_VERSION
        assert image["runtime_pins"] == dict(vllm_compile.RUNTIME_PINS)
        assert image["image_build_inputs"]["base_image_reference"] == (
            vllm_compile.BASE_IMAGE_REFERENCE
        )

    def test_runtime_image_spec_binds_the_exact_sdk_and_base_image(self) -> None:
        spec = modal.RUNTIME_IMAGE_SPEC
        assert spec["provider_sdk_version"] == "1.5.5"
        assert spec["base_image_reference"] == vllm_compile.BASE_IMAGE_REFERENCE
        assert spec["vllm_source_commit"] == vllm_compile.VLLM_SOURCE_COMMIT
        # The commitment is a deterministic hash of exactly those inputs.
        assert modal.RUNTIME_IMAGE_SPEC_COMMITMENT == modal._sha256_json(spec)

    def test_runtime_image_run_commitment_binds_the_source_head(self) -> None:
        block = modal.runtime_image_identity(source_head="a" * 40)
        assert block["source_head"] == "a" * 40
        assert block["runtime_image_run_commitment"].startswith("sha256:")
        other = modal.runtime_image_identity(source_head="b" * 40)
        assert (
            other["runtime_image_run_commitment"]
            != block["runtime_image_run_commitment"]
        )

    def test_modal_surface_is_rpc_only_with_one_live_cell(self) -> None:
        settings = modal.RESOURCE_SETTINGS
        assert settings["surface"] == "modal_functions_rpc_only"
        assert settings["public_web_endpoint"] is False
        assert settings["gpu"] == "L4"
        assert settings["gpu_count"] == 1
        assert settings["cpu_physical_cores"] == 4
        assert settings["memory_gib"] == 32
        assert settings["max_containers"] == 1
        assert settings["min_containers"] == 0
        assert settings["max_concurrent_inputs"] == 1
        assert settings["retries"] == 0
        assert settings["single_use_cell_containers"] is True

    def test_cloudrift_host_cache_requirements_are_removed_here_only(self) -> None:
        assert modal.LIFECYCLE_CONTROLS["host_page_cache_reset"] is False
        assert modal.LIFECYCLE_CONTROLS["dedicated_host_required"] is False
        assert (
            vllm_compile.LIFECYCLE_CONTROLS["between_cell_host_page_cache_resets"] == 31
        )
        assert modal.CLAIM_SURFACE["pure_causal_compilation_effect"] == (
            "unsupported_by_construction"
        )
        assert modal.CLAIM_SURFACE["natural_end_to_end_causal_speedup"] == (
            "unsupported_by_construction"
        )
        # Placement is uncontrolled and the physical host is never identified,
        # but per-pair same/different placement *is* derived and published from
        # anonymized commitment groups, so the limitation says exactly that
        # rather than the stronger, untrue "not observable".
        placement = modal.UNCONTROLLED_CACHE_LIMITATIONS[0]
        assert "is never controlled" in placement
        assert "physical host is never identified" in placement
        assert "anonymized placement group is observable" in placement

    def test_run_scoped_names_are_unique_per_nonce(self) -> None:
        first = modal.run_scoped_names(NONCE)
        second = modal.run_scoped_names("c" * 32)
        assert first["app_name"] != second["app_name"]
        assert set(first.values()).isdisjoint(second.values())
        assert all(NONCE in value for value in first.values())

    def test_run_scoped_names_reject_an_unsafe_nonce(self) -> None:
        with pytest.raises(modal.ModalL4ContractError, match="nonce"):
            modal.run_scoped_names("short")


class TestAuthenticationGuard:
    @pytest.mark.parametrize(
        "name",
        (
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
            "MODAL_PROFILE",
            "MODAL_CONFIG_PATH",
            "MODAL_SERVER_URL",
            "MODAL_ENVIRONMENT",
        ),
    )
    def test_credential_and_routing_overrides_are_refused(self, name: str) -> None:
        with pytest.raises(modal.ModalL4ContractError) as excinfo:
            modal.require_local_profile_authentication({name: "sensitive-value"})
        assert name in str(excinfo.value)
        assert "sensitive-value" not in str(excinfo.value)

    def test_credential_shaped_names_are_refused(self) -> None:
        with pytest.raises(modal.ModalL4ContractError, match="credential-shaped"):
            modal.require_local_profile_authentication({"SOME_API_KEY": "x"})

    def test_clean_environment_uses_the_local_profile(self) -> None:
        modal.require_local_profile_authentication(
            {"PATH": "/usr/bin", "HOME": "/nonexistent", "MODAL_TOKEN_ID": "  "}
        )

    def test_offline_modules_never_import_the_provider_sdk(self) -> None:
        root = Path(__file__).resolve().parents[2]
        for relative in OFFLINE_MODULES:
            text = (root / relative).read_text(encoding="utf-8")
            assert "import modal\n" not in text
            assert "from modal" not in text
        assert not [name for name in sys.modules if name.split(".")[0] == "modal"]

    def test_provider_sdk_presence_fails_closed(self) -> None:
        with pytest.raises(modal.ModalL4ContractError, match="must not be imported"):
            modal.assert_provider_sdk_absent({"modal": object()})

    def test_unrelated_modal_prefixed_module_is_not_the_provider_sdk(self) -> None:
        modal.assert_provider_sdk_absent(
            {"modal_glm_preflight_evidence_bundle": object()}
        )


class TestRateAndBudgetReceipts:
    def test_equal_official_rates_pass(self) -> None:
        result = modal.verify_official_rate_receipt(_rate_receipt())
        assert result["verified"] is True
        assert result["official_rates_at_or_below_committed"] is True

    def test_lower_official_rates_pass(self) -> None:
        receipt = _rate_receipt()
        receipt["rates"]["l4_gpu_second"] = "0.0002"
        assert modal.verify_official_rate_receipt(receipt)["verified"] is True

    def test_higher_official_rate_refuses_the_run(self) -> None:
        receipt = _rate_receipt()
        receipt["rates"]["l4_gpu_second"] = "0.000333"
        with pytest.raises(modal.ModalL4ContractError, match="exceed the committed"):
            modal.verify_official_rate_receipt(receipt)

    def test_new_charge_component_refuses_the_run(self) -> None:
        receipt = _rate_receipt()
        receipt["rates"]["egress_gib"] = "0.01"
        with pytest.raises(modal.ModalL4ContractError, match="uncommitted charge"):
            modal.verify_official_rate_receipt(receipt)

    def test_missing_component_and_additional_charges_refuse(self) -> None:
        receipt = _rate_receipt()
        del receipt["rates"]["volume_gib_month"]
        with pytest.raises(modal.ModalL4ContractError, match="missing committed"):
            modal.verify_official_rate_receipt(receipt)
        with pytest.raises(modal.ModalL4ContractError, match="additional charge"):
            modal.verify_official_rate_receipt(
                _rate_receipt(additional_charges=["surge"])
            )

    @pytest.mark.parametrize(
        ("field", "value", "match"),
        (
            ("source_url", "http://modal.com/pricing", "https"),
            ("source_url", "https://rates.example.com/modal", "official domain"),
            ("document_sha256", "nope", "document hash"),
            ("fetched_at", "yesterday", "timestamp"),
        ),
    )
    def test_rate_receipt_provenance_is_enforced(
        self, field: str, value: str, match: str
    ) -> None:
        with pytest.raises(modal.ModalL4ContractError, match=match):
            modal.verify_official_rate_receipt(_rate_receipt(**{field: value}))

    def test_no_second_headroom_helper_contradicts_the_execution_gate(self) -> None:
        """Headroom has exactly one adjudicator, and absence is never zero risk.

        An earlier helper on this module returned ``supported: False`` for an
        absent receipt, which reads as "no headroom recorded" -- the opposite of
        the execution gate's rule that an absent limit is never an unlimited
        one. It is removed rather than kept alongside, so there is one place
        headroom is decided.
        """

        assert not hasattr(modal, "verify_budget_headroom_receipt")
        with pytest.raises(RateRefreshError, match="refusing to infer"):
            rates_module.account_headroom()


class TestDecodeBandwidthFeasibility:
    """The offline arithmetic proof that the approved design cannot run.

    Every number below is derived from constants that are already sealed
    elsewhere in the protocol, so the proof can be re-done by hand:

        144 controlled requests x 96 output tokens = 13,824 tokens
        13,824 tokens x 14,985,816,064 bytes  = 207,163,921,268,736 bytes
        207,163,921,268,736 / 322,122,547,200 = 643.121455078125 seconds

    against a sealed 480-second controlled-cell timeout, excluding container
    start, weight load, engine init, prefill, and CUDA-graph capture.
    """

    def test_the_sealed_inputs_are_the_protocol_constants(self) -> None:
        assert modal.STAGED_MODEL_BYTES == 16_397_461_266
        assert modal.STAGED_MODEL_BYTES == vllm_compile.EXPECTED_MODEL_BYTES
        assert modal.MODEL_TENSOR_BYTES == 16_381_470_720
        assert modal.INPUT_EMBEDDING_BYTES == 1_244_659_712
        assert modal.NON_DENSE_TENSOR_ALLOWANCE_BYTES == 16 * 1024 * 1024
        assert modal.DENSE_DECODE_WEIGHT_BYTES == 15_120_033_792
        assert modal.ON_CHIP_CACHE_ALLOWANCE_BYTES == 128 * 1024 * 1024
        assert modal.MINIMUM_HBM_WEIGHT_BYTES_PER_TOKEN == 14_985_816_064
        assert modal.CONTROLLED_CELL_OUTPUT_TOKENS == 144 * 96 == 13_824
        assert modal.L4_ADVERTISED_PEAK_BANDWIDTH_BYTES_PER_SECOND == 300_000_000_000
        assert modal.FEASIBILITY_BANDWIDTH_BYTES_PER_SECOND == 300 * 1024**3
        assert modal.CONTROLLED_CELL_TIMEOUT_SECONDS == 480

    def test_model_facts_are_bound_to_the_pinned_inventory(self) -> None:
        manifest_path = (
            Path(__file__).parents[2]
            / "llmtracefx/optimizer/lab/qwen3_8b/data"
            / "qwen3-8b-conversion-manifest-v1.json"
        )
        source = json.loads(manifest_path.read_text(encoding="utf-8"))["source"]
        files = {item["path"]: item for item in source["files"]}
        assert source["official_revision"] == modal.MODEL_REVISION
        assert files["config.json"]["sha256"] == (
            "f7c4eadfbbf522470667b797a3c89be2524832d2d599797248dc304fff447c30"
        )
        assert files["model.safetensors.index.json"]["sha256"] == (
            "f9fdbcb91c23971c13ec5d5f2573d2349e8f61f2f049371ec699281748fdb1bc"
        )
        facts = modal.evaluate_decode_bandwidth_feasibility()["inputs"][
            "model_architecture_facts"
        ]
        assert facts["tensor_payload_bytes"] == modal.MODEL_TENSOR_BYTES
        assert facts["input_embedding_bytes_excluded"] == (
            modal.MODEL_CONFIG_VOCAB_SIZE
            * modal.MODEL_CONFIG_HIDDEN_SIZE
            * modal.BF16_BYTES_PER_ELEMENT
        )
        assert facts["minimum_hbm_weight_bytes_per_token"] == (
            modal.MODEL_TENSOR_BYTES
            - facts["input_embedding_bytes_excluded"]
            - facts["non_dense_tensor_allowance_bytes"]
            - facts["on_chip_cache_allowance_bytes"]
        )

    def test_the_derived_arithmetic_is_exact(self) -> None:
        verdict = modal.evaluate_decode_bandwidth_feasibility()
        derivation = verdict["derivation"]
        assert derivation["weight_bytes_streamed_per_cell"] == 207_163_921_268_736
        assert derivation["bytes_streamable_within_timeout"] == 154_618_822_656_000
        assert derivation["minimum_decode_only_seconds"] == "643.121455078125"
        assert derivation["required_tokens_per_second"] == "28.8"
        assert derivation["theoretical_peak_tokens_per_second"] == "21.495162213676"
        assert derivation["minimum_over_timeout_ratio"] == "1.339836364747"
        assert derivation["rounding"] == (
            "theoretical_peak_tokens_per_second rounds down and "
            "minimum_over_timeout_ratio rounds up to 12 decimal places; "
            "minimum_decode_only_seconds and required_tokens_per_second are "
            "exact decimals; the verdict is exact integer arithmetic"
        )

    def test_the_sealed_design_is_infeasible(self) -> None:
        verdict = modal.evaluate_decode_bandwidth_feasibility()
        assert verdict["feasible"] is False
        assert verdict["uses_sealed_constants"] is True
        assert verdict["computed_offline"] is True
        assert "480s controlled-cell timeout" in verdict["verdict"]

    def test_the_assumptions_and_exclusions_are_published(self) -> None:
        verdict = modal.evaluate_decode_bandwidth_feasibility()
        assert verdict["assumptions"] == list(modal.DECODE_FEASIBILITY_ASSUMPTIONS)
        assert verdict["derivation"]["excluded_from_the_minimum"] == list(
            modal.DECODE_FEASIBILITY_EXCLUSIONS
        )
        assert verdict["inputs"]["model_bytes_provenance"]
        assert verdict["inputs"]["peak_bandwidth_provenance"]
        assert verdict["inputs"]["decode_execution_contract"] == {
            "dtype": "bfloat16",
            "max_num_seqs": 1,
            "enable_prefix_caching": False,
            "speculative_config": None,
            "request_execution": "sequential",
        }

    def test_requiring_feasibility_refuses_the_sealed_design(self) -> None:
        with pytest.raises(modal.ModalL4ContractError, match="infeasible"):
            modal.require_controlled_cell_decode_feasible()

    def test_the_refusal_names_the_deficit(self) -> None:
        with pytest.raises(modal.ModalL4ContractError) as excinfo:
            modal.require_controlled_cell_decode_feasible()
        message = str(excinfo.value)
        assert "643.121455078125s" in message
        assert "480s" in message
        assert "28.8" in message
        assert "21.495162213676" in message

    def test_the_eager_canary_alone_could_not_have_caught_this(self) -> None:
        """One 96-token canary fits easily; only the full cell does not.

        This is why the gate is arithmetic rather than an observation: a
        passing canary says nothing about whether 144 of them fit in 480s.
        """

        canary = modal.evaluate_decode_bandwidth_feasibility(
            output_tokens=modal.DECODE_STEPS,
            timeout_seconds=modal.EAGER_CANARY_TIMEOUT_SECONDS,
        )
        assert canary["feasible"] is True
        assert canary["derivation"]["minimum_decode_only_seconds"] == (
            "4.4661212158203125"
        )

    def test_a_faster_hypothetical_device_is_planning_only(self) -> None:
        verdict = modal.evaluate_decode_bandwidth_feasibility(
            peak_bandwidth_bytes_per_second=1_000_000_000_000
        )
        assert verdict["feasible"] is True
        assert verdict["uses_sealed_constants"] is False
        assert not signature(modal.require_controlled_cell_decode_feasible).parameters

    @pytest.mark.parametrize("field", sorted(("model_bytes", "output_tokens")))
    def test_a_nonpositive_input_is_refused(self, field: str) -> None:
        with pytest.raises(modal.ModalL4ContractError):
            modal.evaluate_decode_bandwidth_feasibility(**{field: 0})

    def test_the_remedy_policy_forbids_resizing_the_experiment(self) -> None:
        policy = modal.evaluate_decode_bandwidth_feasibility()["remedy_policy"]
        for forbidden in ("lowering n", "tuning the runner", "changing the GPU"):
            assert forbidden in policy

    def test_the_plan_carries_the_proof(self) -> None:
        plan = modal.build_default_plan().to_dict()
        assert plan["decode_feasibility"] == (
            modal.evaluate_decode_bandwidth_feasibility()
        )

    def test_the_offline_document_refuses_first(self) -> None:
        document = modal.offline_plan_document()
        assert document["execution_refused_offline"] is True
        assert document["decode_feasibility"]["feasible"] is False
        assert "infeasible on the pinned accelerator" in document["blockers"][0]


class TestCanonicalClaimIdentifiers:
    def test_the_registry_groups_are_disjoint_and_complete(self) -> None:
        offline = set(modal.OFFLINE_ONLY_CLAIM_IDS)
        measured = set(modal.MEASURED_CLAIM_IDS)
        blocked = set(modal.BLOCKED_CLAIM_IDS)
        assert offline & measured == set()
        assert offline & blocked == set()
        assert measured & blocked == set()
        assert set(modal.PREREGISTERED_CLAIM_IDS) == offline | measured | blocked
        assert set(modal.RESULT_CLAIM_IDS) == measured | blocked

    def test_every_result_claim_is_preregistered(self) -> None:
        assert set(modal.RESULT_CLAIM_IDS) <= set(modal.PREREGISTERED_CLAIM_IDS)

    def test_the_memory_gate_and_blocked_claims_are_in_both_matrices(self) -> None:
        for claim_id in (
            "memory-gate-passed",
            "pure-causal-compilation-effect",
            "hardware-matched-comparison",
            "natural-end-to-end-causal-speedup",
            "cache-state-controlled-comparison",
            "compile-cuda-graph-component-timing",
        ):
            assert claim_id in modal.RESULT_CLAIM_IDS
            assert claim_id in modal.PREREGISTERED_CLAIM_IDS

    def test_blocked_ids_derive_from_the_reason_registry(self) -> None:
        assert modal.BLOCKED_CLAIM_IDS == tuple(
            sorted(modal.UNSUPPORTED_BY_CONSTRUCTION_CLAIMS)
        )
        assert all(modal.UNSUPPORTED_BY_CONSTRUCTION_CLAIMS.values())


class TestReusedPrimitiveDeclaration:
    def test_the_plan_and_the_results_path_share_one_list(self) -> None:
        """The protocol's claim and the code that reuses them cannot drift.

        These were two hand-maintained lists that disagreed: the plan named
        ``_pair_effect_distributions``, which the Modal results path never
        calls directly, and omitted the analysis document, the natural
        evaluator, the quality bootstrap, and the two pair-curve primitives it
        does call.
        """

        from llmtracefx.optimizer.lab.qwen3_8b import (
            modal_l4_crossover_results as results,
        )

        assert (
            results.REUSED_STATISTICAL_PRIMITIVES
            is modal.REUSED_PROVIDER_NEUTRAL_PRIMITIVES
        )
        assert modal.STATISTICAL_PUBLICATION[
            "reused_provider_neutral_primitives"
        ] == list(modal.REUSED_PROVIDER_NEUTRAL_PRIMITIVES)
        assert "_pair_effect_distributions" not in "".join(
            modal.REUSED_PROVIDER_NEUTRAL_PRIMITIVES
        )


class TestScaleToZeroSettlingContract:
    def test_the_settling_budget_is_finite_and_exact(self) -> None:
        assert modal.SCALE_ZERO_POLL_ATTEMPTS == 12
        assert modal.SCALE_ZERO_POLL_INTERVAL_SECONDS == 5
        assert modal.SCALE_ZERO_POLL_TIMEOUT_SECONDS == 55
        assert modal.SCALE_ZERO_POLL_TIMEOUT_SECONDS > modal.SCALEDOWN_WINDOW_SECONDS

    def test_the_contract_publishes_the_mechanism_and_bound(self) -> None:
        contract = modal.TEARDOWN_CONTRACT
        assert contract["scale_to_zero_settling_mechanism"] == (
            modal.SCALE_ZERO_SETTLING_MECHANISM
        )
        assert contract["scale_to_zero_settling_timeout_seconds"] == (
            modal.SCALE_ZERO_POLL_TIMEOUT_SECONDS
        )
        assert contract["scale_to_zero_settling_is_scientific_retry"] is False

    def test_a_single_immediate_sample_claim_is_not_enough(self) -> None:
        """A receipt with no settling record cannot adjudicate complete."""

        receipt = _teardown()
        del receipt["scale_zero_settling"]
        verdict = modal.evaluate_teardown_receipt(receipt)
        assert verdict["complete"] is False
        assert "scale_zero_settling" in verdict["failures"]

    def test_settling_beyond_the_bounded_budget_is_refused(self) -> None:
        receipt = _teardown()
        receipt["scale_zero_settling"] = {
            **receipt["scale_zero_settling"],
            "samples_taken": modal.SCALE_ZERO_POLL_ATTEMPTS + 1,
        }
        verdict = modal.evaluate_teardown_receipt(receipt)
        assert verdict["complete"] is False
        assert "scale_zero_settling" in verdict["failures"]


class TestSignedHeadroomReceiptSchema:
    def test_the_receipt_binds_the_run_and_carries_no_identity(self) -> None:
        assert modal.HEADROOM_RECEIPT_FIELDS == (
            "schema_version",
            "kind",
            "protocol_id",
            "plan_sha256",
            "source_head",
            "experiment_nonce",
            "headroom_usd",
            "confirmed_at",
            "expires_at",
        )
        for fragment in ("account", "workspace", "profile", "email"):
            assert fragment in modal.FORBIDDEN_HEADROOM_KEY_FRAGMENTS
            assert not any(fragment in field for field in modal.HEADROOM_RECEIPT_FIELDS)

    def test_the_receipt_window_is_bounded(self) -> None:
        assert modal.MAX_HEADROOM_RECEIPT_WINDOW_SECONDS == 24 * 3600


class TestMemoryGate:
    def test_both_canaries_can_pass(self) -> None:
        for mode in ("eager", "compiled"):
            verdict = modal.evaluate_memory_gate(_memory_observation(mode))
            assert verdict["passed"] is True
            assert verdict["action"] == "proceed"
            assert verdict["tuning_allowed"] is False

    def test_immutable_kwargs_are_frozen_in_the_contract(self) -> None:
        kwargs: Any = modal.MEMORY_GATE["immutable_runner_kwargs"]
        assert kwargs["dtype"] == "bfloat16"
        assert kwargs["tensor_parallel_size"] == 1
        assert kwargs["max_num_seqs"] == 1
        assert kwargs["gpu_memory_utilization"] == "0.94"
        assert kwargs["enable_prefix_caching"] is False
        assert kwargs["speculative_config"] is None
        assert modal.max_model_len(16_384) == 16_480
        assert modal.MEMORY_GATE["tuning_allowed"] is False
        assert modal.MEMORY_GATE["failure_action"] == "publish_refusal_only"
        assert modal.MEMORY_GATE["staging_verification"] == {
            "expected_file_count": 15,
            "expected_bytes": 16_397_461_266,
            "seals_prompt_token_arrays": True,
        }

    @pytest.mark.parametrize(
        ("override", "failure"),
        (
            ({"gpu_name": "NVIDIA A10G"}, "gpu_name"),
            ({"gpu_count": 2}, "gpu_count"),
            ({"runtime_pins": {"vllm_version": "0.29.0"}}, "runtime_pins"),
            ({"peak_vram_mib": 22_600}, "peak_vram_mib"),
            ({"total_vram_mib": 8_000}, "total_vram_mib"),
            ({"kv_cache_blocks": 0}, "kv_cache_blocks"),
            ({"kv_cache_tokens": 100}, "kv_cache_tokens"),
            ({"out_of_memory": True}, "out_of_memory"),
            ({"generated_tokens": 12}, "generated_tokens"),
            ({"terminal": False}, "terminal"),
            (
                {"used_longest_controlled_prompt": False},
                "used_longest_controlled_prompt",
            ),
        ),
    )
    def test_gate_fails_closed_on_every_admission_condition(
        self, override: dict[str, Any], failure: str
    ) -> None:
        verdict = modal.evaluate_memory_gate(_memory_observation(**override))
        assert verdict["passed"] is False
        assert failure in verdict["failures"]
        assert verdict["action"] == "publish_refusal_only"

    def test_tuned_runner_arguments_fail_the_gate(self) -> None:
        observation = _memory_observation()
        observation["runner_kwargs"]["gpu_memory_utilization"] = "0.85"
        verdict = modal.evaluate_memory_gate(observation)
        assert verdict["failures"] == ["runner_kwargs"]

    def test_peak_must_leave_the_reserved_headroom(self) -> None:
        exact = _memory_observation(peak_vram_mib=23_034 - modal.VRAM_HEADROOM_MIB)
        assert modal.evaluate_memory_gate(exact)["passed"] is True
        over = _memory_observation(peak_vram_mib=23_034 - modal.VRAM_HEADROOM_MIB + 1)
        assert modal.evaluate_memory_gate(over)["passed"] is False


class TestAttemptAndTeardownReceipts:
    def test_one_terminal_attempt_per_lifecycle_is_valid(self) -> None:
        verdict = modal.evaluate_attempt_receipts(_attempt_receipts())
        assert verdict["valid"] is True
        assert verdict["action"] == "publish_results"

    @pytest.mark.parametrize(
        ("override", "observation"),
        (
            ({"attempt": 2}, "second_attempt"),
            ({"crashed": True}, "crash"),
            ({"preempted": True}, "preemption"),
            ({"timed_out": True}, "timeout"),
            ({"terminal_receipt": False}, "missing_terminal_receipt"),
        ),
    )
    def test_every_invalidating_observation_terminates_the_run(
        self, override: dict[str, Any], observation: str
    ) -> None:
        verdict = modal.evaluate_attempt_receipts(_attempt_receipts(**override))
        assert verdict["valid"] is False
        assert verdict["action"] == "invalidate_and_tear_down"
        assert observation in {item["observation"] for item in verdict["findings"]}
        assert observation in modal.INVALIDATING_OBSERVATIONS

    def test_missing_lifecycle_receipt_is_not_silence(self) -> None:
        verdict = modal.evaluate_attempt_receipts(_attempt_receipts()[:-1])
        assert verdict["valid"] is False
        assert {item["observation"] for item in verdict["findings"]} == {
            "missing_terminal_receipt"
        }

    def test_unplanned_lifecycle_is_rejected(self) -> None:
        receipts = _attempt_receipts()
        receipts[0]["lifecycle_id"] = "not-a-lifecycle"
        with pytest.raises(modal.ModalL4ContractError, match="unplanned lifecycle"):
            modal.evaluate_attempt_receipts(receipts)

    def test_complete_teardown_reports_null_provider_spend(self) -> None:
        result = modal.evaluate_teardown_receipt(_teardown())
        assert result["complete"] is True
        assert result["provider_reported_spend_usd"] is None
        assert result["provider_reported_spend_null_reason"]
        assert result["storage_allowance_days"] == 4

    def test_teardown_only_scopes_the_listing_to_volumes_and_never_claims_deletion(
        self,
    ) -> None:
        # App context exit is a local action, not provider deletion proof; the
        # empty-listing claim covers only volumes; and any recorded failure
        # (an ambiguous listing) fails closed.
        assert modal.evaluate_teardown_receipt(_teardown())["complete"] is True
        assert (
            modal.evaluate_teardown_receipt(
                _teardown(app_deletion_provider_verified=True)
            )["complete"]
            is False
        )
        assert (
            modal.evaluate_teardown_receipt(
                _teardown(named_resource_listing_scope="all_resources")
            )["complete"]
            is False
        )
        ambiguous = modal.evaluate_teardown_receipt(
            _teardown(teardown_failures=["named_resource_listing_unavailable"])
        )
        assert ambiguous["complete"] is False
        assert "teardown_failures" in ambiguous["failures"]

    @pytest.mark.parametrize(
        "override",
        (
            {"app_context_exited": False},
            {"functions_scaled_to_zero": False, "teardown_failures": ["x"]},
            {"outstanding_calls_cancelled": False},
            {"scale_zero_verified_via_control_plane": False},
            {"volume_deleted": False},
            {"run_created_noncredential_secrets_deleted": False},
            {"sanitized_receipts_retained": False},
            {"credential_secret_created": True},
            {"live_named_volumes": ["llmtracefx-qwen3-8b-modal-l4-b" * 1]},
            {"app_deletion_provider_verified": True},
            {"container_inventory_observable": True},
            {"named_resource_listing_scope": "all"},
            {"teardown_failures": ["named_resource_listing_unavailable"]},
        ),
    )
    def test_incomplete_teardown_fails_closed(self, override: dict[str, Any]) -> None:
        result = modal.evaluate_teardown_receipt(_teardown(**override))
        assert result["complete"] is False
        assert result["failures"]


class TestPlanDocument:
    def test_plan_round_trips_through_canonical_json(self) -> None:
        plan = modal.build_default_plan()
        assert modal.ModalL4Plan.from_json(plan.to_json()).content_sha256 == (
            plan.content_sha256
        )

    def test_plan_rejects_any_drift(self) -> None:
        data = modal.build_default_plan().to_dict()
        data["budget"]["hard_cap_usd"] = "60"
        with pytest.raises(modal.ModalL4ContractError, match="does not exactly match"):
            modal.ModalL4Plan.from_dict(data)

    def test_plan_rejects_missing_and_extra_keys(self) -> None:
        data = modal.build_default_plan().to_dict()
        data.pop("pricing")
        with pytest.raises(modal.ModalL4ContractError, match="missing="):
            modal.ModalL4Plan.from_dict(data)
        extra = modal.build_default_plan().to_dict()
        extra["surprise"] = True
        with pytest.raises(modal.ModalL4ContractError, match="extra="):
            modal.ModalL4Plan.from_dict(extra)

    def test_offline_plan_document_refuses_everything_provider_side(self) -> None:
        document = modal.offline_plan_document()
        assert document["execution_authorized"] is False
        assert document["offline_only"] is True
        assert document["provider_authentication_used"] is False
        assert document["provider_sdk_imported"] is False
        assert document["container_created"] is False
        assert document["model_downloaded"] is False
        assert document["gpu_used"] is False
        assert document["spend_usd"] == "0"
        assert "provider-reported spend" in document["unsupported_claims"]
        assert "causal serving speedup" in document["unsupported_claims"]

    def test_plan_declares_the_ledger_is_not_provider_proof(self) -> None:
        budget = modal.build_default_plan().to_dict()["budget"]
        assert budget["application_ledger_required"] is True
        assert budget["application_ledger_is_provider_proof"] is False
        assert budget["contingency_is_never_spent_on_science"] is True


class TestApplicationLedger:
    def _ledger(self, tmp_path: Path) -> modal.ModalApplicationLedger:
        return modal.ModalApplicationLedger.initialize(
            tmp_path / "ledger.json",
            plan=modal.build_default_plan(),
            git_head=HEAD,
            experiment_nonce=NONCE,
        )

    def test_initial_ledger_pre_reserves_storage_and_denies_provider_proof(
        self, tmp_path: Path
    ) -> None:
        snapshot = self._ledger(tmp_path).snapshot()
        assert snapshot["is_provider_proof"] is False
        assert snapshot["provider_reported_spend_usd"] is None
        assert snapshot["reserved_usd"] == "0.48"
        assert snapshot["remaining_usd"] == "5.52"
        assert len(snapshot["entries"]) == 37

    def test_reserve_and_complete_are_chained_and_priced(self, tmp_path: Path) -> None:
        ledger = self._ledger(tmp_path)
        ledger.reserve("call-01", lifecycle_id="cpu-stage-01", reserved_at=NOW)
        event = ledger.complete("call-01", completed_at=LATER, actual_seconds=600)
        assert event["actual_cost_usd"] == "0.074064"
        snapshot = ledger.snapshot()
        entry = snapshot["entries"][0]
        assert entry["status"] == "completed"
        assert snapshot["reserved_usd"] == "0.702192"

    def test_duplicate_reservation_and_unknown_lifecycle_are_refused(
        self, tmp_path: Path
    ) -> None:
        ledger = self._ledger(tmp_path)
        ledger.reserve("call-01", lifecycle_id="cpu-stage-01", reserved_at=NOW)
        with pytest.raises(modal.ModalL4ContractError, match="already reserved"):
            ledger.reserve("call-01", lifecycle_id="cpu-verify-01", reserved_at=NOW)
        with pytest.raises(modal.ModalL4ContractError, match="not in the plan"):
            ledger.reserve("call-02", lifecycle_id="nope", reserved_at=NOW)

    def test_completion_cannot_exceed_the_planned_ceiling(self, tmp_path: Path) -> None:
        ledger = self._ledger(tmp_path)
        ledger.reserve("call-01", lifecycle_id="cpu-verify-01", reserved_at=NOW)
        with pytest.raises(modal.ModalL4ContractError, match="exceeds the planned"):
            ledger.complete("call-01", completed_at=LATER, actual_seconds=400)

    def test_abort_records_a_reason_and_frees_nothing(self, tmp_path: Path) -> None:
        ledger = self._ledger(tmp_path)
        ledger.reserve("call-01", lifecycle_id="eager-canary-01", reserved_at=NOW)
        ledger.abort("call-01", aborted_at=LATER, reason="memory gate refused")
        snapshot = ledger.snapshot()
        assert snapshot["entries"][2]["status"] == "aborted"
        assert snapshot["reserved_usd"] == "0.583632"

    def test_tampered_event_chain_is_detected(self, tmp_path: Path) -> None:
        ledger = self._ledger(tmp_path)
        ledger.reserve("call-01", lifecycle_id="cpu-stage-01", reserved_at=NOW)
        payload = json.loads(ledger.path.read_text(encoding="utf-8"))
        payload["events"][0]["reserved_usd"] = "0.000001"
        ledger.path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(modal.ModalL4ContractError, match="seal does not verify"):
            ledger.snapshot()

    def test_reseal_after_tamper_still_fails_on_the_event_hash(
        self, tmp_path: Path
    ) -> None:
        ledger = self._ledger(tmp_path)
        ledger.reserve("call-01", lifecycle_id="cpu-stage-01", reserved_at=NOW)
        payload = json.loads(ledger.path.read_text(encoding="utf-8"))
        payload["events"][0]["reserved_usd"] = "0.000001"
        payload.pop("ledger_sha256")
        payload["ledger_sha256"] = modal._sha256_json(payload)
        ledger.path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(modal.ModalL4ContractError, match="event hash"):
            ledger.snapshot()

    def test_rollback_to_an_earlier_revision_is_detected(self, tmp_path: Path) -> None:
        ledger = self._ledger(tmp_path)
        before = ledger.path.read_text(encoding="utf-8")
        ledger.reserve("call-01", lifecycle_id="cpu-stage-01", reserved_at=NOW)
        ledger.path.write_text(before, encoding="utf-8")
        with pytest.raises(modal.ModalL4ContractError, match="rollback"):
            ledger.snapshot()

    def test_ledger_cannot_be_reinitialized(self, tmp_path: Path) -> None:
        self._ledger(tmp_path)
        with pytest.raises(modal.ModalL4ContractError, match="cannot be reset"):
            self._ledger(tmp_path)

    def test_ledger_requires_an_exact_head_and_nonce(self, tmp_path: Path) -> None:
        with pytest.raises(modal.ModalL4ContractError, match="git head"):
            modal.ModalApplicationLedger(
                tmp_path / "l.json",
                plan=modal.build_default_plan(),
                git_head="abc",
                experiment_nonce=NONCE,
            )
        with pytest.raises(modal.ModalL4ContractError, match="nonce"):
            modal.ModalApplicationLedger(
                tmp_path / "l.json",
                plan=modal.build_default_plan(),
                git_head=HEAD,
                experiment_nonce="zz",
            )

    def test_whole_run_reservation_stays_within_the_hard_cap(
        self, tmp_path: Path
    ) -> None:
        ledger = self._ledger(tmp_path)
        for index, lifecycle in enumerate(modal.LIFECYCLES, start=1):
            ledger.reserve(
                f"call-{index:02d}",
                lifecycle_id=lifecycle.lifecycle_id,
                reserved_at=NOW,
            )
        snapshot = ledger.snapshot()
        assert snapshot["reserved_usd"] == "5.0785056"
        assert snapshot["remaining_usd"] == "0.9214944"


class TestCommandLine:
    def test_plan_action_prints_a_zero_spend_document(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        assert modal.main(["plan", "--output", str(tmp_path / "plan.json")]) == 0
        document = json.loads(capsys.readouterr().out)
        assert document["spend_usd"] == "0"
        assert json.loads((tmp_path / "plan.json").read_text(encoding="utf-8")) == (
            document
        )

    def test_verify_plan_prints_the_content_hash(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        plan = modal.build_default_plan()
        path = tmp_path / "plan.json"
        path.write_text(plan.to_json(), encoding="utf-8")
        assert modal.main(["verify-plan", "--plan", str(path)]) == 0
        assert capsys.readouterr().out.strip() == plan.content_sha256

    def test_invalid_plan_exits_nonzero_without_a_traceback(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        path = tmp_path / "plan.json"
        path.write_text("{}", encoding="utf-8")
        assert modal.main(["verify-plan", "--plan", str(path)]) == 1
        assert "llmtracefx-modal-l4-crossover" in capsys.readouterr().err


def _attestation(**overrides: Any) -> dict[str, Any]:
    payload = {
        "schema_version": "1",
        "kind": modal.CREDENTIAL_EXPOSURE_ATTESTATION_KIND,
        "protocol_id": modal.PROTOCOL_ID,
        "exposed_profile_credential_never_used_by_experiment": True,
        "exposed_profile_credential_revocation_confirmed": True,
        "revocation_confirmed_by": "coordinator",
        "fresh_local_profile_created_without_sharing": True,
        "fresh_profile_shared_anywhere": False,
        "confirmed_at": NOW,
        "status": "cleared",
        "reason": "coordinator confirmed revocation and fresh local profile",
    }
    payload.update(overrides)
    return payload


class TestCredentialExposureGate:
    def test_absence_is_refusal_not_permission(self) -> None:
        verdict = modal.evaluate_credential_exposure_attestation(None)
        assert verdict["cleared"] is False
        assert verdict["action"] == "refuse_provider_execution"
        assert verdict["exposed_profile_credential_never_used_by_experiment"] is True
        assert verdict["exposed_profile_credential_revocation_confirmed"] is False
        assert verdict["fresh_local_profile_created_without_sharing"] is False
        assert verdict["records_credential_values"] is False
        with pytest.raises(modal.ModalL4ContractError, match="blocked until"):
            modal.require_credential_exposure_cleared(None)

    def test_a_coordinator_confirmation_clears_the_gate(self) -> None:
        verdict = modal.require_credential_exposure_cleared(_attestation())
        assert verdict["cleared"] is True
        assert verdict["confirmed_by"] == "coordinator"
        assert verdict["action"] == "proceed"

    @pytest.mark.parametrize(
        "override",
        (
            {"exposed_profile_credential_never_used_by_experiment": False},
            {"exposed_profile_credential_revocation_confirmed": False},
            {"fresh_local_profile_created_without_sharing": False},
            {"fresh_profile_shared_anywhere": True},
            {"status": "blocked"},
            {"revocation_confirmed_by": None},
            {"confirmed_at": None},
        ),
    )
    def test_every_unmet_condition_blocks_execution(
        self, override: dict[str, Any]
    ) -> None:
        verdict = modal.evaluate_credential_exposure_attestation(
            _attestation(**override)
        )
        assert verdict["cleared"] is False
        assert verdict["action"] == "refuse_provider_execution"
        with pytest.raises(modal.ModalL4ContractError, match="blocked until"):
            modal.require_credential_exposure_cleared(_attestation(**override))

    @pytest.mark.parametrize(
        "field",
        (
            "revoked_token_id",
            "exposed_token_secret",
            "credential_sha256",
            "token_prefix",
            "screenshot_url",
            "account_email",
        ),
    )
    def test_credential_or_screenshot_fields_are_refused(self, field: str) -> None:
        with pytest.raises(
            modal.ModalL4ContractError, match="credential or screenshot derived"
        ):
            modal.evaluate_credential_exposure_attestation(
                _attestation(**{field: "anything"})
            )

    def test_any_extra_field_is_refused(self) -> None:
        with pytest.raises(modal.ModalL4ContractError, match="closed allowlist"):
            modal.evaluate_credential_exposure_attestation(
                _attestation(operator_note="fine")
            )

    def test_a_missing_field_is_refused(self) -> None:
        payload = _attestation()
        del payload["fresh_profile_shared_anywhere"]
        with pytest.raises(modal.ModalL4ContractError, match="incomplete"):
            modal.evaluate_credential_exposure_attestation(payload)

    @pytest.mark.parametrize(
        "reason",
        (
            "the token was ak-abcdefghijklmnop",
            "sha256:0123456789abcdef of the old credential",
            "old secret rotated",
            "AKIAIOSFODNN7EXAMPLEKEY123456",
        ),
    )
    def test_a_credential_shaped_reason_is_refused_without_being_stored(
        self, reason: str
    ) -> None:
        with pytest.raises(modal.ModalL4ContractError) as excinfo:
            modal.evaluate_credential_exposure_attestation(_attestation(reason=reason))
        assert reason not in str(excinfo.value)

    def test_non_boolean_confirmations_are_refused(self) -> None:
        with pytest.raises(modal.ModalL4ContractError, match="must be a boolean"):
            modal.evaluate_credential_exposure_attestation(
                _attestation(exposed_profile_credential_revocation_confirmed="yes")
            )

    def test_only_the_coordinator_may_confirm_revocation(self) -> None:
        with pytest.raises(modal.ModalL4ContractError, match="coordinator"):
            modal.evaluate_credential_exposure_attestation(
                _attestation(revocation_confirmed_by="the experiment itself")
            )

    def test_an_attestation_for_another_protocol_is_refused(self) -> None:
        with pytest.raises(modal.ModalL4ContractError, match="another protocol"):
            modal.evaluate_credential_exposure_attestation(
                _attestation(protocol_id="qwen3-8b-vllm-crossover-v2")
            )

    def test_the_gate_is_preregistered_in_the_plan(self) -> None:
        gate = modal.build_default_plan().to_dict()["credential_exposure_gate"]
        assert gate["absent_attestation"] == "refuse_provider_execution"
        assert gate["evaluated_before_provider_sdk_import"] is True
        assert gate["records_credential_values"] is False
        assert gate["records_credential_hashes_or_prefixes"] is False
        assert gate["records_screenshot_metadata"] is False
        assert gate["records_credential_derived_identifiers"] is False
        assert set(gate["required_true"]) == {
            "exposed_profile_credential_never_used_by_experiment",
            "exposed_profile_credential_revocation_confirmed",
            "fresh_local_profile_created_without_sharing",
        }
        assert "provider SDK import" in gate["blocks"]

    def test_the_offline_refusal_document_reports_the_block(self) -> None:
        document = modal.offline_plan_document()
        assert document["credential_exposure_gate"]["cleared"] is False
        assert document["exposed_profile_credential_never_used_by_experiment"] is True
        assert any("revocation" in blocker for blocker in document["blockers"])


class TestCredentialExposureAttestationTemplate:
    """Requirement 10: a boolean-only cleared attestation generator + template."""

    def test_the_generator_clears_the_gate_boolean_only(self) -> None:
        attestation = modal.build_credential_exposure_attestation(confirmed_at=NOW)
        verdict = modal.require_credential_exposure_cleared(attestation)
        assert verdict["cleared"] is True
        assert verdict["confirmed_by"] == "coordinator"
        assert verdict["records_credential_values"] is False
        assert (
            attestation["exposed_profile_credential_never_used_by_experiment"] is True
        )
        assert attestation["exposed_profile_credential_revocation_confirmed"] is True
        assert attestation["fresh_local_profile_created_without_sharing"] is True
        assert attestation["fresh_profile_shared_anywhere"] is False
        assert attestation["status"] == "cleared"
        # Every recorded confirmation is a boolean; nothing is a token, hash,
        # prefix, screenshot, or account identifier.
        booleans = {
            "exposed_profile_credential_never_used_by_experiment",
            "exposed_profile_credential_revocation_confirmed",
            "fresh_local_profile_created_without_sharing",
            "fresh_profile_shared_anywhere",
        }
        assert all(isinstance(attestation[name], bool) for name in booleans)
        assert not any(
            fragment in key.lower()
            for key in attestation
            for fragment in (
                "token",
                "secret",
                "hash",
                "sha256",
                "prefix",
                "screenshot",
            )
        )

    def test_the_generated_template_matches_the_closed_schema(self) -> None:
        attestation = modal.build_credential_exposure_attestation(confirmed_at=NOW)
        assert set(attestation) == set(modal.CREDENTIAL_EXPOSURE_ATTESTATION_FIELDS)
        assert attestation["kind"] == modal.CREDENTIAL_EXPOSURE_ATTESTATION_KIND
        assert attestation["protocol_id"] == modal.PROTOCOL_ID

    def test_a_malformed_confirmation_time_is_refused(self) -> None:
        with pytest.raises(modal.ModalL4ContractError):
            modal.build_credential_exposure_attestation(confirmed_at="not-a-timestamp")

    def test_the_cli_prints_a_cleared_template(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert modal.main(["attestation-template", "--confirmed-at", NOW]) == 0
        printed = json.loads(capsys.readouterr().out)
        assert modal.require_credential_exposure_cleared(printed)["cleared"] is True

    def test_the_cli_writes_a_cleared_template(self, tmp_path: Path) -> None:
        output = tmp_path / "attestation.json"
        assert (
            modal.main(
                ["attestation-template", "--confirmed-at", NOW, "--output", str(output)]
            )
            == 0
        )
        written = json.loads(output.read_text(encoding="utf-8"))
        assert modal.require_credential_exposure_cleared(written)["cleared"] is True

    def test_the_cli_refuses_a_malformed_confirmation_time(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert modal.main(["attestation-template", "--confirmed-at", "nope"]) == 1
        assert "llmtracefx-modal-l4-crossover" in capsys.readouterr().err
