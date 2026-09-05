"""Adversarial tests for the provider-specific Modal L4 result path.

Every fact a published result rests on is re-derived by ``analyze_modal_run``:
the orchestration seal, the header bindings, the exact sealed call sequence,
both memory canaries, the application ledger, the teardown, the sealed cell
inventory, and every wrapper and inner seal. These tests tamper with each in
turn and assert the run is refused, and that the reused statistical crossover
inference is emitted for a clean run.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer.lab.qwen3_8b import cloudrift_crossover_results as stats
from llmtracefx.optimizer.lab.qwen3_8b import cloudrift_runner as base_runner
from llmtracefx.optimizer.lab.qwen3_8b import modal_l4_crossover as modal
from llmtracefx.optimizer.lab.qwen3_8b import modal_l4_crossover_results as results
from llmtracefx.optimizer.lab.qwen3_8b import modal_l4_execute as execute

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _modal_result_fixture as fixture  # noqa: E402

CELL_IDS = [cell.cell_id for cell in modal.crossover_schedule()]


def _run(
    orchestration: dict[str, Any] | None = None,
    cells: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return results.analyze_modal_run(
        orchestration=orchestration or fixture.build_orchestration(),
        cells=cells or fixture.build_cells(),
    )


def _reseal_orch(orch: dict[str, Any]) -> dict[str, Any]:
    orch = dict(orch)
    orch.pop("orchestration_sha256", None)
    orch["orchestration_sha256"] = execute._sha256_json(orch)
    return orch


def _reseal_inner(inner: dict[str, Any]) -> dict[str, Any]:
    inner.pop("cell_sha256", None)
    inner["cell_sha256"] = stats._sha256_json(inner)
    return inner


def _reseal_wrapper(wrapper: dict[str, Any]) -> dict[str, Any]:
    wrapper.pop("receipt_sha256", None)
    return base_runner._seal(wrapper, "receipt_sha256")


class TestValidCompletedRun:
    def test_a_valid_run_analyzes_with_reused_primitives(self) -> None:
        result = _run()
        assert result["cell_count"] == 32
        assert result["pair_count"] == 16
        assert result["claims_cloudrift_or_host_cache_proof"] is False
        assert any(
            name.endswith("_analysis_document")
            for name in result["reused_statistical_primitives"]
        )
        assert any(
            name.endswith("analyze_pair_curves")
            for name in result["reused_statistical_primitives"]
        )

    def test_the_crossover_inference_is_emitted_for_all_16_pairs(self) -> None:
        result = _run()
        records = result["pair_records"]
        assert len(records) == 16
        for record in records:
            assert record["lane"] in modal.LANES
            assert record["order"] in ("eager-compiled", "compiled-eager")
            assert record["period_indices"] == [1, 2]
            assert isinstance(record["block_index"], int)
            assert "pair_effects" in record
        controlled = result["crossover_inference"]["controlled"]
        assert controlled["aggregate_first_crossing"]["state"] == "observed"
        assert controlled["simultaneous_band_sustained_crossing"]["state"] == (
            "observed"
        )
        assert controlled["resample_count"] == stats.BOOTSTRAP_RESAMPLES == 20_000
        assert controlled["request_level_resampling"] is False
        assert 0.0 <= controlled["terminal_effect_sign_flip_p_value"] <= 1.0
        assert len(controlled["pair_effects"]) == 8
        assert "natural_timing" in controlled
        distributions = result["crossover_inference"]["pair_effect_distributions"]
        assert set(distributions["lanes"]) == set(modal.LANES)

    def test_the_supported_crossover_claim_requires_a_crossing(self) -> None:
        claims = {c["claim_id"]: c for c in _run()["claim_matrix"]["claims"]}
        crossover = claims["fixed-token-count-provider-conditioned-crossover"]
        assert crossover["state"] == "supported"
        assert crossover["blockers"] == []

    def test_identity_and_quality_claims_are_separate(self) -> None:
        claims = {c["claim_id"]: c["state"] for c in _run()["claim_matrix"]["claims"]}
        assert claims["output-identical-generation-crossover"] == "supported"
        assert claims["numerically-reproducible-generation"] == "supported"
        assert claims["natural-output-quality-preserved"] == "supported"

    def test_causal_and_cache_claims_are_unsupported_by_construction(self) -> None:
        claims = {c["claim_id"]: c["state"] for c in _run()["claim_matrix"]["claims"]}
        assert claims["pure-causal-compilation-effect"] == "unsupported"
        assert claims["natural-end-to-end-causal-speedup"] == "unsupported"
        assert claims["cache-state-controlled-comparison"] == "unsupported"
        assert claims["compile-cuda-graph-component-timing"] == "unsupported"
        for blocked in modal.BLOCKED_CLAIM_IDS:
            assert claims[blocked] == "unsupported"


class TestClaimSemantics:
    def test_output_identity_alone_does_not_support_the_timing_crossover(self) -> None:
        result = _run(cells=fixture.build_cells(flat=True))
        claims = {c["claim_id"]: c for c in result["claim_matrix"]["claims"]}
        crossover = claims["fixed-token-count-provider-conditioned-crossover"]
        assert crossover["state"] == "unsupported"
        assert "no_statistically_supported_controlled_crossing" in crossover["blockers"]
        # Output identity holds under flat timing, but a crossover claim also
        # needs a statistically supported crossing, so the output-identical
        # crossover is unsupported too. Only the standalone numeric
        # reproducibility claim (which never implies a crossover) stays
        # supported.
        identical = claims["output-identical-generation-crossover"]
        assert identical["state"] == "unsupported"
        assert "no_statistically_supported_controlled_crossing" in identical["blockers"]
        assert claims["numerically-reproducible-generation"]["state"] == "supported"

    def test_construction_blocked_claims_are_never_supported(self) -> None:
        result = _run()
        for claim in result["claim_matrix"]["claims"]:
            if claim["claim_id"] in modal.BLOCKED_CLAIM_IDS:
                assert claim["state"] != "supported"


class TestOrchestrationSeal:
    def test_a_missing_seal_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        del orch["orchestration_sha256"]
        with pytest.raises(results.ModalL4ResultsError, match="orchestration seal"):
            _run(orchestration=orch)

    def test_a_tampered_field_without_resealing_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        orch["source_head"] = "f" * 40
        with pytest.raises(results.ModalL4ResultsError, match="seal does not verify"):
            _run(orchestration=orch)

    def test_relying_on_published_is_not_enough(self) -> None:
        orch = fixture.build_orchestration()
        orch["ledger"]["reserved_usd"] = "999"
        orch = _reseal_orch(orch)
        assert orch["published"] is True
        with pytest.raises(results.ModalL4ResultsError, match="ledger"):
            _run(orchestration=orch)


class TestEnvelopeGates:
    def test_a_plan_mismatch_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        orch["plan_sha256"] = "sha256:" + "0" * 64
        orch = _reseal_orch(orch)
        with pytest.raises(results.ModalL4ResultsError, match="different plan"):
            _run(orchestration=orch)

    def test_a_runtime_image_digest_claim_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        orch["runtime_image"]["derived_provider_image_digest"] = "sha256:" + "a" * 64
        orch = _reseal_orch(orch)
        with pytest.raises(results.ModalL4ResultsError, match="runtime image"):
            _run(orchestration=orch)

    def test_a_non_result_kind_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        orch["kind"] = "llmtracefx.modal_l4_crossover.refusal"
        orch = _reseal_orch(orch)
        with pytest.raises(results.ModalL4ResultsError, match="not a result"):
            _run(orchestration=orch)

    def test_a_bad_source_head_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        orch["source_head"] = "not-a-commit"
        orch = _reseal_orch(orch)
        with pytest.raises(results.ModalL4ResultsError, match="source head"):
            _run(orchestration=orch)


class TestCallSequenceGate:
    def test_a_reordered_call_sequence_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        executed = orch["call_sequence_executed"]
        executed[0], executed[1] = executed[1], executed[0]
        orch = _reseal_orch(orch)
        with pytest.raises(results.ModalL4ResultsError, match="call sequence"):
            _run(orchestration=orch)

    def test_a_second_attempt_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        orch["call_sequence_executed"][0]["attempt"] = 2
        orch = _reseal_orch(orch)
        with pytest.raises(results.ModalL4ResultsError, match="call sequence"):
            _run(orchestration=orch)

    def test_a_missing_lifecycle_receipt_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        orch["attempt_receipts"] = orch["attempt_receipts"][:-1]
        orch = _reseal_orch(orch)
        with pytest.raises(results.ModalL4ResultsError, match="attempt"):
            _run(orchestration=orch)

    def test_a_desynced_adjudication_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        orch["attempt_adjudication"]["action"] = "tampered"
        orch = _reseal_orch(orch)
        with pytest.raises(
            results.ModalL4ResultsError, match="adjudication does not recompute"
        ):
            _run(orchestration=orch)


class TestMemoryCanaries:
    def test_a_tampered_canary_seal_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        orch["memory_gate"]["canaries"][0]["receipt"]["observation"][
            "peak_vram_mib"
        ] = 1
        orch = _reseal_orch(orch)
        with pytest.raises(results.ModalL4ResultsError, match="canary receipt seal"):
            _run(orchestration=orch)

    def test_a_failed_canary_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        entry = orch["memory_gate"]["canaries"][0]
        receipt = {k: v for k, v in entry["receipt"].items() if k != "receipt_sha256"}
        receipt["observation"]["out_of_memory"] = True
        entry["receipt"] = base_runner._seal(receipt, "receipt_sha256")
        orch = _reseal_orch(orch)
        with pytest.raises(results.ModalL4ResultsError, match="did not pass|recompute"):
            _run(orchestration=orch)

    def test_a_tuned_memory_gate_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        orch["memory_gate"]["tuning_applied"] = True
        orch = _reseal_orch(orch)
        with pytest.raises(results.ModalL4ResultsError, match="tuning"):
            _run(orchestration=orch)

    def test_a_missing_mode_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        orch["memory_gate"]["canaries"][1] = fixture._memory_gate_entry(
            "eager", index=9
        )
        orch = _reseal_orch(orch)
        with pytest.raises(results.ModalL4ResultsError, match="missing a canary mode"):
            _run(orchestration=orch)


class TestLedgerGate:
    def test_a_tampered_ledger_seal_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        orch["ledger"]["remaining_usd"] = "0"
        orch = _reseal_orch(orch)
        with pytest.raises(results.ModalL4ResultsError, match="ledger does not verify"):
            _run(orchestration=orch)

    def test_a_ledger_claiming_provider_proof_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        ledger = {k: v for k, v in orch["ledger"].items() if k != "ledger_sha256"}
        ledger["is_provider_proof"] = True
        orch["ledger"] = modal._seal(ledger)
        orch = _reseal_orch(orch)
        with pytest.raises(results.ModalL4ResultsError, match="ledger does not verify"):
            _run(orchestration=orch)

    def test_a_ledger_bound_to_a_different_nonce_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        orch["ledger"] = modal.build_completed_ledger_document(
            plan=modal.build_default_plan(),
            source_head=fixture.SOURCE_HEAD,
            experiment_nonce="a" * 32,
            ledger_path_sha256=fixture.LEDGER_PATH_SHA256,
            reserved_at=fixture.RESERVED_AT,
            completed_at=fixture.COMPLETED_AT,
        )
        orch = _reseal_orch(orch)
        with pytest.raises(results.ModalL4ResultsError, match="ledger does not verify"):
            _run(orchestration=orch)

    def test_an_incomplete_ledger_lifecycle_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        ledger = {k: v for k, v in orch["ledger"].items() if k != "ledger_sha256"}
        ledger["events"] = ledger["events"][:-2]
        ledger["revision"] = len(ledger["events"])
        orch["ledger"] = modal._seal(ledger)
        orch = _reseal_orch(orch)
        with pytest.raises(results.ModalL4ResultsError, match="ledger does not verify"):
            _run(orchestration=orch)


class TestTeardownGate:
    def test_an_incomplete_teardown_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        orch["teardown"]["volume_deleted"] = False
        orch["teardown"]["adjudication"] = modal.evaluate_teardown_receipt(
            {k: v for k, v in orch["teardown"].items() if k != "adjudication"}
        )
        orch = _reseal_orch(orch)
        with pytest.raises(results.ModalL4ResultsError, match="teardown is incomplete"):
            _run(orchestration=orch)

    def test_a_desynced_teardown_adjudication_is_refused(self) -> None:
        orch = fixture.build_orchestration()
        orch["teardown"]["adjudication"]["storage_allowance_days"] = 999
        orch = _reseal_orch(orch)
        with pytest.raises(
            results.ModalL4ResultsError, match="adjudication does not recompute"
        ):
            _run(orchestration=orch)


class TestCellInventory:
    def test_an_extra_cell_is_refused(self) -> None:
        cells = fixture.build_cells()
        cells["unexpected-cell"] = cells[CELL_IDS[0]]
        with pytest.raises(results.ModalL4ResultsError, match="no extras"):
            _run(cells=cells)

    def test_a_missing_cell_is_refused(self) -> None:
        cells = fixture.build_cells()
        del cells[CELL_IDS[0]]
        with pytest.raises(results.ModalL4ResultsError, match="no extras|is missing"):
            _run(cells=cells)

    def test_a_tampered_wrapper_seal_is_refused(self) -> None:
        cells = fixture.build_cells()
        cells[CELL_IDS[0]]["container_identity_sha256"] = "sha256:" + "9" * 64
        with pytest.raises(results.ModalL4ResultsError, match="wrapper seal"):
            _run(cells=cells)

    def test_a_reused_container_identity_is_refused(self) -> None:
        cells = fixture.build_cells()
        shared = cells[CELL_IDS[0]]["container_identity_sha256"]
        cells[CELL_IDS[1]]["container_identity_sha256"] = shared
        cells[CELL_IDS[1]] = _reseal_wrapper(cells[CELL_IDS[1]])
        with pytest.raises(results.ModalL4ResultsError, match="reused a container"):
            _run(cells=cells)

    def test_a_non_l4_wrapper_is_refused(self) -> None:
        cells = fixture.build_cells()
        cells[CELL_IDS[0]]["provider_hardware"]["gpu_name"] = "NVIDIA A10G"
        cells[CELL_IDS[0]] = _reseal_wrapper(cells[CELL_IDS[0]])
        with pytest.raises(results.ModalL4ResultsError, match="approved L4"):
            _run(cells=cells)

    def test_a_tampered_inner_seal_is_refused(self) -> None:
        cells = fixture.build_cells()
        cells[CELL_IDS[0]]["cell_receipt"]["measurements"]["initialization_seconds"][
            "value"
        ] = 999.0
        cells[CELL_IDS[0]] = _reseal_wrapper(cells[CELL_IDS[0]])
        with pytest.raises(stats.CrossoverResultsError, match="does not verify"):
            _run(cells=cells)

    def test_a_shared_cache_scope_is_refused(self) -> None:
        cells = fixture.build_cells()
        collide = "sha256:" + "c" * 64
        for cell_id in (CELL_IDS[0], CELL_IDS[1]):
            inner = cells[cell_id]["cell_receipt"]
            inner["deterministic_environment"]["cache_root_role"][
                "path_sha256"
            ] = collide
            _reseal_inner(inner)
            cells[cell_id] = _reseal_wrapper(cells[cell_id])
        with pytest.raises(results.ModalL4ResultsError, match="cache directory"):
            _run(cells=cells)


class TestHardwarePlacement:
    def test_placement_is_reported_not_called_continuity(self) -> None:
        placement = _run()["hardware_placement"]
        assert "continuity" not in placement
        assert placement["distinct_gpu_identity_commitments"] == 1
        assert placement["single_shared_placement"] is True
        assert placement["placement_controlled"] is False
        assert placement["hardware_matched_or_causal_claims_supported"] is False
        assert placement["raw_gpu_identity_exposed"] is False
        assert placement["observed_driver_versions"] == [fixture.DRIVER]

    def test_no_raw_gpu_identity_is_persisted(self) -> None:
        result = _run()
        blob = stats._sha256_json(result)
        assert "gpu_uuid_sha256" not in blob
        commitment = fixture._l4_commitment()["gpu_identity_commitment"]
        assert commitment not in blob

    def test_a_cell_bound_to_a_different_nonce_is_refused(self) -> None:
        cells = fixture.build_cells()
        inner = cells[CELL_IDS[0]]["cell_receipt"]
        inner["hardware_commitment"]["public_experiment_nonce"] = "f" * 32
        _reseal_inner(inner)
        cells[CELL_IDS[0]] = _reseal_wrapper(cells[CELL_IDS[0]])
        with pytest.raises(
            results.ModalL4ResultsError, match="bound to a different experiment"
        ):
            _run(cells=cells)

    def test_differing_placement_does_not_invalidate_but_is_reported(self) -> None:
        cells = fixture.build_cells()
        inner = cells[CELL_IDS[0]]["cell_receipt"]
        inner["hardware_commitment"]["gpu_identity_commitment"] = "sha256:" + "b" * 64
        _reseal_inner(inner)
        cells[CELL_IDS[0]] = _reseal_wrapper(cells[CELL_IDS[0]])
        placement = _run(cells=cells)["hardware_placement"]
        assert placement["distinct_gpu_identity_commitments"] >= 2
        assert placement["single_shared_placement"] is False
        assert placement["hardware_matched_or_causal_claims_supported"] is False

    def test_a_cell_with_drifted_runtime_pins_is_refused(self) -> None:
        cells = fixture.build_cells()
        inner = cells[CELL_IDS[0]]["cell_receipt"]
        inner["runtime"]["pins"] = {"vllm_version": "9.9.9"}
        _reseal_inner(inner)
        cells[CELL_IDS[0]] = _reseal_wrapper(cells[CELL_IDS[0]])
        with pytest.raises(results.ModalL4ResultsError, match="runtime pins"):
            _run(cells=cells)


def test_the_module_imports_no_provider_sdk() -> None:
    source = Path(results.__file__).read_text(encoding="utf-8")
    assert "import modal\n" not in source
    assert "from modal" not in source
