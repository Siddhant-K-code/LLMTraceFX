"""Tests for Modal L4 crossover preregistration and result evidence."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer.lab.qwen3_8b import modal_l4_crossover as modal
from llmtracefx.optimizer.lab.qwen3_8b import modal_l4_crossover_evidence as evidence
from llmtracefx.optimizer.lab.qwen3_8b import modal_l4_crossover_results as results

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _modal_result_fixture as fixture  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NOW = "2026-09-04T19:52:50.511+05:30"
_SEALED_VALIDATE_DECODE_FEASIBILITY = results._validate_decode_feasibility


def _accept_hypothetical_feasibility(
    orchestration: dict[str, Any],
) -> dict[str, Any]:
    receipt = orchestration.get("decode_feasibility")
    if not isinstance(receipt, dict):
        raise results.ModalL4ResultsError(
            "orchestration decode-feasibility verdict is missing"
        )
    return dict(receipt)


@pytest.fixture(autouse=True)
def _exercise_dormant_result_bundle_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test result plumbing without making this protocol result-eligible."""

    monkeypatch.setattr(
        results, "_validate_decode_feasibility", _accept_hypothetical_feasibility
    )


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    output = tmp_path / "bundle"
    evidence.build_offline_bundle(output, repo_root=ROOT)
    return output


def _memory_gate() -> dict[str, Any]:
    def observation(mode: str) -> dict[str, Any]:
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

    return {
        "tuning_applied": False,
        "canaries": [observation("eager"), observation("compiled")],
    }


_DEFAULT_RESULT_ARTIFACTS: dict[str, bytes] | None = None


def _result_artifacts_for(
    orch: dict[str, Any], cell_map: dict[str, Any]
) -> dict[str, bytes]:
    analysis = evidence.analyze_modal_run(orchestration=orch, cells=cell_map)
    return evidence._result_artifact_documents(analysis)


def _result_envelope(
    root: Path,
    *,
    orchestration: dict[str, Any] | None = None,
    cells: dict[str, Any] | None = None,
    **overrides: Any,
) -> Path:
    """Materialize a complete, valid result bundle, then apply any overrides.

    The clean bundle carries the sealed orchestration receipt, 32 cell
    receipts, every standalone envelope projected from the orchestration, the
    four regenerated artifacts (analysis, claim matrix, report, figure), and a
    matching SHA256SUMS. ``overrides`` replace (or, when ``None``, drop)
    individual top-level files; the checksum manifest is recomputed over
    whatever remains unless ``SHA256SUMS`` is itself overridden, so an envelope
    tamper is caught by its own binding rather than by the checksum.
    """

    global _DEFAULT_RESULT_ARTIFACTS
    root.mkdir(parents=True, exist_ok=True)
    orch = orchestration if orchestration is not None else fixture.build_orchestration()
    cell_map = cells if cells is not None else fixture.build_cells()
    reason = "coordinator confirmed revocation and fresh local profile"

    if orchestration is None and cells is None:
        if _DEFAULT_RESULT_ARTIFACTS is None:
            _DEFAULT_RESULT_ARTIFACTS = _result_artifacts_for(orch, cell_map)
        artifacts = dict(_DEFAULT_RESULT_ARTIFACTS)
    else:
        artifacts = _result_artifacts_for(orch, cell_map)

    # Every standalone envelope file is a redundant projection of the sealed
    # orchestration content, so a clean bundle binds exactly.
    envelopes: dict[str, Any] = {
        "application-ledger.json": orch["ledger"],
        "credential-exposure.json": {**orch["credential_exposure"], "reason": reason},
        "decode-feasibility.json": orch["decode_feasibility"],
        "memory-gate.json": {
            "tuning_applied": orch["memory_gate"]["tuning_applied"],
            "canaries": [
                canary["receipt"]["observation"]
                for canary in orch["memory_gate"]["canaries"]
            ],
        },
        "modal-attempt-receipts.json": {"receipts": orch["attempt_receipts"]},
        "modal-limitations.json": {"uncontrolled": orch["uncontrolled_limitations"]},
        "modal-rate-receipt.json": orch["rate_receipt"],
        "modal-rate-refresh.json": orch["rate_refresh"],
        "modal-teardown.json": {
            key: value
            for key, value in orch["teardown"].items()
            if key != "adjudication"
        },
        "profile-authentication.json": orch["profile_authentication"],
        "source-checkout.json": orch["source_checkout"],
    }

    files: dict[str, bytes] = {
        evidence.ORCHESTRATION_FILE: (json.dumps(orch, indent=2) + "\n").encode("utf-8")
    }
    for name, value in envelopes.items():
        files[name] = (json.dumps(value, indent=2) + "\n").encode("utf-8")
    files.update(artifacts)

    cells_override_present = "cells" in overrides
    for name, value in overrides.items():
        if name in ("cells", evidence.RESULT_CHECKSUM_FILE):
            continue
        if value is None:
            files.pop(name, None)
        elif isinstance(value, (bytes, bytearray)):
            files[name] = bytes(value)
        elif isinstance(value, str):
            files[name] = value.encode("utf-8")
        else:
            files[name] = (json.dumps(value, indent=2) + "\n").encode("utf-8")

    if evidence.RESULT_CHECKSUM_FILE in overrides:
        override = overrides[evidence.RESULT_CHECKSUM_FILE]
        if override is not None:
            files[evidence.RESULT_CHECKSUM_FILE] = (
                override.encode("utf-8") if isinstance(override, str) else override
            )
    else:
        checksum_lines = "\n".join(
            f"{evidence._sha256(files[name])}  {name}"
            for name in evidence.RESULT_HASHED_FILES
            if name in files
        )
        files[evidence.RESULT_CHECKSUM_FILE] = (checksum_lines + "\n").encode("utf-8")

    for name, data in files.items():
        (root / name).write_bytes(data)

    cell_source = overrides["cells"] if cells_override_present else cell_map
    if cell_source is not None:
        cells_dir = root / "cells"
        cells_dir.mkdir(exist_ok=True)
        for cell_id, wrapper in cell_source.items():
            (cells_dir / f"{cell_id}.json").write_text(
                json.dumps(wrapper, indent=2) + "\n", encoding="utf-8"
            )
    return root


def _execution_workspace(
    root: Path,
    *,
    orchestration: dict[str, Any] | None = None,
    cells: dict[str, Any] | None = None,
    reason: str = "coordinator confirmed revocation and fresh local profile",
) -> Path:
    """Materialize the workspace a published ``execute()`` run leaves behind.

    ``build_result_bundle`` consumes exactly this: the sealed orchestration
    receipt, the 32 sealed cell receipts, and the credential-exposure verdict
    that carries the coordinator's confirmation reason.
    """

    root.mkdir(parents=True, exist_ok=True)
    orch = orchestration if orchestration is not None else fixture.build_orchestration()
    (root / "orchestration-receipt.json").write_text(
        json.dumps(orch, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cells_dir = root / "cells"
    cells_dir.mkdir(exist_ok=True)
    for cell_id, wrapper in (cells or fixture.build_cells()).items():
        (cells_dir / f"{cell_id}.json").write_text(
            json.dumps(wrapper, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    exposure = {**orch["credential_exposure"], "reason": reason}
    (root / "credential-exposure.json").write_text(
        json.dumps(exposure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


class TestOfflineBundle:
    def test_build_then_verify_is_deterministic(self, bundle: Path) -> None:
        first = {path.name: path.read_bytes() for path in bundle.iterdir()}
        evidence.build_offline_bundle(bundle, repo_root=ROOT)
        second = {path.name: path.read_bytes() for path in bundle.iterdir()}
        assert first == second
        assert set(first) == set(evidence.BUNDLE_FILES)
        evidence.verify_offline_bundle(bundle, repo_root=ROOT)

    def test_refusal_semantics_are_published(self, bundle: Path) -> None:
        preflight = json.loads(
            (bundle / "offline-preflight.json").read_text(encoding="utf-8")
        )
        claims = json.loads((bundle / "claim-matrix.json").read_text(encoding="utf-8"))
        assert preflight["execution_authorized"] is False
        assert preflight["provider_sdk_imported"] is False
        assert preflight["spend_usd"] == "0"
        assert claims["execution_state"] == "not_run"
        states = {item["claim_id"]: item["state"] for item in claims["claims"]}
        assert states["zero-spend-offline-generation"] == "supported"
        assert states["no-provider-authentication"] == "supported"
        assert states["provider-reported-spend-within-hard-cap"] == "unsupported"
        blocked_state = modal.UNSUPPORTED_BY_CONSTRUCTION_STATE
        assert states["cache-state-controlled-comparison"] == blocked_state
        for claim_id in modal.BLOCKED_CLAIM_IDS:
            assert states[claim_id] == blocked_state

    def test_the_offline_matrix_uses_the_canonical_claim_identifiers(
        self, bundle: Path
    ) -> None:
        """Preregistered and result claim identifiers are the same strings.

        Every measured claim (including the memory gate) and every claim
        unsupported by construction (including the causal and hardware-matched
        ones) appears in both matrices under one identifier, so a claim is
        traceable from preregistration to result without a translation table.
        """

        claims = json.loads((bundle / "claim-matrix.json").read_text(encoding="utf-8"))
        published = sorted(item["claim_id"] for item in claims["claims"])
        assert published == sorted(modal.PREREGISTERED_CLAIM_IDS)
        assert claims["claim_ids"] == list(modal.PREREGISTERED_CLAIM_IDS)
        assert claims["result_claim_ids"] == list(modal.RESULT_CLAIM_IDS)
        assert set(modal.RESULT_CLAIM_IDS) <= set(published)
        assert "memory-gate-passed" in published
        assert "hardware-matched-comparison" in published
        assert "pure-causal-compilation-effect" in published

    def test_the_decode_feasibility_refusal_is_published(self, bundle: Path) -> None:
        """The infeasibility proof ships with the preregistration, not just in code."""

        verdict = json.loads(
            (bundle / "decode-feasibility.json").read_text(encoding="utf-8")
        )
        preflight = json.loads(
            (bundle / "offline-preflight.json").read_text(encoding="utf-8")
        )
        plan = json.loads((bundle / "experiment-plan.json").read_text(encoding="utf-8"))
        assert verdict == modal.evaluate_decode_bandwidth_feasibility()
        assert verdict["feasible"] is False
        assert verdict["derivation"]["minimum_decode_only_seconds"] == "754.86029303808"
        assert verdict["derivation"]["required_tokens_per_second"] == "28.8"
        assert preflight["decode_feasibility"] == verdict
        assert preflight["execution_refused_offline"] is True
        assert plan["decode_feasibility"] == verdict
        claims = json.loads((bundle / "claim-matrix.json").read_text(encoding="utf-8"))
        states = {item["claim_id"]: item for item in claims["claims"]}
        feasible_claim = states["controlled-cell-decode-feasible-on-l4"]
        assert feasible_claim["state"] == "unsupported"
        assert feasible_claim["evidence"] == "decode-feasibility.json"

    def test_the_bundle_readme_states_the_arithmetic(self, bundle: Path) -> None:
        readme = (bundle / "README.md").read_text(encoding="utf-8")
        assert "16,381,516,776-byte safetensors" in readme
        assert "300,000,000,000 bytes per second" in readme
        assert "754.86029303808 seconds" in readme
        assert "480-second" in readme
        assert "28.8 tokens per second" in readme

    def test_budget_chain_reconciles_to_the_hard_cap(self, bundle: Path) -> None:
        budget = json.loads((bundle / "budget-plan.json").read_text(encoding="utf-8"))
        chain = budget["budget_chain"]
        assert chain["compute_planned_usd"] == "4.5985056"
        assert chain["storage_planned_usd"] == "0.48"
        assert chain["total_planned_usd"] == "5.0785056"
        assert chain["untouched_margin_usd"] == "0.9214944"
        assert chain["hard_cap_usd"] == "6"
        assert budget["application_ledger_is_provider_proof"] is False
        assert budget["provider_reported_spend_usd"] is None

    def test_result_contract_is_preregistered_and_fail_closed(
        self, bundle: Path
    ) -> None:
        contract = json.loads(
            (bundle / "result-contract.json").read_text(encoding="utf-8")
        )
        assert contract["fail_closed"] is True
        assert contract["provider_native_results_verifier"] == (
            evidence.DELEGATED_STATISTICAL_VERIFIER
        )
        assert contract["orchestration_file"] == "orchestration-receipt.json"
        assert contract["cells_directory"] == "cells"
        assert any(
            name.endswith("_compute_pair_effects")
            for name in contract["reused_statistical_primitives"]
        )
        assert contract["invalidating_observations"] == list(
            modal.INVALIDATING_OBSERVATIONS
        )

    def test_execution_surface_is_preregistered(self, bundle: Path) -> None:
        contract = json.loads(
            (bundle / "evidence-contract.json").read_text(encoding="utf-8")
        )
        surface = contract["execution_surface"]
        assert surface["provider_sdk"]["tested_version"] == modal.TESTED_MODAL_VERSION
        assert surface["provider_sdk"]["forbidden_web_decorators"]
        assert len(surface["functions"]) == len(modal.FUNCTION_SPECS)
        assert len(surface["call_sequence"]) == 37
        assert surface["statistical_publication"]["accepts_modal_workspace"] is False
        assert surface["authorization"]["authentication"] == (
            "openssh_detached_signature"
        )
        assert "modal_l4_app" in surface["provider_module"]

    def test_protocol_sources_bind_the_execution_modules(self, bundle: Path) -> None:
        sources = json.loads(
            (bundle / "protocol-sources.json").read_text(encoding="utf-8")
        )
        paths = {item["path"] for item in sources["files"]}
        for name in (
            "modal_l4_app.py",
            "modal_l4_cell_runner.py",
            "modal_l4_execute.py",
            "modal_l4_rates.py",
        ):
            assert f"llmtracefx/optimizer/lab/qwen3_8b/{name}" in paths

    def test_protocol_sources_bind_the_implementation(self, bundle: Path) -> None:
        sources = json.loads(
            (bundle / "protocol-sources.json").read_text(encoding="utf-8")
        )
        paths = [item["path"] for item in sources["files"]]
        assert "llmtracefx/optimizer/lab/qwen3_8b/modal_l4_crossover.py" in paths
        assert sources["reused_statistical_core"] == (
            evidence.DELEGATED_STATISTICAL_VERIFIER
        )

    @pytest.mark.parametrize(
        "name",
        (
            "README.md",
            "budget-plan.json",
            "claim-matrix.json",
            "evidence-contract.json",
            "experiment-plan.json",
            "offline-preflight.json",
            "report.html",
            "result-contract.json",
        ),
    )
    def test_any_tampered_document_fails_verification(
        self, bundle: Path, name: str
    ) -> None:
        path = bundle / name
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with pytest.raises(evidence.ModalL4EvidenceError):
            evidence.verify_offline_bundle(bundle, repo_root=ROOT)

    def test_missing_or_extra_file_fails_verification(self, bundle: Path) -> None:
        (bundle / "extra.json").write_text("{}\n", encoding="utf-8")
        with pytest.raises(evidence.ModalL4EvidenceError, match="file set differs"):
            evidence.verify_offline_bundle(bundle, repo_root=ROOT)
        (bundle / "extra.json").unlink()
        (bundle / "report.html").unlink()
        with pytest.raises(evidence.ModalL4EvidenceError, match="file set differs"):
            evidence.verify_offline_bundle(bundle, repo_root=ROOT)

    def test_private_content_is_refused(self, bundle: Path) -> None:
        (bundle / "README.md").write_text(
            "/Users/someone/secret-run\n", encoding="utf-8"
        )
        with pytest.raises(evidence.ModalL4EvidenceError, match="private home path"):
            evidence.verify_offline_bundle(bundle, repo_root=ROOT)

    def test_build_refuses_a_directory_with_foreign_files(self, tmp_path: Path) -> None:
        output = tmp_path / "bundle"
        output.mkdir()
        (output / "notes.txt").write_text("hello\n", encoding="utf-8")
        with pytest.raises(evidence.ModalL4EvidenceError, match="unexpected files"):
            evidence.build_offline_bundle(output, repo_root=ROOT)

    def test_checksum_drift_is_refused(self, bundle: Path) -> None:
        (bundle / "SHA256SUMS").write_text("bogus line\n", encoding="utf-8")
        with pytest.raises(evidence.ModalL4EvidenceError, match="SHA256SUMS differs"):
            evidence.verify_offline_bundle(bundle, repo_root=ROOT)

    @pytest.mark.parametrize(
        "url",
        (
            "http://modal.com/pricing",
            "https://pricing.example.com/modal",
            "https://modal.com.evil.test/pricing",
        ),
    )
    def test_provenance_domains_are_closed(self, url: str) -> None:
        with pytest.raises(evidence.ModalL4EvidenceError):
            evidence._require_provenance_domain(url, field="methodology source")

    def test_cli_build_and_verify(self, tmp_path: Path) -> None:
        output = tmp_path / "bundle"
        assert (
            evidence.main(
                ["build", "--output-dir", str(output), "--repo-root", str(ROOT)]
            )
            == 0
        )
        assert (
            evidence.main(
                ["verify", "--bundle-dir", str(output), "--repo-root", str(ROOT)]
            )
            == 0
        )

    def test_cli_reports_failure_without_a_traceback(
        self, capsys: pytest.CaptureFixture[str], tmp_path: Path
    ) -> None:
        assert (
            evidence.main(
                [
                    "verify",
                    "--bundle-dir",
                    str(tmp_path / "absent"),
                    "--repo-root",
                    str(ROOT),
                ]
            )
            == 1
        )
        assert "Modal L4 crossover evidence failed" in capsys.readouterr().err


class TestResultContract:
    def test_statistical_core_is_delegated_not_duplicated(self) -> None:
        # The statistics are no longer delegated to the CloudRift bundle
        # verifier (which needs receipts a Modal run cannot produce); the
        # provider-neutral primitives are reused directly instead.
        assert evidence.DELEGATED_STATISTICAL_VERIFIER.endswith(
            "modal_l4_crossover_results.analyze_modal_run"
        )
        assert any(
            name.endswith("_compute_pair_effects")
            for name in evidence.REUSED_STATISTICAL_PRIMITIVE_NAMES
        )
        assert any(
            name.endswith("_validate_request")
            for name in evidence.REUSED_STATISTICAL_PRIMITIVE_NAMES
        )

    def test_dormant_hypothetical_result_analysis_is_internally_consistent(
        self, tmp_path: Path
    ) -> None:
        root = _result_envelope(tmp_path / "results")
        result = evidence.verify_result_bundle(root)
        assert result["verified"] is True
        assert result["pair_count"] == 16
        assert result["provider_reported_spend_usd"] is None
        supported = {
            claim["claim_id"]
            for claim in result["claim_matrix"]["claims"]
            if claim["state"] == "supported"
        }
        assert "fixed-token-count-provider-conditioned-crossover" in supported
        assert "pure-causal-compilation-effect" not in supported
        for blocked in modal.BLOCKED_CLAIM_IDS:
            assert blocked not in supported

    def test_public_verifier_rejects_any_result_for_the_infeasible_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = _result_envelope(tmp_path / "results")
        monkeypatch.setattr(
            results,
            "_validate_decode_feasibility",
            _SEALED_VALIDATE_DECODE_FEASIBILITY,
        )
        with pytest.raises(
            evidence.ModalL4EvidenceError, match="differs from the sealed plan"
        ):
            evidence.verify_result_bundle(root)

    def test_missing_envelope_document_fails_before_statistics(
        self, tmp_path: Path
    ) -> None:
        root = _result_envelope(tmp_path / "results")
        (root / "modal-teardown.json").unlink()
        with pytest.raises(
            evidence.ModalL4EvidenceError, match="missing modal-teardown.json"
        ):
            evidence.verify_result_bundle(root)

    @pytest.mark.parametrize(
        ("override_index", "field", "value", "match"),
        (
            (0, "attempt", 2, "invalidated"),
            (0, "crashed", True, "invalidated"),
            (0, "preempted", True, "invalidated"),
            (0, "timed_out", True, "invalidated"),
            (0, "terminal_receipt", False, "invalidated"),
        ),
    )
    def test_attempt_observations_invalidate_the_run(
        self,
        tmp_path: Path,
        override_index: int,
        field: str,
        value: Any,
        match: str,
    ) -> None:
        receipts = {
            "receipts": [
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
        }
        receipts["receipts"][override_index][field] = value
        root = _result_envelope(
            tmp_path / "results", **{"modal-attempt-receipts.json": receipts}
        )
        with pytest.raises(evidence.ModalL4EvidenceError, match=match):
            evidence.verify_result_bundle(root)

    def test_a_rate_increase_refuses_publication(self, tmp_path: Path) -> None:
        root = _result_envelope(tmp_path / "results")
        receipt = json.loads(
            (root / "modal-rate-receipt.json").read_text(encoding="utf-8")
        )
        receipt["rates"]["l4_gpu_second"] = "0.0005"
        (root / "modal-rate-receipt.json").write_text(
            json.dumps(receipt), encoding="utf-8"
        )
        with pytest.raises(modal.ModalL4ContractError, match="exceed the committed"):
            evidence.verify_result_bundle(root)

    def test_failed_or_tuned_memory_gate_refuses_publication(
        self, tmp_path: Path
    ) -> None:
        gate = _memory_gate()
        gate["canaries"][1]["out_of_memory"] = True
        root = _result_envelope(tmp_path / "failed", **{"memory-gate.json": gate})
        with pytest.raises(evidence.ModalL4EvidenceError, match="memory gate failed"):
            evidence.verify_result_bundle(root)
        tuned = _memory_gate()
        tuned["tuning_applied"] = True
        root = _result_envelope(tmp_path / "tuned", **{"memory-gate.json": tuned})
        with pytest.raises(evidence.ModalL4EvidenceError, match="tuning"):
            evidence.verify_result_bundle(root)

    def test_ledger_claiming_provider_proof_is_refused(self, tmp_path: Path) -> None:
        orch = fixture.build_orchestration()
        ledger = {
            key: value
            for key, value in orch["ledger"].items()
            if key != "ledger_sha256"
        }
        ledger["is_provider_proof"] = True
        root = _result_envelope(
            tmp_path / "results",
            **{"application-ledger.json": modal._seal(ledger)},
        )
        with pytest.raises(evidence.ModalL4EvidenceError, match="provider proof"):
            evidence.verify_result_bundle(root)

    def test_a_tampered_ledger_total_is_refused(self, tmp_path: Path) -> None:
        # A reserved total edited above the planned envelope no longer verifies:
        # the comprehensive validator recomputes it from the sealed event log.
        orch = fixture.build_orchestration()
        ledger = {
            key: value
            for key, value in orch["ledger"].items()
            if key != "ledger_sha256"
        }
        ledger["reserved_usd"] = "6.5"
        root = _result_envelope(
            tmp_path / "results",
            **{"application-ledger.json": modal._seal(ledger)},
        )
        with pytest.raises(evidence.ModalL4EvidenceError, match="does not verify"):
            evidence.verify_result_bundle(root)

    def test_incomplete_teardown_refuses_publication(self, tmp_path: Path) -> None:
        root = _result_envelope(tmp_path / "results")
        teardown = json.loads(
            (root / "modal-teardown.json").read_text(encoding="utf-8")
        )
        teardown["live_named_volumes"] = ["llmtracefx-run-app"]
        (root / "modal-teardown.json").write_text(
            json.dumps(teardown), encoding="utf-8"
        )
        with pytest.raises(evidence.ModalL4EvidenceError, match="teardown receipt"):
            evidence.verify_result_bundle(root)

    def test_unpublished_limitations_refuse_publication(self, tmp_path: Path) -> None:
        root = _result_envelope(
            tmp_path / "results", **{"modal-limitations.json": {"uncontrolled": []}}
        )
        with pytest.raises(evidence.ModalL4EvidenceError, match="limitations"):
            evidence.verify_result_bundle(root)

    def test_absent_orchestration_is_terminal(self, tmp_path: Path) -> None:
        root = _result_envelope(tmp_path / "results")
        (root / evidence.ORCHESTRATION_FILE).unlink()
        with pytest.raises(
            evidence.ModalL4EvidenceError, match="orchestration-receipt.json"
        ):
            evidence.verify_result_bundle(root)

    def test_absent_cells_directory_is_terminal(self, tmp_path: Path) -> None:
        root = _result_envelope(tmp_path / "results")
        for path in (root / evidence.CELLS_DIRECTORY).iterdir():
            path.unlink()
        (root / evidence.CELLS_DIRECTORY).rmdir()
        with pytest.raises(evidence.ModalL4EvidenceError, match="cells directory"):
            evidence.verify_result_bundle(root)

    def test_a_tampered_cell_receipt_fails_analysis(self, tmp_path: Path) -> None:
        root = _result_envelope(tmp_path / "results")
        cell_path = next((root / evidence.CELLS_DIRECTORY).glob("*.json"))
        wrapper = json.loads(cell_path.read_text(encoding="utf-8"))
        wrapper["cell_receipt"]["requests"][0]["output_token_ids"] = [1, 2, 3]
        cell_path.write_text(json.dumps(wrapper), encoding="utf-8")
        with pytest.raises(
            evidence.ModalL4EvidenceError,
            match="provider result validation or analysis failed",
        ):
            evidence.verify_result_bundle(root)

    def test_a_foreign_file_in_the_bundle_is_refused(self, tmp_path: Path) -> None:
        root = _result_envelope(tmp_path / "results")
        (root / "unexpected.json").write_text("{}", encoding="utf-8")
        with pytest.raises(evidence.ModalL4EvidenceError, match="closed allowlist"):
            evidence.verify_result_bundle(root)

    def test_a_foreign_directory_in_the_bundle_is_refused(self, tmp_path: Path) -> None:
        root = _result_envelope(tmp_path / "results")
        (root / "extra").mkdir()
        with pytest.raises(evidence.ModalL4EvidenceError, match="closed allowlist"):
            evidence.verify_result_bundle(root)

    def test_a_symlinked_envelope_file_is_refused(self, tmp_path: Path) -> None:
        root = _result_envelope(tmp_path / "results")
        real = json.loads((root / "modal-teardown.json").read_text(encoding="utf-8"))
        target = tmp_path / "teardown-target.json"
        target.write_text(json.dumps(real), encoding="utf-8")
        (root / "modal-teardown.json").unlink()
        (root / "modal-teardown.json").symlink_to(target)
        with pytest.raises(evidence.ModalL4EvidenceError, match="closed allowlist"):
            evidence.verify_result_bundle(root)

    def test_a_swapped_ledger_that_still_verifies_is_refused(
        self, tmp_path: Path
    ) -> None:
        # An independently-valid ledger that is not the orchestration's ledger
        # (only its path commitment differs) passes ledger validation but is
        # refused by the cross-binding: no mix-and-match.
        swapped = modal.build_completed_ledger_document(
            plan=modal.build_default_plan(),
            source_head=fixture.SOURCE_HEAD,
            experiment_nonce=fixture.NONCE,
            ledger_path_sha256="sha256:" + "1" * 64,
            reserved_at=fixture.RESERVED_AT,
            completed_at=fixture.COMPLETED_AT,
        )
        root = _result_envelope(
            tmp_path / "results", **{"application-ledger.json": swapped}
        )
        with pytest.raises(
            evidence.ModalL4EvidenceError, match="not bound to the orchestration"
        ):
            evidence.verify_result_bundle(root)

    def test_a_swapped_teardown_that_is_complete_is_refused(
        self, tmp_path: Path
    ) -> None:
        # A teardown receipt that adjudicates complete but differs from the
        # orchestration's teardown is a mix-and-match and is refused.
        orch = fixture.build_orchestration()
        teardown = {
            key: value
            for key, value in orch["teardown"].items()
            if key != "adjudication"
        }
        teardown["observed_at"] = "2020-01-01T00:00:00+00:00"
        root = _result_envelope(
            tmp_path / "results", **{"modal-teardown.json": teardown}
        )
        with pytest.raises(
            evidence.ModalL4EvidenceError, match="not bound to the orchestration"
        ):
            evidence.verify_result_bundle(root)

    def test_a_swapped_rate_receipt_that_is_valid_is_refused(
        self, tmp_path: Path
    ) -> None:
        receipt = dict(fixture.rate_receipt())
        receipt["fetched_at"] = "2020-01-01T00:00:00+00:00"
        root = _result_envelope(
            tmp_path / "results", **{"modal-rate-receipt.json": receipt}
        )
        with pytest.raises(
            evidence.ModalL4EvidenceError, match="not bound to the orchestration"
        ):
            evidence.verify_result_bundle(root)

    @pytest.mark.parametrize(
        "name",
        (
            "source-checkout.json",
            "profile-authentication.json",
            "modal-rate-refresh.json",
        ),
    )
    def test_a_missing_new_envelope_is_refused(self, tmp_path: Path, name: str) -> None:
        root = _result_envelope(tmp_path / "results")
        (root / name).unlink()
        with pytest.raises(evidence.ModalL4EvidenceError, match=f"missing {name}"):
            evidence.verify_result_bundle(root)

    @pytest.mark.parametrize(
        "name",
        (
            "analysis.json",
            "claim-matrix.json",
            "report.html",
            "crossover.svg",
            "SHA256SUMS",
        ),
    )
    def test_a_missing_artifact_is_refused(self, tmp_path: Path, name: str) -> None:
        root = _result_envelope(tmp_path / "results")
        (root / name).unlink()
        with pytest.raises(evidence.ModalL4EvidenceError, match=f"missing {name}"):
            evidence.verify_result_bundle(root)

    def test_a_swapped_source_checkout_is_refused(self, tmp_path: Path) -> None:
        swapped = dict(fixture.build_orchestration()["source_checkout"])
        swapped["ignored_untracked_prefix"] = ".other-traces/"
        root = _result_envelope(
            tmp_path / "results", **{"source-checkout.json": swapped}
        )
        with pytest.raises(
            evidence.ModalL4EvidenceError, match="not bound to the orchestration"
        ):
            evidence.verify_result_bundle(root)

    def test_a_swapped_profile_verdict_is_refused(self, tmp_path: Path) -> None:
        # cli_version == sdk_version keeps the closed schema valid, but the
        # differing version no longer matches the orchestration verdict.
        profile = dict(fixture.build_orchestration()["profile_authentication"])
        profile["cli_version"] = "9.9.9"
        profile["sdk_version"] = "9.9.9"
        root = _result_envelope(
            tmp_path / "results", **{"profile-authentication.json": profile}
        )
        with pytest.raises(
            evidence.ModalL4EvidenceError, match="not bound to the orchestration"
        ):
            evidence.verify_result_bundle(root)

    def test_a_profile_verdict_with_a_bad_schema_is_refused(
        self, tmp_path: Path
    ) -> None:
        profile = dict(fixture.build_orchestration()["profile_authentication"])
        profile["mechanism"] = "some_other_probe"
        root = _result_envelope(
            tmp_path / "results", **{"profile-authentication.json": profile}
        )
        with pytest.raises(
            evidence.ModalL4EvidenceError,
            match="profile authentication verdict is invalid",
        ):
            evidence.verify_result_bundle(root)

    def test_a_swapped_rate_refresh_is_refused(self, tmp_path: Path) -> None:
        refresh = json.loads(json.dumps(fixture.build_orchestration()["rate_refresh"]))
        refresh["capture"]["observed_at"] = "2020-01-01T00:00:00+00:00"
        root = _result_envelope(
            tmp_path / "results", **{"modal-rate-refresh.json": refresh}
        )
        with pytest.raises(
            evidence.ModalL4EvidenceError, match="not bound to the orchestration"
        ):
            evidence.verify_result_bundle(root)

    @pytest.mark.parametrize(
        ("name", "tampered"),
        (
            ("analysis.json", '{"tampered": true}\n'),
            ("claim-matrix.json", '{"claims": []}\n'),
            ("report.html", "<!doctype html><html></html>\n"),
            ("crossover.svg", '<svg xmlns="http://www.w3.org/2000/svg"></svg>\n'),
        ),
    )
    def test_a_tampered_artifact_is_refused(
        self, tmp_path: Path, name: str, tampered: str
    ) -> None:
        root = _result_envelope(tmp_path / "results")
        (root / name).write_text(tampered, encoding="utf-8")
        with pytest.raises(
            evidence.ModalL4EvidenceError,
            match="does not match the regenerated artifact",
        ):
            evidence.verify_result_bundle(root)

    def test_a_drifted_checksum_is_refused(self, tmp_path: Path) -> None:
        root = _result_envelope(tmp_path / "results")
        text = (root / "SHA256SUMS").read_text(encoding="utf-8")
        drifted = [
            "0" * 64 + "  analysis.json" if line.endswith("  analysis.json") else line
            for line in text.splitlines()
        ]
        (root / "SHA256SUMS").write_text("\n".join(drifted) + "\n", encoding="utf-8")
        with pytest.raises(evidence.ModalL4EvidenceError, match="checksum mismatch"):
            evidence.verify_result_bundle(root)

    def test_a_checksum_manifest_naming_a_foreign_file_is_refused(
        self, tmp_path: Path
    ) -> None:
        root = _result_envelope(tmp_path / "results")
        text = (root / "SHA256SUMS").read_text(encoding="utf-8")
        (root / "SHA256SUMS").write_text(
            text + "0" * 64 + "  cells\n", encoding="utf-8"
        )
        with pytest.raises(evidence.ModalL4EvidenceError, match="allowlist differs"):
            evidence.verify_result_bundle(root)


class TestResultBundleBuild:
    def test_build_then_verify_round_trips(self, tmp_path: Path) -> None:
        workspace = _execution_workspace(tmp_path / "workspace")
        output = tmp_path / "bundle"
        summary = evidence.build_result_bundle(workspace, output)
        assert summary["built"] is True
        assert summary["pair_count"] == 16
        assert summary["cell_count"] == 32
        assert summary["cell_files"] == 32
        result = evidence.verify_result_bundle(output)
        assert result["verified"] is True
        assert result["pair_count"] == 16

    def test_the_built_bundle_is_the_closed_file_set(self, tmp_path: Path) -> None:
        workspace = _execution_workspace(tmp_path / "workspace")
        output = tmp_path / "bundle"
        evidence.build_result_bundle(workspace, output)
        top_level = {p.name for p in output.iterdir() if p.name != "cells"}
        assert top_level == set(evidence.RESULT_TOP_LEVEL_FILES)
        assert (output / "cells").is_dir()
        assert len(list((output / "cells").glob("*.json"))) == 32

    def test_two_builds_are_byte_identical(self, tmp_path: Path) -> None:
        workspace = _execution_workspace(tmp_path / "workspace")
        first = tmp_path / "first"
        second = tmp_path / "second"
        evidence.build_result_bundle(workspace, first)
        evidence.build_result_bundle(workspace, second)
        for name in evidence.RESULT_TOP_LEVEL_FILES:
            assert (first / name).read_bytes() == (second / name).read_bytes(), name
        for cell in sorted(p.name for p in (first / "cells").iterdir()):
            assert (first / "cells" / cell).read_bytes() == (
                second / "cells" / cell
            ).read_bytes(), cell

    def test_a_refusal_workspace_is_not_bundled(self, tmp_path: Path) -> None:
        orch = fixture.build_orchestration()
        orch["published"] = False
        orch["kind"] = "llmtracefx.modal_l4_crossover.refusal"
        workspace = _execution_workspace(tmp_path / "workspace", orchestration=orch)
        output = tmp_path / "bundle"
        with pytest.raises(
            evidence.ModalL4EvidenceError, match="published, complete result"
        ):
            evidence.build_result_bundle(workspace, output)
        assert not output.exists()

    def test_a_workspace_missing_cells_is_not_bundled(self, tmp_path: Path) -> None:
        workspace = _execution_workspace(tmp_path / "workspace")
        for path in (workspace / "cells").iterdir():
            path.unlink()
        (workspace / "cells").rmdir()
        output = tmp_path / "bundle"
        with pytest.raises(evidence.ModalL4EvidenceError, match="cells directory"):
            evidence.build_result_bundle(workspace, output)

    def test_a_workspace_missing_the_exposure_reason_is_not_bundled(
        self, tmp_path: Path
    ) -> None:
        workspace = _execution_workspace(tmp_path / "workspace")
        (workspace / "credential-exposure.json").unlink()
        output = tmp_path / "bundle"
        with pytest.raises(evidence.ModalL4EvidenceError, match="credential-exposure"):
            evidence.build_result_bundle(workspace, output)

    def test_a_workspace_that_fails_analysis_is_not_bundled(
        self, tmp_path: Path
    ) -> None:
        cells = fixture.build_cells()
        cell_id = next(iter(cells))
        cells[cell_id]["cell_receipt"]["requests"][0]["output_token_ids"] = [1, 2, 3]
        workspace = _execution_workspace(tmp_path / "workspace", cells=cells)
        output = tmp_path / "bundle"
        with pytest.raises(evidence.ModalL4EvidenceError, match="does not analyze"):
            evidence.build_result_bundle(workspace, output)
        assert not output.exists()


class TestNoProviderAccess:
    def test_no_provider_module_is_imported_by_evidence_flows(
        self, bundle: Path, tmp_path: Path
    ) -> None:
        evidence.verify_offline_bundle(bundle, repo_root=ROOT)
        root = _result_envelope(tmp_path / "results")
        result = evidence.verify_result_bundle(root)
        assert result["verified"] is True
        assert not [name for name in sys.modules if name.split(".")[0] == "modal"]

    def test_offline_paths_refuse_to_run_with_the_sdk_loaded(
        self, monkeypatch: pytest.MonkeyPatch, bundle: Path
    ) -> None:
        monkeypatch.setitem(sys.modules, "modal", object())
        with pytest.raises(modal.ModalL4ContractError, match="must not be imported"):
            evidence.verify_offline_bundle(bundle, repo_root=ROOT)


class TestCredentialExposureEvidence:
    def test_offline_bundle_publishes_the_gate_and_its_claims(
        self, bundle: Path
    ) -> None:
        preflight = json.loads(
            (bundle / "offline-preflight.json").read_text(encoding="utf-8")
        )
        contract = json.loads(
            (bundle / "evidence-contract.json").read_text(encoding="utf-8")
        )
        claims = json.loads((bundle / "claim-matrix.json").read_text(encoding="utf-8"))
        states = {item["claim_id"]: item["state"] for item in claims["claims"]}
        assert preflight["credential_exposure_gate"]["cleared"] is False
        assert preflight["credential_exposure_gate"]["action"] == (
            "refuse_provider_execution"
        )
        assert preflight["exposed_profile_credential_never_used_by_experiment"] is True
        assert contract["credential_exposure_gate"]["absent_attestation"] == (
            "refuse_provider_execution"
        )
        assert (
            contract["credential_exposure_gate"]["records_credential_values"] is False
        )
        assert states["exposed-profile-credential-never-used-by-experiment"] == (
            "supported"
        )
        assert states["exposed-profile-credential-revocation-confirmed"] == (
            "unsupported"
        )
        assert states["fresh-local-profile-created-without-sharing"] == "unsupported"

    def test_result_contract_lists_the_gate_first(self, bundle: Path) -> None:
        contract = json.loads(
            (bundle / "result-contract.json").read_text(encoding="utf-8")
        )
        assert "credential-exposure.json" in contract["modal_envelope_files"]
        # The feasibility proof and the headroom binding are checked before the
        # credential-exposure clearance, so the gate is now third rather than
        # second; it still precedes every statistical step.
        order = contract["verification_order"]
        gate_index = next(
            index for index, step in enumerate(order) if "credential-exposure" in step
        )
        assert gate_index == 3
        assert all(
            "cell" not in step and "statistic" not in step
            for step in order[:gate_index]
        )

    def test_a_missing_gate_verdict_refuses_results(self, tmp_path: Path) -> None:
        root = _result_envelope(tmp_path / "results")
        (root / "credential-exposure.json").unlink()
        with pytest.raises(
            evidence.ModalL4EvidenceError, match="missing credential-exposure.json"
        ):
            evidence.verify_result_bundle(root)

    @pytest.mark.parametrize(
        "override",
        (
            {"cleared": False},
            {"exposed_profile_credential_revocation_confirmed": False},
            {"fresh_local_profile_created_without_sharing": False},
            {"fresh_profile_shared_anywhere": True},
            {"confirmed_by": "self"},
            {"records_credential_values": True},
        ),
    )
    def test_an_uncleared_gate_refuses_results(
        self, tmp_path: Path, override: dict[str, Any]
    ) -> None:
        verdict = {
            "gate": "credential_exposure",
            "cleared": True,
            "exposed_profile_credential_never_used_by_experiment": True,
            "exposed_profile_credential_revocation_confirmed": True,
            "fresh_local_profile_created_without_sharing": True,
            "fresh_profile_shared_anywhere": False,
            "confirmed_by": "coordinator",
            "confirmed_at": "2026-09-04T21:02:06.080+05:30",
            "reason": "coordinator confirmed revocation and fresh local profile",
            "records_credential_values": False,
            "action": "proceed",
        }
        verdict.update(override)
        root = _result_envelope(
            tmp_path / "results", **{"credential-exposure.json": verdict}
        )
        with pytest.raises(evidence.ModalL4EvidenceError, match="credential-exposure"):
            evidence.verify_result_bundle(root)

    def test_a_gate_verdict_carrying_credential_fields_is_refused(
        self, tmp_path: Path
    ) -> None:
        verdict = {
            "gate": "credential_exposure",
            "cleared": True,
            "exposed_profile_credential_never_used_by_experiment": True,
            "exposed_profile_credential_revocation_confirmed": True,
            "fresh_local_profile_created_without_sharing": True,
            "fresh_profile_shared_anywhere": False,
            "confirmed_by": "coordinator",
            "confirmed_at": "2026-09-04T21:02:06.080+05:30",
            "reason": "coordinator confirmed revocation and fresh local profile",
            "records_credential_values": False,
            "action": "proceed",
            "revoked_token_prefix": "ak-1234",
        }
        root = _result_envelope(
            tmp_path / "results", **{"credential-exposure.json": verdict}
        )
        with pytest.raises(evidence.ModalL4EvidenceError, match="closed allowlist"):
            evidence.verify_result_bundle(root)
