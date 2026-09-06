"""Offline, no-GPU tests for the vLLM KV-truth evidence bundling module.

Every fixture here is fabricated in-process; none of it comes from a real
vLLM engine, a real host, or a real GPU. Bundles built in these tests always
carry ``run_mode=RUN_MODE_SYNTHETIC_FIXTURE`` and every report/text artifact
this module renders must say so explicitly -- this suite exists partly to
prove that labeling is never dropped.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.deploy import vllm_kv_truth_evidence as evidence
from llmtracefx.optimizer.lab.qwen3_8b.kv_truth_workload import NESTED_PROBES

VALID_NONCE = "d" * 40


def _record(
    *,
    request_id: str,
    scenario: str,
    num_cached_tokens: int,
    num_cache_creation_tokens: int,
    boundary_valid: bool = True,
    event_batches: list[dict[str, Any]] | None = None,
    finished: bool = True,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "scenario": scenario,
        "prompt_token_count": num_cached_tokens + num_cache_creation_tokens,
        "output_token_count": 1,
        "num_cached_tokens": num_cached_tokens,
        "num_cache_creation_tokens": num_cache_creation_tokens,
        "finished": finished,
        "finish_reason": "stop" if finished else None,
        "event_batches": event_batches or [],
        "boundary_valid": boundary_valid,
        "boundary_reasons": (
            [] if boundary_valid else ["kv_events_out_of_order_for_request"]
        ),
    }


def _all_verified_b_lane_result() -> dict[str, Any]:
    """A fabricated B-lane result where every probe's engine-attested count
    exactly matches its expectation and every request has an event batch --
    i.e. every claim should classify as ``verified``."""

    records = []
    for probe in NESTED_PROBES:
        records.append(
            _record(
                request_id=f"req-{probe.name}",
                scenario=probe.scenario,
                num_cached_tokens=probe.expected_reusable_tokens,
                num_cache_creation_tokens=max(1, 321 - probe.expected_reusable_tokens),
                event_batches=[{"sequence": 0}],
            )
        )
    return {"lane": "B", "records": records, "all_boundaries_valid": True}


class TestClassifyProbeClaim:
    def test_verified_when_engine_and_events_agree(self) -> None:
        verdict = evidence.classify_probe_claim(
            expected_reusable_tokens=256,
            expected_reusable_blocks=16,
            engine_attested_cached_tokens=256,
            event_attested=True,
            boundary_valid=True,
        )
        assert verdict == evidence.VERDICT_VERIFIED

    def test_attested_only_when_no_event_capture_but_engine_matches(self) -> None:
        verdict = evidence.classify_probe_claim(
            expected_reusable_tokens=256,
            expected_reusable_blocks=16,
            engine_attested_cached_tokens=256,
            event_attested=None,
            boundary_valid=True,
        )
        assert verdict == evidence.VERDICT_ATTESTED_ONLY

    def test_partial_when_engine_and_events_disagree(self) -> None:
        verdict = evidence.classify_probe_claim(
            expected_reusable_tokens=256,
            expected_reusable_blocks=16,
            engine_attested_cached_tokens=256,
            event_attested=False,
            boundary_valid=True,
        )
        assert verdict == evidence.VERDICT_PARTIAL

    def test_recomputed_when_engine_reports_less_than_expected(self) -> None:
        verdict = evidence.classify_probe_claim(
            expected_reusable_tokens=256,
            expected_reusable_blocks=16,
            engine_attested_cached_tokens=0,
            event_attested=None,
            boundary_valid=True,
        )
        assert verdict == evidence.VERDICT_RECOMPUTED

    def test_invalid_when_boundary_is_invalid(self) -> None:
        verdict = evidence.classify_probe_claim(
            expected_reusable_tokens=256,
            expected_reusable_blocks=16,
            engine_attested_cached_tokens=256,
            event_attested=True,
            boundary_valid=False,
        )
        assert verdict == evidence.VERDICT_INVALID

    def test_invalid_when_request_binding_is_ambiguous(self) -> None:
        verdict = evidence.classify_probe_claim(
            expected_reusable_tokens=256,
            expected_reusable_blocks=16,
            engine_attested_cached_tokens=256,
            event_attested=True,
            boundary_valid=True,
            request_binding_ambiguous=True,
        )
        assert verdict == evidence.VERDICT_INVALID

    def test_invalid_when_engine_reports_more_than_expected(self) -> None:
        verdict = evidence.classify_probe_claim(
            expected_reusable_tokens=0,
            expected_reusable_blocks=0,
            engine_attested_cached_tokens=64,
            event_attested=None,
            boundary_valid=True,
        )
        assert verdict == evidence.VERDICT_INVALID

    def test_only_seven_fixed_verdicts_exist(self) -> None:
        assert evidence.VALID_VERDICTS == {
            "verified",
            "partial",
            "attested-only",
            "recomputed",
            "evicted",
            "unsupported",
            "invalid",
        }


class TestRenderClaimSentence:
    def test_sentence_includes_expected_attested_and_verdict(self) -> None:
        sentence = evidence.render_claim_sentence(
            expected_reusable_tokens=256,
            expected_reusable_blocks=16,
            engine_attested_cached_tokens=256,
            verdict=evidence.VERDICT_VERIFIED,
        )
        assert "256/16" in sentence
        assert "attested 256" in sentence
        assert "verified" in sentence

    def test_sentence_handles_unsupported_expectation(self) -> None:
        sentence = evidence.render_claim_sentence(
            expected_reusable_tokens=None,
            expected_reusable_blocks=None,
            engine_attested_cached_tokens=0,
            verdict=evidence.VERDICT_UNSUPPORTED,
        )
        assert "null/unsupported" in sentence


class TestBuildClaimMatrix:
    def test_builds_ten_entries_matching_nested_probes(self) -> None:
        matrix = evidence.build_claim_matrix(_all_verified_b_lane_result())
        assert len(matrix) == len(NESTED_PROBES)
        for entry, probe in zip(matrix, NESTED_PROBES, strict=True):
            assert entry.scenario == probe.scenario
            assert entry.expected_reusable_tokens == probe.expected_reusable_tokens

    def test_all_verified_fixture_yields_all_verified_verdicts(self) -> None:
        matrix = evidence.build_claim_matrix(_all_verified_b_lane_result())
        assert all(entry.verdict == evidence.VERDICT_VERIFIED for entry in matrix)

    def test_rejects_wrong_record_count(self) -> None:
        result = _all_verified_b_lane_result()
        result["records"] = result["records"][:5]
        with pytest.raises(evidence.EvidenceError, match="cannot align"):
            evidence.build_claim_matrix(result)

    def test_shared_scenario_probes_are_disambiguated_by_position(self) -> None:
        """Three probes share the 'identical_prefix' scenario; the matrix
        must still align each to its own distinct expectation by position,
        not by scenario lookup."""

        matrix = evidence.build_claim_matrix(_all_verified_b_lane_result())
        identical_prefix_entries = [
            entry for entry in matrix if entry.scenario == "identical_prefix"
        ]
        assert len(identical_prefix_entries) == 3
        expected_tokens = {
            entry.expected_reusable_tokens for entry in identical_prefix_entries
        }
        assert expected_tokens == {256, 128}


class TestBuildEvictionClaim:
    def test_evicted_when_final_probe_is_zero_cached_and_full_work(self) -> None:
        lane_result = {
            "lane": "eviction",
            "records": [
                _record(
                    request_id="evict-final",
                    scenario="eviction_seed",
                    num_cached_tokens=0,
                    num_cache_creation_tokens=257,
                    boundary_valid=True,
                )
            ],
        }
        claim = evidence.build_eviction_claim(lane_result)
        assert claim.verdict == evidence.VERDICT_EVICTED

    def test_unsupported_when_final_probe_is_not_a_clean_eviction(self) -> None:
        lane_result = {
            "lane": "eviction",
            "records": [
                _record(
                    request_id="evict-final",
                    scenario="eviction_seed",
                    num_cached_tokens=128,
                    num_cache_creation_tokens=129,
                    boundary_valid=True,
                )
            ],
        }
        claim = evidence.build_eviction_claim(lane_result)
        assert claim.verdict == evidence.VERDICT_UNSUPPORTED

    def test_invalid_when_boundary_invalid(self) -> None:
        lane_result = {
            "lane": "eviction",
            "records": [
                _record(
                    request_id="evict-final",
                    scenario="eviction_seed",
                    num_cached_tokens=0,
                    num_cache_creation_tokens=257,
                    boundary_valid=False,
                )
            ],
        }
        claim = evidence.build_eviction_claim(lane_result)
        assert claim.verdict == evidence.VERDICT_INVALID

    def test_rejects_empty_records(self) -> None:
        with pytest.raises(evidence.EvidenceError):
            evidence.build_eviction_claim({"lane": "eviction", "records": []})


class TestBuildSaltIsolationClaim:
    def test_unsupported_when_salt_not_supported(self) -> None:
        lane_result = _all_verified_b_lane_result()
        claim = evidence.build_salt_isolation_claim(lane_result, salt_supported=False)
        assert claim.verdict == evidence.VERDICT_UNSUPPORTED

    def test_verified_when_salt_supported_and_isolated(self) -> None:
        lane_result = _all_verified_b_lane_result()
        for record in lane_result["records"]:
            if record["scenario"] == "namespace_isolation":
                record["num_cached_tokens"] = 0
        claim = evidence.build_salt_isolation_claim(lane_result, salt_supported=True)
        assert claim.verdict == evidence.VERDICT_VERIFIED

    def test_invalid_when_salt_supported_but_not_isolated(self) -> None:
        lane_result = _all_verified_b_lane_result()
        for record in lane_result["records"]:
            if record["scenario"] == "namespace_isolation":
                record["num_cached_tokens"] = 256
        claim = evidence.build_salt_isolation_claim(lane_result, salt_supported=True)
        assert claim.verdict == evidence.VERDICT_INVALID

    def test_rejects_missing_namespace_isolation_record(self) -> None:
        with pytest.raises(evidence.EvidenceError):
            evidence.build_salt_isolation_claim({"records": []}, salt_supported=False)


def _fixture_private_bundle() -> evidence.PrivateEvidenceBundle:
    matrix = evidence.build_claim_matrix(_all_verified_b_lane_result())
    ledger = evidence.ListRateLedger(
        entries=(
            evidence.ListRateLedgerEntry(
                elapsed_minutes="1.000000", cost_usd="0.008333"
            ),
        )
    )
    teardown = evidence.TeardownReceipt(
        residual_containers=0,
        residual_gpu_processes=0,
        evidence_transferred_and_verified=True,
        shutdown_issued=True,
        safe_to_terminate_message_emitted=True,
    )
    return evidence.PrivateEvidenceBundle(
        run_mode=evidence.RUN_MODE_SYNTHETIC_FIXTURE,
        experiment_nonce=VALID_NONCE,
        authorization={
            "protocol_id": "qwen3-8b-vllm-kv-truth-v1",
            "nonce": VALID_NONCE,
        },
        ssh_options_public_record={"batch_mode": True},
        lane_receipts={"B": _all_verified_b_lane_result()},
        claim_matrix=matrix,
        list_rate_ledger=ledger,
        teardown_receipt=teardown,
    )


class TestPrivateEvidenceBundle:
    def test_round_trips_through_to_dict_and_verify(self) -> None:
        bundle = _fixture_private_bundle()
        raw = bundle.to_dict()
        again = evidence.PrivateEvidenceBundle.verify_and_parse(raw)
        assert again.experiment_nonce == VALID_NONCE
        assert again.run_mode == evidence.RUN_MODE_SYNTHETIC_FIXTURE

    def test_rejects_tampered_seal(self) -> None:
        bundle = _fixture_private_bundle()
        raw = bundle.to_dict()
        raw["experiment_nonce"] = "e" * 40
        with pytest.raises(evidence.EvidenceError, match="does not verify"):
            evidence.PrivateEvidenceBundle.verify_and_parse(raw)

    def test_rejects_invalid_run_mode(self) -> None:
        with pytest.raises(evidence.EvidenceError, match="run_mode"):
            evidence.PrivateEvidenceBundle(
                run_mode="not-a-real-mode",
                experiment_nonce=VALID_NONCE,
                authorization={},
                ssh_options_public_record={},
                lane_receipts={},
                claim_matrix=(),
                list_rate_ledger=evidence.ListRateLedger(entries=()),
                teardown_receipt=None,
            )

    def test_teardown_receipt_rejects_nonzero_residuals(self) -> None:
        with pytest.raises(evidence.EvidenceError):
            evidence.TeardownReceipt(
                residual_containers=1,
                residual_gpu_processes=0,
                evidence_transferred_and_verified=True,
                shutdown_issued=True,
                safe_to_terminate_message_emitted=True,
            )

    def test_write_and_read_round_trip(self, tmp_path: Path) -> None:
        bundle = _fixture_private_bundle()
        path = tmp_path / "private-bundle.json"
        bundle.write(path)
        again = evidence.PrivateEvidenceBundle.read(path)
        assert again.experiment_nonce == bundle.experiment_nonce

    def test_read_rejects_symlink(self, tmp_path: Path) -> None:
        bundle = _fixture_private_bundle()
        real = tmp_path / "real.json"
        bundle.write(real)
        link = tmp_path / "link.json"
        link.symlink_to(real)
        with pytest.raises(evidence.EvidenceError):
            evidence.PrivateEvidenceBundle.read(link)


class TestPublicRedactedBundle:
    def test_from_private_produces_a_verified_seal(self) -> None:
        private = _fixture_private_bundle()
        public = evidence.PublicRedactedBundle.from_private(private)
        again = evidence.PublicRedactedBundle.verify_and_parse(public.to_dict())
        assert again.to_dict()["experiment_nonce"] == VALID_NONCE

    def test_no_forbidden_keys_survive_in_rendered_bundle(self) -> None:
        private = _fixture_private_bundle()
        public = evidence.PublicRedactedBundle.from_private(private)
        # assert_publication_safe already ran during from_private(); re-run
        # it here explicitly against the rendered dict as a regression guard.
        evidence.assert_publication_safe(public.to_dict())

    def test_exact_private_tokens_and_hashes_are_removed_from_public_bundle(
        self,
    ) -> None:
        private = _fixture_private_bundle()
        secret_hash = "a" * 64
        raw_lane = {
            "lane": "B",
            "records": [
                {
                    "request_id": "req-raw",
                    "scenario": "cold",
                    "prompt_token_ids": [101, 102],
                    "output_token_ids": [201],
                    "event_batches": [
                        {
                            "sequence": 0,
                            "topic": "kv-events",
                            "ts": 1.0,
                            "data_parallel_rank": 0,
                            "events": [
                                {
                                    "type": "BlockStored",
                                    "block_hashes": [secret_hash],
                                    "parent_block_hash": None,
                                    "token_ids": list(range(16)),
                                    "block_size": 16,
                                    "lora_id": None,
                                    "medium": "GPU",
                                    "lora_name": None,
                                    "extra_keys": [None],
                                    "group_idx": 0,
                                    "kv_cache_spec_kind": "full",
                                    "kv_cache_spec_sliding_window": None,
                                    "locality": "LOCAL",
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        private = replace(private, lane_receipts={"B": raw_lane})
        assert secret_hash in json.dumps(private.to_dict())
        public_text = json.dumps(
            evidence.PublicRedactedBundle.from_private(private).to_dict()
        )
        assert secret_hash not in public_text
        assert "prompt_token_ids" not in public_text
        assert "output_token_ids" not in public_text
        assert '"token_ids"' not in public_text

    def test_assert_publication_safe_rejects_forbidden_key(self) -> None:
        with pytest.raises(evidence.EvidenceError, match="forbidden key"):
            evidence.assert_publication_safe({"gpu_uuid": "abc"})

    def test_assert_publication_safe_rejects_nested_forbidden_key(self) -> None:
        with pytest.raises(evidence.EvidenceError, match="forbidden key"):
            evidence.assert_publication_safe({"outer": {"private_key_path": "x"}})

    def test_assert_publication_safe_accepts_clean_payload(self) -> None:
        evidence.assert_publication_safe({"protocol_id": "x", "nested": {"a": 1}})

    def test_assert_publication_safe_accepts_strict_host_key_checking(self) -> None:
        # A real SSH-options public record's field name contains "host" as
        # a substring but is a fixed boolean policy flag, never a
        # host-identifying value -- it must not be flagged as forbidden.
        evidence.assert_publication_safe({"strict_host_key_checking": True})

    def test_assert_publication_safe_still_rejects_other_host_keys(self) -> None:
        # The allowlist is an exact-name exception, not a fragment bypass:
        # any other key containing "host" must still be rejected.
        with pytest.raises(evidence.EvidenceError, match="forbidden key"):
            evidence.assert_publication_safe({"target_host": "example.com"})

    def test_write_and_verify_directory(self, tmp_path: Path) -> None:
        private = _fixture_private_bundle()
        public = evidence.PublicRedactedBundle.from_private(private)
        directory = tmp_path / "public"
        evidence.write_public_bundle_directory(public, directory)
        verified = evidence.verify_public_bundle_directory(directory)
        assert verified.to_dict()["experiment_nonce"] == VALID_NONCE

    def test_verify_directory_rejects_tampered_file(self, tmp_path: Path) -> None:
        private = _fixture_private_bundle()
        public = evidence.PublicRedactedBundle.from_private(private)
        directory = tmp_path / "public"
        evidence.write_public_bundle_directory(public, directory)
        (directory / "report.txt").write_text("tampered", encoding="utf-8")
        with pytest.raises(evidence.EvidenceError, match="SHA256SUMS"):
            evidence.verify_public_bundle_directory(directory)

    def test_report_text_labels_synthetic_fixture(self) -> None:
        private = _fixture_private_bundle()
        public = evidence.PublicRedactedBundle.from_private(private)
        text = evidence.render_report_text(public)
        assert "SYNTHETIC FIXTURE" in text
        assert "never executed" in text

    def test_report_svg_is_deterministic(self) -> None:
        private = _fixture_private_bundle()
        public = evidence.PublicRedactedBundle.from_private(private)
        svg_1 = evidence.render_report_svg(public)
        svg_2 = evidence.render_report_svg(public)
        assert svg_1 == svg_2
        assert svg_1.startswith("<svg")

    def test_sha256sums_helpers_round_trip(self) -> None:
        files = {"a.txt": b"hello", "b.txt": b"world"}
        sums = evidence.compute_sha256sums(files)
        assert evidence.verify_sha256sums(sums, files) is True
        assert (
            evidence.verify_sha256sums(sums, {"a.txt": b"tampered", "b.txt": b"world"})
            is False
        )

    def test_rejects_wrong_key_set(self) -> None:
        with pytest.raises(evidence.EvidenceError, match="required set"):
            evidence.PublicRedactedBundle.verify_and_parse({"protocol_id": "x"})
