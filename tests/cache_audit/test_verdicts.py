from __future__ import annotations

from dataclasses import replace

from llmtracefx.cache_audit.adapters.reference import ReferenceCacheAdapter
from llmtracefx.cache_audit.schema import (
    EligibilityStatus,
    EvidenceBasis,
    EvidenceFact,
    Limitation,
    Verdict,
    unavailable,
)
from llmtracefx.cache_audit.verdicts import classify_request
from llmtracefx.cache_audit.workloads import adversarial_requests, eviction_requests


def _record():
    return ReferenceCacheAdapter().run(adversarial_requests()[:1])[0]


def test_blocking_limitation_is_unsupported() -> None:
    record = replace(
        _record(),
        limitations=(
            Limitation("native_salt_unavailable", "MLX has no native salt", True),
        ),
        verdict=None,
        verdict_reasons=(),
    )
    assert classify_request(record).verdict is Verdict.UNSUPPORTED


def test_unexpected_prompt_overlap_is_recomputed() -> None:
    record = ReferenceCacheAdapter().run(adversarial_requests()[:2])[1]
    required = record.reuse.policy_required_prompt_tokens.value
    assert isinstance(required, int)
    reuse = replace(
        record.reuse,
        observed_prompt_tokens=EvidenceFact(
            value=required + 2,
            basis=EvidenceBasis.OBSERVED,
            source="test",
        ),
        unexpected_recomputed_tokens=EvidenceFact(
            value=2,
            basis=EvidenceBasis.INDEPENDENTLY_DERIVED,
            source="test",
        ),
    )
    classified = classify_request(
        replace(record, reuse=reuse, verdict=None, verdict_reasons=())
    )
    assert classified.verdict is Verdict.RECOMPUTED


def test_positive_attestation_without_oracle_is_attested_only() -> None:
    record = ReferenceCacheAdapter().run(adversarial_requests()[:2])[1]
    reuse = replace(
        record.reuse,
        semantic_prefix_tokens=unavailable("test", "identity_redacted"),
        policy_reusable_tokens=unavailable("test", "identity_redacted"),
        policy_required_prompt_tokens=unavailable("test", "identity_redacted"),
        unexpected_recomputed_tokens=unavailable("test", "identity_redacted"),
    )
    classified = classify_request(
        replace(record, reuse=reuse, verdict=None, verdict_reasons=())
    )
    assert classified.verdict is Verdict.ATTESTED_ONLY


def test_failed_correctness_does_not_change_cache_verdict() -> None:
    record = _record()
    output = replace(
        record.output,
        correctness=EvidenceFact(
            value=False,
            basis=EvidenceBasis.OBSERVED,
            source="test",
        ),
    )
    classified = classify_request(
        replace(record, output=output, verdict=None, verdict_reasons=())
    )
    assert classified.verdict is Verdict.VERIFIED_MISS
    assert classified.eligibility.output_equivalence is EligibilityStatus.INELIGIBLE


def test_failed_token_identity_does_not_change_cache_verdict() -> None:
    record = _record()
    output = replace(
        record.output,
        token_identity=EvidenceFact(
            value=False,
            basis=EvidenceBasis.OBSERVED,
            source="test",
        ),
    )
    classified = classify_request(
        replace(record, output=output, verdict=None, verdict_reasons=())
    )
    assert classified.verdict is Verdict.VERIFIED_MISS
    assert classified.eligibility.output_equivalence is EligibilityStatus.INELIGIBLE


def test_estimated_prompt_work_cannot_produce_a_verified_verdict() -> None:
    record = _record()
    reuse = replace(
        record.reuse,
        observed_prompt_tokens=EvidenceFact(
            value=record.reuse.observed_prompt_tokens.value,
            basis=EvidenceBasis.ESTIMATED,
            source="test",
        ),
    )
    classified = classify_request(
        replace(record, reuse=reuse, verdict=None, verdict_reasons=())
    )
    assert classified.verdict is Verdict.INVALID
    assert classified.verdict_reasons == ("observed_prompt_tokens_basis_invalid",)


def test_cold_miss_is_not_recomputed() -> None:
    record = _record()
    assert record.verdict is Verdict.VERIFIED_MISS
    assert record.reuse.unexpected_recomputed_tokens.value == 0


def test_eviction_requires_residency_absence_and_subsequent_miss() -> None:
    record = _record()
    reuse = replace(
        record.reuse,
        eviction_observed=EvidenceFact(
            value=True,
            basis=EvidenceBasis.OBSERVED,
            source="test",
        ),
    )
    classified = classify_request(
        replace(record, reuse=reuse, verdict=None, verdict_reasons=())
    )
    assert classified.verdict is Verdict.INVALID
    assert classified.verdict_reasons == ("eviction_proof_incomplete",)


def test_missing_semantic_prefix_cannot_verify_hit() -> None:
    record = ReferenceCacheAdapter().run(adversarial_requests()[:2])[1]
    classified = classify_request(
        replace(
            record,
            reuse=replace(
                record.reuse,
                semantic_prefix_tokens=unavailable("test", "semantic_prefix_missing"),
            ),
            verdict=None,
            verdict_reasons=(),
        )
    )
    assert classified.verdict is Verdict.ATTESTED_ONLY


def test_eviction_requires_independent_identity_and_policy_evidence() -> None:
    record = ReferenceCacheAdapter(max_entries=1).run(eviction_requests())[-1]
    assert record.verdict is Verdict.EVICTED
    classified = classify_request(
        replace(
            record,
            reuse=replace(
                record.reuse,
                semantic_prefix_tokens=unavailable("test", "identity_missing"),
            ),
            verdict=None,
            verdict_reasons=(),
        )
    )
    assert classified.verdict is Verdict.INVALID
    assert classified.verdict_reasons == ("eviction_identity_or_policy_unavailable",)


def test_recomputed_requires_attested_expected_reuse_equation() -> None:
    record = ReferenceCacheAdapter().run(adversarial_requests()[:2])[1]
    required = record.reuse.policy_required_prompt_tokens.value
    expected = record.reuse.policy_reusable_tokens.value
    assert isinstance(required, int)
    assert isinstance(expected, int)
    classified = classify_request(
        replace(
            record,
            reuse=replace(
                record.reuse,
                engine_cached_tokens=EvidenceFact(
                    value=expected - 1,
                    basis=EvidenceBasis.ENGINE_ATTESTED,
                    source="test",
                ),
                observed_prompt_tokens=EvidenceFact(
                    value=required + 1,
                    basis=EvidenceBasis.OBSERVED,
                    source="test",
                ),
                unexpected_recomputed_tokens=EvidenceFact(
                    value=1,
                    basis=EvidenceBasis.OBSERVED,
                    source="test",
                ),
            ),
            verdict=None,
            verdict_reasons=(),
        )
    )
    assert classified.verdict is Verdict.INVALID
    assert "engine_counter_equation_mismatch" in classified.verdict_reasons


def test_estimated_output_facts_are_not_claim_eligible() -> None:
    record = _record()
    classified = classify_request(
        replace(
            record,
            output=replace(
                record.output,
                token_identity=EvidenceFact(
                    value=True,
                    basis=EvidenceBasis.ESTIMATED,
                    source="test",
                ),
            ),
            verdict=None,
            verdict_reasons=(),
        )
    )
    assert classified.verdict is Verdict.VERIFIED_MISS
    assert classified.eligibility.output_equivalence is EligibilityStatus.INELIGIBLE


def test_real_output_equivalence_does_not_claim_model_quality() -> None:
    record = _record()
    classified = classify_request(
        replace(
            record,
            output=replace(
                record.output,
                token_identity=replace(
                    record.output.token_identity,
                    source="mlx.baseline_comparison",
                ),
                correctness=replace(
                    record.output.correctness,
                    source="mlx.baseline_comparison",
                ),
            ),
            verdict=None,
            verdict_reasons=(),
        )
    )
    assert classified.eligibility.output_equivalence is EligibilityStatus.ELIGIBLE
    assert classified.eligibility.quality is EligibilityStatus.UNAVAILABLE
