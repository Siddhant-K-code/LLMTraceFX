from __future__ import annotations

from dataclasses import replace

from llmtracefx.cache_audit.adapters.reference import ReferenceCacheAdapter
from llmtracefx.cache_audit.schema import (
    EvidenceBasis,
    EvidenceFact,
    Limitation,
    Verdict,
    unavailable,
)
from llmtracefx.cache_audit.verdicts import classify_request
from llmtracefx.cache_audit.workloads import adversarial_requests


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
    record = _record()
    reuse = replace(
        record.reuse,
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
    record = _record()
    reuse = replace(
        record.reuse,
        policy_reusable_tokens=unavailable("test", "identity_redacted"),
        engine_cached_tokens=EvidenceFact(
            value=4,
            basis=EvidenceBasis.ENGINE_ATTESTED,
            source="test",
        ),
    )
    classified = classify_request(
        replace(record, reuse=reuse, verdict=None, verdict_reasons=())
    )
    assert classified.verdict is Verdict.ATTESTED_ONLY


def test_failed_correctness_is_invalid() -> None:
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
    assert classified.verdict is Verdict.INVALID


def test_failed_token_identity_is_invalid_even_when_correctness_is_true() -> None:
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
    assert classified.verdict is Verdict.INVALID
    assert classified.verdict_reasons == ("output_token_identity_mismatch",)


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
