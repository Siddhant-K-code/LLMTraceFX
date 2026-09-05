"""Fail-closed cache-audit verdict classification."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .schema import (
    EvidenceBasis,
    EvidenceFact,
    RequestEvidence,
    TerminalState,
    Verdict,
)


def _integer(fact: EvidenceFact[Any]) -> int | None:
    value = fact.value
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _truth(fact: EvidenceFact[Any]) -> bool | None:
    return fact.value if isinstance(fact.value, bool) else None


def _basis_invalid(fact: EvidenceFact[Any], allowed: tuple[EvidenceBasis, ...]) -> bool:
    return fact.value is not None and fact.basis not in allowed


def classify_request(record: RequestEvidence) -> RequestEvidence:
    """Return ``record`` with a deterministic verdict and reason codes."""

    reasons: list[str] = []
    if record.terminal_state is TerminalState.FAILED:
        return replace(
            record,
            verdict=Verdict.INVALID,
            verdict_reasons=("terminal_failed",),
        )

    token_identity = _truth(record.output.token_identity)
    correctness = _truth(record.output.correctness)
    if token_identity is False or correctness is False:
        return replace(
            record,
            verdict=Verdict.INVALID,
            verdict_reasons=(
                "output_token_identity_mismatch"
                if token_identity is False
                else "output_incorrect",
            ),
        )

    if any(item.blocks_verdict for item in record.limitations):
        return replace(
            record,
            verdict=Verdict.UNSUPPORTED,
            verdict_reasons=tuple(
                f"unsupported:{item.code}"
                for item in record.limitations
                if item.blocks_verdict
            ),
        )

    if record.terminal_state is TerminalState.REFUSED:
        return replace(
            record,
            verdict=Verdict.UNSUPPORTED,
            verdict_reasons=("runtime_refused",),
        )

    if _truth(record.reuse.eviction_observed):
        return replace(record, verdict=Verdict.EVICTED, verdict_reasons=("evicted",))

    expected = _integer(record.reuse.policy_reusable_tokens)
    semantic_prefix = _integer(record.reuse.semantic_prefix_tokens)
    attested = _integer(record.reuse.engine_cached_tokens)
    observed_prompt = _integer(record.reuse.observed_prompt_tokens)
    required_prompt = _integer(record.reuse.policy_required_prompt_tokens)
    recomputed = _integer(record.reuse.unexpected_recomputed_tokens)

    basis_errors = []
    for name, fact, allowed in (
        (
            "semantic_prefix_tokens",
            record.reuse.semantic_prefix_tokens,
            (EvidenceBasis.INDEPENDENTLY_DERIVED,),
        ),
        (
            "policy_reusable_tokens",
            record.reuse.policy_reusable_tokens,
            (EvidenceBasis.INDEPENDENTLY_DERIVED,),
        ),
        (
            "engine_cached_tokens",
            record.reuse.engine_cached_tokens,
            (EvidenceBasis.ENGINE_ATTESTED,),
        ),
        (
            "observed_prompt_tokens",
            record.reuse.observed_prompt_tokens,
            (EvidenceBasis.OBSERVED,),
        ),
        (
            "policy_required_prompt_tokens",
            record.reuse.policy_required_prompt_tokens,
            (EvidenceBasis.INDEPENDENTLY_DERIVED,),
        ),
        (
            "unexpected_recomputed_tokens",
            record.reuse.unexpected_recomputed_tokens,
            (EvidenceBasis.OBSERVED, EvidenceBasis.INDEPENDENTLY_DERIVED),
        ),
        (
            "output_token_identity",
            record.output.token_identity,
            (EvidenceBasis.OBSERVED,),
        ),
        (
            "output_correctness",
            record.output.correctness,
            (EvidenceBasis.OBSERVED,),
        ),
    ):
        if _basis_invalid(fact, allowed):
            basis_errors.append(f"{name}_basis_invalid")
    if basis_errors:
        return replace(
            record,
            verdict=Verdict.INVALID,
            verdict_reasons=tuple(basis_errors),
        )

    if recomputed is not None and recomputed > 0:
        return replace(
            record,
            verdict=Verdict.RECOMPUTED,
            verdict_reasons=("unexpected_recomputed_tokens",),
        )

    if expected is None or observed_prompt is None:
        if attested is not None and attested > 0:
            return replace(
                record,
                verdict=Verdict.ATTESTED_ONLY,
                verdict_reasons=("independent_or_observed_evidence_unavailable",),
            )
        return replace(
            record,
            verdict=Verdict.UNSUPPORTED,
            verdict_reasons=("required_reuse_evidence_unavailable",),
        )

    if attested is None:
        return replace(
            record,
            verdict=Verdict.UNSUPPORTED,
            verdict_reasons=("engine_attestation_unavailable",),
        )

    if required_prompt is None:
        return replace(
            record,
            verdict=Verdict.UNSUPPORTED,
            verdict_reasons=("required_prompt_expectation_unavailable",),
        )

    if observed_prompt != required_prompt:
        reasons.append("observed_prompt_work_mismatch")
    if attested != expected:
        reasons.append("engine_attestation_mismatch")
    if reasons:
        return replace(
            record,
            verdict=Verdict.INVALID,
            verdict_reasons=tuple(reasons),
        )

    if expected == 0:
        verdict = Verdict.VERIFIED_MISS
        reasons.append("expected_and_observed_miss")
    elif (
        semantic_prefix is not None and semantic_prefix < record.spec.input_token_count
    ):
        verdict = Verdict.PARTIAL_REUSE
        reasons.append("expected_attested_observed_partial_reuse")
    else:
        verdict = Verdict.VERIFIED_HIT
        reasons.append("expected_attested_observed_full_reuse")

    if (
        record.output.token_identity.basis is EvidenceBasis.UNAVAILABLE
        or record.output.correctness.basis is EvidenceBasis.UNAVAILABLE
    ):
        return replace(
            record,
            verdict=Verdict.ATTESTED_ONLY,
            verdict_reasons=("output_identity_or_correctness_unavailable",),
        )
    return replace(record, verdict=verdict, verdict_reasons=tuple(reasons))
