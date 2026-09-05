"""Fail-closed cache verdict and independent claim-eligibility classification."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .expected import longest_common_prefix
from .schema import (
    ClaimEligibility,
    EligibilityStatus,
    EvidenceBasis,
    EvidenceFact,
    RequestEvidence,
    TerminalState,
    Verdict,
)

_DERIVED = (EvidenceBasis.INDEPENDENTLY_DERIVED,)
_ATTESTED = (EvidenceBasis.ENGINE_ATTESTED,)
_OBSERVED = (EvidenceBasis.OBSERVED,)
_OBSERVED_OR_DERIVED = (EvidenceBasis.OBSERVED, EvidenceBasis.INDEPENDENTLY_DERIVED)


def _integer(fact: EvidenceFact[Any]) -> int | None:
    value = fact.value
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _truth(fact: EvidenceFact[Any]) -> bool | None:
    return fact.value if isinstance(fact.value, bool) else None


def _basis_error(
    name: str,
    fact: EvidenceFact[Any],
    allowed: tuple[EvidenceBasis, ...],
) -> str | None:
    if fact.value is not None and fact.basis not in allowed:
        return f"{name}_basis_invalid"
    return None


def _output_eligibility(record: RequestEvidence) -> ClaimEligibility:
    identity = _truth(record.output.token_identity)
    correctness = _truth(record.output.correctness)
    reasons: list[str] = []
    if (
        identity is None
        or correctness is None
        or record.output.output_token_ids is None
        or record.output.baseline_token_ids is None
    ):
        output = EligibilityStatus.UNAVAILABLE
        reasons.append("output_equivalence_unavailable")
    elif (
        record.output.token_identity.basis is not EvidenceBasis.OBSERVED
        or record.output.correctness.basis is not EvidenceBasis.OBSERVED
    ):
        output = EligibilityStatus.INELIGIBLE
        reasons.append("output_comparison_must_be_observed")
    elif identity and correctness:
        output = EligibilityStatus.ELIGIBLE
    else:
        output = EligibilityStatus.INELIGIBLE
        reasons.append("output_equivalence_failed")

    synthetic = any(
        fact.source.startswith("synthetic_engine.")
        or fact.source.startswith("synthetic_baseline.")
        for fact in (record.output.token_identity, record.output.correctness)
    )
    if synthetic:
        quality = EligibilityStatus.NOT_APPLICABLE
        reasons.append("synthetic_equivalence_is_not_model_quality")
    else:
        quality = EligibilityStatus.UNAVAILABLE
        reasons.append("model_quality_evaluator_unavailable")

    performance = EligibilityStatus.INELIGIBLE
    reasons.append("paired_replicated_performance_evidence_required")
    return ClaimEligibility(
        output_equivalence=output,
        performance=performance,
        quality=quality,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def _invalid(record: RequestEvidence, reasons: list[str]) -> RequestEvidence:
    return replace(
        record,
        verdict=Verdict.INVALID,
        verdict_reasons=tuple(reasons),
        eligibility=_output_eligibility(record),
    )


def _eviction_proof_errors(record: RequestEvidence) -> list[str]:
    proof = record.eviction_predecessor
    if proof is None:
        return ["eviction_predecessor_proof_missing"]
    predecessor = proof.predecessor
    current = proof.current
    errors: list[str] = []
    if proof.predecessor_request_id not in record.spec.expected_predecessors:
        errors.append("eviction_predecessor_id_unbound")
    if (
        record.spec.input_token_ids is None
        or current.input_token_ids != record.spec.input_token_ids
        or current.namespace_id != record.spec.namespace_id
    ):
        errors.append("eviction_current_request_identity_mismatch")
    for field in (
        "backend",
        "model_id",
        "tokenizer_id",
        "model_artifact_digest",
        "cache_config_digest",
        "namespace_id",
    ):
        if getattr(predecessor, field) != getattr(current, field):
            errors.append(f"eviction_predecessor_{field}_mismatch")
    reusable = min(
        max(0, len(current.input_token_ids) - 1),
        longest_common_prefix(
            predecessor.input_token_ids,
            current.input_token_ids,
        ),
    )
    if reusable <= 0 or proof.reusable_prefix_tokens != reusable:
        errors.append("eviction_predecessor_not_reuse_producing")
    return errors


def classify_request(record: RequestEvidence) -> RequestEvidence:
    """Classify cache reuse without allowing output results to rewrite cache truth."""

    eligibility = _output_eligibility(record)
    if record.terminal_state is TerminalState.FAILED:
        return replace(
            record,
            verdict=Verdict.INVALID,
            verdict_reasons=("terminal_failed",),
            eligibility=eligibility,
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
            eligibility=eligibility,
        )
    if record.terminal_state is TerminalState.REFUSED:
        return replace(
            record,
            verdict=Verdict.UNSUPPORTED,
            verdict_reasons=("runtime_refused",),
            eligibility=eligibility,
        )

    basis_errors = [
        error
        for error in (
            _basis_error(
                "semantic_prefix_tokens", record.reuse.semantic_prefix_tokens, _DERIVED
            ),
            _basis_error(
                "policy_reusable_tokens", record.reuse.policy_reusable_tokens, _DERIVED
            ),
            _basis_error(
                "engine_cached_tokens", record.reuse.engine_cached_tokens, _ATTESTED
            ),
            _basis_error(
                "engine_created_tokens",
                record.reuse.engine_created_tokens,
                _ATTESTED + _DERIVED,
            ),
            _basis_error(
                "observed_prompt_tokens", record.reuse.observed_prompt_tokens, _OBSERVED
            ),
            _basis_error(
                "policy_required_prompt_tokens",
                record.reuse.policy_required_prompt_tokens,
                _DERIVED,
            ),
            _basis_error(
                "unexpected_recomputed_tokens",
                record.reuse.unexpected_recomputed_tokens,
                _OBSERVED_OR_DERIVED,
            ),
            _basis_error(
                "prior_residency_observed",
                record.reuse.prior_residency_observed,
                _OBSERVED,
            ),
            _basis_error(
                "residency_absence_observed",
                record.reuse.residency_absence_observed,
                _OBSERVED,
            ),
            _basis_error(
                "eviction_observed", record.reuse.eviction_observed, _OBSERVED
            ),
        )
        if error is not None
    ]
    if basis_errors:
        return _invalid(record, basis_errors)

    semantic = _integer(record.reuse.semantic_prefix_tokens)
    expected = _integer(record.reuse.policy_reusable_tokens)
    attested = _integer(record.reuse.engine_cached_tokens)
    created = _integer(record.reuse.engine_created_tokens)
    observed = _integer(record.reuse.observed_prompt_tokens)
    required = _integer(record.reuse.policy_required_prompt_tokens)
    recomputed = _integer(record.reuse.unexpected_recomputed_tokens)
    prior_resident = _truth(record.reuse.prior_residency_observed)
    residency_absent = _truth(record.reuse.residency_absence_observed)
    evicted = _truth(record.reuse.eviction_observed)
    input_count = record.spec.input_token_count

    contradictions: list[str] = []
    for name, value in (
        ("semantic_prefix_tokens", semantic),
        ("policy_reusable_tokens", expected),
        ("engine_cached_tokens", attested),
        ("engine_created_tokens", created),
        ("policy_required_prompt_tokens", required),
        ("unexpected_recomputed_tokens", recomputed),
    ):
        if value is not None and not 0 <= value <= input_count:
            contradictions.append(f"{name}_out_of_range")
    if observed is not None and observed < 0:
        contradictions.append("observed_prompt_tokens_out_of_range")
    if semantic is not None and expected is not None and expected > semantic:
        contradictions.append("policy_reuse_exceeds_semantic_prefix")
    if (
        expected is not None
        and required is not None
        and expected + required != input_count
    ):
        contradictions.append("policy_reuse_equation_mismatch")
    if (
        attested is not None
        and created is not None
        and attested + created != input_count
    ):
        contradictions.append("engine_counter_equation_mismatch")
    if observed is not None and required is not None and recomputed is not None:
        if recomputed != max(0, observed - required):
            contradictions.append("recomputation_equation_mismatch")
    if evicted is True and not (prior_resident is True and residency_absent is True):
        contradictions.append("eviction_proof_incomplete")
    if evicted is not True and record.eviction_predecessor is not None:
        contradictions.append("eviction_predecessor_without_observed_eviction")
    if contradictions:
        return _invalid(record, contradictions)

    if evicted is True:
        proof_errors = _eviction_proof_errors(record)
        if proof_errors:
            return _invalid(record, proof_errors)
        if semantic is None or expected is None:
            return _invalid(record, ["eviction_identity_or_policy_unavailable"])
        if expected != 0:
            return _invalid(record, ["eviction_expected_reuse_nonzero"])
        if attested == 0 and observed == input_count:
            return replace(
                record,
                verdict=Verdict.EVICTED,
                verdict_reasons=(
                    "prior_residency_observed",
                    "controlled_absence_observed",
                    "subsequent_miss_observed",
                ),
                eligibility=eligibility,
            )
        return _invalid(record, ["eviction_subsequent_miss_not_observed"])

    if expected is not None and attested is not None and attested != expected:
        return _invalid(record, ["engine_attestation_mismatch"])

    if recomputed is not None and recomputed > 0:
        if (
            semantic is not None
            and expected is not None
            and expected > 0
            and semantic >= expected
            and attested == expected
            and prior_resident is True
            and observed is not None
            and required is not None
            and observed > required
        ):
            return replace(
                record,
                verdict=Verdict.RECOMPUTED,
                verdict_reasons=(
                    "previously_reusable_state_observed",
                    "observed_prompt_work_exceeds_policy_requirement",
                ),
                eligibility=eligibility,
            )
        return _invalid(record, ["recomputation_without_reusable_resident_state"])

    if expected is None or semantic is None or observed is None:
        verdict = (
            Verdict.ATTESTED_ONLY
            if attested is not None and attested > 0
            else Verdict.UNSUPPORTED
        )
        reason = (
            "independent_or_observed_reuse_evidence_unavailable"
            if verdict is Verdict.ATTESTED_ONLY
            else "required_reuse_evidence_unavailable"
        )
        return replace(
            record,
            verdict=verdict,
            verdict_reasons=(reason,),
            eligibility=eligibility,
        )
    if attested is None:
        return replace(
            record,
            verdict=Verdict.UNSUPPORTED,
            verdict_reasons=("engine_attestation_unavailable",),
            eligibility=eligibility,
        )
    if required is None:
        return replace(
            record,
            verdict=Verdict.UNSUPPORTED,
            verdict_reasons=("required_prompt_expectation_unavailable",),
            eligibility=eligibility,
        )

    mismatches = []
    if observed != required:
        mismatches.append("observed_prompt_work_mismatch")
    if attested != expected:
        mismatches.append("engine_attestation_mismatch")
    if mismatches:
        return _invalid(record, mismatches)

    if expected == 0:
        verdict = Verdict.VERIFIED_MISS
        reasons = ("expected_attested_observed_miss",)
    elif semantic < input_count:
        verdict = Verdict.PARTIAL_REUSE
        reasons = ("expected_attested_observed_partial_reuse",)
    else:
        verdict = Verdict.VERIFIED_HIT
        reasons = ("semantic_policy_attested_observed_full_reuse",)
    return replace(
        record,
        verdict=verdict,
        verdict_reasons=reasons,
        eligibility=eligibility,
    )
