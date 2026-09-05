"""Deterministic cache-audit summaries and claim matrices."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from llmtracefx.optimizer.schema import Measurement

from .schema import AuditManifest, EvidenceFact, PairRole, RequestEvidence


def _value(fact: EvidenceFact[Any]) -> Any:
    return fact.value


def _fact(fact: EvidenceFact[Any]) -> dict[str, Any]:
    return {
        "value": fact.value,
        "basis": fact.basis.value,
        "source": fact.source,
        "scope": fact.scope,
        "limitations": list(fact.limitations),
    }


def _milliseconds(measurement: Measurement | None) -> float | None:
    if measurement is None:
        return None
    if measurement.unit == "ms":
        return measurement.value
    if measurement.unit == "s":
        return measurement.value * 1000
    return None


def claim_statement(record: RequestEvidence) -> str:
    """Render the evidence-first X/Y/Z sentence for one request."""

    reuse = record.reuse
    reported = _value(reuse.engine_cached_tokens)
    expected = _value(reuse.policy_reusable_tokens)
    observed = _value(reuse.observed_prompt_tokens)
    verdict = "unclassified" if record.verdict is None else record.verdict.value
    return (
        f"The engine reported {reported if reported is not None else 'unavailable'} "
        "cached tokens. Given the exact input and cache state, we independently "
        f"expected {expected if expected is not None else 'unavailable'}. "
        f"We observed {observed if observed is not None else 'unavailable'} "
        f"prompt-policy operations. Therefore the cache claim is {verdict}; "
        f"output equivalence is {record.eligibility.output_equivalence.value}, and "
        f"performance attribution is {record.eligibility.performance.value}."
    )


def build_claim_matrix(records: Sequence[RequestEvidence]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "request_id": record.spec.request_id,
                "scenario": record.spec.scenario.value,
                "replicate_id": record.spec.replicate_id,
                "pair_id": record.spec.pair_id,
                "pair_role": record.spec.pair_role.value,
                "semantic_prefix": _fact(record.reuse.semantic_prefix_tokens),
                "expected_policy_reuse": _fact(record.reuse.policy_reusable_tokens),
                "engine_cached": _fact(record.reuse.engine_cached_tokens),
                "observed_prompt_work": _fact(record.reuse.observed_prompt_tokens),
                "unexpected_recomputed": _fact(
                    record.reuse.unexpected_recomputed_tokens
                ),
                "client_ttft_ms": _milliseconds(record.timing.client_ttft),
                "in_process_first_token_ms": _milliseconds(
                    record.timing.in_process_first_token
                ),
                "timing_scope": record.timing.scope,
                "timing_exclusions": list(record.timing.exclusions),
                "logical_cache_memory": _fact(record.memory.logical_cache_bytes),
                "runtime_peak_memory": _fact(record.memory.runtime_peak_bytes),
                "cache_entries_before": (
                    None
                    if record.cache_before is None
                    else _fact(record.cache_before.entry_count)
                ),
                "cache_entries_after": (
                    None
                    if record.cache_after is None
                    else _fact(record.cache_after.entry_count)
                ),
                "cache_event_count": len(record.events),
                "output_identity": _fact(record.output.token_identity),
                "correctness": _fact(record.output.correctness),
                "cache_reuse_verdict": (
                    "unclassified" if record.verdict is None else record.verdict.value
                ),
                "verdict_reasons": list(record.verdict_reasons),
                "claim_eligibility": record.eligibility.to_dict(),
                "limitations": [item.to_dict() for item in record.limitations],
                "statement": claim_statement(record),
            }
        )
    return {"schema_version": "2", "rows": rows}


def _measurement_compatible(
    control: RequestEvidence, treatment: RequestEvidence
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if control.timing.scope != treatment.timing.scope:
        reasons.append("timing_scope_mismatch")
    if control.timing.exclusions != treatment.timing.exclusions:
        reasons.append("timing_exclusions_mismatch")
    left = control.timing.in_process_first_token
    right = treatment.timing.in_process_first_token
    if (left is None) != (right is None):
        reasons.append("timing_availability_mismatch")
    elif left is not None and right is not None:
        if left.unit != right.unit or left.provenance != right.provenance:
            reasons.append("timing_basis_mismatch")
    left_memory = control.memory.runtime_peak_bytes
    right_memory = treatment.memory.runtime_peak_bytes
    if (
        left_memory.basis != right_memory.basis
        or left_memory.scope != right_memory.scope
        or left_memory.source != right_memory.source
    ):
        reasons.append("memory_basis_or_scope_mismatch")
    return not reasons, reasons


def build_summary(
    records: Sequence[RequestEvidence],
    manifest: AuditManifest | None = None,
) -> dict[str, Any]:
    verdicts = Counter(
        "unclassified" if record.verdict is None else record.verdict.value
        for record in records
    )
    pairs: dict[tuple[str, str], list[RequestEvidence]] = {}
    for record in records:
        if record.spec.pair_id is not None:
            pairs.setdefault(
                (record.spec.pair_id, record.spec.replicate_id), []
            ).append(record)
    paired_deltas: list[dict[str, Any]] = []
    for (pair_id, replicate_id), pair_records in sorted(pairs.items()):
        reasons: list[str] = []
        if manifest is None:
            reasons.append("manifest_provenance_unavailable")
        controls = [
            record
            for record in pair_records
            if record.spec.pair_role is PairRole.CONTROL
        ]
        treatments = [
            record
            for record in pair_records
            if record.spec.pair_role is PairRole.TREATMENT
        ]
        if len(controls) != 1 or len(treatments) != 1:
            reasons.append("pair_roles_incomplete_or_ambiguous")
            control = treatment = None
        else:
            control, treatment = controls[0], treatments[0]
            compatible, compatibility_reasons = _measurement_compatible(
                control, treatment
            )
            if not compatible:
                reasons.extend(compatibility_reasons)
            if control.spec.order >= treatment.spec.order:
                reasons.append("pair_order_invalid")
        eligible = not reasons
        control_ttft = (
            None
            if control is None
            else _milliseconds(control.timing.in_process_first_token)
        )
        treatment_ttft = (
            None
            if treatment is None
            else _milliseconds(treatment.timing.in_process_first_token)
        )
        control_memory = (
            None if control is None else _value(control.memory.runtime_peak_bytes)
        )
        treatment_memory = (
            None if treatment is None else _value(treatment.memory.runtime_peak_bytes)
        )
        paired_deltas.append(
            {
                "run_id": None if manifest is None else manifest.run_id,
                "pair_id": pair_id,
                "replicate_id": replicate_id,
                "sample_count": 1,
                "control_request_id": (
                    None if control is None else control.spec.request_id
                ),
                "treatment_request_id": (
                    None if treatment is None else treatment.spec.request_id
                ),
                "compatible": eligible,
                "eligibility_reasons": reasons,
                "timing_basis": (None if control is None else control.timing.scope),
                "memory_basis": (
                    None
                    if control is None
                    else {
                        "basis": control.memory.runtime_peak_bytes.basis.value,
                        "scope": control.memory.runtime_peak_bytes.scope,
                        "source": control.memory.runtime_peak_bytes.source,
                    }
                ),
                "first_token_delta_ms": (
                    treatment_ttft - control_ttft
                    if eligible
                    and control_ttft is not None
                    and treatment_ttft is not None
                    else None
                ),
                "runtime_peak_delta_bytes": (
                    treatment_memory - control_memory
                    if eligible
                    and isinstance(control_memory, int)
                    and isinstance(treatment_memory, int)
                    else None
                ),
                "uncertainty_available": False,
                "causal_interpretation_eligible": False,
                "causal_limitation": "single_pair_has_no_uncertainty_estimate",
            }
        )
    return {
        "schema_version": "2",
        "run_id": None if manifest is None else manifest.run_id,
        "request_count": len(records),
        "verdict_counts": dict(sorted(verdicts.items())),
        "paired_deltas": paired_deltas,
        "limitations": sorted(
            {item.code for record in records for item in record.limitations}
        ),
    }
