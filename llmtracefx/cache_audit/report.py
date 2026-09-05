"""Deterministic cache-audit summaries and claim matrices."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from llmtracefx.optimizer.schema import Measurement

from .schema import EvidenceFact, RequestEvidence


def _value(fact: EvidenceFact[Any]) -> Any:
    return fact.value


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
        f"prompt tokens processed. Therefore the cache claim is {verdict}."
    )


def build_claim_matrix(records: Sequence[RequestEvidence]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "request_id": record.spec.request_id,
                "scenario": record.spec.scenario.value,
                "expected_semantic_prefix_tokens": _value(
                    record.reuse.semantic_prefix_tokens
                ),
                "expected_policy_reusable_tokens": _value(
                    record.reuse.policy_reusable_tokens
                ),
                "engine_cached_tokens": _value(record.reuse.engine_cached_tokens),
                "observed_prompt_tokens": _value(record.reuse.observed_prompt_tokens),
                "unexpected_recomputed_tokens": _value(
                    record.reuse.unexpected_recomputed_tokens
                ),
                "client_ttft_ms": (_milliseconds(record.timing.client_ttft)),
                "in_process_first_token_ms": (
                    _milliseconds(record.timing.in_process_first_token)
                ),
                "logical_cache_bytes": _value(record.memory.logical_cache_bytes),
                "cache_entries_before": (
                    None
                    if record.cache_before is None
                    else _value(record.cache_before.entry_count)
                ),
                "cache_entries_after": (
                    None
                    if record.cache_after is None
                    else _value(record.cache_after.entry_count)
                ),
                "cache_event_count": len(record.events),
                "billed_cost": _value(record.cost.billed),
                "estimated_cost": _value(record.cost.estimated),
                "cost_currency": record.cost.currency,
                "output_identity": _value(record.output.token_identity),
                "correctness": _value(record.output.correctness),
                "verdict": (
                    "unclassified" if record.verdict is None else record.verdict.value
                ),
                "limitations": [item.code for item in record.limitations],
                "statement": claim_statement(record),
            }
        )
    return {"schema_version": "1", "rows": rows}


def build_summary(records: Sequence[RequestEvidence]) -> dict[str, Any]:
    verdicts = Counter(
        "unclassified" if record.verdict is None else record.verdict.value
        for record in records
    )
    pairs: dict[str, list[RequestEvidence]] = {}
    for record in records:
        if record.spec.pair_id is not None:
            pairs.setdefault(record.spec.pair_id, []).append(record)
    paired_deltas: list[dict[str, Any]] = []
    for pair_id, pair_records in sorted(pairs.items()):
        if len(pair_records) != 2:
            continue
        first, second = sorted(pair_records, key=lambda item: item.spec.order)
        first_ttft = first.timing.in_process_first_token
        second_ttft = second.timing.in_process_first_token
        first_ttft_ms = _milliseconds(first_ttft)
        second_ttft_ms = _milliseconds(second_ttft)
        first_memory = _value(first.memory.runtime_peak_bytes)
        second_memory = _value(second.memory.runtime_peak_bytes)
        paired_deltas.append(
            {
                "pair_id": pair_id,
                "first_request_id": first.spec.request_id,
                "second_request_id": second.spec.request_id,
                "first_token_delta_ms": (
                    None
                    if first_ttft_ms is None or second_ttft_ms is None
                    else second_ttft_ms - first_ttft_ms
                ),
                "runtime_peak_delta_bytes": (
                    None
                    if not isinstance(first_memory, int)
                    or not isinstance(second_memory, int)
                    else second_memory - first_memory
                ),
            }
        )
    return {
        "schema_version": "1",
        "request_count": len(records),
        "verdict_counts": dict(sorted(verdicts.items())),
        "paired_deltas": paired_deltas,
        "limitations": sorted(
            {item.code for record in records for item in record.limitations}
        ),
    }
