"""Tests for the versioned pricing manifest and the cost derivation from it.

Everything here is synthetic. No rate in this file is a real price, and no
test in this file contacts a provider.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from llmtracefx.optimizer.compare.cost import (
    MAX_BILLABLE_TOKEN_COUNT,
    MONETARY_BASIS,
    TokenUsage,
    correct_cases_per_currency_unit,
    cost_per_case,
    estimate_run_cost,
)
from llmtracefx.optimizer.compare.pricing import (
    PRICING_MANIFEST_SCHEMA_VERSION,
    PricingEntry,
    PricingError,
    PricingManifest,
)


def _entry(**overrides: object) -> PricingEntry:
    payload: dict[str, object] = {
        "entry_id": "example",
        "provider": "z-ai",
        "model_id": "glm-5.3",
        "currency": "USD",
        "effective_at": "2026-01-01",
        "source": "illustrative example",
        "rates_are_illustrative": True,
        "input_per_million": 1.0,
        "output_per_million": 2.0,
    }
    payload.update(overrides)
    return PricingEntry.from_dict(payload)


def _manifest(*entries: PricingEntry, currency: str = "USD") -> PricingManifest:
    return PricingManifest(
        schema_version=PRICING_MANIFEST_SCHEMA_VERSION,
        currency=currency,
        entries=entries,
    )


# --- Manifest schema ------------------------------------------------------


def test_entry_requires_iso_currency_code() -> None:
    with pytest.raises(PricingError, match="ISO 4217"):
        _entry(currency="usd")
    with pytest.raises(PricingError, match="ISO 4217"):
        _entry(currency="$")


def test_entry_requires_a_parsable_effective_date() -> None:
    with pytest.raises(PricingError, match="effective_at"):
        _entry(effective_at="whenever")
    assert _entry(effective_at="2026-01-01T00:00:00Z").effective_at.endswith("Z")


def test_entry_requires_a_source_reference() -> None:
    payload = {
        "entry_id": "e",
        "provider": "p",
        "model_id": "m",
        "currency": "USD",
        "effective_at": "2026-01-01",
        "rates_are_illustrative": True,
        "input_per_million": 1.0,
    }
    with pytest.raises(PricingError, match="'source'"):
        PricingEntry.from_dict(payload)


def test_entry_requires_an_explicit_illustrative_flag() -> None:
    payload = {
        "entry_id": "e",
        "provider": "p",
        "model_id": "m",
        "currency": "USD",
        "effective_at": "2026-01-01",
        "source": "somewhere",
        "input_per_million": 1.0,
    }
    with pytest.raises(PricingError, match="rates_are_illustrative"):
        PricingEntry.from_dict(payload)


def test_entry_with_no_rate_at_all_is_rejected() -> None:
    with pytest.raises(PricingError, match="declares no rate"):
        _entry(input_per_million=None, output_per_million=None)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_rates_are_rejected(bad: float) -> None:
    with pytest.raises(PricingError, match="finite"):
        _entry(input_per_million=bad)


def test_negative_rates_are_rejected() -> None:
    with pytest.raises(PricingError, match=">= 0"):
        _entry(output_per_million=-0.01)


def test_boolean_rates_are_rejected_rather_than_coerced() -> None:
    with pytest.raises(PricingError, match="must be a number"):
        _entry(input_per_million=True)


def test_zero_is_a_legitimate_rate() -> None:
    entry = _entry(input_per_million=0.0)
    assert entry.input_per_million == 0.0


def test_manifest_rejects_currency_mixing() -> None:
    with pytest.raises(PricingError, match="mixing currencies"):
        _manifest(_entry(currency="EUR"), currency="USD")


def test_manifest_rejects_duplicate_entry_ids() -> None:
    with pytest.raises(PricingError, match="duplicate entry_id"):
        _manifest(_entry(), _entry(model_id="glm-5.3-flash"))


def test_manifest_rejects_ambiguous_entries_at_load_time() -> None:
    with pytest.raises(PricingError, match="ambiguous manifest"):
        _manifest(
            _entry(entry_id="a"),
            _entry(entry_id="b"),
        )


def test_manifest_allows_distinct_revisions_for_one_model() -> None:
    manifest = _manifest(
        _entry(entry_id="a", model_revision="2026-01"),
        _entry(entry_id="b", model_revision="2026-06"),
    )
    resolved = manifest.resolve(
        provider="z-ai", model_id="glm-5.3", model_revision="2026-06"
    )
    assert resolved is not None and resolved.entry_id == "b"


def test_manifest_rejects_a_wildcard_alongside_a_pinned_revision() -> None:
    with pytest.raises(PricingError, match="ambiguous manifest"):
        _manifest(
            _entry(entry_id="wildcard"),
            _entry(entry_id="pinned", model_revision="2026-06"),
        )


def test_manifest_requires_at_least_one_entry() -> None:
    with pytest.raises(PricingError, match="at least one entry"):
        _manifest()


def test_manifest_rejects_unknown_schema_version() -> None:
    with pytest.raises(PricingError, match="unsupported pricing manifest"):
        PricingManifest(schema_version="99", currency="USD", entries=(_entry(),))


def test_manifest_from_file_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "rates.txt"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(PricingError, match="unsupported pricing manifest extension"):
        PricingManifest.from_file(path)


def test_manifest_from_file_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "rates.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(PricingError, match="invalid JSON"):
        PricingManifest.from_file(path)


def test_manifest_round_trips_through_json(tmp_path: Path) -> None:
    manifest = _manifest(_entry())
    path = tmp_path / "rates.json"
    path.write_text(manifest.to_json(), encoding="utf-8")
    assert PricingManifest.from_file(path).to_dict() == manifest.to_dict()


# --- Matching -------------------------------------------------------------


def test_resolve_never_matches_a_local_system_without_a_provider() -> None:
    manifest = _manifest(_entry())
    assert (
        manifest.resolve(provider=None, model_id="glm-5.3", model_revision=None) is None
    )


def test_resolve_requires_an_exact_provider_and_model_match() -> None:
    manifest = _manifest(_entry())
    assert (
        manifest.resolve(provider="Z-AI", model_id="glm-5.3", model_revision=None)
        is None
    )
    assert (
        manifest.resolve(provider="z-ai", model_id="glm-5.3-flash", model_revision=None)
        is None
    )


def test_pinned_revision_entry_does_not_match_another_revision() -> None:
    manifest = _manifest(_entry(model_revision="2026-01"))
    assert (
        manifest.resolve(provider="z-ai", model_id="glm-5.3", model_revision="2026-06")
        is None
    )


def test_resolve_refuses_ambiguity_built_in_code() -> None:
    # Bypasses the constructor check the same way a programmatic caller could.
    manifest = _manifest(_entry(entry_id="a"))
    object.__setattr__(
        manifest, "entries", (_entry(entry_id="a"), _entry(entry_id="b"))
    )
    with pytest.raises(PricingError, match="refusing to pick one"):
        manifest.resolve(provider="z-ai", model_id="glm-5.3", model_revision=None)


def test_entry_hash_changes_when_content_changes() -> None:
    assert _entry().content_sha256 != _entry(input_per_million=1.5).content_sha256


def test_entry_hash_is_stable_across_key_order() -> None:
    first = PricingEntry.from_dict(json.loads(json.dumps(_entry().to_dict())))
    assert first.content_sha256 == _entry().content_sha256


# --- Cost derivation ------------------------------------------------------


def test_simple_cost_uses_per_million_rates() -> None:
    breakdown = estimate_run_cost(
        TokenUsage(prompt_tokens=1_000_000, completion_tokens=500_000), _entry()
    )
    assert breakdown.amount == pytest.approx(1.0 + 1.0)
    assert breakdown.currency == "USD"
    assert breakdown.to_dict()["monetary_basis"] == MONETARY_BASIS
    assert breakdown.to_dict()["estimated"] is True


def test_missing_prompt_tokens_makes_cost_unavailable_not_zero() -> None:
    breakdown = estimate_run_cost(
        TokenUsage(prompt_tokens=None, completion_tokens=10), _entry()
    )
    assert breakdown.amount is None
    assert any("prompt_tokens" in reason for reason in breakdown.reasons)


def test_missing_completion_tokens_makes_cost_unavailable() -> None:
    breakdown = estimate_run_cost(
        TokenUsage(prompt_tokens=10, completion_tokens=None), _entry()
    )
    assert breakdown.amount is None
    assert any("completion_tokens" in reason for reason in breakdown.reasons)


def test_cached_tokens_are_not_double_counted() -> None:
    entry = _entry(cached_input_per_million=0.1)
    breakdown = estimate_run_cost(
        TokenUsage(
            prompt_tokens=1_000_000,
            completion_tokens=0,
            cached_prompt_tokens=400_000,
        ),
        entry,
    )
    # 600k billed at 1.0/M plus 400k billed at 0.1/M.
    assert breakdown.amount == pytest.approx(0.6 + 0.04)
    assert breakdown.billed_input_tokens == 600_000
    assert breakdown.billed_cached_tokens == 400_000


def test_cached_usage_without_a_cached_rate_refuses_to_guess() -> None:
    breakdown = estimate_run_cost(
        TokenUsage(prompt_tokens=1000, completion_tokens=10, cached_prompt_tokens=500),
        _entry(),
    )
    assert breakdown.amount is None
    assert any("cached_input_per_million" in reason for reason in breakdown.reasons)


def test_cached_tokens_exceeding_prompt_tokens_is_refused() -> None:
    breakdown = estimate_run_cost(
        TokenUsage(prompt_tokens=10, completion_tokens=1, cached_prompt_tokens=11),
        _entry(cached_input_per_million=0.1),
    )
    assert breakdown.amount is None
    assert any("internally inconsistent" in reason for reason in breakdown.reasons)


def test_zero_cached_tokens_does_not_require_a_cached_rate() -> None:
    breakdown = estimate_run_cost(
        TokenUsage(
            prompt_tokens=1_000_000, completion_tokens=0, cached_prompt_tokens=0
        ),
        _entry(),
    )
    assert breakdown.amount == pytest.approx(1.0)


def test_unreported_reasoning_is_billed_as_output_and_flagged() -> None:
    breakdown = estimate_run_cost(
        TokenUsage(prompt_tokens=0, completion_tokens=1_000_000), _entry()
    )
    assert breakdown.amount == pytest.approx(2.0)
    assert breakdown.billed_output_tokens == 1_000_000
    assert breakdown.billed_reasoning_tokens is None
    assert any("no hidden reasoning cost" in reason for reason in breakdown.reasons)


def test_reasoning_rate_without_reported_reasoning_refuses_to_infer() -> None:
    breakdown = estimate_run_cost(
        TokenUsage(prompt_tokens=0, completion_tokens=100),
        _entry(reasoning_per_million=5.0),
    )
    assert breakdown.amount is None
    assert any("refusing to infer" in reason for reason in breakdown.reasons)


def test_reported_reasoning_is_split_out_when_a_rate_exists() -> None:
    breakdown = estimate_run_cost(
        TokenUsage(
            prompt_tokens=0, completion_tokens=1_000_000, reasoning_tokens=400_000
        ),
        _entry(reasoning_per_million=5.0),
    )
    # 600k output at 2.0/M plus 400k reasoning at 5.0/M.
    assert breakdown.amount == pytest.approx(1.2 + 2.0)
    assert breakdown.billed_output_tokens == 600_000
    assert breakdown.billed_reasoning_tokens == 400_000


def test_reasoning_exceeding_completion_tokens_is_refused() -> None:
    breakdown = estimate_run_cost(
        TokenUsage(prompt_tokens=0, completion_tokens=10, reasoning_tokens=11),
        _entry(reasoning_per_million=5.0),
    )
    assert breakdown.amount is None
    assert any("internally inconsistent" in reason for reason in breakdown.reasons)


def test_missing_output_rate_for_nonzero_output_makes_cost_unavailable() -> None:
    breakdown = estimate_run_cost(
        TokenUsage(prompt_tokens=10, completion_tokens=10),
        _entry(output_per_million=None),
    )
    assert breakdown.amount is None
    assert any("output_per_million" in reason for reason in breakdown.reasons)


def test_missing_rate_for_zero_tokens_is_harmless() -> None:
    breakdown = estimate_run_cost(
        TokenUsage(prompt_tokens=1_000_000, completion_tokens=0),
        _entry(output_per_million=None),
    )
    assert breakdown.amount == pytest.approx(1.0)


@pytest.mark.parametrize(
    "field",
    ["prompt_tokens", "completion_tokens", "cached_prompt_tokens", "reasoning_tokens"],
)
def test_an_unbillable_token_count_is_refused_not_raised(field: str) -> None:
    """``tokens * rate`` on a huge int raises ``OverflowError``.

    That is an ``ArithmeticError``, so nothing downstream catches it and it
    would escape ``compare`` as a raw traceback. This function is public, so
    it holds the line itself rather than relying on the loader's bound.
    """
    fields: dict[str, int] = {"prompt_tokens": 10, "completion_tokens": 10}
    fields[field] = MAX_BILLABLE_TOKEN_COUNT + 1
    breakdown = estimate_run_cost(
        TokenUsage(**fields), _entry(reasoning_per_million=1.0)
    )
    assert breakdown.amount is None
    assert any("billed exactly" in reason for reason in breakdown.reasons)


def test_an_enormous_count_never_raises_out_of_the_cost_helper() -> None:
    breakdown = estimate_run_cost(
        TokenUsage(prompt_tokens=10**400, completion_tokens=5), _entry()
    )
    assert breakdown.amount is None


def test_a_count_at_the_exact_bound_is_still_billed() -> None:
    breakdown = estimate_run_cost(
        TokenUsage(prompt_tokens=MAX_BILLABLE_TOKEN_COUNT, completion_tokens=0),
        _entry(),
    )
    assert breakdown.amount is not None
    assert math.isfinite(breakdown.amount)


# --- Derived ratios --------------------------------------------------------


def test_cost_per_case_is_unavailable_without_cases() -> None:
    assert cost_per_case(1.0, 0) is None
    assert cost_per_case(None, 5) is None
    assert cost_per_case(2.0, 4) == pytest.approx(0.5)


def test_correct_cases_per_currency_unit_is_undefined_at_zero_cost() -> None:
    assert correct_cases_per_currency_unit(5, 0.0) is None
    assert correct_cases_per_currency_unit(0, 1.0) is None
    assert correct_cases_per_currency_unit(5, None) is None
    assert correct_cases_per_currency_unit(4, 2.0) == pytest.approx(2.0)
