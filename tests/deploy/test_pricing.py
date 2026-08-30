"""Price quotes carry a date, and a stale date is a refusal."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from llmtracefx.deploy.errors import DeploymentPlanError
from llmtracefx.deploy.pricing import (
    DEFAULT_MAX_PRICE_AGE_DAYS,
    MAX_PLAUSIBLE_USD_PER_GIB_MONTH,
    MAX_PLAUSIBLE_USD_PER_GPU_HOUR,
    GpuPriceQuote,
    StorageQuote,
)


def test_a_quote_records_where_and_when_it_was_read() -> None:
    quote = GpuPriceQuote(
        gpu_type="H200",
        usd_per_gpu_hour=4.54,
        effective_date="2026-08-30",
        source="https://modal.com/pricing",
    )
    payload = quote.to_dict()
    assert payload["effective_date"] == "2026-08-30"
    assert payload["source"] == "https://modal.com/pricing"


def test_age_and_staleness_are_measured_against_a_supplied_date() -> None:
    quote = GpuPriceQuote(
        gpu_type="H200",
        usd_per_gpu_hour=4.54,
        effective_date="2026-01-01",
        source="https://modal.com/pricing",
    )
    assert quote.age_days(as_of=date(2026, 1, 31)) == 30
    assert quote.is_stale(as_of=date(2026, 1, 31)) is False
    assert quote.is_stale(as_of=date(2026, 12, 31)) is True
    assert quote.is_stale(as_of=date(2026, 1, 31), max_age_days=10) is True


def test_freshness_limit_boundary_is_inclusive() -> None:
    quote = GpuPriceQuote(
        gpu_type="H200",
        usd_per_gpu_hour=1.0,
        effective_date="2026-01-01",
        source="pricing page",
    )
    exactly_at_limit = date(2026, 1, 1) + timedelta(days=DEFAULT_MAX_PRICE_AGE_DAYS)
    assert quote.is_stale(as_of=exactly_at_limit) is False


def test_future_dated_quote_reports_a_negative_age_rather_than_zero() -> None:
    quote = GpuPriceQuote(
        gpu_type="H200",
        usd_per_gpu_hour=1.0,
        effective_date="2026-12-31",
        source="pricing page",
    )
    assert quote.age_days(as_of=date(2026, 1, 1)) < 0


@pytest.mark.parametrize(
    "bad_date", ["30-08-2026", "2026-8-30", "2026-13-01", "2026-02-30", "today", ""]
)
def test_malformed_effective_dates_are_refused(bad_date: str) -> None:
    with pytest.raises(DeploymentPlanError, match="effective_date"):
        GpuPriceQuote(
            gpu_type="H200",
            usd_per_gpu_hour=1.0,
            effective_date=bad_date,
            source="pricing page",
        )


@pytest.mark.parametrize("bad_rate", [0.0, -1.0])
def test_non_positive_rate_is_refused(bad_rate: float) -> None:
    with pytest.raises(DeploymentPlanError, match="greater than zero"):
        GpuPriceQuote(
            gpu_type="H200",
            usd_per_gpu_hour=bad_rate,
            effective_date="2026-08-30",
            source="pricing page",
        )


def test_implausible_rate_is_refused_as_a_units_mistake() -> None:
    with pytest.raises(DeploymentPlanError, match="check the units"):
        GpuPriceQuote(
            gpu_type="H200",
            usd_per_gpu_hour=MAX_PLAUSIBLE_USD_PER_GPU_HOUR + 1,
            effective_date="2026-08-30",
            source="pricing page",
        )


def test_source_is_mandatory_so_a_number_can_be_rechecked() -> None:
    with pytest.raises(DeploymentPlanError, match="source"):
        GpuPriceQuote(
            gpu_type="H200",
            usd_per_gpu_hour=1.0,
            effective_date="2026-08-30",
            source="   ",
        )


def test_storage_quote_applies_the_same_rules() -> None:
    with pytest.raises(DeploymentPlanError, match="check the units"):
        StorageQuote(
            usd_per_gib_month=MAX_PLAUSIBLE_USD_PER_GIB_MONTH + 1,
            effective_date="2026-08-30",
            source="pricing page",
        )
    with pytest.raises(DeploymentPlanError, match="effective_date"):
        StorageQuote(
            usd_per_gib_month=0.02, effective_date="nope", source="pricing page"
        )
    fresh = StorageQuote(
        usd_per_gib_month=0.02,
        effective_date="2026-08-01",
        source="https://modal.com/pricing",
    )
    assert fresh.age_days(as_of=date(2026, 8, 31)) == 30
