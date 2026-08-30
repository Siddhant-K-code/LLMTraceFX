"""Budget arithmetic, and the ways it is allowed to refuse."""

from __future__ import annotations

import math

import pytest

from llmtracefx.deploy.budget import (
    MAX_CONTAINERS_CEILING,
    MAX_GPU_COUNT_CEILING,
    MAX_RUNTIME_SECONDS_CEILING,
    BudgetRequest,
    assert_within_budget,
    evaluate_budget,
    recommended_session_budget_usd,
)
from llmtracefx.deploy.errors import DeploymentPlanError
from llmtracefx.deploy.pricing import ComputeQuote, GpuPriceQuote, StorageQuote


def quote(rate: float = 1.0, gpu_type: str = "H200") -> GpuPriceQuote:
    return GpuPriceQuote(
        gpu_type=gpu_type,
        usd_per_gpu_hour=rate,
        effective_date="2026-08-01",
        source="https://modal.com/pricing",
    )


def compute(cpu: float = 0.0, memory: float = 0.0) -> ComputeQuote:
    """Zero-cost by default so a test can isolate one term at a time.

    A rate of exactly zero is refused, so the defaults are the smallest
    positive value that still rounds away in the assertions below.
    """
    return ComputeQuote(
        usd_per_cpu_core_hour=cpu or 1e-9,
        usd_per_gib_memory_hour=memory or 1e-9,
        effective_date="2026-08-01",
        source="https://modal.com/pricing",
    )


def storage_quote(rate: float = 1e-9) -> StorageQuote:
    return StorageQuote(
        usd_per_gib_month=rate,
        effective_date="2026-08-01",
        source="https://modal.com/pricing",
    )


def request(**overrides: object) -> BudgetRequest:
    kwargs: dict[str, object] = {
        "max_usd": 10.0,
        "gpu_type": "H200",
        "gpu_count": 4,
        "max_runtime_seconds": 1800,
        "deployment_seconds": 1800,
        "price": quote(),
        "compute": compute(),
        "storage": storage_quote(),
        "stored_gib": 306.0,
        "storage_retention_days": 0.0,
    }
    kwargs.update(overrides)
    return BudgetRequest(**kwargs)  # type: ignore[arg-type]


def test_accelerators_are_priced_against_the_deployment_window() -> None:
    """One container timeout is added to the window on purpose.

    A container that starts a second before the expiry still runs its
    whole lifetime afterwards, so a bound that stopped at the expiry
    would be beatable by a single well-timed request.
    """
    envelope = evaluate_budget(
        request(max_runtime_seconds=3600, deployment_seconds=7200, gpu_count=4)
    )
    assert envelope.billable_seconds == 7200 + 3600
    gpu = next(line for line in envelope.lines if line.name == "serving-gpu")
    assert gpu.hours == pytest.approx(3.0)
    assert gpu.usd == pytest.approx(12.0)
    assert envelope.gpu_worst_case_usd == pytest.approx(12.0)


def test_every_mandatory_resource_appears_in_the_total() -> None:
    """Each line must contribute, and the total must be their sum.

    Deliberately priced with rates large enough that dropping any one
    line moves the total well outside tolerance. The earlier version of
    this test used the near-zero default rates, so removing the CPU-only
    containers from the sum changed nothing it could measure and the
    assertion passed against a total that was missing them.
    """
    envelope = evaluate_budget(
        request(
            compute=compute(cpu=0.5, memory=0.25),
            storage=storage_quote(0.5),
            max_usd=10_000.0,
        )
    )
    names = [line.name for line in envelope.lines]
    assert names == [
        "serving-gpu",
        "serving-compute",
        "staging",
        "verification",
        "storage",
    ]
    for line in envelope.lines:
        assert line.usd > 1.0, f"{line.name} contributes nothing measurable"
    assert envelope.worst_case_usd == pytest.approx(
        sum(line.usd for line in envelope.lines)
    )
    # Dropping any single line has to be detectable in the total.
    for line in envelope.lines:
        assert envelope.worst_case_usd - line.usd != pytest.approx(
            envelope.worst_case_usd
        )


def test_the_cpu_only_containers_are_a_material_part_of_the_total() -> None:
    """Staging and verification are not rounding error.

    They have no accelerator, which is exactly why an earlier version of
    the cost model left them out entirely.
    """
    envelope = evaluate_budget(
        request(compute=compute(cpu=0.5, memory=0.25), max_usd=10_000.0)
    )
    cpu_only = sum(
        line.usd for line in envelope.lines if line.name in {"staging", "verification"}
    )
    assert cpu_only > 0
    assert envelope.worst_case_usd - cpu_only == pytest.approx(
        sum(
            line.usd
            for line in envelope.lines
            if line.name not in {"staging", "verification"}
        )
    )


def test_cpu_and_memory_bill_on_the_cpu_only_containers_too() -> None:
    """Staging and verification have no accelerator and still cost money."""
    free = evaluate_budget(request())
    priced = evaluate_budget(request(compute=compute(cpu=1.0, memory=0.5)))
    for name in ("staging", "verification"):
        before = next(line for line in free.lines if line.name == name).usd
        after = next(line for line in priced.lines if line.name == name).usd
        assert after > before
    assert priced.worst_case_usd > free.worst_case_usd


def test_storage_is_billed_past_deletion() -> None:
    envelope = evaluate_budget(
        request(
            storage=storage_quote(1.0), stored_gib=300.0, storage_retention_days=6.0
        )
    )
    assert envelope.storage_billed_days == pytest.approx(10.0)
    storage = next(line for line in envelope.lines if line.name == "storage")
    # 300 GiB at $1/GiB-month for ten of a thirty day month.
    assert storage.usd == pytest.approx(100.0)
    assert any("after deletion" in note for note in envelope.notes) or any(
        "after deletion" in line.detail for line in envelope.lines
    )


def test_extra_containers_multiply_the_worst_case() -> None:
    one = evaluate_budget(request(max_containers=1))
    three = evaluate_budget(request(max_containers=3, max_usd=100.0))
    assert three.gpu_worst_case_usd == pytest.approx(one.gpu_worst_case_usd * 3)
    assert any("max_containers is 3" in note for note in three.notes)


def test_over_budget_is_reported_and_then_refused() -> None:
    envelope = evaluate_budget(
        request(max_usd=1.0, max_runtime_seconds=3600, deployment_seconds=3600)
    )
    assert envelope.within_budget is False
    assert envelope.headroom_usd < 0
    with pytest.raises(DeploymentPlanError, match="exceeds the authorised budget"):
        assert_within_budget(envelope)


def test_exactly_on_budget_is_allowed() -> None:
    """The boundary is inclusive, and compared before rounding."""
    envelope = evaluate_budget(request())
    on_the_nose = evaluate_budget(request(max_usd=envelope.worst_case_usd))
    assert on_the_nose.within_budget is True
    assert on_the_nose.headroom_usd == pytest.approx(0.0)
    assert_within_budget(on_the_nose)

    just_under = evaluate_budget(request(max_usd=envelope.worst_case_usd * 0.999999))
    assert just_under.within_budget is False


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_budget_is_refused(bad: float) -> None:
    with pytest.raises(DeploymentPlanError, match="finite"):
        request(max_usd=bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_price_is_refused(bad: float) -> None:
    with pytest.raises(DeploymentPlanError, match="finite"):
        quote(rate=bad)


def test_nan_budget_cannot_slip_through_as_within_budget() -> None:
    """A NaN compares False against everything, which reads as 'fine'.

    The guard has to be at construction, because by the time the
    comparison happens a NaN looks exactly like an over-budget plan that
    somehow passed.
    """
    with pytest.raises(DeploymentPlanError):
        request(max_usd=math.nan)


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_budget_is_refused(bad: float) -> None:
    with pytest.raises(DeploymentPlanError, match="greater than zero"):
        request(max_usd=bad)


@pytest.mark.parametrize("bad", [0, -1, MAX_GPU_COUNT_CEILING + 1])
def test_gpu_count_bounds(bad: int) -> None:
    with pytest.raises(DeploymentPlanError, match="gpu_count"):
        request(gpu_count=bad)


@pytest.mark.parametrize("bad", [0, -1, MAX_RUNTIME_SECONDS_CEILING + 1])
def test_runtime_bounds(bad: int) -> None:
    with pytest.raises(DeploymentPlanError, match="max_runtime_seconds"):
        request(max_runtime_seconds=bad)


def test_booleans_are_not_accepted_as_counts() -> None:
    with pytest.raises(DeploymentPlanError, match="integer"):
        request(gpu_count=True)


def test_price_for_a_different_gpu_is_refused() -> None:
    with pytest.raises(DeploymentPlanError, match="price the GPU you are booking"):
        request(gpu_type="H200", price=quote(gpu_type="H100"))


def test_storage_size_and_retention_are_validated() -> None:
    with pytest.raises(DeploymentPlanError, match="stored_gib"):
        request(stored_gib=0.0)
    with pytest.raises(DeploymentPlanError, match="storage_retention_days"):
        request(storage_retention_days=-1.0)
    with pytest.raises(DeploymentPlanError, match="storage_retention_days"):
        request(storage_retention_days=10_000.0)


def test_a_deployment_window_shorter_than_a_container_is_refused() -> None:
    with pytest.raises(DeploymentPlanError, match="shorter than"):
        request(max_runtime_seconds=3600, deployment_seconds=1800)


def test_envelope_records_every_input_needed_to_recheck_it() -> None:
    payload = evaluate_budget(request()).to_dict()
    for field in (
        "gpu_type",
        "gpu_count",
        "max_containers",
        "max_runtime_seconds",
        "usd_per_gpu_hour",
        "price_effective_date",
        "price_source",
        "worst_case_usd",
        "budget_usd",
    ):
        assert payload[field] is not None, field


def test_recommended_budget_holds_two_thirds_in_reserve() -> None:
    assert recommended_session_budget_usd(30.0) == pytest.approx(10.0)
    assert recommended_session_budget_usd(30.0) < 30.0


def test_recommended_budget_rounds_down_so_it_never_exceeds_its_fraction() -> None:
    assert recommended_session_budget_usd(10.0) == pytest.approx(3.33)


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan")])
def test_recommended_budget_refuses_nonsense_credit(bad: float) -> None:
    with pytest.raises(DeploymentPlanError):
        recommended_session_budget_usd(bad)


def test_container_count_is_capped_so_the_product_cannot_overflow() -> None:
    """A large integer count overflows on int-to-float conversion.

    Without a ceiling the multiplication raises OverflowError before the
    finiteness guard can turn it into a refusal, so the guard's own
    advice to reduce max_containers could never be reached for the input
    it names.
    """
    with pytest.raises(DeploymentPlanError, match="max_containers"):
        request(max_containers=10**308)
    with pytest.raises(DeploymentPlanError, match="max_containers"):
        request(max_containers=MAX_CONTAINERS_CEILING + 1)


def test_the_finiteness_guard_is_reachable_for_an_overflowing_product() -> None:
    """Every accepted input must still be priced without raising."""
    envelope = evaluate_budget(
        request(
            max_containers=MAX_CONTAINERS_CEILING,
            max_runtime_seconds=MAX_RUNTIME_SECONDS_CEILING,
            deployment_seconds=MAX_RUNTIME_SECONDS_CEILING,
            max_usd=1e12,
        )
    )
    assert math.isfinite(envelope.worst_case_usd)


def test_a_credit_too_large_to_scale_is_refused_not_raised() -> None:
    """The input being finite does not make the scaled value finite.

    `math.floor` raises on an infinity rather than returning one, so the
    guard has to sit on the product, exactly as it does for the
    worst-case cost.
    """
    with pytest.raises(DeploymentPlanError, match="too large"):
        recommended_session_budget_usd(1e307)
