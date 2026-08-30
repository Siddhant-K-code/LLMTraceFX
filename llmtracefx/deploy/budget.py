"""Worst-case cost arithmetic and the fail-closed budget gate.

Every number here is an upper bound, never an expectation. A plan is
approved on the basis of what the deployment costs if it runs the whole
way to its own timeout on every container it is allowed to start, because
that is the only figure an operator can be held to: an average-case
estimate approves a deployment that can still overspend, and the moment
it does the money is already gone.

Nothing in this module has a default that spends money. There is no
default budget, no default GPU count, no default runtime and no default
price. Each has to be stated by the caller, so an incomplete invocation
fails closed instead of falling back to something plausible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .errors import DeploymentPlanError
from .pricing import ComputeQuote, GpuPriceQuote, StorageQuote, _require_finite
from .resources import (
    POST_DELETE_BILLING_DAYS,
    SERVING_CPU_CORES,
    SERVING_MEMORY_GIB,
    STAGING_CPU_CORES,
    STAGING_MEMORY_GIB,
    STAGING_TIMEOUT_SECONDS,
    VERIFY_CPU_CORES,
    VERIFY_MEMORY_GIB,
    VERIFY_TIMEOUT_SECONDS,
)

SECONDS_PER_HOUR = 3600.0
DAYS_PER_MONTH = 30.0

# This harness refuses to plan a run longer than a day regardless of what
# the platform would accept. A self-host validation that needs more than
# 24 hours of continuous GPU time is not the experiment this harness is
# for, and the failure mode of an over-long timeout (a container nobody
# is watching, billing until it hits the limit) is exactly what the
# budget gate exists to prevent.
MAX_RUNTIME_SECONDS_CEILING = 86_400

# Modal documents up to 8 GPUs per container for H200 ("H200:8").
# See https://modal.com/docs/guide/gpu (read 2026-08-30).
MAX_GPU_COUNT_CEILING = 8

# A ceiling on the autoscaling limit itself. This harness exists for a
# short single-endpoint validation, so any three-digit container count is
# a typo rather than an intention. It also keeps the worst-case product
# inside the float range: without it a large integer count overflows
# during the int-to-float conversion and raises OverflowError before the
# finiteness guard below can turn it into a refusal.
MAX_CONTAINERS_CEILING = 64

# The wall-clock window during which the deployment may serve at all,
# enforced by an expiry the serving container checks before it starts.
# Seven days is the ceiling because this harness is for a short
# validation and the whole point of the expiry is that it arrives.
MAX_DEPLOYMENT_SECONDS_CEILING = 7 * 24 * 60 * 60

# How long weights may be declared to stay on the volume. A year of
# storage for a checkpoint this size is not a validation.
MAX_RETENTION_DAYS_CEILING = 90.0

# Fraction of remaining credit a single session is allowed to put at
# risk, expressed as a divisor. One third leaves two thirds behind for
# the things that are not in the GPU worst case: a startup that fails
# after pulling the image and allocating GPUs, one retry of the whole
# run, and volume storage, which keeps accruing after the GPU stops and
# which Modal bills for up to four days after the data is deleted
# (https://modal.com/docs/guide/volumes, read 2026-08-30).
CREDIT_RESERVE_DIVISOR = 3.0


def recommended_session_budget_usd(available_credit_usd: float) -> float:
    """The largest budget this harness will suggest for one session.

    Deliberately not "all of it". A harness that proposes spending the
    entire balance leaves nothing for the second attempt, and the second
    attempt is the likely one: the first run of an unfamiliar serving
    stack is the run that discovers the flag it needed.

    Rounded down to whole cents so the suggestion never exceeds the
    fraction it claims to be.
    """
    credit = _require_finite(available_credit_usd, field="available_credit_usd")
    if credit <= 0:
        raise DeploymentPlanError("available_credit_usd must be greater than zero")
    scaled = (credit / CREDIT_RESERVE_DIVISOR) * 100
    # The input being finite does not make the scaled value finite, and
    # math.floor raises on an infinity rather than returning one. Same
    # hazard the worst-case product guards against in evaluate_budget.
    if not math.isfinite(scaled):
        raise DeploymentPlanError(
            "available_credit_usd is too large to compute a recommendation from"
        )
    return math.floor(scaled) / 100


def _require_positive_int(value: int, *, field: str, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DeploymentPlanError(f"{field} must be an integer")
    if value < 1:
        raise DeploymentPlanError(f"{field} must be at least 1, got {value}")
    if maximum is not None and value > maximum:
        raise DeploymentPlanError(f"{field} must not exceed {maximum}, got {value}")
    return value


@dataclass(frozen=True)
class BudgetRequest:
    """A fully specified, explicitly authorised spending envelope.

    Every mandatory charge is represented, not just the largest one. The
    accelerators dominate, but the staging container, the verification
    container, the CPU and memory that bill alongside the accelerators,
    and the volume that keeps billing after the run all cost money, and a
    total that omits them is not a budget guard.

    ``deployment_seconds`` is the wall-clock window the deployment may
    serve at all, enforced by an expiry the serving container checks
    before it starts. It is the term the accelerators are priced against,
    because the per-container timeout does not bound spending: a served
    request that arrives after a container exits simply starts another
    one.
    """

    max_usd: float
    gpu_type: str
    gpu_count: int
    max_runtime_seconds: int
    deployment_seconds: int
    price: GpuPriceQuote
    compute: ComputeQuote
    storage: StorageQuote
    stored_gib: float
    storage_retention_days: float
    max_containers: int = 1

    def __post_init__(self) -> None:
        budget = _require_finite(self.max_usd, field="max_usd")
        if budget <= 0:
            raise DeploymentPlanError("max_usd must be greater than zero")
        object.__setattr__(self, "max_usd", budget)

        if not isinstance(self.gpu_type, str) or not self.gpu_type.strip():
            raise DeploymentPlanError("gpu_type must be a non-empty string")
        object.__setattr__(self, "gpu_type", self.gpu_type.strip())

        object.__setattr__(
            self,
            "gpu_count",
            _require_positive_int(
                self.gpu_count, field="gpu_count", maximum=MAX_GPU_COUNT_CEILING
            ),
        )
        object.__setattr__(
            self,
            "max_runtime_seconds",
            _require_positive_int(
                self.max_runtime_seconds,
                field="max_runtime_seconds",
                maximum=MAX_RUNTIME_SECONDS_CEILING,
            ),
        )
        object.__setattr__(
            self,
            "deployment_seconds",
            _require_positive_int(
                self.deployment_seconds,
                field="deployment_seconds",
                maximum=MAX_DEPLOYMENT_SECONDS_CEILING,
            ),
        )
        if self.deployment_seconds < self.max_runtime_seconds:
            raise DeploymentPlanError(
                f"deployment_seconds {self.deployment_seconds} is shorter than "
                f"max_runtime_seconds {self.max_runtime_seconds}; the window "
                "the deployment may serve for cannot be shorter than one "
                "container's own lifetime"
            )
        object.__setattr__(
            self,
            "max_containers",
            _require_positive_int(
                self.max_containers,
                field="max_containers",
                maximum=MAX_CONTAINERS_CEILING,
            ),
        )

        # A quote for a different accelerator is the quietest way to
        # underprice a run: the arithmetic all succeeds and the answer is
        # simply wrong. The two names have to agree.
        if self.price.gpu_type.casefold() != self.gpu_type.casefold():
            raise DeploymentPlanError(
                f"price quote is for {self.price.gpu_type!r} but the plan "
                f"requests {self.gpu_type!r}; price the GPU you are booking"
            )

        size = _require_finite(self.stored_gib, field="stored_gib")
        if size <= 0:
            raise DeploymentPlanError("stored_gib must be greater than zero")
        object.__setattr__(self, "stored_gib", size)

        retention = _require_finite(
            self.storage_retention_days, field="storage_retention_days"
        )
        if retention < 0:
            raise DeploymentPlanError("storage_retention_days must not be negative")
        if retention > MAX_RETENTION_DAYS_CEILING:
            raise DeploymentPlanError(
                "storage_retention_days must not exceed "
                f"{MAX_RETENTION_DAYS_CEILING:.0f}"
            )
        object.__setattr__(self, "storage_retention_days", retention)


@dataclass(frozen=True)
class CostLine:
    """One billable resource, priced, with its inputs carried through."""

    name: str
    detail: str
    hours: float
    usd_per_hour: float
    usd: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "detail": self.detail,
            "hours": round(self.hours, 6),
            "usd_per_hour": round(self.usd_per_hour, 6),
            "usd": round(self.usd, 6),
        }


@dataclass(frozen=True)
class CostEnvelope:
    """The upper bound on what a plan can cost, and how it was reached.

    Every input is carried through to the output so the number can be
    re-derived by hand from the record alone, without re-running the
    planner or trusting it.
    """

    gpu_type: str
    gpu_count: int
    max_containers: int
    max_runtime_seconds: int
    deployment_seconds: int
    billable_seconds: int
    usd_per_gpu_hour: float
    price_effective_date: str
    price_source: str
    compute_effective_date: str
    compute_source: str
    storage_effective_date: str
    storage_source: str
    stored_gib: float
    storage_usd_per_gib_month: float
    storage_retention_days: float
    storage_billed_days: float
    lines: tuple[CostLine, ...]
    worst_case_usd: float
    budget_usd: float
    headroom_usd: float
    within_budget: bool
    bounded: bool = True
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def gpu_worst_case_usd(self) -> float:
        return next(line.usd for line in self.lines if line.name == "serving-gpu")

    def to_dict(self) -> dict[str, Any]:
        def money(value: float) -> float:
            return round(value, 6)

        return {
            "gpu_type": self.gpu_type,
            "gpu_count": self.gpu_count,
            "max_containers": self.max_containers,
            "max_runtime_seconds": self.max_runtime_seconds,
            "deployment_seconds": self.deployment_seconds,
            "billable_seconds": self.billable_seconds,
            "usd_per_gpu_hour": money(self.usd_per_gpu_hour),
            "price_effective_date": self.price_effective_date,
            "price_source": self.price_source,
            "compute_effective_date": self.compute_effective_date,
            "compute_source": self.compute_source,
            "storage_effective_date": self.storage_effective_date,
            "storage_source": self.storage_source,
            "stored_gib": round(self.stored_gib, 3),
            "storage_usd_per_gib_month": money(self.storage_usd_per_gib_month),
            "storage_retention_days": self.storage_retention_days,
            "storage_billed_days": self.storage_billed_days,
            "lines": [line.to_dict() for line in self.lines],
            "worst_case_usd": money(self.worst_case_usd),
            "budget_usd": money(self.budget_usd),
            "headroom_usd": money(self.headroom_usd),
            "within_budget": self.within_budget,
            "bounded": self.bounded,
            "covers": (
                "Accelerators, the CPU and memory billed alongside them, the "
                "staging and verification containers, and volume storage "
                "including the period Modal bills after deletion."
            ),
            "does_not_cover": (
                "Cold starts triggered by requests that reach the platform. A "
                "web server function is scheduled before any code in the "
                "container runs, so a request allocates accelerators first "
                "and is refused second, including after the deployment "
                "expiry. With proxy auth on, only a holder of a workspace "
                "token can do this and the refusal takes seconds. With a "
                "public endpoint anyone can, without limit, and no figure "
                "here bounds it."
            ),
            "notes": list(self.notes),
        }


def evaluate_budget(request: BudgetRequest) -> CostEnvelope:
    """Price the worst case for ``request``. Pure arithmetic, no I/O.

    The serving term assumes every allowed container is occupied for the
    whole deployment window, plus one container timeout, because a
    container that starts just before the expiry still runs its full
    lifetime afterwards.

    This is the cost of the intended path, not an unconditional cap. The
    platform schedules a web server container before any of this
    project's code runs, so a request can allocate accelerators and be
    refused afterwards; what stops that being unbounded is authentication
    at the edge, which is a plan-level gate rather than a term in this
    arithmetic. ``bounded`` records whether that gate is in place.
    """
    billable_seconds = request.deployment_seconds + request.max_runtime_seconds
    serving_hours = request.max_containers * (billable_seconds / SECONDS_PER_HOUR)

    gpu_rate = request.gpu_count * request.price.usd_per_gpu_hour
    serving_compute_rate = request.compute.container_usd_per_hour(
        cpu_cores=SERVING_CPU_CORES, memory_gib=SERVING_MEMORY_GIB
    )
    staging_hours = STAGING_TIMEOUT_SECONDS / SECONDS_PER_HOUR
    staging_rate = request.compute.container_usd_per_hour(
        cpu_cores=STAGING_CPU_CORES, memory_gib=STAGING_MEMORY_GIB
    )
    verify_hours = VERIFY_TIMEOUT_SECONDS / SECONDS_PER_HOUR
    verify_rate = request.compute.container_usd_per_hour(
        cpu_cores=VERIFY_CPU_CORES, memory_gib=VERIFY_MEMORY_GIB
    )

    storage_billed_days = request.storage_retention_days + POST_DELETE_BILLING_DAYS
    storage_hours = storage_billed_days * 24.0
    storage_rate = (
        request.stored_gib * request.storage.usd_per_gib_month / (DAYS_PER_MONTH * 24.0)
    )

    lines = (
        CostLine(
            name="serving-gpu",
            detail=(
                f"{request.gpu_count} x {request.gpu_type} x "
                f"{request.max_containers} container(s)"
            ),
            hours=serving_hours,
            usd_per_hour=gpu_rate,
            usd=serving_hours * gpu_rate,
        ),
        CostLine(
            name="serving-compute",
            detail=(
                f"{SERVING_CPU_CORES:.0f} cores + {SERVING_MEMORY_GIB:.0f} GiB "
                f"x {request.max_containers} container(s)"
            ),
            hours=serving_hours,
            usd_per_hour=serving_compute_rate,
            usd=serving_hours * serving_compute_rate,
        ),
        CostLine(
            name="staging",
            detail=(
                f"{STAGING_CPU_CORES:.0f} cores + {STAGING_MEMORY_GIB:.0f} GiB, "
                "CPU only, to its own timeout"
            ),
            hours=staging_hours,
            usd_per_hour=staging_rate,
            usd=staging_hours * staging_rate,
        ),
        CostLine(
            name="verification",
            detail=(
                f"{VERIFY_CPU_CORES:.0f} cores + {VERIFY_MEMORY_GIB:.0f} GiB, "
                "CPU only, to its own timeout"
            ),
            hours=verify_hours,
            usd_per_hour=verify_rate,
            usd=verify_hours * verify_rate,
        ),
        CostLine(
            name="storage",
            detail=(
                f"{request.stored_gib:.0f} GiB for "
                f"{request.storage_retention_days:.0f} day(s) plus "
                f"{POST_DELETE_BILLING_DAYS:.0f} billed after deletion"
            ),
            hours=storage_hours,
            usd_per_hour=storage_rate,
            usd=storage_hours * storage_rate,
        ),
    )

    worst_case = sum(line.usd for line in lines)

    # Guarding the total as well as each input: the inputs are already
    # finite, but a product of two large finite floats can still overflow
    # to infinity, and an infinite worst case must not be allowed to
    # reach the comparison below as a number.
    if not math.isfinite(worst_case):
        raise DeploymentPlanError(
            "worst-case cost overflowed to a non-finite value; reduce "
            "gpu_count, max_containers or the deployment window"
        )

    # Joined explicitly rather than written as adjacent string literals
    # inside the list, which reads as a missing comma between two notes.
    window_note = " ".join(
        (
            "The accelerator term is priced against the deployment window,",
            "not the container timeout, because a request arriving after a",
            "container exits simply starts another one. One container",
            "timeout is added so a container starting just before the",
            "expiry is still covered.",
        )
    )
    storage_note = " ".join(
        (
            "Volume storage keeps billing after the GPUs stop, and Modal",
            "bills deleted data for up to four days, so both are included.",
        )
    )
    notes: list[str] = [window_note, storage_note]
    if request.max_containers > 1:
        notes.append(
            f"max_containers is {request.max_containers}, so the worst case "
            "is that many concurrent containers for the whole window."
        )

    return CostEnvelope(
        gpu_type=request.gpu_type,
        gpu_count=request.gpu_count,
        max_containers=request.max_containers,
        max_runtime_seconds=request.max_runtime_seconds,
        deployment_seconds=request.deployment_seconds,
        billable_seconds=billable_seconds,
        usd_per_gpu_hour=request.price.usd_per_gpu_hour,
        price_effective_date=request.price.effective_date,
        price_source=request.price.source,
        compute_effective_date=request.compute.effective_date,
        compute_source=request.compute.source,
        storage_effective_date=request.storage.effective_date,
        storage_source=request.storage.source,
        stored_gib=request.stored_gib,
        storage_usd_per_gib_month=request.storage.usd_per_gib_month,
        storage_retention_days=request.storage_retention_days,
        storage_billed_days=storage_billed_days,
        lines=lines,
        worst_case_usd=worst_case,
        budget_usd=request.max_usd,
        headroom_usd=request.max_usd - worst_case,
        within_budget=worst_case <= request.max_usd,
        notes=tuple(notes),
    )


def assert_within_budget(envelope: CostEnvelope) -> None:
    """Refuse an envelope whose worst case exceeds the stated budget."""
    if envelope.within_budget:
        return
    raise DeploymentPlanError(
        f"worst-case cost ${envelope.worst_case_usd:.2f} exceeds the "
        f"authorised budget ${envelope.budget_usd:.2f} "
        f"({envelope.gpu_count} x {envelope.gpu_type} x "
        f"{envelope.max_containers} container(s) for "
        f"{envelope.max_runtime_seconds}s at "
        f"${envelope.usd_per_gpu_hour:.4f}/GPU-hour, priced "
        f"{envelope.price_effective_date}). Lower --max-runtime-seconds or "
        "--gpu-count, or raise --max-usd if you actually intend to spend it."
    )
