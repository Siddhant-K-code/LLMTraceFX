"""Operator-supplied price quotes, with an effective date attached.

Cloud GPU prices are mutable. A price literal compiled into this package
would be a claim about the present that quietly decays into a false claim
about the present, and the number it produces (a cost estimate the
operator uses to decide whether to spend money) is exactly the kind of
number that must not silently go stale.

So there is deliberately no default price anywhere in this module. A
quote has to be supplied by the caller together with the date it was
read and where it was read from, and a quote older than the caller's
tolerance is refused rather than used. The failure mode is "you must go
and look up the current price", which costs a minute, instead of
"budgeting silently used last quarter's price", which costs money.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from .errors import DeploymentPlanError

_ISO_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# A ceiling well above any published accelerator rate. It exists to catch
# a misplaced decimal point or a per-month figure pasted into a per-hour
# field, not to express an opinion about what a GPU should cost.
MAX_PLAUSIBLE_USD_PER_GPU_HOUR = 1000.0

# The same idea for storage, per GiB-month.
MAX_PLAUSIBLE_USD_PER_GIB_MONTH = 100.0

# Per-core-hour and per-GiB-hour rates are small numbers; a three digit
# one is a units mistake.
MAX_PLAUSIBLE_USD_PER_CORE_HOUR = 100.0
MAX_PLAUSIBLE_USD_PER_GIB_HOUR = 100.0

# How old a quote may be before planning refuses it. Ninety days is long
# enough that an operator is not re-reading a pricing page for every
# invocation, and short enough that a quote cannot survive unnoticed
# across a pricing change.
DEFAULT_MAX_PRICE_AGE_DAYS = 90


def _require_finite(value: float, *, field: str) -> float:
    """Reject NaN and both infinities before they reach any arithmetic.

    A NaN price propagates through every multiplication and comparison in
    this package and comes out the other side as a cost that compares
    False against every budget, which reads as "within budget" to any
    naive check. Refusing it at the boundary is the only place where the
    value is still recognisably wrong.
    """
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DeploymentPlanError(f"{field} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise DeploymentPlanError(f"{field} must be a finite number, got {value!r}")
    return numeric


def _require_iso_date(value: str, *, field: str) -> date:
    if not isinstance(value, str) or not _ISO_DATE_PATTERN.match(value):
        raise DeploymentPlanError(f"{field} must be an ISO date (YYYY-MM-DD)")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise DeploymentPlanError(f"{field} is not a valid calendar date") from exc


def _require_text(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DeploymentPlanError(f"{field} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class GpuPriceQuote:
    """One accelerator price, as read by a human on a stated date.

    ``source`` is required and is free text (normally the URL of the
    pricing page). It is what lets a reviewer re-check the number later
    rather than having to trust it.
    """

    gpu_type: str
    usd_per_gpu_hour: float
    effective_date: str
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "gpu_type", _require_text(self.gpu_type, field="gpu_type")
        )
        object.__setattr__(
            self, "source", _require_text(self.source, field="price source")
        )
        rate = _require_finite(self.usd_per_gpu_hour, field="usd_per_gpu_hour")
        if rate <= 0:
            raise DeploymentPlanError("usd_per_gpu_hour must be greater than zero")
        if rate > MAX_PLAUSIBLE_USD_PER_GPU_HOUR:
            raise DeploymentPlanError(
                "usd_per_gpu_hour exceeds "
                f"{MAX_PLAUSIBLE_USD_PER_GPU_HOUR:.0f}; check the units, this "
                "field is dollars per GPU per hour"
            )
        object.__setattr__(self, "usd_per_gpu_hour", rate)
        _require_iso_date(self.effective_date, field="price effective_date")

    def age_days(self, *, as_of: date) -> int:
        """Whole days between the quote's effective date and ``as_of``.

        Negative when the quote is dated in the future, which is reported
        rather than clamped: a future-dated quote is a data-entry error
        worth surfacing, not something to round away to zero.
        """
        return (as_of - date.fromisoformat(self.effective_date)).days

    def is_stale(
        self, *, as_of: date, max_age_days: int = DEFAULT_MAX_PRICE_AGE_DAYS
    ) -> bool:
        return self.age_days(as_of=as_of) > max_age_days

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class StorageQuote:
    """A persistent-volume storage price, per GiB-month.

    Kept separate from :class:`GpuPriceQuote` because it bills on a
    different clock. GPU time stops when the container stops; stored
    weights keep costing money until the volume is deleted, which is the
    charge an operator is most likely to forget.
    """

    usd_per_gib_month: float
    effective_date: str
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source", _require_text(self.source, field="storage source")
        )
        rate = _require_finite(self.usd_per_gib_month, field="usd_per_gib_month")
        if rate <= 0:
            raise DeploymentPlanError("usd_per_gib_month must be greater than zero")
        if rate > MAX_PLAUSIBLE_USD_PER_GIB_MONTH:
            raise DeploymentPlanError(
                "usd_per_gib_month exceeds "
                f"{MAX_PLAUSIBLE_USD_PER_GIB_MONTH:.0f}; check the units, this "
                "field is dollars per GiB per month"
            )
        object.__setattr__(self, "usd_per_gib_month", rate)
        _require_iso_date(self.effective_date, field="storage effective_date")

    def age_days(self, *, as_of: date) -> int:
        return (as_of - date.fromisoformat(self.effective_date)).days

    def is_stale(
        self, *, as_of: date, max_age_days: int = DEFAULT_MAX_PRICE_AGE_DAYS
    ) -> bool:
        return self.age_days(as_of=as_of) > max_age_days

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ComputeQuote:
    """The CPU and memory rates that bill alongside every container.

    Separate from :class:`GpuPriceQuote` because they apply to the
    staging and verification containers too, which have no accelerator
    at all and were previously priced at zero. A harness that calls
    itself a budget guard cannot leave a mandatory charge out of the
    total merely because it is the smaller one.
    """

    usd_per_cpu_core_hour: float
    usd_per_gib_memory_hour: float
    effective_date: str
    source: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "source", _require_text(self.source, field="compute price source")
        )
        cpu = _require_finite(self.usd_per_cpu_core_hour, field="usd_per_cpu_core_hour")
        memory = _require_finite(
            self.usd_per_gib_memory_hour, field="usd_per_gib_memory_hour"
        )
        if cpu <= 0 or memory <= 0:
            raise DeploymentPlanError(
                "CPU and memory rates must both be greater than zero"
            )
        if cpu > MAX_PLAUSIBLE_USD_PER_CORE_HOUR:
            raise DeploymentPlanError(
                "usd_per_cpu_core_hour exceeds "
                f"{MAX_PLAUSIBLE_USD_PER_CORE_HOUR:.0f}; check the units"
            )
        if memory > MAX_PLAUSIBLE_USD_PER_GIB_HOUR:
            raise DeploymentPlanError(
                "usd_per_gib_memory_hour exceeds "
                f"{MAX_PLAUSIBLE_USD_PER_GIB_HOUR:.0f}; check the units"
            )
        object.__setattr__(self, "usd_per_cpu_core_hour", cpu)
        object.__setattr__(self, "usd_per_gib_memory_hour", memory)
        _require_iso_date(self.effective_date, field="compute effective_date")

    def container_usd_per_hour(self, *, cpu_cores: float, memory_gib: float) -> float:
        return (
            cpu_cores * self.usd_per_cpu_core_hour
            + memory_gib * self.usd_per_gib_memory_hour
        )

    def age_days(self, *, as_of: date) -> int:
        return (as_of - date.fromisoformat(self.effective_date)).days

    def is_stale(
        self, *, as_of: date, max_age_days: int = DEFAULT_MAX_PRICE_AGE_DAYS
    ) -> bool:
        return self.age_days(as_of=as_of) > max_age_days

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
