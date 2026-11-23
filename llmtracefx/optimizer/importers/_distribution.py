"""Summarise per-request values into a ``derived`` imported metric."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from .artifact import (
    DERIVED,
    NEAREST_RANK,
    ImportedInterval,
    ImportedMetric,
    ImportedValue,
)

PERCENTILES = (50, 90, 95, 99)


def _interval(found: Mapping[str, Any]) -> ImportedInterval:
    if found["withheld_reason"] is not None:
        return ImportedInterval(None, None, withheld_reason=found["withheld_reason"])
    return ImportedInterval(
        lower=found["lower"],
        upper=found["upper"],
        lower_rank=found["lower_rank"],
        upper_rank=found["upper_rank"],
        coverage=found["coverage"],
    )


def nearest_rank_metric(
    values: Sequence[float],
    *,
    source_key: str,
    unit: str,
    direction: str,
    sample_provenance: str,
    mean_stat: str,
    notes: tuple[str, ...],
) -> ImportedMetric:
    """Mean, min, nearest-rank percentiles with intervals, and max of ``values``."""
    # Imported here so reading artifacts (the gate) does not load the sweep code.
    from ..load.sweep import CONFIDENCE_LEVEL, INTERVAL_METHOD, percentile_interval

    ordered = sorted(values)
    stats = {
        mean_stat: ImportedValue(math.fsum(ordered) / len(ordered), source_key),
        "min": ImportedValue(ordered[0], source_key),
        **{
            f"p{p}": ImportedValue(
                ordered[max(1, math.ceil(p / 100.0 * len(ordered))) - 1], source_key
            )
            for p in PERCENTILES
        },
        "max": ImportedValue(ordered[-1], source_key),
    }
    return ImportedMetric(
        unit=unit,
        direction=direction,
        stats=stats,
        provenance=DERIVED,
        notes=notes,
        sample_count=len(ordered),
        sample_provenance=sample_provenance,
        statistic=NEAREST_RANK,
        intervals={
            f"p{p}": _interval(percentile_interval(ordered, p)) for p in PERCENTILES
        },
        interval_level=CONFIDENCE_LEVEL,
        interval_method=INTERVAL_METHOD,
    )
