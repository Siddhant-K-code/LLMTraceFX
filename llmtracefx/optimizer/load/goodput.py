"""Goodput and SLO attainment over a set of request rows.

A request meets the SLO only when it meets every stated target. Each
request gets exactly one verdict, decided in this order:

1. ``failed``: the collector reported a failure (HTTP status, timeout,
   connection, stream decode, provider error payload, missing content,
   truncated stream) or the client raised. Incomplete streams are failures.
2. ``missed``: it succeeded and at least one stated target was exceeded. A
   missed target is decisive even when another target has no value.
3. ``unverifiable``: it succeeded, nothing was missed, but a stated target
   has no value (for example TPOT with no provider token count).
4. ``met``: every stated target is satisfied. A response of one output
   token or fewer has no inter-token gap, so TPOT is not applicable and
   satisfies the TPOT target; these are counted separately.

The denominator is the number of dispatched requests. Failed requests stay
in it. Requests that were never sent are not in it, but they make the set
incomplete and the headline values are withheld. Nothing is estimated: a
value that cannot be computed from the evidence is ``None`` with a reason.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..schema import MetricProvenance
from .api_worker import TPOT_NOT_APPLICABLE, RequestRow

VERDICT_MET = "met"
VERDICT_MISSED = "missed"
VERDICT_UNVERIFIABLE = "unverifiable"
VERDICT_FAILED = "failed"
VERDICTS = (VERDICT_MET, VERDICT_MISSED, VERDICT_UNVERIFIABLE, VERDICT_FAILED)

WITHHELD_INCOMPLETE = "incomplete: some requests were never sent"
WITHHELD_UNVERIFIABLE = "some requests could not be checked against every target"
WITHHELD_NO_WINDOW = "no measured window: no request was dispatched and completed"
WITHHELD_TOKENS_MISSING = "the provider did not report output tokens for every request"
WITHHELD_NO_TOKENS = "the reported output token total is zero"
WITHHELD_NONE_DISPATCHED = "no request was dispatched"

_DERIVED = MetricProvenance.DERIVED.value


class SloConfigError(ValueError):
    """The SLO targets are invalid."""


@dataclass(frozen=True)
class SloTargets:
    """Upper bounds in milliseconds. A request must satisfy each stated one (``<=``)."""

    ttft_ms: float | None = None
    tpot_ms: float | None = None
    e2e_ms: float | None = None

    def __post_init__(self) -> None:
        stated = self.stated()
        if not stated:
            raise SloConfigError("state at least one of ttft_ms, tpot_ms, e2e_ms")
        for name, value in stated.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise SloConfigError(f"{name} must be a finite number above zero")

    def stated(self) -> dict[str, float]:
        values = {
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "e2e_ms": self.e2e_ms,
        }
        return {name: value for name, value in values.items() if value is not None}

    def to_dict(self) -> dict[str, Any]:
        return {**self.stated(), "comparison": "<="}


def request_verdict(row: RequestRow, slo: SloTargets) -> str | None:
    """The verdict for one row, or ``None`` when the request was never sent."""
    if not row.sent:
        return None
    if row.client_error is not None or row.success is not True:
        return VERDICT_FAILED
    missing = False
    for name, target in slo.stated().items():
        if name == "tpot_ms" and row.tpot_status == TPOT_NOT_APPLICABLE:
            continue
        value = getattr(row, name)
        if value is None:
            missing = True
        elif value > target:
            return VERDICT_MISSED
    return VERDICT_UNVERIFIABLE if missing else VERDICT_MET


def missed_targets(row: RequestRow, slo: SloTargets) -> list[str]:
    """The stated targets this row exceeded, for a succeeded row."""
    if not row.sent or row.client_error is not None or row.success is not True:
        return []
    return [
        name
        for name, target in slo.stated().items()
        if getattr(row, name) is not None and getattr(row, name) > target
    ]


def measured_window_seconds(rows: Sequence[RequestRow]) -> float | None:
    """First dispatch to last completion among sent requests, on the dispatch clock."""
    starts = [row.dispatched_offset_s for row in rows if row.sent]
    ends = [row.completed_offset_s for row in rows if row.sent]
    if not starts or any(value is None for value in starts + ends):
        return None
    window = max(e for e in ends if e is not None) - min(
        s for s in starts if s is not None
    )
    return window if window > 0 else None


def _value(value: float | None, reason: str | None = None) -> dict[str, Any]:
    return {
        "value": value,
        "provenance": _DERIVED,
        "withheld_reason": None if value is not None else reason,
    }


def _token_sum(rows: Sequence[RequestRow]) -> int | None:
    counts = [row.output_tokens for row in rows]
    if any(count is None for count in counts):
        return None
    return sum(count for count in counts if count is not None)


@dataclass(frozen=True)
class GoodputSummary:
    """Goodput for one set of rows against one SLO. All values are ``derived``."""

    slo: SloTargets
    dispatched: int
    unsent: int
    counts: dict[str, int]
    missed_by_target: dict[str, int]
    tpot_not_applicable_met: int
    window_seconds: float | None
    request_goodput_fraction: dict[str, Any]
    request_goodput_fraction_bounds: dict[str, float] | None
    request_goodput_per_second: dict[str, Any]
    token_goodput_fraction: dict[str, Any]
    token_goodput_per_second: dict[str, Any]

    @property
    def incomplete(self) -> bool:
        return self.unsent > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "slo": self.slo.to_dict(),
            "denominator": "dispatched_requests",
            "dispatched": self.dispatched,
            "unsent": self.unsent,
            "incomplete": self.incomplete,
            "counts": dict(self.counts),
            "missed_by_target": dict(self.missed_by_target),
            "tpot_not_applicable_met": self.tpot_not_applicable_met,
            "window_seconds": self.window_seconds,
            "request_goodput_fraction": self.request_goodput_fraction,
            "request_goodput_fraction_bounds": self.request_goodput_fraction_bounds,
            "request_goodput_per_second": self.request_goodput_per_second,
            "token_goodput_fraction": self.token_goodput_fraction,
            "token_goodput_per_second": self.token_goodput_per_second,
        }


def summarize_goodput(rows: Sequence[RequestRow], slo: SloTargets) -> GoodputSummary:
    """Count verdicts and derive goodput, withholding what the evidence cannot support."""
    sent = [row for row in rows if row.sent]
    unsent = len(rows) - len(sent)
    verdicts = [(row, request_verdict(row, slo)) for row in sent]
    counts = dict.fromkeys(VERDICTS, 0)
    for _, verdict in verdicts:
        if verdict is not None:
            counts[verdict] += 1
    missed_by_target = dict.fromkeys(slo.stated(), 0)
    for row in sent:
        for name in missed_targets(row, slo):
            missed_by_target[name] += 1
    met_rows = [row for row, verdict in verdicts if verdict == VERDICT_MET]
    tpot_na = sum(
        1
        for row in met_rows
        if "tpot_ms" in slo.stated() and row.tpot_status == TPOT_NOT_APPLICABLE
    )
    dispatched = len(sent)
    window = measured_window_seconds(rows)
    met = counts[VERDICT_MET]
    unverifiable = counts[VERDICT_UNVERIFIABLE]

    blocker: str | None = None
    if dispatched == 0:
        blocker = WITHHELD_NONE_DISPATCHED
    elif unsent:
        blocker = WITHHELD_INCOMPLETE
    elif unverifiable:
        blocker = WITHHELD_UNVERIFIABLE

    bounds: dict[str, float] | None = None
    if dispatched and not unsent and unverifiable:
        bounds = {
            "lower": met / dispatched,
            "upper": (met + unverifiable) / dispatched,
        }

    fraction = _value(None if blocker else met / dispatched, blocker)
    per_second_blocker = blocker or (None if window else WITHHELD_NO_WINDOW)
    per_second = _value(
        None if per_second_blocker or window is None else met / window,
        per_second_blocker,
    )

    met_tokens = _token_sum(met_rows)
    all_tokens = _token_sum(sent)
    if blocker:
        token_fraction = _value(None, blocker)
    elif all_tokens is None:
        token_fraction = _value(None, WITHHELD_TOKENS_MISSING)
    elif all_tokens == 0 or met_tokens is None:
        token_fraction = _value(None, WITHHELD_NO_TOKENS)
    else:
        token_fraction = _value(met_tokens / all_tokens)
    if per_second_blocker:
        token_per_second = _value(None, per_second_blocker)
    elif met_tokens is None or window is None:
        token_per_second = _value(None, WITHHELD_TOKENS_MISSING)
    else:
        token_per_second = _value(met_tokens / window)

    return GoodputSummary(
        slo=slo,
        dispatched=dispatched,
        unsent=unsent,
        counts=counts,
        missed_by_target=missed_by_target,
        tpot_not_applicable_met=tpot_na,
        window_seconds=window,
        request_goodput_fraction=fraction,
        request_goodput_fraction_bounds=bounds,
        request_goodput_per_second=per_second,
        token_goodput_fraction=token_fraction,
        token_goodput_per_second=token_per_second,
    )
