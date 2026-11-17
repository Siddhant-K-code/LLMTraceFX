"""Goodput and SLO attainment: verdicts, denominators and withheld values."""

from __future__ import annotations

import pytest

from llmtracefx.optimizer.load.api_worker import (
    TPOT_DERIVED,
    TPOT_NOT_APPLICABLE,
    TPOT_UNAVAILABLE,
    RequestRow,
)
from llmtracefx.optimizer.load.goodput import (
    VERDICT_FAILED,
    VERDICT_MET,
    VERDICT_MISSED,
    VERDICT_UNVERIFIABLE,
    WITHHELD_INCOMPLETE,
    WITHHELD_TOKENS_MISSING,
    WITHHELD_UNVERIFIABLE,
    SloConfigError,
    SloTargets,
    measured_window_seconds,
    request_verdict,
    summarize_goodput,
)


def _row(
    index: int = 0,
    *,
    ttft: float | None = 100.0,
    tpot: float | None = 10.0,
    tpot_status: str = TPOT_DERIVED,
    e2e: float | None = 500.0,
    tokens: int | None = 41,
    success: bool | None = True,
    sent: bool = True,
    client_error: str | None = None,
    start: float = 0.0,
    end: float = 1.0,
) -> RequestRow:
    return RequestRow(
        index=index,
        sent=sent,
        dispatched_offset_s=start if sent else None,
        completed_offset_s=end if sent else None,
        client_error=client_error,
        success=success,
        failure_category=None if success else "http_status",
        ttft_ms=ttft,
        tpot_ms=tpot,
        tpot_status=tpot_status,
        e2e_ms=e2e,
        output_tokens=tokens,
    )


SLO = SloTargets(ttft_ms=200.0, tpot_ms=20.0, e2e_ms=1000.0)


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"ttft_ms": 0.0}, {"tpot_ms": -1.0}, {"e2e_ms": float("nan")}],
)
def test_slo_targets_reject_empty_and_non_positive(kwargs: dict[str, float]) -> None:
    with pytest.raises(SloConfigError):
        SloTargets(**kwargs)


def test_verdict_order_failed_then_missed_then_unverifiable() -> None:
    assert request_verdict(_row(success=False), SLO) == VERDICT_FAILED
    assert request_verdict(_row(client_error="TimeoutError", success=None), SLO) == (
        VERDICT_FAILED
    )
    # A missed target decides even when another target has no value.
    missed = _row(ttft=250.0, tpot=None, tpot_status=TPOT_UNAVAILABLE)
    assert request_verdict(missed, SLO) == VERDICT_MISSED
    unverifiable = _row(tpot=None, tpot_status=TPOT_UNAVAILABLE)
    assert request_verdict(unverifiable, SLO) == VERDICT_UNVERIFIABLE
    assert request_verdict(_row(ttft=200.0, tpot=20.0, e2e=1000.0), SLO) == VERDICT_MET
    assert request_verdict(_row(sent=False, success=None), SLO) is None


def test_only_stated_targets_are_checked() -> None:
    slo = SloTargets(e2e_ms=600.0)
    row = _row(ttft=None, tpot=None, tpot_status=TPOT_UNAVAILABLE)
    assert request_verdict(row, slo) == VERDICT_MET


def test_single_token_response_satisfies_tpot_and_is_counted() -> None:
    one = _row(tokens=1, tpot=None, tpot_status=TPOT_NOT_APPLICABLE)
    summary = summarize_goodput([one, _row(1)], SLO)
    assert summary.counts[VERDICT_MET] == 2
    assert summary.tpot_not_applicable_met == 1


def test_failed_requests_stay_in_the_denominator() -> None:
    rows = [_row(0), _row(1, ttft=300.0), _row(2, success=False, tokens=None), _row(3)]
    summary = summarize_goodput(rows, SLO)
    assert summary.dispatched == 4
    assert summary.counts == {
        VERDICT_MET: 2,
        VERDICT_MISSED: 1,
        VERDICT_UNVERIFIABLE: 0,
        VERDICT_FAILED: 1,
    }
    assert summary.missed_by_target == {"ttft_ms": 1, "tpot_ms": 0, "e2e_ms": 0}
    assert summary.request_goodput_fraction["value"] == pytest.approx(0.5)
    assert summary.request_goodput_fraction["provenance"] == "derived"
    assert summary.request_goodput_per_second["value"] == pytest.approx(2.0)
    # The failed request reported no tokens, so the token denominator is unknown.
    assert summary.token_goodput_fraction["value"] is None
    assert summary.token_goodput_fraction["withheld_reason"] == WITHHELD_TOKENS_MISSING
    assert summary.token_goodput_per_second["value"] == pytest.approx(82.0)


def test_token_fraction_when_every_request_reports_tokens() -> None:
    rows = [_row(0, tokens=30), _row(1, e2e=5000.0, tokens=10)]
    summary = summarize_goodput(rows, SLO)
    assert summary.token_goodput_fraction["value"] == pytest.approx(0.75)


def test_unverifiable_withholds_headline_and_reports_bounds() -> None:
    rows = [
        _row(0),
        _row(1, tpot=None, tpot_status=TPOT_UNAVAILABLE, tokens=None),
        _row(2, e2e=2000.0),
        _row(3),
    ]
    summary = summarize_goodput(rows, SLO)
    assert summary.request_goodput_fraction["value"] is None
    assert summary.request_goodput_fraction["withheld_reason"] == WITHHELD_UNVERIFIABLE
    assert summary.request_goodput_fraction_bounds == {"lower": 0.5, "upper": 0.75}
    assert summary.request_goodput_per_second["value"] is None


def test_unsent_requests_make_the_set_incomplete() -> None:
    rows = [_row(0), _row(1, sent=False, success=None)]
    summary = summarize_goodput(rows, SLO)
    assert summary.incomplete
    assert summary.dispatched == 1
    assert summary.request_goodput_fraction["value"] is None
    assert summary.request_goodput_fraction["withheld_reason"] == WITHHELD_INCOMPLETE
    assert summary.request_goodput_fraction_bounds is None
    assert summary.counts[VERDICT_MET] == 1


def test_window_runs_from_first_dispatch_to_last_completion() -> None:
    rows = [_row(0, start=0.5, end=1.0), _row(1, start=0.75, end=2.5)]
    assert measured_window_seconds(rows) == pytest.approx(2.0)
    assert measured_window_seconds([_row(sent=False, success=None)]) is None
