"""Aggregation/reporting across executed verify-pipeline rows.

Reads every ``verification.json`` written by ``verify.execute_row`` under a
``workloads run --output-dir`` results directory and reports a small set of
distinct, well-defined metrics: how many rows landed in each
``RowStatus``, the pass rate among rows that were actually evaluated, and
a throughput figure ("correct cases per minute") computed only from rows
that both passed and have measured timing. Per-``decode_mode``,
per-``context_tier``, per-``backend`` and per-``provider`` breakdowns use
the same definitions.

Every axis exists so that figures which are not the same quantity stay
apart. A row measured on a local checkpoint and a row measured through a
hosted API are separated by ``backend``; two hosted endpoints are
separated by ``provider``. A metric that is undefined for a group is
reported as ``null``, never as ``0``: no evaluated rows means there is no
pass rate, which is not the same statement as a pass rate of zero.

This module deliberately does **not** blend correctness and speed into a
single combined "performance score" -- that would hide which axis (quality
vs. throughput) any one number reflects. Callers that want an overall
picture should read ``pass_rate`` and ``correct_cases_per_minute``
side by side.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..collectors._shared import atomic_write_text
from .verify import RowStatus, RowVerification, VerifyError

AGGREGATE_SCHEMA_VERSION = "2"
"""Schema version for the ``workloads summarize`` document.

v2 adds the ``by_backend`` and ``by_provider`` breakdowns, and adds
``timing_comparable``, ``timing_unavailable_reason`` and
``measurement_contexts`` to every group.

It is additive in shape, but one v1 field changed meaning and a reader
that ignores that will draw the wrong conclusion.
``correct_cases_per_minute`` was null in v1 only when no passing row
carried timing. In v2 it is *also* null when the group spans more than
one measurement context and the figure was deliberately withheld. Those
are different statements -- "nothing to measure" versus "measuring this
together would be wrong" -- and the field alone no longer distinguishes
them, which is exactly the conflation this version exists to prevent.
Consult ``timing_comparable`` to tell them apart, and
``timing_unavailable_reason`` for why.
"""


def _load_verifications(results_dir: Path) -> tuple[RowVerification, ...]:
    runs_dir = results_dir / "runs"
    if not runs_dir.is_dir():
        return ()
    verifications: list[RowVerification] = []
    for verification_path in sorted(runs_dir.glob("*/verification.json")):
        try:
            verification = RowVerification.read_json(verification_path)
            if any(
                value is not None and not math.isfinite(value)
                for value in (verification.quality_score, verification.total_ms)
            ):
                raise VerifyError(
                    "verification.json numeric summary fields must be finite"
                )
            verifications.append(verification)
        except (OSError, VerifyError):
            # A corrupt/partial artifact must not silently skew aggregates;
            # it is simply excluded rather than crashing the whole summary.
            continue
    return tuple(verifications)


def pass_rate(pass_count: int, evaluated_count: int) -> float | None:
    """Fraction of evaluated cases that passed, or ``None`` if none evaluated.

    Public so other modules (e.g. ``optimizer.tune``) that need the exact
    same pass-rate definition can reuse it instead of redefining it.
    """
    return pass_count / evaluated_count if evaluated_count else None


def correct_cases_per_minute(pass_count: int, total_pass_ms: float) -> float | None:
    """Passing cases per minute of measured time, or ``None`` if undefined.

    Public so other modules (e.g. ``optimizer.tune``) that need the exact
    same throughput definition can reuse it instead of redefining it.
    """
    if pass_count == 0 or total_pass_ms <= 0:
        return None
    minutes = total_pass_ms / 1000.0 / 60.0
    return pass_count / minutes


@dataclass(frozen=True)
class GroupSummary:
    """Aggregate metrics for one group (overall, one mode, or one tier)."""

    key: str
    total: int
    completed: int
    failed: int
    unsupported: int
    skipped: int
    inconclusive: int
    evaluated_total: int
    evaluated_pass: int
    pass_rate: float | None
    correct_cases_per_minute: float | None
    timing_comparable: bool = True
    """False when the group mixes execution semantics whose durations are
    not the same quantity."""

    timing_unavailable_reason: str | None = None
    """Why a timing-derived metric was withheld, when it was."""

    measurement_contexts: tuple[str, ...] = ()
    """The distinct ``backend/provider`` semantics present in the group,
    in sorted order. One entry means the group is homogeneous."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "total": self.total,
            "completed": self.completed,
            "failed": self.failed,
            "unsupported": self.unsupported,
            "skipped": self.skipped,
            "inconclusive": self.inconclusive,
            "evaluated_total": self.evaluated_total,
            "evaluated_pass": self.evaluated_pass,
            "pass_rate": self.pass_rate,
            "correct_cases_per_minute": self.correct_cases_per_minute,
            "timing_comparable": self.timing_comparable,
            "timing_unavailable_reason": self.timing_unavailable_reason,
            "measurement_contexts": list(self.measurement_contexts),
        }


def _summarize_group(
    key: str, verifications: Iterable[RowVerification]
) -> GroupSummary:
    items = list(verifications)
    counts = dict.fromkeys(RowStatus, 0)
    for item in items:
        counts[item.status] += 1

    # "Evaluated" rows are ones that actually produced a quality signal:
    # freshly completed or trusted-and-skipped rows with a quality score.
    # Failed/unsupported/inconclusive rows are excluded from pass rate and
    # throughput so they cannot be misread as passing or failing quality
    # checks they never ran.
    evaluated = [
        item
        for item in items
        if item.status in (RowStatus.COMPLETED, RowStatus.SKIPPED)
        and item.quality_score is not None
    ]
    evaluated_pass = [item for item in evaluated if item.outcome_success]
    total_pass_ms = sum(
        item.total_ms for item in evaluated_pass if item.total_ms is not None
    )

    # A duration only means something next to another duration measured the
    # same way. A local row times a model on this machine; an API row times
    # a request to somebody else's, over a network, on hardware nobody here
    # can see. Adding those together produces a "correct cases per minute"
    # that describes no system that exists, and the more rows an aggregate
    # covers the more authoritative that number looks.
    #
    # Quality is not affected: a pass is a pass wherever it was produced, so
    # pass counts and pass rate stay populated and only the timing-derived
    # figure is withheld, with the reason recorded rather than left to be
    # inferred from a null.
    contexts = sorted(
        {
            f"{item.backend}/{item.provider}" if item.provider else item.backend
            for item in evaluated_pass
        }
    )
    timing_comparable = len(contexts) <= 1
    timing_reason = (
        None
        if timing_comparable
        else (
            "withheld: this group mixes measurement contexts ("
            + ", ".join(contexts)
            + "), whose durations are not the same quantity. Read the "
            "per-backend and per-provider groups instead."
        )
    )

    return GroupSummary(
        key=key,
        total=len(items),
        completed=counts[RowStatus.COMPLETED],
        failed=counts[RowStatus.FAILED],
        unsupported=counts[RowStatus.UNSUPPORTED],
        skipped=counts[RowStatus.SKIPPED],
        inconclusive=counts[RowStatus.INCONCLUSIVE],
        evaluated_total=len(evaluated),
        evaluated_pass=len(evaluated_pass),
        pass_rate=pass_rate(len(evaluated_pass), len(evaluated)),
        correct_cases_per_minute=(
            correct_cases_per_minute(len(evaluated_pass), total_pass_ms)
            if timing_comparable
            else None
        ),
        timing_comparable=timing_comparable,
        timing_unavailable_reason=timing_reason,
        measurement_contexts=tuple(contexts),
    )


@dataclass(frozen=True)
class VerificationSummary:
    """Aggregate report across every ``verification.json`` in a results dir."""

    schema_version: str
    results_dir: str
    overall: GroupSummary
    by_decode_mode: tuple[GroupSummary, ...]
    by_context_tier: tuple[GroupSummary, ...]
    by_backend: tuple[GroupSummary, ...] = ()
    """One group per execution backend (``mlx``, ``openai-api``, ...).

    Kept separate because a locally measured row and a row measured
    through a hosted API do not share a hardware definition: the local
    figure times a model on this machine, the remote figure times a
    request to somebody else's. Their throughputs are not the same
    quantity and blending them would produce a number describing neither.
    """

    by_provider: tuple[GroupSummary, ...] = ()
    """One group per provider label, covering only rows that have one.

    Locally executed rows have no provider and are deliberately absent
    rather than being collected under a placeholder key, so these groups
    do not sum to ``overall``. A synthetic "none" provider would read as
    a real one and would silently merge every local row into a bucket
    that invites comparison against a hosted endpoint.
    """

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "results_dir": self.results_dir,
            "overall": self.overall.to_dict(),
            "by_decode_mode": [group.to_dict() for group in self.by_decode_mode],
            "by_context_tier": [group.to_dict() for group in self.by_context_tier],
            "by_backend": [group.to_dict() for group in self.by_backend],
            "by_provider": [group.to_dict() for group in self.by_provider],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


def summarize_results(results_dir: Path) -> VerificationSummary:
    """Aggregate every ``verification.json`` found under ``results_dir``."""
    verifications = _load_verifications(results_dir)
    overall = _summarize_group("overall", verifications)

    by_mode: dict[str, list[RowVerification]] = {}
    by_tier: dict[str, list[RowVerification]] = {}
    by_backend: dict[str, list[RowVerification]] = {}
    by_provider: dict[str, list[RowVerification]] = {}
    for item in verifications:
        by_mode.setdefault(item.decode_mode, []).append(item)
        by_tier.setdefault(item.context_tier, []).append(item)
        by_backend.setdefault(item.backend, []).append(item)
        if item.provider is not None:
            by_provider.setdefault(item.provider, []).append(item)

    return VerificationSummary(
        schema_version=AGGREGATE_SCHEMA_VERSION,
        results_dir=str(results_dir),
        overall=overall,
        by_decode_mode=tuple(
            _summarize_group(key, items) for key, items in sorted(by_mode.items())
        ),
        by_context_tier=tuple(
            _summarize_group(key, items) for key, items in sorted(by_tier.items())
        ),
        by_backend=tuple(
            _summarize_group(key, items) for key, items in sorted(by_backend.items())
        ),
        by_provider=tuple(
            _summarize_group(key, items) for key, items in sorted(by_provider.items())
        ),
    )


def write_summary(summary: VerificationSummary, path: Path) -> None:
    """Atomically write ``summary`` as pretty JSON to ``path``."""
    atomic_write_text(path, summary.to_json() + "\n")
