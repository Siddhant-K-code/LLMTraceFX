"""Output schema for a completed tuning run: the ``tune`` report.

Every number here traces back to a specific accepted run's canonical
``ExperimentRecord``; nothing is fabricated when evidence is missing. See
``tuner.py`` for how these are built and ``explain.py`` for the
human-readable rendering of the same data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..doctor.speculative import SpeculativeRegressionReport
from .identity import CandidateKey, GroupKey
from .loader import ExcludedRun
from .policy import TunePolicy

TUNE_REPORT_SCHEMA_VERSION = "1"


class GroupOutcome(str, Enum):
    """Whether a comparable group produced an actionable recommendation."""

    RECOMMENDED = "recommended"
    """At least one candidate was accepted and a winner was selected."""

    INCONCLUSIVE = "inconclusive"
    """No candidate survived every constraint, or the leading candidates
    are tied within measurement noise; see ``inconclusive_reason``."""


@dataclass(frozen=True)
class CandidateReport:
    """One accepted candidate's measurements, evidence, and rank."""

    candidate_key: CandidateKey
    rank: int
    run_ids: tuple[str, ...]
    verification_paths: tuple[str, ...]
    final_record_paths: tuple[str, ...]
    evidence_count: int
    objective_name: str
    objective_value: float
    mean_total_latency_ms: float | None
    stdev_total_latency_ms: float | None
    coefficient_of_variation: float | None
    correct_cases_per_minute: float | None
    pass_rate: float | None
    mean_quality_score: float | None
    quality_metric: str | None
    mean_peak_memory_bytes: float | None
    max_peak_memory_bytes: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key.to_dict(),
            "rank": self.rank,
            "run_ids": list(self.run_ids),
            "verification_paths": list(self.verification_paths),
            "final_record_paths": list(self.final_record_paths),
            "evidence_count": self.evidence_count,
            "objective_name": self.objective_name,
            "objective_value": self.objective_value,
            "mean_total_latency_ms": self.mean_total_latency_ms,
            "stdev_total_latency_ms": self.stdev_total_latency_ms,
            "coefficient_of_variation": self.coefficient_of_variation,
            "correct_cases_per_minute": self.correct_cases_per_minute,
            "pass_rate": self.pass_rate,
            "mean_quality_score": self.mean_quality_score,
            "quality_metric": self.quality_metric,
            "mean_peak_memory_bytes": self.mean_peak_memory_bytes,
            "max_peak_memory_bytes": self.max_peak_memory_bytes,
        }


@dataclass(frozen=True)
class RejectedCandidateReport:
    """One rejected candidate and every constraint it violated."""

    candidate_key: CandidateKey
    run_ids: tuple[str, ...]
    verification_paths: tuple[str, ...]
    final_record_paths: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_key": self.candidate_key.to_dict(),
            "run_ids": list(self.run_ids),
            "verification_paths": list(self.verification_paths),
            "final_record_paths": list(self.final_record_paths),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class BaselineComparison:
    """Optional autoregressive-baseline comparison for a group's winner."""

    baseline_candidate_key: CandidateKey
    speculative_candidate_key: CandidateKey
    report: SpeculativeRegressionReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_candidate_key": self.baseline_candidate_key.to_dict(),
            "speculative_candidate_key": self.speculative_candidate_key.to_dict(),
            "verdict": self.report.verdict.value,
            "reason": self.report.reason,
            "baseline_run_ids": list(self.report.baseline_run_ids),
            "speculative_run_ids": list(self.report.speculative_run_ids),
            "baseline_mean_total_ms": self.report.baseline_mean_total_ms,
            "speculative_mean_total_ms": self.report.speculative_mean_total_ms,
            "delta_ms": self.report.delta_ms,
            "delta_pct": self.report.delta_pct,
        }


@dataclass(frozen=True)
class GroupReport:
    """One comparable group's full accepted/rejected candidate breakdown."""

    group_key: GroupKey
    outcome: GroupOutcome
    recommended: CandidateReport | None
    accepted: tuple[CandidateReport, ...]
    rejected: tuple[RejectedCandidateReport, ...]
    inconclusive_reason: str | None
    baseline_comparison: BaselineComparison | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_key": self.group_key.to_dict(),
            "group_label": self.group_key.label(),
            "outcome": self.outcome.value,
            "recommended": (
                None if self.recommended is None else self.recommended.to_dict()
            ),
            "accepted": [candidate.to_dict() for candidate in self.accepted],
            "rejected": [candidate.to_dict() for candidate in self.rejected],
            "inconclusive_reason": self.inconclusive_reason,
            "baseline_comparison": (
                None
                if self.baseline_comparison is None
                else self.baseline_comparison.to_dict()
            ),
        }


@dataclass(frozen=True)
class TuneReport:
    """The complete output of one ``tune`` invocation."""

    schema_version: str
    generated_at: str
    results_dirs: tuple[str, ...]
    policy: TunePolicy
    groups: tuple[GroupReport, ...]
    excluded_runs: tuple[ExcludedRun, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "results_dirs": list(self.results_dirs),
            "policy": self.policy.to_dict(),
            "groups": [group.to_dict() for group in self.groups],
            "excluded_runs": [run.to_dict() for run in self.excluded_runs],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @property
    def has_recommendation(self) -> bool:
        return any(group.outcome == GroupOutcome.RECOMMENDED for group in self.groups)
