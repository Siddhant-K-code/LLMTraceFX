"""Output schema for a completed tuning run: the ``tune`` report.

Every number here traces back to a specific accepted run's canonical
``ExperimentRecord``; nothing is fabricated when evidence is missing. See
``tuner.py`` for how these are built and ``explain.py`` for the
human-readable rendering of the same data. ``TuneReport.from_dict``/
``from_json``/``read_json`` are the only supported ways to load a tune
report produced elsewhere (e.g. by the ``tune-report`` HTML viewer CLI):
they never trust arbitrary fields, reject non-finite numeric values, and
raise ``TuneReportValidationError`` on any malformed or inconsistent input.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..doctor.speculative import DoctorVerdict, SpeculativeRegressionReport
from .identity import CandidateKey, GroupKey, IdentityValidationError
from .loader import ExcludedRun, TuneInputError
from .policy import TunePolicy, TunePolicyError

TUNE_REPORT_SCHEMA_VERSION = "1"


class TuneReportValidationError(ValueError):
    """Raised when a ``TuneReport`` loaded from JSON is invalid or malformed."""


def _load_candidate_key(data: Any, *, context: str) -> CandidateKey:
    try:
        return CandidateKey.from_dict(data)
    except IdentityValidationError as exc:
        raise TuneReportValidationError(
            f"{context}.candidate_key is invalid: {exc}"
        ) from exc


def _load_group_key(data: Any, *, context: str) -> GroupKey:
    try:
        return GroupKey.from_dict(data)
    except IdentityValidationError as exc:
        raise TuneReportValidationError(
            f"{context}.group_key is invalid: {exc}"
        ) from exc


def _require(data: Any, key: str, *, context: str) -> Any:
    if not isinstance(data, dict) or key not in data:
        raise TuneReportValidationError(f"{context} is missing required field: {key!r}")
    return data[key]


def _require_str(data: Any, key: str, *, context: str) -> str:
    value = _require(data, key, context=context)
    if not isinstance(value, str) or not value:
        raise TuneReportValidationError(
            f"{context}.{key} must be a non-empty string, got {value!r}"
        )
    return value


def _optional_str(data: dict[str, Any], key: str, *, context: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise TuneReportValidationError(
            f"{context}.{key} must be a string or null, got {value!r}"
        )
    return value


def _string_tuple(data: dict[str, Any], key: str, *, context: str) -> tuple[str, ...]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TuneReportValidationError(
            f"{context}.{key} must be a list of strings, got {value!r}"
        )
    return tuple(value)


def _require_int(
    data: dict[str, Any], key: str, *, context: str, minimum: int | None = None
) -> int:
    value = _require(data, key, context=context)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TuneReportValidationError(
            f"{context}.{key} must be an integer, got {value!r}"
        )
    if minimum is not None and value < minimum:
        raise TuneReportValidationError(
            f"{context}.{key} must be >= {minimum}, got {value}"
        )
    return int(value)


def _require_finite_float(data: dict[str, Any], key: str, *, context: str) -> float:
    value = _require(data, key, context=context)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TuneReportValidationError(
            f"{context}.{key} must be a number, got {value!r}"
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise TuneReportValidationError(
            f"{context}.{key} must be a finite number, got {numeric!r}"
        )
    return numeric


def _optional_finite_float(
    data: dict[str, Any], key: str, *, context: str
) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TuneReportValidationError(
            f"{context}.{key} must be a number or null, got {value!r}"
        )
    numeric = float(value)
    if not math.isfinite(numeric):
        raise TuneReportValidationError(
            f"{context}.{key} must be a finite number, got {numeric!r}"
        )
    return numeric


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

    @classmethod
    def from_dict(cls, data: Any) -> CandidateReport:
        if not isinstance(data, dict):
            raise TuneReportValidationError("candidate report must be a JSON object")
        context = "candidate report"
        return cls(
            candidate_key=_load_candidate_key(
                _require(data, "candidate_key", context=context),
                context=context,
            ),
            rank=_require_int(data, "rank", context=context, minimum=1),
            run_ids=_string_tuple(data, "run_ids", context=context),
            verification_paths=_string_tuple(
                data, "verification_paths", context=context
            ),
            final_record_paths=_string_tuple(
                data, "final_record_paths", context=context
            ),
            evidence_count=_require_int(
                data, "evidence_count", context=context, minimum=0
            ),
            objective_name=_require_str(data, "objective_name", context=context),
            objective_value=_require_finite_float(
                data, "objective_value", context=context
            ),
            mean_total_latency_ms=_optional_finite_float(
                data, "mean_total_latency_ms", context=context
            ),
            stdev_total_latency_ms=_optional_finite_float(
                data, "stdev_total_latency_ms", context=context
            ),
            coefficient_of_variation=_optional_finite_float(
                data, "coefficient_of_variation", context=context
            ),
            correct_cases_per_minute=_optional_finite_float(
                data, "correct_cases_per_minute", context=context
            ),
            pass_rate=_optional_finite_float(data, "pass_rate", context=context),
            mean_quality_score=_optional_finite_float(
                data, "mean_quality_score", context=context
            ),
            quality_metric=_optional_str(data, "quality_metric", context=context),
            mean_peak_memory_bytes=_optional_finite_float(
                data, "mean_peak_memory_bytes", context=context
            ),
            max_peak_memory_bytes=_optional_finite_float(
                data, "max_peak_memory_bytes", context=context
            ),
        )


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

    @classmethod
    def from_dict(cls, data: Any) -> RejectedCandidateReport:
        if not isinstance(data, dict):
            raise TuneReportValidationError(
                "rejected candidate report must be a JSON object"
            )
        context = "rejected candidate report"
        return cls(
            candidate_key=_load_candidate_key(
                _require(data, "candidate_key", context=context),
                context=context,
            ),
            run_ids=_string_tuple(data, "run_ids", context=context),
            verification_paths=_string_tuple(
                data, "verification_paths", context=context
            ),
            final_record_paths=_string_tuple(
                data, "final_record_paths", context=context
            ),
            reasons=_string_tuple(data, "reasons", context=context),
        )


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

    @classmethod
    def from_dict(cls, data: Any) -> BaselineComparison:
        if not isinstance(data, dict):
            raise TuneReportValidationError("baseline comparison must be a JSON object")
        context = "baseline comparison"
        raw_verdict = _require_str(data, "verdict", context=context)
        try:
            verdict = DoctorVerdict(raw_verdict)
        except ValueError as exc:
            raise TuneReportValidationError(
                f"{context}.verdict has an invalid value: {exc}"
            ) from exc
        speculative_report = SpeculativeRegressionReport(
            verdict=verdict,
            reason=_require_str(data, "reason", context=context),
            baseline_run_ids=_string_tuple(data, "baseline_run_ids", context=context),
            speculative_run_ids=_string_tuple(
                data, "speculative_run_ids", context=context
            ),
            baseline_mean_total_ms=_optional_finite_float(
                data, "baseline_mean_total_ms", context=context
            ),
            speculative_mean_total_ms=_optional_finite_float(
                data, "speculative_mean_total_ms", context=context
            ),
            delta_ms=_optional_finite_float(data, "delta_ms", context=context),
            delta_pct=_optional_finite_float(data, "delta_pct", context=context),
        )
        return cls(
            baseline_candidate_key=_load_candidate_key(
                _require(data, "baseline_candidate_key", context=context),
                context=f"{context}.baseline",
            ),
            speculative_candidate_key=_load_candidate_key(
                _require(data, "speculative_candidate_key", context=context),
                context=f"{context}.speculative",
            ),
            report=speculative_report,
        )


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

    @classmethod
    def from_dict(cls, data: Any) -> GroupReport:
        if not isinstance(data, dict):
            raise TuneReportValidationError("group report must be a JSON object")
        context = "group report"

        raw_outcome = _require_str(data, "outcome", context=context)
        try:
            outcome = GroupOutcome(raw_outcome)
        except ValueError as exc:
            raise TuneReportValidationError(
                f"{context}.outcome has an invalid value: {exc}"
            ) from exc

        recommended_raw = data.get("recommended")
        recommended = (
            None
            if recommended_raw is None
            else CandidateReport.from_dict(recommended_raw)
        )

        accepted_raw = data.get("accepted", [])
        if not isinstance(accepted_raw, list):
            raise TuneReportValidationError(f"{context}.accepted must be a list")
        accepted = tuple(
            sorted(
                (CandidateReport.from_dict(item) for item in accepted_raw),
                key=lambda candidate: candidate.rank,
            )
        )
        accepted_ranks = tuple(candidate.rank for candidate in accepted)
        if accepted_ranks != tuple(range(1, len(accepted) + 1)):
            raise TuneReportValidationError(
                f"{context}.accepted ranks must be unique and contiguous "
                f"starting at 1, got {accepted_ranks!r}"
            )

        rejected_raw = data.get("rejected", [])
        if not isinstance(rejected_raw, list):
            raise TuneReportValidationError(f"{context}.rejected must be a list")
        rejected = tuple(
            RejectedCandidateReport.from_dict(item) for item in rejected_raw
        )

        baseline_comparison_raw = data.get("baseline_comparison")
        baseline_comparison = (
            None
            if baseline_comparison_raw is None
            else BaselineComparison.from_dict(baseline_comparison_raw)
        )

        group = cls(
            group_key=_load_group_key(
                _require(data, "group_key", context=context),
                context=context,
            ),
            outcome=outcome,
            recommended=recommended,
            accepted=accepted,
            rejected=rejected,
            inconclusive_reason=_optional_str(
                data, "inconclusive_reason", context=context
            ),
            baseline_comparison=baseline_comparison,
        )
        if group.outcome == GroupOutcome.RECOMMENDED and group.recommended is None:
            raise TuneReportValidationError(
                f"{context} has outcome 'recommended' but 'recommended' is null"
            )
        if group.outcome == GroupOutcome.RECOMMENDED:
            if not group.accepted:
                raise TuneReportValidationError(
                    f"{context} has outcome 'recommended' but accepted is empty"
                )
            if group.recommended != group.accepted[0]:
                raise TuneReportValidationError(
                    f"{context}.recommended must equal the rank 1 accepted candidate"
                )
        if (
            group.outcome == GroupOutcome.INCONCLUSIVE
            and group.inconclusive_reason is None
        ):
            raise TuneReportValidationError(
                f"{context} has outcome 'inconclusive' but 'inconclusive_reason' "
                "is null"
            )
        if group.outcome == GroupOutcome.INCONCLUSIVE:
            if group.recommended is not None:
                raise TuneReportValidationError(
                    f"{context} has outcome 'inconclusive' but recommended is set"
                )
            if group.baseline_comparison is not None:
                raise TuneReportValidationError(
                    f"{context} has outcome 'inconclusive' but "
                    "baseline_comparison is set"
                )
        if (
            group.baseline_comparison is not None
            and group.recommended is not None
            and group.baseline_comparison.speculative_candidate_key
            != group.recommended.candidate_key
        ):
            raise TuneReportValidationError(
                f"{context}.baseline_comparison speculative candidate must "
                "equal the recommended candidate"
            )
        return group


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

    @classmethod
    def from_dict(cls, data: Any) -> TuneReport:
        if not isinstance(data, dict):
            raise TuneReportValidationError("tune report must be a JSON object")
        context = "tune report"

        schema_version = str(data.get("schema_version", TUNE_REPORT_SCHEMA_VERSION))
        if schema_version != TUNE_REPORT_SCHEMA_VERSION:
            raise TuneReportValidationError(
                f"unsupported tune report schema_version {schema_version!r}, "
                f"expected {TUNE_REPORT_SCHEMA_VERSION!r}"
            )

        policy_raw = _require(data, "policy", context=context)
        try:
            policy = TunePolicy.from_dict(policy_raw)
        except TunePolicyError as exc:
            raise TuneReportValidationError(
                f"{context}.policy is invalid: {exc}"
            ) from exc

        groups_raw = data.get("groups", [])
        if not isinstance(groups_raw, list):
            raise TuneReportValidationError(f"{context}.groups must be a list")
        groups = tuple(GroupReport.from_dict(item) for item in groups_raw)

        excluded_raw = data.get("excluded_runs", [])
        if not isinstance(excluded_raw, list):
            raise TuneReportValidationError(f"{context}.excluded_runs must be a list")
        try:
            excluded_runs = tuple(ExcludedRun.from_dict(item) for item in excluded_raw)
        except TuneInputError as exc:
            raise TuneReportValidationError(
                f"{context}.excluded_runs is invalid: {exc}"
            ) from exc

        return cls(
            schema_version=schema_version,
            generated_at=_require_str(data, "generated_at", context=context),
            results_dirs=_string_tuple(data, "results_dirs", context=context),
            policy=policy,
            groups=groups,
            excluded_runs=excluded_runs,
        )

    @classmethod
    def from_json(cls, payload: str) -> TuneReport:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise TuneReportValidationError(
                f"invalid JSON for tune report: {exc}"
            ) from exc
        return cls.from_dict(data)

    @classmethod
    def read_json(cls, path: str | Path) -> TuneReport:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
