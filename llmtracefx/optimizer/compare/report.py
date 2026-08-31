"""Output schema for one ``compare`` run: the cross-system comparison report.

Every number here traces back to a specific accepted run's canonical
``ExperimentRecord``, the provider's own usage block, or an explicitly
supplied pricing entry. Nothing is invented when evidence is missing: an
unavailable measurement stays ``null`` and the reason it is unavailable is
carried alongside it in ``missing_evidence``.

``CompareReport.from_dict``/``from_json``/``read_json`` are the only
supported ways to load a report produced elsewhere (for example by the
``compare-report`` HTML renderer). They never trust arbitrary fields, reject
non-finite numbers, and raise ``CompareReportValidationError`` on any
malformed or internally inconsistent input.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..schema import MetricProvenance
from ..tune.loader import ExcludedRun, TuneInputError
from .cost import MONETARY_BASIS
from .identity import ComparableUnitKey, CompareIdentityError, SystemKey
from .policy import (
    CompareConstraints,
    CompareObjective,
    ComparePolicy,
    ComparePolicyError,
)
from .pricing import CURRENCY_PATTERN

COMPARE_REPORT_SCHEMA_VERSION = "1"


class CompareReportValidationError(ValueError):
    """Raised when a ``CompareReport`` loaded from JSON is invalid."""


def _require(data: Any, key: str, *, context: str) -> Any:
    if not isinstance(data, dict) or key not in data:
        raise CompareReportValidationError(
            f"{context} is missing required field: {key!r}"
        )
    return data[key]


def _require_str(data: Any, key: str, *, context: str) -> str:
    value = _require(data, key, context=context)
    if not isinstance(value, str) or not value:
        raise CompareReportValidationError(
            f"{context}.{key} must be a non-empty string, got {value!r}"
        )
    return value


def _optional_str(data: dict[str, Any], key: str, *, context: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise CompareReportValidationError(
            f"{context}.{key} must be a string or null, got {value!r}"
        )
    return value


def _string_tuple(data: dict[str, Any], key: str, *, context: str) -> tuple[str, ...]:
    value = data.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise CompareReportValidationError(
            f"{context}.{key} must be a list of strings, got {value!r}"
        )
    return tuple(value)


def _require_int(
    data: dict[str, Any], key: str, *, context: str, minimum: int | None = None
) -> int:
    value = _require(data, key, context=context)
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompareReportValidationError(
            f"{context}.{key} must be an integer, got {value!r}"
        )
    if minimum is not None and value < minimum:
        raise CompareReportValidationError(
            f"{context}.{key} must be >= {minimum}, got {value}"
        )
    return int(value)


def _optional_int(
    data: dict[str, Any], key: str, *, context: str, minimum: int | None = None
) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise CompareReportValidationError(
            f"{context}.{key} must be an integer or null, got {value!r}"
        )
    if minimum is not None and value < minimum:
        raise CompareReportValidationError(
            f"{context}.{key} must be >= {minimum}, got {value}"
        )
    return int(value)


def _require_bool(data: dict[str, Any], key: str, *, context: str) -> bool:
    value = _require(data, key, context=context)
    if not isinstance(value, bool):
        raise CompareReportValidationError(
            f"{context}.{key} must be a boolean, got {value!r}"
        )
    return value


def _require_currency(data: dict[str, Any], key: str, *, context: str) -> str:
    """Require an ISO 4217 alphabetic code, exactly as the manifest does.

    A persisted report is untrusted input and this string is rendered into a
    document. Applying the manifest's own pattern here keeps a loaded report
    to the same standard as the manifest it claims to have come from, so a
    hand-edited file cannot smuggle arbitrary text through the one field
    whose validation used to stop at the manifest boundary.
    """
    value = _require_str(data, key, context=context)
    if not CURRENCY_PATTERN.match(value):
        raise CompareReportValidationError(
            f"{context}.{key} must be a three-letter ISO 4217 code "
            f"(e.g. 'USD'), got {value!r}"
        )
    return value


def _require_finite_float(data: dict[str, Any], key: str, *, context: str) -> float:
    value = _require(data, key, context=context)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CompareReportValidationError(
            f"{context}.{key} must be a number, got {value!r}"
        )
    try:
        numeric = float(value)
    except OverflowError as exc:
        # A JSON integer literal too large for a float arrives here as a
        # Python int. ``float()`` raises ``OverflowError``, an
        # ``ArithmeticError`` rather than a ``ValueError``, so no caller
        # catches it and it escapes as a traceback. Every one of these
        # inputs is a user-supplied file this module treats as untrusted,
        # so it becomes the same typed validation failure as any other
        # malformed value.
        raise CompareReportValidationError(
            f"{context}.{key} is too large to represent as a number: {exc}"
        ) from exc
    if not math.isfinite(numeric):
        raise CompareReportValidationError(
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
        raise CompareReportValidationError(
            f"{context}.{key} must be a number or null, got {value!r}"
        )
    try:
        numeric = float(value)
    except OverflowError as exc:
        # A JSON integer literal too large for a float arrives here as a
        # Python int. ``float()`` raises ``OverflowError``, an
        # ``ArithmeticError`` rather than a ``ValueError``, so no caller
        # catches it and it escapes as a traceback. Every one of these
        # inputs is a user-supplied file this module treats as untrusted,
        # so it becomes the same typed validation failure as any other
        # malformed value.
        raise CompareReportValidationError(
            f"{context}.{key} is too large to represent as a number: {exc}"
        ) from exc
    if not math.isfinite(numeric):
        raise CompareReportValidationError(
            f"{context}.{key} must be a finite number, got {numeric!r}"
        )
    return numeric


class StratumOutcome(str, Enum):
    """Whether one comparable unit produced an actionable recommendation."""

    RECOMMENDED = "recommended"
    """At least one system cleared every constraint and won on the objective."""

    INCONCLUSIVE = "inconclusive"
    """No system cleared every constraint, or the leading systems cannot be
    told apart from measurement noise, or the evidence needed for the
    objective is missing; see ``inconclusive_reason``."""


class TtftBasis(str, Enum):
    """Which measurement a reported time-to-first-token actually is.

    These two are never pooled and never ranked against each other. A local
    prefill is model prompt-processing time observed on the host. A hosted
    API's first-content-token offset is a client-side interval that also
    contains DNS, connection setup, TLS, request transfer and any
    server-side queueing.
    """

    LOCAL_PREFILL = "local_prefill"
    CLIENT_OBSERVED_STREAM = "client_observed_stream"


class ParetoAxis(str, Enum):
    """One axis of the evidence frontier, named with its preferred direction."""

    MAX_PASS_RATE = "max_pass_rate"
    MAX_CORRECT_CASES_PER_MINUTE = "max_correct_cases_per_minute"
    MIN_MEAN_TOTAL_LATENCY_MS = "min_mean_total_latency_ms"
    MIN_COST_PER_CORRECT_CASE = "min_cost_per_correct_case"

    @property
    def prefers_lower(self) -> bool:
        return self in (
            ParetoAxis.MIN_MEAN_TOTAL_LATENCY_MS,
            ParetoAxis.MIN_COST_PER_CORRECT_CASE,
        )


@dataclass(frozen=True)
class UsageTotals:
    """Provider-reported token totals across one system's ranked runs."""

    runs_reporting_usage: int
    runs_total: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    reasoning_tokens: int | None = None

    @property
    def complete(self) -> bool:
        """True when every ranked run of this system reported its usage."""
        return self.runs_total > 0 and self.runs_reporting_usage == self.runs_total

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": MetricProvenance.PROVIDER_REPORTED.value,
            "runs_reporting_usage": self.runs_reporting_usage,
            "runs_total": self.runs_total,
            "complete": self.complete,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }

    @classmethod
    def from_dict(cls, data: Any) -> UsageTotals:
        if not isinstance(data, dict):
            raise CompareReportValidationError("usage must be a JSON object")
        context = "usage"
        declared_provenance = data.get(
            "provenance", MetricProvenance.PROVIDER_REPORTED.value
        )
        if declared_provenance != MetricProvenance.PROVIDER_REPORTED.value:
            raise CompareReportValidationError(
                f"{context}.provenance must be "
                f"{MetricProvenance.PROVIDER_REPORTED.value!r}; token usage is "
                "never a client measurement"
            )
        totals = cls(
            runs_reporting_usage=_require_int(
                data, "runs_reporting_usage", context=context, minimum=0
            ),
            runs_total=_require_int(data, "runs_total", context=context, minimum=0),
            input_tokens=_optional_int(
                data, "input_tokens", context=context, minimum=0
            ),
            output_tokens=_optional_int(
                data, "output_tokens", context=context, minimum=0
            ),
            cached_input_tokens=_optional_int(
                data, "cached_input_tokens", context=context, minimum=0
            ),
            reasoning_tokens=_optional_int(
                data, "reasoning_tokens", context=context, minimum=0
            ),
        )
        if totals.runs_reporting_usage > totals.runs_total:
            raise CompareReportValidationError(
                f"{context}.runs_reporting_usage ({totals.runs_reporting_usage}) "
                f"cannot exceed runs_total ({totals.runs_total})"
            )
        return totals


@dataclass(frozen=True)
class CostSummary:
    """One system's estimated spend, always labeled as derived, never measured."""

    currency: str
    pricing_entry_id: str
    pricing_entry_sha256: str
    rates_are_illustrative: bool
    total_amount: float | None = None
    cost_per_case: float | None = None
    cost_per_correct_case: float | None = None
    correct_cases_per_currency_unit: float | None = None
    reasons: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "estimated": True,
            "monetary_basis": MONETARY_BASIS,
            "pricing_entry_id": self.pricing_entry_id,
            "pricing_entry_sha256": self.pricing_entry_sha256,
            "rates_are_illustrative": self.rates_are_illustrative,
            "total_amount": self.total_amount,
            "cost_per_case": self.cost_per_case,
            "cost_per_correct_case": self.cost_per_correct_case,
            "correct_cases_per_currency_unit": self.correct_cases_per_currency_unit,
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: Any) -> CostSummary:
        if not isinstance(data, dict):
            raise CompareReportValidationError("cost must be a JSON object")
        context = "cost"
        if data.get("estimated") is not True:
            raise CompareReportValidationError(
                f"{context}.estimated must be true; every monetary value in this "
                "report is derived from provider-reported usage and supplied "
                "rates, never measured"
            )
        if data.get("monetary_basis", MONETARY_BASIS) != MONETARY_BASIS:
            raise CompareReportValidationError(
                f"{context}.monetary_basis must be {MONETARY_BASIS!r}"
            )
        return cls(
            currency=_require_currency(data, "currency", context=context),
            pricing_entry_id=_require_str(data, "pricing_entry_id", context=context),
            pricing_entry_sha256=_require_str(
                data, "pricing_entry_sha256", context=context
            ),
            rates_are_illustrative=_require_bool(
                data, "rates_are_illustrative", context=context
            ),
            total_amount=_optional_finite_float(data, "total_amount", context=context),
            cost_per_case=_optional_finite_float(
                data, "cost_per_case", context=context
            ),
            cost_per_correct_case=_optional_finite_float(
                data, "cost_per_correct_case", context=context
            ),
            correct_cases_per_currency_unit=_optional_finite_float(
                data, "correct_cases_per_currency_unit", context=context
            ),
            reasons=_string_tuple(data, "reasons", context=context),
        )


@dataclass(frozen=True)
class SystemReport:
    """One system's measurements, evidence and rank within a comparable unit."""

    system_key: SystemKey
    rank: int
    run_ids: tuple[str, ...]
    verification_paths: tuple[str, ...]
    record_paths: tuple[str, ...]
    evidence_count: int
    objective_name: str
    objective_value: float
    pass_rate: float | None = None
    mean_quality_score: float | None = None
    quality_metric: str | None = None
    mean_total_latency_ms: float | None = None
    p50_total_latency_ms: float | None = None
    p95_total_latency_ms: float | None = None
    stdev_total_latency_ms: float | None = None
    coefficient_of_variation: float | None = None
    correct_cases_per_minute: float | None = None
    mean_ttft_ms: float | None = None
    ttft_basis: TtftBasis | None = None
    usage: UsageTotals | None = None
    cost: CostSummary | None = None
    mean_peak_memory_bytes: float | None = None
    max_peak_memory_bytes: float | None = None
    missing_evidence: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_key": self.system_key.to_dict(),
            "system_label": self.system_key.label(),
            "rank": self.rank,
            "run_ids": list(self.run_ids),
            "verification_paths": list(self.verification_paths),
            "record_paths": list(self.record_paths),
            "evidence_count": self.evidence_count,
            "objective_name": self.objective_name,
            "objective_value": self.objective_value,
            "pass_rate": self.pass_rate,
            "mean_quality_score": self.mean_quality_score,
            "quality_metric": self.quality_metric,
            "mean_total_latency_ms": self.mean_total_latency_ms,
            "p50_total_latency_ms": self.p50_total_latency_ms,
            "p95_total_latency_ms": self.p95_total_latency_ms,
            "stdev_total_latency_ms": self.stdev_total_latency_ms,
            "coefficient_of_variation": self.coefficient_of_variation,
            "correct_cases_per_minute": self.correct_cases_per_minute,
            "mean_ttft_ms": self.mean_ttft_ms,
            "ttft_basis": None if self.ttft_basis is None else self.ttft_basis.value,
            "usage": None if self.usage is None else self.usage.to_dict(),
            "cost": None if self.cost is None else self.cost.to_dict(),
            "mean_peak_memory_bytes": self.mean_peak_memory_bytes,
            "max_peak_memory_bytes": self.max_peak_memory_bytes,
            "missing_evidence": list(self.missing_evidence),
        }

    @classmethod
    def from_dict(cls, data: Any) -> SystemReport:
        if not isinstance(data, dict):
            raise CompareReportValidationError("system report must be a JSON object")
        context = "system report"
        try:
            system_key = SystemKey.from_dict(
                _require(data, "system_key", context=context)
            )
        except CompareIdentityError as exc:
            raise CompareReportValidationError(
                f"{context}.system_key is invalid: {exc}"
            ) from exc

        raw_basis = data.get("ttft_basis")
        ttft_basis: TtftBasis | None
        if raw_basis is None:
            ttft_basis = None
        else:
            try:
                ttft_basis = TtftBasis(raw_basis)
            except ValueError as exc:
                raise CompareReportValidationError(
                    f"{context}.ttft_basis has an invalid value: {exc}"
                ) from exc

        mean_ttft = _optional_finite_float(data, "mean_ttft_ms", context=context)
        if mean_ttft is not None and ttft_basis is None:
            raise CompareReportValidationError(
                f"{context} reports mean_ttft_ms without a ttft_basis; a "
                "time-to-first-token figure is meaningless without saying which "
                "measurement it is"
            )

        usage_raw = data.get("usage")
        cost_raw = data.get("cost")
        report = cls(
            system_key=system_key,
            rank=_require_int(data, "rank", context=context, minimum=1),
            run_ids=_string_tuple(data, "run_ids", context=context),
            verification_paths=_string_tuple(
                data, "verification_paths", context=context
            ),
            record_paths=_string_tuple(data, "record_paths", context=context),
            evidence_count=_require_int(
                data, "evidence_count", context=context, minimum=0
            ),
            objective_name=_require_str(data, "objective_name", context=context),
            objective_value=_require_finite_float(
                data, "objective_value", context=context
            ),
            pass_rate=_optional_finite_float(data, "pass_rate", context=context),
            mean_quality_score=_optional_finite_float(
                data, "mean_quality_score", context=context
            ),
            quality_metric=_optional_str(data, "quality_metric", context=context),
            mean_total_latency_ms=_optional_finite_float(
                data, "mean_total_latency_ms", context=context
            ),
            p50_total_latency_ms=_optional_finite_float(
                data, "p50_total_latency_ms", context=context
            ),
            p95_total_latency_ms=_optional_finite_float(
                data, "p95_total_latency_ms", context=context
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
            mean_ttft_ms=mean_ttft,
            ttft_basis=ttft_basis,
            usage=None if usage_raw is None else UsageTotals.from_dict(usage_raw),
            cost=None if cost_raw is None else CostSummary.from_dict(cost_raw),
            mean_peak_memory_bytes=_optional_finite_float(
                data, "mean_peak_memory_bytes", context=context
            ),
            max_peak_memory_bytes=_optional_finite_float(
                data, "max_peak_memory_bytes", context=context
            ),
            missing_evidence=_string_tuple(data, "missing_evidence", context=context),
        )
        if not report.system_key.is_local and (
            report.mean_peak_memory_bytes is not None
            or report.max_peak_memory_bytes is not None
        ):
            raise CompareReportValidationError(
                f"{context} reports peak memory for a system executed by "
                f"provider {report.system_key.provider!r}; peak memory is a "
                "local-only measurement and a hosted API cannot produce one"
            )
        if (
            report.system_key.is_local
            and report.ttft_basis == TtftBasis.CLIENT_OBSERVED_STREAM
        ):
            # These two claims cannot both be true. A client-observed stream
            # offset is only ever produced by talking to a hosted service, so
            # a system carrying one is not local, whatever its provider field
            # says. Checked here as well as in the evidence loader because a
            # report can arrive from anywhere.
            raise CompareReportValidationError(
                f"{context} claims to be local but reports a "
                f"{TtftBasis.CLIENT_OBSERVED_STREAM.value!r} "
                "time-to-first-token, which only a hosted service can "
                "produce; the two claims contradict each other"
            )
        return report


@dataclass(frozen=True)
class RejectedSystemReport:
    """One system that did not clear the constraints, and every reason why."""

    system_key: SystemKey
    run_ids: tuple[str, ...]
    verification_paths: tuple[str, ...]
    record_paths: tuple[str, ...]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_key": self.system_key.to_dict(),
            "system_label": self.system_key.label(),
            "run_ids": list(self.run_ids),
            "verification_paths": list(self.verification_paths),
            "record_paths": list(self.record_paths),
            "reasons": list(self.reasons),
        }

    @classmethod
    def from_dict(cls, data: Any) -> RejectedSystemReport:
        if not isinstance(data, dict):
            raise CompareReportValidationError(
                "rejected system report must be a JSON object"
            )
        context = "rejected system report"
        try:
            system_key = SystemKey.from_dict(
                _require(data, "system_key", context=context)
            )
        except CompareIdentityError as exc:
            raise CompareReportValidationError(
                f"{context}.system_key is invalid: {exc}"
            ) from exc
        reasons = _string_tuple(data, "reasons", context=context)
        if not reasons:
            raise CompareReportValidationError(
                f"{context} must record at least one rejection reason"
            )
        return cls(
            system_key=system_key,
            run_ids=_string_tuple(data, "run_ids", context=context),
            verification_paths=_string_tuple(
                data, "verification_paths", context=context
            ),
            record_paths=_string_tuple(data, "record_paths", context=context),
            reasons=reasons,
        )


@dataclass(frozen=True)
class FrontierEntry:
    """One system's position on the evidence frontier for a comparable unit."""

    system_key: SystemKey
    dominated: bool
    dominated_by: tuple[str, ...] = field(default_factory=tuple)
    """Labels of the systems that beat this one on every frontier axis."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "system_key": self.system_key.to_dict(),
            "system_label": self.system_key.label(),
            "dominated": self.dominated,
            "dominated_by": list(self.dominated_by),
        }

    @classmethod
    def from_dict(cls, data: Any) -> FrontierEntry:
        if not isinstance(data, dict):
            raise CompareReportValidationError("frontier entry must be a JSON object")
        context = "frontier entry"
        try:
            system_key = SystemKey.from_dict(
                _require(data, "system_key", context=context)
            )
        except CompareIdentityError as exc:
            raise CompareReportValidationError(
                f"{context}.system_key is invalid: {exc}"
            ) from exc
        entry = cls(
            system_key=system_key,
            dominated=_require_bool(data, "dominated", context=context),
            dominated_by=_string_tuple(data, "dominated_by", context=context),
        )
        if entry.dominated and not entry.dominated_by:
            raise CompareReportValidationError(
                f"{context} is marked dominated but names nothing that dominates it"
            )
        if not entry.dominated and entry.dominated_by:
            raise CompareReportValidationError(
                f"{context} is not marked dominated yet names dominating systems"
            )
        return entry


#: Which reported metric each objective is defined as. A stratum that claims
#: an ``objective_value`` disagreeing with the metric it says it ranked on is
#: internally false, and a reader has no way to tell which of the two numbers
#: is the real one.
#: The path is read from the ``SystemReport`` itself, or from its nested
#: ``cost`` summary for the two monetary objectives, both of which are
#: ordinary reported columns rather than something derived at render time.
_OBJECTIVE_BACKING_METRIC: dict[str, tuple[str, ...]] = {
    "min_mean_total_latency_ms": ("mean_total_latency_ms",),
    "max_correct_cases_per_minute": ("correct_cases_per_minute",),
    "min_cost_per_correct_case": ("cost", "cost_per_correct_case"),
    "max_correct_cases_per_currency_unit": (
        "cost",
        "correct_cases_per_currency_unit",
    ),
}


def _resolve_metric(system: SystemReport, path: tuple[str, ...]) -> float | None:
    """Follow a backing-metric path, giving up on any missing hop."""
    current: Any = system
    for step in path:
        current = getattr(current, step, None)
        if current is None:
            return None
    return current if isinstance(current, (int, float)) else None


#: Objectives whose best value is the smallest one.
_OBJECTIVES_PREFERRING_LOWER = frozenset(
    {"min_mean_total_latency_ms", "min_cost_per_correct_case"}
)


def _reject_false_ranking(
    ranked: tuple[SystemReport, ...], *, objective_name: str, context: str
) -> None:
    """Refuse a ranking that contradicts the evidence printed beside it.

    Rank order, the objective value and the metric the objective is defined
    as are three statements about the same fact, so a file can assert all
    three and have them disagree. That is not a schema error, which is
    exactly why it needs checking: such a report renders cleanly and reads
    as authoritative while recommending the system that lost.

    Every objective this layer supports is defined as a column reported
    beside it, including the two monetary ones, which read from the nested
    cost summary rather than from the top level of the row. All four are
    checked; a system with no cost summary simply has nothing to check
    against and is skipped.
    """
    backing = _OBJECTIVE_BACKING_METRIC.get(objective_name)
    if backing is not None:
        for system in ranked:
            claimed = system.objective_value
            actual = _resolve_metric(system, backing)
            if claimed is None or actual is None:
                continue
            if not math.isclose(claimed, actual, rel_tol=1e-9, abs_tol=1e-9):
                raise CompareReportValidationError(
                    f"{context}.ranked entry at rank {system.rank} claims "
                    f"objective_value {claimed!r} for objective "
                    f"{objective_name!r}, but its "
                    f"{'.'.join(backing)} is {actual!r}; the ranking "
                    "contradicts the evidence reported beside it"
                )

    values = [
        (system.rank, system.objective_value)
        for system in ranked
        if system.objective_value is not None
    ]
    prefers_lower = objective_name in _OBJECTIVES_PREFERRING_LOWER
    for (rank, value), (next_rank, next_value) in zip(values, values[1:], strict=False):
        better = value <= next_value if prefers_lower else value >= next_value
        if not better:
            direction = "lowest" if prefers_lower else "highest"
            raise CompareReportValidationError(
                f"{context}.ranked is not ordered by {objective_name!r}: rank "
                f"{rank} has objective_value {value!r} but rank {next_rank} "
                f"has {next_value!r}, and this objective prefers the "
                f"{direction} value"
            )


@dataclass(frozen=True)
class StratumReport:
    """One comparable unit: its systems, its ranking, and its frontier."""

    unit_key: ComparableUnitKey
    outcome: StratumOutcome
    objective_name: str
    ranked: tuple[SystemReport, ...] = field(default_factory=tuple)
    rejected: tuple[RejectedSystemReport, ...] = field(default_factory=tuple)
    recommended: SystemReport | None = None
    inconclusive_reason: str | None = None
    frontier_axes: tuple[ParetoAxis, ...] = field(default_factory=tuple)
    frontier: tuple[FrontierEntry, ...] = field(default_factory=tuple)
    missing_evidence: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparable_unit_key": self.unit_key.to_dict(),
            "comparable_unit_label": self.unit_key.label(),
            "outcome": self.outcome.value,
            "objective_name": self.objective_name,
            "recommended": (
                None if self.recommended is None else self.recommended.to_dict()
            ),
            "inconclusive_reason": self.inconclusive_reason,
            "ranked": [system.to_dict() for system in self.ranked],
            "rejected": [system.to_dict() for system in self.rejected],
            "frontier_axes": [axis.value for axis in self.frontier_axes],
            "frontier": [entry.to_dict() for entry in self.frontier],
            "missing_evidence": list(self.missing_evidence),
        }

    @classmethod
    def from_dict(cls, data: Any) -> StratumReport:
        if not isinstance(data, dict):
            raise CompareReportValidationError("stratum report must be a JSON object")
        context = "stratum report"

        try:
            unit_key = ComparableUnitKey.from_dict(
                _require(data, "comparable_unit_key", context=context)
            )
        except CompareIdentityError as exc:
            raise CompareReportValidationError(
                f"{context}.comparable_unit_key is invalid: {exc}"
            ) from exc

        try:
            outcome = StratumOutcome(_require_str(data, "outcome", context=context))
        except ValueError as exc:
            raise CompareReportValidationError(
                f"{context}.outcome has an invalid value: {exc}"
            ) from exc

        ranked_raw = data.get("ranked", [])
        if not isinstance(ranked_raw, list):
            raise CompareReportValidationError(f"{context}.ranked must be a list")
        ranked = tuple(
            sorted(
                (SystemReport.from_dict(item) for item in ranked_raw),
                key=lambda system: system.rank,
            )
        )
        ranks = tuple(system.rank for system in ranked)
        if ranks != tuple(range(1, len(ranked) + 1)):
            raise CompareReportValidationError(
                f"{context}.ranked ranks must be unique and contiguous starting "
                f"at 1, got {ranks!r}"
            )

        objective_name = _require_str(data, "objective_name", context=context)
        _reject_false_ranking(ranked, objective_name=objective_name, context=context)

        rejected_raw = data.get("rejected", [])
        if not isinstance(rejected_raw, list):
            raise CompareReportValidationError(f"{context}.rejected must be a list")

        axes_raw = data.get("frontier_axes", [])
        if not isinstance(axes_raw, list):
            raise CompareReportValidationError(
                f"{context}.frontier_axes must be a list"
            )
        try:
            frontier_axes = tuple(ParetoAxis(value) for value in axes_raw)
        except ValueError as exc:
            raise CompareReportValidationError(
                f"{context}.frontier_axes contains an unknown axis: {exc}"
            ) from exc

        frontier_raw = data.get("frontier", [])
        if not isinstance(frontier_raw, list):
            raise CompareReportValidationError(f"{context}.frontier must be a list")
        if frontier_raw and not frontier_axes:
            raise CompareReportValidationError(
                f"{context} places systems on a frontier without naming the axes "
                "the frontier was computed on"
            )

        recommended_raw = data.get("recommended")
        recommended = (
            None if recommended_raw is None else SystemReport.from_dict(recommended_raw)
        )

        stratum = cls(
            unit_key=unit_key,
            outcome=outcome,
            objective_name=_require_str(data, "objective_name", context=context),
            ranked=ranked,
            rejected=tuple(
                RejectedSystemReport.from_dict(item) for item in rejected_raw
            ),
            recommended=recommended,
            inconclusive_reason=_optional_str(
                data, "inconclusive_reason", context=context
            ),
            frontier_axes=frontier_axes,
            frontier=tuple(FrontierEntry.from_dict(item) for item in frontier_raw),
            missing_evidence=_string_tuple(data, "missing_evidence", context=context),
        )

        if stratum.outcome == StratumOutcome.RECOMMENDED:
            if stratum.recommended is None:
                raise CompareReportValidationError(
                    f"{context} has outcome 'recommended' but 'recommended' is null"
                )
            if not stratum.ranked:
                raise CompareReportValidationError(
                    f"{context} has outcome 'recommended' but 'ranked' is empty"
                )
            if stratum.recommended != stratum.ranked[0]:
                raise CompareReportValidationError(
                    f"{context}.recommended must equal the rank 1 ranked system"
                )
        else:
            if stratum.recommended is not None:
                raise CompareReportValidationError(
                    f"{context} has outcome 'inconclusive' but recommended is set"
                )
            if stratum.inconclusive_reason is None:
                raise CompareReportValidationError(
                    f"{context} has outcome 'inconclusive' but "
                    "'inconclusive_reason' is null"
                )

        objective_names = {system.objective_name for system in stratum.ranked}
        if objective_names and objective_names != {stratum.objective_name}:
            raise CompareReportValidationError(
                f"{context}.ranked mixes objectives {sorted(objective_names)!r} "
                f"but the stratum declares {stratum.objective_name!r}; a single "
                "ranking is only ever produced for one objective"
            )
        return stratum


@dataclass(frozen=True)
class PricingProvenance:
    """Exactly which pricing input produced every monetary value in a report."""

    manifest_path: str
    manifest_sha256: str
    currency: str
    rates_are_illustrative: bool
    entry_ids_used: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "currency": self.currency,
            "rates_are_illustrative": self.rates_are_illustrative,
            "entry_ids_used": list(self.entry_ids_used),
        }

    @classmethod
    def from_dict(cls, data: Any) -> PricingProvenance:
        if not isinstance(data, dict):
            raise CompareReportValidationError("pricing must be a JSON object")
        context = "pricing"
        return cls(
            manifest_path=_require_str(data, "manifest_path", context=context),
            manifest_sha256=_require_str(data, "manifest_sha256", context=context),
            currency=_require_currency(data, "currency", context=context),
            rates_are_illustrative=_require_bool(
                data, "rates_are_illustrative", context=context
            ),
            entry_ids_used=_string_tuple(data, "entry_ids_used", context=context),
        )


def _reject_constraint_violating_rankings(
    strata: tuple[StratumReport, ...], constraints: CompareConstraints
) -> None:
    """Refuse a ranked system that fails the constraints stored beside it.

    A report carries both the bar and the systems said to have cleared it,
    which is two claims about one fact. ``compare()`` rejects a system that
    misses the bar, so a loaded report showing a *ranked* system below it did
    not come from an honest run: either the constraints were edited after the
    fact to look demanding, or the ranking was. Both read as a system having
    earned a recommendation it did not, which is the failure this whole layer
    exists to avoid.

    A ``null`` metric is refused rather than skipped wherever ``compare()``
    provably cannot emit one. For ``min_pass_rate``, ``min_quality_score``,
    ``required_quality_metric`` and ``max_cost_per_correct_case`` the engine
    rejects a system whose value is missing, so ``null`` on a *ranked* system
    is not absent evidence, it is a report no honest run produced -- and
    nulling the field was otherwise the simplest way to evade the check.

    ``coefficient_of_variation`` is the genuine exception: it is legitimately
    ``None`` when the mean is not positive, with no rejection, so a missing
    value there is skipped.
    """
    skip_when_missing = {"coefficient_of_variation"}
    for stratum in strata:
        for system in stratum.ranked:
            checks: tuple[tuple[str, float | None, str, float | None], ...] = (
                ("pass_rate", system.pass_rate, ">=", constraints.min_pass_rate),
                (
                    "mean_quality_score",
                    system.mean_quality_score,
                    ">=",
                    constraints.min_quality_score,
                ),
                (
                    "mean_total_latency_ms",
                    system.mean_total_latency_ms,
                    "<=",
                    constraints.max_mean_total_latency_ms,
                ),
                (
                    "coefficient_of_variation",
                    system.coefficient_of_variation,
                    "<=",
                    constraints.max_coefficient_of_variation,
                ),
                (
                    "cost.cost_per_correct_case",
                    None if system.cost is None else system.cost.cost_per_correct_case,
                    "<=",
                    constraints.max_cost_per_correct_case,
                ),
            )
            for name, value, direction, bound in checks:
                if bound is None:
                    continue
                if value is None:
                    if name in skip_when_missing:
                        continue
                    raise CompareReportValidationError(
                        f"stratum report ranked entry at rank {system.rank} "
                        f"reports no {name}, but this report's own policy "
                        f"constrains it ({name} {direction} {bound!r}); a "
                        "system with no such evidence cannot be ranked as "
                        "having cleared that bar"
                    )
                violated = value < bound if direction == ">=" else value > bound
                if violated:
                    raise CompareReportValidationError(
                        f"stratum report ranked entry at rank {system.rank} "
                        f"reports {name} {value!r}, which does not satisfy the "
                        f"constraint {name} {direction} {bound!r} recorded in "
                        "this report's own policy; a system that misses the "
                        "bar cannot also be ranked as having cleared it"
                    )
            # The metric a score was earned on, not just the score. The
            # policy already refuses ``min_quality_score`` without
            # ``required_quality_metric`` precisely so a score is never
            # compared against a mismatched metric; re-checking the score bar
            # here while skipping the metric would split that pair back apart
            # and let a report assert a bar its evidence was never graded
            # against.
            if (
                constraints.required_quality_metric is not None
                and system.quality_metric != constraints.required_quality_metric
            ):
                raise CompareReportValidationError(
                    f"stratum report ranked entry at rank {system.rank} was "
                    f"graded by {system.quality_metric!r}, but this report's "
                    "own policy requires "
                    f"{constraints.required_quality_metric!r}; a system "
                    "measured by a different evaluator cannot be ranked as "
                    "having met that bar"
                )
            if system.evidence_count < constraints.min_measured_repetitions:
                raise CompareReportValidationError(
                    f"stratum report ranked entry at rank {system.rank} is "
                    f"backed by {system.evidence_count} run(s) but this "
                    "report's own policy requires at least "
                    f"{constraints.min_measured_repetitions}"
                )


@dataclass(frozen=True)
class CompareReport:
    """The complete output of one ``compare`` invocation."""

    schema_version: str
    generated_at: str
    results_dirs: tuple[str, ...]
    policy: ComparePolicy
    strata: tuple[StratumReport, ...] = field(default_factory=tuple)
    tune_report_paths: tuple[str, ...] = field(default_factory=tuple)
    pricing: PricingProvenance | None = None
    excluded_runs: tuple[ExcludedRun, ...] = field(default_factory=tuple)

    @property
    def has_recommendation(self) -> bool:
        return any(
            stratum.outcome == StratumOutcome.RECOMMENDED for stratum in self.strata
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "results_dirs": list(self.results_dirs),
            "tune_report_paths": list(self.tune_report_paths),
            "policy": self.policy.to_dict(),
            "pricing": None if self.pricing is None else self.pricing.to_dict(),
            "strata": [stratum.to_dict() for stratum in self.strata],
            "excluded_runs": [run.to_dict() for run in self.excluded_runs],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Any) -> CompareReport:
        if not isinstance(data, dict):
            raise CompareReportValidationError("compare report must be a JSON object")
        context = "compare report"

        schema_version = str(data.get("schema_version", COMPARE_REPORT_SCHEMA_VERSION))
        if schema_version != COMPARE_REPORT_SCHEMA_VERSION:
            raise CompareReportValidationError(
                f"unsupported compare report schema_version {schema_version!r}, "
                f"expected {COMPARE_REPORT_SCHEMA_VERSION!r}"
            )

        try:
            policy = ComparePolicy.from_dict(_require(data, "policy", context=context))
        except ComparePolicyError as exc:
            raise CompareReportValidationError(
                f"{context}.policy is invalid: {exc}"
            ) from exc

        strata_raw = data.get("strata", [])
        if not isinstance(strata_raw, list):
            raise CompareReportValidationError(f"{context}.strata must be a list")
        strata = tuple(StratumReport.from_dict(item) for item in strata_raw)
        _reject_constraint_violating_rankings(strata, policy.constraints)

        excluded_raw = data.get("excluded_runs", [])
        if not isinstance(excluded_raw, list):
            raise CompareReportValidationError(
                f"{context}.excluded_runs must be a list"
            )
        try:
            excluded_runs = tuple(ExcludedRun.from_dict(item) for item in excluded_raw)
        except TuneInputError as exc:
            raise CompareReportValidationError(
                f"{context}.excluded_runs is invalid: {exc}"
            ) from exc

        pricing_raw = data.get("pricing")
        pricing = (
            None if pricing_raw is None else PricingProvenance.from_dict(pricing_raw)
        )

        report = cls(
            schema_version=schema_version,
            generated_at=_require_str(data, "generated_at", context=context),
            results_dirs=_string_tuple(data, "results_dirs", context=context),
            tune_report_paths=_string_tuple(data, "tune_report_paths", context=context),
            policy=policy,
            strata=strata,
            pricing=pricing,
            excluded_runs=excluded_runs,
        )

        declared_objective = report.policy.objective.value
        for stratum in report.strata:
            if stratum.objective_name != declared_objective:
                raise CompareReportValidationError(
                    f"{context}.strata declares objective "
                    f"{stratum.objective_name!r} but the policy objective is "
                    f"{declared_objective!r}; one compare run ranks on exactly "
                    "one objective"
                )

        if report.pricing is None:
            for stratum in report.strata:
                for system in stratum.ranked:
                    if system.cost is not None:
                        raise CompareReportValidationError(
                            f"{context} carries cost figures but records no "
                            "pricing provenance; a monetary value with no "
                            "manifest behind it is unattributable"
                        )
        else:
            currency = report.pricing.currency
            for stratum in report.strata:
                for system in stratum.ranked:
                    if system.cost is not None and system.cost.currency != currency:
                        raise CompareReportValidationError(
                            f"{context} mixes currencies: system cost is in "
                            f"{system.cost.currency!r} but the pricing manifest "
                            f"declares {currency!r}"
                        )

        if report.policy.objective.requires_cost and report.pricing is None:
            raise CompareReportValidationError(
                f"{context}.policy objective {declared_objective!r} ranks on "
                "money but the report records no pricing provenance"
            )
        return report

    @classmethod
    def from_json(cls, payload: str) -> CompareReport:
        try:
            data = json.loads(payload)
        # ``json`` raises past its own limits with exceptions that are
        # not ``JSONDecodeError``: an integer literal over the
        # interpreter's digit cap raises a plain ``ValueError``, and deep
        # nesting raises ``RecursionError``. Neither is caught by any
        # caller, so both used to escape as a traceback from a merely
        # malformed file.
        except (ValueError, RecursionError) as exc:
            raise CompareReportValidationError(
                f"invalid JSON for compare report: {exc}"
            ) from exc
        return cls.from_dict(data)

    @classmethod
    def read_json(cls, path: str | Path) -> CompareReport:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "COMPARE_REPORT_SCHEMA_VERSION",
    "CompareObjective",
    "CompareReport",
    "CompareReportValidationError",
    "CostSummary",
    "FrontierEntry",
    "ParetoAxis",
    "PricingProvenance",
    "RejectedSystemReport",
    "StratumOutcome",
    "StratumReport",
    "SystemReport",
    "TtftBasis",
    "UsageTotals",
]
