"""Machine-readable orchestration summary for the ``optimize`` command.

``optimize`` composes four already-existing, independently tested phases
(row planning/execution via ``workloads.verify``, offline tuning via
``tune.tuner``, and report rendering via ``tune.report``/
``tune.report_html``) into one invocation. This module defines the schema
for the single artifact that records what actually happened across all of
them, so a caller never has to guess whether a given JSON/HTML report on
disk reflects a fully successful run, a partially failed one, or stale
data from an earlier invocation.

Every phase is reported with an explicit, typed status (see
``PhaseStatus``); a phase is only ever recorded as ``OK`` once its own
artifact has been produced and re-validated. Nothing here recomputes or
duplicates the tuning/rendering logic itself -- it only records the
outcome of calling into it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

OPTIMIZE_SUMMARY_SCHEMA_VERSION = "1"


class OptimizeSummaryValidationError(ValueError):
    """Raised when an ``OptimizeSummary`` loaded from JSON is malformed."""


class PhaseName(str, Enum):
    """The fixed, ordered set of phases one ``optimize`` invocation covers."""

    PLANNED = "planned"
    """Matrix/policy loaded and row selection resolved (dry-run's scope)."""

    EXECUTED = "executed"
    """``workloads.verify.run_selected_rows`` was invoked and returned."""

    VERIFIED = "verified"
    """Whether the executed rows produced trustworthy tunable evidence."""

    TUNED = "tuned"
    """``tune.tuner.tune`` was invoked against the results directory."""

    RENDERED = "rendered"
    """The tune report JSON (and, if requested, HTML) was written."""


class PhaseStatus(str, Enum):
    """Explicit, machine-readable outcome for one orchestration phase."""

    OK = "ok"
    """The phase completed and its artifact (if any) exists and validates."""

    FAILED = "failed"
    """The phase could not complete; see the phase's ``detail``."""

    INCONCLUSIVE = "inconclusive"
    """The phase completed but produced no actionable result."""

    UNSUPPORTED = "unsupported"
    """Every selected row was rejected as unsupported (e.g. native-mtp)."""

    SKIPPED = "skipped"
    """The phase was never attempted (e.g. no ``--report-html`` given)."""

    NOT_RUN = "not_run"
    """The phase was never reached (an earlier phase stopped the run)."""


@dataclass(frozen=True)
class PhaseReport:
    """One phase's final status and an optional human-readable detail."""

    name: PhaseName
    status: PhaseStatus
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "status": self.status.value,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Any) -> PhaseReport:
        if not isinstance(data, dict):
            raise OptimizeSummaryValidationError("phase report must be a JSON object")
        context = "phase report"
        try:
            name = PhaseName(data["name"])
        except KeyError as exc:
            raise OptimizeSummaryValidationError(
                f"{context} is missing required field: 'name'"
            ) from exc
        except ValueError as exc:
            raise OptimizeSummaryValidationError(
                f"{context}.name has an invalid value: {exc}"
            ) from exc
        try:
            status = PhaseStatus(data["status"])
        except KeyError as exc:
            raise OptimizeSummaryValidationError(
                f"{context} is missing required field: 'status'"
            ) from exc
        except ValueError as exc:
            raise OptimizeSummaryValidationError(
                f"{context}.status has an invalid value: {exc}"
            ) from exc
        detail = data.get("detail")
        if detail is not None and not isinstance(detail, str):
            raise OptimizeSummaryValidationError(
                f"{context}.detail must be a string or null"
            )
        return cls(name=name, status=status, detail=detail)


@dataclass(frozen=True)
class RowStatusCounts:
    """Planning and result counts across every row selected for this run."""

    total: int = 0
    ready: int = 0
    blocked: int = 0
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    unsupported: int = 0
    inconclusive: int = 0

    def __post_init__(self) -> None:
        for key in (
            "total",
            "ready",
            "blocked",
            "completed",
            "skipped",
            "failed",
            "unsupported",
            "inconclusive",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise OptimizeSummaryValidationError(
                    f"row_counts.{key} must be a non-negative integer, "
                    f"got {value!r}"
                )

    def to_dict(self) -> dict[str, int]:
        return {
            "total": self.total,
            "ready": self.ready,
            "blocked": self.blocked,
            "completed": self.completed,
            "skipped": self.skipped,
            "failed": self.failed,
            "unsupported": self.unsupported,
            "inconclusive": self.inconclusive,
        }

    @classmethod
    def from_dict(cls, data: Any) -> RowStatusCounts:
        if not isinstance(data, dict):
            raise OptimizeSummaryValidationError("row_counts must be a JSON object")
        values: dict[str, int] = {}
        for key in (
            "total",
            "ready",
            "blocked",
            "completed",
            "skipped",
            "failed",
            "unsupported",
            "inconclusive",
        ):
            raw = data.get(key, 0)
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise OptimizeSummaryValidationError(
                    f"row_counts.{key} must be a non-negative integer, got {raw!r}"
                )
            values[key] = raw
        return cls(**values)


@dataclass(frozen=True)
class RecommendedCandidate:
    """One tuned group's winning candidate, summarized for the orchestration."""

    group_label: str
    run_ids: tuple[str, ...]
    objective_name: str
    objective_value: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.objective_value, bool)
            or not isinstance(self.objective_value, (int, float))
            or not math.isfinite(float(self.objective_value))
        ):
            raise OptimizeSummaryValidationError(
                "recommended candidate.objective_value must be a finite number"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_label": self.group_label,
            "run_ids": list(self.run_ids),
            "objective_name": self.objective_name,
            "objective_value": self.objective_value,
        }

    @classmethod
    def from_dict(cls, data: Any) -> RecommendedCandidate:
        if not isinstance(data, dict):
            raise OptimizeSummaryValidationError(
                "recommended candidate must be a JSON object"
            )
        context = "recommended candidate"
        group_label = data.get("group_label")
        if not isinstance(group_label, str) or not group_label:
            raise OptimizeSummaryValidationError(
                f"{context}.group_label must be a non-empty string"
            )
        run_ids_raw = data.get("run_ids")
        if not isinstance(run_ids_raw, list) or not all(
            isinstance(item, str) for item in run_ids_raw
        ):
            raise OptimizeSummaryValidationError(
                f"{context}.run_ids must be a list of strings"
            )
        objective_name = data.get("objective_name")
        if not isinstance(objective_name, str) or not objective_name:
            raise OptimizeSummaryValidationError(
                f"{context}.objective_name must be a non-empty string"
            )
        objective_value = data.get("objective_value")
        if (
            isinstance(objective_value, bool)
            or not isinstance(objective_value, (int, float))
            or not math.isfinite(float(objective_value))
        ):
            raise OptimizeSummaryValidationError(
                f"{context}.objective_value must be a finite number"
            )
        return cls(
            group_label=group_label,
            run_ids=tuple(run_ids_raw),
            objective_name=objective_name,
            objective_value=float(objective_value),
        )


class OverallStatus(str, Enum):
    """The single, top-level verdict for one ``optimize`` invocation."""

    SUCCESS = "success"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class OptimizeSummary:
    """The complete, atomic record of one ``optimize`` invocation."""

    schema_version: str
    generated_at: str
    dry_run: bool
    matrix_path: str
    results_dir: str | None
    policy_path: str | None
    report_json_path: str | None
    report_html_path: str | None
    phases: tuple[PhaseReport, ...]
    row_counts: RowStatusCounts
    recommendations: tuple[RecommendedCandidate, ...]
    overall_status: OverallStatus
    exit_code: int
    extra_results_dirs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise OptimizeSummaryValidationError(
                "optimize summary.exit_code must be an integer"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "dry_run": self.dry_run,
            "matrix_path": self.matrix_path,
            "results_dir": self.results_dir,
            "extra_results_dirs": list(self.extra_results_dirs),
            "policy_path": self.policy_path,
            "report_json_path": self.report_json_path,
            "report_html_path": self.report_html_path,
            "phases": [phase.to_dict() for phase in self.phases],
            "row_counts": self.row_counts.to_dict(),
            "recommendations": [rec.to_dict() for rec in self.recommendations],
            "overall_status": self.overall_status.value,
            "exit_code": self.exit_code,
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            indent=indent,
            sort_keys=False,
            allow_nan=False,
        )

    def phase(self, name: PhaseName) -> PhaseReport | None:
        for phase in self.phases:
            if phase.name == name:
                return phase
        return None

    @classmethod
    def from_dict(cls, data: Any) -> OptimizeSummary:
        if not isinstance(data, dict):
            raise OptimizeSummaryValidationError(
                "optimize summary must be a JSON object"
            )
        context = "optimize summary"

        schema_version = str(
            data.get("schema_version", OPTIMIZE_SUMMARY_SCHEMA_VERSION)
        )
        if schema_version != OPTIMIZE_SUMMARY_SCHEMA_VERSION:
            raise OptimizeSummaryValidationError(
                f"unsupported optimize summary schema_version {schema_version!r}, "
                f"expected {OPTIMIZE_SUMMARY_SCHEMA_VERSION!r}"
            )

        generated_at = data.get("generated_at")
        if not isinstance(generated_at, str) or not generated_at:
            raise OptimizeSummaryValidationError(
                f"{context}.generated_at must be a non-empty string"
            )
        matrix_path = data.get("matrix_path")
        if not isinstance(matrix_path, str) or not matrix_path:
            raise OptimizeSummaryValidationError(
                f"{context}.matrix_path must be a non-empty string"
            )
        dry_run = data.get("dry_run")
        if not isinstance(dry_run, bool):
            raise OptimizeSummaryValidationError(f"{context}.dry_run must be a boolean")

        def _optional_str(key: str) -> str | None:
            value = data.get(key)
            if value is not None and not isinstance(value, str):
                raise OptimizeSummaryValidationError(
                    f"{context}.{key} must be a string or null"
                )
            return value

        extra_results_raw = data.get("extra_results_dirs", [])
        if not isinstance(extra_results_raw, list) or not all(
            isinstance(item, str) for item in extra_results_raw
        ):
            raise OptimizeSummaryValidationError(
                f"{context}.extra_results_dirs must be a list of strings"
            )

        phases_raw = data.get("phases", [])
        if not isinstance(phases_raw, list):
            raise OptimizeSummaryValidationError(f"{context}.phases must be a list")
        phases = tuple(PhaseReport.from_dict(item) for item in phases_raw)
        phase_names = tuple(phase.name for phase in phases)
        if phase_names != tuple(PhaseName):
            raise OptimizeSummaryValidationError(
                f"{context}.phases must report exactly every phase "
                f"{[p.value for p in PhaseName]!r} in order, got "
                f"{[p.value for p in phase_names]!r}"
            )

        row_counts = RowStatusCounts.from_dict(data.get("row_counts", {}))

        recommendations_raw = data.get("recommendations", [])
        if not isinstance(recommendations_raw, list):
            raise OptimizeSummaryValidationError(
                f"{context}.recommendations must be a list"
            )
        recommendations = tuple(
            RecommendedCandidate.from_dict(item) for item in recommendations_raw
        )

        try:
            overall_status = OverallStatus(data.get("overall_status"))
        except ValueError as exc:
            raise OptimizeSummaryValidationError(
                f"{context}.overall_status has an invalid value: {exc}"
            ) from exc

        exit_code = data.get("exit_code")
        if isinstance(exit_code, bool) or not isinstance(exit_code, int):
            raise OptimizeSummaryValidationError(
                f"{context}.exit_code must be an integer"
            )

        return cls(
            schema_version=schema_version,
            generated_at=generated_at,
            dry_run=dry_run,
            matrix_path=matrix_path,
            results_dir=_optional_str("results_dir"),
            extra_results_dirs=tuple(extra_results_raw),
            policy_path=_optional_str("policy_path"),
            report_json_path=_optional_str("report_json_path"),
            report_html_path=_optional_str("report_html_path"),
            phases=phases,
            row_counts=row_counts,
            recommendations=recommendations,
            overall_status=overall_status,
            exit_code=exit_code,
        )

    @classmethod
    def from_json(cls, payload: str) -> OptimizeSummary:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise OptimizeSummaryValidationError(
                f"invalid JSON for optimize summary: {exc}"
            ) from exc
        return cls.from_dict(data)

    @classmethod
    def read_json(cls, path: str | Path) -> OptimizeSummary:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
