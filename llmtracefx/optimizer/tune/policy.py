"""Typed, strictly-validated tuning policy configuration.

A ``TunePolicy`` is user-authored (JSON or YAML) and declares, explicitly:

* the single ranking objective for this tuning run (never a blended score,
  see ``TuneObjective``), and
* every constraint a candidate configuration must satisfy to be considered
  at all (see ``TuneConstraints``).

Loading mirrors ``llmtracefx.optimizer.runner.RunnerConfig.from_file``: JSON
by default, optional YAML via PyYAML, with an explicit, typed error on any
malformed or inconsistent input. Nothing here is inferred from evidence --
a policy with no configured constraint imposes no limit on that axis (the
only always-on constraints are the required row status and a minimum of
one measured repetition).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..schema import MetricProvenance
from ..workloads.verify import RowStatus

TUNE_POLICY_SCHEMA_VERSION = "1"

#: Row statuses a constraint policy may require. Deliberately excludes
#: "failed", "unsupported", and "inconclusive" -- only rows that were
#: actually executed and evaluated (``completed``) or trusted-and-reused
#: from a hash-matching prior run (``skipped``) can ever back a
#: recommendation.
ALLOWED_REQUIRED_STATUSES: frozenset[RowStatus] = frozenset(
    {RowStatus.COMPLETED, RowStatus.SKIPPED}
)

DEFAULT_REQUIRED_STATUSES: frozenset[RowStatus] = frozenset(
    {RowStatus.COMPLETED, RowStatus.SKIPPED}
)


class TunePolicyError(ValueError):
    """Raised when a tuning policy is invalid or malformed."""


class TuneObjective(str, Enum):
    """The single ranking objective for one tuning run.

    Exactly one objective is used per run; this project never blends
    latency and throughput (or anything else) into one combined score,
    because doing so would hide which axis a recommendation actually
    reflects.
    """

    MIN_MEAN_TOTAL_LATENCY_MS = "min_mean_total_latency_ms"
    """Prefer the candidate with the lowest mean ``timing.total``."""

    MAX_CORRECT_CASES_PER_MINUTE = "max_correct_cases_per_minute"
    """Prefer the candidate with the highest passing-cases-per-minute
    throughput, using the same definition as
    ``workloads.aggregate.correct_cases_per_minute``."""


def _coerce_optional_positive_float(
    data: dict[str, Any], key: str, *, context: str
) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TunePolicyError(f"{context}.{key} must be a number, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise TunePolicyError(f"{context}.{key} must be > 0, got {numeric!r}")
    return numeric


def _coerce_optional_unit_interval(
    data: dict[str, Any], key: str, *, context: str
) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TunePolicyError(f"{context}.{key} must be a number, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or not (0.0 <= numeric <= 1.0):
        raise TunePolicyError(f"{context}.{key} must be within [0, 1], got {numeric!r}")
    return numeric


def _coerce_optional_non_negative_float(
    data: dict[str, Any], key: str, *, context: str
) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TunePolicyError(f"{context}.{key} must be a number, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise TunePolicyError(f"{context}.{key} must be >= 0, got {numeric!r}")
    return numeric


@dataclass(frozen=True)
class TuneConstraints:
    """Every explicit requirement a candidate must satisfy to be accepted.

    A field left at its default imposes no additional limit on that axis;
    the only requirements that are always active are ``required_statuses``
    (defaults to completed/trusted-skipped only) and
    ``min_measured_repetitions`` (defaults to 1).
    """

    required_statuses: frozenset[RowStatus] = field(
        default_factory=lambda: DEFAULT_REQUIRED_STATUSES
    )
    min_pass_rate: float | None = None
    min_quality_score: float | None = None
    required_quality_metric: str | None = None
    max_peak_memory_bytes: float | None = None
    max_total_latency_ms: float | None = None
    allowed_provenances: frozenset[MetricProvenance] | None = None
    min_measured_repetitions: int = 1
    max_coefficient_of_variation: float | None = None

    def __post_init__(self) -> None:
        if not self.required_statuses:
            raise TunePolicyError("constraints.required_statuses must be non-empty")
        disallowed = self.required_statuses - ALLOWED_REQUIRED_STATUSES
        if disallowed:
            raise TunePolicyError(
                "constraints.required_statuses may only contain "
                f"{sorted(s.value for s in ALLOWED_REQUIRED_STATUSES)}, "
                f"got disallowed value(s) {sorted(s.value for s in disallowed)}"
            )
        if self.min_quality_score is not None and self.required_quality_metric is None:
            raise TunePolicyError(
                "constraints.required_quality_metric must be set when "
                "constraints.min_quality_score is configured, so a candidate's "
                "quality_score is never compared against a mismatched metric"
            )
        if self.min_measured_repetitions < 1:
            raise TunePolicyError("constraints.min_measured_repetitions must be >= 1")
        # Defense in depth: every numeric field must be finite even when a
        # caller constructs ``TuneConstraints`` directly instead of going
        # through ``from_dict``/``from_file`` (which already reject
        # non-finite JSON/YAML input before this constructor ever runs). A
        # NaN/Infinity threshold here would otherwise silently disable the
        # ceiling it is supposed to enforce.
        for field_name, numeric_value in (
            ("min_pass_rate", self.min_pass_rate),
            ("min_quality_score", self.min_quality_score),
            ("max_peak_memory_bytes", self.max_peak_memory_bytes),
            ("max_total_latency_ms", self.max_total_latency_ms),
            ("max_coefficient_of_variation", self.max_coefficient_of_variation),
        ):
            if numeric_value is not None and not math.isfinite(numeric_value):
                raise TunePolicyError(
                    f"constraints.{field_name} must be a finite number, got "
                    f"{numeric_value!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_statuses": sorted(s.value for s in self.required_statuses),
            "min_pass_rate": self.min_pass_rate,
            "min_quality_score": self.min_quality_score,
            "required_quality_metric": self.required_quality_metric,
            "max_peak_memory_bytes": self.max_peak_memory_bytes,
            "max_total_latency_ms": self.max_total_latency_ms,
            "allowed_provenances": (
                None
                if self.allowed_provenances is None
                else sorted(p.value for p in self.allowed_provenances)
            ),
            "min_measured_repetitions": self.min_measured_repetitions,
            "max_coefficient_of_variation": self.max_coefficient_of_variation,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TuneConstraints:
        context = "constraints"
        raw_statuses = data.get(
            "required_statuses", [s.value for s in DEFAULT_REQUIRED_STATUSES]
        )
        if not isinstance(raw_statuses, list) or not raw_statuses:
            raise TunePolicyError(
                f"{context}.required_statuses must be a non-empty list of strings"
            )
        try:
            required_statuses = frozenset(RowStatus(value) for value in raw_statuses)
        except ValueError as exc:
            raise TunePolicyError(
                f"{context}.required_statuses contains an unknown status: {exc}"
            ) from exc

        allowed_provenances_raw = data.get("allowed_provenances")
        allowed_provenances: frozenset[MetricProvenance] | None
        if allowed_provenances_raw is None:
            allowed_provenances = None
        else:
            if not isinstance(allowed_provenances_raw, list) or not (
                allowed_provenances_raw
            ):
                raise TunePolicyError(
                    f"{context}.allowed_provenances must be a non-empty list of "
                    "strings, or omitted/null to allow any provenance"
                )
            try:
                allowed_provenances = frozenset(
                    MetricProvenance(value) for value in allowed_provenances_raw
                )
            except ValueError as exc:
                raise TunePolicyError(
                    f"{context}.allowed_provenances contains an unknown "
                    f"provenance: {exc}"
                ) from exc

        required_quality_metric = data.get("required_quality_metric")
        if required_quality_metric is not None and not isinstance(
            required_quality_metric, str
        ):
            raise TunePolicyError(
                f"{context}.required_quality_metric must be a string or null"
            )

        min_measured_repetitions_raw = data.get("min_measured_repetitions", 1)
        if isinstance(min_measured_repetitions_raw, bool) or not isinstance(
            min_measured_repetitions_raw, int
        ):
            raise TunePolicyError(
                f"{context}.min_measured_repetitions must be an integer"
            )

        return cls(
            required_statuses=required_statuses,
            min_pass_rate=_coerce_optional_unit_interval(
                data, "min_pass_rate", context=context
            ),
            min_quality_score=_coerce_optional_non_negative_float(
                data, "min_quality_score", context=context
            ),
            required_quality_metric=required_quality_metric,
            max_peak_memory_bytes=_coerce_optional_positive_float(
                data, "max_peak_memory_bytes", context=context
            ),
            max_total_latency_ms=_coerce_optional_positive_float(
                data, "max_total_latency_ms", context=context
            ),
            allowed_provenances=allowed_provenances,
            min_measured_repetitions=int(min_measured_repetitions_raw),
            max_coefficient_of_variation=_coerce_optional_non_negative_float(
                data, "max_coefficient_of_variation", context=context
            ),
        )


@dataclass(frozen=True)
class TunePolicy:
    """A complete, validated tuning policy: one objective plus constraints."""

    objective: TuneObjective
    constraints: TuneConstraints = field(default_factory=TuneConstraints)
    schema_version: str = TUNE_POLICY_SCHEMA_VERSION
    name: str | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "description": self.description,
            "objective": self.objective.value,
            "constraints": self.constraints.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TunePolicy:
        if not isinstance(data, dict):
            raise TunePolicyError("tune policy must be a JSON/YAML object")
        try:
            objective = TuneObjective(data["objective"])
        except KeyError as exc:
            raise TunePolicyError(
                "tune policy is missing required field: 'objective'"
            ) from exc
        except ValueError as exc:
            raise TunePolicyError(
                f"tune policy has an invalid objective: {exc}"
            ) from exc

        name = data.get("name")
        if name is not None and not isinstance(name, str):
            raise TunePolicyError("tune policy 'name' must be a string or null")
        description = data.get("description")
        if description is not None and not isinstance(description, str):
            raise TunePolicyError("tune policy 'description' must be a string or null")

        constraints_raw = data.get("constraints", {})
        if not isinstance(constraints_raw, dict):
            raise TunePolicyError("tune policy 'constraints' must be an object")

        return cls(
            schema_version=str(data.get("schema_version", TUNE_POLICY_SCHEMA_VERSION)),
            name=name,
            description=description,
            objective=objective,
            constraints=TuneConstraints.from_dict(constraints_raw),
        )

    @classmethod
    def from_json(cls, payload: str) -> TunePolicy:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise TunePolicyError(f"invalid JSON for tune policy: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | Path) -> TunePolicy:
        """Load a policy from a ``.json`` or ``.yaml``/``.yml`` file.

        Mirrors ``RunnerConfig.from_file``'s extension dispatch and explicit
        (never silent) failure on an unsupported extension or missing
        PyYAML dependency.
        """
        config_path = Path(path)
        text = config_path.read_text(encoding="utf-8")
        suffix = config_path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise TunePolicyError(
                    "YAML tune policy requires PyYAML to be installed "
                    "(`uv add pyyaml`); use a .json policy instead if it is "
                    "unavailable"
                ) from exc
            data = yaml.safe_load(text)
        elif suffix == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise TunePolicyError(f"invalid JSON in {config_path}: {exc}") from exc
        else:
            raise TunePolicyError(
                f"unsupported tune policy extension '{suffix}' (use .json or .yaml)"
            )

        if not isinstance(data, dict):
            raise TunePolicyError(
                f"tune policy in {config_path} must be a JSON/YAML object"
            )
        return cls.from_dict(data)
