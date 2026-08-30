"""Typed, strictly-validated policy for one ``compare`` run.

Shares the tuner's discipline verbatim: exactly one ranking objective per
run, and a constraint set a system must clear *before* it is ranked at all.
There is no blended score anywhere in this module, because a single number
mixing latency, quality and price hides which axis a recommendation actually
came from, and that is precisely the axis a reader needs.

The constraint axes that also exist in ``tune.policy.TuneConstraints`` keep
exactly the same meaning here, and the shared vocabulary (``RowStatus``,
``ALLOWED_REQUIRED_STATUSES``, ``MetricProvenance``) is imported from there
rather than restated. The additions are the two things a cross-system
comparison can constrain and a single-machine tuning run cannot: money, and
the requirement that a priced comparison actually be priced.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from ..schema import MetricProvenance
from ..tune.policy import (
    ALLOWED_REQUIRED_STATUSES,
    DEFAULT_REQUIRED_STATUSES,
)
from ..workloads.verify import RowStatus

COMPARE_POLICY_SCHEMA_VERSION = "1"


class ComparePolicyError(ValueError):
    """Raised when a comparison policy is invalid or malformed."""


class CompareObjective(str, Enum):
    """The single ranking objective for one comparison run."""

    MIN_MEAN_TOTAL_LATENCY_MS = "min_mean_total_latency_ms"
    """Prefer the system with the lowest mean end-to-end ``timing.total``."""

    MAX_CORRECT_CASES_PER_MINUTE = "max_correct_cases_per_minute"
    """Prefer the highest passing-cases-per-minute throughput, using
    ``workloads.aggregate.correct_cases_per_minute`` unchanged."""

    MIN_COST_PER_CORRECT_CASE = "min_cost_per_correct_case"
    """Prefer the lowest estimated spend per passing case. Requires a
    pricing manifest and provider-reported usage for every ranked system."""

    MAX_CORRECT_CASES_PER_CURRENCY_UNIT = "max_correct_cases_per_currency_unit"
    """Prefer the most passing cases bought per unit of currency. Same
    evidence requirement as ``min_cost_per_correct_case``."""

    @property
    def prefers_lower(self) -> bool:
        return self in (
            CompareObjective.MIN_MEAN_TOTAL_LATENCY_MS,
            CompareObjective.MIN_COST_PER_CORRECT_CASE,
        )

    @property
    def requires_cost(self) -> bool:
        return self in (
            CompareObjective.MIN_COST_PER_CORRECT_CASE,
            CompareObjective.MAX_CORRECT_CASES_PER_CURRENCY_UNIT,
        )


def _optional_positive_float(
    data: dict[str, Any], key: str, *, context: str
) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparePolicyError(f"{context}.{key} must be a number, got {value!r}")
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
        raise ComparePolicyError(
            f"{context}.{key} is too large to represent as a number: {exc}"
        ) from exc
    if not math.isfinite(numeric) or numeric <= 0:
        raise ComparePolicyError(f"{context}.{key} must be > 0, got {numeric!r}")
    return numeric


def _optional_non_negative_float(
    data: dict[str, Any], key: str, *, context: str
) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparePolicyError(f"{context}.{key} must be a number, got {value!r}")
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
        raise ComparePolicyError(
            f"{context}.{key} is too large to represent as a number: {exc}"
        ) from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise ComparePolicyError(f"{context}.{key} must be >= 0, got {numeric!r}")
    return numeric


def _optional_unit_interval(
    data: dict[str, Any], key: str, *, context: str
) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ComparePolicyError(f"{context}.{key} must be a number, got {value!r}")
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
        raise ComparePolicyError(
            f"{context}.{key} is too large to represent as a number: {exc}"
        ) from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ComparePolicyError(
            f"{context}.{key} must be within [0, 1], got {numeric!r}"
        )
    return numeric


#: Keys a compare policy may carry. Anything else is refused rather than
#: ignored: a policy is the operator's statement of what bar they intended to
#: apply, so silently dropping a misspelled ``max_mean_latency_ms`` publishes
#: a comparison that cleared a constraint nobody actually set, and reads as
#: though the bar had been enforced.
_POLICY_KEYS = frozenset(
    {
        "schema_version",
        "name",
        "description",
        "objective",
        "constraints",
    }
)

#: Same rule, one level down.
_CONSTRAINT_KEYS = frozenset(
    {
        "required_statuses",
        "allowed_provenances",
        "min_pass_rate",
        "min_quality_score",
        "required_quality_metric",
        "max_mean_total_latency_ms",
        "max_cost_per_correct_case",
        "min_measured_repetitions",
        "max_coefficient_of_variation",
    }
)


def _reject_unknown_keys(
    data: dict[str, Any], allowed: frozenset[str], *, context: str
) -> None:
    """Refuse keys this schema does not define."""
    unknown = sorted(key for key in data if key not in allowed)
    if unknown:
        raise ComparePolicyError(
            f"{context} has unknown field(s) {unknown}; allowed fields are "
            f"{sorted(allowed)}. A misspelled constraint would otherwise be "
            "silently ignored and the comparison would report clearing a bar "
            "that was never applied"
        )


@dataclass(frozen=True)
class CompareConstraints:
    """Everything a system must satisfy before it is ranked at all."""

    required_statuses: frozenset[RowStatus] = field(
        default_factory=lambda: DEFAULT_REQUIRED_STATUSES
    )
    min_pass_rate: float | None = None
    min_quality_score: float | None = None
    required_quality_metric: str | None = None
    max_mean_total_latency_ms: float | None = None
    max_cost_per_correct_case: float | None = None
    allowed_provenances: frozenset[MetricProvenance] | None = None
    min_measured_repetitions: int = 1
    max_coefficient_of_variation: float | None = None

    def __post_init__(self) -> None:
        if not self.required_statuses:
            raise ComparePolicyError("constraints.required_statuses must be non-empty")
        disallowed = self.required_statuses - ALLOWED_REQUIRED_STATUSES
        if disallowed:
            raise ComparePolicyError(
                "constraints.required_statuses may only contain "
                f"{sorted(s.value for s in ALLOWED_REQUIRED_STATUSES)}, got "
                f"disallowed value(s) {sorted(s.value for s in disallowed)}"
            )
        if self.min_quality_score is not None and self.required_quality_metric is None:
            raise ComparePolicyError(
                "constraints.required_quality_metric must be set when "
                "constraints.min_quality_score is configured, so a system's "
                "quality_score is never compared against a mismatched metric"
            )
        if isinstance(self.min_measured_repetitions, bool):
            raise ComparePolicyError(
                "constraints.min_measured_repetitions must be an integer, got "
                f"{self.min_measured_repetitions!r}"
            )
        if self.min_measured_repetitions < 1:
            raise ComparePolicyError(
                "constraints.min_measured_repetitions must be >= 1"
            )
        for name, value in (
            ("min_pass_rate", self.min_pass_rate),
            ("min_quality_score", self.min_quality_score),
            ("max_mean_total_latency_ms", self.max_mean_total_latency_ms),
            ("max_cost_per_correct_case", self.max_cost_per_correct_case),
            ("max_coefficient_of_variation", self.max_coefficient_of_variation),
        ):
            if value is not None and not math.isfinite(value):
                raise ComparePolicyError(
                    f"constraints.{name} must be a finite number, got {value!r}"
                )
        # Range rules, enforced here rather than only while parsing, because
        # this dataclass is public: a constraint constructed directly with an
        # unsatisfiable or meaningless bound would silently exclude every
        # system, or admit every system, and be reported as a real threshold.
        for name, value in (
            ("min_pass_rate", self.min_pass_rate),
            ("min_quality_score", self.min_quality_score),
            ("max_coefficient_of_variation", self.max_coefficient_of_variation),
        ):
            if value is not None and value < 0:
                raise ComparePolicyError(
                    f"constraints.{name} must be >= 0, got {value!r}"
                )
        # A ceiling of exactly zero is not a strict bar, it is an
        # unsatisfiable one: nothing costs nothing or takes no time, so it
        # rejects every system while ``active_summary`` renders it as a
        # deliberate threshold. The parser already refuses it; so does this.
        for name, value in (
            ("max_mean_total_latency_ms", self.max_mean_total_latency_ms),
            ("max_cost_per_correct_case", self.max_cost_per_correct_case),
        ):
            if value is not None and value <= 0:
                raise ComparePolicyError(
                    f"constraints.{name} must be > 0, got {value!r}"
                )
        if self.min_pass_rate is not None and self.min_pass_rate > 1:
            raise ComparePolicyError(
                "constraints.min_pass_rate is a fraction between 0 and 1, got "
                f"{self.min_pass_rate!r}"
            )

    def active_summary(self) -> tuple[str, ...]:
        """Every constraint actually in force, as readable phrases.

        A recommendation is only meaningful together with the bar a system
        had to clear to earn it, and the README promises each one is stated
        with the constraints it applies to. Reporting "cleared every
        constraint" without saying which constraints those were leaves a
        reader unable to tell a demanding comparison from a vacuous one.

        Only constraints that are set are listed. An unset constraint
        excludes nothing, so naming it would overstate the bar.
        """
        parts: list[str] = [
            "required run status in "
            + ", ".join(sorted(status.value for status in self.required_statuses))
        ]
        if self.allowed_provenances:
            parts.append(
                "timing provenance in "
                + ", ".join(sorted(item.value for item in self.allowed_provenances))
            )
        if self.min_pass_rate is not None:
            parts.append(f"pass rate >= {self.min_pass_rate}")
        if self.min_quality_score is not None:
            parts.append(
                f"quality score >= {self.min_quality_score} "
                f"on {self.required_quality_metric}"
            )
        elif self.required_quality_metric is not None:
            parts.append(f"quality metric is {self.required_quality_metric}")
        if self.max_mean_total_latency_ms is not None:
            parts.append(f"mean total latency <= {self.max_mean_total_latency_ms} ms")
        if self.max_cost_per_correct_case is not None:
            parts.append(f"cost per correct case <= {self.max_cost_per_correct_case}")
        if self.max_coefficient_of_variation is not None:
            parts.append(
                "coefficient of variation <= " f"{self.max_coefficient_of_variation}"
            )
        parts.append(f"at least {self.min_measured_repetitions} measured run(s)")
        return tuple(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_statuses": sorted(s.value for s in self.required_statuses),
            "min_pass_rate": self.min_pass_rate,
            "min_quality_score": self.min_quality_score,
            "required_quality_metric": self.required_quality_metric,
            "max_mean_total_latency_ms": self.max_mean_total_latency_ms,
            "max_cost_per_correct_case": self.max_cost_per_correct_case,
            "allowed_provenances": (
                None
                if self.allowed_provenances is None
                else sorted(p.value for p in self.allowed_provenances)
            ),
            "min_measured_repetitions": self.min_measured_repetitions,
            "max_coefficient_of_variation": self.max_coefficient_of_variation,
        }

    @classmethod
    def from_dict(cls, data: Any) -> CompareConstraints:
        if not isinstance(data, dict):
            raise ComparePolicyError("compare policy 'constraints' must be an object")
        context = "constraints"
        _reject_unknown_keys(data, _CONSTRAINT_KEYS, context="compare constraints")

        raw_statuses = data.get(
            "required_statuses", [s.value for s in DEFAULT_REQUIRED_STATUSES]
        )
        if not isinstance(raw_statuses, list) or not raw_statuses:
            raise ComparePolicyError(
                f"{context}.required_statuses must be a non-empty list of strings"
            )
        try:
            required_statuses = frozenset(RowStatus(value) for value in raw_statuses)
        except ValueError as exc:
            raise ComparePolicyError(
                f"{context}.required_statuses contains an unknown status: {exc}"
            ) from exc

        raw_provenances = data.get("allowed_provenances")
        allowed_provenances: frozenset[MetricProvenance] | None
        if raw_provenances is None:
            allowed_provenances = None
        else:
            if not isinstance(raw_provenances, list) or not raw_provenances:
                raise ComparePolicyError(
                    f"{context}.allowed_provenances must be a non-empty list of "
                    "strings, or omitted/null to allow any provenance"
                )
            try:
                allowed_provenances = frozenset(
                    MetricProvenance(value) for value in raw_provenances
                )
            except ValueError as exc:
                raise ComparePolicyError(
                    f"{context}.allowed_provenances contains an unknown "
                    f"provenance: {exc}"
                ) from exc

        required_quality_metric = data.get("required_quality_metric")
        if required_quality_metric is not None and not isinstance(
            required_quality_metric, str
        ):
            raise ComparePolicyError(
                f"{context}.required_quality_metric must be a string or null"
            )

        raw_repetitions = data.get("min_measured_repetitions", 1)
        if isinstance(raw_repetitions, bool) or not isinstance(raw_repetitions, int):
            raise ComparePolicyError(
                f"{context}.min_measured_repetitions must be an integer"
            )

        return cls(
            required_statuses=required_statuses,
            min_pass_rate=_optional_unit_interval(
                data, "min_pass_rate", context=context
            ),
            min_quality_score=_optional_non_negative_float(
                data, "min_quality_score", context=context
            ),
            required_quality_metric=required_quality_metric,
            max_mean_total_latency_ms=_optional_positive_float(
                data, "max_mean_total_latency_ms", context=context
            ),
            max_cost_per_correct_case=_optional_positive_float(
                data, "max_cost_per_correct_case", context=context
            ),
            allowed_provenances=allowed_provenances,
            min_measured_repetitions=int(raw_repetitions),
            max_coefficient_of_variation=_optional_non_negative_float(
                data, "max_coefficient_of_variation", context=context
            ),
        )


@dataclass(frozen=True)
class ComparePolicy:
    """A complete, validated comparison policy: one objective plus constraints."""

    objective: CompareObjective
    constraints: CompareConstraints = field(default_factory=CompareConstraints)
    schema_version: str = COMPARE_POLICY_SCHEMA_VERSION
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
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Any) -> ComparePolicy:
        if not isinstance(data, dict):
            raise ComparePolicyError("compare policy must be a JSON/YAML object")
        _reject_unknown_keys(data, _POLICY_KEYS, context="compare policy")
        try:
            objective = CompareObjective(data["objective"])
        except KeyError as exc:
            raise ComparePolicyError(
                "compare policy is missing required field: 'objective'"
            ) from exc
        except ValueError as exc:
            raise ComparePolicyError(
                f"compare policy has an invalid objective: {exc}"
            ) from exc

        name = data.get("name")
        if name is not None and not isinstance(name, str):
            raise ComparePolicyError("compare policy 'name' must be a string or null")
        description = data.get("description")
        if description is not None and not isinstance(description, str):
            raise ComparePolicyError(
                "compare policy 'description' must be a string or null"
            )

        schema_version = str(data.get("schema_version", COMPARE_POLICY_SCHEMA_VERSION))
        if schema_version != COMPARE_POLICY_SCHEMA_VERSION:
            raise ComparePolicyError(
                f"unsupported compare policy schema_version {schema_version!r}, "
                f"expected {COMPARE_POLICY_SCHEMA_VERSION!r}"
            )

        return cls(
            schema_version=schema_version,
            name=name,
            description=description,
            objective=objective,
            constraints=CompareConstraints.from_dict(data.get("constraints", {})),
        )

    @classmethod
    def from_json(cls, payload: str) -> ComparePolicy:
        try:
            data = json.loads(payload)
        # ``json`` raises past its own limits with exceptions that are
        # not ``JSONDecodeError``: an integer literal over the
        # interpreter's digit cap raises a plain ``ValueError``, and deep
        # nesting raises ``RecursionError``. Neither is caught by any
        # caller, so both used to escape as a traceback from a merely
        # malformed file.
        except (ValueError, RecursionError) as exc:
            raise ComparePolicyError(f"invalid JSON for compare policy: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def from_file(cls, path: str | Path) -> ComparePolicy:
        """Load a policy from ``.json`` or ``.yaml``/``.yml``.

        Mirrors ``TunePolicy.from_file``'s extension dispatch and its
        explicit failure on an unsupported extension or missing PyYAML.
        """
        policy_path = Path(path)
        text = policy_path.read_text(encoding="utf-8")
        suffix = policy_path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ComparePolicyError(
                    "YAML compare policy requires PyYAML to be installed "
                    "(`uv add pyyaml`); use a .json policy instead if it is "
                    "unavailable"
                ) from exc
            try:
                data = yaml.safe_load(text)
            # ``yaml.YAMLError`` subclasses ``Exception``, not
            # ``ValueError``, so neither this loader nor the CLI
            # caught it and a merely malformed YAML file escaped as a
            # traceback. YAML is an advertised input format for this
            # flag, and it is far easier to malform than JSON.
            except (yaml.YAMLError, ValueError, RecursionError) as exc:
                raise ComparePolicyError(
                    f"invalid YAML in {policy_path}: {exc}"
                ) from exc
        elif suffix == ".json":
            try:
                data = json.loads(text)
            except (ValueError, RecursionError) as exc:
                raise ComparePolicyError(
                    f"invalid JSON in {policy_path}: {exc}"
                ) from exc
        else:
            raise ComparePolicyError(
                f"unsupported compare policy extension {suffix!r} "
                "(use .json or .yaml)"
            )
        if not isinstance(data, dict):
            raise ComparePolicyError(
                f"compare policy in {policy_path} must be a JSON/YAML object"
            )
        return cls.from_dict(data)
