"""Canonical experiment/evidence schema for the inference optimizer.

Every collector (llama.cpp parser, future MLX/Metal or CUDA collectors) and
every analysis rule (the "doctor") reads and writes this schema, so it is
the single source of truth for what a benchmarking run looks like. Optional
measurements stay ``None`` when they were not observed — nothing here
invents a value.

This is schema ``SCHEMA_VERSION``. Bump the constant and extend
``ExperimentRecord.from_dict`` to stay backward compatible whenever the
shape changes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1"


class SchemaValidationError(ValueError):
    """Raised when an ``ExperimentRecord`` (or a part of it) is invalid."""


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string ending in ``Z``."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


class MetricProvenance(str, Enum):
    """How a metric value was obtained.

    Every optional numeric measurement in this schema is paired with one of
    these so downstream consumers (dashboards, the doctor) can tell a
    hardware-counter measurement apart from a wall-clock timer or an
    estimate, and refuse to compare incompatible provenances.
    """

    MEASURED_NATIVE = "measured_native"
    """Read from a native profiler / hardware counter (e.g. Metal, CUPTI)."""

    MEASURED_WALL_CLOCK = "measured_wall_clock"
    """Timed on the host around a call boundary (e.g. subprocess duration)."""

    DERIVED = "derived"
    """Computed from other measured values (e.g. tokens / seconds)."""

    ESTIMATED = "estimated"
    """Modeled or approximated; not a direct measurement."""


@dataclass(frozen=True)
class Measurement:
    """A single numeric value paired with an explicit unit and provenance."""

    value: float
    provenance: MetricProvenance
    unit: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "provenance": self.provenance.value,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Measurement:
        try:
            return cls(
                value=float(data["value"]),
                provenance=MetricProvenance(data["provenance"]),
                unit=str(data.get("unit", "")),
            )
        except KeyError as exc:
            raise SchemaValidationError(
                f"Measurement is missing required field: {exc}"
            ) from exc
        except ValueError as exc:
            raise SchemaValidationError(
                f"Measurement has an invalid value: {exc}"
            ) from exc


def _measurement_from_optional(data: dict[str, Any] | None) -> Measurement | None:
    if data is None:
        return None
    return Measurement.from_dict(data)


def _measurement_to_optional(measurement: Measurement | None) -> dict[str, Any] | None:
    return None if measurement is None else measurement.to_dict()


def _validate_int(value: Any, *, context: str, key: str) -> int:
    """Require ``value`` to be a genuine ``int`` (not ``bool``/``float``/etc).

    Persisted records are untrusted input: a malformed field (a string, a
    float, a boolean, ``None``, a list, ...) must fail as a
    ``SchemaValidationError`` naming the offending field rather than
    either raising a raw ``ValueError``/``TypeError`` from ``int()`` or
    silently truncating/coercing (e.g. ``int(1.9)`` or ``int(True)``).
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise SchemaValidationError(
            f"{context}.{key} must be an integer, got {value!r}"
        )
    return int(value)


def _coerce_required_int(data: dict[str, Any], key: str, *, context: str) -> int:
    if key not in data:
        raise SchemaValidationError(f"{context} is missing required field: '{key}'")
    return _validate_int(data[key], context=context, key=key)


def _coerce_optional_int(data: dict[str, Any], key: str, *, context: str) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    return _validate_int(value, context=context, key=key)


def _validate_float(value: Any, *, context: str, key: str) -> float:
    """Require ``value`` to be a genuine number (not ``bool``/``str``/etc).

    Same rationale as ``_validate_int``: a malformed persisted float field
    must fail as a ``SchemaValidationError`` naming the offending field
    rather than raising a raw ``ValueError``/``TypeError`` from ``float()``
    or silently accepting a boolean (``float(True) == 1.0``).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaValidationError(f"{context}.{key} must be a number, got {value!r}")
    return float(value)


def _coerce_optional_float(
    data: dict[str, Any], key: str, *, context: str
) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    return _validate_float(value, context=context, key=key)


@dataclass(frozen=True)
class PlatformInfo:
    """Hardware/platform/OS context the run executed under."""

    os_name: str
    os_version: str
    architecture: str
    cpu_model: str | None = None
    cpu_cores: int | None = None
    total_memory_gb: float | None = None
    accelerator: str | None = None
    """e.g. 'Apple M5 Pro (20-core GPU, 24GB unified)' or 'NVIDIA RTX 4090 24GB'."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PlatformInfo:
        try:
            return cls(
                os_name=data["os_name"],
                os_version=data["os_version"],
                architecture=data["architecture"],
                cpu_model=data.get("cpu_model"),
                cpu_cores=_coerce_optional_int(
                    data, "cpu_cores", context="PlatformInfo"
                ),
                total_memory_gb=_coerce_optional_float(
                    data, "total_memory_gb", context="PlatformInfo"
                ),
                accelerator=data.get("accelerator"),
            )
        except KeyError as exc:
            raise SchemaValidationError(
                f"PlatformInfo is missing required field: {exc}"
            ) from exc


@dataclass(frozen=True)
class ModelInfo:
    """Identity of the model under test."""

    model_id: str
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    quantization: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelInfo:
        try:
            return cls(
                model_id=data["model_id"],
                model_revision=data.get("model_revision"),
                tokenizer_revision=data.get("tokenizer_revision"),
                quantization=data.get("quantization"),
            )
        except KeyError as exc:
            raise SchemaValidationError(
                f"ModelInfo is missing required field: {exc}"
            ) from exc


@dataclass(frozen=True)
class RuntimeInfo:
    """Identity of the inference runtime/backend that executed the run."""

    name: str
    version: str | None = None
    backend: str | None = None
    """e.g. 'Metal', 'CUDA', 'CPU'."""
    git_revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeInfo:
        try:
            return cls(
                name=data["name"],
                version=data.get("version"),
                backend=data.get("backend"),
                git_revision=data.get("git_revision"),
            )
        except KeyError as exc:
            raise SchemaValidationError(
                f"RuntimeInfo is missing required field: {exc}"
            ) from exc


@dataclass(frozen=True)
class CommandInfo:
    """The exact invocation that produced this run, plus content hashes."""

    argv: tuple[str, ...]
    config_hash: str | None = None
    workload_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "argv": list(self.argv),
            "config_hash": self.config_hash,
            "workload_hash": self.workload_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommandInfo:
        if "argv" not in data:
            raise SchemaValidationError("CommandInfo is missing required field: 'argv'")
        argv = data["argv"]
        # A scalar string is technically a ``Sequence[str]`` in Python, so
        # ``tuple("llama-cli")`` would silently explode into one-character
        # elements instead of failing. Require an actual list/tuple of
        # non-empty strings.
        if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence):
            raise SchemaValidationError(
                f"CommandInfo.argv must be a non-empty list of strings, got {argv!r}"
            )
        argv_tuple = tuple(argv)
        if not argv_tuple or not all(
            isinstance(item, str) and item for item in argv_tuple
        ):
            raise SchemaValidationError(
                "CommandInfo.argv must be a non-empty list of non-empty strings, "
                f"got {argv!r}"
            )
        return cls(
            argv=argv_tuple,
            config_hash=data.get("config_hash"),
            workload_hash=data.get("workload_hash"),
        )


@dataclass(frozen=True)
class RepetitionInfo:
    """Warmup/repetition/seed metadata for one measured run."""

    warmup_repetitions: int
    measured_repetitions: int
    repetition_index: int
    """0-based index of this record among the measured (non-warmup) reps."""
    seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepetitionInfo:
        return cls(
            warmup_repetitions=_coerce_required_int(
                data, "warmup_repetitions", context="RepetitionInfo"
            ),
            measured_repetitions=_coerce_required_int(
                data, "measured_repetitions", context="RepetitionInfo"
            ),
            repetition_index=_coerce_required_int(
                data, "repetition_index", context="RepetitionInfo"
            ),
            seed=_coerce_optional_int(data, "seed", context="RepetitionInfo"),
        )


@dataclass(frozen=True)
class TokenCounts:
    """Input/context/generated token counts for one run."""

    input_tokens: int | None = None
    context_tokens: int | None = None
    generated_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenCounts:
        return cls(
            input_tokens=_coerce_optional_int(
                data, "input_tokens", context="TokenCounts"
            ),
            context_tokens=_coerce_optional_int(
                data, "context_tokens", context="TokenCounts"
            ),
            generated_tokens=_coerce_optional_int(
                data, "generated_tokens", context="TokenCounts"
            ),
        )


@dataclass(frozen=True)
class TimingMetrics:
    """Model load, tokenize, prefill (TTFT), decode, and total timings."""

    model_load: Measurement | None = None
    tokenize: Measurement | None = None
    prefill: Measurement | None = None
    """Time-to-first-token, a.k.a. prompt processing time."""
    decode: Measurement | None = None
    total: Measurement | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_load": _measurement_to_optional(self.model_load),
            "tokenize": _measurement_to_optional(self.tokenize),
            "prefill": _measurement_to_optional(self.prefill),
            "decode": _measurement_to_optional(self.decode),
            "total": _measurement_to_optional(self.total),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TimingMetrics:
        return cls(
            model_load=_measurement_from_optional(data.get("model_load")),
            tokenize=_measurement_from_optional(data.get("tokenize")),
            prefill=_measurement_from_optional(data.get("prefill")),
            decode=_measurement_from_optional(data.get("decode")),
            total=_measurement_from_optional(data.get("total")),
        )


@dataclass(frozen=True)
class SpeculativeDecodingInfo:
    """Speculative decoding / MTP (multi-token prediction) evidence."""

    enabled: bool = False
    method: str | None = None
    """e.g. 'mtp', 'eagle', 'draft-model', 'lookahead'."""
    configured_depth: int | None = None
    proposed_tokens: int | None = None
    accepted_tokens: int | None = None
    verification_time: Measurement | None = None

    @property
    def acceptance_rate(self) -> float | None:
        """Accepted/proposed ratio, or ``None`` if either count is missing."""
        if not self.proposed_tokens or self.accepted_tokens is None:
            return None
        return self.accepted_tokens / self.proposed_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "method": self.method,
            "configured_depth": self.configured_depth,
            "proposed_tokens": self.proposed_tokens,
            "accepted_tokens": self.accepted_tokens,
            "verification_time": _measurement_to_optional(self.verification_time),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpeculativeDecodingInfo:
        return cls(
            enabled=bool(data.get("enabled", False)),
            method=data.get("method"),
            configured_depth=_coerce_optional_int(
                data, "configured_depth", context="SpeculativeDecodingInfo"
            ),
            proposed_tokens=_coerce_optional_int(
                data, "proposed_tokens", context="SpeculativeDecodingInfo"
            ),
            accepted_tokens=_coerce_optional_int(
                data, "accepted_tokens", context="SpeculativeDecodingInfo"
            ),
            verification_time=_measurement_from_optional(data.get("verification_time")),
        )


@dataclass(frozen=True)
class MemoryMetrics:
    """Active, cache, peak, and wired memory when available."""

    active: Measurement | None = None
    cache: Measurement | None = None
    peak: Measurement | None = None
    wired: Measurement | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "active": _measurement_to_optional(self.active),
            "cache": _measurement_to_optional(self.cache),
            "peak": _measurement_to_optional(self.peak),
            "wired": _measurement_to_optional(self.wired),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryMetrics:
        return cls(
            active=_measurement_from_optional(data.get("active")),
            cache=_measurement_from_optional(data.get("cache")),
            peak=_measurement_from_optional(data.get("peak")),
            wired=_measurement_from_optional(data.get("wired")),
        )


@dataclass(frozen=True)
class PowerMetrics:
    """Power/energy when available (rarely measurable without native tools)."""

    average_power: Measurement | None = None
    energy: Measurement | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "average_power": _measurement_to_optional(self.average_power),
            "energy": _measurement_to_optional(self.energy),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PowerMetrics:
        return cls(
            average_power=_measurement_from_optional(data.get("average_power")),
            energy=_measurement_from_optional(data.get("energy")),
        )


@dataclass(frozen=True)
class OutcomeInfo:
    """Task outcome/quality fields."""

    success: bool = True
    quality_score: float | None = None
    quality_metric: str | None = None
    """Name of the quality metric ``quality_score`` refers to, if any."""
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OutcomeInfo:
        return cls(
            success=bool(data.get("success", True)),
            quality_score=_coerce_optional_float(
                data, "quality_score", context="OutcomeInfo"
            ),
            quality_metric=data.get("quality_metric"),
            notes=data.get("notes"),
        )


@dataclass(frozen=True)
class ErrorInfo:
    """A failure/error encountered while producing this run."""

    category: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ErrorInfo:
        try:
            return cls(category=data["category"], message=data["message"])
        except KeyError as exc:
            raise SchemaValidationError(
                f"ErrorInfo is missing required field: {exc}"
            ) from exc


@dataclass(frozen=True)
class ExperimentRecord:
    """One canonical, versioned record of a single measured run.

    Optional fields are ``None`` when the underlying signal was not
    available; consumers must not treat ``None`` as zero.
    """

    run_id: str
    started_at: str
    platform: PlatformInfo
    model: ModelInfo
    runtime: RuntimeInfo
    command: CommandInfo
    repetition: RepetitionInfo
    schema_version: str = SCHEMA_VERSION
    ended_at: str | None = None
    tokens: TokenCounts = field(default_factory=TokenCounts)
    timing: TimingMetrics = field(default_factory=TimingMetrics)
    speculative: SpeculativeDecodingInfo = field(
        default_factory=SpeculativeDecodingInfo
    )
    memory: MemoryMetrics = field(default_factory=MemoryMetrics)
    power: PowerMetrics = field(default_factory=PowerMetrics)
    outcome: OutcomeInfo = field(default_factory=OutcomeInfo)
    error: ErrorInfo | None = None

    def validate(self) -> None:
        """Raise ``SchemaValidationError`` if this record is inconsistent."""
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError(
                f"Unsupported schema_version '{self.schema_version}', expected '{SCHEMA_VERSION}'"
            )
        if not self.run_id:
            raise SchemaValidationError("run_id must be non-empty")
        if not self.command.argv:
            raise SchemaValidationError(
                "command.argv must contain at least one element"
            )
        if self.repetition.repetition_index < 0:
            raise SchemaValidationError("repetition.repetition_index must be >= 0")
        if (
            self.repetition.warmup_repetitions < 0
            or self.repetition.measured_repetitions < 0
        ):
            raise SchemaValidationError("repetition counts must be >= 0")

        for name, count in (
            ("tokens.input_tokens", self.tokens.input_tokens),
            ("tokens.context_tokens", self.tokens.context_tokens),
            ("tokens.generated_tokens", self.tokens.generated_tokens),
        ):
            if count is not None and count < 0:
                raise SchemaValidationError(f"{name} must be >= 0, got {count}")

        for name, measurement in (
            ("timing.model_load", self.timing.model_load),
            ("timing.tokenize", self.timing.tokenize),
            ("timing.prefill", self.timing.prefill),
            ("timing.decode", self.timing.decode),
            ("timing.total", self.timing.total),
            ("memory.active", self.memory.active),
            ("memory.cache", self.memory.cache),
            ("memory.peak", self.memory.peak),
            ("memory.wired", self.memory.wired),
            ("power.average_power", self.power.average_power),
            ("power.energy", self.power.energy),
            ("speculative.verification_time", self.speculative.verification_time),
        ):
            if measurement is not None and measurement.value < 0:
                raise SchemaValidationError(
                    f"{name} must be >= 0, got {measurement.value}"
                )

        spec = self.speculative
        if spec.enabled:
            if spec.proposed_tokens is not None and spec.proposed_tokens < 0:
                raise SchemaValidationError("speculative.proposed_tokens must be >= 0")
            if spec.accepted_tokens is not None and spec.accepted_tokens < 0:
                raise SchemaValidationError("speculative.accepted_tokens must be >= 0")
            if (
                spec.proposed_tokens is not None
                and spec.accepted_tokens is not None
                and spec.accepted_tokens > spec.proposed_tokens
            ):
                raise SchemaValidationError(
                    "speculative.accepted_tokens cannot exceed speculative.proposed_tokens "
                    f"({spec.accepted_tokens} > {spec.proposed_tokens})"
                )

        if self.error is not None and self.outcome.success:
            raise SchemaValidationError(
                "outcome.success cannot be True when error is set"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "platform": self.platform.to_dict(),
            "model": self.model.to_dict(),
            "runtime": self.runtime.to_dict(),
            "command": self.command.to_dict(),
            "repetition": self.repetition.to_dict(),
            "tokens": self.tokens.to_dict(),
            "timing": self.timing.to_dict(),
            "speculative": self.speculative.to_dict(),
            "memory": self.memory.to_dict(),
            "power": self.power.to_dict(),
            "outcome": self.outcome.to_dict(),
            "error": None if self.error is None else self.error.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRecord:
        try:
            record = cls(
                schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
                run_id=data["run_id"],
                started_at=data["started_at"],
                ended_at=data.get("ended_at"),
                platform=PlatformInfo.from_dict(data["platform"]),
                model=ModelInfo.from_dict(data["model"]),
                runtime=RuntimeInfo.from_dict(data["runtime"]),
                command=CommandInfo.from_dict(data["command"]),
                repetition=RepetitionInfo.from_dict(data["repetition"]),
                tokens=TokenCounts.from_dict(data.get("tokens", {})),
                timing=TimingMetrics.from_dict(data.get("timing", {})),
                speculative=SpeculativeDecodingInfo.from_dict(
                    data.get("speculative", {})
                ),
                memory=MemoryMetrics.from_dict(data.get("memory", {})),
                power=PowerMetrics.from_dict(data.get("power", {})),
                outcome=OutcomeInfo.from_dict(data.get("outcome", {})),
                error=(
                    None
                    if data.get("error") is None
                    else ErrorInfo.from_dict(data["error"])
                ),
            )
        except KeyError as exc:
            raise SchemaValidationError(
                f"ExperimentRecord is missing required field: {exc}"
            ) from exc
        record.validate()
        return record

    @classmethod
    def from_json(cls, payload: str) -> ExperimentRecord:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(
                f"Invalid JSON for ExperimentRecord: {exc}"
            ) from exc
        return cls.from_dict(data)

    def write_json(self, path: str | Path) -> None:
        """Atomically write this record as pretty JSON to ``path``."""
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        tmp_path.write_text(self.to_json() + "\n", encoding="utf-8")
        os.replace(tmp_path, target)

    @classmethod
    def read_json(cls, path: str | Path) -> ExperimentRecord:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
