"""Canonical experiment/evidence schema for the inference optimizer.

Every collector (llama.cpp parser, future MLX/Metal or CUDA collectors) and
every analysis rule (the "doctor") reads and writes this schema, so it is
the single source of truth for what a benchmarking run looks like. Optional
measurements stay ``None`` when they were not observed — nothing here
invents a value.

This is schema ``SCHEMA_VERSION``. New optional fields that older records
simply omit are additive: ``from_dict`` defaults them, old records keep
deserializing, and the constant stays put. Bump the constant, and extend
``ExperimentRecord.from_dict`` to stay backward compatible, whenever an
existing field changes meaning, type, or requiredness.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from ._artifact_io import (
    MAX_EVIDENCE_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)

SCHEMA_VERSION = "1"


class SchemaValidationError(ValueError):
    """Raised when an ``ExperimentRecord`` (or a part of it) is invalid."""


def _require_object(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SchemaValidationError(
            f"{context} must be an object, got {type(value).__name__}"
        )
    return value


def _required_string(data: dict[str, Any], key: str, *, context: str) -> str:
    if key not in data:
        raise SchemaValidationError(f"{context} is missing required field: '{key}'")
    value = data[key]
    if not isinstance(value, str):
        raise SchemaValidationError(f"{context}.{key} must be a string, got {value!r}")
    return value


def _optional_string(data: dict[str, Any], key: str, *, context: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise SchemaValidationError(
            f"{context}.{key} must be a string or null, got {value!r}"
        )
    return value


def _string_with_default(
    data: dict[str, Any], key: str, *, context: str, default: str
) -> str:
    if key not in data:
        return default
    return _required_string(data, key, context=context)


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

    PROVIDER_REPORTED = "provider_reported"
    """Reported by a remote service, not observed by this client.

    Remote inference APIs return their own accounting (token usage in
    particular). Those numbers are evidence, but they are neither a local
    hardware counter nor a host timer: the client cannot verify them and
    cannot decompose them. They are kept under their own provenance so
    they are never mistaken for a native measurement.
    """

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
        data = _require_object(data, context="Measurement")
        try:
            return cls(
                value=_validate_float(
                    data["value"], context="Measurement", key="value"
                ),
                provenance=MetricProvenance(data["provenance"]),
                unit=_string_with_default(
                    data, "unit", context="Measurement", default=""
                ),
            )
        except KeyError as exc:
            raise SchemaValidationError(
                f"Measurement is missing required field: {exc}"
            ) from exc
        except (TypeError, ValueError) as exc:
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
    try:
        return float(value)
    except OverflowError as exc:
        raise SchemaValidationError(
            f"{context}.{key} must be a finite number, got an overflowing integer"
        ) from exc


def _coerce_optional_float(
    data: dict[str, Any], key: str, *, context: str
) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    return _validate_float(value, context=context, key=key)


def _validate_bool(value: Any, *, context: str, key: str) -> bool:
    """Require ``value`` to be a genuine JSON ``bool``.

    ``bool(...)`` is dangerously permissive for persisted input: nearly
    every non-empty value (including the string ``"false"``) is truthy,
    so ``bool("false")`` is ``True``. A malformed persisted boolean field
    (a string, an int, ``None``, a list, ...) must fail as a
    ``SchemaValidationError`` naming the offending field instead of
    silently being coerced to the wrong truth value.
    """
    if not isinstance(value, bool):
        raise SchemaValidationError(f"{context}.{key} must be a boolean, got {value!r}")
    return value


def _coerce_bool_with_default(
    data: dict[str, Any], key: str, *, context: str, default: bool
) -> bool:
    if key not in data:
        return default
    return _validate_bool(data[key], context=context, key=key)


def _coerce_str_tuple(
    data: dict[str, Any], key: str, *, context: str
) -> tuple[str, ...]:
    """Read an optional list-of-strings field as a tuple.

    A bare string is a ``Sequence[str]`` in Python, so ``tuple("abc")``
    would silently become ``("a", "b", "c")``. Require a real list/tuple
    of non-empty strings instead of quietly shredding a scalar.
    """
    value = data.get(key)
    if value is None:
        return ()
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SchemaValidationError(
            f"{context}.{key} must be a list of strings, got {value!r}"
        )
    items = tuple(value)
    if not all(isinstance(item, str) and item for item in items):
        raise SchemaValidationError(
            f"{context}.{key} must contain only non-empty strings, got {value!r}"
        )
    return items


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
        data = _require_object(data, context="PlatformInfo")
        try:
            return cls(
                os_name=_required_string(data, "os_name", context="PlatformInfo"),
                os_version=_required_string(data, "os_version", context="PlatformInfo"),
                architecture=_required_string(
                    data, "architecture", context="PlatformInfo"
                ),
                cpu_model=_optional_string(data, "cpu_model", context="PlatformInfo"),
                cpu_cores=_coerce_optional_int(
                    data, "cpu_cores", context="PlatformInfo"
                ),
                total_memory_gb=_coerce_optional_float(
                    data, "total_memory_gb", context="PlatformInfo"
                ),
                accelerator=_optional_string(
                    data, "accelerator", context="PlatformInfo"
                ),
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
    model_family: str | None = None
    """Architecture family (e.g. mlx-lm/mlx-vlm ``model_type`` such as
    'qwen3_next' or 'qwen4_exp'), distinct from ``model_id`` (which is
    often a user-chosen Hugging Face repo name). Native-MTP collectors use
    this to label which architecture family a capability determination or
    run applies to."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelInfo:
        data = _require_object(data, context="ModelInfo")
        try:
            return cls(
                model_id=_required_string(data, "model_id", context="ModelInfo"),
                model_revision=_optional_string(
                    data, "model_revision", context="ModelInfo"
                ),
                tokenizer_revision=_optional_string(
                    data, "tokenizer_revision", context="ModelInfo"
                ),
                quantization=_optional_string(
                    data, "quantization", context="ModelInfo"
                ),
                model_family=_optional_string(
                    data, "model_family", context="ModelInfo"
                ),
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
    provider: str | None = None
    """Remote service that executed the run, when it was not local.

    Set only by collectors that talk to a hosted API (e.g. an
    OpenAI-compatible endpoint), so a remote run is never mistaken for a
    local one. Local collectors leave this ``None``. It deliberately does
    not reuse ``backend``, which describes a local compute backend."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimeInfo:
        data = _require_object(data, context="RuntimeInfo")
        try:
            return cls(
                name=_required_string(data, "name", context="RuntimeInfo"),
                version=_optional_string(data, "version", context="RuntimeInfo"),
                backend=_optional_string(data, "backend", context="RuntimeInfo"),
                git_revision=_optional_string(
                    data, "git_revision", context="RuntimeInfo"
                ),
                provider=_optional_string(data, "provider", context="RuntimeInfo"),
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
        data = _require_object(data, context="CommandInfo")
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
            config_hash=_optional_string(data, "config_hash", context="CommandInfo"),
            workload_hash=_optional_string(
                data, "workload_hash", context="CommandInfo"
            ),
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
        data = _require_object(data, context="RepetitionInfo")
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
    provenance: MetricProvenance | None = None
    """How these counts were obtained, when that is not self-evident.

    Local collectors tokenize the prompt themselves and count generation
    steps, so they leave this ``None``. A collector that can only repeat
    what a remote service reported must set
    ``MetricProvenance.PROVIDER_REPORTED`` so the counts are never read
    as locally measured."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "context_tokens": self.context_tokens,
            "generated_tokens": self.generated_tokens,
            "provenance": None if self.provenance is None else self.provenance.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenCounts:
        data = _require_object(data, context="TokenCounts")
        provenance = data.get("provenance")
        if provenance is not None:
            try:
                provenance = MetricProvenance(provenance)
            except ValueError as exc:
                raise SchemaValidationError(
                    f"TokenCounts.provenance is invalid: {exc}"
                ) from exc
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
            provenance=provenance,
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
        data = _require_object(data, context="TimingMetrics")
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
    """e.g. 'native-mtp', 'draft-model', 'eagle', 'lookahead'.

    'native-mtp' must only be used when the runtime invoked its own
    multi-token-prediction heads through a verified API path (see
    ``llmtracefx.optimizer.collectors.native_mtp``); generic external
    draft-model speculative decoding must be labeled 'draft-model' even
    when the draft checkpoint happens to be an extracted MTP head, unless
    the runtime is verified to keep the two paths distinguishable."""
    configured_depth: int | None = None
    """Configured speculation/MTP block depth (e.g. draft tokens per step)."""
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
        data = _require_object(data, context="SpeculativeDecodingInfo")
        return cls(
            enabled=_coerce_bool_with_default(
                data, "enabled", context="SpeculativeDecodingInfo", default=False
            ),
            method=_optional_string(data, "method", context="SpeculativeDecodingInfo"),
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
        data = _require_object(data, context="MemoryMetrics")
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
        data = _require_object(data, context="PowerMetrics")
        return cls(
            average_power=_measurement_from_optional(data.get("average_power")),
            energy=_measurement_from_optional(data.get("energy")),
        )


#: The complete set of Instruments metrics this project is willing to
#: persist, with the exact table each must come from, its exact unit and
#: its exact provenance. An allowlist rather than a denylist: a denylist
#: only blocks the overclaims somebody thought to enumerate, and
#: ``metal_power_watts`` sailed straight through one.
INSTRUMENT_METRIC_SPECS: dict[str, tuple[str, str, MetricProvenance]] = {
    "metal_gpu_interval_count": (
        "metal-gpu-intervals",
        "intervals",
        MetricProvenance.MEASURED_NATIVE,
    ),
    "metal_gpu_interval_duration_sum": (
        "metal-gpu-intervals",
        "ms",
        MetricProvenance.MEASURED_NATIVE,
    ),
    "metal_gpu_interval_wall_span": (
        "metal-gpu-intervals",
        "ms",
        MetricProvenance.MEASURED_NATIVE,
    ),
    "metal_gpu_interval_count_all_processes": (
        "metal-gpu-intervals",
        "intervals",
        MetricProvenance.MEASURED_NATIVE,
    ),
}

#: Table schemas this project has a validated parser for. Anything in
#: ``parsed_schemas`` must appear here, so a record cannot claim to have
#: parsed a table no parser exists for.
INSTRUMENT_PARSABLE_SCHEMAS: frozenset[str] = frozenset({"metal-gpu-intervals"})

#: Substrings that must never appear in an ``InstrumentsEvidence``
#: metric name. Redundant with the allowlist above and kept anyway,
#: because it produces a much more specific error for the exact mistake
#: it describes.
FORBIDDEN_INSTRUMENT_METRIC_MARKERS: tuple[str, ...] = (
    "utilization",
    "occupancy",
    "bandwidth",
    "busy_percent",
    "kernel_time",
    "gpu_power",
    "gpu_energy",
    "gpu_memory",
)


@dataclass(frozen=True)
class InstrumentsEvidence:
    """Evidence sourced from an Apple Instruments ``.trace`` bundle.

    Deliberately kept separate from ``MemoryMetrics``/``PowerMetrics``:
    those carry runtime-allocator and host-side numbers (for example the
    MLX allocator's active/cache/peak bytes), whereas everything here
    came out of ``xctrace``. Mixing the two would make it impossible to
    tell an allocator bookkeeping value apart from a profiler
    measurement.

    The common case for this dataclass is "a trace exists, and these are
    the schemas it advertises". ``metrics`` stays empty unless a strict,
    reproducible parser actually derived a value from an exported table,
    so an absent GPU/bandwidth/occupancy/power number is always visibly
    absent rather than silently zero.
    """

    tool: str = "xctrace"
    tool_version: str | None = None
    """Exact ``xctrace version`` output, when it could be read."""
    capability: str | None = None
    """Capability state at collection time, e.g. 'supported'."""
    template: str | None = None
    """Instruments template requested, e.g. 'Metal System Trace'."""
    trace_bundle_name: str | None = None
    """Basename of the ``.trace`` bundle. Never a full path: the bundle
    lives beside the record, and absolute paths leak home directories."""
    available_schemas: tuple[str, ...] = ()
    """Table schemas the bundle's table of contents advertises."""
    parsed_schemas: tuple[str, ...] = ()
    """Schemas a strict parser in this project actually understood."""
    unsupported_schemas: tuple[str, ...] = ()
    """Schemas present in the bundle that this project cannot parse
    reproducibly. Listed so the gap is explicit rather than invisible."""
    metrics: dict[str, Measurement] = field(default_factory=dict)
    """Only values a strict parser derived from an exported table."""
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "tool_version": self.tool_version,
            "capability": self.capability,
            "template": self.template,
            "trace_bundle_name": self.trace_bundle_name,
            "available_schemas": list(self.available_schemas),
            "parsed_schemas": list(self.parsed_schemas),
            "unsupported_schemas": list(self.unsupported_schemas),
            "metrics": {
                name: measurement.to_dict()
                for name, measurement in sorted(self.metrics.items())
            },
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InstrumentsEvidence:
        data = _require_object(data, context="InstrumentsEvidence")
        metrics_data = data.get("metrics", {})
        if not isinstance(metrics_data, dict):
            raise SchemaValidationError(
                f"InstrumentsEvidence.metrics must be an object, got {metrics_data!r}"
            )
        if not all(isinstance(name, str) for name in metrics_data):
            raise SchemaValidationError(
                "InstrumentsEvidence.metrics keys must be strings"
            )
        return cls(
            tool=_string_with_default(
                data, "tool", context="InstrumentsEvidence", default="xctrace"
            ),
            tool_version=_optional_string(
                data, "tool_version", context="InstrumentsEvidence"
            ),
            capability=_optional_string(
                data, "capability", context="InstrumentsEvidence"
            ),
            template=_optional_string(data, "template", context="InstrumentsEvidence"),
            trace_bundle_name=_optional_string(
                data, "trace_bundle_name", context="InstrumentsEvidence"
            ),
            available_schemas=_coerce_str_tuple(
                data, "available_schemas", context="InstrumentsEvidence"
            ),
            parsed_schemas=_coerce_str_tuple(
                data, "parsed_schemas", context="InstrumentsEvidence"
            ),
            unsupported_schemas=_coerce_str_tuple(
                data, "unsupported_schemas", context="InstrumentsEvidence"
            ),
            metrics={
                name: Measurement.from_dict(value)
                for name, value in metrics_data.items()
            },
            notes=_optional_string(data, "notes", context="InstrumentsEvidence"),
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
        data = _require_object(data, context="OutcomeInfo")
        return cls(
            success=_coerce_bool_with_default(
                data, "success", context="OutcomeInfo", default=True
            ),
            quality_score=_coerce_optional_float(
                data, "quality_score", context="OutcomeInfo"
            ),
            quality_metric=_optional_string(
                data, "quality_metric", context="OutcomeInfo"
            ),
            notes=_optional_string(data, "notes", context="OutcomeInfo"),
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
        data = _require_object(data, context="ErrorInfo")
        try:
            return cls(
                category=_required_string(data, "category", context="ErrorInfo"),
                message=_required_string(data, "message", context="ErrorInfo"),
            )
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
    instruments: InstrumentsEvidence | None = None
    """Apple Instruments (``xctrace``) evidence, when a trace was taken.

    Additive and optional: records written before this field existed
    parse unchanged and leave it ``None``. ``None`` means "no trace was
    taken", not "the trace measured zero"."""
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

        self._validate_instruments()

    def _validate_instruments(self) -> None:
        """Enforce that Instruments metrics cannot overclaim.

        The rules, all aimed at the same failure mode (a plausible
        looking GPU number that nothing actually measured):

        1. Every metric name must be in
           :data:`INSTRUMENT_METRIC_SPECS`, with exactly the unit and
           provenance that entry declares. An allowlist, because a
           denylist only blocks the overclaims somebody remembered to
           enumerate.
        2. The table a metric is derived from must actually appear in
           ``parsed_schemas``.
        3. ``parsed_schemas`` must name tables this project has a parser
           for, and must be a subset of the schemas the trace itself
           advertised, so a phantom schema cannot unlock a metric.
        4. A schema cannot be simultaneously parsed and unsupported.
        """
        evidence = self.instruments
        if evidence is None:
            return

        parsed = set(evidence.parsed_schemas)
        overlap = parsed & set(evidence.unsupported_schemas)
        if overlap:
            raise SchemaValidationError(
                "instruments schemas cannot be both parsed and unsupported: "
                + ", ".join(sorted(overlap))
            )

        unparsable = parsed - INSTRUMENT_PARSABLE_SCHEMAS
        if unparsable:
            raise SchemaValidationError(
                "instruments.parsed_schemas names tables this project has "
                "no parser for: " + ", ".join(sorted(unparsable))
            )

        advertised = set(evidence.available_schemas)
        if advertised:
            phantom = parsed - advertised
            if phantom:
                raise SchemaValidationError(
                    "instruments.parsed_schemas is not a subset of "
                    "instruments.available_schemas; the trace never "
                    "advertised: " + ", ".join(sorted(phantom))
                )
        elif parsed:
            raise SchemaValidationError(
                "instruments.parsed_schemas is non-empty while "
                "instruments.available_schemas is empty; a table cannot be "
                "parsed from a trace that advertised nothing"
            )

        for name, measurement in sorted(evidence.metrics.items()):
            if not name:
                raise SchemaValidationError(
                    "instruments.metrics keys must be non-empty strings"
                )
            folded = name.casefold()
            for marker in FORBIDDEN_INSTRUMENT_METRIC_MARKERS:
                if marker in folded:
                    raise SchemaValidationError(
                        f"instruments.metrics.{name} claims a quantity this "
                        f"project cannot derive from an Instruments export "
                        f"(forbidden marker '{marker}'). GPU utilization, "
                        "occupancy, bandwidth, kernel time, power and memory "
                        "are not recoverable from the exported tables "
                        "without unvalidated modelling assumptions."
                    )

            spec = INSTRUMENT_METRIC_SPECS.get(name)
            if spec is None:
                raise SchemaValidationError(
                    f"instruments.metrics.{name} is not a metric this "
                    "project derives. Allowed: "
                    + ", ".join(sorted(INSTRUMENT_METRIC_SPECS))
                )
            source_schema, expected_unit, expected_provenance = spec

            if measurement.unit != expected_unit:
                raise SchemaValidationError(
                    f"instruments.metrics.{name} must be in "
                    f"{expected_unit!r}, got {measurement.unit!r}"
                )
            if measurement.provenance is not expected_provenance:
                raise SchemaValidationError(
                    f"instruments.metrics.{name} must have provenance "
                    f"'{expected_provenance.value}', got "
                    f"'{measurement.provenance.value}'"
                )
            if not math.isfinite(measurement.value):
                raise SchemaValidationError(
                    f"instruments.metrics.{name} must be a finite number, "
                    f"got {measurement.value}. NaN and infinity pass a "
                    "'>= 0' check silently and would propagate as though "
                    "they were measurements."
                )
            if measurement.value < 0:
                raise SchemaValidationError(
                    f"instruments.metrics.{name} must be >= 0, "
                    f"got {measurement.value}"
                )
            if source_schema not in parsed:
                raise SchemaValidationError(
                    f"instruments.metrics.{name} is derived from "
                    f"{source_schema!r}, which is not in "
                    "instruments.parsed_schemas, so no exported table "
                    "could have produced it"
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
            "instruments": (
                None if self.instruments is None else self.instruments.to_dict()
            ),
            "outcome": self.outcome.to_dict(),
            "error": None if self.error is None else self.error.to_dict(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, allow_non_finite: bool = False
    ) -> ExperimentRecord:
        data = _require_object(data, context="ExperimentRecord")
        try:
            record = cls(
                schema_version=_string_with_default(
                    data,
                    "schema_version",
                    context="ExperimentRecord",
                    default=SCHEMA_VERSION,
                ),
                run_id=_required_string(data, "run_id", context="ExperimentRecord"),
                started_at=_required_string(
                    data, "started_at", context="ExperimentRecord"
                ),
                ended_at=_optional_string(data, "ended_at", context="ExperimentRecord"),
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
                instruments=(
                    None
                    if data.get("instruments") is None
                    else InstrumentsEvidence.from_dict(data["instruments"])
                ),
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
        if not allow_non_finite:
            try:
                json.dumps(record.to_dict(), allow_nan=False)
            except ValueError as exc:
                raise SchemaValidationError(
                    "ExperimentRecord numeric fields must be finite"
                ) from exc
        return record

    @classmethod
    def from_json(
        cls, payload: str, *, allow_non_finite: bool = False
    ) -> ExperimentRecord:
        try:
            data = json.loads(
                payload,
                parse_constant=(
                    None if allow_non_finite else reject_non_finite_json_constant
                ),
            )
        except (ValueError, RecursionError) as exc:
            raise SchemaValidationError(
                f"Invalid JSON for ExperimentRecord: {exc}"
            ) from exc
        # Valid JSON is not necessarily an object. Without this, a payload
        # of ``[]`` or ``null`` reaches ``from_dict`` and raises
        # ``AttributeError``, which no caller expects from a parser whose
        # documented failure mode is ``SchemaValidationError``. Mirrors
        # the check ``MatrixManifest.from_json`` already performs.
        if not isinstance(data, dict):
            raise SchemaValidationError(
                f"ExperimentRecord JSON must be an object, got {type(data).__name__}"
            )
        return cls.from_dict(data, allow_non_finite=allow_non_finite)

    def write_json(self, path: str | Path) -> None:
        """Atomically write this record as pretty JSON to ``path``."""
        self.validate()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = target.with_name(f".{target.name}.tmp-{os.getpid()}")
        tmp_path.write_text(self.to_json() + "\n", encoding="utf-8")
        os.replace(tmp_path, target)

    @classmethod
    def read_json(
        cls, path: str | Path, *, allow_non_finite: bool = False
    ) -> ExperimentRecord:
        try:
            payload = read_bounded_regular_text(path, MAX_EVIDENCE_ARTIFACT_BYTES)
        except ArtifactReadError as exc:
            raise SchemaValidationError(
                f"Invalid ExperimentRecord file: {exc}"
            ) from exc
        return cls.from_json(payload, allow_non_finite=allow_non_finite)
