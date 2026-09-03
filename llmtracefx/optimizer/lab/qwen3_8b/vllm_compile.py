"""Pure, offline contract for the Qwen3-8B vLLM compilation experiment.

This module deliberately contains no Modal, vLLM, or network imports.  It
defines the immutable experiment envelope, validates evidence produced by a
separate runner, computes break-even points, and maintains a path-bound
lifecycle budget ledger.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import ROUND_CEILING, Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any

from ..._artifact_io import (
    MAX_EVIDENCE_ARTIFACT_BYTES,
    MAX_METADATA_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from ...collectors._shared import atomic_write_text

MODEL_ID = "Qwen/Qwen3-8B"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
EXPECTED_MODEL_FILE_COUNT = 15
EXPECTED_MODEL_BYTES = 16_397_461_266
HARD_CAP_USD = Decimal("28")
APPROVED_PLAN_SHA256 = (
    "sha256:cf11310288784e10ccbb364bca6874ac53100b747b0b76375524b4b4ae013ec0"
)
ALLOWANCE_SECONDS = 2700
POST_DELETE_STORAGE_DAYS = 4
CPU_CORES = 4
MEMORY_GIB = 32
REQUESTS_PER_CELL = 12
MAX_PRICE_AGE_DAYS = 30

RATE_FIELDS = (
    "l40s_gpu_second_usd",
    "h100_gpu_second_usd",
    "cpu_core_second_usd",
    "memory_gib_second_usd",
    "volume_gib_month_usd",
)
CURRENT_RATES = {
    "l40s_gpu_second_usd": Decimal("0.000542"),
    "h100_gpu_second_usd": Decimal("0.001097"),
    "cpu_core_second_usd": Decimal("0.0000131"),
    "memory_gib_second_usd": Decimal("0.00000222"),
    "volume_gib_month_usd": Decimal("0.09"),
}
OFFICIAL_VLLM_IMAGE_DIGEST = (
    "sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635"
)
REQUIRED_RUNTIME_PINS = (
    "python_version",
    "vllm_version",
    "torch_version",
    "cuda_version",
    "typing_extensions_version",
)
WORKLOAD_IDS = (
    "structured-json-profile-extraction",
    "prose-reasoning-two-train-gap",
)
CONTEXT_TIERS = ("2k", "8k", "16k")
EXPECTED_GPU_NAMES = {
    "L40S": "NVIDIA L40S",
    "H100!": "NVIDIA H100",
}
TERMINAL_FINISH_REASONS = frozenset({"stop", "length"})

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_GIT_HEAD = re.compile(r"^[0-9a-f]{40}$")
_EXACT_VERSION = re.compile(r"^\d+(?:\.\d+)+(?:[-+][0-9A-Za-z.-]+)?$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_LEDGER_SCHEMA_VERSION = "1"
_PLAN_SCHEMA_VERSION = "1"
_PROMPT_MANIFEST = (
    Path(__file__).parent / "data" / "qwen3-8b-control-manifest-template-v1.json"
)


class VLLMCompileContractError(ValueError):
    """Raised whenever the offline contract cannot prove an input safe."""


def canonical_json(value: Any) -> str:
    """Return stable, finite JSON used for all contract hashes."""

    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VLLMCompileContractError(f"value is not canonical JSON: {exc}") from exc


def canonical_decimal(value: Decimal) -> str:
    """Serialize a finite Decimal without exponent or insignificant zeroes."""

    if not isinstance(value, Decimal) or not value.is_finite():
        raise VLLMCompileContractError("decimal value must be finite")
    if len(value.as_tuple().digits) > 128 or abs(value.adjusted()) > 128:
        raise VLLMCompileContractError("decimal value exceeds the supported magnitude")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in ("", "-0") else rendered


def _decimal(value: Any, *, field: str, positive: bool = False) -> Decimal:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise VLLMCompileContractError(f"{field} must be a decimal string")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise VLLMCompileContractError(f"{field} must be a decimal string") from exc
    if not result.is_finite() or result < 0 or (positive and result <= 0):
        relation = "> 0" if positive else ">= 0"
        raise VLLMCompileContractError(f"{field} must be finite and {relation}")
    if canonical_decimal(result) != value:
        raise VLLMCompileContractError(
            f"{field} must use canonical decimal spelling "
            f"{canonical_decimal(result)!r}"
        )
    return result


def _sha256_json(value: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _parse_date(value: str, *, field: str) -> date:
    if not isinstance(value, str):
        raise VLLMCompileContractError(f"{field} must be an ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise VLLMCompileContractError(f"{field} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise VLLMCompileContractError(f"{field} must be a canonical ISO date")
    return parsed


@dataclass(frozen=True)
class PricePins:
    """Exact current rates plus their caller-supplied provenance."""

    rates: tuple[tuple[str, Decimal], ...]
    effective_date: str
    source: str
    source_sha256: str

    @classmethod
    def create(
        cls,
        rates: Mapping[str, str],
        *,
        effective_date: str,
        source: str,
        source_sha256: str,
        as_of_date: str,
        max_age_days: int = MAX_PRICE_AGE_DAYS,
    ) -> PricePins:
        if set(rates) != set(RATE_FIELDS):
            raise VLLMCompileContractError(
                f"prices must contain exactly {list(RATE_FIELDS)!r}"
            )
        if not isinstance(source, str) or not source.strip():
            raise VLLMCompileContractError("price source must be non-empty")
        if not isinstance(source_sha256, str) or not _SHA256.fullmatch(source_sha256):
            raise VLLMCompileContractError(
                "price source response must have an immutable sha256 pin"
            )
        if isinstance(max_age_days, bool) or not isinstance(max_age_days, int):
            raise VLLMCompileContractError("max price age must be an integer")
        if max_age_days < 0:
            raise VLLMCompileContractError("max price age must be non-negative")
        if max_age_days > MAX_PRICE_AGE_DAYS:
            raise VLLMCompileContractError(
                f"max price age cannot exceed {MAX_PRICE_AGE_DAYS} days"
            )
        effective = _parse_date(effective_date, field="effective_date")
        as_of = _parse_date(as_of_date, field="as_of_date")
        age = (as_of - effective).days
        if age < 0:
            raise VLLMCompileContractError("prices cannot be effective in the future")
        if age > max_age_days:
            raise VLLMCompileContractError(
                f"prices are stale ({age} days old; maximum is {max_age_days})"
            )
        parsed: list[tuple[str, Decimal]] = []
        for name in RATE_FIELDS:
            rate = _decimal(rates[name], field=f"prices.{name}", positive=True)
            baseline = CURRENT_RATES[name]
            if rate < baseline / 2 or rate > baseline * 2:
                raise VLLMCompileContractError(
                    f"prices.{name} differs by more than 2x from the approved "
                    "baseline; refresh the reviewed price guard"
                )
            plausible_maximum = (
                Decimal("0.1")
                if name.endswith("gpu_second_usd")
                else (
                    Decimal("100")
                    if name == "volume_gib_month_usd"
                    else Decimal("0.01")
                )
            )
            if rate > plausible_maximum:
                raise VLLMCompileContractError(
                    f"prices.{name} exceeds its plausible unit-rate ceiling"
                )
            parsed.append((name, rate))
        return cls(tuple(parsed), effective_date, source, source_sha256)

    def rate(self, name: str) -> Decimal:
        return dict(self.rates)[name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": "USD",
            "effective_date": self.effective_date,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "rates": {name: canonical_decimal(value) for name, value in self.rates},
        }

    @property
    def content_sha256(self) -> str:
        return _sha256_json(self.to_dict())


@dataclass(frozen=True)
class RuntimePins:
    """Caller-supplied exact versions; mutable selectors are never accepted."""

    values: tuple[tuple[str, str], ...]

    @classmethod
    def create(cls, values: Mapping[str, str]) -> RuntimePins:
        if set(values) != set(REQUIRED_RUNTIME_PINS):
            raise VLLMCompileContractError(
                f"runtime pins must contain exactly {list(REQUIRED_RUNTIME_PINS)!r}"
            )
        pins: list[tuple[str, str]] = []
        for key in REQUIRED_RUNTIME_PINS:
            value = values[key]
            if (
                not isinstance(value, str)
                or not _EXACT_VERSION.fullmatch(value)
                or re.search(r"latest|current|main|master|snapshot", value, re.I)
            ):
                raise VLLMCompileContractError(
                    f"runtime pin {key!r} must be an exact immutable version"
                )
            pins.append((key, value))
        return cls(tuple(pins))

    def to_dict(self) -> dict[str, str]:
        return dict(self.values)


@dataclass(frozen=True)
class ExperimentCell:
    cell_id: str
    accelerator: str
    execution_mode: str
    compile_enabled: bool
    gpu_count: int = 1
    cpu_cores: int = CPU_CORES
    memory_gib: int = MEMORY_GIB
    max_containers: int = 1
    min_containers: int = 0
    concurrency: int = 1
    retries: int = 0
    allowance_seconds: int = ALLOWANCE_SECONDS

    def to_dict(self) -> dict[str, Any]:
        return {
            "cell_id": self.cell_id,
            "accelerator": self.accelerator,
            "execution_mode": self.execution_mode,
            "compile_enabled": self.compile_enabled,
            "gpu_count": self.gpu_count,
            "cpu_cores": self.cpu_cores,
            "memory_gib": self.memory_gib,
            "max_containers": self.max_containers,
            "min_containers": self.min_containers,
            "concurrency": self.concurrency,
            "retries": self.retries,
            "allowance_seconds": self.allowance_seconds,
        }


CELLS = (
    ExperimentCell("l40s-eager", "L40S", "eager", False),
    ExperimentCell("l40s-compiled", "L40S", "compiled", True),
    ExperimentCell("h100-eager", "H100!", "eager", False),
    ExperimentCell("h100-compiled", "H100!", "compiled", True),
)


@dataclass(frozen=True)
class PlanLine:
    line_id: str
    kind: str
    amount_usd: Decimal
    allowance_seconds: int | None = None
    cell_id: str | None = None
    retained_gib: Decimal | None = None
    retained_days: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "line_id": self.line_id,
            "kind": self.kind,
            "amount_usd": canonical_decimal(self.amount_usd),
            "allowance_seconds": self.allowance_seconds,
            "cell_id": self.cell_id,
            "retained_gib": (
                canonical_decimal(self.retained_gib)
                if self.retained_gib is not None
                else None
            ),
            "retained_days": self.retained_days,
        }


def _plan_lines(prices: PricePins) -> tuple[PlanLine, ...]:
    with localcontext() as context:
        context.prec = 28
        seconds = Decimal(ALLOWANCE_SECONDS)
        cpu_memory_per_second = prices.rate("cpu_core_second_usd") * Decimal(
            CPU_CORES
        ) + prices.rate("memory_gib_second_usd") * Decimal(MEMORY_GIB)
        non_gpu_allowance = cpu_memory_per_second * seconds
        lines: list[PlanLine] = [
            PlanLine(
                "image-allowance",
                "image_allowance",
                non_gpu_allowance,
                allowance_seconds=ALLOWANCE_SECONDS,
            ),
            PlanLine(
                "staging",
                "staging",
                non_gpu_allowance,
                allowance_seconds=ALLOWANCE_SECONDS,
            ),
        ]
        for cell in CELLS:
            gpu_rate = prices.rate(
                "l40s_gpu_second_usd"
                if cell.accelerator == "L40S"
                else "h100_gpu_second_usd"
            )
            lines.append(
                PlanLine(
                    f"cell-{cell.cell_id}",
                    "cell",
                    (gpu_rate + cpu_memory_per_second) * seconds,
                    allowance_seconds=ALLOWANCE_SECONDS,
                    cell_id=cell.cell_id,
                )
            )
        retained_gib = Decimal(EXPECTED_MODEL_BYTES) / Decimal(2**30)
        storage = (
            retained_gib
            * prices.rate("volume_gib_month_usd")
            * Decimal(POST_DELETE_STORAGE_DAYS)
            / Decimal(30)
        )
        lines.append(
            PlanLine(
                "storage",
                "storage",
                storage,
                retained_gib=retained_gib,
                retained_days=POST_DELETE_STORAGE_DAYS,
            )
        )
        return tuple(lines)


@dataclass(frozen=True)
class VLLMCompilePlan:
    prices: PricePins
    image_digest: str
    runtime_pins: RuntimePins
    validation_as_of_date: str
    max_price_age_days: int
    lines: tuple[PlanLine, ...]
    first_pass_usd: Decimal
    full_retry_usd: Decimal
    contingency_usd: Decimal
    envelope_usd: Decimal

    @classmethod
    def create(
        cls,
        *,
        prices: Mapping[str, str],
        effective_date: str,
        price_source: str,
        price_source_sha256: str,
        image_digest: str,
        runtime_pins: Mapping[str, str],
        as_of_date: str,
        max_price_age_days: int = MAX_PRICE_AGE_DAYS,
        hard_cap_usd: Decimal = HARD_CAP_USD,
    ) -> VLLMCompilePlan:
        if hard_cap_usd != HARD_CAP_USD:
            raise VLLMCompileContractError("the approved hard cap is exactly USD 28")
        if image_digest != OFFICIAL_VLLM_IMAGE_DIGEST:
            raise VLLMCompileContractError(
                "image digest must match the approved official vLLM 0.28.0 amd64 pin"
            )
        price_pins = PricePins.create(
            prices,
            effective_date=effective_date,
            source=price_source,
            source_sha256=price_source_sha256,
            as_of_date=as_of_date,
            max_age_days=max_price_age_days,
        )
        pins = RuntimePins.create(runtime_pins)
        lines = _plan_lines(price_pins)
        with localcontext() as context:
            context.prec = 28
            first = sum((line.amount_usd for line in lines), Decimal())
            retry = first
            if first + retry > hard_cap_usd:
                raise VLLMCompileContractError(
                    "approved first pass and full retry exceed the USD 28 envelope"
                )
            contingency = hard_cap_usd - first - retry
            envelope = first + retry + contingency
        return cls(
            prices=price_pins,
            image_digest=image_digest,
            runtime_pins=pins,
            validation_as_of_date=as_of_date,
            max_price_age_days=max_price_age_days,
            lines=lines,
            first_pass_usd=first,
            full_retry_usd=retry,
            contingency_usd=contingency,
            envelope_usd=envelope,
        )

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _PLAN_SCHEMA_VERSION,
            "experiment_id": "qwen3-8b-vllm-compile-break-even-v1",
            "model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "expected_file_count": EXPECTED_MODEL_FILE_COUNT,
                "expected_bytes": EXPECTED_MODEL_BYTES,
            },
            "hard_cap_usd": canonical_decimal(HARD_CAP_USD),
            "prices": self.prices.to_dict(),
            "pricing_sha256": self.prices.content_sha256,
            "validation_as_of_date": self.validation_as_of_date,
            "max_price_age_days": self.max_price_age_days,
            "image_digest": self.image_digest,
            "runtime_pins": self.runtime_pins.to_dict(),
            "cells": [cell.to_dict() for cell in CELLS],
            "workload": {
                "requests_per_cell": REQUESTS_PER_CELL,
                "warmups": 0,
                "contract_sha256": _sha256_json(
                    [item.to_dict() for item in workload_descriptors()]
                ),
            },
            "first_pass_lines": [line.to_dict() for line in self.lines],
            "first_pass_usd": canonical_decimal(self.first_pass_usd),
            "full_retry_usd": canonical_decimal(self.full_retry_usd),
            "contingency_usd": canonical_decimal(self.contingency_usd),
            "envelope_usd": canonical_decimal(self.envelope_usd),
        }

    @property
    def content_sha256(self) -> str:
        return _sha256_json(self._content_dict())

    def to_dict(self) -> dict[str, Any]:
        result = self._content_dict()
        result["plan_sha256"] = self.content_sha256
        return result

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, data: Any) -> VLLMCompilePlan:
        if not isinstance(data, dict):
            raise VLLMCompileContractError("plan must be an object")
        try:
            prices = data["prices"]
            plan = cls.create(
                prices=prices["rates"],
                effective_date=prices["effective_date"],
                price_source=prices["source"],
                price_source_sha256=prices["source_sha256"],
                image_digest=data["image_digest"],
                runtime_pins=data["runtime_pins"],
                as_of_date=data["validation_as_of_date"],
                max_price_age_days=data["max_price_age_days"],
            )
        except (KeyError, TypeError) as exc:
            raise VLLMCompileContractError(
                f"plan is missing a required pin: {exc}"
            ) from exc
        if data != plan.to_dict():
            raise VLLMCompileContractError(
                "plan does not exactly match its immutable canonical contract"
            )
        return plan

    @classmethod
    def from_json(cls, payload: str) -> VLLMCompilePlan:
        try:
            data = json.loads(payload, parse_constant=reject_non_finite_json_constant)
        except (ValueError, RecursionError) as exc:
            raise VLLMCompileContractError(f"invalid plan JSON: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def read_json(cls, path: str | Path) -> VLLMCompilePlan:
        try:
            payload = read_bounded_regular_text(path, MAX_METADATA_ARTIFACT_BYTES)
        except (OSError, ArtifactReadError) as exc:
            raise VLLMCompileContractError(f"failed to read plan: {exc}") from exc
        return cls.from_json(payload)


build_plan = VLLMCompilePlan.create


@dataclass(frozen=True)
class HardwareIdentity:
    gpu_name: str
    gpu_count: int


def validate_hardware_identity(
    cell: ExperimentCell, observed: HardwareIdentity
) -> None:
    """Refuse fallback/substitution hardware, including adjacent NVIDIA SKUs."""

    expected = EXPECTED_GPU_NAMES[cell.accelerator]
    normalized = " ".join(observed.gpu_name.split())
    name_matches = (
        normalized == expected
        if cell.accelerator == "L40S"
        else re.fullmatch(r"NVIDIA H100(?:[ -].+)?", normalized) is not None
    )
    if observed.gpu_count != cell.gpu_count or not name_matches:
        raise VLLMCompileContractError(
            f"{cell.cell_id} requires exactly {cell.gpu_count} {expected}; observed "
            f"{observed.gpu_count} x {observed.gpu_name!r}"
        )


def validate_model_identity(
    *, observed_revision: str, observed_file_count: int, observed_bytes: int
) -> None:
    """Refuse any staged source inventory that differs from the public pin."""

    if (
        observed_revision != MODEL_REVISION
        or observed_file_count != EXPECTED_MODEL_FILE_COUNT
        or observed_bytes != EXPECTED_MODEL_BYTES
    ):
        raise VLLMCompileContractError(
            "staged model identity does not match the approved revision inventory"
        )


@dataclass(frozen=True)
class WorkloadDescriptor:
    ordinal: int
    workload_id: str
    workload_version: str
    context_tier: str
    repetition: int
    prompt_sha256: str
    warmup: bool = False

    @property
    def request_id(self) -> str:
        return f"{self.context_tier}-{self.workload_id}-rep-{self.repetition:02d}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "request_id": self.request_id,
            "workload_id": self.workload_id,
            "workload_version": self.workload_version,
            "context_tier": self.context_tier,
            "repetition": self.repetition,
            "prompt_sha256": self.prompt_sha256,
            "warmup": self.warmup,
        }


def _packaged_prompt_hashes() -> dict[str, dict[str, str]]:
    try:
        text = read_bounded_regular_text(_PROMPT_MANIFEST, MAX_METADATA_ARTIFACT_BYTES)
        payload = json.loads(text, parse_constant=reject_non_finite_json_constant)
        workloads = payload["workloads"]
    except (OSError, ArtifactReadError, ValueError, RecursionError, KeyError) as exc:
        raise VLLMCompileContractError(
            f"packaged Qwen control manifest is unavailable or invalid: {exc}"
        ) from exc
    if not isinstance(workloads, list) or len(workloads) != len(WORKLOAD_IDS):
        raise VLLMCompileContractError(
            "packaged Qwen workloads do not match the contract"
        )
    result: dict[str, dict[str, str]] = {}
    for expected_id, item in zip(WORKLOAD_IDS, workloads, strict=True):
        if not isinstance(item, dict) or item.get("workload_id") != expected_id:
            raise VLLMCompileContractError(
                "packaged Qwen workload order does not match the contract"
            )
        hashes = item.get("prompt_hashes")
        if not isinstance(hashes, dict) or set(hashes) != set(CONTEXT_TIERS):
            raise VLLMCompileContractError("packaged prompt hashes are incomplete")
        result[expected_id] = dict(hashes)
    return result


def workload_descriptors(
    prompt_hashes: Mapping[str, Mapping[str, str]] | None = None,
    *,
    prompt_payloads: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[WorkloadDescriptor, ...]:
    """Build the exact tier/workload/repetition order, with no warmups.

    Custom hash pins are allowed for an explicitly supplied materialization.
    In that case ``prompt_payloads`` is mandatory and every UTF-8 payload must
    verify against its pin.  Without custom pins the packaged Qwen control
    manifest is the sole source.
    """

    packaged = _packaged_prompt_hashes()
    selected: Mapping[str, Mapping[str, str]]
    if prompt_hashes is None:
        if prompt_payloads is not None:
            raise VLLMCompileContractError(
                "prompt payloads require explicit prompt hash pins"
            )
        selected = packaged
    else:
        if prompt_payloads is None:
            raise VLLMCompileContractError(
                "supplied prompt hashes require payloads for verification"
            )
        selected = prompt_hashes
    if set(selected) != set(WORKLOAD_IDS):
        raise VLLMCompileContractError("prompt hashes must cover the two workloads")
    if prompt_payloads is not None and set(prompt_payloads) != set(WORKLOAD_IDS):
        raise VLLMCompileContractError("prompt payloads must cover the two workloads")

    descriptors: list[WorkloadDescriptor] = []
    for tier in CONTEXT_TIERS:
        for workload_id in WORKLOAD_IDS:
            hashes = selected[workload_id]
            if set(hashes) != set(CONTEXT_TIERS):
                raise VLLMCompileContractError(
                    f"prompt hashes for {workload_id} must cover every tier"
                )
            prompt_hash = hashes[tier]
            if not isinstance(prompt_hash, str) or not _SHA256.fullmatch(prompt_hash):
                raise VLLMCompileContractError(
                    "prompt hash must be sha256 lowercase hex"
                )
            if prompt_payloads is not None:
                payloads = prompt_payloads[workload_id]
                if set(payloads) != set(CONTEXT_TIERS):
                    raise VLLMCompileContractError(
                        f"prompt payloads for {workload_id} must cover every tier"
                    )
                actual = _sha256_text(payloads[tier])
                if actual != prompt_hash:
                    raise VLLMCompileContractError(
                        f"prompt payload hash mismatch for {workload_id}/{tier}"
                    )
            for repetition in (1, 2):
                descriptors.append(
                    WorkloadDescriptor(
                        ordinal=len(descriptors) + 1,
                        workload_id=workload_id,
                        workload_version="1",
                        context_tier=tier,
                        repetition=repetition,
                        prompt_sha256=prompt_hash,
                    )
                )
    if len(descriptors) != REQUESTS_PER_CELL:
        raise VLLMCompileContractError("workload descriptor count drifted from 12")
    return tuple(descriptors)


def _timestamp(value: Any, *, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise VLLMCompileContractError(f"{field} is required")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise VLLMCompileContractError(f"{field} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise VLLMCompileContractError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _optional_metric(value: Any, *, field: str) -> Decimal | None:
    if value is None:
        return None
    return _decimal(value, field=field)


@dataclass(frozen=True)
class TerminalRequest:
    request_id: str
    finish_reason: str
    token_ids: tuple[int, ...]
    started_at: str
    ended_at: str
    correctness: bool
    ttft_seconds: Decimal | None = None
    output_tokens_per_second: Decimal | None = None
    gpu_memory_gib: Decimal | None = None

    @classmethod
    def from_dict(cls, data: Any) -> TerminalRequest:
        if not isinstance(data, dict):
            raise VLLMCompileContractError("terminal request must be an object")
        request_id = data.get("request_id")
        finish_reason = data.get("finish_reason")
        token_ids = data.get("token_ids")
        correctness = data.get("correctness")
        if not isinstance(request_id, str) or not request_id:
            raise VLLMCompileContractError("request_id is required")
        if finish_reason not in TERMINAL_FINISH_REASONS:
            raise VLLMCompileContractError(
                "finish_reason must be a complete stop or bounded length terminal"
            )
        if (
            not isinstance(token_ids, list)
            or not token_ids
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in token_ids
            )
        ):
            raise VLLMCompileContractError(
                "token_ids must be a non-empty list of non-negative integers"
            )
        if not isinstance(correctness, bool):
            raise VLLMCompileContractError(
                "correctness must be terminal boolean evidence"
            )
        started = _timestamp(data.get("started_at"), field="started_at")
        ended = _timestamp(data.get("ended_at"), field="ended_at")
        if ended < started:
            raise VLLMCompileContractError("ended_at must not precede started_at")
        return cls(
            request_id=request_id,
            finish_reason=finish_reason,
            token_ids=tuple(token_ids),
            started_at=data["started_at"],
            ended_at=data["ended_at"],
            correctness=correctness,
            ttft_seconds=_optional_metric(
                data.get("ttft_seconds"), field="ttft_seconds"
            ),
            output_tokens_per_second=_optional_metric(
                data.get("output_tokens_per_second"),
                field="output_tokens_per_second",
            ),
            gpu_memory_gib=_optional_metric(
                data.get("gpu_memory_gib"), field="gpu_memory_gib"
            ),
        )

    @property
    def terminal(self) -> bool:
        return True

    @property
    def latency_seconds(self) -> Decimal:
        started = _timestamp(self.started_at, field="started_at")
        ended = _timestamp(self.ended_at, field="ended_at")
        delta = ended - started
        return Decimal(delta.days * 86_400 + delta.seconds) + Decimal(
            delta.microseconds
        ) / Decimal(1_000_000)

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "finish_reason": self.finish_reason,
            "token_ids": list(self.token_ids),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "correctness": self.correctness,
            "ttft_seconds": (
                canonical_decimal(self.ttft_seconds)
                if self.ttft_seconds is not None
                else None
            ),
            "output_tokens_per_second": (
                canonical_decimal(self.output_tokens_per_second)
                if self.output_tokens_per_second is not None
                else None
            ),
            "gpu_memory_gib": (
                canonical_decimal(self.gpu_memory_gib)
                if self.gpu_memory_gib is not None
                else None
            ),
        }


@dataclass(frozen=True)
class LatencyRecord:
    request_id: str
    latency_seconds: Decimal

    def __post_init__(self) -> None:
        if not self.request_id:
            raise VLLMCompileContractError("latency request_id is required")
        if (
            not isinstance(self.latency_seconds, Decimal)
            or not self.latency_seconds.is_finite()
            or self.latency_seconds < 0
            or abs(self.latency_seconds.adjusted()) > 128
        ):
            raise VLLMCompileContractError(
                "latency_seconds must be a finite non-negative Decimal"
            )


@dataclass(frozen=True)
class BreakEvenResult:
    observed_requests: int | None
    observed_lower_bound_requests: int | None
    extrapolated_requests: int | None
    full_cycle_saving_seconds: Decimal

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_requests": self.observed_requests,
            "observed_lower_bound_requests": self.observed_lower_bound_requests,
            "extrapolated_requests": self.extrapolated_requests,
            "full_cycle_saving_seconds": canonical_decimal(
                self.full_cycle_saving_seconds
            ),
            "extrapolation": "repeated_exact_12_request_cycle",
        }


def _latency_records(
    records: Sequence[LatencyRecord | TerminalRequest],
) -> tuple[LatencyRecord, ...]:
    converted = tuple(
        (
            item
            if isinstance(item, LatencyRecord)
            else LatencyRecord(item.request_id, item.latency_seconds)
        )
        for item in records
    )
    if len(converted) != REQUESTS_PER_CELL:
        raise VLLMCompileContractError(
            "break-even requires exactly 12 records per mode"
        )
    return converted


def calculate_break_even(
    eager_records: Sequence[LatencyRecord | TerminalRequest],
    compiled_records: Sequence[LatencyRecord | TerminalRequest],
    *,
    eager_cell: ExperimentCell,
    compiled_cell: ExperimentCell,
    compilation_overhead_seconds: Decimal = Decimal(),
) -> BreakEvenResult:
    """Calculate observed and repeated-cycle break-even without SSE inference."""

    if (
        eager_cell.accelerator != compiled_cell.accelerator
        or eager_cell.compile_enabled
        or not compiled_cell.compile_enabled
    ):
        raise VLLMCompileContractError(
            "break-even requires eager and compiled cells on the same accelerator"
        )
    eager = _latency_records(eager_records)
    compiled = _latency_records(compiled_records)
    if tuple(item.request_id for item in eager) != tuple(
        item.request_id for item in compiled
    ):
        raise VLLMCompileContractError("paired latency request identities/order differ")
    if (
        not isinstance(compilation_overhead_seconds, Decimal)
        or not compilation_overhead_seconds.is_finite()
        or compilation_overhead_seconds < 0
        or abs(compilation_overhead_seconds.adjusted()) > 128
    ):
        raise VLLMCompileContractError(
            "compilation overhead must be a finite non-negative Decimal"
        )
    with localcontext() as context:
        context.prec = 28
        prefix_savings: list[Decimal] = []
        cumulative = Decimal()
        observed: int | None = None
        for index, (eager_item, compiled_item) in enumerate(
            zip(eager, compiled, strict=True), start=1
        ):
            cumulative += eager_item.latency_seconds - compiled_item.latency_seconds
            prefix_savings.append(cumulative)
            if observed is None and cumulative >= compilation_overhead_seconds:
                observed = index
        cycle_saving = prefix_savings[-1]
        extrapolated: int | None = None
        if observed is None and cycle_saving > 0:
            # Find the first repeated cycle/prefix whose exact accumulated
            # saving repays the one-time compilation overhead.
            candidates: list[int] = []
            for prefix, saving in enumerate(prefix_savings, start=1):
                required = compilation_overhead_seconds - saving
                cycles = max(
                    0,
                    int(
                        (required / cycle_saving).to_integral_value(
                            rounding=ROUND_CEILING
                        )
                    ),
                )
                candidates.append(cycles * REQUESTS_PER_CELL + prefix)
            extrapolated = min(candidates)
        return BreakEvenResult(
            observed_requests=observed,
            observed_lower_bound_requests=(
                REQUESTS_PER_CELL if observed is None else None
            ),
            extrapolated_requests=extrapolated,
            full_cycle_saving_seconds=cycle_saving,
        )


def _event_hash(event: dict[str, Any]) -> str:
    material = dict(event)
    material.pop("event_sha256", None)
    return _sha256_json(material)


def _seal(payload: dict[str, Any]) -> dict[str, Any]:
    material = dict(payload)
    material.pop("ledger_sha256", None)
    material["ledger_sha256"] = _sha256_json(material)
    return material


@contextmanager
def _locked(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class LifecycleBudgetLedger:
    """Path-bound, atomically written, append-only reservation ledger."""

    def __init__(
        self,
        path: str | Path,
        *,
        plan: VLLMCompilePlan,
        git_head: str,
        workspace_path: str | Path,
    ) -> None:
        if not _GIT_HEAD.fullmatch(git_head):
            raise VLLMCompileContractError("git head must be an exact 40-hex commit")
        workspace = Path(workspace_path)
        if not workspace.exists() or not workspace.is_dir() or workspace.is_symlink():
            raise VLLMCompileContractError(
                "workspace path must be an existing non-symlink directory"
            )
        self.path = Path(path).resolve()
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")
        self.plan = plan
        self.git_head = git_head
        self.workspace_path = workspace.resolve()
        self._high_water_revision = 0
        self._high_water_reserved = Decimal()

    @classmethod
    def initialize(
        cls,
        path: str | Path,
        *,
        plan: VLLMCompilePlan,
        git_head: str,
        workspace_path: str | Path,
    ) -> LifecycleBudgetLedger:
        ledger = cls(path, plan=plan, git_head=git_head, workspace_path=workspace_path)
        with _locked(ledger.lock_path):
            if ledger.path.exists():
                raise VLLMCompileContractError(
                    "lifecycle ledger already exists and cannot be reset"
                )
            payload = ledger._initial_payload()
            ledger._write(payload)
        return ledger

    def _initial_payload(self) -> dict[str, Any]:
        return {
            "schema_version": _LEDGER_SCHEMA_VERSION,
            "revision": 0,
            "plan_sha256": self.plan.content_sha256,
            "git_head": self.git_head,
            "workspace_path_sha256": _sha256_text(str(self.workspace_path)),
            "ledger_path_sha256": _sha256_text(str(self.path)),
            "pricing_sha256": self.plan.prices.content_sha256,
            "hard_cap_usd": canonical_decimal(HARD_CAP_USD),
            "reserved_usd": "0",
            "remaining_usd": canonical_decimal(HARD_CAP_USD),
            "events": [],
        }

    def _write(self, payload: dict[str, Any]) -> None:
        sealed = _seal(payload)
        atomic_write_text(
            self.path,
            json.dumps(sealed, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )

    def _read(self) -> dict[str, Any]:
        try:
            text = read_bounded_regular_text(self.path, MAX_EVIDENCE_ARTIFACT_BYTES)
            payload = json.loads(text, parse_constant=reject_non_finite_json_constant)
        except (OSError, ArtifactReadError, ValueError, RecursionError) as exc:
            raise VLLMCompileContractError(
                f"failed to read lifecycle ledger: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise VLLMCompileContractError("lifecycle ledger must be an object")
        expected_seal = payload.get("ledger_sha256")
        if (
            not isinstance(expected_seal, str)
            or _seal(payload).get("ledger_sha256") != expected_seal
        ):
            raise VLLMCompileContractError(
                "lifecycle ledger integrity seal does not verify"
            )
        expected = self._initial_payload()
        for field in (
            "schema_version",
            "plan_sha256",
            "git_head",
            "workspace_path_sha256",
            "ledger_path_sha256",
            "pricing_sha256",
            "hard_cap_usd",
        ):
            if payload.get(field) != expected[field]:
                raise VLLMCompileContractError(
                    f"lifecycle ledger {field} binding does not match"
                )
        revision = payload.get("revision")
        events = payload.get("events")
        if (
            isinstance(revision, bool)
            or not isinstance(revision, int)
            or revision < 0
            or not isinstance(events, list)
            or revision != len(events)
        ):
            raise VLLMCompileContractError(
                "lifecycle ledger revision/event count is invalid"
            )
        previous: str | None = None
        ids: set[str] = set()
        total = Decimal()
        for index, event in enumerate(events, start=1):
            if not isinstance(event, dict):
                raise VLLMCompileContractError("lifecycle event must be an object")
            if event.get("index") != index or event.get("stage") != "pre_command":
                raise VLLMCompileContractError("lifecycle event order/stage is invalid")
            if event.get("previous_event_sha256") != previous:
                raise VLLMCompileContractError("lifecycle event chain is broken")
            if event.get("event_sha256") != _event_hash(event):
                raise VLLMCompileContractError("lifecycle event hash does not verify")
            command_id = event.get("command_id")
            if (
                not isinstance(command_id, str)
                or not _SAFE_ID.fullmatch(command_id)
                or command_id in ids
            ):
                raise VLLMCompileContractError(
                    "lifecycle event command identity is invalid or duplicated"
                )
            ids.add(command_id)
            line_id = event.get("line_id")
            matching_lines = [
                line for line in self.plan.lines if line.line_id == line_id
            ]
            if len(matching_lines) != 1:
                raise VLLMCompileContractError(
                    "lifecycle event line_id is not in the plan"
                )
            reserved = _decimal(
                event.get("reserved_usd"),
                field="lifecycle event reserved_usd",
                positive=True,
            )
            if reserved != matching_lines[0].amount_usd:
                raise VLLMCompileContractError(
                    "lifecycle event reservation differs from its plan line"
                )
            if (
                sum(prior.get("line_id") == line_id for prior in events[: index - 1])
                >= 2
            ):
                raise VLLMCompileContractError(
                    "lifecycle plan line is reserved more than twice"
                )
            total += reserved
            previous = event["event_sha256"]
        if total > HARD_CAP_USD:
            raise VLLMCompileContractError("lifecycle reservations exceed hard cap")
        if payload.get("reserved_usd") != canonical_decimal(total):
            raise VLLMCompileContractError("lifecycle reserved total does not verify")
        if payload.get("remaining_usd") != canonical_decimal(HARD_CAP_USD - total):
            raise VLLMCompileContractError("lifecycle remaining total does not verify")
        if revision < self._high_water_revision or total < self._high_water_reserved:
            raise VLLMCompileContractError("lifecycle ledger rollback detected")
        self._high_water_revision = revision
        self._high_water_reserved = total
        return payload

    def snapshot(self) -> dict[str, Any]:
        with _locked(self.lock_path):
            return self._read()

    def reserve(
        self,
        command_id: str,
        *,
        line_id: str,
        ceiling_usd: Decimal,
        argv: Sequence[str],
        reserved_at: str,
    ) -> dict[str, Any]:
        """Irreversibly reserve a ceiling before a command may be launched."""

        if not isinstance(command_id, str) or not _SAFE_ID.fullmatch(command_id):
            raise VLLMCompileContractError("command_id is not a safe stable identity")
        if (
            not isinstance(ceiling_usd, Decimal)
            or not ceiling_usd.is_finite()
            or ceiling_usd <= 0
        ):
            raise VLLMCompileContractError(
                "command ceiling must be a finite positive Decimal"
            )
        matching_lines = [line for line in self.plan.lines if line.line_id == line_id]
        if len(matching_lines) != 1:
            raise VLLMCompileContractError("reservation line_id is not in the plan")
        if ceiling_usd != matching_lines[0].amount_usd:
            raise VLLMCompileContractError(
                "command ceiling must exactly match its approved plan line"
            )
        if (
            isinstance(argv, (str, bytes))
            or not argv
            or any(not isinstance(item, str) or not item for item in argv)
        ):
            raise VLLMCompileContractError("argv must be non-empty immutable strings")
        _timestamp(reserved_at, field="reserved_at")
        with _locked(self.lock_path):
            payload = self._read()
            if any(event["command_id"] == command_id for event in payload["events"]):
                raise VLLMCompileContractError(
                    f"command {command_id!r} is already reserved"
                )
            if sum(event["line_id"] == line_id for event in payload["events"]) >= 2:
                raise VLLMCompileContractError(
                    "a plan line cannot be reserved more than once for first pass "
                    "and once for retry"
                )
            already = _decimal(payload["reserved_usd"], field="ledger reserved_usd")
            if already + ceiling_usd > HARD_CAP_USD:
                raise VLLMCompileContractError(
                    "pre-command reservation refused: USD 28 hard cap would be exceeded"
                )
            previous = (
                payload["events"][-1]["event_sha256"] if payload["events"] else None
            )
            event = {
                "index": len(payload["events"]) + 1,
                "command_id": command_id,
                "line_id": line_id,
                "stage": "pre_command",
                "reserved_usd": canonical_decimal(ceiling_usd),
                "reserved_at": reserved_at,
                "argv_sha256": _sha256_json(list(argv)),
                "previous_event_sha256": previous,
            }
            event["event_sha256"] = _event_hash(event)
            payload["events"].append(event)
            payload["revision"] += 1
            total = already + ceiling_usd
            payload["reserved_usd"] = canonical_decimal(total)
            payload["remaining_usd"] = canonical_decimal(HARD_CAP_USD - total)
            self._write(payload)
            return dict(event)


SealedLifecycleLedger = LifecycleBudgetLedger


__all__ = [
    "ALLOWANCE_SECONDS",
    "CELLS",
    "CONTEXT_TIERS",
    "CPU_CORES",
    "CURRENT_RATES",
    "EXPECTED_MODEL_BYTES",
    "EXPECTED_MODEL_FILE_COUNT",
    "HARD_CAP_USD",
    "HardwareIdentity",
    "LatencyRecord",
    "LifecycleBudgetLedger",
    "MEMORY_GIB",
    "MODEL_ID",
    "MODEL_REVISION",
    "OFFICIAL_VLLM_IMAGE_DIGEST",
    "POST_DELETE_STORAGE_DAYS",
    "PricePins",
    "REQUESTS_PER_CELL",
    "REQUIRED_RUNTIME_PINS",
    "RuntimePins",
    "SealedLifecycleLedger",
    "TerminalRequest",
    "TERMINAL_FINISH_REASONS",
    "VLLMCompileContractError",
    "VLLMCompilePlan",
    "WorkloadDescriptor",
    "build_plan",
    "calculate_break_even",
    "canonical_decimal",
    "canonical_json",
    "validate_hardware_identity",
    "validate_model_identity",
    "workload_descriptors",
]
