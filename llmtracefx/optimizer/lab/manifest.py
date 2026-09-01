"""Strict, versioned configuration for the M5 Pro local inference lab."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .._artifact_io import (
    MAX_METADATA_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)

LAB_MANIFEST_SCHEMA_VERSION = "1"


class LabManifestError(ValueError):
    """Raised when a lab manifest is malformed or internally inconsistent."""


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LabManifestError(f"{context} must be an object")
    return value


def _string(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise LabManifestError(f"{context}.{key} must be a non-empty string")
    return value


def _integer(data: dict[str, Any], key: str, context: str, *, minimum: int = 0) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise LabManifestError(f"{context}.{key} must be an integer >= {minimum}")
    return value


def _number(
    data: dict[str, Any],
    key: str,
    context: str,
    *,
    minimum: float = 0.0,
    exclusive_minimum: bool = False,
) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabManifestError(f"{context}.{key} must be a number")
    numeric = float(value)
    valid_minimum = numeric > minimum if exclusive_minimum else numeric >= minimum
    if not math.isfinite(numeric) or not valid_minimum:
        operator = ">" if exclusive_minimum else ">="
        raise LabManifestError(
            f"{context}.{key} must be finite and {operator} {minimum}"
        )
    return numeric


def _boolean(data: dict[str, Any], key: str, context: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise LabManifestError(f"{context}.{key} must be a boolean")
    return value


def _sha256(value: str, context: str) -> str:
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise LabManifestError(f"{context} must be a lowercase SHA-256 hex digest")
    return value


def _git_revision(value: str, context: str) -> str:
    if len(value) != 40 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise LabManifestError(f"{context} must be a 40-character git revision")
    return value


@dataclass(frozen=True)
class ModelFilePin:
    path: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelFilePin:
        context = "model.files[]"
        path = _string(data, "path", context)
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise LabManifestError(
                "model.files[].path must be a repository-relative path"
            )
        return cls(
            path=path,
            size_bytes=_integer(data, "size_bytes", context, minimum=1),
            sha256=_sha256(_string(data, "sha256", context), f"{context}.sha256"),
        )


@dataclass(frozen=True)
class ModelPin:
    official_id: str
    official_revision: str
    repository_id: str
    revision: str
    license: str
    quantization: str
    converter: str
    converter_revision: str
    expected_download_bytes: int
    files: tuple[ModelFilePin, ...]
    sources: tuple[str, ...]
    model_family: str = "qwen3_5"
    """The mlx-lm/mlx-vlm model-family identifier this pin's checkpoint
    loads as (for example ``mlx_lm/models/qwen3_5.py``'s ``qwen3_5``, or
    plain ``qwen3`` for a dense non-MTP mlx-lm checkpoint such as a
    self-converted Qwen3-8B). Optional and defaulted to the historical
    ``"qwen3_5"`` so every manifest packaged before this field existed
    keeps parsing to the exact identity it always resolved to."""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelPin:
        context = "model"
        files_raw = data.get("files")
        if not isinstance(files_raw, list) or not files_raw:
            raise LabManifestError("model.files must be a non-empty list")
        files = tuple(
            ModelFilePin.from_dict(_object(item, "model.files[]")) for item in files_raw
        )
        sources_raw = data.get("sources")
        if (
            not isinstance(sources_raw, list)
            or not sources_raw
            or not all(
                isinstance(source, str)
                and (parts := urlsplit(source)).scheme == "https"
                and bool(parts.netloc)
                and parts.username is None
                and parts.password is None
                and not parts.query
                and not parts.fragment
                and parts.hostname in {"huggingface.co", "github.com"}
                and "%" not in parts.path
                for source in sources_raw
            )
        ):
            raise LabManifestError(
                "model.sources must be credential-free Hugging Face or GitHub "
                "HTTPS URLs without encoded paths, queries, or fragments"
            )
        if len({item.path for item in files}) != len(files):
            raise LabManifestError("model.files paths must be unique")
        expected_download_bytes = _integer(
            data, "expected_download_bytes", context, minimum=1
        )
        if sum(item.size_bytes for item in files) != expected_download_bytes:
            raise LabManifestError(
                "model.expected_download_bytes must equal the pinned file sizes"
            )
        model_family_raw = data.get("model_family", "qwen3_5")
        if not isinstance(model_family_raw, str) or not model_family_raw:
            raise LabManifestError("model.model_family must be a non-empty string")
        return cls(
            official_id=_string(data, "official_id", context),
            official_revision=_git_revision(
                _string(data, "official_revision", context),
                "model.official_revision",
            ),
            repository_id=_string(data, "repository_id", context),
            revision=_git_revision(
                _string(data, "revision", context), "model.revision"
            ),
            license=_string(data, "license", context),
            quantization=_string(data, "quantization", context),
            converter=_string(data, "converter", context),
            converter_revision=_git_revision(
                _string(data, "converter_revision", context),
                "model.converter_revision",
            ),
            expected_download_bytes=expected_download_bytes,
            files=files,
            sources=tuple(sources_raw),
            model_family=model_family_raw,
        )


@dataclass(frozen=True)
class RuntimePin:
    name: str
    version: str
    mlx_version: str
    mlx_lm_version: str
    transformers_version: str
    prefill_step_size: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RuntimePin:
        context = "runtime"
        return cls(
            name=_string(data, "name", context),
            version=_string(data, "version", context),
            mlx_version=_string(data, "mlx_version", context),
            mlx_lm_version=_string(data, "mlx_lm_version", context),
            transformers_version=_string(data, "transformers_version", context),
            prefill_step_size=_integer(data, "prefill_step_size", context, minimum=1),
        )


@dataclass(frozen=True)
class GenerationConfig:
    max_output_tokens: int
    seed: int
    temperature: float
    top_p: float
    enable_thinking: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerationConfig:
        context = "generation"
        top_p = _number(data, "top_p", context, exclusive_minimum=True)
        if top_p > 1:
            raise LabManifestError("generation.top_p must be <= 1")
        return cls(
            max_output_tokens=_integer(data, "max_output_tokens", context, minimum=1),
            seed=_integer(data, "seed", context),
            temperature=_number(data, "temperature", context),
            top_p=top_p,
            enable_thinking=_boolean(data, "enable_thinking", context),
        )


@dataclass(frozen=True)
class RepetitionConfig:
    warmup_per_tier: int
    measured_per_workload: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RepetitionConfig:
        context = "repetitions"
        return cls(
            warmup_per_tier=_integer(data, "warmup_per_tier", context),
            measured_per_workload=_integer(
                data, "measured_per_workload", context, minimum=1
            ),
        )


@dataclass(frozen=True)
class SafetyConfig:
    required_chip: str
    required_total_memory_bytes: int
    minimum_free_disk_after_download_bytes: int
    minimum_memory_free_percent: float
    maximum_peak_memory_bytes: int
    maximum_swap_used_bytes: int
    stop_on_any_failed_row: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SafetyConfig:
        context = "safety"
        memory_percent = _number(data, "minimum_memory_free_percent", context)
        if memory_percent > 100:
            raise LabManifestError("safety.minimum_memory_free_percent must be <= 100")
        return cls(
            required_chip=_string(data, "required_chip", context),
            required_total_memory_bytes=_integer(
                data, "required_total_memory_bytes", context, minimum=1
            ),
            minimum_free_disk_after_download_bytes=_integer(
                data,
                "minimum_free_disk_after_download_bytes",
                context,
                minimum=1,
            ),
            minimum_memory_free_percent=memory_percent,
            maximum_peak_memory_bytes=_integer(
                data, "maximum_peak_memory_bytes", context, minimum=1
            ),
            maximum_swap_used_bytes=_integer(
                data, "maximum_swap_used_bytes", context, minimum=1
            ),
            stop_on_any_failed_row=_boolean(data, "stop_on_any_failed_row", context),
        )


@dataclass(frozen=True)
class WorkloadPin:
    workload_id: str
    version: str
    prompt_hashes: dict[str, str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkloadPin:
        context = "workloads[]"
        raw_hashes = data.get("prompt_hashes")
        if not isinstance(raw_hashes, dict) or not raw_hashes:
            raise LabManifestError(
                "workloads[].prompt_hashes must be a non-empty object"
            )
        hashes: dict[str, str] = {}
        for tier, digest in raw_hashes.items():
            if not isinstance(tier, str) or not isinstance(digest, str):
                raise LabManifestError(
                    "workloads[].prompt_hashes must map strings to strings"
                )
            hashes[tier] = _sha256(digest, f"workloads[].prompt_hashes.{tier}")
        return cls(
            workload_id=_string(data, "workload_id", context),
            version=_string(data, "version", context),
            prompt_hashes=hashes,
        )


@dataclass(frozen=True)
class ContextTierPin:
    name: str
    target_tokens: int
    order: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextTierPin:
        context = "context_tiers[]"
        return cls(
            name=_string(data, "name", context),
            target_tokens=_integer(data, "target_tokens", context, minimum=1),
            order=_integer(data, "order", context),
        )


@dataclass(frozen=True)
class ArtifactConfig:
    workspace: str
    model_cache: str
    shareable_example_dir: str
    commit_raw_artifacts: bool

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactConfig:
        context = "artifacts"
        values = {
            key: _string(data, key, context)
            for key in ("workspace", "model_cache", "shareable_example_dir")
        }
        for key, value in values.items():
            if Path(value).is_absolute() or ".." in Path(value).parts:
                raise LabManifestError(
                    f"artifacts.{key} must be a repository-relative path"
                )
        commit_raw = _boolean(data, "commit_raw_artifacts", context)
        if commit_raw:
            raise LabManifestError("artifacts.commit_raw_artifacts must remain false")
        return cls(
            workspace=values["workspace"],
            model_cache=values["model_cache"],
            shareable_example_dir=values["shareable_example_dir"],
            commit_raw_artifacts=commit_raw,
        )


@dataclass(frozen=True)
class LabManifest:
    schema_version: str
    lab_id: str
    model: ModelPin
    runtime: RuntimePin
    generation: GenerationConfig
    repetitions: RepetitionConfig
    cooperative_timeout_seconds: float
    safety: SafetyConfig
    workloads: tuple[WorkloadPin, ...]
    context_tiers: tuple[ContextTierPin, ...]
    artifacts: ArtifactConfig
    environment_capture: tuple[str, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LabManifest:
        data = _object(data, "manifest")
        schema_version = _string(data, "schema_version", "manifest")
        if schema_version != LAB_MANIFEST_SCHEMA_VERSION:
            raise LabManifestError(
                f"unsupported lab manifest schema_version {schema_version!r}"
            )
        workloads_raw = data.get("workloads")
        tiers_raw = data.get("context_tiers")
        if not isinstance(workloads_raw, list) or not workloads_raw:
            raise LabManifestError("workloads must be a non-empty list")
        if not isinstance(tiers_raw, list) or not tiers_raw:
            raise LabManifestError("context_tiers must be a non-empty list")
        workloads = tuple(
            WorkloadPin.from_dict(_object(item, "workloads[]"))
            for item in workloads_raw
        )
        tiers = tuple(
            ContextTierPin.from_dict(_object(item, "context_tiers[]"))
            for item in tiers_raw
        )
        tier_names = [tier.name for tier in tiers]
        if len(set(tier_names)) != len(tier_names):
            raise LabManifestError("context tier names must be unique")
        if [tier.order for tier in tiers] != sorted(tier.order for tier in tiers):
            raise LabManifestError("context_tiers must be ordered by ascending order")
        if len({workload.workload_id for workload in workloads}) != len(workloads):
            raise LabManifestError("workload IDs must be unique")
        for workload in workloads:
            if set(workload.prompt_hashes) != set(tier_names):
                raise LabManifestError(
                    f"workload {workload.workload_id!r} must pin every context tier"
                )
        capture_raw = data.get("environment_capture")
        if (
            not isinstance(capture_raw, list)
            or not capture_raw
            or not all(isinstance(item, str) and item for item in capture_raw)
        ):
            raise LabManifestError(
                "environment_capture must be a non-empty list of strings"
            )
        forbidden_capture = {
            "serial",
            "serial_number",
            "hardware_uuid",
            "hostname",
            "username",
            "environment_variables",
        }
        if forbidden_capture.intersection(item.lower() for item in capture_raw):
            raise LabManifestError("environment_capture includes a private field")
        return cls(
            schema_version=schema_version,
            lab_id=_string(data, "lab_id", "manifest"),
            model=ModelPin.from_dict(_object(data.get("model"), "model")),
            runtime=RuntimePin.from_dict(_object(data.get("runtime"), "runtime")),
            generation=GenerationConfig.from_dict(
                _object(data.get("generation"), "generation")
            ),
            repetitions=RepetitionConfig.from_dict(
                _object(data.get("repetitions"), "repetitions")
            ),
            cooperative_timeout_seconds=_number(
                data,
                "cooperative_timeout_seconds",
                "manifest",
                exclusive_minimum=True,
            ),
            safety=SafetyConfig.from_dict(_object(data.get("safety"), "safety")),
            workloads=workloads,
            context_tiers=tiers,
            artifacts=ArtifactConfig.from_dict(
                _object(data.get("artifacts"), "artifacts")
            ),
            environment_capture=tuple(capture_raw),
        )

    @classmethod
    def from_json(cls, payload: str) -> LabManifest:
        try:
            data = json.loads(payload, parse_constant=reject_non_finite_json_constant)
        except (ValueError, RecursionError) as exc:
            raise LabManifestError(f"invalid lab manifest JSON: {exc}") from exc
        return cls.from_dict(data)

    @classmethod
    def read_json(cls, path: str | Path) -> LabManifest:
        try:
            payload = read_bounded_regular_text(path, MAX_METADATA_ARTIFACT_BYTES)
        except ArtifactReadError as exc:
            raise LabManifestError(f"invalid lab manifest file: {exc}") from exc
        return cls.from_json(payload)

    def tier(self, name: str) -> ContextTierPin:
        for tier in self.context_tiers:
            if tier.name == name:
                return tier
        raise LabManifestError(f"unknown context tier {name!r}")
