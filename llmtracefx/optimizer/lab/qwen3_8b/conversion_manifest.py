"""Strict, versioned self-conversion specification for Qwen3-8B.

Binds the exact official upstream source, the exact converter package
and its own upstream git revision, and the exact quantization
parameters used to produce a local MLX checkpoint. This module never
downloads anything or runs a conversion; it only parses and validates
the pinned specification (see ``conversion.py`` for execution).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ..._artifact_io import (
    MAX_METADATA_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)

CONVERSION_MANIFEST_SCHEMA_VERSION = "1"

#: Quantization modes ``mlx_lm.convert`` accepts for ``--q-mode``.
SUPPORTED_QUANT_MODES = ("affine", "mxfp4", "nvfp4", "mxfp8")

#: ``mlx_lm.convert --dtype`` accepts only these three values (or None,
#: meaning "use the source config's dtype"), see ``mlx_lm/convert.py``.
SUPPORTED_CONVERSION_DTYPES = ("float16", "bfloat16", "float32")


class ConversionManifestError(ValueError):
    """Raised when the self-conversion manifest is malformed or inconsistent."""


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConversionManifestError(f"{context} must be an object")
    return value


def _string(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ConversionManifestError(f"{context}.{key} must be a non-empty string")
    return value


def _optional_string(data: dict[str, Any], key: str, context: str) -> str | None:
    value = data.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise ConversionManifestError(
            f"{context}.{key} must be a non-empty string or null"
        )
    return value


def _integer(data: dict[str, Any], key: str, context: str, *, minimum: int = 0) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConversionManifestError(
            f"{context}.{key} must be an integer >= {minimum}"
        )
    return value


def _number(
    data: dict[str, Any],
    key: str,
    context: str,
    *,
    minimum: float,
    exclusive: bool = False,
) -> float:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConversionManifestError(f"{context}.{key} must be a number")
    numeric = float(value)
    valid = numeric > minimum if exclusive else numeric >= minimum
    if not math.isfinite(numeric) or not valid:
        operator = ">" if exclusive else ">="
        raise ConversionManifestError(
            f"{context}.{key} must be finite and {operator} {minimum}"
        )
    return numeric


def _boolean(data: dict[str, Any], key: str, context: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ConversionManifestError(f"{context}.{key} must be a boolean")
    return value


def _sha256(value: str, context: str) -> str:
    digest = value.removeprefix("sha256:")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ConversionManifestError(
            f"{context} must be a lowercase SHA-256 hex digest"
        )
    return digest


def _git_revision(value: str, context: str) -> str:
    if len(value) != 40 or any(c not in "0123456789abcdef" for c in value):
        raise ConversionManifestError(f"{context} must be a 40-character git revision")
    return value


def _relative_path(value: str, context: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ConversionManifestError(f"{context} must be a repository-relative path")
    return value


def _https_url(value: str, *, allowed_hosts: frozenset[str], context: str) -> str:
    parts = urlsplit(value)
    if (
        parts.scheme != "https"
        or not parts.netloc
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
        or parts.hostname not in allowed_hosts
        or "%" in parts.path
    ):
        raise ConversionManifestError(
            f"{context} must be a credential-free HTTPS URL on "
            f"{sorted(allowed_hosts)} without encoded paths, queries, or fragments"
        )
    return value


@dataclass(frozen=True)
class SourceFilePin:
    """One official-source file with exact size and SHA-256 pins."""

    path: str
    size_bytes: int
    sha256: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceFilePin:
        context = "source.files[]"
        path = _string(data, "path", context)
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise ConversionManifestError(
                "source.files[].path must be a repository-relative path"
            )
        return cls(
            path=path,
            size_bytes=_integer(data, "size_bytes", context, minimum=1),
            sha256=_sha256(_string(data, "sha256", context), f"{context}.sha256"),
        )


@dataclass(frozen=True)
class SourcePin:
    """Exact upstream identity for the official Qwen3-8B checkpoint."""

    official_id: str
    official_revision: str
    repository_id: str
    license: str
    expected_source_bytes: int
    files: tuple[SourceFilePin, ...]
    sources: tuple[str, ...]

    @property
    def fully_pinned(self) -> bool:
        return all(pin.sha256 is not None for pin in self.files)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourcePin:
        context = "source"
        files_raw = data.get("files")
        if not isinstance(files_raw, list) or not files_raw:
            raise ConversionManifestError("source.files must be a non-empty list")
        files = tuple(
            SourceFilePin.from_dict(_object(item, "source.files[]"))
            for item in files_raw
        )
        if len({pin.path for pin in files}) != len(files):
            raise ConversionManifestError("source.files paths must be unique")
        expected_source_bytes = _integer(
            data, "expected_source_bytes", context, minimum=1
        )
        if sum(pin.size_bytes for pin in files) != expected_source_bytes:
            raise ConversionManifestError(
                "source.expected_source_bytes must equal the pinned file sizes"
            )
        sources_raw = data.get("sources")
        if not isinstance(sources_raw, list) or not sources_raw:
            raise ConversionManifestError("source.sources must be a non-empty list")
        sources = tuple(
            _https_url(
                item,
                allowed_hosts=frozenset({"huggingface.co", "github.com"}),
                context="source.sources[]",
            )
            for item in sources_raw
            if isinstance(item, str)
        )
        if len(sources) != len(sources_raw):
            raise ConversionManifestError("source.sources[] must all be strings")
        return cls(
            official_id=_string(data, "official_id", context),
            official_revision=_git_revision(
                _string(data, "official_revision", context),
                "source.official_revision",
            ),
            repository_id=_string(data, "repository_id", context),
            license=_string(data, "license", context),
            expected_source_bytes=expected_source_bytes,
            files=files,
            sources=sources,
        )


@dataclass(frozen=True)
class ConverterPin:
    """Exact converter package/version and its own upstream git revision."""

    package: str
    version: str
    git_repository: str
    git_revision: str
    entrypoint: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConverterPin:
        context = "converter"
        return cls(
            package=_string(data, "package", context),
            version=_string(data, "version", context),
            git_repository=_https_url(
                _string(data, "git_repository", context),
                allowed_hosts=frozenset({"github.com"}),
                context="converter.git_repository",
            ),
            git_revision=_git_revision(
                _string(data, "git_revision", context), "converter.git_revision"
            ),
            entrypoint=_string(data, "entrypoint", context),
        )


@dataclass(frozen=True)
class ConversionParameters:
    """Explicit, deterministic ``mlx_lm.convert`` parameters.

    Every field mirrors an actual ``mlx_lm.convert`` keyword argument
    (see ``mlx_lm/convert.py``) so the recorded provenance and the
    literal subprocess argv this project builds can never silently
    diverge. ``dtype``/``quant_predicate`` of ``None`` mean "use the
    converter's own default for this source model", which is itself a
    deterministic function of the pinned converter version and source
    config -- never an unrecorded ambient default.
    """

    quantize: bool
    q_group_size: int
    q_bits: int
    q_mode: str
    dtype: str | None
    quant_predicate: str | None
    dequantize: bool
    trust_remote_code: bool
    upload_repo: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversionParameters:
        context = "parameters"
        quantize = _boolean(data, "quantize", context)
        if not quantize:
            raise ConversionManifestError(
                "parameters.quantize must be true for this self-conversion control"
            )
        dequantize = _boolean(data, "dequantize", context)
        if dequantize:
            raise ConversionManifestError("parameters.dequantize must be false")
        trust_remote_code = _boolean(data, "trust_remote_code", context)
        if trust_remote_code:
            raise ConversionManifestError(
                "parameters.trust_remote_code must stay false"
            )
        upload_repo = _optional_string(data, "upload_repo", context)
        if upload_repo is not None:
            raise ConversionManifestError(
                "parameters.upload_repo must be null; this control never "
                "uploads the converted checkpoint anywhere"
            )
        q_mode = _string(data, "q_mode", context)
        if q_mode not in SUPPORTED_QUANT_MODES:
            raise ConversionManifestError(
                f"parameters.q_mode must be one of {SUPPORTED_QUANT_MODES}"
            )
        dtype = _optional_string(data, "dtype", context)
        if dtype is not None and dtype not in SUPPORTED_CONVERSION_DTYPES:
            raise ConversionManifestError(
                f"parameters.dtype must be null or one of {SUPPORTED_CONVERSION_DTYPES}"
            )
        return cls(
            quantize=quantize,
            q_group_size=_integer(data, "q_group_size", context, minimum=1),
            q_bits=_integer(data, "q_bits", context, minimum=1),
            q_mode=q_mode,
            dtype=dtype,
            quant_predicate=_optional_string(data, "quant_predicate", context),
            dequantize=dequantize,
            trust_remote_code=trust_remote_code,
            upload_repo=upload_repo,
        )

    def argv(self, *, hf_path: str, mlx_path: str) -> tuple[str, ...]:
        """The exact, deterministic ``mlx_lm convert`` argv this control runs."""
        argv = [
            "convert",
            "--hf-path",
            hf_path,
            "--mlx-path",
            mlx_path,
            "--quantize",
            "--q-group-size",
            str(self.q_group_size),
            "--q-bits",
            str(self.q_bits),
            "--q-mode",
            self.q_mode,
        ]
        if self.dtype is not None:
            argv.extend(("--dtype", self.dtype))
        if self.quant_predicate is not None:
            argv.extend(("--quant-predicate", self.quant_predicate))
        return tuple(argv)


@dataclass(frozen=True)
class ConversionSafety:
    """Conservative preflight thresholds for the conversion process only.

    ``minimum_residual_free_disk_bytes`` is the free-disk floor that must
    remain *after* the largest write this conversion could still need to
    perform: the official source download (only when not already cached)
    plus the self-converted output. ``assess_conversion_safety`` adds
    ``ConversionManifest.expected_output_bytes`` (and, only when a
    download is actually still needed, ``source.expected_source_bytes``)
    on top of this floor -- so a missing-source preflight requires
    exactly ``source + output + residual`` bytes free, never less.
    """

    required_chip: str
    required_total_memory_bytes: int
    minimum_residual_free_disk_bytes: int
    minimum_memory_free_percent: float
    maximum_swap_used_bytes: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversionSafety:
        context = "safety"
        memory_percent = _number(
            data, "minimum_memory_free_percent", context, minimum=0
        )
        if memory_percent > 100:
            raise ConversionManifestError(
                "safety.minimum_memory_free_percent must be <= 100"
            )
        return cls(
            required_chip=_string(data, "required_chip", context),
            required_total_memory_bytes=_integer(
                data, "required_total_memory_bytes", context, minimum=1
            ),
            minimum_residual_free_disk_bytes=_integer(
                data, "minimum_residual_free_disk_bytes", context, minimum=1
            ),
            minimum_memory_free_percent=memory_percent,
            maximum_swap_used_bytes=_integer(
                data, "maximum_swap_used_bytes", context, minimum=1
            ),
        )


@dataclass(frozen=True)
class ConversionArtifacts:
    """Repo-relative cache/workspace paths, fully separate from the 27B lab."""

    source_cache: str
    output_cache: str
    workspace: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversionArtifacts:
        context = "artifacts"
        values = {
            key: _relative_path(_string(data, key, context), f"artifacts.{key}")
            for key in ("source_cache", "output_cache", "workspace")
        }
        return cls(**values)


@dataclass(frozen=True)
class ConversionManifest:
    schema_version: str
    conversion_id: str
    source: SourcePin
    converter: ConverterPin
    parameters: ConversionParameters
    expected_output_bytes: int
    safety: ConversionSafety
    timeout_seconds: float
    cleanup_grace_seconds: float
    max_journal_bytes: int
    artifacts: ConversionArtifacts

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversionManifest:
        data = _object(data, "manifest")
        schema_version = _string(data, "schema_version", "manifest")
        if schema_version != CONVERSION_MANIFEST_SCHEMA_VERSION:
            raise ConversionManifestError(
                f"unsupported conversion manifest schema_version {schema_version!r}"
            )
        return cls(
            schema_version=schema_version,
            conversion_id=_string(data, "conversion_id", "manifest"),
            source=SourcePin.from_dict(_object(data.get("source"), "source")),
            converter=ConverterPin.from_dict(
                _object(data.get("converter"), "converter")
            ),
            parameters=ConversionParameters.from_dict(
                _object(data.get("parameters"), "parameters")
            ),
            expected_output_bytes=_integer(
                data, "expected_output_bytes", "manifest", minimum=1
            ),
            safety=ConversionSafety.from_dict(_object(data.get("safety"), "safety")),
            timeout_seconds=_number(
                data, "timeout_seconds", "manifest", minimum=0, exclusive=True
            ),
            cleanup_grace_seconds=_number(
                data, "cleanup_grace_seconds", "manifest", minimum=0, exclusive=True
            ),
            max_journal_bytes=_integer(
                data, "max_journal_bytes", "manifest", minimum=1024
            ),
            artifacts=ConversionArtifacts.from_dict(
                _object(data.get("artifacts"), "artifacts")
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> ConversionManifest:
        try:
            data = json.loads(payload, parse_constant=reject_non_finite_json_constant)
        except (ValueError, RecursionError) as exc:
            raise ConversionManifestError(
                f"invalid conversion manifest JSON: {exc}"
            ) from exc
        return cls.from_dict(data)

    @classmethod
    def read_json(cls, path: str | Path) -> ConversionManifest:
        try:
            payload = read_bounded_regular_text(path, MAX_METADATA_ARTIFACT_BYTES)
        except ArtifactReadError as exc:
            raise ConversionManifestError(
                f"invalid conversion manifest file: {exc}"
            ) from exc
        return cls.from_json(payload)
