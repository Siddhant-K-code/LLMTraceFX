"""Deterministic workload matrix generation.

Generates the set of (workload, context tier, decode mode) combinations
this project can plan for a given target model, without ever executing
anything or downloading model weights. Each entry carries the exact
planned CLI invocation (reusing the existing ``collect-mlx`` /
``native-mtp collect`` commands) and a ready-to-run
``llmtracefx.optimizer.runner.RunnerConfig``-compatible JSON file, so the
PR #3 runner can execute it once local checkpoints are available.

Native-MTP rows are always included for visibility, but marked
``runnable=False`` with an explicit ``unsupported_reason`` whenever
capability detection (see ``collectors.native_mtp``) reports the
runtime cannot produce trustworthy native-MTP evidence for the
requested model family -- never silently omitted, never fabricated as
runnable.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._artifact_io import (
    MAX_EVIDENCE_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from ..collectors._shared import atomic_write_text
from ..collectors.native_mtp import detect_native_mtp_capability
from .catalog import WORKLOADS
from .materialize import MaterializedPrompt, materialize_prompt
from .schema import ContextTier, Workload

MATRIX_SCHEMA_VERSION = "1"

#: Default MTP block depths planned for when a runtime does support
#: native MTP. Matches the depths already accepted by
#: ``NativeMTPCollectionConfig.configured_depth`` and by generic
#: draft-model speculation's ``num_draft_tokens``.
DEFAULT_MTP_DEPTHS: tuple[int, ...] = (2, 4)

DECODE_MODE_AUTOREGRESSIVE = "autoregressive"
DECODE_MODE_NATIVE_MTP = "native-mtp"

DEFAULT_MAX_TOKENS = 128


class MatrixSchemaError(ValueError):
    """Raised when a persisted matrix manifest is malformed."""


def _required_string(data: dict[str, Any], key: str, *, context: str) -> str:
    try:
        value = data[key]
    except KeyError as exc:
        raise MatrixSchemaError(
            f"{context} is missing required field: '{key}'"
        ) from exc
    if not isinstance(value, str):
        raise MatrixSchemaError(f"{context}.{key} must be a string")
    return value


def _optional_string(data: dict[str, Any], key: str, *, context: str) -> str | None:
    value = data.get(key)
    if value is not None and not isinstance(value, str):
        raise MatrixSchemaError(f"{context}.{key} must be a string or null")
    return value


def _integer(data: dict[str, Any], key: str, *, context: str, default: int) -> int:
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise MatrixSchemaError(f"{context}.{key} must be an integer")
    return value


@dataclass(frozen=True)
class MatrixEntry:
    """One planned (workload, context tier, decode mode) combination."""

    run_id: str
    workload_id: str
    workload_version: str
    category: str
    context_tier: str
    decode_mode: str
    configured_depth: int | None
    prompt: MaterializedPrompt
    prompt_path: str
    runner_results_dir: str
    collector_output_dir: str
    runnable: bool
    unsupported_reason: str | None
    command_argv: tuple[str, ...]
    max_tokens: int = DEFAULT_MAX_TOKENS

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "workload_id": self.workload_id,
            "workload_version": self.workload_version,
            "category": self.category,
            "context_tier": self.context_tier,
            "decode_mode": self.decode_mode,
            "configured_depth": self.configured_depth,
            "prompt": self.prompt.to_dict(),
            "prompt_path": self.prompt_path,
            "runner_results_dir": self.runner_results_dir,
            "collector_output_dir": self.collector_output_dir,
            "runnable": self.runnable,
            "unsupported_reason": self.unsupported_reason,
            "command_argv": list(self.command_argv),
            "max_tokens": self.max_tokens,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatrixEntry:
        if not isinstance(data, dict):
            raise MatrixSchemaError(
                f"MatrixEntry must be an object, got {type(data).__name__}"
            )
        try:
            command_argv = data["command_argv"]
            if isinstance(command_argv, (str, bytes)) or not isinstance(
                command_argv, Sequence
            ):
                raise MatrixSchemaError(
                    f"MatrixEntry.command_argv must be a list of strings, "
                    f"got {command_argv!r}"
                )
            if not all(isinstance(item, str) and item for item in command_argv):
                raise MatrixSchemaError(
                    "MatrixEntry.command_argv must contain non-empty strings"
                )
            configured_depth = data.get("configured_depth")
            if configured_depth is not None and (
                isinstance(configured_depth, bool)
                or not isinstance(configured_depth, int)
            ):
                raise MatrixSchemaError(
                    "MatrixEntry.configured_depth must be an integer or null"
                )
            runnable = data["runnable"]
            if not isinstance(runnable, bool):
                raise MatrixSchemaError("MatrixEntry.runnable must be a boolean")
            return cls(
                run_id=_required_string(data, "run_id", context="MatrixEntry"),
                workload_id=_required_string(
                    data, "workload_id", context="MatrixEntry"
                ),
                workload_version=_required_string(
                    data, "workload_version", context="MatrixEntry"
                ),
                category=_required_string(data, "category", context="MatrixEntry"),
                context_tier=_required_string(
                    data, "context_tier", context="MatrixEntry"
                ),
                decode_mode=_required_string(
                    data, "decode_mode", context="MatrixEntry"
                ),
                configured_depth=configured_depth,
                prompt=MaterializedPrompt.from_dict(data["prompt"]),
                prompt_path=_required_string(
                    data, "prompt_path", context="MatrixEntry"
                ),
                runner_results_dir=_required_string(
                    data, "runner_results_dir", context="MatrixEntry"
                ),
                collector_output_dir=_required_string(
                    data, "collector_output_dir", context="MatrixEntry"
                ),
                runnable=runnable,
                unsupported_reason=_optional_string(
                    data, "unsupported_reason", context="MatrixEntry"
                ),
                command_argv=tuple(command_argv),
                max_tokens=_integer(
                    data,
                    "max_tokens",
                    context="MatrixEntry",
                    default=DEFAULT_MAX_TOKENS,
                ),
            )
        except KeyError as exc:
            raise MatrixSchemaError(
                f"MatrixEntry is missing required field: {exc}"
            ) from exc
        except (TypeError, ValueError) as exc:
            if isinstance(exc, MatrixSchemaError):
                raise
            raise MatrixSchemaError(f"invalid MatrixEntry: {exc}") from exc


@dataclass(frozen=True)
class MatrixManifest:
    """The full deterministic matrix for one target model/family."""

    schema_version: str
    model_id: str
    model_family: str
    output_dir: str
    entries: tuple[MatrixEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_family": self.model_family,
            "output_dir": self.output_dir,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MatrixManifest:
        if not isinstance(data, dict):
            raise MatrixSchemaError(
                f"MatrixManifest must be an object, got {type(data).__name__}"
            )
        try:
            entries_raw = data["entries"]
            if not isinstance(entries_raw, list):
                raise MatrixSchemaError("MatrixManifest.entries must be a list")
            return cls(
                schema_version=_required_string(
                    {
                        "schema_version": data.get(
                            "schema_version", MATRIX_SCHEMA_VERSION
                        )
                    },
                    "schema_version",
                    context="MatrixManifest",
                ),
                model_id=_required_string(data, "model_id", context="MatrixManifest"),
                model_family=_required_string(
                    data, "model_family", context="MatrixManifest"
                ),
                output_dir=_required_string(
                    data, "output_dir", context="MatrixManifest"
                ),
                entries=tuple(MatrixEntry.from_dict(entry) for entry in entries_raw),
            )
        except KeyError as exc:
            raise MatrixSchemaError(
                f"MatrixManifest is missing required field: {exc}"
            ) from exc

    @classmethod
    def from_json(cls, payload: str) -> MatrixManifest:
        try:
            data = json.loads(payload, parse_constant=reject_non_finite_json_constant)
        except (ValueError, RecursionError) as exc:
            raise MatrixSchemaError(f"Invalid JSON for MatrixManifest: {exc}") from exc
        if not isinstance(data, dict):
            raise MatrixSchemaError("MatrixManifest JSON must be an object")
        return cls.from_dict(data)

    @classmethod
    def read_json(cls, path: str | Path) -> MatrixManifest:
        try:
            payload = read_bounded_regular_text(path, MAX_EVIDENCE_ARTIFACT_BYTES)
        except ArtifactReadError as exc:
            raise MatrixSchemaError(f"Invalid MatrixManifest file: {exc}") from exc
        return cls.from_json(payload)


def _collect_mlx_argv(
    *,
    run_id: str,
    model_path: str,
    model_id: str,
    prompt_path: str,
    output_dir: str,
    max_tokens: int,
) -> tuple[str, ...]:
    return (
        "llmtracefx-optimizer",
        "collect-mlx",
        "--run-id",
        run_id,
        "--model-path",
        model_path,
        "--model-id",
        model_id,
        "--prompt-file",
        prompt_path,
        "--output-dir",
        output_dir,
        "--max-tokens",
        str(max_tokens),
    )


def _native_mtp_collect_argv(
    *,
    run_id: str,
    target_model_path: str,
    mtp_sidecar_path: str,
    model_id: str,
    prompt_path: str,
    output_dir: str,
    max_tokens: int,
    configured_depth: int,
) -> tuple[str, ...]:
    return (
        "llmtracefx-optimizer",
        "native-mtp",
        "collect",
        "--run-id",
        run_id,
        "--target-model-path",
        target_model_path,
        "--mtp-sidecar-path",
        mtp_sidecar_path,
        "--model-id",
        model_id,
        "--prompt-file",
        prompt_path,
        "--output-dir",
        output_dir,
        "--max-tokens",
        str(max_tokens),
        "--configured-depth",
        str(configured_depth),
    )


def generate_matrix(
    *,
    model_id: str,
    model_family: str,
    output_dir: str,
    target_model_path: str | None = None,
    mtp_sidecar_path: str | None = None,
    workloads: Sequence[Workload] = WORKLOADS,
    context_tiers: Sequence[ContextTier] = tuple(ContextTier),
    mtp_depths: Sequence[int] = DEFAULT_MTP_DEPTHS,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> MatrixManifest:
    """Deterministically build the full planned matrix.

    Purely computational: materializes prompts and derives planned
    commands/paths from the given identifiers, but performs no file
    I/O, no model loading, and no downloads. ``output_dir`` is used only
    to compute the absolute paths later written by ``write_matrix`` (and
    baked into each entry's planned command); pass the same value to
    both functions. ``target_model_path``/``mtp_sidecar_path`` are
    optional placeholders substituted into the planned commands; when
    omitted, an explicit placeholder string is used so the manifest is
    still generated but visibly incomplete.
    """
    resolved_target_path = target_model_path or "<TARGET_MODEL_PATH>"
    resolved_sidecar_path = mtp_sidecar_path or "<MTP_SIDECAR_PATH>"
    # Persist absolute artifact paths so a manifest generated with a relative
    # --output-dir remains consumable from any later working directory.
    output_dir_path = Path(output_dir).expanduser().resolve()

    capability = detect_native_mtp_capability(
        model_family, mlx_lm_version=None, mlx_vlm_version=None
    )

    entries: list[MatrixEntry] = []
    for workload in workloads:
        for tier in context_tiers:
            materialized = materialize_prompt(workload, tier)
            prompt_path = str(
                output_dir_path / "prompts" / f"{workload.workload_id}-{tier.value}.txt"
            )

            ar_run_id = (
                f"{workload.workload_id}-{tier.value}-{DECODE_MODE_AUTOREGRESSIVE}"
            )
            ar_output_dir = str(output_dir_path / "runs" / ar_run_id)
            entries.append(
                MatrixEntry(
                    run_id=ar_run_id,
                    workload_id=workload.workload_id,
                    workload_version=workload.version,
                    category=workload.category.value,
                    context_tier=tier.value,
                    decode_mode=DECODE_MODE_AUTOREGRESSIVE,
                    configured_depth=None,
                    prompt=materialized,
                    prompt_path=prompt_path,
                    runner_results_dir=str(output_dir_path / "runner" / ar_run_id),
                    collector_output_dir=ar_output_dir,
                    runnable=True,
                    unsupported_reason=None,
                    command_argv=_collect_mlx_argv(
                        run_id=ar_run_id,
                        model_path=resolved_target_path,
                        model_id=model_id,
                        prompt_path=prompt_path,
                        output_dir=ar_output_dir,
                        max_tokens=max_tokens,
                    ),
                    max_tokens=max_tokens,
                )
            )

            for depth in mtp_depths:
                mtp_run_id = (
                    f"{workload.workload_id}-{tier.value}-"
                    f"{DECODE_MODE_NATIVE_MTP}-depth{depth}"
                )
                mtp_output_dir = str(output_dir_path / "runs" / mtp_run_id)
                entries.append(
                    MatrixEntry(
                        run_id=mtp_run_id,
                        workload_id=workload.workload_id,
                        workload_version=workload.version,
                        category=workload.category.value,
                        context_tier=tier.value,
                        decode_mode=DECODE_MODE_NATIVE_MTP,
                        configured_depth=depth,
                        prompt=materialized,
                        prompt_path=prompt_path,
                        runner_results_dir=str(output_dir_path / "runner" / mtp_run_id),
                        collector_output_dir=mtp_output_dir,
                        runnable=capability.supported,
                        unsupported_reason=(
                            None if capability.supported else capability.reason
                        ),
                        command_argv=_native_mtp_collect_argv(
                            run_id=mtp_run_id,
                            target_model_path=resolved_target_path,
                            mtp_sidecar_path=resolved_sidecar_path,
                            model_id=model_id,
                            prompt_path=prompt_path,
                            output_dir=mtp_output_dir,
                            max_tokens=max_tokens,
                            configured_depth=depth,
                        ),
                        max_tokens=max_tokens,
                    )
                )

    return MatrixManifest(
        schema_version=MATRIX_SCHEMA_VERSION,
        model_id=model_id,
        model_family=model_family,
        output_dir=str(output_dir_path),
        entries=tuple(entries),
    )


def write_matrix(manifest: MatrixManifest) -> None:
    """Materialize the manifest, prompts, and per-entry runner configs.

    Never executes a command or loads a model. Writes, under
    ``manifest.output_dir`` (the same value passed to
    ``generate_matrix``):

    * ``manifest.json``: the full machine-readable matrix.
    * ``prompts/<workload_id>-<tier>.txt``: each fully materialized
      prompt (deduplicated across decode modes/depths).
    * ``configs/<run_id>.json``: a ``RunnerConfig``-compatible JSON per
      entry (absolute paths), ready for
      ``llmtracefx-optimizer run --config ...``.
    """
    output_dir_path = Path(manifest.output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir_path / "manifest.json", manifest.to_json() + "\n")

    written_prompts: set[str] = set()
    for entry in manifest.entries:
        if entry.prompt_path not in written_prompts:
            atomic_write_text(Path(entry.prompt_path), entry.prompt.text)
            written_prompts.add(entry.prompt_path)

        runner_config = {
            "run_id": entry.run_id,
            "command": list(entry.command_argv),
            "results_dir": entry.runner_results_dir,
            "warmup_repetitions": 0,
            "measured_repetitions": 1,
        }
        atomic_write_text(
            output_dir_path / "configs" / f"{entry.run_id}.json",
            json.dumps(runner_config, indent=2, sort_keys=False) + "\n",
        )
