"""Process-isolated context-fit frontier for the pinned M5 Pro lab."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import html
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from .._artifact_io import (
    MAX_METADATA_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from ..collectors._shared import atomic_write_text, config_hash, sha256_text
from ..collectors.mlx import MLXCollectionConfig, MLXVLMRuntime, collect_mlx
from ..schema import ExperimentRecord, utc_now_iso
from ..workloads.catalog import CONTEXT_FILLER_CORPUS, workload_by_id
from ..workloads.evaluators import evaluate_workload
from .core import (
    LabError,
    SafetyDecision,
    assert_shareable,
    assess_safety,
    model_files_present,
    verify_model,
)
from .manifest import LabManifest

FRONTIER_MANIFEST_SCHEMA_VERSION = "1"
FRONTIER_STATE_SCHEMA_VERSION = "1"
FRONTIER_RESULT_SCHEMA_VERSION = "1"
FRONTIER_REPORT_SCHEMA_VERSION = "1"
DEFAULT_FRONTIER_MANIFEST_RESOURCE = "data/fit-frontier-manifest-v1.json"
RUN_MODES = ("exploratory", "publication")
RESUMABLE_RESULT_STATUSES = ("completed",)


class FrontierManifestError(ValueError):
    """Raised when the fit-frontier manifest is invalid."""


@dataclass(frozen=True)
class FrontierTier:
    name: str
    requested_tokens: int
    order: int


@dataclass(frozen=True)
class FrontierArtifacts:
    workspace: str
    shareable_example_dir: str
    commit_raw_artifacts: bool


@dataclass(frozen=True)
class FrontierManifest:
    schema_version: str
    frontier_id: str
    base_manifest_resource: str
    base_manifest_sha256: str
    base_lab_id: str
    model_repository_id: str
    model_revision: str
    model_expected_download_bytes: int
    workload_id: str
    workload_version: str
    prompt_construction: str
    maximum_token_shortfall: int
    tiers: tuple[FrontierTier, ...]
    max_output_tokens: int
    tier_timeout_seconds: float
    process_cleanup_grace_seconds: float
    artifacts: FrontierArtifacts

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> FrontierManifest:
        try:
            model = _object(raw["model"], "model")
            workload = _object(raw["workload"], "workload")
            generation = _object(raw["generation"], "generation")
            artifacts_raw = _object(raw["artifacts"], "artifacts")
            tiers_raw = raw["tiers"]
        except KeyError as exc:
            raise FrontierManifestError(f"missing frontier field: {exc}") from exc
        if not isinstance(tiers_raw, list) or not tiers_raw:
            raise FrontierManifestError("tiers must be a non-empty list")
        tiers = tuple(_tier(item) for item in tiers_raw)
        if len({tier.name for tier in tiers}) != len(tiers):
            raise FrontierManifestError("tier names must be unique")
        if [tier.order for tier in tiers] != sorted(tier.order for tier in tiers):
            raise FrontierManifestError("tiers must be ordered")
        if [tier.requested_tokens for tier in tiers] != sorted(
            tier.requested_tokens for tier in tiers
        ):
            raise FrontierManifestError("requested token targets must ascend")
        schema_version = _string(raw, "schema_version", "manifest")
        if schema_version != FRONTIER_MANIFEST_SCHEMA_VERSION:
            raise FrontierManifestError(
                f"unsupported frontier schema_version {schema_version!r}"
            )
        base_resource = _relative_path(
            _string(raw, "base_manifest_resource", "manifest"),
            "base_manifest_resource",
        )
        workspace = _relative_path(
            _string(artifacts_raw, "workspace", "artifacts"), "artifacts.workspace"
        )
        shareable = _relative_path(
            _string(artifacts_raw, "shareable_example_dir", "artifacts"),
            "artifacts.shareable_example_dir",
        )
        commit_raw = artifacts_raw.get("commit_raw_artifacts")
        if commit_raw is not False:
            raise FrontierManifestError(
                "artifacts.commit_raw_artifacts must remain false"
            )
        revision = _string(model, "revision", "model")
        if len(revision) != 40 or any(
            char not in "0123456789abcdef" for char in revision
        ):
            raise FrontierManifestError("model.revision must be a 40-character git SHA")
        digest = _string(raw, "base_manifest_sha256", "manifest")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise FrontierManifestError(
                "base_manifest_sha256 must be lowercase SHA-256"
            )
        return cls(
            schema_version=schema_version,
            frontier_id=_string(raw, "frontier_id", "manifest"),
            base_manifest_resource=base_resource,
            base_manifest_sha256=digest,
            base_lab_id=_string(raw, "base_lab_id", "manifest"),
            model_repository_id=_string(model, "repository_id", "model"),
            model_revision=revision,
            model_expected_download_bytes=_integer(
                model, "expected_download_bytes", "model", minimum=1
            ),
            workload_id=_string(workload, "workload_id", "workload"),
            workload_version=_string(workload, "version", "workload"),
            prompt_construction=_string(workload, "prompt_construction", "workload"),
            maximum_token_shortfall=_integer(
                workload, "maximum_token_shortfall", "workload"
            ),
            tiers=tiers,
            max_output_tokens=_integer(
                generation, "max_output_tokens", "generation", minimum=1
            ),
            tier_timeout_seconds=_number(
                raw, "tier_timeout_seconds", "manifest", minimum=1
            ),
            process_cleanup_grace_seconds=_number(
                raw, "process_cleanup_grace_seconds", "manifest", minimum=0.1
            ),
            artifacts=FrontierArtifacts(
                workspace=workspace,
                shareable_example_dir=shareable,
                commit_raw_artifacts=False,
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> FrontierManifest:
        try:
            raw = json.loads(payload, parse_constant=reject_non_finite_json_constant)
        except (ValueError, RecursionError) as exc:
            raise FrontierManifestError(f"invalid frontier JSON: {exc}") from exc
        return cls.from_dict(_object(raw, "manifest"))

    @classmethod
    def read_json(cls, path: Path) -> FrontierManifest:
        try:
            payload = read_bounded_regular_text(path, MAX_METADATA_ARTIFACT_BYTES)
        except ArtifactReadError as exc:
            raise FrontierManifestError(f"invalid frontier file: {exc}") from exc
        return cls.from_json(payload)

    def tier(self, name: str) -> FrontierTier:
        for tier in self.tiers:
            if tier.name == name:
                return tier
        raise FrontierManifestError(f"unknown frontier tier {name!r}")


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise FrontierManifestError(f"{context} must be an object")
    return value


def _string(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise FrontierManifestError(f"{context}.{key} must be a non-empty string")
    return value


def _integer(data: dict[str, Any], key: str, context: str, *, minimum: int = 0) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise FrontierManifestError(f"{context}.{key} must be an integer >= {minimum}")
    return value


def _number(data: dict[str, Any], key: str, context: str, *, minimum: float) -> float:
    value = data.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise FrontierManifestError(f"{context}.{key} must be >= {minimum}")
    return float(value)


def _relative_path(value: str, context: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise FrontierManifestError(f"{context} must be a relative safe path")
    return value


def _tier(value: Any) -> FrontierTier:
    raw = _object(value, "tiers[]")
    return FrontierTier(
        name=_string(raw, "name", "tiers[]"),
        requested_tokens=_integer(raw, "requested_tokens", "tiers[]", minimum=1),
        order=_integer(raw, "order", "tiers[]", minimum=1),
    )


def load_packaged_frontier_manifest() -> tuple[FrontierManifest, str]:
    resource = resources.files("llmtracefx.optimizer.lab").joinpath(
        DEFAULT_FRONTIER_MANIFEST_RESOURCE
    )
    payload = resource.read_text(encoding="utf-8")
    return (
        FrontierManifest.from_json(payload),
        f"package:llmtracefx.optimizer.lab/{DEFAULT_FRONTIER_MANIFEST_RESOURCE}",
    )


def load_frontier_manifest(path: Path | None) -> tuple[FrontierManifest, str]:
    if path is None:
        return load_packaged_frontier_manifest()
    return FrontierManifest.read_json(path), str(path)


def _resource_text(name: str) -> str:
    resource = resources.files("llmtracefx.optimizer.lab").joinpath(name)
    return resource.read_text(encoding="utf-8")


def load_bound_base_manifest(frontier: FrontierManifest) -> LabManifest:
    payload = _resource_text(frontier.base_manifest_resource)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if digest != frontier.base_manifest_sha256:
        raise FrontierManifestError(
            "packaged base manifest does not match fit-frontier binding"
        )
    base = LabManifest.from_json(payload)
    identity = (
        base.lab_id,
        base.model.repository_id,
        base.model.revision,
        base.model.expected_download_bytes,
    )
    expected = (
        frontier.base_lab_id,
        frontier.model_repository_id,
        frontier.model_revision,
        frontier.model_expected_download_bytes,
    )
    if identity != expected:
        raise FrontierManifestError(
            "fit-frontier identity does not match the pinned base manifest"
        )
    workload = workload_by_id(frontier.workload_id)
    if workload.version != frontier.workload_version:
        raise FrontierManifestError("fit-frontier workload version drifted")
    if frontier.prompt_construction != "deterministic-filler-before-verbatim-base-v1":
        raise FrontierManifestError("unsupported prompt construction")
    return base


def frontier_manifest_hash(frontier: FrontierManifest) -> str:
    return config_hash(
        {
            "schema_version": frontier.schema_version,
            "frontier_id": frontier.frontier_id,
            "base_manifest_sha256": frontier.base_manifest_sha256,
            "model_revision": frontier.model_revision,
            "workload_id": frontier.workload_id,
            "workload_version": frontier.workload_version,
            "prompt_construction": frontier.prompt_construction,
            "maximum_token_shortfall": frontier.maximum_token_shortfall,
            "tiers": [asdict(tier) for tier in frontier.tiers],
            "max_output_tokens": frontier.max_output_tokens,
            "tier_timeout_seconds": frontier.tier_timeout_seconds,
        }
    )


def machine_state(decision: SafetyDecision) -> dict[str, Any]:
    snapshot = decision.snapshot
    available_estimate = None
    if (
        snapshot.total_memory_bytes is not None
        and snapshot.memory_free_percent is not None
    ):
        available_estimate = int(
            snapshot.total_memory_bytes * snapshot.memory_free_percent / 100
        )
    return {
        "collected_at": snapshot.collected_at,
        "os_name": snapshot.os_name,
        "os_release": snapshot.os_release,
        "architecture": snapshot.architecture,
        "chip": snapshot.chip,
        "physical_memory_bytes": snapshot.total_memory_bytes,
        "available_memory_estimate_bytes": available_estimate,
        "available_memory_estimate_provenance": (
            "derived from macOS memory_pressure free percentage multiplied by "
            "physical memory; approximate system headroom, not free GPU memory"
            if available_estimate is not None
            else None
        ),
        "memory_free_percent": snapshot.memory_free_percent,
        "memory_free_percent_provenance": "macOS memory_pressure",
        "swap_used_bytes": snapshot.swap_used_bytes,
        "swap_provenance": "sysctl vm.swapusage",
        "package_versions": {
            name: version
            for name, version in snapshot.package_versions.items()
            if name in {"mlx", "mlx-lm", "mlx-vlm", "transformers"}
        },
    }


def _filler(characters: int) -> str:
    parts: list[str] = []
    index = 0
    length = 0
    while length < characters:
        part = CONTEXT_FILLER_CORPUS.format(index=index) + "\n"
        parts.append(part)
        length += len(part)
        index += 1
    return "".join(parts)[:characters]


def _candidate_prompt(base_prompt: str, filler_characters: int) -> str:
    if filler_characters == 0:
        return base_prompt
    return (
        "The following numbered context is inert padding. Ignore it when "
        "answering the profile task.\n"
        + _filler(filler_characters)
        + "\n\n"
        + base_prompt
    )


def fit_prompt(
    encode: Callable[[str], list[int]],
    *,
    base_prompt: str,
    requested_tokens: int,
    maximum_shortfall: int,
) -> tuple[str, list[int]]:
    base_tokens = encode(base_prompt)
    if len(base_tokens) > requested_tokens:
        raise LabError(
            f"base prompt tokenized to {len(base_tokens)} tokens, above requested "
            f"tier {requested_tokens}"
        )
    low = 0
    high = requested_tokens * 8
    best = (base_prompt, base_tokens)
    while low <= high:
        middle = (low + high) // 2
        prompt = _candidate_prompt(base_prompt, middle)
        tokens = encode(prompt)
        if len(tokens) <= requested_tokens:
            if len(tokens) > len(best[1]):
                best = (prompt, tokens)
            low = middle + 1
        else:
            high = middle - 1
    start = max(0, high - 96)
    for characters in range(start, high + 97):
        prompt = _candidate_prompt(base_prompt, characters)
        tokens = encode(prompt)
        if len(best[1]) < len(tokens) <= requested_tokens:
            best = (prompt, tokens)
    shortfall = requested_tokens - len(best[1])
    if shortfall > maximum_shortfall:
        raise LabError(
            f"tokenizer could only construct {len(best[1])} tokens for requested "
            f"tier {requested_tokens}; maximum allowed shortfall is "
            f"{maximum_shortfall}"
        )
    return best


def _measurement(record: ExperimentRecord, field: str) -> float | None:
    value = getattr(record.timing, field)
    return None if value is None else value.value


def _classify_error(category: str | None, message: str | None) -> tuple[str, str]:
    text = f"{category or ''} {message or ''}".lower()
    if (
        category == "MemoryError"
        or "outofmemory" in text
        or "out of memory" in text
        or "insufficient memory" in text
    ):
        return "oom", "MLX/Metal reported insufficient memory"
    if "timeout" in text:
        return "timeout", "tier exceeded its configured timeout"
    return "failed", "runtime collection failed"


def execute_tier_child(
    frontier: FrontierManifest,
    base: LabManifest,
    *,
    tier: FrontierTier,
    model_path: Path,
    output_dir: Path,
    run_mode: str,
    clean_boot_confirmed: bool,
) -> dict[str, Any]:
    runtime = MLXVLMRuntime(
        temperature=base.generation.temperature,
        top_p=base.generation.top_p,
        enable_thinking=base.generation.enable_thinking,
        prefill_step_size=base.runtime.prefill_step_size,
    )
    model, processor = runtime.load_model(model_path)
    del model
    workload = workload_by_id(frontier.workload_id)
    prompt, prompt_tokens = fit_prompt(
        lambda text: runtime.encode(processor, text),
        base_prompt=workload.base_prompt,
        requested_tokens=tier.requested_tokens,
        maximum_shortfall=frontier.maximum_token_shortfall,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / "prompt.txt", prompt)
    collection_dir = output_dir / "collection"
    run_id = f"{frontier.frontier_id}-{run_mode}-{tier.name}"
    collected = collect_mlx(
        MLXCollectionConfig(
            run_id=run_id,
            model_path=model_path,
            model_id=base.model.repository_id,
            prompt=prompt,
            output_dir=collection_dir,
            command_argv=(
                "llmtracefx-m5-frontier",
                "run",
                "--mode",
                run_mode,
                "--max-tier",
                tier.name,
            ),
            max_tokens=frontier.max_output_tokens,
            seed=base.generation.seed,
            temperature=base.generation.temperature,
            top_p=base.generation.top_p,
            enable_thinking=base.generation.enable_thinking,
            prefill_step_size=base.runtime.prefill_step_size,
            model_revision=base.model.revision,
            tokenizer_revision=base.model.revision,
            quantization=base.model.quantization,
            model_family="qwen3_5",
            accelerator=base.safety.required_chip,
            timeout_seconds=frontier.tier_timeout_seconds,
        ),
        runtime=runtime,
    )
    record = collected.record
    actual_tokens = record.tokens.input_tokens
    if actual_tokens != len(prompt_tokens):
        raise LabError("collector token count disagrees with tokenizer construction")
    evaluator: dict[str, Any] | None = None
    if record.outcome.success:
        outcome = evaluate_workload(workload, collected.response_text)
        record = dataclasses.replace(record, outcome=outcome)
        record.write_json(output_dir / "final_record.json")
        evaluator = {
            "success": outcome.success,
            "quality_score": outcome.quality_score,
            "quality_metric": outcome.quality_metric,
        }
        status = "completed"
        reason = None
    else:
        status, reason = _classify_error(
            record.error.category if record.error else None,
            record.error.message if record.error else None,
        )
        record.write_json(output_dir / "final_record.json")
    result = {
        "schema_version": FRONTIER_RESULT_SCHEMA_VERSION,
        "frontier_id": frontier.frontier_id,
        "frontier_manifest_hash": frontier_manifest_hash(frontier),
        "run_mode": run_mode,
        "clean_boot_confirmed": clean_boot_confirmed,
        "tier": tier.name,
        "requested_tokens": tier.requested_tokens,
        "actual_tokens": actual_tokens,
        "prompt_hash": sha256_text(prompt),
        "status": status,
        "reason": reason,
        "timing": {
            "total_ms": _measurement(record, "total"),
            "prefill_ms": _measurement(record, "prefill"),
            "decode_ms": _measurement(record, "decode"),
        },
        "quality": evaluator,
        "record_sha256": _file_sha256(output_dir / "final_record.json"),
    }
    atomic_write_text(
        output_dir / "child-result.json",
        json.dumps(result, indent=2, sort_keys=False) + "\n",
    )
    return result


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = read_bounded_regular_text(path, MAX_METADATA_ARTIFACT_BYTES)
        value = json.loads(payload, parse_constant=reject_non_finite_json_constant)
    except (ArtifactReadError, ValueError, RecursionError) as exc:
        raise LabError(f"invalid artifact {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise LabError(f"invalid artifact {path.name}: expected an object")
    return value


def _validate_result(
    result: dict[str, Any],
    frontier: FrontierManifest,
    *,
    tier: FrontierTier,
    run_mode: str,
    clean_boot_confirmed: bool,
    allow_skipped: bool = True,
) -> None:
    expected = {
        "schema_version": FRONTIER_RESULT_SCHEMA_VERSION,
        "frontier_id": frontier.frontier_id,
        "frontier_manifest_hash": frontier_manifest_hash(frontier),
        "run_mode": run_mode,
        "clean_boot_confirmed": clean_boot_confirmed,
        "tier": tier.name,
        "requested_tokens": tier.requested_tokens,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise LabError(f"stale tier artifact: {key} does not match")
    status = result.get("status")
    allowed = {"completed", "oom", "timeout", "failed"}
    if allow_skipped:
        allowed.add("skipped")
    if status not in allowed:
        raise LabError("invalid tier artifact: unsupported status")
    actual = result.get("actual_tokens")
    if actual is not None and (
        isinstance(actual, bool)
        or not isinstance(actual, int)
        or actual > tier.requested_tokens
        or tier.requested_tokens - actual > frontier.maximum_token_shortfall
    ):
        raise LabError("invalid tier artifact: actual token count is out of bounds")
    if status == "completed" or (status == "skipped" and actual is not None):
        quality = result.get("quality")
        if actual is None or not isinstance(quality, dict):
            raise LabError("invalid tier artifact: evaluator result is missing")
        digest = result.get("record_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise LabError("invalid tier artifact: record digest is missing")


@dataclass(frozen=True)
class ChildProcessResult:
    exit_code: int | None
    timed_out: bool
    descendants_cleaned: bool


def _group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _signal_process_group(process_group: int, requested_signal: signal.Signals) -> bool:
    try:
        os.killpg(process_group, requested_signal)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return True


def _wait_for_process_group_exit(
    process: subprocess.Popen[bytes], process_group: int, grace_seconds: float
) -> bool:
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        process.poll()
        if not _group_exists(process_group):
            return True
        time.sleep(0.05)
    process.poll()
    return not _group_exists(process_group)


def _clean_process_group(
    process: subprocess.Popen[bytes], process_group: int, grace_seconds: float
) -> bool:
    process.poll()
    if not _group_exists(process_group):
        return True
    if not _signal_process_group(process_group, signal.SIGTERM):
        return False
    if _wait_for_process_group_exit(process, process_group, grace_seconds):
        return True
    if not _signal_process_group(process_group, signal.SIGKILL):
        return False
    return _wait_for_process_group_exit(process, process_group, grace_seconds)


def launch_tier_subprocess(
    *,
    manifest_path: Path | None,
    tier: FrontierTier,
    model_path: Path,
    output_dir: Path,
    run_mode: str,
    clean_boot_confirmed: bool,
    timeout_seconds: float,
    cleanup_grace_seconds: float,
) -> ChildProcessResult:
    argv = [
        sys.executable,
        "-m",
        "llmtracefx.optimizer.lab.frontier",
        "_child",
        "--tier",
        tier.name,
        "--model-path",
        str(model_path),
        "--output-dir",
        str(output_dir),
        "--mode",
        run_mode,
    ]
    if manifest_path is not None:
        argv.extend(("--manifest", str(manifest_path)))
    if clean_boot_confirmed:
        argv.append("--confirm-clean-boot")
    process = subprocess.Popen(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        shell=False,
    )
    process_group = process.pid
    timed_out = False
    try:
        process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
    descendants_cleaned = _clean_process_group(
        process, process_group, cleanup_grace_seconds
    )
    if process.poll() is None:
        try:
            process.wait(timeout=cleanup_grace_seconds)
        except subprocess.TimeoutExpired:
            descendants_cleaned = False
    return ChildProcessResult(
        exit_code=process.returncode,
        timed_out=timed_out,
        descendants_cleaned=descendants_cleaned,
    )


def _skipped_result(
    frontier: FrontierManifest,
    tier: FrontierTier,
    *,
    run_mode: str,
    clean_boot_confirmed: bool,
    reason: str,
) -> dict[str, Any]:
    return {
        "schema_version": FRONTIER_RESULT_SCHEMA_VERSION,
        "frontier_id": frontier.frontier_id,
        "frontier_manifest_hash": frontier_manifest_hash(frontier),
        "run_mode": run_mode,
        "clean_boot_confirmed": clean_boot_confirmed,
        "tier": tier.name,
        "requested_tokens": tier.requested_tokens,
        "actual_tokens": None,
        "prompt_hash": None,
        "status": "skipped",
        "reason": reason,
        "timing": {"total_ms": None, "prefill_ms": None, "decode_ms": None},
        "quality": None,
        "record_sha256": None,
        "process": None,
    }


def run_frontier(
    frontier: FrontierManifest,
    base: LabManifest,
    *,
    manifest_path: Path | None,
    workspace: Path,
    model_path: Path,
    max_tier: str,
    run_mode: str,
    clean_boot_confirmed: bool,
    resume: bool,
    launcher: Callable[..., ChildProcessResult] = launch_tier_subprocess,
) -> dict[str, Any]:
    if run_mode not in RUN_MODES:
        raise LabError(f"unsupported run mode {run_mode!r}")
    if run_mode == "publication" and not clean_boot_confirmed:
        raise LabError(
            "publication mode requires the operator assertion --confirm-clean-boot"
        )
    if run_mode == "exploratory" and clean_boot_confirmed:
        raise LabError("--confirm-clean-boot is only valid in publication mode")
    verify_model(base, model_path)
    preflight = assess_safety(base, workspace, include_download=False)
    if not preflight.safe:
        raise LabError(
            "frontier run blocked by safety preflight: " + "; ".join(preflight.blockers)
        )
    selected = frontier.tier(max_tier)
    tiers = tuple(tier for tier in frontier.tiers if tier.order <= selected.order)
    mode_workspace = workspace / run_mode
    state_path = mode_workspace / "state.json"
    if resume and state_path.is_file():
        prior_state = _read_json(state_path)
        prior_max = prior_state.get("requested_max_tier")
        if isinstance(prior_max, str):
            try:
                prior_selected = frontier.tier(prior_max)
            except FrontierManifestError as exc:
                raise LabError("existing state has an unknown maximum tier") from exc
            if prior_selected.order > selected.order:
                raise LabError(
                    "cannot lower --max-tier while resuming a wider existing "
                    "sweep; use --no-resume for a distinct replacement run"
                )
    if resume:
        for tier in tiers:
            result_path = mode_workspace / "tiers" / tier.name / "result.json"
            if not result_path.is_file():
                continue
            _validate_result(
                _read_json(result_path),
                frontier,
                tier=tier,
                run_mode=run_mode,
                clean_boot_confirmed=clean_boot_confirmed,
            )
    state: dict[str, Any] = {
        "schema_version": FRONTIER_STATE_SCHEMA_VERSION,
        "frontier_id": frontier.frontier_id,
        "frontier_manifest_hash": frontier_manifest_hash(frontier),
        "run_mode": run_mode,
        "clean_boot_confirmed": clean_boot_confirmed,
        "started_at": utc_now_iso(),
        "ended_at": None,
        "status": "running",
        "requested_max_tier": max_tier,
        "pre_run_machine_state": machine_state(preflight),
        "results": [],
        "stop_reason": None,
    }
    mode_workspace.mkdir(parents=True, exist_ok=True)
    atomic_write_text(state_path, json.dumps(state, indent=2) + "\n")
    stopped = False
    for index, tier in enumerate(tiers):
        if stopped:
            state["results"].append(
                _skipped_result(
                    frontier,
                    tier,
                    run_mode=run_mode,
                    clean_boot_confirmed=clean_boot_confirmed,
                    reason="not attempted after first failed tier",
                )
            )
            continue
        tier_dir = mode_workspace / "tiers" / tier.name
        result_path = tier_dir / "result.json"
        child_path = tier_dir / "child-result.json"
        if resume and result_path.is_file():
            prior = _read_json(result_path)
            _validate_result(
                prior,
                frontier,
                tier=tier,
                run_mode=run_mode,
                clean_boot_confirmed=clean_boot_confirmed,
            )
            if prior["status"] in RESUMABLE_RESULT_STATUSES:
                resumed = dict(prior)
                resumed["status"] = "skipped"
                resumed["reason"] = "resumed verified completed artifact"
                state["results"].append(resumed)
                continue
        tier_preflight = assess_safety(base, workspace, include_download=False)
        if not tier_preflight.safe:
            result = _skipped_result(
                frontier,
                tier,
                run_mode=run_mode,
                clean_boot_confirmed=clean_boot_confirmed,
                reason="tier blocked by current machine safety gate",
            )
            result["status"] = "failed"
            state["results"].append(result)
            state["stop_reason"] = "machine safety gate failed before tier"
            stopped = True
            continue
        child = launcher(
            manifest_path=manifest_path,
            tier=tier,
            model_path=model_path,
            output_dir=tier_dir,
            run_mode=run_mode,
            clean_boot_confirmed=clean_boot_confirmed,
            timeout_seconds=frontier.tier_timeout_seconds,
            cleanup_grace_seconds=frontier.process_cleanup_grace_seconds,
        )
        process_evidence = {
            "fresh_subprocess": True,
            "new_session": True,
            "child_exit_code": child.exit_code,
            "timed_out": child.timed_out,
            "descendants_cleaned": child.descendants_cleaned,
        }
        if not child.descendants_cleaned:
            result = _skipped_result(
                frontier,
                tier,
                run_mode=run_mode,
                clean_boot_confirmed=clean_boot_confirmed,
                reason="child process group cleanup could not be verified",
            )
            result["status"] = "failed"
        elif child.timed_out:
            result = _skipped_result(
                frontier,
                tier,
                run_mode=run_mode,
                clean_boot_confirmed=clean_boot_confirmed,
                reason="tier exceeded parent-enforced timeout",
            )
            result["status"] = "timeout"
        elif not child_path.is_file():
            result = _skipped_result(
                frontier,
                tier,
                run_mode=run_mode,
                clean_boot_confirmed=clean_boot_confirmed,
                reason="child exited without a complete artifact",
            )
            result["status"] = "failed"
        else:
            try:
                result = _read_json(child_path)
                _validate_result(
                    result,
                    frontier,
                    tier=tier,
                    run_mode=run_mode,
                    clean_boot_confirmed=clean_boot_confirmed,
                    allow_skipped=False,
                )
            except LabError:
                result = _skipped_result(
                    frontier,
                    tier,
                    run_mode=run_mode,
                    clean_boot_confirmed=clean_boot_confirmed,
                    reason="child artifact failed validation",
                )
                result["status"] = "failed"
            else:
                if result["status"] == "completed" and child.exit_code != 0:
                    result = _skipped_result(
                        frontier,
                        tier,
                        run_mode=run_mode,
                        clean_boot_confirmed=clean_boot_confirmed,
                        reason="successful child artifact has a nonzero exit code",
                    )
                    result["status"] = "failed"
                elif result["status"] != "completed" and child.exit_code == 0:
                    result = _skipped_result(
                        frontier,
                        tier,
                        run_mode=run_mode,
                        clean_boot_confirmed=clean_boot_confirmed,
                        reason="failed child artifact has a zero exit code",
                    )
                    result["status"] = "failed"
        result["process"] = process_evidence
        atomic_write_text(
            result_path, json.dumps(result, indent=2, sort_keys=False) + "\n"
        )
        state["results"].append(result)
        if result["status"] != "completed":
            state["stop_reason"] = result["reason"]
            stopped = True
        atomic_write_text(state_path, json.dumps(state, indent=2) + "\n")
        if stopped:
            for remaining in tiers[index + 1 :]:
                state["results"].append(
                    _skipped_result(
                        frontier,
                        remaining,
                        run_mode=run_mode,
                        clean_boot_confirmed=clean_boot_confirmed,
                        reason="not attempted after first failed tier",
                    )
                )
            break
    state["ended_at"] = utc_now_iso()
    state["status"] = "stopped" if stopped else "completed"
    atomic_write_text(state_path, json.dumps(state, indent=2) + "\n")
    return state


def verify_frontier_evidence(
    frontier: FrontierManifest, *, workspace: Path, run_mode: str
) -> dict[str, Any]:
    state_path = workspace / run_mode / "state.json"
    failures: list[str] = []
    if not state_path.is_file():
        return {"verified": False, "failures": ["state artifact is missing"]}
    state = _read_json(state_path)
    if state.get("status") == "running":
        failures.append("state artifact is incomplete")
    if state.get("frontier_manifest_hash") != frontier_manifest_hash(frontier):
        failures.append("state manifest binding is stale")
    if state.get("run_mode") != run_mode:
        failures.append("state run mode does not match")
    clean_boot = state.get("clean_boot_confirmed")
    if run_mode == "publication" and clean_boot is not True:
        failures.append("publication state lacks operator clean-boot confirmation")
    for raw in state.get("results", []):
        if not isinstance(raw, dict):
            failures.append("state contains a non-object result")
            continue
        try:
            tier = frontier.tier(str(raw.get("tier")))
            _validate_result(
                raw,
                frontier,
                tier=tier,
                run_mode=run_mode,
                clean_boot_confirmed=bool(clean_boot),
            )
            digest = raw.get("record_sha256")
            if isinstance(digest, str):
                record_path = (
                    workspace / run_mode / "tiers" / tier.name / "final_record.json"
                )
                if not record_path.is_file() or _file_sha256(record_path) != digest:
                    failures.append(f"{tier.name} final record digest does not match")
        except (FrontierManifestError, LabError) as exc:
            failures.append(str(exc))
    return {"verified": not failures, "failures": failures}


def build_frontier_report(
    frontier: FrontierManifest, *, workspace: Path, run_mode: str
) -> dict[str, Any]:
    verification = verify_frontier_evidence(
        frontier, workspace=workspace, run_mode=run_mode
    )
    if not verification["verified"]:
        raise LabError(
            "frontier report refused invalid evidence: "
            + "; ".join(verification["failures"])
        )
    state = _read_json(workspace / run_mode / "state.json")
    rows = []
    by_tier = {
        result["tier"]: result
        for result in state["results"]
        if isinstance(result, dict)
    }
    for tier in frontier.tiers:
        result = by_tier.get(tier.name)
        if result is None:
            rows.append(
                {
                    "tier": tier.name,
                    "requested_tokens": tier.requested_tokens,
                    "actual_tokens": None,
                    "outcome": "skipped",
                    "total_ms": None,
                    "prefill_ms": None,
                    "decode_ms": None,
                    "quality_success": None,
                    "quality_score": None,
                }
            )
            continue
        quality = result.get("quality") or {}
        timing = result.get("timing") or {}
        rows.append(
            {
                "tier": tier.name,
                "requested_tokens": tier.requested_tokens,
                "actual_tokens": result.get("actual_tokens"),
                "outcome": result.get("status"),
                "total_ms": timing.get("total_ms"),
                "prefill_ms": timing.get("prefill_ms"),
                "decode_ms": timing.get("decode_ms"),
                "quality_success": quality.get("success"),
                "quality_score": quality.get("quality_score"),
            }
        )
    report = {
        "schema_version": FRONTIER_REPORT_SCHEMA_VERSION,
        "frontier_id": frontier.frontier_id,
        "generated_at": state.get("ended_at"),
        "run_mode": run_mode,
        "clean_boot_confirmed": state.get("clean_boot_confirmed"),
        "model": {
            "repository_id": frontier.model_repository_id,
            "revision": frontier.model_revision,
        },
        "pre_run_machine_state": state.get("pre_run_machine_state"),
        "rows": rows,
        "maximum_completed": next(
            (
                {
                    "tier": row["tier"],
                    "requested_tokens": row["requested_tokens"],
                    "actual_tokens": row["actual_tokens"],
                }
                for row in reversed(rows)
                if row["outcome"] in {"completed", "skipped"}
                and row["actual_tokens"] is not None
            ),
            None,
        ),
        "stop_reason": state.get("stop_reason"),
        "limitations": [
            "This is bounded evidence for one recorded machine state, exact checkpoint, runtime, prompt construction, and run mode; it is not a universal memory-capacity claim.",
            "Requested tokens are targets; actual_tokens is the model tokenizer observation and is never relabeled as the requested tier.",
            "Available memory is an approximate system-headroom estimate derived from macOS memory_pressure, not precise free unified GPU memory.",
            "Total, prefill, and decode values are supported host wall-clock boundaries; missing measurements remain null.",
            "No GPU utilization, peak system memory, bandwidth, power, energy, or kernel time is measured or inferred.",
            "Clean-boot confirmation is an operator assertion and is never inferred.",
        ],
    }
    assert_shareable(report)
    return report


def render_frontier_report_html(report: dict[str, Any]) -> str:
    assert_shareable(report)

    def esc(value: Any) -> str:
        return html.escape(str(value), quote=True)

    rows = "".join(
        "<tr>"
        f"<td>{esc(row['tier'])}</td>"
        f"<td>{esc(row['requested_tokens'])}</td>"
        f"<td>{esc(row['actual_tokens'] if row['actual_tokens'] is not None else 'n/a')}</td>"
        f"<td>{esc(row['outcome'])}</td>"
        f"<td>{esc(row['total_ms'] if row['total_ms'] is not None else 'n/a')}</td>"
        f"<td>{esc(row['prefill_ms'] if row['prefill_ms'] is not None else 'n/a')}</td>"
        f"<td>{esc(row['quality_score'] if row['quality_score'] is not None else 'n/a')}</td>"
        "</tr>"
        for row in report["rows"]
    )
    bars = "".join(
        f'<div class="bar-row"><span>{esc(row["actual_tokens"] or 0)}</span>'
        f'<i style="width:{(row["actual_tokens"] or 0) / 2048 * 100:.2f}%"></i>'
        f"<b>{esc(row['outcome'])}</b></div>"
        for row in report["rows"]
    )
    limitations = "".join(f"<li>{esc(item)}</li>" for item in report["limitations"])
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>Qwen3.8-27B fit frontier</title>
<style>
body{{margin:0;background:#f7f3ea;color:#17202a;font:15px/1.5 ui-monospace,monospace}}
main{{max-width:980px;margin:auto;padding:36px 20px}}section{{background:#fffdf8;border:1px
solid #d9d1c3;padding:18px;margin:18px 0;overflow:auto}}table{{width:100%;
border-collapse:collapse}}th,td{{padding:9px;text-align:right;border-bottom:1px solid
#ddd5c8}}th:first-child,td:first-child{{text-align:left}}.bar-row{{display:grid;
grid-template-columns:55px 1fr 90px;gap:10px;align-items:center;margin:8px 0}}
.bar-row i{{display:block;height:18px;background:#f05d23;min-width:1px}}
</style></head><body><main><p>LLMTraceFX / {esc(report['run_mode'])} evidence</p>
<h1>Context-fit frontier</h1><p><code>{esc(report['model']['repository_id'])}@
{esc(report['model']['revision'])}</code></p><section><h2>Sanitized outcomes</h2>
<table><thead><tr><th>Tier</th><th>Requested</th><th>Actual</th><th>Outcome</th>
<th>Total ms</th><th>Prefill ms</th><th>Quality</th></tr></thead><tbody>{rows}
</tbody></table></section><section><h2>Actual token counts</h2>{bars}</section>
<section><h2>Limits</h2><ul>{limitations}</ul></section></main></body></html>"""


def write_frontier_report(
    frontier: FrontierManifest,
    *,
    workspace: Path,
    run_mode: str,
    shareable_dir: Path | None,
) -> dict[str, Any]:
    report = build_frontier_report(frontier, workspace=workspace, run_mode=run_mode)
    reports = workspace / run_mode / "reports"
    atomic_write_text(
        reports / "fit-frontier-summary.json",
        json.dumps(report, indent=2, sort_keys=False) + "\n",
    )
    atomic_write_text(
        reports / "fit-frontier-report.html",
        render_frontier_report_html(report),
    )
    if shareable_dir is not None:
        destination = shareable_dir / run_mode
        atomic_write_text(
            destination / "fit-frontier-summary.json",
            json.dumps(report, indent=2, sort_keys=False) + "\n",
        )
        atomic_write_text(
            destination / "fit-frontier-report.html",
            render_frontier_report_html(report),
        )
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmtracefx-m5-frontier",
        description=(
            "Process-isolated context-fit frontier for the exact pinned "
            "Qwen3.8-27B MLX checkpoint. The default action is a no-load plan."
        ),
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="plan",
        choices=("plan", "run", "report", "verify", "_child"),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument(
        "--max-tier",
        choices=("t256", "t512", "t1024", "t1536", "t2048"),
        default="t2048",
    )
    parser.add_argument("--tier", choices=("t256", "t512", "t1024", "t1536", "t2048"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=RUN_MODES, default="exploratory")
    parser.add_argument("--confirm-clean-boot", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--shareable-dir", type=Path)
    return parser


def _paths(
    args: argparse.Namespace, frontier: FrontierManifest, base: LabManifest
) -> tuple[Path, Path]:
    workspace = args.workspace or Path(frontier.artifacts.workspace)
    model_path = args.model_path or Path(base.artifacts.model_cache)
    return workspace, model_path


def _plan(args: argparse.Namespace) -> int:
    frontier, source = load_frontier_manifest(args.manifest)
    base = load_bound_base_manifest(frontier)
    workspace, model_path = _paths(args, frontier, base)
    decision = assess_safety(base, workspace, include_download=False)
    payload = {
        "action": "plan",
        "weights_loaded": False,
        "downloads_performed": False,
        "manifest": source,
        "frontier_id": frontier.frontier_id,
        "run_mode": args.mode,
        "clean_boot_confirmed": args.confirm_clean_boot,
        "publication_ready": (
            args.mode == "publication" and args.confirm_clean_boot and decision.safe
        ),
        "model_present_by_size": model_files_present(base, model_path),
        "model": {
            "repository_id": frontier.model_repository_id,
            "revision": frontier.model_revision,
        },
        "tiers": [asdict(tier) for tier in frontier.tiers],
        "machine_state": machine_state(decision),
        "safety": {"safe": decision.safe, "blockers": list(decision.blockers)},
    }
    print(json.dumps(payload, indent=2, sort_keys=False))
    if args.mode == "publication" and not args.confirm_clean_boot:
        return 2
    return 0 if decision.safe else 2


def _run(args: argparse.Namespace) -> int:
    frontier, _ = load_frontier_manifest(args.manifest)
    base = load_bound_base_manifest(frontier)
    workspace, model_path = _paths(args, frontier, base)
    state = run_frontier(
        frontier,
        base,
        manifest_path=args.manifest,
        workspace=workspace,
        model_path=model_path,
        max_tier=args.max_tier,
        run_mode=args.mode,
        clean_boot_confirmed=args.confirm_clean_boot,
        resume=not args.no_resume,
    )
    report = write_frontier_report(
        frontier,
        workspace=workspace,
        run_mode=args.mode,
        shareable_dir=args.shareable_dir,
    )
    print(json.dumps({"state": state, "report": report}, indent=2))
    return 0 if state["status"] == "completed" else 2


def _report(args: argparse.Namespace) -> int:
    frontier, _ = load_frontier_manifest(args.manifest)
    base = load_bound_base_manifest(frontier)
    workspace, _ = _paths(args, frontier, base)
    report = write_frontier_report(
        frontier,
        workspace=workspace,
        run_mode=args.mode,
        shareable_dir=args.shareable_dir,
    )
    print(json.dumps(report, indent=2))
    return 0


def _verify(args: argparse.Namespace) -> int:
    frontier, _ = load_frontier_manifest(args.manifest)
    base = load_bound_base_manifest(frontier)
    workspace, model_path = _paths(args, frontier, base)
    verify_model(base, model_path)
    result = verify_frontier_evidence(frontier, workspace=workspace, run_mode=args.mode)
    print(json.dumps(result, indent=2))
    return 0 if result["verified"] else 2


def _child(args: argparse.Namespace) -> int:
    if args.tier is None or args.output_dir is None or args.model_path is None:
        raise LabError("_child requires --tier, --output-dir, and --model-path")
    frontier, _ = load_frontier_manifest(args.manifest)
    base = load_bound_base_manifest(frontier)
    result = execute_tier_child(
        frontier,
        base,
        tier=frontier.tier(args.tier),
        model_path=args.model_path,
        output_dir=args.output_dir,
        run_mode=args.mode,
        clean_boot_confirmed=args.confirm_clean_boot,
    )
    return 0 if result["status"] == "completed" else 2


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    commands = {
        "plan": _plan,
        "run": _run,
        "report": _report,
        "verify": _verify,
        "_child": _child,
    }
    try:
        return commands[args.action](args)
    except (
        FrontierManifestError,
        LabError,
        OSError,
        UnicodeError,
        ValueError,
    ) as exc:
        print(f"M5 fit frontier failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
