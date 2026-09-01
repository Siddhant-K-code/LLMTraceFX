"""Focused MLX OOM autopsy for the pinned M5 Pro Qwen3.8-27B checkpoint.

Runs exactly one process-isolated attempt at the frontier's ``t256`` tier and
records discrete stage-boundary checkpoints (never periodic samples) of the
MLX allocator, host process, and host system swap scopes. The evidence is a
bounded, atomically-persisted journal that remains usable failure evidence
even after an abrupt timeout, signal, or forced parent cleanup.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from .._artifact_io import (
    MAX_METADATA_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_text,
    reject_non_finite_json_constant,
)
from ..collectors._shared import atomic_write_text, config_hash
from ..collectors.mlx import MLXVLMRuntime
from ..schema import utc_now_iso
from ..workloads.catalog import workload_by_id
from .core import (
    LabError,
    assert_shareable,
    assess_safety,
    model_files_present,
    verify_model,
)
from .frontier import (
    RUN_MODES,
    ChildProcessResult,
    FrontierManifest,
    FrontierManifestError,
    _classify_error,
    _clean_process_group,
    _file_sha256,
    _read_json,
    fit_prompt,
    frontier_manifest_hash,
    load_bound_base_manifest,
    machine_state,
)
from .manifest import LabManifest

AUTOPSY_MANIFEST_SCHEMA_VERSION = "1"
AUTOPSY_JOURNAL_SCHEMA_VERSION = "1"
AUTOPSY_RESULT_SCHEMA_VERSION = "1"
AUTOPSY_STATE_SCHEMA_VERSION = "1"
AUTOPSY_REPORT_SCHEMA_VERSION = "1"
DEFAULT_AUTOPSY_MANIFEST_RESOURCE = "data/autopsy-manifest-v1.json"
DEFAULT_FRONTIER_MANIFEST_RESOURCE_NAME = "data/fit-frontier-manifest-v1.json"
RESUMABLE_TERMINAL_STATUSES = ("completed", "oom", "timeout", "failed")

# Ordered "main path" stages: each may appear at most once and only in this
# relative order. ``caught_oom``, ``signal_received``, and ``cleanup`` are
# terminal-only stages that may follow the main path but never each other in
# more than one instance, and ``cleanup`` (when present) must be the final
# checkpoint.
MAIN_STAGE_SEQUENCE = (
    "child_start",
    "before_model_load",
    "after_model_load",
    "after_prompt_tokenization",
    "immediately_before_prefill_generation",
    "after_first_token",
    "completion",
)
TERMINAL_ONLY_STAGES = ("caught_oom", "signal_received", "cleanup")
STAGE_SEQUENCE = MAIN_STAGE_SEQUENCE + TERMINAL_ONLY_STAGES
_HANDLED_SIGNAL_NAMES = frozenset({"SIGTERM", "SIGINT"})
_COMPLETE_TERMINALS = frozenset({"completed", "oom", "timeout", "failed"})
_INCOMPLETE_TERMINALS = frozenset({"signal"})
_JOURNAL_TOP_LEVEL_KEYS = frozenset(
    {
        "schema_version",
        "autopsy_id",
        "autopsy_manifest_hash",
        "frontier_manifest_hash",
        "model",
        "tier",
        "run_mode",
        "clean_boot_confirmed",
        "synthetic",
        "started_at",
        "sampling",
        "peak_memory_reset",
        "checkpoints",
        "complete",
        "terminal",
        "envelope_sha256",
    }
)
_CHECKPOINT_TOP_LEVEL_KEYS = frozenset(
    {
        "stage",
        "sequence",
        "wall_clock_utc",
        "wall_clock_provenance",
        "monotonic_offset_seconds",
        "monotonic_offset_unit",
        "monotonic_offset_provenance",
        "mlx_allocator",
        "host_process",
        "host_system",
        "signal_received",
    }
)


class AutopsyManifestError(ValueError):
    """Raised when the autopsy manifest is invalid or its bindings drift."""


def _object(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AutopsyManifestError(f"{context} must be an object")
    return value


def _string(data: dict[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise AutopsyManifestError(f"{context}.{key} must be a non-empty string")
    return value


def _integer(data: dict[str, Any], key: str, context: str, *, minimum: int = 0) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise AutopsyManifestError(f"{context}.{key} must be an integer >= {minimum}")
    return value


def _number(data: dict[str, Any], key: str, context: str, *, minimum: float) -> float:
    value = data.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise AutopsyManifestError(f"{context}.{key} must be >= {minimum}")
    return float(value)


def _relative_path(value: str, context: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise AutopsyManifestError(f"{context} must be a relative safe path")
    return value


def _sha256_hex(value: str, context: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise AutopsyManifestError(f"{context} must be a lowercase SHA-256 hex digest")
    return value


def _git_revision(value: str, context: str) -> str:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise AutopsyManifestError(f"{context} must be a 40-character git revision")
    return value


@dataclass(frozen=True)
class AutopsyArtifacts:
    workspace: str
    shareable_example_dir: str
    commit_raw_artifacts: bool


@dataclass(frozen=True)
class AutopsyManifest:
    schema_version: str
    autopsy_id: str
    frontier_manifest_resource: str
    frontier_manifest_sha256: str
    frontier_id: str
    base_lab_id: str
    model_repository_id: str
    model_revision: str
    model_expected_download_bytes: int
    tier_name: str
    tier_requested_tokens: int
    child_timeout_seconds: float
    process_cleanup_grace_seconds: float
    journal_max_checkpoints: int
    journal_max_bytes: int
    artifacts: AutopsyArtifacts

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> AutopsyManifest:
        try:
            model = _object(raw["model"], "model")
            tier = _object(raw["tier"], "tier")
            journal = _object(raw["journal"], "journal")
            artifacts_raw = _object(raw["artifacts"], "artifacts")
        except KeyError as exc:
            raise AutopsyManifestError(f"missing autopsy field: {exc}") from exc
        schema_version = _string(raw, "schema_version", "manifest")
        if schema_version != AUTOPSY_MANIFEST_SCHEMA_VERSION:
            raise AutopsyManifestError(
                f"unsupported autopsy schema_version {schema_version!r}"
            )
        commit_raw = artifacts_raw.get("commit_raw_artifacts")
        if commit_raw is not False:
            raise AutopsyManifestError(
                "artifacts.commit_raw_artifacts must remain false"
            )
        journal_max_bytes = _integer(journal, "max_bytes", "journal", minimum=1)
        if journal_max_bytes > MAX_METADATA_ARTIFACT_BYTES:
            raise AutopsyManifestError(
                "journal.max_bytes must not exceed the "
                f"{MAX_METADATA_ARTIFACT_BYTES}-byte metadata artifact limit"
            )
        return cls(
            schema_version=schema_version,
            autopsy_id=_string(raw, "autopsy_id", "manifest"),
            frontier_manifest_resource=_relative_path(
                _string(raw, "frontier_manifest_resource", "manifest"),
                "frontier_manifest_resource",
            ),
            frontier_manifest_sha256=_sha256_hex(
                _string(raw, "frontier_manifest_sha256", "manifest"),
                "frontier_manifest_sha256",
            ),
            frontier_id=_string(raw, "frontier_id", "manifest"),
            base_lab_id=_string(raw, "base_lab_id", "manifest"),
            model_repository_id=_string(model, "repository_id", "model"),
            model_revision=_git_revision(
                _string(model, "revision", "model"), "model.revision"
            ),
            model_expected_download_bytes=_integer(
                model, "expected_download_bytes", "model", minimum=1
            ),
            tier_name=_string(tier, "name", "tier"),
            tier_requested_tokens=_integer(tier, "requested_tokens", "tier", minimum=1),
            child_timeout_seconds=_number(
                raw, "child_timeout_seconds", "manifest", minimum=1
            ),
            process_cleanup_grace_seconds=_number(
                raw, "process_cleanup_grace_seconds", "manifest", minimum=0.1
            ),
            journal_max_checkpoints=_integer(
                journal, "max_checkpoints", "journal", minimum=1
            ),
            journal_max_bytes=journal_max_bytes,
            artifacts=AutopsyArtifacts(
                workspace=_relative_path(
                    _string(artifacts_raw, "workspace", "artifacts"),
                    "artifacts.workspace",
                ),
                shareable_example_dir=_relative_path(
                    _string(artifacts_raw, "shareable_example_dir", "artifacts"),
                    "artifacts.shareable_example_dir",
                ),
                commit_raw_artifacts=False,
            ),
        )

    @classmethod
    def from_json(cls, payload: str) -> AutopsyManifest:
        try:
            raw = json.loads(payload, parse_constant=reject_non_finite_json_constant)
        except (ValueError, RecursionError) as exc:
            raise AutopsyManifestError(f"invalid autopsy JSON: {exc}") from exc
        return cls.from_dict(_object(raw, "manifest"))

    @classmethod
    def read_json(cls, path: Path) -> AutopsyManifest:
        try:
            payload = read_bounded_regular_text(path, MAX_METADATA_ARTIFACT_BYTES)
        except ArtifactReadError as exc:
            raise AutopsyManifestError(f"invalid autopsy file: {exc}") from exc
        return cls.from_json(payload)


def load_packaged_autopsy_manifest() -> tuple[AutopsyManifest, str]:
    resource = resources.files("llmtracefx.optimizer.lab").joinpath(
        DEFAULT_AUTOPSY_MANIFEST_RESOURCE
    )
    payload = resource.read_text(encoding="utf-8")
    return (
        AutopsyManifest.from_json(payload),
        f"package:llmtracefx.optimizer.lab/{DEFAULT_AUTOPSY_MANIFEST_RESOURCE}",
    )


def load_autopsy_manifest(path: Path | None) -> tuple[AutopsyManifest, str]:
    if path is None:
        return load_packaged_autopsy_manifest()
    return AutopsyManifest.read_json(path), str(path)


def autopsy_manifest_hash(autopsy: AutopsyManifest) -> str:
    """Deterministic binding hash of the autopsy manifest's own fields.

    Evidence must be bound to the exact autopsy manifest, not only to the
    frontier manifest it is layered on top of: a change to this manifest
    (for example its timeout, checkpoint, or byte bounds) that leaves the
    frontier manifest untouched must still stale-out any prior journal,
    result, state, or report. Local artifact paths are excluded, mirroring
    ``frontier_manifest_hash``'s exclusion of its own artifact section.
    """
    return config_hash(
        {
            "schema_version": autopsy.schema_version,
            "autopsy_id": autopsy.autopsy_id,
            "frontier_manifest_resource": autopsy.frontier_manifest_resource,
            "frontier_manifest_sha256": autopsy.frontier_manifest_sha256,
            "frontier_id": autopsy.frontier_id,
            "base_lab_id": autopsy.base_lab_id,
            "model_repository_id": autopsy.model_repository_id,
            "model_revision": autopsy.model_revision,
            "model_expected_download_bytes": autopsy.model_expected_download_bytes,
            "tier_name": autopsy.tier_name,
            "tier_requested_tokens": autopsy.tier_requested_tokens,
            "child_timeout_seconds": autopsy.child_timeout_seconds,
            "process_cleanup_grace_seconds": autopsy.process_cleanup_grace_seconds,
            "journal_max_checkpoints": autopsy.journal_max_checkpoints,
            "journal_max_bytes": autopsy.journal_max_bytes,
        }
    )


def _packaged_frontier_text() -> str:
    resource = resources.files("llmtracefx.optimizer.lab").joinpath(
        DEFAULT_FRONTIER_MANIFEST_RESOURCE_NAME
    )
    return resource.read_text(encoding="utf-8")


def load_bound_frontier(
    autopsy: AutopsyManifest, *, frontier_manifest_path: Path | None
) -> tuple[FrontierManifest, LabManifest]:
    if frontier_manifest_path is None:
        payload = _packaged_frontier_text()
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        frontier = FrontierManifest.from_json(payload)
    else:
        payload = read_bounded_regular_text(
            frontier_manifest_path, MAX_METADATA_ARTIFACT_BYTES
        )
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        frontier = FrontierManifest.from_json(payload)
    if digest != autopsy.frontier_manifest_sha256:
        raise AutopsyManifestError(
            "frontier manifest does not match the autopsy binding"
        )
    if frontier.frontier_id != autopsy.frontier_id:
        raise AutopsyManifestError("autopsy frontier_id binding drifted")
    if frontier.base_lab_id != autopsy.base_lab_id:
        raise AutopsyManifestError("autopsy base_lab_id binding drifted")
    identity = (
        frontier.model_repository_id,
        frontier.model_revision,
        frontier.model_expected_download_bytes,
    )
    expected = (
        autopsy.model_repository_id,
        autopsy.model_revision,
        autopsy.model_expected_download_bytes,
    )
    if identity != expected:
        raise AutopsyManifestError("autopsy model identity binding drifted")
    try:
        tier = frontier.tier(autopsy.tier_name)
    except FrontierManifestError as exc:
        raise AutopsyManifestError(
            f"autopsy tier {autopsy.tier_name!r} is not defined by the frontier"
        ) from exc
    if tier.requested_tokens != autopsy.tier_requested_tokens:
        raise AutopsyManifestError("autopsy tier token binding drifted")
    base = load_bound_base_manifest(frontier)
    return frontier, base


# ---------------------------------------------------------------------------
# Direct MLX allocator counter probes. Never estimate: an absent API records
# a null value with a null API name rather than a fabricated zero.
# ---------------------------------------------------------------------------

_MLX_COUNTER_FUNCTIONS: tuple[tuple[str, str], ...] = (
    ("get_active_memory", "active_bytes"),
    ("get_cache_memory", "cache_bytes"),
    ("get_peak_memory", "peak_bytes"),
)


def _qualified_name(function: Any, fallback: str) -> str:
    module = getattr(function, "__module__", None) or "mlx.core"
    qualname = getattr(function, "__qualname__", None) or fallback
    return f"{module}.{qualname}"


def probe_mlx_counters(mlx_module: Any | None) -> dict[str, dict[str, Any]]:
    """Return one probe entry per direct MLX allocator counter.

    Each entry records the exact callable's qualified name and whether it is
    available on the installed MLX build, alongside its current value in
    bytes. A missing or failing counter is recorded as ``None``, never a
    guessed zero. ``error_category`` distinguishes "the API call raised" from
    "the API is simply unavailable", recording only the exception's type
    name -- never its message, which could carry unsanitized detail.
    """
    result: dict[str, dict[str, Any]] = {}
    for function_name, output_key in _MLX_COUNTER_FUNCTIONS:
        function = getattr(mlx_module, function_name, None) if mlx_module else None
        if not callable(function):
            result[output_key] = {
                "api": None,
                "value": None,
                "unit": "bytes",
                "error_category": None,
            }
            continue
        error_category: str | None = None
        try:
            value: int | None = int(function())
        except (RuntimeError, ValueError, TypeError, OSError, MemoryError) as exc:
            value = None
            error_category = type(exc).__name__
        result[output_key] = {
            "api": _qualified_name(function, function_name),
            "value": value,
            "unit": "bytes",
            "error_category": error_category,
        }
    return result


def probe_mlx_reset_peak_memory_api(mlx_module: Any | None) -> str | None:
    function = getattr(mlx_module, "reset_peak_memory", None) if mlx_module else None
    if not callable(function):
        return None
    return _qualified_name(function, "reset_peak_memory")


def _import_mlx_core() -> Any | None:
    try:
        import mlx.core as mlx_module
    except ImportError:
        return None
    return mlx_module


# ---------------------------------------------------------------------------
# Host process/system probes. Every provenance string names the exact source
# and unit; unavailable measurements are null, never a silent default.
# ---------------------------------------------------------------------------


def _run_probe_text(argv: list[str]) -> str | None:
    try:
        return subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            shell=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def host_process_rss_bytes(
    *,
    system: str | None = None,
    run_text: Callable[[list[str]], str | None] = _run_probe_text,
    linux_status_path: Path = Path("/proc/self/status"),
) -> tuple[int | None, str | None]:
    import platform as _platform

    resolved = system or _platform.system()
    if resolved == "Darwin":
        text = run_text(["ps", "-o", "rss=", "-p", str(os.getpid())])
        if text is None:
            return None, None
        try:
            kibibytes = int(text.strip())
        except ValueError:
            return None, None
        return kibibytes * 1024, "ps -o rss= (KiB, current process RSS)"
    if resolved == "Linux":
        try:
            status_text = read_bounded_regular_text(
                linux_status_path, MAX_METADATA_ARTIFACT_BYTES
            )
        except (ArtifactReadError, OSError):
            return None, None
        for line in status_text.splitlines():
            if line.startswith("VmRSS:"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        kibibytes = int(parts[1])
                    except ValueError:
                        return None, None
                    return (
                        kibibytes * 1024,
                        "Linux procfs self status VmRSS (kB, current process RSS)",
                    )
        return None, None
    return None, None


def host_process_max_rss_bytes(
    *,
    system: str | None = None,
    getrusage: Callable[[], Any] | None = None,
) -> tuple[int | None, str | None]:
    import platform as _platform

    resolved = system or _platform.system()
    if getrusage is None:
        try:
            import resource

            getrusage = lambda: resource.getrusage(resource.RUSAGE_SELF)  # noqa: E731
        except ImportError:
            return None, None
    try:
        usage = getrusage()
        raw = int(usage.ru_maxrss)
    except (OSError, ValueError, AttributeError, TypeError):
        return None, None
    if resolved == "Darwin":
        return raw, "getrusage(RUSAGE_SELF).ru_maxrss (bytes, macOS, process max RSS)"
    if resolved == "Linux":
        return (
            raw * 1024,
            "getrusage(RUSAGE_SELF).ru_maxrss (KiB, Linux, process max RSS)",
        )
    return None, None


def _parse_byte_quantity(value: str) -> int | None:
    import re

    match = re.search(r"([0-9]+(?:\.[0-9]+)?)([KMG])", value)
    if match is None:
        return None
    factors = {"K": 1024, "M": 1024**2, "G": 1024**3}
    return int(float(match.group(1)) * factors[match.group(2)])


def host_swap_bytes(
    *,
    system: str | None = None,
    run_text: Callable[[list[str]], str | None] = _run_probe_text,
    linux_meminfo_path: Path = Path("/proc/meminfo"),
) -> tuple[int | None, int | None, str | None]:
    import platform as _platform
    import re

    resolved = system or _platform.system()
    if resolved == "Darwin":
        text = run_text(["sysctl", "-n", "vm.swapusage"])
        if text is None:
            return None, None, None
        total_match = re.search(r"total\s*=\s*([0-9.]+[KMG])", text)
        used_match = re.search(r"used\s*=\s*([0-9.]+[KMG])", text)
        total = _parse_byte_quantity(total_match.group(1)) if total_match else None
        used = _parse_byte_quantity(used_match.group(1)) if used_match else None
        return total, used, "sysctl vm.swapusage (macOS)"
    if resolved == "Linux":
        try:
            meminfo_text = read_bounded_regular_text(
                linux_meminfo_path, MAX_METADATA_ARTIFACT_BYTES
            )
        except (ArtifactReadError, OSError):
            return None, None, None
        values: dict[str, int] = {}
        for line in meminfo_text.splitlines():
            key, _, remainder = line.partition(":")
            if key.strip() in ("SwapTotal", "SwapFree"):
                match = re.search(r"([0-9]+)", remainder)
                if match:
                    values[key.strip()] = int(match.group(1)) * 1024
        total = values.get("SwapTotal")
        free = values.get("SwapFree")
        used = None if total is None or free is None else total - free
        return total, used, "Linux procfs meminfo SwapTotal/SwapFree"
    return None, None, None


def build_checkpoint(
    stage: str,
    sequence: int,
    *,
    mlx_module: Any | None,
    started_monotonic: float,
    extra: dict[str, Any] | None = None,
    rss_probe: Callable[[], tuple[int | None, str | None]] = host_process_rss_bytes,
    max_rss_probe: Callable[
        [], tuple[int | None, str | None]
    ] = host_process_max_rss_bytes,
    swap_probe: Callable[
        [], tuple[int | None, int | None, str | None]
    ] = host_swap_bytes,
) -> dict[str, Any]:
    """Build one privacy-safe stage checkpoint with exact scopes.

    Never carries a PID, path, hostname, username, prompt, or response.
    """
    counters = probe_mlx_counters(mlx_module)
    try:
        rss_value, rss_provenance = rss_probe()
    except (MemoryError, OSError, RuntimeError, TypeError, ValueError) as exc:
        rss_value, rss_provenance = None, f"probe failed ({type(exc).__name__})"
    try:
        max_rss_value, max_rss_provenance = max_rss_probe()
    except (MemoryError, OSError, RuntimeError, TypeError, ValueError) as exc:
        max_rss_value, max_rss_provenance = (
            None,
            f"probe failed ({type(exc).__name__})",
        )
    try:
        swap_total, swap_used, swap_provenance = swap_probe()
    except (MemoryError, OSError, RuntimeError, TypeError, ValueError) as exc:
        swap_total, swap_used, swap_provenance = (
            None,
            None,
            f"probe failed ({type(exc).__name__})",
        )
    entry: dict[str, Any] = {
        "stage": stage,
        "sequence": sequence,
        "wall_clock_utc": utc_now_iso(),
        "wall_clock_provenance": "schema.utc_now_iso (ISO-8601, UTC)",
        "monotonic_offset_seconds": max(0.0, time.monotonic() - started_monotonic),
        "monotonic_offset_unit": "seconds",
        "monotonic_offset_provenance": (
            "time.monotonic() delta since the child process's own start"
        ),
        "mlx_allocator": {
            "scope": "mlx_allocator",
            "unit": "bytes",
            "active_bytes": counters["active_bytes"],
            "cache_bytes": counters["cache_bytes"],
            "peak_bytes": counters["peak_bytes"],
        },
        "host_process": {
            "scope": "host_process",
            "unit": "bytes",
            "rss_bytes": {"value": rss_value, "provenance": rss_provenance},
            "max_rss_bytes": {"value": max_rss_value, "provenance": max_rss_provenance},
        },
        "host_system": {
            "scope": "host_system_swap",
            "unit": "bytes",
            "swap_total_bytes": {"value": swap_total, "provenance": swap_provenance},
            "swap_used_bytes": {"value": swap_used, "provenance": swap_provenance},
        },
    }
    if extra:
        entry.update(extra)
    return entry


# ---------------------------------------------------------------------------
# Bounded, atomically-persisted journal.
# ---------------------------------------------------------------------------


class AutopsyJournal:
    """Bounded, replace-safe checkpoint journal for one isolated attempt.

    Persisted through ``atomic_write_text`` after every checkpoint so an
    abrupt timeout, signal, or forced parent cleanup leaves the last valid
    checkpoint on disk as usable failure evidence rather than nothing.
    """

    def __init__(
        self,
        *,
        path: Path,
        autopsy_id: str,
        autopsy_manifest_hash_value: str,
        frontier_manifest_hash_value: str,
        model_repository_id: str,
        model_revision: str,
        tier: str,
        run_mode: str,
        clean_boot_confirmed: bool,
        max_checkpoints: int,
        max_bytes: int,
    ) -> None:
        self.path = path
        self.autopsy_id = autopsy_id
        self.autopsy_manifest_hash = autopsy_manifest_hash_value
        self.frontier_manifest_hash = frontier_manifest_hash_value
        self.model_repository_id = model_repository_id
        self.model_revision = model_revision
        self.tier = tier
        self.run_mode = run_mode
        self.clean_boot_confirmed = clean_boot_confirmed
        self.max_checkpoints = max_checkpoints
        self.max_bytes = max_bytes
        self.checkpoints: list[dict[str, Any]] = []
        self.complete = False
        self.terminal: str | None = None
        self.started_monotonic = time.monotonic()
        self.started_at = utc_now_iso()

    def checkpoint(
        self, stage: str, mlx_module: Any | None, *, extra: dict[str, Any] | None = None
    ) -> None:
        if len(self.checkpoints) >= self.max_checkpoints:
            raise LabError("autopsy journal exceeded the maximum checkpoint count")
        entry = build_checkpoint(
            stage,
            len(self.checkpoints),
            mlx_module=mlx_module,
            started_monotonic=self.started_monotonic,
            extra=extra,
        )
        self.checkpoints.append(entry)
        try:
            self._persist()
        except (LabError, MemoryError, OSError):
            self.checkpoints.pop()
            raise

    def finalize(self, *, complete: bool, terminal: str) -> None:
        self.complete = complete
        self.terminal = terminal
        self._persist()

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": AUTOPSY_JOURNAL_SCHEMA_VERSION,
            "autopsy_id": self.autopsy_id,
            "autopsy_manifest_hash": self.autopsy_manifest_hash,
            "frontier_manifest_hash": self.frontier_manifest_hash,
            "model": {
                "repository_id": self.model_repository_id,
                "revision": self.model_revision,
            },
            "tier": self.tier,
            "run_mode": self.run_mode,
            "clean_boot_confirmed": self.clean_boot_confirmed,
            "synthetic": False,
            "started_at": self.started_at,
            "sampling": {
                "periodic_sampling_enabled": False,
                "note": (
                    "Only discrete stage-boundary checkpoints are recorded; "
                    "no interval polling occurs."
                ),
            },
            "peak_memory_reset": {
                "applied": False,
                "api": None,
                "note": (
                    "Peak memory is never reset after model load, so a fresh "
                    "subprocess's peak reading includes load growth."
                ),
            },
            "checkpoints": self.checkpoints,
            "complete": self.complete,
            "terminal": self.terminal,
        }

    def _persist(self) -> None:
        body = self._payload()
        envelope_hash = config_hash(body)
        body_with_hash = dict(body)
        body_with_hash["envelope_sha256"] = envelope_hash
        text = json.dumps(body_with_hash, indent=2, sort_keys=False) + "\n"
        if len(text.encode("utf-8")) > self.max_bytes:
            raise LabError("autopsy journal exceeded the maximum artifact size")
        atomic_write_text(self.path, text)


def _record_signal_checkpoint(
    journal: AutopsyJournal, mlx_module: Any | None, signum: int
) -> None:
    """Best-effort signal evidence: each write is independent of the others.

    The ``signal_received`` stage is never mislabeled as ``cleanup``; the two
    are recorded as distinct checkpoints so an abrupt-termination trace is
    unambiguous. A failure writing one checkpoint (for example the
    checkpoint or byte bound already being exhausted) must not prevent the
    remaining best-effort writes, and must never prevent the process from
    still exiting.
    """
    name = signal.Signals(signum).name
    try:
        journal.checkpoint(
            "signal_received", mlx_module, extra={"signal_received": name}
        )
    except (LabError, MemoryError, OSError):
        # Preserve any earlier checkpoint and continue the best-effort sequence.
        pass
    try:
        journal.checkpoint("cleanup", mlx_module)
    except (LabError, MemoryError, OSError):
        # A signal must still terminate even when cleanup evidence cannot persist.
        pass
    try:
        journal.finalize(complete=False, terminal="signal")
    except (LabError, MemoryError, OSError):
        # The last valid atomic journal remains usable if finalization fails.
        pass


def install_signal_handlers(journal: AutopsyJournal, mlx_module: Any | None) -> None:
    def handler(signum: int, frame: Any) -> None:
        try:
            _record_signal_checkpoint(journal, mlx_module, signum)
        finally:
            os._exit(128 + signum)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def _validate_measurement(value: Any, *, allow_provenance: bool) -> str | None:
    if not isinstance(value, dict):
        return "measurement is not an object"
    raw = value.get("value")
    if raw is not None and (
        isinstance(raw, bool) or not isinstance(raw, int) or raw < 0
    ):
        return "measurement value must be a non-negative integer or null"
    if allow_provenance:
        provenance = value.get("provenance")
        if provenance is not None and not isinstance(provenance, str):
            return "measurement provenance must be a string or null"
    else:
        api = value.get("api")
        if api is not None and not isinstance(api, str):
            return "measurement api must be a string or null"
        if value.get("unit") != "bytes":
            return "measurement unit must be 'bytes'"
        error_category = value.get("error_category")
        if error_category is not None and not isinstance(error_category, str):
            return "measurement error_category must be a string or null"
    return None


def _validate_scope(
    scope: Any, *, key: str, expected_scope: str, sub_keys: tuple[str, ...]
) -> str | None:
    if not isinstance(scope, dict):
        return f"checkpoint {key} scope must be an object"
    if scope.get("scope") != expected_scope:
        return f"checkpoint {key} scope label must be {expected_scope!r}"
    if scope.get("unit") != "bytes":
        return f"checkpoint {key} unit must be 'bytes'"
    for sub_key in sub_keys:
        reason = _validate_measurement(
            scope.get(sub_key), allow_provenance=(key != "mlx_allocator")
        )
        if reason is not None:
            return f"checkpoint {key}.{sub_key}: {reason}"
    return None


def _validate_checkpoint_shape(entry: dict[str, Any], *, stage: str) -> str | None:
    unknown = set(entry) - _CHECKPOINT_TOP_LEVEL_KEYS
    if unknown:
        return f"checkpoint carries unexpected field(s): {sorted(unknown)}"
    wall_clock = entry.get("wall_clock_utc")
    if not isinstance(wall_clock, str) or not wall_clock:
        return "checkpoint wall_clock_utc must be a non-empty string"
    if entry.get("wall_clock_provenance") is not None and not isinstance(
        entry.get("wall_clock_provenance"), str
    ):
        return "checkpoint wall_clock_provenance must be a string or null"
    offset = entry.get("monotonic_offset_seconds")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, (int, float))
        or not math.isfinite(float(offset))
        or offset < 0
    ):
        return "checkpoint monotonic_offset_seconds must be finite and non-negative"
    if entry.get("monotonic_offset_unit") not in (None, "seconds"):
        return "checkpoint monotonic_offset_unit must be 'seconds' or null"
    reason = _validate_scope(
        entry.get("mlx_allocator"),
        key="mlx_allocator",
        expected_scope="mlx_allocator",
        sub_keys=("active_bytes", "cache_bytes", "peak_bytes"),
    )
    if reason is not None:
        return reason
    reason = _validate_scope(
        entry.get("host_process"),
        key="host_process",
        expected_scope="host_process",
        sub_keys=("rss_bytes", "max_rss_bytes"),
    )
    if reason is not None:
        return reason
    reason = _validate_scope(
        entry.get("host_system"),
        key="host_system",
        expected_scope="host_system_swap",
        sub_keys=("swap_total_bytes", "swap_used_bytes"),
    )
    if reason is not None:
        return reason
    signal_received = entry.get("signal_received")
    if stage == "signal_received":
        if signal_received not in _HANDLED_SIGNAL_NAMES:
            return "signal_received checkpoint must record a known signal name"
    elif signal_received is not None:
        return "only the signal_received checkpoint may carry signal_received"
    return None


def _validate_checkpoint_sequence(checkpoints: Any) -> str | None:
    if not isinstance(checkpoints, list) or not checkpoints:
        return "journal checkpoints must be a non-empty list"
    if checkpoints[0].get("stage") != "child_start":
        return "the first checkpoint must be child_start"
    highest_main_index = -1
    seen_terminal_only: set[str] = set()
    for index, entry in enumerate(checkpoints):
        if not isinstance(entry, dict):
            return "checkpoint entries must be objects"
        sequence = entry.get("sequence")
        if isinstance(sequence, bool) or sequence != index:
            return "checkpoint sequence numbers must be contiguous from zero"
        stage = entry.get("stage")
        if stage not in STAGE_SEQUENCE:
            return f"checkpoint stage {stage!r} is not an allowed stage"
        if stage in MAIN_STAGE_SEQUENCE:
            stage_index = MAIN_STAGE_SEQUENCE.index(stage)
            if stage_index <= highest_main_index:
                return "main-path checkpoint stages are out of order or repeated"
            highest_main_index = stage_index
        else:
            if stage in seen_terminal_only:
                return f"checkpoint stage {stage!r} appears more than once"
            seen_terminal_only.add(stage)
            if stage == "cleanup" and index != len(checkpoints) - 1:
                return "a cleanup checkpoint must be the final checkpoint"
        reason = _validate_checkpoint_shape(entry, stage=stage)
        if reason is not None:
            return reason
    return None


def _validate_terminal_semantics(
    checkpoints: list[dict[str, Any]], *, complete: Any, terminal: Any
) -> str | None:
    if not isinstance(complete, bool):
        return "journal complete flag must be a boolean"
    if terminal is not None and not isinstance(terminal, str):
        return "journal terminal must be a string or null"
    stages = {entry.get("stage") for entry in checkpoints}
    has_completion = "completion" in stages
    has_caught_oom = "caught_oom" in stages
    has_signal = "signal_received" in stages
    if terminal is None:
        if complete:
            return "a journal cannot be complete with no terminal outcome"
        if has_completion or has_caught_oom or has_signal:
            return (
                "a journal without a terminal outcome must not record a "
                "terminal-only checkpoint"
            )
        return None
    if terminal in _COMPLETE_TERMINALS:
        if not complete:
            return f"journal terminal {terminal!r} requires complete=true"
        if terminal == "completed":
            if not has_completion or has_caught_oom or has_signal:
                return (
                    "a completed journal must record completion and no "
                    "failure checkpoints"
                )
        elif terminal == "oom":
            if not has_caught_oom or has_completion or has_signal:
                return (
                    "an oom journal must record caught_oom and no "
                    "completion/signal checkpoints"
                )
        elif has_completion or has_caught_oom or has_signal:
            return (
                f"a {terminal} journal must not record completion/caught_oom/"
                "signal checkpoints"
            )
        return None
    if terminal in _INCOMPLETE_TERMINALS:
        if complete:
            return "a signal journal must have complete=false"
        if not has_signal:
            return "a signal journal must record a signal_received checkpoint"
        if has_completion or has_caught_oom:
            return (
                "a signal journal must not record completion/caught_oom " "checkpoints"
            )
        return None
    return f"journal terminal {terminal!r} is not a recognized outcome"


def _journal_invalid_reason(
    journal: Any,
    autopsy: AutopsyManifest,
    frontier: FrontierManifest,
    *,
    run_mode: str,
    clean_boot_confirmed: bool,
) -> str | None:
    """Return why ``journal`` is not usable evidence, or ``None`` if valid.

    Every check is explicit so a stale, tampered, or foreign journal is
    never silently treated as merely absent (see ``JournalLoadResult``).
    """
    if not isinstance(journal, dict):
        return "journal is not an object"
    unknown = set(journal) - _JOURNAL_TOP_LEVEL_KEYS
    if unknown:
        return f"journal carries unexpected field(s): {sorted(unknown)}"
    envelope = journal.get("envelope_sha256")
    canonical = {
        key: value for key, value in journal.items() if key != "envelope_sha256"
    }
    if not isinstance(envelope, str) or config_hash(canonical) != envelope:
        return "journal envelope hash does not match its own contents"
    if journal.get("schema_version") != AUTOPSY_JOURNAL_SCHEMA_VERSION:
        return "journal schema_version is unsupported"
    if journal.get("autopsy_id") != autopsy.autopsy_id:
        return "journal autopsy_id does not match the bound manifest"
    if journal.get("autopsy_manifest_hash") != autopsy_manifest_hash(autopsy):
        return "journal autopsy_manifest_hash does not match the bound manifest"
    if journal.get("frontier_manifest_hash") != frontier_manifest_hash(frontier):
        return "journal frontier_manifest_hash does not match the bound frontier"
    model = journal.get("model")
    if (
        not isinstance(model, dict)
        or set(model) != {"repository_id", "revision"}
        or model.get("repository_id") != frontier.model_repository_id
        or model.get("revision") != frontier.model_revision
    ):
        return "journal model identity does not match the bound frontier"
    if journal.get("tier") != autopsy.tier_name:
        return "journal tier does not match the bound manifest"
    if journal.get("run_mode") != run_mode:
        return "journal run_mode does not match the requested run"
    if journal.get("clean_boot_confirmed") is not clean_boot_confirmed:
        return "journal clean_boot_confirmed does not match the requested run"
    if journal.get("synthetic") is not False:
        return "journal synthetic flag must be false"
    if not isinstance(journal.get("started_at"), str) or not journal["started_at"]:
        return "journal started_at must be a non-empty string"
    sampling = journal.get("sampling")
    if (
        not isinstance(sampling, dict)
        or set(sampling) - {"periodic_sampling_enabled", "note"}
        or sampling.get("periodic_sampling_enabled") is not False
    ):
        return "journal sampling.periodic_sampling_enabled must be false"
    peak_reset = journal.get("peak_memory_reset")
    if (
        not isinstance(peak_reset, dict)
        or set(peak_reset) - {"applied", "api", "note"}
        or peak_reset.get("applied") is not False
        or peak_reset.get("api") is not None
    ):
        return (
            "journal peak_memory_reset must explicitly record that peak "
            "memory was never reset after model load"
        )
    checkpoints = journal.get("checkpoints")
    if not isinstance(checkpoints, list):
        return "journal checkpoints must be a list"
    reason = _validate_checkpoint_sequence(checkpoints)
    if reason is not None:
        return reason
    if len(checkpoints) > autopsy.journal_max_checkpoints:
        return "journal checkpoints exceed the manifest-bound maximum count"
    return _validate_terminal_semantics(
        checkpoints, complete=journal.get("complete"), terminal=journal.get("terminal")
    )


@dataclass(frozen=True)
class JournalLoadResult:
    """Explicit outcome of loading a journal artifact from disk.

    ``status`` is one of ``"missing"`` (no journal artifact exists),
    ``"invalid"`` (an artifact exists but is stale, tampered, foreign, a
    symlink, or otherwise fails full contract validation -- ``reason``
    explains why), or ``"valid"`` (fully parsed and validated, regardless of
    whether it records a complete or an incomplete attempt).
    """

    status: str
    journal: dict[str, Any] | None
    digest: str | None
    reason: str | None


def _load_journal_if_valid(
    path: Path,
    autopsy: AutopsyManifest,
    frontier: FrontierManifest,
    *,
    run_mode: str,
    clean_boot_confirmed: bool,
) -> JournalLoadResult:
    if path.is_symlink():
        return JournalLoadResult("invalid", None, None, "journal path is a symlink")
    if not path.is_file():
        return JournalLoadResult("missing", None, None, None)
    try:
        journal = _read_json(path)
    except LabError as exc:
        return JournalLoadResult("invalid", None, None, f"journal is unreadable: {exc}")
    reason = _journal_invalid_reason(
        journal,
        autopsy,
        frontier,
        run_mode=run_mode,
        clean_boot_confirmed=clean_boot_confirmed,
    )
    if reason is not None:
        return JournalLoadResult("invalid", None, None, reason)
    return JournalLoadResult("valid", journal, _file_sha256(path), None)


# ---------------------------------------------------------------------------
# Isolated child execution.
# ---------------------------------------------------------------------------


def execute_autopsy_child(
    autopsy: AutopsyManifest,
    frontier: FrontierManifest,
    base: LabManifest,
    *,
    model_path: Path,
    output_dir: Path,
    run_mode: str,
    clean_boot_confirmed: bool,
    runtime_factory: Callable[[], Any] | None = None,
    mlx_module_factory: Callable[[], Any | None] = _import_mlx_core,
) -> dict[str, Any]:
    tier = frontier.tier(autopsy.tier_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    journal_path = output_dir / "journal.json"
    result_path = output_dir / "result.json"
    mlx_module = mlx_module_factory()
    journal = AutopsyJournal(
        path=journal_path,
        autopsy_id=autopsy.autopsy_id,
        autopsy_manifest_hash_value=autopsy_manifest_hash(autopsy),
        frontier_manifest_hash_value=frontier_manifest_hash(frontier),
        model_repository_id=frontier.model_repository_id,
        model_revision=frontier.model_revision,
        tier=autopsy.tier_name,
        run_mode=run_mode,
        clean_boot_confirmed=clean_boot_confirmed,
        max_checkpoints=autopsy.journal_max_checkpoints,
        max_bytes=autopsy.journal_max_bytes,
    )
    install_signal_handlers(journal, mlx_module)
    journal.checkpoint("child_start", mlx_module)
    actual_tokens: int | None = None
    status = "failed"
    reason: str | None = "autopsy child did not reach a terminal stage"
    try:
        factory = runtime_factory or (
            lambda: MLXVLMRuntime(
                temperature=base.generation.temperature,
                top_p=base.generation.top_p,
                enable_thinking=base.generation.enable_thinking,
                prefill_step_size=base.runtime.prefill_step_size,
            )
        )
        runtime = factory()
        journal.checkpoint("before_model_load", mlx_module)
        model, processor = runtime.load_model(model_path)
        runtime.synchronize()
        journal.checkpoint("after_model_load", mlx_module)
        workload = workload_by_id(frontier.workload_id)
        prompt, prompt_tokens = fit_prompt(
            lambda text: runtime.encode(processor, text),
            base_prompt=workload.base_prompt,
            requested_tokens=tier.requested_tokens,
            maximum_shortfall=frontier.maximum_token_shortfall,
        )
        del prompt
        actual_tokens = len(prompt_tokens)
        journal.checkpoint("after_prompt_tokenization", mlx_module)
        runtime.seed(base.generation.seed)
        runtime.synchronize()
        journal.checkpoint("immediately_before_prefill_generation", mlx_module)
        first_token_seen = False
        for response in runtime.stream_generate(
            model,
            processor,
            prompt_tokens,
            max_tokens=frontier.max_output_tokens,
            draft_model=None,
            num_draft_tokens=0,
        ):
            if response is None:
                raise LabError("MLX-VLM generation yielded an invalid response")
            if not first_token_seen:
                journal.checkpoint("after_first_token", mlx_module)
                first_token_seen = True
        runtime.synchronize()
        journal.checkpoint("completion", mlx_module)
        status, reason = "completed", None
    except (
        KeyError,
        RuntimeError,
        ValueError,
        OSError,
        MemoryError,
        TimeoutError,
    ) as exc:
        classified_status, classified_reason = _classify_error(
            type(exc).__name__, str(exc)
        )
        if classified_status == "oom":
            journal.checkpoint("caught_oom", mlx_module)
        status, reason = classified_status, classified_reason
    finally:
        try:
            journal.checkpoint("cleanup", mlx_module)
        except (LabError, MemoryError, OSError):
            # Finalization can still bind the preceding valid checkpoints.
            pass
        journal.finalize(complete=True, terminal=status)

    result = {
        "schema_version": AUTOPSY_RESULT_SCHEMA_VERSION,
        "autopsy_id": autopsy.autopsy_id,
        "autopsy_manifest_hash": autopsy_manifest_hash(autopsy),
        "frontier_manifest_hash": frontier_manifest_hash(frontier),
        "run_mode": run_mode,
        "clean_boot_confirmed": clean_boot_confirmed,
        "tier": autopsy.tier_name,
        "requested_tokens": tier.requested_tokens,
        "actual_tokens": actual_tokens,
        "status": status,
        "reason": reason,
        "journal_sha256": _file_sha256(journal_path),
        "journal_complete": journal.complete,
        "journal_terminal": journal.terminal,
        "synthetic": False,
    }
    atomic_write_text(result_path, json.dumps(result, indent=2, sort_keys=False) + "\n")
    return result


def _validate_autopsy_result(
    result: dict[str, Any],
    autopsy: AutopsyManifest,
    frontier: FrontierManifest,
    *,
    run_mode: str,
    clean_boot_confirmed: bool,
) -> None:
    if not isinstance(result, dict):
        raise LabError("invalid autopsy artifact: result is not an object")
    expected = {
        "schema_version": AUTOPSY_RESULT_SCHEMA_VERSION,
        "autopsy_id": autopsy.autopsy_id,
        "autopsy_manifest_hash": autopsy_manifest_hash(autopsy),
        "frontier_manifest_hash": frontier_manifest_hash(frontier),
        "run_mode": run_mode,
        "clean_boot_confirmed": clean_boot_confirmed,
        "tier": autopsy.tier_name,
        "requested_tokens": frontier.tier(autopsy.tier_name).requested_tokens,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise LabError(f"stale autopsy artifact: {key} does not match")
    status = result.get("status")
    if status not in RESUMABLE_TERMINAL_STATUSES:
        raise LabError("invalid autopsy artifact: unsupported status")
    if result.get("synthetic") is not False:
        raise LabError("invalid autopsy artifact: synthetic must be false")
    reason = result.get("reason")
    if status == "completed":
        if reason is not None:
            raise LabError(
                "invalid autopsy artifact: completed result must have no reason"
            )
    elif not isinstance(reason, str) or not reason:
        raise LabError(
            "invalid autopsy artifact: non-completed result must carry a reason"
        )
    journal_complete = result.get("journal_complete")
    if journal_complete is not None and not isinstance(journal_complete, bool):
        raise LabError(
            "invalid autopsy artifact: journal_complete must be a boolean or null"
        )
    journal_terminal = result.get("journal_terminal")
    if journal_terminal is not None and not isinstance(journal_terminal, str):
        raise LabError(
            "invalid autopsy artifact: journal_terminal must be a string or null"
        )
    journal_digest = result.get("journal_sha256")
    if journal_digest is not None and (
        not isinstance(journal_digest, str)
        or len(journal_digest) != 64
        or any(char not in "0123456789abcdef" for char in journal_digest)
    ):
        raise LabError("invalid autopsy artifact: journal digest is malformed")
    journal_claimed = journal_complete is not None or journal_terminal is not None
    if journal_claimed and journal_digest is None:
        raise LabError(
            "invalid autopsy artifact: a claimed journal outcome requires a digest"
        )
    if status == "completed" and not (
        journal_complete is True
        and journal_terminal == "completed"
        and journal_digest is not None
    ):
        raise LabError(
            "invalid autopsy artifact: completed status requires a valid, "
            "complete, terminal=completed journal"
        )
    actual = result.get("actual_tokens")
    requested = frontier.tier(autopsy.tier_name).requested_tokens
    if actual is not None:
        if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
            raise LabError(
                "invalid autopsy artifact: actual token count must be a "
                "non-negative integer or null"
            )
        if actual > requested:
            raise LabError(
                "invalid autopsy artifact: actual token count exceeds the "
                "requested tier"
            )
        if requested - actual > frontier.maximum_token_shortfall:
            raise LabError(
                "invalid autopsy artifact: actual token count shortfall "
                "exceeds the frontier's maximum"
            )


# ---------------------------------------------------------------------------
# Process-isolated parent orchestration.
# ---------------------------------------------------------------------------


def launch_autopsy_subprocess(
    *,
    autopsy_manifest_path: Path | None,
    frontier_manifest_path: Path | None,
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
        "llmtracefx.optimizer.lab.autopsy",
        "_child",
        "--model-path",
        str(model_path),
        "--output-dir",
        str(output_dir),
        "--mode",
        run_mode,
    ]
    if autopsy_manifest_path is not None:
        argv.extend(("--manifest", str(autopsy_manifest_path)))
    if frontier_manifest_path is not None:
        argv.extend(("--frontier-manifest", str(frontier_manifest_path)))
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


def _finalize_autopsy_attempt(
    autopsy: AutopsyManifest,
    frontier: FrontierManifest,
    *,
    run_mode: str,
    clean_boot_confirmed: bool,
    attempt_dir: Path,
    child: ChildProcessResult,
) -> dict[str, Any]:
    result_path = attempt_dir / "result.json"
    journal_path = attempt_dir / "journal.json"
    process_evidence = {
        "fresh_subprocess": True,
        "new_session": True,
        "child_exit_code": child.exit_code,
        "timed_out": child.timed_out,
        "descendants_cleaned": child.descendants_cleaned,
    }
    journal_load = _load_journal_if_valid(
        journal_path,
        autopsy,
        frontier,
        run_mode=run_mode,
        clean_boot_confirmed=clean_boot_confirmed,
    )
    journal_complete = (
        journal_load.journal.get("complete")
        if journal_load.status == "valid" and journal_load.journal is not None
        else None
    )
    journal_terminal = (
        journal_load.journal.get("terminal")
        if journal_load.status == "valid" and journal_load.journal is not None
        else None
    )

    def synth(status: str, reason: str) -> dict[str, Any]:
        return {
            "schema_version": AUTOPSY_RESULT_SCHEMA_VERSION,
            "autopsy_id": autopsy.autopsy_id,
            "autopsy_manifest_hash": autopsy_manifest_hash(autopsy),
            "frontier_manifest_hash": frontier_manifest_hash(frontier),
            "run_mode": run_mode,
            "clean_boot_confirmed": clean_boot_confirmed,
            "tier": autopsy.tier_name,
            "requested_tokens": frontier.tier(autopsy.tier_name).requested_tokens,
            "actual_tokens": None,
            "status": status,
            "reason": reason,
            "journal_sha256": journal_load.digest,
            "journal_complete": journal_complete,
            "journal_terminal": journal_terminal,
            "synthetic": False,
        }

    if not child.descendants_cleaned:
        result = synth("failed", "child process group cleanup could not be verified")
    elif child.timed_out:
        # A parent-enforced timeout may legitimately race a valid-but-
        # incomplete journal (the child was mid-attempt); that evidence is
        # preserved rather than discarded. A missing or invalid journal
        # stays explicitly null.
        result = synth("timeout", "autopsy exceeded parent-enforced timeout")
    elif not result_path.is_file():
        result = synth("failed", "child exited without producing a result artifact")
    else:
        try:
            candidate = _read_json(result_path)
            _validate_autopsy_result(
                candidate,
                autopsy,
                frontier,
                run_mode=run_mode,
                clean_boot_confirmed=clean_boot_confirmed,
            )
        except LabError:
            result = synth("failed", "child artifact failed validation")
        else:
            if journal_load.status != "valid":
                result = synth(
                    "failed",
                    "child artifact claims a journal that is missing or invalid: "
                    f"{journal_load.reason or 'no journal was written'}",
                )
            elif candidate.get("journal_sha256") != journal_load.digest:
                result = synth(
                    "failed",
                    "child artifact journal digest does not match the actual "
                    "validated journal on disk",
                )
            elif (
                candidate.get("journal_complete") != journal_complete
                or candidate.get("journal_terminal") != journal_terminal
            ):
                result = synth(
                    "failed",
                    "child artifact journal completion/terminal claims do not "
                    "match the actual validated journal",
                )
            elif candidate["status"] != journal_terminal:
                result = synth(
                    "failed",
                    "child artifact status does not match the actual validated "
                    "journal terminal",
                )
            elif candidate["status"] == "completed" and not (
                journal_complete is True and journal_terminal == "completed"
            ):
                result = synth(
                    "failed",
                    "completed status requires a valid, complete, "
                    "terminal=completed journal",
                )
            elif candidate["status"] == "completed" and child.exit_code != 0:
                result = synth(
                    "failed", "successful child artifact has a nonzero exit code"
                )
            elif candidate["status"] != "completed" and child.exit_code == 0:
                result = synth("failed", "failed child artifact has a zero exit code")
            else:
                result = candidate
    result["process"] = process_evidence
    atomic_write_text(result_path, json.dumps(result, indent=2, sort_keys=False) + "\n")
    return result


def run_autopsy(
    autopsy: AutopsyManifest,
    frontier: FrontierManifest,
    base: LabManifest,
    *,
    autopsy_manifest_path: Path | None,
    frontier_manifest_path: Path | None,
    workspace: Path,
    model_path: Path,
    run_mode: str,
    clean_boot_confirmed: bool,
    resume: bool,
    launcher: Callable[..., ChildProcessResult] = launch_autopsy_subprocess,
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
            "autopsy run blocked by safety preflight: " + "; ".join(preflight.blockers)
        )
    mode_workspace = workspace / run_mode
    result_path = mode_workspace / "result.json"
    state_path = mode_workspace / "state.json"
    if resume and result_path.is_file():
        prior = _read_json(result_path)
        _validate_autopsy_result(
            prior,
            autopsy,
            frontier,
            run_mode=run_mode,
            clean_boot_confirmed=clean_boot_confirmed,
        )
        if prior.get("status") in RESUMABLE_TERMINAL_STATUSES:
            journal_path = mode_workspace / "journal.json"
            prior_digest = prior.get("journal_sha256")
            journal_load = _load_journal_if_valid(
                journal_path,
                autopsy,
                frontier,
                run_mode=run_mode,
                clean_boot_confirmed=clean_boot_confirmed,
            )
            if journal_load.status == "invalid":
                raise LabError(
                    "cannot resume: prior checkpoint journal is invalid: "
                    f"{journal_load.reason}"
                )
            if prior_digest is not None:
                if (
                    journal_load.status != "valid"
                    or journal_load.digest != prior_digest
                    or journal_load.journal is None
                    or journal_load.journal.get("complete")
                    != prior.get("journal_complete")
                    or journal_load.journal.get("terminal")
                    != prior.get("journal_terminal")
                ):
                    raise LabError(
                        "cannot resume: prior result's journal binding no longer "
                        "matches a valid, consistent journal on disk"
                    )
            elif journal_load.status == "valid":
                raise LabError(
                    "cannot resume: a valid checkpoint journal exists but the "
                    "prior result does not bind it"
                )
            resumed = dict(prior)
            resumed["reason"] = (
                resumed.get("reason") or "resumed prior terminal attempt"
            )
            state = {
                "schema_version": AUTOPSY_STATE_SCHEMA_VERSION,
                "autopsy_id": autopsy.autopsy_id,
                "autopsy_manifest_hash": autopsy_manifest_hash(autopsy),
                "frontier_manifest_hash": frontier_manifest_hash(frontier),
                "run_mode": run_mode,
                "clean_boot_confirmed": clean_boot_confirmed,
                "synthetic": False,
                "started_at": utc_now_iso(),
                "ended_at": utc_now_iso(),
                "status": resumed["status"],
                "resumed": True,
                "pre_run_machine_state": machine_state(preflight),
                "result": resumed,
            }
            atomic_write_text(state_path, json.dumps(state, indent=2) + "\n")
            return state
    mode_workspace.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": AUTOPSY_STATE_SCHEMA_VERSION,
        "autopsy_id": autopsy.autopsy_id,
        "autopsy_manifest_hash": autopsy_manifest_hash(autopsy),
        "frontier_manifest_hash": frontier_manifest_hash(frontier),
        "run_mode": run_mode,
        "clean_boot_confirmed": clean_boot_confirmed,
        "synthetic": False,
        "started_at": utc_now_iso(),
        "ended_at": None,
        "status": "running",
        "resumed": False,
        "pre_run_machine_state": machine_state(preflight),
        "result": None,
    }
    atomic_write_text(state_path, json.dumps(state, indent=2) + "\n")
    child = launcher(
        autopsy_manifest_path=autopsy_manifest_path,
        frontier_manifest_path=frontier_manifest_path,
        model_path=model_path,
        output_dir=mode_workspace,
        run_mode=run_mode,
        clean_boot_confirmed=clean_boot_confirmed,
        timeout_seconds=autopsy.child_timeout_seconds,
        cleanup_grace_seconds=autopsy.process_cleanup_grace_seconds,
    )
    result = _finalize_autopsy_attempt(
        autopsy,
        frontier,
        run_mode=run_mode,
        clean_boot_confirmed=clean_boot_confirmed,
        attempt_dir=mode_workspace,
        child=child,
    )
    state["result"] = result
    state["ended_at"] = utc_now_iso()
    state["status"] = result["status"]
    atomic_write_text(state_path, json.dumps(state, indent=2) + "\n")
    return state


def verify_autopsy_evidence(
    autopsy: AutopsyManifest,
    frontier: FrontierManifest,
    *,
    workspace: Path,
    run_mode: str,
) -> dict[str, Any]:
    state_path = workspace / run_mode / "state.json"
    failures: list[str] = []
    if not state_path.is_file():
        return {"verified": False, "failures": ["state artifact is missing"]}
    state = _read_json(state_path)
    if state.get("status") == "running":
        failures.append("state artifact is incomplete")
    if state.get("autopsy_manifest_hash") != autopsy_manifest_hash(autopsy):
        failures.append("state autopsy manifest binding is stale")
    if state.get("frontier_manifest_hash") != frontier_manifest_hash(frontier):
        failures.append("state frontier manifest binding is stale")
    if state.get("run_mode") != run_mode:
        failures.append("state run mode does not match")
    if state.get("autopsy_id") != autopsy.autopsy_id:
        failures.append("state autopsy_id does not match")
    if state.get("synthetic") is not False:
        failures.append("state synthetic flag must be false")
    clean_boot = state.get("clean_boot_confirmed")
    if run_mode == "publication" and clean_boot is not True:
        failures.append("publication state lacks operator clean-boot confirmation")
    result = state.get("result")
    if not isinstance(result, dict):
        failures.append("state is missing a result")
        return {"verified": not failures, "failures": failures}
    if state.get("status") != result.get("status"):
        failures.append("state status does not match its result")
    try:
        _validate_autopsy_result(
            result,
            autopsy,
            frontier,
            run_mode=run_mode,
            clean_boot_confirmed=bool(clean_boot),
        )
    except LabError as exc:
        failures.append(str(exc))
    journal_path = workspace / run_mode / "journal.json"
    digest = result.get("journal_sha256")
    journal_load = _load_journal_if_valid(
        journal_path,
        autopsy,
        frontier,
        run_mode=run_mode,
        clean_boot_confirmed=bool(clean_boot),
    )
    if journal_load.status == "invalid":
        failures.append(
            "journal fails full contract validation: "
            f"{journal_load.reason or 'unknown validation failure'}"
        )
    if isinstance(digest, str):
        if journal_load.status != "valid":
            if journal_load.status == "missing":
                failures.append("result binds a journal that is missing")
        else:
            if journal_load.digest != digest:
                failures.append("journal digest does not match the persisted artifact")
            if journal_load.journal is not None:
                if journal_load.journal.get("complete") != result.get(
                    "journal_complete"
                ):
                    failures.append(
                        "result's claimed journal completion does not match "
                        "the actual journal"
                    )
                if journal_load.journal.get("terminal") != result.get(
                    "journal_terminal"
                ):
                    failures.append(
                        "result's claimed journal terminal does not match "
                        "the actual journal"
                    )
    elif journal_load.status == "valid":
        failures.append("a valid journal exists but the result does not bind it")
    return {"verified": not failures, "failures": failures}


# ---------------------------------------------------------------------------
# Sanitized, deterministic reporting.
# ---------------------------------------------------------------------------

_LIMITATIONS: tuple[str, ...] = (
    "This is bounded evidence for one recorded machine state, one exact "
    "checkpoint, runtime, and the t256 tier of one workload; it is not a "
    "universal memory-capacity boundary and not a universal 24GB boundary.",
    "There is no unified-memory free-space or GPU capacity precision here: "
    "process RSS is host process memory, not GPU memory, and is never "
    "substituted for a GPU free-memory measurement.",
    "No GPU utilization, free GPU memory, memory bandwidth, power, energy, "
    "or kernel time is measured or inferred.",
    "Stage deltas are observed checkpoint-to-checkpoint changes only; no "
    "causal claim is made from one checkpoint's change to another.",
    "Operator clean boot is never inferred; publication mode requires an "
    "explicit --confirm-clean-boot assertion.",
    "Observer overhead consists only of the stage-boundary probes recorded "
    "here (ps, getrusage, sysctl, or procfs reads); it is not separately "
    "measured or subtracted.",
    "Periodic sampling is disabled; only discrete stage-boundary checkpoints "
    "are recorded.",
)


def _checkpoint_row(entry: dict[str, Any]) -> dict[str, Any]:
    mlx_scope = entry.get("mlx_allocator", {})
    host_scope = entry.get("host_process", {})
    swap_scope = entry.get("host_system", {})

    def value_of(scope: dict[str, Any], key: str) -> Any:
        item = scope.get(key)
        if isinstance(item, dict):
            return item.get("value")
        return None

    def provenance_of(scope: dict[str, Any], key: str) -> Any:
        item = scope.get(key)
        if isinstance(item, dict):
            return item.get("provenance")
        return None

    def mlx_field(key: str, field: str) -> Any:
        item = mlx_scope.get(key)
        if isinstance(item, dict):
            return item.get(field)
        return None

    return {
        "stage": entry.get("stage"),
        "sequence": entry.get("sequence"),
        "wall_clock_utc": entry.get("wall_clock_utc"),
        "wall_clock_provenance": entry.get("wall_clock_provenance"),
        "monotonic_offset_seconds": entry.get("monotonic_offset_seconds"),
        "monotonic_offset_unit": entry.get("monotonic_offset_unit"),
        "mlx_active_bytes": mlx_field("active_bytes", "value"),
        "mlx_active_api": mlx_field("active_bytes", "api"),
        "mlx_active_error_category": mlx_field("active_bytes", "error_category"),
        "mlx_cache_bytes": mlx_field("cache_bytes", "value"),
        "mlx_cache_api": mlx_field("cache_bytes", "api"),
        "mlx_cache_error_category": mlx_field("cache_bytes", "error_category"),
        "mlx_peak_bytes": mlx_field("peak_bytes", "value"),
        "mlx_peak_api": mlx_field("peak_bytes", "api"),
        "mlx_peak_error_category": mlx_field("peak_bytes", "error_category"),
        "host_rss_bytes": value_of(host_scope, "rss_bytes"),
        "host_rss_provenance": provenance_of(host_scope, "rss_bytes"),
        "host_max_rss_bytes": value_of(host_scope, "max_rss_bytes"),
        "host_max_rss_provenance": provenance_of(host_scope, "max_rss_bytes"),
        "swap_total_bytes": value_of(swap_scope, "swap_total_bytes"),
        "swap_used_bytes": value_of(swap_scope, "swap_used_bytes"),
        "swap_provenance": provenance_of(swap_scope, "swap_total_bytes"),
        "signal_received": entry.get("signal_received"),
    }


_DELTA_FIELDS = (
    "mlx_active_bytes",
    "mlx_cache_bytes",
    "mlx_peak_bytes",
    "host_rss_bytes",
    "host_max_rss_bytes",
    "swap_used_bytes",
)


def _observed_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    for before, after in zip(rows[:-1], rows[1:], strict=True):
        row: dict[str, Any] = {
            "from_stage": before.get("stage"),
            "to_stage": after.get("stage"),
        }
        for field in _DELTA_FIELDS:
            before_value = before.get(field)
            after_value = after.get(field)
            row[f"{field}_delta"] = (
                after_value - before_value
                if isinstance(before_value, int)
                and not isinstance(before_value, bool)
                and isinstance(after_value, int)
                and not isinstance(after_value, bool)
                else None
            )
        deltas.append(row)
    return deltas


def build_autopsy_report(
    autopsy: AutopsyManifest,
    frontier: FrontierManifest,
    *,
    workspace: Path,
    run_mode: str,
) -> dict[str, Any]:
    verification = verify_autopsy_evidence(
        autopsy, frontier, workspace=workspace, run_mode=run_mode
    )
    if not verification["verified"]:
        raise LabError(
            "autopsy report refused invalid evidence: "
            + "; ".join(verification["failures"])
        )
    state = _read_json(workspace / run_mode / "state.json")
    result = state["result"]
    journal_path = workspace / run_mode / "journal.json"
    checkpoints_raw: list[dict[str, Any]] = []
    if journal_path.is_file():
        journal_load = _load_journal_if_valid(
            journal_path,
            autopsy,
            frontier,
            run_mode=run_mode,
            clean_boot_confirmed=bool(state.get("clean_boot_confirmed")),
        )
        # ``verify_autopsy_evidence`` above already refused invalid evidence,
        # so a non-valid load here would be an internal inconsistency; the
        # report still must never fabricate checkpoints from an unvalidated
        # journal, so an unexpected non-valid status yields an empty list
        # rather than raw, unchecked data.
        if journal_load.status == "valid" and journal_load.journal is not None:
            raw = journal_load.journal.get("checkpoints")
            if isinstance(raw, list):
                checkpoints_raw = raw
    rows = [_checkpoint_row(entry) for entry in checkpoints_raw]

    def first_present(field: str) -> Any:
        return next((row.get(field) for row in rows if row.get(field)), None)

    provenance = {
        "mlx_active_bytes": {
            "scope": "mlx_allocator",
            "unit": "bytes",
            "api": first_present("mlx_active_api"),
            "error_category": first_present("mlx_active_error_category"),
        },
        "mlx_cache_bytes": {
            "scope": "mlx_allocator",
            "unit": "bytes",
            "api": first_present("mlx_cache_api"),
            "error_category": first_present("mlx_cache_error_category"),
        },
        "mlx_peak_bytes": {
            "scope": "mlx_allocator",
            "unit": "bytes",
            "api": first_present("mlx_peak_api"),
            "error_category": first_present("mlx_peak_error_category"),
        },
        "host_rss_bytes": {
            "scope": "host_process",
            "unit": "bytes",
            "provenance": first_present("host_rss_provenance"),
        },
        "host_max_rss_bytes": {
            "scope": "host_process",
            "unit": "bytes",
            "provenance": first_present("host_max_rss_provenance"),
        },
        "swap_bytes": {
            "scope": "host_system_swap",
            "unit": "bytes",
            "provenance": first_present("swap_provenance"),
        },
        "wall_clock": {"provenance": first_present("wall_clock_provenance")},
        "monotonic_offset": {"unit": "seconds"},
    }
    report = {
        "schema_version": AUTOPSY_REPORT_SCHEMA_VERSION,
        "autopsy_id": autopsy.autopsy_id,
        "generated_at": state.get("ended_at"),
        "run_mode": run_mode,
        "clean_boot_confirmed": state.get("clean_boot_confirmed"),
        "synthetic": bool(state.get("synthetic", False)),
        "bindings": {
            "autopsy_manifest_hash": autopsy_manifest_hash(autopsy),
            "frontier_manifest_hash": frontier_manifest_hash(frontier),
        },
        "model": {
            "repository_id": frontier.model_repository_id,
            "revision": frontier.model_revision,
        },
        "tier": {
            "name": autopsy.tier_name,
            "requested_tokens": frontier.tier(autopsy.tier_name).requested_tokens,
        },
        "terminal_outcome": result.get("status"),
        "reason": result.get("reason"),
        "actual_tokens": result.get("actual_tokens"),
        "journal_complete": result.get("journal_complete"),
        "journal_terminal": result.get("journal_terminal"),
        "pre_run_machine_state": state.get("pre_run_machine_state"),
        "provenance": provenance,
        "sampling": {"periodic_sampling_enabled": False},
        "checkpoints": rows,
        "observed_deltas": _observed_deltas(rows),
        "limitations": list(_LIMITATIONS),
    }
    assert_shareable(report)
    return report


def build_autopsy_csv(report: dict[str, Any]) -> str:
    header = (
        "stage",
        "sequence",
        "wall_clock_utc",
        "wall_clock_provenance",
        "monotonic_offset_seconds",
        "monotonic_offset_unit",
        "mlx_active_bytes",
        "mlx_active_api",
        "mlx_cache_bytes",
        "mlx_cache_api",
        "mlx_peak_bytes",
        "mlx_peak_api",
        "host_rss_bytes",
        "host_rss_provenance",
        "host_max_rss_bytes",
        "host_max_rss_provenance",
        "swap_total_bytes",
        "swap_used_bytes",
        "swap_provenance",
    )
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    for row in report["checkpoints"]:
        values = [row.get(key) for key in header]
        writer.writerow(["n/a" if value is None else value for value in values])
    return stream.getvalue()


def _svg_series(
    rows: list[dict[str, Any]], key: str, *, color: str, height: float, max_value: float
) -> str:
    points = [
        (index, row[key]) for index, row in enumerate(rows) if row.get(key) is not None
    ]
    if not points or max_value <= 0:
        return ""
    step = 60.0

    def coordinate(index: int, value: float) -> tuple[float, float]:
        return index * step, height - (value / max_value) * height

    coords = " ".join(
        f"{x:.1f},{y:.1f}" for x, y in (coordinate(i, v) for i, v in points)
    )
    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>'
        for x, y in (coordinate(i, v) for i, v in points)
    )
    return (
        f'<polyline points="{coords}" fill="none" stroke="{color}" '
        f'stroke-width="2"/>{circles}'
    )


def render_autopsy_chart_svg(report: dict[str, Any]) -> str:
    rows = report["checkpoints"]
    height = 120.0
    width = max(60.0 * len(rows), 60.0)
    series = (
        ("mlx_peak_bytes", "#c0392b", "MLX allocator (peak)"),
        ("host_rss_bytes", "#2471a3", "Host process (RSS)"),
        ("swap_used_bytes", "#7d3c98", "Host system (swap used)"),
    )
    max_value = 0.0
    for key, _color, _label in series:
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)):
                max_value = max(max_value, float(value))
    body = "".join(
        _svg_series(rows, key, color=color, height=height, max_value=max_value)
        for key, color, _label in series
    )
    legend = "".join(
        f'<span style="color:{color}">&#9632;</span> {html.escape(label)}'
        for _key, color, label in series
    )
    return (
        f'<svg viewBox="0 0 {max(width, 60.0):.1f} {height + 20:.1f}" '
        f'role="img" aria-label="MLX allocator, host process, and host swap '
        f'checkpoint series">{body}</svg>'
        f'<div class="chart-legend">{legend}</div>'
    )


def render_autopsy_report_html(report: dict[str, Any]) -> str:
    assert_shareable(report)

    def esc(value: Any) -> str:
        return html.escape(str(value), quote=True)

    def cell(value: Any) -> str:
        return esc(value) if value is not None else "n/a"

    rows = "".join(
        "<tr>"
        f"<td>{esc(row['stage'])}</td>"
        f"<td>{cell(row['mlx_active_bytes'])}</td>"
        f"<td>{cell(row['mlx_cache_bytes'])}</td>"
        f"<td>{cell(row['mlx_peak_bytes'])}</td>"
        f"<td>{cell(row['host_rss_bytes'])}</td>"
        f"<td>{cell(row['host_max_rss_bytes'])}</td>"
        f"<td>{cell(row['swap_total_bytes'])}</td>"
        f"<td>{cell(row['swap_used_bytes'])}</td>"
        "</tr>"
        for row in report["checkpoints"]
    )
    delta_rows = "".join(
        "<tr>"
        f"<td>{esc(row['from_stage'])} &rarr; {esc(row['to_stage'])}</td>"
        f"<td>{cell(row['mlx_active_bytes_delta'])}</td>"
        f"<td>{cell(row['mlx_cache_bytes_delta'])}</td>"
        f"<td>{cell(row['mlx_peak_bytes_delta'])}</td>"
        f"<td>{cell(row['host_rss_bytes_delta'])}</td>"
        f"<td>{cell(row['host_max_rss_bytes_delta'])}</td>"
        f"<td>{cell(row['swap_used_bytes_delta'])}</td>"
        "</tr>"
        for row in report["observed_deltas"]
    )
    limitations = "".join(f"<li>{esc(item)}</li>" for item in report["limitations"])
    chart = render_autopsy_chart_svg(report)
    provenance = report.get("provenance", {})
    provenance_rows = "".join(
        f"<tr><td>{esc(name)}</td><td>{cell(details.get('scope'))}</td>"
        f"<td>{cell(details.get('unit'))}</td><td>{cell(details.get('api'))}</td>"
        f"<td>{cell(details.get('provenance'))}</td>"
        f"<td>{cell(details.get('error_category'))}</td></tr>"
        for name, details in provenance.items()
        if isinstance(details, dict)
    )
    bindings = report.get("bindings", {})
    synthetic_banner = (
        "<p><strong>SYNTHETIC</strong> fixture evidence, not a real run.</p>"
        if report.get("synthetic")
        else ""
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>Qwen3.8-27B OOM autopsy</title>
<style>
body{{margin:0;background:#f7f3ea;color:#17202a;font:15px/1.5 ui-monospace,monospace}}
main{{max-width:980px;margin:auto;padding:36px 20px}}section{{background:#fffdf8;border:1px
solid #d9d1c3;padding:18px;margin:18px 0;overflow:auto}}table{{width:100%;
border-collapse:collapse}}th,td{{padding:9px;text-align:right;border-bottom:1px solid
#ddd5c8}}th:first-child,td:first-child{{text-align:left}}.chart-legend{{display:flex;
gap:16px;margin-top:8px;font-size:13px}}
</style></head><body><main><p>LLMTraceFX / {esc(report['run_mode'])} evidence</p>
<h1>MLX OOM autopsy</h1>{synthetic_banner}<p><code>{esc(report['model']['repository_id'])}@
{esc(report['model']['revision'])}</code> tier <code>{esc(report['tier']['name'])}</code>
({esc(report['tier']['requested_tokens'])} requested tokens)</p>
<section><h2>Terminal outcome</h2><p>{esc(report['terminal_outcome'])}
&mdash; {cell(report['reason'])}</p><p>actual tokens: {cell(report['actual_tokens'])}</p>
</section><section><h2>Checkpoint series</h2>{chart}<table><thead><tr><th>Stage</th>
<th>MLX active</th><th>MLX cache</th><th>MLX peak</th><th>Host RSS</th>
<th>Host max RSS</th><th>Swap total</th><th>Swap used</th></tr></thead>
<tbody>{rows}</tbody></table></section>
<section><h2>Observed checkpoint deltas</h2><p>These differences locate observed
growth between stage boundaries; they do not identify a cause.</p><table><thead><tr>
<th>Boundary</th><th>MLX active</th><th>MLX cache</th><th>MLX peak</th>
<th>Host RSS</th><th>Host max RSS</th><th>Swap used</th></tr></thead>
<tbody>{delta_rows}</tbody></table></section>
<section><h2>Provenance</h2><table><thead><tr><th>Measurement</th><th>Scope</th>
<th>Unit</th><th>API</th><th>Provenance</th><th>Error category</th></tr></thead>
<tbody>{provenance_rows}</tbody></table></section>
<section><h2>Bindings</h2><p>autopsy_manifest_hash: {cell(bindings.get('autopsy_manifest_hash'))}</p>
<p>frontier_manifest_hash: {cell(bindings.get('frontier_manifest_hash'))}</p></section>
<section><h2>Limits</h2><ul>{limitations}</ul></section></main></body></html>"""


def write_autopsy_report(
    autopsy: AutopsyManifest,
    frontier: FrontierManifest,
    *,
    workspace: Path,
    run_mode: str,
    shareable_dir: Path | None,
) -> dict[str, Any]:
    report = build_autopsy_report(
        autopsy, frontier, workspace=workspace, run_mode=run_mode
    )
    reports = workspace / run_mode / "reports"
    atomic_write_text(
        reports / "oom-autopsy-summary.json",
        json.dumps(report, indent=2, sort_keys=False) + "\n",
    )
    atomic_write_text(
        reports / "oom-autopsy-report.html", render_autopsy_report_html(report)
    )
    atomic_write_text(
        reports / "oom-autopsy-checkpoints.csv", build_autopsy_csv(report)
    )
    if shareable_dir is not None:
        destination = shareable_dir / run_mode
        atomic_write_text(
            destination / "oom-autopsy-summary.json",
            json.dumps(report, indent=2, sort_keys=False) + "\n",
        )
        atomic_write_text(
            destination / "oom-autopsy-report.html",
            render_autopsy_report_html(report),
        )
        atomic_write_text(
            destination / "oom-autopsy-checkpoints.csv", build_autopsy_csv(report)
        )
    return report


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmtracefx-m5-lab autopsy",
        description=(
            "Focused MLX OOM autopsy for the exact pinned Qwen3.8-27B MLX "
            "checkpoint at the t256 tier. The default action is a no-load "
            "plan that only probes MLX allocator counter APIs."
        ),
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="plan",
        choices=("plan", "run", "report", "verify", "_child"),
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--frontier-manifest", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=RUN_MODES, default="exploratory")
    parser.add_argument("--confirm-clean-boot", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--shareable-dir", type=Path)
    return parser


def _paths(
    args: argparse.Namespace, autopsy: AutopsyManifest, base: LabManifest
) -> tuple[Path, Path]:
    workspace = args.workspace or Path(autopsy.artifacts.workspace)
    model_path = args.model_path or Path(base.artifacts.model_cache)
    return workspace, model_path


def _plan(args: argparse.Namespace) -> int:
    autopsy, source = load_autopsy_manifest(args.manifest)
    frontier, base = load_bound_frontier(
        autopsy, frontier_manifest_path=args.frontier_manifest
    )
    workspace, model_path = _paths(args, autopsy, base)
    decision = assess_safety(base, workspace, include_download=False)
    mlx_module = _import_mlx_core()
    payload = {
        "action": "plan",
        "weights_loaded": False,
        "downloads_performed": False,
        "manifest": source,
        "autopsy_id": autopsy.autopsy_id,
        "autopsy_manifest_hash": autopsy_manifest_hash(autopsy),
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
        "tier": {
            "name": autopsy.tier_name,
            "requested_tokens": frontier.tier(autopsy.tier_name).requested_tokens,
        },
        "mlx_counter_apis": probe_mlx_counters(mlx_module),
        "mlx_reset_peak_memory_api": probe_mlx_reset_peak_memory_api(mlx_module),
        "sampling": {"periodic_sampling_enabled": False},
        "machine_state": machine_state(decision),
        "safety": {"safe": decision.safe, "blockers": list(decision.blockers)},
    }
    print(json.dumps(payload, indent=2, sort_keys=False))
    if args.mode == "publication" and not args.confirm_clean_boot:
        return 2
    return 0 if decision.safe else 2


def _run(args: argparse.Namespace) -> int:
    autopsy, _ = load_autopsy_manifest(args.manifest)
    frontier, base = load_bound_frontier(
        autopsy, frontier_manifest_path=args.frontier_manifest
    )
    workspace, model_path = _paths(args, autopsy, base)
    state = run_autopsy(
        autopsy,
        frontier,
        base,
        autopsy_manifest_path=args.manifest,
        frontier_manifest_path=args.frontier_manifest,
        workspace=workspace,
        model_path=model_path,
        run_mode=args.mode,
        clean_boot_confirmed=args.confirm_clean_boot,
        resume=not args.no_resume,
    )
    report = write_autopsy_report(
        autopsy,
        frontier,
        workspace=workspace,
        run_mode=args.mode,
        shareable_dir=args.shareable_dir,
    )
    print(json.dumps({"state": state, "report": report}, indent=2))
    return 0 if state["status"] == "completed" else 2


def _report(args: argparse.Namespace) -> int:
    autopsy, _ = load_autopsy_manifest(args.manifest)
    frontier, base = load_bound_frontier(
        autopsy, frontier_manifest_path=args.frontier_manifest
    )
    workspace, _ = _paths(args, autopsy, base)
    report = write_autopsy_report(
        autopsy,
        frontier,
        workspace=workspace,
        run_mode=args.mode,
        shareable_dir=args.shareable_dir,
    )
    print(json.dumps(report, indent=2))
    return 0


def _verify(args: argparse.Namespace) -> int:
    autopsy, _ = load_autopsy_manifest(args.manifest)
    frontier, base = load_bound_frontier(
        autopsy, frontier_manifest_path=args.frontier_manifest
    )
    workspace, model_path = _paths(args, autopsy, base)
    verify_model(base, model_path)
    result = verify_autopsy_evidence(
        autopsy, frontier, workspace=workspace, run_mode=args.mode
    )
    print(json.dumps(result, indent=2))
    return 0 if result["verified"] else 2


def _child(args: argparse.Namespace) -> int:
    if args.output_dir is None or args.model_path is None:
        raise LabError("_child requires --output-dir and --model-path")
    autopsy, _ = load_autopsy_manifest(args.manifest)
    frontier, base = load_bound_frontier(
        autopsy, frontier_manifest_path=args.frontier_manifest
    )
    result = execute_autopsy_child(
        autopsy,
        frontier,
        base,
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
        AutopsyManifestError,
        FrontierManifestError,
        LabError,
        OSError,
        UnicodeError,
        ValueError,
    ) as exc:
        print(f"M5 OOM autopsy failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
