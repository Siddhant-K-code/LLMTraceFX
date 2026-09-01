"""Self-conversion runner for the planned Qwen3-8B M5 Pro control.

No conversion has completed on any machine yet; this module is the
offline framework for one. A live, conservative safety gate (chip,
physical memory, free memory percent, swap, free disk, installed
converter version) must pass, with no network call, subprocess launch,
or filesystem write to any cache/output path, before anything else
happens. Only once that gate and a pre-existing-output check both pass
does this download (or reuse a verified local cache of) the official
Qwen3-8B checkpoint and convert it with the repository's own pinned
``mlx-lm``, in a fresh, no-shell subprocess/process group with a
parent-enforced bounded timeout and TERM->KILL cleanup escalation. On
success, the converted output directory is recursively inventoried
(regular files only; symlinks and unsafe paths are rejected) and a
conversion receipt is written. On any failure -- a refused safety gate,
non-zero exit, timeout, or cleanup failure -- a bounded, privacy-safe
failure record is written instead; this module never retries
automatically and never deletes or overwrites an existing cache or
output directory on the caller's behalf.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from ..._artifact_io import (
    MAX_METADATA_ARTIFACT_BYTES,
    ArtifactReadError,
    read_bounded_regular_text,
)
from ...collectors._shared import atomic_write_text, config_hash
from ..core import HostSnapshot, collect_host_snapshot
from ..frontier import _clean_process_group
from .conversion_manifest import ConversionManifest

CONVERSION_RECEIPT_SCHEMA_VERSION = "1"
CONVERSION_JOURNAL_SCHEMA_VERSION = "1"


class ConversionError(RuntimeError):
    """Raised when a conversion cannot proceed or cannot be trusted."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def source_files_present(manifest: ConversionManifest, source_path: Path) -> bool:
    """Whether every pinned source file already exists at the expected size."""
    return all(
        (source_path / pin.path).is_file()
        and not (source_path / pin.path).is_symlink()
        and (source_path / pin.path).stat().st_size == pin.size_bytes
        for pin in manifest.source.files
    )


def verify_source(manifest: ConversionManifest, source_path: Path) -> dict[str, Any]:
    """Verify a local source directory against every pinned file.

    Files with no pinned SHA-256 (``SourceFilePin.sha256 is None``) are
    size-checked only; the returned payload marks them
    ``"exactly_pinned": false`` so callers never mistake a size-only
    match for a byte-identical one. Symlinks and unpinned root entries
    are rejected the same way ``lab.core.verify_model`` rejects them.
    """
    failures: list[str] = []
    files: list[dict[str, Any]] = []
    expected_paths = {pin.path for pin in manifest.source.files}
    if source_path.is_dir():
        unexpected = sorted(
            path.name
            for path in source_path.iterdir()
            if path.name != ".cache" and path.name not in expected_paths
        )
        failures.extend(f"unpinned source-root entry: {name}" for name in unexpected)
    for pin in manifest.source.files:
        path = source_path / pin.path
        if not path.is_file() or path.is_symlink():
            failures.append(f"missing regular source file: {pin.path}")
            continue
        size = path.stat().st_size
        if size != pin.size_bytes:
            failures.append(f"{pin.path} size {size} does not match {pin.size_bytes}")
            continue
        digest = _sha256_file(path)
        if digest != pin.sha256:
            failures.append(f"{pin.path} sha256 {digest} does not match {pin.sha256}")
            continue
        files.append(
            {
                "path": pin.path,
                "size_bytes": size,
                "sha256": digest,
                "exactly_pinned": True,
            }
        )
    result = {
        "schema_version": "1",
        "official_id": manifest.source.official_id,
        "official_revision": manifest.source.official_revision,
        "verified": not failures,
        "fully_pinned": manifest.source.fully_pinned,
        "failures": failures,
        "files": files,
    }
    if failures:
        raise ConversionError("source verification failed: " + "; ".join(failures))
    return result


def source_cache_has_existing_content(source_path: Path) -> bool:
    return source_path.is_dir() and any(source_path.iterdir())


def acquire_source(
    manifest: ConversionManifest, *, source_path: Path, workspace: Path
) -> dict[str, Any]:
    """Download the pinned official source, reusing a verified cache.

    ``source_path`` is trusted only after every pinned file verifies
    exactly (``verify_source``). If anything already exists at
    ``source_path`` -- any file or subdirectory at all, not only a
    complete, correctly sized set -- and that content fails
    verification, this refuses (``ConversionError`` propagates) instead
    of falling through to a network write that would land on top of a
    possibly corrupt or stale cache. A network download is only
    attempted when ``source_path`` does not exist yet or is empty.
    ``source_path`` itself is never deleted or overwritten by this
    function.
    """
    if source_cache_has_existing_content(source_path):
        return verify_source(manifest, source_path)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ConversionError(
            "huggingface-hub is required; install the `mlx` extra"
        ) from exc
    downloaded = Path(
        snapshot_download(
            repo_id=manifest.source.repository_id,
            revision=manifest.source.official_revision,
            local_dir=source_path,
            allow_patterns=[pin.path for pin in manifest.source.files],
        )
    )
    result = verify_source(manifest, downloaded)
    workspace.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        workspace / "source-verification.json",
        json.dumps(result, indent=2, sort_keys=False) + "\n",
    )
    return result


def _installed_version(distribution: str) -> str | None:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class ConversionSafetyDecision:
    safe: bool
    blockers: tuple[str, ...]
    snapshot: HostSnapshot
    required_free_disk_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "safe": self.safe,
            "blockers": list(self.blockers),
            "snapshot": self.snapshot.to_dict(),
            "required_free_disk_bytes": self.required_free_disk_bytes,
            "observed_free_disk_bytes": self.snapshot.disk_free_bytes,
        }


def assess_conversion_safety(
    manifest: ConversionManifest,
    path: Path,
    *,
    include_source_download: bool,
) -> ConversionSafetyDecision:
    """Preflight check reusing ``lab.core.collect_host_snapshot`` verbatim.

    ``required_free_disk_bytes`` is always
    ``safety.minimum_residual_free_disk_bytes + expected_output_bytes``,
    plus ``source.expected_source_bytes`` on top of that when
    ``include_source_download`` is true (a source that still needs
    downloading). This is deliberately conservative: it is the disk this
    conversion could still need to write, not an estimate of what is
    already cached.
    """
    snapshot = collect_host_snapshot(path)
    blockers: list[str] = []
    required_free = (
        manifest.safety.minimum_residual_free_disk_bytes
        + manifest.expected_output_bytes
    )
    if include_source_download:
        required_free += manifest.source.expected_source_bytes
    if snapshot.disk_free_bytes < required_free:
        blockers.append(
            f"disk free {snapshot.disk_free_bytes} bytes is below required "
            f"{required_free} bytes"
        )
    if snapshot.os_name != "Darwin" or snapshot.architecture != "arm64":
        blockers.append("self-conversion requires Apple Silicon macOS")
    if snapshot.chip != manifest.safety.required_chip:
        blockers.append(
            f"chip is {snapshot.chip or 'unavailable'}, expected "
            f"{manifest.safety.required_chip}"
        )
    if snapshot.total_memory_bytes is None:
        blockers.append("total physical memory could not be measured")
    elif snapshot.total_memory_bytes != manifest.safety.required_total_memory_bytes:
        blockers.append(
            f"physical memory is {snapshot.total_memory_bytes} bytes, expected "
            f"{manifest.safety.required_total_memory_bytes} bytes"
        )
    if snapshot.memory_free_percent is None:
        blockers.append("current memory headroom could not be measured")
    elif snapshot.memory_free_percent < manifest.safety.minimum_memory_free_percent:
        blockers.append(
            f"memory free {snapshot.memory_free_percent:g}% is below "
            f"{manifest.safety.minimum_memory_free_percent:g}%"
        )
    if snapshot.swap_used_bytes is None:
        blockers.append("current swap usage could not be measured")
    elif snapshot.swap_used_bytes > manifest.safety.maximum_swap_used_bytes:
        blockers.append(
            f"swap used {snapshot.swap_used_bytes} bytes exceeds "
            f"{manifest.safety.maximum_swap_used_bytes} bytes"
        )
    installed_mlx_lm = snapshot.package_versions.get("mlx-lm")
    if installed_mlx_lm != manifest.converter.version:
        blockers.append(
            f"installed mlx-lm is {installed_mlx_lm or 'not installed'}, expected "
            f"pinned converter version {manifest.converter.version}"
        )
    return ConversionSafetyDecision(
        safe=not blockers,
        blockers=tuple(blockers),
        snapshot=snapshot,
        required_free_disk_bytes=required_free,
    )


def _bounded_tail(text: str, max_bytes: int) -> str:
    encoded = text.encode("utf-8", errors="replace")
    if len(encoded) <= max_bytes:
        return text
    return encoded[-max_bytes:].decode("utf-8", errors="replace")


def _argv(manifest: ConversionManifest, *, hf_path: Path, mlx_path: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "mlx_lm",
        *manifest.parameters.argv(hf_path=str(hf_path), mlx_path=str(mlx_path)),
    ]


def _unsafe_output_reason(root: Path, entry: Path) -> str | None:
    try:
        resolved = entry.resolve()
        resolved_root = root.resolve()
    except OSError:
        return "path could not be resolved"
    if resolved_root not in resolved.parents and resolved != resolved_root:
        return "escapes the output directory"
    return None


def _inventory_output(output_path: Path) -> tuple[dict[str, Any], ...]:
    """Recursively inventory regular output files; reject symlinks/unsafe paths."""
    entries: list[dict[str, Any]] = []
    for entry in sorted(output_path.rglob("*")):
        if entry.is_dir():
            continue
        if entry.is_symlink():
            raise ConversionError(f"conversion output contains a symlink: {entry}")
        unsafe = _unsafe_output_reason(output_path, entry)
        if unsafe is not None:
            raise ConversionError(
                f"conversion output path is unsafe ({unsafe}): {entry}"
            )
        if not entry.is_file():
            raise ConversionError(
                f"conversion output contains a non-regular file: {entry}"
            )
        relative = entry.relative_to(output_path)
        entries.append(
            {
                "path": str(relative),
                "size_bytes": entry.stat().st_size,
                "sha256": _sha256_file(entry),
            }
        )
    if not entries:
        raise ConversionError("conversion output directory is empty")
    return tuple(entries)


def _required_string(data: dict[str, Any], key: str, *, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value:
        raise ConversionError(
            f"conversion receipt {context}.{key} must be a non-empty string"
        )
    return value


def _required_object(data: dict[str, Any], key: str, *, context: str) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict) or not value:
        raise ConversionError(
            f"conversion receipt {context}.{key} must be a non-empty object"
        )
    return value


def _required_positive_int(data: dict[str, Any], key: str, *, context: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConversionError(
            f"conversion receipt {context}.{key} must be a positive integer"
        )
    return value


def _required_sha256(value: Any, *, context: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ConversionError(
            f"conversion receipt {context} must be a lowercase SHA-256 hex digest"
        )
    return value


def _required_relative_path(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConversionError(
            f"conversion receipt {context} must be a non-empty string"
        )
    candidate = Path(value)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or any(part in ("", ".") for part in candidate.parts)
    ):
        raise ConversionError(
            f"conversion receipt {context} must be a safe relative path: {value!r}"
        )
    return value


def _validate_output_file_entry(item: Any, *, index: int) -> dict[str, Any]:
    context = f"output_files[{index}]"
    if not isinstance(item, dict):
        raise ConversionError(f"conversion receipt {context} must be an object")
    return {
        "path": _required_relative_path(item.get("path"), context=f"{context}.path"),
        "size_bytes": _required_positive_int(item, "size_bytes", context=context),
        "sha256": _required_sha256(item.get("sha256"), context=f"{context}.sha256"),
    }


#: Sub-keys that must be present (identity, not full typing) on a
#: receipt's ``source``/``converter``/``parameters`` objects so a receipt
#: can never bind against a packaged spec it does not actually describe.
_RECEIPT_SOURCE_IDENTITY_KEYS = ("official_id", "official_revision", "license")
_RECEIPT_CONVERTER_IDENTITY_KEYS = ("package", "version", "git_revision")
_RECEIPT_PARAMETER_KEYS = (
    "quantize",
    "q_group_size",
    "q_bits",
    "q_mode",
    "dtype",
    "quant_predicate",
    "dequantize",
    "trust_remote_code",
    "upload_repo",
)


@dataclass(frozen=True)
class ConversionReceipt:
    """The durable, verifiable record of one completed self-conversion.

    Strictly typed and bounded on parse: every output file entry is a
    unique, safe, repository-relative path with a positive size and a
    lowercase SHA-256 digest, the recorded total equals their sum, and
    the ``source``/``converter``/``parameters`` identity sub-keys this
    binds against later must actually be present. None of this alone
    proves the receipt matches a *specific* packaged conversion spec --
    that cross-check is ``bind_control_manifest``'s job (see
    ``control_manifest.py``), using ``conversion_manifest_hash`` and the
    individual identity fields recorded here.
    """

    schema_version: str
    conversion_id: str
    conversion_manifest_hash: str
    status: str
    started_at: str
    ended_at: str
    source: dict[str, Any]
    converter: dict[str, Any]
    parameters: dict[str, Any]
    argv: tuple[str, ...]
    output_files: tuple[dict[str, Any], ...]
    output_total_bytes: int
    host: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "conversion_id": self.conversion_id,
            "conversion_manifest_hash": self.conversion_manifest_hash,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "source": self.source,
            "converter": self.converter,
            "parameters": self.parameters,
            "argv": list(self.argv),
            "output_files": [dict(item) for item in self.output_files],
            "output_total_bytes": self.output_total_bytes,
            "host": self.host,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ConversionReceipt:
        if not isinstance(data, dict):
            raise ConversionError("conversion receipt must be an object")
        if data.get("schema_version") != CONVERSION_RECEIPT_SCHEMA_VERSION:
            raise ConversionError("unsupported conversion receipt schema_version")
        if data.get("status") != "completed":
            raise ConversionError(
                "conversion receipt does not describe a completed conversion"
            )
        conversion_id = _required_string(data, "conversion_id", context="receipt")
        started_at = _required_string(data, "started_at", context="receipt")
        ended_at = _required_string(data, "ended_at", context="receipt")

        source = _required_object(data, "source", context="receipt")
        for key in _RECEIPT_SOURCE_IDENTITY_KEYS:
            _required_string(source, key, context="receipt.source")

        converter = _required_object(data, "converter", context="receipt")
        for key in _RECEIPT_CONVERTER_IDENTITY_KEYS:
            _required_string(converter, key, context="receipt.converter")

        parameters = _required_object(data, "parameters", context="receipt")
        missing_parameters = [
            key for key in _RECEIPT_PARAMETER_KEYS if key not in parameters
        ]
        if missing_parameters:
            raise ConversionError(
                f"conversion receipt.parameters is missing {missing_parameters}"
            )

        argv_raw = data.get("argv")
        if not isinstance(argv_raw, list) or not argv_raw:
            raise ConversionError("conversion receipt.argv must be a non-empty list")
        if not all(isinstance(item, str) and item for item in argv_raw):
            raise ConversionError(
                "conversion receipt.argv must be all non-empty strings"
            )

        output_files_raw = data.get("output_files")
        if not isinstance(output_files_raw, list) or not output_files_raw:
            raise ConversionError("conversion receipt output_files must be non-empty")
        output_files = tuple(
            _validate_output_file_entry(item, index=index)
            for index, item in enumerate(output_files_raw)
        )
        paths = [item["path"] for item in output_files]
        if len(set(paths)) != len(paths):
            raise ConversionError(
                "conversion receipt output_files paths must be unique"
            )
        output_total_bytes = _required_positive_int(
            data, "output_total_bytes", context="receipt"
        )
        computed_total = sum(item["size_bytes"] for item in output_files)
        if computed_total != output_total_bytes:
            raise ConversionError(
                "conversion receipt output_total_bytes "
                f"({output_total_bytes}) does not equal the sum of "
                f"output_files sizes ({computed_total})"
            )

        host = _required_object(data, "host", context="receipt")
        manifest_hash = data.get("conversion_manifest_hash")
        if (
            not isinstance(manifest_hash, str)
            or not manifest_hash.startswith("sha256:")
            or len(manifest_hash) != 71
            or any(
                char not in "0123456789abcdef"
                for char in manifest_hash.removeprefix("sha256:")
            )
        ):
            raise ConversionError(
                "conversion receipt.conversion_manifest_hash must be a "
                "sha256:-prefixed lowercase digest"
            )

        return cls(
            schema_version=data["schema_version"],
            conversion_id=conversion_id,
            conversion_manifest_hash=manifest_hash,
            status=data["status"],
            started_at=started_at,
            ended_at=ended_at,
            source=source,
            converter=converter,
            parameters=parameters,
            argv=tuple(argv_raw),
            output_files=output_files,
            output_total_bytes=output_total_bytes,
            host=host,
        )

    @classmethod
    def read_json(cls, path: Path) -> ConversionReceipt:
        try:
            payload = read_bounded_regular_text(path, MAX_METADATA_ARTIFACT_BYTES)
        except ArtifactReadError as exc:
            raise ConversionError(f"invalid conversion receipt file: {exc}") from exc
        try:
            data = json.loads(payload)
        except ValueError as exc:
            raise ConversionError(f"invalid conversion receipt JSON: {exc}") from exc
        return cls.from_dict(data)


def run_conversion(
    manifest: ConversionManifest,
    *,
    source_path: Path,
    output_path: Path,
    workspace: Path,
    utc_now: Any = None,
) -> dict[str, Any]:
    """Run exactly one self-conversion attempt; never retries automatically.

    Order of operations, all before any network activity or subprocess
    launch: (1) the live conservative safety gate (chip, physical memory,
    free memory percent, swap, free disk, installed converter version),
    which never downloads or converts anything and never deletes or
    modifies anything on disk; (2) the pre-existing-output-path check.
    Only once both pass does this acquire (download or verify-in-place)
    the source and launch the isolated conversion subprocess.

    Returns a bounded, privacy-safe journal dict describing the attempt.
    On a ``"completed"`` status, ``workspace/conversion-receipt.json`` is
    also written. On any other status -- including a safety-gate refusal
    -- a bounded ``workspace/conversion-failure.json`` is written instead
    and no receipt is produced or implied.
    """
    from ...schema import utc_now_iso

    now = utc_now or utc_now_iso
    workspace.mkdir(parents=True, exist_ok=True)

    # --- Phase 1: live safety gate. No network call, no subprocess, no
    # write to any cache or output path may happen before this passes. ---
    include_source_download = not source_cache_has_existing_content(source_path)
    preflight = assess_conversion_safety(
        manifest, source_path.parent, include_source_download=include_source_download
    )
    if not preflight.safe:
        journal: dict[str, Any] = {
            "schema_version": CONVERSION_JOURNAL_SCHEMA_VERSION,
            "conversion_id": manifest.conversion_id,
            "conversion_manifest_hash": conversion_manifest_hash(manifest),
            "phase": "pre_conversion_safety",
            "status": "safety_blocked",
            "started_at": now(),
            "ended_at": now(),
            "downloads_performed": False,
            "conversion_process_started": False,
            "retried": False,
            "blockers": list(preflight.blockers),
            "required_free_disk_bytes": preflight.required_free_disk_bytes,
            "host": preflight.snapshot.to_dict(),
            "stages": ["pre_conversion_safety"],
        }
        atomic_write_text(
            workspace / "conversion-failure.json",
            json.dumps(journal, indent=2, sort_keys=False) + "\n",
        )
        return journal

    # --- Phase 2: pre-existing-output check, still before any network
    # activity. Never deletes or overwrites an existing output path. ---
    if output_path.exists():
        raise ConversionError(
            f"conversion output path already exists: {output_path}. This "
            "project never deletes an existing output directory on the "
            "caller's behalf; remove or relocate it explicitly first."
        )

    # --- Phase 3: acquire (download or verify-in-place) the source. Only
    # reached once the safety gate and output check both passed. ---
    downloads_performed = include_source_download
    source_result = acquire_source(
        manifest, source_path=source_path, workspace=workspace
    )

    argv = _argv(manifest, hf_path=source_path, mlx_path=output_path)
    started_at = now()
    stdout_path = workspace / "convert-stdout.log"
    stderr_path = workspace / "convert-stderr.log"
    journal = {
        "schema_version": CONVERSION_JOURNAL_SCHEMA_VERSION,
        "conversion_id": manifest.conversion_id,
        "conversion_manifest_hash": conversion_manifest_hash(manifest),
        "phase": "conversion_process",
        "started_at": started_at,
        "argv": argv,
        "converter_version_installed": _installed_version("mlx-lm"),
        "converter_version_expected": manifest.converter.version,
        "downloads_performed": downloads_performed,
        "conversion_process_started": True,
        "retried": False,
        "stages": ["source_verified", "preflight_passed", "launching"],
    }

    with (
        stdout_path.open("wb") as stdout_handle,
        stderr_path.open("wb") as stderr_handle,
    ):
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
            shell=False,
        )
        process_group = process.pid
        timed_out = False
        try:
            process.wait(timeout=manifest.timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        descendants_cleaned = _clean_process_group(
            process, process_group, manifest.cleanup_grace_seconds
        )
        if process.poll() is None:
            try:
                process.wait(timeout=manifest.cleanup_grace_seconds)
            except subprocess.TimeoutExpired:
                descendants_cleaned = False
    ended_at = now()
    exit_code = process.returncode
    stdout_tail = _bounded_tail(
        stdout_path.read_text(encoding="utf-8", errors="replace"),
        manifest.max_journal_bytes // 2,
    )
    stderr_tail = _bounded_tail(
        stderr_path.read_text(encoding="utf-8", errors="replace"),
        manifest.max_journal_bytes // 2,
    )
    journal.update(
        {
            "ended_at": ended_at,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "descendants_cleaned": descendants_cleaned,
            "stdout_tail": stdout_tail,
            "stderr_tail": stderr_tail,
        }
    )

    if timed_out:
        journal["status"] = "timeout"
    elif not descendants_cleaned:
        journal["status"] = "cleanup_failed"
    elif exit_code != 0:
        journal["status"] = "failed"
    else:
        journal["status"] = "completed"
    journal["stages"].append(journal["status"])

    if journal["status"] != "completed":
        atomic_write_text(
            workspace / "conversion-failure.json",
            json.dumps(journal, indent=2, sort_keys=False) + "\n",
        )
        return journal

    try:
        output_files = _inventory_output(output_path)
    except ConversionError as exc:
        journal["status"] = "unsafe_output"
        journal["stages"].append(journal["status"])
        journal["unsafe_output_reason"] = str(exc)
        atomic_write_text(
            workspace / "conversion-failure.json",
            json.dumps(journal, indent=2, sort_keys=False) + "\n",
        )
        return journal
    host_snapshot = collect_host_snapshot(workspace)
    receipt = ConversionReceipt(
        schema_version=CONVERSION_RECEIPT_SCHEMA_VERSION,
        conversion_id=manifest.conversion_id,
        conversion_manifest_hash=conversion_manifest_hash(manifest),
        status="completed",
        started_at=started_at,
        ended_at=ended_at,
        source={
            "official_id": manifest.source.official_id,
            "official_revision": manifest.source.official_revision,
            "repository_id": manifest.source.repository_id,
            "license": manifest.source.license,
            "expected_source_bytes": manifest.source.expected_source_bytes,
            "files_verified": source_result["files"],
        },
        converter={
            "package": manifest.converter.package,
            "version": manifest.converter.version,
            "git_repository": manifest.converter.git_repository,
            "git_revision": manifest.converter.git_revision,
            "installed_version": journal["converter_version_installed"],
        },
        parameters=asdict(manifest.parameters),
        argv=tuple(argv),
        output_files=output_files,
        output_total_bytes=sum(item["size_bytes"] for item in output_files),
        host=host_snapshot.to_dict(),
    )
    atomic_write_text(
        workspace / "conversion-receipt.json",
        json.dumps(receipt.to_dict(), indent=2, sort_keys=False) + "\n",
    )
    journal["receipt_path"] = str((workspace / "conversion-receipt.json").resolve())
    return journal


def conversion_manifest_hash(manifest: ConversionManifest) -> str:
    return config_hash(asdict(manifest))


__all__ = [
    "ConversionError",
    "ConversionReceipt",
    "ConversionSafetyDecision",
    "acquire_source",
    "assess_conversion_safety",
    "conversion_manifest_hash",
    "run_conversion",
    "source_cache_has_existing_content",
    "source_files_present",
    "verify_source",
]
