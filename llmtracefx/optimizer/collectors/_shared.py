"""Helpers shared by the MLX-LM and native-MTP collectors.

Kept separate from ``collectors.mlx`` so new collectors extend the same
primitives (hashing, atomic writes, wall-clock/byte measurements, platform
recording) instead of duplicating them.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

from ..manifest import collect_environment_manifest
from ..schema import Measurement, MetricProvenance, PlatformInfo


def sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def milliseconds(started: float | None, ended: float | None) -> Measurement | None:
    if started is None or ended is None:
        return None
    return Measurement(
        value=max(0.0, ended - started) * 1000,
        provenance=MetricProvenance.MEASURED_WALL_CLOCK,
        unit="ms",
    )


def bytes_measurement(value: int | None) -> Measurement | None:
    if value is None:
        return None
    return Measurement(
        value=float(value),
        provenance=MetricProvenance.MEASURED_NATIVE,
        unit="bytes",
    )


def atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` so the bytes on disk are exactly its UTF-8 encoding.

    ``newline=""`` disables the translation text mode would otherwise
    apply. Without it a ``\\n`` becomes the platform line ending on write
    while ``\\r\\n`` and a lone ``\\r`` collapse back to ``\\n`` on read, so
    a response that legitimately contains carriage returns would not hash
    to the same value it was written with.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8", newline="")
    os.replace(temporary, path)


def record_platform(
    *, accelerator: str | None, extra_packages: tuple[str, ...] = ("mlx", "mlx-lm")
) -> PlatformInfo:
    manifest = collect_environment_manifest(extra_packages=extra_packages)
    return PlatformInfo(
        os_name=manifest.os_name,
        os_version=manifest.os_release,
        architecture=manifest.architecture,
        cpu_cores=manifest.cpu_count,
        total_memory_gb=manifest.total_memory_gb,
        accelerator=accelerator,
    )


def config_hash(payload: dict[str, Any]) -> str:
    import json

    return sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
