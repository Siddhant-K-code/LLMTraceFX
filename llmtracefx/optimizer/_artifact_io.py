"""Bounded reads for persisted optimizer artifacts."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import NoReturn

MAX_METADATA_ARTIFACT_BYTES = 64 * 1024
MAX_EVIDENCE_ARTIFACT_BYTES = 64 * 1024 * 1024


class ArtifactReadError(ValueError):
    """Raised when an artifact is not a bounded, regular UTF-8 file."""


def reject_non_finite_json_constant(value: str) -> NoReturn:
    """Reject Python's non-standard JSON NaN and infinity extensions."""
    raise ValueError(f"non-finite JSON number {value}")


def read_bounded_regular_bytes(path: str | Path, max_bytes: int) -> bytes:
    """Read a regular file without following symlinks or exceeding a limit.

    Filesystem failures remain ``OSError`` so callers can distinguish a missing
    artifact from one that exists but is structurally unsafe.
    """
    artifact_path = Path(path)
    if artifact_path.is_symlink():
        raise ArtifactReadError(f"{artifact_path} must not be a symlink")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(artifact_path, flags)

    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ArtifactReadError(f"{artifact_path} must be a regular file")
        if metadata.st_size > max_bytes:
            raise ArtifactReadError(
                f"{artifact_path} exceeds the {max_bytes}-byte size limit"
            )

        chunks: list[bytes] = []
        bytes_read = 0
        while True:
            chunk = os.read(descriptor, min(64 * 1024, max_bytes - bytes_read + 1))
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > max_bytes:
                raise ArtifactReadError(
                    f"{artifact_path} exceeds the {max_bytes}-byte size limit"
                )
            chunks.append(chunk)
    finally:
        os.close(descriptor)

    return b"".join(chunks)


def read_bounded_regular_text(path: str | Path, max_bytes: int) -> str:
    """Read a bounded regular file and require strict UTF-8."""
    raw = read_bounded_regular_bytes(path, max_bytes)
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactReadError(f"{path} is not valid UTF-8") from exc
