"""Bounded reads for persisted optimizer artifacts."""

from __future__ import annotations

import os
import stat
from pathlib import Path

MAX_METADATA_ARTIFACT_BYTES = 64 * 1024
MAX_EVIDENCE_ARTIFACT_BYTES = 64 * 1024 * 1024


class ArtifactReadError(ValueError):
    """Raised when an artifact is not a bounded, regular UTF-8 file."""


def read_bounded_regular_text(path: str | Path, max_bytes: int) -> str:
    """Read a regular UTF-8 file without following symlinks or exceeding a limit.

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

    try:
        return b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ArtifactReadError(f"{artifact_path} is not valid UTF-8") from exc
