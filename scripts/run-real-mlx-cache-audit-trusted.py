#!/usr/bin/env python3
"""Verify the external MLX runtime before importing the real audit CLI."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import runpy
import stat
import sys
from pathlib import Path
from typing import Any

EXPECTED_IDENTITY_SHA256 = (
    "sha256:391e14ce1b09b5de11b96ab24f98dd73c4c8334625599e1cda5dd5f4906c16a5"
)
EXPECTED_DISTRIBUTIONS = {
    "anyio": "4.9.0",
    "certifi": "2025.7.9",
    "click": "8.1.8",
    "filelock": "3.32.4",
    "fsspec": "2026.7.0",
    "h11": "0.16.0",
    "hf-xet": "1.6.0",
    "httpcore": "1.0.9",
    "httpx": "0.28.1",
    "huggingface-hub": "1.13.0",
    "idna": "3.15",
    "jinja2": "3.1.6",
    "markdown-it-py": "3.0.0",
    "markupsafe": "3.0.2",
    "mdurl": "0.1.2",
    "mlx": "0.32.2",
    "mlx-lm": "0.31.3",
    "mlx-metal": "0.32.2",
    "numpy": "2.2.6",
    "packaging": "26.3",
    "protobuf": "6.33.5",
    "pygments": "2.20.0",
    "pyyaml": "6.0.2",
    "regex": "2026.7.19",
    "rich": "14.0.0",
    "safetensors": "0.8.0",
    "sentencepiece": "0.2.2",
    "shellingham": "1.5.4",
    "sniffio": "1.3.1",
    "tokenizers": "0.23.1",
    "tqdm": "4.70.0",
    "transformers": "5.16.1",
    "typer": "0.16.0",
    "typing-extensions": "4.14.1",
}
_FORBIDDEN_FILENAMES = {"sitecustomize.py", "usercustomize.py"}
_FORBIDDEN_SUFFIXES = {".pth", ".pyc", ".pyo"}


class BootstrapError(RuntimeError):
    """A fail-closed bootstrap verification error."""


def _normalized_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).lower()


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _safe_existing_directory(path: Path, parent: Path | None = None) -> Path:
    absolute = Path(os.path.abspath(path))
    if absolute.is_symlink() or not absolute.is_dir():
        raise BootstrapError("trusted directory is missing or symlinked")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise BootstrapError("trusted directory is unavailable") from exc
    if resolved != absolute or (
        parent is not None and not _is_relative_to(resolved, parent)
    ):
        raise BootstrapError("trusted directory is outside its expected root")
    return resolved


def _external_site_root() -> tuple[Path, Path]:
    executable = Path(os.path.abspath(sys.executable))
    if executable.parent.name != "bin" or executable.name not in {
        "python",
        f"python{sys.version_info.major}",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
    }:
        raise BootstrapError("bootstrap requires an external venv bin/python")
    venv_root = _safe_existing_directory(executable.parent.parent)
    site_root = _safe_existing_directory(
        venv_root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages",
        venv_root,
    )
    return venv_root, site_root


def _repository_paths() -> tuple[Path, Path]:
    script = Path(os.path.abspath(__file__))
    if script.is_symlink() or not script.is_file():
        raise BootstrapError("bootstrap script is missing or symlinked")
    try:
        resolved_script = script.resolve(strict=True)
    except OSError as exc:
        raise BootstrapError("bootstrap script is unavailable") from exc
    if resolved_script != script:
        raise BootstrapError("bootstrap script path is unsafe")
    repo_root = _safe_existing_directory(script.parent.parent)
    expected_script = repo_root / "scripts" / "run-real-mlx-cache-audit-trusted.py"
    if script != expected_script:
        raise BootstrapError("bootstrap script is outside the repository")
    identity = (
        repo_root
        / "llmtracefx"
        / "cache_audit"
        / "data"
        / "apple-silicon-python313-mlx-lm-runtime-v1.json"
    )
    identity_absolute = Path(os.path.abspath(identity))
    if (
        identity_absolute.is_symlink()
        or not identity_absolute.is_file()
        or identity_absolute.resolve(strict=True) != identity_absolute
        or not _is_relative_to(identity_absolute, repo_root)
    ):
        raise BootstrapError("committed runtime identity is unavailable")
    return repo_root, identity_absolute


def _scan_site_root(site_root: Path) -> None:
    def fail_walk(error: OSError) -> None:
        raise BootstrapError("site-packages could not be scanned") from error

    for current, directories, filenames in os.walk(
        site_root, followlinks=False, onerror=fail_walk
    ):
        current_path = Path(current)
        for name in directories:
            candidate = current_path / name
            if name == "__pycache__":
                raise BootstrapError("site-packages contains forbidden bytecode")
            if candidate.is_symlink():
                raise BootstrapError("site-packages contains a symlink")
        for name in filenames:
            candidate = current_path / name
            if (
                name in _FORBIDDEN_FILENAMES
                or candidate.suffix.lower() in _FORBIDDEN_SUFFIXES
            ):
                raise BootstrapError(
                    "site-packages contains a startup hook or bytecode"
                )
            if candidate.is_symlink():
                raise BootstrapError("site-packages contains a symlink")


def _regular_file_identity(path: Path) -> tuple[int, str]:
    absolute = Path(os.path.abspath(path))
    if absolute.is_symlink() or not absolute.is_file():
        raise BootstrapError("runtime distribution contains a missing or unsafe file")
    before = absolute.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise BootstrapError("runtime distribution contains a non-regular file")
    digest = hashlib.sha256()
    with absolute.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    after = absolute.lstat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise BootstrapError("runtime distribution changed while hashing")
    return before.st_size, "sha256:" + digest.hexdigest()


def _distribution_identity(
    distribution: importlib.metadata.Distribution,
    name: str,
    version: str,
    site_root: Path,
) -> dict[str, str | int]:
    declared = distribution.files
    if not declared:
        raise BootstrapError(f"{name} distribution files are unavailable")
    files: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for item in declared:
        declared_path = Path(str(item))
        if declared_path.is_absolute() or not declared_path.parts:
            raise BootstrapError(f"{name} declares an unsafe path")
        if (
            declared_path.suffix.lower() in {".pyc", ".pyo"}
            or "__pycache__" in declared_path.parts
        ):
            raise BootstrapError(f"{name} declares forbidden bytecode")
        located = Path(os.path.abspath(str(distribution.locate_file(item))))
        if not _is_relative_to(located, site_root):
            scripts_root = site_root.parents[2] / "bin"
            if _is_relative_to(located, scripts_root):
                continue
            raise BootstrapError(f"{name} declares an unsafe out-of-root path")
        relative = located.relative_to(site_root)
        if relative.name == "RECORD" and relative.parent.name.endswith(".dist-info"):
            continue
        logical_path = "site-packages/" + relative.as_posix()
        if logical_path in seen:
            raise BootstrapError(f"{name} declares a duplicate package path")
        seen.add(logical_path)
        try:
            resolved = located.resolve(strict=True)
        except OSError as exc:
            raise BootstrapError(f"{name} contains a missing package file") from exc
        if resolved != located or not _is_relative_to(resolved, site_root):
            raise BootstrapError(f"{name} contains a symlinked package file")
        files.append((logical_path, resolved))
    if not files:
        raise BootstrapError(f"{name} has no declared site-packages files")
    tree_digest = hashlib.sha256()
    total_bytes = 0
    for logical_path, path in sorted(files):
        size, digest = _regular_file_identity(path)
        total_bytes += size
        payload = json.dumps(
            {"path": logical_path, "sha256": digest, "size": size},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        tree_digest.update(payload)
        tree_digest.update(b"\n")
    return {
        "distribution": name,
        "version": version,
        "trusted_root": "purelib",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "tree_sha256": "sha256:" + tree_digest.hexdigest(),
    }


def _actual_identity(site_root: Path) -> dict[str, dict[str, str | int]]:
    discovered: dict[str, list[importlib.metadata.Distribution]] = {}
    for distribution in importlib.metadata.distributions(path=[str(site_root)]):
        raw_name = distribution.metadata["Name"]
        if isinstance(raw_name, str):
            discovered.setdefault(_normalized_name(raw_name), []).append(distribution)
    identities: dict[str, dict[str, str | int]] = {}
    for name, version in EXPECTED_DISTRIBUTIONS.items():
        matches = discovered.get(name, [])
        if len(matches) != 1:
            raise BootstrapError(
                f"{name} expected exactly one normalized .dist-info distribution"
            )
        distribution = matches[0]
        if distribution.version != version:
            raise BootstrapError(f"{name} distribution version mismatch")
        try:
            root = Path(str(distribution.locate_file(""))).resolve(strict=True)
        except OSError as exc:
            raise BootstrapError(f"{name} distribution root is unavailable") from exc
        if root != site_root:
            raise BootstrapError(f"{name} distribution root is untrusted")
        identities[name] = _distribution_identity(
            distribution, name, version, site_root
        )
    return identities


def _load_expected(identity_path: Path) -> dict[str, Any]:
    raw = identity_path.read_bytes()
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_IDENTITY_SHA256:
        raise BootstrapError("committed runtime identity digest mismatch")

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise BootstrapError("runtime identity contains duplicate JSON keys")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("ascii"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BootstrapError("committed runtime identity is invalid") from exc
    if not isinstance(value, dict) or set(value) != set(EXPECTED_DISTRIBUTIONS):
        raise BootstrapError("committed runtime identity closure is invalid")
    return value


def _verify() -> tuple[Path, Path]:
    if sys.flags.isolated != 1 or sys.flags.no_site != 1:
        raise BootstrapError("bootstrap requires Python -I -S")
    _venv_root, site_root = _external_site_root()
    repo_root, identity_path = _repository_paths()
    _scan_site_root(site_root)
    expected = _load_expected(identity_path)
    if _actual_identity(site_root) != expected:
        raise BootstrapError("installed runtime identity does not match committed tree")
    _scan_site_root(site_root)
    return repo_root, site_root


def main() -> None:
    try:
        repo_root, site_root = _verify()
        if sys.argv[1:] == ["--bootstrap-self-test"]:
            print(
                json.dumps(
                    {
                        "bootstrap_verified": True,
                        "distribution_count": len(EXPECTED_DISTRIBUTIONS),
                    },
                    sort_keys=True,
                )
            )
            return
        if not sys.argv[1:]:
            raise BootstrapError("real CLI arguments are required")
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ["PYTHONSAFEPATH"] = "1"
        os.environ["LLMTRACEFX_TRUSTED_REPO_ROOT"] = str(repo_root)
        os.environ["LLMTRACEFX_TRUSTED_BOOTSTRAP"] = "1"
        sys.dont_write_bytecode = True
        # Keep verified third-party packages ahead of the repository so an
        # untracked top-level module cannot shadow the pinned runtime. The
        # repository is added only for the reviewed llmtracefx source tree.
        sys.path.extend((str(site_root), str(repo_root)))
        sys.argv = ["llmtracefx-real-mlx-cache-audit", *sys.argv[1:]]
        runpy.run_module("llmtracefx.cache_audit.real_mlx", run_name="__main__")
    except BootstrapError as exc:
        print(json.dumps({"error": str(exc), "ok": False}, sort_keys=True))
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
