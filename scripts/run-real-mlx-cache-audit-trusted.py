#!/usr/bin/env python3
"""Verify and snapshot the canonical MLX audit before importing project code."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import re
import runpy
import stat
import subprocess
import sys
import tarfile
import tempfile
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
_CANONICAL_COMMANDS = {
    "aggregate",
    "calibrate",
    "compile",
    "preflight",
    "replicate",
    "run-all",
    "sandbox-probe",
    "sanitize",
    "verify",
}
_FORBIDDEN_FILENAMES = {"sitecustomize.py", "usercustomize.py"}
_FORBIDDEN_SUFFIXES = {".pth", ".pyc", ".pyo"}
_SNAPSHOT_PREFIXES = ("llmtracefx/", "vllm_kv_truth/", "scripts/")
_IDENTITY_PATH = (
    "llmtracefx/cache_audit/data/" "apple-silicon-python313-mlx-lm-runtime-v1.json"
)
_BOOTSTRAP_PATH = "scripts/run-real-mlx-cache-audit-trusted.py"


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


def _running_script() -> Path:
    script = Path(os.path.abspath(__file__))
    if script.is_symlink() or not script.is_file():
        raise BootstrapError("bootstrap script is missing or symlinked")
    try:
        resolved_script = script.resolve(strict=True)
    except OSError as exc:
        raise BootstrapError("bootstrap script is unavailable") from exc
    if resolved_script != script:
        raise BootstrapError("bootstrap script path is unsafe")
    return script


def _trusted_repository(argv: list[str]) -> tuple[Path, list[str]]:
    values: list[str] = []
    forwarded: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--trusted-repo-root":
            if index + 1 >= len(argv):
                raise BootstrapError("--trusted-repo-root requires an absolute path")
            values.append(argv[index + 1])
            index += 2
            continue
        if item.startswith("--trusted-repo-root="):
            values.append(item.split("=", 1)[1])
            index += 1
            continue
        forwarded.append(item)
        index += 1
    if len(values) != 1:
        raise BootstrapError("exactly one --trusted-repo-root is required")
    raw = Path(values[0])
    if not raw.is_absolute():
        raise BootstrapError("--trusted-repo-root must be absolute")
    return _safe_existing_directory(raw), forwarded


def _scan_site_root(
    site_root: Path,
    *,
    owned_files: set[str] | None = None,
    generated_records: set[str] | None = None,
) -> None:
    allowed = (owned_files or set()) | (generated_records or set())

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
            before = candidate.lstat()
            if not stat.S_ISREG(before.st_mode):
                raise BootstrapError("site-packages contains a non-regular file")
            relative = candidate.relative_to(site_root).as_posix()
            if owned_files is not None and relative not in allowed:
                raise BootstrapError(
                    f"site-packages contains unowned regular file: {relative}"
                )


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


def _regular_file_bytes(path: Path) -> bytes:
    absolute = Path(os.path.abspath(path))
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise BootstrapError("bootstrap script is unavailable") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise BootstrapError("bootstrap script is not a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw = handle.read()
        after = os.fstat(descriptor)
        path_state = absolute.lstat()
    except OSError as exc:
        raise BootstrapError("bootstrap script could not be read") from exc
    finally:
        os.close(descriptor)
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_size,
        before.st_mtime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_size,
        after.st_mtime_ns,
    ) or identity != (
        path_state.st_dev,
        path_state.st_ino,
        path_state.st_mode,
        path_state.st_size,
        path_state.st_mtime_ns,
    ):
        raise BootstrapError("bootstrap script changed while being authenticated")
    return raw


def _identity_record_bytes(logical_path: str, size: int, digest: str) -> bytes:
    return json.dumps(
        {"path": logical_path, "sha256": digest, "size": size},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _distribution_files(
    distribution: importlib.metadata.Distribution,
    name: str,
    site_root: Path,
) -> tuple[list[tuple[str, Path]], set[str]]:
    declared = distribution.files
    if not declared:
        raise BootstrapError(f"{name} distribution files are unavailable")
    files: list[tuple[str, Path]] = []
    records: set[str] = set()
    seen: set[str] = set()
    scripts_root = site_root.parents[2] / "bin"
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
            if _is_relative_to(located, scripts_root):
                continue
            raise BootstrapError(f"{name} declares an unsafe out-of-root path")
        relative = located.relative_to(site_root)
        relative_name = relative.as_posix()
        if relative.name == "RECORD" and relative.parent.name.endswith(".dist-info"):
            if relative_name in records:
                raise BootstrapError(f"{name} declares a duplicate RECORD path")
            records.add(relative_name)
            continue
        logical_path = "site-packages/" + relative_name
        if relative_name in seen:
            raise BootstrapError(f"{name} declares a duplicate package path")
        seen.add(relative_name)
        try:
            resolved = located.resolve(strict=True)
        except OSError as exc:
            raise BootstrapError(f"{name} contains a missing package file") from exc
        if resolved != located or not _is_relative_to(resolved, site_root):
            raise BootstrapError(f"{name} contains a symlinked package file")
        files.append((logical_path, resolved))
    if not files:
        raise BootstrapError(f"{name} has no declared site-packages files")
    return files, records


def _distribution_identity(
    distribution: importlib.metadata.Distribution,
    name: str,
    version: str,
    site_root: Path,
) -> dict[str, str | int]:
    files, _records = _distribution_files(distribution, name, site_root)
    tree_digest = hashlib.sha256()
    total_bytes = 0
    for logical_path, path in sorted(files):
        size, digest = _regular_file_identity(path)
        total_bytes += size
        tree_digest.update(_identity_record_bytes(logical_path, size, digest))
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
        if not isinstance(raw_name, str):
            raise BootstrapError("runtime distribution has no valid name")
        discovered.setdefault(_normalized_name(raw_name), []).append(distribution)
    if set(discovered) != set(EXPECTED_DISTRIBUTIONS):
        raise BootstrapError("installed distribution closure is not exact")
    identities: dict[str, dict[str, str | int]] = {}
    owned_files: set[str] = set()
    generated_records: set[str] = set()
    for name, version in EXPECTED_DISTRIBUTIONS.items():
        matches = discovered[name]
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
        files, records = _distribution_files(distribution, name, site_root)
        relative_files = {
            logical.removeprefix("site-packages/") for logical, _path in files
        }
        if owned_files.intersection(relative_files):
            raise BootstrapError("runtime distributions claim the same package file")
        if generated_records.intersection(records):
            raise BootstrapError("runtime distributions claim the same RECORD file")
        owned_files.update(relative_files)
        generated_records.update(records)
        identities[name] = _distribution_identity(
            distribution, name, version, site_root
        )
    _scan_site_root(
        site_root,
        owned_files=owned_files,
        generated_records=generated_records,
    )
    return identities


def _load_expected_bytes(raw: bytes) -> dict[str, Any]:
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


def _expected_commit(argv: list[str]) -> str:
    command = argv[0] if argv else ""
    if command == "--bootstrap-self-test":
        canonical = True
    else:
        canonical = command in _CANONICAL_COMMANDS
    values: list[str] = []
    for index, item in enumerate(argv):
        if item == "--expected-commit" and index + 1 < len(argv):
            values.append(argv[index + 1])
        elif item.startswith("--expected-commit="):
            values.append(item.split("=", 1)[1])
    if not canonical:
        raise BootstrapError("bootstrap accepts canonical commands only")
    if len(values) != 1 or re.fullmatch(r"[0-9a-f]{40}", values[0]) is None:
        raise BootstrapError("exactly one full lowercase --expected-commit is required")
    return values[0]


def _git_environment() -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "HOME": "/dev/null",
        "LC_ALL": "C",
        "LANG": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_NO_LAZY_FETCH": "1",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }


def _git(
    repo_root: Path, args: list[str], *, text: bool = False
) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        [
            "/usr/bin/git",
            "--no-replace-objects",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.attributesFile=/dev/null",
            "-C",
            str(repo_root),
            *args,
        ],
        capture_output=True,
        check=False,
        env=_git_environment(),
        text=text,
    )


def _verify_repository(
    repo_root: Path,
    script_path: Path,
    expected_commit: str,
) -> bytes:
    head = _git(repo_root, ["rev-parse", "--verify", "HEAD^{commit}"], text=True)
    if head.returncode != 0 or head.stdout.strip() != expected_commit:
        raise BootstrapError("current Git HEAD does not match expected commit")
    committed_script = _git(repo_root, ["show", f"{expected_commit}:{_BOOTSTRAP_PATH}"])
    if (
        committed_script.returncode != 0
        or committed_script.stdout != _regular_file_bytes(script_path)
    ):
        raise BootstrapError(f"{_BOOTSTRAP_PATH} does not match expected commit")
    committed_identity = _git(
        repo_root, ["show", f"{expected_commit}:{_IDENTITY_PATH}"]
    )
    if committed_identity.returncode != 0 or not isinstance(
        committed_identity.stdout, bytes
    ):
        raise BootstrapError("committed runtime identity is unavailable")
    identity_bytes = committed_identity.stdout
    _load_expected_bytes(identity_bytes)
    return identity_bytes


def _output_paths(argv: list[str]) -> tuple[Path, ...]:
    paths: list[Path] = []
    output_flags = {"--output", "--output-dir", "--output-workspace"}
    for index, item in enumerate(argv):
        raw: str | None = None
        if item in output_flags and index + 1 < len(argv):
            raw = argv[index + 1]
        else:
            for flag in output_flags:
                if item.startswith(flag + "="):
                    raw = item.split("=", 1)[1]
                    break
        if raw is not None:
            candidate = Path(raw).expanduser()
            try:
                parent = candidate.parent.resolve(strict=True)
            except OSError:
                continue
            paths.append(parent / candidate.name)
    return tuple(paths)


def _materialize_snapshot(
    repo_root: Path,
    expected_commit: str,
    destination: Path,
) -> Path:
    archived = _git(
        repo_root,
        [
            "archive",
            "--format=tar",
            expected_commit,
            "--",
            "llmtracefx",
            "vllm_kv_truth",
            _BOOTSTRAP_PATH,
        ],
    )
    if archived.returncode != 0:
        raise BootstrapError("trusted source archive is unavailable")
    snapshot = destination / "snapshot"
    snapshot.mkdir(mode=0o700)
    seen: set[str] = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(archived.stdout), mode="r:") as archive:
            for member in archive:
                name = member.name.rstrip("/")
                pure = Path(name)
                if (
                    not name
                    or pure.is_absolute()
                    or ".." in pure.parts
                    or not any(
                        name == prefix.rstrip("/") or name.startswith(prefix)
                        for prefix in _SNAPSHOT_PREFIXES
                    )
                    or name in seen
                    or not (member.isdir() or member.isfile())
                    or member.issym()
                    or member.islnk()
                ):
                    raise BootstrapError("Git archive contains an unsafe entry")
                seen.add(name)
                target = snapshot.joinpath(*pure.parts)
                if member.isdir():
                    target.mkdir(mode=0o700, parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise BootstrapError("Git archive file is unavailable")
                with target.open("xb") as handle:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                target.chmod(0o400)
    except (OSError, tarfile.TarError) as exc:
        raise BootstrapError("trusted source archive is invalid") from exc
    for required in ("llmtracefx", "vllm_kv_truth"):
        if not (snapshot / required).is_dir():
            raise BootstrapError("trusted source snapshot is incomplete")
    if not (snapshot / _BOOTSTRAP_PATH).is_file():
        raise BootstrapError("trusted source snapshot is incomplete")
    for directory in sorted(
        (path for path in snapshot.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o500)
    snapshot.chmod(0o500)
    return snapshot


def _verify(
    argv: list[str],
) -> tuple[
    Path,
    Path,
    str,
    tempfile.TemporaryDirectory[str],
    Path,
    list[str],
]:
    if sys.flags.isolated != 1 or sys.flags.no_site != 1:
        raise BootstrapError("bootstrap requires Python -I -S")
    repo_root, forwarded = _trusted_repository(argv)
    expected_commit = _expected_commit(forwarded)
    venv_root, site_root = _external_site_root()
    script_path = _running_script()
    outputs = _output_paths(forwarded)
    if _is_relative_to(venv_root, repo_root) or _is_relative_to(repo_root, venv_root):
        raise BootstrapError("trusted runtime venv must be external to the repository")
    if any(
        _is_relative_to(venv_root, output) or _is_relative_to(output, venv_root)
        for output in outputs
    ):
        raise BootstrapError(
            "trusted runtime venv and output workspace must not overlap"
        )
    identity_bytes = _verify_repository(repo_root, script_path, expected_commit)
    _scan_site_root(site_root)
    expected = _load_expected_bytes(identity_bytes)
    if _actual_identity(site_root) != expected:
        raise BootstrapError("installed runtime identity does not match committed tree")
    _scan_site_root(site_root)
    owner = tempfile.TemporaryDirectory(
        prefix="llmtracefx-trusted-source-", dir=venv_root.parent
    )
    owner_path = Path(owner.name).resolve(strict=True)
    if _is_relative_to(owner_path, repo_root) or any(
        _is_relative_to(owner_path, output) or _is_relative_to(output, owner_path)
        for output in outputs
    ):
        owner.cleanup()
        raise BootstrapError("trusted source snapshot overlaps repository or output")
    snapshot = _materialize_snapshot(repo_root, expected_commit, owner_path)
    return repo_root, site_root, expected_commit, owner, snapshot, forwarded


def main() -> None:
    owner: tempfile.TemporaryDirectory[str] | None = None
    try:
        (
            repo_root,
            site_root,
            expected_commit,
            owner,
            snapshot,
            forwarded,
        ) = _verify(sys.argv[1:])
        if forwarded[0] == "--bootstrap-self-test":
            print(
                json.dumps(
                    {
                        "bootstrap_verified": True,
                        "distribution_count": len(EXPECTED_DISTRIBUTIONS),
                        "snapshot_verified": True,
                    },
                    sort_keys=True,
                )
            )
            return
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        os.environ["PYTHONSAFEPATH"] = "1"
        os.environ["LLMTRACEFX_TRUSTED_REPO_ROOT"] = str(repo_root)
        os.environ["LLMTRACEFX_TRUSTED_COMMIT"] = expected_commit
        os.environ["LLMTRACEFX_TRUSTED_SNAPSHOT_ROOT"] = str(snapshot)
        os.environ["LLMTRACEFX_TRUSTED_BOOTSTRAP"] = "1"
        sys.dont_write_bytecode = True
        sys.path.extend((str(site_root), str(snapshot)))
        sys.argv = ["llmtracefx-real-mlx-cache-audit", *forwarded]
        runpy.run_module("llmtracefx.cache_audit.real_mlx", run_name="__main__")
    except BootstrapError as exc:
        print(json.dumps({"error": str(exc), "ok": False}, sort_keys=True))
        raise SystemExit(2) from None
    finally:
        if owner is not None:
            owner.cleanup()


if __name__ == "__main__":
    main()
