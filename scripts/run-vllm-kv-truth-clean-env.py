#!/usr/bin/env python3
"""Verify a dedicated installation before importing or running LLMTraceFX.

Invoke this file only through native ``/usr/bin/env -i`` and the dedicated
virtual environment's absolute Python path with ``-I -S -B``.  ``-S`` is
important: it prevents site-packages and executable ``.pth`` hooks from being
processed before this bootstrap has verified the environment. ``-B`` keeps the
verified environment immutable while package code is imported.
"""

import base64
import csv
import hashlib
import io
import json
import os
import stat
import sys
import sysconfig
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

SAFE_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin:/usr/local/bin",
    "LANG": "C",
    "LC_ALL": "C",
}
PLATFORM_ENVIRONMENT_NAMES = {"__CF_USER_TEXT_ENCODING"}
SCHEMA_VERSION = 2
MAX_SYMLINKS = 32


class BootstrapError(Exception):
    """A non-sensitive bootstrap refusal."""


def _fail(message: str) -> NoReturn:
    raise BootstrapError(message)


def _require_clean_environment() -> None:
    changed = [
        name
        for name, expected in SAFE_ENVIRONMENT.items()
        if os.environ.get(name) != expected
    ]
    unexpected = sorted(
        set(os.environ) - set(SAFE_ENVIRONMENT) - PLATFORM_ENVIRONMENT_NAMES
    )
    if changed or unexpected:
        _fail("process environment is not the fixed clean environment")


def _require_python_flags() -> None:
    if (
        not sys.flags.isolated
        or not sys.flags.no_site
        or not sys.flags.dont_write_bytecode
    ):
        _fail("Python must be invoked with -I -S -B")


def _absolute_path(value: str, label: str) -> Path:
    if not value.startswith("/") or "\x00" in value:
        _fail(f"{label} must be an absolute path")
    if value.startswith("//") or os.path.normpath(value) != value:
        _fail(f"{label} path is ambiguous")
    return Path(value)


def _mode_writable(mode: int) -> bool:
    return bool(stat.S_IMODE(mode) & 0o022)


def _trusted_owner(owner: int) -> bool:
    return owner in {0, os.geteuid()}


def _check_trusted_directory(path: Path, label: str) -> None:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
        _fail(f"{label} has an unsafe directory component")
    if not _trusted_owner(info.st_uid):
        _fail(f"{label} directory has an untrusted owner")
    if _mode_writable(info.st_mode):
        if info.st_uid != 0 or not (info.st_mode & stat.S_ISVTX):
            _fail(f"{label} directory is group- or world-writable")


def _require_nonsymlink_parents(path: Path, label: str) -> None:
    cursor = Path("/")
    for component in path.parent.parts[1:]:
        cursor /= component
        try:
            _check_trusted_directory(cursor, label)
        except FileNotFoundError:
            _fail(f"{label} parent directory does not exist")


def _open_trusted_file(
    path: Path,
    label: str,
    *,
    private: bool = False,
    executable: bool = False,
    allow_root: bool = False,
) -> int:
    _require_nonsymlink_parents(path, label)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError:
        _fail(f"{label} must be a regular non-symlink file")
    try:
        opened = os.fstat(descriptor)
        named = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_ISLNK(named.st_mode)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            _fail(f"{label} must be a regular non-symlink file")
        if opened.st_uid != os.geteuid() and not (allow_root and opened.st_uid == 0):
            _fail(f"{label} must be owned by the current user")
        mode = stat.S_IMODE(opened.st_mode)
        if private and mode != 0o600:
            _fail(f"{label} must have mode 0600")
        if not private and mode & 0o022:
            _fail(f"{label} must not be group- or world-writable")
        if executable and not mode & 0o100:
            _fail(f"{label} must be executable by its owner")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_descriptor(descriptor: int) -> bytes:
    chunks: list[bytes] = []
    while True:
        chunk = os.read(descriptor, 1024 * 1024)
        if not chunk:
            return b"".join(chunks)
        chunks.append(chunk)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_trusted_bytes(
    path: Path,
    label: str,
    *,
    private: bool = False,
    executable: bool = False,
    allow_root: bool = False,
) -> bytes:
    descriptor = _open_trusted_file(
        path,
        label,
        private=private,
        executable=executable,
        allow_root=allow_root,
    )
    try:
        return _read_descriptor(descriptor)
    finally:
        os.close(descriptor)


def _resolve_trusted_interpreter(path: Path) -> Path:
    pending = path
    links = 0
    while True:
        cursor = Path("/")
        parts = list(pending.parts[1:])
        restarted = False
        for index, component in enumerate(parts):
            parent = cursor
            cursor /= component
            try:
                info = cursor.lstat()
            except OSError:
                _fail("Python interpreter path does not exist")
            if stat.S_ISLNK(info.st_mode):
                if not _trusted_owner(info.st_uid):
                    _fail("Python interpreter link has an untrusted owner")
                _check_trusted_directory(parent, "Python interpreter")
                links += 1
                if links > MAX_SYMLINKS:
                    _fail("Python interpreter has too many symlink levels")
                target = Path(os.readlink(cursor))
                if not target.is_absolute():
                    target = parent / target
                pending = Path(
                    os.path.normpath(str(target.joinpath(*parts[index + 1 :])))
                )
                if not pending.is_absolute():
                    _fail("Python interpreter link target is ambiguous")
                restarted = True
                break
            if index < len(parts) - 1:
                try:
                    _check_trusted_directory(cursor, "Python interpreter")
                except FileNotFoundError:
                    _fail("Python interpreter path does not exist")
        if restarted:
            continue
        break

    info = pending.lstat()
    if not stat.S_ISREG(info.st_mode) or not info.st_mode & 0o111:
        _fail("Python interpreter target is not an executable regular file")
    if not _trusted_owner(info.st_uid):
        _fail("Python interpreter target has an untrusted owner")
    if _mode_writable(info.st_mode):
        _fail("Python interpreter target is group- or world-writable")
    return pending


def _venv_root(interpreter: Path, bootstrap: Path) -> Path:
    if interpreter.parent.name != "bin":
        _fail("Python interpreter must be in the dedicated environment bin directory")
    root = interpreter.parent.parent
    if bootstrap.parent != interpreter.parent:
        _fail(
            "bootstrap and Python interpreter must share the environment bin directory"
        )
    _require_nonsymlink_parents(root / "sentinel", "virtual environment")
    if not (root / "pyvenv.cfg").is_file():
        _fail("Python interpreter must belong to a virtual environment")
    return root


def _validate_environment_symlink(path: Path, root: Path) -> None:
    resolved = path.resolve(strict=True)
    if path.parent == root / "bin" and path.name.startswith("python"):
        _resolve_trusted_interpreter(path)
        return
    try:
        resolved.relative_to(root)
    except ValueError:
        _fail("virtual environment contains an external symbolic link")
    _require_nonsymlink_parents(resolved / "sentinel", "virtual environment link")
    info = resolved.lstat()
    if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
        _fail("virtual environment symbolic link has an unsafe target")
    if not _trusted_owner(info.st_uid) or _mode_writable(info.st_mode):
        _fail("virtual environment symbolic link has an unsafe target")


def _inventory_environment(root: Path) -> tuple[str, int]:
    entries: list[dict[str, Any]] = []
    for directory, names, files in os.walk(root, topdown=True, followlinks=False):
        directory_path = Path(directory)
        names[:] = sorted(names)
        for name in sorted([*names, *files]):
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            info = path.lstat()
            mode = stat.S_IMODE(info.st_mode)
            if stat.S_ISLNK(info.st_mode):
                if not _trusted_owner(info.st_uid):
                    _fail("virtual environment contains an untrusted symbolic link")
                try:
                    _validate_environment_symlink(path, root)
                except (OSError, RuntimeError):
                    _fail("virtual environment contains an unsafe symbolic link")
                entries.append(
                    {
                        "kind": "symlink",
                        "mode": mode,
                        "path": relative,
                        "target": os.readlink(path),
                    }
                )
            elif stat.S_ISDIR(info.st_mode):
                if not _trusted_owner(info.st_uid) or _mode_writable(info.st_mode):
                    _fail("virtual environment contains an unsafe directory")
                entries.append({"kind": "directory", "mode": mode, "path": relative})
            elif stat.S_ISREG(info.st_mode):
                uv_lock = relative == ".lock" and info.st_size == 0
                if not _trusted_owner(info.st_uid) or (
                    _mode_writable(info.st_mode) and not uv_lock
                ):
                    _fail("virtual environment contains an unsafe file")
                entries.append(
                    {
                        "kind": "file",
                        "mode": mode,
                        "path": relative,
                        "sha256": _sha256_bytes(path.read_bytes()),
                        "size": info.st_size,
                    }
                )
            else:
                _fail("virtual environment contains an unsupported file type")
    encoded = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return _sha256_bytes(encoded), len(entries)


def _runtime_paths() -> tuple[Path, tuple[Path, ...], tuple[Path, ...]]:
    base = Path(sys.base_prefix).resolve(strict=True)
    if not base.is_absolute():
        _fail("Python base runtime path is not absolute")
    configured = sysconfig.get_paths(vars={"base": str(base), "platbase": str(base)})
    roots = tuple(
        sorted(
            {
                Path(configured[name]).resolve(strict=True)
                for name in ("stdlib", "platstdlib")
            },
            key=str,
        )
    )
    if not roots:
        _fail("Python base runtime has no versioned standard-library root")
    for root in roots:
        try:
            root.relative_to(base)
        except ValueError:
            _fail("Python standard-library root is outside the base runtime")
        if root.name != f"python{sys.version_info.major}.{sys.version_info.minor}":
            _fail("Python standard-library root is not version-specific")

    companion_candidates = [
        base / "Python",
        base / "pyvenv.cfg",
        *sorted(
            (base / "lib").glob(
                f"libpython{sys.version_info.major}.{sys.version_info.minor}*"
            )
        ),
    ]
    companions = tuple(path for path in companion_candidates if path.exists())
    return base, roots, companions


def _runtime_inventory() -> tuple[str, tuple[str, ...], str, int]:
    base, roots, companions = _runtime_paths()
    entries: list[dict[str, Any]] = []

    def add_tree(
        directory: Path, logical_directory: str, ancestors: frozenset[Path]
    ) -> None:
        resolved_directory = directory.resolve(strict=True)
        if resolved_directory in ancestors:
            _fail("Python runtime contains a symbolic-link directory cycle")
        try:
            resolved_directory.relative_to(base)
        except ValueError:
            _fail("Python runtime contains an external symbolic-link target")
        directory_info = resolved_directory.lstat()
        if (
            not stat.S_ISDIR(directory_info.st_mode)
            or not _trusted_owner(directory_info.st_uid)
            or _mode_writable(directory_info.st_mode)
        ):
            _fail("Python runtime contains an unsafe directory")
        next_ancestors = ancestors | {resolved_directory}
        for path in sorted(resolved_directory.iterdir(), key=lambda item: item.name):
            info = path.lstat()
            logical_path = f"{logical_directory}/{path.name}"
            mode = stat.S_IMODE(info.st_mode)
            if path.name in {"site-packages", "dist-packages"} and (
                stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
            ):
                continue
            if not _trusted_owner(info.st_uid):
                _fail("Python runtime contains an entry with an untrusted owner")
            if stat.S_ISLNK(info.st_mode):
                resolved = path.resolve(strict=True)
                try:
                    resolved.relative_to(base)
                except ValueError:
                    _fail("Python runtime contains an external symbolic link")
                target_info = resolved.lstat()
                if _mode_writable(target_info.st_mode):
                    _fail("Python runtime symbolic-link target is writable")
                entry: dict[str, Any] = {
                    "kind": "symlink",
                    "mode": mode,
                    "path": logical_path,
                    "target": os.readlink(path),
                }
                if stat.S_ISREG(target_info.st_mode):
                    entry["target_sha256"] = _sha256_bytes(resolved.read_bytes())
                    entry["target_size"] = target_info.st_size
                elif stat.S_ISDIR(target_info.st_mode):
                    entry["target_kind"] = "directory"
                else:
                    _fail("Python runtime symbolic link has an unsafe target")
                entries.append(entry)
                if stat.S_ISDIR(target_info.st_mode):
                    add_tree(
                        resolved,
                        f"{logical_path}/@target",
                        next_ancestors,
                    )
            elif stat.S_ISDIR(info.st_mode):
                if _mode_writable(info.st_mode):
                    _fail("Python runtime contains a writable directory")
                entries.append(
                    {"kind": "directory", "mode": mode, "path": logical_path}
                )
                add_tree(path, logical_path, next_ancestors)
            elif stat.S_ISREG(info.st_mode):
                if _mode_writable(info.st_mode):
                    _fail("Python runtime contains a writable file")
                entries.append(
                    {
                        "kind": "file",
                        "mode": mode,
                        "path": logical_path,
                        "sha256": _sha256_bytes(path.read_bytes()),
                        "size": info.st_size,
                    }
                )
            else:
                _fail("Python runtime contains an unsupported file type")

    for index, root in enumerate(roots):
        logical_root = f"stdlib-{index}"
        root_info = root.lstat()
        entries.append(
            {
                "kind": "directory",
                "mode": stat.S_IMODE(root_info.st_mode),
                "path": logical_root,
            }
        )
        add_tree(root, logical_root, frozenset())

    for path in companions:
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(base)
        except ValueError:
            _fail("Python runtime companion is outside the base runtime")
        info = path.lstat()
        resolved_info = resolved.lstat()
        if (
            not _trusted_owner(info.st_uid)
            or not _trusted_owner(resolved_info.st_uid)
            or _mode_writable(resolved_info.st_mode)
            or not stat.S_ISREG(resolved_info.st_mode)
        ):
            _fail("Python runtime companion is unsafe")
        entries.append(
            {
                "kind": "symlink" if stat.S_ISLNK(info.st_mode) else "file",
                "mode": stat.S_IMODE(info.st_mode),
                "path": f"companion/{path.relative_to(base).as_posix()}",
                "sha256": _sha256_bytes(resolved.read_bytes()),
                "size": resolved_info.st_size,
                **({"target": os.readlink(path)} if stat.S_ISLNK(info.st_mode) else {}),
            }
        )

    encoded = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return (
        str(base),
        tuple(str(root) for root in roots),
        _sha256_bytes(encoded),
        len(entries),
    )


def _site_packages(root: Path) -> Path:
    path = (
        root
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    if not path.is_dir() or path.is_symlink():
        _fail("dedicated environment site-packages directory is missing or unsafe")
    return path


def _record_digest(value: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, TypeError):
        _fail("wheel RECORD contains an invalid digest")


def _verify_wheel_install(wheel_bytes: bytes, root: Path, interpreter: Path) -> None:
    site_packages = _site_packages(root)
    try:
        with zipfile.ZipFile(io.BytesIO(wheel_bytes)) as archive:
            archive_file_list = [
                info.filename for info in archive.infolist() if not info.is_dir()
            ]
            archive_files = set(archive_file_list)
            if len(archive_file_list) != len(archive_files):
                _fail("wheel contains duplicate archive paths")
            record_names = [
                name
                for name in archive_files
                if name.endswith(".dist-info/RECORD")
                and PurePosixPath(name).name == "RECORD"
            ]
            if len(record_names) != 1:
                _fail("wheel must contain exactly one RECORD")
            rows = csv.reader(
                io.StringIO(archive.read(record_names[0]).decode("utf-8"))
            )
            bootstrap_payloads: list[bytes] = []
            checked_packages: set[str] = set()
            recorded_names: set[str] = set()
            for row in rows:
                if len(row) != 3:
                    _fail("wheel RECORD row has the wrong field count")
                name, digest, size_text = row
                if name in recorded_names:
                    _fail("wheel RECORD contains a duplicate path")
                recorded_names.add(name)
                pure = PurePosixPath(name)
                if pure.is_absolute() or ".." in pure.parts:
                    _fail("wheel RECORD contains an unsafe path")
                top_level = pure.parts[0]
                expected_dist_info = PurePosixPath(record_names[0]).parts[0]
                expected_data = expected_dist_info.removesuffix(".dist-info") + ".data"
                if top_level not in {
                    "llmtracefx",
                    "vllm_kv_truth",
                    expected_dist_info,
                    expected_data,
                }:
                    _fail("wheel contains an unexpected top-level path")
                if name == record_names[0]:
                    if digest or size_text:
                        _fail("wheel RECORD self-entry must be unhashed")
                    continue
                if not digest or not size_text:
                    _fail("wheel RECORD payload entries must have hashes and sizes")
                payload = archive.read(name)
                if len(payload) != int(size_text):
                    _fail("wheel payload does not match its RECORD size")
                algorithm, separator, encoded = digest.partition("=")
                if (
                    separator != "="
                    or algorithm != "sha256"
                    or hashlib.sha256(payload).digest() != _record_digest(encoded)
                ):
                    _fail("wheel payload does not match its RECORD digest")
                if pure.parts[0] in {"llmtracefx", "vllm_kv_truth"}:
                    installed = site_packages.joinpath(*pure.parts)
                    if (
                        _read_trusted_bytes(installed, "installed package file")
                        != payload
                    ):
                        _fail(
                            "installed package contents do not match the trusted wheel"
                        )
                    checked_packages.add(pure.parts[0])
                elif pure.parts[0].endswith(".dist-info"):
                    installed = site_packages.joinpath(*pure.parts)
                    if (
                        _read_trusted_bytes(
                            installed, "installed distribution metadata"
                        )
                        != payload
                    ):
                        _fail(
                            "installed distribution metadata does not match "
                            "the trusted wheel"
                        )
                if (
                    ".data" in pure.parts[0]
                    and len(pure.parts) >= 3
                    and pure.parts[1] == "scripts"
                    and pure.name == "run-vllm-kv-truth-clean-env.py"
                ):
                    bootstrap_payloads.append(payload)
            if recorded_names != archive_files:
                _fail("wheel contents do not exactly match RECORD")
    except (KeyError, OSError, UnicodeError, ValueError, zipfile.BadZipFile):
        _fail("wheel is malformed or unreadable")
    if checked_packages != {"llmtracefx", "vllm_kv_truth"}:
        _fail("wheel does not contain both required packages")
    if len(bootstrap_payloads) != 1:
        _fail("wheel does not contain exactly one clean bootstrap")
    installed_bootstrap = _read_trusted_bytes(
        root / "bin" / "run-vllm-kv-truth-clean-env.py",
        "installed bootstrap",
        executable=True,
    )
    wheel_bootstrap = bootstrap_payloads[0]
    if not wheel_bootstrap.startswith(b"#!python\n"):
        _fail("wheel bootstrap does not use the PEP 427 Python placeholder")
    wheel_body = wheel_bootstrap.split(b"\n", 1)[1]
    installed_lines = installed_bootstrap.splitlines(keepends=True)
    direct_shebang = f"#!{interpreter}\n".encode()
    trampoline_exec = f"'''exec' '{interpreter}' \"$0\" \"$@\"\n".encode()
    if installed_lines and installed_lines[0] == direct_shebang:
        installed_body = b"".join(installed_lines[1:])
    elif (
        len(installed_lines) >= 4
        and installed_lines[0] == b"#!/bin/sh\n"
        and installed_lines[1] == trampoline_exec
        and installed_lines[2] == b"' '''\n"
    ):
        installed_body = b"".join(installed_lines[3:])
    else:
        _fail("installed bootstrap has an invalid PEP 427 shebang rewrite")
    if installed_body != wheel_body:
        _fail("installed bootstrap source does not match the trusted wheel")


def _parse_options(argv: list[str]) -> tuple[str, dict[str, str]]:
    if not argv or argv[0] not in {"record-trust", "preflight", "run"}:
        _fail("usage error")
    command = argv[0]
    required = {
        "record-trust": {"--wheel", "--output"},
        "preflight": {
            "--wheel",
            "--trusted-manifest",
            "--trusted-manifest-sha256",
        },
        "run": {
            "--wheel",
            "--trusted-manifest",
            "--trusted-manifest-sha256",
            "--execution-config",
            "--authorization",
            "--output-dir",
        },
    }[command]
    values: dict[str, str] = {}
    tail = argv[1:]
    if len(tail) != 2 * len(required):
        _fail("usage error")
    for index in range(0, len(tail), 2):
        option = tail[index]
        if option not in required or option in values:
            _fail("usage error")
        values[option] = tail[index + 1]
    if set(values) != required:
        _fail("usage error")
    return command, values


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    _require_nonsymlink_parents(path, "trusted manifest output")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except OSError:
        _fail("trusted manifest output must be a new non-symlink file")
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _record_trust(values: dict[str, str], bootstrap: Path, interpreter: Path) -> int:
    wheel = _absolute_path(values["--wheel"], "wheel")
    output = _absolute_path(values["--output"], "trusted manifest output")
    wheel_bytes = _read_trusted_bytes(wheel, "wheel")
    bootstrap_bytes = _read_trusted_bytes(bootstrap, "bootstrap", executable=True)
    root = _venv_root(interpreter, bootstrap)
    if wheel.is_relative_to(root):
        _fail("wheel must be external to the environment")
    if output == wheel or output.is_relative_to(root):
        _fail("trusted manifest output must be outside the wheel and environment")
    _verify_wheel_install(wheel_bytes, root, interpreter)
    environment_sha256, entry_count = _inventory_environment(root)
    (
        runtime_base_prefix,
        runtime_roots,
        runtime_sha256,
        runtime_entry_count,
    ) = _runtime_inventory()
    target = _resolve_trusted_interpreter(interpreter)
    target_bytes = _read_trusted_bytes(
        target, "Python interpreter target", executable=True, allow_root=True
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "bootstrap_sha256": _sha256_bytes(bootstrap_bytes),
        "environment_entry_count": entry_count,
        "environment_sha256": environment_sha256,
        "python_path": str(interpreter),
        "python_runtime_base_prefix": runtime_base_prefix,
        "python_runtime_entry_count": runtime_entry_count,
        "python_runtime_roots": list(runtime_roots),
        "python_runtime_sha256": runtime_sha256,
        "python_target_sha256": _sha256_bytes(target_bytes),
        "wheel_path": str(wheel),
        "wheel_sha256": _sha256_bytes(wheel_bytes),
    }
    _write_manifest(output, payload)
    print("trusted launch manifest recorded")
    return 0


def _load_and_verify_trust(
    values: dict[str, str], bootstrap: Path, interpreter: Path
) -> tuple[Path, Path]:
    manifest_path = _absolute_path(values["--trusted-manifest"], "trusted manifest")
    expected_manifest_sha256 = values["--trusted-manifest-sha256"]
    if len(expected_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_manifest_sha256
    ):
        _fail("trusted manifest SHA-256 must be 64 lowercase hexadecimal characters")
    manifest_bytes = _read_trusted_bytes(
        manifest_path, "trusted manifest", private=True
    )
    if _sha256_bytes(manifest_bytes) != expected_manifest_sha256:
        _fail("trusted manifest does not match its externally recorded SHA-256")
    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeError, ValueError):
        _fail("trusted manifest is not valid JSON")
    expected_keys = {
        "schema_version",
        "bootstrap_sha256",
        "environment_entry_count",
        "environment_sha256",
        "python_path",
        "python_runtime_base_prefix",
        "python_runtime_entry_count",
        "python_runtime_roots",
        "python_runtime_sha256",
        "python_target_sha256",
        "wheel_path",
        "wheel_sha256",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        _fail("trusted manifest has the wrong schema")
    if manifest["schema_version"] != SCHEMA_VERSION:
        _fail("trusted manifest schema version is unsupported")
    if not all(
        isinstance(manifest[name], str)
        for name in expected_keys
        - {
            "schema_version",
            "environment_entry_count",
            "python_runtime_entry_count",
            "python_runtime_roots",
        }
    ) or not all(
        isinstance(manifest[name], int)
        for name in {"environment_entry_count", "python_runtime_entry_count"}
    ):
        _fail("trusted manifest has invalid field types")
    if not isinstance(manifest["python_runtime_roots"], list) or not all(
        isinstance(root_name, str) for root_name in manifest["python_runtime_roots"]
    ):
        _fail("trusted manifest has invalid field types")

    wheel = _absolute_path(values["--wheel"], "wheel")
    if (
        str(wheel) != manifest["wheel_path"]
        or str(interpreter) != manifest["python_path"]
    ):
        _fail("launch paths do not match the trusted manifest")
    bootstrap_bytes = _read_trusted_bytes(bootstrap, "bootstrap", executable=True)
    wheel_bytes = _read_trusted_bytes(wheel, "wheel")
    if _sha256_bytes(bootstrap_bytes) != manifest["bootstrap_sha256"]:
        _fail("bootstrap does not match the trusted manifest")
    if _sha256_bytes(wheel_bytes) != manifest["wheel_sha256"]:
        _fail("wheel does not match the trusted manifest")

    root = _venv_root(interpreter, bootstrap)
    if wheel.is_relative_to(root):
        _fail("wheel must be external to the environment")
    if manifest_path.is_relative_to(root):
        _fail("trusted manifest must be external to the environment")
    target = _resolve_trusted_interpreter(interpreter)
    target_bytes = _read_trusted_bytes(
        target, "Python interpreter target", executable=True, allow_root=True
    )
    if _sha256_bytes(target_bytes) != manifest["python_target_sha256"]:
        _fail("Python interpreter target does not match the trusted manifest")
    (
        runtime_base_prefix,
        runtime_roots,
        runtime_sha256,
        runtime_entry_count,
    ) = _runtime_inventory()
    if (
        runtime_base_prefix != manifest["python_runtime_base_prefix"]
        or list(runtime_roots) != manifest["python_runtime_roots"]
        or runtime_sha256 != manifest["python_runtime_sha256"]
        or runtime_entry_count != manifest["python_runtime_entry_count"]
    ):
        _fail("Python base runtime does not match the trusted manifest")
    _verify_wheel_install(wheel_bytes, root, interpreter)
    environment_sha256, entry_count = _inventory_environment(root)
    if (
        environment_sha256 != manifest["environment_sha256"]
        or entry_count != manifest["environment_entry_count"]
    ):
        _fail("installed environment does not match the trusted manifest")
    return root, wheel


def _require_private_input(value: str, label: str) -> Path:
    path = _absolute_path(value, label)
    descriptor = _open_trusted_file(path, label, private=True)
    os.close(descriptor)
    return path


def _require_output_directory(value: str) -> Path:
    path = _absolute_path(value, "output directory")
    _require_nonsymlink_parents(path / "sentinel", "output directory")
    try:
        info = path.lstat()
    except OSError:
        _fail("output directory must already exist")
    if not stat.S_ISDIR(info.st_mode) or path.is_symlink():
        _fail("output directory must be a non-symlink directory")
    if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
        _fail("output directory must be current-user-owned with mode 0700")
    if any(path.iterdir()):
        _fail("output directory must be empty")
    return path


def main(argv: list[str] | None = None) -> int:
    _require_clean_environment()
    _require_python_flags()
    command, values = _parse_options(list(sys.argv[1:] if argv is None else argv))
    bootstrap = _absolute_path(str(Path(__file__)), "bootstrap")
    interpreter = _absolute_path(sys.executable, "Python interpreter")
    if command == "record-trust":
        return _record_trust(values, bootstrap, interpreter)

    root, _wheel = _load_and_verify_trust(values, bootstrap, interpreter)
    cli_args: list[str]
    if command == "preflight":
        cli_args = ["preflight"]
    else:
        config = _require_private_input(
            values["--execution-config"], "execution config"
        )
        authorization = _require_private_input(
            values["--authorization"], "authorization"
        )
        output = _require_output_directory(values["--output-dir"])
        cli_args = [
            "run",
            "--execution-config",
            str(config),
            "--authorization",
            str(authorization),
            "--output-dir",
            str(output),
        ]

    site_packages = _site_packages(root)
    sys.path.insert(0, str(site_packages))
    from vllm_kv_truth.cli import bootstrap_dispatch

    return int(bootstrap_dispatch(cli_args))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BootstrapError as exc:
        print(f"clean bootstrap: error: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
