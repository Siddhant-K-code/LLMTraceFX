"""Run a closed evidence verifier with network sockets disabled."""

from __future__ import annotations

import importlib.abc
import os
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

_BLOCKED_IMPORT_ROOTS = {
    "ctypes",
    "huggingface_hub",
    "mlx",
    "mlx_lm",
    "modal",
    "multiprocessing",
    "safetensors",
    "tensorflow",
    "torch",
    "transformers",
}
_BLOCKED_AUDIT_EVENTS = {
    "os.exec",
    "os.fork",
    "os.forkpty",
    "os.link",
    "os.mkdir",
    "os.posix_spawn",
    "os.remove",
    "os.rename",
    "os.rmdir",
    "os.spawn",
    "os.symlink",
    "os.system",
    "os.truncate",
    "shutil.copyfile",
    "socket.__new__",
    "socket.bind",
    "socket.connect",
    "socket.getaddrinfo",
    "subprocess.Popen",
}


class _BlockedImportFinder(importlib.abc.MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None,
        target: object | None = None,
    ) -> None:
        del path, target
        if fullname.split(".", 1)[0] in _BLOCKED_IMPORT_ROOTS:
            raise RuntimeError(
                f"model/cloud module {fullname!r} is disabled during verification"
            )
        return None


def _contained(path: Path, roots: tuple[Path, ...]) -> bool:
    try:
        resolved = path.resolve(strict=False)
    except (OSError, RuntimeError):
        return False
    return any(resolved == root or root in resolved.parents for root in roots)


def _install_audit_guard(repo_root: Path) -> None:
    roots = tuple(
        path.resolve()
        for path in {
            repo_root,
            Path(sys.prefix),
            Path(sys.base_prefix),
        }
    )

    def audit(event: str, args: tuple[object, ...]) -> None:
        if event in _BLOCKED_AUDIT_EVENTS or event.startswith("socket."):
            raise RuntimeError(f"audit event {event!r} is disabled during verification")
        if event != "open" or not args:
            return
        raw_path = args[0]
        if isinstance(raw_path, int):
            return
        if not isinstance(raw_path, (str, bytes, os.PathLike)):
            raise RuntimeError("invalid file access during verification")
        mode = args[1] if len(args) > 1 else "r"
        flags = args[2] if len(args) > 2 else 0
        if isinstance(mode, str) and any(marker in mode for marker in "wax+"):
            raise RuntimeError("file writes are disabled during verification")
        write_flags = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
        if isinstance(flags, int) and flags & write_flags:
            raise RuntimeError("file writes are disabled during verification")
        try:
            candidate = Path(os.fsdecode(raw_path))
        except (TypeError, ValueError) as exc:
            raise RuntimeError("invalid file access during verification") from exc
        if not _contained(candidate, roots):
            raise RuntimeError("file access outside verified roots is disabled")

    sys.addaudithook(audit)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("offline verifier requires a repository root and script path")
    repo_root = Path(sys.argv[1]).resolve(strict=True)
    script = Path(sys.argv[2]).resolve(strict=True)
    if not _contained(script, (repo_root,)):
        raise SystemExit("offline verifier script must be inside the repository")
    sys.argv = [str(script), *sys.argv[3:]]
    sys.dont_write_bytecode = True
    sys.meta_path.insert(0, _BlockedImportFinder())
    _install_audit_guard(repo_root)
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
