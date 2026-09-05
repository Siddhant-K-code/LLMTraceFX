"""Offline verifier bound to one audited source implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

EXPECTED_COMMIT = '3ee184486845bf9b30c37b9aaf7153d93f88a4be'
EXPECTED_PACKAGE_DIGEST = 'sha256:41eee925488778e1b81ee5f339ea35adad411857ec973eee94b5f36e81454a02'
BUNDLE_DATA_FILES = ('audit-manifest.json', 'request-evidence.jsonl', 'cache-events.jsonl', 'claim-matrix.json', 'summary.json', 'reuse-alignment.svg', 'report.html', 'evidence_bundle.py')
BUNDLE_FILES = BUNDLE_DATA_FILES + ("SHA256SUMS",)
MAX_FILE_BYTES = 67108864


def fail(message: str) -> None:
    raise SystemExit(message)


def safe_bytes(path: Path) -> bytes:
    if path.is_symlink():
        fail(f"unsafe symlink: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        fail(f"cannot open {path}: {exc}")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_FILE_BYTES:
            fail(f"unsafe or oversized file: {path}")
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65536, MAX_FILE_BYTES - total + 1))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_FILE_BYTES:
                fail(f"oversized file: {path}")
        after = os.fstat(descriptor)
        signature = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_size,
            value.st_mtime_ns,
        )
        if signature(before) != signature(after):
            fail(f"file changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def verify_bundle_envelope(bundle: Path) -> None:
    if bundle.is_symlink() or not bundle.is_dir():
        fail("bundle must be a regular directory")
    names = {item.name for item in bundle.iterdir()}
    if names != set(BUNDLE_FILES):
        fail("bundle file allowlist mismatch")
    checksum_lines = safe_bytes(bundle / "SHA256SUMS").decode("ascii").splitlines()
    checksums = {}
    for line in checksum_lines:
        parts = line.split("  ")
        if len(parts) != 2 or len(parts[0]) != 64 or any(
            character not in "0123456789abcdef" for character in parts[0]
        ):
            fail("invalid checksum manifest")
        digest, name = parts
        if name in checksums:
            fail("duplicate checksum entry")
        checksums[name] = digest
    if set(checksums) != set(BUNDLE_DATA_FILES):
        fail("checksum file allowlist mismatch")
    for name, expected in checksums.items():
        if hashlib.sha256(safe_bytes(bundle / name)).hexdigest() != expected:
            fail(f"checksum mismatch: {name}")
    try:
        manifest = json.loads(safe_bytes(bundle / "audit-manifest.json"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        fail(f"invalid audit manifest: {exc}")
    if manifest.get("schema_version") != "2":
        fail("unsupported audit schema")
    if manifest.get("generator_commit") != EXPECTED_COMMIT:
        fail("bundle generator commit does not match embedded trust anchor")
    if manifest.get("generator_package_digest") != EXPECTED_PACKAGE_DIGEST:
        fail("bundle package digest does not match embedded trust anchor")


def normalized_source(relative: str, content: bytes) -> bytes:
    if relative != "evidence/registry.py":
        return content
    import re

    text = content.decode("utf-8")
    text = re.sub(
        r'^_CACHE_AUDIT_SOURCE_COMMIT = (?:"[^"]*"|\(\n\s*"[^"]*"\n\))$',
        '_CACHE_AUDIT_SOURCE_COMMIT = "<bound-at-generation>"',
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r'^_CACHE_AUDIT_PACKAGE_DIGEST = (?:"[^"]*"|\(\n\s*"[^"]*"\n\))$',
        '_CACHE_AUDIT_PACKAGE_DIGEST = "<bound-at-generation>"',
        text,
        flags=re.MULTILINE,
    )
    return text.encode("utf-8")


def snapshot_package(root: Path) -> tuple[Path, str]:
    package = root / "llmtracefx"
    if package.is_symlink() or not package.is_dir():
        fail("candidate has no regular llmtracefx package")
    temporary = Path(tempfile.mkdtemp(prefix="llmtracefx-verifier-"))
    snapshot = temporary / "llmtracefx"
    snapshot.mkdir(mode=0o700)
    digest = hashlib.sha256()
    paths = sorted(package.rglob("*.py"))
    if not paths:
        fail("candidate llmtracefx package has no Python sources")
    for source in paths:
        relative_path = source.relative_to(package)
        for parent in (source, *source.parents):
            if parent == package.parent:
                break
            if parent.is_symlink():
                fail(f"candidate package contains symlink: {parent}")
        content = safe_bytes(source)
        digest_content = normalized_source(relative_path.as_posix(), content)
        relative = relative_path.as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(digest_content).to_bytes(8, "big"))
        digest.update(digest_content)
        destination = snapshot / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            os.write(descriptor, content)
        finally:
            os.close(descriptor)
    return temporary, "sha256:" + digest.hexdigest()


def commit_matches(root: Path) -> bool:
    if EXPECTED_COMMIT is None or not (root / ".git").exists():
        return True
    import subprocess

    listing = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-tree",
            "-r",
            "--name-only",
            EXPECTED_COMMIT,
            "--",
            "llmtracefx",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if listing.returncode != 0:
        return False
    digest = hashlib.sha256()
    paths = sorted(
        line.decode("utf-8").removeprefix("llmtracefx/")
        for line in listing.stdout.splitlines()
        if line.endswith(b".py")
    )
    for relative_text in paths:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "show",
                EXPECTED_COMMIT + ":llmtracefx/" + relative_text,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0:
            return False
        content = normalized_source(relative_text, result.stdout)
        relative = relative_text.encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return "sha256:" + digest.hexdigest() == EXPECTED_PACKAGE_DIGEST


def resolve_package(bundle: Path, requested: Path | None) -> Path:
    candidates = [requested.resolve()] if requested is not None else []
    if requested is None:
        candidates.extend((bundle.resolve(), *bundle.resolve().parents))
    for root in candidates:
        package = root / "llmtracefx"
        if package.is_symlink() or not package.is_dir():
            continue
        temporary, digest = snapshot_package(root)
        if digest == EXPECTED_PACKAGE_DIGEST and commit_matches(root):
            return temporary
        import shutil

        shutil.rmtree(temporary)
    fail(
        "matching llmtracefx source not found; pass --package-root for the "
        "repository or installed package recorded by this bundle"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--public-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--package-root", type=Path)
    args = parser.parse_args()
    bundle = args.public_dir.resolve()
    verify_bundle_envelope(bundle)
    snapshot_root = resolve_package(bundle, args.package_root)
    sys.path.insert(0, str(snapshot_root))
    from llmtracefx.cache_audit.bundle import verify_bundle

    print(json.dumps({"verified": True, **verify_bundle(bundle)}, sort_keys=True))


if __name__ == "__main__":
    main()
