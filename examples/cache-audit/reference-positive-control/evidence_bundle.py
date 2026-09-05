"""Offline verifier bound to this bundle's generating package."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def package_digest(root: Path) -> str:
    package = root / "llmtracefx" / "cache_audit"
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return "sha256:" + digest.hexdigest()


def resolve_package(bundle: Path, requested: Path | None) -> Path:
    manifest = json.loads((bundle / "audit-manifest.json").read_text(encoding="utf-8"))
    expected = manifest.get("generator_package_digest")
    if not isinstance(expected, str):
        raise SystemExit("bundle has no generating-package digest")
    candidates = [requested.resolve()] if requested is not None else []
    candidates.extend((bundle.resolve(), *bundle.resolve().parents))
    for root in candidates:
        if (root / "llmtracefx" / "cache_audit" / "bundle.py").is_file():
            if package_digest(root) == expected:
                return root
    raise SystemExit(
        "matching llmtracefx source not found; pass --package-root for the "
        "repository or installed package recorded by this bundle"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--public-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--package-root", type=Path)
    args = parser.parse_args()
    root = resolve_package(args.public_dir, args.package_root)
    sys.path.insert(0, str(root))
    from llmtracefx.cache_audit.bundle import verify_bundle
    print(json.dumps({"verified": True, **verify_bundle(args.public_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
