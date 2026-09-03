"""Command-line interface for the offline evidence catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core import CatalogError, generate_catalog_artifacts, verify_catalog


def _default_repo_root() -> Path:
    candidates = (Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents)
    for candidate in candidates:
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "examples").is_dir()
            and (candidate / "llmtracefx" / "evidence" / "registry.py").is_file()
        ):
            return candidate
    raise CatalogError(
        "repository root is unavailable; pass --repo-root or --catalog explicitly"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmtracefx-evidence",
        description="Index and verify committed public evidence without network or models",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("index", "graph"):
        command_parser = subparsers.add_parser(
            command,
            help=(
                "generate the canonical catalog and all graph views"
                if command == "index"
                else "verify and deterministically regenerate graph views"
            ),
        )
        command_parser.add_argument("--repo-root", type=Path, default=None)
        command_parser.add_argument("--output-dir", type=Path, default=None)
    verify_parser = subparsers.add_parser(
        "verify", help="verify the catalog, lineage, claims, and every bundle"
    )
    verify_parser.add_argument("--repo-root", type=Path, default=None)
    verify_parser.add_argument("--catalog", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    try:
        if args.command in {"index", "graph"}:
            root = (args.repo_root or _default_repo_root()).resolve()
            catalog = generate_catalog_artifacts(root, args.output_dir)
            result = {
                "generated": True,
                "catalog_hash": catalog["catalog_hash"],
                "entries": len(catalog["entries"]),
                "edges": len(catalog["edges"]),
                "unregistered_candidates": catalog["unregistered_candidates"],
            }
            if args.command == "graph":
                output_dir = args.output_dir or root / "examples" / "evidence-catalog"
                result.update(verify_catalog(Path(output_dir) / "catalog.json", root))
        else:
            if args.catalog is None:
                root = (args.repo_root or _default_repo_root()).resolve()
                catalog_path = root / "examples" / "evidence-catalog" / "catalog.json"
            else:
                catalog_path = args.catalog
                root = args.repo_root
            result = verify_catalog(catalog_path, root)
    except (CatalogError, OSError, ValueError) as exc:
        print(json.dumps({"verified": False, "error": str(exc)}, sort_keys=True))
        raise SystemExit(2) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(2 if result.get("unregistered_candidates") else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
