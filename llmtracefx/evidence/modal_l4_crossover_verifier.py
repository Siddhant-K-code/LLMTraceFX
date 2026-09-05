"""Trusted catalog entry point for Modal L4 crossover bundles."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    from llmtracefx.optimizer.lab.qwen3_8b import modal_l4_crossover_evidence as module

    parser = argparse.ArgumentParser(allow_abbrev=False)
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action in ("verify-protocol", "verify-results"):
        sub = subparsers.add_parser(action, allow_abbrev=False)
        sub.add_argument("--bundle", required=True, type=Path)
    args = parser.parse_args(argv)
    repo_root = Path(module.__file__).resolve().parents[4]
    try:
        if args.action == "verify-protocol":
            module.verify_offline_bundle(args.bundle, repo_root=repo_root)
            print("Modal L4 crossover protocol verified")
        else:
            module.verify_result_bundle(args.bundle)
            print("Modal L4 crossover results verified")
    except (OSError, ValueError) as exc:
        print(f"Modal L4 crossover verification failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
