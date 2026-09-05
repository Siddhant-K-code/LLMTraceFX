"""Offline verifier for this cache-audit bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmtracefx.cache_audit.bundle import verify_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify",))
    parser.add_argument("--public-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    print(json.dumps({"verified": True, **verify_bundle(args.public_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
