"""Verify the committed CloudRift vLLM compilation evidence."""

import importlib
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

verify_bundle = importlib.import_module(
    "llmtracefx.optimizer.lab.qwen3_8b.cloudrift_compile_evidence"
).verify_bundle

if __name__ == "__main__":
    if sys.argv[1:] != ["verify"]:
        raise SystemExit("usage: evidence_bundle.py verify")
    verify_bundle(Path(__file__).resolve().parent)
    print("CloudRift vLLM compilation evidence verified")
