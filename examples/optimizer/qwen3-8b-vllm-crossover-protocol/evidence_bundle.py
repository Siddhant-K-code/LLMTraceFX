"""Verify the committed offline vLLM crossover protocol bundle."""

import importlib
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

verify_offline_bundle = importlib.import_module(
    "llmtracefx.optimizer.lab.qwen3_8b.cloudrift_crossover_evidence"
).verify_offline_bundle

if __name__ == "__main__":
    if sys.argv[1:] != ["verify"]:
        raise SystemExit("usage: evidence_bundle.py verify")
    verify_offline_bundle(Path(__file__).resolve().parent)
    print("Offline vLLM crossover protocol verified")
