"""Verify the committed CloudRift vLLM compilation evidence."""

from pathlib import Path
import sys

from llmtracefx.optimizer.lab.qwen3_8b.cloudrift_compile_evidence import verify_bundle

if __name__ == "__main__":
    if sys.argv[1:] != ["verify"]:
        raise SystemExit("usage: evidence_bundle.py verify")
    verify_bundle(Path(__file__).resolve().parent)
    print("CloudRift vLLM compilation evidence verified")
