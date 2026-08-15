"""Create a small LLMTraceFX trace from real MLX work on Apple Silicon."""

import argparse
from pathlib import Path

import mlx.core as mx

from llmtracefx.profiler import MLXTraceRecorder

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("mlx_trace.json"))
parser.add_argument("--metal-capture", type=Path)
args = parser.parse_args()

left = mx.random.uniform(shape=(1024, 1024))
right = mx.random.uniform(shape=(1024, 1024))
mx.eval(left, right)

with MLXTraceRecorder(args.output, metal_capture_path=args.metal_capture) as trace:
    for token_id, token_text in enumerate(("Hello", "MLX")):
        with trace.token(token_id, token_text):
            result = trace.measure("matmul", mx.matmul, left, right)
            trace.measure("softmax", mx.softmax, result, axis=-1)

print(f"Wrote {args.output}")
print(f"Analyze it with: llmtracefx --trace {args.output} --gpu-type MLX --no-claude")
