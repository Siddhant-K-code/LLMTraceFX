"""Create a small LLMTraceFX trace from real MLX work on Apple Silicon."""

import mlx.core as mx

from llmtracefx.profiler import MLXTraceRecorder

left = mx.random.uniform(shape=(1024, 1024))
right = mx.random.uniform(shape=(1024, 1024))
mx.eval(left, right)

with MLXTraceRecorder("mlx_trace.json") as trace:
    for token_id, token_text in enumerate(("Hello", "MLX")):
        with trace.token(token_id, token_text):
            result = trace.measure("matmul", mx.matmul, left, right)
            trace.measure("softmax", mx.softmax, result, axis=-1)

print("Wrote mlx_trace.json")
print("Analyze it with: llmtracefx --trace mlx_trace.json --gpu-type MLX --no-claude")
