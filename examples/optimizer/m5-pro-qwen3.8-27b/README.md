# M5 Pro local inference lab

This lab measures useful, deterministic work on one pinned local system:
`mlx-community/Qwen3.8-27B-4bit` at
`3e6447f082e89cc7f0bc6e5441afd38dfce760ff`, using `mlx-vlm==0.6.8`.
The official upstream is `Qwen/Qwen3.8-27B` at
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`; both repositories declare
Apache-2.0.

The conversion is affine 4-bit with group size 64. The 15 pinned files total
16,081,490,933 bytes (14.98 GiB). That leaves limited but plausible runtime
headroom on a 24 GB M5 Pro at 2K-16K because only 16 of the language model's 64
layers use full attention. Fit is verified by measurement, not assumed:
execution starts at 2K and stops before 8K or 16K if the previous tier fails,
MLX peak memory exceeds 20 GiB, free-memory pressure drops below 25%, or swap
use exceeds 12 GiB.

## Commands

```bash
# Default: offline plan only. No model load or download.
make m5-lab

# Explicitly acquire and SHA-256 verify the public pinned snapshot.
make m5-lab-acquire

# Resume the bounded experiment, starting at 2K.
make m5-lab-run MAX_TIER=2k

# Request gated escalation through 8K and 16K.
make m5-lab-run MAX_TIER=16k

# Re-verify model/evidence bindings and render reports.
make m5-lab-verify
make m5-lab-report
```

Local weights, prompts, raw responses, and absolute-path-bearing artifacts live
under `.cache/llmtracefx/` and are ignored by Git. Only sanitized aggregate
evidence and the self-contained report are suitable to copy into this example
directory.

## What is measured

- host wall-clock total, prefill/time-to-first-token, and decode duration;
- tokenizer-reported input and generated token counts;
- decode tokens/second derived from measured tokens and decode time;
- MLX allocator peak memory when exposed by the runtime;
- deterministic structured-extraction and reasoning evaluator results;
- pass rate and correct cases per minute.

Missing measurements remain `null`. The lab does not infer GPU utilization,
bandwidth, power, energy, or kernel timing. The report applies only to the
pinned model, runtime, workloads, context tier, output cap, sampling settings,
and safety constraints; it does not claim a universally fastest setup.
The configured 1,800-second timeout is cooperative: it is checked after model
load, tokenization, and each streamed generation response, so it cannot
preempt a blocked native model load or prefill call.

## Recorded result on this machine

The pinned checkpoint passed file verification and loaded with
`mlx-vlm==0.6.8`, `mlx==0.32.2`, `mlx-lm==0.31.3`, and
`transformers==5.16.1`. A direct lazy reload of the exact checkpoint also
constructed `Qwen3VLProcessor` successfully. The 2K warmup prompt tokenized to
1,657 tokens. Metal then stopped during prefill after 22.731 seconds with:

```text
[METAL] Command buffer execution failed: Insufficient Memory
(00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).
```

No output token or evaluator result was produced, so prefill latency, decode
latency, token rate, allocator peak memory, pass rate, quality, and correct
cases per minute remain `null`. The safety gate did not run measured
repetitions and did not attempt 8K or 16K. This is evidence that this exact
27B/4-bit checkpoint and runtime did not fit the observed 24 GB environment,
not a claim that every 24 GB M5 Pro will always fail.

The safest next attempt is the same pinned experiment after a clean boot with
other GPU/memory-heavy applications closed. Any smaller model or different
quantization must use a separate, independently researched manifest and must
not be blended into this result.

## Provenance caveat

The MLX conversion card publishes the upstream model ID and converter version,
but not the exact source-model commit. The manifest therefore pins the produced
MLX snapshot and official upstream revision independently; it does not claim a
cryptographic link between them.
