# Qwen3-8B M5 Pro self-conversion control (planned, preparatory)

**Status: no conversion or benchmark has run yet.** This directory ships the
offline framework -- manifests, the self-conversion runner, the bound-manifest
binder, the subprocess-isolated benchmark runner, and their tests -- for a
*future* self-converted Qwen3-8B positive control. It is not the pinned
Qwen3.8-27B lab in `../m5-pro-qwen3.8-27b/`, and unlike that lab, no model has
been downloaded, converted, or benchmarked here: the one real attempt on this
machine so far was refused before any download by the pre-conversion safety
gate (see `conversion-preflight-refusal-example.json` below). Execution
remains blocked pending a clean reboot and a passing safe preflight.

Once it does run, this control is intended to self-convert the official,
public, ungated `Qwen/Qwen3-8B` at `b968826d9c46dd6066d109eabc6255188de91218`
(Apache-2.0) with this repository's own pinned `mlx-lm==0.31.3`
(`ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd`), using explicit, recorded
quantization parameters: `quantize=true`, `q_group_size=64`, `q_bits=4`,
`q_mode=affine`. It is designed to never claim byte-equivalence with any
third-party conversion (e.g. `mlx-community`); a completed run
cryptographically binds its own source revision, converter revision, and
output file hashes instead. None of that has happened yet.

Every manifest, cache path, and CLI entrypoint here (`llmtracefx-m5-control`)
lives in its own `qwen3-8b` namespace, fully separate from the packaged 27B
lab's `llmtracefx-m5-lab`/`llmtracefx-m5-frontier` artifacts.

## Commands

```bash
# Default: offline plan only. No download or conversion.
make m5-control-plan

# Self-convert the official source once. Never retried automatically.
# Refuses before any download if the live safety gate is not satisfied.
make m5-control-convert

# Materialize a bound benchmark manifest from a completed conversion receipt.
make m5-control-bind

# Resume the subprocess-isolated benchmark, starting at 2K.
make m5-control-run

# Request gated escalation through 8K and 16K.
make m5-control-run CONTROL_MAX_TIER=16k

# Re-verify model/evidence bindings and render reports.
make m5-control-verify
make m5-control-report
```

Local weights, prompts, raw responses, and absolute-path-bearing artifacts
live under `.cache/llmtracefx/` and `.cache/models/` and are ignored by Git.
Only sanitized aggregate evidence and self-contained reports are suitable to
copy into this example directory -- and only once a real conversion and
benchmark run have actually produced them.

## How this is designed to differ from the 27B lab, once it runs

- **Self-converted, not pinned pre-quantized.** The 27B lab downloads an
  already-quantized `mlx-community` checkpoint byte-for-byte. This control is
  designed to download the official *unquantized* source and run the
  conversion itself, in a fresh, no-shell subprocess/process group with a
  bounded timeout and TERM->KILL cleanup escalation; it is never retried
  automatically, and a live safety gate (chip, memory, swap, free disk,
  installed converter version) must pass before any download or subprocess
  launch.
- **Requested vs. actual tokenizer counts.** The workload catalog's context
  tiers target an approximate, model-independent token count. This control's
  own mlx-lm chat template (`enable_thinking=false`) and tokenizer generally
  produce a different *actual* input token count than that *requested*
  target; every tier is designed to report both, never conflated.
- **Per-row subprocess isolation.** Every warmup and every measured
  repetition is designed to run in its own fresh subprocess and process
  group, with a parent-enforced wall-clock timeout independent of the
  collector's own cooperative in-process timeout.
- **Not comparable to the 27B lab.** Different model, different quantized
  checkpoint, different memory/timing envelope. Any future report from this
  control would only show completion under the host state actually observed
  during that run.

## What would be measured, once a run exists

Identical evidence surface to the 27B lab: host wall-clock phase timings,
requested/actual input and generated token counts, decode tokens/second,
MLX allocator active/cache/peak memory when exposed by the runtime,
deterministic structured-extraction and reasoning evaluator results, pass
rate, and correct cases per minute. Missing measurements would remain `null`.
No GPU utilization, bandwidth, power, energy, or kernel timing is measured or
inferred. None of this has been produced yet.

## Recorded refusal on this machine

`conversion-preflight-refusal-example.json` is a sanitized **refusal
artifact**, not benchmark evidence: it records that the live, conservative
pre-conversion safety gate refused with the host at 39.0% free memory
(40.0% required) before any download, conversion, or benchmark attempt.
No model weights were downloaded, no conversion subprocess was started, no
retry was attempted, and no existing cache was created, deleted, or
modified. `memory_free_percent`/`swap_used_bytes` are macOS
`memory_pressure`/`sysctl` host memory headroom, not free GPU memory. This
reflects only the host state observed at one point in time on one machine.

No `evidence-summary.json` or `report.html` exists in this directory: those
require an actual self-conversion and benchmark run on real M5 Pro hardware,
which has not happened. Populate them (and rewrite this section as recorded
*results*, not a refusal) only after `make m5-control-convert && make
m5-control-bind && make m5-control-run` actually complete.
