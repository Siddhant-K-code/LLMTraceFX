# Qwen3.8-27B clean-boot OOM autopsy

This namespace now contains real, sanitized publication evidence from the
separately namespaced `m5-pro-qwen3.8-27b-oom-autopsy-v1` run. The implementation
and contract existed previously, but that implementation-only state contained no
real publication checkpoints. The files under [`publication/`](publication/)
are the first clean-boot publication bundle and do not alter or reinterpret the
earlier exploratory or PR #51/#52 evidence.

The run used code checkout
`2519bc8da309656d2e2ce2a7063f19b0dfb4c9ed`, completed at
`2026-09-01T17:45:36.921331Z`, and recorded the operator's explicit clean-boot
assertion. It used publication mode and the exact 15-file,
16,081,490,933-byte (14.977056 GiB) checkpoint
`mlx-community/Qwen3.8-27B-4bit@3e6447f082e89cc7f0bc6e5441afd38dfce760ff`.

## Public bundle

- [`autopsy-plan.json`](publication/autopsy-plan.json) - reviewed no-load plan;
  `weights_loaded=false`, `downloads_performed=false`,
  `publication_ready=true`, `model_present_by_size=true`, and no safety blockers.
- [`oom-autopsy-summary.json`](publication/oom-autopsy-summary.json) - sanitized
  report data and explicit measurement provenance.
- [`oom-autopsy-checkpoints.csv`](publication/oom-autopsy-checkpoints.csv) -
  checkpoint rows for independent analysis.
- [`oom-autopsy-report.html`](publication/oom-autopsy-report.html) -
  self-contained source report with no external resources.
- [`mlx-memory-by-stage.svg`](publication/mlx-memory-by-stage.svg) - deterministic
  chart generated only from the public summary JSON. MLX allocator, process RSS,
  and system swap use separate panels and independent axes.
- [`evidence-manifest.json`](publication/evidence-manifest.json) - run identity,
  source hashes, outcome, scope contract, and limitations.
- [`SHA256SUMS`](publication/SHA256SUMS) - complete content hashes for the public
  bundle.

The generator and fail-closed verifier are
[`evidence_bundle.py`](evidence_bundle.py). They do not load a model:

```bash
make m5-autopsy-evidence-verify
uv run --offline python \
  examples/optimizer/m5-pro-qwen3.8-27b-oom-autopsy/evidence_bundle.py generate
git diff --exit-code -- \
  examples/optimizer/m5-pro-qwen3.8-27b-oom-autopsy/publication
```

## Recorded state and outcome

The no-load plan observed 82% free memory pressure, approximately
21,131,239,096 bytes (19.680000 GiB) of system headroom, and 0 bytes of swap in
use. The run preflight observed 79% free, approximately 20,358,144,983 bytes
(18.960000 GiB) of system headroom, and 0 bytes of swap in use. These available
memory values are estimates derived from macOS `memory_pressure`; they are
approximate system headroom, not GPU memory.

The host was Apple M5 Pro arm64 with 25,769,803,776 physical bytes (24 GiB),
Darwin 25.6.0, MLX 0.32.2, mlx-lm 0.31.3, mlx-vlm 0.6.8, and transformers
5.16.1. The exact `t256` workload requested and actually tokenized 256 prompt
tokens. The process ended OOM because `MLX/Metal reported insufficient memory`;
the child exit code was 2, it did not time out, descendants were cleaned, and the
terminal journal was complete. No first token, generation completion, evaluator,
quality, or throughput measurement exists for this attempt.

The complete observed stage order was:

`child_start -> before_model_load -> after_model_load ->
after_prompt_tokenization -> immediately_before_prefill_generation ->
caught_oom -> cleanup`

Authoritative selected allocator values are:

| Stage | MLX active bytes | MLX cache bytes | MLX peak bytes |
|---|---:|---:|---:|
| after model load | 16,055,717,352 (14.953052 GiB) | 18,548 (0.000017 GiB) | 16,055,717,352 (14.953052 GiB) |
| immediately before prefill | 16,055,717,352 (14.953052 GiB) | 18,556 (0.000017 GiB) | 16,055,717,360 (14.953052 GiB) |
| caught OOM | 18,727,905,294 (17.441721 GiB) | 76,438,712 (0.071189 GiB) | 18,894,739,574 (17.597098 GiB) |
| cleanup | 18,341,875,182 (17.082202 GiB) | 462,468,824 (0.430708 GiB) | 18,894,739,574 (17.597098 GiB) |

From the immediately-before-prefill checkpoint to the caught-OOM checkpoint,
the observed deltas were active +2,672,187,942 bytes (2.488669 GiB), cache
+76,420,156 bytes (0.071172 GiB), and peak +2,839,022,214 bytes
(2.644045 GiB). They are checkpoint-to-checkpoint differences, not causal
allocation attribution.

At the prefill boundary, process RSS was 929,824,768 bytes, process max RSS was
1,908,129,792 bytes, and system swap used was 5,923,531,653 of
6,442,450,944 bytes. At caught OOM, those values were 527,089,664 bytes,
1,908,129,792 bytes, and 4,070,372,802 of 5,368,709,120 bytes. Swap grew after
model load and changed dynamically; these are only the exact stage observations.

## Scope and limitations

MLX active/cache/peak values are MLX allocator counters. Current/max RSS is host
process scope. Swap is host system scope. They are not additive, must not be
summed, and none is labeled as free GPU memory or GPU capacity. Bytes are
authoritative; GiB values are only binary-unit conversions (`bytes / 2^30`).

This is bounded evidence for one recorded machine state, exact checkpoint,
runtime, and `t256` workload. It is not a universal memory-capacity or 24 GB
boundary. It measures no GPU utilization, free GPU memory, bandwidth, power,
energy, occupancy, or kernel time. The stage probes add observer overhead that
was not separately measured or subtracted, and periodic sampling was disabled.
