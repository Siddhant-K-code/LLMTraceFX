# Qwen3-8B M5 Pro exploratory positive control

This bundle records one provenance-safe local self-conversion and a completed
exploratory control run on the same 24 GiB Apple M5 Pro used for the separate
Qwen3.8-27B investigation.

The source is the official, public Apache-2.0
`Qwen/Qwen3-8B@b968826d9c46dd6066d109eabc6255188de91218`. It was converted
exactly once with `mlx-lm==0.31.3` at
`ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd`, using affine 4-bit
quantization, group size 64. The 15-file source inventory totals
16,397,461,266 bytes. The 8-file converted output totals 4,619,328,159 bytes
and has binding fingerprint `df71c0372db25213fc9ee4efe23b3502ba6fc6d0`.
That fingerprint is a self-conversion identity, not a Git commit.

No byte-equivalence with any public or third-party MLX checkpoint is claimed.

## Evidence bundle

- [Checksums](SHA256SUMS)
- [Exact conversion provenance and inventories](conversion-summary.json)
- [Bound control manifest](control-manifest.json)
- [Evidence manifest](evidence-manifest.json)
- [Sanitized aggregate evidence](evidence-summary.json)
- [Self-contained report and charts](report.html)
- [Earlier preflight refusal](conversion-preflight-refusal-example.json)

Local model weights, prompts, responses, raw logs, receipts, and
absolute-path-bearing artifacts are not committed.

## Observed exploratory results

Each tier used one warmup followed by two measured repetitions for each of the
two deterministic safe workloads. Every warmup and measured repetition ran in
a fresh process group. Higher tiers ran only after the lower tier's evidence,
quality, cleanup, memory, swap, and disk gates passed.

| Requested tier | Mean actual input tokens | Passing runs | Pass rate | Mean total ms | Correct cases/min | Max MLX active bytes | Max MLX cache bytes | Max MLX peak bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2,048 | 1,613 | 4/4 | 1.00 | 2,595.61 | 23.12 | 4,608,043,016 | 277,713,890 | 5,605,114,236 |
| 8,192 | 6,373 | 4/4 | 1.00 | 4,977.31 | 12.05 | 4,608,043,016 | 1,944,972,034 | 6,201,600,040 |
| 16,384 | 12,697 | 4/4 | 1.00 | 10,287.57 | 5.83 | 4,608,043,016 | 1,895,709,294 | 7,107,602,472 |

Requested tiers are workload-catalog targets. Actual counts are tokenizer
observations and remain separate. Durations are host wall-clock boundaries.
Active, cache, and peak values are MLX allocator counters. Correct cases per
minute and decode token rates are derived only from measured counts and
wall-clock durations. Missing values remain null.

The final sweep began at 58% free memory and 4,003,788,226 swap bytes and ended
at 60% free memory and 4,555,800,576 swap bytes. These are macOS
`memory_pressure`/`sysctl` host observations, not free GPU memory.

## Comparison boundary

This is a **different model and system identity** from the pinned
Qwen3.8-27B 4-bit run. It demonstrates only that this self-converted Qwen3-8B
control completed through a requested 16K tier under the recorded host state.
It does not replace the 27B result and does not prove that the 27B OOM was
caused only by parameter count.

The run was exploratory: no clean-boot operator assertion was made. These
numbers are not publication-mode performance claims, not a universal capacity
boundary, and not a fastest-model claim. No utilization, bandwidth, power,
energy, kernel time, or free unified GPU memory is measured or inferred.

## Safety history

The earlier [refusal artifact](conversion-preflight-refusal-example.json)
records a separate attempt that stopped before download at 39% free memory
against the 40% conversion threshold. On 2026-09-02 the tool's own preflight
passed at 59% free memory, 1,967,065,661 swap bytes, and 730,086,916,096 free
disk bytes. The workflow then performed exactly one conversion attempt with no
automatic retry, cache deletion, alternate checkpoint, or hosted action.

## Reproduction commands

```bash
# No-download plan and safety preflight.
make m5-control-plan

# Exactly one gated conversion attempt.
make m5-control-convert

# Bind the verified output inventory and run gated tiers.
make m5-control-bind
make m5-control-run CONTROL_MAX_TIER=2k
make m5-control-verify
make m5-control-run CONTROL_MAX_TIER=8k
make m5-control-verify
make m5-control-run CONTROL_MAX_TIER=16k

# Verify and deterministically regenerate sanitized reports.
make m5-control-verify
make m5-control-report
```
