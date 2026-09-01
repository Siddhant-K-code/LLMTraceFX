# Qwen3.8-27B OOM autopsy

This directory is reserved for sanitized reports from the separately
namespaced `m5-pro-qwen3.8-27b-oom-autopsy-v1` autopsy. It binds by SHA-256
and identity to the packaged `m5-pro-qwen3.8-27b-fit-frontier-v1` manifest and
runs only the exact pinned `t256` tier (256 requested tokens, 48 max output
tokens) that the fit-frontier exploratory evidence in
`../m5-pro-qwen3.8-27b-fit-frontier/exploratory/` already showed failing with
insufficient memory before a first token. It never modifies or reinterprets
that existing #51/#52 evidence.

The no-load exploratory plan passed the current host safety gates, but the
pinned checkpoint was not present in this worktree's cache. The model run was
therefore refused without a download, and this change contains no invented
checkpoint values or synthetic stand-in. This README documents the evidence
contract a later real run must satisfy; `oom-autopsy-summary.json`,
`oom-autopsy-checkpoints.csv`, and `oom-autopsy-report.html` land here only once
an operator runs the autopsy on the pinned hardware and copies the sanitized
output over.

## What the autopsy adds beyond the fit frontier

The fit frontier records only pass/fail per tier. The autopsy instruments the
exact same `t256` request with privacy-safe *stage checkpoints* (no periodic
sampling) so a failure can be read as a sequence of allocator/process/system
observations instead of a single opaque error:

`child_start -> before_model_load -> after_model_load ->
after_prompt_tokenization -> immediately_before_prefill_generation ->
[after_first_token, if reached] -> completion or caught_oom -> cleanup`

Each checkpoint records three independent scopes, each with explicit API
provenance, and `null` (never `0` or an estimate) when a probe is unavailable
or errors:

- **MLX allocator** - `mlx.core.get_active_memory` / `get_cache_memory` /
  `get_peak_memory`, probed as callables on the installed MLX build rather
  than assumed present.
- **Host process** - current and max RSS in bytes, normalized from
  platform-specific units and tools (macOS `ps -o rss=` in KiB plus
  `getrusage().ru_maxrss` in bytes; Linux `/proc/self/status` `VmRSS` in kB
  plus `getrusage().ru_maxrss` in KiB).
- **Host system** - swap total/used in bytes (macOS `sysctl vm.swapusage`;
  Linux `/proc/meminfo` `SwapTotal`/`SwapFree`).

MLX allocator scope is not GPU capacity or free memory, and host process RSS
is not GPU memory. No PIDs, paths, hostnames, usernames, prompts, or response
text ever appear in the journal or the report.

## Commands

```bash
# Default: offline plan only. Probes MLX counter APIs, never constructs a
# runtime or loads weights.
make m5-autopsy

# Resume (or start) the exploratory, process-isolated t256 autopsy. Refuses
# unless the pinned model is already cached and verified, and unless the
# existing safety gates pass. There is no acquire/download surface here; use
# `make m5-lab-acquire` or `make m5-frontier` first.
make m5-autopsy-run

# After an operator-confirmed clean boot only:
make m5-autopsy-publication
```

Passing `--confirm-clean-boot` (wired through `m5-autopsy-publication`) is an
operator assertion; the autopsy never infers clean-boot status, and
exploratory mode refuses the flag outright. Raw journals, per-attempt working
directories, and any path- or host-bearing artifacts stay under
`.cache/llmtracefx/m5-pro-qwen3.8-27b-oom-autopsy-v1/` and are ignored by Git.
Only the sanitized `oom-autopsy-summary.json` / `oom-autopsy-checkpoints.csv` /
`oom-autopsy-report.html` produced by `llmtracefx-m5-lab autopsy report` are
suitable to copy into this example directory.

## Strict limitations that carry into every report

- No unified-memory free-space or GPU capacity precision; process RSS is not
  GPU memory.
- No GPU utilization, free GPU memory, memory bandwidth, power, energy, or
  kernel time.
- Stage deltas are observed checkpoint-to-checkpoint changes, not a causal
  explanation of what allocated the memory in between.
- Not evidence of a universal 24 GB boundary; this is one pinned model,
  runtime, and request on one machine.
- A committed report's `synthetic` field is `false` only for real evidence
  from an actual run; synthetic reports (used only in tests) are always
  `synthetic: true` and are never placed in this directory.
- Observer overhead is limited to the stage-boundary probes themselves and is
  not separately measured or subtracted.
