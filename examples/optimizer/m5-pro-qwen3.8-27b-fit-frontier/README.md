# Qwen3.8-27B fit-frontier evidence

This directory contains only sanitized reports from the versioned
`m5-pro-qwen3.8-27b-fit-frontier-v1` lab. Raw prompts, model paths, responses,
process identifiers, and machine identifiers stay in the ignored workspace.

The committed `exploratory/` report is not publication evidence. In its
recorded machine state, the exact pinned checkpoint tokenized the first request
to 256 tokens and Metal reported insufficient memory before a first token was
observed. The 512, 1,024, 1,536, and 2,048 requested tiers were therefore
skipped. No successful bounded tier was established by this exploratory run.

After a clean boot, an operator can run the separately namespaced publication
sweep from a source checkout with:

```bash
make m5-frontier-publication \
  M5_FRONTIER_MODEL=/absolute/path/to/qwen3.8-27b-4bit-3e6447f
```

Passing `--confirm-clean-boot` through that target is an operator assertion.
The lab does not infer clean-boot status, and it still refuses publication mode
if the pinned hardware, runtime, memory-headroom, swap, disk, model-hash, or
artifact-integrity gates fail.
