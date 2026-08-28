# Fixture provenance

The `.log` files in this directory are **synthetic, representative**
llama.cpp text output modeled on the well-known `llama_perf_context_print`
timing format and the `n_draft` / `n_predict` / `n_drafted` / `n_accept`
counters used by llama.cpp's speculative-decoding examples.

They are **not** benchmark evidence. They were hand-written for this PR to
exercise the parser and the "doctor" comparability/regression logic
deterministically in CI, and do not represent measured performance of
Qwen3.8-27B, llama.cpp, or any specific hardware. Do not cite these
numbers as real benchmark results.

- `qwen3_8b_baseline_run1.log` / `qwen3_8b_baseline_run2.log`:
  synthetic autoregressive (no speculative decoding) runs.
- `qwen3_8b_mtp_regression_run1.log` / `qwen3_8b_mtp_regression_run2.log`:
  synthetic MTP/speculative-decoding runs that are slower than the
  baseline above, for exercising the regression-detection doctor rule.
- `qwen3_8b_mtp_improvement_run1.log` / `qwen3_8b_mtp_improvement_run2.log`:
  synthetic MTP/speculative-decoding runs that are faster than the
  baseline above, for exercising the improvement path.
- `malformed_load_time.log`: a deliberately corrupted line used to test
  that the parser surfaces malformed values explicitly instead of
  silently ignoring them.
