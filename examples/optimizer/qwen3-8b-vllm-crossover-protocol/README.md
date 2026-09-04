# Controlled Qwen3-8B vLLM crossover protocol

This is a verified offline protocol bundle, not benchmark evidence. It defines
two separate fresh-lifecycle lanes, eight eager/compiled pairs per lane,
counterbalanced ABBA/BAAB order, exact fixed-token-count and natural-output
workloads, whole-pair inference, a strict list-rate budget, and fail-closed
claim gates.

No CloudRift or Modal authentication occurred. No instance was created, model
downloaded, GPU used, or paid operation performed. All performance, crossover,
quality, runtime-component, and provider-spend claims remain unsupported until
a separately authorized complete execution bundle passes verification.

The controlled lane fixes 144 prompt requests and exactly 96 generated token
steps per request. This is fixed token count, not output control. Only observed
token-array identity can support an output-identical qualification. The
natural lane uses separate fresh lifecycles and gates end-to-end serving claims;
unequal outputs are never used for a causal speedup claim.

The independent analysis unit is a whole adjacent eager/compiled lifecycle
pair. Requests are repeated measures and are never bootstrapped independently.
The protocol reports first and sustained integer-request crossings, preserves
no-crossing outcomes as right-censored through request 144, and performs no
headline extrapolation.

Run `uv run --offline --no-sync python -I evidence_bundle.py verify` from this
directory in a clean checkout.
