# Modal GLM-5.3-Flash preflight refusal

This bundle records the no-spend preflight performed on 2026-09-02 for one
proposed Modal deployment of the official `zai-org/GLM-5.3-Flash` checkpoint.
The paid lifecycle did not run.

The plan was refused for three independent reasons:

1. The local Modal 1.5.4 CLI was not authenticated. The exact login step is
   `uv run modal setup`.
2. The complete conservative envelope was **$48.22**, above the explicitly
   authorized **$10.00** hard cap.
3. The digest-pinned dedicated vLLM image reports its embedded build revision
   as `unknown`, so the exact framework source revision could not be verified
   from primary image metadata.

The official model repository resolved to commit
`03eb5366286afd40d2221b1d9c63a6dd1ba4832e`: 72 files totaling
328,366,172,318 bytes (305.815 GiB), including 62 safetensors shards. The
repository publishes SHA-256 metadata for all 62 shards and `tokenizer.json`;
the nine remaining repository files do not carry published SHA-256 metadata.

No Modal app, secret, volume, container, endpoint, GPU, staging job,
verification job, readiness probe, or inference request was created or run.
No teardown was necessary, and estimated Modal credit use is $0.00.

Run the offline verifier from the repository root:

```bash
uv run --offline python \
  examples/optimizer/modal-glm53flash-preflight/evidence_bundle.py verify
```

`budget-plan.json` is the direct output of the merged deployment planner. Its
cost model is intentionally conservative: it prices staging and verification
to their configured six-hour timeouts, the one-hour deployment window plus a
last-started one-hour container lifetime, and four days of post-delete storage
at the published list rate. It does not assume the advertised included storage
allowance is unused because authentication was absent and account usage could
not be checked. The model is an operator planning gate, not a provider billing
cap.
