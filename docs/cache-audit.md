# KV-cache truth auditing

`llmtracefx-cache-audit` verifies cache claims as a chain of evidence. A runtime
counter by itself does not prove that the intended token prefix matched, that KV
state was reused, that prompt work was skipped, that latency or memory improved,
or that the output stayed correct.

For every request, the auditor reports:

- the exact semantic token prefix independently found in the recorded cache
  state;
- the tokens or blocks eligible under the pinned runtime's cache policy;
- the tokens or blocks the runtime attested as cached;
- the prompt tokens actually submitted or observed as processed;
- timing and memory values in their original measurement domains;
- output-token identity and correctness against a no-cache control;
- a fail-closed verdict and explicit limitations.

Missing measurements remain `null`. A cached-token count is never converted to
latency saved, and estimated saved tokens are never represented as observed
compute.

## Quick start

Create the deterministic synthetic workload and run the download-free reference
positive control:

```console
uv run llmtracefx-cache-audit compile --output workload.json
uv run llmtracefx-cache-audit run \
  --backend reference \
  --workload workload.json \
  --publication-mode public_synthetic \
  --output-dir cache-audit-bundle
uv run llmtracefx-cache-audit verify cache-audit-bundle
uv run llmtracefx-cache-audit report cache-audit-bundle
```

The reference backend tests the evidence pipeline; it is not evidence about MLX
or vLLM. Inspect real-backend readiness before attempting a run:

```console
uv run llmtracefx-cache-audit capabilities --backend mlx
uv run llmtracefx-cache-audit capabilities --backend vllm
```

The CLI is offline by default. Real MLX execution accepts an existing local
model path only. It does not resolve Hub IDs or download weights. The vLLM
adapter supports only offline capability/configuration and event parsing; it
does not execute requests.

```console
uv run llmtracefx-cache-audit run \
  --backend mlx \
  --model-path /already/approved/local/checkpoint \
  --model-id public-safe-model-label \
  --tokenizer-id public-safe-tokenizer-label \
  --output-dir private-mlx-cache-audit
```

The local path is used only to load the approved checkpoint and is never
persisted in the evidence manifest.

## Verdicts

| Verdict | Meaning |
|---|---|
| `verified_hit` | Independently expected policy reuse, engine attestation, and observed prompt work agree for a full semantic prefix. |
| `partial_reuse` | The same evidence agrees for a proper prefix shorter than the request. |
| `verified_miss` | Independent expectation, engine attestation, and observed work all show no reuse. |
| `attested_only` | The engine reports reuse but identity or prompt-work corroboration is unavailable. |
| `recomputed` | Prompt work overlaps attested reuse beyond the runtime's declared policy-required work. |
| `evicted` | Controlled state/events prove a previously resident candidate was removed before the miss. |
| `unsupported` | The backend or available instrumentation cannot support the requested claim. |
| `invalid` | Cache evidence is contradictory, identity is ambiguous, or execution failed. |

Timing, memory, cost, and output correctness are independent claim dimensions.
They do not promote or rewrite a cache verdict. Failed or missing output
equivalence makes output/performance claims ineligible without changing cache
truth.

## MLX-LM 0.31.3 semantics

The MLX adapter is pinned to `mlx-lm==0.31.3` and `mlx==0.32.2`. It uses exact
token arrays and the version-pinned `LRUPromptCache` instrumentation surface.
MLX-LM's in-memory server cache is token-granular, not block-hashed.

When a longer trimmable cache matches the complete request, MLX-LM trims it to
at most `len(request) - 1`; one prompt token remains to produce sampling logits.
The auditor records:

- the semantic common prefix;
- the policy-reusable token count;
- the engine's `len(prompt) - len(remainder)` attestation;
- the remainder observed by the prompt-progress hook.

That required sampling token is not classified as hidden recomputation.

MLX block-boundary, native cache-salt, and native multimodal-key claims are
unsupported. Rotating caches become non-trimmable after rotation, and rotating
cache quantization is unsupported by the pinned release. Those cases fail
closed rather than borrowing vLLM semantics.

Saved MLX prompt caches do not contain a cryptographic token-sequence or
model-weight binding. LLMTraceFX therefore requires its own sidecar binding the
exact token sequence, model artifacts, runtime versions, and cache payload.
An absent, stale, or mismatched sidecar is refused before generation.

## vLLM 0.28.0 semantics

The vLLM adapter is pinned to tag `v0.28.0`. Its offline oracle handles complete
hash units, final partial units, physical-group alignment, and identity inputs.
The runtime capability gate requires:

- `vllm==0.28.0`;
- automatic prefix caching explicitly enabled;
- `sha256_cbor`;
- a fixed `PYTHONHASHSEED`;
- KV-cache events enabled;
- `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0`, avoiding the default truncated
  external hash representation;
- valid hash and physical block sizes.
- a runtime-exported, version-bound attestation digest;
- full 256-bit `sha256_bytes` event-hash representation.

Caller or environment strings do not satisfy the runtime-attestation gate.
Arrival-to-first-token is reported as a TTFT-like duration; queue duration is
unavailable without a distinct scheduling timestamp.

Aggregate hit counters and sampled residency histograms are corroboration only.
They cannot prove a per-request hit. Preemption/recompute, hybrid group
internals, multimodal identity, and speculative decoding remain unavailable
unless an approved runtime run captures the required evidence.

## Evidence bundles

A bundle contains:

- `audit-manifest.json`
- `request-evidence.jsonl`
- `cache-events.jsonl` (empty when the backend exposes no events)
- `claim-matrix.json`
- `summary.json`
- `reuse-alignment.svg`
- `report.html`
- `SHA256SUMS`

The offline verifier checks the exact file allowlist, checksums, strict schemas,
request order, verdict predicates, derived claim matrix and summary, deterministic
HTML/SVG rendering, public privacy rules, and the digest of the generating
`llmtracefx.cache_audit` package. The portable wrapper searches the containing
repository or requires `--package-root`; it never imports whichever unrelated
`llmtracefx` happens to be installed.

Backend version, runtime identity, cache type, and cache limits are supplied by
the adapter. Caller-provided values are checked for agreement and are never
persisted as authoritative metadata. The MLX adapter performs its pinned-version
capability check for local-path loading, already-loaded models, and every saved
cache load.

The report's primary sentence is:

> The engine reported X cached tokens/blocks. Given the exact input and cache
> state, we independently expected Y. We observed Z prompt/timing/memory/output
> behavior. Therefore the claim is supported, unsupported, or attested-only.

## Privacy modes

`private` bundles may retain exact token arrays locally. They are not
catalog-eligible.

`public_synthetic` bundles may include exact arrays only for the built-in,
approved synthetic reference workload and identities, preserving independent
verification without allowing arbitrary token arrays to be published.

`public_redacted` bundles cannot be produced directly by `run`. First write and
verify a private bundle, then sanitize it:

```console
uv run llmtracefx-cache-audit run --backend reference --output-dir private-audit
uv run llmtracefx-cache-audit sanitize private-audit --output-dir public-audit
```

The redacted bundle removes exact input/output token arrays, replaces
request, pair, namespace, model, tokenizer, runtime, and limitation identifiers,
and downgrade identity-dependent verdicts to `attested_only` or `unsupported`.
They also exclude prompts, native cache hashes, cache tensors, salts,
credentials, host/account identities, and local paths. Sanitization is followed
by complete bundle verification.

## First article evidence gate

The first article may use only a verified public-synthetic bundle. Each
published request must include exact synthetic token arrays, deterministic
request order, independent reuse calculation, engine attestation, observed
prompt work, output comparison, limitations, and the generated claim sentence.

Latency, allocator memory, cost, or cross-runtime superiority are not
publishable from the reference control. A later MLX result may make a timing or
memory claim only when its own claim-matrix cell has compatible raw paired
samples. vLLM and MLX are never ranked as interchangeable cache
implementations.
