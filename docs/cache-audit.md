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
In pinned MLX-LM 0.31.3, `fetch_nearest_cache` reads and copies the selected
entry but does not update `CacheOrder`; eviction order changes only when
`insert_cache` removes/replaces and pushes an entry. The independent oracle and
synthetic control implement that same documented policy separately.
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
exact token sequence, namespace, model/tokenizer artifacts, runtime versions,
cache configuration, and cache payload.
The model directory and saved-cache payload are copied through verified file
descriptors into private immutable snapshots; the adapter hashes and loads those
same snapshot bytes. An absent, stale, mismatched, or concurrently modified
artifact is refused before generation.
The runtime cache key never contains the randomized snapshot pathname. It is a
stable digest of the verified model-and-tokenizer artifact set, pinned MLX/MLX-LM
versions, cache type, and cache limits. Reopening identical verified artifacts
therefore accepts a matching saved-cache sidecar, while changed artifacts,
runtime versions, or cache configuration refuse.

## vLLM 0.28.0 semantics

The vLLM adapter is pinned to tag `v0.28.0`. Its offline oracle handles complete
hash units, final partial units, physical-group alignment, and identity inputs.
The offline parser validates a proposed configuration against:

- `vllm==0.28.0`;
- automatic prefix caching explicitly enabled;
- `sha256_cbor`;
- a fixed `PYTHONHASHSEED`;
- KV-cache events enabled;
- `VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES=0`, avoiding the default truncated
  external hash representation;
- valid hash and physical block sizes.
- full 256-bit `sha256_bytes` event-hash representation.

No credible runtime-exported attestation artifact is available to this offline
adapter, so vLLM capabilities are unconditionally reported as unsupported.
Caller booleans, environment strings, digests, and configuration labels cannot
open a supported path. Event parsing requires canonical 256-bit SHA-256 hashes,
exact capture boundaries, monotonic contiguous sequences, and consistent block
metadata, but its output remains claim-ineligible without a future
runtime-exported artifact bound to installed binaries and engine configuration.
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
HTML/SVG rendering, public privacy rules, and the digest of every Python source
in the generating `llmtracefx` package. Before importing package code, the
portable wrapper verifies the bundle allowlist and checksums against embedded
commit/package trust anchors, snapshots the exact matching source bytes, and
imports only that snapshot. It never imports a bundle-local, ancestor, or
otherwise unrelated `llmtracefx` package.

Backend version, runtime identity, model artifact digest, cache type, and cache
limits are supplied by the adapter. Caller-provided values are checked for
agreement and are never persisted as authoritative metadata. The MLX adapter
performs its pinned-version capability check for local-path loading,
already-loaded models, and every saved-cache load.

An `evicted` verdict also persists an exact predecessor proof. The verifier
requires the predecessor to be an earlier request with the same backend, model
artifact, tokenizer, cache configuration, and namespace, and requires its exact
tokens to have produced reusable policy prefix state for the current request.
Cross-tenant, cross-model, configuration-mismatched, and unrelated predecessor
claims are invalid.

`created_at` records when the evidence request sequence was captured.
`generated_at` separately records when that evidence was bound to the generating
implementation and artifacts. `generator_commit_at` persists the bound commit
timestamp so repository, installed-package, and portable verification all
refuse a `generated_at` timestamp earlier than the generator commit. Catalog
`captured_at` remains the evidence capture time rather than the later
implementation-binding time.

Verification reports `repository_chronology_corroboration` as `verified` when
the exact generator commit is available and its timestamp and package tree
match. It reports `unavailable` for an installed package without Git metadata
or a shallow/partial checkout missing that object; the checksum-bound embedded
timestamps and exact package digest remain mandatory. An available but
conflicting Git object, timestamp, or tree always fails verification. Git
corroboration disables replacement objects and lazy network fetching, and only
an explicitly configured promisor remote can establish a partial checkout.

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
replaces every evidence/timing scope and timing exclusion with fixed public
constants, normalizes retained timing measurements to seconds, and downgrades
identity-dependent verdicts to `attested_only` or `unsupported`.
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
