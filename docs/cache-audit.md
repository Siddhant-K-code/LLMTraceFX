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

## Private real-MLX experiment

`llmtracefx-real-mlx-cache-audit` is the fail-closed Apple Silicon workflow.
It never downloads a model. The pinned artifact is the eight-file local
self-conversion of `Qwen/Qwen3-4B` revision
`1cfa9a7208912126459214e8b04321603b3df60c` (Apache-2.0), produced with
`mlx-lm==0.31.3` revision
`ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd` using affine 4-bit quantization
and group size 64. `compile` copies and re-verifies that contract in a private
temporary snapshot, loads only its tokenizer, and freezes both lanes:

- `1k`: 1025-token base and 513-token eviction prompts;
- `4k`: 4097-token base and 2049-token eviction prompts.

Each lane contains exactly seven cases: cold/exact/duplicate,
interior mutation at index 137, allocation-step mutation at index 256,
same-length different IDs, suffix-only change, namespace isolation, and
capacity eviction. The same-length different-ID case is an early-divergence
same-length case; it does not promise zero reuse. Index 256 is an allocation-step
boundary probe, not a block cache claim. The suffix case mutates tokens
`len-32:len-16` while preserving the final 16 generation-template tokens. No
directional shorter/longer prompt claim is made. Namespace isolation means
harness-enforced cache-key separation; it is not a claim that MLX provides
native tenant isolation.

`calibrate` runs each lane's eight distinct valid arrays (`base`,
`different_ids`, both mutation arrays, `suffix_change`, and eviction A/B/C)
once in separate fresh caches, plus a second fresh-cache base repeat. This is
nine adapters per lane and 18 total. Every result must contain exactly three
tokens, decode exactly to `CACHE_OK`, match its independent baseline, and the
two base results must be byte-for-byte stable. The deterministic
`calibration_outputs` mapping is frozen in each lane; calibration output is
never appended to a prompt.

After the calibrated workload digests have been reviewed and pinned, `run-all`
is the canonical execution path:

```console
uv run llmtracefx-real-mlx-cache-audit compile --model-dir MODEL --output workload.json
uv run llmtracefx-real-mlx-cache-audit calibrate --model-dir MODEL \
  --workload workload.json --output calibrated.json
uv run llmtracefx-real-mlx-cache-audit run-all --model-dir MODEL \
  --workload calibrated.json --expected-commit FULL_40_CHARACTER_GIT_SHA \
  --output-workspace real-mlx-run
```

`run-all` resolves all input and output paths before creating anything and
refuses missing, non-regular, or symlinked inputs, a mismatched Git HEAD,
tracked changes, package-source drift, top-level import-shadow candidates, an
existing output workspace, or an unavailable macOS network sandbox. Before
creating the workspace, it runs an isolated (`python -I`) model-free probe from
a resolved non-repository directory under the exact network-denied sandbox.
The probe verifies installed MLX, MLX-LM, NumPy, Tokenizers, Transformers, and safetensors
versions and complete package-tree identities, including the trusted install
root, import origin, regular-file count, total bytes, and deterministic tree
digest. Bytecode and mutable cache directories are excluded from the identity;
symlinks and other unsafe or missing package files are rejected. The probe
also verifies source/package identity, current process RSS, system swap, and
system memory pressure. It then applies one global machine gate. A failure returns
`NEEDS_CLEAN_BOOT:<reason>` without creating a ledger or consuming a replicate
ID. The same checks can be recorded without canonical execution by writing an
explicitly new receipt:

```console
uv run llmtracefx-real-mlx-cache-audit preflight --model-dir MODEL \
  --workload calibrated.json --expected-commit FULL_40_CHARACTER_GIT_SHA \
  --output-workspace real-mlx-run --output preflight.json
```

`run-all` repeats this gate rather than trusting a prior receipt. After it
passes, `run-all` creates a workspace containing exactly
`attempts/`, `private-artifacts/`, and `run-ledger.jsonl`. The append-only
ledger binds the expected commit, package-source digest, installed runtime
package-tree identities, calibrated workload and lane digests, model digest,
conversion-summary digest, exact sandbox-policy digest, and all six immutable
replicate IDs. Every replicate must follow one
canonical lifecycle: planned, preflight, optional started plus passed monitors,
postflight, then finalized. Complete replicates have exactly one started row
with a distinct privacy-safe child-instance digest. Failed-before-start and
failed-after-start transitions have separate schemas and reason rules. Every
finalized row binds the status, safe reason code, and deterministic digest of
its final attempt directory. Each child receives the expected commit and
revalidates the clean source/package tree and trusted runtime-package origins
before loading the model and immediately before finalizing evidence. The parent
repeats source validation before and after every child.

Each replicate loads the model once, then performs exactly one untimed,
discarded, direct runtime generation against a fresh cache per lane. Warm-ups
must reproduce the corresponding frozen calibration output, decode to
`CACHE_OK`, and emit no baseline, insertion, request, stage, or bundle
evidence; their caches and runtimes are torn down before measured blocks. Six
deterministic full 14-block permutations interleave the lanes,
spread the six first positions, and avoid fixing the interior/allocation-step
pair in one adjacent order. The supervisor launches each ID once in a fresh, sequential, network-denied
child process. Its environment is rebuilt from a minimal explicit allowlist:
a fixed system `PATH`, offline flags, and a fresh instance ID. Parent `HOME`,
`PYTHONPATH`, `DYLD*`, cloud/auth/token-file variables, and unknown variables
are never inherited. Preflight requires the Apple M5 Pro/24 GiB host contract,
at least 25% `vm_stat` availability, no more than 12 GiB swap, at least 20 GiB
free disk, and no non-excluded process using at least 1 GiB RSS. Process
observations retain only a count and the generic `other_large_process`
category, never names or PIDs. The two-second runtime monitor uses 15%, 14 GiB,
and 12 GiB limits respectively. Timing margin is bounded by a 12-minute
per-child deadline and a 90-minute total monotonic deadline; no performance
dry-run was used. The only preliminary child is the model-free sandbox probe.
Failed IDs retain bounded private logs and
partial artifacts in `private-artifacts/`, receive exactly one failed marker,
and are never replaced. Failed process-group cleanup or a surviving orphan
aborts all later launches while still finalizing every planned ID exactly once.
Five-of-six eligibility permits at most one failed replicate, and only when it
never started and its finalized reason is an explicit preflight machine-policy
refusal. A started failure, launch failure, timeout, source failure, or
`supervisor_aborted_before_start` cascade disqualifies the full run even when
five replicates completed; failed IDs are never replaced.
After the replicates finish, a terminal ledger row is appended even if public
results derivation or verification fails; that state is recorded as
`results_derivation_failed` with no results digest. The command exits nonzero
unless at least five of six replicates complete and public results derivation
succeeds.

Aggregation and sanitization remain explicit:

```console
PINNED_ENV/bin/llmtracefx-real-mlx-cache-audit aggregate \
  --run-workspace real-mlx-run --output-dir private-aggregate
PINNED_ENV/bin/llmtracefx-real-mlx-cache-audit verify private-aggregate
PINNED_ENV/bin/llmtracefx-real-mlx-cache-audit sanitize private-aggregate \
  --output-dir public-aggregate
PINNED_ENV/bin/llmtracefx-real-mlx-cache-audit verify public-aggregate
```

Aggregation accepts only the complete three-entry run workspace. It verifies
each standard source bundle, then reproduces nested copies through the
data-only bundle path without `evidence_bundle.py`; neither private nor public
aggregates may contain scripts or executable files. It never copies
`private-artifacts/`. Both aggregates retain every privacy-safe lifecycle row
in `run-ledger.jsonl` and contain a strict `results.json`; both files are
covered by the recursive `SHA256SUMS`.
`results.json` is derived from verified private records before nested bundle
redaction and preserves per-lane/case paired reuse, engine verdict, timing,
memory-level, output-count, identity, and correctness observations plus
descriptive medians and ranges. It contains no token arrays, prompts, paths,
host IDs, or secrets.

Real-MLX aggregate bundles deliberately contain no executable verifier.
Offline verification must invoke the independently installed, version-pinned
`llmtracefx-real-mlx-cache-audit verify` command. The verifier checks the
ledger against every attempt, reproduces `results.json` from private
aggregates, strictly validates public result schemas and bounds, and verifies
the result digest bound by the ledger and experiment contract. Unkeyed
`SHA256SUMS` provides integrity only, not authenticity. The Git commit that
contains the final public evidence is the external authenticity anchor; the
measurement commit and package-source digest remain visible in public
evidence even though standard public-redacted nested manifests omit the
generator commit.

Within each lifecycle, all cache-assisted requests finish before the independent
fresh-cache correctness baselines run. Reported client timing sums only the
runtime cache fetch and generation clocks. TTFT ends when the yielded response
token reaches the client iterator; process-wide synchronization is retained
only for total completion. Oracle work, stage collection, insertion, and
baseline generation are excluded. Paired latency comparability additionally
requires adjacent requests and equal generated output-token counts in both
arms. Eviction control/revisit pairs are always incomparable because pressure
requests intervene, while reuse and eviction facts remain reportable.
Allocator active/cache, process RSS, system swap, and system-memory pressure
remain explicitly scoped control/treatment level observations only and are not
causal deltas. Only the per-request-reset allocator peak has a paired
difference. The two-second monitoring and stage-observation subprocesses can
perturb host scheduling, but remain outside client timing clocks.

Article claims may use only a verified compatible claim-matrix cell and its raw
paired observations. A hit alone never proves saved work or latency; missing
facts remain null. Token index 256 is an MLX KV allocation-step boundary, not a
block-cache claim. Public tables must use the verified `results.json`, not
unavailable fields in generic redacted nested bundles, as proof of exact reuse
or output agreement.

The fixed interpretation limits are one observation per cell per replicate,
descriptive medians and ranges only, possible schedule/order and thermal
effects, non-causal allocator-active/cache, RSS, swap, and pressure levels,
monitoring/stage-observation scheduling perturbation, no block-cache
interpretation of allocation step 256, token-granular MLX cache behavior, and
no power, energy, kernel, or utilization claims. The constant-target
`CACHE_OK` identity/correctness check is a low-power guard and cannot rule out
all KV corruption. Evidence is scoped to one host, model, and conversion. The
output workspace must be outside every repository and free of top-level import
shadows for every runtime-tree package.
