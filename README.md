<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/llmtracefx-lockup-inverse.svg">
    <img src="assets/brand/llmtracefx-lockup.svg" alt="LLMTraceFX" width="309">
  </picture>
</p>

<p align="center">
  <strong>Measure what happened. Verify that it was correct. Optimize only what the evidence supports.</strong>
</p>

<p align="center">
  <a href="#quickstart">Quickstart</a>
  ·
  <a href="#current-capabilities">Capabilities</a>
  ·
  <a href="SELF_HOST_GLM_RUNBOOK.md">Modal runbook</a>
  ·
  <a href="https://siddhantkhare.com/writing/ttft-http-client">Methods example</a>
  ·
  <a href="DESIGN.md">Design system</a>
</p>

<p align="center">
  <img src="assets/brand/social-preview.png" alt="LLMTraceFX evidence and optimization report preview using synthetic example data" width="820">
</p>

<p align="center"><sub>The report values in this preview are synthetic interface examples, not benchmark results.</sub></p>

LLMTraceFX is an evidence-first inference toolkit for local models and
OpenAI-compatible streaming APIs. It collects measurements into one canonical
schema, checks model output with deterministic workloads, and recommends a
configuration only when it satisfies an explicit policy.

The main workflow is:

1. **Measure** with a collector or import an existing runtime artifact.
2. **Verify** the response against a pinned workload and retain failed or
   incomplete evidence.
3. **Compare and tune** candidates under one objective and stated constraints.
4. **Optimize** only after the collected evidence supports a recommendation.

## Quickstart

The LLMTraceFX commands in this path use no API key, model download, network
request, or accelerator. Cloning the repository and installing uncached
dependencies can use the network. The commands then audit the environment,
materialize a deterministic workload plan, and write an inspectable API request
plan.

```bash
git clone https://github.com/Siddhant-K-code/LLMTraceFX.git
cd LLMTraceFX
uv sync --locked

mkdir -p output/quickstart

uv run llmtracefx-optimizer manifest \
  --output output/quickstart/environment.json

uv run llmtracefx-optimizer workloads generate-matrix \
  --model-id example/local-model \
  --model-family qwen3_next \
  --context-tiers 2k \
  --max-tokens 16 \
  --output-dir output/quickstart/matrix

env -u LLMTRACEFX_QUICKSTART_NO_KEY uv run llmtracefx-optimizer collect-api \
  --run-id api-plan \
  --provider example \
  --endpoint https://example.com/v1/chat/completions \
  --model-id example-model \
  --prompt-file examples/optimizer/api-smoke-prompt.txt \
  --output-dir output/quickstart/api-plan \
  --api-key-env LLMTRACEFX_QUICKSTART_NO_KEY \
  --dry-run
```

The last command writes `output/quickstart/api-plan/request_plan.json`. Its
`network_request_performed` field is `false`, and the dry run does not read an
API credential because the named variable is explicitly unset. In general, a
dry run does not require or transmit a credential. If its named variable is
already set, it reads the value only to detect unsafe embedding and redact the
plan.

Two more offline commands describe the optional Modal deployment harness:

```bash
uv run llmtracefx-deploy recipe
uv run llmtracefx-deploy budget --credit-usd 30
```

Neither command imports Modal, authenticates, creates resources, or performs a
network request.

## Current capabilities

### Canonical evidence

Inference collectors and the llama.cpp importer produce `ExperimentRecord`
schema version 1. Instruments collection writes a separate, typed evidence
record for trace capabilities and supported Metal tables. Numeric measurements
carry a unit and a provenance:

- `measured_native`
- `measured_wall_clock`
- `provider_reported`
- `derived`
- `estimated`

Unavailable observations remain absent. They are not converted to zeros or
relabelled as measurements.

### Collection

- **MLX and MLX-LM:** run an existing local model directory on Apple silicon
  and record normalized timing and memory evidence. LLMTraceFX does not
  download the model.
- **OpenAI-compatible streaming APIs:** record client-observed headers,
  first-byte timing, first visible content, inter-event timing, completion
  state, and provider-reported usage without persisting the credential.
- **llama.cpp:** convert captured stdout and stderr into the canonical evidence
  schema. This parser does not launch or configure llama.cpp.
- **Apple Instruments:** check `xctrace` capability, print an execution plan,
  record a local command, or import an existing trace bundle. Supported Metal
  tables are reported narrowly rather than treated as generic GPU utilization.
- **Native Qwen MTP:** emit a capability report and an explicit unsupported
  record when the installed stack cannot produce trustworthy native-MTP
  evidence.

### Verification and resumability

The workload catalog covers code completion, structured JSON, and prose
reasoning across pinned context tiers. `workloads generate-matrix` writes the
prompts, hashes, runner configs, and planned commands without loading a model.

`workloads run` and `workloads run-api` verify that the prompt, workload
version, run binding, and artifact hashes still match. A complete hash-matching
run is resumed by default. Failed, partial, mismatched, and unsupported rows
remain visible instead of being scored as successes.

### Controlled vLLM crossover protocol

`llmtracefx-vllm-crossover plan` renders the preregistered Qwen3-8B vLLM
compilation crossover offline. It defines separate fixed-token-count and
natural-output lanes, eight fresh eager/compiled lifecycle pairs per lane,
counterbalanced order, whole-pair uncertainty, and a strict list-rate budget.

```bash
uv run --offline --no-sync llmtracefx-vllm-crossover plan
make vllm-crossover-verify
```

These commands do not authenticate, contact CloudRift or Modal, download a
model, use a GPU, or authorize spend. The controlled lane fixes decode-step
count, not output identity; unequal output token arrays are never described as
output-controlled. A paid `run` requires a separate exact-plan authorization
receipt and controls Docker only on an already provisioned local host. It
contains no provider or SSH client. Any temporary public-key access is managed
out of band; passwords, private keys, API tokens, host addresses, usernames,
ports, and provider credentials are not accepted. Authorization is
content-hashed, bound to the exact resolved workspace, and authenticated with
an OpenSSH detached signature against an operator-managed authorized-signers
file. It also binds the pinned image, billing start, zero-retry rule, and
external shutdown deadline. Every Docker command targets only
`unix:///var/run/docker.sock`; Docker/SSH routing environment variables are
rejected and host subprocesses receive a fixed minimal environment.

Only a fully completed 32-cell workspace can be published with
`llmtracefx-vllm-crossover-results build --workspace ... --output ...`. The
builder revalidates lifecycle, hardware, prompt, output, correctness, budget,
and teardown evidence; it resamples whole lifecycle pairs and preserves
unobservable request/compile fields as explicit nulls.

### Tune within one target

`tune` reads existing `verification.json` and `final_record.json` files. It
does not load a model or execute a benchmark. A policy chooses one objective
and can constrain pass rate, quality metric, peak memory, latency, provenance,
repetition count, and coefficient of variation.

```bash
uv run llmtracefx-optimizer tune \
  --results output/results \
  --policy examples/optimizer/tune-policy-fastest-under-20gb-m5-pro.json \
  --output output/tune-report.json \
  --explain

uv run llmtracefx-optimizer tune-report \
  --input output/tune-report.json \
  --output output/tune-report.html
```

The HTML report is a deterministic, self-contained view with inline CSS, no
JavaScript, and no CDN. Local paths are redacted unless `--include-paths` is
set.

### Cross-system comparison

`tune` compares configurations within one model and hardware target. `compare`
works across already-collected local and hosted systems. It is offline: it
loads no model, calls no API, deploys nothing, and runs no benchmark.

The command accepts only result directories shaped by `workloads run` or
`workloads run-api`: each row must have `verification.json` and the referenced
`final_record.json`. API rows also validate the collector's
`api_evidence.json` and completion marker. Flat `collect-api` output is not
accepted because it has no workload verification or quality result.

Use separate result directories for repetitions. Reusing one directory resumes
the completed row instead of measuring it again. Pass every repetition to
`--results`; duplicate paths count once.

```bash
uv run llmtracefx-optimizer compare \
  --results \
    artifacts/local/rep-1 artifacts/local/rep-2 \
    artifacts/frontier-api/rep-1 artifacts/frontier-api/rep-2 \
    artifacts/flash-api/rep-1 artifacts/flash-api/rep-2 \
  --policy examples/optimizer/compare-policy-local-vs-api-latency.json \
  --output artifacts/cross-system-compare.json \
  --explain

uv run llmtracefx-optimizer compare-report \
  --input artifacts/cross-system-compare.json \
  --output artifacts/cross-system-compare.html
```

Systems are compared only within a comparable stratum: identical workload and
version, prompt hash, context tier, evaluator, output cap, sampling, and request
shape. Different or unknown settings remain separate. System identity retains
model and revision, provider, runtime and backend, accelerator, quantization,
reasoning settings, endpoint route, decode mode, and collection configuration.

The report can carry pass rate, quality, total latency, client or local
first-token timing, correct cases per minute, provider-reported usage, and
estimated cost metrics where those values exist. It does not:

- rank local prefill timing against hosted client-observed time to first visible
  content;
- invent hosted peak memory or replace missing evidence with zero;
- combine objectives into a blended score;
- choose a winner when the evidence is tied, within noise, or limited to one
  system.

Cost objectives require `--pricing` with a versioned manifest. No rates are
built in or fetched. Every monetary result is estimated from provider-reported
usage and the supplied rate entry. The file
`examples/optimizer/pricing-manifest-illustrative.json` contains invented
demonstration values and must not be used as a current price list.

`compare-report` renders a deterministic, self-contained HTML file. Paths and
endpoint hosts are redacted by default; `--include-paths` is explicit opt-in.
Prompt text, response text, reasoning content, and credentials are not fields
in the comparison schema.

See the synthetic, non-benchmark
`examples/optimizer/compare-report-example.json` and the example policies
`compare-policy-local-vs-api-latency.json` and
`compare-policy-cost-per-correct-case.json`.

### Optimize an approved path

`optimize` composes matrix execution, tuning, and optional HTML rendering. Its
`--dry-run` path lists selected rows, blockers, and expected artifacts without
loading a model or tuning:

```bash
uv run llmtracefx-optimizer optimize \
  --matrix output/quickstart/matrix/manifest.json \
  --model-path /existing/local/mlx/model \
  --results output/results \
  --policy examples/optimizer/tune-policy-fastest-under-20gb-m5-pro.json \
  --dry-run
```

## Prerequisites by path

| Path | Requirements | Install or check |
| --- | --- | --- |
| Offline planning and report inspection | Python 3.10+ and `uv` | `uv sync --locked` |
| Local MLX collection | macOS on arm64, MLX, MLX-LM, and an existing local model directory | `uv sync --locked --extra mlx` |
| Metal and `xctrace` | macOS, full Xcode command-line tools, and a locally available Instruments template | `uv run llmtracefx-optimizer instruments capability` |
| Hosted API collection | An HTTPS OpenAI-compatible endpoint and a provider key stored in an environment variable | Use `collect-api --dry-run` before making a request |
| Modal planning | No Modal account or SDK is required | `uv run llmtracefx-deploy plan --help` |
| Modal execution | An approved plan, current price inputs, Modal account and auth, optional `modal` extra, proxy auth token, pinned model revision, and pinned serving image | `uv sync --locked --extra modal` |

Hosted API requests, Modal staging, deployment, health checks, and inference can
incur provider charges. None runs from the quickstart or from
`llmtracefx-deploy recipe`, `budget`, or `plan`.

## Measure

### Local MLX

```bash
uv run llmtracefx-optimizer collect-mlx \
  --run-id local-baseline \
  --model-path /existing/local/mlx/model \
  --model-id organization/model \
  --model-revision <pinned-revision> \
  --prompt-file examples/optimizer/mlx-smoke-prompt.txt \
  --output-dir output/local-baseline \
  --max-tokens 64 \
  --seed 0
```

The model path must already exist. Generic external draft-model speculation is
available through `--draft-model-path`; it is labelled `draft-model`, not
native MTP.

### OpenAI-compatible API

Start with `--dry-run`. For a real request, load the key without putting it in
the command line or shell history, then remove `--dry-run`. This Bash example
uses a silent prompt:

```bash
read -rsp "Provider API key: " PROVIDER_API_KEY
printf "\n"
export PROVIDER_API_KEY

uv run llmtracefx-optimizer collect-api \
  --run-id hosted-baseline \
  --provider provider-name \
  --endpoint https://provider.example/v1/chat/completions \
  --model-id provider-model-id \
  --prompt-file examples/optimizer/api-smoke-prompt.txt \
  --output-dir output/hosted-baseline \
  --api-key-env PROVIDER_API_KEY \
  --max-output-tokens 64

unset PROVIDER_API_KEY
```

The key value is accepted only through the named environment variable. A
credential manager or shell integration is preferable for repeated use. Check
the provider's current prices and data-handling terms before sending a request.

The article
[What your TTFT benchmark is really measuring](https://siddhantkhare.com/writing/ttft-http-client)
is a practical methods example. It explains why client buffering, empty SSE
events, hidden reasoning, and truncated streams change what a time-to-first-
token number means.

### Apple Instruments and Metal

Check support before recording:

```bash
uv run llmtracefx-optimizer instruments capability

uv run llmtracefx-optimizer instruments plan \
  --output-trace output/metal.trace \
  --output-dir output/metal \
  --time-limit 30s \
  -- /path/to/local-command --its-argument
```

`plan` prints the exact `xctrace` invocation and runs no target command. The
recording path uses `record` in place of `plan`. Current parsing supports the
`metal-gpu-intervals` table and reports interval counts, duration sums, and
wall spans. It does not infer GPU utilization, occupancy, bandwidth, power, or
energy from those intervals.

A public, reproducible Apple Silicon example with a deterministic Metal
workload, sanitized measured evidence, integrity hashes, and charts lives in
[`examples/metal_evidence/`](examples/metal_evidence/README.md). Raw trace
bundles and XML exports are excluded by design.

## Verify, tune, and optimize

Generate a matrix for an existing model:

```bash
uv run llmtracefx-optimizer workloads generate-matrix \
  --model-id organization/model \
  --model-family qwen3_next \
  --target-model-path /existing/local/mlx/model \
  --output-dir output/matrix
```

Inspect execution without loading the model:

```bash
uv run llmtracefx-optimizer workloads run \
  --matrix output/matrix/manifest.json \
  --model-path /existing/local/mlx/model \
  --output-dir output/results \
  --mode autoregressive \
  --dry-run
```

Remove `--dry-run` to execute the selected MLX rows. Re-running the same command
resumes complete hash-matching rows. Pass `--no-resume` only when a deliberate
rerun is required.

For a hosted API, use `workloads run-api`. Its `--dry-run` validates selection,
endpoint configuration, and credential handling without a request:

```bash
uv run llmtracefx-optimizer workloads run-api \
  --matrix output/matrix/manifest.json \
  --output-dir output/api-results \
  --provider provider-name \
  --endpoint https://provider.example/v1/chat/completions \
  --model-id provider-model-id \
  --api-key-env PROVIDER_API_KEY \
  --mode autoregressive \
  --dry-run
```

Then run `tune`, `tune-report`, or `optimize` against the verified result
directory. The example policy files under `examples/optimizer/` are labelled
examples and contain no benchmark claim.

## Modal deployment

`llmtracefx-deploy` is a planning CLI for the pinned GLM-5.3-Flash harness. It
prints model facts, recommends a session cap from an operator-supplied credit
balance, and evaluates a proposed deployment from operator-supplied prices and
limits.

The planner is no-spend by construction. It does not deploy, authenticate,
open a socket, import Modal, download weights, call an API, or allocate an
accelerator. If required inputs or safety gates fail, it withholds paid
commands from its executable set.

Its calculated cost envelope is planning arithmetic, not a Modal billing
guarantee. Provider scheduling, billing granularity, traffic that reaches the
deployment, price changes, failed starts, and resources outside the declared
inputs can still affect the bill. Follow the full
[Modal GLM-5.3-Flash runbook](SELF_HOST_GLM_RUNBOOK.md), review every generated
command, and tear down the app and volume explicitly.

The older public Modal analyzer endpoint is retired and is not part of the
current quickstart.

## Command index

The console scripts below come from `pyproject.toml`.

| Command | Status | Purpose |
| --- | --- | --- |
| `llmtracefx-optimizer` | Current | Evidence collection, deterministic workloads, verification, comparison, tuning, and optimization |
| `llmtracefx-deploy` | Current | No-spend planning for the optional Modal GLM-5.3-Flash harness |
| `llmtracefx` | Legacy compatibility | Earlier token trace analyzer |
| `llmtracefx-serve` | Legacy compatibility | Local FastAPI surface for the earlier analyzer |
| `llmtracefx-dashboard` | Legacy compatibility | Earlier Streamlit dashboard; not the current evidence workflow |

Use `uv run llmtracefx-optimizer --help` and
`uv run llmtracefx-deploy --help` as the source of truth for current flags.
The legacy scripts are listed for package inventory only. Do not assume they
implement a side-effect-free `--help` path.

## Current status and limitations

- The repository contains synthetic fixtures and interface examples, but no
  real model benchmark result that should be treated as a performance claim.
- MLX collection requires Apple silicon and an existing local model. There is
  no direct CUDA collector; NVIDIA llama.cpp evidence is imported from captured
  output.
- API timing is observed at the client. It cannot expose provider queueing,
  prefill, kernel execution, or server-side clocks.
- Native Qwen MTP execution is not supported by the current MLX-LM path.
  LLMTraceFX records that limitation instead of substituting generic
  draft-model speculation.
- Instruments table availability varies by macOS, Xcode, hardware, and
  template. Unsupported schemas remain unsupported.
- Tuning is only as sound as the supplied workload, repetitions, provenance,
  and policy. An `inconclusive` outcome is expected when evidence is missing,
  noisy, tied, or fails every constraint.
- Modal planning reduces accidental spend but does not impose a provider-side
  account budget or guarantee a final bill.
- The legacy analyzer, local API, and dashboard remain in the package for
  compatibility. Their synthetic GPU scoring and optional explanation path are
  not the recommended optimizer workflow.
- The legacy `deploy-modal`, `serve-modal`, and `test-modal` Make targets operate
  on the earlier analyzer and can create billable Modal resources. They are not
  part of the budget-guarded GLM harness.

## Development

```bash
uv sync --locked --extra dev --extra test
uv run pytest
make lint-changed
```

The project supports Python 3.10 through 3.13. The `mlx` extra is installed
only on macOS arm64, and the `modal` extra is optional.

## License

[Apache-2.0](LICENSE)
