# Self-hosting GLM-5.3-Flash on Modal: runbook

A budget-guarded, pinned harness for standing GLM-5.3-Flash up on rented
accelerators for long enough to measure it with the LLMTraceFX API
collector, and then taking it down again.

Read the next section before anything else. It is the one that decides
whether you should be doing this at all.

## Use OpenRouter first. Self-host only to answer what OpenRouter cannot

Self-hosting a 320B model is not the cheap way to get GLM-5.3-Flash
tokens. It is the expensive way, and it is only worth it for questions a
hosted API cannot answer.

Use the hosted route for anything about the model:

```bash
uv run llmtracefx-optimizer collect-api \
  --run-id glm53flash-hosted \
  --provider openrouter \
  --endpoint https://openrouter.ai/api/v1/chat/completions \
  --model-id z-ai/glm-5.3-flash \
  --prompt-file prompts/smoke.txt \
  --output-dir output/glm53flash-hosted \
  --api-key-env OPENROUTER_API_KEY
```

Per-token pricing for `z-ai/glm-5.3-flash` is published on the OpenRouter
model page (<https://openrouter.ai/z-ai/glm-5.3-flash>). Check it there
rather than trusting a number written down here: it has changed at least
once, and a stale figure in a runbook is worse than no figure.

Self-host only when the question is about *serving*, and therefore cannot
be asked of somebody else's endpoint:

- how long this checkpoint takes to load onto a given accelerator count,
- what tensor-parallel degree and context cap actually fit,
- how the serving framework behaves under your own request shape,
- whether the numbers a provider reports match your own measurement.

Those are short experiments. Budget for an afternoon, not for a month.

## What is pinned, what you must supply, and why

The harness compiles in the facts that describe the checkpoint, because
they cannot drift without the checkpoint changing:

| Fact | Value | Source |
| --- | --- | --- |
| Repository | `zai-org/GLM-5.3-Flash` | <https://huggingface.co/zai-org/GLM-5.3-Flash> |
| Parameters | 320B total, 18B active | model card |
| Layers | 45 | `config.json` |
| Experts | 288 routed, 8 per token | `config.json` |
| Quantization | FP8, E4M3, dynamic activation scaling | `config.json` |
| Attention | hybrid linear with periodic sparse layers | `config.json` |
| Max context | 1,048,576 tokens | `config.json` |
| Multimodal | yes | `config.json` |

Run `llmtracefx-deploy recipe` to print these with their
sources.

Two notes on scope. The default `zai-org/GLM-5.3-Flash` repository *is*
the FP8 checkpoint; there is no separate `-FP8` slug. And the harness
refuses `zai-org/GLM-5.3-Flash-BF16` and full `zai-org/GLM-5.3` by name,
because both are roughly twice the size and neither fits the budget this
tool is built for.

Everything mutable is *not* compiled in, and the harness will not run
without it:

| You supply | Flag | Why it is not a constant |
| --- | --- | --- |
| Model revision | `--model-revision` | A branch is not a revision. `main` moves. |
| Serving image | `--image` | Tags can be repointed; `latest` is meaningless. |
| Framework version | `--framework-version` | Records what actually served. |
| GPU price and its date | `--usd-per-gpu-hour`, `--price-effective-date`, `--price-source` | Cloud prices change. A stale literal is a silently wrong cost. |
| Budget, GPU type and count, runtime | `--max-usd`, `--gpu-type`, `--gpu-count`, `--max-runtime-seconds` | These are the spending authority. A default would be an assumption about your money. |

Resolve the revision once and pin it:

```bash
curl -s https://huggingface.co/api/models/zai-org/GLM-5.3-Flash \
  | python -c "import json,sys; print(json.load(sys.stdin)['sha'])"
```

### Why vLLM by default

Both SGLang and vLLM are listed as supported on the GLM-5.3-Flash model
card, and the harness supports both (`--framework vllm|sglang`). vLLM is
the default, and the deciding reason is where each one puts your API key.

vLLM reads it from the environment. `VLLM_API_KEY` is declared in
`vllm/envs.py` and consumed by the authentication middleware in
`vllm/entrypoints/serve/middleware/register.py`, so the value never
appears on a command line.

SGLang accepts it only as `--api-key` on argv, and its engine logs its
own resolved configuration with
`logger.info(f"server_args={server_args.resolved_dict()}")` in
`python/sglang/srt/entrypoints/engine.py`, unredacted. The key therefore
lands in the container's stdout, and from there in Modal's app logs, as
well as being readable from `/proc/<pid>/cmdline`.

So `--framework sglang` is refused unless you pass
`--accept-argv-credential-exposure`, which records that you understood
the trade. Both sources read 2026-08-30.

SGLang does have one advantage worth knowing: the model card publishes a
complete container recipe for it, including the image, shared-memory size
and IPC mode, whereas the vLLM snippet is a bare `vllm serve`. If you
take the SGLang route, expect to supply those container details
yourself.

Be aware of what is *not* published. As of 2026-08-30 the model card does
not state a minimum SGLang or vLLM version, does not show multi-GPU or
FP8 flags, and pins no image digest (it shows `lmsysorg/sglang:latest`).
The `--tp` / `--tensor-parallel-size` and `--context-length` /
`--max-model-len` flags this harness generates come from each framework's
own general documentation, not from the GLM-5.3-Flash card. Verify the
current recipe at <https://docs.sglang.io/cookbook/autoregressive/GLM/GLM-5.3-Flash>
or <https://recipes.vllm.ai/zai-org/GLM-5.3-Flash> before you commit
money to a version.

### Whether 4x H200 is enough

`deploy plan` computes it: roughly 306 GiB of weights against 4 x 141 GiB
of advertised VRAM leaves about 258 GiB, or 45 percent, for KV cache and
activations. The harness requires at least 20 percent and so approves it.

This is arithmetic, not vendor guidance. No official hardware requirement
for GLM-5.3-Flash was published on the model card or in
<https://github.com/zai-org/GLM-5> as of 2026-08-30. The check can prove a
configuration *cannot* work; it cannot prove one will. The smoke test
settles that.

## Budgeting on a small balance

```bash
uv run llmtracefx-deploy budget --credit-usd 30
```

On $30 this recommends a $10 cap and holds $20 back. The reserve is not
timidity: it covers a start up that fails after the image is pulled and
the GPUs are allocated, one full retry, and volume storage, which keeps
billing after the GPUs stop and which Modal bills for up to four days
after you delete the data.

What $10 buys is a short window. Price the resources yourself, then
divide.

The total is not GPU-only. It is the sum of five mandatory charges, and
`deploy plan` prints them as separate lines so you can check each:

| Line | What it is |
| --- | --- |
| `serving-gpu` | the accelerators, for the whole deployment window |
| `serving-compute` | the CPU and memory billed alongside them |
| `staging` | the CPU-only download container, to its own timeout |
| `verification` | the CPU-only inventory check, to its own timeout |
| `storage` | the volume, for your retention plus four days |

The accelerators are priced against `--max-deployment-seconds`, not
against the container timeout. That matters: Modal's `timeout` bounds a
Function's *execution*, and this server's function returns as soon as it
has launched the framework, so the timeout alone would not stop a request
arriving tomorrow from allocating accelerators again.

What does stop it is an expiry. The deploy bakes an absolute instant into
the container image, and the serving container refuses to start once it
has passed. A container that starts a second before the expiry still runs
its own lifetime afterwards, so one container timeout is added to the
window rather than assuming a clean cut. That is what makes the number an
upper bound rather than an estimate.

Be precise about what the expiry does and does not do. It stops a
container from *serving* past the window. It cannot stop one from being
*started*, because the platform schedules a web server container before
any of its code runs, so a request allocates accelerators first and is
refused second. Proxy auth is what makes that bounded: without
credentials the request never reaches the scheduler, and with them only
you can trigger it. A request bearing your token after the expiry still
cold-starts a container that refuses within seconds, and that residual is
deliberately not priced.

`min_containers` above zero is refused outright for a related reason: a
warm container bills from deploy whether or not anything calls it, so the
whole window is spent by definition rather than in the worst case.

## The run

### 0. Plan. This is free, offline and unauthenticated

```bash
uv run llmtracefx-deploy plan \
  --max-usd 10.00 \
  --gpu-type H200 --gpu-count 4 \
  --max-runtime-seconds 1800 \
  --max-deployment-seconds 3600 \
  --usd-per-gpu-hour "<rate you read>" \
  --usd-per-cpu-core-hour "<rate you read>" \
  --usd-per-gib-memory-hour "<rate you read>" \
  --storage-usd-per-gib-month "<rate you read>" \
  --storage-retention-days 1 \
  --price-effective-date "<YYYY-MM-DD you read them>" \
  --price-source https://modal.com/pricing \
  --model-revision "<40-character SHA>" \
  --image "vllm/vllm-openai:<tag>@sha256:<digest>" \
  --framework-version "<version>" \
  --context-length 131072 \
  --startup-timeout-seconds 900 \
  --output output/glm-plan.json
```

This needs no Modal account, no token and no network. It prints the
decision, the worst-case cost envelope, the capacity check and every
command below, then exits non-zero if anything is blocking.

Run it before you install the Modal CLI. If it refuses, nothing you do
next can cost money, because the paid commands are withheld from the
executable set rather than merely annotated.

### 1. Authenticate

```bash
modal setup
```

Install the SDK first if you have not: `uv sync --extra modal`. It is an
optional extra; the planner and the test suite do not need it.

### 2. Create a proxy token, and store it as the secret

The endpoint is served with Modal proxy auth on, and there is no flag to
turn that off. The reason is a cost one rather than a security one: a web
server container is scheduled by the platform *before* any of its code
runs, so on a public URL every request from anyone allocates accelerators
first and is refused second, without limit and including after the
deployment expiry. Nothing in the cost envelope bounds that, so the
harness will not configure it. With proxy auth on, Modal returns 401 at
its edge and never schedules a container.

Requiring it costs nothing, because Modal's token pair can be sent as a
single `Authorization: Bearer` value, which is the scheme the OpenAI API
uses. The collector and every OpenAI-compatible client work unchanged.

Create a token in the dashboard under Proxy Auth Tokens, or with
`modal workspace proxy-tokens`. Then join the pair with a period and
store it:

```bash
export GLM_SELFHOST_API_KEY="wk-yourtokenid.ws-yourtokensecret"
modal secret create glm-selfhost-api-key GLM_SELFHOST_API_KEY="$GLM_SELFHOST_API_KEY"
```

Creating the secret in the Modal dashboard instead keeps it out of shell
history altogether. Either way, no plan, manifest, log line or generated
command produced by this harness contains the value: they contain the
string `$GLM_SELFHOST_API_KEY`.

Check that the export actually took. Running the `modal secret create`
line in a shell where `GLM_SELFHOST_API_KEY` is not set expands it to an
empty value and creates a secret that exists but holds nothing. Modal's
own `required_keys` assertion does not catch that, because it checks the
variable is present, not that it contains anything. The server does: it
refuses to start on an empty key rather than serving without
authentication.

If the repository is gated, create a Hugging Face token secret too and
name it in `LLMTRACEFX_GLM_HF_SECRET_NAME`.

### 3. Export the plan environment

`modal run` and `modal deploy` import the app, and the app rebuilds the
adjudicated plan from the environment. If any spending authority, price or
pin is missing, or if the plan is refused, the import raises and nothing is
registered. There is no configuration in which a default GPU count gets
deployed.

```bash
export LLMTRACEFX_GLM_MAX_USD=10.00
export LLMTRACEFX_GLM_GPU_TYPE=H200
export LLMTRACEFX_GLM_GPU_COUNT=4
export LLMTRACEFX_GLM_MAX_RUNTIME_SECONDS=1800
export LLMTRACEFX_GLM_MAX_DEPLOYMENT_SECONDS=3600
export LLMTRACEFX_GLM_USD_PER_GPU_HOUR="<rate you read>"
export LLMTRACEFX_GLM_USD_PER_CPU_CORE_HOUR="<rate you read>"
export LLMTRACEFX_GLM_USD_PER_GIB_MEMORY_HOUR="<rate you read>"
export LLMTRACEFX_GLM_STORAGE_USD_PER_GIB_MONTH="<rate you read>"
export LLMTRACEFX_GLM_STORAGE_RETENTION_DAYS=1
export LLMTRACEFX_GLM_PRICE_EFFECTIVE_DATE="<YYYY-MM-DD>"
export LLMTRACEFX_GLM_PRICE_SOURCE=https://modal.com/pricing
export LLMTRACEFX_GLM_MODEL_REVISION="<40-character SHA>"
export LLMTRACEFX_GLM_IMAGE="vllm/vllm-openai:<tag>@sha256:<digest>"
export LLMTRACEFX_GLM_FRAMEWORK_VERSION="<version>"
export LLMTRACEFX_GLM_CONTEXT_LENGTH=131072
export LLMTRACEFX_GLM_STARTUP_TIMEOUT_SECONDS=900
```

Optional, all with safe defaults: `LLMTRACEFX_GLM_FRAMEWORK` (vllm),
`LLMTRACEFX_GLM_MAX_CONTAINERS` (1), `LLMTRACEFX_GLM_MIN_CONTAINERS` (0),
`LLMTRACEFX_GLM_SCALEDOWN_WINDOW_SECONDS` (300),
`LLMTRACEFX_GLM_MAX_CONCURRENT_INPUTS` (1),
`LLMTRACEFX_GLM_TENSOR_PARALLEL_SIZE` (the GPU count),
`LLMTRACEFX_GLM_APP_NAME`, `LLMTRACEFX_GLM_VOLUME_NAME`,
`LLMTRACEFX_GLM_API_KEY_ENV`, `LLMTRACEFX_GLM_MODAL_SECRET_NAME`,
`LLMTRACEFX_GLM_STORED_GIB`.

You do not have to re-export these for the containers. The deploy bakes
the adjudicated configuration into both container images, so a remote
worker rebuilds the same plan from the image rather than from a shell it
has never seen. Nothing baked there is a secret: names, prices,
revisions and limits only. The API key still travels separately, in the
Modal Secret.

### 4. Stage the weights on CPU

```bash
modal volume create llmtracefx-glm53flash-weights

modal run llmtracefx/deploy/modal_glm_app.py::stage_weights \
  --repo-id zai-org/GLM-5.3-Flash \
  --revision "<40-character SHA>" \
  --confirm "<the same SHA>" \
  --volume-name llmtracefx-glm53flash-weights
```

This function has no `gpu=` argument at all, so several hundred GiB are
never transferred while accelerators are billing. It is CPU and network
only, and it takes a long time.

`--confirm` repeats `--revision` rather than being a yes/no flag. A yes/no
flag survives being copied onto a different command line and confirms
whatever that line now says; restating the SHA means an edited command
with a stale confirmation fails instead of fetching something nobody
asked for.

It writes `manifest.json` next to the weights recording the repository,
the revision, every file and its size, and each file's sha256 where the
repository published one. Hashes are not recomputed locally: hashing
hundreds of GiB would double the container time you are billed for, and a
locally computed digest proves the bytes are self-consistent, not that
they are the bytes upstream published.

Re-running with the same revision is a no-op. Inspect it any time:

```bash
modal run llmtracefx/deploy/modal_glm_app.py::read_manifest \
  --volume-name llmtracefx-glm53flash-weights \
  --revision "<40-character SHA>"
```

### 4b. Verify the staged inventory, still on CPU

```bash
modal run llmtracefx/deploy/modal_glm_app.py::verify_weights \
  --volume-name llmtracefx-glm53flash-weights \
  --revision "<40-character SHA>"
```

A manifest naming the right revision proves a download was started, not
that it finished. An interruption near the end leaves the manifest and a
short shard behind, and the only cheap place to find that is a CPU
container. This re-checks every file's existence and size, and its
published sha256 where the repository provided one, then records the
result on the volume.

The serving container requires that record. If verification has not
passed for the revision it is configured with, it refuses to start, so a
truncated volume is never discovered on four accelerators.

### 5. Deploy

```bash
modal deploy llmtracefx/deploy/modal_glm_app.py
```

Deploying registers the app. Accelerators are allocated on the first
request and released after the scaledown window, so nothing bills between
step 5 and step 6.

The server is pointed at the volume path, never at the repository id, so
it cannot quietly fall back to downloading weights on GPUs. It verifies
the staging manifest before launching the framework: if the weights are
not staged you lose seconds of accelerator time, not an entire start up.

The deploy prints the app URL. Feed it back in to get exact client
commands:

```bash
uv run llmtracefx-deploy plan ... \
  --endpoint-base-url https://<workspace>--llmtracefx-glm53flash-serve.modal.run
```

### 6. Health, readiness, one bounded smoke request

```bash
curl -fsS --max-time 30 "$URL/health"

curl -fsS --max-time 60 \
  -H "Authorization: Bearer $GLM_SELFHOST_API_KEY" "$URL/v1/models"

curl -fsS --max-time 120 \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GLM_SELFHOST_API_KEY" \
  -d '{"model":"zai-org/GLM-5.3-Flash","max_tokens":32,"stream":false,"messages":[{"role":"user","content":"Reply with the word: ready"}]}' \
  "$URL/v1/chat/completions"
```

The first of these cold-starts the container and therefore allocates
GPUs. The smoke request is capped at 32 output tokens so a misbehaving
server cannot generate indefinitely.

None of this is a benchmark. It proves the endpoint serves.

### 7. Measure with the collector

```bash
uv run llmtracefx-optimizer collect-api \
  --run-id glm53flash-selfhost \
  --provider modal-selfhost \
  --endpoint "$URL/v1/chat/completions" \
  --model-id zai-org/GLM-5.3-Flash \
  --model-revision "<40-character SHA>" \
  --prompt-file prompts/smoke.txt \
  --output-dir output/glm53flash-selfhost \
  --api-key-env GLM_SELFHOST_API_KEY
```

Timing evidence is produced from outside, by the collector that already
exists, and not by the server. A server that measures and grades itself
produces numbers with no independent provenance. The server records only
what it can honestly observe about its own configuration: GPU model and
count, framework and version, model revision, quantization, tensor
parallel degree, context cap, and start up duration, each marked as
configured or observed. It makes no performance claim.

### 8. Tear down. This is the step people skip

```bash
modal app stop llmtracefx-glm53flash
modal volume delete llmtracefx-glm53flash-weights
```

Stop the app first: that is what prevents a request from starting a new
GPU container.

Then delete the volume. Storage is the charge that outlives the
experiment, roughly 306 GiB of it, billed until deletion and for up to
four days afterwards. Keep the volume only if you intend to serve again
soon, because re-staging means transferring the whole checkpoint again.

Both teardown commands stay available even when a plan is refused. They
are the commands that make spending stop, so gating them on approval
would withhold them exactly when they are most needed.

## What the harness refuses, and how to proceed anyway

| Refusal | Meaning | If you meant it |
| --- | --- | --- |
| Worst case exceeds budget | `gpu_count x containers x runtime x rate > --max-usd` | Lower the runtime or GPU count, or raise `--max-usd` |
| Price quote is stale | Older than 90 days | Re-read the price, or `--accept-stale-price` |
| Price is future dated | Data entry error | Fix the date. No override. |
| Revision is not a SHA | `main` or a tag was passed | Resolve and pin the SHA |
| Image tag is `latest` | Not reproducible | Pin a tag and digest. No override. |
| Image has no digest | A tag can be repointed | `--accept-mutable-image` |
| Accelerators too small | Under 20 percent left after weights | Add GPUs |
| `min_containers` above zero | Warm GPUs bill from deploy whether or not anything calls them | Nothing. Set it to 0 and accept the cold start. |
| Deployment window shorter than the container timeout | The window cannot be shorter than one container's life | Raise `--max-deployment-seconds` |
| Weights not verified | The inventory check has not passed for this revision | Run the verify-weights step |
| Deployment expired | The window baked at deploy has passed | `modal app stop`, then redeploy with a fresh window |
| Framework takes the key on argv | SGLang logs its own config unredacted | `--accept-argv-credential-exposure` |

## Safety properties, and where they are tested

- Planning is pure. `deploy plan` opens no socket, reads no credential and
  never imports the Modal SDK. `tests/deploy/test_deploy_cli.py` asserts
  this by making `socket.socket` raise.
- Nothing spends by default. Every spending parameter is required with no
  default; omitting one is an error.
- Downloads never touch a GPU. `tests/deploy/test_modal_app.py` imports
  the real entrypoint against a fake Modal SDK and asserts the staging
  function receives no `gpu` argument, and that serving receives exactly
  the planned GPU string, timeout, container limits and concurrency.
- No credential is ever persisted. Every field in a plan is a name by
  construction, and on top of that the rendered document is scrubbed
  against the resolved value of the variable you named, using the same
  redactor `collect-api --dry-run` uses. `tests/deploy/test_deploy_cli.py`
  plants a sentinel key and asserts it appears in neither stdout, stderr
  nor the written artifact, on the success path, the JSON path, the
  refusal path, and the path where the key was pasted where a name
  belongs. Shape validation alone would not catch that last one: a
  credential can be a well-formed Secret name.
- Diagnostics repeat nothing the caller typed. A credential-shaped flag
  is refused with a fixed message that does not name the offending
  option, so no part of your argv can reach a log through it.
- Modal is optional. The whole suite passes with the SDK uninstalled.

```bash
uv run pytest tests/deploy
```
