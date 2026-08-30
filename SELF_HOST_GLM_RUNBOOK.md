# Self-host GLM-5.3-Flash on Modal

This runbook covers the optional, budget-guarded Modal harness for
GLM-5.3-Flash. Its purpose is to answer serving questions that a hosted API
cannot answer, collect evidence briefly, and remove the deployment.

Planning is offline and no-spend. Staging, deployment, health checks, and
inference are not.

## Start with the no-spend path

The planning CLI needs neither the Modal SDK nor Modal authentication. The
initial dependency install can use the network when packages are not cached;
the three `llmtracefx-deploy` commands shown here do not:

```bash
uv sync --locked

uv run llmtracefx-deploy recipe
uv run llmtracefx-deploy budget --credit-usd 30
uv run llmtracefx-deploy plan --help
```

These commands do not open a socket, read a credential, download weights,
create resources, or allocate an accelerator.

Use a hosted OpenAI-compatible endpoint first when the question is about model
output or client-visible streaming behavior. The API collector also has a
no-network dry run:

```bash
env -u LLMTRACEFX_HOSTED_DRY_RUN_NO_KEY \
  uv run llmtracefx-optimizer collect-api \
  --run-id hosted-plan \
  --provider provider-name \
  --endpoint https://provider.example/v1/chat/completions \
  --model-id provider-model-id \
  --prompt-file examples/optimizer/api-smoke-prompt.txt \
  --output-dir output/hosted-plan \
  --api-key-env LLMTRACEFX_HOSTED_DRY_RUN_NO_KEY \
  --dry-run
```

Self-host only when the measurement depends on control of the serving stack,
such as checkpoint load time, tensor parallelism, context capacity, framework
behavior, or hardware placement.

## What the planner does

`llmtracefx-deploy plan`:

- requires the operator to supply every price and spending input;
- requires a full model revision and rejects moving branch names;
- rejects `latest` images and warns or refuses mutable image references;
- checks approximate checkpoint capacity against the supplied accelerator
  memory profile;
- defaults to zero warm containers, one maximum container, and one concurrent
  request;
- requires Modal proxy authentication for the endpoint;
- includes CPU, memory, staging, verification, accelerator, and volume storage
  estimates;
- writes a credential-free plan and withholds paid commands when a blocker is
  present.

The planner is a refusal gate and estimate. It is not a provider-side account
budget, billing control, or guarantee of the final charge. In particular:

- Modal can schedule a container before application code checks its expiry;
- a request that passes edge authentication can cause allocation;
- provider prices and billing granularity can differ from the values supplied
  to the planner;
- failed starts, retries, network transfer, retained storage, and resources not
  represented by the declared inputs can affect the bill;
- the capacity check is arithmetic, not proof that a framework and checkpoint
  will start successfully.

Treat `--max-usd` as the operator's planning threshold for the supplied cost
model, not a hard cloud billing cap.

## Required inputs

Read current values from their primary sources before each experiment.

| Input | Flag | Requirement |
| --- | --- | --- |
| Available session amount | `--max-usd` | An amount the operator is willing to risk, independent of provider credits |
| Accelerator and count | `--gpu-type`, `--gpu-count` | Match the intended Modal configuration |
| Container and deployment windows | `--max-runtime-seconds`, `--max-deployment-seconds` | Keep both short and explicit |
| Accelerator price | `--usd-per-gpu-hour` | Current operator-read rate |
| CPU and memory prices | `--usd-per-cpu-core-hour`, `--usd-per-gib-memory-hour` | Current operator-read rates |
| Storage price and retention | `--storage-usd-per-gib-month`, `--storage-retention-days` | Include deletion and retention policy |
| Price provenance | `--price-effective-date`, `--price-source` | Record when and where every rate was read |
| Model revision | `--model-revision` | Full 40-character repository commit SHA |
| Serving image | `--image` | Prefer a tag plus `sha256` digest; `latest` is refused |
| Framework version | `--framework-version` | Version inside the selected image |
| Context cap | `--context-length` | Must not exceed the model maximum |

The harness is pinned to `zai-org/GLM-5.3-Flash`. Run
`llmtracefx-deploy recipe` for the model facts and their recorded sources.
Verify those sources again before spending:

- [GLM-5.3-Flash model card](https://huggingface.co/zai-org/GLM-5.3-Flash)
- [Modal pricing](https://modal.com/pricing)
- [Modal documentation](https://modal.com/docs)

Resolve the model revision outside the planner, then review it before use. A
branch such as `main` is intentionally not accepted.

## Build and review a plan

Export the non-secret deployment configuration before planning. The Modal app
rebuilds the approved plan from these variables when `modal deploy` imports it.
If a required variable is absent, deployment fails closed.

Replace every angle-bracketed value with a value you verified. This block is
intentionally not executable as written:

```bash
export LLMTRACEFX_GLM_MAX_USD="<session-threshold-usd>"
export LLMTRACEFX_GLM_GPU_TYPE="<modal-gpu-type>"
export LLMTRACEFX_GLM_GPU_COUNT="<count>"
export LLMTRACEFX_GLM_MAX_RUNTIME_SECONDS="<container-seconds>"
export LLMTRACEFX_GLM_MAX_DEPLOYMENT_SECONDS="<deployment-seconds>"
export LLMTRACEFX_GLM_USD_PER_GPU_HOUR="<current-gpu-rate>"
export LLMTRACEFX_GLM_USD_PER_CPU_CORE_HOUR="<current-cpu-rate>"
export LLMTRACEFX_GLM_USD_PER_GIB_MEMORY_HOUR="<current-memory-rate>"
export LLMTRACEFX_GLM_STORAGE_USD_PER_GIB_MONTH="<current-storage-rate>"
export LLMTRACEFX_GLM_STORAGE_RETENTION_DAYS="<planned-days>"
export LLMTRACEFX_GLM_PRICE_EFFECTIVE_DATE="<YYYY-MM-DD>"
export LLMTRACEFX_GLM_PRICE_SOURCE="https://modal.com/pricing"
export LLMTRACEFX_GLM_MODEL_REVISION="<40-character-model-sha>"
export LLMTRACEFX_GLM_IMAGE="<repository:tag@sha256:digest>"
export LLMTRACEFX_GLM_FRAMEWORK_VERSION="<version>"
export LLMTRACEFX_GLM_CONTEXT_LENGTH="<tokens>"
export LLMTRACEFX_GLM_AS_OF="<YYYY-MM-DD-planning-date>"

uv run llmtracefx-deploy plan \
  --max-usd "$LLMTRACEFX_GLM_MAX_USD" \
  --gpu-type "$LLMTRACEFX_GLM_GPU_TYPE" \
  --gpu-count "$LLMTRACEFX_GLM_GPU_COUNT" \
  --max-runtime-seconds "$LLMTRACEFX_GLM_MAX_RUNTIME_SECONDS" \
  --max-deployment-seconds "$LLMTRACEFX_GLM_MAX_DEPLOYMENT_SECONDS" \
  --usd-per-gpu-hour "$LLMTRACEFX_GLM_USD_PER_GPU_HOUR" \
  --usd-per-cpu-core-hour "$LLMTRACEFX_GLM_USD_PER_CPU_CORE_HOUR" \
  --usd-per-gib-memory-hour "$LLMTRACEFX_GLM_USD_PER_GIB_MEMORY_HOUR" \
  --storage-usd-per-gib-month "$LLMTRACEFX_GLM_STORAGE_USD_PER_GIB_MONTH" \
  --storage-retention-days "$LLMTRACEFX_GLM_STORAGE_RETENTION_DAYS" \
  --price-effective-date "$LLMTRACEFX_GLM_PRICE_EFFECTIVE_DATE" \
  --price-source "$LLMTRACEFX_GLM_PRICE_SOURCE" \
  --model-revision "$LLMTRACEFX_GLM_MODEL_REVISION" \
  --image "$LLMTRACEFX_GLM_IMAGE" \
  --framework-version "$LLMTRACEFX_GLM_FRAMEWORK_VERSION" \
  --context-length "$LLMTRACEFX_GLM_CONTEXT_LENGTH" \
  --as-of "$LLMTRACEFX_GLM_AS_OF" \
  --output output/glm-plan.json
```

The block uses the planner's defaults for framework, tensor parallel size,
served model name, port, mount path, container limits, scaledown window,
startup timeout, concurrency, app name, volume name, credential variable name,
and Modal Secret name. If you change any optional planner flag, export its
matching `LLMTRACEFX_GLM_...` variable before both planning and deployment. For
example, `--startup-timeout-seconds 900` must be paired with
`LLMTRACEFX_GLM_STARTUP_TIMEOUT_SECONDS=900`. Re-run the plan after every
change.

Review both the terminal decision and `output/glm-plan.json`:

1. The decision must be `APPROVED`.
2. The model revision and image must match the intended artifacts.
3. Every rate, date, and source must match a current primary source.
4. The plan must use proxy authentication, zero warm containers, and a
   container limit you intend.
5. The capacity result must be plausible, but it must not be read as a
   successful startup test.
6. The list of paid steps must be short enough to execute and tear down during
   the planned window.
7. No secret value may appear in the plan or terminal output.

A refused plan is a successful safety outcome. Change the configuration or stop
the experiment. Do not bypass a refusal merely to reproduce an earlier recipe.

## Paid execution boundary

Everything below this heading can create a charge. Do not continue unless an
approved plan has been reviewed by the person responsible for the account.

Install and authenticate Modal only at this point:

```bash
uv sync --locked --extra modal
modal setup
```

Keep the reviewed `LLMTRACEFX_GLM_*` variables in the shell used for
deployment. If deployment happens in a new shell, re-export the same reviewed
values and re-run the plan there before any paid step.

The initial plan prints the exact lifecycle commands for its app name, volume
name, and model revision. Endpoint-dependent commands contain a
`<deployed-url>` placeholder until Modal assigns the URL. It also does not
print the required environment exports. Use the block above for those values,
then use the generated lifecycle commands rather than copying an old command
from a document.

### Protect endpoint credentials

The harness requires Modal proxy authentication so unauthenticated traffic is
rejected at the edge before a serving container is scheduled. Create the proxy
token in Modal and store the joined token value in the configured Modal Secret.

Prefer creating the secret in the Modal dashboard so the value does not enter
shell history. If the CLI is used, export the value once, verify the variable is
non-empty, create the named secret, and clear the shell variable afterwards.
The plan accepts only an environment variable name and a Modal Secret name. It
must never contain the credential value.

The default vLLM path passes the endpoint key through an environment variable.
The SGLang path requires `--accept-argv-credential-exposure` because the key
can appear in process arguments and framework logs. Do not select SGLang unless
that exposure is acceptable for the environment.

### Execute the generated lifecycle

An approved plan orders the work as follows:

1. Create the named Modal volume.
2. Stage the pinned model revision in a CPU-only function.
3. Verify the staged file inventory in a CPU-only function.
4. Deploy the serving application with the adjudicated environment.
5. Re-run the no-spend plan with the assigned endpoint URL.
6. Run health and readiness checks from the refreshed plan.
7. Send one output-capped smoke request.
8. Measure the endpoint with `llmtracefx-optimizer collect-api`.
9. Stop the app.
10. Delete the volume unless there is an explicit, reviewed reason to retain it.

After `modal deploy` prints the URL, export it and repeat the exact approved
plan command from the previous section with one additional flag:

```bash
export LLMTRACEFX_GLM_ENDPOINT_BASE_URL="https://<assigned-modal-host>"

# Add this to the same llmtracefx-deploy plan invocation:
--endpoint-base-url "$LLMTRACEFX_GLM_ENDPOINT_BASE_URL"
```

The replan is still offline and no-spend. Review that it remains approved, then
use its refreshed health, readiness, smoke, and collection commands. They must
contain the assigned host, not `<deployed-url>`. The generated collector uses
the repository's existing `examples/optimizer/api-smoke-prompt.txt`.

Staging and verification avoid spending accelerator time on the large model
download, but they still use billable compute, storage, and network resources.
Deploying registers the app; a request can allocate accelerators. A health
request is not free merely because it generates no model output.

The smoke request proves only that the endpoint responds. It is not a
benchmark. Keep its output cap small, then collect timing from outside the
server:

```bash
uv run llmtracefx-optimizer collect-api \
  --run-id glm53flash-selfhost \
  --provider modal-selfhost \
  --endpoint "$URL/v1/chat/completions" \
  --model-id zai-org/GLM-5.3-Flash \
  --model-revision <40-character-model-sha> \
  --prompt-file examples/optimizer/api-smoke-prompt.txt \
  --output-dir output/glm53flash-selfhost \
  --api-key-env GLM_SELFHOST_API_KEY \
  --max-output-tokens 32
```

The collector observes the client boundary. It does not claim server-side
queue, prefill, kernel, or GPU timing.

## Teardown

Use the names from the approved plan:

```bash
modal app stop <app-name>
modal volume delete <volume-name>
```

Stop the app first so a later request cannot start another serving container.
Delete the volume to end ongoing storage retention. Confirm both resources are
gone in Modal. Do not infer successful teardown from a local command timeout or
closed terminal.

If any step behaves unexpectedly, stop new requests and begin teardown before
debugging. Preserve the credential-free plan and evidence artifacts, not secret
values or copied provider logs that may contain them.

## Common refusals

| Refusal | Meaning | Response |
| --- | --- | --- |
| Calculated envelope exceeds `--max-usd` | The supplied cost model is over the operator's threshold | Reduce scope or stop |
| Price is stale or future dated | The estimate lacks valid price provenance | Re-read and correct the source |
| Revision is not a full SHA | The model input can move | Pin an exact revision |
| Image is `latest` | The serving environment can move | Pin a version and digest |
| Image has no digest | A tag can be repointed | Pin a digest, or explicitly accept the recorded risk |
| Capacity check fails | Approximate weights leave too little advertised memory | Change the accelerator plan |
| Warm containers are requested | Idle accelerator allocation is possible | Keep `min_containers` at zero |
| Proxy authentication is absent | Public traffic can schedule containers | Keep proxy authentication enabled |
| SGLang credential exposure is not accepted | The key can enter argv and logs | Use vLLM or explicitly review the exposure |

## Repository validation

The deployment test suite uses a fake Modal SDK and no real credentials,
downloads, API calls, or accelerator execution:

```bash
uv run pytest tests/deploy
```

It covers offline planning, required spending inputs, credential redaction,
paid-command withholding, CPU-only staging and verification, endpoint wiring,
and generated teardown commands.
