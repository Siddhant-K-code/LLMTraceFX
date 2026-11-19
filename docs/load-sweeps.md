# Load sweeps, goodput and scheduling diagnostics

`llmtracefx-optimizer sweep-api` sends the same request mix to an
OpenAI-compatible streaming endpoint at several offered loads and records,
for each load point:

- client-observed latency distributions (TTFT, first content, TPOT, E2E),
- throughput,
- goodput against optional TTFT, TPOT and E2E targets,
- diagnostics that show whether the load generator kept up,
- optionally, the server's own Prometheus window for the point.

`sweep-llama-cpp` and `sweep-mlx` do the same against llama.cpp's native
`/completion` endpoint and in-process mlx-lm; see
[Sweeps against llama.cpp and MLX](#sweeps-against-llamacpp-and-mlx).

Everything else in LLMTraceFX measures one request at a time. A sweep does
not. Read the next section before using its numbers.

## What a sweep can and cannot claim

Under load, requests share the server, so a request's latency includes time
spent behind other requests. A sweep result describes the server under a
given load. It does not describe any single request in isolation.

A sweep can claim:

- the latency distribution and goodput this client observed, for this
  request mix, at this offered load, against this endpoint, from this client
  host;
- that each per-request value is a real wall clock reading for that request
  while it shared the server with the other requests of its point;
- server-side counters and histograms for a point's window, when server
  metrics were paired and the window is attributed.

A sweep cannot claim:

- what any single request would cost on an idle server;
- how a request's time splits into queueing, prefill and decode, from client
  timings alone;
- that a latency change between points comes from the server rather than the
  client or the network. The scheduling diagnostics are there to rule out
  the client;
- comparability with single-request records.

The same lists are written into every `sweep.json` under `attribution`.

### Provenance

Values keep the existing provenance labels: latencies and offsets are
`measured_wall_clock`, token counts and the server's own per-request numbers
are `provider_reported`, and TPOT,
percentiles, rates and goodput are `derived`. No new label was added. What
separates sweep evidence from single-request evidence is:

- the artifact kind, `llmtracefx.load_sweep` (or `llmtracefx.load_sweep_plan`
  for a dry run);
- the `attribution` block, with `requests_isolated: false`;
- a `load_context` block on every point that names the load policy and the
  effective in-flight cap.

A sweep writes no `record.json`, so its requests never reach a reader that
expects isolated single-request evidence.

### Sweeps are refused by gate, compare and export

`gate`, `compare` and `export` refuse a sweep artifact by name. This covers a
`sweep.json` or `sweep_plan.json` given directly, a sweep file under another
name, and any directory that holds a sweep file somewhere below it. They
refuse rather than skip, because skipping would silently leave out results
the user asked for. Mixing sweep and single-request numbers needs an explicit
decision that none of these commands make. If you want to compare two
sweeps, read the two `sweep.json` files side by side; each point's
`load_context` says what was offered.

## Running a sweep

Closed loop, at 1, 4 and 16 requests in flight:

```bash
llmtracefx-optimizer sweep-api \
  --run-id qwen-sweep \
  --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --model-id Qwen/Qwen3-8B \
  --prompt-file prompt.txt \
  --concurrency 1,4,16 \
  --requests-per-point 64 \
  --stream-include-usage \
  --slo-ttft-ms 500 --slo-tpot-ms 50 --slo-e2e-ms 8000 \
  --output-dir results/sweep
```

Open loop, at 2, 4 and 8 requests per second with Poisson arrivals:

```bash
llmtracefx-optimizer sweep-api ... \
  --request-rate 2,4,8 --arrival poisson --arrival-seed 7 \
  --requests-per-point 64
```

Use `--arrival constant` for evenly spaced arrivals; it takes no seed. Add
`--dry-run` to write `sweep_plan.json` with the planned arrival offsets for
every point and send nothing.

Options worth knowing:

- `--prompts-jsonl` takes one `{"prompt": "..."}` object per line, and the
  requests of each point cycle through them in order. Any other key is
  refused.
- `--max-in-flight` caps requests in flight for a rate point (default 256).
  Arrivals held back by the cap are counted, and the point is marked
  degraded.
- `--cooldown-seconds` pauses between points so one point's tail does not
  reach the next.
- `--max-dispatch-lag-ms` sets the threshold for the scheduling diagnostics
  (default 10).
- `--stream-include-usage` asks the server for token usage in the stream.
  Without token counts, TPOT and token goodput are not available.

Remote endpoints are refused unless you pass `--allow-remote-load`. Only
`localhost`, `*.localhost` and loopback addresses count as local. A sweep
sends many requests, so pointing it at a paid API by accident is costly.

### Output

`<output-dir>/sweep.json` holds the whole sweep and is rewritten after each
point, so an interrupted sweep keeps the points it finished (`summary.finished`
is false and `summary.interrupted` is true). Each point has:

- `load_context`, `status` (`complete`, `failures` or `incomplete`) and
  `counts`, including failure categories and finish reason codes;
- `latency`, `throughput`, `goodput` (null without targets), `scheduling`
  and `dispatch`;
- `requests`, one row per request. Rows hold timings, token counts, the HTTP
  status and the raw finish reason. They hold no prompt or response text,
  only the prompt's sha256;
- `server_reported`, only when at least one succeeded request carried the
  server's own numbers (see [Per-request server numbers](#per-request-server-numbers)).

An output directory that already holds sweep output is refused.

The command exits 0 when every point finished with every request sent and
succeeded and every server metrics window was recorded. It exits 1 otherwise,
and 130 when interrupted. Degraded scheduling prints a warning but does not
change the exit code.

## Sweeps against llama.cpp and MLX

Two more commands run the same sweep against backends that do not speak the
OpenAI API. They use the same dispatcher, load flags, goodput, scheduling
diagnostics and `sweep.json` layout as `sweep-api`, and `gate`, `compare`
and `export` refuse their output in the same way. Each `attribution` block
gets a `backend` entry that names the backend, the pinned upstream commit,
what the backend reported about itself and the in-flight limit it supports.

Their rows are not comparable with `sweep-api` rows, even against the same
server or model. Neither command applies a chat template, and each
`attribution.cannot_claim` says so.

### `sweep-llama-cpp`

Sends each request to `llama-server`'s native `/completion` endpoint
(ggml-org/llama.cpp at `ebc9fb933e774bb4ac0ba497a67948e16b7e5656`).

```bash
llmtracefx-optimizer sweep-llama-cpp \
  --run-id llama-sweep \
  --endpoint http://127.0.0.1:8080 \
  --prompt-file prompt.txt \
  --max-output-tokens 256 \
  --concurrency 1,2,4 \
  --requests-per-point 32 \
  --output-dir results/llama-sweep
```

- `--endpoint` is the server root, including any API prefix. The command
  adds `/completion` and `/props` itself, and refuses a URL that already
  ends in `/completion` or has a query string.
- `--max-output-tokens` is required and is sent as `n_predict`.
- The model is whatever `llama-server` loaded. The server reports it only as
  a file path, which is not written; the backend block records
  `build_info` and `total_slots` from `/props`.
- The prompt is sent as is. There is no chat template and no system prompt.
- `cache_prompt` is sent as false unless you pass `--cache-prompt`.
  Sampling options (`--temperature`, `--top-p`, `--top-k`, `--seed`) are
  sent only when given.
- `--api-key-env` names an environment variable that holds a key for
  `--api-key`. The key is sent as a Bearer header and never written.
- Remote endpoints need `--allow-remote-load`, and non-local hosts need
  https, as with `sweep-api`.
- Before a real sweep, the command reads `/props`. A point that could have
  more requests in flight than the server's `total_slots` is refused, since
  the extra requests would wait in the server's queue. Pass
  `--allow-server-queue` to run it anyway; the choice is recorded in the
  backend block. A dry run sends nothing and does not read `/props`.
- The server metrics flags work as in `sweep-api` (start `llama-server`
  with `--metrics`).

Rows add the server's own accounting from the final chunk: `truncated`,
`id_slot`, `server_prompt_n`, `server_prompt_ms` and `server_predicted_ms`.
The server timers are `measured_native`. They exclude queueing and network
time, so they are not the client's TTFT and decode time. Token counts
(`tokens_predicted`, `tokens_evaluated`, `timings.cache_n`) are
`provider_reported`. With speculative decoding, `server_draft_tokens` and
`server_accepted_draft_tokens` hold `timings.draft_n` and
`timings.draft_n_accepted`, also `provider_reported`. The server sends them
only when it drafted at least one token
([`server_slot_stats::to_json`](https://github.com/ggml-org/llama.cpp/blob/ebc9fb933e774bb4ac0ba497a67948e16b7e5656/tools/server/server-common.cpp#L146-L167)),
so without a draft model they stay null. No acceptance rate is derived from
them. `finish_reason` holds the raw `stop_type`: `eos` and
`word` count as `stop`, `limit` counts as `length`, and anything else fails
the request as `unrecognized_stop_type`.

Two things to know when reading these fields:

- `tokens_predicted` counts the end-of-sequence token, which the server
  streams as a chunk with empty content. `sweep-mlx` and `collect-mlx` do
  not count it, so for the same text an `eos` stop reports one more output
  token here than under MLX.
- With more than one slot busy, `llama-server` evaluates prompts in a shared
  batch and each slot's `server_prompt_ms` is the time of that whole batch.
  Do not sum it across rows; the server metrics window has the server's
  own total.

`llama-bench` is not a worker. It runs its own fixed prompt and generation
lengths and takes no requests, so the dispatcher cannot drive it.

### `sweep-mlx`

Runs mlx-lm in the same process (ml-explore/mlx-lm at
`c4c9c762cc797fd979fc98645f5db5112753bbe9`, ml-explore/mlx at
`fbedcee5f4a0c7b2cbb642ea91d4c555460d3e48`).

```bash
llmtracefx-optimizer sweep-mlx \
  --run-id mlx-sweep \
  --model-path models/Qwen3-8B-4bit \
  --model-id qwen3-8b-4bit \
  --prompt-file prompt.txt \
  --request-rate 0.5,1 --max-in-flight 1 \
  --requests-per-point 32 \
  --output-dir results/mlx-sweep
```

`--model-path` must be a local directory. A Hugging Face repo ID such as
`mlx-community/Qwen3-8B-4bit` is refused, so download the model first.

In-process mlx-lm runs one generation at a time. `stream_generate` does not
batch, and its generation stream belongs to the thread that loaded the
model. So the command:

- refuses any point that could have more than one request in flight. Closed
  loop takes only `--concurrency 1`, and rate points need
  `--max-in-flight 1`. Arrivals held back by that cap show up as dispatch
  lag in the scheduling diagnostics;
- loads the model and runs every generation on one owner thread;
- refuses, rather than queues, a request that arrives while another is
  running. It would be a `client_error` row, and should not happen with the
  limit above.

Other behavior:

- Tokenization is inside TTFT. The seed is set before every request.
- Token counts use the `collect-mlx` rule and are `measured_native`. There
  is no prompt cache, so `cached_prompt_tokens` is empty.
- `--request-timeout` is checked between tokens.
- A failed generation keeps only the exception type, in
  `runtime_error_type`.
- There is no warmup, so the first request includes any one-time cost.
  There is no draft model, chat template, server metrics or
  `--allow-remote-load`, since nothing goes over the network.
- A dry run does not load the model, and `--model-path` is optional for it.
  The backend block records the model load time as `measured_wall_clock`.
- The model runs in the load generator's process, so process CPU time
  cannot separate client work from model work. `client_cpu_seconds` and
  `client_cpu_utilization` are `null`, `client_cpu_withheld_reason` says
  why, and client CPU never marks an MLX point degraded. Dispatch lag and
  slot refill are still judged.

### Checked on real hardware

Both commands were run on an Apple M5 Pro against
Qwen2.5-0.5B-Instruct: the Q4_K_M GGUF under `llama-server` built at the
pinned commit (`build_info` `b12485-ebc9fb933`, `--parallel 2 --metrics`),
and `mlx-community/Qwen2.5-0.5B-Instruct-4bit` under mlx 0.32.2 and
mlx-lm 0.31.3 (the releases in `pyproject.toml`, older than the pinned
commits). What matched what the backends did:

- llama.cpp: per-row `output_tokens` and `prompt_tokens` summed to the
  `/metrics` `tokens_predicted` and `prompt_tokens` deltas. `eos` and
  `limit` mapped to `stop` and `length`, a full context returned `limit`
  with `truncated` true, and `--cache-prompt` reported `cache_n` as
  `cached_prompt_tokens` with `server_prompt_n` 1. A point with more
  requests in flight than `total_slots` was refused before anything was
  written, and with `--allow-server-queue` the queued rows showed TTFT far
  above `server_prompt_ms`.
- MLX: `prompt_tokens` matched the tokenizer, an `eos` stop counted 1 token
  for "Yes" and a `length` stop counted `--max-output-tokens`. Concurrency 2
  and rate points without `--max-in-flight 1` were refused, timeouts were
  recorded per request, and the backend block named the accelerator.
- `gate`, `compare` and `export` refused both commands' `sweep.json`, its
  directory and a renamed copy.

## Definitions

- **TTFT** is the time from sending the request to the first generated
  delta, reasoning included. This matches the OTLP export's
  `gen_ai.response.time_to_first_chunk`.
- **First content** is the time to the first visible content delta.
- **E2E** is the time from sending the request to the end of the stream.
- **TPOT** is `(e2e_ms - ttft_ms) / (output_tokens - 1)`. It uses the
  provider's output token count, so it is `derived`, and it is not
  available when the provider did not report one. For a response of one
  token or less it is not applicable.
- **Percentiles** use the nearest rank method over the requests that
  succeeded.
- **The measured window** for a point runs from the first dispatch to the
  last completion. Rates divide by it.

## Goodput

Goodput is the share of requests that met every stated target. Targets are
given with `--slo-ttft-ms`, `--slo-tpot-ms` and `--slo-e2e-ms`; any subset is
allowed, and only stated targets are checked. A value equal to its target
meets it.

Every dispatched request gets one verdict, checked in this order:

1. `failed`: the request did not succeed, including HTTP errors and client
   errors such as timeouts.
2. `missed`: a stated target was exceeded.
3. `unverifiable`: a stated target could not be checked because the value is
   missing.
4. `met`: every stated target was checked and met.

A response of one token or less has no TPOT. It counts as meeting the TPOT
target and is counted separately under `tpot_not_applicable_met`.

The denominator is every dispatched request. Failed requests stay in it and
count against goodput. Requests that were never sent are counted as
`unsent`. They are not in the denominator, but they make the point
incomplete.

Values are withheld, never guessed:

- When some requests were never sent, the fractions are withheld.
- When some requests are `unverifiable`, the headline fraction is withheld and
  `request_goodput_fraction_bounds` gives the lower bound (treat them as
  missed) and the upper bound (treat them as met).
- Goodput per second is withheld when there is no measured window.
- Token goodput is the share of reported output tokens that came from
  requests that met their targets. It is withheld when any dispatched request
  lacks a token count, which includes most failed requests. Token goodput per
  second needs token counts only for the requests that met their targets.

Every withheld value carries a `withheld_reason`.

## Scheduling diagnostics

A slow load generator looks like a slow server. Each point's `scheduling`
block reports:

- `dispatch_lag` (open loop): how late each request left compared with its
  planned arrival, as count, p50, p99, max and the number over the threshold;
- `slot_refill` (closed loop): the gap between a request finishing and the
  next one starting in its slot;
- `max_in_flight`, `max_observed_in_flight` and `deferred_by_in_flight_cap`;
- `client_cpu_seconds` and `client_cpu_utilization` for the window, or
  `null` with a `client_cpu_withheld_reason` when the backend shares the
  client's process (`sweep-mlx`);
- `dispatch_rate` (open loop): the target rate, the rate of the planned
  schedule and the achieved dispatch rate.

`status` is `degraded` when the dispatch lag or slot refill p99 is over the
threshold, when any arrival was held back by the in-flight cap, or when the
client used more than 0.9 CPU seconds per wall second. `reasons` lists which.
CPython threads share one interpreter lock, so a busy client shows up here
before it shows up anywhere else.

A Poisson schedule over a finite run does not hit its target rate exactly.
Compare the achieved rate with the planned schedule's rate, not only with the
target.

## Per-request server numbers

A Prometheus window covers every request in it, so under load it cannot say
which request queued or how many drafts one request accepted. Some engines
return those numbers inside the response to each request. `collect-api`,
`sweep-api` and `replay run` read them when they are there and record
nothing when they are not; no field is filled with zeros.

| Engine | What it sends | How to get it |
| --- | --- | --- |
| vLLM | `metrics` on the final usage chunk ([protocol](https://github.com/vllm-project/vllm/blob/1ae1b64807ff6be2a0d3f942dbad0a59a663a02a/vllm/entrypoints/generate/base/protocol.py#L105-L138), [docs](https://github.com/vllm-project/vllm/blob/1ae1b64807ff6be2a0d3f942dbad0a59a663a02a/docs/features/per_request_metrics.md)) | Start vLLM with `--enable-per-request-metrics` (and `--per-request-spec-decode-metrics summary` for the speculative block). Pass `--stream-include-usage` (on by default for `replay run`). |
| SGLang | `sglext` with `spec_tokens_details` and `cached_tokens_details` ([chunk](https://github.com/sgl-project/sglang/blob/b6d4389a77970e31817536c3e182256fae6a6f73/python/sglang/srt/entrypoints/openai/serving_chat.py#L2690-L2761), [request fields](https://github.com/sgl-project/sglang/blob/b6d4389a77970e31817536c3e182256fae6a6f73/python/sglang/srt/entrypoints/openai/protocol.py#L1038-L1039)) | Pass `--sglang-return-details`, which adds `return_spec_tokens_details` and `return_cached_tokens_details` to the request body. |
| TensorRT-LLM | a `trtllm.perf_metrics` SSE event after `[DONE]`, holding the `Server-Timing` and `X-TRTLLM-*` values ([perf_metrics.py](https://github.com/NVIDIA/TensorRT-LLM/blob/e6ecd280fd1f77cad6496f81a7194afe29f8d5e8/tensorrt_llm/serve/perf_metrics.py#L322-L400), [streaming](https://github.com/NVIDIA/TensorRT-LLM/blob/e6ecd280fd1f77cad6496f81a7194afe29f8d5e8/tensorrt_llm/serve/perf_metrics.py#L842-L914)) | Start the server with `return_perf_metrics` ([server](https://github.com/NVIDIA/TensorRT-LLM/blob/e6ecd280fd1f77cad6496f81a7194afe29f8d5e8/tensorrt_llm/serve/openai_server.py#L760-L766)) and pass `--trtllm-return-metrics`, which sends `X-TRTLLM-return-metrics: 1`. |
| llama.cpp | `timings.draft_n` and `timings.draft_n_accepted` | `sweep-llama-cpp`, see above. |

Both opt-in flags change the request, so each is part of the config hash, the
recorded command and, for sweeps, the `workload` block. A run with a flag is
never compared silently with a run without it. Without the flags nothing
changes in the request or in the output.

Rows gain `server_metrics_source` (the sources joined with `+`) and these
fields, all `provider_reported` and null when the server did not send them:

| Field | vLLM | SGLang | TensorRT-LLM |
| --- | --- | --- | --- |
| `server_queue_ms` | `queue_time_ms` | | `server_queue` |
| `server_ttft_ms` | | | `server_ttft`, from arrival, so it includes the queue |
| `server_scheduled_ttft_ms` | `time_to_first_token_ms`, from scheduling, so it excludes the queue | | |
| `server_decode_ms` | `generation_time_ms` | | |
| `server_e2e_ms` | | | `server_e2e` |
| `server_mean_itl_ms` | `mean_itl_ms` | | |
| `server_tokens_per_second` | `tokens_per_second` | | |
| `server_draft_tokens` | `num_draft_tokens` | `spec_num_proposed_drafts` | |
| `server_accepted_draft_tokens` | `num_accepted_draft_tokens` | `spec_num_correct_drafts` | |
| `server_draft_acceptance_rate` | `draft_acceptance_rate` | `spec_accept_rate` | |
| `server_mean_acceptance_length` | `mean_acceptance_length` | `spec_accept_length` | |
| `server_verify_steps` | `num_spec_steps` | `spec_verify_ct` | |

The two TTFT fields are kept apart because the engines start the clock at
different points. Both acceptance lengths count the bonus token. None of
these replaces the client's `ttft_ms` or `e2e_ms`: the server's clock starts
when the request reaches it and ends when it hands off the last token, so it
leaves out the network and the client. SGLang's speculative values are kept
only when `spec_verify_ct` is above zero, because the server defaults every
one of them to 0 when no verify step ran.

`collect-api` writes the parsed block to `api_evidence.json` under
`server_reported`, with the normalized values, the raw fields each source
sent (token ids and routed experts are never kept) and `malformed_fields`,
the names of values that could not be read (not a finite, non-negative
number of the expected kind). A malformed
value is named and left out, and never fails the request. A sweep point
adds a `server_reported` block with each field's distribution over the
succeeded requests that had it.

TensorRT-LLM sends its event after `[DONE]`. The client's completion time is
taken at `[DONE]`, so waiting for the event does not add to `e2e_ms`. The
wait is capped at 5 seconds and 256 KiB, and a broken or missing event
leaves the request successful with no TensorRT-LLM values.

## Pairing server metrics

Pass `--server-metrics-url http://127.0.0.1:8000/metrics` to take a
Prometheus snapshot before and after each point. Each window is written to
`server_metrics_point_<k>.json`, and the point records its file name, sha256
and attribution status. The window is judged the same way as the
`server-metrics` command, with the expected request count set to the number of
requests that succeeded. If the server also counted failed requests, the
window comes out `contaminated`, which is the honest result. A point where
nothing succeeded gets no window. A failed snapshot is recorded as an error
on its point, the sweep goes on, and the exit code is 1. For `trtllm-serve`,
pass its `/prometheus/metrics` URL and enable `return_perf_metrics`, or every
window comes out `stats_not_enabled`.

## Prefix cache warning

A single prompt file repeats the same prompt for every request. Servers with
prefix caching will serve most of the prompt from cache after the first
request, so TTFT will look better than it would for distinct prompts. Rows
record `cached_prompt_tokens` when the server reports it. Use
`--prompts-jsonl` with distinct prompts when you want uncached prefill under
load.

## Reusing the dispatcher

The load generator lives in `llmtracefx/optimizer/load/dispatcher.py` and
knows nothing about HTTP or sweeps. Trace replay builds on it.

- `ClosedLoop(concurrency)` keeps a fixed number of requests in flight.
- `OpenLoopRate(rate, distribution, seed, max_in_flight)` sends at a target
  rate with Poisson or constant arrivals.
- `TimedArrivals(offsets, max_in_flight)` sends at given offsets in seconds.
  Requests with equal offsets are released together.
- `SessionArrivals(turns, max_in_flight)` sends each session's first turn at
  its offset, and each later turn only after the previous turn of its session
  completes, plus its delay or not before its offset.

`dispatch(requests, send, policy, ...)` calls `send(request, context)` from
worker threads and returns a `DispatchReport` with one outcome per request,
the scheduling diagnostics, the effective `max_in_flight`, the UTC start time
and the dispatcher name and version. `load/api_worker.py` provides
`ApiRequestSpec` (prompt, system prompt, per-request `max_tokens` and
`extra_body`) and `make_api_sender`, which streams one chat completion per
request through the same client as `collect-api`.
