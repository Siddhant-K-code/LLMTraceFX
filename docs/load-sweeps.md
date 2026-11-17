# Load sweeps, goodput and scheduling diagnostics

`llmtracefx-optimizer sweep-api` sends the same request mix to an
OpenAI-compatible streaming endpoint at several offered loads and records,
for each load point:

- client-observed latency distributions (TTFT, first content, TPOT, E2E),
- throughput,
- goodput against optional TTFT, TPOT and E2E targets,
- diagnostics that show whether the load generator kept up,
- optionally, the server's own Prometheus window for the point.

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
`measured_wall_clock`, token counts are `provider_reported`, and TPOT,
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
  only the prompt's sha256.

An output directory that already holds sweep output is refused.

The command exits 0 when every point finished with every request sent and
succeeded and every server metrics window was recorded. It exits 1 otherwise,
and 130 when interrupted. Degraded scheduling prints a warning but does not
change the exit code.

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
- `client_cpu_seconds` and `client_cpu_utilization` for the window;
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

## Pairing server metrics

Pass `--server-metrics-url http://127.0.0.1:8000/metrics` to take a
Prometheus snapshot before and after each point. Each window is written to
`server_metrics_point_<k>.json`, and the point records its file name, sha256
and attribution status. The window is judged the same way as the
`server-metrics` command, with the expected request count set to the number of
requests that succeeded. If the server also counted failed requests, the
window comes out `contaminated`, which is the honest result. A point where
nothing succeeded gets no window. A failed snapshot is recorded as an error
on its point, the sweep goes on, and the exit code is 1.

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

`dispatch(requests, send, policy, ...)` calls `send(request, context)` from
worker threads and returns a `DispatchReport` with one outcome per request,
the scheduling diagnostics, the effective `max_in_flight`, the UTC start time
and the dispatcher name and version. `load/api_worker.py` provides
`ApiRequestSpec` (prompt, system prompt, per-request `max_tokens` and
`extra_body`) and `make_api_sender`, which streams one chat completion per
request through the same client as `collect-api`.
