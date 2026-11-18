# Ecosystem gap analysis

This note compares LLMTraceFX with the wider inference benchmarking and
observability ecosystem. It covers what users of neighbouring tools expect,
what LLMTraceFX already does that they do not, and which channels would make
the project easier to find. It closes with the two features this analysis
led to: `server-metrics` and `export`.

The survey was carried out in late 2026 against upstream source where
possible. Metric names for vLLM, SGLang and llama.cpp were checked against
current source (`vllm/v1/metrics/loggers.py`,
`python/sglang/srt/observability/metrics_collector.py`, and the llama.cpp
server README). TGI names come from its Rust source as summarised by the
research pass and are less thoroughly verified.

## Tools compared

| Tool | Load model | Server-side signals | Goodput or SLO | Energy | Correctness | Provenance |
|---|---|---|---|---|---|---|
| `vllm bench serve` | rate, Poisson or gamma, concurrency, ramp | partial (`/metrics` for spec decode) | yes | no | no | CLI echoed in JSON |
| SGLang `bench_serving` | as vLLM, plus Mooncake replay | `--cache-report` | yes, rate and attainment | no | no | CLI echoed |
| NVIDIA AIPerf (GenAI-Perf successor) | concurrency, rate, trace replay, sweeps | `--server-metrics` Prometheus scrape, `--gpu-telemetry` | yes | GPU power via DCGM or NVML | partial | `run_info` with seed, sweep and trial |
| GuideLLM (vllm-project) | rate, concurrency, sweep, warmup windows | collector metrics from Prometheus | richest model (request and token goodput) | no | no | windowed exclusion |
| HF inference-benchmarker | sweep, open-loop rate, closed-loop VUs | no | no | no | no | config driven |
| Ray LLMPerf (archived Dec 2025) | fixed concurrency | no | no | no | separate correctness script | none |
| `llama-bench` | single stream | no | no | no | no | repetitions |
| Ollama response timings | none (per request fields) | no `/metrics` endpoint | no | no | no | none |
| mlx-lm server benchmark | concurrency | no | no | no | no | none |
| MLPerf Inference | formal scenarios | n/a | latency caps | MLPerf Power | mandatory accuracy | audited |
| LLMTraceFX | single request; **now: concurrency and rate sweeps** (`sweep-api`) | **now: `server-metrics`** | **now: request and token goodput** (`sweep-api`) | Apple Instruments, **now: NVML (`gpu-energy`)** | every workload verified | per value provenance |

## Gaps, ranked by how strongly users expect them

1. **Concurrency and request rate sweeps.** Table stakes for every serving
   benchmark. LLMTraceFX measures one request at a time by design. A sweep
   mode would need its own attribution story because per request timings
   stop being isolated. **Built in this branch as `sweep-api`.**
2. **Goodput and SLO attainment.** vLLM, SGLang, GuideLLM and AIPerf report
   the fraction of requests or tokens that meet joint TTFT, TPOT and E2E
   targets. This depends on item 1. **Built in this branch as part of
   `sweep-api`.**
3. **Server-side metrics during a run.** GuideLLM and AIPerf scrape the
   server's Prometheus endpoint so users can tell "the client saw slow TTFT"
   apart from "the server queue was the bottleneck". **Built in this branch
   as `server-metrics`.**
4. **Native vLLM, SGLang, TGI and Ollama collectors.** Partly addressed by
   item 3, which reads the server's own histograms instead of inferring
   phases at the client.
5. **GPU telemetry and energy (NVML or DCGM).** LLMTraceFX covers Apple
   silicon through Instruments. **Built in this branch as `gpu-energy`, for
   NVIDIA power and energy through NVML, with dcgm-exporter as a second
   source.**
6. **Trace replay (Mooncake, ShareGPT timed replay, BurstGPT).** SGLang,
   AIPerf and AISBench all added Mooncake replay within the last year.
   **Built in this branch as `replay`.**
7. **Prefix cache hit rate under load.** `server-metrics` now reports the
   vLLM `prefix_cache_queries` and `prefix_cache_hits` counter deltas, and
   SGLang's `cached_tokens_total`, for the measured window.
8. **Export to observability backends.** Datadog, Honeycomb, Langfuse,
   Grafana (through OpenLIT), MLflow and W&B Weave ingest OTLP spans that use
   the OpenTelemetry GenAI semantic conventions. Perfetto is the common
   viewer for timelines. **Built in this branch as `export otlp`,
   `export otlp-metrics` and `export perfetto`.**
9. **Reasoning aware first token timing.** AIPerf separates time to first
   token from time to first output. The OTLP export follows the GenAI
   convention for `gen_ai.response.time_to_first_chunk`, which counts
   reasoning chunks, and keeps the first visible content offset as a
   separate attribute.
10. **Scheduling fidelity diagnostics.** GuideLLM reports dispatch and
    scheduling delay so users can trust the load generator itself. Only
    relevant once item 1 exists. **Built in this branch as part of
    `sweep-api`.**

Importers for `vllm bench serve --save-result` JSON and AIPerf
`profile_export_aiperf.json` let users bring existing results into
LLMTraceFX without rerunning. Both formats carry percentile fields
(`p99_ttft_ms`, per metric `avg`, `p50`, `p99`). These are statistics another
tool computed, so they feed the perf gate rather than `compare`, and they carry
their own provenance labels. See "`import vllm-bench` and `import aiperf`"
below.

## What LLMTraceFX already does that others do not

- **Provenance on every value.** AIPerf classifies metric types, and
  GenAI-Perf splits raw and derived files, but no surveyed tool labels each
  number with how it was obtained.
- **Refusal as an outcome.** Other tools print a number. LLMTraceFX returns
  `inconclusive` or `unsupported` when the evidence does not support one.
  `server-metrics diff` follows this: a window that other traffic touched is
  reported as contaminated and its highlights are withheld.
- **Single objective `tune` without composite scores.**
- **Verified deterministic workloads.** Correctness is checked on the same
  request that was timed.
- **Local and hosted runs in one `compare`.** Most tools are either local
  only or server only.
- **Apple Instruments Metal intervals and the KV cache truth auditor.**
- **Offline, self-contained HTML reports.**

## Adoption and discoverability

Ranked by expected return for effort:

Items 1 to 5 are now done; see below.

1. **Publish a perf gate GitHub Action** that runs a pinned workload, compares
   against a committed baseline and fails a pull request on regression. A
   similar action (`waynehacking8/inference-perf-gate`) shows demand but has
   no provenance model.
2. **Submit to `tensorchord/Awesome-LLMOps`** (observability and performance
   sections) and **`rafska/Awesome-local-LLM`** (benchmarks section). Both
   have light PR processes.
3. **Submit to an `awesome-mlx` list** (`antranapp/awesome-mlx` already lists
   `mlx-bench`).
4. **Add PyPI and CI status badges** to the README header, as GuideLLM does.
5. **Host a small Hugging Face Space** that renders published evidence. The
   LLM-Perf leaderboard shows this is the strongest distribution channel for
   a measurement tool.
6. **OTLP export makes LLMTraceFX visible inside tools people already run.**
   A run exported with `export otlp` can be loaded by an OpenTelemetry
   Collector (`otlpjsonfile` receiver) and forwarded to Datadog, Honeycomb,
   Langfuse or Grafana Tempo.
7. The OpenTelemetry ecosystem registry is frozen
   (open-telemetry/opentelemetry.io#11377). Track it rather than submitting.
8. LLMPerf and its leaderboard were archived in Dec 2025 and Jan 2026. Its
   former users need a home for correctness-checked performance testing,
   which LLMTraceFX already offers.

## Features built from this analysis

### `server-metrics`

`server-metrics snapshot` saves one scrape of a server's Prometheus endpoint.
`server-metrics diff` compares two snapshots taken around a measured request
and writes server-side evidence: counter deltas, histogram count, sum and
bucket deltas, and gauge values. Server histograms are labelled
`measured_native`, the same provenance the llama.cpp parser gives the
runtime's own timers. Bucket-bound quantiles are `estimated`, and means are
`derived`.

A scrape is server wide. Every delta counts all traffic in the window, not
only the measured request. The diff therefore checks the server's own
request counter against the expected request count and reports one of
`attributed`, `contaminated`, `incomplete`, `counter_reset`, `unverifiable`
or `operator_asserted`. Only `attributed` and `operator_asserted` windows
produce highlights. llama.cpp exposes no request counter, so its windows are
`unverifiable` unless the operator asserts isolation.

`collect-api --server-metrics-url` takes the two snapshots automatically
around a single request. It re-scrapes for up to
`--server-metrics-settle-timeout` seconds while the server has not yet counted
the request, and it deletes any `server_metrics_evidence.json` left by an
earlier run in the same directory. `export` also refuses server evidence whose
scrape window does not bracket the run.

### `export otlp` and `export perfetto`

`export otlp` turns a record, and optionally its API evidence and server
metrics evidence, into one OTLP/JSON `TracesData` line. The output follows the
file exporter format, so the OpenTelemetry Collector `otlpjsonfile` receiver
can read it. Spans use the GenAI semantic conventions (`gen_ai.operation.name`,
`gen_ai.provider.name`, `gen_ai.request.model`, `gen_ai.usage.*`,
`gen_ai.response.time_to_first_chunk`). Every timing keeps its provenance as
`llmtracefx.*` attributes.

`export perfetto` writes the same spans as Chrome trace event JSON for
ui.perfetto.dev or `chrome://tracing`.

API runs use real client offsets for their child spans. Local runs only
record phase durations, so their phases are laid out back to back and marked
`llmtracefx.span.placement=sequential_from_durations`. That layout is an
assumption, not a measurement.

### `export otlp-metrics`

`export otlp-metrics` aggregates API runs into one OTLP/JSON `MetricsData`
line in the same file exporter format. Names, units, bucket boundaries and
attributes follow the GenAI metrics in
`open-telemetry/semantic-conventions-genai` at commit `e6658f1`, where the
GenAI conventions now live. It records the client inference metrics
(`gen_ai.client.inference.duration`, `.time_to_first_chunk`,
`.time_per_output_chunk`, `.operation.input_tokens` and `.output_tokens`) and
the server metrics (`gen_ai.server.time_to_first_token`,
`.time_per_output_token` and `.request.duration`). The conventions say
`gen_ai.client.operation.duration` should not be reported for inference, and
`gen_ai.client.token.usage` has been replaced, so neither is emitted.

Each point adds `llmtracefx.provenance` and `llmtracefx.source`, so values
measured in different ways never share a histogram, and each run is an
exemplar that points at its `export otlp` root span. Failed runs keep their
duration and token points with `error.type`. Server values come only from an
attributable window that held exactly one request, from vLLM and SGLang
histograms whose definitions match the conventions, and for TGI only from the
end to end duration. These mappings were checked against vLLM `c26c9bf`,
SGLang `e55e2ee` and TGI `b4adbf2`.

Where it departs from the conventions on purpose:

- Missing token usage is left out. The conventions ask for 0.
- vLLM's TPOT of 0 for a one-token reply is left out, since no time per token
  was measured.
- TGI's per-token mean divides by all generated tokens and includes prefill,
  so it is not reported as `gen_ai.server.time_per_output_token`.
- Local records are skipped. Their phases are not one inference call.
- The `gen_ai.client.inference.usage.*` counters are not emitted. They are
  sums, not histograms, and the records do not say which modality the tokens
  were, so every value would be `unknown`.

The output was checked with OpenTelemetry Collector contrib 0.165.0: the
`otlpjsonfile` receiver parsed it with no errors, the `prometheus` exporter
produced the expected `_bucket`, `_sum` and `_count` series, and the `file`
exporter wrote it back unchanged apart from the receiver's own file name
metadata.

### `gate` and the perf gate Action

`gate` compares candidate records with baseline records and exits 0 (pass),
1 (regression) or 2 (inconclusive). Both groups must share the model,
quantization, runtime name and backend, platform, workload hash and token
counts. The runtime version and config hash may differ, because those are
usually what the pull request changes. The metric's unit and provenance must
match on both sides. A regression is reported only when the relative change
exceeds `--max-regression` and the absolute difference exceeds the summed
population standard deviations of the two groups. Otherwise a worse result
is inconclusive and the report asks for more runs. `decode_tokens_per_second`
is derived from `timing.decode` and the generated token count, and it is
labelled `derived_from_<provenance>`.

`action.yml` at the repository root wraps the command as a composite action.
The gate imports only the standard library, so the action installs nothing
beyond Python. It writes a Markdown table to the job summary and exposes
`verdict` and `report` outputs. The `perf gate action self-test` CI job runs
the action against committed fixtures and checks all three verdicts.

### `import vllm-bench` and `import aiperf`

`llmtracefx-optimizer import vllm-bench <file>` reads a
`vllm bench serve --save-result` JSON file, and
`llmtracefx-optimizer import aiperf <file>` reads an AIPerf
`profile_export_aiperf.json` (schema versions 1.0 to 1.7 and older files
without a version). Each writes an `llmtracefx.imported_benchmark` artifact.
The artifact records the source tool, its version when the file states one
(vLLM files do not), the source schema version, the source file name and the
sha256 of its bytes.

Every value keeps the key it came from. Statistics the tool computed (means,
medians, percentiles, standard deviations, rates) are labelled
`imported_derived`. Counts and totals the tool reported are labelled
`imported_reported`. Neither is `derived`, which in LLMTraceFX means
LLMTraceFX computed the value, and neither is a measurement. Keys the importer
does not map are listed in `unmapped_keys`. Load settings it cannot find are
listed in `load_unresolved`. Nothing missing is filled in. For example, vLLM
writes only the percentiles passed to `--percentile-metrics`, so a file saved
with the default `99` has no p50.

The importers refuse a file rather than guess when it is not one JSON object
(`--append-result` files hold several), holds NaN or Infinity, has the wrong
type for a key, has no successful requests, is a pooling or embedding result,
or comes from the other tool. AIPerf files are also refused when the run was
cancelled or incomplete, used more than one model, counted more than one phase
toward results, or are a timeslice, aggregate or collated export. Endpoint
URLs, headers and API keys are never copied.

Imported artifacts go to `gate`, not `compare`. `compare` reads sealed
`workloads run` directories and only compares systems that ran the identical
workload: same prompt hashes, evaluator, output cap and sampling, with a
correctness result for every row. Neither export has per-prompt hashes or
correctness results, so an imported file cannot join a comparable unit, and a
unit of its own would compare nothing. Imports are not turned into
`ExperimentRecord`s either, because a record holds per-run phase measurements
and platform facts that these files do not have.

With imported artifacts a user can archive a provenance-tagged summary of an
existing run and gate a pull request on N baseline and N candidate runs of the
same tool, model, endpoint, load and token totals, for example
`--metric ttft_ms.p99` or `--metric time_to_first_token.p99`. Each imported
file counts as one run, and the same file cannot be counted twice. A run with
failed requests, an unstated failure count or unresolved load is excluded and
listed. Imports cannot be used with `compare`, `tune` or the lab, cannot be
mixed with LLMTraceFX records in one gate, and vLLM and AIPerf results cannot
be gated against each other, because the two tools define their metrics
differently. Neither format has per-request data or correctness results, so
neither is available from an import.

### `gpu-energy`

`gpu-energy record` runs one command and reads NVIDIA power and energy
through NVML (the optional `nvml` extra). It reads each GPU's total energy
counter before and after the command and polls power while it runs. The
counter delta is `energy`, labelled `measured_native`. A trapezoid over the
power samples is `sampled_power_integral`, labelled `estimated`, and is never
called energy. A counter that went backwards, did not move or could not be
read leaves `energy` null with a reason. Too few samples, a failed read or a
full sample buffer does the same for the sampled values.

NVML readings are device wide. On each poll the collector also lists the GPU
processes and walks their parent PIDs. A process outside the command, its
children and any `--gpu-pid` makes the window `contaminated`. A window that
cannot be checked (PID namespaces, MIG, a failed listing, or the command never
listed) is `unverifiable` unless the operator passes `--assume-isolated`.
Only attributable windows of a command that exited 0 produce highlights.
`gpu-energy capability` reports what the machine supports and exits 3 when
NVML is unavailable, including on macOS.

NVML is the default source. `--source dcgm-exporter` reads a running
dcgm-exporter instead (pinned against v4.9.0, `4e663a5`, and v4.8.4,
`fafd151`). Its scrapes carry no timestamp and it refreshes every 30 s by
default, so `record` brackets the run with the first counter change seen
before and after it and states the bracket. The energy delta is refused with
`bracket_too_wide` when the command is under 90 % of the widest span the
delta can cover. Power is listed per scrape but not integrated, and the
window is `unverifiable` because the exporter has no process list. The DCGM
Python bindings are not on PyPI and need `nv-hostengine`, and `dcgmi dmon`
prints no timestamps, so neither is used.

### `sweep-api`

`llmtracefx-optimizer sweep-api` runs one request mix at several
concurrency levels (closed loop) or request rates (open loop, Poisson or
constant arrivals) against an OpenAI-compatible endpoint, using the same
streaming client as `collect-api`. Each point reports nearest rank latency
percentiles for TTFT, first content, TPOT and E2E, throughput, and goodput.

Goodput is the share of dispatched requests that met every stated TTFT, TPOT
and E2E target. Failed requests stay in the denominator. A request whose value
for a stated target is missing is unverifiable: the headline fraction is then
withheld and lower and upper bounds are reported instead. Token goodput is
withheld when any request lacks a provider token count.

Requests in a sweep overlap, so a sweep is its own artifact kind,
`llmtracefx.load_sweep`, with an `attribution` block that lists what it can
and cannot claim. It writes no `record.json`, and `gate`, `compare` and
`export` refuse sweep artifacts rather than mix them with single-request
records. Values keep the existing provenance labels.

Each point also carries scheduling diagnostics (dispatch lag, closed loop slot
refill, arrivals held by the in-flight cap, client CPU use) and is marked
degraded when the client could not keep to its schedule.
`--server-metrics-url` pairs each point with a Prometheus window. The load
generator is a separate module, `load/dispatcher.py`, that also supports
arrivals at given timestamps for trace replay. See
[load-sweeps.md](load-sweeps.md).

`sweep-llama-cpp` (llama.cpp's native `/completion`) and `sweep-mlx`
(in-process mlx-lm) reuse the same dispatcher and artifact. `sweep-mlx`
refuses more than one request in flight, and `sweep-llama-cpp` refuses more
than the server's slots unless told to queue. Each records its limit in the
attribution block.

### `replay`

`llmtracefx-optimizer replay validate|plan|run` reads Mooncake JSONL,
BurstGPT CSV and AIPerf `single_turn` JSONL. Field names follow the upstream
sources: the Mooncake `FAST25-release` traces, the BurstGPT README, and
AIPerf's `SingleTurn` model. There is no separate "ShareGPT timed" format
upstream; AIPerf `single_turn` with `timestamp` and `text` is how other tools
replay ShareGPT prompts at given times, so that is the format read here.

Parsing is strict. A row with a missing, extra or mistyped field, a timestamp
that goes backwards, or a Mooncake hash id that names two different block
lengths refuses the whole trace with its line number. Nothing is skipped or
repaired. BurstGPT rows with zero tokens are refused unless
`--exclude-failed` drops them and counts them by reason.

The schedule records the trace sha256, the window, the time scale and the
prompt salt. Mooncake prompts are built from `hash_ids`, so shared prefixes in
the trace are shared prefixes on the wire, and the plan reports the share of
prompt tokens that could at best be served from a prefix cache. Prompt
lengths are word counts labelled `estimated`, and the provider's
`prompt_tokens` and `cached_tokens` are recorded next to them. With
`--tokenizer` (optional `tokenizer` extra) every synthetic prompt is encoded
and checked to be exactly the trace length in that tokenizer, labelled
`tokenizer_exact`, with the tokenizer.json sha256 and hub commit recorded.
The chat template is not counted.

`run` sends requests through the shared load dispatcher at the scheduled
offsets and checks itself: dispatch lag percentiles go into the result, and
the verdict is `faithful`, `lagged` (p99 lag over `--max-dispatch-lag-s`) or
`incomplete`. With `--server-metrics-url` the run is bracketed by
`server-metrics` snapshots, so vLLM's prefix cache counters can be read
against the trace's planned reuse.

`--format aiperf-multi-turn` reads AIPerf `multi_turn` JSONL, following
`ai-dynamo/aiperf` at commit `fc7d61e`. A later turn is sent only after the
previous turn of its session completes, then after its `delay` or not before
its `timestamp`, as in AIPerf's fixed schedule. Its prompt resends the earlier
turns. No response text is stored, so each earlier answer is synthetic text
whose length is that turn's `output_length` (labelled `trace_declared`;
with `--tokenizer` it is exactly that many tokens and is checked).
AIPerf resends the live answers instead. Mooncake publishes no multi-turn
format: its `conversation_trace.jsonl` (`kvcache-ai/Mooncake` at `eca2c76`)
has no session field, so it is replayed open-loop with `--format mooncake`.

### Evidence viewer Space

`spaces/evidence-viewer/` is a Gradio app for Hugging Face Spaces. It shows a
record's measurements with their provenance and its span tree, and it offers
the Perfetto and OTLP exports for download. The logic lives in `viewer.py`,
which does not import Gradio and is tested in CI. The bundled example record
is synthetic. `deploy-space.yml` deploys it on every release; the README in
that directory has the manual deploy commands.

It runs at
[siddhant-k-code/llmtracefx-evidence-viewer](https://huggingface.co/spaces/siddhant-k-code/llmtracefx-evidence-viewer)
on ZeroGPU. `app.py` defines one `@spaces.GPU` function that is never called,
because ZeroGPU will not start an app without one.

### README badges

CI, CodeQL, PyPI version, PyPI downloads (pepy.tech), license, supported
Python versions and a link to the perf gate.

### Awesome list entries

One line per list, in each list's existing format. Each list asks for one
pull request per entry and alphabetical placement.

| List | Pull request | Status |
| --- | --- | --- |
| `tensorchord/Awesome-LLMOps` | tensorchord/Awesome-LLMOps#1136 | Merged |
| `antranapp/awesome-mlx` | antranapp/awesome-mlx#62 | Merged |
| `rafska/Awesome-local-LLM` | rafska/awesome-local-llm#364 | Open |

Entries as submitted:

- `tensorchord/Awesome-LLMOps`, Observability section:
  `| [LLMTraceFX](https://github.com/Siddhant-K-code/LLMTraceFX) | Evidence-first LLM inference measurement: provenance on every metric, OTLP/Perfetto export, server /metrics attribution and a CI perf gate. | ![GitHub Badge](https://img.shields.io/github/stars/Siddhant-K-code/LLMTraceFX.svg?style=flat-square) |`
- `rafska/Awesome-local-LLM`, Explorers, Benchmarks, Leaderboards:
  `- [LLMTraceFX](https://github.com/Siddhant-K-code/LLMTraceFX) - measures local MLX, llama.cpp and OpenAI-compatible inference with per-value provenance and correctness-checked workloads, and gates regressions in CI.`
- `antranapp/awesome-mlx`, next to `mlx-bench`:
  `- [LLMTraceFX](https://github.com/Siddhant-K-code/LLMTraceFX): MLX-LM and Apple Instruments (Metal) evidence collection with provenance, Perfetto export and a regression gate.`

## Follow ups not built here

- Getting rafska/awesome-local-llm#364 reviewed.
