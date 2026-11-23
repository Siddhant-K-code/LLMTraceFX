# Ecosystem gap analysis

This note compares LLMTraceFX with the wider inference benchmarking and
observability ecosystem. It covers what users of neighbouring tools expect,
what LLMTraceFX already does that they do not, and which channels would make
the project easier to find. It then describes the features this analysis
led to, from `server-metrics` and `export` to `sweep-api` and `replay`.

Every follow-up from that first survey has been built. A second survey on
2026-11-19 checked what changed upstream since then and what LLMTraceFX now
lacks. Its findings and a new prioritized follow-up list are in
[November 2026 refresh](#november-2026-refresh) at the end.

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
   phases at the client. **Ollama is built as `collect-ollama` and
   `sweep-ollama`.**
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

Importers for `vllm bench serve --save-result` JSON, AIPerf
`profile_export_aiperf.json` and GuideLLM `benchmarks.json` let users bring
existing results into LLMTraceFX without rerunning. All three formats carry
percentile fields (`p99_ttft_ms`, per metric `avg`, `p50`, `p99`, or
`percentiles.p99`). These are statistics another tool computed, so they feed
the perf gate rather than `compare`, and they carry their own provenance
labels. See "`import vllm-bench`, `import aiperf` and `import guidellm`"
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
   Collector (`otlp_json_file` receiver) and forwarded to Datadog, Honeycomb,
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
file exporter format, so the OpenTelemetry Collector `otlp_json_file` receiver
can read it with `start_at: beginning` (see the README for a full config). Spans use the GenAI semantic conventions (`gen_ai.operation.name`,
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
`open-telemetry/semantic-conventions-genai` at commit `76e0c19`, where the
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
end to end duration. These mappings were checked against vLLM `1ae1b64`,
SGLang `b6d4389` and TGI `b4adbf2`. TGI is archived upstream, so its
profile is legacy.

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
`otlp_json_file` receiver (`start_at: beginning`) parsed it with no errors, the `prometheus` exporter
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

### `import vllm-bench`, `import aiperf` and `import guidellm`

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

`llmtracefx-optimizer import guidellm <file> [--benchmark INDEX]` reads one
benchmark from a GuideLLM `benchmarks.json` report. Report version 1
(GuideLLM 0.5 and 0.6) and version 2 (0.7 and later, including 0.8.1) are
read; any other `metadata.version`, and reports from 0.4 or earlier that state
none, are refused. A report with several benchmarks (a sweep, or several rates
or stream counts) is refused unless `--benchmark` picks one, and the refusal
lists them. Each GuideLLM metric is a distribution split by request status;
only the `successful` part is read, and a zero filled placeholder for an
empty status is skipped. Latency metrics keep mean, median, `std_dev` as
`std`, min, max and the p0.1 to p99.9 percentiles. Rate metrics keep only the
mean, which is the count over the measurement window, because GuideLLM's
percentiles of a rate are instantaneous rates between events. GuideLLM weights
`time_per_output_token_ms` by output tokens and `inter_token_latency_ms` by
output tokens minus one, and counts a request with no value as 0; each metric
carries these as notes. `request_totals` gives completed, failed (errored) and
incomplete counts. The strategy fields, warmup and cooldown settings and each
constraint (as canonical JSON) form the load, so only runs with the same
settings compare. A sweep stores each limit as a list with a `current_index`;
the entry that applied is picked the way GuideLLM does, so a sweep benchmark
and a single run with the same limit match. The endpoint is the backend, API
route and streaming mode. Version 2 reports state them; version 1 reports
state only what the user passed, so GuideLLM's defaults
(`/v1/chat/completions`, streaming on) fill the rest. A setting that still
cannot be resolved marks the run unresolved and the gate refuses it. The
per-request `requests` lists are not read in this version: they hold prompt
and output text, and reading them the way `import aiperf --records` reads
AIPerf's records is a follow-up. Endpoint URLs and API keys are never copied.

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
`--metric ttft_ms.p99`, `--metric time_to_first_token.p99` or
`--metric time_to_first_token_ms.p99`. Each imported
file counts as one run, and the same file cannot be counted twice. A run with
failed requests, an unstated failure count or unresolved load is excluded and
listed. Imports cannot be used with `compare`, `tune` or the lab, cannot be
mixed with LLMTraceFX records in one gate, and results from different tools
cannot be gated against each other, because the tools define their metrics
differently. No summary import holds per-request data or correctness results.

`import aiperf <summary> --records profile_export.jsonl` reads AIPerf's
per-request records (the `MetricRecordInfo` lines in
[`record_models.py`](https://github.com/ai-dynamo/aiperf/blob/ebcdc98ad69388b86564e25bc1a6976fc50051a7/src/aiperf/common/models/record_models.py),
unchanged in the fields read from AIPerf 0.2.0 to 0.14.0) and writes an
artifact with tool `aiperf-records`. LLMTraceFX computes the statistics:
nearest rank percentiles over successful profiling records with
distribution-free 95% intervals, throughput over the records' window, and,
with `--slo-ttft-ms`, `--slo-tpot-ms` or `--slo-e2e-ms`, goodput and the share
of requests that met every target. These values are labelled `derived`; each
percentile also states its sample count, whose per-request inputs are
`imported_reported`. The records hold no model, endpoint, load or version, so
the summary from the same run is required, goes through the summary checks,
and is recorded as the companion with its sha256. The import is refused when
the records and summary disagree on successful or failed requests or token
totals, when the records span more than one profiling phase or repeat an
`x_request_id`, when a used field has the wrong type or unit or a duplicate
key, and when goodput would need a value a record lacks. `gate` keeps
`aiperf-records` artifacts apart from summary imports, uses runs with failures
for goodput only, and compares goodput only across runs with the same targets
and number of requests sent (token totals leave failed requests out, so they
are not used for goodput).

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

## November 2026 refresh

This survey was carried out on 2026-11-19. Each upstream claim below was read
in source at the commit or tag linked next to it, using `gh api` against the
file at that ref. Claims that rest only on release notes are marked as such.
It is analysis only: nothing listed here is built.

### Upstream versions checked

| Project | Release | Commit read |
| --- | --- | --- |
| vLLM | [v0.34.0](https://github.com/vllm-project/vllm/releases/tag/v0.34.0) | [`1ae1b64`](https://github.com/vllm-project/vllm/tree/1ae1b64807ff6be2a0d3f942dbad0a59a663a02a) |
| SGLang | [v0.5.23](https://github.com/sgl-project/sglang/releases/tag/v0.5.23) | [`b6d4389`](https://github.com/sgl-project/sglang/tree/b6d4389a77970e31817536c3e182256fae6a6f73) |
| TensorRT-LLM | [v1.2.1](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.2.1), [v1.4.0rc4](https://github.com/NVIDIA/TensorRT-LLM/releases/tag/v1.4.0rc4) | [`e6ecd28`](https://github.com/NVIDIA/TensorRT-LLM/tree/e6ecd280fd1f77cad6496f81a7194afe29f8d5e8) |
| NVIDIA Dynamo | [v1.6.2](https://github.com/ai-dynamo/dynamo/releases/tag/v1.6.2) | [`8c7ca7b`](https://github.com/ai-dynamo/dynamo/tree/8c7ca7b348c26c2d012c5767972690588dea7f52) |
| llama.cpp | [b12535](https://github.com/ggml-org/llama.cpp/releases/tag/b12535) | [`f8caa5e`](https://github.com/ggml-org/llama.cpp/tree/f8caa5ea30b5775512906ae3b7ee53b21a7a6e1d) |
| mlx-lm | [v0.32.2](https://github.com/ml-explore/mlx-lm/releases/tag/v0.32.2) | [`c4c9c76`](https://github.com/ml-explore/mlx-lm/tree/c4c9c762cc797fd979fc98645f5db5112753bbe9) |
| Ollama | [v0.40.16](https://github.com/ollama/ollama/releases/tag/v0.40.16) | [`42ac2c0`](https://github.com/ollama/ollama/tree/42ac2c0c1e132c0b4f0d9cfcd4e7950f0c5470bc) |
| AIPerf | [v0.13.0](https://github.com/ai-dynamo/aiperf/releases/tag/v0.13.0) | [`ebcdc98`](https://github.com/ai-dynamo/aiperf/tree/ebcdc98ad69388b86564e25bc1a6976fc50051a7) |
| GuideLLM | [v0.8.1](https://github.com/vllm-project/guidellm/releases/tag/v0.8.1) | [`c3204f3`](https://github.com/vllm-project/guidellm/tree/c3204f37c5dbd51b5a4f3154c79894d5d2470d21) |
| inference-perf | [v0.7.0](https://github.com/kubernetes-sigs/inference-perf/releases/tag/v0.7.0) | [`c34e956`](https://github.com/kubernetes-sigs/inference-perf/tree/c34e9562620e24581b77641d4a76ee3ada4eb1ec) |
| llm-d-benchmark | v1.0.0 | [`0fb187f`](https://github.com/llm-d/llm-d-benchmark/tree/0fb187fbd34d571ee97c2383375a2e3b2491aa40) |
| OTel GenAI semconv | no tags | [`76e0c19`](https://github.com/open-telemetry/semantic-conventions-genai/tree/76e0c19e7fe11dc693fe258642363a438af4d3aa) |
| OTel semconv (hardware) | [v1.45.0](https://github.com/open-telemetry/semantic-conventions/releases/tag/v1.45.0) | [`fb30ca1`](https://github.com/open-telemetry/semantic-conventions/tree/fb30ca17193b551473b1f43c315482918ab6987c) |
| OTel Collector contrib | [v0.165.1](https://github.com/open-telemetry/opentelemetry-collector-contrib/releases/tag/v0.165.1) | tag |
| Perfetto | [v59.1](https://github.com/google/perfetto/releases/tag/v59.1) | [`1e9c61e`](https://github.com/google/perfetto/tree/1e9c61e9844affd7dccd3d42d9ea9f3a3d10cf6d) |
| DCGM | [v4.7.0](https://github.com/NVIDIA/DCGM/tree/v4.7.0) (tag only) | [`9fae286`](https://github.com/NVIDIA/DCGM/tree/9fae2867e6e5264fa0fca22a07dc898d591baa21) |
| dcgm-exporter | [4.9.0](https://github.com/NVIDIA/dcgm-exporter/releases/tag/4.9.0) | [`4e663a5`](https://github.com/NVIDIA/dcgm-exporter/tree/4e663a526c4624ad0e18ad816f09b7be2c5348a7) |
| nvidia-ml-py | [13.615.71](https://pypi.org/project/nvidia-ml-py/13.615.71/) | PyPI wheel |
| amdsmi | [27.2.0](https://pypi.org/project/amdsmi/27.2.0/) | [`1e91f3c`](https://github.com/ROCm/amdsmi/tree/1e91f3c1527617066f50c22f9ec4368fe82e1a3c) (rocm-7.2.4) |

### What is still current

The pins in use at the survey were not stale. The gaps below are about
signals LLMTraceFX does not read yet, not about names that moved. Follow up 7
has since moved the pins to the commits above.

- vLLM: every `name="vllm:..."` string in `vllm/v1/metrics/loggers.py` is the
  same at `c26c9bf` and `1ae1b64`, so the `server-metrics` vLLM profile still
  matches.
- SGLang: the only metric names added since `e55e2ee` are the
  `sglang:linker_*` family. Counter names end in `_total` in source, and the
  Prometheus parser already strips the suffix.
- llama.cpp: the server `/metrics` definitions in `tools/server/server-task.cpp`
  are unchanged between the `ebc9fb9` pin and `f8caa5e`. There is still no
  request counter, so `unverifiable` remains correct for llama.cpp windows.
- mlx-lm `main` is still `c4c9c76`, the pin used by `sweep-mlx`. Native MTP is
  still not merged (open PR
  [ml-explore/mlx-lm#990](https://github.com/ml-explore/mlx-lm/pull/990)), so
  the unsupported native-MTP record is still right.
- AIPerf `main` is 2 commits ahead of `fc7d61e`, and
  `profile_export_aiperf.json` is still schema `1.7`
  ([`export_models.py#L296`](https://github.com/ai-dynamo/aiperf/blob/ebcdc98ad69388b86564e25bc1a6976fc50051a7/src/aiperf/common/models/export_models.py#L296)).
- The GenAI conventions moved 4 commits past `e6658f1`, none of which change
  the metrics `export otlp-metrics` writes. Everything in them is still
  `development` stability.
- dcgm-exporter 4.9.0 is still the latest release. It still writes no sample
  timestamps and still defaults to a 30 s collect interval
  ([`defaults.go#L27`](https://github.com/NVIDIA/dcgm-exporter/blob/4e663a526c4624ad0e18ad816f09b7be2c5348a7/internal/pkg/appconfig/defaults.go#L27)).
- The DCGM Python bindings are still shipped only in the
  `datacenter-gpu-manager-python3` deb and rpm, not on PyPI.
- Ollama still has no Prometheus endpoint
  ([ollama/ollama#3144](https://github.com/ollama/ollama/issues/3144) is
  open).
- Perfetto still documents Chrome JSON as a supported input
  ([`other-formats.md`](https://github.com/google/perfetto/blob/1e9c61e9844affd7dccd3d42d9ea9f3a3d10cf6d/docs/getting-started/other-formats.md)).

Two status changes affect the tables above:

- TGI is archived and in maintenance mode, with vLLM or SGLang recommended
  instead
  ([README](https://github.com/huggingface/text-generation-inference/blob/b4adbf2f6e2e721280bd0ea5f91d70f7d033f5ed/README.md)).
  The TGI profile keeps working against existing servers but should not grow.
- The Collector receiver is now registered as `otlp_json_file`, and
  `otlpjsonfile` is a deprecated alias that logs a warning. `start_at`
  defaults to `end`, so a Collector started after the file was written reads
  nothing unless it is set to `beginning`
  ([README](https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/v0.165.1/receiver/otlpjsonfilereceiver/README.md)).

### New gaps by area

#### Speculative decoding telemetry

Every major server now exports draft and accepted token counts. LLMTraceFX
reads none of them, and the speculative decoding doctor compares client
latency only.

- vLLM: `vllm:spec_decode_num_drafts`, `_num_draft_tokens`,
  `_num_accepted_tokens`, `_num_verified_tokens`, and per position
  `_num_accepted_tokens_per_pos`
  ([`spec_decode/metrics.py`](https://github.com/vllm-project/vllm/blob/1ae1b64807ff6be2a0d3f942dbad0a59a663a02a/vllm/v1/spec_decode/metrics.py#L251-L292)).
  `vllm bench serve` already writes `spec_decode_acceptance_rate`,
  `spec_decode_acceptance_length` and the per-position rates into its result
  file
  ([`serve.py`](https://github.com/vllm-project/vllm/blob/1ae1b64807ff6be2a0d3f942dbad0a59a663a02a/vllm/benchmarks/serve.py#L1386-L1399)),
  and `import vllm-bench` ignores them.
- SGLang: `sglang:spec_accept_length` and `spec_accept_rate` are gauges that
  keep the most recent value, so they cannot be diffed. The counter
  `sglang:spec_verify_calls_total` with `generation_tokens` gives a diffable
  accept length
  ([gauges](https://github.com/sgl-project/sglang/blob/b6d4389a77970e31817536c3e182256fae6a6f73/python/sglang/srt/observability/metrics_collector.py#L548-L583),
  [counters](https://github.com/sgl-project/sglang/blob/b6d4389a77970e31817536c3e182256fae6a6f73/python/sglang/srt/observability/metrics_collector.py#L1939-L1948)).
- TensorRT-LLM: `trtllm_spec_decode_num_draft_tokens_total`,
  `_num_accepted_tokens_total`, `trtllm_spec_decode_acceptance_length` and
  `trtllm_spec_decode_draft_overhead`
  ([`collector.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/e6ecd280fd1f77cad6496f81a7194afe29f8d5e8/tensorrt_llm/metrics/collector.py#L550-L566)).
- llama.cpp: `llamacpp:spec_decode_num_drafts_total`,
  `_num_draft_tokens_total`, `_num_accepted_tokens_total` and
  `_num_accepted_tokens_per_pos_total{position}` were already present at the
  previous `ebc9fb9` pin
  ([`server-task.cpp`](https://github.com/ggml-org/llama.cpp/blob/f8caa5ea30b5775512906ae3b7ee53b21a7a6e1d/tools/server/server-task.cpp#L1566-L1660))
  but are not in the profile. The response `timings` adds `draft_n` and
  `draft_n_accepted`
  ([`server-common.cpp`](https://github.com/ggml-org/llama.cpp/blob/f8caa5ea30b5775512906ae3b7ee53b21a7a6e1d/tools/server/server-common.cpp#L146-L168)),
  which `sweep-llama-cpp` does not read.
- mlx-lm: `GenerationResponse.from_draft` marks each drafted token
  ([`generate.py`](https://github.com/ml-explore/mlx-lm/blob/c4c9c762cc797fd979fc98645f5db5112753bbe9/mlx_lm/generate.py#L287-L312)),
  so an in-process acceptance count is possible.

#### Per-request server timing in the response

Under load a Prometheus window is contaminated by design, so `sweep-api`
and `replay` get no server-side split. Several servers now return their own
per-request numbers with the response.

- vLLM `--enable-per-request-metrics` (since v0.25.0) adds a `metrics` object
  with `time_to_first_token_ms`, `queue_time_ms`, `generation_time_ms`,
  `mean_itl_ms` and, with `--per-request-spec-decode-metrics`, a
  `speculative_decoding` block
  ([`protocol.py`](https://github.com/vllm-project/vllm/blob/1ae1b64807ff6be2a0d3f942dbad0a59a663a02a/vllm/entrypoints/generate/base/protocol.py#L104-L136),
  [`cli_args.py`](https://github.com/vllm-project/vllm/blob/1ae1b64807ff6be2a0d3f942dbad0a59a663a02a/vllm/entrypoints/common/cli_args.py#L158-L159)).
  It covers `n == 1` and single prompts only.
- SGLang `meta_info` carries `cached_tokens`, `cached_tokens_details`,
  `reasoning_tokens` and the `spec_accept_rate`, `spec_accept_length` and
  `spec_verify_ct` fields
  ([`tokenizer_manager.py`](https://github.com/sgl-project/sglang/blob/b6d4389a77970e31817536c3e182256fae6a6f73/python/sglang/srt/managers/tokenizer_manager.py#L2929-L2950),
  [`spec_decoding_metrics.py`](https://github.com/sgl-project/sglang/blob/b6d4389a77970e31817536c3e182256fae6a6f73/python/sglang/srt/speculative/spec_decoding_metrics.py#L48-L65)).
- TensorRT-LLM returns `X-TRTLLM-Start-End-Time`, `X-TRTLLM-Step-Metrics`
  and `X-TRTLLM-Ctx-Chunk-Metrics` headers in Server-Timing format when the
  request sends `X-TRTLLM-return-metrics: 1`
  ([`perf_metrics.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/e6ecd280fd1f77cad6496f81a7194afe29f8d5e8/tensorrt_llm/serve/perf_metrics.py#L20-L46)).
  `collect-api` stores headers but does not parse these.
- Ollama native responses add `prompt_eval_cached_count`
  ([`api/types.go`](https://github.com/ollama/ollama/blob/42ac2c0c1e132c0b4f0d9cfcd4e7950f0c5470bc/api/types.go#L683-L691))
  next to `total_duration`, `load_duration`, `prompt_eval_duration` and
  `eval_duration`. The OpenAI route drops the durations. `collect-ollama` and
  `sweep-ollama` read them from the native API.

#### KV and prefix cache tiers

`server-metrics` reads local prefix cache hits only. Servers now report where
a hit came from and how full the cache is.

- vLLM: `vllm:kv_cache_usage_perc`, `vllm:prompt_tokens_cached`,
  `vllm:prompt_tokens_by_source`, `vllm:prompt_tokens_cached_by_source`,
  `vllm:external_prefix_cache_queries` and `_hits`, and opt-in
  `vllm:kv_block_lifetime_seconds`
  ([`loggers.py`](https://github.com/vllm-project/vllm/blob/1ae1b64807ff6be2a0d3f942dbad0a59a663a02a/vllm/v1/metrics/loggers.py#L634),
  [L772 to L802](https://github.com/vllm-project/vllm/blob/1ae1b64807ff6be2a0d3f942dbad0a59a663a02a/vllm/v1/metrics/loggers.py#L772-L802)).
  KV offload counters `vllm:kv_offload_{load,store}_*` replace the deprecated
  `vllm:kv_offload_total_*`
  ([`offloading/metrics.py`](https://github.com/vllm-project/vllm/blob/1ae1b64807ff6be2a0d3f942dbad0a59a663a02a/vllm/distributed/kv_transfer/kv_connector/v1/offloading/metrics.py#L42-L55)).
- SGLang: `sglang:token_usage`, `kv_{available,evictable,used}_tokens`,
  HiCache host and storage counters, and the new `linker_*` family
  ([`metrics_collector.py`](https://github.com/sgl-project/sglang/blob/b6d4389a77970e31817536c3e182256fae6a6f73/python/sglang/srt/observability/metrics_collector.py#L418-L470)).
- TensorRT-LLM: `trtllm_prompt_cached_tokens_total`,
  `trtllm_prompt_cache_hit_tokens_total{cache_level,cache_tier}` and a KV
  cache utilization gauge
  ([`collector.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/e6ecd280fd1f77cad6496f81a7194afe29f8d5e8/tensorrt_llm/metrics/collector.py#L323-L610)).
- llama.cpp: `llamacpp:prompt_tokens_cached_total`, while
  `llamacpp:prompt_tokens_total` excludes cached tokens
  ([`server-task.cpp`](https://github.com/ggml-org/llama.cpp/blob/f8caa5ea30b5775512906ae3b7ee53b21a7a6e1d/tools/server/server-task.cpp#L1566-L1580)).
  The profile therefore under-reports prompt work on cache hits.

Usage gauges are point samples and must be recorded as before and after
values, not diffed.

#### Engines with no profile

- TensorRT-LLM `trtllm-serve` serves Prometheus text on
  `/prometheus/metrics` (its `/metrics` is JSON iteration stats) with
  `trtllm_request_success_total`, `trtllm_time_to_first_token_seconds`,
  `trtllm_request_queue_time_seconds`, prefill and decode histograms, and
  token counters
  ([`collector.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/e6ecd280fd1f77cad6496f81a7194afe29f8d5e8/tensorrt_llm/metrics/collector.py#L226-L321)).
  It has a request counter, so windows can be attributed. The metrics only
  appear with `return_perf_metrics` and `enable_iter_perf_stats` on
  ([docs](https://github.com/NVIDIA/TensorRT-LLM/blob/e6ecd280fd1f77cad6496f81a7194afe29f8d5e8/docs/source/commands/trtllm-serve/performance-metrics-collection.md)),
  so a profile needs a "stats not enabled" result.
- Dynamo's frontend exposes `dynamo_frontend_requests_total`,
  `time_to_first_token_seconds`, `inter_token_latency_seconds`,
  `queued_requests` and KV router metrics such as `kv_hit_rate`
  ([`prometheus_names.rs`](https://github.com/ai-dynamo/dynamo/blob/8c7ca7b348c26c2d012c5767972690588dea7f52/lib/runtime/src/metrics/prometheus_names.rs#L173-L290)).
  Engine metrics stay on a separate worker port, so a run needs two scrapes.

#### Benchmark harnesses and result formats

- AIPerf writes `profile_export.jsonl` with one record per request, plus
  `gpu_telemetry_export.json` and server metrics exports
  ([`working-with-profile-exports.md`](https://github.com/ai-dynamo/aiperf/blob/ebcdc98ad69388b86564e25bc1a6976fc50051a7/docs/tutorials/working-with-profile-exports.md)).
  `import aiperf` read only the summary, so its percentiles stayed
  `imported_derived`; `--records` now reads the per-request file. AIPerf also
  added multi-run confidence intervals (`--num-profile-runs`,
  `--confidence-level`) and HTTP connection timing
  ([`multi-run-confidence.md`](https://github.com/ai-dynamo/aiperf/blob/ebcdc98ad69388b86564e25bc1a6976fc50051a7/docs/tutorials/multi-run-confidence.md),
  [`http-trace-metrics.md`](https://github.com/ai-dynamo/aiperf/blob/ebcdc98ad69388b86564e25bc1a6976fc50051a7/docs/tutorials/http-trace-metrics.md)).
- GuideLLM reports confidence intervals for per-request metrics, warmup and
  cooldown windows, over-saturation stopping and throughput knee detection
  ([`metrics.md`](https://github.com/vllm-project/guidellm/blob/c3204f37c5dbd51b5a4f3154c79894d5d2470d21/docs/en/guides/metrics.md),
  [`knee_detection.md`](https://github.com/vllm-project/guidellm/blob/c3204f37c5dbd51b5a4f3154c79894d5d2470d21/docs/en/guides/knee_detection.md)).
  `sweep-api` reports nearest rank percentiles with no interval and no
  warmup exclusion.
- inference-perf separates server `usage` from client tokenization and counts
  mismatches, and reports price per million tokens
  ([`metrics.md`](https://github.com/kubernetes-sigs/inference-perf/blob/c34e9562620e24581b77641d4a76ee3ada4eb1ec/docs/metrics.md)).
- llm-d's Benchmark Report v0.2.1 is the closest thing to a shared result
  format, with converters from GuideLLM, inference-perf, vLLM bench and AIPerf
  ([`schema.py`](https://github.com/llm-d/llm-d-benchmark/blob/0fb187fbd34d571ee97c2383375a2e3b2491aa40/benchmark-report/llmd_benchmark_report/artifacts/benchmark_report/v0_2/schema.py)).
- `import` covers vLLM bench, AIPerf and GuideLLM (report versions 1 and 2).
  SGLang's bench now lives in
  `sglang/benchmark/serving.py`, fetches `/server_info` into its result and
  has a `--cache-report` option
  ([`serving.py`](https://github.com/sgl-project/sglang/blob/b6d4389a77970e31817536c3e182256fae6a6f73/python/sglang/benchmark/serving.py)).
  GuideLLM `benchmarks.json` is version 2
  ([`report.py`](https://github.com/vllm-project/guidellm/blob/c3204f37c5dbd51b5a4f3154c79894d5d2470d21/src/guidellm/benchmark/schemas/report.py#L35-L45))
  and is now read by `import guidellm`. `trtllm-bench` and `llama-bench -o jsonl` have their own JSON reports.

#### Trace replay formats

Mooncake, Weka (Claude Code sessions) and OpenTelemetry GenAI spans are now
the common replay formats. GuideLLM reads all three
([`trace_replay.md`](https://github.com/vllm-project/guidellm/blob/c3204f37c5dbd51b5a4f3154c79894d5d2470d21/docs/en/guides/trace_replay.md#L9-L14)),
inference-perf reads Weka and OTel
([`otel_trace_replay.md`](https://github.com/kubernetes-sigs/inference-perf/blob/c34e9562620e24581b77641d4a76ee3ada4eb1ec/docs/otel_trace_replay.md)),
and AIPerf added `bailian_trace`, `weka_trace` and DAG replay
([`plugins.yaml`](https://github.com/ai-dynamo/aiperf/blob/ebcdc98ad69388b86564e25bc1a6976fc50051a7/src/aiperf/plugin/plugins.yaml)).
LLMTraceFX reads Mooncake only of the three. Since `export otlp` already writes
GenAI spans, OTel replay would let a user replay an exported run. Azure
published GitHub Copilot coding agent traces with per-call cached token counts
([`GitHubCopilotCodingAgentDataset2026.md`](https://github.com/Azure/AzurePublicDataset/blob/5d7424495ca07a30c89d30a2a534ac0d0d3a5e3e/GitHubCopilotCodingAgentDataset2026.md)),
and BurstGPT v2.0 added a `Session ID` column that `replay` ignores.

#### OpenTelemetry and Perfetto

- The GenAI conventions define `gen_ai.usage.cache_read.input_tokens` and
  `gen_ai.usage.reasoning.output_tokens` span attributes
  ([`registry.yaml`](https://github.com/open-telemetry/semantic-conventions-genai/blob/76e0c19e7fe11dc693fe258642363a438af4d3aa/model/gen-ai/registry.yaml#L312-L359))
  and `gen_ai.client.inference.usage.*` counters including cache read, cache
  write and reasoning tokens
  ([`token-metrics.yaml`](https://github.com/open-telemetry/semantic-conventions-genai/blob/76e0c19e7fe11dc693fe258642363a438af4d3aa/model/gen-ai/token-metrics.yaml#L17-L101)).
  `export otlp` already wrote these span attributes from API evidence usage,
  though only the cache read was tested, and `replay --format otel-genai`
  ignored them. Both are now covered (see "Built from this refresh"). The
  counters were skipped earlier because modality is unknown; the cache and
  reasoning attributes have no such problem.
- The conventions still have no server queue, KV cache, speculative decoding
  or energy metrics. Only open proposals exist
  ([semantic-conventions-genai#87](https://github.com/open-telemetry/semantic-conventions-genai/issues/87)).
  `llmtracefx.*` names remain the right choice for those.
- The hardware conventions define `hw.energy` (J, counter) and `hw.power`
  (W, gauge) with `hw.type=gpu`, at development stability
  ([`common-metrics.yaml`](https://github.com/open-telemetry/semantic-conventions/blob/fb30ca17193b551473b1f43c315482918ab6987c/model/hardware/common-metrics.yaml#L4-L55)).
  `llmtracefx.gpu.energy` could be written as `hw.energy` too.
  `hw.power` is instantaneous, so the average stays custom.
- vLLM's own traces use one `llm_request` span with custom
  `gen_ai.latency.*` attributes and the older `gen_ai.usage.prompt_tokens`
  name
  ([`tracing/utils.py`](https://github.com/vllm-project/vllm/blob/1ae1b64807ff6be2a0d3f942dbad0a59a663a02a/vllm/tracing/utils.py#L15-L44)).
  SGLang and TensorRT-LLM also export OTLP spans. LLMTraceFX cannot import
  any of them.
- Perfetto recommends its protobuf format for custom tools, with named
  counter tracks, async slices, flows and debug annotations, written with the
  pure Python `perfetto.trace_builder` (PyPI `perfetto` 0.59.1)
  ([`converting.md`](https://github.com/google/perfetto/blob/1e9c61e9844affd7dccd3d42d9ea9f3a3d10cf6d/docs/getting-started/converting.md)).
  `export perfetto` writes Chrome JSON, where `X` slices must nest.

#### GPU telemetry

- NVML: `nvmlDeviceGetFieldValues` with `NVML_FI_DEV_POWER_INSTANT` returns
  timestamped instantaneous power. `gpu-energy` uses
  `nvmlDeviceGetPowerUsage`, which is a smoothed average on recent drivers.
  `nvmlDeviceGetProcessesUtilizationInfo` reports per-PID SM use since a
  timestamp, which would catch a process that started and finished between
  two process-list polls. GPM (`nvmlGpmSampleGet`) gives SM occupancy and
  tensor activity on Hopper and newer. All are in
  [nvidia-ml-py 13.615.71](https://pypi.org/project/nvidia-ml-py/13.615.71/).
- dcgm-exporter 4.9.0 lists `DCGM_FI_DEV_GPU_ENERGY_JOULES_TOTAL` as its
  default energy counter, where older releases listed the millijoule
  `DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION`
  ([`default-counters.csv`](https://github.com/NVIDIA/dcgm-exporter/blob/4e663a526c4624ad0e18ad816f09b7be2c5348a7/etc/default-counters.csv)),
  which `gpu-energy` already reads. The collect interval in effect is not
  recorded in the evidence.
- AMD: `amdsmi_get_energy_count` returns the energy accumulator with its own
  timestamp, and `amdsmi_get_gpu_process_list` gives a process list
  ([`amdsmi_interface.py`](https://github.com/ROCm/amdsmi/blob/1e91f3c1527617066f50c22f9ec4368fe82e1a3c/py-interface/amdsmi_interface.py#L4269-L4288)).
  `gpu-energy` is NVIDIA only.

#### Not gaps

- Prometheus native histograms: vLLM and SGLang still expose classic
  histograms in text format, which the parser reads, including optional
  sample timestamps.
- MLX moved memory APIs from `mx.metal.*` to `mx.*`
  ([`memory.cpp`](https://github.com/ml-explore/mlx/blob/14dc9c673bce9a51dca4e58f5b2ccae80736431d/python/src/memory.cpp#L14-L150)).
  The MLX collector already looks for the top level names first.
- llama.cpp `timings.cache_n`, `prompt_ms` and `predicted_ms` are already read
  by `sweep-llama-cpp`.

### Built from this refresh

- Speculative decoding acceptance: `server-metrics` derives acceptance rate,
  length and per-position rates from the vLLM, SGLang, llama.cpp and
  TensorRT-LLM counters, the speculative doctor uses them when present, and
  `import vllm-bench` reads the `spec_decode_*` keys (#171, #172).
- Per-request server numbers from the response: vLLM `metrics`, SGLang
  `sglext`, TensorRT-LLM `trtllm.perf_metrics` and llama.cpp `draft_n` in
  `collect-api`, `sweep-api`, `replay run` and `sweep-llama-cpp` (#174).
- Cached prompt tokens by tier, KV usage point samples, vLLM KV offload,
  llama.cpp cached prompt tokens, and a TensorRT-LLM profile (#172).
- `replay --format weka` and `--format otel-genai`, which replays
  `export otlp` output (#173).
- GenAI cache and reasoning token attributes: `export otlp` writes
  `gen_ai.usage.cache_read.input_tokens`,
  `gen_ai.usage.cache_write.input_tokens` and
  `gen_ai.usage.reasoning.output_tokens` only when the provider reported
  them, and `replay --format otel-genai` reads them back, accepts the
  pre-v0.1.0 `cache_creation` name, refuses parts larger than their total and
  records their sums. Checked against
  [`registry.yaml`](https://github.com/open-telemetry/semantic-conventions-genai/blob/4181b1ebe55da7edb3538636c45ae64af369b5ef/model/gen-ai/registry.yaml#L315-L371),
  where all three are at development stability.
- AIPerf per-request import: `import aiperf --records profile_export.jsonl`
  computes LLMTraceFX's own percentiles, intervals and goodput (#190).
- Native Ollama collection: `collect-ollama` and `sweep-ollama` stream
  `/api/chat` or `/api/generate` and record `load_duration`,
  `prompt_eval_duration`, `eval_duration`, `total_duration` and the token
  counts, including `prompt_eval_cached_count`, as `provider_reported`
  (follow up 11).
- Re-pins (follow up 7): `sweep-llama-cpp` records llama.cpp `f8caa5e`
  (b12535) as its `upstream_reference`, `export otlp-metrics` cites GenAI
  semconv `76e0c19`, and the vLLM and SGLang citations point at `1ae1b64`
  and `b6d4389`. Every metric and attribute name LLMTraceFX reads exists
  unchanged at the new commits, so no name needed dual handling. The README
  shows a Collector config with the `otlp_json_file` receiver and
  `start_at: beginning`, checked with otelcol-contrib 0.165.0.
- TGI is a legacy profile (follow up 17): the `--runtime` and
  `--server-metrics-runtime` help says so, and TGI evidence carries a note
  that the server is archived upstream, which `server-metrics diff` also
  prints to stderr. Parsing, attribution and exit codes
  are unchanged.

## Follow ups not built here

Sizes: **S** is about a day, one module and its tests. **M** is a few days: a
new profile, importer or subcommand with fixtures. **L** is a week or more: a
new execution model or output format.

Ranked by value to the evidence model against cost:

1. **GenAI cache and reasoning token attributes in `export otlp` (S).**
   **Built**, see "Built from this refresh".
2. **Dynamo frontend profile with a worker scrape (M).** Its frontend has a
   request counter and KV router hit rates, and engine metrics sit on a
   second port.
3. **Extend `import vllm-bench` to `server_ttfts` fields (S).** The
   fields are already in the files users have, and the importer drops them.
4. **NVML instant power and per-process utilization (S, then M).** Timestamped
   instant power is a better sample than the smoothed reading, and per-PID
   utilization closes a contamination blind spot between polls.
5. **Confidence intervals and warmup exclusion in sweeps (M). Built.**
    Sweep percentiles now get a distribution-free 95% interval when there
    are enough requests, and
    `--warmup-requests` leaves the first requests of each point out of the
    summaries. See [`docs/load-sweeps.md`](load-sweeps.md#confidence-intervals).
6. **AIPerf `profile_export.jsonl` import (M). Built.**
    `import aiperf --records` lets `gate` use LLMTraceFX's own percentiles,
    intervals and goodput instead of `imported_derived` ones. See
    [`import vllm-bench` and `import aiperf`](#import-vllm-bench-and-import-aiperf).
7. **Re-pins and the Collector receiver rename (S). Built.** Re-pin vLLM, SGLang,
    llama.cpp and the GenAI conventions to the commits above, and document
    `otlp_json_file` with `start_at: beginning`. See "Built from this
    refresh".
8. **`hw.energy` alongside `llmtracefx.gpu.energy` (S).** Backends that know
    the hardware conventions could chart LLMTraceFX energy with no mapping.
9. **Record the dcgm-exporter collect interval (S).** The bracket width
    depends on it, and the evidence does not say what it was.
10. **Importers for SGLang bench, GuideLLM, `trtllm-bench` and `llama-bench`
    (M each).** Each brings results users already have into `gate`.
    **GuideLLM built as `import guidellm`** (summary statistics; reading its
    per-request lists is a follow-up).
11. **Native Ollama collector (M).** Its `/api/*` responses carry load,
    prompt and eval durations and the new cached count, which the OpenAI
    route drops. **Built as `collect-ollama` and `sweep-ollama`.**
12. **Perfetto protobuf export (M to L).** Counter tracks, async slices and
    flows would show power, sweeps and requests together; JSON stays as the
    fallback.
13. **AMD energy through `amdsmi` (M).** It has a timestamped energy counter
    and a process list, so it fits the existing attribution model.
14. **Azure Copilot agent traces, Bailian traces and BurstGPT sessions in
    `replay` (M, M, S).** These are the newest public traces with sessions
    and cached token counts.
15. **Benchmark Report v0.2 export (M).** llm-d uses it to compare harnesses,
    so it is a route into that tooling.
16. **Client and server token reconciliation, and cost per token in sweeps
    (S to M).** inference-perf reports both; LLMTraceFX has provider usage
    and pricing data but does not join them.
17. **Mark the TGI profile as legacy (S). Built.** TGI is archived. See
    "Built from this refresh".
18. **Import engine OTLP traces from vLLM, SGLang and TensorRT-LLM (L).**
    Server span timings would be `measured_native`, but each engine uses its
    own span layout.
19. **Tool-calling, multimodal and embeddings workloads (L each).** Only if
    the scope widens beyond text generation.
20. **vLLM KV event subscriber and profiler artifact capture (M).** Optional
    deeper evidence for the KV cache auditor.
26. Getting rafska/awesome-local-llm#364 reviewed. It has been open with no
    review since it was filed (last updated 2026-11-09).
