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
| LLMTraceFX | single request | **now: `server-metrics`** | no (single objective `tune`) | Apple Instruments | every workload verified | per value provenance |

## Gaps, ranked by how strongly users expect them

1. **Concurrency and request rate sweeps.** Table stakes for every serving
   benchmark. LLMTraceFX measures one request at a time by design. A sweep
   mode would need its own attribution story because per request timings
   stop being isolated.
2. **Goodput and SLO attainment.** vLLM, SGLang, GuideLLM and AIPerf report
   the fraction of requests or tokens that meet joint TTFT, TPOT and E2E
   targets. This depends on item 1.
3. **Server-side metrics during a run.** GuideLLM and AIPerf scrape the
   server's Prometheus endpoint so users can tell "the client saw slow TTFT"
   apart from "the server queue was the bottleneck". **Built in this branch
   as `server-metrics`.**
4. **Native vLLM, SGLang, TGI and Ollama collectors.** Partly addressed by
   item 3, which reads the server's own histograms instead of inferring
   phases at the client.
5. **GPU telemetry and energy (NVML or DCGM).** LLMTraceFX covers Apple
   silicon through Instruments. NVIDIA power is absent.
6. **Trace replay (Mooncake, ShareGPT timed replay, BurstGPT).** SGLang,
   AIPerf and AISBench all added Mooncake replay within the last year.
7. **Prefix cache hit rate under load.** `server-metrics` now reports the
   vLLM `prefix_cache_queries` and `prefix_cache_hits` counter deltas, and
   SGLang's `cached_tokens_total`, for the measured window.
8. **Export to observability backends.** Datadog, Honeycomb, Langfuse,
   Grafana (through OpenLIT), MLflow and W&B Weave ingest OTLP spans that use
   the OpenTelemetry GenAI semantic conventions. Perfetto is the common
   viewer for timelines. **Built in this branch as `export otlp` and
   `export perfetto`.**
9. **Reasoning aware first token timing.** AIPerf separates time to first
   token from time to first output. The OTLP export follows the GenAI
   convention for `gen_ai.response.time_to_first_chunk`, which counts
   reasoning chunks, and keeps the first visible content offset as a
   separate attribute.
10. **Scheduling fidelity diagnostics.** GuideLLM reports dispatch and
    scheduling delay so users can trust the load generator itself. Only
    relevant once item 1 exists.

Importers for `vllm bench serve --save-result` JSON and AIPerf
`profile_export_aiperf.json` would let users bring existing results into
`compare` without rerunning. Both formats carry percentile fields
(`p99_ttft_ms`, per metric `avg`, `p50`, `p99`) that map cleanly to
`derived` provenance.

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

Items 1, 4 (CI badges only) and 5 are now built; see below.

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

### Evidence viewer Space

`spaces/evidence-viewer/` is a Gradio app for Hugging Face Spaces. It shows a
record's measurements with their provenance and its span tree, and it offers
the Perfetto and OTLP exports for download. The logic lives in `viewer.py`,
which does not import Gradio and is tested in CI. The bundled example record
is synthetic. The README in that directory has the deploy commands.

### README badges

CI, CodeQL, license, supported Python versions and a link to the perf gate.
There is no PyPI badge because the package is not on PyPI yet.

### Awesome list entries (drafts)

One line per list, in each list's existing format. Each list asks for one
pull request per entry and alphabetical placement.

- `tensorchord/Awesome-LLMOps`, Observability section:
  `| [LLMTraceFX](https://github.com/Siddhant-K-code/LLMTraceFX) | Evidence-first LLM inference measurement: provenance on every metric, OTLP/Perfetto export, server /metrics attribution and a CI perf gate. | ![GitHub Badge](https://img.shields.io/github/stars/Siddhant-K-code/LLMTraceFX.svg?style=flat-square) |`
- `rafska/Awesome-local-LLM`, Explorers, Benchmarks, Leaderboards:
  `- [LLMTraceFX](https://github.com/Siddhant-K-code/LLMTraceFX) - measures local MLX, llama.cpp and OpenAI-compatible inference with per-value provenance and correctness-checked workloads, and gates regressions in CI.`
- `antranapp/awesome-mlx`, next to `mlx-bench`:
  `- [LLMTraceFX](https://github.com/Siddhant-K-code/LLMTraceFX): MLX-LM and Apple Instruments (Metal) evidence collection with provenance, Perfetto export and a regression gate.`

## Follow ups not built here

- OTLP metrics export (histograms for TTFT and duration).
- Concurrency sweeps, goodput and trace replay.
- Importers for `vllm bench serve` and AIPerf exports.
- NVML or DCGM energy collection.
- Publishing to PyPI (needed for a PyPI badge and `pip install llmtracefx`).
- Tagging a `v1` release so the Action can be pinned as `@v1`.
- Deploying the Space, which needs a Hugging Face account.
- Following up on the awesome list submissions:
  tensorchord/Awesome-LLMOps#1136, rafska/awesome-local-llm#364 and
  antranapp/awesome-mlx#62. These describe only features already on `main`.
  Once the gate and exports merge, they can be mentioned in follow-up edits.
