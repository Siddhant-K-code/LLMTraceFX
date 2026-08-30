<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/llmtracefx-lockup-inverse.svg">
    <img src="assets/brand/llmtracefx-lockup.svg" alt="LLMTraceFX" width="309">
  </picture>
</p>

<p align="center">
  <strong>GPU-level LLM inference profiler</strong> that analyzes token-level performance and provides AI-powered explanations.
</p>

<p align="center">
  <a href="https://youtu.be/8tBpqgQIEG4">Video demo</a>
  ·
  <a href="https://siddhant-k-code--llmtracefx-web-app.modal.run">Live demo</a>
  ·
  <a href="#-inference-optimizer-foundation">Optimizer</a>
  ·
  <a href="DESIGN.md">Design system</a>
</p>

<p align="center">
  <img src="assets/brand/social-preview.png" alt="LLMTraceFX: find out why inference is slow, with evidence. A tune report readout showing a 42.5 percent mean latency reduction with the pass rate held at 100 percent." width="820">
</p>

## 🎬 Video Demo

[![LLMTraceFX Demo](https://img.youtube.com/vi/8tBpqgQIEG4/maxresdefault.jpg)](https://youtu.be/8tBpqgQIEG4)

## 🌐 **Live Demo**
**Try it now:** https://siddhant-k-code--llmtracefx-web-app.modal.run (might be not available at all times due to Modal's free tier limitations 🙈)

**Quick API test:**
```bash
curl -X POST "https://siddhant-k-code--llmtracefx-web-app.modal.run/analyze-trace" \
-H "Content-Type: application/json" \
-d '{"trace_data": {"tokens": [{"id": 0, "text": "Hello", "operations": [{"name": "matmul", "start_time": 0, "duration": 15.3}]}]}, "gpu_type": "A10G", "enable_claude": false}'
```

**Upload your trace file:**
```bash
curl -X POST "https://siddhant-k-code--llmtracefx-web-app.modal.run/upload-trace?gpu_type=A10G&enable_claude=true" \
     -F "file=@your_trace.json"
```

---

<details>
<summary><strong>📋 Full Demo Walkthrough — CloudRift GPU (end-to-end)</strong></summary>

This walkthrough runs LLMTraceFX on a real cloud GPU (CloudRift RTX 4090/5090) from scratch. All commands are copy-paste ready.

---

### Step 1 — Connect to the CloudRift instance

```bash
ssh riftuser@<YOUR_INSTANCE_IP> -o PreferredAuthentications=password -o PubkeyAuthentication=no
```

If you hit "Too many authentication failures":

```bash
ssh riftuser@<YOUR_INSTANCE_IP> \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no
```

---

### Step 2 — Verify the GPU

```bash
nvidia-smi
```

---

### Step 3 — Install uv and clone the repo

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
git clone https://github.com/Siddhant-K-code/LLMTraceFX.git && cd LLMTraceFX
uv sync
```

---

### Step 4 — Generate synthetic traces

```bash
# Memory-bound profile — simulates long-context decode bottleneck
python generate_trace.py \
  --tokens "The" "transformer" "model" "generates" "tokens" \
            "auto" "regressively" "one" "at" "a" "time" \
  --profile memory_bound \
  --output demo_memory_bound.json

# Optimized profile — for comparison
python generate_trace.py \
  --tokens "The" "transformer" "model" "generates" "tokens" \
            "auto" "regressively" "one" "at" "a" "time" \
  --profile optimized \
  --output demo_optimized.json
```

---

### Step 5 — Run the profiler

```bash
uv run llmtracefx \
  --trace demo_memory_bound.json \
  --gpu-type A10G \
  --no-claude \
  --output-dir output/memory_bound_run

uv run llmtracefx \
  --trace demo_optimized.json \
  --gpu-type A10G \
  --no-claude \
  --output-dir output/optimized_run
```

---

### Step 6 — Compare performance scores

```bash
grep "Avg performance" \
  output/memory_bound_run/report.txt \
  output/optimized_run/report.txt
```

---

### Step 7 — Read the bottleneck report

```bash
cat output/memory_bound_run/report.txt
```

---

### Step 8 — 4-way comparison across real experiment outputs

```bash
grep "Average Performance Score" \
  output/output_hf_2k_b1/report.txt \
  output/output_hf_8k_b1/report.txt \
  output/output_opt_2k_b1/report.txt \
  output/output_opt_8k_b1/report.txt
```

```bash
for d in output/output_hf_2k_b1 output/output_hf_8k_b1 output/output_opt_2k_b1 output/output_opt_8k_b1; do
  echo "--- $d ---"
  grep -A3 "Bottleneck Distribution" $d/report.txt
done
```

---

### Step 9 — Serve the dashboard

```bash
cd output/output_hf_8k_b1 && python3 -m http.server 8080
```

Open an SSH tunnel on your **local machine** (new terminal tab):

```bash
ssh -L 8080:localhost:8080 riftuser@<YOUR_INSTANCE_IP> \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no -N
```

Then open in your browser: `http://localhost:8080/dashboard.html`

---

### Step 10 — Launch the real-time Streamlit dashboard

On the remote instance (stop the http.server first with `Ctrl+C`):

```bash
cd ~/LLMTraceFX && uv run python launch_dashboard.py
```

SSH tunnel on your local machine:

```bash
ssh -L 8501:localhost:8501 riftuser@<YOUR_INSTANCE_IP> \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no -N
```

Then open: `http://localhost:8501`

---

### Both tunnels at once (optional)

```bash
ssh -L 8080:localhost:8080 -L 8501:localhost:8501 \
  riftuser@<YOUR_INSTANCE_IP> \
  -o PreferredAuthentications=password \
  -o PubkeyAuthentication=no -N
```

</details>

---

## 🎯 Features

- **Token-level profiling** of LLM inference with kernel timing analysis
- **GPU bottleneck heuristics** from measured metadata or deterministic estimates
- **AI explanations** using Claude API for performance insights
- **Interactive visualizations** with flame graphs and dashboards
- **Modal.com deployment** with GPU acceleration
- **Multiple input formats** (vLLM, generic trace logs)
- **Apple Silicon and NVIDIA GB10 profiles** with an optional MLX recorder

## 📦 Installation

### Using uv (Recommended)

```bash
git clone https://github.com/Siddhant-K-code/LLMTraceFX.git
cd LLMTraceFX

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Install development and test tools
uv sync --extra dev --extra test

# Apple Silicon only: install the optional MLX recorder dependency
uv sync --extra mlx
```

### Using pip

```bash
git clone https://github.com/Siddhant-K-code/LLMTraceFX.git
cd LLMTraceFX
pip install -r llmtracefx/requirements.txt

# Or install as editable package
pip install -e .

# Apple Silicon only
pip install -e ".[mlx]"
```

## 🔧 Quick Start

### 1. CLI Usage

```bash
# With uv
uv run llmtracefx --trace sample --no-claude
uv run llmtracefx --trace your_trace.json --gpu-type A10G --no-claude

# Or activate virtual environment first
uv sync
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
llmtracefx --trace sample --no-claude

# With pip/python
python -m llmtracefx.main --trace sample --no-claude
python -m llmtracefx.main --trace your_trace.json --gpu-type A10G --no-claude
python -m llmtracefx.main --trace sample --no-claude
```

### Apple Silicon with MLX

`MLXTraceRecorder` runs on Apple Silicon. It evaluates lazy MLX results,
synchronizes the device, and writes JSON that the existing CLI can read.

```bash
uv sync --extra mlx
uv run python examples/mlx_trace.py
uv run llmtracefx --trace mlx_trace.json --gpu-type MLX --no-claude
```

Use the recorder around token generation in your own model:

```python
from llmtracefx.profiler import MLXTraceRecorder

with MLXTraceRecorder("mlx_trace.json") as trace:
    with trace.token(0, "Hello"):
        logits = trace.measure("model_forward", model, input_ids)
        next_token = trace.measure("sample", sample, logits)
```

Each duration is synchronized wall-clock time for the callable plus forced MLX
evaluation. It is not an individual Metal kernel-counter measurement. The
recorder also saves MLX allocator snapshots after each operation.

For a native Xcode GPU capture of the same region:

```bash
MTL_CAPTURE_ENABLED=1 uv run python examples/mlx_trace.py \
  --metal-capture mlx_trace.gputrace
```

The `.gputrace` path must not already exist. Open it in Xcode for kernel-level
inspection. See the [MLX Metal debugger guide](https://ml-explore.github.io/mlx/build/html/dev/metal_debugger.html).

### NVIDIA GB10 / DGX Spark

No extra dependency is needed for GB10. Analyze an existing LLMTraceFX, vLLM,
or event trace with the static GB10 hardware profile:

```bash
uv run llmtracefx --trace trace.json --gpu-type GB10 --no-claude
```

`DGX-SPARK` is accepted as an alias. The profile uses the GB10's 128 GB
coherent unified memory and 273 GB/s memory bandwidth from the
[NVIDIA DGX Spark hardware guide](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).
It does not capture a CUDA trace by itself.

### 2. FastAPI Server

```bash
# With uv
uv run llmtracefx-serve

# Or with python
python -m llmtracefx.api.serve

# Access at http://localhost:8000
```

### 3. Modal Deployment

```bash
# Setup Modal secrets
uv run modal secret create claude-api-key CLAUDE_API_KEY=your_api_key

# Deploy to Modal
uv run modal deploy llmtracefx/modal_app.py

# Test with sample data
uv run modal run llmtracefx/modal_app.py
```

#### **🌐 Live Web API**
Once deployed, your app is available at:
```
https://siddhant-k-code--llmtracefx-web-app.modal.run
```

#### **Quick API Test**
```bash
# Test the deployed API
curl -X POST "https://siddhant-k-code--llmtracefx-web-app.modal.run/analyze-trace" \
-H "Content-Type: application/json" \
-d '{
  "trace_data": {
    "tokens": [
      {
        "id": 0,
        "text": "Hello",
        "operations": [
          {"name": "matmul", "start_time": 0, "duration": 15.3}
        ]
      }
    ]
  },
  "gpu_type": "A10G",
  "enable_claude": false
}'
```

## 🔑 Configuration

### Environment Variables

```bash
export CLAUDE_API_KEY="your_claude_api_key"
export DEFAULT_GPU_TYPE="A10G"  # A10G, A100, H100, GB10, or MLX
export ENABLE_CLAUDE="true"
export DASHBOARD_PORT="8000"
```

### Claude API Setup

1. Get API key from [Anthropic](https://console.anthropic.com/)
2. Set environment variable: `export CLAUDE_API_KEY="your_key"`
3. Or create Modal secret: `modal secret create claude-api-key CLAUDE_API_KEY=your_key`

## 📊 Output Examples

### CLI Output
```
🔍 Analyzing trace: sample
📊 Using sample trace data
🔧 Analyzing GPU performance (NVIDIA A10G)
📈 Analysis complete:
   Total tokens: 5
   Total latency: 120.5ms
   Avg latency per token: 24.1ms
   Avg performance score: 67.3/100
```

### Dashboard Features
- **Flame Graph**: Token vs operations timeline
- **Bottleneck Distribution**: Types of performance issues
- **Performance Trends**: Latency and score over time
- **Heatmap**: Operation duration patterns
- **GPU Metrics**: Radar charts for detailed analysis

## 🎮 API Endpoints

### **🌐 Deployed API (Modal)**
Base URL: `https://siddhant-k-code--llmtracefx-web-app.modal.run`

```bash
POST /upload-trace          # Upload trace file
POST /analyze-trace         # Analyze trace data
GET  /hardware              # List hardware profiles
GET  /analysis/{id}         # Get analysis summary
GET  /token/{id}/{token}    # Get token details
GET  /explain/{id}/{token}  # Get Claude explanation
GET  /dashboard/{id}        # Get HTML dashboard
GET  /export/{id}           # Export JSON data
```

### **🏠 Local FastAPI Server**
For local development: `http://localhost:8000`

### Example Usage

#### **Production API (Deployed)**
```python
import requests

# Analyze trace data directly
response = requests.post(
    'https://siddhant-k-code--llmtracefx-web-app.modal.run/analyze-trace',
    json={
        "trace_data": {
            "tokens": [
                {
                    "id": 0,
                    "text": "Hello",
                    "operations": [
                        {"name": "matmul", "start_time": 0, "duration": 15.3}
                    ]
                }
            ]
        },
        "gpu_type": "A10G",
        "enable_claude": True
    }
)

analysis_id = response.json()['analysis_id']

# Get dashboard
dashboard = requests.get(f'https://siddhant-k-code--llmtracefx-web-app.modal.run/dashboard/{analysis_id}')
with open('dashboard.html', 'w') as f:
    f.write(dashboard.text)

print(f"Performance score: {response.json()['avg_performance_score']:.1f}/100")
```

#### **Upload Trace File**
```bash
# Upload your vLLM trace file
curl -X POST "https://siddhant-k-code--llmtracefx-web-app.modal.run/upload-trace?gpu_type=A10G&enable_claude=true" \
     -F "file=@your_trace.json"
```

#### **Local Development**
```python
import requests

# Upload trace (local server)
with open('trace.json', 'rb') as f:
    response = requests.post('http://localhost:8000/upload-trace', files={'file': f})

analysis_id = response.json()['analysis_id']

# Get dashboard (local server)
dashboard = requests.get(f'http://localhost:8000/dashboard/{analysis_id}')
```

## 🔬 Trace Format

### vLLM Format
```json
{
  "tokens": [
    {
      "id": 0,
      "text": "Hello",
      "operations": [
        {"name": "embedding", "start_time": 0, "duration": 2.1},
        {"name": "rmsnorm", "start_time": 2.1, "duration": 1.8},
        {"name": "matmul", "start_time": 3.9, "duration": 15.3}
      ]
    }
  ]
}
```

### Event Format
```json
{
  "events": [
    {
      "token_id": 0,
      "token_text": "Hello",
      "op_name": "matmul",
      "timestamp": 12.1,
      "duration": 15.3,
      "metadata": {}
    }
  ]
}
```

## 🎯 GPU Analysis

### Supported Operations
- `rmsnorm` / `layernorm` - Normalization layers
- `linear` / `matmul` - Matrix operations
- `softmax` - Attention computations
- `kvload` / `kvstore` - Key-Value cache operations
- `attention` - Attention mechanisms
- `embedding` - Token embeddings

### GPU Metrics
- **Stall Percentage**: Memory-bound bottlenecks
- **Launch Delay**: Kernel launch overhead
- **Occupancy**: SM occupancy on CUDA and GPU occupancy on Metal
- **Cache Hit Rate**: Memory access efficiency
- **Compute Utilization**: GPU computational usage

Measured values can be supplied in operation metadata using `stall_pct`,
`launch_delay_ms`, `memory_latency_ms`, `occupancy_pct`, `cache_hit_rate`, and
`compute_utilization`. Missing fields use deterministic estimates. A report is
`measured` only when every operation supplies all six metrics; otherwise it is
labelled `mixed` or `estimated`. MLX recorder timings and memory snapshots are
measured, but the six analyzer metrics remain estimated unless you add them.

### Supported Hardware
- **A10G**: 24GB VRAM, 600 GB/s bandwidth
- **H100**: 80GB VRAM, 3350 GB/s bandwidth
- **A100**: 80GB VRAM, 1935 GB/s bandwidth
- **GB10 / DGX Spark**: 128GB coherent unified memory, 273 GB/s bandwidth
- **MLX / Apple Silicon**: Metal backend with runtime device metadata

## 🤖 Claude Integration

### Explanation Types
1. **Performance Summary**: High-level bottleneck analysis
2. **Technical Details**: GPU-specific explanations
3. **Optimization Suggestions**: Actionable improvements
4. **Severity Assessment**: Priority ranking

### Example Claude Output
```
🔍 Token 42 Analysis

**Summary:** MatMul operation shows 33% memory stall due to poor coalescing

**Technical Details:** The matrix multiplication kernel is experiencing
significant memory bandwidth limitations due to non-coalesced memory access
patterns. This is causing the GPU to wait for memory operations.

**Optimization Recommendations:**
• Consider transposing matrices for better memory layout
• Implement tiling strategies to improve cache utilization
• Use tensor cores if available for better compute efficiency

**Severity:** HIGH
```

## 📈 Performance Optimization

### Bottleneck Types
- `memory_stall`: High memory latency
- `launch_overhead`: Kernel launch delays
- `low_occupancy`: Underutilized GPU cores
- `cache_miss`: Poor memory locality
- `compute_underutilization`: Low computational throughput

### Optimization Flags
- `high_memory_stall`: Memory bandwidth issues
- `kernel_fusion_candidate`: Multiple small kernels
- `increase_occupancy`: Low SM utilization
- `improve_data_locality`: Cache optimization needed
- `norm_linear_fusion`: Specific fusion opportunity

## 🚀 Modal Deployment

> **Modal is an optional extra.** It is no longer a runtime dependency of
> `llmtracefx`, because nothing in the library imports it at module scope.
> Install it with `uv sync --extra modal` (or `make install-modal`) before
> running any `modal` command below.

### **🌐 Live Deployment**
- **Web API**: https://siddhant-k-code--llmtracefx-web-app.modal.run
- **Modal Dashboard**: https://modal.com/apps/siddhant-k-code/main/deployed/llmtracefx
- **GPU**: A10G acceleration available
- **Claude Integration**: AI explanations ready

### Functions
- `analyze_trace_modal`: Full trace analysis with GPU
- `explain_token_modal`: Individual token explanations
- `web_app`: FastAPI web endpoint (deployed)
- `run_server`: FastAPI server for local development
- `create_sample_trace`: Generate test data

### Deployment Commands
```bash
# Deploy app
uv run modal deploy llmtracefx/modal_app.py

# Run analysis
uv run modal run llmtracefx/modal_app.py

# Test deployed API
curl -X POST "https://siddhant-k-code--llmtracefx-web-app.modal.run/analyze-trace" \
-H "Content-Type: application/json" \
-d '{"trace_data": {"tokens": [{"id": 0, "text": "test", "operations": [{"name": "matmul", "start_time": 0, "duration": 10.0}]}]}, "gpu_type": "A10G", "enable_claude": false}'
```

### Management
```bash
# View deployment status
uv run modal app list

# Check function logs
uv run modal app logs llmtracefx

# Stop deployment
uv run modal app stop llmtracefx
```

### Self-hosting GLM-5.3-Flash

A separate, budget-guarded harness deploys the 320B-A18B FP8
GLM-5.3-Flash checkpoint on rented accelerators for long enough to
measure it with `collect-api`, then tears it down. Full sequence:
[SELF_HOST_GLM_RUNBOOK.md](SELF_HOST_GLM_RUNBOOK.md).

Start with the offline commands. They need no Modal account, make no
network request and cannot spend anything:

```bash
# The pinned model facts and where each one came from
uv run llmtracefx-deploy recipe

# A conservative session cap that leaves a retry reserve
uv run llmtracefx-deploy budget --credit-usd 30

# Adjudicate a full deployment: worst-case cost, capacity check,
# pinning, and every command it would take to run it
uv run llmtracefx-deploy plan --help
```

`deploy plan` refuses and withholds every money-spending command unless
you supply an explicit budget, GPU type and count, maximum runtime, and a
GPU price with the date you read it. There are no defaults for any of
those, so an incomplete invocation fails rather than assuming.

Note that self-hosting is the *expensive* way to get GLM-5.3-Flash
tokens. For anything about the model rather than about serving it, use
the hosted `z-ai/glm-5.3-flash` endpoint on OpenRouter with `collect-api`
instead; the runbook explains when each is appropriate.

## 🧪 Testing

### Create Sample Data

```bash
python -m llmtracefx.main --create-sample
```

### Run Tests

```bash
uv sync --extra dev --extra test
uv run pytest
uv run ruff check llmtracefx/hardware.py \
  llmtracefx/profiler/mlx_tracer.py \
  llmtracefx/profiler/gpu_analyzer.py tests
```

### Continuous Integration

GitHub Actions runs three jobs on every pull request and on pushes to `main`
(see `.github/workflows/ci.yml`):

| Job | What it does |
| --- | --- |
| `test` | Full `pytest` suite on Python 3.10, 3.11, 3.12 and 3.13 on Linux, plus 3.10 and 3.13 on macOS |
| `quality ratchet` | `ruff check`, `black`, `isort` and `mypy` on changed files only |
| `build` | `uv build`, then installs the wheel into a clean environment and imports it |

macOS is included because GitHub's macOS runners are Apple Silicon. MLX
collection is gated on Darwin plus arm64 in
`llmtracefx/optimizer/collectors/mlx.py`, and `optimizer/manifest.py` records
platform details, so a Linux-only matrix never executes those branches.

The macOS jobs install the `mlx` extra, whose dependency markers only resolve on
Darwin plus arm64, and then run a smoke check that constructs `MLXLMRuntime`.
That constructor is the real runtime boundary: it rejects any other platform and
raises if `mlx` or `mlx_lm` cannot be imported. The check reads the resolved
versions, the accelerator name and a memory snapshot, and deliberately never
calls `load_model` or `stream_generate`, so nothing is downloaded. Without the
extra installed the MLX tests exercise a monkeypatched stand-in and the real
import path never runs, which would leave the macOS jobs costing time without
proving anything.

A separate `.github/workflows/codeql.yml` runs CodeQL static analysis for Python
on pull requests, pushes to `main`, and weekly. It reports into the repository
Security tab rather than blocking pull requests. `.github/dependabot.yml` keeps
Python dependencies and the SHA-pinned GitHub Actions up to date.

The quality job checks only the Python files a change touches, rather than the
whole repository. Older modules still carry lint and typing debt, so a repo-wide
gate would fail every pull request regardless of its contents. Narrowing the
gate keeps new code at the intended standard while that debt is paid down
separately.

Note that the unit is the whole file, not the changed lines. Touching a module
that still carries debt means inheriting all of it: a one line edit to
`llmtracefx/modal_app.py` currently fails on 39 ruff findings and 10 mypy
errors that the change did not introduce.

Most of that is mechanical. `make format` satisfies black and isort and clears
36 of the 39 ruff findings, and `ruff check --fix llmtracefx/modal_app.py`
clears the last 3, which leaves the 10 mypy errors as the only part that needs
real edits. That is deliberate, since it is what makes the debt shrink rather
than persist, but it is worth knowing before editing an older module. Files
added by recent work are already clean and stay that way.

Run the same check locally before pushing:

```bash
make lint-changed
```

That compares against `origin/main`. To compare against something else, call the
script directly:

```bash
./scripts/lint-changed.sh <base-ref-or-sha>
```

The script fails closed. If it cannot work out what changed, because the base is
missing or shares no history with `HEAD`, it exits non-zero instead of reporting
that there is nothing to check. That distinction is the whole point: a gate that
runs its tools over an empty file list reports success, so a ratchet that guesses
when it is confused stops gating without anyone noticing. `scripts/test-lint-changed.sh`
pins that behaviour down, together with renames, deletions, paths containing
spaces, multi-commit pushes and the first push of a branch:

```bash
make test-ratchet
```

It stubs `uv` on `PATH` rather than running the real linters, so each case
asserts on the exact file list the script builds and the status it returns.

Note that `ruff format` is not used anywhere in this project. It and `black`
disagree on a few files here and each undoes the other, so running both can
never pass. `black` is the formatter of record, and `make format` matches what
CI enforces. `ruff check` is still used for linting, which does not conflict.

`make format` and `make format-check` cover the whole tree rather than just
`llmtracefx/`, because the ratchet checks every changed `.py` file wherever it
lives. Two files at the repository root, `launch_dashboard.py` and
`generate_trace.py`, fail black and isort today and were invisible to the older
`llmtracefx/`-only scope, so editing either one meant CI rejecting formatting
that `make format` had declined to fix. `make lint` stays scoped to
`llmtracefx/`, which is how the ratchet scopes mypy.

## 🧭 Inference Optimizer Foundation

LLMTraceFX is evolving from a trace *analyzer* into a workload-aware
inference *optimizer* for open models (initial target: Qwen3.8-27B on
Apple M5 Pro via MLX/llama.cpp Metal, and on CloudRift RTX 4090 via
llama.cpp CUDA). This PR lays the **foundation only** — reliable, tested
primitives that later native Metal/CUDA collectors and tuning logic will
build on:

- **Canonical evidence schema** (`llmtracefx.optimizer.schema`): a
  versioned `ExperimentRecord` capturing run identity, hardware/platform,
  model/tokenizer/quantization, runtime/backend + git revision, exact
  command and config/workload hashes, warmup/repetition/seed metadata,
  token counts, load/tokenize/prefill/decode/total timing, speculative
  decoding (MTP) counters, memory/power when available, task outcome, and
  errors — every measurement is tagged with a `MetricProvenance`
  (`measured_native`, `measured_wall_clock`, `derived`, `estimated`) so
  nothing is silently invented.
- **CPU-only environment manifest** (`llmtracefx.optimizer.manifest`):
  deterministic, non-sensitive OS/arch/CPU/package-version metadata for
  comparability checks. Never collects secrets, usernames, hostnames, or
  full environment dumps.
- **Reproducible experiment runner** (`llmtracefx.optimizer.runner`):
  executes a configured command (argv list, never a shell string) for N
  warmup + N measured repetitions, with timeouts, atomic JSON/JSONL
  artifacts, and resume that skips already-completed repetitions while
  retrying failed/timed-out ones.
- **llama.cpp collector** (`llmtracefx.optimizer.parsers.llama_cpp`):
  parses `llama_print_timings` / `llama_perf_context_print` timing lines
  and speculative-decoding counters into the canonical schema, tolerating
  missing optional lines but raising explicitly on malformed values.
- **First "doctor" rule** (`llmtracefx.optimizer.doctor.speculative`):
  compares speculative-decoding (MTP) runs against a comparable
  autoregressive baseline and reports `regression` / `improvement` /
  `no_significant_difference` / `inconclusive` — it refuses to guess when
  runs aren't comparable, repetitions are too few, or the delta is
  smaller than run-to-run noise.
- **Native Qwen MTP capability detection**
  (`llmtracefx.optimizer.collectors.native_mtp`): checks, against verified
  upstream mlx-lm/mlx-vlm source, whether this environment can produce
  trustworthy native multi-token-prediction evidence for a checkpoint's
  architecture family, validates target/sidecar checkpoint compatibility,
  and records an explicit unsupported result rather than mislabeling
  generic draft-model speculation as native MTP.
- **Deterministic workload matrix** (`llmtracefx.optimizer.workloads`): a
  pinned, versioned catalog of code-completion, structured-JSON, and
  prose-reasoning workloads with deterministic evaluators, materialized to
  2K/8K/16K context tiers with documented, hashed padding, and a dry-run
  matrix generator that plans (never executes) `collect-mlx`/`native-mtp`
  commands across decode modes.

Try it end-to-end with a CPU-only example (no GPU or model download
required):

```bash
uv sync --extra dev --extra test

# 1. Collect a non-sensitive environment manifest
uv run llmtracefx-optimizer manifest

# 2. Convert a llama.cpp log into a canonical ExperimentRecord
uv run llmtracefx-optimizer parse-llama-cpp \
  --run-id baseline-1 --model-id "Qwen/Qwen3.8-27B" --quantization Q4_K_M \
  --stdout-file tests/optimizer/fixtures/llama_cpp/qwen3_8b_baseline_run1.log \
  --output /tmp/baseline-1.json -- llama-cli -m qwen3.8-27b-q4.gguf

uv run llmtracefx-optimizer parse-llama-cpp \
  --run-id baseline-2 --model-id "Qwen/Qwen3.8-27B" --quantization Q4_K_M \
  --stdout-file tests/optimizer/fixtures/llama_cpp/qwen3_8b_baseline_run2.log \
  --output /tmp/baseline-2.json -- llama-cli -m qwen3.8-27b-q4.gguf

uv run llmtracefx-optimizer parse-llama-cpp \
  --run-id mtp-1 --model-id "Qwen/Qwen3.8-27B" --quantization Q4_K_M \
  --speculative-method mtp \
  --stdout-file tests/optimizer/fixtures/llama_cpp/qwen3_8b_mtp_improvement_run1.log \
  --output /tmp/mtp-1.json -- llama-cli -m qwen3.8-27b-q4.gguf --spec-type draft-mtp

uv run llmtracefx-optimizer parse-llama-cpp \
  --run-id mtp-2 --model-id "Qwen/Qwen3.8-27B" --quantization Q4_K_M \
  --speculative-method mtp \
  --stdout-file tests/optimizer/fixtures/llama_cpp/qwen3_8b_mtp_improvement_run2.log \
  --output /tmp/mtp-2.json -- llama-cli -m qwen3.8-27b-q4.gguf --spec-type draft-mtp

# 3. Ask the doctor whether speculative decoding changed performance
uv run llmtracefx-optimizer doctor speculative \
  --baseline /tmp/baseline-1.json /tmp/baseline-2.json \
  --speculative /tmp/mtp-1.json /tmp/mtp-2.json
```

### Collect an MLX-LM run on Apple Silicon

Install the optional MLX runtime dependencies, then point the collector at an
existing local MLX model directory:

```bash
uv sync --extra mlx

uv run llmtracefx-optimizer collect-mlx \
  --run-id local-smoke-1 \
  --model-path "$HOME/.cache/huggingface/hub/models--mlx-community--Llama-3.2-3B-Instruct-4bit/snapshots/<revision>" \
  --model-id mlx-community/Llama-3.2-3B-Instruct-4bit \
  --model-revision <revision> \
  --quantization mlx-affine-4bit \
  --prompt-file examples/optimizer/mlx-smoke-prompt.txt \
  --max-tokens 64 \
  --output-dir artifacts/local-smoke-1
```

The model directory must already exist. `collect-mlx` never downloads or
converts model weights. It writes:

- `record.json`: the canonical experiment record
- `response.txt`: generated output
- `environment.json`: non-sensitive package and platform metadata

The collector records host wall-clock timing around model load, tokenization,
time to first token, decode, and total execution. MLX allocator values are
recorded as native measurements. The normal path synchronizes at phase
boundaries only and does not force evaluation per layer or per token.

An optional existing MLX-LM draft model can be supplied with
`--draft-model-path`. Accepted draft tokens are counted from MLX-LM's
`from_draft` signal. MLX-LM does not expose the number of proposed tokens or
verification time through this API, so those fields remain absent. This is
generic draft-model speculation, not native Qwen3.8 MTP.

### Collect a streaming OpenAI-compatible API run

`collect-api` is the remote counterpart to `collect-mlx`. It streams one
chat completion from any OpenAI-compatible `/chat/completions` endpoint and
writes the same canonical evidence artifacts, so a local run and an API run
can later be compared on identical record structures. The collector itself
is provider neutral: the endpoint, model ID, credential environment variable
name and provider-specific request fields are all configuration.

The examples below are commands, not measured results. This repository
publishes no latency, throughput or cost numbers for any hosted API.

**Dry run first.** `--dry-run` validates the configuration, prints the
credential-free request plan and performs no network request at all:

```bash
uv run llmtracefx-optimizer collect-api \
  --run-id zai-glm-5.3-dry \
  --provider z.ai \
  --endpoint https://api.z.ai/api/paas/v4/chat/completions \
  --model-id glm-5.3 \
  --prompt-file examples/optimizer/api-smoke-prompt.txt \
  --output-dir artifacts/zai-glm-5.3-dry \
  --reasoning-effort high \
  --dry-run
```

**Collecting real evidence.** Export the key first; it is never accepted as
a command argument:

```bash
export ZAI_API_KEY=...   # read by name, never persisted or echoed

# Frontier model
uv run llmtracefx-optimizer collect-api \
  --run-id zai-glm-5.3-1 \
  --provider z.ai \
  --endpoint https://api.z.ai/api/paas/v4/chat/completions \
  --model-id glm-5.3 \
  --prompt-file examples/optimizer/api-smoke-prompt.txt \
  --output-dir artifacts/zai-glm-5.3-1 \
  --max-output-tokens 256 \
  --reasoning-effort high \
  --clear-thinking true

# Efficiency model
uv run llmtracefx-optimizer collect-api \
  --run-id zai-glm-5.3-flash-1 \
  --provider z.ai \
  --endpoint https://api.z.ai/api/paas/v4/chat/completions \
  --model-id glm-5.3-flash \
  --prompt-file examples/optimizer/api-smoke-prompt.txt \
  --output-dir artifacts/zai-glm-5.3-flash-1 \
  --max-output-tokens 256 \
  --reasoning-effort low
```

#### Provider-specific request fields

`reasoning_effort` and `thinking` are not portable OpenAI chat-completions
parameters. They are kept in a typed `ProviderExtensions` block and are only
sent when explicitly requested, so a record never implies a default the
provider did not actually apply. Per Z.ai's published documentation
([chat completions](https://docs.z.ai/api-reference/llm/chat-completion),
[GLM-5.3-Flash guide](https://docs.z.ai/guides/vlm/glm-5.3-flash)):

| Flag | Body field | Documented values | Notes |
| --- | --- | --- | --- |
| `--reasoning-effort` | `reasoning_effort` | `low`, `high`, `max` | `max` is the provider default for `glm-5.3` and `glm-5.3-flash`. Other values are rejected by the API. |
| `--thinking` | `thinking.type` | `enabled`, `disabled` | `glm-5.3` and `glm-5.3-flash` accept only `enabled`. |
| `--clear-thinking` | `thinking.clear_thinking` | `true`, `false` | Provider default is `true`. It controls whether `reasoning_content` from *previous* turns is cleared. It does not change whether the current turn thinks. |
| `--provider-request-id` | `request_id` | any string | Optional caller-supplied ID. Leave unset to let the provider generate one. |

Two flags shape the collection rather than the request:

| Flag | Default | Notes |
| --- | --- | --- |
| `--request-timeout` | `120.0` | Whole-response budget in seconds, covering connect, TLS and every read. No retries are performed. |
| `--retained-event-limit` | `20000` | How many per-event timing rows the timeline keeps. Counters, rates and the inter-token distribution stay exact past the bound; only the individual rows stop. Part of the config identity hash. |

#### Artifacts

```
artifacts/<run-id>/
  record.json         canonical ExperimentRecord, identical schema to collect-mlx
  response.txt        final answer text only
  api_evidence.json   streaming timeline, statistics, provider usage, failure detail
  environment.json    non-sensitive client package and platform metadata
  artifacts.json      completion marker listing the evidence files with sha256
```

`--dry-run` writes only `request_plan.json`.

The four evidence files are written one at a time, so a crash or a full disk
partway through would otherwise leave a fresh `record.json` sitting next to
stale or missing evidence from an earlier run, which reads as a success. To
make that detectable, `artifacts.json` is deleted before the set is written
and written last. A reader can call
`llmtracefx.optimizer.collectors.artifact_set_is_complete(output_dir)`, which
returns `True` only when a bounded regular, non-symlink marker names the exact
canonical artifact set and every bounded regular file still matches its
sha256. Paths outside the run directory, partial sets, duplicates, symlinks
and special files are rejected. Publication uses randomized exclusive
temporary files in the destination directory, so a pre-created predictable
symlink cannot redirect an atomic write. Both sides of the integrity check
work on raw bytes, so an answer containing CRLF or a lone CR verifies
correctly instead of looking tampered with because text mode rewrote it.

#### What is measured and how it is labelled

Client-observed timing and provider-reported usage are kept strictly apart
and are never mixed into a single unlabelled number.

- **Client-measured** (`timeline`, `provenance: measured_wall_clock`), taken
  from a monotonic clock: request start, response headers, first body byte,
  first content token, last event, completion.
- **Time to first content token** is the offset of the first non-empty
  `delta.content` string. Empty content deltas (GLM's role-only opening
  chunk and its final chunk both carry `""`), metadata chunks,
  reasoning-only deltas and `:` keepalive comments do not count.
- **Provider-reported** (`usage`, `provenance: provider_reported`):
  `prompt_tokens`, `completion_tokens`, `total_tokens`,
  `prompt_tokens_details.cached_tokens`, and
  `completion_tokens_details.reasoning_tokens` when the provider sends it.
  A metric the provider does not report stays `null`. It is never inferred
  as zero. Values that arrive malformed are listed in
  `usage.malformed_fields` rather than silently discarded.
- **Derived** (`content_delta_rate_per_second`, `inter_content_delta`):
  computed from SSE content deltas, which are not necessarily one token
  each, so the field is labelled `derived` and named after deltas rather
  than tokens. `provider_completion_tokens_per_second` combines a
  provider-reported count with a client-measured window and carries an
  explicit mixed-provenance note.
- **Two windows, each matching its numerator.** `content_window_ms` runs
  from the first content delta arrival to the last and is the denominator
  for `content_delta_rate_per_second`. `generation_window_ms` starts at the
  first generated event of any kind, reasoning or content, and is the
  denominator for `provider_completion_tokens_per_second`, because Z.ai
  counts reasoning tokens inside `completion_tokens` and dividing them by
  the visible window would credit a long silent reasoning phase to a short
  answer. With no reasoning deltas the two windows are identical. Both
  exclude the request, the response headers, any leading metadata chunk and
  the trailing usage, finish-reason and `[DONE]` events. The windows are
  persisted alongside the rates because a rate is only as trustworthy as
  the window it came from: two deltas a few microseconds apart produce a
  very large number that is arithmetic rather than evidence, and only the
  window makes that visible. A window with no measurable width leaves its
  rate `null` rather than zero.
- **Reasoning that was billed but never observed withdraws the rate.** The
  window above only works because a reasoning phase announces itself as
  reasoning deltas. Z.ai can report a positive `reasoning_tokens` while
  streaming no reasoning delta at all, which is what happens when the
  thinking is done server-side and only the answer is sent. The numerator
  still contains those tokens, but the window now starts at the first
  visible content delta, so the entire reasoning phase sits outside the
  denominator and the rate reads high in proportion to how long the model
  spent thinking. There is no honest window to divide by, because the
  evidence never showed when that work began. So
  `provider_completion_tokens_per_second` is left `null` in that case and
  `provider_completion_tokens_per_second_unavailable_reason` records why,
  rather than publishing a number that is wrong in a flattering direction.
  The rate remains available only when thinking was explicitly disabled, the
  provider reported zero reasoning tokens without contradicting itself, or
  reasoning was observed from the first generated event. A late reasoning
  delta does not validate an earlier window and overrides disabled or
  zero-count assurances.
- **A visible-token rate is published only when it can be.**
  `provider_visible_completion_tokens_per_second` divides
  `completion_tokens` minus the provider-reported reasoning tokens by
  `content_window_ms`. When the provider does not report a reasoning token
  count the field is `null`, because a missing count is not zero. It is also
  `null` when a reported zero contradicts a streamed reasoning delta, because
  that leaves no trustworthy visible-token numerator.
- **The token rate is an estimate, not a measurement.** Its window starts
  at the first generated delta, so that delta's own generation time is
  outside it, and when the delta count is far below `completion_tokens` the
  window endpoints are delta boundaries rather than token boundaries. The
  persisted note says so.
- **One clock read per network chunk.** Several SSE events can arrive in a
  single chunk. Every event decoded from a chunk is stamped with that
  chunk's arrival time, so the parser's own CPU time never appears as
  inter-token latency. Deltas that shared a packet therefore show a zero
  gap, which is what was actually observed.
- **The finish reason is classified before it is redacted.** Redaction
  rewrites provider-controlled text, so classifying from the redacted
  string would let a credential that happens to contain `error` dissolve
  `network_error` and turn an aborted generation into a success.
  `finish_reason` holds the redacted provider text,
  `finish_reason_classification` holds `terminal`, `failure` or
  `unrecognized`, and `finish_reason_code` holds the documented spelling
  this collector recognized. The code is drawn from a configured set held
  in this repository rather than from provider bytes. A reported failure is
  sticky: nothing in the wire format stops a provider sending a second
  `finish_reason`, and last-write-wins would let a trailing `stop` erase an
  earlier `network_error` and publish an aborted generation as a success with
  a full latency timeline. Two ordinary terminal reasons still take the later
  one.
- **Which reasons end a stream is configuration, not collector policy.**
  `finish_reason` is not fully standardized: OpenAI documents one set and
  providers add their own, so the meaning of a reason belongs to the
  endpoint being measured. `FinishReasonVocabulary` carries the two sets as
  typed configuration, the applied vocabulary is written into
  `plan.finish_reasons` so a record says which reading produced its verdict,
  and it feeds `config_hash`, because two runs that read the same reason
  differently are not the same measurement configuration even though they
  send identical requests. The default is the union of the OpenAI reasons
  and Z.ai's documented additions (`sensitive` as terminal,
  `network_error` and `model_context_window_exceeded` as failures), which
  is what this collector was validated against. Those strings appear in no
  other OpenAI-compatible vocabulary known at the time of writing, so the
  union classifies a non-Z.ai stream exactly as the OpenAI set alone would;
  `FinishReasonVocabulary.openai_only()` drops them for an endpoint known to
  reuse one differently. A reason in neither set is `unrecognized`, which is
  deliberately not a synonym for terminal: an unknown reason is no evidence
  that generation completed, so a stream ending on one without `[DONE]` is
  still reported as truncated. Custom vocabulary entries also pass credential
  preflight before the plan is built, since those strings are persisted as
  configuration and may become a finish-reason code.
- **The canonical `prefill` and `decode` fields are left unset.** They name
  model phases, prompt processing and generation, and neither is observable
  from outside a hosted API. The client-side interval before the first
  content token also contains DNS, connection setup, TLS, request transfer
  and any server-side queueing. The interval after it runs to the last SSE
  event, which can be a usage chunk or `[DONE]` sent long after generation
  ended: a provider that waits a minute before its sentinel would add that
  minute to `decode`, with no generation in it. Publishing those two numbers
  under those names would assert a decomposition the evidence does not
  support. `total` is genuinely measurable end to end and is kept, and the
  client-observed offsets are kept in the API evidence timeline under names
  that say they are client-observed and include transport.
- **`runtime.backend` stays empty for a hosted run.** That field is the
  local compute backend (`Metal`, `CUDA`, `CPU`). `runtime.provider` exists
  so a remote run is recorded without overloading it, and writing a
  transport into `backend` would put `remote-http` everywhere a reader
  expects hardware.
- **The request timeout is a whole-response budget, not just an idle
  timeout.** Passed to the transport it applies per socket operation, so a
  server that emits a keepalive comment before each one expires resets it
  forever and the run neither completes nor fails. The stream is checked
  against a monotonic deadline as well, and a response that outlives its
  budget is abandoned as a `timeout` failure.
- **Silence about reasoning suppresses the provider completion rate.** Not
  asking for reasoning is not the same as reasoning being off. Omitting
  `reasoning_effort` leaves the provider free to apply its own default, and
  for `glm-5.3` and `glm-5.3-flash` that default is `max`, so the plainest
  request is a thinking request. A provider that then reports neither a
  reasoning delta nor a reasoning token count has said nothing either way,
  and dividing completion tokens that may include reasoning by a window
  that opens at the first visible character would overstate throughput.
  Three things provide enough evidence: `--thinking disabled` on the request,
  an explicit reasoning token count of zero, or reasoning observed from the
  first generated event. Otherwise the rate is left unavailable with the
  reason recorded.
- **The request timeout bounds the whole response.** DNS resolution,
  address attempts, connection setup, TLS, request upload, response headers
  and every body read draw from one monotonic deadline. A watchdog closes
  the live socket at that deadline, so byte-dripped status lines, headers or
  chunk framing cannot reset an idle timeout forever. EOF and read errors
  are checked against the deadline before they can become a successful
  terminal stream. The socket is also
  found through both success and `HTTPError` wrapper shapes before each body
  read, and its timeout is reduced to the remaining budget. DNS uses one
  bounded daemon worker rather than starting an uncancellable thread per
  timeout, and the TLS socket is registered before its handshake so the
  watchdog can interrupt that phase too.
- **The per-event timeline is capped by explicit configuration.** One
  timing row per SSE event with no limit lets a chatty provider decide how
  large the artifact gets, and serialization amplifies it again. The bound
  is a typed field on the collection configuration with a documented
  default, settable with `--retained-event-limit` and validated as a
  positive integer, so a run cannot silently discard its whole timeline.
  Past the bound the rows stop accumulating and the timeline records
  `events_truncated`, exact total, retained and dropped counts, and the
  `retained_event_limit` that produced them. Error frames and `[DONE]`
  participate in the same accounting. Content, reasoning and metadata
  counters and the first and last offsets used by derived metrics stay
  exact. The bound is recorded in the request plan and config identity hash:
  two runs that kept different amounts of evidence are not the same
  configuration.
- **Asking for reasoning and hearing nothing back suppresses the rate too.**
  When the request enabled thinking or set `reasoning_effort` and the
  provider returns neither reasoning deltas nor a reasoning token count,
  reasoning cannot be ruled out of `completion_tokens`, so
  `provider_completion_tokens_per_second` stays `null` with a recorded
  reason. The test is what this collector asked for, not who the provider
  is, so it stays meaningful for any OpenAI-compatible endpoint.
- **A request id in an error body is kept.** On a non-200 the stream never
  runs, so headers used to be the only source. A `request_id` at the top
  level of the error payload or inside its `error` object is now read as
  well, redacted like any other identifier, and the header value still wins
  when both are present.

#### Privacy guarantees

- The API key is read only from the environment variable named by
  `--api-key-env` (default `ZAI_API_KEY`). There is no `--api-key` flag, and
  prefix abbreviation is disabled on this subcommand so `--api-key` cannot
  resolve to `--api-key-env`. A credential-shaped flag such as `--api-key`,
  `--api_key`, `--token` or `--secret`, in either the separate or the
  `--flag=value` form, is refused before argparse sees it, because argparse
  quotes an unrecognized argument straight back into stderr, value and all.
  The refusal names the flag and points at `--api-key-env`, never the value.
  The scan stops at a bare `--`, because everything after it belongs to a
  recorded external command rather than to this program, and `llama-server`
  has its own `--api-key`. Those values are redacted where `parse-llama-cpp`
  persists them instead, so the flag stays visible as evidence and the
  credential does not reach `record.command.argv`. A separate value is
  redacted whatever it looks like, because such a flag always takes one and
  the base64url alphabet starts a value with `-` often enough to matter.
  The one exception is another credential flag: letting the first flag
  consume it would skip the second flag's own handler and append the real
  credential verbatim.
- No parse diagnostic repeats a value the caller supplied. A token is a name
  only when this program defined it, which the parser itself is asked; token
  syntax is not evidence. Option names and the usage block are kept, since
  they carry no caller input and are what make the error actionable, and
  everything else is replaced. That includes the tail of an attached short
  cluster such as `-p<secret>`, which is a value wearing an option's clothes,
  including when the value ends in the `=` padding a base64 key carries. It
  also includes a long option with a dropped space, `--api-key<secret>`: when
  a defined option is a prefix of the token only the tail is replaced, so
  `--dry-run<secret>` still reads as `--dry-run[REDACTED]`, and otherwise the
  whole token goes. This program's own vocabulary is put beyond reach first,
  so mistyping `collect-ap` does not rewrite the valid `collect-api` in the
  list of choices; a literal is only protected when no supplied value contains
  it, so a secret that embeds an option name is still replaced whole. A secret
  pasted into the wrong option is still a secret.
- The credential value is never written to an artifact, never logged, never
  hashed and never included in the reconstructed command. `HTTPRequest`
  overrides `repr` so a traceback cannot surface the `Authorization` header.
  Only header *names* are persisted.
- **A credential pasted into the name slot is contained as well.** The refusal
  for `--api-key` points the caller at `--api-key-env`, and the mechanical
  response is to keep the value and change the flag, which puts the credential
  exactly where a variable name is expected. Two independent rules apply.
  First, the value must be a conventional exported variable name, uppercase
  with digits and underscores, which rejects the `sk-`, `sk_live_` and `ghp_`
  shapes real keys take; the refusal never repeats the rejected value. Second,
  a name is only treated as a name when the environment defines it, which is
  the one thing a caller cannot fake and which catches an all-uppercase key
  such as an AWS access key id. An unproven name is replaced by `[REDACTED]`
  in `credential_env_var` and in the reconstructed command, in both the
  separate and the `--flag=value` spelling. This is the same rule the parse
  diagnostics use one level down: syntax is not evidence, so the authoritative
  source is asked instead.
- The variable name is not part of the request identity hash. It does not
  change the bytes on the wire, and hashing it would persist a derivation of
  a value that may be the credential itself. The missing-variable diagnostic
  does not name the variable either, for the same reason.
- Collection aborts before any request if the credential value appears in the
  run id, endpoint, provider label, model id, model revision, prompt, system
  prompt, any provider extension string or command arguments. The check also
  decodes percent-encoded forms, because a key pasted into a URL is normally
  encoded and `abc%2Fdef` is trivially reversible once persisted. Case
  variants, `+` for space and a few rounds of double encoding are covered by
  decoding the candidate rather than enumerating encodings of the key. Very
  short values are compared literally only, so the refusal does not fire on
  coincidence.
- Endpoint query keys and values are stripped from every persisted form of
  the command, including `record.json`, and from `HTTPRequest.__repr__`.
  `endpoint_query_keys` records one redaction marker per query pair, while
  the config hash still distinguishes the original keys and values. This
  contains opaque key-shaped names even when no configured credential is
  available to match them. Identity uses the exact raw query representation,
  including field order, separators, invalid percent-encoded bytes and an
  empty `?` delimiter, because routers, signatures and caches may distinguish
  those request targets.
- A malformed endpoint is reported as a sanitized error rather than an
  escaping `ValueError`. `urlsplit` raises on an unclosed IPv6 bracket and
  `SplitResult.port` raises on a port that is not an integer in range, both
  with messages that can quote the netloc, so every parse and every lazy
  property access is guarded.
- Surrounding whitespace is stripped from the credential, and a value that
  cannot be sent as an HTTP header value (control characters, non latin-1) is
  rejected by name before the request is built. An unencodable header would
  otherwise make `http.client` raise an error whose message contains the whole
  header value.
- Prompt and system prompt are hashed, never copied into artifacts.
- Redirects are refused, so the credential is never replayed to a host you
  did not name. Plain `http` is rejected except for loopback hosts, and
  loopback HTTP disables environment proxies so the Authorization header
  cannot be routed off-host in clear text.
- Reasoning content is counted, never stored: `api_evidence.json` records
  `reasoning_delta_count`, `reasoning_characters` and
  `reasoning_text_persisted: false`, and `response.txt` holds the final
  answer only.
- Every provider-controlled string is passed through a redactor before it is
  persisted, not only error messages: generated content, response and request
  IDs, the echoed model name, the finish reason, response header request IDs,
  provider error codes, and both the names and the values of rate-limit
  headers. A provider that echoes your key back in any of those fields cannot
  get it into an artifact or onto the terminal. The redactor removes the known
  credential and any bearer-token shape, and it preserves whitespace in
  generated text so redaction does not quietly alter the answer.
- Redaction runs before any transform that could hide a match, and it matches
  more than the literal value. The credential is matched case insensitively,
  because header names are lowercased before they are persisted, and each run
  of whitespace inside the credential is matched flexibly, because a space is
  a legal header value character and different sinks treat whitespace
  differently. `response.txt` preserves the answer's own spacing and still
  gets the same coverage as a collapsed diagnostic. Provider payloads that a
  byte cap or a cut connection truncated are repaired at the cut point,
  because truncation can slice through an echoed credential and leave a
  trailing fragment that an exact-substring match would not recognize. That
  repair is applied to truncated evidence only, so a complete answer is never
  altered, and it uses the same flexible matching as the whole-value scrub so
  the two controls guarding the same threat have equal strength.
- Redaction also recognizes percent-encoded echoes. A provider that
  reflects a key it received in a URL sends back `sk-slash%2Fcredential`
  rather than `sk-slash/credential`, which is reversible and therefore
  still a leak. Each credential character is matched as its literal form,
  its `%XX` form or its double-encoded `%25XX` form, case insensitively, so
  a mixed encoding is covered without enumerating whole-string variants.
  Matching starts from candidate positions found by a single scan rather
  than by compiling one pattern per prefix length, which keeps the cost of
  a long credential flat. Truncation is repaired inside an encoding too, not
  only between characters. A cut that lands after `%` or `%2` leaves a
  fragment that is not yet a character in any spelling, so an exact match
  fails and, without the repair, everything before the cut survives: a
  20 character key truncated to `secret%2` would keep 6 of its characters in
  the artifact. A trailing fragment that is a proper prefix of the next
  credential character's literal, `%XX` or `%25XX` form is treated as that
  character having been cut, and redaction starts from where the credential
  began. Single and double encodings and mixed-case hex are all covered.
- Backslash escapes count as spellings of the credential, alongside
  percent-encoding. A JSON encoder writes the key as `\u0073\u006b...`
  and a Python `repr` writes it as `\x73\x6b...`; both are one mechanical
  decode away from the key. This matters most where the text is not
  re-parsed: a non-JSON error body is persisted as it arrived, so an escape
  in it stays an escape rather than being decoded back into characters the
  literal matcher would catch. `\xXX`, `\uXXXX`, `\UXXXXXXXX` and the
  UTF-16 surrogate pair a JSON encoder emits outside the BMP are all
  matched, in either hex case, and a cut inside one is repaired the same
  way a cut inside `%2F` is. The same applies to a space spelled `%2520` or
  `\u0020`: a cut inside an encoded whitespace run used to leave every
  credential character before it exposed. The matcher and the truncation
  repair are generated from one list of spellings, so a form cannot be
  added to one without the other.
- The one-character escapes count too. A JSON encoder may write `/` as
  `\/`, and every encoder writes a backslash as `\\` and a double quote as
  `\"`; Python adds `\'`. These are shorter spellings of the same leak the
  numeric escapes carry, and `json.loads` recovers the key from them
  exactly, so they are matched and their truncation is repaired alongside
  the numeric forms. Raw, once-escaped and twice-escaped forms are covered.
  A non-ASCII whitespace character keeps its own percent and JSON spellings,
  such as `%C2%A0` and `\u00A0`, rather than being reduced to ASCII space
  forms. The truncation floor counts decoded credential characters, including
  every character in a collapsed whitespace run.
- Whole-value re-encodings count as spellings too. Percent-encoding and
  backslash escapes keep the credential's own characters, so a matcher that
  walks characters sees them. Base64, base64url, hex and octal do not: they
  share no character with the key, and a logging proxy or a request echo
  that returns one of them defeats a character-wise matcher completely
  while staying trivially reversible for anyone reading the artifact. The
  three byte alignments of base64 are each derived, so a key embedded in a
  longer encoded blob is still found, minus the leading and trailing
  characters whose bits are shared with the surrounding bytes. Unicode
  normalization is included for the same reason: an intermediary that
  normalizes an accented key returns a different sequence of codepoints
  that renders identically. These extra spellings apply only above a
  minimum credential length, because a short value's encodings collide with
  ordinary text and a redactor that fires on noise destroys the evidence it
  is guarding.
- Matching is a deterministic walk over positions, not a backtracking
  search. Several spellings of the same character are prefixes of each
  other: a literal backslash is a prefix of `\\`, of `\x5C`, of `\u005C` and
  of the octal `\134`, and `%` is a prefix of `%25`. Expressing that as a
  regex alternation and repeating it is the classic exponential
  backtracking shape, and provider text is untrusted, so a key made of
  backslashes would be a denial of service against the redactor itself.
  Committing to the longest spelling instead is not a fix, because it
  produces real misses: a credential containing a literal `%25` would never
  match. Carrying the set of reachable positions forward one element at a
  time is exact and stays linear in the length of the text.
- Repairing a credential cut short by truncation searches only a bounded
  window at the end of the text. Truncation removes the tail, so a cut
  credential runs to the last character, and a match consumes a bounded
  number of characters; a candidate starting further back than that cannot
  reach the end and so cannot be the one. Searching the whole body was
  linear in candidate starts and linear again in the walk from each, and
  provider text can put a candidate start every few characters: 256 KiB of
  near misses took about six seconds, quadrupling with each doubling. That
  is reachable from the network and it runs after the read loop, where the
  request deadline no longer applies, so it was a way to spend unbounded
  CPU on a response that had already failed.
- The pre-flight check that refuses to persist a credential uses that same
  matcher, not a plain substring test. The two controls guard the same
  threat from opposite ends, so a difference between them is a gap: a
  provider request ID echoing the key in lowercase, or a provider extension
  holding it with a tab where a space was, would be scrubbed on the way out
  by the redactor and yet pass the check that decides whether a
  `RequestPlan`, its reconstructed command or a persisted config may be
  written at all. Both now match case insensitively, tolerate whitespace
  differences and see through percent-encoding, and the credential
  environment variable's own name is part of what is checked. The match runs
  at every credential length. A minimum length still applies, but only to the
  extra rounds of percent-decoding, which is where a coincidental hit is
  plausible; gating the ordinary match on it meant a short key was scrubbed
  by the redactor and waved through by the check. The output directory is
  checked too, since a credential there is written into the filesystem as a
  pathname, where no downstream redactor can reach it.
- A credential flag that swallowed its value is redacted in a recorded
  external command. Separate, equals, dropped-space long and attached `-k`
  or `-p` forms are covered, including values beginning with `-` or `_`.
  Glued prefixes come from an explicit credential-option vocabulary and the
  tail must also have credential evidence, such as a case boundary, digit or
  known key prefix. This keeps `--authentication-method`,
  `--authorization-policy`, `--tokenizer-model` and `--api-key-file`
  reproducible instead of corrupting legitimate command literals, including
  their `--option=value` forms.
- The refusal to accept a credential-shaped query parameter does not repeat
  the parameter name. The name is caller-controlled and is exactly where a
  key ends up when someone puts it in the URL, so quoting it back into a
  message that reaches stderr and a failure record would republish the
  thing the refusal exists to prevent. The diagnostic names no part of the
  endpoint at all.
- Because that diagnostic says nothing, a false positive is expensive: the
  operator is refused and cannot tell which parameter caused it. The name is
  therefore tokenized on separators, on case changes and between letters and
  digits, and each component is judged whole. A bare substring search refused
  `design`, `assignment`, `monkey`, `insignia`, `signal` and `keyword`, all
  ordinary parameter names, on the strength of `sig` or `key` buried inside
  them. A glued compound carries no separator and no case change, so a
  component also counts when recognized words cover it end to end and at
  least one of them names a credential outright. The cover has to be
  complete, which is what keeps `apikey`, `xapikey`, `secretkey`,
  `clientsecretkey` and `sessiontoken` refused while letting `keyword` and
  `monkey` through: their leftover `word` and `mon` are not words this
  recognizes. A cover made only of qualifiers, such as `appid`, does not
  count either. Splitting a component once into two parts was not enough,
  because `xapikey` is three words and the substring search it replaced had
  caught it.
- The credential vocabulary covers more than words spelled `key` or `token`:
  `sid`, `jwt`, `pwd` and `passphrase` each name authentication material
  outright, so all four are credential nouns. The cover rule keeps that from
  spreading: `sidebar` and `sidecar` are still accepted, because `ebar` and
  `car` are not words this recognizes.
- The credential vocabulary also covers session and authorization-code
  material. `sessionid`, `session_id`, `authcode` and `authorizationcode` are
  credential compounds, while bare `session` and `code` remain ordinary
  parameter names.
- A complete cover cannot see a credential glued to a word the tables do not
  know, so `openaiapikey`, `myapikey` and `zaiapikey` were accepted. A
  component is also refused when a contiguous run of two recognized words,
  at least one of them a credential noun, sits anywhere inside it. Two words
  is what makes the surrounding text safe to ignore: one ambiguous noun
  proves nothing, which is why `keyword` and `monkey` still pass on `key` and
  `design`, `signal`, `insignia` and `assignment` still pass on `sig`. The
  scan carries at most six states per position, so it stays linear and does
  not undo the bound below.
- `param` and `value` are not credential words. They were briefly
  qualifiers, so that `tokenvalue` and `secretparam` would be refused, and
  they refused `hotkeyvalue`, `partitionkeyvalue`, `sortkeyvalue`,
  `rowkeyvalue` and `keyvalue` along with them. Those are ordinary key-value
  store parameter names and there is no principled rule that separates them
  from `tokenvalue`, which is built the same way out of the same parts. Both
  words are gone, and `tokenvalue` and `secretparam` are accepted as a
  result, on the same bounded-cost reasoning as `sessionid` above.
- The cover search looks ahead no further than the longest word either table
  spells. Without that bound every reachable offset rescanned every remaining
  substring, so a query key made of a few thousand repeated qualifier
  characters cost seconds of work before the endpoint it belonged to was even
  rejected: an 8,003 character key took over seventeen seconds, and the cost
  quadrupled with each doubling. No cover step can use a word longer than the
  longest one there is, so nothing is missed, and the bound is derived from
  the tables rather than written down, which keeps a longer noun added later
  reachable.
- The whitespace run cap is a floor, not a ceiling. It stops a matcher being
  walked across unbounded whitespace hunting for its next element, but
  applying it to a credential whose own spelling contains a longer run left
  the matcher unable to consume its own value, and the exact secret survived
  verbatim. Each matcher raises its cap to its own longest literal run, so a
  credential can always match itself while ordinary ones are unaffected.
- Transport encodings are built from every literal spelling, not only the one
  supplied. Normalization and encoding compose in both orders, and an
  intermediary that normalized an accented key before base64 encoding it
  produced bytes that decode straight back to the credential yet matched no
  spelling the redactor knew.
- The cover search looks ahead no further than the longest word either table
  spells. Without that bound every reachable offset rescanned every remaining
  substring, so a query key made of a few thousand repeated qualifier
  characters cost seconds of work before the endpoint it belonged to was even
  rejected: an 8,003 character key took over seventeen seconds, and the cost
  quadrupled with each doubling. No cover step can use a word longer than the
  longest one there is, so nothing is missed, and the bound is derived from
  the tables rather than written down, which keeps a longer noun added later
  reachable.
- A provider cannot end a run with no evidence at all by sending JSON that
  `json` refuses past its own limits. An integer literal over the
  interpreter's digit cap raises a plain `ValueError` and deep nesting raises
  `RecursionError`, neither of which is a `JSONDecodeError`, so both used to
  escape as an unhandled crash rather than a failure record. Both are now
  stream decode failures with the usual canonical artifacts.
- A JSON string containing an unpaired surrogate is also a stream decode
  failure. Python accepts that escape during `json.loads`, but it cannot be
  encoded as UTF-8 for `response.txt`; rejecting it while handling the event
  prevents a partial artifact set during publication.
- Stream bytes, accumulated visible content and retained content-arrival
  timings have explicit safety bounds. A fast provider therefore cannot
  exhaust memory inside the wall-clock deadline with an unterminated frame or
  millions of tiny content deltas, and every successfully published artifact
  stays within the verifier's own size limit.
- Valid events before malformed UTF-8 are yielded before the decoder reports
  the suffix. Once `[DONE]` is accepted, trailing bytes are ignored
  consistently whether the transport delivered them together or across two
  reads. EOF also terminates a final comment line without turning an otherwise
  completed stream into a false truncation.
- A provider token count above the largest integer a float represents exactly
  is recorded as malformed and dropped. Smaller but still enormous counts
  parsed fine and then raised `OverflowError` in the rate arithmetic; beyond
  that bound a count either cannot become a float at all or becomes one with
  silent precision loss, so any rate derived from it would be fiction. The
  metric stays missing, which is the same rule applied everywhere else.
- A null provider completion token rate always records why it is null. The
  reasons that a hidden reasoning phase makes the window unusable were
  already recorded, but a run whose generated tokens all arrived inside one
  network chunk has no window with any width to divide by, and that case left
  the rate null with nothing said about it, which reads as a metric the
  provider never sent rather than one that could not be measured.
- Provider usage that implies zero visible tokens while non-empty visible
  content was streamed is internally inconsistent. Completion and visible
  rates remain null with an explicit reason rather than becoming valid-looking
  zeroes.
- No parse diagnostic repeats a value the caller supplied, in any
  rendering. argparse formats several of its messages with `%r`, so a value
  containing a newline, a tab, a zero-width space or a backslash reaches
  stderr as an escape sequence that does not match the raw string. Both the
  value and its `repr` body are scrubbed, longest rendering first. Short
  values are scrubbed as well. Quoted renderings are replaced directly, and
  bare short values are replaced only as whole diagnostic tokens, so
  `unrecognized arguments: xy` is contained without deleting `xy` from
  unrelated words. The scrub state lives in a `ContextVar`
  installed per parse rather than in globals set once by `main`, so
  `build_parser().parse_args(...)` is as safe as the real entrypoint and
  `main` no longer leaves state standing after it exits. An absent
  `ContextVar` is distinguishable from an empty one, which an emptiness test
  was not: stale state from `main` looked exactly like an active enclosing
  scope and suppressed the installation of a real one.
  `parse_args` and `parse_known_args` are both public and both reachable on
  their own, and `parse_args` reports unrecognized arguments itself after the
  inner call has returned, so each installs the scope and a nested parse
  inherits the enclosing one instead of narrowing it. Parsed handlers run
  inside the same scope, including normal returns, `SystemExit` and errors.
  Returned collection failures and dry-run write errors use the same scrub,
  and dry-run output is printed only after its plan was written successfully.
- `--dry-run` applies the same refusal a real run does. If the configured
  environment variable holds a value that appears in the endpoint or the
  command, the pre-flight check fails instead of printing a plan that a real
  run would reject, and the rendered plan is scrubbed as a second line of
  defence. Endpoint rejection is generic and never repeats its netloc, path,
  query key or query value.

#### Failure evidence

Transport and protocol failures produce a failure-shaped record instead of
an exception or a silent success: `outcome.success = false`, an
`error.category`, and the same four artifacts. Categories are
`http_status`, `timeout`, `connection`, `stream_decode`,
`stream_truncated`, `provider_error_payload` and `missing_content`. Safe status code, provider
error code, provider request ID and rate-limit headers are preserved where
the provider returns them. Both the OpenAI `{"error": {...}}` shape and
Z.ai's bare `{"code": ..., "message": ...}` shape are recognized. Protocol
level failures that Python raises as `http.client.HTTPException` rather than
`OSError`, such as an `IncompleteRead` when a proxy hangs up mid chunk or a
`BadStatusLine` from a garbled response, are recorded as `connection`
failures.

A stream is only accepted when it reaches a terminal condition the provider
documents: a `[DONE]` sentinel, or a `finish_reason` of `stop`, `length`,
`content_filter`, `tool_calls`, `function_call` or `sensitive`. A connection
that closes cleanly after some content but before either of those is recorded
as `stream_truncated`, not as a short success. Accepting it would silently
convert a dropped connection into a real answer and, worse, into a plausible
latency measurement. A body shorter than its own `Content-Length` is a
`connection` failure: `http.client` returns a clean end of file there rather
than raising, so the collector compares bytes read against the declared
length itself.

Z.ai documents `network_error` and `model_context_window_exceeded` alongside
those successful reasons, so a stream can carry content, one of these, and
`[DONE]` all at once
([API reference](https://docs.z.ai/api-reference/llm/chat-completion)). The
sentinel does not outrank them. It reports that the transport finished, not
that the generation did, so a run ending on either reason is recorded as
`provider_error_payload` with the reason as the provider error code.

An event the stream leaves pending is discarded rather than dispatched. The
event-stream rules dispatch on a blank line, and end of stream is not one, so a
frame still buffered when the body ends was cut in transit. Dispatching it
anyway would let an unterminated `data: [DONE]` close a truncated collection as
though the provider had ended it cleanly. The run is recorded as
`stream_truncated` instead. A lone `\r` at the very end of a body is the
exception: it is deferred while streaming because the next chunk may start with
`\n`, but at end of stream nothing can follow it, so it is resolved as the line
terminator the rules say it is rather than treated as a cut line. One leading
U+FEFF byte order mark is ignored, as
the rules require, including when its bytes arrive split across chunks;
without that the first field name is not `data` and the whole first event is
silently dropped.

A named `event: error` frame is treated as a provider error even when its
payload is only a message string, is empty, or is the `[DONE]` sentinel. The
event name is resolved before its data is interpreted, because handling
`[DONE]` first would let a provider close a failed stream as though it had
finished cleanly. A `choices` field that is present but not a list of objects
is a `stream_decode` failure rather than an ignorable metadata chunk. All of
these remain failures when they arrive after partial content has already been
received.

Invalid configuration and a missing credential remain hard errors with no
artifacts, because neither describes a request that was actually attempted.

#### Limitations

- **No retries.** Exactly one request is issued per invocation. A retry
  policy would have to represent every attempt as separate evidence, which
  is out of scope here.
- **Transport is not separable from generation.** Time to first content
  token includes DNS, TLS, queueing and network transit. The headers and
  first-body-byte offsets are recorded separately so the transport share is
  visible, but the remote decode time cannot be isolated from a client-side
  observation.
- **No provider hardware or memory is recorded.** It is not observable, so
  the record claims no accelerator and no memory rather than substituting a
  local value.
- **Model revision is usually unavailable.** Hosted APIs generally do not
  expose a build identifier. `--model-revision` stays unset in that case
  instead of guessing, and the config hash pins the request identity that
  *is* observable. The hash covers the exact raw endpoint query representation,
  so versions, field order, separators, invalid percent-encoded bytes and an
  empty `?` delimiter remain distinct. The raw query is hashed rather than
  recorded, so sensitive keys and values affect identity without being
  persisted.
- **`reasoning_tokens` is not documented for GLM.** It is captured if the
  provider sends it and stays `null` otherwise.
- **No pricing.** Cost per correct case belongs to a later versioned
  comparison layer. Baking mutable prices into a collector would make old
  evidence silently wrong.

### Native Qwen MTP: capability report and honest evidence collection

Native multi-token-prediction (MTP) is architecturally different from the
generic draft-model speculation above: the *same* model predicts several
tokens ahead using its own MTP heads, instead of a smaller external draft
model proposing tokens for the target to verify. Before adding an
`llmtracefx-optimizer native-mtp collect` run, verify this project can
actually trust what it would report:

```bash
uv run llmtracefx-optimizer native-mtp capability-report \
  --target-model-path /path/to/local/qwen-checkpoint
```

This inspects the checkpoint's `config.json` model family and checks it
against verified upstream facts (exit code `0` if a trustworthy native-MTP
path exists, `3` if not, `1` on error):

- **mlx-lm** (the runtime `collect-mlx` wraps) strips multi-token-prediction
  weights during model loading for every family that ships them
  (`qwen3_next`, `qwen3_5`, `qwen3_5_moe`, and others -- see
  `llmtracefx.optimizer.collectors.native_mtp` for the full, referenced
  list). There is no code path in mlx-lm that loads or invokes an MTP head.
- **mlx-vlm** has an experimental, git-main-only `mlx_vlm.speculative.drafters`
  module for a narrow set of families (e.g. `qwen4_exp`), dispatched through
  the same `draft_model` request path as generic speculative decoding. It is
  not a stable release, and this project cannot reliably distinguish its
  metrics from generic draft-model speculation from the generation response
  alone -- only checkpoint provenance does, and that is a best-effort,
  metadata-only check.

Given that, `native-mtp collect` never fabricates native-MTP evidence. It
validates the target/sidecar checkpoints are locally present and
architecturally compatible (`hidden_size`/`vocab_size` match, failing
clearly otherwise), runs the same capability check, and:

```bash
uv run llmtracefx-optimizer native-mtp collect \
  --run-id native-mtp-smoke-1 \
  --target-model-path /path/to/local/qwen-target \
  --mtp-sidecar-path /path/to/local/qwen-mtp-sidecar \
  --model-id "Qwen/Qwen3.8-27B" \
  --prompt-file examples/optimizer/mlx-smoke-prompt.txt \
  --output-dir artifacts/native-mtp-smoke-1
```

writes an explicit, honest `record.json` (`outcome.success = false`,
`error.category = "NativeMTPUnsupported"`, `speculative.enabled = false`)
plus a standalone `capability_report.json`, rather than silently running
generic draft-model speculation and mislabeling it as MTP. The collector's
capable code path (`speculative.method = "native-mtp"`, recording only
whatever proposed/accepted/depth fields a runtime actually exposes) is fully
implemented and tested against a fake runtime so it is ready to wrap a
genuinely stable, metrics-differentiated API if one is published upstream --
no production adapter claims that today.

### Apple Instruments / Metal traces via `xctrace`

`llmtracefx-optimizer instruments` wraps Apple's Instruments CLI to record a
**Metal System Trace** around a local inference command and turn the
resulting `.trace` bundle into canonical `ExperimentRecord` evidence.

**Setup.** `xctrace` ships only with the full Xcode, never with Command Line
Tools alone. On a Command Line Tools machine `/usr/bin/xctrace` still exists
and is executable, so a plain "is it on PATH" check is misleading; the
capability probe therefore always invokes the tool:

```bash
# Install Xcode from the Mac App Store, then select it:
sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
sudo xcodebuild -runFirstLaunch      # run these yourself; the project
sudo xcodebuild -license accept      # never invokes sudo on your behalf

uv run llmtracefx-optimizer instruments capability
```

`capability` exits `0` when supported, `3` when a known cause blocks it, and
`1` on error. Each cause is reported separately with its own remediation
rather than collapsed into "unavailable": not macOS, not arm64, xctrace
absent from PATH, Command Line Tools only, license not accepted, first launch
incomplete, template unavailable, permission denied, and probe failed for an
unrecognized reason.

**Dry run first.** `plan` validates everything and executes nothing, printing
the exact argv and the exact artifact paths it would write:

```bash
uv run llmtracefx-optimizer instruments plan \
  --output-trace artifacts/metal/run.trace \
  --output-dir artifacts/metal \
  --time-limit 45s \
  -- ./your-inference-command --tokens 128
```

**Record and import.** `record` performs the capture and then exports and
parses it; `import` does the export and parse half against a `.trace` bundle
you already have:

```bash
uv run llmtracefx-optimizer instruments record \
  --output-trace artifacts/metal/run.trace \
  --output-dir artifacts/metal \
  --time-limit 45s \
  -- ./your-inference-command --tokens 128

uv run llmtracefx-optimizer instruments import \
  --trace artifacts/metal/run.trace --output-dir artifacts/metal
```

#### What is actually measured

Validated live on this hardware, and reproducible with the commands above:

| Fact | Value |
| --- | --- |
| Tool | `xctrace version 16.0 (17F113)` |
| Host | macOS 26.6.2 (25G83), Apple M5 Pro, arm64 |
| Template | `Metal System Trace` |
| Table schemas advertised by one capture | 82 |
| Table this project parses | `metal-gpu-intervals` |

The four metrics emitted, all with provenance `measured_native` and all
scoped to a single pid:

- `metal_gpu_interval_count` -- GPU intervals attributed to the profiled
  process.
- `metal_gpu_interval_duration_sum` (ms) -- the sum of those intervals'
  durations.
- `metal_gpu_interval_wall_span` (ms) -- last interval end minus first
  interval start for that process.
- `metal_gpu_interval_count_all_processes` -- the trace-wide interval count,
  so the attributed share is visible rather than implied.

Correctness of the parser was cross-checked against a workload that issues a
known number of GPU dispatches: 400, 250, 120 and 77 dispatches produced
exactly 400, 250, 120 and 77 attributed intervals.

#### What is explicitly not claimed

This wrapper reports **no** GPU utilization, GPU busy percentage, kernel
time, memory bandwidth, occupancy, GPU power, GPU energy or GPU memory
footprint. Metal System Trace advertises tables whose names gesture at some
of these, but deriving those numbers needs modelling assumptions this project
has not validated against ground truth, so they stay absent rather than
approximated. `ExperimentRecord.validate` enforces this structurally, with an
allowlist rather than a denylist: only the four metric names above may be
persisted, each with exactly the unit and provenance it declares, each
requiring the table it is derived from to appear in `parsed_schemas`, and
`parsed_schemas` itself must be a subset of the schemas the trace actually
advertised. A fabricated hardware number therefore cannot be persisted even by
a caller that tries.

Two further limits worth stating plainly:

- `metal_gpu_interval_duration_sum` is **not** GPU busy time and **not** a
  utilization numerator. Metal runs Vertex, Fragment and Compute channels
  concurrently, so overlapping intervals are counted more than once and the
  sum can exceed wall-clock time.
- Metal System Trace records GPU work for **every** process on the system.
  A capture of a trivial local Metal program also contained intervals
  belonging to `WindowServer` and `com.apple.WebKit.GPU`, and in one capture
  81% of all intervals were `WindowServer`'s. Metrics are therefore always
  attributed per pid, read from the `<process>` cell's structured `pid` child
  rather than scraped from its display label. Without a target pid, and when
  one pid appears under two different labels (pid reuse, or a process that
  renamed itself), no scalar metric is emitted at all. Nothing here is a
  benchmark result or a comparison between runtimes.

The other 81 schemas found in a capture are listed in
`instruments.unsupported_schemas` so the gap is explicit. Instruments
evidence is kept separate from `memory` and `power`, which carry
runtime-allocator and host-side values (the MLX allocator's active, cache and
peak bytes), so a bookkeeping figure can never be mistaken for a profiler
measurement.

#### Safety and privacy

- **Traces are never overwritten.** The output path is resolved (symlinks and
  `..` collapsed, and case-insensitively on a default macOS filesystem) and
  claimed with a single atomic `O_CREAT | O_EXCL` reservation that is held for
  the whole run, so two concurrent recordings cannot both pass a check and
  then race into the same path. `--append-run` is never passed.
- **No shell.** Every invocation is an argv list, so no template name, path
  or inference argument can be reinterpreted as shell syntax. Schema names
  are validated against a conservative character set before entering an XPath
  so they cannot break out of the query.
- **Bounded and cleaned up.** Each recording has a host deadline strictly
  greater than `--time-limit`, leaving room for xctrace to finalize the
  bundle. The child runs in its own process group, whose id is captured at
  spawn so it stays reachable after xctrace itself exits, and teardown
  escalates SIGINT then SIGTERM then SIGKILL until the group is actually
  empty. The leader exiting is not treated as cleanup: `xctrace record
  --launch` starts the profiled program itself, and that program can outlive
  xctrace or ignore SIGINT.
- **Failures are preserved,** not cleaned away: stdout, stderr, run metadata
  and any partial bundle stay on disk. The whole request is validated before
  the first byte is written, so an invalid time limit or output path leaves
  nothing behind, and a refused rerun leaves an earlier run's artifacts byte
  for byte intact rather than mixing the two. The CLI says plainly that
  nothing was written instead of naming an evidence file it did not produce.
  Imports stage their exports and promote them together, and a run that fails
  before promoting clears the exported files an earlier run left, so a
  directory never holds one run's metadata beside another run's GPU table.
- **A run that measured nothing does not exit 0.** `instruments record`
  returns a nonzero status when the export failed, and when the profiled
  program survived every stop signal, while still recording truthfully in the
  metadata that the recording itself completed.
- **No prompt or completion capture.** `--target-stdin` and `--target-stdout`
  are never constructed, so the profiled program's own input and output never
  flow through xctrace into captured logs. `--all-processes` is never used,
  and attaching is offered by numeric pid only, since attaching by name
  resolves ambiguously.
- **Credential redaction.** Secrets are redacted from every argv this project
  stores or prints, in all three shapes they arrive in: `NAME=value`,
  `--api-key=value`, and `--api-key value` (where the following argument is
  replaced even if it starts with a dash). Name matching is case and separator
  insensitive, so `--API_KEY` and `--api-key` are the same. It deliberately
  distinguishes credentials from quantities and settings: `--hf-token` and
  `--basic-auth` are redacted, while `--max-tokens`, `--token-count` and
  `--auth-mode` are not, because destroying ordinary parameters would defeat
  the reproducibility the recorded argv exists to provide. The real value
  still reaches the process.

  **Treat this as a safety net, not a guarantee.** It classifies option names,
  so it can only catch spellings it recognizes, and it cannot see a secret
  embedded in a positional argument. Two limits are pinned by tests: a value
  attached to a single-dash short option (`-ksecret`) is not detected, and a
  name ending in a configuration or location suffix (`--auth-mode`,
  `--private-key-path`) is treated as naming a setting or a file. The reliable
  protection is not to put secrets on the command line at all: pass them
  through the profiled program's environment, or have it read them from a
  file.
- **Identity is not ingested, and not copied.** A trace's table of contents
  contains the device's display name (routinely a person's name), its hardware
  UUID, and the target's full argument list. None are read, and the copy of
  the table of contents written into your output directory has all three
  stripped, since the raw file would otherwise carry them even though the
  parser ignored them. Only the launched process's own pid and name are kept,
  because attribution is impossible without them, and records store the trace
  bundle's basename rather than an absolute path.
- **The `.trace` bundle itself is sensitive.** It is raw evidence, so nothing
  in it is sanitized: it contains the device name and UUID, the profiled
  command's arguments, and the names of every other process that used the GPU
  while it was recording. Treat a bundle like a memory dump. Do not attach one
  to a public issue. The same applies to `trace_table.xml`, the raw exported
  GPU table, which lists every process that produced a GPU interval. The
  derived artifacts do not: `instruments_evidence.json` and `trace_toc.json`
  carry only your own process's pid and counts, never another process's name.
- **One run per artifact directory.** The trace path and the output directory
  are both claimed with atomic `O_CREAT | O_EXCL` reservations held for the
  whole run, so two concurrent runs cannot interleave metadata or exports even
  when their trace names differ.
- **Malformed input is refused.** Exports declaring a `DOCTYPE` or `ENTITY`
  are rejected before parsing (entity expansion denial of service), oversized
  exports are refused with a suggestion to shorten `--time-limit`, and a row
  whose value count or engineering types disagree with the declared schema is
  an error rather than being mapped onto the wrong columns.

The test suite covers all of the above without Xcode installed, by injecting
the subprocess and process-launch boundaries and parsing small synthetic
export fixtures.

### Deterministic code/JSON/reasoning workload matrix

`llmtracefx.optimizer.workloads` pins a small, versioned, redistributable
catalog of three workload categories -- code completion, constrained
structured JSON, and prose/reasoning -- and a deterministic evaluator for
each (exact unit-test pass/fail, exact required-field/type match, and exact
expected-answer pattern match; no LLM-as-judge). Generate the full matrix for
a target model without loading or downloading anything:

```bash
uv run llmtracefx-optimizer workloads generate-matrix \
  --model-id "Qwen/Qwen3.8-27B" \
  --model-family qwen3_next \
  --output-dir artifacts/qwen3.8-matrix
```

This materializes each workload's prompt toward the 2K/8K/16K context tiers
using a documented, deterministic filler-padding scheme (the base task prompt
is always preserved verbatim and never truncated), hashes the fully
materialized prompt, and writes:

- `manifest.json`: every (workload, context tier, decode mode) combination,
  each with its prompt hash, whether it is `runnable`, and -- for MTP rows
  the capability report marks unsupported -- an explicit
  `unsupported_reason` instead of a silently-omitted row.
- `prompts/<workload>-<tier>.txt`: the exact materialized prompt text.
- `configs/<run_id>.json`: a ready-to-use config for the PR #3 runner
  (`llmtracefx-optimizer run --config ...`), wrapping the exact
  `collect-mlx`/`native-mtp collect` invocation for that row.

Evaluate a captured response against one workload's deterministic checker:

```bash
uv run llmtracefx-optimizer workloads evaluate \
  --workload-id structured-json-profile-extraction \
  --response-file /tmp/response.txt
```

### Executing the matrix: `workloads run` and `workloads summarize`

`workloads generate-matrix` only plans; it never loads a model or downloads
anything. `workloads run` consumes that manifest and actually executes the
runnable rows through the same `collect-mlx` collector used above, evaluates
each response with the deterministic evaluator, and writes a canonical
result per row:

```bash
uv run llmtracefx-optimizer workloads run \
  --matrix artifacts/qwen3.8-matrix/manifest.json \
  --model-path /path/to/local/qwen3.8-checkpoint \
  --output-dir artifacts/qwen3.8-run \
  --mode autoregressive \
  --context-tier 2k
```

- `--model-path` (and the optional `--draft-model-path` for generic
  draft-model speculation) must already exist on disk; nothing is ever
  downloaded.
- `--run-id`/`--category`/`--context-tier`/`--mode` filter which manifest
  rows are selected; omit all of them to select every row.
- `native-mtp` rows are always rejected as `unsupported` and never
  executed -- they are never silently downgraded to generic draft-model
  speculation, matching the capability report above.
- `--dry-run` prints every selected row's required local paths, expected
  artifacts, and blockers (missing model path, a prompt file whose hash no
  longer matches the manifest, workload catalog drift) without loading a
  model.
- Re-running with the same `--output-dir` resumes: a row is only skipped
  if its prior `verification.json` is `completed`/`skipped` *and* its
  prompt hash, run-binding hash (model paths, seed, `max_tokens`,
  `num_draft_tokens`), and workload version all still match; any mismatch
  reruns the row instead of trusting stale data.

Each row writes `runs/<run_id>/collection/{record.json,response.txt,
environment.json}` (the raw collector output), `final_record.json` (the
canonical `ExperimentRecord`, with the evaluator's outcome layered onto a
successful collection -- a runtime failure is always persisted as-is and
never overwritten by an evaluator result), and `verification.json` (a
machine-readable summary: status, hashes, quality, timing, and artifact
paths). Aggregate a completed run:

```bash
uv run llmtracefx-optimizer workloads summarize --results artifacts/qwen3.8-run
```

which reports pass rate and "correct cases per minute" (computed only from
rows that both passed and have measured timing) overall and broken down by
decode mode, context tier, backend and provider -- deliberately not blended
into one combined score.

The `backend` and `provider` breakdowns exist so that figures which are not
the same quantity stay apart. A row measured on a local checkpoint times a
model on your machine; a row measured through a hosted API times a request
to somebody else's, on hardware you cannot see. Locally executed rows have
no provider and are left out of `by_provider` entirely rather than being
gathered under a placeholder key, so those groups do not sum to `overall`.
A metric that is undefined for a group is reported as `null`, never `0`: no
evaluated rows means there is no pass rate, which is a different statement
from a pass rate of zero.

### Executing the matrix against an API: `workloads run-api`

`workloads run-api` is the remote counterpart to `workloads run`. It takes
the same matrix manifest, the same selection filters, the same
deterministic evaluators and the same canonical `ExperimentRecord`, but
executes each selected row through the provider-neutral streaming
`collect-api` collector instead of a local MLX checkpoint. Transport, SSE
decoding, failure classification and credential redaction are that
collector's, used unmodified.

> All commands in this section are **unmeasured examples**. They show the
> shape of an invocation; no number in this repository was produced by
> running them, and running one measures your endpoint on that day, not a
> published result.

Providers are named *profiles*, not hardcoded behaviour. A profile only
supplies defaults for `--provider`, `--endpoint` and `--api-key-env`:

```bash
uv run llmtracefx-optimizer workloads list-api-profiles
```

```json
{
  "profiles": [
    {
      "name": "openrouter",
      "provider_label": "openrouter",
      "endpoint": "https://openrouter.ai/api/v1/chat/completions",
      "credential_env_var": "OPENROUTER_API_KEY",
      "documented_model_ids": ["z-ai/glm-5.3", "z-ai/glm-5.3-flash"]
    },
    {
      "name": "z.ai",
      "provider_label": "z.ai",
      "endpoint": "https://api.z.ai/api/paas/v4/chat/completions",
      "credential_env_var": "ZAI_API_KEY",
      "documented_model_ids": ["glm-5.3", "glm-5.3-flash"]
    }
  ]
}
```

Validate a selection and see exactly what would be sent, with no network
request and no secret in the output (**unmeasured example**):

```bash
uv run llmtracefx-optimizer workloads run-api \
  --matrix artifacts/qwen3.8-matrix/manifest.json \
  --output-dir artifacts/glm-api-run \
  --profile openrouter \
  --model-id z-ai/glm-5.3 \
  --mode autoregressive \
  --context-tier 2k \
  --dry-run
```

The plan goes to stdout as JSON (so it can be piped into `jq`) and is also
written to `<output-dir>/api_request_plan.json`; the human-readable row
count goes to stderr. It carries message *digests* rather than prompt text,
endpoint query *keys* rather than values, and header *names* rather than
header values, and the whole document is passed through the collector's
redactor before it is printed or written.

Then execute against OpenRouter (**unmeasured example**):

```bash
export OPENROUTER_API_KEY=...   # read by name, never persisted or echoed

uv run llmtracefx-optimizer workloads run-api \
  --matrix artifacts/qwen3.8-matrix/manifest.json \
  --output-dir artifacts/glm-api-run \
  --profile openrouter \
  --model-id z-ai/glm-5.3-flash \
  --mode autoregressive \
  --reasoning-effort high \
  --thinking enabled
```

Or directly against Z.ai, whose model IDs carry no vendor prefix
(**unmeasured example**):

```bash
export ZAI_API_KEY=...

uv run llmtracefx-optimizer workloads run-api \
  --matrix artifacts/qwen3.8-matrix/manifest.json \
  --output-dir artifacts/glm-direct-run \
  --profile z.ai \
  --model-id glm-5.3 \
  --mode autoregressive
```

An unlisted provider is a first-class citizen, not a special case: drop
`--profile` and give the three fields it would have filled in
(**unmeasured example**):

```bash
uv run llmtracefx-optimizer workloads run-api \
  --matrix artifacts/qwen3.8-matrix/manifest.json \
  --output-dir artifacts/self-hosted-run \
  --provider self-hosted \
  --endpoint https://vllm.internal.example/v1/chat/completions \
  --api-key-env SELF_HOSTED_API_KEY \
  --model-id local-glm \
  --mode autoregressive
```

What the command guarantees:

- **The matrix row decides the request.** The prompt is read from the
  entry's `prompt_path` and its sha256 is checked against the manifest
  before anything is sent, and the row's `max_tokens` becomes the
  request's `max_tokens`. A mismatch fails the row instead of quietly
  measuring a different prompt.
- **`native-mtp` rows stay `unsupported`.** Native multi-token prediction
  is a decoding mechanism inside a local runtime. A hosted API exposes no
  such control, and its reasoning or "thinking" settings are a different
  mechanism measuring something else, so those rows are rejected rather
  than re-labelled. Passing `--reasoning-effort` does not change this.
- **`code_completion` rows stay `unsupported` too, and this one is a
  security boundary.** That category is graded by writing the model's
  answer to disk and running it with this interpreter. Locally that is a
  considered trade, because the answer came from a checkpoint on your own
  machine. Over an API it is not: the answer comes from a remote endpoint,
  and the evaluator has no network namespace, no filesystem confinement
  and no seccomp policy, only a minimal environment and POSIX resource
  limits. Executing it would hand the provider local code execution. The
  row is refused before a request is sent and the evaluator is never
  invoked. There is deliberately no opt-in flag; the honest time to add
  one is when a real sandbox exists to put behind it. Run this category
  locally with `workloads run`.
- **Only the final answer is graded.** The evaluator sees the assembled
  content stream, never `reasoning_content`, so a model that reasons its
  way to the answer and then states something else is graded on what it
  stated.
- **A provider failure is never overwritten by an evaluator verdict.** A
  non-200, a timeout, a decode error, a provider error frame or a stream
  that ends without a clean termination fails the row and the evaluator is
  not run at all -- including when the failed response body happens to
  contain a passing answer. A stream counts as cleanly terminated when it
  sent either the `[DONE]` sentinel or a terminal `finish_reason`, since
  not every OpenAI-compatible provider sends both; a documented failure
  reason or a frame left pending at end of stream is truncation either
  way. If collection succeeded but the evaluator itself raised, the row is
  `inconclusive`: the timing evidence is kept and `quality_score` is left
  unset rather than guessed.
- **`--max-stream-events`** bounds a chatty endpoint that stays under the
  request timeout while emitting far more events than any answer needs.
  Tripping the cap fails the row as truncated with the cap named in the
  reason; the timing and usage evidence is still persisted, and only the
  outcome refuses to claim success. The budget is denominated in
  *dispatched SSE events*, counting the terminal sentinel. Chunks that
  dispatch no event, such as keepalive comments or a stray blank line,
  are not charged, and the count is taken before each chunk is handed on,
  so two runs over identical bytes agree regardless of how the network
  split them into reads.
- **A credential is refused before it can be written down, not after.**
  Both `--dry-run` and a real run apply the collector's pre-flight check,
  which fails the row if the key from `--api-key-env` appears in the
  endpoint, provider label, model ID, prompt, output path or reconstructed
  command. This runs *before* the request plan is built, because a
  credential sitting in an endpoint query value would otherwise be folded
  into the config hash as its sha256 and no redactor can undo a hash.
- **No credential is ever hashed, persisted or echoed.** There is no
  `--api-key` flag; `--api-key-env` takes a *name*. The API binding hash
  deliberately excludes that name, both because two runs differing only in
  which variable held the key issue byte-identical requests and are graded
  identically, and because a caller who pastes a key into that slot must
  not have a derivation of it written into an artifact.

Each row writes `runs/<run_id>/collection/{record.json,response.txt,
api_evidence.json,environment.json,artifacts.json}` (the collector's own
artifact set, with `artifacts.json` as its completion marker),
`final_record.json` (the canonical `ExperimentRecord` carrying the
evaluator's outcome), `verification.json`, and `run.json` sealing all
three. Aggregate it with the same `workloads summarize` used for local
runs.

`summarize` withholds `correct_cases_per_minute` for any group that mixes
measurement contexts, reporting `timing_comparable: false` with a reason
and listing the contexts involved. A duration only means something beside
another measured the same way: a local row times a model on your machine,
an API row times a request to somebody else's over a network. Quality is
unaffected, since a pass is a pass wherever it was produced, so pass counts
and pass rate stay populated and only the timing figure is withheld. Read
the per-backend and per-provider groups for comparable throughput.

Re-running with the same `--output-dir` resumes, but only on evidence that
is provably whole. A row is skipped only if *every* one of these holds: the
prior `verification.json` is `completed`/`skipped`, it was produced by this
same backend, and its prompt hash, workload version and API binding hash
all still match, **and** the collector's `artifacts.json` marker verifies
the sha256 of every file in the collection directory, **and** the run-level
`run.json` marker verifies the collection marker together with
`final_record.json` and `verification.json`.

That second marker exists because the collector's own covers only the four
files it writes. The record carrying the graded outcome and the summary
resume actually reads both sat outside every integrity check, so either
could be edited and would still be trusted. `run.json` is removed before a
row is rewritten and written last, so an interrupted run leaves a directory
that is rejected rather than one that reads as trustworthy. A stale
binding, an interrupted write, a missing marker, or a file edited *without*
regenerating the marker all rerun the row. `--no-resume` reruns everything
regardless.

This is tamper evidence, not tamper proofing, and the distinction is worth
stating plainly: `run.json` is unsigned and derives from nothing outside
the run directory, so anyone who can edit `final_record.json` can also
rewrite the marker over it and resume will trust the result. What the
marker buys is that the failures which actually happen -- a crash between
files, a half-written directory, a hand edit nobody thought to re-seal --
stop resume rather than sail through it.

The API binding hash covers the sanitized endpoint identity (origin, path,
query keys and hashed query values), the provider label, the model ID and
revision, the request parameters including this row's `max_tokens`, the
provider extensions and reasoning settings, the finish-reason vocabulary,
the request timeout, the system prompt's hash, the event cap, and the
workload version the answer is graded against. Changing any of them reruns
the row rather than trusting evidence gathered under different conditions.

#### Repeating a run

The matrix deliberately has no repetition axis: `run_id` is derived from
`(workload, context tier, decode mode)`, so one manifest row is one
invocation. Rather than invent synthetic repetition IDs that no other part
of the pipeline understands, repeat by giving each repetition its own
results directory (**unmeasured example**):

```bash
for repetition in 1 2 3; do
  uv run llmtracefx-optimizer workloads run-api \
    --matrix artifacts/qwen3.8-matrix/manifest.json \
    --output-dir "artifacts/glm-api-run/rep-${repetition}" \
    --profile openrouter \
    --model-id z-ai/glm-5.3 \
    --mode autoregressive
done
```

Each repetition is then summarized on its own with `workloads summarize`.
Keeping them in separate directories is what makes repeated sampling
honest here: resume is keyed on the run directory, so reusing one directory
would skip the second repetition rather than measure it, and the spread
across repetitions stays visible instead of being averaged away by a
pipeline that never saw it as a spread.

This command deliberately computes no cost, no price and no cross-provider
ranking. The provider's own usage counters are persisted exactly as
reported; turning them into money or into a comparison is a separate
concern with its own correctness burden.

### Offline tuning and the `tune-report` HTML viewer

Once a `workloads run` results directory has been collected, `tune`
recommends the best verified configuration for one explicit objective
under an explicit set of constraints -- it never blends multiple objectives
and never loads a model or executes anything itself:

```bash
uv run llmtracefx-optimizer tune \
  --results artifacts/qwen3.8-run \
  --policy examples/optimizer/tune-policy-fastest-under-20gb-m5-pro.json \
  --output artifacts/qwen3.8-tune-report.json
```

`tune-report` then renders that JSON report as a single, self-contained,
portable HTML file (inline CSS, no JavaScript, no CDN) so the recommendation,
the full accepted/rejected candidate breakdown, any speculative-vs-baseline
comparison, and every excluded run can be inspected offline in a browser --
no Streamlit dashboard, no re-scoring, no new tuning logic:

```bash
uv run llmtracefx-optimizer tune-report \
  --input artifacts/qwen3.8-tune-report.json \
  --output artifacts/qwen3.8-tune-report.html
```

Local artifact paths (results directories, `verification.json`/
`final_record.json` locations) are redacted to stable
`runs/<run_id>/<file>`-style labels by default, so the HTML file is safe to
share without leaking a machine's home-directory layout; pass
`--include-paths` to include the full paths instead. See
`examples/optimizer/tune-report-example.json` for a synthetic (non-benchmark)
example report that exercises every section of the viewer.

The `optimize` command composes workload execution, verification, tuning, and
optional HTML rendering without changing any phase's underlying logic:

```bash
uv run llmtracefx-optimizer optimize \
  --matrix artifacts/qwen3.8-matrix/manifest.json \
  --model-path /existing/local/mlx/model \
  --results artifacts/qwen3.8-run \
  --policy examples/optimizer/tune-policy-fastest-under-20gb-m5-pro.json \
  --report-json artifacts/qwen3.8-tune-report.json \
  --report-html artifacts/qwen3.8-tune-report.html \
  --mode autoregressive \
  --context-tier 2k
```

It writes `optimize_summary.json` under `--results` by default. This atomic,
machine-readable summary records every phase, row count, recommendation, and
the final exit status. `--dry-run` writes the same summary with planning counts
and marks execution, verification, tuning, and rendering as `not_run`; it does
not load MLX or write tune reports. Existing unrelated runs under `--results`
are excluded from tuning, while repeatable `--extra-results` paths explicitly
opt additional evidence into the comparison.

**Not yet included** (tracked as a follow-up PR): a genuinely capable
native-MTP runtime adapter (none exists upstream today), native Metal/CUDA
performance-counter ingestion, CUDA/vLLM/SGLang collectors, and any actual
Qwen3.8-27B benchmark results -- everything in this section was verified
against fake runtimes and small local checkpoints, never a real Qwen3.8-27B
run. The fixtures under `tests/optimizer/fixtures/llama_cpp/` are synthetic,
hand-written logs for testing the parser and doctor rule -- not benchmark
evidence (see the `PROVENANCE.md` in that directory).

## 📄 License

This project is licensed under the GNU General Public License v3.0 License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Modal.com Documentation](https://modal.com/docs)
- [Claude API Documentation](https://docs.anthropic.com/claude/reference)
