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

Note that `ruff format` is not used anywhere in this project. It and `black`
disagree on a few files here and each undoes the other, so running both can
never pass. `black` is the formatter of record, and `make format` matches what
CI enforces. `ruff check` is still used for linting, which does not conflict.

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
decode mode and context tier -- deliberately not blended into one combined
score.

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
