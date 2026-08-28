# 🚀 LLMTraceFX

**GPU-level LLM inference profiler** that analyzes token-level performance and provides AI-powered explanations.

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

**Not yet included** (tracked as follow-up work): native Metal/CUDA
performance-counter ingestion, native Qwen3.8 MTP collection, CUDA/vLLM/SGLang
collectors, automatic tuning/search, and any actual Qwen3.8-27B benchmark results.
The fixtures under `tests/optimizer/fixtures/llama_cpp/` are synthetic,
hand-written logs for testing the parser and doctor rule — not
benchmark evidence (see the `PROVENANCE.md` in that directory).

## 📄 License

This project is licensed under the GNU General Public License v3.0 License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Modal.com Documentation](https://modal.com/docs)
- [Claude API Documentation](https://docs.anthropic.com/claude/reference)
