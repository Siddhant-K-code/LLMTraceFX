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
curl -X POST "https://siddhant-k-code--llmtracefx-web-app.modal.run/upload-trace" \
     -F "file=@your_trace.json" -F "gpu_type=A10G" -F "enable_claude=true"
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
- **GPU bottleneck detection** (stall %, launch delays, memory issues)
- **AI explanations** using Claude API for performance insights
- **Interactive visualizations** with flame graphs and dashboards
- **Modal.com deployment** with GPU acceleration
- **Multiple input formats** (vLLM, generic trace logs)

## 📦 Installation

### Using uv (Recommended)

```bash
git clone https://github.com/Siddhant-K-code/LLMTraceFX.git
cd LLMTraceFX

# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync

# Or install in development mode with optional dependencies
uv sync --extra dev --extra test
```

### Using pip

```bash
git clone https://github.com/Siddhant-K-code/LLMTraceFX.git
cd LLMTraceFX
pip install -r llmtracefx/requirements.txt

# Or install as editable package
pip install -e .
```

## 🔧 Quick Start

### 1. CLI Usage

```bash
# With uv
uv run llmtracefx --trace sample
uv run llmtracefx --trace your_trace.json --gpu-type A10G
uv run llmtracefx --trace sample --no-claude

# Or activate virtual environment first
uv sync
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
llmtracefx --trace sample

# With pip/python
python -m llmtracefx.main --trace sample
python -m llmtracefx.main --trace your_trace.json --gpu-type A10G
python -m llmtracefx.main --trace sample --no-claude
```

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
export DEFAULT_GPU_TYPE="A10G"  # or H100, A100
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
🔧 Analyzing GPU performance (GPU: A10G)
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
curl -X POST "https://siddhant-k-code--llmtracefx-web-app.modal.run/upload-trace" \
     -F "file=@your_trace.json" \
     -F "gpu_type=A10G" \
     -F "enable_claude=true"
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
- **SM Occupancy**: Streaming multiprocessor utilization
- **Cache Hit Rate**: Memory access efficiency
- **Compute Utilization**: GPU computational usage

### Supported GPUs
- **A10G**: 24GB VRAM, 600 GB/s bandwidth
- **H100**: 80GB VRAM, 3350 GB/s bandwidth
- **A100**: 80GB VRAM, 1935 GB/s bandwidth

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
python -m llmtracefx.main --trace test_traces/sample_vllm_trace.json
```

## 📄 License

This project is licensed under the GNU General Public License v3.0 License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Modal.com Documentation](https://modal.com/docs)
- [Claude API Documentation](https://docs.anthropic.com/claude/reference)
