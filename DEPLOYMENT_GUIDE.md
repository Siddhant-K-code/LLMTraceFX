# 🚀 LLMTraceFX Deployment Guide

Your LLMTraceFX application is now successfully deployed on Modal! Here's how to use it.

## 🌐 **Live Web Application**

**🔗 Your deployed web app is available at:**
```
https://siddhant-k-code--llmtracefx-web-app.modal.run
```

### **Web Interface Features:**
- **Interactive API Documentation**: Visit the root URL for API docs
- **Upload trace files**: Use `/upload-trace` endpoint
- **View dashboards**: Get HTML visualizations
- **Export data**: Download JSON reports
- **Real-time analysis**: With GPU acceleration

---

## 📡 **API Endpoints**

### **1. Upload & Analyze Trace File**
```bash
curl -X POST "https://siddhant-k-code--llmtracefx-web-app.modal.run/upload-trace" \
     -F "file=@your_trace.json" \
     -F "gpu_type=A10G" \
     -F "enable_claude=true"
```

### **2. Analyze Trace Data Directly**
```bash
curl -X POST "https://siddhant-k-code--llmtracefx-web-app.modal.run/analyze-trace" \
     -H "Content-Type: application/json" \
     -d '{
       "trace_data": {
         "tokens": [
           {
             "id": 0,
             "text": "Hello",
             "operations": [
               {"name": "embedding", "start_time": 0, "duration": 2.1},
               {"name": "matmul", "start_time": 2.1, "duration": 15.3}
             ]
           }
         ]
       },
       "gpu_type": "A10G",
       "enable_claude": true
     }'
```

### **3. Get Analysis Results**
```bash
# Get analysis summary
curl "https://siddhant-k-code--llmtracefx-web-app.modal.run/analysis/{analysis_id}"

# Get detailed token analysis
curl "https://siddhant-k-code--llmtracefx-web-app.modal.run/token/{analysis_id}/{token_id}"

# Get Claude AI explanation
curl "https://siddhant-k-code--llmtracefx-web-app.modal.run/explain/{analysis_id}/{token_id}"
```

### **4. Get Visualizations**
```bash
# Get interactive dashboard (HTML)
curl "https://siddhant-k-code--llmtracefx-web-app.modal.run/dashboard/{analysis_id}" -o dashboard.html

# Get flame graph (HTML)
curl "https://siddhant-k-code--llmtracefx-web-app.modal.run/flame/{analysis_id}" -o flame.html

# Export data (JSON)
curl "https://siddhant-k-code--llmtracefx-web-app.modal.run/export/{analysis_id}" -o analysis.json
```

---

## 🐍 **Python Integration**

### **Using requests library:**
```python
import requests
import json

# Upload and analyze trace
with open('your_trace.json', 'rb') as f:
    response = requests.post(
        'https://siddhant-k-code--llmtracefx-web-app.modal.run/upload-trace',
        files={'file': f},
        data={'gpu_type': 'A10G', 'enable_claude': 'true'}
    )

result = response.json()
analysis_id = result['analysis_id']

print(f"Analysis completed! ID: {analysis_id}")
print(f"Total tokens: {result['total_tokens']}")
print(f"Total latency: {result['total_latency_ms']:.1f}ms")
print(f"Avg performance: {result['avg_performance_score']:.1f}/100")

# Get dashboard
dashboard = requests.get(f'https://siddhant-k-code--llmtracefx-web-app.modal.run/dashboard/{analysis_id}')
with open('dashboard.html', 'w') as f:
    f.write(dashboard.text)

print("Dashboard saved to dashboard.html")
```

### **Direct trace data analysis:**
```python
import requests

trace_data = {
    "tokens": [
        {
            "id": 0,
            "text": "Hello world",
            "operations": [
                {"name": "embedding", "start_time": 0, "duration": 2.1},
                {"name": "rmsnorm", "start_time": 2.1, "duration": 1.8},
                {"name": "matmul", "start_time": 3.9, "duration": 15.3},
                {"name": "softmax", "start_time": 19.2, "duration": 3.1}
            ]
        }
    ]
}

response = requests.post(
    'https://siddhant-k-code--llmtracefx-web-app.modal.run/analyze-trace',
    json={
        "trace_data": trace_data,
        "gpu_type": "A10G",
        "enable_claude": True
    }
)

result = response.json()
print(f"Analysis ID: {result['analysis_id']}")
```

---

## 🔧 **Modal Function Calls**

You can also call individual Modal functions directly:

### **1. Analyze Trace with GPU**
```bash
uv run modal run llmtracefx/modal_app.py::analyze_trace_modal \
    --trace-data '{"tokens": [...]}' \
    --gpu-type A10G \
    --enable-claude true
```

### **2. Get Claude Explanation**
```bash
uv run modal run llmtracefx/modal_app.py::explain_token_modal \
    --token-analysis-data '{"token_id": 0, "operations": [...], "gpu_metrics": {...}}'
```

### **3. Create Sample Data**
```bash
uv run modal run llmtracefx/modal_app.py::create_sample_trace
```

---

## 📊 **Expected Outputs**

### **Analysis Response:**
```json
{
  "analysis_id": "analysis_0",
  "total_tokens": 5,
  "total_latency_ms": 120.5,
  "avg_performance_score": 67.3,
  "bottleneck_summary": {
    "memory_stall": 2,
    "launch_overhead": 1,
    "optimal": 2
  },
  "status": "completed"
}
```

### **Token Detail Response:**
```json
{
  "token_id": 0,
  "token_text": "Hello",
  "total_latency_ms": 30.7,
  "performance_score": 65.2,
  "bottleneck_type": "memory_stall",
  "optimization_flags": ["high_memory_stall", "kernel_fusion_candidate"],
  "operations": [
    {
      "name": "matmul",
      "duration": 15.3,
      "start_time": 3.9
    }
  ],
  "gpu_metrics": {
    "stall_pct": 35.2,
    "launch_delay_ms": 0.3,
    "sm_occupancy_pct": 72.1,
    "cache_hit_rate": 84.5
  },
  "claude_explanation": "The MatMul operation shows 35% memory stall..."
}
```

---

## 🎯 **Use Cases**

### **1. LLM Optimization Teams**
- Upload production inference traces
- Identify GPU bottlenecks automatically
- Get AI-powered optimization suggestions
- Generate reports for stakeholders

### **2. Research & Development**
- Compare different model architectures
- Analyze token-level performance patterns
- Optimize kernel fusion strategies
- Benchmark GPU utilization

### **3. Production Monitoring**
- Integrate with inference pipelines
- Real-time performance analysis
- Automated bottleneck detection
- Performance regression testing

### **4. Educational/Learning**
- Understand GPU performance concepts
- Learn about LLM inference optimization
- Visualize kernel execution patterns
- Study AI-generated explanations

---

## 🔧 **Advanced Configuration**

### **Custom GPU Types**
```python
# Analyze with different GPU types
for gpu_type in ["A10G", "H100", "A100"]:
    response = requests.post(
        'https://siddhant-k-code--llmtracefx-web-app.modal.run/analyze-trace',
        json={"trace_data": data, "gpu_type": gpu_type}
    )
```

### **Batch Processing**
```python
import concurrent.futures

def analyze_trace(trace_file):
    with open(trace_file, 'rb') as f:
        response = requests.post(url, files={'file': f})
    return response.json()

# Process multiple traces in parallel
trace_files = ['trace1.json', 'trace2.json', 'trace3.json']
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(analyze_trace, trace_files))
```

---

## 📈 **Monitoring & Management**

### **Check App Status**
Visit: https://modal.com/apps/siddhant-k-code/main/deployed/llmtracefx

### **View Logs**
```bash
uv run modal logs llmtracefx
```

### **App Management**
```bash
# Stop the app
uv run modal app stop llmtracefx

# Restart the app
uv run modal deploy llmtracefx/modal_app.py
```

---

## 🎉 **Next Steps**

1. **Try the web interface**: Visit your deployed URL
2. **Upload a real trace**: Use your vLLM trace files
3. **Integrate with your pipeline**: Use the API endpoints
4. **Optimize your models**: Follow Claude's suggestions
5. **Scale up**: Handle multiple traces concurrently

**Your LLMTraceFX deployment is ready for production use!** 🚀
