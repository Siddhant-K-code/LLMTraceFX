"""
Modal deployment for LLMTraceFX
"""
import modal
from modal import App, Image, Secret, Volume, gpu

# Create the Modal app
app = App("llmtracefx")

# Define the container image
image = (
    Image.debian_slim(python_version="3.10")
    .pip_install([
        "fastapi==0.104.1",
        "uvicorn[standard]==0.24.0",
        "aiohttp==3.9.0",
        "plotly==5.17.0",
        "pandas==2.1.3",
        "numpy==1.25.2",
        "python-multipart==0.0.6"
    ])
    .copy_local_dir("./llmtracefx", "/app/llmtracefx")
    .workdir("/app")
)

# Create a volume for storing analysis results
volume = Volume.from_name("llmtracefx-data", create_if_missing=True)


@app.function(
    image=image,
    gpu=gpu.A10G(),
    timeout=600,
    secrets=[Secret.from_name("claude-api-key")],
    volumes={"/data": volume}
)
def analyze_trace_modal(trace_data: dict, gpu_type: str = "A10G", enable_claude: bool = True):
    """
    Modal function to analyze trace data
    """
    import os
    import json
    import asyncio
    
    # Import local modules
    from llmtracefx.profiler.trace_parser import TraceParser
    from llmtracefx.profiler.gpu_analyzer import GPUAnalyzer
    from llmtracefx.explainer.claude import ClaudeExplainer
    from llmtracefx.visualize.flame import FlameGraphGenerator
    
    # Set up Claude API key
    claude_api_key = os.environ.get("CLAUDE_API_KEY")
    
    try:
        # Parse trace data
        parser = TraceParser()
        tokens = parser.parse_trace_data(trace_data)
        
        # Analyze tokens
        analyzer = GPUAnalyzer(gpu_type)
        analyses = analyzer.analyze_sequence(tokens)
        
        # Generate Claude explanations if enabled
        explanations = {}
        if enable_claude and claude_api_key:
            explainer = ClaudeExplainer(claude_api_key)
            
            async def generate_explanations():
                tasks = []
                for analysis in analyses:
                    tasks.append(explainer.explain_token_performance(analysis))
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        explanations[analyses[i].token_id] = f"Error: {str(result)}"
                    else:
                        explanations[analyses[i].token_id] = explainer.format_explanation_for_display(result)
            
            asyncio.run(generate_explanations())
        
        # Generate visualizations
        visualizer = FlameGraphGenerator()
        dashboard_html = visualizer.generate_comprehensive_dashboard(analyses)
        export_json = visualizer.export_data_json(analyses)
        
        # Calculate summary statistics
        total_latency = sum(a.total_latency_ms for a in analyses)
        avg_performance = sum(a.performance_score for a in analyses) / len(analyses)
        
        bottleneck_summary = {}
        for analysis in analyses:
            bottleneck_summary[analysis.bottleneck_type] = bottleneck_summary.get(analysis.bottleneck_type, 0) + 1
        
        # Prepare response
        response = {
            "total_tokens": len(analyses),
            "total_latency_ms": total_latency,
            "avg_performance_score": avg_performance,
            "bottleneck_summary": bottleneck_summary,
            "dashboard_html": dashboard_html,
            "export_json": export_json,
            "explanations": explanations,
            "status": "completed"
        }
        
        return response
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }


@app.function(
    image=image,
    timeout=120,
    secrets=[Secret.from_name("claude-api-key")]
)
def explain_token_modal(token_analysis_data: dict):
    """
    Modal function to explain single token performance
    """
    import os
    import asyncio
    
    # Import local modules
    from llmtracefx.explainer.claude import ClaudeExplainer
    from llmtracefx.profiler.gpu_analyzer import TokenAnalysis, GPUMetrics
    from llmtracefx.profiler.trace_parser import Operation
    
    # Set up Claude API key
    claude_api_key = os.environ.get("CLAUDE_API_KEY")
    
    try:
        # Reconstruct TokenAnalysis object from data
        operations = []
        for op_data in token_analysis_data["operations"]:
            operations.append(Operation(
                name=op_data["name"],
                start_time=op_data["start_time"],
                duration=op_data["duration"],
                dependencies=op_data.get("dependencies", []),
                metadata=op_data.get("metadata", {})
            ))
        
        gpu_metrics = GPUMetrics(
            stall_pct=token_analysis_data["gpu_metrics"]["stall_pct"],
            launch_delay_ms=token_analysis_data["gpu_metrics"]["launch_delay_ms"],
            memory_latency_ms=token_analysis_data["gpu_metrics"]["memory_latency_ms"],
            sm_occupancy_pct=token_analysis_data["gpu_metrics"]["sm_occupancy_pct"],
            cache_hit_rate=token_analysis_data["gpu_metrics"]["cache_hit_rate"],
            memory_bandwidth_gb_s=token_analysis_data["gpu_metrics"]["memory_bandwidth_gb_s"],
            compute_utilization=token_analysis_data["gpu_metrics"]["compute_utilization"]
        )
        
        analysis = TokenAnalysis(
            token_id=token_analysis_data["token_id"],
            token_text=token_analysis_data["token_text"],
            total_latency_ms=token_analysis_data["total_latency_ms"],
            operations=operations,
            gpu_metrics=gpu_metrics,
            bottleneck_type=token_analysis_data["bottleneck_type"],
            optimization_flags=token_analysis_data["optimization_flags"],
            performance_score=token_analysis_data["performance_score"]
        )
        
        # Generate explanation
        if claude_api_key:
            explainer = ClaudeExplainer(claude_api_key)
            
            async def get_explanation():
                return await explainer.explain_token_performance(analysis)
            
            explanation = asyncio.run(get_explanation())
            formatted_explanation = explainer.format_explanation_for_display(explanation)
            
            return {
                "explanation": formatted_explanation,
                "status": "completed"
            }
        else:
            return {
                "error": "Claude API key not available",
                "status": "error"
            }
            
    except Exception as e:
        return {
            "error": str(e),
            "status": "error"
        }


@app.function(
    image=image,
    timeout=300,
    secrets=[Secret.from_name("claude-api-key")],
    volumes={"/data": volume},
    keep_warm=1
)
@modal.asgi_app()
def web_app():
    """
    Modal web endpoint for the FastAPI app
    """
    from llmtracefx.api.serve import app as fastapi_app
    return fastapi_app


@app.function(
    image=image,
    timeout=300,
    secrets=[Secret.from_name("claude-api-key")],
    volumes={"/data": volume}
)
def run_server():
    """
    Modal function to run the FastAPI server locally
    """
    import uvicorn
    from llmtracefx.api.serve import app as fastapi_app
    
    # Run the FastAPI server
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)


@app.function(
    image=image,
    timeout=60
)
def create_sample_trace():
    """
    Modal function to create sample trace data for testing
    """
    sample_trace = {
        "tokens": [
            {
                "id": 0,
                "text": "Hello",
                "operations": [
                    {"name": "embedding", "start_time": 0, "duration": 2.1},
                    {"name": "rmsnorm", "start_time": 2.1, "duration": 1.8},
                    {"name": "linear", "start_time": 3.9, "duration": 8.2},
                    {"name": "matmul", "start_time": 12.1, "duration": 15.3},
                    {"name": "softmax", "start_time": 27.4, "duration": 3.1}
                ]
            },
            {
                "id": 1,
                "text": "world",
                "operations": [
                    {"name": "embedding", "start_time": 30.5, "duration": 2.3},
                    {"name": "rmsnorm", "start_time": 32.8, "duration": 2.1},
                    {"name": "linear", "start_time": 34.9, "duration": 7.8},
                    {"name": "matmul", "start_time": 42.7, "duration": 18.1},
                    {"name": "kvload", "start_time": 60.8, "duration": 9.2},
                    {"name": "attention", "start_time": 70.0, "duration": 12.4},
                    {"name": "softmax", "start_time": 82.4, "duration": 2.9}
                ]
            }
        ]
    }
    
    return sample_trace


# CLI function for local testing
@app.local_entrypoint()
def main(trace_file: str = "test_traces/sample_vllm_trace.json", gpu_type: str = "A10G"):
    """
    Local entrypoint for testing Modal functions
    """
    import json
    
    # Load trace file
    try:
        with open(trace_file, 'r') as f:
            trace_data = json.load(f)
    except FileNotFoundError:
        print(f"Creating sample trace since {trace_file} not found...")
        trace_data = create_sample_trace.remote()
    
    # Analyze trace
    print("Analyzing trace with Modal...")
    result = analyze_trace_modal.remote(trace_data, gpu_type, enable_claude=True)
    
    if result["status"] == "completed":
        print(f"✅ Analysis completed!")
        print(f"Total tokens: {result['total_tokens']}")
        print(f"Total latency: {result['total_latency_ms']:.1f}ms")
        print(f"Avg performance score: {result['avg_performance_score']:.1f}")
        print(f"Bottlenecks: {result['bottleneck_summary']}")
        
        # Save dashboard
        with open("dashboard.html", "w") as f:
            f.write(result["dashboard_html"])
        print("💾 Dashboard saved to dashboard.html")
        
        # Save export
        with open("export.json", "w") as f:
            f.write(result["export_json"])
        print("💾 Export saved to export.json")
        
    else:
        print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()
