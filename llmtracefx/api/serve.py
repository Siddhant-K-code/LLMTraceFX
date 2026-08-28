"""
FastAPI server for LLMTraceFX web endpoints
"""
from ..brand import LOCKUP_SVG, TOKENS_CSS
import asyncio
import json
import tempfile
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..hardware import hardware_profiles, normalize_hardware_name
from ..profiler.trace_parser import TraceParser
from ..profiler.gpu_analyzer import GPUAnalyzer, TokenAnalysis
from ..explainer.claude import ClaudeExplainer
from ..visualize.flame import FlameGraphGenerator


class AnalysisRequest(BaseModel):
    """Request model for trace analysis"""
    trace_data: Dict[str, Any]
    gpu_type: str = "A10G"
    enable_claude: bool = True


class AnalysisResponse(BaseModel):
    """Response model for trace analysis"""
    analysis_id: str
    total_tokens: int
    total_latency_ms: float
    avg_performance_score: float
    bottleneck_summary: Dict[str, int]
    status: str


class TokenDetailResponse(BaseModel):
    """Response model for token detail"""
    token_id: int
    token_text: str
    total_latency_ms: float
    performance_score: float
    bottleneck_type: str
    optimization_flags: List[str]
    operations: List[Dict[str, Any]]
    gpu_metrics: Dict[str, Any]
    claude_explanation: Optional[str] = None


app = FastAPI(
    title="LLMTraceFX API",
    description="GPU-level LLM inference profiler",
    version="1.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for storing analyses
analyses_store: Dict[str, List[TokenAnalysis]] = {}
explanations_store: Dict[str, Dict[int, str]] = {}
gpu_results_store: Dict[str, Dict[str, Any]] = {}  # Store GPU results for dashboard/export

# Initialize components
parser = TraceParser()
analyzer = GPUAnalyzer()
visualizer = FlameGraphGenerator()

# Initialize Claude explainer if API key is available
try:
    explainer = ClaudeExplainer()
except ValueError:
    explainer = None


@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <meta name="color-scheme" content="light">
        <title>API index - LLMTraceFX</title>
        <style>
            {TOKENS_CSS}
            * {{ box-sizing: border-box; }}
            body {{
                font-family: var(--sans);
                margin: 0;
                padding: clamp(16px, 3vw, 40px) clamp(12px, 3vw, 36px) 64px;
                background-color: var(--field);
                background-image:
                    repeating-linear-gradient(to right, var(--graticule) 0 1px, transparent 1px 48px),
                    repeating-linear-gradient(to bottom, var(--graticule) 0 1px, transparent 1px 48px);
                color: var(--ink);
                font-size: 15px;
                line-height: 1.55;
            }}
            .sheet {{
                max-width: 760px;
                margin: 0 auto;
                background: var(--sheet);
                border: 1px solid var(--rule);
                box-shadow: 0 1px 1px #16181a0f, 0 26px 52px -30px #16181a5c;
                padding: clamp(20px, 4vw, 48px);
            }}
            .masthead {{
                display: flex;
                justify-content: space-between;
                align-items: flex-start;
                gap: 16px 32px;
                flex-wrap: wrap;
                border-bottom: 1px solid var(--ink);
                padding-bottom: 14px;
                margin-bottom: 32px;
            }}
            .lockup {{ display: block; height: 19px; width: auto; color: var(--ink); }}
            .stamp {{
                margin: 0;
                font-family: var(--mono);
                font-size: 10.5px;
                letter-spacing: 0.07em;
                text-transform: uppercase;
                color: var(--muted);
            }}
            h1 {{
                font-size: clamp(1.5rem, 1.1rem + 1.6vw, 2.1rem);
                font-weight: 600;
                letter-spacing: -0.022em;
                line-height: 1.15;
                margin: 0 0 10px;
            }}
            .lede {{ margin: 0 0 36px; max-width: 62ch; color: var(--muted); }}
            h2 {{
                position: relative;
                font-size: 1.0625rem;
                font-weight: 600;
                letter-spacing: -0.01em;
                border-top: 1px solid var(--ink);
                padding-top: 13px;
                margin: 0 0 4px;
            }}
            h2::before {{
                content: "";
                position: absolute;
                top: 0; left: 0;
                width: 2px; height: 7px;
                background: var(--signal);
            }}
            /* Routes are a ruled index, not a stack of grey boxes: the path is
               the thing you scan for, so it gets the mono column. */
            .routes {{
                border-top: 1px solid var(--rule);
                margin-top: 12px;
            }}
            .endpoint {{
                display: grid;
                grid-template-columns: minmax(0, 4.25rem) minmax(0, 1fr);
                gap: 2px 18px;
                padding: 13px 0;
                border-bottom: 1px solid var(--rule-soft);
            }}
            .method {{
                font-family: var(--mono);
                font-size: 10.5px;
                letter-spacing: 0.09em;
                line-height: 1.7;
                color: var(--signal);
            }}
            .path {{
                font-family: var(--mono);
                font-size: 0.9375rem;
                overflow-wrap: anywhere;
            }}
            .about {{ grid-column: 2; color: var(--muted); font-size: 0.875rem; }}
            footer {{
                margin-top: 44px;
                border-top: 1px solid var(--ink);
                padding-top: 14px;
                font-family: var(--mono);
                font-size: 10.5px;
                letter-spacing: 0.07em;
                text-transform: uppercase;
                color: var(--muted);
            }}
            @media (max-width: 520px) {{
                .endpoint {{ grid-template-columns: minmax(0, 1fr); }}
                .about {{ grid-column: 1; }}
            }}
        </style>
    </head>
    <body>
    <div class="sheet">
        <header class="masthead">
            {LOCKUP_SVG}
            <p class="stamp">HTTP API index</p>
        </header>
        <h1>GPU level LLM inference profiler</h1>
        <p class="lede">This service accepts a trace, attributes its latency to
        GPU operations, and returns the analysis as JSON or as a rendered
        dashboard. The routes below are the whole surface.</p>

        <h2>Routes</h2>
        <div class="routes">
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/hardware</span>
                <span class="about">List supported CUDA and Metal hardware profiles.</span>
            </div>
            <div class="endpoint">
                <span class="method">POST</span>
                <span class="path">/upload-trace</span>
                <span class="about">Upload a trace file for analysis.</span>
            </div>
            <div class="endpoint">
                <span class="method">POST</span>
                <span class="path">/analyze-trace</span>
                <span class="about">Analyze trace data supplied directly in the request.</span>
            </div>
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/analysis/{{analysis_id}}</span>
                <span class="about">Get the analysis summary.</span>
            </div>
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/token/{{analysis_id}}/{{token_id}}</span>
                <span class="about">Get the detailed analysis for a single token.</span>
            </div>
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/explain/{{analysis_id}}/{{token_id}}</span>
                <span class="about">Get a Claude explanation for a token.</span>
            </div>
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/dashboard/{{analysis_id}}</span>
                <span class="about">Get the rendered HTML dashboard.</span>
            </div>
            <div class="endpoint">
                <span class="method">GET</span>
                <span class="path">/export/{{analysis_id}}</span>
                <span class="about">Export the analysis data as JSON.</span>
            </div>
        </div>

        <footer>llmtracefx serve</footer>
    </div>
    </body>
    </html>
    """


@app.get("/hardware")
async def list_hardware():
    """List supported hardware profiles."""
    return {"hardware": hardware_profiles()}


@app.post("/upload-trace", response_model=AnalysisResponse)
async def upload_trace(
    file: UploadFile = File(...),
    gpu_type: str = "A10G",
    enable_claude: bool = True,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload and analyze trace file"""
    try:
        gpu_type = _validate_hardware(gpu_type)
        # Read uploaded file
        content = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp.write(content.decode('utf-8'))
            tmp_path = tmp.name
        
        try:
            # Parse trace file
            tokens = parser.parse_trace_file(tmp_path)
            _require_tokens(tokens)
            
            # Analyze tokens
            analyzer_instance = GPUAnalyzer(gpu_type)
            analyses = analyzer_instance.analyze_sequence(tokens)
            
            # Generate analysis ID
            analysis_id = f"analysis_{len(analyses_store)}"
            
            # Store results
            analyses_store[analysis_id] = analyses
            
            # Schedule Claude explanations in background if enabled
            if enable_claude:
                background_tasks.add_task(generate_claude_explanations, analysis_id, analyses)
            
            # Calculate summary stats
            total_latency = sum(a.total_latency_ms for a in analyses)
            avg_performance = sum(a.performance_score for a in analyses) / len(analyses)
            
            bottleneck_summary = {}
            for analysis in analyses:
                bottleneck_summary[analysis.bottleneck_type] = bottleneck_summary.get(analysis.bottleneck_type, 0) + 1
            
            return AnalysisResponse(
                analysis_id=analysis_id,
                total_tokens=len(analyses),
                total_latency_ms=total_latency,
                avg_performance_score=avg_performance,
                bottleneck_summary=bottleneck_summary,
                status="completed" if not enable_claude else "processing_explanations"
            )
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing trace: {str(e)}")


@app.post("/analyze-trace", response_model=AnalysisResponse)
async def analyze_trace(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Analyze trace data directly - uses GPU when available"""
    try:
        gpu_type = _validate_hardware(request.gpu_type)
        # Check if we should use Modal GPU functions
        try:
            # Try to import Modal and get the GPU function
            import modal
            
            # Try to get Modal app and function
            try:
                from llmtracefx.modal_app import app as modal_app
                
                # Call the GPU function remotely
                gpu_result = modal_app.analyze_trace_modal.remote(
                    trace_data=request.trace_data,
                    gpu_type=gpu_type,
                    enable_claude=request.enable_claude
                )
                
                # Check if successful and has required fields
                if gpu_result and gpu_result.get("status") == "completed" and "total_tokens" in gpu_result:
                    # Generate analysis ID and store the results for later retrieval
                    analysis_id = f"analysis_{len(analyses_store)}"
                    
                    # Create dummy analyses for storage (we'll reconstruct from GPU result when needed)
                    # This is a simplified approach - in production you'd want to store full analysis objects
                    analyses_store[analysis_id] = []  # Placeholder
                    
                    # Store GPU results for dashboard/export
                    gpu_results_store[analysis_id] = gpu_result
                    
                    # Store explanations if available
                    if gpu_result.get("explanations"):
                        explanations_store[analysis_id] = gpu_result["explanations"]
                    
                    return AnalysisResponse(
                        analysis_id=analysis_id,
                        total_tokens=gpu_result["total_tokens"],
                        total_latency_ms=gpu_result["total_latency_ms"],
                        avg_performance_score=gpu_result["avg_performance_score"],
                        bottleneck_summary=gpu_result["bottleneck_summary"],
                        status=gpu_result["status"]
                    )
                elif gpu_result and gpu_result.get("status") == "error":
                    print(f"GPU analysis failed: {gpu_result.get('error')}, using CPU fallback")
                else:
                    print("GPU analysis returned unexpected result, using CPU fallback")
                    
            except Exception as gpu_error:
                # Log GPU error but continue with CPU fallback
                print(f"GPU analysis failed: {gpu_error}, using CPU fallback")
                    
        except ImportError:
            # Modal not available, use local processing
            print("Modal not available, using CPU processing")
        
        # CPU fallback processing (original code)
        # Parse trace data
        tokens = parser.parse_trace_data(request.trace_data)
        _require_tokens(tokens)
        
        # Analyze tokens
        analyzer_instance = GPUAnalyzer(gpu_type)
        analyses = analyzer_instance.analyze_sequence(tokens)
        
        # Generate analysis ID
        analysis_id = f"analysis_{len(analyses_store)}"
        
        # Store results
        analyses_store[analysis_id] = analyses
        
        # Schedule Claude explanations in background if enabled
        if request.enable_claude:
            background_tasks.add_task(generate_claude_explanations, analysis_id, analyses)
        
        # Calculate summary stats
        total_latency = sum(a.total_latency_ms for a in analyses)
        avg_performance = sum(a.performance_score for a in analyses) / len(analyses)
        
        bottleneck_summary = {}
        for analysis in analyses:
            bottleneck_summary[analysis.bottleneck_type] = bottleneck_summary.get(analysis.bottleneck_type, 0) + 1
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            total_tokens=len(analyses),
            total_latency_ms=total_latency,
            avg_performance_score=avg_performance,
            bottleneck_summary=bottleneck_summary,
            status="completed" if not request.enable_claude else "processing_explanations"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing trace: {str(e)}")


@app.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str):
    """Get analysis summary"""
    if analysis_id not in analyses_store:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analyses = analyses_store[analysis_id]
    
    # Calculate summary stats
    total_latency = sum(a.total_latency_ms for a in analyses)
    avg_performance = sum(a.performance_score for a in analyses) / len(analyses)
    
    bottleneck_summary = {}
    for analysis in analyses:
        bottleneck_summary[analysis.bottleneck_type] = bottleneck_summary.get(analysis.bottleneck_type, 0) + 1
    
    # Check if Claude explanations are available
    has_explanations = analysis_id in explanations_store
    
    return AnalysisResponse(
        analysis_id=analysis_id,
        total_tokens=len(analyses),
        total_latency_ms=total_latency,
        avg_performance_score=avg_performance,
        bottleneck_summary=bottleneck_summary,
        status="completed" if has_explanations else "processing_explanations"
    )


@app.get("/token/{analysis_id}/{token_id}", response_model=TokenDetailResponse)
async def get_token_detail(analysis_id: str, token_id: int):
    """Get detailed token analysis"""
    if analysis_id not in analyses_store:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analyses = analyses_store[analysis_id]
    
    # Find token analysis
    token_analysis = None
    for analysis in analyses:
        if analysis.token_id == token_id:
            token_analysis = analysis
            break
    
    if not token_analysis:
        raise HTTPException(status_code=404, detail="Token not found")
    
    # Get Claude explanation if available
    claude_explanation = None
    if analysis_id in explanations_store and token_id in explanations_store[analysis_id]:
        claude_explanation = explanations_store[analysis_id][token_id]
    
    # Format operations
    operations = []
    for op in token_analysis.operations:
        operations.append({
            "name": op.name,
            "duration": op.duration,
            "start_time": op.start_time,
            "dependencies": op.dependencies,
            "metadata": op.metadata
        })
    
    return TokenDetailResponse(
        token_id=token_analysis.token_id,
        token_text=token_analysis.token_text,
        total_latency_ms=token_analysis.total_latency_ms,
        performance_score=token_analysis.performance_score,
        bottleneck_type=token_analysis.bottleneck_type,
        optimization_flags=token_analysis.optimization_flags,
        operations=operations,
        gpu_metrics={
            "stall_pct": token_analysis.gpu_metrics.stall_pct,
            "launch_delay_ms": token_analysis.gpu_metrics.launch_delay_ms,
            "memory_latency_ms": token_analysis.gpu_metrics.memory_latency_ms,
            "sm_occupancy_pct": token_analysis.gpu_metrics.sm_occupancy_pct,
            "cache_hit_rate": token_analysis.gpu_metrics.cache_hit_rate,
            "memory_bandwidth_gb_s": token_analysis.gpu_metrics.memory_bandwidth_gb_s,
            "compute_utilization": token_analysis.gpu_metrics.compute_utilization,
            "occupancy_label": token_analysis.gpu_metrics.occupancy_label,
            "metrics_source": token_analysis.gpu_metrics.metrics_source
        },
        claude_explanation=claude_explanation
    )


@app.get("/explain/{analysis_id}/{token_id}")
async def get_token_explanation(analysis_id: str, token_id: int):
    """Get Claude AI explanation for token"""
    if analysis_id not in analyses_store:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    if analysis_id not in explanations_store:
        raise HTTPException(status_code=404, detail="Explanations not available yet")
    
    if token_id not in explanations_store[analysis_id]:
        raise HTTPException(status_code=404, detail="Token explanation not found")
    
    return {"explanation": explanations_store[analysis_id][token_id]}


@app.get("/dashboard/{analysis_id}", response_class=HTMLResponse)
async def get_dashboard(analysis_id: str):
    """Get HTML dashboard"""
    if analysis_id not in analyses_store:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Check if we have GPU-generated dashboard
    if analysis_id in gpu_results_store and "dashboard_html" in gpu_results_store[analysis_id]:
        return HTMLResponse(content=gpu_results_store[analysis_id]["dashboard_html"])
    
    # Fallback to local generation
    analyses = analyses_store[analysis_id]
    
    try:
        dashboard_html = visualizer.generate_comprehensive_dashboard(analyses)
        return HTMLResponse(content=dashboard_html)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating dashboard: {str(e)}")


@app.get("/export/{analysis_id}")
async def export_analysis(analysis_id: str):
    """Export analysis data as JSON"""
    if analysis_id not in analyses_store:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    # Check if we have GPU-generated export data
    if analysis_id in gpu_results_store and "export_json" in gpu_results_store[analysis_id]:
        return JSONResponse(content=json.loads(gpu_results_store[analysis_id]["export_json"]))
    
    # Fallback to local generation
    analyses = analyses_store[analysis_id]
    
    try:
        json_data = visualizer.export_data_json(analyses)
        return JSONResponse(content=json.loads(json_data))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting data: {str(e)}")


@app.get("/flame/{analysis_id}", response_class=HTMLResponse)
async def get_flame_graph(analysis_id: str):
    """Get flame graph visualization"""
    if analysis_id not in analyses_store:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analyses = analyses_store[analysis_id]
    
    try:
        flame_html = visualizer.generate_token_flame_graph(analyses)
        return HTMLResponse(content=flame_html)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating flame graph: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "analyses_count": len(analyses_store)}


def _validate_hardware(gpu_type: str) -> str:
    try:
        return normalize_hardware_name(gpu_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _require_tokens(tokens: list[Any]) -> None:
    if not tokens:
        raise HTTPException(
            status_code=400, detail="Trace does not contain any tokens"
        )


async def generate_claude_explanations(analysis_id: str, analyses: List[TokenAnalysis]):
    """Background task to generate Claude explanations"""
    try:
        explanations = {}
        
        # Generate explanations for each token
        for analysis in analyses:
            try:
                explanation = await explainer.explain_token_performance(analysis)
                explanations[analysis.token_id] = explainer.format_explanation_for_display(explanation)
            except Exception as e:
                explanations[analysis.token_id] = f"Error generating explanation: {str(e)}"
        
        # Store explanations
        explanations_store[analysis_id] = explanations
        
    except Exception as e:
        # Store error message for all tokens
        explanations_store[analysis_id] = {
            analysis.token_id: f"Error generating explanations: {str(e)}"
            for analysis in analyses
        }


def main():
    """Main function for CLI entry point"""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
