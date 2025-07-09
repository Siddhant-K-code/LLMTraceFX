"""
FastAPI server for LLMTraceFX web endpoints
"""
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
    version="1.0.0"
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
    return """
    <html>
    <head>
        <title>LLMTraceFX API</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .endpoint { background: #f0f0f0; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #007bff; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>🚀 LLMTraceFX API</h1>
        <p>GPU-level LLM inference profiler</p>
        
        <h2>Available Endpoints:</h2>
        
        <div class="endpoint">
            <span class="method">POST</span> /upload-trace
            <br>Upload trace file for analysis
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> /analyze-trace
            <br>Analyze trace data directly
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> /analysis/{analysis_id}
            <br>Get analysis summary
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> /token/{analysis_id}/{token_id}
            <br>Get detailed token analysis
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> /explain/{analysis_id}/{token_id}
            <br>Get Claude AI explanation for token
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> /dashboard/{analysis_id}
            <br>Get HTML dashboard
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> /export/{analysis_id}
            <br>Export analysis data as JSON
        </div>
    </body>
    </html>
    """


@app.post("/upload-trace", response_model=AnalysisResponse)
async def upload_trace(
    file: UploadFile = File(...),
    gpu_type: str = "A10G",
    enable_claude: bool = True,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Upload and analyze trace file"""
    try:
        # Read uploaded file
        content = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            tmp.write(content.decode('utf-8'))
            tmp_path = tmp.name
        
        try:
            # Parse trace file
            tokens = parser.parse_trace_file(tmp_path)
            
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
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing trace: {str(e)}")


@app.post("/analyze-trace", response_model=AnalysisResponse)
async def analyze_trace(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """Analyze trace data directly"""
    try:
        # Parse trace data
        tokens = parser.parse_trace_data(request.trace_data)
        
        # Analyze tokens
        analyzer_instance = GPUAnalyzer(request.gpu_type)
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
            "compute_utilization": token_analysis.gpu_metrics.compute_utilization
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
