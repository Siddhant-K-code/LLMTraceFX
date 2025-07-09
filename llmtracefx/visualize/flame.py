"""
Flame graph visualization for token-level GPU performance
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any, Optional
import colorsys
import json

from ..profiler.gpu_analyzer import TokenAnalysis
from ..profiler.trace_parser import TokenTrace


class FlameGraphGenerator:
    """Generate flame graphs and performance visualizations"""
    
    def __init__(self):
        self.color_map = {
            "rmsnorm": "#FF6B6B",
            "layernorm": "#FF8E8E", 
            "linear": "#4ECDC4",
            "matmul": "#45B7D1",
            "softmax": "#96CEB4",
            "kvload": "#FFEAA7",
            "kvstore": "#DDA0DD",
            "attention": "#98D8C8",
            "activation": "#F7DC6F",
            "embedding": "#BB8FCE"
        }
    
    def generate_token_flame_graph(self, analyses: List[TokenAnalysis]) -> str:
        """Generate flame graph showing token vs operations timeline"""
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=["Token-Level GPU Performance Timeline"],
            specs=[[{"secondary_y": False}]]
        )
        
        # Create timeline data
        timeline_data = []
        y_position = 0
        
        for analysis in analyses:
            current_time = 0
            
            for op in analysis.operations:
                color = self.color_map.get(op.name, "#95A5A6")
                
                # Add operation bar
                fig.add_trace(go.Bar(
                    x=[op.duration],
                    y=[f"Token {analysis.token_id}"],
                    name=op.name,
                    orientation='h',
                    marker_color=color,
                    text=f"{op.name}: {op.duration:.1f}ms",
                    textposition="inside",
                    hovertemplate=f"<b>{op.name}</b><br>" +
                                f"Duration: {op.duration:.1f}ms<br>" +
                                f"Token: {analysis.token_id}<br>" +
                                f"Performance Score: {analysis.performance_score:.1f}<extra></extra>",
                    showlegend=True if analysis.token_id == 0 else False
                ))
                
                current_time += op.duration
        
        # Update layout
        fig.update_layout(
            title="LLM Token Inference Performance Timeline",
            xaxis_title="Time (ms)",
            yaxis_title="Tokens",
            barmode='stack',
            height=max(400, len(analyses) * 40),
            showlegend=True,
            legend=dict(x=1.05, y=1),
            margin=dict(r=150)
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="flame-graph")
    
    def generate_bottleneck_distribution(self, analyses: List[TokenAnalysis]) -> str:
        """Generate bottleneck distribution chart"""
        bottleneck_counts = {}
        for analysis in analyses:
            bottleneck_counts[analysis.bottleneck_type] = bottleneck_counts.get(analysis.bottleneck_type, 0) + 1
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(bottleneck_counts.keys()),
                y=list(bottleneck_counts.values()),
                marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            )
        ])
        
        fig.update_layout(
            title="GPU Bottleneck Distribution",
            xaxis_title="Bottleneck Type",
            yaxis_title="Number of Tokens",
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="bottleneck-chart")
    
    def generate_performance_heatmap(self, analyses: List[TokenAnalysis]) -> str:
        """Generate performance heatmap"""
        # Create matrix data
        operations = set()
        for analysis in analyses:
            for op in analysis.operations:
                operations.add(op.name)
        
        operations = sorted(list(operations))
        
        # Create matrix
        matrix = []
        token_ids = []
        
        for analysis in analyses:
            row = []
            token_ids.append(f"Token {analysis.token_id}")
            
            for op_name in operations:
                # Find operation duration
                duration = 0
                for op in analysis.operations:
                    if op.name == op_name:
                        duration = op.duration
                        break
                row.append(duration)
            matrix.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=operations,
            y=token_ids,
            colorscale='Viridis',
            hovertemplate='<b>%{x}</b><br>%{y}<br>Duration: %{z:.1f}ms<extra></extra>'
        ))
        
        fig.update_layout(
            title="Operation Duration Heatmap",
            xaxis_title="GPU Operations",
            yaxis_title="Tokens",
            height=max(400, len(analyses) * 20)
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="heatmap")
    
    def generate_latency_trend(self, analyses: List[TokenAnalysis]) -> str:
        """Generate latency trend chart"""
        token_ids = [analysis.token_id for analysis in analyses]
        latencies = [analysis.total_latency_ms for analysis in analyses]
        performance_scores = [analysis.performance_score for analysis in analyses]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Token Latency Trend", "Performance Score Trend"],
            vertical_spacing=0.1
        )
        
        # Latency trend
        fig.add_trace(
            go.Scatter(
                x=token_ids, 
                y=latencies,
                mode='lines+markers',
                name='Latency (ms)',
                line=dict(color='#FF6B6B', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        # Performance score trend
        fig.add_trace(
            go.Scatter(
                x=token_ids, 
                y=performance_scores,
                mode='lines+markers',
                name='Performance Score',
                line=dict(color='#4ECDC4', width=2),
                marker=dict(size=6)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Token Performance Trends",
            height=600,
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Token ID", row=2, col=1)
        fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
        fig.update_yaxes(title_text="Score (0-100)", row=2, col=1)
        
        return fig.to_html(include_plotlyjs='cdn', div_id="trend-chart")
    
    def generate_gpu_metrics_radar(self, analysis: TokenAnalysis) -> str:
        """Generate radar chart for GPU metrics"""
        metrics = analysis.gpu_metrics
        
        categories = [
            'SM Occupancy %',
            'Cache Hit Rate %', 
            'Compute Utilization %',
            'Memory Efficiency %',
            'Launch Efficiency %'
        ]
        
        values = [
            metrics.sm_occupancy_pct,
            metrics.cache_hit_rate,
            metrics.compute_utilization,
            100 - (metrics.stall_pct),  # Memory efficiency
            100 - min(metrics.launch_delay_ms * 10, 100)  # Launch efficiency
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=f'Token {analysis.token_id}',
            line_color='#4ECDC4'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title=f"GPU Metrics Profile - Token {analysis.token_id}",
            height=500
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="radar-chart")
    
    def generate_operation_breakdown(self, analysis: TokenAnalysis) -> str:
        """Generate pie chart for operation breakdown"""
        op_names = [op.name for op in analysis.operations]
        op_durations = [op.duration for op in analysis.operations]
        
        colors = [self.color_map.get(name, "#95A5A6") for name in op_names]
        
        fig = go.Figure(data=[go.Pie(
            labels=op_names,
            values=op_durations,
            hole=0.3,
            marker_colors=colors,
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{percent}<br>%{value:.1f}ms'
        )])
        
        fig.update_layout(
            title=f"Operation Breakdown - Token {analysis.token_id}",
            height=400
        )
        
        return fig.to_html(include_plotlyjs='cdn', div_id="breakdown-chart")
    
    def generate_comprehensive_dashboard(self, analyses: List[TokenAnalysis]) -> str:
        """Generate comprehensive HTML dashboard"""
        if not analyses:
            return "<html><body><h1>No data to display</h1></body></html>"
        
        # Generate all charts
        flame_graph = self.generate_token_flame_graph(analyses)
        bottleneck_dist = self.generate_bottleneck_distribution(analyses)
        heatmap = self.generate_performance_heatmap(analyses)
        trend_chart = self.generate_latency_trend(analyses)
        
        # Sample detailed analysis for first token
        sample_radar = self.generate_gpu_metrics_radar(analyses[0])
        sample_breakdown = self.generate_operation_breakdown(analyses[0])
        
        # Generate summary stats
        total_latency = sum(a.total_latency_ms for a in analyses)
        avg_latency = total_latency / len(analyses)
        avg_performance = sum(a.performance_score for a in analyses) / len(analyses)
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>LLMTraceFX - GPU Performance Dashboard</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f8f9fa;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 30px;
                    padding: 20px;
                    background-color: #2c3e50;
                    color: white;
                    border-radius: 10px;
                }}
                .stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .stat-card {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .stat-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .stat-label {{
                    color: #7f8c8d;
                    margin-top: 10px;
                }}
                .chart-container {{
                    background: white;
                    margin: 20px 0;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                .chart-grid {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 LLMTraceFX - GPU Performance Dashboard</h1>
                <p>Comprehensive analysis of LLM token inference performance</p>
            </div>
            
            <div class="stats">
                <div class="stat-card">
                    <div class="stat-value">{len(analyses)}</div>
                    <div class="stat-label">Total Tokens</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_latency:.1f}ms</div>
                    <div class="stat-label">Total Latency</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_latency:.1f}ms</div>
                    <div class="stat-label">Avg Latency/Token</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{avg_performance:.1f}</div>
                    <div class="stat-label">Avg Performance Score</div>
                </div>
            </div>
            
            <div class="chart-container">
                <h2>🔥 Token Performance Timeline</h2>
                {flame_graph}
            </div>
            
            <div class="chart-grid">
                <div class="chart-container">
                    <h2>📊 Bottleneck Distribution</h2>
                    {bottleneck_dist}
                </div>
                <div class="chart-container">
                    <h2>📈 Performance Trends</h2>
                    {trend_chart}
                </div>
            </div>
            
            <div class="chart-container">
                <h2>🌡️ Operation Performance Heatmap</h2>
                {heatmap}
            </div>
            
            <div class="chart-grid">
                <div class="chart-container">
                    <h2>🎯 GPU Metrics Profile (Token 0)</h2>
                    {sample_radar}
                </div>
                <div class="chart-container">
                    <h2>🥧 Operation Breakdown (Token 0)</h2>
                    {sample_breakdown}
                </div>
            </div>
            
            <div class="chart-container">
                <h2>📋 Analysis Summary</h2>
                <p>Generated {len(analyses)} token analyses with comprehensive GPU performance metrics.</p>
                <p>Primary bottlenecks and optimization opportunities identified.</p>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def export_data_json(self, analyses: List[TokenAnalysis]) -> str:
        """Export analysis data as JSON"""
        data = []
        
        for analysis in analyses:
            ops_data = []
            for op in analysis.operations:
                ops_data.append({
                    "name": op.name,
                    "duration": op.duration,
                    "start_time": op.start_time
                })
            
            data.append({
                "token_id": analysis.token_id,
                "token_text": analysis.token_text,
                "total_latency_ms": analysis.total_latency_ms,
                "performance_score": analysis.performance_score,
                "bottleneck_type": analysis.bottleneck_type,
                "optimization_flags": analysis.optimization_flags,
                "operations": ops_data,
                "gpu_metrics": {
                    "stall_pct": analysis.gpu_metrics.stall_pct,
                    "launch_delay_ms": analysis.gpu_metrics.launch_delay_ms,
                    "memory_latency_ms": analysis.gpu_metrics.memory_latency_ms,
                    "sm_occupancy_pct": analysis.gpu_metrics.sm_occupancy_pct,
                    "cache_hit_rate": analysis.gpu_metrics.cache_hit_rate,
                    "compute_utilization": analysis.gpu_metrics.compute_utilization
                }
            })
        
        return json.dumps(data, indent=2)
