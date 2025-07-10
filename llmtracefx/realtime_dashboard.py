"""
Real-time Interactive Dashboard for LLMTraceFX
GPU-level LLM inference profiler with interactive controls
Connected to live Modal API for real GPU analysis
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import time
import json
import requests
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('llmtracefx_dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="LLMTraceFX Real-Time Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
MODAL_API_BASE = "https://siddhant-k-code--llmtracefx-web-app.modal.run"

class LLMTraceFXClient:
    """Client for LLMTraceFX Modal API"""
    
    def __init__(self, api_base: str = MODAL_API_BASE):
        self.api_base = api_base
        self.time_series = []
        self.analysis_cache = {}
        self.analysis_ids = []  # Store analysis IDs for detailed exploration
        logger.info(f"Initialized LLMTraceFX client with API base: {api_base}")
        
    def create_trace_data(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Create trace data based on parameters for analysis"""
        logger.debug(f"Creating trace data with params: {params}")
        
        # Generate realistic operations based on parameters
        operations = []
        
        # Base timings affected by parameters
        base_timings = {
            'embedding': 2.0,
            'rmsnorm': 1.5,
            'matmul': 15.0,
            'softmax': 3.0,
            'kvload': 2.5
        }
        
        # Apply parameter effects
        stall_factor = params['memory_stall'] / 100.0
        occupancy_factor = params['sm_occupancy'] / 100.0
        cache_factor = params['cache_hit_rate'] / 100.0
        
        current_time = 0.0
        for op_name, base_time in base_timings.items():
            # Calculate affected duration
            if op_name == 'matmul':
                # MatMul most affected by memory stall and occupancy
                duration = base_time * (1 + stall_factor * 0.8) * (2 - occupancy_factor)
            elif op_name in ['embedding', 'kvload']:
                # Memory operations affected by cache hit rate
                duration = base_time * (1.5 - cache_factor * 0.4)
            else:
                # Other operations less affected
                duration = base_time * (1 + stall_factor * 0.3)
            
            operations.append({
                'name': op_name,
                'start_time': current_time,
                'duration': max(0.1, duration)  # Ensure positive duration
            })
            current_time += duration
        
        # Create trace data structure
        trace_data = {
            'tokens': [
                {
                    'id': 0,
                    'text': f'Token_{len(self.time_series)}',
                    'operations': operations
                }
            ]
        }
        
        return trace_data
    
    def analyze_trace(self, trace_data: Dict[str, Any], gpu_type: str = 'A10G', 
                     enable_claude: bool = False) -> Optional[Dict[str, Any]]:
        """Analyze trace data using Modal API"""
        start_time = time.time()
        logger.info(f"Starting trace analysis - GPU: {gpu_type}, Claude: {enable_claude}")
        
        try:
            payload = {
                'trace_data': trace_data,
                'gpu_type': gpu_type,
                'enable_claude': enable_claude
            }
            
            logger.debug(f"Sending request to {self.api_base}/analyze-trace")
            response = requests.post(
                f"{self.api_base}/analyze-trace",
                json=payload,
                timeout=30
            )
            
            elapsed_time = time.time() - start_time
            logger.info(f"API request completed in {elapsed_time:.2f}s - Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Analysis successful - Analysis ID: {result.get('analysis_id', 'N/A')}")
                return result
            else:
                error_msg = f"API Error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                st.error(error_msg)
                return None
                
        except requests.exceptions.Timeout:
            error_msg = "Request timeout - API took too long to respond"
            logger.error(error_msg)
            st.error(error_msg)
            return None
        except requests.exceptions.RequestException as e:
            error_msg = f"Connection error: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            return None
    
    def get_real_metrics(self, params: Dict[str, float], gpu_type: str = 'A10G', enable_claude: bool = False) -> Dict[str, Any]:
        """Get real metrics from LLMTraceFX API"""
        logger.info(f"Getting real metrics - Token #{len(self.time_series) + 1}")
        
        # Create trace data based on parameters
        trace_data = self.create_trace_data(params)
        
        # Analyze using real API
        analysis = self.analyze_trace(trace_data, gpu_type, enable_claude=enable_claude)
        
        if analysis:
            # Extract metrics from analysis
            current_time = datetime.now()
            
            # Get token analysis
            token_analysis = analysis.get('token_analyses', [{}])[0]
            
            metrics = {
                'timestamp': current_time,
                'token_id': len(self.time_series) + 1,
                'latency': analysis.get('total_latency', 0.0),
                'performance_score': analysis.get('avg_performance_score', 0.0),
                'stall_percentage': params['memory_stall'],
                'sm_occupancy': params['sm_occupancy'],
                'cache_hit_rate': params['cache_hit_rate'],
                'compute_utilization': params['compute_utilization'],
                'memory_efficiency': 100 - params['memory_stall'],
                'launch_efficiency': params['launch_efficiency'],
                'analysis_id': analysis.get('analysis_id', ''),
                'total_tokens': analysis.get('total_tokens', 1),
                'operations': trace_data['tokens'][0]['operations']
            }
            
            logger.info(f"Real metrics obtained - Latency: {metrics['latency']:.2f}ms, Score: {metrics['performance_score']:.1f}")
            
            # Store analysis ID for detailed exploration
            analysis_id = metrics.get('analysis_id', '')
            if analysis_id and analysis_id not in self.analysis_ids:
                self.analysis_ids.append(analysis_id)
                # Keep only last 20 analysis IDs
                if len(self.analysis_ids) > 20:
                    self.analysis_ids = self.analysis_ids[-20:]
            
            # Store for time series
            self.time_series.append(metrics)
            
            # Keep only last 50 points
            if len(self.time_series) > 50:
                self.time_series = self.time_series[-50:]
            
            return metrics
        
        # Fallback to simulated data if API fails
        logger.warning("API analysis failed, falling back to simulated metrics")
        return self.fallback_metrics(params)
    
    def upload_trace_file(self, file_content: bytes, filename: str, gpu_type: str = 'A10G', 
                         enable_claude: bool = False) -> Optional[Dict[str, Any]]:
        """Upload trace file using /upload-trace endpoint"""
        logger.info(f"Uploading trace file: {filename}")
        
        try:
            files = {'file': (filename, file_content, 'application/json')}
            data = {
                'gpu_type': gpu_type,
                'enable_claude': str(enable_claude).lower()
            }
            
            response = requests.post(
                f"{self.api_base}/upload-trace",
                files=files,
                data=data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"File upload successful - Analysis ID: {result.get('analysis_id', 'N/A')}")
                return result
            else:
                error_msg = f"Upload Error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                st.error(error_msg)
                return None
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Upload error: {str(e)}"
            logger.error(error_msg)
            st.error(error_msg)
            return None
    
    def get_analysis_summary(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis summary using /analysis/{analysis_id} endpoint"""
        logger.info(f"Getting analysis summary for ID: {analysis_id}")
        
        try:
            response = requests.get(
                f"{self.api_base}/analysis/{analysis_id}",
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Analysis summary retrieved successfully")
                return result
            else:
                error_msg = f"Analysis Error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Analysis summary error: {str(e)}")
            return None
    
    def get_token_details(self, analysis_id: str, token_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed token analysis using /token/{analysis_id}/{token_id} endpoint"""
        logger.info(f"Getting token details for Analysis: {analysis_id}, Token: {token_id}")
        
        try:
            response = requests.get(
                f"{self.api_base}/token/{analysis_id}/{token_id}",
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Token details retrieved successfully")
                return result
            else:
                error_msg = f"Token Error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Token details error: {str(e)}")
            return None
    
    def get_claude_explanation(self, analysis_id: str, token_id: int) -> Optional[str]:
        """Get Claude AI explanation using /explain/{analysis_id}/{token_id} endpoint"""
        logger.info(f"Getting Claude explanation for Analysis: {analysis_id}, Token: {token_id}")
        
        try:
            response = requests.get(
                f"{self.api_base}/explain/{analysis_id}/{token_id}",
                timeout=60  # Claude can take longer
            )
            
            if response.status_code == 200:
                result = response.json()
                explanation = result.get('explanation', 'No explanation available')
                logger.info(f"Claude explanation retrieved successfully")
                return explanation
            else:
                error_msg = f"Explanation Error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Claude explanation error: {str(e)}")
            return None
    
    def get_html_dashboard(self, analysis_id: str) -> Optional[str]:
        """Get HTML dashboard using /dashboard/{analysis_id} endpoint"""
        logger.info(f"Getting HTML dashboard for Analysis: {analysis_id}")
        
        try:
            response = requests.get(
                f"{self.api_base}/dashboard/{analysis_id}",
                timeout=30
            )
            
            if response.status_code == 200:
                logger.info(f"HTML dashboard retrieved successfully")
                return response.text
            else:
                error_msg = f"Dashboard Error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"HTML dashboard error: {str(e)}")
            return None
    
    def export_analysis_data(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Export analysis data using /export/{analysis_id} endpoint"""
        logger.info(f"Exporting analysis data for ID: {analysis_id}")
        
        try:
            response = requests.get(
                f"{self.api_base}/export/{analysis_id}",
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Analysis data exported successfully")
                return result
            else:
                error_msg = f"Export Error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Export error: {str(e)}")
            return None
    
    def fallback_metrics(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Fallback metrics when API is unavailable"""
        current_time = datetime.now()
        
        # Simple calculation based on parameters
        base_latency = 25.0
        base_score = 70.0
        
        stall_factor = params['memory_stall'] / 100.0
        occupancy_factor = params['sm_occupancy'] / 100.0
        cache_factor = params['cache_hit_rate'] / 100.0
        
        latency = base_latency * (1 + stall_factor * 0.5) * (2 - occupancy_factor)
        performance_score = base_score * occupancy_factor * cache_factor * (1 - stall_factor * 0.3)
        
        # Ensure bounds
        latency = max(5.0, min(200.0, latency))
        performance_score = max(0.0, min(100.0, performance_score))
        
        return {
            'timestamp': current_time,
            'token_id': len(self.time_series) + 1,
            'latency': latency,
            'performance_score': performance_score,
            'stall_percentage': params['memory_stall'],
            'sm_occupancy': params['sm_occupancy'],
            'cache_hit_rate': params['cache_hit_rate'],
            'compute_utilization': params['compute_utilization'],
            'memory_efficiency': 100 - params['memory_stall'],
            'launch_efficiency': params['launch_efficiency'],
            'analysis_id': 'fallback',
            'total_tokens': 1,
            'operations': []
        }

def create_flame_graph(metrics_data: List[Dict]) -> go.Figure:
    """Create interactive flame graph showing token operations"""
    fig = go.Figure()
    
    colors = {
        'embedding': '#FF6B6B',
        'rmsnorm': '#4ECDC4', 
        'matmul': '#45B7D1',
        'softmax': '#96CEB4',
        'kvload': '#FFEAA7',
        'attention': '#DDA0DD',
        'layernorm': '#FFA15A'
    }
    
    for i, token_data in enumerate(metrics_data[-10:]):  # Last 10 tokens
        token_id = token_data['token_id']
        operations = token_data.get('operations', [])
        
        if not operations:
            # Fallback to basic display
            fig.add_trace(go.Bar(
                x=[token_data['latency']],
                y=[f'Token {token_id}'],
                orientation='h',
                name='Total',
                marker_color='#45B7D1',
                text=f'Total: {token_data["latency"]:.1f}ms',
                textposition='inside',
                hovertemplate=f'<b>Total</b><br>Duration: {token_data["latency"]:.1f}ms<br>Token: {token_id}<extra></extra>',
                showlegend=i == 0,
                offsetgroup=i
            ))
        else:
            # Use real operation data
            for op in operations:
                op_name = op['name']
                duration = op['duration']
                start_time = op['start_time']
                color = colors.get(op_name, '#999999')
                
                fig.add_trace(go.Bar(
                    x=[duration],
                    y=[f'Token {token_id}'],
                    orientation='h',
                    name=op_name,
                    marker_color=color,
                    text=f'{op_name}: {duration:.1f}ms',
                    textposition='inside',
                    hovertemplate=f'<b>{op_name}</b><br>Duration: {duration:.1f}ms<br>Start: {start_time:.1f}ms<br>Token: {token_id}<extra></extra>',
                    showlegend=i == 0,  # Only show legend for first token
                    base=start_time,
                    offsetgroup=i
                ))
    
    fig.update_layout(
        title='🔥 Real-Time Token Operations Timeline (Live Data)',
        xaxis_title='Time (ms)',
        yaxis_title='Tokens',
        barmode='overlay',
        height=400,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_metrics_radar(metrics: Dict[str, Any]) -> go.Figure:
    """Create radar chart for GPU metrics"""
    categories = ['SM Occupancy', 'Cache Hit Rate', 'Compute Utilization', 
                  'Memory Efficiency', 'Launch Efficiency']
    
    values = [
        metrics['sm_occupancy'],
        metrics['cache_hit_rate'],
        metrics['compute_utilization'],
        metrics['memory_efficiency'],
        metrics['launch_efficiency']
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Current Metrics',
        line_color='#4ECDC4',
        fillcolor='rgba(78, 205, 196, 0.3)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title='🎯 GPU Metrics Profile',
        height=400
    )
    
    return fig

def create_performance_trends(time_series: List[Dict]) -> go.Figure:
    """Create performance trend charts"""
    if not time_series:
        return go.Figure()
    
    df = pd.DataFrame(time_series)
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Latency Trend', 'Performance Score Trend'),
        vertical_spacing=0.1,
        shared_xaxes=True
    )
    
    # Latency trend
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['latency'],
            mode='lines+markers',
            name='Latency (ms)',
            line=dict(color='#FF6B6B', width=2),
            marker=dict(size=4)
        ),
        row=1, col=1
    )
    
    # Performance score trend
    fig.add_trace(
        go.Scatter(
            x=df['timestamp'],
            y=df['performance_score'],
            mode='lines+markers',
            name='Performance Score',
            line=dict(color='#4ECDC4', width=2),
            marker=dict(size=4)
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title='📈 Real-Time Performance Trends',
        height=500,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time", row=2, col=1)
    fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
    fig.update_yaxes(title_text="Score (0-100)", row=2, col=1)
    
    return fig

def create_bottleneck_distribution(metrics: Dict[str, Any]) -> go.Figure:
    """Create bottleneck distribution chart"""
    bottlenecks = []
    values = []
    
    # Determine bottlenecks based on metrics
    if metrics['stall_percentage'] > 30:
        bottlenecks.append('Memory Stall')
        values.append(metrics['stall_percentage'])
    
    if metrics['sm_occupancy'] < 70:
        bottlenecks.append('Low Occupancy')
        values.append(100 - metrics['sm_occupancy'])
    
    if metrics['cache_hit_rate'] < 80:
        bottlenecks.append('Cache Miss')
        values.append(100 - metrics['cache_hit_rate'])
    
    if metrics['compute_utilization'] < 75:
        bottlenecks.append('Compute Underutilization')
        values.append(100 - metrics['compute_utilization'])
    
    if not bottlenecks:
        bottlenecks = ['Optimal']
        values = [100]
    
    colors = ['#FF6B6B', '#FFA15A', '#FFEAA7', '#96CEB4', '#4ECDC4'][:len(bottlenecks)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=bottlenecks,
            y=values,
            marker_color=colors,
            text=[f'{v:.1f}%' for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title='📊 Current Bottleneck Analysis',
        xaxis_title='Bottleneck Type',
        yaxis_title='Severity (%)',
        height=400
    )
    
    return fig

def main():
    logger.info("Starting LLMTraceFX Dashboard")
    
    # Initialize session state
    if 'llm_client' not in st.session_state:
        st.session_state.llm_client = LLMTraceFXClient()
        logger.info("Initialized new LLMTraceFX client")
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False
    
    # Header
    st.title("🚀 LLMTraceFX Real-Time Dashboard")
    st.markdown("**GPU-level LLM inference profiler with interactive controls**")
    st.markdown("🔗 **Connected to:** https://siddhant-k-code--llmtracefx-web-app.modal.run")
    
    # API Status check
    try:
        logger.info("Checking API status...")
        response = requests.get(f"{MODAL_API_BASE}/", timeout=5)
        if response.status_code == 200:
            st.success("✅ API Status: Online")
            logger.info("API status: Online")
        else:
            st.warning("⚠️ API Status: Limited functionality")
            logger.warning(f"API status: Limited functionality (Status: {response.status_code})")
    except Exception as e:
        st.error("❌ API Status: Offline - Using fallback mode")
        logger.error(f"API status check failed: {str(e)}")
    
    # Sidebar controls
    st.sidebar.header("🎛️ GPU Parameters")
    
    # Parameter sliders first (to define variables)
    memory_stall = st.sidebar.slider(
        "Memory Stall %", 
        min_value=0, max_value=80, value=25, step=1,
        help="Percentage of time GPU is stalled waiting for memory"
    )
    
    sm_occupancy = st.sidebar.slider(
        "SM Occupancy %", 
        min_value=20, max_value=100, value=75, step=1,
        help="Streaming Multiprocessor occupancy percentage"
    )
    
    cache_hit_rate = st.sidebar.slider(
        "Cache Hit Rate %", 
        min_value=50, max_value=98, value=85, step=1,
        help="L1/L2 cache hit rate percentage"
    )
    
    compute_utilization = st.sidebar.slider(
        "Compute Utilization %", 
        min_value=30, max_value=100, value=80, step=1,
        help="GPU compute units utilization"
    )
    
    launch_efficiency = st.sidebar.slider(
        "Launch Efficiency %", 
        min_value=60, max_value=100, value=90, step=1,
        help="Kernel launch efficiency"
    )
    
    st.sidebar.markdown("---")
    
    # GPU Type selection
    gpu_type = st.sidebar.selectbox(
        "GPU Type",
        ["A10G", "H100", "A100"],
        index=0
    )
    
    # Claude AI toggle
    st.sidebar.markdown("### 🤖 AI Analysis")
    enable_claude = st.sidebar.checkbox(
        "Enable Claude AI Explanations",
        value=False,
        help="Get AI-powered insights (slower but more detailed)"
    )
    
    st.sidebar.markdown("---")
    
    # File Upload Section
    st.sidebar.markdown("### 📤 Upload Trace File")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a trace file",
        type=['json'],
        help="Upload a vLLM trace file for analysis"
    )
    
    if uploaded_file is not None:
        if st.sidebar.button("🚀 Analyze Uploaded File"):
            with st.spinner("📤 Uploading and analyzing trace file..."):
                file_content = uploaded_file.getvalue()
                result = st.session_state.llm_client.upload_trace_file(
                    file_content, uploaded_file.name, gpu_type, enable_claude
                )
                
                if result:
                    st.sidebar.success(f"✅ Analysis complete! ID: {result.get('analysis_id', 'N/A')}")
                    analysis_id = result.get('analysis_id')
                    if analysis_id:
                        st.session_state.llm_client.analysis_ids.append(analysis_id)
                else:
                    st.sidebar.error("❌ Upload failed")
    
    st.sidebar.markdown("---")
    

    
    # Refresh controls
    st.sidebar.markdown("### 🔄 Refresh Controls")
    auto_refresh = st.sidebar.checkbox("Auto Refresh", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh
    
    refresh_interval = st.sidebar.slider(
        "Refresh Interval (seconds)",
        min_value=1, max_value=10, value=2
    )
    
    manual_refresh = st.sidebar.button("🔄 Manual Refresh")
    
    # Generate metrics
    params = {
        'memory_stall': memory_stall,
        'sm_occupancy': sm_occupancy,
        'cache_hit_rate': cache_hit_rate,
        'compute_utilization': compute_utilization,
        'launch_efficiency': launch_efficiency
    }
    
    # Auto refresh logic
    if auto_refresh:
        logger.debug(f"Auto-refresh triggered, interval: {refresh_interval}s")
        time.sleep(refresh_interval)
        st.rerun()
    
    if manual_refresh or auto_refresh:
        logger.info(f"Manual/Auto refresh triggered - GPU: {gpu_type}, Claude: {enable_claude}")
        spinner_text = "🔄 Fetching real-time data from LLMTraceFX API..."
        if enable_claude:
            spinner_text += " (with Claude AI analysis)"
        with st.spinner(spinner_text):
            current_metrics = st.session_state.llm_client.get_real_metrics(params, gpu_type, enable_claude)
    else:
        # Use last metrics or generate initial ones
        if hasattr(st.session_state.llm_client, 'time_series') and st.session_state.llm_client.time_series:
            current_metrics = st.session_state.llm_client.time_series[-1]
            logger.debug("Using cached metrics from time series")
        else:
            logger.info("No cached metrics found, initializing with API call")
            with st.spinner("🔄 Initializing with LLMTraceFX API..."):
                current_metrics = st.session_state.llm_client.get_real_metrics(params, gpu_type, enable_claude)
    
    # Main dashboard
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Latency",
            f"{current_metrics['latency']:.1f}ms",
            delta=f"{current_metrics['latency'] - 25:.1f}ms"
        )
    
    with col2:
        st.metric(
            "Performance Score",
            f"{current_metrics['performance_score']:.1f}",
            delta=f"{current_metrics['performance_score'] - 70:.1f}"
        )
    
    with col3:
        st.metric(
            "Tokens Processed",
            f"{current_metrics['token_id']}",
            delta=f"+1"
        )
    
    with col4:
        st.metric(
            "GPU Type",
            gpu_type,
            delta=None
        )
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Flame graph
        flame_fig = create_flame_graph(st.session_state.llm_client.time_series)
        st.plotly_chart(flame_fig, use_container_width=True)
        
        # Bottleneck distribution
        bottleneck_fig = create_bottleneck_distribution(current_metrics)
        st.plotly_chart(bottleneck_fig, use_container_width=True)
    
    with col2:
        # Radar chart
        radar_fig = create_metrics_radar(current_metrics)
        st.plotly_chart(radar_fig, use_container_width=True)
        
        # Performance trends
        trends_fig = create_performance_trends(st.session_state.llm_client.time_series)
        st.plotly_chart(trends_fig, use_container_width=True)
    
    # Real-time data table
    st.markdown("### 📊 Real-Time Metrics")
    
    if st.session_state.llm_client.time_series:
        recent_data = st.session_state.llm_client.time_series[-10:]  # Last 10 entries
        df = pd.DataFrame([{
            'Token ID': d['token_id'],
            'Latency (ms)': f"{d['latency']:.1f}",
            'Performance Score': f"{d['performance_score']:.1f}",
            'Memory Stall %': f"{d['stall_percentage']:.1f}",
            'SM Occupancy %': f"{d['sm_occupancy']:.1f}",
            'Cache Hit Rate %': f"{d['cache_hit_rate']:.1f}",
            'Timestamp': d['timestamp'].strftime('%H:%M:%S')
        } for d in recent_data])
        
        st.dataframe(df, use_container_width=True)
    
    # AI Insights
    st.markdown("### 🤖 AI Performance Insights")
    
    insights = []
    
    if current_metrics['stall_percentage'] > 30:
        insights.append("🔴 **High Memory Stall**: Consider optimizing memory access patterns or increasing memory bandwidth")
    
    if current_metrics['sm_occupancy'] < 70:
        insights.append("🟡 **Low SM Occupancy**: Try increasing block size or improving parallelism")
    
    if current_metrics['cache_hit_rate'] < 80:
        insights.append("🟡 **Poor Cache Performance**: Optimize data locality and access patterns")
    
    if current_metrics['performance_score'] > 85:
        insights.append("🟢 **Excellent Performance**: Current configuration is well-optimized")
    
    if not insights:
        insights.append("🟢 **Good Performance**: No major bottlenecks detected")
    
    for insight in insights:
        st.markdown(insight)
    
    # Analysis Explorer Section
    if st.session_state.llm_client.analysis_ids:
        st.markdown("---")
        st.markdown("## 🔍 Analysis Explorer")
        
        # Analysis selection
        selected_analysis = st.selectbox(
            "Select Analysis to Explore",
            st.session_state.llm_client.analysis_ids,
            help="Choose from recent analyses for detailed exploration"
        )
        
        if selected_analysis:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Get Analysis Summary"):
                    with st.spinner("Fetching analysis summary..."):
                        summary = st.session_state.llm_client.get_analysis_summary(selected_analysis)
                        if summary:
                            st.json(summary)
            
            with col2:
                if st.button("📄 Get HTML Dashboard"):
                    with st.spinner("Fetching HTML dashboard..."):
                        html_dashboard = st.session_state.llm_client.get_html_dashboard(selected_analysis)
                        if html_dashboard:
                            # Offer multiple viewing options
                            view_option = st.radio(
                                "How would you like to view the dashboard?",
                                ["🪟 Open in New Tab", "🔗 Download HTML", "📊 Native View", "📱 Embedded View"],
                                horizontal=True
                            )
                            
                            if view_option == "📊 Native View":
                                # Create a native Streamlit version using the analysis data
                                st.markdown("#### 📊 Analysis Dashboard (Native Streamlit)")
                                
                                # Get analysis summary for native rendering
                                with st.spinner("Loading analysis data..."):
                                    summary = st.session_state.llm_client.get_analysis_summary(selected_analysis)
                                    
                                if summary:
                                    # Display key metrics
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Total Tokens", summary.get('total_tokens', 0))
                                    with col2:
                                        st.metric("Total Latency", f"{summary.get('total_latency', 0):.1f}ms")
                                    with col3:
                                        st.metric("Avg Latency/Token", f"{summary.get('avg_latency_per_token', 0):.1f}ms")
                                    with col4:
                                        st.metric("Avg Performance Score", f"{summary.get('avg_performance_score', 0):.1f}")
                                    
                                    # Create token breakdown chart
                                    if 'token_analyses' in summary:
                                        st.markdown("##### 🔥 Token Performance Breakdown")
                                        token_data = []
                                        for token in summary['token_analyses']:
                                            token_data.append({
                                                'Token ID': token.get('token_id', 0),
                                                'Text': token.get('token_text', ''),
                                                'Latency (ms)': token.get('total_latency', 0),
                                                'Performance Score': token.get('performance_score', 0),
                                                'Primary Bottleneck': token.get('primary_bottleneck', 'unknown')
                                            })
                                        
                                        df = pd.DataFrame(token_data)
                                        st.dataframe(df, use_container_width=True)
                                        
                                        # Create performance chart
                                        if len(token_data) > 0:
                                            fig = go.Figure()
                                            fig.add_trace(go.Bar(
                                                x=[f"Token {d['Token ID']}" for d in token_data],
                                                y=[d['Latency (ms)'] for d in token_data],
                                                name='Latency (ms)',
                                                marker_color='lightblue'
                                            ))
                                            fig.update_layout(
                                                title='Token Latency Analysis',
                                                xaxis_title='Tokens',
                                                yaxis_title='Latency (ms)',
                                                height=400
                                            )
                                            st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show any available recommendations
                                    if 'recommendations' in summary:
                                        st.markdown("##### 💡 Optimization Recommendations")
                                        for rec in summary['recommendations']:
                                            st.info(f"• {rec}")
                                    
                                else:
                                    st.error("Could not load analysis data for native view")
                                    
                            elif view_option == "📱 Embedded View":
                                # Improved embedded view with better styling
                                st.markdown("#### 📊 LLMTraceFX Dashboard (Embedded)")
                                st.warning("⚠️ **Note**: Embedded view may have styling limitations. For the best experience, use 'Download HTML' option.")
                                
                                # Height selector for better customization
                                embed_height = st.slider("Adjust height", 400, 1200, 800, step=50)
                                
                                # Clean up the HTML for better embedding
                                cleaned_html = f"""
                                <div style="
                                    width: 100%;
                                    height: {embed_height-20}px;
                                    border: 1px solid #ddd;
                                    border-radius: 8px;
                                    overflow: auto;
                                    background: white;
                                    padding: 10px;
                                    box-sizing: border-box;
                                ">
                                    {html_dashboard}
                                </div>
                                """
                                
                                st.components.v1.html(cleaned_html, height=embed_height, scrolling=True)
                                
                                # Add fullscreen option
                                if st.button("🔍 Try Fullscreen Mode"):
                                    st.components.v1.html(html_dashboard, height=1000, scrolling=True)
                                
                            elif view_option == "🔗 Download HTML":
                                # Download option
                                st.download_button(
                                    label="📥 Download HTML Dashboard",
                                    data=html_dashboard,
                                    file_name=f"dashboard_{selected_analysis}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                                    mime="text/html"
                                )
                                st.info("💡 Download the HTML file and open it in your browser for the best experience")
                                
                            elif view_option == "🪟 Open in New Tab":
                                # Best experience option - prominently displayed
                                st.success("🌟 **Recommended**: This provides the full interactive dashboard experience!")
                                
                                # Create a prominent download button
                                st.download_button(
                                    label="🚀 Download Full Dashboard",
                                    data=html_dashboard,
                                    file_name=f"llmtracefx_dashboard_{selected_analysis}.html",
                                    mime="text/html",
                                    help="Download and open in your browser for the complete interactive experience"
                                )
                                
                                # Clear instructions with better formatting
                                st.markdown("""
                                ### 📋 **How to View:**
                                1. **Click** the "🚀 Download Full Dashboard" button above
                                2. **Open** the downloaded HTML file in your browser
                                3. **Enjoy** the complete interactive LLMTraceFX dashboard!
                                
                                ### ✨ **What you'll get:**
                                - 🔥 **Flame graphs** with token operations
                                - 📊 **Interactive charts** and visualizations  
                                - 🎯 **GPU metrics** radar charts
                                - 📈 **Performance trends** over time
                                - 🌡️ **Operation heatmaps**
                                - 💡 **AI insights** and recommendations
                                """)
                                
                                # Show a preview
                                with st.expander("👁️ Quick Preview (Full experience in downloaded file)"):
                                    st.components.v1.html(html_dashboard, height=400, scrolling=True)
            
            with col3:
                if st.button("💾 Export Analysis Data"):
                    with st.spinner("Exporting analysis data..."):
                        export_data = st.session_state.llm_client.export_analysis_data(selected_analysis)
                        if export_data:
                            st.download_button(
                                label="📥 Download Export",
                                data=json.dumps(export_data, indent=2),
                                file_name=f"analysis_{selected_analysis}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
            
            # Token Explorer
            st.markdown("### 🔍 Token Explorer")
            
            # Get analysis summary first to know available tokens
            with st.spinner("Loading token information..."):
                summary = st.session_state.llm_client.get_analysis_summary(selected_analysis)
                
            if summary:
                total_tokens = summary.get('total_tokens', 0)
                if total_tokens > 0:
                    selected_token = st.number_input(
                        "Select Token ID",
                        min_value=0,
                        max_value=total_tokens-1,
                        value=0,
                        help="Choose token to analyze in detail"
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("🔬 Get Token Details"):
                            with st.spinner("Fetching token details..."):
                                token_details = st.session_state.llm_client.get_token_details(
                                    selected_analysis, selected_token
                                )
                                if token_details:
                                    st.markdown("#### Token Analysis")
                                    st.json(token_details)
                    
                    with col2:
                        if st.button("🤖 Get Claude Explanation"):
                            with st.spinner("Getting Claude AI explanation..."):
                                explanation = st.session_state.llm_client.get_claude_explanation(
                                    selected_analysis, selected_token
                                )
                                if explanation:
                                    st.markdown("#### Claude AI Analysis")
                                    st.markdown(explanation)
                else:
                    st.warning("No tokens found in this analysis")
            else:
                st.error("Could not load analysis summary")
    
    # Export functionality
    st.markdown("### 📥 Export Data")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Export Metrics JSON"):
            export_data = {
                'metrics': st.session_state.llm_client.time_series,
                'parameters': params,
                'gpu_type': gpu_type,
                'api_endpoint': MODAL_API_BASE,
                'export_time': datetime.now().isoformat()
            }
            
            st.download_button(
                label="Download JSON",
                data=json.dumps(export_data, indent=2, default=str),
                file_name=f"llmtracefx_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("📊 Export CSV"):
            if st.session_state.llm_client.time_series:
                df = pd.DataFrame(st.session_state.llm_client.time_series)
                csv = df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"llmtracefx_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    # Debug/Logging section
    with st.expander("🔍 Debug & Logging Info"):
        st.markdown("### 📊 Session Statistics")
        total_requests = len(st.session_state.llm_client.time_series)
        st.metric("Total API Requests", total_requests)
        
        if st.session_state.llm_client.time_series:
            last_request = st.session_state.llm_client.time_series[-1]['timestamp']
            st.metric("Last Request", last_request.strftime('%H:%M:%S'))
        
        st.markdown("### 📝 Recent Log Entries")
        try:
            with open('llmtracefx_dashboard.log', 'r') as f:
                log_lines = f.readlines()
                recent_logs = log_lines[-10:]  # Last 10 log entries
                for line in recent_logs:
                    if 'ERROR' in line:
                        st.error(line.strip())
                    elif 'WARNING' in line:
                        st.warning(line.strip())
                    elif 'INFO' in line:
                        st.info(line.strip())
                    else:
                        st.text(line.strip())
        except FileNotFoundError:
            st.info("Log file not found yet")
        
        st.markdown("### ⚙️ Current Configuration")
        config_data = {
            'API Endpoint': MODAL_API_BASE,
            'GPU Type': gpu_type,
            'Claude Enabled': enable_claude,
            'Auto Refresh': auto_refresh,
            'Refresh Interval': f"{refresh_interval}s",
            'Available Analysis IDs': len(st.session_state.llm_client.analysis_ids)
        }
        st.json(config_data)
        
        st.markdown("### 🔌 API Endpoints Used")
        endpoints = [
            "POST /analyze-trace - Real-time trace analysis",
            "POST /upload-trace - File upload analysis", 
            "GET /analysis/{id} - Analysis summary",
            "GET /token/{id}/{token} - Token details",
            "GET /explain/{id}/{token} - Claude explanations",
            "GET /dashboard/{id} - HTML dashboard",
            "GET /export/{id} - Export analysis data"
        ]
        for endpoint in endpoints:
            st.text(f"✅ {endpoint}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using **LLMTraceFX** | "
        "🚀 [GitHub](https://github.com/Siddhant-K-code/LLMTraceFX) | "
        "📊 Real-time GPU Performance Analysis | "
        f"📝 Logs: llmtracefx_dashboard.log"
    )

if __name__ == "__main__":
    main()
