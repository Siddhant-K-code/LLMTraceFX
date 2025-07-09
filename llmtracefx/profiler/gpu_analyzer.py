"""
GPU performance analyzer for kernel timing and bottleneck detection
"""
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from .trace_parser import Operation, TokenTrace


@dataclass
class GPUMetrics:
    """GPU performance metrics for an operation"""
    stall_pct: float
    launch_delay_ms: float
    memory_latency_ms: float
    sm_occupancy_pct: float
    cache_hit_rate: float
    memory_bandwidth_gb_s: float
    compute_utilization: float


@dataclass
class TokenAnalysis:
    """Complete analysis for a token"""
    token_id: int
    token_text: str
    total_latency_ms: float
    operations: List[Operation]
    gpu_metrics: GPUMetrics
    bottleneck_type: str
    optimization_flags: List[str]
    performance_score: float  # 0-100, higher is better


class GPUAnalyzer:
    """Analyze GPU performance and identify bottlenecks"""
    
    def __init__(self, gpu_type: str = "A10G"):
        self.gpu_type = gpu_type
        self.gpu_specs = self._get_gpu_specs(gpu_type)
        
        # Performance models for different operations
        self.op_models = {
            "rmsnorm": {"base_time": 2.0, "stall_range": (10, 25), "memory_bound": True},
            "layernorm": {"base_time": 2.2, "stall_range": (10, 25), "memory_bound": True},
            "linear": {"base_time": 8.0, "stall_range": (15, 35), "memory_bound": False},
            "matmul": {"base_time": 12.0, "stall_range": (20, 45), "memory_bound": False},
            "softmax": {"base_time": 3.0, "stall_range": (5, 20), "memory_bound": True},
            "kvload": {"base_time": 9.0, "stall_range": (30, 60), "memory_bound": True},
            "kvstore": {"base_time": 7.0, "stall_range": (25, 50), "memory_bound": True},
            "attention": {"base_time": 15.0, "stall_range": (20, 40), "memory_bound": False},
            "activation": {"base_time": 1.5, "stall_range": (5, 15), "memory_bound": True},
            "embedding": {"base_time": 5.0, "stall_range": (15, 30), "memory_bound": True}
        }
    
    def _get_gpu_specs(self, gpu_type: str) -> Dict[str, Any]:
        """Get GPU specifications"""
        specs = {
            "A10G": {
                "memory_bandwidth_gb_s": 600,
                "compute_units": 80,
                "base_clock_mhz": 1695,
                "memory_size_gb": 24,
                "l2_cache_mb": 6
            },
            "H100": {
                "memory_bandwidth_gb_s": 3350,
                "compute_units": 132,
                "base_clock_mhz": 1980,
                "memory_size_gb": 80,
                "l2_cache_mb": 50
            },
            "A100": {
                "memory_bandwidth_gb_s": 1935,
                "compute_units": 108,
                "base_clock_mhz": 1410,
                "memory_size_gb": 80,
                "l2_cache_mb": 40
            }
        }
        return specs.get(gpu_type, specs["A10G"])
    
    def analyze_token(self, token_trace: TokenTrace) -> TokenAnalysis:
        """Analyze a single token's performance"""
        # Analyze each operation
        analyzed_ops = []
        total_stall_time = 0
        total_memory_time = 0
        
        for op in token_trace.operations:
            gpu_metrics = self._analyze_operation(op)
            analyzed_ops.append(op)
            
            total_stall_time += op.duration * (gpu_metrics.stall_pct / 100)
            total_memory_time += gpu_metrics.memory_latency_ms
        
        # Calculate aggregate metrics
        avg_stall_pct = (total_stall_time / token_trace.total_latency) * 100 if token_trace.total_latency > 0 else 0
        avg_launch_delay = sum(self._get_launch_delay(op) for op in analyzed_ops) / len(analyzed_ops)
        avg_sm_occupancy = sum(self._get_sm_occupancy(op) for op in analyzed_ops) / len(analyzed_ops)
        avg_cache_hit = sum(self._get_cache_hit_rate(op) for op in analyzed_ops) / len(analyzed_ops)
        
        aggregate_metrics = GPUMetrics(
            stall_pct=avg_stall_pct,
            launch_delay_ms=avg_launch_delay,
            memory_latency_ms=total_memory_time,
            sm_occupancy_pct=avg_sm_occupancy,
            cache_hit_rate=avg_cache_hit,
            memory_bandwidth_gb_s=self.gpu_specs["memory_bandwidth_gb_s"],
            compute_utilization=self._calculate_compute_utilization(analyzed_ops)
        )
        
        # Identify bottleneck
        bottleneck = self._identify_bottleneck(aggregate_metrics, analyzed_ops)
        
        # Generate optimization flags
        optimization_flags = self._generate_optimization_flags(aggregate_metrics, analyzed_ops)
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(aggregate_metrics, token_trace.total_latency)
        
        return TokenAnalysis(
            token_id=token_trace.token_id,
            token_text=token_trace.token_text,
            total_latency_ms=token_trace.total_latency,
            operations=analyzed_ops,
            gpu_metrics=aggregate_metrics,
            bottleneck_type=bottleneck,
            optimization_flags=optimization_flags,
            performance_score=performance_score
        )
    
    def _analyze_operation(self, op: Operation) -> GPUMetrics:
        """Analyze a single operation's GPU performance"""
        model = self.op_models.get(op.name, self.op_models["linear"])
        
        # Simulate stall percentage based on operation type
        stall_pct = random.uniform(*model["stall_range"])
        
        # Simulate realistic metrics
        launch_delay = self._get_launch_delay(op)
        memory_latency = self._get_memory_latency(op)
        sm_occupancy = self._get_sm_occupancy(op)
        cache_hit_rate = self._get_cache_hit_rate(op)
        compute_util = self._get_compute_utilization(op)
        
        return GPUMetrics(
            stall_pct=stall_pct,
            launch_delay_ms=launch_delay,
            memory_latency_ms=memory_latency,
            sm_occupancy_pct=sm_occupancy,
            cache_hit_rate=cache_hit_rate,
            memory_bandwidth_gb_s=self.gpu_specs["memory_bandwidth_gb_s"],
            compute_utilization=compute_util
        )
    
    def _get_launch_delay(self, op: Operation) -> float:
        """Simulate kernel launch delay"""
        base_delay = 0.1  # 0.1ms base launch overhead
        
        # Longer operations have slightly higher launch overhead
        size_factor = min(op.duration / 10.0, 2.0)
        return base_delay + (size_factor * random.uniform(0.05, 0.3))
    
    def _get_memory_latency(self, op: Operation) -> float:
        """Simulate memory access latency"""
        model = self.op_models.get(op.name, self.op_models["linear"])
        
        if model["memory_bound"]:
            # Memory-bound operations have higher memory latency
            return op.duration * random.uniform(0.3, 0.6)
        else:
            # Compute-bound operations have lower memory latency
            return op.duration * random.uniform(0.1, 0.3)
    
    def _get_sm_occupancy(self, op: Operation) -> float:
        """Simulate SM occupancy percentage"""
        model = self.op_models.get(op.name, self.op_models["linear"])
        
        if model["memory_bound"]:
            # Memory-bound ops typically have lower occupancy
            return random.uniform(40, 70)
        else:
            # Compute-bound ops can achieve higher occupancy
            return random.uniform(60, 90)
    
    def _get_cache_hit_rate(self, op: Operation) -> float:
        """Simulate cache hit rate"""
        # KV operations typically have good cache locality
        if "kv" in op.name:
            return random.uniform(80, 95)
        # Other operations vary more
        return random.uniform(60, 85)
    
    def _get_compute_utilization(self, op: Operation) -> float:
        """Simulate compute utilization"""
        model = self.op_models.get(op.name, self.op_models["linear"])
        
        if model["memory_bound"]:
            return random.uniform(30, 60)
        else:
            return random.uniform(70, 95)
    
    def _calculate_compute_utilization(self, ops: List[Operation]) -> float:
        """Calculate average compute utilization"""
        if not ops:
            return 0.0
        
        total_util = sum(self._get_compute_utilization(op) for op in ops)
        return total_util / len(ops)
    
    def _identify_bottleneck(self, metrics: GPUMetrics, ops: List[Operation]) -> str:
        """Identify the primary bottleneck type"""
        if metrics.stall_pct > 40:
            return "memory_stall"
        elif metrics.launch_delay_ms > 2.0:
            return "launch_overhead"
        elif metrics.sm_occupancy_pct < 50:
            return "low_occupancy"
        elif metrics.cache_hit_rate < 70:
            return "cache_miss"
        elif metrics.compute_utilization < 60:
            return "compute_underutilization"
        else:
            return "optimal"
    
    def _generate_optimization_flags(self, metrics: GPUMetrics, ops: List[Operation]) -> List[str]:
        """Generate optimization suggestions"""
        flags = []
        
        if metrics.stall_pct > 35:
            flags.append("high_memory_stall")
        
        if metrics.launch_delay_ms > 1.5:
            flags.append("kernel_fusion_candidate")
        
        if metrics.sm_occupancy_pct < 60:
            flags.append("increase_occupancy")
        
        if metrics.cache_hit_rate < 75:
            flags.append("improve_data_locality")
        
        if metrics.compute_utilization < 70:
            flags.append("underutilized_compute")
        
        # Check for fusion opportunities
        if len(ops) > 3:
            flags.append("multi_kernel_fusion")
        
        # Check for specific patterns
        op_names = [op.name for op in ops]
        if "rmsnorm" in op_names and "linear" in op_names:
            flags.append("norm_linear_fusion")
        
        if "kvload" in op_names and "attention" in op_names:
            flags.append("attention_optimization")
        
        return flags
    
    def _calculate_performance_score(self, metrics: GPUMetrics, total_latency: float) -> float:
        """Calculate overall performance score (0-100)"""
        score = 100.0
        
        # Penalize high stall percentage
        score -= metrics.stall_pct * 0.8
        
        # Penalize low occupancy
        score -= max(0, (80 - metrics.sm_occupancy_pct) * 0.5)
        
        # Penalize low cache hit rate
        score -= max(0, (85 - metrics.cache_hit_rate) * 0.3)
        
        # Penalize low compute utilization
        score -= max(0, (80 - metrics.compute_utilization) * 0.4)
        
        # Penalize high launch delay
        score -= min(metrics.launch_delay_ms * 10, 20)
        
        return max(0, min(100, score))
    
    def analyze_sequence(self, tokens: List[TokenTrace]) -> List[TokenAnalysis]:
        """Analyze a sequence of tokens"""
        return [self.analyze_token(token) for token in tokens]
    
    def get_aggregate_stats(self, analyses: List[TokenAnalysis]) -> Dict[str, Any]:
        """Get aggregate statistics across all tokens"""
        if not analyses:
            return {}
        
        avg_latency = sum(a.total_latency_ms for a in analyses) / len(analyses)
        avg_stall_pct = sum(a.gpu_metrics.stall_pct for a in analyses) / len(analyses)
        avg_performance = sum(a.performance_score for a in analyses) / len(analyses)
        
        bottleneck_counts = {}
        for analysis in analyses:
            bottleneck_counts[analysis.bottleneck_type] = bottleneck_counts.get(analysis.bottleneck_type, 0) + 1
        
        all_flags = []
        for analysis in analyses:
            all_flags.extend(analysis.optimization_flags)
        
        flag_counts = {}
        for flag in all_flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
        
        return {
            "total_tokens": len(analyses),
            "avg_latency_ms": avg_latency,
            "avg_stall_pct": avg_stall_pct,
            "avg_performance_score": avg_performance,
            "bottleneck_distribution": bottleneck_counts,
            "optimization_flags": flag_counts,
            "total_latency_ms": sum(a.total_latency_ms for a in analyses)
        }
