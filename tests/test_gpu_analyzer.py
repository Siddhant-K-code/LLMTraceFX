import pytest

from llmtracefx.profiler.gpu_analyzer import GPUAnalyzer
from llmtracefx.profiler.trace_parser import Operation, TokenTrace


def make_token(metadata=None):
    operation = Operation(
        name="matmul",
        start_time=0,
        duration=10,
        metadata=metadata or {},
    )
    return TokenTrace(
        token_id=1,
        token_text="test",
        total_latency=10,
        operations=[operation],
        start_time=0,
        end_time=10,
    )


def test_gb10_analysis_is_deterministic():
    analyzer = GPUAnalyzer("DGX-SPARK")

    first = analyzer.analyze_token(make_token())
    second = analyzer.analyze_token(make_token())

    assert analyzer.gpu_type == "GB10"
    assert first.gpu_metrics == second.gpu_metrics
    assert first.performance_score == second.performance_score
    assert first.gpu_metrics.memory_bandwidth_gb_s == 273
    assert first.gpu_metrics.metrics_source == "estimated"


def test_measured_operation_metadata_overrides_estimates():
    metadata = {
        "stall_pct": 12,
        "launch_delay_ms": 0.2,
        "memory_latency_ms": 1.5,
        "occupancy_pct": 88,
        "cache_hit_rate": 91,
        "compute_utilization": 84,
    }

    analysis = GPUAnalyzer("MLX").analyze_token(make_token(metadata))

    assert analysis.gpu_metrics.stall_pct == 12
    assert analysis.gpu_metrics.launch_delay_ms == 0.2
    assert analysis.gpu_metrics.memory_latency_ms == 1.5
    assert analysis.gpu_metrics.sm_occupancy_pct == 88
    assert analysis.gpu_metrics.cache_hit_rate == 91
    assert analysis.gpu_metrics.compute_utilization == 84
    assert analysis.gpu_metrics.occupancy_label == "GPU occupancy"
    assert analysis.gpu_metrics.metrics_source == "measured"


def test_partial_metadata_is_marked_mixed():
    analysis = GPUAnalyzer("MLX").analyze_token(make_token({"stall_pct": 15}))

    assert analysis.gpu_metrics.stall_pct == 15
    assert analysis.gpu_metrics.metrics_source == "mixed"


def test_non_numeric_metadata_is_rejected():
    with pytest.raises(ValueError, match="must be numeric"):
        GPUAnalyzer("GB10").analyze_token(make_token({"stall_pct": "high"}))
