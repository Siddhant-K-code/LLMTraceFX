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


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_metadata_is_rejected(value):
    with pytest.raises(ValueError, match="must be finite"):
        GPUAnalyzer("MLX").analyze_token(make_token({"stall_pct": value}))


def test_percentage_metadata_is_clamped_to_valid_range():
    analysis = GPUAnalyzer("MLX").analyze_token(
        make_token({"occupancy_pct": 110, "cache_hit_rate": -2})
    )

    assert analysis.gpu_metrics.sm_occupancy_pct == 100
    assert analysis.gpu_metrics.cache_hit_rate == 0
    assert analysis.gpu_metrics.metrics_source == "mixed"


def test_empty_token_has_explicit_unavailable_analysis():
    token = TokenTrace(
        token_id=2,
        token_text="",
        total_latency=0,
        operations=[],
        start_time=0,
        end_time=0,
    )

    analysis = GPUAnalyzer("GB10").analyze_token(token)

    assert analysis.bottleneck_type == "no_operations"
    assert analysis.performance_score == 0
    assert analysis.gpu_metrics.metrics_source == "unavailable"


def test_aggregate_metrics_are_duration_weighted():
    operations = [
        Operation(
            name="matmul",
            start_time=0,
            duration=9,
            metadata={"stall_pct": 10},
        ),
        Operation(
            name="softmax",
            start_time=9,
            duration=1,
            metadata={"stall_pct": 90},
        ),
    ]
    token = TokenTrace(1, "test", 10, operations, 0, 10)

    analysis = GPUAnalyzer("GB10").analyze_token(token)

    assert analysis.gpu_metrics.stall_pct == pytest.approx(18)
    assert analysis.gpu_metrics.metrics_source == "mixed"
