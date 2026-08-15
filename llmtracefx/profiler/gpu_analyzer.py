"""GPU performance analysis for token-level inference traces."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any, TypedDict

from llmtracefx.hardware import HardwareProfile, get_hardware_profile

from .trace_parser import Operation, TokenTrace


class OperationModel(TypedDict):
    stall_range: tuple[float, float]
    memory_bound: bool


@dataclass
class GPUMetrics:
    """GPU metrics for an operation or token."""

    stall_pct: float
    launch_delay_ms: float
    memory_latency_ms: float
    sm_occupancy_pct: float
    cache_hit_rate: float
    memory_bandwidth_gb_s: float | None
    compute_utilization: float
    occupancy_label: str = "SM occupancy"
    metrics_source: str = "estimated"


@dataclass
class TokenAnalysis:
    """Complete analysis for a token."""

    token_id: int
    token_text: str
    total_latency_ms: float
    operations: list[Operation]
    gpu_metrics: GPUMetrics
    bottleneck_type: str
    optimization_flags: list[str]
    performance_score: float


class GPUAnalyzer:
    """Analyze token traces for CUDA or Metal hardware profiles.

    Operation metadata can provide measured values using ``stall_pct``,
    ``launch_delay_ms``, ``memory_latency_ms``, ``occupancy_pct`` (or
    ``sm_occupancy_pct``), ``cache_hit_rate``, and ``compute_utilization``.
    Missing values are filled with deterministic operation-level estimates.
    """

    def __init__(self, gpu_type: str = "A10G"):
        self.hardware: HardwareProfile = get_hardware_profile(gpu_type)
        self.gpu_type = self.hardware.name
        # Keep the existing attribute available for callers that inspect it.
        self.gpu_specs = self.hardware.to_dict()

        self.op_models: dict[str, OperationModel] = {
            "rmsnorm": {"stall_range": (10, 25), "memory_bound": True},
            "layernorm": {"stall_range": (10, 25), "memory_bound": True},
            "linear": {"stall_range": (15, 35), "memory_bound": False},
            "matmul": {"stall_range": (20, 45), "memory_bound": False},
            "softmax": {"stall_range": (5, 20), "memory_bound": True},
            "kvload": {"stall_range": (30, 60), "memory_bound": True},
            "kvstore": {"stall_range": (25, 50), "memory_bound": True},
            "attention": {"stall_range": (20, 40), "memory_bound": False},
            "activation": {"stall_range": (5, 15), "memory_bound": True},
            "embedding": {"stall_range": (15, 30), "memory_bound": True},
        }

    def _get_gpu_specs(self, gpu_type: str) -> dict[str, Any]:
        """Return validated hardware specifications for compatibility."""
        return get_hardware_profile(gpu_type).to_dict()

    def analyze_token(self, token_trace: TokenTrace) -> TokenAnalysis:
        """Analyze one token's operations."""
        operations = token_trace.operations
        if not operations:
            metrics = GPUMetrics(
                stall_pct=0,
                launch_delay_ms=0,
                memory_latency_ms=0,
                sm_occupancy_pct=0,
                cache_hit_rate=0,
                memory_bandwidth_gb_s=self.hardware.memory_bandwidth_gb_s,
                compute_utilization=0,
                occupancy_label=self.hardware.occupancy_label,
                metrics_source="unavailable",
            )
            return TokenAnalysis(
                token_id=token_trace.token_id,
                token_text=token_trace.token_text,
                total_latency_ms=token_trace.total_latency,
                operations=[],
                gpu_metrics=metrics,
                bottleneck_type="no_operations",
                optimization_flags=[],
                performance_score=0,
            )

        operation_metrics = [self._analyze_operation(op) for op in operations]
        total_duration = sum(max(op.duration, 0) for op in operations)

        def weighted_average(values: list[float]) -> float:
            if total_duration == 0:
                return sum(values) / len(values)
            return (
                sum(
                    value * max(op.duration, 0)
                    for op, value in zip(operations, values, strict=True)
                )
                / total_duration
            )

        sources = {metrics.metrics_source for metrics in operation_metrics}
        if sources == {"measured"}:
            metrics_source = "measured"
        elif sources == {"estimated"}:
            metrics_source = "estimated"
        else:
            metrics_source = "mixed"

        aggregate_metrics = GPUMetrics(
            stall_pct=weighted_average(
                [metrics.stall_pct for metrics in operation_metrics]
            ),
            launch_delay_ms=sum(
                metrics.launch_delay_ms for metrics in operation_metrics
            )
            / len(operation_metrics),
            memory_latency_ms=sum(
                metrics.memory_latency_ms for metrics in operation_metrics
            ),
            sm_occupancy_pct=weighted_average(
                [metrics.sm_occupancy_pct for metrics in operation_metrics]
            ),
            cache_hit_rate=weighted_average(
                [metrics.cache_hit_rate for metrics in operation_metrics]
            ),
            memory_bandwidth_gb_s=self.hardware.memory_bandwidth_gb_s,
            compute_utilization=weighted_average(
                [metrics.compute_utilization for metrics in operation_metrics]
            ),
            occupancy_label=self.hardware.occupancy_label,
            metrics_source=metrics_source,
        )

        return TokenAnalysis(
            token_id=token_trace.token_id,
            token_text=token_trace.token_text,
            total_latency_ms=token_trace.total_latency,
            operations=operations,
            gpu_metrics=aggregate_metrics,
            bottleneck_type=self._identify_bottleneck(aggregate_metrics, operations),
            optimization_flags=self._generate_optimization_flags(
                aggregate_metrics, operations
            ),
            performance_score=self._calculate_performance_score(
                aggregate_metrics, token_trace.total_latency
            ),
        )

    def _analyze_operation(self, op: Operation) -> GPUMetrics:
        """Use measured metadata when present and estimate missing metrics."""
        model = self.op_models.get(op.name, self.op_models["linear"])
        stall_min, stall_max = model["stall_range"]

        values: list[tuple[float, bool]] = [
            self._metadata_value(op, ("stall_pct",), (stall_min + stall_max) / 2),
            self._metadata_value(op, ("launch_delay_ms",), self._get_launch_delay(op)),
            self._metadata_value(
                op, ("memory_latency_ms",), self._get_memory_latency(op)
            ),
            self._metadata_value(
                op,
                ("occupancy_pct", "sm_occupancy_pct"),
                self._get_sm_occupancy(op),
            ),
            self._metadata_value(op, ("cache_hit_rate",), self._get_cache_hit_rate(op)),
            self._metadata_value(
                op,
                ("compute_utilization",),
                self._get_compute_utilization(op),
            ),
        ]
        measured_count = sum(measured for _, measured in values)
        if measured_count == len(values):
            source = "measured"
        elif measured_count == 0:
            source = "estimated"
        else:
            source = "mixed"

        return GPUMetrics(
            stall_pct=self._clamp(values[0][0]),
            launch_delay_ms=max(0, values[1][0]),
            memory_latency_ms=max(0, values[2][0]),
            sm_occupancy_pct=self._clamp(values[3][0]),
            cache_hit_rate=self._clamp(values[4][0]),
            memory_bandwidth_gb_s=self.hardware.memory_bandwidth_gb_s,
            compute_utilization=self._clamp(values[5][0]),
            occupancy_label=self.hardware.occupancy_label,
            metrics_source=source,
        )

    @staticmethod
    def _metadata_value(
        op: Operation, keys: Sequence[str], default: float
    ) -> tuple[float, bool]:
        for key in keys:
            if key in op.metadata:
                try:
                    value = float(op.metadata[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Operation '{op.name}' metadata '{key}' must be numeric"
                    ) from exc
                if not isfinite(value):
                    raise ValueError(
                        f"Operation '{op.name}' metadata '{key}' must be finite"
                    )
                return value, True
        return float(default), False

    @staticmethod
    def _clamp(value: float, minimum: float = 0, maximum: float = 100) -> float:
        return max(minimum, min(maximum, value))

    def _get_launch_delay(self, op: Operation) -> float:
        """Estimate launch overhead deterministically from operation duration."""
        size_factor = min(max(op.duration, 0) / 10.0, 2.0)
        return 0.1 + (size_factor * 0.175)

    def _get_memory_latency(self, op: Operation) -> float:
        """Estimate the memory component of operation latency."""
        model = self.op_models.get(op.name, self.op_models["linear"])
        factor = 0.45 if model["memory_bound"] else 0.2
        return max(op.duration, 0) * factor

    def _get_sm_occupancy(self, op: Operation) -> float:
        """Estimate accelerator occupancy from operation type."""
        model = self.op_models.get(op.name, self.op_models["linear"])
        return 55.0 if model["memory_bound"] else 75.0

    def _get_cache_hit_rate(self, op: Operation) -> float:
        """Estimate cache hit rate from operation type."""
        return 87.5 if "kv" in op.name else 72.5

    def _get_compute_utilization(self, op: Operation) -> float:
        """Estimate compute utilization from operation type."""
        model = self.op_models.get(op.name, self.op_models["linear"])
        return 45.0 if model["memory_bound"] else 82.5

    def _calculate_compute_utilization(self, ops: list[Operation]) -> float:
        """Calculate average estimated compute utilization."""
        if not ops:
            return 0.0
        return sum(self._get_compute_utilization(op) for op in ops) / len(ops)

    def _identify_bottleneck(self, metrics: GPUMetrics, ops: list[Operation]) -> str:
        if metrics.stall_pct > 40:
            return "memory_stall"
        if metrics.launch_delay_ms > 2.0:
            return "launch_overhead"
        if metrics.sm_occupancy_pct < 50:
            return "low_occupancy"
        if metrics.cache_hit_rate < 70:
            return "cache_miss"
        if metrics.compute_utilization < 60:
            return "compute_underutilization"
        return "optimal"

    def _generate_optimization_flags(
        self, metrics: GPUMetrics, ops: list[Operation]
    ) -> list[str]:
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
        if len(ops) > 3:
            flags.append("multi_kernel_fusion")

        op_names = [op.name for op in ops]
        if "rmsnorm" in op_names and "linear" in op_names:
            flags.append("norm_linear_fusion")
        if "kvload" in op_names and "attention" in op_names:
            flags.append("attention_optimization")
        return flags

    def _calculate_performance_score(
        self, metrics: GPUMetrics, total_latency: float
    ) -> float:
        score = 100.0
        score -= metrics.stall_pct * 0.8
        score -= max(0, (80 - metrics.sm_occupancy_pct) * 0.5)
        score -= max(0, (85 - metrics.cache_hit_rate) * 0.3)
        score -= max(0, (80 - metrics.compute_utilization) * 0.4)
        score -= min(metrics.launch_delay_ms * 10, 20)
        return max(0, min(100, score))

    def analyze_sequence(self, tokens: list[TokenTrace]) -> list[TokenAnalysis]:
        """Analyze a sequence of tokens."""
        return [self.analyze_token(token) for token in tokens]

    def get_aggregate_stats(self, analyses: list[TokenAnalysis]) -> dict[str, Any]:
        """Get aggregate statistics across tokens."""
        if not analyses:
            return {}

        bottleneck_counts: dict[str, int] = {}
        flag_counts: dict[str, int] = {}
        for analysis in analyses:
            bottleneck_counts[analysis.bottleneck_type] = (
                bottleneck_counts.get(analysis.bottleneck_type, 0) + 1
            )
            for flag in analysis.optimization_flags:
                flag_counts[flag] = flag_counts.get(flag, 0) + 1

        return {
            "total_tokens": len(analyses),
            "avg_latency_ms": sum(a.total_latency_ms for a in analyses) / len(analyses),
            "avg_stall_pct": sum(a.gpu_metrics.stall_pct for a in analyses)
            / len(analyses),
            "avg_performance_score": sum(a.performance_score for a in analyses)
            / len(analyses),
            "bottleneck_distribution": bottleneck_counts,
            "optimization_flags": flag_counts,
            "total_latency_ms": sum(a.total_latency_ms for a in analyses),
        }
