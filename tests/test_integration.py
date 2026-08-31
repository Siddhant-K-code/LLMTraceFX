import json
import subprocess
import sys

import pytest

from llmtracefx.profiler.gpu_analyzer import GPUAnalyzer
from llmtracefx.profiler.trace_parser import TraceParser
from llmtracefx.visualize.flame import FlameGraphGenerator


def mlx_trace_payload():
    return {
        "format": "llmtracefx.mlx.v1",
        "framework": "mlx",
        "hardware": "MLX",
        "device": {"device_name": "Test Apple GPU"},
        "time_unit": "ms",
        "tokens": [
            {
                "id": 0,
                "text": "hello",
                "operations": [
                    {
                        "name": "model_forward",
                        "start_time": 0,
                        "duration": 2.5,
                        "metadata": {
                            "backend": "metal",
                            "active_memory_bytes": 1024,
                        },
                    }
                ],
            }
        ],
    }


def test_mlx_trace_parser_analyzer_and_visualizer_pipeline():
    tokens = TraceParser().parse_trace_data(mlx_trace_payload())
    analyses = GPUAnalyzer("MLX").analyze_sequence(tokens)
    visualizer = FlameGraphGenerator()

    exported = json.loads(visualizer.export_data_json(analyses))
    dashboard = visualizer.generate_comprehensive_dashboard(analyses)

    assert exported[0]["total_latency_ms"] == 2.5
    assert exported[0]["gpu_metrics"]["occupancy_label"] == "GPU occupancy"
    assert exported[0]["gpu_metrics"]["metrics_source"] == "estimated"
    assert "GPU occupancy" in dashboard
    assert dashboard.lower().count("<!doctype html>") == 1
    assert dashboard.lower().count("<html") == 1
    assert '"showSendToCloud": false' in dashboard


@pytest.mark.parametrize(
    ("hardware", "display_name", "occupancy_label"),
    [
        ("apple-silicon", "Apple Silicon (MLX / Metal)", "GPU occupancy"),
        ("DGX-SPARK", "NVIDIA GB10 / DGX Spark", "SM occupancy"),
    ],
)
def test_cli_analyzes_new_hardware_aliases(
    tmp_path, hardware, display_name, occupancy_label
):
    trace_path = tmp_path / "mlx_trace.json"
    output_dir = tmp_path / "nested" / "report"
    trace_path.write_text(json.dumps(mlx_trace_payload()), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "llmtracefx.main",
            "--trace",
            str(trace_path),
            "--gpu-type",
            hardware,
            "--no-claude",
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    report = (output_dir / "report.txt").read_text(encoding="utf-8")
    exported = json.loads((output_dir / "analysis_data.json").read_text())
    assert display_name in report
    assert "Metrics Source: estimated" in report
    assert exported[0]["gpu_metrics"]["occupancy_label"] == occupancy_label
