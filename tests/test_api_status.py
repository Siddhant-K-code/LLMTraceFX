"""Regression tests for the explicit analysis status in llmtracefx.api.serve."""

import asyncio
import sys
import types
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from llmtracefx.api import serve
from llmtracefx.api.serve import AnalysisResponse, app
from llmtracefx.profiler.gpu_analyzer import GPUAnalyzer
from llmtracefx.profiler.trace_parser import TraceParser
from llmtracefx.visualize.flame import FlameGraphGenerator

TRACE = {
    "tokens": [
        {
            "id": 1,
            "text": "hello",
            "operations": [
                {"name": "matmul", "start_time": 0, "duration": 2.5},
                {"name": "softmax", "start_time": 2.5, "duration": 0.5},
            ],
        },
        {
            "id": 2,
            "text": "world",
            "operations": [{"name": "matmul", "start_time": 3, "duration": 4.0}],
        },
    ]
}


class OkExplainer:
    async def explain_token_performance(self, analysis: Any) -> str:
        return f"explained {analysis.token_id}"

    def format_explanation_for_display(self, explanation: str) -> str:
        return explanation


class FailingExplainer:
    async def explain_token_performance(self, analysis: Any) -> str:
        raise RuntimeError("Claude API error 401: invalid key")

    def format_explanation_for_display(self, explanation: str) -> str:
        return explanation


class CancelledExplainer:
    async def explain_token_performance(self, analysis: Any) -> str:
        raise asyncio.CancelledError()

    def format_explanation_for_display(self, explanation: str) -> str:
        return explanation


@pytest.fixture
def cpu_only(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", None)


def _post(client: TestClient, enable_claude: bool) -> dict[str, Any]:
    response = client.post(
        "/analyze-trace",
        json={"trace_data": TRACE, "gpu_type": "A10G", "enable_claude": enable_claude},
    )
    assert response.status_code == 200
    return response.json()


def test_explanation_success_reports_completed(cpu_only, monkeypatch):
    monkeypatch.setattr(serve, "explainer", OkExplainer())
    client = TestClient(app)

    created = _post(client, enable_claude=True)
    assert created["status"] == "processing_explanations"

    summary = client.get(f"/analysis/{created['analysis_id']}").json()
    assert summary["status"] == "completed"
    assert summary["error"] is None

    explanation = client.get(f"/explain/{created['analysis_id']}/2")
    assert explanation.json() == {"explanation": "explained 2"}


def test_explanation_failure_reports_error_state(cpu_only, monkeypatch):
    monkeypatch.setattr(serve, "explainer", FailingExplainer())
    client = TestClient(app)

    created = _post(client, enable_claude=True)
    assert created["status"] == "processing_explanations"

    summary = client.get(f"/analysis/{created['analysis_id']}").json()
    assert summary["status"] == "explanations_failed"
    assert "invalid key" in summary["error"]
    assert summary["total_tokens"] == 2
    assert summary["total_latency_ms"] > 0


def test_missing_explainer_reports_error_state(cpu_only, monkeypatch):
    monkeypatch.setattr(serve, "explainer", None)
    client = TestClient(app)

    created = _post(client, enable_claude=True)
    analysis_id = created["analysis_id"]

    summary = client.get(f"/analysis/{analysis_id}").json()
    assert summary["status"] == "explanations_failed"
    assert "CLAUDE_API_KEY" in summary["error"]

    explanation = client.get(f"/explain/{analysis_id}/1")
    assert explanation.status_code == 404
    assert "CLAUDE_API_KEY" in explanation.json()["detail"]


def test_cancelled_explanation_task_does_not_stay_processing(monkeypatch):
    monkeypatch.setattr(serve, "explainer", CancelledExplainer())
    analyses = GPUAnalyzer("A10G").analyze_sequence(
        TraceParser().parse_trace_data(TRACE)
    )
    analysis_id = serve._register_analysis(analyses, explanations_pending=True)

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(serve.generate_claude_explanations(analysis_id, analyses))

    state = serve.analysis_states[analysis_id]
    assert state.status == "explanations_failed"
    assert state.error


def test_analysis_without_claude_reports_completed(cpu_only):
    client = TestClient(app)

    created = _post(client, enable_claude=False)
    assert created["status"] == "completed"

    summary = client.get(f"/analysis/{created['analysis_id']}").json()
    assert summary["status"] == "completed"


def test_status_values_are_constrained():
    with pytest.raises(ValidationError):
        AnalysisResponse(
            analysis_id="analysis_x",
            total_tokens=1,
            total_latency_ms=1.0,
            avg_performance_score=1.0,
            bottleneck_summary={},
            status="done",
        )


def _install_fake_modal(monkeypatch, result: dict[str, Any]) -> None:
    def remote(**kwargs: Any) -> dict[str, Any]:
        return result

    modal_app = types.ModuleType("llmtracefx.modal_app")
    modal_app.app = types.SimpleNamespace(  # type: ignore[attr-defined]
        analyze_trace_modal=types.SimpleNamespace(remote=remote)
    )
    monkeypatch.setitem(sys.modules, "modal", types.ModuleType("modal"))
    monkeypatch.setitem(sys.modules, "llmtracefx.modal_app", modal_app)


def _gpu_result() -> dict[str, Any]:
    analyses = GPUAnalyzer("A10G").analyze_sequence(
        TraceParser().parse_trace_data(TRACE)
    )
    visualizer = FlameGraphGenerator()
    bottlenecks: dict[str, int] = {}
    for analysis in analyses:
        bottlenecks[analysis.bottleneck_type] = (
            bottlenecks.get(analysis.bottleneck_type, 0) + 1
        )
    return {
        "total_tokens": len(analyses),
        "total_latency_ms": sum(a.total_latency_ms for a in analyses),
        "avg_performance_score": sum(a.performance_score for a in analyses)
        / len(analyses),
        "bottleneck_summary": bottlenecks,
        "dashboard_html": "<html>gpu dashboard</html>",
        "export_json": visualizer.export_data_json(analyses),
        "explanations": {1: "gpu explanation 1"},
        "status": "completed",
    }


def test_gpu_path_serves_real_results(monkeypatch):
    gpu_result = _gpu_result()
    _install_fake_modal(monkeypatch, gpu_result)
    client = TestClient(app)

    created = _post(client, enable_claude=True)
    analysis_id = created["analysis_id"]
    assert created["status"] == "completed"

    summary = client.get(f"/analysis/{analysis_id}").json()
    assert summary["status"] == "completed"
    assert summary["total_tokens"] == gpu_result["total_tokens"]
    assert summary["total_latency_ms"] == pytest.approx(gpu_result["total_latency_ms"])
    assert summary["avg_performance_score"] == pytest.approx(
        gpu_result["avg_performance_score"]
    )
    assert summary["bottleneck_summary"] == gpu_result["bottleneck_summary"]

    token = client.get(f"/token/{analysis_id}/1")
    assert token.status_code == 200
    assert token.json()["token_text"] == "hello"
    assert token.json()["total_latency_ms"] > 0
    assert [op["name"] for op in token.json()["operations"]] == ["matmul", "softmax"]
    assert token.json()["claude_explanation"] == "gpu explanation 1"

    assert client.get(f"/flame/{analysis_id}").status_code == 200
    assert "gpu dashboard" in client.get(f"/dashboard/{analysis_id}").text
    exported = client.get(f"/export/{analysis_id}").json()
    assert [record["token_id"] for record in exported] == [1, 2]


def test_gpu_result_without_export_falls_back_to_cpu(monkeypatch):
    gpu_result = _gpu_result()
    del gpu_result["export_json"]
    _install_fake_modal(monkeypatch, gpu_result)
    client = TestClient(app)

    created = _post(client, enable_claude=False)
    summary = client.get(f"/analysis/{created['analysis_id']}").json()
    assert summary["status"] == "completed"
    assert summary["total_tokens"] == 2
    assert summary["total_latency_ms"] > 0


def _post_gpu(
    monkeypatch, explanations: dict[int, str], enable_claude: bool
) -> tuple[TestClient, dict[str, Any]]:
    gpu_result = _gpu_result()
    gpu_result["explanations"] = explanations
    _install_fake_modal(monkeypatch, gpu_result)
    client = TestClient(app)
    return client, _post(client, enable_claude=enable_claude)


def test_gpu_claude_requested_but_none_returned_reports_failed(monkeypatch):
    client, created = _post_gpu(monkeypatch, {}, enable_claude=True)
    analysis_id = created["analysis_id"]
    assert created["status"] == "explanations_failed"
    assert "CLAUDE_API_KEY" in created["error"]

    summary = client.get(f"/analysis/{analysis_id}").json()
    assert summary["status"] == "explanations_failed"
    assert summary["total_tokens"] == 2

    explanation = client.get(f"/explain/{analysis_id}/1")
    assert explanation.status_code == 404
    assert "not available yet" not in explanation.json()["detail"]
    assert "CLAUDE_API_KEY" in explanation.json()["detail"]


def test_gpu_claude_all_tokens_failed_reports_failed(monkeypatch):
    explanations = {1: "Error: Claude API error 401", 2: "Error: Claude API error 401"}
    client, created = _post_gpu(monkeypatch, explanations, enable_claude=True)
    assert created["status"] == "explanations_failed"
    assert created["error"] == (
        "Explanation generation failed for all 2 tokens: Claude API error 401"
    )

    explanation = client.get(f"/explain/{created['analysis_id']}/1")
    assert explanation.json() == {"explanation": "Error: Claude API error 401"}


def test_gpu_claude_some_tokens_succeeded_reports_completed(monkeypatch):
    explanations = {1: "gpu explanation 1", 2: "Error: Claude API error 529"}
    client, created = _post_gpu(monkeypatch, explanations, enable_claude=True)
    assert created["status"] == "completed"
    assert created["error"] is None

    explanation = client.get(f"/explain/{created['analysis_id']}/1")
    assert explanation.json() == {"explanation": "gpu explanation 1"}


def test_gpu_claude_not_requested_reports_completed(monkeypatch):
    client, created = _post_gpu(monkeypatch, {}, enable_claude=False)
    assert created["status"] == "completed"
    assert created["error"] is None

    summary = client.get(f"/analysis/{created['analysis_id']}").json()
    assert summary["status"] == "completed"
