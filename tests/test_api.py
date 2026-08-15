import json
import sys

from fastapi.testclient import TestClient

from llmtracefx.api.serve import app


def test_hardware_endpoint_lists_gb10_and_mlx():
    response = TestClient(app).get("/hardware")

    assert response.status_code == 200
    names = [item["name"] for item in response.json()["hardware"]]
    assert "GB10" in names
    assert "MLX" in names


def test_hardware_endpoint_exposes_backend_and_memory_model():
    response = TestClient(app).get("/hardware")

    profiles = {item["name"]: item for item in response.json()["hardware"]}
    assert profiles["GB10"]["backend"] == "CUDA"
    assert profiles["GB10"]["unified_memory"] is True
    assert profiles["MLX"]["backend"] == "Metal"


def test_analysis_rejects_unknown_hardware_before_processing():
    response = TestClient(app).post(
        "/analyze-trace",
        json={"trace_data": {"tokens": []}, "gpu_type": "unknown"},
    )

    assert response.status_code == 400
    assert "Choose one of" in response.json()["detail"]


def test_analysis_accepts_apple_silicon_alias_and_exports_metrics(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", None)
    client = TestClient(app)
    response = client.post(
        "/analyze-trace",
        json={
            "trace_data": {
                "tokens": [
                    {
                        "id": 3,
                        "text": "hello",
                        "operations": [
                            {"name": "matmul", "start_time": 0, "duration": 2.5}
                        ],
                    }
                ]
            },
            "gpu_type": "apple-silicon",
            "enable_claude": False,
        },
    )

    assert response.status_code == 200
    analysis_id = response.json()["analysis_id"]
    token = client.get(f"/token/{analysis_id}/3")
    exported = client.get(f"/export/{analysis_id}")
    assert token.status_code == 200
    assert token.json()["gpu_metrics"]["occupancy_label"] == "GPU occupancy"
    assert token.json()["gpu_metrics"]["metrics_source"] == "estimated"
    assert exported.status_code == 200
    assert exported.json()[0]["gpu_metrics"]["memory_bandwidth_gb_s"] is None


def test_analysis_rejects_empty_trace(monkeypatch):
    monkeypatch.setitem(sys.modules, "modal", None)
    response = TestClient(app).post(
        "/analyze-trace",
        json={
            "trace_data": {"tokens": []},
            "gpu_type": "MLX",
            "enable_claude": False,
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Trace does not contain any tokens"


def test_upload_rejects_empty_trace():
    response = TestClient(app).post(
        "/upload-trace?gpu_type=GB10&enable_claude=false",
        files={
            "file": (
                "trace.json",
                json.dumps({"tokens": []}),
                "application/json",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Trace does not contain any tokens"
