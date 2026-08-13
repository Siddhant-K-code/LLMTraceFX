from fastapi.testclient import TestClient

from llmtracefx.api.serve import app


def test_hardware_endpoint_lists_gb10_and_mlx():
    response = TestClient(app).get("/hardware")

    assert response.status_code == 200
    names = [item["name"] for item in response.json()["hardware"]]
    assert "GB10" in names
    assert "MLX" in names


def test_analysis_rejects_unknown_hardware_before_processing():
    response = TestClient(app).post(
        "/analyze-trace",
        json={"trace_data": {"tokens": []}, "gpu_type": "unknown"},
    )

    assert response.status_code == 400
    assert "Choose one of" in response.json()["detail"]
