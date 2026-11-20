from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SPACE = Path(__file__).resolve().parents[2] / "spaces" / "evidence-viewer"


def _load_viewer():
    spec = importlib.util.spec_from_file_location("space_viewer", SPACE / "viewer.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolve annotations through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


viewer = _load_viewer()
EXAMPLE = SPACE / "examples" / "synthetic-record.json"


def render(*args, roots, **kwargs):
    return viewer.render(*args, allowed_roots=roots, **kwargs)


def test_renders_bundled_example(tmp_path):
    result = render(EXAMPLE, roots=[SPACE / "examples"], output_dir=tmp_path)
    assert "`synthetic-example-0` succeeded" in result.summary_markdown
    assert "| `timing.total` | 5160.2 | ms | `measured_wall_clock` |" in (
        result.measurements_markdown
    )
    assert "prefill" in result.spans_markdown
    trace = json.loads(result.perfetto_path.read_text())
    assert trace["traceEvents"]
    otlp = json.loads(result.otlp_path.read_text())
    assert otlp["resourceSpans"]
    assert result.perfetto_path.parent == tmp_path


def test_rejects_invalid_record(tmp_path):
    bad = tmp_path / "record.json"
    bad.write_text('{"not": "a record"}')
    with pytest.raises(viewer.ViewerError, match="not a valid ExperimentRecord"):
        render(bad, roots=[tmp_path], output_dir=tmp_path)


def test_rejects_non_object_evidence(tmp_path):
    evidence = tmp_path / "api.json"
    evidence.write_text("[]")
    with pytest.raises(viewer.ViewerError, match="must be a JSON object"):
        render(EXAMPLE, evidence, roots=[tmp_path, EXAMPLE.parent], output_dir=tmp_path)


def test_output_names_are_sanitized(tmp_path):
    record = json.loads(EXAMPLE.read_text())
    record["run_id"] = "../../etc/x y"
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))
    result = render(path, roots=[tmp_path], output_dir=tmp_path / "out")
    assert result.perfetto_path.parent == tmp_path / "out"
    assert result.perfetto_path.name == "etc_x_y.perfetto.json"


def test_rejects_paths_outside_allowed_roots(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "link.json").symlink_to(EXAMPLE)
    with pytest.raises(viewer.ViewerError, match="outside the upload directory"):
        render(EXAMPLE, roots=[uploads], output_dir=tmp_path)
    with pytest.raises(viewer.ViewerError, match="outside the upload directory"):
        render(uploads / "link.json", roots=[uploads], output_dir=tmp_path)
    with pytest.raises(viewer.ViewerError, match="outside the upload directory"):
        render(uploads / ".." / "x.json", roots=[uploads], output_dir=tmp_path)


def test_rejects_oversized_upload(tmp_path):
    big = tmp_path / "record.json"
    big.write_bytes(b" " * (viewer.MAX_UPLOAD_BYTES + 1))
    with pytest.raises(viewer.ViewerError, match="larger than"):
        render(big, roots=[tmp_path], output_dir=tmp_path)


def test_default_roots_include_examples_and_gradio_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("GRADIO_TEMP_DIR", str(tmp_path))
    assert viewer.default_allowed_roots() == (EXAMPLE.parent, tmp_path)
    assert (
        "synthetic-example-0"
        in viewer.render(EXAMPLE, output_dir=tmp_path / "out").summary_markdown
    )


LLAMACPP = {
    name: SPACE / "examples" / f"llamacpp-{name}.json"
    for name in ("record", "api_evidence", "server_metrics_evidence")
}


def test_renders_request_and_server_evidence(tmp_path):
    result = render(*LLAMACPP.values(), roots=[SPACE / "examples"], output_dir=tmp_path)
    assert "provider `llamacpp-local`" in result.summary_markdown
    assert "input not recorded, generated not recorded" in result.summary_markdown
    evidence = result.evidence_markdown
    assert "| finish reason | `stop` | `provider_reported` |" in evidence
    assert "| first content token (ms) | 88.8856 | `measured_wall_clock` |" in evidence
    assert "The provider sent no token usage for this request." in evidence
    assert (
        "| `generation_tokens` | 93 | tokens | `measured_native` "
        "| llamacpp:tokens_predicted |"
    ) in evidence
    assert "Speculative decoding: `no_drafts`." in evidence


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def test_withholds_server_values_when_window_not_attributable(tmp_path):
    server = json.loads(LLAMACPP["server_metrics_evidence"].read_text())
    server["attribution"].update(attributable=False, status="unverifiable")
    server["highlights_withheld_reason"] = "other traffic may have reached it"
    path = _write(tmp_path, "server.json", server)
    result = render(
        LLAMACPP["record"],
        None,
        path,
        roots=[tmp_path, EXAMPLE.parent],
        output_dir=tmp_path,
    )
    assert "window `unverifiable`" in result.evidence_markdown
    assert "withheld" in result.evidence_markdown
    assert "other traffic may have reached it" in result.evidence_markdown
    assert "generation_tokens" not in result.evidence_markdown


def test_renders_reported_usage_measured_speculation_and_kv_usage(tmp_path):
    api = json.loads(LLAMACPP["api_evidence"].read_text())
    api["usage"] = {"reported": True, "prompt_tokens": 47, "completion_tokens": 93}
    api["server_reported"] = {"normalized": {"server_ttft_ms": 12.5, "queue": None}}
    server = json.loads(LLAMACPP["server_metrics_evidence"].read_text())
    entry = {"provenance": "measured_native", "unit": "tokens", "source": "s"}
    server["speculative"] = {
        "status": "measured",
        "reason": "drafts ran",
        "counters": {"accepted_tokens": {**entry, "value": 30}},
        "acceptance_rate": {**entry, "value": 0.75, "provenance": "derived"},
    }
    server["kv_usage"] = {
        "kv_cache_usage": {"unit": "ratio", "before": 0.1, "after": 0.4}
    }
    result = render(
        LLAMACPP["record"],
        _write(tmp_path, "api.json", api),
        _write(tmp_path, "server.json", server),
        roots=[tmp_path, EXAMPLE.parent],
        output_dir=tmp_path,
    )
    evidence = result.evidence_markdown
    assert "| completion_tokens | 93 | `provider_reported` |" in evidence
    assert "| server server_ttft_ms | 12.5 | `provider_reported` |" in evidence
    assert "server queue" not in evidence
    assert "no token usage" not in evidence
    assert "| `accepted_tokens` | 30 | tokens | `measured_native` | s |" in evidence
    assert "| `acceptance_rate` | 0.75 | tokens | `derived` | s |" in evidence
    assert "| `kv_cache_usage` | 0.1 | 0.4 | ratio |" in evidence


def test_record_only_has_no_evidence_section(tmp_path):
    result = render(EXAMPLE, roots=[SPACE / "examples"], output_dir=tmp_path)
    assert result.evidence_markdown == ""
