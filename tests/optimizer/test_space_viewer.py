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
