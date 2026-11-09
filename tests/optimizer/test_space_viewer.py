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


def test_renders_bundled_example(tmp_path):
    result = viewer.render(
        SPACE / "examples" / "synthetic-record.json", output_dir=tmp_path
    )
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
        viewer.render(bad, output_dir=tmp_path)


def test_rejects_non_object_evidence(tmp_path):
    evidence = tmp_path / "api.json"
    evidence.write_text("[]")
    with pytest.raises(viewer.ViewerError, match="must be a JSON object"):
        viewer.render(
            SPACE / "examples" / "synthetic-record.json",
            evidence,
            output_dir=tmp_path,
        )


def test_output_names_are_sanitized(tmp_path):
    record = json.loads((SPACE / "examples" / "synthetic-record.json").read_text())
    record["run_id"] = "../../etc/x y"
    path = tmp_path / "record.json"
    path.write_text(json.dumps(record))
    result = viewer.render(path, output_dir=tmp_path / "out")
    assert result.perfetto_path.parent == tmp_path / "out"
    assert result.perfetto_path.name == "etc_x_y.perfetto.json"
