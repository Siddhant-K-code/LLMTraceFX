import json

import pytest

from llmtracefx.profiler import MLXTraceRecorder
from llmtracefx.profiler.trace_parser import TraceParser


class FakeClock:
    def __init__(self, *values):
        self.values = iter(values)

    def __call__(self):
        return next(self.values)


class FakeMetal:
    def __init__(self):
        self.started = None
        self.stopped = False

    def start_capture(self, path):
        self.started = path

    def stop_capture(self):
        self.stopped = True


class FakeMLX:
    def __init__(self):
        self.eval_calls = []
        self.synchronize_calls = 0
        self.metal = FakeMetal()

    def eval(self, *values):
        self.eval_calls.append(values)

    def synchronize(self):
        self.synchronize_calls += 1

    def device_info(self):
        return {
            "device_name": "Fake M4 Max",
            "architecture": "applegpu_g16s",
            "memory_size": 64 * 1024**3,
        }

    def get_active_memory(self):
        return 1024

    def get_peak_memory(self):
        return 2048

    def get_cache_memory(self):
        return 512


def test_mlx_recorder_writes_parseable_trace(tmp_path):
    output_path = tmp_path / "mlx_trace.json"
    fake_mlx = FakeMLX()
    result = object()
    clock = FakeClock(0, 1_000_000, 4_000_000)

    with MLXTraceRecorder(output_path, mlx_module=fake_mlx, clock_ns=clock) as trace:
        with trace.token(7, "hello"):
            returned = trace.measure("matmul", lambda: result)

    payload = json.loads(output_path.read_text())
    operation = payload["tokens"][0]["operations"][0]

    assert returned is result
    assert payload["format"] == "llmtracefx.mlx.v1"
    assert payload["hardware"] == "MLX"
    assert payload["device"]["device_name"] == "Fake M4 Max"
    assert operation["start_time"] == 1.0
    assert operation["duration"] == 3.0
    assert operation["metadata"]["active_memory_bytes"] == 1024
    assert fake_mlx.eval_calls == [(result,)]
    assert fake_mlx.synchronize_calls == 2

    tokens = TraceParser().parse_trace_file(str(output_path))
    assert tokens[0].token_id == 7
    assert tokens[0].operations[0].name == "matmul"
    assert tokens[0].total_latency == 3.0


def test_optional_native_metal_capture_uses_same_region(tmp_path):
    output_path = tmp_path / "mlx_trace.json"
    capture_path = tmp_path / "mlx_trace.gputrace"
    fake_mlx = FakeMLX()
    clock = FakeClock(0, 1_000_000, 2_000_000)

    with MLXTraceRecorder(
        output_path,
        metal_capture_path=capture_path,
        mlx_module=fake_mlx,
        clock_ns=clock,
    ) as trace:
        with trace.token(0):
            trace.measure("softmax", lambda: object())

    assert fake_mlx.metal.started == str(capture_path)
    assert fake_mlx.metal.stopped is True


def test_measure_requires_an_active_token(tmp_path):
    recorder = MLXTraceRecorder(tmp_path / "trace.json", mlx_module=FakeMLX())

    with pytest.raises(RuntimeError, match="Start a token trace"):
        recorder.measure("matmul", lambda: object())


def test_native_capture_refuses_to_overwrite_existing_trace(tmp_path):
    capture_path = tmp_path / "existing.gputrace"
    capture_path.mkdir()
    recorder = MLXTraceRecorder(
        tmp_path / "trace.json",
        metal_capture_path=capture_path,
        mlx_module=FakeMLX(),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        with recorder:
            pass
