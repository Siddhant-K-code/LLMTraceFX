"""sweep-mlx with a fake in-process runtime. Runs on Linux without mlx."""

from __future__ import annotations

import contextlib
import io
import json
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer import cli
from llmtracefx.optimizer.collectors.mlx import MLXCollectorError, MLXMemorySnapshot
from llmtracefx.optimizer.load import backend_cli
from llmtracefx.optimizer.load.api_worker import ApiRequestSpec
from llmtracefx.optimizer.load.artifact import SWEEP_KIND, SWEEP_PLAN_KIND
from llmtracefx.optimizer.load.dispatcher import ClosedLoop, DispatchContext, dispatch
from llmtracefx.optimizer.load.mlx_worker import (
    MLXBusyError,
    MLXLoadConfig,
    MLXSession,
    make_mlx_sender,
)


@dataclass
class Response:
    text: str
    generation_tokens: int
    finish_reason: str | None = None
    from_draft: bool = False
    prompt_tokens: int = 4


EOS_SCRIPT = (
    Response("Hel", 1),
    Response("lo", 2),
    Response("!", 3),
    Response("", 4, "stop"),
)
LENGTH_SCRIPT = (Response("a", 1), Response("b", 2), Response("c", 3, "length"))


class FakeRuntime:
    """Records which thread touched the model."""

    instances: list[FakeRuntime] = []

    def __init__(self, *, temperature: float = 0.0, top_p: float = 1.0) -> None:
        self.temperature = temperature
        self.top_p = top_p
        self.script: tuple[Response, ...] = EOS_SCRIPT
        self.raise_in_generate: BaseException | None = None
        self.threads: dict[str, set[int]] = {}
        self.seeds: list[int] = []
        self.created_on = threading.get_ident()
        FakeRuntime.instances.append(self)

    def _touch(self, name: str) -> None:
        self.threads.setdefault(name, set()).add(threading.get_ident())

    mlx_version = "0.0-fake"
    mlx_lm_version = "0.0-fake"
    runtime_name = "mlx-lm"
    runtime_version = "0.0-fake"

    def load_model(self, path: Path) -> tuple[Any, Any]:
        self._touch("load")
        return object(), object()

    def encode(self, tokenizer: Any, prompt: str) -> list[int]:
        self._touch("encode")
        return list(range(len(prompt.split())))

    def seed(self, seed: int) -> None:
        self.seeds.append(seed)

    def synchronize(self) -> None:
        self._touch("synchronize")

    def reset_peak_memory(self) -> None:
        pass

    def memory_snapshot(self) -> MLXMemorySnapshot:
        return MLXMemorySnapshot()

    def accelerator_name(self) -> str | None:
        return "Fake M9"

    def stream_generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt_tokens: list[int],
        *,
        max_tokens: int,
        draft_model: Any,
        num_draft_tokens: int,
    ) -> Iterator[Response]:
        self._touch("generate")
        assert draft_model is None
        if self.raise_in_generate is not None:
            raise self.raise_in_generate
        yield from self.script


@pytest.fixture(autouse=True)
def _fresh_instances() -> Iterator[None]:
    FakeRuntime.instances.clear()
    yield
    FakeRuntime.instances.clear()


def config(**changes: Any) -> MLXLoadConfig:
    values: dict[str, Any] = {
        "run_id": "mlx-load",
        "model_path": Path("."),
        "model_id": "fake/model",
        "max_tokens": 8,
    }
    values.update(changes)
    return MLXLoadConfig(**values)


@contextlib.contextmanager
def session(runtime: FakeRuntime | None = None, **changes: Any):
    runtime = runtime or FakeRuntime()
    live = MLXSession(lambda: runtime, config(**changes), clock=time.perf_counter)
    live.start()
    try:
        yield live, runtime
    finally:
        live.close()


def one_row(live: MLXSession, cfg: MLXLoadConfig | None = None) -> dict[str, Any]:
    send = make_mlx_sender(live, cfg or config())
    report = dispatch([ApiRequestSpec(prompt="one two three")], send, ClosedLoop(1))
    outcome = report.outcomes[0]
    assert outcome.client_error is None
    return outcome.result.to_dict()


def test_model_loads_and_generates_on_one_owner_thread() -> None:
    with session() as (live, runtime):
        one_row(live)
        one_row(live)
        owner = live.owner_thread
        assert owner is not None
    used = set().union(*runtime.threads.values())
    assert used == {owner.ident}
    assert owner.ident != threading.get_ident()
    assert runtime.seeds == [0, 0]


def test_eos_summary_is_not_counted_as_a_token() -> None:
    with session() as (live, _):
        row = one_row(live)
    assert row["success"] is True
    assert row["output_tokens"] == 3
    assert row["prompt_tokens"] == 3
    assert row["finish_reason"] == "stop"
    assert row["finish_reason_code"] == "stop"
    assert row["content_delta_count"] == 3
    assert row["ttft_source"] == "first_counted_token"
    assert row["http_status"] is None
    assert row["cached_prompt_tokens"] is None
    assert row["tpot_status"] == "derived"
    assert row["ttft_ms"] <= row["e2e_ms"]


def test_length_stop_counts_the_last_token() -> None:
    runtime = FakeRuntime()
    runtime.script = LENGTH_SCRIPT
    with session(runtime) as (live, _):
        row = one_row(live)
    assert row["output_tokens"] == 3
    assert row["finish_reason_code"] == "length"


def test_runtime_error_keeps_type_not_message() -> None:
    runtime = FakeRuntime()
    runtime.raise_in_generate = RuntimeError("secret prompt text in message")
    with session(runtime) as (live, _):
        row = one_row(live)
    assert row["success"] is False
    assert row["failure_category"] == "runtime_error"
    assert row["runtime_error_type"] == "RuntimeError"
    assert "secret" not in json.dumps(row)


def test_generation_without_finish_reason_fails() -> None:
    runtime = FakeRuntime()
    runtime.script = (Response("a", 1), Response("b", 2))
    with session(runtime) as (live, _):
        row = one_row(live)
    assert row["success"] is False
    assert row["failure_category"] == "no_finish_reason"


def test_timeout_is_checked_between_tokens() -> None:
    ticks = iter(float(n) for n in range(1000))
    context = DispatchContext(
        index=0,
        scheduled_offset_s=0.0,
        dispatched_offset_s=0.0,
        in_flight_at_dispatch=1,
        clock=lambda: next(ticks),
        clock_origin=0.0,
    )
    cfg = config(request_timeout_seconds=1.5)
    with session() as (live, _):
        row = make_mlx_sender(live, cfg)(ApiRequestSpec(prompt="x"), context)
    assert row.success is False
    assert row.failure_category == "timeout"
    assert row.output_tokens is None


def test_second_request_while_busy_is_refused_not_queued() -> None:
    with session() as (live, _):
        assert live.acquire()
        try:
            send = make_mlx_sender(live, config())
            report = dispatch([ApiRequestSpec(prompt="x")], send, ClosedLoop(1))
        finally:
            live.release()
    assert report.outcomes[0].client_error == "MLXBusyError"
    with pytest.raises(MLXBusyError):
        with session() as (live, _):
            live.acquire()
            make_mlx_sender(live, config())(
                ApiRequestSpec(prompt="x"),
                DispatchContext(0, 0.0, 0.0, 1, time.perf_counter, 0.0),
            )


# CLI


def run_main(argv: list[str]) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            cli.main(argv)
    except SystemExit as exit_error:
        code = int(exit_error.code or 0)
    return code, stdout.getvalue(), stderr.getvalue()


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend_cli, "MLXLMRuntime", FakeRuntime)


def sweep_argv(tmp_path: Path, *extra: str) -> list[str]:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Tell me a story", encoding="utf-8")
    return [
        "sweep-mlx",
        "--run-id",
        "mlx-load",
        "--model-path",
        str(model),
        "--model-id",
        "fake/model",
        "--output-dir",
        str(tmp_path / "out"),
        "--prompt-file",
        str(prompt_file),
        "--requests-per-point",
        "3",
        *extra,
    ]


def test_cli_sweep_writes_comparable_rows(tmp_path: Path, fake_runtime: None) -> None:
    code, out, err = run_main(
        sweep_argv(
            tmp_path,
            "--concurrency",
            "1",
            "--slo-ttft-ms",
            "5000",
            "--temperature",
            "0.7",
        )
    )
    assert code == 0, err
    assert "concurrency=1: 3/3 succeeded" in out
    (runtime,) = FakeRuntime.instances
    assert runtime.temperature == 0.7
    assert runtime.created_on != threading.get_ident()
    text = (tmp_path / "out" / "sweep.json").read_text(encoding="utf-8")
    assert "Tell me a story" not in text
    sweep = json.loads(text)
    assert sweep["kind"] == SWEEP_KIND
    assert sweep["summary"]["clean"] is True
    backend = sweep["attribution"]["backend"]
    assert backend["max_supported_in_flight"] == 1
    assert "thread-local" in backend["max_supported_in_flight_reason"]
    assert backend["runtime"]["accelerator"] == "Fake M9"
    assert backend["runtime"]["model_load_seconds"] >= 0
    assert backend["upstream_reference"]["mlx_lm"]["commit"] == (
        "c4c9c762cc797fd979fc98645f5db5112753bbe9"
    )
    provenance = sweep["field_provenance"]
    assert provenance["output_tokens"] == "measured_native"
    assert "cached_prompt_tokens" not in provenance
    point = sweep["points"][0]
    assert point["latency"]["tpot_ms"]["count"] == 3
    assert point["goodput"]["request_goodput_fraction"]["value"] == 1.0
    assert point["requests"][0]["output_tokens"] == 3


def test_cli_refuses_concurrency_above_one(tmp_path: Path, fake_runtime: None) -> None:
    code, _, err = run_main(sweep_argv(tmp_path, "--concurrency", "1,2"))
    assert code == 1
    assert "in-process MLX supports 1" in err
    assert "thread-local" in err
    assert FakeRuntime.instances == []


def test_cli_rate_mode_needs_max_in_flight_one(
    tmp_path: Path, fake_runtime: None
) -> None:
    rate = ("--request-rate", "20", "--arrival", "constant")
    code, _, err = run_main(sweep_argv(tmp_path, *rate))
    assert code == 1
    assert "--max-in-flight 1" in err

    code, _, err = run_main(
        sweep_argv(
            tmp_path, *rate, "--max-in-flight", "1", "--max-dispatch-lag-ms", "100000"
        )
    )
    assert code == 0, err
    point = json.loads((tmp_path / "out" / "sweep.json").read_text())["points"][0]
    assert point["scheduling"]["max_observed_in_flight"] == 1


class BusyRuntime(FakeRuntime):
    """Spends CPU on the owner thread, as a real in-process model does."""

    def stream_generate(self, *args: Any, **kwargs: Any) -> Iterator[Response]:
        started = time.thread_time()
        while time.thread_time() - started < 0.1:
            pass
        yield from super().stream_generate(*args, **kwargs)


def test_cli_does_not_blame_model_cpu_on_the_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(backend_cli, "MLXLMRuntime", BusyRuntime)
    code, _, err = run_main(sweep_argv(tmp_path, "--concurrency", "1"))
    assert code == 0, err
    sweep = json.loads((tmp_path / "out" / "sweep.json").read_text())
    scheduling = sweep["points"][0]["scheduling"]
    assert scheduling["status"] == "ok"
    assert scheduling["client_cpu_seconds"] is None
    assert scheduling["client_cpu_utilization"] is None
    assert "load generator's process" in scheduling["client_cpu_withheld_reason"]
    assert sweep["summary"]["degraded_points"] == []


def test_cli_dry_run_loads_nothing(tmp_path: Path, monkeypatch) -> None:
    def refuse(**_: Any) -> None:
        raise AssertionError("dry run must not create the runtime")

    monkeypatch.setattr(backend_cli, "MLXLMRuntime", refuse)
    argv = sweep_argv(tmp_path, "--concurrency", "1", "--dry-run")
    model_index = argv.index("--model-path")
    del argv[model_index : model_index + 2]
    code, out, err = run_main(argv)
    assert code == 0, err
    assert "nothing sent" in out
    plan = json.loads((tmp_path / "out" / "sweep_plan.json").read_text())
    assert plan["kind"] == SWEEP_PLAN_KIND
    assert plan["model_loaded"] is False
    assert plan["attribution"]["backend"]["runtime"]["status"] == "not loaded"


def test_cli_needs_model_path_without_dry_run(tmp_path: Path) -> None:
    argv = sweep_argv(tmp_path, "--concurrency", "1")
    model_index = argv.index("--model-path")
    del argv[model_index : model_index + 2]
    code, _, err = run_main(argv)
    assert code == 1
    assert "--model-path is required" in err


def test_cli_reports_unsupported_platform(tmp_path: Path, monkeypatch) -> None:
    def unsupported(**_: Any) -> None:
        raise MLXCollectorError("MLX collection requires Apple Silicon running macOS")

    monkeypatch.setattr(backend_cli, "MLXLMRuntime", unsupported)
    code, _, err = run_main(sweep_argv(tmp_path, "--concurrency", "1"))
    assert code == 1
    assert "Failed to load the model" in err
    assert "Apple Silicon" in err
    assert not (tmp_path / "out" / "sweep.json").exists()
