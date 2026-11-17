"""The sweep-api command against a local fake server. Nothing leaves the machine."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest
from _load_fakes import API_KEY, API_KEY_ENV, FakeOpenAIServer, fake_openai_server

from llmtracefx.optimizer import cli
from llmtracefx.optimizer.compare.evidence import (
    CompareEvidenceError,
    load_comparison_evidence,
)
from llmtracefx.optimizer.load.artifact import SWEEP_KIND, SWEEP_PLAN_KIND
from llmtracefx.optimizer.load.cli import is_loopback_endpoint

VLLM_METRICS = (
    "# TYPE vllm:request_success_total counter\n"
    'vllm:request_success_total{finished_reason="stop"} @COMPLETED@\n'
    "# TYPE vllm:num_requests_running gauge\n"
    "vllm:num_requests_running 0\n"
)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Any:
    monkeypatch.setenv(API_KEY_ENV, API_KEY)
    with fake_openai_server() as fake:
        yield fake


def run_main(argv: list[str]) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            cli.main(argv)
    except SystemExit as exit_error:
        code = int(exit_error.code or 0)
    return code, stdout.getvalue(), stderr.getvalue()


def sweep_argv(
    tmp_path: Path, server: FakeOpenAIServer, *extra: str, prompt: str = "Hi"
) -> list[str]:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    return [
        "sweep-api",
        "--run-id",
        "load-run-7",
        "--endpoint",
        server.endpoint,
        "--model-id",
        "fake-model",
        "--api-key-env",
        API_KEY_ENV,
        "--output-dir",
        str(tmp_path / "out"),
        *(() if "--prompts-jsonl" in extra else ("--prompt-file", str(prompt_file))),
        *extra,
    ]


def read_sweep(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "out" / "sweep.json").read_text(encoding="utf-8"))


def test_closed_loop_sweep_reaches_each_concurrency_level(
    tmp_path: Path, server: FakeOpenAIServer
) -> None:
    server.behaviour.chunk_delay_s = 0.03
    code, out, err = run_main(
        sweep_argv(
            tmp_path,
            server,
            "--concurrency",
            "1,4",
            "--requests-per-point",
            "8",
            "--stream-include-usage",
            "--slo-ttft-ms",
            "5000",
            "--slo-tpot-ms",
            "5000",
            "--slo-e2e-ms",
            "10000",
            "--max-dispatch-lag-ms",
            "1000",
        )
    )
    assert code == 0, err
    assert "concurrency=4: 8/8 succeeded" in out
    assert server.peak_active == 4
    assert len(server.bodies) == 16
    assert all(
        body["stream_options"] == {"include_usage": True} for body in server.bodies
    )

    sweep = read_sweep(tmp_path)
    assert sweep["kind"] == SWEEP_KIND
    assert sweep["attribution"]["requests_isolated"] is False
    assert sweep["summary"]["clean"] is True
    assert [point["label"] for point in sweep["points"]] == [
        "concurrency=1",
        "concurrency=4",
    ]
    point = sweep["points"][1]
    assert point["load_context"]["measurement_mode"] == "concurrent_load"
    assert point["scheduling"]["max_observed_in_flight"] == 4
    assert point["counts"]["succeeded"] == 8
    assert point["latency"]["ttft_ms"]["count"] == 8
    assert point["latency"]["tpot_ms"]["count"] == 8
    goodput = point["goodput"]
    assert goodput["request_goodput_fraction"]["value"] == 1.0
    assert goodput["token_goodput_fraction"]["value"] == 1.0
    row = point["requests"][0]
    assert row["output_tokens"] == 3
    assert row["cached_prompt_tokens"] == 4
    assert row["http_status"] == 200
    assert row["tpot_status"] == "derived"
    # No single-request artifacts that a reader could mistake for isolated evidence.
    names = {path.name for path in (tmp_path / "out").rglob("*")}
    assert names == {"sweep.json"}
    assert API_KEY not in (tmp_path / "out" / "sweep.json").read_text(encoding="utf-8")


def test_constant_rate_sweep_records_target_and_achieved_rate(
    tmp_path: Path, server: FakeOpenAIServer
) -> None:
    code, _, err = run_main(
        sweep_argv(
            tmp_path,
            server,
            "--request-rate",
            "40",
            "--arrival",
            "constant",
            "--requests-per-point",
            "5",
            "--max-dispatch-lag-ms",
            "1000",
        )
    )
    assert code == 0, err
    point = read_sweep(tmp_path)["points"][0]
    assert point["policy"]["kind"] == "open_loop_rate"
    assert point["policy"]["distribution"] == "constant"
    rate = point["scheduling"]["dispatch_rate"]
    assert rate["target_rate_per_second"] == 40.0
    assert rate["planned_schedule_rate_per_second"]["value"] == pytest.approx(40.0)
    # Wall clock: slow CI runners fall behind schedule, so only bound it.
    assert 0 < rate["achieved_dispatch_rate_per_second"]["value"] < 80.0
    # The fake server reports usage without being asked; no SLO, no goodput.
    assert point["goodput"] is None


def test_failed_requests_count_against_goodput_and_exit_nonzero(
    tmp_path: Path, server: FakeOpenAIServer
) -> None:
    server.behaviour.fail_every = 3
    server.behaviour.metrics_text = VLLM_METRICS
    code, _, err = run_main(
        sweep_argv(
            tmp_path,
            server,
            "--concurrency",
            "1",
            "--requests-per-point",
            "6",
            "--slo-e2e-ms",
            "10000",
            "--server-metrics-url",
            server.metrics_url,
        )
    )
    assert code == 1
    assert "Points with failed requests: concurrency=1" in err
    point = read_sweep(tmp_path)["points"][0]
    assert point["status"] == "failures"
    assert point["counts"]["failure_categories"] == {"http_status": 2}
    assert point["goodput"]["counts"]["failed"] == 2
    assert point["goodput"]["request_goodput_fraction"]["value"] == pytest.approx(4 / 6)
    failed = [row for row in point["requests"] if row["success"] is False]
    assert {row["http_status"] for row in failed} == {503}
    # The server counted the failed requests too, so the window cannot be
    # attributed to the four that succeeded.
    metrics = point["server_metrics"]
    assert metrics["attribution_status"] == "contaminated"
    assert metrics["expected_requests"] == 4


def test_server_metrics_window_is_written_and_hashed(
    tmp_path: Path, server: FakeOpenAIServer
) -> None:
    server.behaviour.metrics_text = VLLM_METRICS
    code, out, err = run_main(
        sweep_argv(
            tmp_path,
            server,
            "--concurrency",
            "2",
            "--requests-per-point",
            "4",
            "--server-metrics-url",
            server.metrics_url,
        )
    )
    assert code == 0, err
    metrics = read_sweep(tmp_path)["points"][0]["server_metrics"]
    assert metrics["attribution_status"] == "attributed"
    evidence = tmp_path / "out" / metrics["file"]
    assert metrics["file"] == "server_metrics_point_0.json"
    assert metrics["sha256"] == hashlib.sha256(evidence.read_bytes()).hexdigest()
    assert "server metrics attributed" in out


def test_dry_run_writes_a_plan_and_sends_nothing(
    tmp_path: Path, server: FakeOpenAIServer
) -> None:
    code, out, err = run_main(
        sweep_argv(
            tmp_path,
            server,
            "--request-rate",
            "2,4",
            "--arrival-seed",
            "7",
            "--requests-per-point",
            "3",
            "--dry-run",
        )
    )
    assert code == 0, err
    assert "nothing sent" in out
    assert server.bodies == []
    plan = json.loads((tmp_path / "out" / "sweep_plan.json").read_text())
    assert plan["kind"] == SWEEP_PLAN_KIND
    assert plan["network_request_performed"] is False
    assert plan["total_requests"] == 6
    assert plan["credential_env_var_present"] is True
    assert [len(point["planned_offsets_s"]) for point in plan["points"]] == [3, 3]
    assert plan["points"][0]["planned_offsets_s"][0] == 0.0
    assert plan["request_plan"]["messages"][0]["content_sha256"]
    assert not (tmp_path / "out" / "sweep.json").exists()


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--request-rate", "1"], "--arrival-seed"),
        (["--concurrency", "2", "--arrival-seed", "3"], "only applies"),
        (["--concurrency", "0"], "concurrency"),
        (["--concurrency", "2", "--slo-ttft-ms", "0"], "ttft_ms"),
    ],
)
def test_invalid_load_settings_are_refused_before_sending(
    tmp_path: Path, server: FakeOpenAIServer, extra: list[str], message: str
) -> None:
    code, _, err = run_main(
        sweep_argv(tmp_path, server, *extra, "--requests-per-point", "2")
    )
    assert code == 1
    assert message in err
    assert server.bodies == []


def test_remote_endpoint_needs_explicit_opt_in(
    tmp_path: Path, server: FakeOpenAIServer
) -> None:
    argv = sweep_argv(
        tmp_path, server, "--concurrency", "1", "--requests-per-point", "1"
    )
    argv[argv.index(server.endpoint)] = "https://api.example.com/v1/chat/completions"
    code, _, err = run_main(argv)
    assert code == 1
    assert "--allow-remote-load" in err
    code, out, err = run_main(argv + ["--allow-remote-load", "--dry-run"])
    assert code == 0, err


def test_existing_sweep_output_is_refused(
    tmp_path: Path, server: FakeOpenAIServer
) -> None:
    argv = sweep_argv(
        tmp_path, server, "--concurrency", "1", "--requests-per-point", "1"
    )
    assert run_main(argv)[0] == 0
    code, _, err = run_main(argv)
    assert code == 1
    assert "already holds sweep output" in err
    assert len(server.bodies) == 1


def test_prompts_jsonl_cycles_in_order(
    tmp_path: Path, server: FakeOpenAIServer
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text(
        '{"prompt": "zebra alpha"}\n\n{"prompt": "zebra beta"}\n', encoding="utf-8"
    )
    code, _, err = run_main(
        sweep_argv(
            tmp_path,
            server,
            "--prompts-jsonl",
            str(prompts),
            "--concurrency",
            "1",
            "--requests-per-point",
            "3",
        )
    )
    assert code == 0, err
    sent = [body["messages"][-1]["content"] for body in server.bodies]
    assert sent == ["zebra alpha", "zebra beta", "zebra alpha"]
    sweep = read_sweep(tmp_path)
    assert len(sweep["workload"]["distinct_prompt_sha256"]) == 2
    assert "zebra" not in json.dumps(sweep)


def test_prompts_jsonl_with_unknown_keys_is_refused(
    tmp_path: Path, server: FakeOpenAIServer
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text('{"prompt": "a", "max_tokens": 5}\n', encoding="utf-8")
    code, _, err = run_main(
        sweep_argv(
            tmp_path,
            server,
            "--prompts-jsonl",
            str(prompts),
            "--concurrency",
            "1",
            "--requests-per-point",
            "1",
        )
    )
    assert code == 1
    assert 'only a "prompt" key' in err
    assert server.bodies == []


@pytest.mark.parametrize(
    ("endpoint", "loopback"),
    [
        ("http://127.0.0.1:8000/v1/chat/completions", True),
        ("http://127.5.0.1/v1", True),
        ("http://[::1]:8000/v1", True),
        ("http://localhost:8000/v1", True),
        ("http://api.localhost/v1", True),
        ("https://api.openai.com/v1/chat/completions", False),
        ("http://10.0.0.5:8000/v1", False),
        ("http://localhost.example.com/v1", False),
    ],
)
def test_is_loopback_endpoint(endpoint: str, loopback: bool) -> None:
    assert is_loopback_endpoint(endpoint) is loopback


def test_gate_compare_and_export_refuse_sweep_artifacts(
    tmp_path: Path, server: FakeOpenAIServer
) -> None:
    argv = sweep_argv(
        tmp_path, server, "--concurrency", "1", "--requests-per-point", "1"
    )
    assert run_main(argv)[0] == 0
    out_dir = tmp_path / "out"
    sweep_file = out_dir / "sweep.json"
    renamed = tmp_path / "renamed.json"
    renamed.write_bytes(sweep_file.read_bytes())

    for target in (out_dir, sweep_file, renamed, tmp_path):
        code, _, err = run_main(
            ["gate", "--baseline", str(target), "--candidate", str(target)]
        )
        assert code == 2
        assert "was refused: it is a concurrent load sweep" in err

    with pytest.raises(CompareEvidenceError, match="concurrent load sweep"):
        load_comparison_evidence((tmp_path,))

    code, _, err = run_main(
        ["export", "otlp", "--record", str(renamed), "--output", str(tmp_path / "t")]
    )
    assert code == 1
    assert "concurrent load sweep" in err
    assert not (tmp_path / "t").exists()

    code, _, err = run_main(
        ["export", "otlp-metrics", str(out_dir), "--output", str(tmp_path / "m")]
    )
    assert code == 1
    assert "concurrent load sweep" in err
    assert not (tmp_path / "m").exists()
