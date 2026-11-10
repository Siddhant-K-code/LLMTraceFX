"""Tests for Prometheus server metrics snapshots and window attribution.

No test opens a socket: every scrape goes through an injected fetcher.
Metric names follow the vLLM v1 ``loggers.py``, SGLang ``metrics/collector.py``
and llama.cpp ``server.cpp`` exposition at the time of writing.
"""

from __future__ import annotations

import contextlib
import functools
import io
import json
import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer import cli
from llmtracefx.optimizer.collectors.openai_api import artifact_set_is_complete
from llmtracefx.optimizer.server_metrics import (
    ExpositionParseError,
    ServerMetricsError,
    ServerMetricsSnapshot,
    capture_settled_after,
    capture_snapshot,
    detect_runtime,
    diff_snapshots,
    parse_exposition,
    snapshot_from_body,
    validate_metrics_url,
)

URL = "http://127.0.0.1:8000/metrics"
MODEL_PATH = "/Users/someone/models/private-llama"


def vllm_body(
    *,
    requests: int,
    ttft_sum: float,
    generated: int,
    running: int = 0,
    waiting: int = 0,
    process_start: float = 1.7e9,
    created: float | None = 1.7e9,
) -> bytes:
    labels = f'engine="0",model_name="{MODEL_PATH}"'
    lines = [
        "# HELP process_start_time_seconds Start time of the process.",
        "# TYPE process_start_time_seconds gauge",
        f"process_start_time_seconds {process_start}",
        "# TYPE vllm:num_requests_running gauge",
        f"vllm:num_requests_running{{{labels}}} {running}",
        "# TYPE vllm:num_requests_waiting gauge",
        f"vllm:num_requests_waiting{{{labels}}} {waiting}",
        "# TYPE vllm:generation_tokens_total counter",
        f"vllm:generation_tokens_total{{{labels}}} {generated}",
    ]
    if created is not None:
        lines += [
            "# TYPE vllm:generation_tokens_created gauge",
            f"vllm:generation_tokens_created{{{labels}}} {created}",
        ]
    lines += [
        "# TYPE vllm:time_to_first_token_seconds histogram",
        f'vllm:time_to_first_token_seconds_bucket{{{labels},le="0.05"}} {requests}',
        f'vllm:time_to_first_token_seconds_bucket{{{labels},le="0.1"}} {requests}',
        f'vllm:time_to_first_token_seconds_bucket{{{labels},le="+Inf"}} {requests}',
        f"vllm:time_to_first_token_seconds_count{{{labels}}} {requests}",
        f"vllm:time_to_first_token_seconds_sum{{{labels}}} {ttft_sum}",
        "# TYPE vllm:e2e_request_latency_seconds histogram",
        f'vllm:e2e_request_latency_seconds_bucket{{{labels},le="1.0"}} {requests}',
        f'vllm:e2e_request_latency_seconds_bucket{{{labels},le="+Inf"}} {requests}',
        f"vllm:e2e_request_latency_seconds_count{{{labels}}} {requests}",
        f"vllm:e2e_request_latency_seconds_sum{{{labels}}} {ttft_sum * 10}",
    ]
    return ("\n".join(lines) + "\n").encode()


def llamacpp_body(*, prompt_tokens: int, prompt_seconds: float) -> bytes:
    return (
        "# TYPE llamacpp:prompt_tokens_total counter\n"
        f"llamacpp:prompt_tokens_total {prompt_tokens}\n"
        "# TYPE llamacpp:prompt_seconds_total counter\n"
        f"llamacpp:prompt_seconds_total {prompt_seconds}\n"
        "# TYPE llamacpp:requests_processing gauge\n"
        "llamacpp:requests_processing 0\n"
    ).encode()


def snap(body: bytes, captured_at: str = "2026-01-01T00:00:00.000000Z") -> Any:
    return snapshot_from_body(
        URL, body, "text/plain", captured_at=captured_at, scrape_duration_ms=1.0
    )


BEFORE = snap(vllm_body(requests=3, ttft_sum=0.09, generated=100))
ONE_MORE = snap(
    vllm_body(requests=4, ttft_sum=0.12, generated=150),
    "2026-01-01T00:00:01.500000Z",
)


# --- Exposition parser ------------------------------------------------------


def test_parser_handles_help_type_escapes_exemplars_and_eof() -> None:
    families = parse_exposition(
        "# HELP demo_total A \\\\ help line\n"
        "# TYPE demo_total counter\n"
        'demo_total{path="a\\"b\\\\c\\nd"} 3 # {trace_id="x"} 1.0\n'
        "# TYPE temp gauge\n"
        "temp NaN\n"
        "temp_other +Inf\n"
        "# EOF\n"
    )
    demo = families["demo"]
    assert demo.type == "counter"
    assert demo.samples[0].labels == (("path", 'a"b\\c\nd'),)
    assert demo.samples[0].value == 3.0
    assert math.isnan(families["temp"].samples[0].value)


@pytest.mark.parametrize(
    "text",
    [
        "a 1\na 1\n",
        "# TYPE a counter\n# TYPE a gauge\na 1\n",
        'a{b="c} 1\n',
        "a one\n",
        "# EOF\na 1\n",
    ],
    ids=[
        "duplicate-series",
        "duplicate-type",
        "unterminated-label",
        "bad-value",
        "after-eof",
    ],
)
def test_parser_refuses_malformed_exposition(text: str) -> None:
    with pytest.raises(ExpositionParseError):
        parse_exposition(text)


def test_parser_bounds_the_sample_count() -> None:
    with pytest.raises(ExpositionParseError):
        parse_exposition("".join(f"m{i} 1\n" for i in range(5)), max_samples=4)


# --- URL validation and snapshots -------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "ftp://host/metrics",
        "http://user:pw@host/metrics",
        "http://host/metrics?token=abc",
        "http://host/metrics#x",
        "http:///metrics",
    ],
)
def test_metrics_url_validation_refuses_unsafe_urls(url: str) -> None:
    with pytest.raises(ServerMetricsError):
        validate_metrics_url(url)


def test_metrics_url_validation_keeps_host_port_and_path() -> None:
    assert validate_metrics_url("http://LOCALHOST:8000/metrics") == (
        "http://localhost:8000/metrics"
    )


def test_snapshot_round_trips_through_json(tmp_path: Path) -> None:
    path = tmp_path / "before.json"
    BEFORE.write_json(path)
    restored = ServerMetricsSnapshot.read_json(path)
    assert restored == BEFORE
    assert json.loads(path.read_text())["kind"] == "llmtracefx.server_metrics.snapshot"


def test_capture_reports_fetch_and_parse_failures() -> None:
    def broken(url: str, timeout: float) -> tuple[bytes, str | None]:
        raise OSError("connection refused")

    with pytest.raises(ServerMetricsError, match="connection refused"):
        capture_snapshot(URL, fetcher=broken)
    with pytest.raises(ServerMetricsError, match="did not parse"):
        capture_snapshot(URL, fetcher=lambda url, timeout: (b"a one\n", None))


# --- Window attribution -----------------------------------------------------


def test_single_request_window_is_attributed_with_native_highlights() -> None:
    evidence = diff_snapshots(BEFORE, ONE_MORE, expected_requests=1)

    assert evidence["runtime"] == "vllm"
    assert evidence["attribution"]["status"] == "attributed"
    assert evidence["attribution"]["observed_requests"] == 1
    assert evidence["window"]["value"] == pytest.approx(1500.0)
    ttft = evidence["highlights"]["time_to_first_token"]
    assert ttft["value"] == pytest.approx(30.0)
    assert ttft["unit"] == "ms"
    assert ttft["provenance"] == "measured_native"
    assert ttft["statistic"] == "single_observation"
    assert evidence["highlights"]["generation_tokens"]["value"] == 50


def test_model_path_labels_never_reach_the_evidence() -> None:
    evidence = diff_snapshots(BEFORE, ONE_MORE, expected_requests=1)
    assert MODEL_PATH not in json.dumps(evidence)
    assert "model_name" not in evidence["kept_labels"]


@pytest.mark.parametrize(
    ("after", "status"),
    [
        (vllm_body(requests=5, ttft_sum=0.15, generated=150), "contaminated"),
        (vllm_body(requests=3, ttft_sum=0.09, generated=100), "incomplete"),
        (
            vllm_body(requests=4, ttft_sum=0.12, generated=150, running=1),
            "incomplete",
        ),
        (vllm_body(requests=1, ttft_sum=0.03, generated=20), "counter_reset"),
        (
            vllm_body(requests=4, ttft_sum=0.12, generated=150, process_start=1.8e9),
            "counter_reset",
        ),
    ],
    ids=["extra-request", "not-counted", "still-running", "decrease", "restart"],
)
def test_unattributable_windows_withhold_highlights(after: bytes, status: str) -> None:
    evidence = diff_snapshots(
        BEFORE, snap(after, "2026-01-01T00:00:01Z"), expected_requests=1
    )
    assert evidence["attribution"]["status"] == status
    assert evidence["attribution"]["attributable"] is False
    assert evidence["highlights"] == {}
    assert evidence["highlights_withheld_reason"]
    if status == "contaminated":
        # Raw deltas stay available for inspection when highlights are withheld.
        assert evidence["histograms"]


def test_traffic_in_flight_before_the_window_is_contamination() -> None:
    busy = snap(vllm_body(requests=3, ttft_sum=0.09, generated=100, waiting=2))
    evidence = diff_snapshots(busy, ONE_MORE, expected_requests=1)
    assert evidence["attribution"]["status"] == "contaminated"


def test_a_lazily_created_series_is_not_a_reset() -> None:
    before = snap(vllm_body(requests=3, ttft_sum=0.09, generated=100, created=None))
    evidence = diff_snapshots(before, ONE_MORE, expected_requests=1)
    assert evidence["attribution"]["status"] == "attributed"


def test_a_changed_created_timestamp_is_a_reset() -> None:
    after = snap(
        vllm_body(requests=4, ttft_sum=0.12, generated=150, created=1.75e9),
        "2026-01-01T00:00:01Z",
    )
    evidence = diff_snapshots(BEFORE, after, expected_requests=1)
    assert evidence["attribution"]["status"] == "counter_reset"


def test_llama_cpp_has_no_request_counter_so_attribution_needs_an_assertion() -> None:
    before = snap(llamacpp_body(prompt_tokens=10, prompt_seconds=0.5))
    after = snap(
        llamacpp_body(prompt_tokens=30, prompt_seconds=0.7), "2026-01-01T00:00:01Z"
    )

    unverified = diff_snapshots(before, after, expected_requests=1)
    assert unverified["runtime"] == "llamacpp"
    assert unverified["attribution"]["status"] == "unverifiable"
    assert unverified["highlights"] == {}

    asserted = diff_snapshots(before, after, expected_requests=1, assume_isolated=True)
    assert asserted["attribution"]["status"] == "operator_asserted"
    assert asserted["attribution"]["assume_isolated"] is True
    assert asserted["highlights"]["prompt_tokens"]["value"] == 20


def tgi_body(
    *,
    started: int,
    finished: int,
    succeeded: int,
    queued: int = 0,
    running: int = 0,
) -> bytes:
    # TGI counts a request when it starts and records its duration and
    # success when it finishes.
    return (
        "# TYPE tgi_request_count counter\n"
        f"tgi_request_count {started}\n"
        "# TYPE tgi_request_success counter\n"
        f"tgi_request_success {succeeded}\n"
        "# TYPE tgi_request_duration histogram\n"
        f'tgi_request_duration_bucket{{le="1.0"}} {finished}\n'
        f'tgi_request_duration_bucket{{le="+Inf"}} {finished}\n'
        f"tgi_request_duration_count {finished}\n"
        f"tgi_request_duration_sum {finished * 0.5}\n"
        "# TYPE tgi_queue_size gauge\n"
        f"tgi_queue_size {queued}\n"
        "# TYPE tgi_batch_current_size gauge\n"
        f"tgi_batch_current_size {running}\n"
    ).encode()


def test_tgi_request_already_running_at_the_before_scrape_is_contamination() -> None:
    # A foreign request is running (not queued) at the before scrape and
    # finishes in the window. tgi_request_count moves by exactly one, ours.
    before = snap(tgi_body(started=10, finished=8, succeeded=8, running=1))
    after = snap(
        tgi_body(started=11, finished=10, succeeded=10), "2026-01-01T00:00:01Z"
    )
    evidence = diff_snapshots(before, after, expected_requests=1)
    assert evidence["runtime"] == "tgi"
    assert evidence["attribution"]["observed_requests"] == 1
    assert evidence["attribution"]["in_flight_before"] == 1
    assert evidence["attribution"]["status"] == "contaminated"
    assert evidence["highlights"] == {}


def test_finish_time_families_must_agree_with_the_start_counter() -> None:
    # Same overlap, but with no gauge evidence of it: the duration histogram
    # and success counter still expose the extra finished request.
    before = snap(tgi_body(started=10, finished=9, succeeded=9))
    after = snap(
        tgi_body(started=11, finished=11, succeeded=11), "2026-01-01T00:00:01Z"
    )
    evidence = diff_snapshots(before, after, expected_requests=1)
    attribution = evidence["attribution"]
    assert attribution["status"] == "contaminated"
    assert "tgi_request_duration counted 2" in attribution["reason"]
    assert attribution["request_checks"] == [
        {"family": "tgi_request_count", "observed": 1},
        {"family": "tgi_request_duration", "observed": 2},
        {"family": "tgi_request_success", "observed": 2},
    ]
    assert evidence["highlights"] == {}


def test_tgi_window_is_attributed_when_every_request_family_agrees() -> None:
    before = snap(tgi_body(started=10, finished=10, succeeded=10))
    after = snap(
        tgi_body(started=11, finished=11, succeeded=11), "2026-01-01T00:00:01Z"
    )
    evidence = diff_snapshots(before, after, expected_requests=1)
    assert evidence["attribution"]["status"] == "attributed"
    assert evidence["highlights"]["e2e_latency"]["value"] == pytest.approx(500.0)
    assert evidence["highlights"]["successful_requests"]["value"] == 1


def test_a_lagging_success_counter_downgrades_a_vllm_window() -> None:
    def with_success(body: bytes, count: int) -> bytes:
        return (
            body
            + (
                "# TYPE vllm:request_success_total counter\n"
                f'vllm:request_success_total{{finished_reason="stop"}} {count}\n'
            ).encode()
        )

    before = snap(with_success(vllm_body(requests=3, ttft_sum=0.09, generated=100), 3))
    lagging = snap(
        with_success(vllm_body(requests=4, ttft_sum=0.12, generated=150), 3),
        "2026-01-01T00:00:01Z",
    )
    evidence = diff_snapshots(before, lagging, expected_requests=1)
    assert evidence["attribution"]["status"] == "incomplete"
    assert "vllm:request_success counted 0" in evidence["attribution"]["reason"]

    agreeing = snap(
        with_success(vllm_body(requests=4, ttft_sum=0.12, generated=150), 4),
        "2026-01-01T00:00:01Z",
    )
    assert (
        diff_snapshots(before, agreeing, expected_requests=1)["attribution"]["status"]
        == "attributed"
    )


def test_settled_capture_waits_for_finish_time_families() -> None:
    before = snap(tgi_body(started=10, finished=10, succeeded=10))
    bodies = iter(
        [
            tgi_body(started=11, finished=10, succeeded=10),
            tgi_body(started=11, finished=11, succeeded=11),
        ]
    )
    ticks = iter(float(i) for i in range(100))
    after = capture_settled_after(
        before,
        URL,
        expected_requests=1,
        settle_timeout_seconds=10.0,
        fetcher=lambda url, timeout: (next(bodies), None),
        clock=lambda: next(ticks),
        sleep=lambda seconds: None,
    )
    evidence = diff_snapshots(before, after, expected_requests=1)
    assert evidence["attribution"]["status"] == "attributed"


def test_runtime_detection_uses_metric_prefixes() -> None:
    assert detect_runtime(["sglang:num_requests", "process_cpu_seconds"]) == "sglang"
    assert detect_runtime(["vllm:num_requests_running"]) == "vllm"
    assert detect_runtime(["llamacpp:prompt_tokens"]) == "llamacpp"
    assert detect_runtime(["http_requests"]) == "generic"


def test_settled_capture_rescrapes_until_the_request_is_counted() -> None:
    bodies = iter(
        [
            vllm_body(requests=3, ttft_sum=0.09, generated=100, running=1),
            vllm_body(requests=4, ttft_sum=0.12, generated=150),
        ]
    )
    ticks = iter(float(i) for i in range(100))
    after = capture_settled_after(
        BEFORE,
        URL,
        expected_requests=1,
        settle_timeout_seconds=10.0,
        fetcher=lambda url, timeout: (next(bodies), None),
        clock=lambda: next(ticks),
        sleep=lambda seconds: None,
    )
    assert diff_snapshots(BEFORE, after, expected_requests=1)["attribution"][
        "status"
    ] == ("attributed")


# --- CLI ----------------------------------------------------------------------


def run_cli(argv: list[str]) -> tuple[int, str, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            args = cli.build_parser().parse_args(argv)
            code = int(args.func(args))
    except SystemExit as exc:
        code = int(exc.code or 0)
    return code, stdout.getvalue(), stderr.getvalue()


def test_diff_cli_exits_zero_when_attributed_and_two_when_not(
    tmp_path: Path,
) -> None:
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    BEFORE.write_json(before)
    ONE_MORE.write_json(after)
    output = tmp_path / "evidence.json"

    code, out, _ = run_cli(
        [
            "server-metrics",
            "diff",
            "--before",
            str(before),
            "--after",
            str(after),
            "--expected-requests",
            "1",
            "--output",
            str(output),
        ]
    )
    assert code == 0
    assert "Attribution: attributed" in out
    assert "time_to_first_token: 30 ms" in out
    assert json.loads(output.read_text())["attribution"]["status"] == "attributed"

    code, out, _ = run_cli(
        [
            "server-metrics",
            "diff",
            "--before",
            str(before),
            "--after",
            str(after),
            "--expected-requests",
            "2",
            "--output",
            str(output),
        ]
    )
    assert code == 2
    assert "Highlights withheld" in out
    assert json.loads(output.read_text())["attribution"]["status"] == "incomplete"


@pytest.mark.parametrize("count", ["0", "-1"])
def test_diff_cli_requires_at_least_one_expected_request(
    tmp_path: Path, count: str
) -> None:
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    BEFORE.write_json(before)
    ONE_MORE.write_json(after)
    code, _, err = run_cli(
        [
            "server-metrics",
            "diff",
            "--before",
            str(before),
            "--after",
            str(after),
            "--expected-requests",
            count,
        ]
    )
    assert code == 1
    assert "--expected-requests must be at least 1" in err


def test_snapshot_cli_refuses_a_credentialed_url_without_scraping(
    tmp_path: Path,
) -> None:
    code, _, err = run_cli(
        [
            "server-metrics",
            "snapshot",
            "--url",
            "http://user:secret@127.0.0.1:8000/metrics",
            "--output",
            str(tmp_path / "s.json"),
        ]
    )
    assert code == 1
    assert "credentials" in err
    assert not (tmp_path / "s.json").exists()


# --- collect-api --server-metrics-url ----------------------------------------

STREAM = [
    b'data: {"id":"c1","model":"m","choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n',
    b'data: {"id":"c1","choices":[{"index":0,"delta":{"content":""},'
    + b'"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,'
    + b'"total_tokens":6}}\n\n',
    b"data: [DONE]\n\n",
]


class _Response:
    status_code = 200
    headers: dict[str, str] = {}

    def iter_bytes(self) -> Iterator[bytes]:
        yield from STREAM

    def close(self) -> None:
        return None


class _Transport:
    def open_stream(self, request: Any) -> _Response:
        return _Response()


def _patch_scrapes(
    monkeypatch: pytest.MonkeyPatch, bodies: list[bytes]
) -> Callable[[], int]:
    remaining = iter(bodies)
    calls = {"n": 0}

    def fetcher(url: str, timeout: float) -> tuple[bytes, str | None]:
        calls["n"] += 1
        return next(remaining), "text/plain"

    monkeypatch.setattr(
        cli, "capture_snapshot", functools.partial(capture_snapshot, fetcher=fetcher)
    )
    monkeypatch.setattr(
        cli,
        "capture_settled_after",
        functools.partial(
            capture_settled_after, fetcher=fetcher, sleep=lambda seconds: None
        ),
    )
    return lambda: calls["n"]


def _collect_argv(tmp_path: Path, *extra: str) -> list[str]:
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("hi", encoding="utf-8")
    return [
        "collect-api",
        "--run-id",
        "sm-run",
        "--provider",
        "local-vllm",
        "--endpoint",
        "http://127.0.0.1:8000/v1/chat/completions",
        "--model-id",
        "m",
        "--prompt-file",
        str(prompt),
        "--output-dir",
        str(tmp_path / "artifacts"),
        "--api-key-env",
        "SM_TEST_KEY",
        *extra,
    ]


def test_collect_api_brackets_the_request_with_server_scrapes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SM_TEST_KEY", "not-a-real-key")
    monkeypatch.setattr(cli, "UrllibStreamingTransport", _Transport)
    scrapes = _patch_scrapes(
        monkeypatch,
        [
            vllm_body(requests=3, ttft_sum=0.09, generated=100),
            vllm_body(requests=4, ttft_sum=0.12, generated=101),
        ],
    )

    code, out, err = run_cli(_collect_argv(tmp_path, "--server-metrics-url", URL))

    assert code == 0, err
    assert scrapes() == 2
    assert "Attribution: attributed" in out
    artifacts = tmp_path / "artifacts"
    evidence = json.loads((artifacts / "server_metrics_evidence.json").read_text())
    assert evidence["attribution"]["expected_requests"] == 1
    assert evidence["highlights"]["generation_tokens"]["value"] == 1
    # The collector's own artifact set is untouched by the extra file.
    assert artifact_set_is_complete(artifacts)


def test_collect_api_removes_stale_server_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SM_TEST_KEY", "not-a-real-key")
    monkeypatch.setattr(cli, "UrllibStreamingTransport", _Transport)
    stale = tmp_path / "artifacts" / "server_metrics_evidence.json"
    stale.parent.mkdir()
    stale.write_text("{}", encoding="utf-8")

    code, _, err = run_cli(_collect_argv(tmp_path))

    assert code == 0, err
    assert not stale.exists()


def test_collect_api_fails_before_the_request_when_the_server_is_unreachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SM_TEST_KEY", "not-a-real-key")

    class Exploding:
        def open_stream(self, request: Any) -> Any:
            raise AssertionError("no request should be sent")

    def refused(url: str, timeout: float) -> tuple[bytes, str | None]:
        raise OSError("connection refused")

    monkeypatch.setattr(cli, "UrllibStreamingTransport", Exploding)
    monkeypatch.setattr(
        cli, "capture_snapshot", functools.partial(capture_snapshot, fetcher=refused)
    )

    code, _, err = run_cli(_collect_argv(tmp_path, "--server-metrics-url", URL))

    assert code == 1
    assert "before the request" in err


def test_collect_api_dry_run_validates_the_metrics_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    code, _, err = run_cli(
        _collect_argv(
            tmp_path, "--server-metrics-url", "http://h/metrics?x=1", "--dry-run"
        )
    )
    assert code == 1
    assert "query" in err
