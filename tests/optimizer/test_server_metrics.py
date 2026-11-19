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


def llamacpp_body(
    *, prompt_tokens: int, prompt_seconds: float, cached: int | None = None
) -> bytes:
    cached_lines = (
        ""
        if cached is None
        else (
            "# TYPE llamacpp:prompt_tokens_cached_total counter\n"
            f"llamacpp:prompt_tokens_cached_total {cached}\n"
        )
    )
    return (
        "# TYPE llamacpp:prompt_tokens_total counter\n"
        f"llamacpp:prompt_tokens_total {prompt_tokens}\n"
        f"{cached_lines}"
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
    before = snap(llamacpp_body(prompt_tokens=10, prompt_seconds=0.5, cached=0))
    after = snap(
        llamacpp_body(prompt_tokens=30, prompt_seconds=0.7, cached=0),
        "2026-01-01T00:00:01Z",
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


# --- Speculative decoding counters ------------------------------------------


def vllm_spec_lines(drafts: int, draft_tokens: int, accepted: int, per_pos) -> str:
    labels = f'engine="0",model_name="{MODEL_PATH}"'
    lines = []
    for name, value in (
        ("num_drafts", drafts),
        ("num_draft_tokens", draft_tokens),
        ("num_accepted_tokens", accepted),
    ):
        lines += [
            f"# TYPE vllm:spec_decode_{name} counter",
            f"vllm:spec_decode_{name}_total{{{labels}}} {value}",
        ]
    lines.append("# TYPE vllm:spec_decode_num_accepted_tokens_per_pos counter")
    lines += [
        f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{{labels},position="{i}"}}'
        f" {value}"
        for i, value in enumerate(per_pos)
    ]
    return "\n".join(lines) + "\n"


def test_vllm_acceptance_is_derived_from_counter_deltas() -> None:
    before = snap(
        vllm_body(requests=3, ttft_sum=0.09, generated=100)
        + vllm_spec_lines(10, 30, 15, [8, 5, 2]).encode()
    )
    after = snap(
        vllm_body(requests=4, ttft_sum=0.12, generated=150)
        + vllm_spec_lines(30, 90, 45, [24, 15, 6]).encode(),
        "2026-01-01T00:00:01Z",
    )
    spec = diff_snapshots(before, after, expected_requests=1)["speculative"]

    assert spec["status"] == "measured"
    assert spec["counters"]["verify_steps"]["value"] == 20
    assert spec["counters"]["verify_steps"]["provenance"] == "measured_native"
    rate, length = spec["acceptance_rate"], spec["acceptance_length"]
    assert rate["value"] == pytest.approx(0.5)
    assert rate["provenance"] == "derived"
    assert rate["basis"]
    assert length["value"] == pytest.approx(2.5)
    assert length["source"].startswith("1 + vllm:spec_decode_num_accepted_tokens")
    positions = spec["per_position_acceptance_rate"]
    assert [p["position"] for p in positions] == [0, 1, 2]
    assert [p["value"] for p in positions] == pytest.approx([0.8, 0.5, 0.2])


def test_vllm_without_speculative_counters_reports_not_exported() -> None:
    spec = diff_snapshots(BEFORE, ONE_MORE, expected_requests=1)["speculative"]

    assert spec["status"] == "not_exported"
    assert "acceptance_rate" not in spec and "counters" not in spec


def test_speculative_values_are_withheld_from_an_unattributed_window() -> None:
    before = snap(
        vllm_body(requests=3, ttft_sum=0.09, generated=100)
        + vllm_spec_lines(10, 30, 15, []).encode()
    )
    after = snap(
        vllm_body(requests=5, ttft_sum=0.15, generated=150)
        + vllm_spec_lines(30, 90, 45, []).encode(),
        "2026-01-01T00:00:01Z",
    )
    spec = diff_snapshots(before, after, expected_requests=1)["speculative"]

    assert spec["status"] == "withheld"
    assert "acceptance_length" not in spec


def sglang_body(*, requests: int, generated: int, verify_calls: int | None) -> bytes:
    lines = [
        "# TYPE sglang:num_requests counter",
        f"sglang:num_requests_total {requests}",
        "# TYPE sglang:num_running_reqs gauge",
        "sglang:num_running_reqs 0",
        "# TYPE sglang:num_queue_reqs gauge",
        "sglang:num_queue_reqs 0",
        "# TYPE sglang:generation_tokens counter",
        f'sglang:generation_tokens_total{{is_streaming="true"}} {generated}',
        # Last-value gauges: a diff of these would be meaningless.
        "# TYPE sglang:spec_accept_length gauge",
        "sglang:spec_accept_length 9.5",
        "# TYPE sglang:spec_accept_rate gauge",
        "sglang:spec_accept_rate 0.99",
    ]
    if verify_calls is not None:
        lines += [
            "# TYPE sglang:spec_verify_calls counter",
            f"sglang:spec_verify_calls_total {verify_calls}",
        ]
    return ("\n".join(lines) + "\n").encode()


def test_sglang_length_comes_from_counters_and_never_from_gauges() -> None:
    before = snap(sglang_body(requests=2, generated=100, verify_calls=40))
    after = snap(
        sglang_body(requests=3, generated=160, verify_calls=60),
        "2026-01-01T00:00:01Z",
    )
    spec = diff_snapshots(before, after, expected_requests=1)["speculative"]

    assert spec["status"] == "measured"
    assert spec["acceptance_length"]["value"] == pytest.approx(3.0)
    assert spec["acceptance_length"]["source"] == (
        "sglang:generation_tokens / sglang:spec_verify_calls"
    )
    assert "acceptance_rate" not in spec
    assert "acceptance_rate" in spec["not_derived"]
    assert spec["ignored_gauges"] == [
        "sglang:spec_accept_length",
        "sglang:spec_accept_rate",
    ]


def test_sglang_without_verify_calls_reports_not_exported() -> None:
    before = snap(sglang_body(requests=2, generated=100, verify_calls=None))
    after = snap(
        sglang_body(requests=3, generated=160, verify_calls=None),
        "2026-01-01T00:00:01Z",
    )
    spec = diff_snapshots(before, after, expected_requests=1)["speculative"]

    assert spec["status"] == "not_exported"
    assert "acceptance_length" not in spec


def llamacpp_spec_body(drafts: int, draft_tokens: int, accepted: int) -> bytes:
    lines = [
        "# TYPE llamacpp:spec_decode_num_draft_tokens_total counter",
        f"llamacpp:spec_decode_num_draft_tokens_total {draft_tokens}",
        "# TYPE llamacpp:spec_decode_num_accepted_tokens_total counter",
        f"llamacpp:spec_decode_num_accepted_tokens_total {accepted}",
        "# TYPE llamacpp:spec_decode_num_drafts_total counter",
        f"llamacpp:spec_decode_num_drafts_total {drafts}",
    ]
    return (
        llamacpp_body(prompt_tokens=10, prompt_seconds=0.5)
        + ("\n".join(lines) + "\n").encode()
    )


def test_llama_cpp_zero_speculative_counters_are_not_reported_as_acceptance() -> None:
    # llama.cpp exports these counters at zero when speculation is off.
    spec = diff_snapshots(
        snap(llamacpp_spec_body(0, 0, 0)),
        snap(llamacpp_spec_body(0, 0, 0), "2026-01-01T00:00:01Z"),
        expected_requests=1,
        assume_isolated=True,
    )["speculative"]

    assert spec["status"] == "no_drafts"
    assert "acceptance_rate" not in spec and "acceptance_length" not in spec


def test_llama_cpp_acceptance_uses_verification_steps() -> None:
    spec = diff_snapshots(
        snap(llamacpp_spec_body(4, 16, 4)),
        snap(llamacpp_spec_body(14, 56, 24), "2026-01-01T00:00:01Z"),
        expected_requests=1,
        assume_isolated=True,
    )["speculative"]

    assert spec["acceptance_rate"]["value"] == pytest.approx(0.5)
    assert spec["acceptance_length"]["value"] == pytest.approx(3.0)
    assert spec["not_derived"] == {
        "per_position_acceptance_rate": (
            "llamacpp:spec_decode_num_accepted_tokens_per_pos has no position series"
        )
    }


def test_runtime_detection_uses_metric_prefixes() -> None:
    assert detect_runtime(["sglang:num_requests", "process_cpu_seconds"]) == "sglang"
    assert detect_runtime(["vllm:num_requests_running"]) == "vllm"
    assert detect_runtime(["llamacpp:prompt_tokens"]) == "llamacpp"
    assert detect_runtime(["trtllm_request_success"]) == "trtllm"
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


# --- Cache tiers and KV usage -------------------------------------------------


def with_lines(body: bytes, *lines: str) -> bytes:
    return body + ("\n".join(lines) + "\n").encode()


def vllm_window(before_lines: list[str], after_lines: list[str]) -> dict[str, Any]:
    before = snap(
        with_lines(vllm_body(requests=3, ttft_sum=0.09, generated=100), *before_lines)
    )
    after = snap(
        with_lines(vllm_body(requests=4, ttft_sum=0.12, generated=150), *after_lines),
        "2026-01-01T00:00:01Z",
    )
    return diff_snapshots(before, after, expected_requests=1)


def vllm_cache_lines(
    *, prompt: int, cached: int, device: int, host: int, ext_q: int, ext_h: int
) -> list[str]:
    return [
        "# TYPE vllm:prompt_tokens_total counter",
        f'vllm:prompt_tokens_total{{engine="0"}} {prompt}',
        "# TYPE vllm:prompt_tokens_cached_total counter",
        f'vllm:prompt_tokens_cached_total{{engine="0"}} {cached}',
        "# TYPE vllm:prompt_tokens_cached_by_source_total counter",
        f'vllm:prompt_tokens_cached_by_source_total{{engine="0",source="device"}} '
        f"{device}",
        f'vllm:prompt_tokens_cached_by_source_total{{engine="0",source="host"}} '
        f"{host}",
        "# TYPE vllm:external_prefix_cache_queries_total counter",
        f'vllm:external_prefix_cache_queries_total{{engine="0"}} {ext_q}',
        "# TYPE vllm:external_prefix_cache_hits_total counter",
        f'vllm:external_prefix_cache_hits_total{{engine="0"}} {ext_h}',
        "# TYPE vllm:kv_cache_usage_perc gauge",
        f'vllm:kv_cache_usage_perc{{engine="0"}} {0.25 if prompt < 100 else 0.5}',
    ]


def test_vllm_reports_cached_prompt_tokens_by_tier_and_kv_usage() -> None:
    evidence = vllm_window(
        vllm_cache_lines(prompt=50, cached=10, device=10, host=0, ext_q=0, ext_h=0),
        vllm_cache_lines(prompt=150, cached=70, device=40, host=30, ext_q=40, ext_h=10),
    )
    highlights = evidence["highlights"]
    assert highlights["cached_prompt_tokens"]["value"] == 60
    assert highlights["cached_prompt_token_ratio"]["value"] == pytest.approx(0.6)
    assert highlights["cached_prompt_tokens_by_tier"]["breakdown"] == {
        "source": {"device": 30, "host": 30}
    }
    assert highlights["external_prefix_cache_hit_rate"]["value"] == pytest.approx(0.25)
    # Server state is a before and an after sample, never a delta, and is
    # reported even when highlights would be withheld.
    usage = evidence["kv_usage"]["kv_cache_usage"]
    assert usage["statistic"] == "point_sample"
    assert usage["before"]["value"] == 0.25
    assert usage["after"]["value"] == 0.5


def test_kv_usage_is_reported_for_an_unattributable_window() -> None:
    lines = ["# TYPE vllm:kv_cache_usage_perc gauge", "vllm:kv_cache_usage_perc 0.4"]
    before = snap(
        with_lines(vllm_body(requests=3, ttft_sum=0.09, generated=100), *lines)
    )
    after = snap(
        with_lines(vllm_body(requests=5, ttft_sum=0.15, generated=150), *lines),
        "2026-01-01T00:00:01Z",
    )
    evidence = diff_snapshots(before, after, expected_requests=1)
    assert evidence["attribution"]["status"] == "contaminated"
    assert evidence["highlights"] == {}
    usage = evidence["kv_usage"]["kv_cache_usage"]
    assert (
        usage["before"]
        == usage["after"]
        == {
            "series": 1,
            "min": 0.4,
            "max": 0.4,
            "value": 0.4,
        }
    )


def test_vllm_offload_reads_new_counters_and_falls_back_to_deprecated_ones() -> None:
    def offload(load: int, store: int, *, deprecated: bool) -> list[str]:
        if deprecated:
            return [
                "# TYPE vllm:kv_offload_load_bytes_total counter",
                "# TYPE vllm:kv_offload_total_bytes_total counter",
                f'vllm:kv_offload_total_bytes_total{{transfer_type="CPU_to_GPU"}} '
                f"{load}",
                f'vllm:kv_offload_total_bytes_total{{transfer_type="GPU_to_CPU"}} '
                f"{store}",
                "# TYPE vllm:kv_offload_total_time_total counter",
                'vllm:kv_offload_total_time_total{transfer_type="CPU_to_GPU"} 0.5',
                'vllm:kv_offload_total_time_total{transfer_type="GPU_to_CPU"} 0.5',
            ]
        return [
            "# TYPE vllm:kv_offload_load_bytes_total counter",
            f"vllm:kv_offload_load_bytes_total {load}",
            "# TYPE vllm:kv_offload_store_bytes_total counter",
            f"vllm:kv_offload_store_bytes_total {store}",
        ]

    new = vllm_window(
        offload(100, 200, deprecated=False), offload(400, 300, deprecated=False)
    )["highlights"]
    assert new["kv_offload_load_bytes"]["value"] == 300
    assert new["kv_offload_store_bytes"]["value"] == 100
    assert new["kv_offload_load_bytes"]["source"] == "vllm:kv_offload_load_bytes"

    # The new family is declared with no samples; the deprecated totals carry
    # the transfers, one direction per transfer_type.
    old = vllm_window(
        offload(100, 200, deprecated=True), offload(400, 300, deprecated=True)
    )["highlights"]
    assert old["kv_offload_load_bytes"]["value"] == 300
    assert old["kv_offload_load_bytes"]["source"] == "vllm:kv_offload_total_bytes"
    assert old["kv_offload_load_bytes"]["match"] == {"transfer_type": "CPU_to_GPU"}
    assert old["kv_offload_store_bytes"]["value"] == 100
    assert old["kv_offload_load_time"]["value"] == 0


def sglang_body(*, requests: int, extra: list[str]) -> bytes:
    labels = 'tp_rank="0",pp_rank="0",dp_rank="0"'
    return with_lines(
        b"",
        "# TYPE sglang:num_requests_total counter",
        f"sglang:num_requests_total{{{labels}}} {requests}",
        "# TYPE sglang:num_running_reqs gauge",
        f"sglang:num_running_reqs{{{labels}}} 0",
        "# TYPE sglang:num_queue_reqs gauge",
        f"sglang:num_queue_reqs{{{labels}}} 0",
        *extra,
    )


def sglang_cache_lines(
    *, device: int, host: int, load_back: tuple[int, int], prefetched: int
) -> list[str]:
    def ranked(name: str, values: tuple[int, int], extra: str = "") -> list[str]:
        return [
            f'{name}{{tp_rank="{rank}",pp_rank="0",dp_rank="0"{extra}}} {value}'
            for rank, value in enumerate(values)
        ]

    return [
        "# TYPE sglang:cached_tokens_total counter",
        f'sglang:cached_tokens_total{{tp_rank="0",cache_source="device"}} {device}',
        f'sglang:cached_tokens_total{{tp_rank="0",cache_source="host"}} {host}',
        "# TYPE sglang:load_back_tokens_total counter",
        *ranked("sglang:load_back_tokens_total", load_back, ',pool="kv"'),
        *ranked("sglang:load_back_tokens_total", load_back, ',pool="mamba"'),
        "# TYPE sglang:prefetched_tokens_total counter",
        *ranked("sglang:prefetched_tokens_total", (prefetched, prefetched)),
        "# TYPE sglang:token_usage gauge",
        'sglang:token_usage{tp_rank="0"} 0.1',
        'sglang:token_usage{tp_rank="1"} 0.3',
    ]


def test_sglang_splits_cached_tokens_and_reads_hicache_from_one_rank() -> None:
    before = snap(
        sglang_body(
            requests=3,
            extra=sglang_cache_lines(device=10, host=0, load_back=(5, 5), prefetched=0),
        )
    )
    after = snap(
        sglang_body(
            requests=4,
            extra=sglang_cache_lines(
                device=30, host=16, load_back=(21, 21), prefetched=8
            ),
        ),
        "2026-01-01T00:00:01Z",
    )
    evidence = diff_snapshots(before, after, expected_requests=1)
    assert evidence["attribution"]["status"] == "attributed"
    highlights = evidence["highlights"]
    assert highlights["cached_prompt_tokens_by_tier"]["breakdown"] == {
        "cache_source": {"device": 20, "host": 16}
    }
    # Two TP ranks report the same 16 tokens; summing them would claim 32,
    # and the mamba pool repeats the same tokens again.
    load_back = highlights["hicache_load_back_tokens"]
    assert load_back["value"] == 16
    assert load_back["statistic"] == "delta_one_rank"
    assert load_back["ranks_agree"] is True
    assert highlights["hicache_storage_prefetched_tokens"]["value"] == 8
    token_usage = evidence["kv_usage"]["token_usage"]["after"]
    assert token_usage == {"series": 2, "min": 0.1, "max": 0.3}


def test_disagreeing_ranks_are_flagged() -> None:
    before = snap(
        sglang_body(
            requests=3,
            extra=sglang_cache_lines(device=0, host=0, load_back=(0, 0), prefetched=0),
        )
    )
    after = snap(
        sglang_body(
            requests=4,
            extra=sglang_cache_lines(device=0, host=0, load_back=(4, 6), prefetched=0),
        ),
        "2026-01-01T00:00:01Z",
    )
    load_back = diff_snapshots(before, after, expected_requests=1)["highlights"][
        "hicache_load_back_tokens"
    ]
    assert load_back["value"] == 4
    assert load_back["ranks_agree"] is False


def test_llama_cpp_prompt_tokens_include_cached_tokens() -> None:
    before = snap(llamacpp_body(prompt_tokens=10, prompt_seconds=0.5, cached=0))
    after = snap(
        llamacpp_body(prompt_tokens=30, prompt_seconds=0.7, cached=60),
        "2026-01-01T00:00:01Z",
    )
    highlights = diff_snapshots(
        before, after, expected_requests=1, assume_isolated=True
    )["highlights"]
    assert highlights["prompt_tokens"]["value"] == 80
    assert highlights["computed_prompt_tokens"]["value"] == 20
    assert highlights["cached_prompt_tokens"]["value"] == 60
    assert highlights["cached_prompt_token_ratio"]["value"] == pytest.approx(0.75)
    assert highlights["prompt_tokens_per_second"]["value"] == pytest.approx(100.0)


def test_llama_cpp_without_the_cached_counter_does_not_call_it_prompt_tokens() -> None:
    # Builds before the cached counter split still count only computed tokens.
    before = snap(llamacpp_body(prompt_tokens=10, prompt_seconds=0.5))
    after = snap(
        llamacpp_body(prompt_tokens=30, prompt_seconds=0.7), "2026-01-01T00:00:01Z"
    )
    highlights = diff_snapshots(
        before, after, expected_requests=1, assume_isolated=True
    )["highlights"]
    assert "prompt_tokens" not in highlights
    assert highlights["computed_prompt_tokens"]["value"] == 20


# --- TensorRT-LLM -------------------------------------------------------------


def trtllm_body(
    *,
    succeeded: int | None,
    running: int | None = 0,
    gpu_hits: int = 0,
    host_hits: int = 0,
    utilization: float = 0.0,
) -> bytes:
    # trtllm-serve runs prometheus_client in multiprocess mode, so a family
    # no process has touched is absent rather than declared.
    lines = [
        "# TYPE trtllm_prompt_cached_tokens_total counter",
        f'trtllm_prompt_cached_tokens_total{{model_name="m"}} {gpu_hits + host_hits}',
        "# TYPE trtllm_prompt_cache_hit_tokens_total counter",
        'trtllm_prompt_cache_hit_tokens_total{model_name="m",dp_rank="0",'
        f'cache_level="0",cache_tier="gpu"}} {gpu_hits}',
        'trtllm_prompt_cache_hit_tokens_total{model_name="m",dp_rank="0",'
        f'cache_level="1",cache_tier="host"}} {host_hits}',
        "# TYPE trtllm_kv_cache_utilization gauge",
        f'trtllm_kv_cache_utilization{{model_name="m",dp_rank="0",pid="7"}} '
        f"{utilization}",
    ]
    if succeeded is not None:
        lines += [
            "# TYPE trtllm_request_success_total counter",
            f'trtllm_request_success_total{{model_name="m",finished_reason="stop"}} '
            f"{succeeded}",
            "# TYPE trtllm_e2e_request_latency_seconds histogram",
            f'trtllm_e2e_request_latency_seconds_bucket{{le="+Inf"}} {succeeded}',
            f"trtllm_e2e_request_latency_seconds_count {succeeded}",
            f"trtllm_e2e_request_latency_seconds_sum {succeeded * 0.4}",
        ]
    if running is not None:
        lines += [
            "# TYPE trtllm_num_requests_running gauge",
            f'trtllm_num_requests_running{{model_name="m",pid="7"}} {running}',
            "# TYPE trtllm_num_requests_waiting gauge",
            'trtllm_num_requests_waiting{model_name="m",pid="7"} 0',
        ]
    return with_lines(b"", *lines)


TRT_URL = "http://127.0.0.1:8000/prometheus/metrics"


def trt_snap(body: bytes, captured_at: str = "2026-01-01T00:00:00Z") -> Any:
    return snapshot_from_body(
        TRT_URL, body, "text/plain", captured_at=captured_at, scrape_duration_ms=1.0
    )


def test_trtllm_window_is_attributed_with_cache_tiers() -> None:
    before = trt_snap(trtllm_body(succeeded=2, gpu_hits=100, utilization=0.1))
    after = trt_snap(
        trtllm_body(succeeded=3, gpu_hits=164, host_hits=32, utilization=0.3),
        "2026-01-01T00:00:01Z",
    )
    evidence = diff_snapshots(before, after, expected_requests=1)
    assert evidence["runtime"] == "trtllm"
    attribution = evidence["attribution"]
    assert attribution["status"] == "attributed"
    assert attribution["request_counter"] == "trtllm_request_success"
    assert attribution["missing_stats"] == []
    highlights = evidence["highlights"]
    assert highlights["e2e_latency"]["value"] == pytest.approx(400.0)
    assert highlights["cached_prompt_tokens"]["value"] == 96
    assert highlights["cached_prompt_tokens_by_tier"]["breakdown"] == {
        "cache_tier": {"gpu": 64, "host": 32}
    }
    usage = evidence["kv_usage"]["kv_cache_utilization"]
    assert (usage["before"]["value"], usage["after"]["value"]) == (0.1, 0.3)


def test_trtllm_without_perf_metrics_says_stats_are_not_enabled() -> None:
    before = trt_snap(trtllm_body(succeeded=None))
    after = trt_snap(trtllm_body(succeeded=None), "2026-01-01T00:00:01Z")
    for assume_isolated in (False, True):
        evidence = diff_snapshots(
            before, after, expected_requests=1, assume_isolated=assume_isolated
        )
        attribution = evidence["attribution"]
        assert attribution["status"] == "stats_not_enabled"
        assert attribution["attributable"] is False
        assert "return_perf_metrics: true" in attribution["reason"]
        assert "/prometheus/metrics" in attribution["reason"]
        assert evidence["highlights"] == {}


def test_trtllm_without_iteration_stats_needs_an_assertion() -> None:
    before = trt_snap(trtllm_body(succeeded=2, running=None))
    after = trt_snap(trtllm_body(succeeded=3, running=None), "2026-01-01T00:00:01Z")

    unchecked = diff_snapshots(before, after, expected_requests=1)
    attribution = unchecked["attribution"]
    assert attribution["status"] == "stats_not_enabled"
    assert "enable_iter_perf_stats: true" in attribution["reason"]
    assert attribution["missing_stats"] == [
        {
            "setting": "enable_iter_perf_stats: true",
            "families": ["trtllm_num_requests_running", "trtllm_num_requests_waiting"],
            "required": False,
        }
    ]

    asserted = diff_snapshots(before, after, expected_requests=1, assume_isolated=True)
    attribution = asserted["attribution"]
    assert attribution["status"] == "operator_asserted"
    assert attribution["in_flight_before"] is None
    assert "operator asserted" in attribution["reason"]
    assert asserted["highlights"]["e2e_latency"]["value"] == pytest.approx(400.0)

    extra = trt_snap(trtllm_body(succeeded=4, running=None), "2026-01-01T00:00:01Z")
    assert (
        diff_snapshots(before, extra, expected_requests=1, assume_isolated=True)[
            "attribution"
        ]["status"]
        == "contaminated"
    )


def test_settled_capture_waits_for_the_trtllm_request_counter() -> None:
    # On a fresh server the request family only appears once a request has
    # finished, so its absence alone must not end the wait.
    before = trt_snap(trtllm_body(succeeded=None))
    bodies = iter([trtllm_body(succeeded=None), trtllm_body(succeeded=1)])
    ticks = iter(float(i) for i in range(100))
    after = capture_settled_after(
        before,
        TRT_URL,
        expected_requests=1,
        settle_timeout_seconds=10.0,
        fetcher=lambda url, timeout: (next(bodies), None),
        clock=lambda: next(ticks),
        sleep=lambda seconds: None,
    )
    evidence = diff_snapshots(before, after, expected_requests=1)
    assert evidence["attribution"]["status"] == "attributed"


def test_a_json_metrics_body_points_at_the_prometheus_endpoint() -> None:
    with pytest.raises(ServerMetricsError, match="/prometheus/metrics"):
        snap(b"[]")
    with pytest.raises(ServerMetricsError, match="JSON"):
        snap(b'[{"iter": 1, "numActiveRequests": 0}]')


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


def test_diff_cli_prints_tier_breakdowns_and_kv_usage(tmp_path: Path) -> None:
    before_path, after_path = tmp_path / "before.json", tmp_path / "after.json"
    snap(
        with_lines(
            vllm_body(requests=3, ttft_sum=0.09, generated=100),
            *vllm_cache_lines(
                prompt=50, cached=10, device=10, host=0, ext_q=0, ext_h=0
            ),
        )
    ).write_json(before_path)
    snap(
        with_lines(
            vllm_body(requests=4, ttft_sum=0.12, generated=150),
            *vllm_cache_lines(
                prompt=150, cached=70, device=40, host=30, ext_q=40, ext_h=10
            ),
        ),
        "2026-01-01T00:00:01Z",
    ).write_json(after_path)

    code, out, _ = run_cli(
        [
            "server-metrics",
            "diff",
            "--before",
            str(before_path),
            "--after",
            str(after_path),
            "--expected-requests",
            "1",
        ]
    )
    assert code == 0
    assert "by source: device=30, host=30" in out
    assert "KV usage (point samples at each scrape, not attributed):" in out
    assert "kv_cache_usage: 0.25 -> 0.5" in out


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
