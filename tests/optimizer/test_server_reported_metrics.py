"""Per-request numbers a server reports about itself in the response.

Covers vLLM ``metrics``, SGLang ``sglext`` and TensorRT-LLM's
``trtllm.perf_metrics`` event through the API collector, the load worker
rows, the sweep point summary and the ``collect-api`` and ``sweep-api``
command lines. Streams come from an in-process fake transport or a local
fake server; no request leaves the machine.
"""

from __future__ import annotations

import contextlib
import io
import json
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from _load_fakes import API_KEY, API_KEY_ENV, ENVIRON, fake_openai_server

from llmtracefx.optimizer import cli
from llmtracefx.optimizer.collectors.openai_api import (
    APICollectionConfig,
    HTTPRequest,
    OpenAIStreamCollectorError,
    TransportConnectionError,
    build_request_plan,
    stream_chat_completion,
)
from llmtracefx.optimizer.load.api_worker import (
    ROW_FIELD_PROVENANCE,
    SERVER_REPORTED_ROW_FIELDS,
    RequestRow,
)
from llmtracefx.optimizer.load.mlx_worker import MLX_FIELD_PROVENANCE
from llmtracefx.optimizer.load.sweep import server_reported_summary

ENDPOINT = "http://127.0.0.1:9/v1/chat/completions"
TRT_HEADER = "X-TRTLLM-return-metrics"

# vLLM ``PerRequestMetrics`` and ``SpeculativeDecodingMetrics``
# (vllm/entrypoints/generate/base/protocol.py L105-138 at 1ae1b64).
VLLM_METRICS = {
    "time_to_first_token_ms": 41.5,
    "queue_time_ms": 7.25,
    "generation_time_ms": 120.0,
    "mean_itl_ms": 12.0,
    "tokens_per_second": 61.0,
    "speculative_decoding": {
        "mean_acceptance_length": 2.5,
        "draft_acceptance_rate": 0.75,
        "acceptance_histogram": [1, 2, 1],
        "num_spec_steps": 4,
        "num_accepted_draft_tokens": 6,
        "num_draft_tokens": 8,
    },
}

# The ``trtllm.perf_metrics`` event data: header name to header value
# (tensorrt_llm/serve/perf_metrics.py L322-399 and L842-910 at e6ecd28).
TRT_EVENT = (
    b"event: trtllm.perf_metrics\n"
    + b"data: "
    + json.dumps(
        {
            "Server-Timing": (
                "server_queue;dur=3.5, server_ttft;dur=50.25, server_e2e;dur=180, "
                "server_input_tokens;dur=11"
            ),
            "X-TRTLLM-Start-End-Time": "server-start;ts=100.5, server-end;ts=100.68",
        }
    ).encode()
    + b"\n\n"
)


class StepClock:
    def __init__(self, step: float = 0.001) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


class FakeResponse:
    def __init__(
        self,
        chunks: list[bytes],
        *,
        on_chunk: Callable[[int], None] | None = None,
        raise_after: Exception | None = None,
    ) -> None:
        self._chunks = chunks
        self._on_chunk = on_chunk
        self._raise_after = raise_after

    @property
    def status_code(self) -> int:
        return 200

    @property
    def headers(self) -> Mapping[str, str]:
        return {}

    def iter_bytes(self) -> Iterator[bytes]:
        for index, chunk in enumerate(self._chunks):
            if self._on_chunk is not None:
                self._on_chunk(index)
            yield chunk
        if self._raise_after is not None:
            raise self._raise_after

    def close(self) -> None:
        return None


class FakeTransport:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.requests: list[HTTPRequest] = []

    def open_stream(self, request: HTTPRequest) -> FakeResponse:
        self.requests.append(request)
        return self.response


def sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def stream(final_extra: Mapping[str, Any] | None = None) -> list[bytes]:
    """Two content deltas, a finish chunk, a usage chunk, then ``[DONE]``."""
    usage: dict[str, Any] = {
        "choices": [],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }
    usage.update(final_extra or {})
    return [
        sse({"id": "c", "choices": [{"index": 0, "delta": {"content": "Hi"}}]}),
        sse({"id": "c", "choices": [{"index": 0, "delta": {"content": " there"}}]}),
        sse(
            {"id": "c", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        ),
        sse(usage),
        b"data: [DONE]\n\n",
    ]


def config(tmp_path: Path, **overrides: Any) -> APICollectionConfig:
    kwargs: dict[str, Any] = {
        "run_id": "server-reported",
        "provider": "local",
        "endpoint": ENDPOINT,
        "model_id": "m",
        "prompt": "hello",
        "output_dir": tmp_path / "out",
        "command_argv": ("llmtracefx-optimizer", "collect-api"),
        "credential_env_var": API_KEY_ENV,
        "stream_include_usage": True,
    }
    kwargs.update(overrides)
    return APICollectionConfig(**kwargs)


def collect(
    tmp_path: Path,
    chunks: list[bytes] | FakeResponse,
    *,
    clock: StepClock | None = None,
    **overrides: Any,
) -> tuple[Any, FakeTransport]:
    response = chunks if isinstance(chunks, FakeResponse) else FakeResponse(chunks)
    transport = FakeTransport(response)
    result = stream_chat_completion(
        config(tmp_path, **overrides),
        transport=transport,
        environ=ENVIRON,
        clock=clock or StepClock(),
    )
    return result, transport


# --- vLLM ``metrics`` --------------------------------------------------------


def test_vllm_metrics_are_kept_beside_client_timing(tmp_path: Path) -> None:
    plain, _ = collect(tmp_path, stream())
    reported, _ = collect(tmp_path, stream({"metrics": VLLM_METRICS}))

    assert reported.evidence.success is True
    server = reported.evidence.server_reported
    assert server is not None
    assert server.sources == ("vllm_metrics",)
    # vLLM's TTFT starts at scheduling, so it is not the arrival-based TTFT.
    assert server.value("scheduled_ttft_ms") == 41.5
    assert server.value("ttft_ms") is None
    assert server.value("queue_ms") == 7.25
    assert server.value("decode_ms") == 120.0
    assert server.value("draft_tokens") == 8
    assert server.value("accepted_draft_tokens") == 6
    assert server.value("draft_acceptance_rate") == 0.75
    assert server.value("mean_acceptance_length") == 2.5
    assert server.value("verify_steps") == 4

    # The client's own measurements are untouched by what the server said.
    assert reported.evidence.timeline == plain.evidence.timeline

    persisted = reported.evidence.to_dict()["server_reported"]
    assert persisted["provenance"] == "provider_reported"
    assert persisted["raw"]["vllm_metrics"]["speculative_decoding"][
        "acceptance_histogram"
    ] == [1, 2, 1]
    assert persisted["malformed_fields"] == []


def test_no_server_block_records_nothing(tmp_path: Path) -> None:
    result, _ = collect(tmp_path, stream({"metrics": None}))
    assert result.evidence.server_reported is None
    assert "server_reported" not in result.evidence.to_dict()


def test_malformed_vllm_values_are_named_not_coerced(tmp_path: Path) -> None:
    chunks = stream(
        {
            "metrics": {
                "time_to_first_token_ms": 12.5,
                "queue_time_ms": -1,
                "mean_itl_ms": "fast",
                "tokens_per_second": True,
                "speculative_decoding": {"draft_acceptance_rate": 1.5},
            }
        }
    )
    # json.dumps cannot write NaN as a literal a strict parser accepts, but
    # Python's parser does, and so do some servers.
    chunks.insert(
        3, b'data: {"choices": [], "metrics": {"generation_time_ms": NaN}}\n\n'
    )
    result, _ = collect(tmp_path, chunks)

    assert result.evidence.success is True
    server = result.evidence.server_reported
    assert server is not None
    assert server.value("scheduled_ttft_ms") == 12.5
    assert server.value("decode_ms") is None
    assert server.value("queue_ms") is None
    assert set(server.malformed_fields) == {
        "metrics.generation_time_ms",
        "metrics.queue_time_ms",
        "metrics.mean_itl_ms",
        "metrics.tokens_per_second",
        "metrics.speculative_decoding.draft_acceptance_rate",
    }
    json.dumps(result.evidence.to_dict(), allow_nan=False)


def test_non_object_metrics_block_is_malformed_not_fatal(tmp_path: Path) -> None:
    result, _ = collect(tmp_path, stream({"metrics": [1, 2]}))
    assert result.evidence.success is True
    server = result.evidence.server_reported
    assert server is not None
    assert server.sources == ()
    assert server.malformed_fields == ("metrics",)


# --- SGLang ``sglext`` -------------------------------------------------------


def test_sglang_spec_defaults_without_a_verify_step_are_dropped(
    tmp_path: Path,
) -> None:
    # SpecTokensDetails defaults every number to 0; only a request that ran a
    # verify step has real values (spec_decoding_metrics.py L40-65 at b6d4389).
    sglext = {
        "spec_tokens_details": {
            "spec_verify_ct": 0,
            "spec_accept_rate": 0.0,
            "spec_accept_length": 0.0,
            "spec_num_correct_drafts": 0,
            "spec_num_proposed_drafts": 0,
            "spec_correct_drafts_histogram": [],
        },
        "cached_tokens_details": {"device": 3, "host": 0},
    }
    result, _ = collect(tmp_path, stream({"sglext": sglext}))

    server = result.evidence.server_reported
    assert server is not None
    assert server.sources == ("sglang_sglext",)
    assert server.values == {}
    assert server.raw["sglang_sglext"] == {
        "cached_tokens_details": {"device": 3, "host": 0}
    }


def test_sglang_spec_details_after_a_verify_step(tmp_path: Path) -> None:
    sglext = {
        "spec_tokens_details": {
            "spec_verify_ct": 5,
            "spec_accept_rate": 0.6,
            "spec_accept_length": 2.8,
            "spec_num_correct_drafts": 9,
            "spec_num_proposed_drafts": 15,
            "spec_correct_drafts_histogram": [1, 2, 2],
        },
        "routed_experts": "never persisted",
    }
    result, _ = collect(tmp_path, stream({"sglext": sglext}))

    server = result.evidence.server_reported
    assert server is not None
    assert server.value("verify_steps") == 5
    assert server.value("draft_acceptance_rate") == 0.6
    assert server.value("mean_acceptance_length") == 2.8
    assert server.value("accepted_draft_tokens") == 9
    assert server.value("draft_tokens") == 15
    assert "routed_experts" not in json.dumps(result.evidence.to_dict())


# --- TensorRT-LLM ``trtllm.perf_metrics`` ------------------------------------


def test_trtllm_metrics_need_the_opt_in(tmp_path: Path) -> None:
    chunks = [*stream(), TRT_EVENT]
    default, default_transport = collect(tmp_path, chunks)
    opted, opted_transport = collect(tmp_path, chunks, trtllm_return_metrics=True)

    assert TRT_HEADER not in default_transport.requests[0].headers
    assert default.evidence.server_reported is None
    assert default.evidence.plan.header_names == (
        "Accept",
        "Authorization",
        "Content-Type",
    )

    assert opted_transport.requests[0].headers[TRT_HEADER] == "1"
    assert TRT_HEADER in opted.evidence.plan.header_names
    assert opted.evidence.plan.config_hash != default.evidence.plan.config_hash
    assert opted.evidence.success is True
    server = opted.evidence.server_reported
    assert server is not None
    assert server.sources == ("trtllm_perf_metrics",)
    # TensorRT-LLM's TTFT runs from arrival, so it includes the queue.
    assert server.value("queue_ms") == 3.5
    assert server.value("ttft_ms") == 50.25
    assert server.value("e2e_ms") == 180.0
    assert server.value("scheduled_ttft_ms") is None
    raw = server.raw["trtllm_perf_metrics"]
    assert raw["server_timing"]["server_input_tokens"] == 11.0
    assert raw["start_end_time"] == {"server-start": 100.5, "server-end": 100.68}


def test_trtllm_wait_after_done_is_not_client_latency(tmp_path: Path) -> None:
    clock = StepClock()
    done_chunk = len(stream())

    def jump(index: int) -> None:
        # The trailing event arrives a long time after [DONE].
        if index == done_chunk:
            clock.value += 3.0

    chunks = [*stream(), TRT_EVENT]
    result, _ = collect(
        tmp_path,
        FakeResponse(chunks, on_chunk=jump),
        clock=clock,
        trtllm_return_metrics=True,
    )

    assert result.evidence.server_reported is not None
    assert result.evidence.timeline.completed_offset_ms is not None
    assert result.evidence.timeline.completed_offset_ms < 1000.0
    assert result.completed_monotonic is not None
    assert result.completed_monotonic < 1.0


def test_trouble_after_done_never_fails_the_request(tmp_path: Path) -> None:
    chunks = [*stream(), b"event: trtllm.perf_metrics\ndata: {not json\n\n"]
    broken, _ = collect(
        tmp_path,
        FakeResponse(chunks, raise_after=TransportConnectionError("reset")),
        trtllm_return_metrics=True,
    )
    assert broken.evidence.success is True
    assert broken.evidence.failure is None
    assert broken.evidence.server_reported is not None
    assert broken.evidence.server_reported.malformed_fields == ("trtllm.perf_metrics",)

    oversized = [*stream(), b": " + b"x" * (300 * 1024) + b"\n\n", TRT_EVENT]
    capped, _ = collect(tmp_path, oversized, trtllm_return_metrics=True)
    assert capped.evidence.success is True
    assert capped.evidence.server_reported is None


def test_trtllm_entry_echoing_the_credential_is_dropped(tmp_path: Path) -> None:
    event = (
        b"event: trtllm.perf_metrics\ndata: "
        + json.dumps({"server-timing": f"{API_KEY};dur=1, server_e2e;dur=9"}).encode()
        + b"\n\n"
    )
    result, _ = collect(tmp_path, [*stream(), event], trtllm_return_metrics=True)
    server = result.evidence.server_reported
    assert server is not None
    assert server.value("e2e_ms") == 9.0
    assert API_KEY not in json.dumps(result.evidence.to_dict())
    assert server.malformed_fields == ("trtllm.perf_metrics.server-timing",)


def test_trtllm_flag_must_be_a_boolean(tmp_path: Path) -> None:
    with pytest.raises(OpenAIStreamCollectorError, match="trtllm_return_metrics"):
        config(tmp_path, trtllm_return_metrics="yes")


def test_default_plan_has_no_trtllm_identity(tmp_path: Path) -> None:
    plain = build_request_plan(config(tmp_path), environ=ENVIRON)
    explicit_off = build_request_plan(
        config(tmp_path, trtllm_return_metrics=False), environ=ENVIRON
    )
    assert plain.config_hash == explicit_off.config_hash


# --- Rows, provenance and the sweep summary ----------------------------------


def test_server_fields_are_provider_reported_and_absent_for_mlx() -> None:
    for name in SERVER_REPORTED_ROW_FIELDS:
        assert ROW_FIELD_PROVENANCE[name] == "provider_reported"
        assert name not in MLX_FIELD_PROVENANCE


def test_sweep_summary_covers_only_rows_with_server_values() -> None:
    plain = RequestRow(index=0, sent=True, success=True)
    assert server_reported_summary([plain]) is None

    rows = [
        plain,
        RequestRow(
            index=1,
            sent=True,
            success=True,
            server_metrics_source="vllm_metrics",
            server_queue_ms=4.0,
        ),
        RequestRow(
            index=2,
            sent=True,
            success=False,
            server_metrics_source="vllm_metrics",
            server_queue_ms=900.0,
        ),
    ]
    summary = server_reported_summary(rows)
    assert summary is not None
    assert summary["sources"] == {"vllm_metrics": 1}
    assert summary["server_queue_ms"]["count"] == 1
    assert summary["server_queue_ms"]["max"] == 4.0
    assert summary["server_queue_ms"]["sample_provenance"] == "provider_reported"
    assert "server_ttft_ms" not in summary


# --- Command lines against a local server ------------------------------------


def run_main(argv: list[str]) -> tuple[int, str]:
    stderr = io.StringIO()
    code = 0
    try:
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(stderr),
        ):
            cli.main(argv)
    except SystemExit as exit_error:
        code = int(exit_error.code or 0)
    return code, stderr.getvalue()


@pytest.fixture(autouse=True)
def _credential(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, API_KEY)


def test_sweep_api_rows_and_points_carry_server_numbers(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Hi", encoding="utf-8")
    with fake_openai_server() as server:
        server.behaviour.final_extra = {"metrics": VLLM_METRICS}
        server.behaviour.after_done = TRT_EVENT
        code, err = run_main(
            [
                "sweep-api",
                "--run-id",
                "sweep",
                "--endpoint",
                server.endpoint,
                "--model-id",
                "fake-model",
                "--api-key-env",
                API_KEY_ENV,
                "--output-dir",
                str(tmp_path / "out"),
                "--prompt-file",
                str(prompt_file),
                "--concurrency",
                "2",
                "--requests-per-point",
                "2",
                "--stream-include-usage",
                "--trtllm-return-metrics",
            ]
        )
        headers = [{k.lower(): v for k, v in h.items()} for h in server.request_headers]
    assert code == 0, err
    assert all(h.get(TRT_HEADER.lower()) == "1" for h in headers)

    sweep = json.loads((tmp_path / "out" / "sweep.json").read_text(encoding="utf-8"))
    assert "--trtllm-return-metrics" in sweep["request_plan"]["command"]
    assert TRT_HEADER in sweep["request_plan"]["header_names"]
    assert sweep["workload"]["trtllm_return_metrics"] is True
    point = sweep["points"][0]
    row = point["requests"][0]
    assert row["server_metrics_source"] == "vllm_metrics+trtllm_perf_metrics"
    assert row["server_scheduled_ttft_ms"] == 41.5
    assert row["server_ttft_ms"] == 50.25
    assert row["server_draft_tokens"] == 8
    # Client latency is measured, not copied from the server.
    assert row["ttft_ms"] != row["server_ttft_ms"]
    summary = point["server_reported"]
    assert summary["sources"] == {"vllm_metrics+trtllm_perf_metrics": 2}
    assert summary["server_ttft_ms"]["count"] == 2


def test_collect_api_records_the_opt_in(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Hi", encoding="utf-8")
    with fake_openai_server() as server:
        server.behaviour.after_done = TRT_EVENT
        code, err = run_main(
            [
                "collect-api",
                "--run-id",
                "one",
                "--endpoint",
                server.endpoint,
                "--model-id",
                "fake-model",
                "--api-key-env",
                API_KEY_ENV,
                "--prompt-file",
                str(prompt_file),
                "--output-dir",
                str(tmp_path / "out"),
                "--trtllm-return-metrics",
            ]
        )
        headers = [{k.lower(): v for k, v in h.items()} for h in server.request_headers]
    assert code == 0, err
    assert headers[0].get(TRT_HEADER.lower()) == "1"
    evidence = json.loads(
        next((tmp_path / "out").rglob("api_evidence.json")).read_text("utf-8")
    )
    assert "--trtllm-return-metrics" in evidence["plan"]["command"]
    assert evidence["server_reported"]["normalized"]["e2e_ms"] == 180.0


def test_sglang_details_are_requested_only_with_the_opt_in(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Hi", encoding="utf-8")
    sglext = {
        "spec_tokens_details": {
            "spec_verify_ct": 2,
            "spec_accept_rate": 0.5,
            "spec_accept_length": 2.0,
            "spec_num_correct_drafts": 2,
            "spec_num_proposed_drafts": 4,
        }
    }

    def argv(run_id: str, *extra: str) -> list[str]:
        return [
            "collect-api",
            "--run-id",
            run_id,
            "--endpoint",
            server.endpoint,
            "--model-id",
            "fake-model",
            "--api-key-env",
            API_KEY_ENV,
            "--prompt-file",
            str(prompt_file),
            "--output-dir",
            str(tmp_path / run_id),
            *extra,
        ]

    with fake_openai_server() as server:
        server.behaviour.final_extra = {"sglext": sglext}
        assert run_main(argv("plain"))[0] == 0
        code, err = run_main(argv("details", "--sglang-return-details"))
        bodies = list(server.bodies)
    assert code == 0, err
    assert "return_spec_tokens_details" not in bodies[0]
    assert bodies[1]["return_spec_tokens_details"] is True
    assert bodies[1]["return_cached_tokens_details"] is True

    def evidence(run_id: str) -> dict[str, Any]:
        path = next((tmp_path / run_id).rglob("api_evidence.json"))
        return json.loads(path.read_text("utf-8"))

    plain, details = evidence("plain"), evidence("details")
    assert plain["plan"]["config_hash"] != details["plan"]["config_hash"]
    assert "--sglang-return-details" in details["plan"]["command"]
    assert details["plan"]["provider_extensions"]["return_spec_tokens_details"]
    reported = details["server_reported"]
    assert reported["sources"] == ["sglang_sglext"]
    assert reported["normalized"]["draft_acceptance_rate"] == 0.5
