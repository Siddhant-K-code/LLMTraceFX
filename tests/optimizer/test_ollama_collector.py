"""Native Ollama collector against a fake transport. Nothing is sent."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer.collectors.ollama import (
    API_GENERATE,
    NDJSONDecoder,
    OllamaCollectionConfig,
    OllamaError,
    OllamaSettings,
    collect_ollama,
    parse_server_metrics,
    request_body,
    stream_request,
)
from llmtracefx.optimizer.collectors.openai_api import (
    HTTPRequest,
    TransportConnectionError,
)
from llmtracefx.optimizer.schema import ExperimentRecord

ROOT = "http://127.0.0.1:11434"
KEY = "ollama-proxy-key-not-a-real-credential"

FINAL_METRICS: dict[str, Any] = {
    "total_duration": 250_000_000,
    "load_duration": 12_500_000,
    "prompt_eval_count": 9,
    "prompt_eval_cached_count": 4,
    "prompt_eval_duration": 40_000_000,
    "eval_count": 3,
    "eval_duration": 150_000_000,
}


def _line(payload: Any) -> bytes:
    return json.dumps(payload).encode() + b"\n"


def chat_lines(
    pieces: Sequence[str] = ("Hel", "lo", "!"),
    *,
    thinking: Sequence[str] = (),
    done_reason: str = "stop",
    metrics: Mapping[str, Any] = FINAL_METRICS,
) -> list[bytes]:
    lines = [
        _line(
            {
                "model": "m",
                "message": {"role": "assistant", "content": "", "thinking": t},
                "done": False,
            }
        )
        for t in thinking
    ]
    lines += [
        _line(
            {
                "model": "m",
                "message": {"role": "assistant", "content": p},
                "done": False,
            }
        )
        for p in pieces
    ]
    lines.append(
        _line(
            {
                "model": "m",
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "done_reason": done_reason,
                **metrics,
            }
        )
    )
    return lines


@dataclass
class FakeResponse:
    status_code: int
    chunks: Sequence[bytes]
    headers: Mapping[str, str] = field(default_factory=dict)
    closed: bool = False

    def iter_bytes(self) -> Iterator[bytes]:
        yield from self.chunks

    def close(self) -> None:
        self.closed = True


@dataclass
class FakeTransport:
    chunks: Sequence[bytes]
    status: int = 200
    version: Any = None
    requests: list[HTTPRequest] = field(default_factory=list)
    responses: list[FakeResponse] = field(default_factory=list)

    def open_stream(self, request: HTTPRequest) -> FakeResponse:
        self.requests.append(request)
        if request.url.endswith("/api/version"):
            if self.version is None:
                raise TransportConnectionError("refused")
            response = FakeResponse(200, [json.dumps(self.version).encode()])
        else:
            response = FakeResponse(self.status, self.chunks)
        self.responses.append(response)
        return response


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        self.now += 0.01
        return self.now


def settings(**overrides: Any) -> OllamaSettings:
    values: dict[str, Any] = {"base_url": ROOT, "model": "qwen3:0.6b", "num_predict": 3}
    values.update(overrides)
    return OllamaSettings(**values)


def run_stream(chunks: Sequence[bytes], status: int = 200, **overrides: Any) -> Any:
    stream, _ = stream_request(
        settings(**overrides),
        prompt="hi",
        system_prompt=None,
        transport=FakeTransport(chunks, status=status),
        credential=None,
        clock=Clock(),
        keep_text=True,
    )
    return stream


def test_chat_stream_records_client_timing_and_server_metrics() -> None:
    # Split lines across chunks to exercise the incremental decoder.
    data = b"".join(chat_lines(thinking=("Let me",)))
    chunks = [data[i : i + 7] for i in range(0, len(data), 7)]
    stream = run_stream(chunks)
    assert stream.failure is None
    assert stream.response_text() == "Hello!"
    assert stream.thinking_chunks == 1
    assert stream.content_chunks == 3
    assert stream.first_output_at == stream.first_thinking_at
    assert stream.first_output_at < stream.first_content_at < stream.final_at
    assert stream.done_reason() == "stop"
    assert stream.finish_code() == "stop"

    metrics = parse_server_metrics(stream.final)
    assert metrics.ms("load_duration") == 12.5
    assert metrics.ms("prompt_eval_duration") == 40.0
    assert metrics.ms("eval_duration") == 150.0
    assert metrics.ms("total_duration") == 250.0
    assert metrics.count("prompt_eval_cached_count") == 4


def test_generate_stream_reads_response_field_and_drops_context() -> None:
    lines = [
        _line({"model": "m", "response": "Hi", "done": False}),
        _line({"model": "m", "response": " there", "done": False}),
        _line(
            {
                "model": "m",
                "response": "",
                "done": True,
                "done_reason": "length",
                "context": list(range(5000)),
                **FINAL_METRICS,
            }
        ),
    ]
    stream = run_stream(lines, api=API_GENERATE)
    assert stream.failure is None
    assert stream.response_text() == "Hi there"
    assert stream.finish_code() == "length"
    assert "context" not in parse_server_metrics(stream.final).to_dict()["raw"]


def test_absent_metric_fields_stay_absent_and_cached_zero_is_kept() -> None:
    metrics = parse_server_metrics(
        {"eval_count": 3, "prompt_eval_cached_count": 0, "load_duration": True}
    )
    payload = metrics.to_dict()
    assert payload["raw"]["prompt_eval_cached_count"] == 0
    assert payload["raw"]["total_duration"] is None
    assert payload["malformed_fields"] == ["load_duration"]
    assert "total_duration" in payload["absent_fields"]
    assert "load_duration" not in payload["absent_fields"]
    assert metrics.ms("load_duration") is None
    assert parse_server_metrics({"eval_count": -1}).malformed_fields == ("eval_count",)


@pytest.mark.parametrize(
    ("chunks", "status", "category", "message"),
    [
        (chat_lines()[:2], 200, "stream_truncated", "before the done chunk"),
        (
            [*chat_lines()[:1], _line({"error": f"runner died {KEY}"})],
            200,
            "provider_error_payload",
            "runner died",
        ),
        ([b"{not json\n"], 200, "stream_decode", "not JSON"),
        ([_line({"message": "x", "done": False})], 200, "stream_decode", "shape"),
        ([_line({"error": "model 'x' not found"})], 404, "http_status", "HTTP 404"),
        (chat_lines(done_reason="unload"), 200, "unrecognized_done_reason", "unload"),
    ],
)
def test_failure_categories(
    chunks: list[bytes], status: int, category: str, message: str
) -> None:
    stream, _ = stream_request(
        settings(),
        prompt="hi",
        system_prompt=None,
        transport=FakeTransport(chunks, status=status),
        credential=KEY,
        clock=Clock(),
        keep_text=False,
    )
    assert stream.failure == category
    assert message in stream.failure_message
    assert KEY not in stream.failure_message


def test_stream_stops_reading_after_the_done_chunk() -> None:
    stream = run_stream([*chat_lines(), b"{not json\n"])
    assert stream.failure is None


def test_ndjson_decoder_refuses_oversized_lines() -> None:
    decoder = NDJSONDecoder(max_line_bytes=8)
    assert decoder.feed(b"{}\n\n{}") == [b"{}"]
    assert decoder.close() == [b"{}"]
    with pytest.raises(ValueError, match="size limit"):
        decoder.feed(b"x" * 9)


def test_request_body_shapes() -> None:
    chat = request_body(
        settings(temperature=0.0, seed=7, think=False), "hi", system_prompt="be brief"
    )
    assert chat == {
        "model": "qwen3:0.6b",
        "stream": True,
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ],
        "options": {"num_predict": 3, "temperature": 0.0, "seed": 7},
        "think": False,
    }
    generate = request_body(settings(api=API_GENERATE), "hi", system_prompt="sys")
    assert generate == {
        "model": "qwen3:0.6b",
        "stream": True,
        "prompt": "hi",
        "system": "sys",
        "options": {"num_predict": 3},
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"base_url": f"{ROOT}/api/chat"}, "server root"),
        ({"base_url": f"{ROOT}/v1"}, "server root"),
        ({"base_url": f"{ROOT}/?a=1"}, "query"),
        ({"num_predict": 0}, "max-output-tokens"),
        ({"api": "embed"}, "--api"),
        ({"top_p": 0}, "top-p"),
        ({"model": " m"}, "surrounding spaces"),
    ],
)
def test_settings_are_validated(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(OllamaError, match=message):
        settings(**overrides)


def _config(tmp_path: Path, **overrides: Any) -> OllamaCollectionConfig:
    return OllamaCollectionConfig(
        run_id="ollama-run",
        settings=settings(**overrides),
        prompt="Say hello",
        output_dir=tmp_path / "out",
        command_argv=("llmtracefx-optimizer", "collect-ollama"),
    )


def test_collect_writes_record_with_provider_reported_phases(tmp_path: Path) -> None:
    transport = FakeTransport(chat_lines(), version={"version": "0.12.3"})
    result = collect_ollama(
        _config(tmp_path, credential_env_var="OLLAMA_PROXY_KEY"),
        transport=transport,
        environ={"OLLAMA_PROXY_KEY": KEY},
        clock=Clock(),
    )
    out = tmp_path / "out"
    assert {p.name for p in out.iterdir()} == {
        "record.json",
        "response.txt",
        "ollama_evidence.json",
        "environment.json",
    }
    assert all(r.closed for r in transport.responses)
    assert [r.headers["Authorization"] for r in transport.requests] == [
        f"Bearer {KEY}"
    ] * 2
    for path in out.iterdir():
        assert KEY not in path.read_text(encoding="utf-8")
    assert (out / "response.txt").read_text(encoding="utf-8") == "Hello!"

    record = ExperimentRecord.from_dict(
        json.loads((out / "record.json").read_text(encoding="utf-8"))
    )
    assert record == result.record
    assert record.outcome.success is True
    assert record.runtime.name == "ollama"
    assert record.runtime.version == "0.12.3"
    assert record.tokens.input_tokens == 9
    assert record.tokens.generated_tokens == 3
    assert record.tokens.provenance is not None
    assert record.tokens.provenance.value == "provider_reported"
    assert record.timing.model_load is not None
    assert record.timing.model_load.value == 12.5
    assert record.timing.model_load.provenance.value == "provider_reported"
    assert record.timing.prefill is not None
    assert record.timing.prefill.value == 40.0
    assert record.timing.decode is not None
    assert record.timing.decode.value == 150.0
    assert record.timing.total is not None
    assert record.timing.total.provenance.value == "measured_wall_clock"

    evidence = json.loads((out / "ollama_evidence.json").read_text(encoding="utf-8"))
    assert evidence["server"]["version"] == "0.12.3"
    assert evidence["server_reported"]["source"] == "ollama_metrics"
    assert evidence["server_reported"]["provenance"] == "provider_reported"
    assert evidence["server_reported"]["raw"]["prompt_eval_cached_count"] == 4
    assert evidence["server_reported"]["milliseconds"]["total_duration"] == 250.0
    timing = evidence["client_timing"]
    assert timing["ttft_source"] == "first_output_chunk"
    assert timing["ttft_ms"] <= timing["done_chunk_ms"] <= timing["e2e_ms"]
    assert timing["inter_output_chunk_gap_ms"]["count"] == 2
    assert "Say hello" not in json.dumps(evidence)
    assert evidence["request"]["header_names"] == [
        "Accept",
        "Authorization",
        "Content-Type",
    ]


def test_collect_records_a_failed_request_and_unknown_version(tmp_path: Path) -> None:
    transport = FakeTransport(chat_lines()[:1])
    result = collect_ollama(
        _config(tmp_path), transport=transport, environ={}, clock=Clock()
    )
    assert result.record.outcome.success is False
    assert result.record.error is not None
    assert result.record.error.category == "stream_truncated"
    assert result.record.runtime.version is None
    assert result.record.timing.prefill is None
    assert result.evidence["server_reported"] is None
    assert result.evidence["server"]["error"].startswith("TransportConnectionError")


def test_missing_credential_raises_before_sending(tmp_path: Path) -> None:
    transport = FakeTransport(chat_lines())
    with pytest.raises(OllamaError, match="not set"):
        collect_ollama(
            _config(tmp_path, credential_env_var="OLLAMA_PROXY_KEY"),
            transport=transport,
            environ={},
        )
    assert transport.requests == []
