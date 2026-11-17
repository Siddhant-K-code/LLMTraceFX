"""Tests for the API load worker: the dispatcher bound to the stream collector.

The end-to-end tests stream from a local server on 127.0.0.1. Edge cases
use an in-process fake transport. No request leaves the machine.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
from _load_fakes import API_KEY, API_KEY_ENV, ENVIRON, fake_openai_server

from llmtracefx.optimizer.collectors.openai_api import (
    FAILURE_HTTP_STATUS,
    APICollectionConfig,
    HTTPRequest,
    OpenAIStreamCollectorError,
    UrllibStreamingTransport,
)
from llmtracefx.optimizer.load.api_worker import (
    ROW_FIELD_PROVENANCE,
    TPOT_DERIVED,
    TPOT_NOT_APPLICABLE,
    TPOT_UNAVAILABLE,
    ApiRequestSpec,
    make_api_sender,
    preflight,
    request_config,
    request_rows,
)
from llmtracefx.optimizer.load.dispatcher import ClosedLoop, TimedArrivals, dispatch


def base_config(tmp_path: Path, endpoint: str, **overrides: Any) -> APICollectionConfig:
    kwargs: dict[str, Any] = {
        "run_id": "sweep",
        "provider": "local-test-server",
        "endpoint": endpoint,
        "model_id": "fake-model",
        "prompt": "unused base prompt",
        "output_dir": tmp_path / "unused",
        "command_argv": ("llmtracefx-optimizer", "sweep-api"),
        "credential_env_var": API_KEY_ENV,
        "request_timeout_seconds": 10.0,
        "stream_include_usage": True,
    }
    kwargs.update(overrides)
    return APICollectionConfig(**kwargs)


class _Response:
    def __init__(self, chunks: list[bytes], status_code: int = 200) -> None:
        self._chunks = chunks
        self._status_code = status_code

    @property
    def status_code(self) -> int:
        return self._status_code

    @property
    def headers(self) -> Mapping[str, str]:
        return {}

    def iter_bytes(self) -> Iterator[bytes]:
        yield from self._chunks

    def close(self) -> None:
        return None


class _ScriptedTransport:
    """Returns one canned stream for every request and keeps the bodies."""

    def __init__(
        self,
        *,
        usage: dict[str, Any] | None,
        parts: tuple[str, ...] = ("a", "b"),
        status_code: int = 200,
        error: Exception | None = None,
    ) -> None:
        self.usage = usage
        self.parts = parts
        self.status_code = status_code
        self.error = error
        self.bodies: list[dict[str, Any]] = []

    def open_stream(self, request: HTTPRequest) -> _Response:
        self.bodies.append(json.loads(request.body))
        if self.error is not None:
            raise self.error
        if self.status_code != 200:
            return _Response([b'{"error": "nope"}'], self.status_code)
        chunks = [
            f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': p}}]})}\n\n".encode()
            for p in self.parts
        ]
        final: dict[str, Any] = {
            "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]
        }
        if self.usage is not None:
            final["usage"] = self.usage
        chunks.append(f"data: {json.dumps(final)}\n\n".encode())
        chunks.append(b"data: [DONE]\n\n")
        return _Response(chunks)


def _run(
    tmp_path: Path, transport: Any, specs: list[ApiRequestSpec], **overrides: Any
) -> Any:
    config = base_config(
        tmp_path, "http://127.0.0.1:9/v1/chat/completions", **overrides
    )
    sender = make_api_sender(config, transport=transport, environ=ENVIRON)
    return request_rows(dispatch(specs, sender, ClosedLoop(1)))


# --- End to end against a local server ---------------------------------------


def test_concurrent_requests_stream_and_reduce_to_text_free_rows(
    tmp_path: Path,
) -> None:
    specs = [
        ApiRequestSpec(prompt=f"secret prompt {i}", max_tokens=16 + i) for i in range(8)
    ]
    with fake_openai_server() as server:
        server.behaviour.chunk_delay_s = 0.02
        config = base_config(tmp_path, server.endpoint)
        configs = preflight(config, specs, environ=ENVIRON)
        sender = make_api_sender(
            config, transport=UrllibStreamingTransport(), environ=ENVIRON
        )
        report = dispatch(specs, sender, ClosedLoop(4))
        rows = request_rows(report)

    assert len(configs) == 8
    assert server.peak_active >= 2
    assert sorted(body["max_tokens"] for body in server.bodies) == list(range(16, 24))
    assert all(
        body["stream_options"] == {"include_usage": True} for body in server.bodies
    )
    for row in rows:
        assert row.sent and row.success is True and row.client_error is None
        assert row.http_status == 200
        assert row.finish_reason == "stop"
        assert row.output_tokens == 3
        assert row.prompt_tokens == 11
        assert row.cached_prompt_tokens == 4
        assert row.ttft_ms is not None and row.e2e_ms is not None
        assert 0 < row.ttft_ms <= row.e2e_ms
        assert row.tpot_status == TPOT_DERIVED
        assert row.tpot_ms == pytest.approx((row.e2e_ms - row.ttft_ms) / 2)
        assert row.dispatched_offset_s is not None
        assert row.completed_offset_s is not None
        assert row.completed_offset_s >= row.dispatched_offset_s
        assert row.in_flight_at_dispatch is not None
        assert 1 <= row.in_flight_at_dispatch <= 4
    assert [row.run_id for row in rows] == [f"sweep-{i:06d}" for i in range(8)]
    dumped = json.dumps([row.to_dict() for row in rows])
    assert "secret prompt" not in dumped and "Hello" not in dumped
    assert API_KEY not in dumped
    assert not (tmp_path / "unused").exists()


def test_a_rejected_request_becomes_a_failed_row(tmp_path: Path) -> None:
    with fake_openai_server() as server:
        server.behaviour.status_code = 503
        config = base_config(tmp_path, server.endpoint)
        sender = make_api_sender(
            config, transport=UrllibStreamingTransport(), environ=ENVIRON
        )
        rows = request_rows(
            dispatch([ApiRequestSpec(prompt="p")], sender, TimedArrivals((0,)))
        )
    (row,) = rows
    assert row.sent and row.success is False
    assert row.failure_category == FAILURE_HTTP_STATUS
    assert row.http_status == 503
    assert row.tpot_status == TPOT_UNAVAILABLE
    assert row.output_tokens is None


# --- Per-request parameters --------------------------------------------------


def test_spec_max_tokens_and_extra_body_override_the_base(tmp_path: Path) -> None:
    transport = _ScriptedTransport(usage={"completion_tokens": 2})
    specs = [
        ApiRequestSpec(prompt="one", max_tokens=7, extra_body={"ignore_eos": True}),
        ApiRequestSpec(prompt="two", system_prompt="be brief", run_id_suffix="r2"),
    ]
    rows = _run(
        tmp_path,
        transport,
        specs,
        max_output_tokens=64,
        extra_body={"ignore_eos": False, "min_tokens": 1},
    )

    first, second = transport.bodies
    assert first["max_tokens"] == 7
    assert first["ignore_eos"] is True and first["min_tokens"] == 1
    assert second["max_tokens"] == 64 and second["ignore_eos"] is False
    assert second["messages"][0] == {"role": "system", "content": "be brief"}
    assert rows[1].run_id == "sweep-r2"
    assert rows[0].finish_reason == "length"


@pytest.mark.parametrize(
    ("usage", "status", "tokens"),
    [
        ({"completion_tokens": 1}, TPOT_NOT_APPLICABLE, 1),
        (None, TPOT_UNAVAILABLE, None),
    ],
)
def test_tpot_is_never_guessed(
    tmp_path: Path, usage: dict[str, Any] | None, status: str, tokens: int | None
) -> None:
    rows = _run(tmp_path, _ScriptedTransport(usage=usage), [ApiRequestSpec(prompt="p")])
    assert rows[0].tpot_status == status
    assert rows[0].tpot_ms is None
    assert rows[0].output_tokens == tokens
    assert rows[0].prompt_tokens is None and rows[0].cached_prompt_tokens is None


def test_client_side_exceptions_are_rows_with_a_type_name(tmp_path: Path) -> None:
    transport = _ScriptedTransport(usage=None, error=RuntimeError("boom"))
    rows = _run(tmp_path, transport, [ApiRequestSpec(prompt="p")])
    assert rows[0].sent and rows[0].client_error == "RuntimeError"
    assert rows[0].success is None and rows[0].http_status is None


def test_provenance_covers_every_numeric_row_metric() -> None:
    assert ROW_FIELD_PROVENANCE["ttft_ms"] == "measured_wall_clock"
    assert ROW_FIELD_PROVENANCE["output_tokens"] == "provider_reported"
    assert ROW_FIELD_PROVENANCE["tpot_ms"] == "derived"


# --- Validation --------------------------------------------------------------


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"prompt": ""}, "prompt"),
        ({"prompt": "p", "max_tokens": 0}, "max_tokens"),
        ({"prompt": "p", "max_tokens": True}, "max_tokens"),
        ({"prompt": "p", "run_id_suffix": "../x"}, "run_id_suffix"),
        ({"prompt": "p", "extra_body": ["x"]}, "extra_body"),
    ],
)
def test_bad_specs_are_refused(kwargs: dict[str, Any], match: str) -> None:
    with pytest.raises(OpenAIStreamCollectorError, match=match):
        ApiRequestSpec(**kwargs)


def test_preflight_refuses_before_anything_is_sent(tmp_path: Path) -> None:
    config = base_config(tmp_path, "http://127.0.0.1:9/v1/chat/completions")
    duplicate = [
        ApiRequestSpec(prompt="a", run_id_suffix="x"),
        ApiRequestSpec(prompt="b", run_id_suffix="x"),
    ]
    with pytest.raises(OpenAIStreamCollectorError, match="unique"):
        preflight(config, duplicate, environ=ENVIRON)
    with pytest.raises(OpenAIStreamCollectorError):
        preflight(config, [ApiRequestSpec(prompt="a")], environ={})
    with pytest.raises(OpenAIStreamCollectorError):
        preflight(config, [ApiRequestSpec(prompt=f"key is {API_KEY}")], environ=ENVIRON)
    with pytest.raises(OpenAIStreamCollectorError, match="collector controls"):
        request_config(config, ApiRequestSpec(prompt="a", extra_body={"model": "x"}), 0)
