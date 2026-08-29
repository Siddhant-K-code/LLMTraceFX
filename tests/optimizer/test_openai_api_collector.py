"""Tests for the provider-neutral OpenAI-compatible streaming API collector.

Every test here injects a fake transport. No request ever leaves the
process, no API key is used, and Z.ai is never contacted. The GLM request
profiles exercised below follow Z.ai's published chat-completions
documentation:

* https://docs.z.ai/guides/vlm/glm-5.3-flash
* https://docs.z.ai/api-reference/llm/chat-completion
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer.collectors.openai_api import (
    FAILURE_CONNECTION,
    FAILURE_HTTP_STATUS,
    FAILURE_MISSING_CONTENT,
    FAILURE_PROVIDER_ERROR,
    FAILURE_STREAM_DECODE,
    FAILURE_TIMEOUT,
    APICollectionConfig,
    HTTPRequest,
    OpenAIStreamCollectorError,
    ProviderExtensions,
    TransportConnectionError,
    TransportTimeout,
    build_request_plan,
    collect_openai_stream,
)
from llmtracefx.optimizer.schema import ExperimentRecord, MetricProvenance

ENDPOINT = "https://api.z.ai/api/paas/v4/chat/completions"
API_KEY = "test-key-not-a-real-credential"
ENVIRON = {"ZAI_API_KEY": API_KEY}


class StepClock:
    """Monotonic clock that advances a fixed amount per read."""

    def __init__(self, step: float = 0.001) -> None:
        self._value = 0.0
        self._step = step

    def __call__(self) -> float:
        current = self._value
        self._value += self._step
        return current


class FakeResponse:
    """A streaming response whose body is a fixed list of byte chunks."""

    def __init__(
        self,
        chunks: list[bytes],
        *,
        status_code: int = 200,
        headers: Mapping[str, str] | None = None,
        raise_after: Exception | None = None,
    ) -> None:
        self._chunks = chunks
        self._status_code = status_code
        self._headers = dict(headers or {})
        self._raise_after = raise_after
        self.closed = False

    @property
    def status_code(self) -> int:
        return self._status_code

    @property
    def headers(self) -> Mapping[str, str]:
        return self._headers

    def iter_bytes(self) -> Iterator[bytes]:
        yield from self._chunks
        if self._raise_after is not None:
            raise self._raise_after

    def close(self) -> None:
        self.closed = True


class FakeTransport:
    def __init__(self, response: FakeResponse | Exception) -> None:
        self._response = response
        self.requests: list[HTTPRequest] = []

    def open_stream(self, request: HTTPRequest) -> FakeResponse:
        self.requests.append(request)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class ExplodingTransport:
    """Fails the test if the collector ever opens a stream."""

    def open_stream(self, request: HTTPRequest) -> FakeResponse:
        raise AssertionError("no network request should have been attempted")


def sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def make_config(tmp_path: Path, **overrides: Any) -> APICollectionConfig:
    kwargs: dict[str, Any] = {
        "run_id": "api-run",
        "provider": "z.ai",
        "endpoint": ENDPOINT,
        "model_id": "glm-5.3",
        "prompt": "Explain a stack in one sentence.",
        "output_dir": tmp_path / "artifacts",
        "command_argv": (
            "llmtracefx-optimizer",
            "collect-api",
            "--run-id",
            "api-run",
            "--api-key-env",
            "ZAI_API_KEY",
        ),
        "credential_env_var": "ZAI_API_KEY",
    }
    kwargs.update(overrides)
    return APICollectionConfig(**kwargs)


def glm_stream(
    *,
    content_parts: tuple[str, ...] = ("Hello", " world"),
    reasoning_parts: tuple[str, ...] = ("weighing options",),
    usage: dict[str, Any] | None = None,
) -> list[bytes]:
    """A GLM-shaped stream: metadata, reasoning, content, final usage, DONE."""
    chunks = [
        b": keepalive\n\n",
        sse(
            {
                "id": "chatcmpl-abc",
                "request_id": "req-123",
                "model": "glm-5.3",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": ""}}
                ],
            }
        ),
    ]
    chunks.extend(
        sse(
            {
                "id": "chatcmpl-abc",
                "choices": [{"index": 0, "delta": {"reasoning_content": part}}],
            }
        )
        for part in reasoning_parts
    )
    chunks.extend(
        sse(
            {
                "id": "chatcmpl-abc",
                "choices": [{"index": 0, "delta": {"content": part}}],
            }
        )
        for part in content_parts
    )
    chunks.append(
        sse(
            {
                "id": "chatcmpl-abc",
                "choices": [
                    {"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}
                ],
                "usage": (
                    usage
                    if usage is not None
                    else {
                        "prompt_tokens": 17,
                        "completion_tokens": 5,
                        "total_tokens": 22,
                        "prompt_tokens_details": {"cached_tokens": 4},
                    }
                ),
            }
        )
    )
    chunks.append(b"data: [DONE]\n\n")
    return chunks


def run(
    config: APICollectionConfig,
    chunks: list[bytes] | FakeResponse | Exception,
    *,
    environ: Mapping[str, str] | None = None,
) -> tuple[Any, FakeTransport]:
    if isinstance(chunks, list):
        response: FakeResponse | Exception = FakeResponse(chunks)
    else:
        response = chunks
    transport = FakeTransport(response)
    result = collect_openai_stream(
        config,
        transport=transport,
        environ=ENVIRON if environ is None else environ,
        clock=StepClock(),
    )
    return result, transport


# --- Configuration validation ------------------------------------------------


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("run_id", "", "run_id"),
        ("provider", "not a label!", "provider must be a short label"),
        ("model_id", "  ", "model_id"),
        ("prompt", "", "prompt must be non-empty"),
        ("system_prompt", "", "system_prompt must be non-empty"),
        ("credential_env_var", "9BAD", "credential_env_var"),
        ("max_output_tokens", 0, "max_output_tokens"),
        ("max_output_tokens", True, "max_output_tokens"),
        ("seed", 1.5, "seed must be an integer"),
        ("temperature", 2.5, "temperature must be between"),
        ("temperature", float("nan"), "temperature must be a finite number"),
        ("top_p", 1.5, "top_p must be between"),
        ("top_p", float("inf"), "top_p must be a finite number"),
        ("request_timeout_seconds", 0, "request_timeout_seconds"),
        ("request_timeout_seconds", float("inf"), "request_timeout_seconds"),
        ("extensions", {"reasoning_effort": "high"}, "ProviderExtensions"),
        ("command_argv", (), "command_argv"),
        ("command_argv", ("ok", ""), "command_argv"),
    ],
)
def test_invalid_configuration_is_rejected(
    tmp_path: Path, field_name: str, value: Any, match: str
) -> None:
    with pytest.raises(OpenAIStreamCollectorError, match=match):
        make_config(tmp_path, **{field_name: value})


@pytest.mark.parametrize(
    ("endpoint", "match"),
    [
        ("", "endpoint must be non-empty"),
        ("ftp://api.z.ai/v4", "http or https"),
        ("https:///v4/chat", "must include a host"),
        ("http://api.z.ai/v4/chat", "must use https"),
        ("https://user:pass@api.z.ai/v4/chat", "must not embed credentials"),
        ("https://api.z.ai/v4/chat#frag", "must not contain a fragment"),
        ("https://api.z.ai/v4/chat?api_key=abc", "looks like a credential"),
        ("https://api.z.ai/v4/chat?access_token=abc", "looks like a credential"),
    ],
)
def test_invalid_endpoint_is_rejected(
    tmp_path: Path, endpoint: str, match: str
) -> None:
    with pytest.raises(OpenAIStreamCollectorError, match=match):
        make_config(tmp_path, endpoint=endpoint)


def test_plain_http_is_allowed_for_local_test_servers(tmp_path: Path) -> None:
    config = make_config(tmp_path, endpoint="http://127.0.0.1:8080/v1/chat/completions")

    assert config.endpoint.startswith("http://127.0.0.1")


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("reasoning_effort", ""),
        ("reasoning_effort", 3),
        ("thinking_type", "   "),
        ("provider_request_id", 7),
        ("clear_thinking", "true"),
    ],
)
def test_invalid_provider_extensions_are_rejected(field_name: str, value: Any) -> None:
    with pytest.raises(OpenAIStreamCollectorError):
        ProviderExtensions(**{field_name: value})


# --- Request plan and command reconstruction ---------------------------------


def test_request_plan_records_identity_without_the_credential(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        max_output_tokens=256,
        temperature=0.2,
        top_p=0.9,
        seed=7,
        system_prompt="You are terse.",
        extensions=ProviderExtensions(reasoning_effort="high", clear_thinking=False),
    )

    plan = build_request_plan(config)
    serialized = plan.to_json()

    assert plan.endpoint_origin == "https://api.z.ai"
    assert plan.endpoint_path == "/api/paas/v4/chat/completions"
    assert plan.credential_env_var == "ZAI_API_KEY"
    assert plan.credential_header_name == "Authorization"
    assert "Authorization" in plan.header_names
    assert plan.request_parameters["max_tokens"] == 256
    assert plan.provider_extensions == {
        "reasoning_effort": "high",
        "thinking": {"clear_thinking": False},
    }
    assert tuple(plan.command) == config.command_argv
    assert API_KEY not in serialized
    # Prompts are hashed, never copied.
    assert "You are terse." not in serialized
    assert "Explain a stack" not in serialized
    assert [message.role for message in plan.messages] == ["system", "user"]
    assert all(
        message.content_sha256.startswith("sha256:") for message in plan.messages
    )


def test_config_hash_is_stable_and_parameter_sensitive(tmp_path: Path) -> None:
    baseline = build_request_plan(make_config(tmp_path)).config_hash

    assert baseline == build_request_plan(make_config(tmp_path)).config_hash
    assert (
        baseline
        != build_request_plan(make_config(tmp_path, temperature=0.1)).config_hash
    )
    assert (
        baseline
        != build_request_plan(
            make_config(tmp_path, model_id="glm-5.3-flash")
        ).config_hash
    )
    assert (
        baseline
        != build_request_plan(
            make_config(tmp_path, extensions=ProviderExtensions(reasoning_effort="low"))
        ).config_hash
    )
    # The output directory is not part of request identity.
    assert (
        baseline == build_request_plan(make_config(tmp_path / "elsewhere")).config_hash
    )


def test_building_a_plan_requires_no_credential(tmp_path: Path) -> None:
    plan = build_request_plan(make_config(tmp_path))

    assert plan.model_id == "glm-5.3"


@pytest.mark.parametrize(
    ("model_id", "extensions", "expected"),
    [
        (
            "glm-5.3",
            ProviderExtensions(reasoning_effort="max", thinking_type="enabled"),
            {"reasoning_effort": "max", "thinking": {"type": "enabled"}},
        ),
        (
            "glm-5.3-flash",
            ProviderExtensions(reasoning_effort="low", clear_thinking=True),
            {"reasoning_effort": "low", "thinking": {"clear_thinking": True}},
        ),
    ],
)
def test_glm_request_profiles_reach_the_wire(
    tmp_path: Path,
    model_id: str,
    extensions: ProviderExtensions,
    expected: dict[str, Any],
) -> None:
    config = make_config(tmp_path, model_id=model_id, extensions=extensions)

    _, transport = run(config, glm_stream())

    body = json.loads(transport.requests[0].body.decode("utf-8"))
    assert body["model"] == model_id
    assert body["stream"] is True
    for key, value in expected.items():
        assert body[key] == value
    assert transport.requests[0].headers["Accept"] == "text/event-stream"


# --- Credential handling -----------------------------------------------------


def test_missing_credential_variable_is_an_explicit_error(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    with pytest.raises(OpenAIStreamCollectorError, match="ZAI_API_KEY is not set"):
        collect_openai_stream(config, transport=ExplodingTransport(), environ={})

    assert not config.output_dir.exists()


def test_blank_credential_variable_is_an_explicit_error(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    with pytest.raises(OpenAIStreamCollectorError, match="is empty"):
        collect_openai_stream(
            config, transport=ExplodingTransport(), environ={"ZAI_API_KEY": "  "}
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"prompt": f"leak {API_KEY}"},
        {"system_prompt": f"leak {API_KEY}"},
        {"command_argv": ("llmtracefx-optimizer", "collect-api", f"--x={API_KEY}")},
        {"endpoint": f"https://api.z.ai/v4/chat?trace={API_KEY}"},
        {"run_id": f"run-{API_KEY}"},
        {"model_id": f"glm-5.3-{API_KEY}"},
        {"model_revision": API_KEY},
        {"extensions": ProviderExtensions(provider_request_id=API_KEY)},
        {"extensions": ProviderExtensions(reasoning_effort=API_KEY)},
        {"extensions": ProviderExtensions(thinking_type=API_KEY)},
    ],
)
def test_collector_refuses_to_run_when_the_secret_would_be_persisted(
    tmp_path: Path, overrides: dict[str, Any]
) -> None:
    config = make_config(tmp_path, **overrides)

    with pytest.raises(OpenAIStreamCollectorError, match="refusing to run"):
        collect_openai_stream(config, transport=ExplodingTransport(), environ=ENVIRON)


def test_surrounding_whitespace_is_stripped_from_the_credential(
    tmp_path: Path,
) -> None:
    """A key read from a file or a ``.env`` routinely carries a newline.

    Sending it unstripped makes ``http.client`` raise a ``ValueError``
    whose message embeds the whole header value, which would print the
    secret in a traceback.
    """
    config = make_config(tmp_path)

    _, transport = run(config, glm_stream(), environ={"ZAI_API_KEY": f"  {API_KEY}\n"})

    assert transport.requests[0].headers["Authorization"] == f"Bearer {API_KEY}"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("sk-zai-\x00-abc", "control character"),
        ("sk-zai-\x7f-abc", "control character"),
        ("sk-zai-\u2603-abc", "non latin-1"),
    ],
)
def test_a_credential_that_cannot_be_a_header_value_is_refused(
    tmp_path: Path, value: str, expected: str
) -> None:
    config = make_config(tmp_path)

    with pytest.raises(OpenAIStreamCollectorError, match=expected) as excinfo:
        collect_openai_stream(
            config, transport=ExplodingTransport(), environ={"ZAI_API_KEY": value}
        )

    assert value not in str(excinfo.value)
    assert "ZAI_API_KEY" in str(excinfo.value)
    assert not config.output_dir.exists()


def test_credential_is_sent_but_never_persisted(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    result, transport = run(config, glm_stream())

    assert transport.requests[0].headers["Authorization"] == f"Bearer {API_KEY}"
    assert API_KEY not in repr(transport.requests[0])
    for artifact in sorted(config.output_dir.iterdir()):
        assert API_KEY not in artifact.read_text(encoding="utf-8")
    assert API_KEY not in result.evidence.to_json()


def test_secret_echoed_in_an_error_body_is_redacted(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    body = json.dumps(
        {"code": 401, "message": f"invalid key {API_KEY} for Bearer {API_KEY}"}
    ).encode()

    result, _ = run(config, FakeResponse([body], status_code=401))

    failure = result.evidence.failure
    assert failure is not None
    assert API_KEY not in failure.message
    assert "[REDACTED]" in failure.message
    assert API_KEY not in (config.output_dir / "record.json").read_text(
        encoding="utf-8"
    )


# --- Successful streaming ----------------------------------------------------


def test_successful_stream_produces_evidence_and_artifacts(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    result, transport = run(config, glm_stream())

    assert result.response_text == "Hello world"
    assert result.record.outcome.success is True
    assert result.record.error is None
    evidence = result.evidence
    assert evidence.success is True
    assert evidence.finish_reason == "stop"
    assert evidence.response_id == "chatcmpl-abc"
    assert evidence.provider_request_id == "req-123"
    assert evidence.response_model == "glm-5.3"
    assert evidence.stream_terminated_with_done is True
    assert evidence.stream_had_unterminated_event is False
    assert evidence.statistics.comment_count == 1
    assert transport.requests[0].timeout_seconds == 120.0

    assert sorted(path.name for path in config.output_dir.iterdir()) == [
        "api_evidence.json",
        "environment.json",
        "record.json",
        "response.txt",
    ]
    assert (config.output_dir / "response.txt").read_text(
        encoding="utf-8"
    ) == "Hello world"
    restored = ExperimentRecord.from_dict(
        json.loads((config.output_dir / "record.json").read_text(encoding="utf-8"))
    )
    assert restored.runtime.provider == "z.ai"
    assert restored.runtime.backend == "remote-http"


def test_time_to_first_token_ignores_empty_and_reasoning_only_chunks(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    result, _ = run(config, glm_stream())

    timeline = result.evidence.timeline
    kinds = [event.kind for event in timeline.events]
    assert kinds == ["metadata", "reasoning", "content", "content", "metadata"]

    first_content = next(event for event in timeline.events if event.kind == "content")
    assert timeline.first_content_token_offset_ms == first_content.offset_ms
    assert timeline.first_body_chunk_offset_ms is not None
    assert timeline.first_body_chunk_offset_ms < timeline.first_content_token_offset_ms
    # The role-only chunk carried content="" and must not count as a token.
    assert timeline.events[0].offset_ms < timeline.first_content_token_offset_ms
    assert result.record.timing.prefill is not None
    assert result.record.timing.prefill.value == timeline.first_content_token_offset_ms


def test_stream_split_across_arbitrary_byte_boundaries(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    body = b"".join(glm_stream(content_parts=("Ünï", "çodé 🙂")))
    fragments = [body[index : index + 3] for index in range(0, len(body), 3)]

    result, _ = run(config, fragments)

    assert result.response_text == "Ünïçodé 🙂"
    assert result.evidence.success is True
    assert result.evidence.statistics.content_characters == len("Ünïçodé 🙂")


def test_usage_delivered_in_a_separate_final_chunk(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "ok"}}]}),
        sse(
            {
                "id": "c1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        ),
        sse(
            {
                "id": "c1",
                "choices": [],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            }
        ),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    usage = result.evidence.usage
    assert usage.reported is True
    assert usage.prompt_tokens == 3
    assert usage.completion_tokens == 1
    assert usage.total_tokens is None
    assert usage.cached_prompt_tokens is None
    assert usage.reasoning_tokens is None
    assert result.evidence.finish_reason == "stop"


def test_absent_usage_stays_missing_and_is_never_inferred_as_zero(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "ok"}}]}),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    usage = result.evidence.usage
    assert usage.reported is False
    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert result.record.tokens.input_tokens is None
    assert result.record.tokens.generated_tokens is None
    assert result.record.tokens.provenance is None
    statistics = result.evidence.statistics
    assert statistics.provider_completion_tokens_per_second is None


def test_provider_usage_is_labelled_provider_reported(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    result, _ = run(config, glm_stream())

    assert result.record.tokens.provenance is MetricProvenance.PROVIDER_REPORTED
    assert result.record.tokens.input_tokens == 17
    assert result.record.tokens.generated_tokens == 5
    assert result.evidence.usage.cached_prompt_tokens == 4
    payload = json.loads(
        (config.output_dir / "api_evidence.json").read_text(encoding="utf-8")
    )
    assert payload["usage"]["provenance"] == "provider_reported"
    assert payload["timeline"]["provenance"] == "measured_wall_clock"


def test_reasoning_tokens_are_captured_when_the_provider_reports_them(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    usage = {
        "prompt_tokens": 4,
        "completion_tokens": 9,
        "total_tokens": 13,
        "completion_tokens_details": {"reasoning_tokens": 6},
    }

    result, _ = run(config, glm_stream(usage=usage))

    assert result.evidence.usage.reasoning_tokens == 6


def test_malformed_usage_values_are_recorded_not_silently_dropped(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    usage = {
        "prompt_tokens": "seventeen",
        "completion_tokens": -3,
        "total_tokens": 22,
        "prompt_tokens_details": "not-an-object",
    }

    result, _ = run(config, glm_stream(usage=usage))

    reported = result.evidence.usage
    assert reported.prompt_tokens is None
    assert reported.completion_tokens is None
    assert reported.total_tokens == 22
    assert set(reported.malformed_fields) == {
        "prompt_tokens",
        "completion_tokens",
        "prompt_tokens_details",
    }


def test_reasoning_text_is_counted_but_never_persisted(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    secret_thought = "internal chain of thought that must not be stored"

    result, _ = run(config, glm_stream(reasoning_parts=(secret_thought,)))

    assert result.evidence.reasoning_content_returned is True
    assert result.evidence.statistics.reasoning_delta_count == 1
    assert result.evidence.statistics.reasoning_characters == len(secret_thought)
    payload = json.loads(
        (config.output_dir / "api_evidence.json").read_text(encoding="utf-8")
    )
    assert payload["reasoning_text_persisted"] is False
    for artifact in sorted(config.output_dir.iterdir()):
        assert secret_thought not in artifact.read_text(encoding="utf-8")
    assert secret_thought not in result.response_text


def test_reasoning_delta_alias_is_also_recognized(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    chunks = [
        sse(
            {"id": "c1", "choices": [{"index": 0, "delta": {"reasoning": "thinking"}}]}
        ),
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "answer"}}]}),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    assert result.evidence.statistics.reasoning_delta_count == 1
    assert result.response_text == "answer"


def test_inter_token_latency_is_derived_from_content_deltas_only(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    result, _ = run(config, glm_stream(content_parts=("a", "b", "c")))

    statistics = result.evidence.statistics
    assert statistics.content_delta_count == 3
    distribution = statistics.inter_content_delta
    assert distribution is not None
    assert distribution.count == 2
    assert distribution.min_ms > 0
    assert statistics.content_delta_rate_per_second is not None
    payload = json.loads(
        (config.output_dir / "api_evidence.json").read_text(encoding="utf-8")
    )
    assert payload["statistics"]["content_delta_rate_provenance"] == "derived"


def test_stream_without_a_done_sentinel_is_recorded_honestly(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "partial"}}]})
    ]

    result, _ = run(config, chunks)

    assert result.evidence.success is True
    assert result.evidence.stream_terminated_with_done is False
    assert result.response_text == "partial"


def test_unterminated_final_event_is_flagged(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    chunks = [
        b'data: {"id": "c1", "choices": [{"index": 0, "delta": {"content": "x"}}]}\n\n',
        b"data: [DONE]",
    ]

    result, _ = run(config, chunks)

    assert result.evidence.stream_terminated_with_done is True
    assert result.evidence.stream_had_unterminated_event is True


def test_response_is_closed_even_though_evidence_is_kept(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    response = FakeResponse(glm_stream())

    collect_openai_stream(
        config, transport=FakeTransport(response), environ=ENVIRON, clock=StepClock()
    )

    assert response.closed is True


# --- Failure evidence --------------------------------------------------------


def assert_failure_artifacts(
    config: APICollectionConfig, category: str
) -> dict[str, Any]:
    payload = json.loads(
        (config.output_dir / "api_evidence.json").read_text(encoding="utf-8")
    )
    assert payload["success"] is False
    assert payload["failure"]["category"] == category
    record = ExperimentRecord.from_dict(
        json.loads((config.output_dir / "record.json").read_text(encoding="utf-8"))
    )
    assert record.outcome.success is False
    assert record.error is not None
    assert record.error.category == category
    return payload


@pytest.mark.parametrize(
    ("status_code", "body", "expected_code", "expected_fragment"),
    [
        (
            400,
            json.dumps(
                {"code": 1210, "message": "reasoning_effort is invalid"}
            ).encode(),
            "1210",
            "reasoning_effort is invalid",
        ),
        (
            401,
            json.dumps(
                {"error": {"code": "invalid_api_key", "message": "bad key"}}
            ).encode(),
            "invalid_api_key",
            "bad key",
        ),
        (500, b"upstream exploded", None, "upstream exploded"),
        (503, b"", None, "HTTP 503"),
    ],
)
def test_http_error_bodies_become_failure_evidence(
    tmp_path: Path,
    status_code: int,
    body: bytes,
    expected_code: str | None,
    expected_fragment: str,
) -> None:
    config = make_config(tmp_path)

    result, _ = run(config, FakeResponse([body], status_code=status_code))

    failure = result.evidence.failure
    assert failure is not None
    assert failure.category == FAILURE_HTTP_STATUS
    assert failure.status_code == status_code
    assert failure.provider_error_code == expected_code
    assert expected_fragment in failure.message
    assert result.record.outcome.success is False
    assert_failure_artifacts(config, FAILURE_HTTP_STATUS)


def test_rate_limit_headers_and_request_id_are_captured_on_429(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    response = FakeResponse(
        [json.dumps({"code": 1302, "message": "too many requests"}).encode()],
        status_code=429,
        headers={
            "X-Request-Id": "req-from-header",
            "X-RateLimit-Limit-Requests": "60",
            "X-RateLimit-Remaining-Requests": "0",
            "Retry-After": "12",
            "Content-Type": "application/json",
        },
    )

    result, _ = run(config, response)

    evidence = result.evidence
    assert evidence.failure is not None
    assert evidence.failure.status_code == 429
    assert evidence.provider_request_id == "req-from-header"
    assert evidence.rate_limit_headers == {
        "x-ratelimit-limit-requests": "60",
        "x-ratelimit-remaining-requests": "0",
        "retry-after": "12",
    }
    assert "content-type" not in evidence.rate_limit_headers


def test_redirects_are_not_followed_and_surface_as_status_failures(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    response = FakeResponse(
        [b""], status_code=302, headers={"Location": "https://elsewhere"}
    )

    result, _ = run(config, response)

    assert result.evidence.failure is not None
    assert result.evidence.failure.status_code == 302


def test_connection_failure_is_failure_shaped(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    result, _ = run(config, TransportConnectionError("dns lookup failed"))

    assert result.evidence.failure is not None
    assert result.evidence.failure.category == FAILURE_CONNECTION
    assert result.evidence.failure.status_code is None
    assert result.evidence.timeline.first_content_token_offset_ms is None
    assert result.record.timing.prefill is None
    assert_failure_artifacts(config, FAILURE_CONNECTION)


def test_timeout_before_headers_is_failure_shaped(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    result, _ = run(config, TransportTimeout("request timed out"))

    assert result.evidence.failure is not None
    assert result.evidence.failure.category == FAILURE_TIMEOUT
    assert result.evidence.timeline.response_headers_offset_ms is None
    assert_failure_artifacts(config, FAILURE_TIMEOUT)


def test_timeout_mid_stream_keeps_partial_evidence(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    response = FakeResponse(
        [sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "half"}}]})],
        raise_after=TransportTimeout("read timed out"),
    )

    result, _ = run(config, response)

    assert result.evidence.failure is not None
    assert result.evidence.failure.category == FAILURE_TIMEOUT
    assert result.evidence.statistics.content_delta_count == 1
    assert result.response_text == "half"
    assert result.record.outcome.success is False
    assert_failure_artifacts(config, FAILURE_TIMEOUT)


@pytest.mark.parametrize(
    ("chunk", "fragment"),
    [
        (b"data: {not json}\n\n", "not valid JSON"),
        (b"data: [1, 2, 3]\n\n", "not a JSON object"),
        (b'data: {"choices": [{"delta": {"content": 5}}]}\n\n', "delta.content"),
        (
            b'data: {"choices": [{"delta": ["nope"]}]}\n\n',
            "'delta' is not a JSON object",
        ),
        (
            b'data: {"choices": [{"delta": {}, "finish_reason": 7}]}\n\n',
            "finish_reason",
        ),
        (b'data: {"choices": [], "usage": 5}\n\n', "'usage' is not a JSON object"),
        (
            b'data: {"choices": [{"delta": {"reasoning_content": 5}}]}\n\n',
            "reasoning delta is not a string",
        ),
    ],
)
def test_malformed_stream_payloads_become_decode_failures(
    tmp_path: Path, chunk: bytes, fragment: str
) -> None:
    config = make_config(tmp_path)

    result, _ = run(config, [chunk])

    failure = result.evidence.failure
    assert failure is not None
    assert failure.category == FAILURE_STREAM_DECODE
    assert fragment in failure.message
    assert_failure_artifacts(config, FAILURE_STREAM_DECODE)


def test_invalid_utf8_in_the_stream_becomes_a_decode_failure(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    result, _ = run(config, [b'data: {"a": "\xff\xfe"}\n\n'])

    assert result.evidence.failure is not None
    assert result.evidence.failure.category == FAILURE_STREAM_DECODE


@pytest.mark.parametrize(
    ("payload", "expected_code"),
    [
        (
            {"error": {"code": "server_error", "message": "backend unavailable"}},
            "server_error",
        ),
        ({"code": 1113, "message": "account balance exhausted"}, "1113"),
    ],
)
def test_provider_error_inside_a_200_stream_is_failure_shaped(
    tmp_path: Path, payload: dict[str, Any], expected_code: str
) -> None:
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "partial"}}]}),
        sse(payload),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    failure = result.evidence.failure
    assert failure is not None
    assert failure.category == FAILURE_PROVIDER_ERROR
    assert failure.provider_error_code == expected_code
    assert failure.status_code is None
    assert result.evidence.stream_terminated_with_done is False
    assert_failure_artifacts(config, FAILURE_PROVIDER_ERROR)


def test_stream_without_any_content_is_a_missing_content_failure(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"role": "assistant"}}]}),
        sse(
            {
                "id": "c1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
            }
        ),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    failure = result.evidence.failure
    assert failure is not None
    assert failure.category == FAILURE_MISSING_CONTENT
    assert "length" in failure.message
    assert result.evidence.finish_reason == "length"
    assert_failure_artifacts(config, FAILURE_MISSING_CONTENT)


def test_reasoning_only_stream_is_a_missing_content_failure(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    chunks = [
        sse(
            {
                "id": "c1",
                "choices": [{"index": 0, "delta": {"reasoning_content": "hmm"}}],
            }
        ),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    assert result.evidence.failure is not None
    assert result.evidence.failure.category == FAILURE_MISSING_CONTENT


def test_transport_is_only_opened_once_because_there_are_no_retries(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    _, transport = run(config, FakeResponse([b""], status_code=500))

    assert len(transport.requests) == 1


def test_evidence_numbers_are_finite_and_json_serializable(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    result, _ = run(config, glm_stream())

    # ``allow_nan=False`` makes a non-finite value raise instead of emitting
    # the non-standard ``NaN``/``Infinity`` tokens.
    json.dumps(result.evidence.to_dict(), allow_nan=False)
    json.loads(result.record.to_json())


def test_a_non_finite_measurement_is_refused_rather_than_persisted(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    def broken_clock() -> float:
        return float("nan")

    with pytest.raises(OpenAIStreamCollectorError, match="non-finite measurement"):
        collect_openai_stream(
            config,
            transport=FakeTransport(FakeResponse(glm_stream())),
            environ=ENVIRON,
            clock=broken_clock,
        )

    assert not config.output_dir.exists()
