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
    _MAX_PERSISTED_HEADER_CHARS,
    ARTIFACT_MANIFEST_NAME,
    FAILURE_CONNECTION,
    FAILURE_HTTP_STATUS,
    FAILURE_MISSING_CONTENT,
    FAILURE_PROVIDER_ERROR,
    FAILURE_STREAM_DECODE,
    FAILURE_STREAM_TRUNCATED,
    FAILURE_TIMEOUT,
    APICollectionConfig,
    HTTPRequest,
    OpenAIStreamCollectorError,
    ProviderExtensions,
    TransportConnectionError,
    TransportTimeout,
    _safe_endpoint_for_message,
    artifact_set_is_complete,
    build_request_plan,
    collect_openai_stream,
    redact_text_for_dry_run,
)
from llmtracefx.optimizer.schema import ExperimentRecord, MetricProvenance

ENDPOINT = "https://api.z.ai/api/paas/v4/chat/completions"
API_KEY = "test-key-not-a-real-credential"
# Shortest credential prefix whose appearance in an artifact is a real leak
# rather than an incidental character match.
_MIN_LEAKED_PREFIX_CHARS = 8
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
        "artifacts.json",
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


def test_both_rates_share_one_content_window(tmp_path: Path) -> None:
    """The delta rate and the token rate must use the same denominator.

    Anchoring the token rate on the last event of any kind would fold the
    trailing usage, finish-reason and ``[DONE]`` events into a decode
    window, making the two published rates incomparable.
    """
    config = make_config(tmp_path)
    usage = {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17}

    result, _ = run(config, glm_stream(content_parts=("a", "b", "c", "d"), usage=usage))

    statistics = result.evidence.statistics
    timeline = result.evidence.timeline
    content_offsets = [
        event.offset_ms for event in timeline.events if event.kind == "content"
    ]
    expected_window = content_offsets[-1] - content_offsets[0]

    assert statistics.content_window_ms == pytest.approx(expected_window)
    # Three gaps across four arrivals, over the same window.
    assert statistics.content_delta_rate_per_second == pytest.approx(
        3 / (expected_window / 1000)
    )
    assert statistics.provider_completion_tokens_per_second == pytest.approx(
        12 / (expected_window / 1000)
    )


def test_the_content_window_ignores_trailing_usage_and_done_events(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    usage = {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17}

    result, _ = run(config, glm_stream(content_parts=("a", "b", "c", "d"), usage=usage))

    statistics = result.evidence.statistics
    timeline = result.evidence.timeline
    assert statistics.content_window_ms is not None
    # The stream keeps running after the last content delta, so a window
    # that reached the end of the stream would be strictly wider.
    assert timeline.completed_offset_ms > statistics.content_window_ms
    assert timeline.last_event_offset_ms is not None
    assert timeline.last_event_offset_ms > statistics.content_window_ms


def test_a_single_content_delta_yields_no_window_and_no_rates(
    tmp_path: Path,
) -> None:
    """One arrival bounds zero intervals, so no rate is observable."""
    config = make_config(tmp_path)
    usage = {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17}

    result, _ = run(config, glm_stream(content_parts=("only",), usage=usage))

    statistics = result.evidence.statistics
    assert statistics.content_delta_count == 1
    assert statistics.content_window_ms is None
    assert statistics.content_delta_rate_per_second is None
    assert statistics.provider_completion_tokens_per_second is None
    assert statistics.inter_content_delta is None


def test_the_content_window_is_persisted_with_its_definition(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    run(config, glm_stream(content_parts=("a", "b")))

    payload = json.loads(
        (config.output_dir / "api_evidence.json").read_text(encoding="utf-8")
    )
    statistics = payload["statistics"]
    assert statistics["content_window_ms"] is not None
    assert "last content delta arrival" in statistics["content_window_definition"]
    note = statistics["provider_completion_tokens_per_second_note"]
    assert "content_window_ms" in note
    assert "coarse estimate" in note


def test_stream_without_a_done_sentinel_is_recorded_honestly(tmp_path: Path) -> None:
    """A stream cut short mid-answer is a failure, not a short success.

    Content alone is not evidence that the answer is whole, so without
    ``[DONE]`` or a terminal ``finish_reason`` the run is failure-shaped
    and the partial text is preserved for inspection rather than
    published as the model's answer.
    """
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "partial"}}]})
    ]

    result, _ = run(config, chunks)

    assert result.evidence.success is False
    assert result.evidence.failure is not None
    assert result.evidence.failure.category == FAILURE_STREAM_TRUNCATED
    assert result.evidence.stream_terminated_with_done is False
    assert result.response_text == "partial"
    assert_failure_artifacts(config, FAILURE_STREAM_TRUNCATED)


def test_a_terminal_finish_reason_terminates_a_stream_without_done(
    tmp_path: Path,
) -> None:
    """Not every OpenAI-compatible provider sends the ``[DONE]`` sentinel."""
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "whole"}}]}),
        sse(
            {
                "id": "c1",
                "choices": [
                    {"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}
                ],
            }
        ),
    ]

    result, _ = run(config, chunks)

    assert result.evidence.success is True
    assert result.evidence.stream_terminated_with_done is False
    assert result.evidence.finish_reason == "stop"


def test_a_non_terminal_finish_reason_does_not_terminate_a_stream(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "partial"}}]}),
        sse(
            {
                "id": "c1",
                "choices": [
                    {"index": 0, "delta": {"content": ""}, "finish_reason": "null"}
                ],
            }
        ),
    ]

    result, _ = run(config, chunks)

    assert result.evidence.success is False
    assert result.evidence.failure is not None
    assert result.evidence.failure.category == FAILURE_STREAM_TRUNCATED


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


# --- Provider-controlled string redaction ------------------------------------


def test_a_provider_echoing_the_credential_everywhere_leaks_nothing(
    tmp_path: Path,
) -> None:
    """A hostile or misconfigured provider must not get the key onto disk.

    Every provider-controlled string is echoed back containing the
    credential: the response id, the request id, the model name, the
    finish reason and the generated text itself. None of them may survive
    into any artifact.
    """
    config = make_config(tmp_path)
    chunks = [
        sse(
            {
                "id": f"resp-{API_KEY}",
                "request_id": f"req-{API_KEY}",
                "model": f"glm-5.3-{API_KEY}",
                "choices": [{"index": 0, "delta": {"content": f"key is {API_KEY}!"}}],
            }
        ),
        sse(
            {
                "id": f"resp-{API_KEY}",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": ""},
                        "finish_reason": f"stop-{API_KEY}",
                    }
                ],
                "usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7,
                },
            }
        ),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    for path in sorted(config.output_dir.iterdir()):
        assert API_KEY not in path.read_text(encoding="utf-8"), path.name
    assert API_KEY not in result.response_text
    assert "[REDACTED]" in result.response_text
    assert result.evidence.response_id is not None
    assert API_KEY not in result.evidence.response_id
    assert result.evidence.provider_request_id is not None
    assert API_KEY not in result.evidence.provider_request_id
    assert result.evidence.response_model is not None
    assert API_KEY not in result.evidence.response_model
    assert result.evidence.finish_reason is not None
    assert API_KEY not in result.evidence.finish_reason


def test_a_credential_split_across_deltas_still_never_reaches_disk(
    tmp_path: Path,
) -> None:
    """Per-delta scrubbing alone is not enough.

    A provider that wants the key back can dribble it out a few characters
    per SSE delta, so no single delta ever contains it while the assembled
    answer does. The scrub therefore has to run on the joined text.
    """
    config = make_config(tmp_path)
    fragments = [API_KEY[i : i + 4] for i in range(0, len(API_KEY), 4)]
    assert len(fragments) > 2
    for fragment in fragments:
        assert fragment not in ("", API_KEY)

    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": fragment}}]})
        for fragment in fragments
    ]
    chunks.append(
        sse(
            {
                "id": "c1",
                "choices": [
                    {"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}
                ],
            }
        )
    )
    chunks.append(b"data: [DONE]\n\n")

    result, _ = run(config, chunks)

    assert API_KEY not in result.response_text
    for path in sorted(config.output_dir.iterdir()):
        assert API_KEY not in path.read_text(encoding="utf-8"), path.name


def test_persisted_content_length_matches_the_scrubbed_response(
    tmp_path: Path,
) -> None:
    """The recorded character count must describe what was actually written."""
    config = make_config(tmp_path)
    fragments = [API_KEY[i : i + 4] for i in range(0, len(API_KEY), 4)]
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": fragment}}]})
        for fragment in fragments
    ]
    chunks.append(
        sse(
            {
                "id": "c1",
                "choices": [
                    {"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}
                ],
            }
        )
    )
    chunks.append(b"data: [DONE]\n\n")

    result, _ = run(config, chunks)

    written = (config.output_dir / "response.txt").read_text(encoding="utf-8")
    assert result.response_text == written
    assert result.evidence.statistics.content_characters == len(written)


def test_redacting_generated_text_preserves_shape(tmp_path: Path) -> None:
    """Scrubbing the answer must not collapse it into a single line."""
    config = make_config(tmp_path)
    chunks = [
        sse(
            {"id": "c1", "choices": [{"index": 0, "delta": {"content": "line one\n"}}]}
        ),
        sse(
            {"id": "c1", "choices": [{"index": 0, "delta": {"content": "  line two"}}]}
        ),
        sse(
            {
                "id": "c1",
                "choices": [
                    {"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}
                ],
            }
        ),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    assert result.response_text == "line one\n  line two"


def test_a_credential_echoed_in_a_rate_limit_header_is_redacted(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    response = FakeResponse(
        list(glm_stream()),
        headers={
            "content-type": "text/event-stream",
            "x-ratelimit-remaining-requests": f"{API_KEY}",
            "x-request-id": f"hdr-{API_KEY}",
        },
    )

    result, _ = run(config, response)

    for path in sorted(config.output_dir.iterdir()):
        assert API_KEY not in path.read_text(encoding="utf-8"), path.name
    for value in result.evidence.rate_limit_headers.values():
        assert API_KEY not in value


# --- Named error events and malformed choices --------------------------------


def test_a_named_error_event_carrying_json_is_a_failure(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    chunks = [
        b'event: error\ndata: {"code": "1301", "message": "content blocked"}\n\n',
    ]

    result, _ = run(config, chunks)

    payload = assert_failure_artifacts(config, FAILURE_PROVIDER_ERROR)
    assert "content blocked" in payload["failure"]["message"]
    assert result.evidence.failure is not None
    assert result.evidence.failure.provider_error_code == "1301"


def test_a_named_error_event_carrying_a_bare_message_is_a_failure(
    tmp_path: Path,
) -> None:
    """Not every provider wraps an error event payload in JSON."""
    config = make_config(tmp_path)
    chunks = [b"event: error\ndata: upstream capacity exceeded\n\n"]

    run(config, chunks)

    payload = assert_failure_artifacts(config, FAILURE_PROVIDER_ERROR)
    assert "upstream capacity exceeded" in payload["failure"]["message"]


def test_a_named_error_event_with_an_empty_payload_is_a_failure(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    chunks = [b"event: error\ndata:\n\n"]

    run(config, chunks)

    payload = assert_failure_artifacts(config, FAILURE_PROVIDER_ERROR)
    assert "empty payload" in payload["failure"]["message"]


def test_a_named_error_event_after_partial_content_is_a_failure(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "half"}}]}),
        b'event: error\ndata: {"code": 500, "message": "upstream died"}\n\n',
    ]

    result, _ = run(config, chunks)

    assert_failure_artifacts(config, FAILURE_PROVIDER_ERROR)
    assert result.response_text == "half"


@pytest.mark.parametrize(
    "choices",
    [
        pytest.param({}, id="object"),
        pytest.param([7], id="list-of-scalar"),
        pytest.param("stop", id="string"),
        pytest.param(3, id="number"),
    ],
)
def test_a_present_but_malformed_choices_is_a_decode_failure(
    tmp_path: Path, choices: Any
) -> None:
    """``{"choices": {}}`` is a broken frame, not a metadata frame."""
    config = make_config(tmp_path)
    chunks = [sse({"id": "c1", "choices": choices})]

    run(config, chunks)

    payload = assert_failure_artifacts(config, FAILURE_STREAM_DECODE)
    assert "malformed" in payload["failure"]["message"]


def test_a_malformed_choices_after_partial_content_is_not_a_success(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "half"}}]}),
        sse({"id": "c1", "choices": {}}),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    assert_failure_artifacts(config, FAILURE_STREAM_DECODE)
    assert result.response_text == "half"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"id": "c1"}, id="absent"),
        pytest.param({"id": "c1", "choices": []}, id="empty-list"),
        pytest.param({"id": "c1", "choices": None}, id="null"),
    ],
)
def test_an_absent_or_empty_choices_stays_a_metadata_frame(
    tmp_path: Path, payload: dict[str, Any]
) -> None:
    """GLM's usage-only final chunk legitimately carries no choices."""
    config = make_config(tmp_path)
    chunks = [
        sse(payload),
        *glm_stream(),
    ]

    result, _ = run(config, chunks)

    assert result.evidence.success is True
    assert result.evidence.statistics.metadata_event_count >= 1


# --- Config identity ---------------------------------------------------------


def test_query_values_change_the_config_hash(tmp_path: Path) -> None:
    """Two API versions on one endpoint must not share an identity."""
    base = "https://example.test/v1/chat/completions"
    first = build_request_plan(
        make_config(tmp_path, endpoint=f"{base}?api-version=2024-01")
    )
    second = build_request_plan(
        make_config(tmp_path, endpoint=f"{base}?api-version=2025-06")
    )

    assert first.config_hash != second.config_hash
    assert first.endpoint_query_keys == second.endpoint_query_keys


def test_an_identical_query_keeps_a_stable_config_hash(tmp_path: Path) -> None:
    endpoint = "https://example.test/v1/chat/completions?api-version=2024-01"
    first = build_request_plan(make_config(tmp_path, endpoint=endpoint))
    second = build_request_plan(make_config(tmp_path, endpoint=endpoint))

    assert first.config_hash == second.config_hash


def test_raw_query_values_are_never_persisted_in_the_plan(tmp_path: Path) -> None:
    config = make_config(
        tmp_path,
        endpoint="https://example.test/v1/chat?deployment=super-secret-deployment",
        command_argv=(
            "llmtracefx-optimizer",
            "collect-api",
            "--endpoint",
            "https://example.test/v1/chat?deployment=super-secret-deployment",
        ),
    )

    plan = build_request_plan(config)
    rendered = plan.to_json()

    assert "super-secret-deployment" not in rendered
    assert plan.endpoint_query_keys == ("deployment",)
    assert "[REDACTED]" in " ".join(plan.command)


# --- Artifact set publication ------------------------------------------------


def test_a_successful_run_publishes_a_complete_artifact_set(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    run(config, glm_stream())

    assert artifact_set_is_complete(config.output_dir) is True
    marker = json.loads(
        (config.output_dir / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8")
    )
    assert marker["run_id"] == "api-run"
    assert sorted(entry["name"] for entry in marker["artifacts"]) == [
        "api_evidence.json",
        "environment.json",
        "record.json",
        "response.txt",
    ]


def test_an_artifact_replaced_independently_fails_the_completeness_check(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    run(config, glm_stream())
    assert artifact_set_is_complete(config.output_dir) is True

    (config.output_dir / "response.txt").write_text("tampered", encoding="utf-8")

    assert artifact_set_is_complete(config.output_dir) is False


def test_a_missing_marker_fails_the_completeness_check(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run(config, glm_stream())

    (config.output_dir / ARTIFACT_MANIFEST_NAME).unlink()

    assert artifact_set_is_complete(config.output_dir) is False


def test_a_failed_late_write_leaves_no_complete_marker_beside_stale_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure the marker exists to catch.

    A first run publishes a complete set. A second run crashes after
    record.json is replaced, so the directory now holds a new record next
    to the previous run's evidence. That set must not read as complete.
    """
    config = make_config(tmp_path)
    run(config, glm_stream(content_parts=("first",)))
    assert artifact_set_is_complete(config.output_dir) is True
    first_evidence = (config.output_dir / "api_evidence.json").read_text(
        encoding="utf-8"
    )

    import llmtracefx.optimizer.collectors.openai_api as module

    real_write = module.atomic_write_text
    calls: list[str] = []

    def failing_write(path: Path, text: str) -> None:
        calls.append(path.name)
        if path.name == "api_evidence.json":
            raise OSError("disk full")
        real_write(path, text)

    monkeypatch.setattr(module, "atomic_write_text", failing_write)

    with pytest.raises(OSError, match="disk full"):
        run(config, glm_stream(content_parts=("second",)))

    assert "record.json" in calls
    assert artifact_set_is_complete(config.output_dir) is False
    assert not (config.output_dir / ARTIFACT_MANIFEST_NAME).exists()
    # The new record really is sitting next to the old evidence, which is
    # exactly why the set has to be rejected rather than trusted.
    assert (config.output_dir / "api_evidence.json").read_text(
        encoding="utf-8"
    ) == first_evidence


# --- Second review pass -------------------------------------------------------


def test_the_persisted_command_never_carries_raw_query_values(
    tmp_path: Path,
) -> None:
    """record.json must use the sanitized command, like the plan does.

    The plan already strips query values from the reconstructed command.
    Rebuilding the record's argv from the raw config put them straight
    back into record.json, which is the artifact most likely to be shared.
    """
    endpoint = f"{ENDPOINT}?deployment=private-value"
    config = make_config(
        tmp_path,
        endpoint=endpoint,
        command_argv=(
            "llmtracefx-optimizer",
            "collect-api",
            "--endpoint",
            endpoint,
            "--api-key-env",
            "ZAI_API_KEY",
        ),
    )

    result, _ = run(config, glm_stream())

    assert "private-value" not in json.dumps(result.record.to_dict())
    for path in sorted(config.output_dir.iterdir()):
        assert "private-value" not in path.read_text(encoding="utf-8"), path.name
    assert "deployment" in " ".join(result.record.command.argv)


@pytest.mark.parametrize(
    "encoded",
    [
        "sk-slash%2Fcredential",
        "sk-slash%2fcredential",
        "sk-slash%252Fcredential",
    ],
    ids=["upper-hex", "lower-hex", "double-encoded"],
)
def test_a_percent_encoded_credential_in_the_endpoint_is_refused(
    tmp_path: Path, encoded: str
) -> None:
    """Percent encoding must not defeat the pre-flight refusal.

    A credential pasted into a URL is normally encoded, and the encoded
    form is trivially reversible once persisted, so containment has to be
    checked against the decoded representations too.
    """
    credential = "sk-slash/credential"
    config = make_config(
        tmp_path, endpoint=f"https://api.z.ai/v1/{encoded}/completions"
    )

    with pytest.raises(OpenAIStreamCollectorError) as excinfo:
        collect_openai_stream(
            config,
            transport=ExplodingTransport(),
            environ={"ZAI_API_KEY": credential},
            clock=StepClock(),
        )

    assert "appears in endpoint" in str(excinfo.value)
    assert credential not in str(excinfo.value)
    assert not config.output_dir.exists()


def test_a_percent_encoded_credential_in_the_query_is_refused(tmp_path: Path) -> None:
    credential = "sk-slash/credential"
    config = make_config(
        tmp_path, endpoint=f"{ENDPOINT}?deployment=sk-slash%2Fcredential"
    )

    with pytest.raises(OpenAIStreamCollectorError) as excinfo:
        collect_openai_stream(
            config,
            transport=ExplodingTransport(),
            environ={"ZAI_API_KEY": credential},
            clock=StepClock(),
        )

    assert "appears in endpoint" in str(excinfo.value)


def test_an_unrelated_encoded_endpoint_is_not_falsely_refused(tmp_path: Path) -> None:
    """Decoding must not turn ordinary escaped URLs into refusals."""
    config = make_config(tmp_path, endpoint=f"{ENDPOINT}?filter=a%2Fb&q=one+two")

    result, _ = run(config, glm_stream())

    assert result.record.outcome.success is True


def test_a_short_credential_does_not_trigger_encoded_false_positives(
    tmp_path: Path,
) -> None:
    """A tiny value would match a decoded byte by coincidence."""
    config = make_config(tmp_path, endpoint=f"{ENDPOINT}?path=a%2Fb")

    result, _ = run(config, glm_stream(), environ={"ZAI_API_KEY": "a/b"})

    assert result.record.outcome.success is True


def test_an_http_error_code_echoing_the_credential_is_redacted(
    tmp_path: Path,
) -> None:
    """The non-200 path builds its own failure and must redact it too."""
    config = make_config(tmp_path)
    body = json.dumps(
        {"error": {"code": f"invalid_key_{API_KEY}", "message": f"bad {API_KEY}"}}
    ).encode()

    result, _ = run(config, FakeResponse([body], status_code=401))

    failure = result.evidence.failure
    assert failure is not None
    assert failure.category == FAILURE_HTTP_STATUS
    assert failure.provider_error_code is not None
    assert API_KEY not in failure.provider_error_code
    assert API_KEY not in failure.message
    for path in sorted(config.output_dir.iterdir()):
        assert API_KEY not in path.read_text(encoding="utf-8"), path.name


def test_a_bare_zai_error_code_echoing_the_credential_is_redacted(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    body = json.dumps({"code": API_KEY, "message": "rate limited"}).encode()

    result, _ = run(config, FakeResponse([body], status_code=429))

    failure = result.evidence.failure
    assert failure is not None
    assert failure.provider_error_code is not None
    assert API_KEY not in failure.provider_error_code
    for path in sorted(config.output_dir.iterdir()):
        assert API_KEY not in path.read_text(encoding="utf-8"), path.name


def test_an_error_event_carrying_done_is_not_a_clean_termination(
    tmp_path: Path,
) -> None:
    """The event name must be resolved before its data is interpreted.

    Handling ``[DONE]`` first let a provider close a failed stream as
    though it had finished normally, producing a success record from an
    explicit error frame.
    """
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "partial"}}]}),
        b"event: error\ndata: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    assert result.record.outcome.success is False
    failure = result.evidence.failure
    assert failure is not None
    assert failure.category == FAILURE_PROVIDER_ERROR
    assert "[DONE]" in failure.message


def test_an_error_event_carrying_done_as_the_first_event_is_a_failure(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    result, _ = run(config, [b"event: error\ndata: [DONE]\n\n"])

    assert result.record.outcome.success is False
    assert result.evidence.failure is not None
    assert result.evidence.failure.category == FAILURE_PROVIDER_ERROR


def test_repeated_query_keys_keep_their_value_order_in_the_hash(
    tmp_path: Path,
) -> None:
    """``?a=1&a=2`` and ``?a=2&a=1`` are not necessarily the same request."""
    first = build_request_plan(make_config(tmp_path, endpoint=f"{ENDPOINT}?a=1&a=2"))
    second = build_request_plan(make_config(tmp_path, endpoint=f"{ENDPOINT}?a=2&a=1"))

    assert first.config_hash != second.config_hash


def test_distinct_query_keys_are_order_insensitive_in_the_hash(
    tmp_path: Path,
) -> None:
    """The order of distinct keys carries no meaning in a query string."""
    first = build_request_plan(make_config(tmp_path, endpoint=f"{ENDPOINT}?a=1&b=2"))
    second = build_request_plan(make_config(tmp_path, endpoint=f"{ENDPOINT}?b=2&a=1"))

    assert first.config_hash == second.config_hash


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.z.ai:99999/v1/chat",
        "https://api.z.ai:notaport/v1/chat",
        "https://[unclosed/v1/chat",
    ],
    ids=["port-out-of-range", "port-not-an-integer", "malformed-ipv6"],
)
def test_a_malformed_endpoint_raises_a_sanitized_collector_error(
    tmp_path: Path, endpoint: str
) -> None:
    """Parsing failures must not escape as a raw ValueError.

    ``urlsplit`` raises on a malformed authority and ``SplitResult.port``
    raises on a bad port, both after the safe renderer has already run, so
    the guard has to cover the property access as well as the split.
    """
    with pytest.raises(OpenAIStreamCollectorError) as excinfo:
        make_config(tmp_path, endpoint=endpoint)

    message = str(excinfo.value)
    assert "/v1/chat" not in message
    assert "notaport" not in message
    assert "unclosed" not in message


def test_a_malformed_endpoint_does_not_leak_a_secret_path(tmp_path: Path) -> None:
    with pytest.raises(OpenAIStreamCollectorError) as excinfo:
        make_config(tmp_path, endpoint=f"https://api.z.ai:99999/v1/{API_KEY}/chat")

    assert API_KEY not in str(excinfo.value)


def test_the_safe_endpoint_renderer_never_raises() -> None:
    """It builds error messages, so raising would replace the diagnostic."""
    for endpoint in (
        "https://api.z.ai:99999/v1",
        "https://api.z.ai:notaport/v1",
        "https://[unclosed/v1",
        "not-a-url-at-all",
        "",
    ):
        assert isinstance(_safe_endpoint_for_message(endpoint), str)


def test_the_request_repr_does_not_expose_endpoint_query_values(
    tmp_path: Path,
) -> None:
    """The same rule the persisted command follows applies to a traceback.

    Query values are stripped from every artifact, so a debugger dump or
    an accidental log line must not be the one place that prints them.
    """
    endpoint = f"{ENDPOINT}?deployment=private-value"
    config = make_config(tmp_path, endpoint=endpoint)

    _, transport = run(config, glm_stream())

    rendered = repr(transport.requests[0])
    assert "private-value" not in rendered
    assert "deployment" in rendered
    assert API_KEY not in rendered
    assert transport.requests[0].url == endpoint


def test_a_long_request_id_header_cannot_leak_a_credential_prefix(
    tmp_path: Path,
) -> None:
    """Truncating a header before redacting it would slice the credential.

    Redaction matches the credential as an exact substring, so a value cut
    at the persistence limit mid credential leaves a prefix the scrub can no
    longer recognise. A provider, proxy or CDN that can set a response header
    chooses the padding, so it can position the boundary anywhere.
    """
    config = make_config(tmp_path)
    # Place the credential so the 128 character limit falls inside it.
    padding = "A" * (_MAX_PERSISTED_HEADER_CHARS - len(API_KEY) + 1)
    # No body level request_id, so the header supplies the persisted value.
    chunks = [
        sse({"choices": [{"index": 0, "delta": {"content": "Hi"}}]}),
        sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    ]
    response = FakeResponse(
        chunks,
        headers={
            "content-type": "text/event-stream",
            "x-request-id": f"{padding}{API_KEY}",
        },
    )

    result, _ = run(config, response)

    persisted = result.evidence.provider_request_id or ""
    assert len(persisted) <= _MAX_PERSISTED_HEADER_CHARS
    assert "[REDACTED]" in persisted
    prefixes = [API_KEY[:n] for n in range(_MIN_LEAKED_PREFIX_CHARS, len(API_KEY) + 1)]
    for prefix in prefixes:
        assert prefix not in persisted, f"{len(prefix)} credential characters survived"
    for path in sorted(config.output_dir.iterdir()):
        text = path.read_text(encoding="utf-8")
        for prefix in prefixes:
            assert prefix not in text, f"{path.name} leaked {len(prefix)} characters"


# --- Third review pass: transform-before-redact orderings ---------------------


def test_a_credential_echoed_in_a_rate_limit_header_name_is_redacted(
    tmp_path: Path,
) -> None:
    """Header names are provider controlled and were persisted as dict keys.

    The HTTP token alphabet covers the alphabet most API keys use, so a
    server can put the credential in the name rather than the value. Names
    are lowercased before persistence, so the lowered form has to be
    matched too or an uppercase key is written in reversible form.
    """
    config = make_config(tmp_path)
    response = FakeResponse(
        list(glm_stream()),
        headers={
            "content-type": "text/event-stream",
            f"X-RateLimit-{API_KEY}": "1",
            f"X-RateLimit-Remaining-{API_KEY.upper()}": "0",
        },
    )

    result, _ = run(config, response)

    for name, value in result.evidence.rate_limit_headers.items():
        assert API_KEY not in name
        assert API_KEY.lower() not in name
        assert API_KEY not in value
    for path in sorted(config.output_dir.iterdir()):
        text = path.read_text(encoding="utf-8")
        assert API_KEY not in text, path.name
        assert API_KEY.lower() not in text, path.name


@pytest.mark.parametrize("padding", [" ", "\n", "\t", "x"])
def test_an_error_body_cut_through_a_credential_does_not_leak_a_prefix(
    tmp_path: Path, padding: str
) -> None:
    """The body byte cap can slice an echoed credential in half.

    Whitespace padding makes this reachable: the redactor collapses runs of
    whitespace after scrubbing, which pulls the truncated tail back into the
    persisted window. The cut always lands at the end of the buffer, so the
    surviving fragment is a trailing prefix of the credential.
    """
    config = make_config(tmp_path)
    cap = 64 * 1024
    surviving = len(API_KEY) - 1
    body = (padding * (cap - surviving) + API_KEY).encode("utf-8")
    assert body[:cap].endswith(API_KEY[:surviving].encode("utf-8"))

    result, _ = run(config, FakeResponse([body], status_code=402))

    failure = result.evidence.failure
    assert failure is not None
    prefixes = [API_KEY[:n] for n in range(_MIN_LEAKED_PREFIX_CHARS, len(API_KEY) + 1)]
    for prefix in prefixes:
        assert prefix not in failure.message, f"{len(prefix)} characters survived"
    for path in sorted(config.output_dir.iterdir()):
        text = path.read_text(encoding="utf-8")
        for prefix in prefixes:
            assert prefix not in text, f"{path.name} leaked {len(prefix)} characters"


@pytest.mark.parametrize(
    "echoed",
    [
        "key with spaces here",
        "key\twith\tspaces\there",
        "key  with  spaces  here",
        "key\nwith\nspaces\nhere",
        "key \t with \t spaces \t here",
    ],
)
def test_a_credential_containing_spaces_survives_whitespace_normalization(
    tmp_path: Path, echoed: str
) -> None:
    """A space is a legal header value character, so it is a legal key.

    Diagnostics collapse whitespace after scrubbing, so a provider that
    echoes the key with tabs or doubled spaces would evade the exact match
    and then be normalized back into the credential.
    """
    credential = "key with spaces here"
    config = make_config(tmp_path)
    body = json.dumps({"error": {"message": echoed, "code": "1002"}}).encode("utf-8")

    result, _ = run(
        config,
        FakeResponse([body], status_code=401),
        environ={"ZAI_API_KEY": credential},
    )

    failure = result.evidence.failure
    assert failure is not None
    assert credential not in failure.message
    assert "[REDACTED]" in failure.message
    for path in sorted(config.output_dir.iterdir()):
        assert credential not in path.read_text(encoding="utf-8"), path.name


def test_an_uppercase_credential_echoed_in_a_header_name_is_redacted(
    tmp_path: Path,
) -> None:
    """Header names are lowercased before persistence.

    Matching only the literal credential would spare an uppercase key
    cosmetically while still writing the lowered form, which is fully
    reversible for the hex and base32 alphabets keys usually use.
    """
    credential = "SK-9F3A2B7C1D4E5F60ZAIKEY"
    config = make_config(tmp_path)
    response = FakeResponse(
        list(glm_stream()),
        headers={
            "content-type": "text/event-stream",
            f"X-RateLimit-{credential}": "1",
        },
    )

    result, _ = run(config, response, environ={"ZAI_API_KEY": credential})

    for name in result.evidence.rate_limit_headers:
        assert credential.lower() not in name
    for path in sorted(config.output_dir.iterdir()):
        assert credential.lower() not in path.read_text(encoding="utf-8"), path.name


def test_a_credential_with_doubled_spaces_is_redacted_from_response_text(
    tmp_path: Path,
) -> None:
    """``response.txt`` is scrubbed without whitespace normalization.

    Preserving the answer's own whitespace is correct, but it means a
    credential whose stored form has doubled spaces would not match an echo
    that uses single spaces unless the normalized form is matched too.
    """
    credential = "key  with  doubled  spaces"
    normalized = "key with doubled spaces"
    config = make_config(tmp_path)
    chunks = [
        sse({"choices": [{"index": 0, "delta": {"content": normalized}}]}),
        sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    ]

    run(config, FakeResponse(chunks), environ={"ZAI_API_KEY": credential})

    answer = (config.output_dir / "response.txt").read_text(encoding="utf-8")
    assert normalized not in answer
    assert "[REDACTED]" in answer
    for path in sorted(config.output_dir.iterdir()):
        text = path.read_text(encoding="utf-8")
        assert credential not in text, path.name
        assert normalized not in text, path.name


@pytest.mark.parametrize("keep", [29, 20, 12, 6])
def test_a_truncated_error_event_payload_does_not_leak_a_credential_prefix(
    tmp_path: Path, keep: int
) -> None:
    """An unterminated ``error`` event is dispatched with its raw data line.

    A complete data line parses as JSON and takes a safe branch, so the raw
    interpolation is reached mainly when the provider cut the payload. The
    cut can fall inside an echoed credential, and no oversized body or
    padding trick is needed: the provider just closes the connection at a
    chosen offset.
    """
    config = make_config(tmp_path)
    chunks = [
        b'event: error\ndata: {"error":{"message":"' + API_KEY[:keep].encode("utf-8")
    ]

    result, _ = run(config, FakeResponse(chunks))

    failure = result.evidence.failure
    assert failure is not None
    prefixes = [API_KEY[:n] for n in range(_MIN_LEAKED_PREFIX_CHARS, len(API_KEY) + 1)]
    for prefix in prefixes:
        assert prefix not in failure.message, f"{len(prefix)} characters survived"
    for path in sorted(config.output_dir.iterdir()):
        text = path.read_text(encoding="utf-8")
        for prefix in prefixes:
            assert prefix not in text, f"{path.name} leaked {len(prefix)} characters"


def test_content_cut_mid_credential_does_not_leak_into_the_response_file(
    tmp_path: Path,
) -> None:
    """A truncated run still persists ``response.txt``.

    The credential is dribbled across deltas and the stream is cut partway
    through the last one, so no delta and no assembled string contains the
    whole value. The assembled tail is a credential prefix, which only the
    boundary repair can see.
    """
    config = make_config(tmp_path)
    # Cut one character short so the assembled text is a prefix, not the key.
    partial = API_KEY[:-1]
    parts = [partial[i : i + 5] for i in range(0, len(partial), 5)]
    assert "".join(parts) != API_KEY
    chunks = [
        sse({"choices": [{"index": 0, "delta": {"content": part}}]}) for part in parts
    ]

    result, _ = run(config, FakeResponse(chunks))

    assert result.evidence.failure is not None
    answer = (config.output_dir / "response.txt").read_text(encoding="utf-8")
    prefixes = [API_KEY[:n] for n in range(_MIN_LEAKED_PREFIX_CHARS, len(API_KEY) + 1)]
    for prefix in prefixes:
        assert prefix not in answer, f"{len(prefix)} characters survived"
    for path in sorted(config.output_dir.iterdir()):
        text = path.read_text(encoding="utf-8")
        for prefix in prefixes:
            assert prefix not in text, f"{path.name} leaked {len(prefix)} characters"


@pytest.mark.parametrize(
    "echoed",
    [
        "ZQXJV\tKWPMB\tGHFDS\tRTNCL",
        "ZQXJV\nKWPMB\nGHFDS\nRTNCL",
        "ZQXJV  KWPMB  GHFDS  RTNCL",
        "ZQXJV \t KWPMB \t GHFDS \t RTNCL",
    ],
)
def test_response_text_redacts_a_whitespace_bearing_credential(
    tmp_path: Path, echoed: str
) -> None:
    """``response.txt`` preserves whitespace, so it cannot normalize first.

    Matching each internal whitespace run flexibly gives the answer sink the
    same coverage as a collapsed diagnostic without altering what is written.
    """
    credential = "ZQXJV KWPMB GHFDS RTNCL"
    config = make_config(tmp_path)
    chunks = [
        sse({"choices": [{"index": 0, "delta": {"content": f"key is {echoed}"}}]}),
        sse({"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    ]

    run(config, FakeResponse(chunks), environ={"ZAI_API_KEY": credential})

    answer = (config.output_dir / "response.txt").read_text(encoding="utf-8")
    assert "[REDACTED]" in answer
    assert "ZQXJV" not in " ".join(answer.split())
    for path in sorted(config.output_dir.iterdir()):
        assert "ZQXJV" not in path.read_text(encoding="utf-8"), path.name


@pytest.mark.parametrize("fold", ["lower", "upper", "swapcase"])
def test_a_case_folded_truncated_tail_is_repaired(tmp_path: Path, fold: str) -> None:
    """The boundary repair must be as strong as the whole-value scrub.

    Both guard the same threat. A literal, case sensitive prefix comparison
    would let any case folding defeat the repair while the scrub would have
    caught the same echo in full.
    """
    credential = "ZqXjVkWpMbGhFdSaTrNcLy0123"
    config = make_config(tmp_path)
    cap = 64 * 1024
    surviving = 20
    tail: str = getattr(credential[:surviving], fold)()
    body = (" " * (cap - surviving) + tail).encode("utf-8")

    result, _ = run(
        config,
        FakeResponse([body], status_code=402),
        environ={"ZAI_API_KEY": credential},
    )

    failure = result.evidence.failure
    assert failure is not None
    assert "[REDACTED]" in failure.message
    for path in sorted(config.output_dir.iterdir()):
        text = path.read_text(encoding="utf-8").lower()
        for length in range(_MIN_LEAKED_PREFIX_CHARS, surviving + 1):
            assert credential[:length].lower() not in text, path.name


@pytest.mark.parametrize("substitute", ["\t", "  ", "\u00a0"])
def test_a_whitespace_altered_truncated_tail_is_repaired(
    tmp_path: Path, substitute: str
) -> None:
    """Collapsing whitespace turns an altered tail back into the prefix.

    The persisted diagnostic normalizes runs of whitespace, so a tab in the
    echo becomes a space again and the result is exactly the credential
    prefix unless the repair matches whitespace flexibly.
    """
    credential = "ZQXJVK WPMBGH FDSART"
    config = make_config(tmp_path)
    surviving = 12
    tail = credential[:surviving].replace(" ", substitute)
    body = tail.encode("utf-8")

    result, _ = run(
        config,
        FakeResponse([body], status_code=402),
        environ={"ZAI_API_KEY": credential},
    )

    failure = result.evidence.failure
    assert failure is not None
    assert "[REDACTED]" in failure.message
    normalized = " ".join(failure.message.split())
    for length in range(_MIN_LEAKED_PREFIX_CHARS, surviving + 1):
        assert credential[:length] not in normalized
    for path in sorted(config.output_dir.iterdir()):
        text = " ".join(path.read_text(encoding="utf-8").split())
        for length in range(_MIN_LEAKED_PREFIX_CHARS, surviving + 1):
            assert credential[:length] not in text, path.name


def test_a_whitespace_only_credential_does_not_crash_the_redactor(
    tmp_path: Path,
) -> None:
    """``redact_text_for_dry_run`` is exported, so the constructor is public.

    A whitespace-only value is unreachable through the normal resolution
    path, which strips and then rejects an empty credential, but the
    constructor must not raise for a caller that builds one directly.
    """
    assert redact_text_for_dry_run("nothing to hide", "   ") == "nothing to hide"
    assert redact_text_for_dry_run("nothing to hide", "") == "nothing to hide"
    assert redact_text_for_dry_run("nothing to hide", None) == "nothing to hide"
