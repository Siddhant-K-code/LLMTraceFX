"""Tests for the provider-neutral OpenAI-compatible streaming API collector.

Every test here injects a fake transport. No request ever leaves the
process, no API key is used, and Z.ai is never contacted. The GLM request
profiles exercised below follow Z.ai's published chat-completions
documentation:

* https://docs.z.ai/guides/vlm/glm-5.3-flash
* https://docs.z.ai/api-reference/llm/chat-completion
"""

from __future__ import annotations

import base64
import codecs
import json
import time
import unicodedata
from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer.collectors.openai_api import (
    _MAX_PERSISTED_HEADER_CHARS,
    ARTIFACT_MANIFEST_NAME,
    DEFAULT_RETAINED_EVENT_LIMIT,
    FAILURE_CONNECTION,
    FAILURE_HTTP_STATUS,
    FAILURE_MISSING_CONTENT,
    FAILURE_PROVIDER_ERROR,
    FAILURE_STREAM_DECODE,
    FAILURE_STREAM_TRUNCATED,
    FAILURE_TIMEOUT,
    APICollectionConfig,
    FinishReasonVocabulary,
    HTTPRequest,
    OpenAIStreamCollectorError,
    ProviderExtensions,
    TransportConnectionError,
    TransportTimeout,
    _contains_credential,
    _Redactor,
    _response_socket,
    _safe_endpoint_for_message,
    artifact_set_is_complete,
    assert_credential_not_embedded,
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
        ("credential_env_var", "9BAD", "--api-key-env"),
        ("credential_env_var", "lower_case_name", "--api-key-env"),
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
    assert plan.credential_env_var == "[REDACTED]"
    assert plan.credential_header_name == "Authorization"
    assert "Authorization" in plan.header_names
    assert plan.request_parameters["max_tokens"] == 256
    assert plan.provider_extensions == {
        "reasoning_effort": "high",
        "thinking": {"clear_thinking": False},
    }
    # The command matches the invocation apart from the credential variable
    # name, which is masked because nothing here proved it is a name.
    assert tuple(plan.command) == tuple(
        "[REDACTED]" if argument == "ZAI_API_KEY" else argument
        for argument in config.command_argv
    )
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

    with pytest.raises(
        OpenAIStreamCollectorError, match="named by --api-key-env is not set"
    ):
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
    # No local compute backend ran this. The hosted service goes in
    # ``provider``, which exists so a transport is never mistaken for
    # hardware.
    assert restored.runtime.backend is None
    assert restored.runtime.provider == "z.ai"


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
    # The canonical model-phase fields stay unset: neither prompt
    # processing nor generation is separable from transport here. The
    # client-observed offsets live in the evidence timeline instead.
    assert result.record.timing.prefill is None
    assert result.record.timing.decode is None
    assert result.record.timing.total is not None


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


def test_the_two_rates_use_the_windows_that_match_their_numerators(
    tmp_path: Path,
) -> None:
    """Each rate is divided by the window its numerator was produced over.

    The delta rate counts content deltas, so it uses the content window.
    The provider's ``completion_tokens`` counts reasoning tokens too, so it
    uses the wider generation window that starts at the first reasoning
    delta. Anchoring either on the last event of any kind would fold the
    trailing usage, finish-reason and ``[DONE]`` events into a decode
    window.
    """
    config = make_config(tmp_path)
    usage = {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17}

    result, _ = run(config, glm_stream(content_parts=("a", "b", "c", "d"), usage=usage))

    statistics = result.evidence.statistics
    timeline = result.evidence.timeline
    content_offsets = [
        event.offset_ms for event in timeline.events if event.kind == "content"
    ]
    generated_offsets = [
        event.offset_ms
        for event in timeline.events
        if event.kind in {"content", "reasoning"}
    ]
    expected_window = content_offsets[-1] - content_offsets[0]
    expected_generation = generated_offsets[-1] - generated_offsets[0]

    assert statistics.content_window_ms == pytest.approx(expected_window)
    assert statistics.generation_window_ms == pytest.approx(expected_generation)
    # Reasoning was streamed first, so the generation window is strictly wider.
    assert expected_generation > expected_window
    # Three gaps across four arrivals, over the content window.
    assert statistics.content_delta_rate_per_second == pytest.approx(
        3 / (expected_window / 1000)
    )
    assert statistics.provider_completion_tokens_per_second == pytest.approx(
        12 / (expected_generation / 1000)
    )


def test_reasoning_before_content_does_not_inflate_the_token_rate(
    tmp_path: Path,
) -> None:
    """A long silent reasoning phase must not be credited to the answer.

    ``completion_tokens`` covers the reasoning tokens as well, so the rate
    published for it has to span the time those tokens were generated in.
    """
    config = make_config(tmp_path)
    usage = {
        "prompt_tokens": 5,
        "completion_tokens": 100,
        "total_tokens": 105,
        "completion_tokens_details": {"reasoning_tokens": 90},
    }

    result, _ = run(
        config,
        glm_stream(
            content_parts=("a", "b"),
            reasoning_parts=tuple(f"step {index}" for index in range(8)),
            usage=usage,
        ),
    )

    statistics = result.evidence.statistics
    assert statistics.content_window_ms is not None
    assert statistics.generation_window_ms is not None
    assert statistics.generation_window_ms > statistics.content_window_ms
    assert statistics.provider_completion_tokens_per_second is not None
    naive = 100 / (statistics.content_window_ms / 1000)
    assert statistics.provider_completion_tokens_per_second < naive
    assert statistics.provider_completion_tokens_per_second == pytest.approx(
        100 / (statistics.generation_window_ms / 1000)
    )
    # The visible rate strips the reasoning tokens the provider reported and
    # is measured over the window the visible answer actually arrived in.
    assert statistics.provider_visible_completion_tokens_per_second == pytest.approx(
        10 / (statistics.content_window_ms / 1000)
    )


def test_a_missing_reasoning_token_count_leaves_the_visible_rate_null(
    tmp_path: Path,
) -> None:
    """A count the provider did not report is not zero."""
    config = make_config(tmp_path)
    usage = {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17}

    result, _ = run(config, glm_stream(content_parts=("a", "b", "c"), usage=usage))

    statistics = result.evidence.statistics
    assert result.evidence.usage.reasoning_tokens is None
    assert statistics.provider_visible_completion_tokens_per_second is None
    assert statistics.provider_completion_tokens_per_second is not None


def test_reasoning_tokens_exceeding_completion_tokens_leaves_the_rate_null(
    tmp_path: Path,
) -> None:
    """An inconsistent provider report yields no rate, not a negative one.

    ``completion_tokens`` is documented as including reasoning tokens, so
    a reasoning count above it is a provider bug. Subtracting anyway would
    publish a negative tokens-per-second, which reads as evidence.
    """
    config = make_config(tmp_path)
    usage = {
        "prompt_tokens": 5,
        "completion_tokens": 4,
        "total_tokens": 9,
        "completion_tokens_details": {"reasoning_tokens": 40},
    }

    result, _ = run(config, glm_stream(content_parts=("a", "b", "c"), usage=usage))

    statistics = result.evidence.statistics
    assert result.evidence.usage.reasoning_tokens == 40
    assert statistics.provider_visible_completion_tokens_per_second is None


def test_without_reasoning_deltas_the_two_windows_are_identical(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    usage = {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17}

    result, _ = run(
        config,
        glm_stream(content_parts=("a", "b", "c"), reasoning_parts=(), usage=usage),
    )

    statistics = result.evidence.statistics
    assert statistics.content_window_ms is not None
    assert statistics.generation_window_ms == pytest.approx(
        statistics.content_window_ms
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


def test_a_single_content_delta_yields_no_content_window_and_no_delta_rate(
    tmp_path: Path,
) -> None:
    """One arrival bounds zero intervals, so no delta rate is observable."""
    config = make_config(tmp_path)
    usage = {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17}

    result, _ = run(config, glm_stream(content_parts=("only",), usage=usage))

    statistics = result.evidence.statistics
    assert statistics.content_delta_count == 1
    assert statistics.content_window_ms is None
    assert statistics.content_delta_rate_per_second is None
    assert statistics.provider_visible_completion_tokens_per_second is None
    assert statistics.inter_content_delta is None
    # A reasoning delta preceded the answer, so a generation window was
    # still observed and the provider's token count has a denominator.
    assert statistics.generation_window_ms is not None
    assert statistics.provider_completion_tokens_per_second is not None


def test_a_single_generated_delta_yields_no_window_and_no_rates(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    usage = {"prompt_tokens": 5, "completion_tokens": 12, "total_tokens": 17}

    result, _ = run(
        config,
        glm_stream(content_parts=("only",), reasoning_parts=(), usage=usage),
    )

    statistics = result.evidence.statistics
    assert statistics.content_window_ms is None
    assert statistics.generation_window_ms is None
    assert statistics.content_delta_rate_per_second is None
    assert statistics.provider_completion_tokens_per_second is None


def test_the_content_window_is_persisted_with_its_definition(tmp_path: Path) -> None:
    config = make_config(tmp_path)

    run(config, glm_stream(content_parts=("a", "b")))

    payload = json.loads(
        (config.output_dir / "api_evidence.json").read_text(encoding="utf-8")
    )
    statistics = payload["statistics"]
    assert statistics["content_window_ms"] is not None
    assert "last content delta arrival" in statistics["content_window_definition"]
    assert statistics["generation_window_ms"] is not None
    assert "reasoning or content" in statistics["generation_window_definition"]
    note = statistics["provider_completion_tokens_per_second_note"]
    assert "generation_window_ms" in note
    assert "includes reasoning tokens" in note
    assert "coarse estimate" in note
    visible_note = statistics["provider_visible_completion_tokens_per_second_note"]
    assert "content_window_ms" in visible_note
    assert "missing count is not zero" in visible_note


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


def test_unterminated_done_sentinel_is_not_a_clean_ending(tmp_path: Path) -> None:
    """A ``[DONE]`` the stream never terminated cannot end a run cleanly."""
    config = make_config(tmp_path)
    chunks = [
        b'data: {"id": "c1", "choices": [{"index": 0, "delta": {"content": "x"}}]}\n\n',
        b"data: [DONE]",
    ]

    result, _ = run(config, chunks)

    assert result.evidence.stream_terminated_with_done is False
    assert result.evidence.stream_had_unterminated_event is True
    assert result.evidence.success is False
    assert result.evidence.failure is not None
    assert result.evidence.failure.category == FAILURE_STREAM_TRUNCATED


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
    assert isinstance(payload, dict)
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


def test_a_short_credential_is_still_matched_the_way_the_redactor_matches_it(
    tmp_path: Path,
) -> None:
    """Length must not buy a value weaker containment than redaction.

    ``a/b`` is not a plausible key, but ``?path=a%2Fb`` is a genuine
    percent-encoded echo of it rather than a coincidence: the redactor
    turns that endpoint into ``path=[REDACTED]``. If the pre-flight check
    let it through, a value the redactor considers the credential would
    reach the persisted plan, which is the asymmetry this pairing exists
    to prevent. The extra multi-round decodings, where a short value
    really can collide with a decoded byte, stay behind the length gate.
    """
    config = make_config(tmp_path, endpoint=f"{ENDPOINT}?path=a%2Fb")

    with pytest.raises(OpenAIStreamCollectorError, match="refusing to run"):
        run(config, glm_stream(), environ={"ZAI_API_KEY": "a/b"})

    assert _Redactor("a/b").text("path=a%2Fb") == "path=[REDACTED]"


def test_a_short_credential_absent_from_the_endpoint_is_not_refused(
    tmp_path: Path,
) -> None:
    """Refusal must follow a real match, not merely a short credential.

    The matcher runs at every length now, so this pins the other side of
    that change: a short value that appears in no spelling still runs.
    """
    config = make_config(tmp_path, endpoint=f"{ENDPOINT}?q=one+two")

    result, _ = run(config, glm_stream(), environ={"ZAI_API_KEY": "xyz"})

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
    """A cut ``error`` payload must not reach an artifact in any form.

    The frame is now discarded at end of stream rather than dispatched, so
    the cut payload no longer reaches the message interpolation at all and
    the run is reported as truncated. The assertions stay because the
    guarantee under test is the absence of the credential, not the route
    the bytes would otherwise have taken.
    """
    config = make_config(tmp_path)
    chunks = [
        b'event: error\ndata: {"error":{"message":"' + API_KEY[:keep].encode("utf-8")
    ]

    result, _ = run(config, FakeResponse(chunks))

    failure = result.evidence.failure
    assert failure is not None
    assert failure.category == FAILURE_STREAM_TRUNCATED
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


# --- Sixth review pass: terminal conditions and artifact bytes ---------------


@pytest.mark.parametrize(
    "failure_reason", ["network_error", "model_context_window_exceeded"]
)
def test_a_documented_failure_finish_reason_outranks_the_done_sentinel(
    tmp_path: Path, failure_reason: str
) -> None:
    """Z.ai documents these next to the successful reasons, so both can arrive.

    ``[DONE]`` says the transport finished, not that the generation did.
    Letting the sentinel win would publish an aborted generation as a
    successful measurement.

    https://docs.z.ai/api-reference/llm/chat-completion
    """
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "partial"}}]}),
        sse(
            {
                "id": "c1",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": ""},
                        "finish_reason": failure_reason,
                    }
                ],
            }
        ),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    assert result.evidence.success is False
    assert result.evidence.stream_terminated_with_done is True
    payload = assert_failure_artifacts(config, FAILURE_PROVIDER_ERROR)
    assert payload["failure"]["provider_error_code"] == failure_reason
    assert payload["finish_reason"] == failure_reason


def test_a_failure_finish_reason_without_content_is_still_failure_shaped(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    chunks = [
        sse(
            {
                "id": "c1",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "network_error"}
                ],
            }
        ),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    assert result.evidence.success is False
    assert result.evidence.failure is not None
    assert result.evidence.failure.category == FAILURE_PROVIDER_ERROR


def test_a_failure_finish_reason_without_done_is_reported_as_the_provider_error(
    tmp_path: Path,
) -> None:
    """The provider's own reason explains more than a generic truncation."""
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "partial"}}]}),
        sse(
            {
                "id": "c1",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "network_error"}
                ],
            }
        ),
    ]

    result, _ = run(config, chunks)

    assert result.evidence.failure is not None
    assert result.evidence.failure.category == FAILURE_PROVIDER_ERROR


@pytest.mark.parametrize(
    "terminal_reason", ["stop", "length", "tool_calls", "sensitive"]
)
def test_documented_successful_finish_reasons_still_terminate_cleanly(
    tmp_path: Path, terminal_reason: str
) -> None:
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "answer"}}]}),
        sse(
            {
                "id": "c1",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": terminal_reason}
                ],
            }
        ),
    ]

    result, _ = run(config, chunks)

    assert result.evidence.success is True
    assert result.evidence.finish_reason == terminal_reason


def test_a_stream_cut_after_partial_content_is_truncated_not_successful(
    tmp_path: Path,
) -> None:
    """A clean socket close mid-frame still loses bytes."""
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "half"}}]}),
        b'data: {"id": "c1", "choices": [{"index": 0, "delta": {"content": " an',
    ]

    result, _ = run(config, chunks)

    assert result.evidence.success is False
    assert result.evidence.stream_had_unterminated_event is True
    assert_failure_artifacts(config, FAILURE_STREAM_TRUNCATED)


def test_a_leading_byte_order_mark_does_not_lose_the_first_content_event(
    tmp_path: Path,
) -> None:
    """Providers may prefix the body with a BOM; the answer must survive it."""
    config = make_config(tmp_path)
    chunks = [
        "\ufeff".encode()
        + sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "first"}}]}),
        sse(
            {
                "id": "c1",
                "choices": [{"index": 0, "delta": {"content": " second"}}],
            }
        ),
        sse(
            {
                "id": "c1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        ),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    assert result.evidence.success is True
    assert result.response_text == "first second"


@pytest.mark.parametrize("newline", ["\r\n", "\r", "\n"])
def test_an_artifact_set_verifies_whatever_line_endings_the_answer_used(
    tmp_path: Path, newline: str
) -> None:
    """Text mode rewrites newlines on the way in and on the way out.

    Hashing the string and verifying a re-read string made a legitimate
    CRLF answer look tampered with, because the read collapsed it back to
    ``\\n`` and the digests stopped matching.
    """
    config = make_config(tmp_path)
    answer = f"line one{newline}line two"
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": answer}}]}),
        sse(
            {
                "id": "c1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        ),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    assert result.evidence.success is True
    assert artifact_set_is_complete(config.output_dir) is True
    assert (config.output_dir / "response.txt").read_bytes() == answer.encode("utf-8")


def test_a_tampered_artifact_is_still_rejected(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    run(config, glm_stream())

    assert artifact_set_is_complete(config.output_dir) is True
    (config.output_dir / "response.txt").write_bytes(b"replaced\r\n")

    assert artifact_set_is_complete(config.output_dir) is False


def test_the_attached_endpoint_form_is_sanitized_in_every_artifact(
    tmp_path: Path,
) -> None:
    """``--endpoint=<url>`` is as ordinary as the separate form."""
    endpoint = f"{ENDPOINT}?deployment=private-value"
    config = make_config(
        tmp_path,
        endpoint=endpoint,
        command_argv=(
            "llmtracefx-optimizer",
            "collect-api",
            f"--endpoint={endpoint}",
            "--api-key-env",
            "ZAI_API_KEY",
        ),
    )

    result, _ = run(config, glm_stream())

    plan_command = list(result.evidence.plan.command)
    assert f"--endpoint={endpoint}" not in plan_command
    assert any(argument.startswith("--endpoint=") for argument in plan_command)
    for path in sorted(config.output_dir.iterdir()):
        assert "private-value" not in path.read_text(encoding="utf-8"), path.name


def test_the_separate_endpoint_form_is_still_sanitized(tmp_path: Path) -> None:
    endpoint = f"{ENDPOINT}?deployment=private-value"
    config = make_config(
        tmp_path,
        endpoint=endpoint,
        command_argv=("llmtracefx-optimizer", "collect-api", "--endpoint", endpoint),
    )

    result, _ = run(config, glm_stream())

    assert endpoint not in result.evidence.plan.command
    for path in sorted(config.output_dir.iterdir()):
        assert "private-value" not in path.read_text(encoding="utf-8"), path.name


# --- Seventh review pass -----------------------------------------------------


def test_a_carriage_return_framed_stream_collects_as_a_success(tmp_path: Path) -> None:
    """Lone CR framing is legal SSE and must not read as a truncated stream."""
    body = b"".join(glm_stream()).replace(b"\n", b"\r")
    result, _ = run(make_config(tmp_path), [body])

    assert result.evidence.success is True
    assert result.response_text == "Hello world"
    assert result.evidence.stream_had_unterminated_event is False


def test_a_stream_ending_on_a_stray_carriage_return_is_still_a_success(
    tmp_path: Path,
) -> None:
    chunks = [*glm_stream(), b"\r"]
    result, _ = run(make_config(tmp_path), chunks)

    assert result.evidence.success is True
    assert result.evidence.stream_had_unterminated_event is False
    assert result.record.outcome.success is True


# --- Ninth review pass -------------------------------------------------------


@pytest.mark.parametrize(
    "credential",
    [
        "error",
        "network",
        "network_error",
        "_error",
        "work_err",
    ],
)
def test_a_credential_overlapping_a_finish_reason_cannot_erase_a_failure(
    tmp_path: Path, credential: str
) -> None:
    """Meaning is decided before redaction rewrites the text.

    Redaction is a text transform on a provider-controlled string. A
    credential that happens to contain ``error`` turns ``network_error``
    into ``network_[REDACTED]``; if the failure were classified from the
    redacted text it would stop being recognized, and the ``[DONE]`` that
    follows would publish an aborted generation as a whole answer.

    These credentials are English fragments of this program's own
    vocabulary, so no artifact-wide substring scan is asserted here: a
    credential that spells a word this collector writes itself cannot be
    kept out of its own diagnostics, and the encoded-echo tests cover the
    scan against a realistic key.
    """
    config = make_config(tmp_path)
    environ = {"ZAI_API_KEY": credential}
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "partial"}}]}),
        sse(
            {
                "id": "c1",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "network_error"}
                ],
            }
        ),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks, environ=environ)

    assert result.evidence.success is False
    assert result.record.outcome.success is False
    assert result.evidence.failure is not None
    assert result.evidence.failure.provider_error_code == "network_error"
    assert result.evidence.finish_reason_classification == "failure"
    # The persisted text is still redacted, only the classification is not
    # derived from it.
    assert result.evidence.finish_reason is not None
    assert "[REDACTED]" in result.evidence.finish_reason


@pytest.mark.parametrize("credential", ["stop", "top", "sensitive", "tool_calls"])
def test_a_credential_overlapping_a_terminal_reason_keeps_the_success(
    tmp_path: Path, credential: str
) -> None:
    """The same ordering must not turn a completed answer into a failure."""
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "done"}}]}),
        sse(
            {
                "id": "c1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        ),
    ]

    result, _ = run(config, chunks, environ={"ZAI_API_KEY": credential})

    assert result.evidence.success is True
    assert result.evidence.finish_reason_classification == "terminal"
    assert result.response_text == "done"


def test_the_finish_reason_classification_is_persisted_beside_the_text(
    tmp_path: Path,
) -> None:
    """The persisted code is a value this collector defined, not the wire text."""
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "hi"}}]}),
        sse(
            {
                "id": "c1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "STOP "}],
            }
        ),
    ]

    result, _ = run(config, chunks)

    assert result.evidence.success is True
    payload = json.loads(
        (config.output_dir / "api_evidence.json").read_text(encoding="utf-8")
    )
    assert payload["finish_reason"] == "STOP"
    assert payload["finish_reason_classification"] == "terminal"
    assert payload["finish_reason_code"] == "stop"


def test_an_unrecognized_finish_reason_is_classified_as_such(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "hi"}}]}),
        sse(
            {
                "id": "c1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "invented"}],
            }
        ),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    payload = json.loads(
        (config.output_dir / "api_evidence.json").read_text(encoding="utf-8")
    )
    assert payload["finish_reason_classification"] == "unrecognized"
    assert payload["finish_reason_code"] is None
    # [DONE] is still an accepted terminal condition on its own.
    assert result.evidence.success is True


PERCENT_CREDENTIAL = "sk-slash/credential+plus space"


def percent_encode(value: str, *, lower: bool = False, double: bool = False) -> str:
    encoded = "".join(
        (
            character
            if character.isalnum() or character in "-_."
            else "".join(f"%{byte:02X}" for byte in character.encode("utf-8"))
        )
        for character in value
    )
    if double:
        encoded = encoded.replace("%", "%25")
    return encoded.lower() if lower else encoded


@pytest.mark.parametrize(
    ("lower", "double"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_a_percent_encoded_credential_echo_is_scrubbed_everywhere(
    tmp_path: Path, lower: bool, double: bool
) -> None:
    """An encoded echo is one mechanical decode away from the key.

    A provider that reflects the credential through a URL builder returns
    ``sk-slash%2Fcredential``. A literal substring scrub sees nothing, and
    the artifact carries a reversible secret.
    """
    config = make_config(tmp_path)
    echo = percent_encode(PERCENT_CREDENTIAL, lower=lower, double=double)
    chunks = [
        sse(
            {
                "id": f"resp-{echo}",
                "request_id": f"req-{echo}",
                "model": f"glm-{echo}",
                "choices": [{"index": 0, "delta": {"content": f"key is {echo}"}}],
            }
        ),
        sse(
            {
                "id": "resp-2",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        ),
        b"data: [DONE]\n\n",
    ]
    response = FakeResponse(
        chunks,
        headers={"content-type": "text/event-stream", "x-request-id": echo},
    )

    result, _ = run(config, response, environ={"ZAI_API_KEY": PERCENT_CREDENTIAL})

    assert echo not in result.response_text
    for path in sorted(config.output_dir.iterdir()):
        text = path.read_text(encoding="utf-8")
        assert echo not in text, path.name
        assert PERCENT_CREDENTIAL not in text, path.name


def test_a_partially_percent_encoded_echo_is_scrubbed(tmp_path: Path) -> None:
    """Real quoting functions encode only what they consider unsafe."""
    config = make_config(tmp_path)
    echo = PERCENT_CREDENTIAL.replace("/", "%2F").replace(" ", "%20")
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": echo}}]}),
        sse(
            {
                "id": "c1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        ),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks, environ={"ZAI_API_KEY": PERCENT_CREDENTIAL})

    assert "%2F" not in result.response_text
    assert result.response_text == "[REDACTED]"


def test_an_encoded_credential_cut_by_truncation_is_repaired(tmp_path: Path) -> None:
    """The boundary repair matches the same encodings as the whole scrub."""
    config = make_config(tmp_path)
    encoded = percent_encode(PERCENT_CREDENTIAL)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": encoded[:-4]}}]})
    ]

    result, _ = run(config, chunks, environ={"ZAI_API_KEY": PERCENT_CREDENTIAL})

    assert result.evidence.success is False
    for path in sorted(config.output_dir.iterdir()):
        text = path.read_text(encoding="utf-8")
        assert "sk-slash" not in text, path.name


def test_an_encoded_bearer_prefix_is_scrubbed(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    chunks = [
        sse(
            {
                "id": "c1",
                "choices": [
                    {"index": 0, "delta": {"content": "Bearer%20some-other-token"}}
                ],
            }
        ),
        sse(
            {
                "id": "c1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        ),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    assert "some-other-token" not in result.response_text


def test_events_from_one_network_chunk_share_one_arrival_time(
    tmp_path: Path,
) -> None:
    """Parser CPU time is not inter-token latency.

    Reading the clock per decoded event rather than per network chunk
    invents positive gaps between deltas that arrived in the same packet,
    which would show up as observable throughput that no observer could
    have measured.
    """
    config = make_config(tmp_path)
    together = sse(
        {"id": "c1", "choices": [{"index": 0, "delta": {"content": "aa"}}]}
    ) + sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "bb"}}]})
    chunks = [
        together,
        sse(
            {
                "id": "c1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        ),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    offsets = [
        event.offset_ms
        for event in result.evidence.timeline.events
        if event.kind == "content"
    ]
    assert len(offsets) == 2
    assert offsets[0] == offsets[1]
    statistics = result.evidence.statistics
    # A zero-width window is not a window, so no rate is published from it.
    assert statistics.content_window_ms is None
    assert statistics.content_delta_rate_per_second is None
    assert statistics.provider_completion_tokens_per_second is None


def test_events_in_separate_network_chunks_keep_their_gaps(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    chunks = [
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "aa"}}]}),
        sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "bb"}}]}),
        sse(
            {
                "id": "c1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        ),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    offsets = [
        event.offset_ms
        for event in result.evidence.timeline.events
        if event.kind == "content"
    ]
    assert offsets[1] > offsets[0]
    assert result.evidence.statistics.content_window_ms == pytest.approx(
        offsets[1] - offsets[0]
    )


def test_a_fragmented_event_is_timed_by_the_chunk_that_completed_it(
    tmp_path: Path,
) -> None:
    """A delta split across packets arrives when its last byte does."""
    config = make_config(tmp_path)
    first = sse({"id": "c1", "choices": [{"index": 0, "delta": {"content": "aa"}}]})
    chunks = [
        first[:10],
        first[10:],
        sse(
            {
                "id": "c1",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        ),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, chunks)

    content = [
        event for event in result.evidence.timeline.events if event.kind == "content"
    ]
    assert len(content) == 1
    assert result.evidence.timeline.first_body_chunk_offset_ms is not None
    assert content[0].offset_ms > result.evidence.timeline.first_body_chunk_offset_ms


# --- Tenth review pass -------------------------------------------------------


def _finish_chunk(reason: str) -> bytes:
    return sse({"choices": [{"index": 0, "delta": {}, "finish_reason": reason}]})


@pytest.mark.parametrize(
    "order",
    [
        ("network_error", "stop"),
        ("network_error", "length"),
        ("model_context_window_exceeded", "stop"),
        ("stop", "network_error"),
    ],
)
def test_a_later_finish_reason_does_not_erase_a_reported_failure(
    tmp_path: Path, order: tuple[str, str]
) -> None:
    """Failure wins regardless of which chunk carried it.

    Nothing in the wire format stops a provider sending a second
    ``finish_reason``. Last write wins would let a trailing ``stop`` erase
    ``network_error`` and publish an aborted generation as a success with a
    full latency timeline, which is the outcome the terminal-condition
    check exists to prevent.
    """
    config = make_config(tmp_path)
    chunks = [
        sse({"choices": [{"index": 0, "delta": {"content": "partial answ"}}]}),
        *(_finish_chunk(reason) for reason in order),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, FakeResponse(chunks))

    assert result.record.outcome.success is False
    assert result.evidence.finish_reason_code in _FAILURE_ORDER
    assert result.evidence.finish_reason_classification == "failure"
    assert result.evidence.failure is not None
    assert result.evidence.failure.category == FAILURE_PROVIDER_ERROR


_FAILURE_ORDER = {"network_error", "model_context_window_exceeded"}


def test_the_persisted_finish_reason_agrees_with_the_classified_code(
    tmp_path: Path,
) -> None:
    """The evidence must not show ``stop`` next to a failure code."""
    config = make_config(tmp_path)
    chunks = [
        sse({"choices": [{"index": 0, "delta": {"content": "partial"}}]}),
        _finish_chunk("network_error"),
        _finish_chunk("stop"),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, FakeResponse(chunks))

    assert result.evidence.finish_reason == "network_error"
    assert result.evidence.finish_reason_code == "network_error"


def test_two_terminal_reasons_still_take_the_later_one(tmp_path: Path) -> None:
    """Only a failure is sticky, so ordinary streams are unaffected."""
    config = make_config(tmp_path)
    chunks = [
        sse({"choices": [{"index": 0, "delta": {"content": "answer"}}]}),
        _finish_chunk("length"),
        _finish_chunk("stop"),
        b"data: [DONE]\n\n",
    ]

    result, _ = run(config, FakeResponse(chunks))

    assert result.record.outcome.success is True
    assert result.evidence.finish_reason_code == "stop"


# The credential in the name slot. Uppercase so it passes the shape rule and
# only the presence rule can stop it, which is the harder of the two cases.
NAME_SLOT_CREDENTIAL = "AKIA1234567890ABCDEF"


@pytest.mark.parametrize(
    "name",
    [
        "sk-3f0a1c2b-9d8e-7f6a-5b4c",
        "sk_live_9f2b7d41ca6e4b8f",
        "zai_api_key",
        "9BAD",
        "has space",
        "WITH-HYPHEN",
        "",
    ],
)
def test_a_name_that_is_not_a_conventional_variable_is_refused_silently(
    tmp_path: Path, name: str
) -> None:
    """The rejected value is never repeated, because it may be the key."""
    with pytest.raises(OpenAIStreamCollectorError) as raised:
        make_config(tmp_path, credential_env_var=name)

    message = str(raised.value)
    assert "--api-key-env" in message
    if name:
        assert name not in message


def test_an_unproven_variable_name_is_masked_in_the_plan(tmp_path: Path) -> None:
    """Presence is the only thing a caller cannot fake.

    An uppercase credential passes the shape rule, so the plan falls back
    to asking the environment. A name the environment does not define was
    never proven to be a name.
    """
    config = make_config(tmp_path, credential_env_var=NAME_SLOT_CREDENTIAL)

    plan = build_request_plan(config, environ={})

    assert plan.credential_env_var == "[REDACTED]"
    assert NAME_SLOT_CREDENTIAL not in plan.to_json()


def test_a_proven_variable_name_is_persisted(tmp_path: Path) -> None:
    """A real exported variable is not a secret and stays readable."""
    config = make_config(tmp_path, credential_env_var="ZAI_API_KEY")

    plan = build_request_plan(config, environ={"ZAI_API_KEY": "value"})

    assert plan.credential_env_var == "ZAI_API_KEY"


@pytest.mark.parametrize("attached", [False, True])
def test_an_unproven_name_is_masked_in_both_command_spellings(
    tmp_path: Path, attached: bool
) -> None:
    argv = (
        ("llmtracefx-optimizer", "collect-api", f"--api-key-env={NAME_SLOT_CREDENTIAL}")
        if attached
        else (
            "llmtracefx-optimizer",
            "collect-api",
            "--api-key-env",
            NAME_SLOT_CREDENTIAL,
        )
    )
    config = make_config(
        tmp_path, credential_env_var=NAME_SLOT_CREDENTIAL, command_argv=argv
    )

    plan = build_request_plan(config, environ={})

    assert NAME_SLOT_CREDENTIAL not in plan.to_json()
    assert "[REDACTED]" in " ".join(plan.command)


def test_the_variable_name_is_not_part_of_request_identity(tmp_path: Path) -> None:
    """Two runs differing only in the variable issue identical requests.

    Hashing the name would also persist a derivation of a value that may be
    the credential, which this collector promises never to hash.
    """
    first = build_request_plan(
        make_config(tmp_path, credential_env_var="ZAI_API_KEY"), environ={}
    )
    second = build_request_plan(
        make_config(tmp_path, credential_env_var="OTHER_PROVIDER_KEY"), environ={}
    )

    assert first.config_hash == second.config_hash


def test_a_missing_variable_is_reported_without_naming_it(tmp_path: Path) -> None:
    config = make_config(tmp_path, credential_env_var="ZAI_API_KEY")

    with pytest.raises(OpenAIStreamCollectorError) as raised:
        collect_openai_stream(config, transport=ExplodingTransport(), environ={})

    assert "ZAI_API_KEY" not in str(raised.value)
    assert not config.output_dir.exists()


def test_the_embedded_credential_refusal_names_the_option_not_the_variable(
    tmp_path: Path,
) -> None:
    """The rule is applied uniformly rather than per call site.

    Every caller of this check happens to hold a proven name today, so
    naming the variable would be safe by construction. Referring to the
    option instead removes the dependency on that reasoning surviving a
    later refactor, and stays equally actionable.
    """
    config = make_config(tmp_path, prompt=f"summarize this: {API_KEY}")

    with pytest.raises(OpenAIStreamCollectorError) as raised:
        collect_openai_stream(
            config,
            transport=ExplodingTransport(),
            environ={"ZAI_API_KEY": API_KEY},
        )

    message = str(raised.value)
    assert "--api-key-env" in message
    assert "ZAI_API_KEY" not in message
    assert API_KEY not in message


# --- Eleventh review pass ----------------------------------------------------


# The pre-flight refusal and the redactor guard the same threat: the
# credential ending up in an artifact. They only agree if they recognise the
# same spellings, so these cases are stated once and asserted against both.
_EQUIVALENT_SPELLINGS = (
    ("lowercased", lambda value: value.lower()),
    ("uppercased", lambda value: value.upper()),
    ("tab for space", lambda value: value.replace(" ", "\t")),
    ("newline for space", lambda value: value.replace(" ", "\n")),
    ("percent encoded slash", lambda value: value.replace("/", "%2F")),
    ("lowercase hex", lambda value: value.replace("/", "%2f")),
    ("double encoded", lambda value: value.replace("/", "%252F")),
    ("plus for space", lambda value: value.replace(" ", "+")),
)

# Mixed case with a slash and a space, so every transformation above
# produces a genuinely different string from the original.
_SPELLING_CREDENTIAL = "sk-Live/Key Value-8712"


@pytest.mark.parametrize("label,transform", _EQUIVALENT_SPELLINGS)
def test_preflight_and_redactor_agree_on_credential_spellings(
    label: str, transform: Any
) -> None:
    """Both matchers recognise every spelling, or neither is trustworthy.

    A pre-flight that is merely literal passes a provider identifier
    echoing the key in lower case, or an extension value with a tab where
    the key has a space. The refusal exists precisely to stop such a value
    being persisted, so it has to be as flexible as the redaction it backs.
    """
    echo = transform(_SPELLING_CREDENTIAL)
    assert echo != _SPELLING_CREDENTIAL, label

    assert _contains_credential(echo, _SPELLING_CREDENTIAL), label
    assert _Redactor(_SPELLING_CREDENTIAL).text(echo) != echo, label


@pytest.mark.parametrize(
    "value",
    [
        "ZAI_API_KEY",
        "glm-5.3",
        "https://api.z.ai/api/paas/v4/chat/completions",
        "Explain a stack in one sentence.",
        "chatcmpl-abc123",
        "",
    ],
)
def test_preflight_does_not_fire_on_unrelated_values(value: str) -> None:
    """Flexibility must not turn into matching everything.

    A refusal that fires on ordinary configuration teaches people to work
    around the check, which costs more than the check buys.
    """
    assert not _contains_credential(value, _SPELLING_CREDENTIAL)


@pytest.mark.parametrize(
    "field,build",
    [
        ("run_id", lambda echo: {"run_id": echo}),
        ("provider", lambda echo: {"provider": echo}),
        ("model_id", lambda echo: {"model_id": echo}),
        ("prompt", lambda echo: {"prompt": echo}),
        ("model_revision", lambda echo: {"model_revision": echo}),
        ("system_prompt", lambda echo: {"system_prompt": echo}),
        (
            "endpoint",
            lambda echo: {"endpoint": f"https://api.z.ai/v4/chat?x={echo}"},
        ),
        (
            "provider_request_id",
            lambda echo: {"extensions": ProviderExtensions(provider_request_id=echo)},
        ),
        (
            "command_argv",
            lambda echo: {
                "command_argv": ("llmtracefx-optimizer", "collect-api", echo)
            },
        ),
    ],
)
def test_every_persisted_field_refuses_a_case_folded_credential(
    tmp_path: Path, field: str, build: Any
) -> None:
    """Each field that reaches an artifact is checked, not just the obvious ones.

    ``build_request_plan`` is the sink under test because it renders the
    plan and the reconstructed command, which is where a value that slips
    past the refusal would be written.
    """
    credential = "SECRETVALUE1234567890"
    echo = credential.lower()
    config = make_config(tmp_path, **build(echo))
    environ = {"ZAI_API_KEY": credential}

    with pytest.raises(OpenAIStreamCollectorError) as raised:
        assert_credential_not_embedded(config, environ)

    message = str(raised.value)
    assert field.split("[")[0] in message
    assert credential not in message
    assert echo not in message


def test_preflight_refuses_a_credential_used_as_the_variable_name() -> None:
    """The variable name is persisted, so it is checked like any other field.

    Contrived, but the name is written into the plan whenever it resolves,
    and a rule that holds for every other persisted field should not have
    an exception carved out of it.
    """
    credential = "SELFNAMINGSECRET1234"
    config = APICollectionConfig(
        run_id="api-run",
        provider="z.ai",
        endpoint=ENDPOINT,
        model_id="glm-5.3",
        prompt="hi",
        output_dir=Path("/tmp/does-not-matter"),
        command_argv=("llmtracefx-optimizer", "collect-api"),
        credential_env_var=credential,
    )

    with pytest.raises(OpenAIStreamCollectorError) as raised:
        assert_credential_not_embedded(config, {credential: credential})

    assert "credential_env_var" in str(raised.value)
    assert credential not in str(raised.value)


# A credential whose seventh character is a slash, so a cut immediately
# after the sixth lands inside that character's percent escape. Six is the
# threshold, so this is the shortest prefix the repair may act on.
_CUT_CREDENTIAL = "secret/key-abcdefgh"


@pytest.mark.parametrize(
    "cut",
    ["%", "%2", "%2f", "%2F", "%25", "%252", "%252f", "%252F"],
)
def test_boundary_repairs_a_cut_inside_a_percent_escape(cut: str) -> None:
    """Truncation inside an escape must not strand the credential head.

    ``boundary`` walks the credential element by element, and an element
    matcher rejects a half-written escape. Without the repair the walk
    stops at the cut and every character before it survives, which is
    almost the whole key for a cut near the end.
    """
    text = f"provider said error secret{cut}"

    scrubbed = _Redactor(_CUT_CREDENTIAL).boundary(text)

    assert scrubbed == "provider said error [REDACTED]"
    assert "secret" not in scrubbed


def test_boundary_repair_covers_every_cut_position() -> None:
    """No offset into the encoded credential leaves a usable prefix behind."""
    encoded = "secret" + "%252F" + "key-abcdefgh"
    for length in range(6, len(encoded) + 1):
        text = f"error {encoded[:length]}"
        scrubbed = _Redactor(_CUT_CREDENTIAL).boundary(text)
        assert scrubbed == "error [REDACTED]", encoded[:length]


@pytest.mark.parametrize(
    "text",
    [
        "all done 100%",
        "progress 50%2 of the way",
        "sec%2",
        "a discount of 25% applies",
        "%",
    ],
)
def test_boundary_repair_leaves_innocent_text_alone(text: str) -> None:
    """A trailing percent sign is ordinary text far more often than a cut key.

    The repair only fires once enough whole credential elements have
    already matched, so a bare percent, and a prefix shorter than the
    threshold, are both left as they are.
    """
    assert _Redactor(_CUT_CREDENTIAL).boundary(text) == text


def test_boundary_repair_survives_the_transformations_applied_after_it() -> None:
    """The repaired string is still clean once collapsed and truncated."""
    redactor = _Redactor(_CUT_CREDENTIAL)
    repaired = redactor.boundary("x" * 400 + " error secret%2")

    collapsed = redactor(repaired)

    assert "secret" not in collapsed
    assert "[REDACTED]" in collapsed or collapsed.endswith("...")


def test_hidden_reasoning_leaves_the_provider_token_rate_unavailable(
    tmp_path: Path,
) -> None:
    """Tokens generated in an unobserved period have no window to divide by.

    Z.ai counts reasoning tokens inside ``completion_tokens``. When it
    reports them but streams no reasoning delta, the generation window
    collapses onto the visible answer while the numerator still carries
    the hidden work, so any rate computed here overstates throughput by
    whatever fraction of the response was silent reasoning.
    """
    config = make_config(tmp_path)
    usage = {
        "prompt_tokens": 5,
        "completion_tokens": 100,
        "total_tokens": 105,
        "completion_tokens_details": {"reasoning_tokens": 90},
    }

    result, _ = run(
        config,
        glm_stream(content_parts=("a", "b"), reasoning_parts=(), usage=usage),
    )

    statistics = result.evidence.statistics
    assert statistics.reasoning_delta_count == 0
    assert statistics.generation_window_ms == statistics.content_window_ms
    assert statistics.provider_completion_tokens_per_second is None
    reason = statistics.provider_completion_tokens_per_second_unavailable_reason
    assert reason is not None
    assert "never observed" in reason
    # The visible rate stays available: it subtracts the reasoning tokens
    # from its numerator, so the window it uses does match what it counts.
    assert statistics.provider_visible_completion_tokens_per_second is not None


@pytest.mark.parametrize(
    "reasoning_tokens,reasoning_parts",
    [
        (90, ("thinking", "harder")),
        (0, ()),
    ],
)
def test_provider_token_rate_is_published_when_the_window_is_observable(
    tmp_path: Path, reasoning_tokens: int | None, reasoning_parts: tuple[str, ...]
) -> None:
    """Streamed reasoning and an explicit zero both leave the rate usable.

    These are the two shapes that answer the question with evidence. A
    streamed reasoning delta puts the reasoning inside the observed window,
    and an explicit zero says there was none. A *missing* count answers
    nothing and is covered separately, because silence is the shape a
    provider that thinks by default produces.
    """
    config = make_config(tmp_path)
    usage: dict[str, Any] = {
        "prompt_tokens": 5,
        "completion_tokens": 100,
        "total_tokens": 105,
    }
    if reasoning_tokens is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning_tokens}

    result, _ = run(
        config,
        glm_stream(
            content_parts=("a", "b"), reasoning_parts=reasoning_parts, usage=usage
        ),
    )

    statistics = result.evidence.statistics
    assert statistics.provider_completion_tokens_per_second is not None
    assert statistics.provider_completion_tokens_per_second_unavailable_reason is None


def test_the_unavailable_reason_is_persisted_in_the_record(tmp_path: Path) -> None:
    """A consumer reading the artifact sees why the rate is missing."""
    config = make_config(tmp_path)
    usage = {
        "prompt_tokens": 5,
        "completion_tokens": 100,
        "total_tokens": 105,
        "completion_tokens_details": {"reasoning_tokens": 90},
    }

    result, _ = run(
        config,
        glm_stream(content_parts=("a", "b"), reasoning_parts=(), usage=usage),
    )

    payload = result.evidence.statistics.to_dict()
    assert payload["provider_completion_tokens_per_second"] is None
    assert payload["provider_completion_tokens_per_second_unavailable_reason"]


# --- Twelfth review pass -----------------------------------------------------


@pytest.mark.parametrize(
    "credential,field,value",
    [
        ("ABCDE", "model_id", "abcde"),
        ("ABCDE", "run_id", "abcde"),
        ("Ab-Cd", "provider", "ab-cd"),
        ("SHORT", "system_prompt", "short"),
        ("a/b", "model_revision", "a%2Fb"),
    ],
)
def test_a_short_credential_cannot_slip_past_the_preflight(
    tmp_path: Path, credential: str, field: str, value: str
) -> None:
    """The length gate belonged to the decodings, never to the matcher.

    ``_contains_credential`` used to return early for anything shorter
    than the encoded-credential threshold, comparing literally and case
    sensitively, while ``_Redactor`` matched every length case
    insensitively and through percent-encoding. That gap let a value the
    redactor treats as the credential be written into the plan, the
    reconstructed command and the persisted config.
    """
    config = make_config(tmp_path, **{field: value})

    assert _Redactor(credential).text(value) == "[REDACTED]"
    with pytest.raises(OpenAIStreamCollectorError, match="refusing to run"):
        run(config, glm_stream(), environ={"ZAI_API_KEY": credential})


def test_the_output_directory_cannot_be_named_after_the_credential(
    tmp_path: Path,
) -> None:
    """A path is written into the filesystem, where no redactor reaches.

    Every other persisted field was checked, but ``output_dir`` becomes a
    directory name on disk. Creating it would put the key in the
    filesystem itself, visible to anything that can list the parent.
    """
    config = make_config(tmp_path, output_dir=tmp_path / API_KEY)

    with pytest.raises(OpenAIStreamCollectorError, match="refusing to run"):
        run(config, glm_stream())

    assert not (tmp_path / API_KEY).exists()


def test_requested_reasoning_with_no_accounting_leaves_the_rate_unavailable(
    tmp_path: Path,
) -> None:
    """Asking for reasoning and being told nothing is not evidence of none.

    A provider that streams no reasoning delta and reports no reasoning
    token count has accounted for the thinking in neither way it can. The
    window may therefore miss tokens the numerator counts, and treating
    the absent count as zero is the inference this collector refuses to
    make everywhere else.
    """
    config = make_config(
        tmp_path, extensions=ProviderExtensions(thinking_type="enabled")
    )
    usage = {"prompt_tokens": 5, "completion_tokens": 100, "total_tokens": 105}

    result, _ = run(
        config,
        glm_stream(content_parts=("a", "b"), reasoning_parts=(), usage=usage),
    )

    statistics = result.evidence.statistics
    assert result.evidence.usage.reasoning_tokens is None
    assert statistics.provider_completion_tokens_per_second is None
    reason = statistics.provider_completion_tokens_per_second_unavailable_reason
    assert reason is not None
    assert "cannot be ruled out" in reason


def test_reasoning_effort_alone_also_counts_as_requesting_reasoning(
    tmp_path: Path,
) -> None:
    """Either control asks the model to think, so either arms the check."""
    config = make_config(
        tmp_path, extensions=ProviderExtensions(reasoning_effort="high")
    )
    usage = {"prompt_tokens": 5, "completion_tokens": 100, "total_tokens": 105}

    result, _ = run(
        config,
        glm_stream(content_parts=("a", "b"), reasoning_parts=(), usage=usage),
    )

    assert result.evidence.statistics.provider_completion_tokens_per_second is None


def test_silence_about_reasoning_withholds_the_rate(tmp_path: Path) -> None:
    """Not asking for reasoning does not mean reasoning did not happen.

    Omitting ``reasoning_effort`` leaves the provider free to apply its own
    default, and for ``glm-5.3`` and ``glm-5.3-flash`` that default is
    ``max``. So the plainest possible request, which sets nothing, is a
    thinking request, and a provider that then reports no reasoning delta
    and no reasoning token count has told us nothing either way. Publishing
    a rate here would divide a numerator that may include reasoning tokens
    by a window that begins at the first visible character.
    """
    config = make_config(tmp_path, extensions=ProviderExtensions())
    usage = {"prompt_tokens": 5, "completion_tokens": 100, "total_tokens": 105}

    result, _ = run(
        config,
        glm_stream(content_parts=("a", "b"), reasoning_parts=(), usage=usage),
    )

    statistics = result.evidence.statistics
    assert statistics.provider_completion_tokens_per_second is None
    reason = statistics.provider_completion_tokens_per_second_unavailable_reason
    assert reason is not None
    assert "cannot be ruled out" in reason
    # The visible rate is unavailable too, and for its own reason: it works
    # by subtracting reasoning tokens from the numerator, which a provider
    # that reported no count has not given us anything to subtract.
    assert statistics.provider_visible_completion_tokens_per_second is None


def test_explicitly_disabled_thinking_publishes_the_rate(tmp_path: Path) -> None:
    """Turning thinking off is the request-side way to rule reasoning out.

    This is the escape hatch that keeps the metric available for ordinary
    non-reasoning models: say so explicitly and the rate is published.
    """
    config = make_config(
        tmp_path, extensions=ProviderExtensions(thinking_type="disabled")
    )
    usage = {"prompt_tokens": 5, "completion_tokens": 100, "total_tokens": 105}

    result, _ = run(
        config,
        glm_stream(content_parts=("a", "b"), reasoning_parts=(), usage=usage),
    )

    statistics = result.evidence.statistics
    assert statistics.provider_completion_tokens_per_second is not None
    assert statistics.provider_completion_tokens_per_second_unavailable_reason is None


def test_requested_reasoning_that_was_streamed_still_publishes_the_rate(
    tmp_path: Path,
) -> None:
    """Observed reasoning is inside the window, so the rate is honest."""
    config = make_config(
        tmp_path, extensions=ProviderExtensions(thinking_type="enabled")
    )
    usage = {"prompt_tokens": 5, "completion_tokens": 100, "total_tokens": 105}

    result, _ = run(
        config,
        glm_stream(content_parts=("a", "b"), reasoning_parts=("think",), usage=usage),
    )

    statistics = result.evidence.statistics
    assert statistics.reasoning_delta_count == 1
    assert statistics.provider_completion_tokens_per_second is not None


@pytest.mark.parametrize(
    "body",
    [
        {"request_id": "req-body-1", "error": {"message": "bad", "code": "1210"}},
        {"error": {"message": "bad", "code": "1210", "request_id": "req-body-1"}},
    ],
)
def test_a_body_level_request_id_survives_an_http_failure(
    tmp_path: Path, body: dict[str, Any]
) -> None:
    """The id is what makes a failed call traceable with the provider.

    Z.ai returns it in the body and an error response need not carry the
    header form, so consulting only headers threw away the one field the
    caller needs when raising a support ticket.
    """
    config = make_config(tmp_path)

    result, _ = run(config, FakeResponse([json.dumps(body).encode()], status_code=400))

    assert result.record.outcome.success is False
    assert result.evidence.failure is not None
    assert result.evidence.provider_request_id == "req-body-1"


def test_a_body_request_id_echoing_the_credential_is_redacted(
    tmp_path: Path,
) -> None:
    """It is provider-controlled text like any other."""
    config = make_config(tmp_path)
    body = json.dumps(
        {"request_id": f"req-{API_KEY}", "error": {"message": "bad"}}
    ).encode()

    result, _ = run(config, FakeResponse([body], status_code=400))

    request_id = result.evidence.provider_request_id
    assert request_id is not None
    assert API_KEY not in request_id
    assert "[REDACTED]" in request_id


def test_a_header_request_id_is_not_overwritten_by_a_body_without_one(
    tmp_path: Path,
) -> None:
    """The existing header path must keep working unchanged."""
    config = make_config(tmp_path)

    result, _ = run(
        config,
        FakeResponse(
            [json.dumps({"error": {"message": "bad"}}).encode()],
            status_code=400,
            headers={"x-request-id": "req-header-9"},
        ),
    )

    assert result.evidence.provider_request_id == "req-header-9"


# --- Finish reason vocabulary as configuration -------------------------------
#
# ``finish_reason`` is not fully standardized. Z.ai documents "sensitive",
# "network_error" and "model_context_window_exceeded" alongside the OpenAI
# set, so which strings end a stream is a property of the endpoint rather
# than of this collector.
# https://docs.z.ai/api-reference/llm/chat-completion


def _finish_stream(reason: str, *, done: bool = True) -> list[bytes]:
    chunks = [
        sse(
            {"id": "chatcmpl-f", "choices": [{"index": 0, "delta": {"content": "hi"}}]}
        ),
        sse(
            {
                "id": "chatcmpl-f",
                "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
            }
        ),
    ]
    if done:
        chunks.append(b"data: [DONE]\n\n")
    return chunks


@pytest.mark.parametrize(
    ("reason", "classification"),
    [
        ("stop", "terminal"),
        ("length", "terminal"),
        ("sensitive", "terminal"),
        ("network_error", "failure"),
        ("model_context_window_exceeded", "failure"),
        ("eos", "unrecognized"),
    ],
)
def test_the_default_vocabulary_keeps_the_documented_zai_semantics(
    tmp_path: Path, reason: str, classification: str
) -> None:
    result, _ = run(make_config(tmp_path), _finish_stream(reason))

    assert result.evidence.finish_reason_classification == classification


def test_a_provider_finish_reason_can_be_declared_terminal_in_configuration(
    tmp_path: Path,
) -> None:
    """A different provider is supported by configuring it, not by editing
    the collector."""
    config = make_config(
        tmp_path,
        provider="other",
        finish_reasons=FinishReasonVocabulary(
            terminal=frozenset({"stop", "eos"}), failure=frozenset()
        ),
    )

    result, _ = run(config, _finish_stream("eos"))

    assert result.evidence.finish_reason_classification == "terminal"
    assert result.evidence.success is True


def test_the_zai_additions_can_be_dropped_for_an_endpoint_that_reuses_them(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path, finish_reasons=FinishReasonVocabulary.openai_only())

    result, _ = run(config, _finish_stream("sensitive"))

    assert result.evidence.finish_reason_classification == "unrecognized"


def test_an_unrecognized_reason_without_done_is_still_reported_as_truncated(
    tmp_path: Path,
) -> None:
    """Unknown is not a synonym for finished: an unrecognized reason is no
    evidence that generation completed."""
    result, _ = run(make_config(tmp_path), _finish_stream("eos", done=False))

    assert result.evidence.success is False
    assert result.evidence.failure is not None
    assert result.evidence.failure.category == FAILURE_STREAM_TRUNCATED


def test_a_configured_failure_reason_outranks_a_trailing_done(
    tmp_path: Path,
) -> None:
    config = make_config(
        tmp_path,
        finish_reasons=FinishReasonVocabulary(
            terminal=frozenset({"stop"}), failure=frozenset({"aborted"})
        ),
    )

    result, _ = run(config, _finish_stream("aborted"))

    assert result.evidence.success is False
    assert result.evidence.finish_reason_classification == "failure"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {"terminal": frozenset({"stop"}), "failure": frozenset({"stop"})},
            "cannot be both terminal and a failure",
        ),
        (
            {"terminal": frozenset(), "failure": frozenset()},
            "at least one terminal finish reason",
        ),
        (
            {"terminal": frozenset({"Stop"})},
            "stripped lowercase",
        ),
        (
            {"terminal": frozenset({" stop"})},
            "stripped lowercase",
        ),
        (
            {"terminal": frozenset({""})},
            "must be non-empty strings",
        ),
        (
            {"terminal": ["stop"]},
            "must be a set of strings",
        ),
    ],
)
def test_an_incoherent_vocabulary_is_rejected(
    kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(OpenAIStreamCollectorError, match=match):
        FinishReasonVocabulary(**kwargs)


def test_the_vocabulary_is_recorded_in_the_plan(tmp_path: Path) -> None:
    plan = build_request_plan(make_config(tmp_path), environ=ENVIRON)

    recorded = plan.to_dict()["finish_reasons"]

    assert "sensitive" in recorded["terminal"]
    assert recorded["failure"] == [
        "model_context_window_exceeded",
        "network_error",
    ]
    assert recorded["terminal"] == sorted(recorded["terminal"])


def test_changing_the_vocabulary_changes_the_config_identity(
    tmp_path: Path,
) -> None:
    """Two runs that read the same reason differently are not the same
    measurement configuration, even though they send identical requests."""
    default = build_request_plan(make_config(tmp_path), environ=ENVIRON)
    narrowed = build_request_plan(
        make_config(tmp_path, finish_reasons=FinishReasonVocabulary.openai_only()),
        environ=ENVIRON,
    )

    assert default.request_parameters == narrowed.request_parameters
    assert default.config_hash != narrowed.config_hash


# --- Thirteenth review pass ---------------------------------------------------


def assert_artifacts_are_credential_free(config: APICollectionConfig) -> None:
    """No artifact holds the key, literally or in any escaped spelling."""
    spellings = (
        API_KEY,
        _u_escape(API_KEY),
        _x_escape(API_KEY),
        "".join(f"%{byte:02X}" for byte in API_KEY.encode()),
    )
    for path in sorted(config.output_dir.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for spelling in spellings:
            assert spelling not in text, f"{spelling[:12]}... in {path.name}"


def _u_escape(value: str) -> str:
    return "".join(f"\\u{ord(character):04x}" for character in value)


def _x_escape(value: str) -> str:
    return "".join(f"\\x{ord(character):02x}" for character in value)


@pytest.mark.parametrize(
    ("label", "encode"),
    [
        ("json unicode escape", _u_escape),
        ("uppercase hex", lambda v: "".join(f"\\u{ord(c):04X}" for c in v)),
        ("wide unicode escape", lambda v: "".join(f"\\U{ord(c):08X}" for c in v)),
        ("python byte escape", _x_escape),
        ("mixed literal and escape", lambda v: v[:3] + _u_escape(v[3:])),
    ],
)
def test_a_backslash_escaped_credential_is_redacted(
    label: str, encode: Callable[[str], str]
) -> None:
    """A JSON encoder or a Python repr renders the key as an escape
    sequence. It is one mechanical decode away from the key, so an artifact
    holding it is holding the credential."""
    redactor = _Redactor(API_KEY)
    encoded = encode(API_KEY)

    assert encoded not in redactor.text(f"provider said {encoded}")


def test_an_escaped_credential_in_a_non_json_error_body_is_not_persisted(
    tmp_path: Path,
) -> None:
    """The path that matters: a body that is not re-parsed is persisted as
    it arrived, so an escape in it stays an escape."""
    config = make_config(tmp_path)
    encoded = _u_escape(API_KEY)
    body = f"upstream rejected token {encoded}".encode()

    result, _ = run(
        config,
        FakeResponse([body], status_code=400, headers={"content-type": "text/plain"}),
    )

    assert result.evidence.failure is not None
    assert encoded not in result.evidence.failure.message
    assert_artifacts_are_credential_free(config)


@pytest.mark.parametrize("encode", [_u_escape, _x_escape])
def test_truncation_inside_an_escape_redacts_from_the_credential_start(
    encode: Callable[[str], str],
) -> None:
    """A cut inside ``\\u0073`` leaves ``\\u00``, which no element matcher
    accepts. Without the repair the scrub stops at the cut and every
    credential character before it survives."""
    redactor = _Redactor(API_KEY)
    encoded = encode(API_KEY)

    for cut in range(len(encoded) // 2, len(encoded) + 1):
        cleaned = redactor.boundary(f"body {encoded[:cut]}")

        assert encoded[:cut] not in cleaned, f"cut at {cut} survived"


def test_a_cut_inside_an_encoded_whitespace_run_is_repaired() -> None:
    """A space spelled ``%2520`` or ``\\u0020`` can be cut too, and the run
    then matches nothing, leaving every character before it exposed."""
    credential = "sk-Ab9 zQ7"
    redactor = _Redactor(credential)
    head = "sk-Ab9"

    for spelling in ("%2520", "\\u0020", "%20"):
        encoded = head + spelling
        for cut in range(len(head) + 1, len(encoded) + 1):
            cleaned = redactor.boundary(f"body {encoded[:cut]}")

            assert head not in cleaned, f"{spelling!r} cut at {cut} kept the head"


def test_an_escaped_credential_is_caught_by_the_preflight_check() -> None:
    """The pre-flight check and the redactor must not disagree."""
    assert _contains_credential(f"?trace={_u_escape(API_KEY)}", API_KEY) is True


def test_escapes_that_are_not_the_credential_are_left_alone() -> None:
    """Over-matching would corrupt ordinary provider text."""
    redactor = _Redactor(API_KEY)
    message = "decode error at \\u0041\\u0042, byte \\x41 unexpected"

    assert redactor.text(message) == message


def test_the_canonical_record_leaves_the_model_phase_timings_unset(
    tmp_path: Path,
) -> None:
    """``prefill`` and ``decode`` name model phases. Neither is separable
    from transport for a hosted API, and ``decode`` would additionally
    absorb a usage chunk or ``[DONE]`` sent after generation ended."""
    config = make_config(tmp_path)

    result, _ = run(config, glm_stream())

    assert result.record.timing.prefill is None
    assert result.record.timing.decode is None
    assert result.record.timing.total is not None
    # The client-observed decomposition is kept, under names that say it is
    # client-observed and includes transport.
    timeline = result.evidence.timeline
    assert timeline.first_content_token_offset_ms is not None
    assert timeline.response_headers_offset_ms is not None
    assert timeline.first_body_chunk_offset_ms is not None


def test_a_hosted_run_records_no_local_compute_backend(tmp_path: Path) -> None:
    """``runtime.backend`` is the local backend ('Metal', 'CUDA', 'CPU').
    ``runtime.provider`` exists so a transport is not stored as hardware."""
    config = make_config(tmp_path)

    result, _ = run(config, glm_stream())

    assert result.record.runtime.backend is None
    assert result.record.runtime.provider == "z.ai"
    assert result.record.runtime.name == "openai-compatible-api"


def test_a_stream_that_outlives_the_timeout_budget_fails_as_a_timeout(
    tmp_path: Path,
) -> None:
    """A server emitting a keepalive before each socket timeout expires
    resets the transport timeout forever, so the run would otherwise never
    complete and never fail."""
    config = make_config(tmp_path, request_timeout_seconds=0.05)
    chunks = [b": keepalive\n\n"] * 200

    result, _ = run(config, chunks)

    assert result.evidence.failure is not None
    assert result.evidence.failure.category == FAILURE_TIMEOUT
    assert "request timeout" in result.evidence.failure.message
    assert result.record.outcome.success is False
    assert_failure_artifacts(config, FAILURE_TIMEOUT)


def test_the_deadline_does_not_cut_short_a_stream_inside_its_budget(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path, request_timeout_seconds=600.0)

    result, _ = run(config, glm_stream())

    assert result.evidence.failure is None
    assert result.record.outcome.success is True


def test_the_timeout_message_never_repeats_provider_text(tmp_path: Path) -> None:
    config = make_config(tmp_path, request_timeout_seconds=0.05)
    chunks = [b": keepalive\n\n"] * 200

    result, _ = run(config, chunks)

    assert result.evidence.failure is not None
    assert API_KEY not in result.evidence.failure.message
    assert_artifacts_are_credential_free(config)


# --- Fourteenth review pass ---------------------------------------------------


@pytest.mark.parametrize(
    ("credential", "spelling", "decoder"),
    [
        ("sk-slash/abc123def456", "sk-slash\\/abc123def456", "json"),
        ("sk-back\\abc123def456", "sk-back\\\\abc123def456", "json"),
        ('sk-quote"abc123def456', 'sk-quote\\"abc123def456', "json"),
        ("sk-tick'abc123def456", "sk-tick\\'abc123def456", "python"),
    ],
)
def test_json_short_escapes_are_a_spelling_of_the_credential(
    credential: str, spelling: str, decoder: str
) -> None:
    """``\\/``, ``\\\\``, ``\\"`` and ``\\'`` decode straight back to the key.

    A JSON encoder is free to write ``/`` as ``\\/``, and every encoder
    writes a backslash and a double quote that way. The numeric escapes were
    already covered; these short ones are the same leak with a shorter
    spelling, and ``json.loads`` recovers the credential exactly.
    """
    redactor = _Redactor(credential)
    persisted = redactor(redactor.boundary(f"error {spelling} denied"))

    assert spelling not in persisted
    assert credential not in persisted
    # The spelling really is one mechanical decode from the key, which is
    # what makes persisting it a leak rather than a cosmetic issue.
    if decoder == "json":
        assert json.loads(f'"{spelling}"') == credential
    else:
        assert codecs.decode(spelling, "unicode_escape") == credential
    assert "[REDACTED]" in persisted


def test_no_cut_through_a_short_escape_leaks_a_usable_prefix() -> None:
    """Truncation inside ``\\/`` must repair like truncation inside ``%2F``."""
    credential = "sk-slash/abc123def456"
    redactor = _Redactor(credential)
    body = "error sk-slash\\/abc123def456 denied"

    for cut in range(len(body) + 1):
        persisted = redactor(redactor.boundary(body[:cut]))
        for length in range(_MIN_LEAKED_PREFIX_CHARS, len(credential) + 1):
            assert credential[:length] not in persisted, f"cut {cut}"


def test_an_innocent_escape_is_not_redacted() -> None:
    """The added spellings must not swallow unrelated text."""
    persisted = _Redactor("sk-slash/abc123def456").text(
        "traceback: opened C:\\temp and wrote \\u0041 ok"
    )

    assert "[REDACTED]" not in persisted


def test_silence_about_reasoning_leaves_the_rate_unavailable_end_to_end(
    tmp_path: Path,
) -> None:
    """The suppressed rate must be absent from the persisted artifact too."""
    config = make_config(tmp_path, extensions=ProviderExtensions())
    usage = {"prompt_tokens": 5, "completion_tokens": 100, "total_tokens": 105}

    run(config, glm_stream(content_parts=("a", "b"), reasoning_parts=(), usage=usage))

    payload = json.loads(
        (config.output_dir / "api_evidence.json").read_text(encoding="utf-8")
    )
    statistics = payload["statistics"]
    assert statistics["provider_completion_tokens_per_second"] is None
    assert "cannot be ruled out" in (
        statistics["provider_completion_tokens_per_second_unavailable_reason"]
    )


def test_the_retained_event_timeline_is_bounded(tmp_path: Path) -> None:
    """A chatty provider must not grow the timeline without limit.

    The counters and the offsets every derived metric reads stay exact; only
    the per-event rows stop accumulating, and the timeline records that.
    """
    limit = 12
    config = replace(make_config(tmp_path), retained_event_limit=limit)
    chatty = limit + 250
    stream: list[bytes] = [
        sse({"id": "c", "choices": [{"index": 0, "delta": {"role": "assistant"}}]})
    ]
    # Metadata chunks carry no content, which is exactly the cheap event a
    # provider can emit without limit.
    stream.extend(
        sse({"id": "c", "choices": [{"index": 0, "delta": {}}]}) for _ in range(chatty)
    )
    stream.append(
        sse({"id": "c", "choices": [{"index": 0, "delta": {"content": "a"}}]})
    )
    stream.append(
        sse(
            {
                "id": "c",
                "choices": [
                    {"index": 0, "delta": {"content": ""}, "finish_reason": "stop"}
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )
    )
    stream.append(b"data: [DONE]\n\n")

    result, _ = run(config, stream)

    timeline = result.evidence.timeline
    assert len(timeline.events) == limit
    # One role chunk, every metadata chunk, one content chunk and the final
    # usage chunk. ``[DONE]`` is a sentinel, not an event.
    assert timeline.total_event_count == chatty + 3
    assert timeline.events_truncated is True
    assert timeline.retained_event_limit == limit
    assert timeline.completed_offset_ms is not None
    # The rows stop, the accounting does not: the answer and the provider
    # usage are unaffected by having dropped per-event detail.
    assert result.response_text == "a"
    assert result.evidence.usage is not None
    assert result.evidence.usage.completion_tokens == 1
    # The bound is part of what produced this evidence, so it is part of the
    # configuration identity and part of the persisted timeline.
    payload = json.loads(
        (config.output_dir / "api_evidence.json").read_text(encoding="utf-8")
    )
    assert payload["timeline"]["retained_event_limit"] == limit
    assert payload["timeline"]["events_truncated"] is True
    assert (
        len(payload["timeline"]["events"]) == limit
    ), "the artifact must be bounded, not only the in-memory timeline"
    assert DEFAULT_RETAINED_EVENT_LIMIT > limit


# --- Fifteenth review pass ---------------------------------------------------


def _octal(value: str) -> str:
    return "".join(f"\\{ord(char):o}" for char in value)


def _hexed(value: str) -> str:
    return value.encode("utf-8").hex()


def _b64(value: str) -> str:
    return base64.b64encode(value.encode("utf-8")).decode("ascii")


def _b64url(value: str) -> str:
    return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii")


# Whole-value re-encodings. Unlike the percent and backslash spellings these
# do not keep the credential's own characters, so a literal or character-wise
# matcher sees nothing at all and the value is persisted verbatim.
_TRANSPORT_SPELLINGS = (
    ("octal escaped", _octal),
    ("hex encoded", _hexed),
    ("base64", _b64),
    ("base64url", _b64url),
    ("base64 embedded", lambda value: f"trace-{_b64(value)}-end"),
    ("base64 offset by one", lambda value: _b64("x" + value)),
    ("base64 offset by two", lambda value: _b64("xy" + value)),
    ("hex uppercase", lambda value: _hexed(value).upper()),
)


@pytest.mark.parametrize("label,transform", _TRANSPORT_SPELLINGS)
def test_transport_encodings_are_matched_and_redacted(
    label: str, transform: Any
) -> None:
    """A re-encoded credential is still the credential.

    Anything that logs or echoes a request can hand back the key
    base64 encoded, hex encoded or escaped, and none of those share a
    single character with the original. Matching only the literal spelling
    means the pre-flight passes and the value lands in an artifact intact
    and trivially reversible.
    """
    echo = transform(_SPELLING_CREDENTIAL)
    assert _SPELLING_CREDENTIAL not in echo, label

    assert _contains_credential(echo, _SPELLING_CREDENTIAL), label
    scrubbed = _Redactor(_SPELLING_CREDENTIAL).text(echo)
    assert scrubbed != echo, label
    assert _b64(_SPELLING_CREDENTIAL) not in scrubbed, label
    assert _hexed(_SPELLING_CREDENTIAL) not in scrubbed.lower(), label


def test_unicode_normalisation_forms_are_matched() -> None:
    """The same key in NFC and NFD is byte-different and character-equal.

    A provider that normalises what it echoes returns a string that no
    literal comparison matches, while a human reading the artifact sees the
    key.
    """
    credential = "clé-Sécrète-9182"
    composed = unicodedata.normalize("NFC", credential)
    decomposed = unicodedata.normalize("NFD", credential)
    assert composed != decomposed

    for spelling in (composed, decomposed):
        assert _contains_credential(f"echo={spelling}", composed)
        assert _Redactor(composed).text(f"echo={spelling}") == "echo=[REDACTED]"
        assert _contains_credential(f"echo={spelling}", decomposed)


def test_ordinary_text_is_not_matched_as_a_transport_encoding() -> None:
    """Decoding every candidate must not turn into matching everything."""
    credential = "sk-Live/Key Value-8712"
    for value in (
        "deadbeefdeadbeefdeadbeef",
        "aGVsbG8gd29ybGQgdGhpcyBpcyBmaW5l",
        "Explain a stack in one sentence.",
        "chatcmpl-abc123",
        "https://api.z.ai/api/paas/v4/chat/completions",
    ):
        assert not _contains_credential(value, credential), value
        assert _Redactor(credential).text(value) == value, value


def test_a_short_credential_does_not_enable_encoded_matching() -> None:
    """Encoded matching on a tiny value would collide with normal text.

    The encoded spellings of a three character key appear inside ordinary
    identifiers, so they stay off below the documented minimum rather than
    producing a redactor that eats the artifact.
    """
    assert _Redactor("abc").text("YWJj and 616263") == "YWJj and 616263"
    assert _Redactor("abc").text("literal abc here") == "literal [REDACTED] here"


@pytest.mark.parametrize("label,transform", _TRANSPORT_SPELLINGS)
def test_transport_encodings_survive_boundary_truncation(
    label: str, transform: Any
) -> None:
    """A cut inside a re-encoded credential must not leak the rest.

    Truncation is how these values reach an artifact in practice: a
    provider message is clipped to a length budget, and the clip lands
    mid-token.
    """
    encoded = transform(_SPELLING_CREDENTIAL)
    redactor = _Redactor(_SPELLING_CREDENTIAL)
    # From half the encoding onward the surviving prefix decodes to well
    # over the documented minimum number of credential characters, which is
    # the point at which the repair is required to fire. Shorter prefixes
    # are deliberately left alone; that floor is asserted separately.
    for cut in range(len(encoded) // 2, len(encoded)):
        text = f"provider said {encoded[:cut]}"
        repaired = redactor.text(redactor.boundary(text))
        assert encoded[:cut] not in repaired, f"{label} cut={cut}: {repaired}"


@pytest.mark.parametrize("label,transform", _TRANSPORT_SPELLINGS)
def test_a_tiny_encoded_fragment_is_deliberately_left_alone(
    label: str, transform: Any
) -> None:
    """The repair has a floor, and it is a choice rather than an oversight.

    A handful of characters is ambiguous with ordinary text, so blanking on
    that evidence would let any provider message ending in a plausible byte
    erase the artifact. The floor costs a few leading characters of a
    truncated key and buys a redactor that does not eat real evidence.
    """
    encoded = transform(_SPELLING_CREDENTIAL)
    redactor = _Redactor(_SPELLING_CREDENTIAL)
    text = f"provider said {encoded[:4]}"

    assert redactor.boundary(text) == text, label


def test_backslash_spellings_do_not_take_exponential_time() -> None:
    """The matcher must be a walk, not a backtracking search.

    Every backslash has several spellings that are prefixes of each other,
    which is the exact shape that makes a regex alternation backtrack
    exponentially. A credential of backslashes is therefore a denial of
    service against anything that redacts untrusted provider text.
    """
    redactor = _Redactor("\\" * 40)
    text = "provider said " + ("\\" * 40) + " end"

    started = time.perf_counter()
    redactor.text(text)
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0, f"redaction took {elapsed:.2f}s"


def test_the_socket_is_found_under_an_error_response_too() -> None:
    """A non-2xx body must be bounded by the same deadline as a success.

    ``urlopen`` raises an ``HTTPError`` wrapping the response, so the
    socket sits one wrapper deeper than on the success path. Naming a
    single depth silently skips the tightening on exactly the responses a
    provider is most likely to stall on.
    """

    class _Sock:
        def __init__(self) -> None:
            self.timeouts: list[float] = []

        def settimeout(self, value: float) -> None:
            self.timeouts.append(value)

    sock = _Sock()
    raw = type("_Raw", (), {"_sock": sock})()
    inner = type("_Inner", (), {"raw": raw})()
    success_shape = type("_Response", (), {"raw": raw})()
    error_shape = type("_HTTPError", (), {"fp": inner})()

    assert _response_socket(success_shape) is sock
    assert _response_socket(error_shape) is sock
    assert _response_socket(object()) is None
    assert _response_socket(None) is None


def test_a_credential_like_query_key_is_refused_without_being_echoed() -> None:
    """The refusal must not become the leak it is preventing.

    The key of a query parameter is caller controlled and is exactly where
    a credential ends up when someone puts it in the URL, so quoting it
    back into a message or a traceback republishes it.
    """
    canary = "CANARY-SECRET-VALUE-91827"
    endpoint = f"https://api.z.ai/api/paas/v4/chat/completions?api_key_{canary}=1"

    with pytest.raises(OpenAIStreamCollectorError) as raised:
        APICollectionConfig(
            run_id="r",
            provider="zai",
            endpoint=endpoint,
            model_id="glm-5.3",
            prompt="hi",
            output_dir=Path("/tmp/does-not-matter"),
            command_argv=("llmtracefx-optimizer",),
            credential_env_var="ZAI_API_KEY",
        )

    message = str(raised.value)
    assert canary not in message
    assert "query" in message.lower()
    assert _safe_endpoint_for_message(endpoint).find(canary) == -1


def test_the_retained_event_limit_changes_the_config_hash(tmp_path: Path) -> None:
    """Two runs that kept different evidence are not the same configuration.

    The config hash is what later analysis groups by. A run whose timeline
    was cut at ten rows and one that kept every row produced materially
    different artifacts, so collapsing them into one identity would let a
    comparison average incomparable evidence.
    """
    baseline = make_config(tmp_path)
    narrowed = replace(baseline, retained_event_limit=10)

    assert build_request_plan(baseline).config_hash != (
        build_request_plan(narrowed).config_hash
    )
    # The bound changes the identity without changing the request itself.
    assert build_request_plan(baseline).request_parameters == (
        build_request_plan(narrowed).request_parameters
    )
    assert build_request_plan(baseline).workload_hash == (
        build_request_plan(narrowed).workload_hash
    )


@pytest.mark.parametrize("limit", [0, -1, 1.5, True, "20"])
def test_a_bad_retained_event_limit_is_refused(tmp_path: Path, limit: Any) -> None:
    """A bound that is not a positive integer silently breaks the timeline."""
    with pytest.raises(OpenAIStreamCollectorError) as raised:
        replace(make_config(tmp_path), retained_event_limit=limit)

    assert "retained_event_limit must be a positive integer" in str(raised.value)
