"""Tests for executing the workload matrix through an OpenAI-compatible API.

Nothing here touches the network. Every test injects a fake transport, and
the two tests that must prove no request was attempted inject a transport
that fails the test if it is ever opened. No real API key is used and no
real endpoint is contacted: the OpenRouter and Z.ai profiles are exercised
only as configuration.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer.collectors.openai_api import (
    ARTIFACT_MANIFEST_NAME,
    FAILURE_HTTP_STATUS,
    FAILURE_STREAM_DECODE,
    FAILURE_STREAM_TRUNCATED,
    HTTPRequest,
    ProviderExtensions,
    TransportConnectionError,
    artifact_set_is_complete,
)
from llmtracefx.optimizer.workloads import api_verify
from llmtracefx.optimizer.workloads.api_profiles import (
    API_PROFILES,
    OPENROUTER_PROFILE,
    ZAI_PROFILE,
    APIProfileError,
    profile_by_name,
)
from llmtracefx.optimizer.workloads.api_verify import (
    APIBinding,
    APIVerifyError,
    execute_api_row,
    plan_selected_api_rows,
    render_plan_document,
    run_selected_api_rows,
)
from llmtracefx.optimizer.workloads.catalog import (
    PROSE_REASONING_TRAIN_PROBLEM,
    STRUCTURED_JSON_PROFILE_EXTRACTION,
)
from llmtracefx.optimizer.workloads.matrix import (
    DECODE_MODE_AUTOREGRESSIVE,
    DECODE_MODE_NATIVE_MTP,
    MatrixManifest,
    generate_matrix,
    write_matrix,
)
from llmtracefx.optimizer.workloads.schema import ContextTier
from llmtracefx.optimizer.workloads.verify import (
    BACKEND_MLX,
    BACKEND_OPENAI_API,
    RowSelection,
    RowStatus,
    RowVerification,
)

API_KEY = "api-verify-test-key-not-a-real-credential"
ENV_VAR = "LLMTRACEFX_TEST_API_KEY"
ENVIRON: Mapping[str, str] = {ENV_VAR: API_KEY}

GOOD_JSON_ANSWER = '{"name": "Priya", "age": 34, "is_active": true}'
BAD_JSON_ANSWER = "there is no json here at all"


# --- Fakes -------------------------------------------------------------------


class FakeResponse:
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
    """Replays one recorded response, recording what was requested."""

    def __init__(self, response: FakeResponse | Exception) -> None:
        self._response = response
        self.requests: list[HTTPRequest] = []

    def open_stream(self, request: HTTPRequest) -> FakeResponse:
        self.requests.append(request)
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class ExplodingTransport:
    """Fails the test if a request is ever attempted."""

    def open_stream(self, request: HTTPRequest) -> FakeResponse:
        raise AssertionError("no network request should have been attempted")


def sse(payload: dict[str, Any]) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode()


def answer_stream(
    answer: str, *, reasoning: str = "weighing the options"
) -> list[bytes]:
    """A GLM-shaped stream whose *content* is ``answer``.

    The reasoning delta deliberately says something other than the answer,
    so a test asserting the evaluator's verdict is asserting that only the
    content stream was graded.
    """
    return [
        sse(
            {
                "id": "chatcmpl-test",
                "model": "glm-5.3",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": ""}}
                ],
            }
        ),
        sse({"choices": [{"index": 0, "delta": {"reasoning_content": reasoning}}]}),
        sse({"choices": [{"index": 0, "delta": {"content": answer}}]}),
        sse(
            {
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                },
            }
        ),
        b"data: [DONE]\n\n",
    ]


# --- Fixtures ----------------------------------------------------------------


def build_manifest(
    tmp_path: Path,
    *,
    workloads: tuple[Any, ...] = (STRUCTURED_JSON_PROFILE_EXTRACTION,),
    context_tiers: tuple[ContextTier, ...] = (ContextTier.TIER_2K,),
) -> tuple[MatrixManifest, Path, Path]:
    """Generate and write a small matrix; return manifest, dir and path."""
    output_dir = tmp_path / "matrix"
    manifest = generate_matrix(
        model_id="local/test-model",
        model_family="qwen3_next",
        output_dir=str(output_dir),
        workloads=workloads,
        context_tiers=context_tiers,
        mtp_depths=(2,),
    )
    write_matrix(manifest)
    matrix_path = output_dir / "manifest.json"
    return MatrixManifest.read_json(matrix_path), output_dir, matrix_path


def make_binding(**overrides: Any) -> APIBinding:
    kwargs: dict[str, Any] = {
        "provider": OPENROUTER_PROFILE.provider_label,
        "endpoint": OPENROUTER_PROFILE.endpoint,
        "model_id": "z-ai/glm-5.3",
        "credential_env_var": ENV_VAR,
    }
    kwargs.update(overrides)
    return APIBinding(**kwargs)


def autoregressive_entry(manifest: MatrixManifest) -> Any:
    return next(
        entry
        for entry in manifest.entries
        if entry.decode_mode == DECODE_MODE_AUTOREGRESSIVE
    )


def native_mtp_entry(manifest: MatrixManifest) -> Any:
    return next(
        entry
        for entry in manifest.entries
        if entry.decode_mode == DECODE_MODE_NATIVE_MTP
    )


def run_one(
    tmp_path: Path,
    *,
    transport: Any,
    binding: APIBinding | None = None,
    resume: bool = True,
    entry: Any = None,
    environ: Mapping[str, str] = ENVIRON,
    manifest_bundle: tuple[MatrixManifest, Path, Path] | None = None,
) -> Any:
    manifest, manifest_dir, matrix_path = manifest_bundle or build_manifest(tmp_path)
    return execute_api_row(
        entry or autoregressive_entry(manifest),
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        binding=binding or make_binding(),
        resume=resume,
        transport_factory=lambda: transport,
        environ=environ,
    )


# --- Profiles ----------------------------------------------------------------


def test_openrouter_profile_matches_documented_values():
    assert OPENROUTER_PROFILE.endpoint == (
        "https://openrouter.ai/api/v1/chat/completions"
    )
    assert OPENROUTER_PROFILE.credential_env_var == "OPENROUTER_API_KEY"
    assert OPENROUTER_PROFILE.documented_model_ids == (
        "z-ai/glm-5.3",
        "z-ai/glm-5.3-flash",
    )


def test_zai_profile_is_retained_alongside_openrouter():
    assert ZAI_PROFILE.endpoint == "https://api.z.ai/api/paas/v4/chat/completions"
    assert ZAI_PROFILE.credential_env_var == "ZAI_API_KEY"
    assert {profile.name for profile in API_PROFILES} == {"openrouter", "z.ai"}


def test_profile_lookup_rejects_unknown_name_without_echoing_it():
    with pytest.raises(APIProfileError) as excinfo:
        profile_by_name("not-a-profile-and-possibly-a-secret")
    message = str(excinfo.value)
    assert "not-a-profile-and-possibly-a-secret" not in message
    assert "openrouter" in message


def test_profiles_are_defaults_not_hardcoded_behaviour(tmp_path):
    """An unlisted provider is a first-class citizen, not a special case."""
    binding = make_binding(
        provider="self-hosted",
        endpoint="https://vllm.internal.example/v1/chat/completions",
        model_id="local-glm",
    )
    result = run_one(
        tmp_path,
        transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER))),
        binding=binding,
    )
    assert result.verification.status is RowStatus.COMPLETED
    assert result.verification.provider == "self-hosted"


# --- Binding -----------------------------------------------------------------


def test_binding_rejects_non_positive_event_cap():
    with pytest.raises(APIVerifyError):
        make_binding(max_stream_events=0)


def test_binding_validate_rejects_plain_http_remote_endpoint():
    binding = make_binding(endpoint="http://openrouter.ai/api/v1/chat/completions")
    with pytest.raises(APIVerifyError):
        binding.validate()


def test_binding_validate_accepts_documented_profiles():
    for profile in API_PROFILES:
        make_binding(
            provider=profile.provider_label,
            endpoint=profile.endpoint,
            model_id=profile.documented_model_ids[0],
            credential_env_var=profile.credential_env_var,
        ).validate()


def _binding_hash_for(tmp_path: Path, binding: APIBinding) -> str:
    manifest, manifest_dir, matrix_path = build_manifest(tmp_path)
    plans = plan_selected_api_rows(
        manifest,
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        selection=RowSelection(decode_modes=frozenset({DECODE_MODE_AUTOREGRESSIVE})),
        binding=binding,
        environ=ENVIRON,
    )
    assert plans[0].binding_hash is not None
    return plans[0].binding_hash


@pytest.mark.parametrize(
    "overrides",
    [
        {"model_id": "z-ai/glm-5.3-flash"},
        {"model_revision": "2026-01-01"},
        {"endpoint": "https://api.z.ai/api/paas/v4/chat/completions"},
        {"provider": "z.ai"},
        {"temperature": 0.25},
        {"top_p": 0.9},
        {"seed": 7},
        {"request_timeout_seconds": 30.0},
        {"max_stream_events": 42},
        {"system_prompt": "Answer tersely."},
        {"extensions": ProviderExtensions(reasoning_effort="low")},
        {"extensions": ProviderExtensions(thinking_type="enabled")},
    ],
)
def test_binding_hash_changes_with_every_request_affecting_value(tmp_path, overrides):
    baseline = _binding_hash_for(tmp_path / "a", make_binding())
    changed = _binding_hash_for(tmp_path / "b", make_binding(**overrides))
    assert baseline != changed


def test_binding_hash_ignores_credential_variable_name(tmp_path):
    """Two runs differing only in which variable held the key are identical.

    They issue byte-identical requests and are graded identically, so the
    name affects neither request, evaluation nor resume -- and hashing it
    would persist a derivation of a value that may be the credential.
    """
    baseline = _binding_hash_for(tmp_path / "a", make_binding())
    renamed = _binding_hash_for(
        tmp_path / "b", make_binding(credential_env_var="OTHER_KEY_VAR")
    )
    assert baseline == renamed


def test_binding_hash_never_contains_the_credential(tmp_path):
    binding = make_binding()
    assert API_KEY not in _binding_hash_for(tmp_path, binding)


# --- Successful execution ----------------------------------------------------


def test_successful_row_is_completed_and_evaluated(tmp_path):
    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    result = run_one(tmp_path, transport=transport)

    verification = result.verification
    assert verification.status is RowStatus.COMPLETED
    assert verification.backend == BACKEND_OPENAI_API
    assert verification.provider == "openrouter"
    assert verification.api_model_id == "z-ai/glm-5.3"
    assert verification.artifacts_verified is True
    assert verification.outcome_success is True
    assert verification.quality_score == 1.0
    assert result.final_record is not None
    assert result.final_record.outcome.quality_metric is not None


def test_row_uses_matrix_max_tokens_for_the_request(tmp_path):
    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    manifest, _, _ = build_manifest(tmp_path)
    entry = autoregressive_entry(manifest)
    run_one(tmp_path, transport=transport)

    body = json.loads(transport.requests[0].body.decode("utf-8"))
    assert body["max_tokens"] == entry.max_tokens
    assert body["stream"] is True


def test_reasoning_settings_are_sent_when_configured(tmp_path):
    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    binding = make_binding(
        extensions=ProviderExtensions(
            reasoning_effort="high", thinking_type="enabled", clear_thinking=False
        )
    )
    run_one(tmp_path, transport=transport, binding=binding)

    body = json.loads(transport.requests[0].body.decode("utf-8"))
    assert body["reasoning_effort"] == "high"
    assert body["thinking"] == {"type": "enabled", "clear_thinking": False}


def test_evaluation_grades_the_content_not_the_reasoning(tmp_path):
    """A model that reasons correctly but answers wrongly must fail."""
    transport = FakeTransport(
        FakeResponse(answer_stream(BAD_JSON_ANSWER, reasoning=GOOD_JSON_ANSWER))
    )
    result = run_one(tmp_path, transport=transport)

    assert result.verification.status is RowStatus.COMPLETED
    assert result.verification.outcome_success is False
    assert result.verification.quality_score == 0.0


def test_quality_failure_is_completed_not_failed(tmp_path):
    """A wrong answer is a measured result, not a broken run."""
    transport = FakeTransport(FakeResponse(answer_stream(BAD_JSON_ANSWER)))
    result = run_one(tmp_path, transport=transport)
    assert result.verification.status is RowStatus.COMPLETED
    assert result.verification.outcome_success is False


def test_artifacts_are_written_and_marked_complete(tmp_path):
    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    result = run_one(tmp_path, transport=transport)

    collection_dir = Path(result.verification.collection_dir or "")
    assert artifact_set_is_complete(collection_dir)
    for name in (
        "record.json",
        "response.txt",
        "api_evidence.json",
        "environment.json",
    ):
        assert (collection_dir / name).exists()
    assert Path(result.verification.final_record_path or "").exists()


# --- Provider failures -------------------------------------------------------


def test_http_status_failure_is_failed_and_never_evaluated(tmp_path):
    transport = FakeTransport(
        FakeResponse([b'{"error": {"message": "no"}}'], status_code=429)
    )
    result = run_one(tmp_path, transport=transport)

    assert result.verification.status is RowStatus.FAILED
    assert FAILURE_HTTP_STATUS in (result.verification.reason or "")
    # The evaluator never ran, so no quality verdict was invented.
    assert result.verification.quality_score is None
    assert result.verification.outcome_success is False


def test_provider_failure_is_not_overwritten_by_a_passing_answer(tmp_path):
    """A 500 whose body happens to contain a passing answer still fails.

    This is the case the short circuit exists for: the content is exactly
    what the evaluator wants, so a pipeline that graded before checking the
    outcome would publish it as a pass.
    """
    transport = FakeTransport(
        FakeResponse([GOOD_JSON_ANSWER.encode("utf-8")], status_code=500)
    )
    result = run_one(tmp_path, transport=transport)

    assert result.verification.status is RowStatus.FAILED
    assert result.verification.outcome_success is False
    assert result.verification.quality_score is None


def test_connection_failure_is_failed(tmp_path):
    transport = FakeTransport(TransportConnectionError("connection refused"))
    result = run_one(tmp_path, transport=transport)
    assert result.verification.status is RowStatus.FAILED


def test_truncated_stream_is_failed(tmp_path):
    """A stream that stops before [DONE] is never published as an answer."""
    chunks = answer_stream(GOOD_JSON_ANSWER)[:-2]
    transport = FakeTransport(FakeResponse(chunks))
    result = run_one(tmp_path, transport=transport)

    assert result.verification.status is RowStatus.FAILED
    assert FAILURE_STREAM_TRUNCATED in (result.verification.reason or "")


def test_malformed_sse_is_propagated_as_a_decode_failure(tmp_path):
    """The collector's decode error reaches the row verdict unchanged.

    The event cap feeds a second copy of every chunk to its own decoder,
    so this also proves that counting never swallows or pre-empts the
    collector's authoritative diagnostic.
    """
    transport = FakeTransport(FakeResponse([b"data: {\xff\xfe not utf-8 \n\n"]))
    result = run_one(tmp_path, transport=transport)

    assert result.verification.status is RowStatus.FAILED
    assert FAILURE_STREAM_DECODE in (result.verification.reason or "")


def test_missing_credential_fails_the_row_without_evidence(tmp_path):
    result = run_one(
        tmp_path,
        transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER))),
        environ={},
    )
    assert result.verification.status is RowStatus.FAILED
    assert "could not be attempted" in (result.verification.reason or "")
    assert result.final_record is None


# --- Event cap ---------------------------------------------------------------


def test_event_cap_abandons_a_chatty_stream_and_says_so(tmp_path):
    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    result = run_one(
        tmp_path, transport=transport, binding=make_binding(max_stream_events=2)
    )

    assert result.verification.status is RowStatus.FAILED
    reason = result.verification.reason or ""
    assert "2-event cap" in reason


@pytest.mark.parametrize("cap", [1, 2, 3, 4])
def test_no_cap_below_the_stream_length_ever_yields_a_graded_pass(tmp_path, cap):
    """Cutting the stream short must never be published as a verdict.

    The cap can land after a terminal ``finish_reason`` but before
    ``[DONE]``, which the collector alone would call a clean ending. This
    pipeline stopped reading, so the answer it holds is a prefix and the
    row has to fail however tidy the fragment looks.
    """
    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    result = run_one(
        tmp_path / f"cap{cap}",
        transport=transport,
        binding=make_binding(max_stream_events=cap),
    )

    assert result.verification.status is RowStatus.FAILED
    assert result.verification.outcome_success is False
    assert result.verification.quality_score is None
    assert f"{cap}-event cap" in (result.verification.reason or "")
    # The measurement survives even though the outcome refuses to claim it.
    assert result.final_record is not None
    assert result.final_record.outcome.success is False
    assert result.verification.total_ms is not None


def test_event_cap_high_enough_does_not_interfere(tmp_path):
    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    result = run_one(
        tmp_path, transport=transport, binding=make_binding(max_stream_events=1000)
    )
    assert result.verification.status is RowStatus.COMPLETED


def test_a_fully_delivered_stream_is_not_failed_at_the_exact_cap(tmp_path):
    """Reaching the cap is not the same as being cut short by it.

    A provider may end with a terminal ``finish_reason`` and no ``[DONE]``
    sentinel, which this project documents as supported. The collector
    keeps pulling after the final chunk, so a cap equal to the number of
    events the stream actually sent used to trip on a stream where every
    byte had already been delivered, and a clean measurement was
    published as a truncation.
    """
    chunks = answer_stream(GOOD_JSON_ANSWER)[:-1]
    transport = FakeTransport(FakeResponse(chunks))
    result = run_one(
        tmp_path,
        transport=transport,
        binding=make_binding(max_stream_events=len(chunks)),
    )

    assert result.verification.status is RowStatus.COMPLETED
    assert result.verification.outcome_success is True


def test_a_stream_one_event_past_the_cap_still_fails(tmp_path):
    """The boundary moved by one; it did not disappear."""
    chunks = answer_stream(GOOD_JSON_ANSWER)[:-1]
    transport = FakeTransport(FakeResponse(chunks))
    result = run_one(
        tmp_path,
        transport=transport,
        binding=make_binding(max_stream_events=len(chunks) - 1),
    )

    assert result.verification.status is RowStatus.FAILED
    assert f"{len(chunks) - 1}-event cap" in (result.verification.reason or "")


# --- Credential pre-flight ---------------------------------------------------


@pytest.mark.parametrize(
    "field", ["provider", "model_id", "endpoint", "credential_env_var"]
)
def test_a_credential_in_a_binding_field_never_reaches_an_artifact(tmp_path, field):
    """A key pasted into any persisted slot is refused, not written down.

    ``verification.json`` is written on paths that never issue a request,
    so it cannot rely on the collector having refused first.
    """
    # Shaped to satisfy every validator it has to pass on the way in: the
    # provider label pattern, the model ID check and the uppercase
    # environment-variable rule all accept this string.
    pasted = "AKIAIOSFODNN7EXAMPLE"
    overrides: dict[str, Any] = {
        "provider": pasted,
        "model_id": pasted,
        "endpoint": f"https://openrouter.ai/api/v1/chat/completions?deployment={pasted}",
        "credential_env_var": pasted,
    }
    binding = make_binding(**{field: overrides[field]})
    result = run_one(
        tmp_path,
        transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER))),
        binding=binding,
        environ={binding.credential_env_var: pasted},
    )

    assert result.verification.status is RowStatus.FAILED
    for path in sorted((tmp_path / "results").rglob("*")):
        if path.is_file():
            assert pasted not in path.read_text(encoding="utf-8", errors="replace")


def test_a_credential_in_an_endpoint_query_is_never_hashed_into_the_plan(tmp_path):
    """A hash of the secret is still a derivation of the secret.

    ``config_hash`` folds in sha256 of every endpoint query value, so the
    pre-flight has to run before the plan is built. If it did not, two
    runs differing only in the pasted key would produce different plan
    documents, which is exactly what a stored derivation looks like.
    """
    manifest, manifest_dir, matrix_path = build_manifest(tmp_path)
    documents = []
    for pasted in ("AKIAIOSFODNN7EXAMPLE", "AKIAI44QH8DHBEXAMPLE"):
        binding = make_binding(
            endpoint=(
                f"https://openrouter.ai/api/v1/chat/completions?deployment={pasted}"
            )
        )
        plans = plan_selected_api_rows(
            manifest,
            manifest_dir=manifest_dir,
            matrix_path=matrix_path,
            output_dir=tmp_path / "results",
            selection=RowSelection(
                decode_modes=frozenset({DECODE_MODE_AUTOREGRESSIVE})
            ),
            binding=binding,
            environ={ENV_VAR: pasted},
        )
        assert plans[0].ready is False
        assert plans[0].binding_hash is None
        assert any("refusing to run" in blocker for blocker in plans[0].blockers)
        document = render_plan_document(
            plans, binding=binding, environ={ENV_VAR: pasted}
        )
        assert pasted not in document
        documents.append(document)

    # Identical apart from the pasted key, so nothing derived from it was
    # written into either document.
    assert documents[0] == documents[1]


def test_dry_run_and_a_real_run_agree_about_an_embedded_credential(tmp_path):
    """A pre-flight that green-lights what the real run refuses is useless."""
    pasted = "AKIAIOSFODNN7EXAMPLE"
    bundle = build_manifest(tmp_path)
    manifest, manifest_dir, matrix_path = bundle
    binding = make_binding(provider=pasted)
    environ = {ENV_VAR: pasted}

    plans = plan_selected_api_rows(
        manifest,
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        selection=RowSelection(decode_modes=frozenset({DECODE_MODE_AUTOREGRESSIVE})),
        binding=binding,
        environ=environ,
    )
    assert plans[0].ready is False

    result = run_one(
        tmp_path,
        transport=ExplodingTransport(),
        binding=binding,
        environ=environ,
        manifest_bundle=bundle,
    )
    assert result.verification.status is RowStatus.FAILED


# --- Inconclusive ------------------------------------------------------------


def test_evaluator_error_is_inconclusive_with_evidence_preserved(tmp_path, monkeypatch):
    def exploding_evaluator(workload, response_text):
        raise RuntimeError("sandbox unavailable")

    monkeypatch.setattr(api_verify, "evaluate_workload", exploding_evaluator)
    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    result = run_one(tmp_path, transport=transport)

    verification = result.verification
    assert verification.status is RowStatus.INCONCLUSIVE
    # Quality is never guessed, but the measurement survives.
    assert verification.quality_score is None
    assert verification.total_ms is not None
    assert verification.artifacts_verified is True
    assert artifact_set_is_complete(Path(verification.collection_dir or ""))


# --- Unsupported rows --------------------------------------------------------


def test_native_mtp_row_is_unsupported_and_never_sent(tmp_path):
    manifest, manifest_dir, matrix_path = build_manifest(tmp_path)
    result = execute_api_row(
        native_mtp_entry(manifest),
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        binding=make_binding(),
        resume=True,
        transport_factory=ExplodingTransport,
        environ=ENVIRON,
    )
    assert result.verification.status is RowStatus.UNSUPPORTED
    assert "native multi-token prediction" in (result.verification.reason or "").lower()


def test_native_mtp_row_stays_unsupported_even_with_reasoning_configured(tmp_path):
    """Reasoning settings are never a stand-in for native MTP."""
    manifest, manifest_dir, matrix_path = build_manifest(tmp_path)
    result = execute_api_row(
        native_mtp_entry(manifest),
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        binding=make_binding(
            extensions=ProviderExtensions(
                reasoning_effort="max", thinking_type="enabled"
            )
        ),
        resume=True,
        transport_factory=ExplodingTransport,
        environ=ENVIRON,
    )
    assert result.verification.status is RowStatus.UNSUPPORTED


# --- Prompt and catalog integrity -------------------------------------------


def test_prompt_hash_mismatch_fails_without_sending(tmp_path):
    manifest, manifest_dir, matrix_path = build_manifest(tmp_path)
    entry = autoregressive_entry(manifest)
    Path(entry.prompt_path).write_text("a different prompt entirely", encoding="utf-8")

    result = execute_api_row(
        entry,
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        binding=make_binding(),
        resume=True,
        transport_factory=ExplodingTransport,
        environ=ENVIRON,
    )
    assert result.verification.status is RowStatus.FAILED
    assert "prompt hash mismatch" in (result.verification.reason or "")


def test_missing_prompt_file_fails_without_sending(tmp_path):
    manifest, manifest_dir, matrix_path = build_manifest(tmp_path)
    entry = autoregressive_entry(manifest)
    Path(entry.prompt_path).unlink()

    result = execute_api_row(
        entry,
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        binding=make_binding(),
        resume=True,
        transport_factory=ExplodingTransport,
        environ=ENVIRON,
    )
    assert result.verification.status is RowStatus.FAILED
    assert "prompt file missing" in (result.verification.reason or "")


def test_workload_version_drift_fails_without_sending(tmp_path, monkeypatch):
    manifest, manifest_dir, matrix_path = build_manifest(tmp_path)
    entry = autoregressive_entry(manifest)
    drifted = dict(entry.to_dict())
    drifted["workload_version"] = "999"

    from llmtracefx.optimizer.workloads.matrix import MatrixEntry

    result = execute_api_row(
        MatrixEntry.from_dict(drifted),
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        binding=make_binding(),
        resume=True,
        transport_factory=ExplodingTransport,
        environ=ENVIRON,
    )
    assert result.verification.status is RowStatus.FAILED
    assert "version drift" in (result.verification.reason or "")


# --- Resume ------------------------------------------------------------------


def test_resume_trusts_a_complete_hash_matching_artifact_set(tmp_path):
    bundle = build_manifest(tmp_path)
    first = run_one(
        tmp_path,
        transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER))),
        manifest_bundle=bundle,
    )
    assert first.verification.status is RowStatus.COMPLETED

    second = run_one(tmp_path, transport=ExplodingTransport(), manifest_bundle=bundle)
    assert second.verification.status is RowStatus.SKIPPED
    assert second.verification.resumed is True
    assert second.verification.artifacts_verified is True


def test_no_resume_reruns_even_a_valid_artifact_set(tmp_path):
    bundle = build_manifest(tmp_path)
    run_one(
        tmp_path,
        transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER))),
        manifest_bundle=bundle,
    )
    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    second = run_one(
        tmp_path, transport=transport, resume=False, manifest_bundle=bundle
    )
    assert second.verification.status is RowStatus.COMPLETED
    assert transport.requests, "a re-run must actually issue a request"


def test_resume_reruns_when_the_binding_changed(tmp_path):
    bundle = build_manifest(tmp_path)
    run_one(
        tmp_path,
        transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER))),
        manifest_bundle=bundle,
    )
    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    second = run_one(
        tmp_path,
        transport=transport,
        binding=make_binding(model_id="z-ai/glm-5.3-flash"),
        manifest_bundle=bundle,
    )
    assert second.verification.status is RowStatus.COMPLETED
    assert transport.requests


def test_resume_reruns_when_the_prompt_changed(tmp_path):
    bundle = build_manifest(tmp_path)
    manifest, _, _ = bundle
    entry = autoregressive_entry(manifest)
    run_one(
        tmp_path,
        transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER))),
        manifest_bundle=bundle,
    )

    Path(entry.prompt_path).write_text("tampered prompt", encoding="utf-8")
    second = run_one(tmp_path, transport=ExplodingTransport(), manifest_bundle=bundle)
    assert second.verification.status is RowStatus.FAILED
    assert "prompt hash mismatch" in (second.verification.reason or "")


def test_resume_reruns_when_the_artifact_marker_is_missing(tmp_path):
    bundle = build_manifest(tmp_path)
    first = run_one(
        tmp_path,
        transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER))),
        manifest_bundle=bundle,
    )
    collection_dir = Path(first.verification.collection_dir or "")
    (collection_dir / ARTIFACT_MANIFEST_NAME).unlink()

    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    second = run_one(tmp_path, transport=transport, manifest_bundle=bundle)
    assert second.verification.status is RowStatus.COMPLETED
    assert transport.requests, "an incomplete artifact set must rerun"


def test_resume_reruns_when_an_artifact_was_tampered_with(tmp_path):
    bundle = build_manifest(tmp_path)
    first = run_one(
        tmp_path,
        transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER))),
        manifest_bundle=bundle,
    )
    collection_dir = Path(first.verification.collection_dir or "")
    (collection_dir / "response.txt").write_text("edited by hand", encoding="utf-8")
    assert not artifact_set_is_complete(collection_dir)

    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    second = run_one(tmp_path, transport=transport, manifest_bundle=bundle)
    assert second.verification.status is RowStatus.COMPLETED
    assert transport.requests, "a tampered artifact set must rerun"


def test_resume_never_trusts_a_row_from_another_backend(tmp_path):
    """An MLX verification.json must not satisfy an API resume check."""
    bundle = build_manifest(tmp_path)
    manifest, _, _ = bundle
    entry = autoregressive_entry(manifest)
    first = run_one(
        tmp_path,
        transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER))),
        manifest_bundle=bundle,
    )
    verification_path = (
        tmp_path / "results" / "runs" / entry.run_id / "verification.json"
    )
    payload = json.loads(verification_path.read_text(encoding="utf-8"))
    payload["backend"] = BACKEND_MLX
    verification_path.write_text(json.dumps(payload), encoding="utf-8")
    assert first.verification.status is RowStatus.COMPLETED

    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    second = run_one(tmp_path, transport=transport, manifest_bundle=bundle)
    assert second.verification.status is RowStatus.COMPLETED
    assert transport.requests


def test_resume_ignores_a_corrupt_verification_file(tmp_path):
    bundle = build_manifest(tmp_path)
    manifest, _, _ = bundle
    entry = autoregressive_entry(manifest)
    run_one(
        tmp_path,
        transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER))),
        manifest_bundle=bundle,
    )
    verification_path = (
        tmp_path / "results" / "runs" / entry.run_id / "verification.json"
    )
    verification_path.write_text("{ not json", encoding="utf-8")

    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    second = run_one(tmp_path, transport=transport, manifest_bundle=bundle)
    assert second.verification.status is RowStatus.COMPLETED
    assert transport.requests


# --- Selection and batch execution -------------------------------------------


def test_run_selected_rows_respects_filters_and_orders_by_manifest(tmp_path):
    manifest, manifest_dir, matrix_path = build_manifest(
        tmp_path,
        workloads=(STRUCTURED_JSON_PROFILE_EXTRACTION, PROSE_REASONING_TRAIN_PROBLEM),
    )
    results = run_selected_api_rows(
        manifest,
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        selection=RowSelection(categories=frozenset({"structured_json"})),
        binding=make_binding(),
        resume=True,
        transport_factory=lambda: FakeTransport(
            FakeResponse(answer_stream(GOOD_JSON_ANSWER))
        ),
        environ=ENVIRON,
    )
    assert results
    assert {r.entry.category for r in results} == {"structured_json"}


def test_selecting_only_unsupported_rows_never_builds_a_transport(tmp_path):
    manifest, manifest_dir, matrix_path = build_manifest(tmp_path)

    def exploding_factory():
        raise AssertionError("no transport should be constructed")

    results = run_selected_api_rows(
        manifest,
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        selection=RowSelection(decode_modes=frozenset({DECODE_MODE_NATIVE_MTP})),
        binding=make_binding(),
        resume=True,
        transport_factory=exploding_factory,
        environ=ENVIRON,
    )
    assert results
    assert all(r.verification.status is RowStatus.UNSUPPORTED for r in results)


def test_empty_selection_returns_no_results(tmp_path):
    manifest, manifest_dir, matrix_path = build_manifest(tmp_path)
    results = run_selected_api_rows(
        manifest,
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        selection=RowSelection(run_ids=frozenset({"no-such-row"})),
        binding=make_binding(),
        resume=True,
        transport_factory=ExplodingTransport,
        environ=ENVIRON,
    )
    assert results == ()


# --- Dry run -----------------------------------------------------------------


def test_dry_run_plans_every_selected_row_without_network(tmp_path):
    manifest, manifest_dir, matrix_path = build_manifest(tmp_path)
    plans = plan_selected_api_rows(
        manifest,
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        selection=RowSelection(),
        binding=make_binding(),
        environ=ENVIRON,
    )
    assert len(plans) == len(manifest.entries)
    ready = [plan for plan in plans if plan.ready]
    assert ready and all(plan.request_plan is not None for plan in ready)
    assert all(plan.binding_hash is not None for plan in ready)
    unsupported = [plan for plan in plans if plan.unsupported]
    assert unsupported and all(plan.request_plan is None for plan in unsupported)


def test_dry_run_blocks_when_the_credential_variable_is_unset(tmp_path):
    manifest, manifest_dir, matrix_path = build_manifest(tmp_path)
    plans = plan_selected_api_rows(
        manifest,
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        selection=RowSelection(decode_modes=frozenset({DECODE_MODE_AUTOREGRESSIVE})),
        binding=make_binding(),
        environ={},
    )
    assert plans[0].ready is False
    assert plans[0].credential_env_var_present is False
    assert any("--api-key-env" in blocker for blocker in plans[0].blockers)


def test_dry_run_blocker_never_echoes_the_credential_variable_slot(tmp_path):
    """A key pasted into the name slot must not reach the plan document.

    The uppercase shape rule stops most keys, so this uses one shaped
    like a variable name to prove the message does not repeat it.
    """
    pasted = "AKIAIOSFODNN7EXAMPLE"
    manifest, manifest_dir, matrix_path = build_manifest(tmp_path)
    binding = make_binding(credential_env_var=pasted)
    plans = plan_selected_api_rows(
        manifest,
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        selection=RowSelection(decode_modes=frozenset({DECODE_MODE_AUTOREGRESSIVE})),
        binding=binding,
        environ={},
    )
    document = render_plan_document(plans, binding=binding, environ={})
    assert pasted not in document


def test_plan_document_holds_no_prompt_text_and_no_credential(tmp_path):
    manifest, manifest_dir, matrix_path = build_manifest(tmp_path)
    entry = autoregressive_entry(manifest)
    prompt_text = Path(entry.prompt_path).read_text(encoding="utf-8")
    binding = make_binding()
    plans = plan_selected_api_rows(
        manifest,
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        selection=RowSelection(decode_modes=frozenset({DECODE_MODE_AUTOREGRESSIVE})),
        binding=binding,
        environ=ENVIRON,
    )
    document = render_plan_document(plans, binding=binding, environ=ENVIRON)

    assert API_KEY not in document
    assert prompt_text[:200] not in document
    payload = json.loads(document)
    assert payload["network_request_performed"] is False
    assert payload["backend"] == BACKEND_OPENAI_API
    assert payload["rows"][0]["request_plan"]["messages"][0]["characters"] == len(
        prompt_text
    )


def test_dry_run_reports_prompt_and_catalog_blockers(tmp_path):
    manifest, manifest_dir, matrix_path = build_manifest(tmp_path)
    entry = autoregressive_entry(manifest)
    Path(entry.prompt_path).write_text("edited", encoding="utf-8")

    plans = plan_selected_api_rows(
        manifest,
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        selection=RowSelection(run_ids=frozenset({entry.run_id})),
        binding=make_binding(),
        environ=ENVIRON,
    )
    assert plans[0].ready is False
    assert any("prompt hash mismatch" in blocker for blocker in plans[0].blockers)
    assert plans[0].binding_hash is None


# --- Secret containment ------------------------------------------------------


def test_no_persisted_artifact_contains_the_credential(tmp_path):
    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    result = run_one(tmp_path, transport=transport)
    assert result.verification.status is RowStatus.COMPLETED

    for path in sorted((tmp_path / "results").rglob("*")):
        if path.is_file():
            assert API_KEY not in path.read_text(encoding="utf-8", errors="replace")


def test_a_provider_that_echoes_the_key_cannot_leak_it(tmp_path):
    """Evidence is redacted even when the provider replays the credential."""
    transport = FakeTransport(FakeResponse(answer_stream(f"key is {API_KEY}")))
    result = run_one(tmp_path, transport=transport)

    for path in sorted((tmp_path / "results").rglob("*")):
        if path.is_file():
            assert API_KEY not in path.read_text(encoding="utf-8", errors="replace")
    assert result.verification.status is RowStatus.COMPLETED


def test_the_recorded_command_masks_an_undefined_credential_variable(tmp_path):
    pasted = "AKIAIOSFODNN7EXAMPLE"
    binding = make_binding(credential_env_var=pasted)
    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    result = run_one(
        tmp_path, transport=transport, binding=binding, environ={pasted: API_KEY}
    )
    # Defined in the environment, so it is a real variable name and is kept.
    assert result.verification.status is RowStatus.COMPLETED
    evidence = json.loads(
        (
            Path(result.verification.collection_dir or "") / "api_evidence.json"
        ).read_text(encoding="utf-8")
    )
    assert evidence["plan"]["credential_env_var"] == pasted


# --- Verification schema -----------------------------------------------------


def test_schema_v1_verification_still_parses_as_a_local_row():
    """v2 fields are additive: a v1 document keeps its v1 meaning."""
    legacy = {
        "schema_version": "1",
        "run_id": "legacy-row",
        "workload_id": "structured-json-profile-extraction",
        "workload_version": "1",
        "category": "structured_json",
        "context_tier": "2k",
        "decode_mode": "autoregressive",
        "status": "completed",
        "reason": None,
        "recorded_prompt_hash": "sha256:abc",
        "verified_prompt_hash": "sha256:abc",
        "run_binding_hash": "sha256:def",
        "resumed": False,
        "outcome_success": True,
        "quality_score": 1.0,
        "total_ms": 12.0,
        "started_at": "2026-01-01T00:00:00Z",
        "ended_at": "2026-01-01T00:00:01Z",
        "final_record_path": "/tmp/final.json",
        "collection_dir": "/tmp/collection",
    }
    verification = RowVerification.from_dict(legacy)
    assert verification.backend == BACKEND_MLX
    assert verification.provider is None
    assert verification.api_model_id is None
    assert verification.artifacts_verified is None
