"""Tests for executing the workload matrix through an OpenAI-compatible API.

Nothing here touches the network. Every test injects a fake transport, and
the two tests that must prove no request was attempted inject a transport
that fails the test if it is ever opened. No real API key is used and no
real endpoint is contacted: the OpenRouter and Z.ai profiles are exercised
only as configuration.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from llmtracefx.optimizer.collectors._shared import sha256_bytes
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
from llmtracefx.optimizer.workloads import api_verify, evaluators
from llmtracefx.optimizer.workloads.api_profiles import (
    API_PROFILES,
    OPENROUTER_PROFILE,
    ZAI_PROFILE,
    APIProfileError,
    profile_by_name,
)
from llmtracefx.optimizer.workloads.api_verify import (
    RUN_MANIFEST_NAME,
    APIBinding,
    APIVerifyError,
    execute_api_row,
    plan_api_row,
    plan_selected_api_rows,
    render_plan_document,
    run_artifacts_are_complete,
    run_selected_api_rows,
)
from llmtracefx.optimizer.workloads.catalog import (
    CODE_COMPLETION_PALINDROME,
    PROSE_REASONING_TRAIN_PROBLEM,
    STRUCTURED_JSON_PROFILE_EXTRACTION,
)
from llmtracefx.optimizer.workloads.matrix import (
    DECODE_MODE_AUTOREGRESSIVE,
    DECODE_MODE_NATIVE_MTP,
    MatrixEntry,
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


def sse_comment() -> bytes:
    """A keepalive comment: a chunk that dispatches no event at all."""
    return b": keepalive\n\n"


def test_event_cap_abandons_a_chatty_stream_and_says_so(tmp_path):
    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    result = run_one(
        tmp_path, transport=transport, binding=make_binding(max_stream_events=2)
    )

    assert result.verification.status is RowStatus.FAILED
    assert "2-event cap" in (result.verification.reason or "")


@pytest.mark.parametrize("cap", [1, 2, 3, 4])
def test_a_cap_the_stream_exceeds_never_yields_a_graded_pass(tmp_path, cap):
    """Exceeding the cap discards the verdict but keeps the measurement."""
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
    assert result.final_record is not None
    assert result.verification.total_ms is not None


def test_event_cap_high_enough_does_not_interfere(tmp_path):
    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    result = run_one(
        tmp_path, transport=transport, binding=make_binding(max_stream_events=1000)
    )
    assert result.verification.status is RowStatus.COMPLETED


@pytest.mark.parametrize(
    ("label", "trailer"),
    [
        ("nothing", []),
        ("a keepalive comment", [sse_comment()]),
        ("a stray blank line", [b"\n"]),
        ("a keepalive split across reads", [b": keep", b"alive\n\n"]),
        ("several non-event chunks", [sse_comment(), b"\n", sse_comment()]),
    ],
)
def test_a_trailing_non_event_chunk_never_trips_the_cap(tmp_path, label, trailer):
    """A chunk is not an event, and only events are charged to the cap.

    A provider may close with a terminal ``finish_reason`` and no
    ``[DONE]`` sentinel, then leave a keepalive or a blank line in the
    pipe. Deciding truncation on "another chunk exists" rather than on the
    event count failed these streams even though every byte of the answer
    had already been delivered.
    """
    chunks = answer_stream(GOOD_JSON_ANSWER)[:-1] + list(trailer)
    transport = FakeTransport(FakeResponse(chunks))
    result = run_one(
        tmp_path,
        transport=transport,
        # Four events were dispatched; the trailer adds none.
        binding=make_binding(max_stream_events=4),
    )

    assert result.verification.status is RowStatus.COMPLETED, label
    assert result.verification.outcome_success is True
    assert result.verification.quality_score == 1.0


def segmentations(parts: list[bytes]) -> dict[str, list[bytes]]:
    """The same wire bytes, split the ways a network might deliver them."""
    body = b"".join(parts)
    return {
        "one read per event": list(parts),
        "sentinel glued to the last event": [*parts[:-2], parts[-2] + parts[-1]],
        "whole body in one read": [body],
        "sixteen byte reads": [body[i : i + 16] for i in range(0, len(body), 16)],
        "sixty four byte reads": [body[i : i + 64] for i in range(0, len(body), 64)],
    }


@pytest.mark.parametrize("cap", [2, 3, 4, 5, 6])
def test_the_verdict_never_depends_on_network_segmentation(tmp_path, cap):
    """Identical wire bytes must produce one verdict, at every cap.

    Asserting a fixed outcome at a cap the stream cannot exceed proves
    nothing: every implementation of this class agrees there. The property
    is that all segmentations agree *with each other*, so it is swept
    across the boundary rather than pinned to one side of it, and the
    caps either side of the boundary are included.
    """
    parts = answer_stream(GOOD_JSON_ANSWER)
    verdicts = {}
    for label, chunks in segmentations(parts).items():
        result = run_one(
            tmp_path / f"cap{cap}" / label.replace(" ", "-"),
            transport=FakeTransport(FakeResponse(chunks)),
            binding=make_binding(max_stream_events=cap),
        )
        verdicts[label] = result.verification.status

    assert len(set(verdicts.values())) == 1, verdicts


@pytest.mark.parametrize("trailer_label", ["duplicate sentinel", "trailing content"])
def test_frames_after_the_sentinel_are_never_charged(tmp_path, trailer_label):
    """A gateway may append frames the collector will never read.

    The collector returns at ``[DONE]``, so anything after it is only
    ever observed when it shares a socket read with the sentinel. Charging
    it would let the network decide the verdict, and would fail a
    complete, correct answer as a truncation.
    """
    parts = answer_stream(GOOD_JSON_ANSWER)
    trailer = (
        b"data: [DONE]\n\n"
        if trailer_label == "duplicate sentinel"
        else sse({"choices": [{"index": 0, "delta": {"content": " extra"}}]})
    )
    parts = [*parts, trailer]

    verdicts = {}
    for label, chunks in segmentations(parts).items():
        result = run_one(
            tmp_path / label.replace(" ", "-"),
            transport=FakeTransport(FakeResponse(chunks)),
            binding=make_binding(
                max_stream_events=len(answer_stream(GOOD_JSON_ANSWER))
            ),
        )
        verdicts[label] = (
            result.verification.status,
            result.verification.quality_score,
        )

    assert set(verdicts.values()) == {(RowStatus.COMPLETED, 1.0)}, verdicts


def test_a_stream_past_the_cap_still_fails_whatever_the_segmentation(tmp_path):
    """The boundary still exists; it is just denominated in events."""
    parts = answer_stream(GOOD_JSON_ANSWER)
    for label, chunks in (
        ("one read per event", list(parts)),
        ("whole body in one read", [b"".join(parts)]),
        (
            "sixteen byte reads",
            [b"".join(parts)[i : i + 16] for i in range(0, len(b"".join(parts)), 16)],
        ),
    ):
        transport = FakeTransport(FakeResponse(chunks))
        result = run_one(
            tmp_path / label.replace(" ", "-"),
            transport=transport,
            binding=make_binding(max_stream_events=2),
        )
        assert result.verification.status is RowStatus.FAILED, label


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


# --- Executable evaluators are refused over an API ----------------------------


def test_a_code_workload_is_unsupported_and_never_sent(tmp_path):
    """Grading this category runs the answer as a local program.

    Over an API that answer is produced by a remote party, so executing it
    hands that party local code execution. The row is refused before a
    transport exists and before the evaluator is chosen.
    """
    manifest, manifest_dir, matrix_path = build_manifest(
        tmp_path, workloads=(CODE_COMPLETION_PALINDROME,)
    )
    entry = autoregressive_entry(manifest)
    assert entry.category == "code_completion"

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

    assert result.verification.status is RowStatus.UNSUPPORTED
    reason = (result.verification.reason or "").lower()
    assert "executing the model's answer" in reason
    assert "sandbox" in reason
    assert result.final_record is None


def test_a_code_workload_never_reaches_the_evaluator(tmp_path, monkeypatch):
    """Belt and braces: the evaluator must not be invoked at all."""

    def forbidden(workload, response_text):
        raise AssertionError("the evaluator must never run for an API code row")

    monkeypatch.setattr(api_verify, "evaluate_workload", forbidden)
    manifest, manifest_dir, matrix_path = build_manifest(
        tmp_path, workloads=(CODE_COMPLETION_PALINDROME,)
    )
    result = execute_api_row(
        autoregressive_entry(manifest),
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        binding=make_binding(),
        resume=True,
        transport_factory=ExplodingTransport,
        environ=ENVIRON,
    )
    assert result.verification.status is RowStatus.UNSUPPORTED


def test_a_code_workload_is_blocked_in_the_dry_run_plan(tmp_path):
    manifest, manifest_dir, matrix_path = build_manifest(
        tmp_path, workloads=(CODE_COMPLETION_PALINDROME,)
    )
    plans = plan_selected_api_rows(
        manifest,
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        selection=RowSelection(decode_modes=frozenset({DECODE_MODE_AUTOREGRESSIVE})),
        binding=make_binding(),
        environ=ENVIRON,
    )

    assert plans[0].unsupported is True
    assert plans[0].request_plan is None
    assert "local code execution" in (plans[0].unsupported_reason or "")


def test_non_executing_categories_are_still_runnable(tmp_path):
    """The refusal is narrow: only the category that executes is refused."""
    for workload in (STRUCTURED_JSON_PROFILE_EXTRACTION, PROSE_REASONING_TRAIN_PROBLEM):
        bundle = build_manifest(tmp_path / workload.workload_id, workloads=(workload,))
        answer = (
            GOOD_JSON_ANSWER
            if workload is STRUCTURED_JSON_PROFILE_EXTRACTION
            else "The gap closes after 2 hours."
        )
        result = run_one(
            tmp_path / workload.workload_id,
            transport=FakeTransport(FakeResponse(answer_stream(answer))),
            manifest_bundle=bundle,
        )
        assert result.verification.status is RowStatus.COMPLETED, workload.workload_id


# --- Run-level completion marker ---------------------------------------------


def test_a_completed_row_writes_a_run_level_marker(tmp_path):
    result = run_one(
        tmp_path, transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    )
    run_dir = Path(result.verification.collection_dir or "").parent

    assert (run_dir / RUN_MANIFEST_NAME).exists()
    assert run_artifacts_are_complete(run_dir, expected_run_id=run_dir.name)
    sealed = {
        entry["name"]
        for entry in json.loads(
            (run_dir / RUN_MANIFEST_NAME).read_text(encoding="utf-8")
        )["artifacts"]
    }
    # The two files the collector's own marker does not cover.
    assert "final_record.json" in sealed
    assert "verification.json" in sealed


def test_an_unsafe_artifact_during_marker_creation_does_not_crash(
    tmp_path, monkeypatch
):
    def refuse_marker(**_kwargs):
        raise api_verify.ArtifactReadError("unsafe sealed artifact")

    monkeypatch.setattr(api_verify, "_run_marker_payload", refuse_marker)

    result = run_one(
        tmp_path, transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    )
    run_dir = Path(result.verification.collection_dir or "").parent

    assert result.verification.status is RowStatus.COMPLETED
    assert not (run_dir / RUN_MANIFEST_NAME).exists()


@pytest.mark.parametrize("victim", ["final_record.json", "verification.json"])
def test_editing_a_file_outside_the_collection_marker_forces_a_rerun(tmp_path, victim):
    """These two sit outside the collector's marker and were trusted blind."""
    bundle = build_manifest(tmp_path)
    first = run_one(
        tmp_path,
        transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER))),
        manifest_bundle=bundle,
    )
    run_dir = Path(first.verification.collection_dir or "").parent

    target = run_dir / victim
    payload = json.loads(target.read_text(encoding="utf-8"))
    if victim == "verification.json":
        payload["quality_score"] = 1.0
        payload["outcome_success"] = True
    else:
        payload["outcome"]["quality_score"] = 1.0
    target.write_text(json.dumps(payload), encoding="utf-8")

    assert not run_artifacts_are_complete(run_dir, expected_run_id=run_dir.name)

    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    second = run_one(tmp_path, transport=transport, manifest_bundle=bundle)
    assert transport.requests, "a tampered run directory must rerun"
    assert second.verification.status is RowStatus.COMPLETED


def test_a_missing_run_marker_forces_a_rerun(tmp_path):
    """An interrupted write leaves no marker, and no marker means no trust."""
    bundle = build_manifest(tmp_path)
    first = run_one(
        tmp_path,
        transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER))),
        manifest_bundle=bundle,
    )
    run_dir = Path(first.verification.collection_dir or "").parent
    (run_dir / RUN_MANIFEST_NAME).unlink()

    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    second = run_one(tmp_path, transport=transport, manifest_bundle=bundle)
    assert transport.requests, "a run with no marker must rerun"
    assert second.verification.status is RowStatus.COMPLETED


def test_a_corrupt_run_marker_forces_a_rerun(tmp_path):
    bundle = build_manifest(tmp_path)
    first = run_one(
        tmp_path,
        transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER))),
        manifest_bundle=bundle,
    )
    run_dir = Path(first.verification.collection_dir or "").parent
    (run_dir / RUN_MANIFEST_NAME).write_text("{ not json", encoding="utf-8")

    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    second = run_one(tmp_path, transport=transport, manifest_bundle=bundle)
    assert transport.requests
    assert second.verification.status is RowStatus.COMPLETED


def test_a_manifest_that_mislabels_a_code_workload_cannot_bypass_the_refusal(
    tmp_path, monkeypatch
):
    """The catalog is the authority, because the evaluator dispatches on it.

    A matrix manifest is a file on disk. One that declares a
    code-completion row as ``structured_json`` passes a check that trusts
    the manifest, then resolves the real workload from the catalog and is
    graded by executing the answer. The manifest must not be able to talk
    the pipeline into running a remote endpoint's code.
    """

    def forbidden(spec, response_text):
        raise AssertionError("the code evaluator must never run for an API row")

    monkeypatch.setattr(evaluators, "evaluate_code_completion", forbidden)

    manifest, manifest_dir, matrix_path = build_manifest(
        tmp_path, workloads=(CODE_COMPLETION_PALINDROME,)
    )
    doctored = dict(autoregressive_entry(manifest).to_dict())
    doctored["category"] = "structured_json"

    result = execute_api_row(
        MatrixEntry.from_dict(doctored),
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        binding=make_binding(),
        resume=True,
        transport_factory=lambda: FakeTransport(
            FakeResponse(answer_stream("def is_palindrome(s): return True"))
        ),
        environ=ENVIRON,
    )

    assert result.verification.status is RowStatus.UNSUPPORTED
    assert "executing the model's answer" in (result.verification.reason or "")


def test_a_mislabelled_code_workload_is_also_blocked_in_the_plan(tmp_path):
    manifest, manifest_dir, matrix_path = build_manifest(
        tmp_path, workloads=(CODE_COMPLETION_PALINDROME,)
    )
    doctored = dict(autoregressive_entry(manifest).to_dict())
    doctored["category"] = "structured_json"

    plan = plan_api_row(
        MatrixEntry.from_dict(doctored),
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        binding=make_binding(),
        environ=ENVIRON,
    )

    assert plan.unsupported is True
    assert plan.request_plan is None


@pytest.mark.parametrize(
    "rewritten_name",
    ["../escaped.json", "/etc/hostname", "collection/record.json"],
)
def test_a_marker_naming_another_file_is_rejected(tmp_path, rewritten_name):
    """The marker's names are not authority over what this process opens.

    Redirecting an entry at a file nobody edited would otherwise let a
    tampered run verify clean, which is worse than having no marker.
    """
    result = run_one(
        tmp_path, transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    )
    run_dir = Path(result.verification.collection_dir or "").parent
    marker_path = run_dir / RUN_MANIFEST_NAME

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["artifacts"][1]["name"] = rewritten_name
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    assert not run_artifacts_are_complete(run_dir, expected_run_id=run_dir.name)


def test_a_marker_that_omits_an_artifact_is_rejected(tmp_path):
    """Dropping an entry must not silently narrow what is sealed."""
    result = run_one(
        tmp_path, transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    )
    run_dir = Path(result.verification.collection_dir or "").parent
    marker_path = run_dir / RUN_MANIFEST_NAME

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["artifacts"] = [
        entry for entry in marker["artifacts"] if entry["name"] != "final_record.json"
    ]
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    assert not run_artifacts_are_complete(run_dir, expected_run_id=run_dir.name)


# --- Unsafe artifact paths ----------------------------------------------------


def _entry_with_run_id(manifest, run_id: str) -> MatrixEntry:
    doctored = dict(autoregressive_entry(manifest).to_dict())
    doctored["run_id"] = run_id
    return MatrixEntry.from_dict(doctored)


@pytest.mark.parametrize(
    "run_id",
    [
        "../escaped",
        "../../escaped",
        "nested/child",
        "..",
        ".",
        "",
        "with space",
        "semi;colon",
        # `$` matches before a trailing newline, so `re.match` accepted
        # this and created a directory the refusal text calls impossible.
        "trailing-newline\n",
        "embedded\nnewline",
    ],
)
def test_an_unsafe_run_id_is_refused_and_writes_nothing(tmp_path, run_id):
    """A run_id is read from a manifest and becomes a directory name.

    An absolute value replaces the output directory outright and ``..``
    climbs out of it, so an edited manifest could plant artifacts
    anywhere the user can write. The refusal itself must not be written
    either, since the only path available is the unsafe one.
    """
    manifest, manifest_dir, matrix_path = build_manifest(tmp_path)
    output_dir = tmp_path / "results"

    result = execute_api_row(
        _entry_with_run_id(manifest, run_id),
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=output_dir,
        binding=make_binding(),
        resume=True,
        transport_factory=ExplodingTransport,
        environ=ENVIRON,
    )

    assert result.verification.status is RowStatus.FAILED
    assert "unsafe artifact path" in (result.verification.reason or "")
    assert result.final_record is None
    # Nothing was created anywhere, including the traversal target.
    assert not output_dir.exists()
    assert not (tmp_path / "escaped").exists()


def test_an_absolute_run_id_cannot_escape_the_output_directory(tmp_path):
    """The starkest case: an absolute id discards the output directory."""
    manifest, manifest_dir, matrix_path = build_manifest(tmp_path)
    target = tmp_path / "planted"

    result = execute_api_row(
        _entry_with_run_id(manifest, str(target)),
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        binding=make_binding(),
        resume=True,
        transport_factory=ExplodingTransport,
        environ=ENVIRON,
    )

    assert result.verification.status is RowStatus.FAILED
    assert not target.exists()
    assert not (tmp_path / "results").exists()


def test_an_unsafe_run_id_is_refused_in_the_plan_without_echoing_it(tmp_path):
    """The rendered plan withholds the value, not just the paths.

    ``to_dict`` used to emit ``entry.run_id`` on the refusal path while
    blanking only the four path fields, so the rejected value still
    reached stdout and ``api_request_plan.json``. The credential case was
    saved by the whole-document redactor, which is incidental cover
    rather than the property this asserts.
    """
    manifest, manifest_dir, matrix_path = build_manifest(tmp_path)
    plan = plan_api_row(
        _entry_with_run_id(manifest, "../escaped"),
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        binding=make_binding(),
        environ=ENVIRON,
    )

    assert plan.ready is False
    assert plan.request_plan is None
    assert any("unsafe artifact path" in blocker for blocker in plan.blockers)
    assert not (tmp_path / "results").exists()

    # The whole rendered row, which is what a caller actually sees.
    rendered = json.dumps(plan.to_dict())
    assert "../escaped" not in rendered
    assert plan.to_dict()["run_id"] == "[REJECTED]"


def test_a_credential_in_the_effective_row_path_is_refused(tmp_path):
    """--output-dir may be clean while the run_id carries the key."""
    manifest, manifest_dir, matrix_path = build_manifest(tmp_path)
    output_dir = tmp_path / "results"

    result = execute_api_row(
        _entry_with_run_id(manifest, f"row-{API_KEY}"),
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=output_dir,
        binding=make_binding(),
        resume=True,
        transport_factory=ExplodingTransport,
        environ=ENVIRON,
    )

    assert result.verification.status is RowStatus.FAILED
    assert API_KEY not in (result.verification.reason or "")
    assert API_KEY not in (result.verification.run_id or "")
    assert not output_dir.exists()


# --- Resume binds the row's identity ------------------------------------------


@pytest.mark.parametrize("corruption", ["oversized", "symlink"])
def test_api_prompt_must_be_a_bounded_regular_file(tmp_path, monkeypatch, corruption):
    bundle = build_manifest(tmp_path)
    manifest, manifest_dir, matrix_path = bundle
    entry = autoregressive_entry(manifest)
    prompt_path = Path(entry.prompt_path)
    if corruption == "oversized":
        monkeypatch.setattr(api_verify, "MAX_EVIDENCE_ARTIFACT_BYTES", 8)
    else:
        target_path = tmp_path / "prompt-target.txt"
        target_path.write_text(
            prompt_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        prompt_path.unlink()
        prompt_path.symlink_to(target_path)
    binding = make_binding()

    plan = plan_api_row(
        entry,
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        binding=binding,
        environ=ENVIRON,
    )
    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    result = execute_api_row(
        entry,
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=tmp_path / "results",
        binding=binding,
        resume=True,
        transport_factory=lambda: transport,
        environ=ENVIRON,
    )

    assert not plan.ready
    assert "prompt file unreadable" in plan.blockers[0]
    assert result.verification.status is RowStatus.FAILED
    assert "prompt file unreadable" in (result.verification.reason or "")
    assert transport.requests == []


def test_a_run_directory_copied_from_another_row_is_never_trusted(tmp_path):
    """Every hash checks out; the evidence still describes another row.

    Integrity says the bytes were not edited. It says nothing about which
    row they belong to, so identity has to be asserted separately.
    """
    bundle = build_manifest(
        tmp_path,
        workloads=(STRUCTURED_JSON_PROFILE_EXTRACTION,),
        context_tiers=(ContextTier.TIER_2K, ContextTier.TIER_8K),
    )
    manifest, manifest_dir, matrix_path = bundle
    rows = [e for e in manifest.entries if e.decode_mode == DECODE_MODE_AUTOREGRESSIVE]
    source, destination = rows[0], rows[1]
    output_dir = tmp_path / "results"

    first = execute_api_row(
        source,
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=output_dir,
        binding=make_binding(),
        resume=True,
        transport_factory=lambda: FakeTransport(
            FakeResponse(answer_stream(GOOD_JSON_ANSWER))
        ),
        environ=ENVIRON,
    )
    assert first.verification.status is RowStatus.COMPLETED

    source_dir = Path(first.verification.collection_dir or "").parent
    destination_dir = output_dir / "runs" / destination.run_id
    shutil.copytree(source_dir, destination_dir)

    # The copy is byte-identical, so every hash in it still verifies.
    assert artifact_set_is_complete(destination_dir / "collection")
    # But it is not this row's evidence.
    assert not run_artifacts_are_complete(
        destination_dir, expected_run_id=destination.run_id
    )

    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    second = execute_api_row(
        destination,
        manifest_dir=manifest_dir,
        matrix_path=matrix_path,
        output_dir=output_dir,
        binding=make_binding(),
        resume=True,
        transport_factory=lambda: transport,
        environ=ENVIRON,
    )
    assert transport.requests, "a copied run directory must rerun"
    assert second.verification.status is RowStatus.COMPLETED


def test_a_marker_renamed_to_another_run_is_rejected(tmp_path):
    result = run_one(
        tmp_path, transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    )
    run_dir = Path(result.verification.collection_dir or "").parent

    assert run_artifacts_are_complete(run_dir, expected_run_id=run_dir.name)
    assert not run_artifacts_are_complete(run_dir, expected_run_id="some-other-row")


# --- Corrupt, non-UTF-8 artifacts -------------------------------------------


def _reseal(run_dir: Path) -> None:
    """Rewrite the marker so it matches whatever is on disk now."""
    marker_path = run_dir / RUN_MANIFEST_NAME
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    for entry in marker["artifacts"]:
        entry["sha256"] = sha256_bytes((run_dir / entry["name"]).read_bytes())
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


# --- Malformed but syntactically valid artifacts -------------------------------

#: Payloads that are valid JSON, or valid nothing, but are not the object
#: every one of these readers assumes. ``json.loads`` returns a list, a
#: scalar or ``None`` without complaint, and the field access that follows
#: raises ``AttributeError`` or ``TypeError`` -- neither of which is a
#: parse error any caller was catching.
_MALFORMED_ROOTS = {
    "empty list": b"[]",
    "populated list": b'["run_id"]',
    "null": b"null",
    "integer": b"5",
    "string": b'"not an object"',
    "bare true": b"true",
    "invalid utf-8": b"\xff\xfe\x00 not utf-8",
    "truncated json": b'{"run_id": ',
}


@pytest.mark.parametrize("payload_label", sorted(_MALFORMED_ROOTS))
@pytest.mark.parametrize(
    "victim", [RUN_MANIFEST_NAME, "verification.json", "final_record.json"]
)
def test_a_malformed_artifact_reruns_instead_of_crashing(
    tmp_path, victim, payload_label
):
    """Unreadable evidence is untrusted evidence, never an exception.

    Corrupt is not only "not JSON": ``[]`` parses fine and then fails on
    the first field access, which is a different exception type from the
    one the handlers were written for. Every reader on the resume path
    has to survive all of it and simply decline to trust the row.
    """
    bundle = build_manifest(tmp_path)
    first = run_one(
        tmp_path,
        transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER))),
        manifest_bundle=bundle,
    )
    run_dir = Path(first.verification.collection_dir or "").parent
    (run_dir / victim).write_bytes(_MALFORMED_ROOTS[payload_label])

    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    second = run_one(tmp_path, transport=transport, manifest_bundle=bundle)

    assert transport.requests, f"{victim} / {payload_label} must rerun"
    assert second.verification.status is RowStatus.COMPLETED


@pytest.mark.parametrize("payload_label", sorted(_MALFORMED_ROOTS))
def test_a_malformed_final_record_behind_a_valid_marker_reruns(tmp_path, payload_label):
    """Reaching the record parse at all requires the marker to agree.

    Corrupting the record alone breaks the marker, so resume stops
    earlier. Resealing puts the run into the one state where the record
    is actually parsed, which is what an attacker rewriting the marker
    would leave behind, and is the only way to exercise that reader.
    """
    bundle = build_manifest(tmp_path)
    first = run_one(
        tmp_path,
        transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER))),
        manifest_bundle=bundle,
    )
    run_dir = Path(first.verification.collection_dir or "").parent
    (run_dir / "final_record.json").write_bytes(_MALFORMED_ROOTS[payload_label])
    _reseal(run_dir)

    assert run_artifacts_are_complete(run_dir, expected_run_id=run_dir.name)

    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    second = run_one(tmp_path, transport=transport, manifest_bundle=bundle)

    assert transport.requests, f"an unparsable record ({payload_label}) must rerun"
    assert second.verification.status is RowStatus.COMPLETED


def test_a_non_finite_final_record_behind_a_valid_marker_reruns(tmp_path):
    bundle = build_manifest(tmp_path)
    first = run_one(
        tmp_path,
        transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER))),
        manifest_bundle=bundle,
    )
    run_dir = Path(first.verification.collection_dir or "").parent
    record_path = run_dir / "final_record.json"
    payload = json.loads(record_path.read_text(encoding="utf-8"))
    payload["timing"]["total"]["value"] = "NON_FINITE"
    record_path.write_text(
        json.dumps(payload).replace('"NON_FINITE"', "1e400"), encoding="utf-8"
    )
    _reseal(run_dir)

    assert run_artifacts_are_complete(run_dir, expected_run_id=run_dir.name)

    transport = FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    second = run_one(tmp_path, transport=transport, manifest_bundle=bundle)

    assert transport.requests
    assert second.verification.status is RowStatus.COMPLETED


@pytest.mark.parametrize("payload_label", sorted(_MALFORMED_ROOTS))
def test_a_malformed_run_marker_is_rejected_not_raised(tmp_path, payload_label):
    """The marker reader is the one gate that must never raise."""
    result = run_one(
        tmp_path, transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    )
    run_dir = Path(result.verification.collection_dir or "").parent
    (run_dir / RUN_MANIFEST_NAME).write_bytes(_MALFORMED_ROOTS[payload_label])

    assert not run_artifacts_are_complete(run_dir, expected_run_id=run_dir.name)


@pytest.mark.parametrize("corruption", ["recursion", "oversized", "symlink"])
def test_an_unsafe_run_marker_is_rejected_not_raised(tmp_path, corruption, monkeypatch):
    result = run_one(
        tmp_path, transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    )
    run_dir = Path(result.verification.collection_dir or "").parent
    marker_path = run_dir / RUN_MANIFEST_NAME
    if corruption == "recursion":
        marker_path.write_text("[" * 10_000 + "0" + "]" * 10_000, encoding="utf-8")
    elif corruption == "oversized":
        monkeypatch.setattr(api_verify, "MAX_METADATA_ARTIFACT_BYTES", 16)
        marker_path.write_bytes(b"x" * 17)
    else:
        target = marker_path.with_name("run-target.json")
        target.write_bytes(marker_path.read_bytes())
        marker_path.unlink()
        marker_path.symlink_to(target)

    assert not run_artifacts_are_complete(run_dir, expected_run_id=run_dir.name)


@pytest.mark.parametrize("corruption", ["oversized", "symlink"])
def test_an_unsafe_sealed_artifact_is_rejected_not_raised(
    tmp_path, corruption, monkeypatch
):
    result = run_one(
        tmp_path, transport=FakeTransport(FakeResponse(answer_stream(GOOD_JSON_ANSWER)))
    )
    run_dir = Path(result.verification.collection_dir or "").parent
    artifact_path = run_dir / "verification.json"
    if corruption == "oversized":
        monkeypatch.setitem(api_verify._SEALED_ARTIFACT_LIMITS, "verification.json", 16)
        artifact_path.write_bytes(b"x" * 17)
    else:
        target = artifact_path.with_name("verification-target.json")
        target.write_bytes(artifact_path.read_bytes())
        artifact_path.unlink()
        artifact_path.symlink_to(target)

    assert not run_artifacts_are_complete(run_dir, expected_run_id=run_dir.name)
