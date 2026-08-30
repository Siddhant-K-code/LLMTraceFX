"""Contract tests between the API collector's artifacts and the compare loader.

``compare/evidence.py`` reads ``api_evidence.json``, which the OpenAI-compatible
collector writes and which has no loader of its own. The other compare tests
build that sidecar from a hand-written fixture, which would keep passing if the
collector renamed or moved a field tomorrow.

These tests close that gap: they serialize evidence produced by the *real*
collector types and feed the result straight into the compare loader, so a
drift in either direction fails here rather than silently producing a
comparison with no usage, no decode settings or no time-to-first-token.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmtracefx.optimizer.collectors import openai_api
from llmtracefx.optimizer.compare.evidence import ApiEvidence
from llmtracefx.optimizer.schema import utc_now_iso

_ENDPOINT = "https://example.invalid/v1/chat/completions"


def _collector_evidence(
    *,
    usage: openai_api.ProviderUsage,
    reasoning_effort: str | None = "high",
    max_output_tokens: int | None = 512,
    temperature: float | None = 0.0,
    top_p: float | None = 1.0,
    first_content_token_offset_ms: float | None = 240.0,
) -> openai_api.APIEvidence:
    config = openai_api.APICollectionConfig(
        run_id="api-run",
        provider="z.ai",
        endpoint=_ENDPOINT,
        model_id="glm-5.3",
        model_revision="2026-06",
        prompt="synthetic prompt",
        output_dir=Path("/tmp/compare-contract-does-not-matter"),
        command_argv=("llmtracefx-optimizer", "collect-api"),
        credential_env_var="EXAMPLE_API_KEY",
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        extensions=openai_api.ProviderExtensions(reasoning_effort=reasoning_effort),
    )
    return openai_api.APIEvidence(
        schema_version=openai_api.API_EVIDENCE_SCHEMA_VERSION,
        run_id="api-run",
        collected_at=utc_now_iso(),
        plan=openai_api.build_request_plan(
            config, environ={"EXAMPLE_API_KEY": "secret"}
        ),
        success=True,
        usage=usage,
        timeline=openai_api.StreamTimeline(
            first_content_token_offset_ms=first_content_token_offset_ms
        ),
    )


def _round_trip(evidence: openai_api.APIEvidence) -> ApiEvidence:
    """Serialize exactly as the collector persists it, then load it back."""
    return ApiEvidence.from_dict(json.loads(evidence.to_json()))


def test_the_loader_reads_a_real_collector_usage_block() -> None:
    loaded = _round_trip(
        _collector_evidence(
            usage=openai_api.ProviderUsage(
                reported=True,
                prompt_tokens=2100,
                completion_tokens=480,
                total_tokens=2580,
                cached_prompt_tokens=600,
                reasoning_tokens=120,
            )
        )
    )
    assert loaded.usage_reported is True
    assert loaded.usage.prompt_tokens == 2100
    assert loaded.usage.completion_tokens == 480
    assert loaded.usage.cached_prompt_tokens == 600
    assert loaded.usage.reasoning_tokens == 120


def test_an_unreported_usage_block_stays_unreported() -> None:
    loaded = _round_trip(_collector_evidence(usage=openai_api.ProviderUsage()))
    assert loaded.usage_reported is False
    assert loaded.usage.prompt_tokens is None
    assert loaded.usage.completion_tokens is None


def test_malformed_usage_fields_survive_the_round_trip() -> None:
    loaded = _round_trip(
        _collector_evidence(
            usage=openai_api.ProviderUsage(
                reported=True,
                prompt_tokens=10,
                completion_tokens=5,
                malformed_fields=("prompt_tokens_details.cached_tokens",),
            )
        )
    )
    assert loaded.usage_malformed_fields == ("prompt_tokens_details.cached_tokens",)


def test_the_loader_reads_the_real_request_plan_decode_settings() -> None:
    loaded = _round_trip(_collector_evidence(usage=openai_api.ProviderUsage()))
    assert loaded.decode_settings.max_output_tokens == 512
    assert loaded.decode_settings.temperature == pytest.approx(0.0)
    assert loaded.decode_settings.top_p == pytest.approx(1.0)
    assert loaded.decode_settings.source == "api_request_plan"


def test_unset_decode_settings_stay_unrecorded_rather_than_defaulted() -> None:
    loaded = _round_trip(
        _collector_evidence(
            usage=openai_api.ProviderUsage(),
            max_output_tokens=None,
            temperature=None,
            top_p=None,
        )
    )
    assert loaded.decode_settings.max_output_tokens is None
    assert loaded.decode_settings.temperature is None
    assert loaded.decode_settings.top_p is None


def test_the_loader_reads_the_real_provider_model_and_reasoning_effort() -> None:
    loaded = _round_trip(_collector_evidence(usage=openai_api.ProviderUsage()))
    assert loaded.provider == "z.ai"
    assert loaded.model_id == "glm-5.3"
    assert loaded.model_revision == "2026-06"
    assert loaded.reasoning_effort == "high"


def test_an_absent_reasoning_effort_stays_none() -> None:
    loaded = _round_trip(
        _collector_evidence(usage=openai_api.ProviderUsage(), reasoning_effort=None)
    )
    assert loaded.reasoning_effort is None


def test_the_loader_reads_the_real_client_observed_ttft() -> None:
    loaded = _round_trip(_collector_evidence(usage=openai_api.ProviderUsage()))
    assert loaded.client_ttft_ms == pytest.approx(240.0)


def test_a_stream_with_no_content_token_reports_no_ttft() -> None:
    loaded = _round_trip(
        _collector_evidence(
            usage=openai_api.ProviderUsage(), first_content_token_offset_ms=None
        )
    )
    assert loaded.client_ttft_ms is None


def test_the_persisted_sidecar_carries_no_credential_or_prompt_text() -> None:
    """The compare layer inherits the collector's redaction, so pin it here."""
    payload = _collector_evidence(usage=openai_api.ProviderUsage()).to_json()
    assert "secret" not in payload
    assert "synthetic prompt" not in payload
    assert '"reasoning_text_persisted": false' in payload
