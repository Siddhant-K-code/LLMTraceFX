from __future__ import annotations

import pytest

from llmtracefx.cache_audit.schema import (
    AuditManifest,
    CacheConfig,
    EvidenceBasis,
    EvidenceFact,
    PublicationMode,
    RequestSpec,
    ScenarioKind,
)
from llmtracefx.optimizer.schema import SchemaValidationError


def test_unavailable_fact_requires_null() -> None:
    with pytest.raises(SchemaValidationError, match="null"):
        EvidenceFact(
            value=0,
            basis=EvidenceBasis.UNAVAILABLE,
            source="test",
        )


def test_available_fact_requires_value() -> None:
    with pytest.raises(SchemaValidationError, match="non-null"):
        EvidenceFact(
            value=None,
            basis=EvidenceBasis.OBSERVED,
            source="test",
        )


def test_request_spec_round_trip_and_redaction() -> None:
    request = RequestSpec(
        request_id="r1",
        scenario=ScenarioKind.COLD,
        order=0,
        input_token_ids=(1, 2, 3),
        input_token_count=3,
    )
    assert RequestSpec.from_dict(request.to_dict()) == request
    redacted = request.to_dict(include_tokens=False)
    loaded = RequestSpec.from_dict(redacted)
    assert loaded.input_token_ids is None
    assert loaded.input_token_count == 3


def test_manifest_is_strict_and_round_trips() -> None:
    manifest = AuditManifest(
        run_id="run",
        created_at="2026-01-01T00:00:00Z",
        generated_at="2026-01-01T00:00:01Z",
        backend="synthetic_reference",
        backend_version="1",
        adapter_version="1",
        model_id="synthetic-model",
        tokenizer_id="integer-tokenizer",
        model_artifact_digest=None,
        runtime_identity={"python": "3.12"},
        cache_config=CacheConfig(
            namespace_id="test",
            cache_type="token_trie",
        ),
        publication_mode=PublicationMode.PUBLIC_SYNTHETIC,
        request_order=("r1",),
        workload_digest="sha256:" + "a" * 64,
        seed=0,
    )
    assert AuditManifest.from_dict(manifest.to_dict()) == manifest
    malformed = manifest.to_dict()
    malformed["unexpected"] = True
    with pytest.raises(SchemaValidationError, match="extra"):
        AuditManifest.from_dict(malformed)
